/**
 * @file AdjointDifferentiation.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Adjoint (reverse-mode) differentiation of Pauli expectation values on the
 * statevector, for unitary circuits.
 *
 * The circuit is compiled once into a parameterized program; its gate angles
 * can then be replaced row by row (one row per batch element), so a batch of
 * inputs that share a circuit structure is evaluated in a single call.
 *
 * For a gate U(t) = exp(-i t G / 2), up to a global phase, the derivative of
 * E = <psi|O|psi> is dE/dt = Im <lambda|G|phi>, where phi is the state right
 * after the gate and lambda = U_{>k}^dag O psi. The backward sweep undoes
 * every gate on both states, so the cost is O(gates), independent of the
 * number of parameters.
 *
 * Gates outside the causal cone of an observable cannot change it; they are
 * dropped and the register is compacted to the cone qubits. Their parameters
 * get an exactly zero derivative.
 *
 * Gate matrices and kernels are QCSim's, so conventions match the QCSim
 * statevector simulator.
 */

#pragma once

#ifndef _ADJOINT_DIFFERENTIATION_H_
#define _ADJOINT_DIFFERENTIATION_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigen>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "../Circuit/Circuit.h"

#include "QuantumGate.h"
#include "QubitRegisterCalculator.h"
#include "SimpleGates.h"

namespace MaestroExecution::Adjoint
{

using Complex = std::complex<double>;
using State = Eigen::VectorXcd;
using Calculator = QC::QubitRegisterCalculator<>;

/** One weighted Pauli string; qubit q is character q of the string. */
using PauliTerm = std::pair<double, std::string>;

/** An observable is a real linear combination of Pauli strings. */
using Observable = std::vector<PauliTerm>;

/**
 * What to read from the final state: the expectation value of an observable
 * (one value), or the probabilities of the basis states of some qubits (2^k
 * values, the first listed qubit being the least significant bit of the
 * outcome index). Probabilities are expectation values of diagonal
 * projectors, so they are differentiated the same way.
 */
struct Readout
{
    enum class Kind : uint8_t
    {
        Expectation,
        Probabilities
    };
    Kind kind = Kind::Expectation;
    Observable observable;
    std::vector<Types::qubit_t> qubits;

    static Readout Expectation(Observable observable)
    {
        return {Kind::Expectation, std::move(observable), {}};
    }

    static Readout Probabilities(std::vector<Types::qubit_t> qubits)
    {
        return {Kind::Probabilities, {}, std::move(qubits)};
    }

    /** Number of values this readout produces. */
    size_t Columns() const;
};

// Up to this many statevector qubits, the rows of a batch run in parallel with
// serial kernels (up to three states per thread, 16 MiB each at 20 qubits);
// above it, rows run one at a time with parallel kernels.
inline constexpr size_t kRowParallelMaxQubits = 20;

// States whose qubit masks must fit in 64 bits.
inline constexpr size_t kMaxQubits = 63;

// Probabilities readouts are limited to this many qubits (2^k values per row).
inline constexpr size_t kMaxProbabilityQubits = 24;

// Validated here, not only when parsing, because callers size their output
// from it before anything else runs.
inline size_t Readout::Columns() const
{
    if (kind == Kind::Expectation)
        return 1;
    if (qubits.empty() || qubits.size() > kMaxProbabilityQubits)
        throw std::invalid_argument("Probabilities need between 1 and 24 qubits.");
    return size_t{1} << qubits.size();
}

/** Total number of values produced by a list of readouts. */
inline size_t Columns(const std::vector<Readout> &readouts)
{
    size_t n = 0;
    for (const auto &r : readouts)
        n += r.Columns();
    return n;
}

/**
 * Thrown before a call would need more memory than the machine has. It is a
 * std::bad_alloc, so Python sees a MemoryError, but carries a message.
 */
class MemoryLimitError : public std::bad_alloc
{
  public:
    explicit MemoryLimitError(std::string message) : message(std::move(message))
    {
    }

    const char *what() const noexcept override
    {
        return message.c_str();
    }

  private:
    std::string message;
};

namespace Detail
{

/** Physical memory of the machine in bytes, or 0 if unknown. */
inline double PhysicalMemoryBytes()
{
#if defined(__APPLE__)
    uint64_t bytes = 0;
    size_t len = sizeof(bytes);
    if (sysctlbyname("hw.memsize", &bytes, &len, nullptr, 0) == 0)
        return static_cast<double>(bytes);
#elif defined(__linux__)
    const long pages = sysconf(_SC_PHYS_PAGES), pageSize = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && pageSize > 0)
        return static_cast<double>(pages) * static_cast<double>(pageSize);
#endif
    return 0.;
}

inline unsigned Popcount(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_popcountll(x));
#else
    unsigned n = 0;
    for (; x; x &= x - 1)
        ++n;
    return n;
#endif
}

inline unsigned LowestBit(uint64_t x)
{
    unsigned n = 0;
    while (!(x & 1))
    {
        x >>= 1;
        ++n;
    }
    return n;
}

inline size_t BitWidth(uint64_t x)
{
    size_t n = 0;
    for (; x; x >>= 1)
        ++n;
    return n;
}

/** A Pauli string as bit masks over a register. */
struct PauliMasks
{
    uint64_t flip = 0;  // X or Y
    uint64_t phase = 0; // Z or Y
    unsigned numY = 0;
    double coeff = 0.;
};

/** Generator of one rotation inside a gate, as used by the backward sweep. */
struct Generator
{
    enum class Kind : uint8_t
    {
        X,
        Y,
        Z,
        Projector // -2 * |1..1><1..1| on target and controls, for phase gates
    };
    Kind kind = Kind::Z;
    uint64_t target = 0;   // target bit (compact register)
    uint64_t controls = 0; // control bits that must be 1
};

enum class GateKind : uint8_t
{
    Fixed,
    P,
    Rx,
    Ry,
    Rz,
    U,
    CP,
    CRx,
    CRy,
    CRz,
    CU
};

/** A circuit gate, with its differentiable parameters as slots of a row. */
struct GateSpec
{
    GateKind kind = GateKind::Fixed;
    unsigned numQubits = 0;
    // Qubits in QCSim kernel order: target first (least significant), then controls.
    std::array<Types::qubit_t, 3> kernelQubits{};
    uint64_t mask = 0; // all qubits the gate acts on (original register)
    size_t firstParam = 0;
    size_t numParams = 0;
    double gamma = 0.;      // fixed global phase of a U gate
    Eigen::MatrixXcd fixed; // matrix of a non-parameterized gate
};

inline size_t ParamCount(GateKind kind)
{
    switch (kind)
    {
    case GateKind::Fixed:
        return 0;
    case GateKind::U:
        return 3;
    case GateKind::CU:
        return 4;
    default:
        return 1;
    }
}

template <class Gate> Eigen::MatrixXcd FixedMatrix()
{
    static const Gate gate;
    return gate.getRawOperatorMatrix();
}

inline Eigen::MatrixXcd RxMatrix(double t)
{
    QC::Gates::RxGate<> g;
    g.SetTheta(t);
    return g.getRawOperatorMatrix();
}

inline Eigen::MatrixXcd RyMatrix(double t)
{
    QC::Gates::RyGate<> g;
    g.SetTheta(t);
    return g.getRawOperatorMatrix();
}

inline Eigen::MatrixXcd RzMatrix(double t)
{
    QC::Gates::RzGate<> g;
    g.SetTheta(t);
    return g.getRawOperatorMatrix();
}

inline Eigen::MatrixXcd PMatrix(double t)
{
    QC::Gates::PhaseShiftGate<> g;
    g.SetPhaseShift(t);
    return g.getRawOperatorMatrix();
}

/** Controlled version of a one-qubit matrix; the control is the high bit. */
inline Eigen::MatrixXcd Controlled(const Eigen::MatrixXcd &m)
{
    Eigen::MatrixXcd c = Eigen::MatrixXcd::Identity(4, 4);
    c.block(2, 2, 2, 2) = m;
    return c;
}

/** Full matrix of a gate for one parameter row. */
inline Eigen::MatrixXcd GateMatrix(const GateSpec &g, const double *p)
{
    switch (g.kind)
    {
    case GateKind::Fixed:
        return g.fixed;
    case GateKind::P:
        return PMatrix(p[0]);
    case GateKind::Rx:
        return RxMatrix(p[0]);
    case GateKind::Ry:
        return RyMatrix(p[0]);
    case GateKind::Rz:
        return RzMatrix(p[0]);
    case GateKind::U: {
        QC::Gates::UGate<> u;
        u.SetParams(p[0], p[1], p[2], g.gamma);
        return u.getRawOperatorMatrix();
    }
    case GateKind::CP:
        return Controlled(PMatrix(p[0]));
    case GateKind::CRx:
        return Controlled(RxMatrix(p[0]));
    case GateKind::CRy:
        return Controlled(RyMatrix(p[0]));
    case GateKind::CRz:
        return Controlled(RzMatrix(p[0]));
    case GateKind::CU: {
        QC::Gates::ControlledUGate<> u;
        u.SetParams(p[0], p[1], p[2], p[3]);
        return u.getRawOperatorMatrix();
    }
    }
    throw std::logic_error("Unknown gate kind");
}

/** Sum f(i) over the basis states, in parallel for large states. */
template <class F> Complex Reduce(size_t count, bool parallel, F f)
{
    double re = 0., im = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : re, im) schedule(static) if (parallel)
#endif
    for (long long i = 0; i < static_cast<long long>(count); ++i)
    {
        const Complex v = f(static_cast<uint64_t>(i));
        re += v.real();
        im += v.imag();
    }
    (void)parallel;
    return {re, im};
}

/** <lambda|G|phi> for a rotation generator. */
inline Complex GeneratorElement(const Generator &g, const State &lambda, const State &phi, bool parallel)
{
    const size_t count = static_cast<size_t>(phi.size());
    const Complex *l = lambda.data();
    const Complex *f = phi.data();
    const uint64_t c = g.controls, t = g.target;
    switch (g.kind)
    {
    case Generator::Kind::X:
        return Reduce(count, parallel, [=](uint64_t i) { return (i & c) == c ? std::conj(l[i]) * f[i ^ t] : Complex(0.); });
    case Generator::Kind::Y:
        return Reduce(count, parallel, [=](uint64_t i) {
            if ((i & c) != c)
                return Complex(0.);
            const Complex v = std::conj(l[i]) * f[i ^ t];
            return (i & t) ? Complex(-v.imag(), v.real()) : Complex(v.imag(), -v.real());
        });
    case Generator::Kind::Z:
        return Reduce(count, parallel, [=](uint64_t i) {
            if ((i & c) != c)
                return Complex(0.);
            const Complex v = std::conj(l[i]) * f[i];
            return (i & t) ? -v : v;
        });
    case Generator::Kind::Projector: {
        const uint64_t m = c | t;
        return Reduce(count, parallel, [=](uint64_t i) { return (i & m) == m ? -2. * std::conj(l[i]) * f[i] : Complex(0.); });
    }
    }
    throw std::logic_error("Unknown generator kind");
}

/** out += coeff * P |in> for a Pauli string P. */
inline void AddPauli(const PauliMasks &p, double coeff, const State &in, State &out, bool parallel)
{
    static const Complex iPow[4] = {{1., 0.}, {0., 1.}, {-1., 0.}, {0., -1.}};
    const Complex c = coeff * iPow[p.numY & 3];
    const uint64_t flip = p.flip, phase = p.phase;
    const long long count = static_cast<long long>(in.size());
    const Complex *src = in.data();
    Complex *dst = out.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (long long i = 0; i < count; ++i)
    {
        const uint64_t j = static_cast<uint64_t>(i) ^ flip;
        const Complex v = c * src[j];
        dst[i] += (Popcount(j & phase) & 1) ? -v : v;
    }
}

/** <psi|P|psi> for a Pauli string P. */
inline double PauliExpectation(const PauliMasks &p, const State &psi, bool parallel)
{
    static const Complex iPow[4] = {{1., 0.}, {0., 1.}, {-1., 0.}, {0., -1.}};
    const uint64_t flip = p.flip, phase = p.phase;
    const Complex *s = psi.data();
    const Complex sum = Reduce(static_cast<size_t>(psi.size()), parallel, [=](uint64_t i) {
        const uint64_t j = i ^ flip;
        const Complex v = std::conj(s[i]) * s[j];
        return (Popcount(j & phase) & 1) ? -v : v;
    });
    return (iPow[p.numY & 3] * sum).real();
}

/** Index of the outcome of the qubits at `bits` (bits[0] least significant). */
inline size_t Outcome(uint64_t i, const std::vector<uint64_t> &bits)
{
    size_t o = 0;
    for (size_t j = 0; j < bits.size(); ++j)
        if (i & bits[j])
            o |= size_t{1} << j;
    return o;
}

// Up to this many outcomes, Marginal sums into one small histogram per thread;
// above it, each thread owns whole outcomes, so no histogram grows with 2^k.
inline constexpr size_t kMarginalHistogramMaxOutcomes = 4096;

/** out[o] = probability of outcome o of the qubits at `bits`. */
inline void Marginal(const State &psi, const std::vector<uint64_t> &bits, double *out, bool parallel)
{
    const size_t numOutcomes = size_t{1} << bits.size();
    const long long count = static_cast<long long>(psi.size());
    const Complex *s = psi.data();
    if (numOutcomes > kMarginalHistogramMaxOutcomes)
    {
        uint64_t measured = 0;
        for (const auto bit : bits)
            measured |= bit;
        const uint64_t rest = (static_cast<uint64_t>(count) - 1) & ~measured;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
        for (long long o = 0; o < static_cast<long long>(numOutcomes); ++o)
        {
            uint64_t base = 0;
            for (size_t j = 0; j < bits.size(); ++j)
                if ((static_cast<uint64_t>(o) >> j) & 1)
                    base |= bits[j];
            // Visit every subset of the unmeasured bits.
            double p = 0.;
            uint64_t sub = 0;
            do
            {
                p += std::norm(s[base | sub]);
                sub = (sub - rest) & rest;
            } while (sub);
            out[o] = p;
        }
        (void)parallel;
        return;
    }

    int threads = 1;
#ifdef _OPENMP
    if (parallel)
        threads = omp_get_max_threads();
#endif
    // Allocated up front: an exception must not escape a parallel region.
    std::vector<double> local(static_cast<size_t>(threads) * numOutcomes, 0.);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if (parallel)
#endif
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        double *mine = local.data() + static_cast<size_t>(t) * numOutcomes;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (long long i = 0; i < count; ++i)
            mine[Outcome(static_cast<uint64_t>(i), bits)] += std::norm(s[i]);
    }
    std::fill(out, out + numOutcomes, 0.);
    for (int t = 0; t < threads; ++t)
        for (size_t o = 0; o < numOutcomes; ++o)
            out[o] += local[static_cast<size_t>(t) * numOutcomes + o];
}

/** out[i] += weights[outcome(i)] * in[i]: a weighted sum of projectors. */
inline void AddProjectors(const std::vector<uint64_t> &bits, const double *weights, const State &in, State &out, bool parallel)
{
    const long long count = static_cast<long long>(in.size());
    const Complex *src = in.data();
    Complex *dst = out.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
    for (long long i = 0; i < count; ++i)
        dst[i] += weights[Outcome(static_cast<uint64_t>(i), bits)] * src[i];
    (void)parallel;
}

/**
 * M(a, b) = sum over the other qubits of conj(lambda[a]) phi[b], with a, b the
 * value of `bit`. Then <lambda|A|phi> = sum_ab A(a, b) M(a, b) for any
 * one-qubit operator A on that qubit.
 */
inline Eigen::Matrix2cd CrossMatrix(const State &lambda, const State &phi, uint64_t bit, bool parallel)
{
    const long long half = static_cast<long long>(phi.size() / 2);
    const Complex *l = lambda.data();
    const Complex *f = phi.data();
    const uint64_t low = bit - 1;
    double r00 = 0., i00 = 0., r01 = 0., i01 = 0., r10 = 0., i10 = 0., r11 = 0., i11 = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : r00, i00, r01, i01, r10, i10, r11, i11) schedule(static) if (parallel)
#endif
    for (long long k = 0; k < half; ++k)
    {
        const uint64_t i0 = ((static_cast<uint64_t>(k) & ~low) << 1) | (static_cast<uint64_t>(k) & low);
        const uint64_t i1 = i0 | bit;
        const Complex l0 = std::conj(l[i0]), l1 = std::conj(l[i1]);
        const Complex m00 = l0 * f[i0], m01 = l0 * f[i1], m10 = l1 * f[i0], m11 = l1 * f[i1];
        r00 += m00.real();
        i00 += m00.imag();
        r01 += m01.real();
        i01 += m01.imag();
        r10 += m10.real();
        i10 += m10.imag();
        r11 += m11.real();
        i11 += m11.imag();
    }
    (void)parallel;
    Eigen::Matrix2cd m;
    m << Complex(r00, i00), Complex(r01, i01), Complex(r10, i10), Complex(r11, i11);
    return m;
}

/**
 * A step of a cone program: one gate, or a run of one-qubit gates on the same
 * qubit with no other gate on that qubit in between. A run is applied as one
 * fused matrix and differentiated from one CrossMatrix pass.
 */
struct Block
{
    bool run = false;
    std::vector<size_t> gates; // indices into Program::gates, in circuit order
};

/** The gates of a circuit that lie in a causal cone, on a compact register. */
struct ConeProgram
{
    std::vector<size_t> gates; // indices into Program::gates, in circuit order
    std::vector<Block> blocks; // the same gates, with one-qubit runs fused
    std::vector<int> compact;  // original qubit -> compact qubit, -1 outside
    size_t numQubits = 0;
};

} // namespace Detail

/**
 * @class Program
 * @brief A unitary circuit compiled for batched expectation values and
 * adjoint derivatives.
 *
 * The differentiable parameters, in circuit order, are: one angle for each
 * P, Rx, Ry, Rz, CP, CRx, CRy and CRz gate; theta, phi, lambda for U
 * (its gamma stays fixed); theta, phi, lambda, gamma for CU. Delays are
 * ignored; any other non-gate operation is rejected.
 */
class Program
{
  public:
    explicit Program(const Circuits::Circuit<double> &circuit)
    {
        using Circuits::QuantumGateType;
        using namespace Detail;
        for (const auto &op : circuit.GetOperations())
        {
            const auto type = op->GetType();
            if (type == Circuits::OperationType::kDelay || type == Circuits::OperationType::kNoOp)
                continue;
            if (type != Circuits::OperationType::kGate)
                throw std::invalid_argument("Adjoint differentiation requires a unitary circuit; found a non-gate operation.");
            const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
            if (!gate)
                throw std::invalid_argument("Adjoint differentiation: unsupported gate operation.");

            GateSpec spec;
            spec.numQubits = gate->GetNumQubits();
            std::array<Types::qubit_t, 3> q{};
            for (unsigned i = 0; i < spec.numQubits; ++i)
            {
                q[i] = gate->GetQubit(i);
                numQubits = std::max(numQubits, static_cast<size_t>(q[i]) + 1);
                spec.mask |= uint64_t{1} << q[i];
            }
            // Maestro lists controls first; QCSim kernels take the target first.
            if (spec.numQubits == 1)
                spec.kernelQubits = {q[0], 0, 0};
            else if (spec.numQubits == 2)
                spec.kernelQubits = {q[1], q[0], 0};
            else
                spec.kernelQubits = {q[2], q[1], q[0]};

            const auto params = gate->GetParams();
            switch (gate->GetGateType())
            {
            case QuantumGateType::kPhaseGateType:
                spec.kind = GateKind::P;
                break;
            case QuantumGateType::kRxGateType:
                spec.kind = GateKind::Rx;
                break;
            case QuantumGateType::kRyGateType:
                spec.kind = GateKind::Ry;
                break;
            case QuantumGateType::kRzGateType:
                spec.kind = GateKind::Rz;
                break;
            case QuantumGateType::kUGateType:
                spec.kind = GateKind::U;
                spec.gamma = params.size() > 3 ? params[3] : 0.;
                break;
            case QuantumGateType::kCPGateType:
                spec.kind = GateKind::CP;
                break;
            case QuantumGateType::kCRxGateType:
                spec.kind = GateKind::CRx;
                break;
            case QuantumGateType::kCRyGateType:
                spec.kind = GateKind::CRy;
                break;
            case QuantumGateType::kCRzGateType:
                spec.kind = GateKind::CRz;
                break;
            case QuantumGateType::kCUGateType:
                spec.kind = GateKind::CU;
                break;
            case QuantumGateType::kXGateType:
                spec.fixed = FixedMatrix<QC::Gates::PauliXGate<>>();
                break;
            case QuantumGateType::kYGateType:
                spec.fixed = FixedMatrix<QC::Gates::PauliYGate<>>();
                break;
            case QuantumGateType::kZGateType:
                spec.fixed = FixedMatrix<QC::Gates::PauliZGate<>>();
                break;
            case QuantumGateType::kHadamardGateType:
                spec.fixed = FixedMatrix<QC::Gates::HadamardGate<>>();
                break;
            case QuantumGateType::kSGateType:
                spec.fixed = FixedMatrix<QC::Gates::SGate<>>();
                break;
            case QuantumGateType::kSdgGateType:
                spec.fixed = FixedMatrix<QC::Gates::SDGGate<>>();
                break;
            case QuantumGateType::kTGateType:
                spec.fixed = FixedMatrix<QC::Gates::TGate<>>();
                break;
            case QuantumGateType::kTdgGateType:
                spec.fixed = FixedMatrix<QC::Gates::TDGGate<>>();
                break;
            case QuantumGateType::kSxGateType:
                spec.fixed = FixedMatrix<QC::Gates::SquareRootNOTGate<>>();
                break;
            case QuantumGateType::kSxDagGateType:
                spec.fixed = FixedMatrix<QC::Gates::SquareRootNOTDagGate<>>();
                break;
            case QuantumGateType::kKGateType:
                spec.fixed = FixedMatrix<QC::Gates::HyGate<>>();
                break;
            case QuantumGateType::kSwapGateType:
                spec.fixed = FixedMatrix<QC::Gates::SwapGate<>>();
                break;
            case QuantumGateType::kCXGateType:
                spec.fixed = FixedMatrix<QC::Gates::CNOTGate<>>();
                break;
            case QuantumGateType::kCYGateType:
                spec.fixed = FixedMatrix<QC::Gates::ControlledYGate<>>();
                break;
            case QuantumGateType::kCZGateType:
                spec.fixed = FixedMatrix<QC::Gates::ControlledZGate<>>();
                break;
            case QuantumGateType::kCHGateType:
                spec.fixed = FixedMatrix<QC::Gates::ControlledHadamardGate<>>();
                break;
            case QuantumGateType::kCSxGateType:
                spec.fixed = FixedMatrix<QC::Gates::ControlledSquareRootNOTGate<>>();
                break;
            case QuantumGateType::kCSxDagGateType:
                spec.fixed = FixedMatrix<QC::Gates::ControlledSquareRootNOTDagGate<>>();
                break;
            case QuantumGateType::kCCXGateType:
                spec.fixed = FixedMatrix<QC::Gates::ToffoliGate<>>();
                break;
            case QuantumGateType::kCSwapGateType:
                spec.fixed = FixedMatrix<QC::Gates::FredkinGate<>>();
                break;
            default:
                throw std::invalid_argument("Adjoint differentiation: unsupported gate type.");
            }

            spec.firstParam = defaults.size();
            spec.numParams = ParamCount(spec.kind);
            for (size_t i = 0; i < spec.numParams; ++i)
                defaults.push_back(i < params.size() ? params[i] : 0.);
            gates.push_back(std::move(spec));
        }
        if (numQubits > kMaxQubits)
            throw std::invalid_argument("Adjoint differentiation supports at most 63 qubits.");
    }

    /** Number of differentiable parameters (columns of a parameter row). */
    size_t NumParams() const
    {
        return defaults.size();
    }

    /** The parameter values stored in the circuit. */
    const std::vector<double> &DefaultParams() const
    {
        return defaults;
    }

    size_t NumQubits() const
    {
        return numQubits;
    }

    /**
     * The values of every readout for each row: out[row * nCols + c], with the
     * readouts' values side by side in order. All readouts are evaluated on
     * one statevector (their joint cone).
     */
    std::vector<double> Values(const std::vector<Readout> &readouts, const std::vector<double> &rows, size_t numRows, bool causalCone) const
    {
        const auto parsed = ParseAll(readouts);
        const auto cone = Cone(Support(parsed), causalCone);
        const auto compact = CompactAll(parsed, cone);
        const size_t nCols = Columns(readouts);
        CheckMemory(numRows, cone.numQubits, 1, static_cast<double>(numRows) * nCols);

        std::vector<double> out(numRows * nCols, 0.);
        ForRows(numRows, cone.numQubits, [&](size_t row, bool parallel) {
            const State psi = Forward(cone, rows.data() + row * NumParams(), parallel);
            WriteValues(compact, psi, out.data() + row * nCols, parallel);
        });
        return out;
    }

    /**
     * Jacobian of every value with respect to every parameter, for each row:
     * jac[(row * nCols + c) * nParams + k]; values as in Values. Each readout
     * gets its own causal cone and one backward sweep per value.
     */
    void Jacobian(const std::vector<Readout> &readouts, const std::vector<double> &rows, size_t numRows, bool causalCone, std::vector<double> &values,
                  std::vector<double> &jac) const
    {
        const size_t nCols = Columns(readouts), nParams = NumParams();
        const auto parsed = ParseAll(readouts);
        std::vector<Detail::ConeProgram> cones;
        cones.reserve(parsed.size());
        size_t widest = 0;
        for (const auto &r : parsed)
        {
            cones.push_back(Cone(r.support, causalCone));
            widest = std::max(widest, cones.back().numQubits);
        }
        // psi, plus phi and lambda for the sweep in progress.
        CheckMemory(numRows, widest, 3, static_cast<double>(numRows) * nCols * (nParams + 1));

        values.assign(numRows * nCols, 0.);
        jac.assign(numRows * nCols * nParams, 0.);
        size_t first = 0;
        for (size_t i = 0; i < parsed.size(); ++i)
        {
            const auto &cone = cones[i];
            const auto compact = Compact(parsed[i], cone);
            const size_t width = compact.columns;
            ForRows(numRows, cone.numQubits, [&](size_t row, bool parallel) {
                const double *p = rows.data() + row * nParams;
                const State psi = Forward(cone, p, parallel);
                WriteValues({compact}, psi, values.data() + row * nCols + first, parallel);
                std::vector<double> weights(width, 0.);
                for (size_t c = 0; c < width; ++c)
                {
                    weights[c] = 1.;
                    State phi = psi;
                    State lambda = State::Zero(phi.size());
                    AddReadout(compact, weights.data(), phi, lambda, parallel);
                    weights[c] = 0.;
                    Backward(cone, p, phi, lambda, jac.data() + (row * nCols + first + c) * nParams, parallel);
                }
            });
            first += width;
        }
    }

    /**
     * Vector-Jacobian product for each row: vjp[row * nParams + k] =
     * sum_c cotangents[row * nCols + c] * d value_c / dp_k, in one backward
     * sweep for all readouts.
     */
    void VectorJacobianProduct(const std::vector<Readout> &readouts, const std::vector<double> &cotangents, const std::vector<double> &rows, size_t numRows,
                               bool causalCone, std::vector<double> &values, std::vector<double> &vjp) const
    {
        const auto parsed = ParseAll(readouts);
        const auto cone = Cone(Support(parsed), causalCone);
        const auto compact = CompactAll(parsed, cone);
        const size_t nCols = Columns(readouts), nParams = NumParams();
        CheckMemory(numRows, cone.numQubits, 2, static_cast<double>(numRows) * (nCols + nParams));
        values.assign(numRows * nCols, 0.);
        vjp.assign(numRows * nParams, 0.);
        ForRows(numRows, cone.numQubits, [&](size_t row, bool parallel) {
            const double *p = rows.data() + row * nParams;
            State phi = Forward(cone, p, parallel);
            State lambda = State::Zero(phi.size());
            WriteValues(compact, phi, values.data() + row * nCols, parallel);
            const double *dy = cotangents.data() + row * nCols;
            for (const auto &r : compact)
            {
                AddReadout(r, dy, phi, lambda, parallel);
                dy += r.columns;
            }
            Backward(cone, p, phi, lambda, vjp.data() + row * nParams, parallel);
        });
    }

  private:
    using Masks = std::vector<Detail::PauliMasks>;

    /** A readout as masks and bits; on a compact register after Compact. */
    struct ParsedReadout
    {
        Readout::Kind kind = Readout::Kind::Expectation;
        Masks terms;                // Expectation
        std::vector<uint64_t> bits; // Probabilities: one bit per qubit, in order
        uint64_t support = 0;
        size_t columns = 1;
    };

    ParsedReadout Parse(const Readout &readout) const
    {
        ParsedReadout out;
        out.kind = readout.kind;
        out.columns = readout.Columns();
        if (readout.kind == Readout::Kind::Expectation)
        {
            out.terms = ParseObservable(readout.observable);
            for (const auto &term : out.terms)
                out.support |= term.flip | term.phase;
            return out;
        }
        for (const auto q : readout.qubits)
        {
            if (q >= kMaxQubits)
                throw std::invalid_argument("Probabilities qubit index is out of range.");
            const uint64_t bit = uint64_t{1} << q;
            if (out.support & bit)
                throw std::invalid_argument("Probabilities qubits must be distinct.");
            out.support |= bit;
            out.bits.push_back(bit);
        }
        return out;
    }

    std::vector<ParsedReadout> ParseAll(const std::vector<Readout> &readouts) const
    {
        std::vector<ParsedReadout> out;
        out.reserve(readouts.size());
        for (const auto &r : readouts)
            out.push_back(Parse(r));
        return out;
    }

    static uint64_t Support(const std::vector<ParsedReadout> &parsed)
    {
        uint64_t support = 0;
        for (const auto &r : parsed)
            support |= r.support;
        return support;
    }

    Masks ParseObservable(const Observable &observable) const
    {
        Masks out;
        for (const auto &[coeff, pauli] : observable)
        {
            Detail::PauliMasks m;
            m.coeff = coeff;
            for (size_t q = 0; q < pauli.size(); ++q)
            {
                const char c = pauli[q];
                if (c == 'I' || c == 'i')
                    continue;
                if (q >= kMaxQubits)
                    throw std::invalid_argument("Pauli string is longer than 63 qubits.");
                const uint64_t bit = uint64_t{1} << q;
                if (c == 'X' || c == 'x')
                    m.flip |= bit;
                else if (c == 'Z' || c == 'z')
                    m.phase |= bit;
                else if (c == 'Y' || c == 'y')
                {
                    m.flip |= bit;
                    m.phase |= bit;
                    ++m.numY;
                }
                else
                    throw std::invalid_argument("Invalid Pauli character '" + std::string(1, c) + "' in observable.");
            }
            out.push_back(m);
        }
        return out;
    }

    /** Backward light cone of the qubits in `support`. */
    Detail::ConeProgram Cone(uint64_t support, bool causalCone) const
    {
        Detail::ConeProgram cone;
        uint64_t qubits = 0;
        if (causalCone)
        {
            qubits = support;
            for (size_t k = gates.size(); k-- > 0;)
                if (gates[k].mask & qubits)
                {
                    qubits |= gates[k].mask;
                    cone.gates.push_back(k);
                }
            std::reverse(cone.gates.begin(), cone.gates.end());
        }
        else
        {
            for (size_t k = 0; k < gates.size(); ++k)
            {
                qubits |= gates[k].mask;
                cone.gates.push_back(k);
            }
            qubits |= support;
        }
        const size_t width = std::max(numQubits, Detail::BitWidth(support));
        cone.compact.assign(width, -1);
        for (size_t q = 0; q < width; ++q)
            if (qubits & (uint64_t{1} << q))
                cone.compact[q] = static_cast<int>(cone.numQubits++);

        // Gates on other qubits commute with a pending run, so each run is
        // emitted just before the next multi-qubit gate on its qubit.
        std::vector<std::vector<size_t>> pending(width);
        auto flush = [&](size_t q) {
            if (pending[q].empty())
                return;
            cone.blocks.push_back({true, std::move(pending[q])});
            pending[q].clear();
        };
        for (const size_t k : cone.gates)
        {
            const auto &g = gates[k];
            if (g.numQubits == 1)
            {
                pending[g.kernelQubits[0]].push_back(k);
                continue;
            }
            for (unsigned i = 0; i < g.numQubits; ++i)
                flush(g.kernelQubits[i]);
            cone.blocks.push_back({false, {k}});
        }
        for (size_t q = 0; q < width; ++q)
            flush(q);
        return cone;
    }

    static uint64_t CompactMask(uint64_t mask, const Detail::ConeProgram &cone)
    {
        uint64_t out = 0;
        for (; mask; mask &= mask - 1)
        {
            const int q = cone.compact[Detail::LowestBit(mask)];
            out |= uint64_t{1} << q;
        }
        return out;
    }

    static ParsedReadout Compact(const ParsedReadout &readout, const Detail::ConeProgram &cone)
    {
        ParsedReadout out = readout;
        for (auto &m : out.terms)
        {
            m.flip = CompactMask(m.flip, cone);
            m.phase = CompactMask(m.phase, cone);
        }
        for (auto &bit : out.bits)
            bit = CompactMask(bit, cone);
        out.support = CompactMask(out.support, cone);
        return out;
    }

    static std::vector<ParsedReadout> CompactAll(const std::vector<ParsedReadout> &all, const Detail::ConeProgram &cone)
    {
        std::vector<ParsedReadout> out;
        out.reserve(all.size());
        for (const auto &r : all)
            out.push_back(Compact(r, cone));
        return out;
    }

    /** Write the values of compacted readouts, side by side. */
    static void WriteValues(const std::vector<ParsedReadout> &readouts, const State &psi, double *out, bool parallel)
    {
        for (const auto &r : readouts)
        {
            if (r.kind == Readout::Kind::Expectation)
                *out = Expectation(r.terms, psi, parallel);
            else
                Detail::Marginal(psi, r.bits, out, parallel);
            out += r.columns;
        }
    }

    /** lambda += sum_c weights[c] * (operator of value c) phi. */
    static void AddReadout(const ParsedReadout &r, const double *weights, const State &phi, State &lambda, bool parallel)
    {
        if (r.kind == Readout::Kind::Probabilities)
        {
            Detail::AddProjectors(r.bits, weights, phi, lambda, parallel);
            return;
        }
        if (weights[0] != 0.)
            for (const auto &term : r.terms)
                Detail::AddPauli(term, weights[0] * term.coeff, phi, lambda, parallel);
    }

    static double Expectation(const Masks &obs, const State &psi, bool parallel)
    {
        double value = 0.;
        for (const auto &term : obs)
            value += term.coeff * (term.flip || term.phase ? Detail::PauliExpectation(term, psi, parallel) : psi.squaredNorm());
        return value;
    }

    static bool KernelsParallel(size_t coneQubits)
    {
        return (size_t{1} << coneQubits) >= QC::ParallelMinBasisStatesSetting().load();
    }

    /** Number of rows that run at once: all threads for small registers, else one. */
    static size_t ConcurrentRows(size_t numRows, size_t coneQubits)
    {
#ifdef _OPENMP
        const size_t threads = static_cast<size_t>(omp_get_max_threads());
        if (numRows <= 1 || coneQubits > kRowParallelMaxQubits)
            return 1;
        // Registers big enough for parallel kernels only go row-parallel when
        // the rows can keep every thread busy.
        if (KernelsParallel(coneQubits) && numRows < threads)
            return 1;
        return std::min(numRows, threads);
#else
        (void)numRows;
        (void)coneQubits;
        return 1;
#endif
    }

    /**
     * Refuse a call whose states and outputs would not fit in physical memory,
     * rather than let the allocation swap the machine or get it OOM-killed.
     */
    static void CheckMemory(size_t numRows, size_t coneQubits, size_t statesPerRow, double outputValues)
    {
        const double stateBytes = std::ldexp(static_cast<double>(sizeof(Complex)), static_cast<int>(coneQubits));
        const double rows = static_cast<double>(ConcurrentRows(numRows, coneQubits));
        const double needed = rows * statesPerRow * stateBytes + outputValues * sizeof(double);
        double limit = Detail::PhysicalMemoryBytes();
        // Unknown machine: still refuse sizes that cannot even be indexed.
        if (limit <= 0.)
            limit = static_cast<double>(std::numeric_limits<Eigen::Index>::max());
        if (needed <= limit)
            return;
        auto gib = [](double bytes) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.3g", bytes / (1024. * 1024. * 1024.));
            return std::string(buf);
        };
        throw MemoryLimitError("Adjoint differentiation needs about " + gib(needed) + " GiB (" + std::to_string(statesPerRow) + " statevectors of " +
                               std::to_string(coneQubits) + " qubits per concurrent row), more than the " + gib(limit) +
                               " GiB of physical memory. Reduce the qubits in the causal cone or the batch size.");
    }

    /** Run rows in parallel for small registers, else kernels in parallel. */
    template <class F> static void ForRows(size_t numRows, size_t coneQubits, F f)
    {
#ifdef _OPENMP
        if (ConcurrentRows(numRows, coneQubits) > 1)
        {
            // An exception must not escape the parallel region (that would
            // terminate); the first one is rethrown after it.
            std::exception_ptr error;
#pragma omp parallel for schedule(dynamic)
            for (long long row = 0; row < static_cast<long long>(numRows); ++row)
            {
                try
                {
                    f(static_cast<size_t>(row), false);
                }
                catch (...)
                {
#pragma omp critical(adjoint_row_error)
                    if (!error)
                        error = std::current_exception();
                }
            }
            if (error)
                std::rethrow_exception(error);
            return;
        }
#endif
        const bool kernelsParallel = KernelsParallel(coneQubits);
        for (size_t row = 0; row < numRows; ++row)
            f(row, kernelsParallel);
    }

    void Apply(const Detail::GateSpec &g, const Eigen::MatrixXcd &m, const Detail::ConeProgram &cone, State &state, bool parallel) const
    {
        std::array<size_t, 3> bits{0, 0, 0};
        for (unsigned i = 0; i < g.numQubits; ++i)
            bits[i] = size_t{1} << cone.compact[g.kernelQubits[i]];
        Calculator::ApplyGateInPlace(state, m, QC::Gates::ClassifyGateMatrix(m), bits, g.numQubits, static_cast<size_t>(state.size()), parallel);
    }

    /** Product of the matrices of a one-qubit run, in application order. */
    Eigen::MatrixXcd RunMatrix(const Detail::Block &block, const double *p) const
    {
        Eigen::MatrixXcd m = Eigen::MatrixXcd::Identity(2, 2);
        for (const size_t k : block.gates)
            m = Detail::GateMatrix(gates[k], p + gates[k].firstParam) * m;
        return m;
    }

    State Forward(const Detail::ConeProgram &cone, const double *p, bool parallel) const
    {
        State psi = State::Zero(size_t{1} << cone.numQubits);
        psi(0) = 1.;
        for (const auto &block : cone.blocks)
        {
            const auto &g = gates[block.gates.front()];
            Apply(g, block.run ? RunMatrix(block, p) : Detail::GateMatrix(g, p + g.firstParam), cone, psi, parallel);
        }
        return psi;
    }

    /**
     * Derivatives of a one-qubit run from the states after it. For a rotation
     * with generator G followed by V (the rest of the run), the derivative is
     * Im <lambda|V G V^dag|phi> = Im sum_ab (V G V^dag)(a, b) M(a, b).
     */
    void BackwardRun(const Detail::Block &block, const Detail::ConeProgram &cone, const double *p, State &phi, State &lambda, double *grad, bool parallel) const
    {
        using Detail::GateKind;
        const auto &first = gates[block.gates.front()];
        const uint64_t bit = uint64_t{1} << cone.compact[first.kernelQubits[0]];

        static const Eigen::Matrix2cd X = (Eigen::Matrix2cd() << 0., 1., 1., 0.).finished();
        static const Eigen::Matrix2cd Y = (Eigen::Matrix2cd() << 0., Complex(0., -1.), Complex(0., 1.), 0.).finished();
        static const Eigen::Matrix2cd Z = (Eigen::Matrix2cd() << 1., 0., 0., -1.).finished();
        static const Eigen::Matrix2cd Proj = (Eigen::Matrix2cd() << 0., 0., 0., -2.).finished();

        bool anyParams = false;
        for (const size_t k : block.gates)
            anyParams |= gates[k].numParams > 0;
        const Eigen::MatrixXcd total = RunMatrix(block, p);
        if (anyParams)
        {
            const Eigen::Matrix2cd M = Detail::CrossMatrix(lambda, phi, bit, parallel);
            Eigen::Matrix2cd V = Eigen::Matrix2cd::Identity();
            auto im = [&](const Eigen::Matrix2cd &G) { return (V * G * V.adjoint()).cwiseProduct(M).sum().imag(); };
            for (size_t idx = block.gates.size(); idx-- > 0;)
            {
                const auto &g = gates[block.gates[idx]];
                const double *gp = p + g.firstParam;
                double *gg = grad + g.firstParam;
                switch (g.kind)
                {
                case GateKind::P:
                    gg[0] += im(Proj);
                    break;
                case GateKind::Rx:
                    gg[0] += im(X);
                    break;
                case GateKind::Ry:
                    gg[0] += im(Y);
                    break;
                case GateKind::Rz:
                    gg[0] += im(Z);
                    break;
                case GateKind::U:
                    // U = e^{i(f + l)/2} Rz(f) Ry(t) Rz(l); the phase cancels in V G V^dag.
                    gg[1] += im(Z);
                    V = V * Detail::RzMatrix(gp[1]);
                    gg[0] += im(Y);
                    V = V * Detail::RyMatrix(gp[0]);
                    gg[2] += im(Z);
                    V = V * Detail::RzMatrix(gp[2]);
                    continue;
                default:
                    break;
                }
                V = V * Detail::GateMatrix(g, gp);
            }
        }
        Undo(first, total, cone, phi, lambda, parallel);
    }

    /** Undo a matrix on both states. */
    void Undo(const Detail::GateSpec &g, const Eigen::MatrixXcd &m, const Detail::ConeProgram &cone, State &phi, State &lambda, bool parallel) const
    {
        const Eigen::MatrixXcd inverse = m.adjoint();
        Apply(g, inverse, cone, phi, parallel);
        Apply(g, inverse, cone, lambda, parallel);
    }

    /**
     * Backward sweep from phi = U|0>, lambda = O phi. Adds dE/dp_k into
     * grad[k] for the parameters of every gate in the cone.
     */
    void Backward(const Detail::ConeProgram &cone, const double *p, State &phi, State &lambda, double *grad, bool parallel) const
    {
        using Detail::GateKind;
        using Detail::Generator;
        for (size_t idx = cone.blocks.size(); idx-- > 0;)
        {
            const auto &block = cone.blocks[idx];
            if (block.run)
            {
                BackwardRun(block, cone, p, phi, lambda, grad, parallel);
                continue;
            }
            const auto &g = gates[block.gates.front()];
            const double *gp = p + g.firstParam;
            double *gg = grad + g.firstParam;
            const uint64_t target = uint64_t{1} << cone.compact[g.kernelQubits[0]];
            const uint64_t control = g.numQubits > 1 ? uint64_t{1} << cone.compact[g.kernelQubits[1]] : 0;
            auto im = [&](Generator::Kind kind, uint64_t controls = 0) {
                return Detail::GeneratorElement({kind, target, controls}, lambda, phi, parallel).imag();
            };
            // Undo a rotation-shaped piece of the gate, embedded on its qubits.
            auto undoPart = [&](const Eigen::MatrixXcd &m) { Undo(g, g.numQubits > 1 ? Detail::Controlled(m) : m, cone, phi, lambda, parallel); };

            switch (g.kind)
            {
            case GateKind::Fixed:
                Undo(g, g.fixed, cone, phi, lambda, parallel);
                break;
            case GateKind::P:
            case GateKind::CP:
                gg[0] += im(Generator::Kind::Projector, control);
                Undo(g, Detail::GateMatrix(g, gp), cone, phi, lambda, parallel);
                break;
            case GateKind::Rx:
            case GateKind::CRx:
                gg[0] += im(Generator::Kind::X, control);
                Undo(g, Detail::GateMatrix(g, gp), cone, phi, lambda, parallel);
                break;
            case GateKind::Ry:
            case GateKind::CRy:
                gg[0] += im(Generator::Kind::Y, control);
                Undo(g, Detail::GateMatrix(g, gp), cone, phi, lambda, parallel);
                break;
            case GateKind::Rz:
            case GateKind::CRz:
                gg[0] += im(Generator::Kind::Z, control);
                Undo(g, Detail::GateMatrix(g, gp), cone, phi, lambda, parallel);
                break;
            case GateKind::U:
            case GateKind::CU: {
                // U(t, f, l) = e^{i(f + l)/2} Rz(f) Ry(t) Rz(l), up to gamma. A
                // phase common to both states cancels in every overlap, so the
                // pieces are undone without it. CU adds the controlled phase
                // P(gamma + (f + l)/2) on the control qubit, which is not global
                // and so is undone explicitly.
                if (g.kind == GateKind::CU)
                {
                    const double d = Detail::GeneratorElement({Generator::Kind::Projector, control, 0}, lambda, phi, parallel).imag();
                    gg[3] += d;
                    gg[1] += 0.5 * d;
                    gg[2] += 0.5 * d;
                    const Complex phase = std::polar(1., gp[3] + 0.5 * (gp[1] + gp[2]));
                    undoPart(phase * Eigen::MatrixXcd::Identity(2, 2));
                }
                gg[1] += im(Generator::Kind::Z, control);
                undoPart(Detail::RzMatrix(gp[1]));
                gg[0] += im(Generator::Kind::Y, control);
                undoPart(Detail::RyMatrix(gp[0]));
                gg[2] += im(Generator::Kind::Z, control);
                undoPart(Detail::RzMatrix(gp[2]));
                break;
            }
            }
        }
    }

    std::vector<Detail::GateSpec> gates;
    std::vector<double> defaults;
    size_t numQubits = 0;
};

} // namespace MaestroExecution::Adjoint

#endif // _ADJOINT_DIFFERENTIATION_H_
