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
#include <fstream>
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
#elif defined(_WIN32)
#include <windows.h>
#undef min
#undef max
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

    /** Throws std::invalid_argument for an unusable Probabilities readout. */
    void Validate() const;

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

inline void Readout::Validate() const
{
    if (kind == Kind::Expectation)
        return;
    if (qubits.empty() || qubits.size() > kMaxProbabilityQubits)
        throw std::invalid_argument("Probabilities need between 1 and " + std::to_string(kMaxProbabilityQubits) + " qubits.");
    uint64_t seen = 0;
    for (const auto q : qubits)
    {
        if (q >= kMaxQubits)
            throw std::invalid_argument("Probabilities qubit index is out of range.");
        const uint64_t bit = uint64_t{1} << q;
        if (seen & bit)
            throw std::invalid_argument("Probabilities qubits must be distinct.");
        seen |= bit;
    }
}

// Validates, because callers size their output from it before anything else runs.
inline size_t Readout::Columns() const
{
    Validate();
    return kind == Kind::Expectation ? 1 : size_t{1} << qubits.size();
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
    double mem = 0.;
    if (pages > 0 && pageSize > 0)
        mem = static_cast<double>(pages) * static_cast<double>(pageSize);

    auto readLimit = [](const std::string &path) -> double {
        std::ifstream f(path);
        if (!f.is_open())
            return 0.;
        std::string token;
        if (!(f >> token) || token == "max")
            return 0.;
        try
        {
            const auto limit = std::stoull(token);
            // Treat limits >= LONG_MAX (typical cgroup v1 unlimited indicator) as unlimited.
            if (limit > 0 && limit < 9223372036854771712ULL)
                return static_cast<double>(limit);
        }
        catch (...)
        {
        }
        return 0.;
    };

    double limit = readLimit("/sys/fs/cgroup/memory.max"); // cgroup v2
    if (limit <= 0.)
        limit = readLimit("/sys/fs/cgroup/memory/memory.limit_in_bytes"); // cgroup v1
    if (limit <= 0.)
        limit = readLimit("/sys/fs/cgroup/memory.limit_in_bytes");

    std::ifstream cgroups("/proc/self/cgroup");
    if (cgroups.is_open())
    {
        std::string line;
        while (std::getline(cgroups, line))
        {
            if (line.rfind("0::", 0) == 0)
            {
                std::string sub = line.substr(3);
                while (!sub.empty() && sub.front() == '/')
                    sub.erase(sub.begin());
                if (!sub.empty())
                {
                    double l = readLimit("/sys/fs/cgroup/" + sub + "/memory.max");
                    if (l > 0.)
                        limit = (limit > 0.) ? std::min(limit, l) : l;
                }
            }
        }
    }

    if (limit > 0.)
        mem = (mem > 0.) ? std::min(mem, limit) : limit;

    return mem;
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return static_cast<double>(status.ullTotalPhys);
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

/** Generator of a one-angle rotation: P, Rx, Ry, Rz or their controlled forms. */
inline Generator::Kind RotationGenerator(GateKind kind)
{
    switch (kind)
    {
    case GateKind::P:
    case GateKind::CP:
        return Generator::Kind::Projector;
    case GateKind::Rx:
    case GateKind::CRx:
        return Generator::Kind::X;
    case GateKind::Ry:
    case GateKind::CRy:
        return Generator::Kind::Y;
    case GateKind::Rz:
    case GateKind::CRz:
        return Generator::Kind::Z;
    default:
        throw std::logic_error("Not a one-angle rotation");
    }
}

/** A generator as a 2x2 matrix on its target qubit. */
inline const Eigen::Matrix2cd &GeneratorMatrix(Generator::Kind kind)
{
    // In Generator::Kind order.
    static const Eigen::Matrix2cd matrices[4] = {
        (Eigen::Matrix2cd() << 0., 1., 1., 0.).finished(),
        (Eigen::Matrix2cd() << 0., Complex(0., -1.), Complex(0., 1.), 0.).finished(),
        (Eigen::Matrix2cd() << 1., 0., 0., -1.).finished(),
        (Eigen::Matrix2cd() << 0., 0., 0., -2.).finished(),
    };
    return matrices[static_cast<size_t>(kind)];
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

/**
 * OpenMP team for a loop over `count` basis states: QCSim's kernel team, so
 * every region of a sweep has the same size (see QC::ParallelTeamSize).
 */
inline int TeamSize(size_t count, bool parallel)
{
    return parallel ? QC::ParallelTeamSize(count) : 1;
}

/** Sum f(i) over the basis states, in parallel for large states. */
template <class F> Complex Reduce(size_t count, bool parallel, F f)
{
    double re = 0., im = 0.;
    [[maybe_unused]] const int threads = TeamSize(count, parallel);
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : re, im) schedule(static) num_threads(threads) if (threads > 1)
#endif
    for (long long i = 0; i < static_cast<long long>(count); ++i)
    {
        const Complex v = f(static_cast<uint64_t>(i));
        re += v.real();
        im += v.imag();
    }
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
    [[maybe_unused]] const int threads = TeamSize(static_cast<size_t>(count), parallel);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
#endif
    for (long long i = 0; i < count; ++i)
    {
        const uint64_t j = static_cast<uint64_t>(i) ^ flip;
        const Complex v = c * src[j];
        dst[i] += (Popcount(j & phase) & 1) ? -v : v;
    }
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
    const int threads = TeamSize(static_cast<size_t>(count), parallel);
    if (numOutcomes > kMarginalHistogramMaxOutcomes)
    {
        uint64_t measured = 0;
        for (const auto bit : bits)
            measured |= bit;
        const uint64_t rest = (static_cast<uint64_t>(count) - 1) & ~measured;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
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
        return;
    }

    // Allocated up front: an exception must not escape a parallel region.
    std::vector<double> local(static_cast<size_t>(threads) * numOutcomes, 0.);
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if (threads > 1)
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
    [[maybe_unused]] const int threads = TeamSize(static_cast<size_t>(count), parallel);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
#endif
    for (long long i = 0; i < count; ++i)
        dst[i] += weights[Outcome(static_cast<uint64_t>(i), bits)] * src[i];
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
    [[maybe_unused]] const int threads = TeamSize(static_cast<size_t>(phi.size()), parallel);
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : r00, i00, r01, i01, r10, i10, r11, i11) schedule(static) num_threads(threads) if (threads > 1)
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
    std::vector<Block> blocks; // the cone's gates, with one-qubit runs fused
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
            for (unsigned i = 0; i < spec.numQubits; ++i)
            {
                const auto q = gate->GetQubit(i);
                if (static_cast<size_t>(q) >= kMaxQubits)
                    throw std::invalid_argument("Adjoint differentiation supports at most " + std::to_string(kMaxQubits) + " qubits.");
                numQubits = std::max(numQubits, static_cast<size_t>(q) + 1);
                spec.mask |= uint64_t{1} << q;
                // Maestro lists controls first; QCSim kernels take the target first.
                spec.kernelQubits[spec.numQubits - 1 - i] = q;
            }

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
                spec.gamma = params[3];
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
            case QuantumGateType::kYGateType:
            case QuantumGateType::kZGateType:
            case QuantumGateType::kHadamardGateType:
            case QuantumGateType::kSGateType:
            case QuantumGateType::kSdgGateType:
            case QuantumGateType::kTGateType:
            case QuantumGateType::kTdgGateType:
            case QuantumGateType::kSxGateType:
            case QuantumGateType::kSxDagGateType:
            case QuantumGateType::kKGateType:
            case QuantumGateType::kSwapGateType:
            case QuantumGateType::kCXGateType:
            case QuantumGateType::kCYGateType:
            case QuantumGateType::kCZGateType:
            case QuantumGateType::kCHGateType:
            case QuantumGateType::kCSxGateType:
            case QuantumGateType::kCSxDagGateType:
            case QuantumGateType::kCCXGateType:
            case QuantumGateType::kCSwapGateType:
                spec.fixed = gate->GetMatrix();
                break;
            default:
                throw std::invalid_argument("Adjoint differentiation: unsupported gate type.");
            }

            spec.firstParam = defaults.size();
            spec.numParams = ParamCount(spec.kind);
            defaults.insert(defaults.end(), params.begin(), params.begin() + static_cast<std::ptrdiff_t>(spec.numParams));
            gates.push_back(std::move(spec));
        }
        if (numQubits > kMaxQubits)
            throw std::invalid_argument("Adjoint differentiation supports at most " + std::to_string(kMaxQubits) + " qubits.");
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
        CheckRows(rows, numRows);
        const auto parsed = ParseAll(readouts);
        const auto cone = Cone(Support(parsed), causalCone);
        const auto compact = CompactAll(parsed, cone);
        const size_t nCols = Columns(readouts);
        CheckMemory(StateBytes(numRows, cone.numQubits, 1), static_cast<double>(numRows) * nCols);

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
        CheckRows(rows, numRows);
        const size_t nCols = Columns(readouts), nParams = NumParams();
        const auto parsed = ParseAll(readouts);
        std::vector<Detail::ConeProgram> cones;
        cones.reserve(parsed.size());
        // Readouts run one after another, each with its own row concurrency:
        // the peak is the largest of them. psi, plus phi and lambda per sweep.
        double stateBytes = 0.;
        for (const auto &r : parsed)
        {
            cones.push_back(Cone(r.support, causalCone));
            stateBytes = std::max(stateBytes, StateBytes(numRows, cones.back().numQubits, 3));
        }
        CheckMemory(stateBytes, static_cast<double>(numRows) * nCols * (nParams + 1));

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
                WriteValue(compact, psi, values.data() + row * nCols + first, parallel);
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
        CheckRows(rows, numRows);
        const auto parsed = ParseAll(readouts);
        const auto cone = Cone(Support(parsed), causalCone);
        const auto compact = CompactAll(parsed, cone);
        const size_t nCols = Columns(readouts), nParams = NumParams();
        if (cotangents.size() != numRows * nCols)
            throw std::invalid_argument("Cotangents must hold one value per row and readout value.");
        CheckMemory(StateBytes(numRows, cone.numQubits, 2), static_cast<double>(numRows) * (nCols + nParams));
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
        uint64_t support = 0;       // qubits read, on the original register
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
            const uint64_t bit = uint64_t{1} << q;
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
                    throw std::invalid_argument("Pauli string is longer than " + std::to_string(kMaxQubits) + " qubits.");
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
        std::vector<size_t> coneGates; // indices into gates, in circuit order
        uint64_t qubits = 0;
        if (causalCone)
        {
            qubits = support;
            for (size_t k = gates.size(); k-- > 0;)
                if (gates[k].mask & qubits)
                {
                    qubits |= gates[k].mask;
                    coneGates.push_back(k);
                }
            std::reverse(coneGates.begin(), coneGates.end());
        }
        else
        {
            for (size_t k = 0; k < gates.size(); ++k)
            {
                qubits |= gates[k].mask;
                coneGates.push_back(k);
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
        for (const size_t k : coneGates)
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

    /** Write the values of a compacted readout. */
    static void WriteValue(const ParsedReadout &r, const State &psi, double *out, bool parallel)
    {
        if (r.kind == Readout::Kind::Expectation)
            *out = Expectation(r.terms, psi, parallel);
        else
            Detail::Marginal(psi, r.bits, out, parallel);
    }

    /** Write the values of compacted readouts, side by side. */
    static void WriteValues(const std::vector<ParsedReadout> &readouts, const State &psi, double *out, bool parallel)
    {
        for (const auto &r : readouts)
        {
            WriteValue(r, psi, out, parallel);
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
        const auto size = static_cast<size_t>(psi.size());
        for (const auto &term : obs)
            value += term.coeff * Calculator::PauliExpectationValue(size, psi, term.flip, term.phase, term.numY, parallel).real();
        return value;
    }

    void CheckRows(const std::vector<double> &rows, size_t numRows) const
    {
        if (rows.size() != numRows * NumParams())
            throw std::invalid_argument("Parameter rows must hold NumParams() values per row.");
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

    /** Bytes of the statevectors alive at once while ForRows runs rows on this cone. */
    static double StateBytes(size_t numRows, size_t coneQubits, size_t statesPerRow)
    {
        const double rows = static_cast<double>(ConcurrentRows(numRows, coneQubits));
        return rows * statesPerRow * std::ldexp(static_cast<double>(sizeof(Complex)), static_cast<int>(coneQubits));
    }

    /**
     * Refuse a call whose states and outputs would not fit in physical memory,
     * rather than let the allocation swap the machine or get it OOM-killed.
     */
    static void CheckMemory(double stateBytes, double outputValues)
    {
        const double needed = stateBytes + outputValues * sizeof(double);
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
        throw MemoryLimitError("Adjoint differentiation needs about " + gib(needed) + " GiB for its statevectors and outputs, more than the " + gib(limit) +
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
        using Detail::Generator;
        const auto &first = gates[block.gates.front()];
        const uint64_t bit = uint64_t{1} << cone.compact[first.kernelQubits[0]];

        bool anyParams = false;
        for (const size_t k : block.gates)
            anyParams |= gates[k].numParams > 0;
        const Eigen::MatrixXcd total = RunMatrix(block, p);
        if (anyParams)
        {
            const Eigen::Matrix2cd M = Detail::CrossMatrix(lambda, phi, bit, parallel);
            Eigen::Matrix2cd V = Eigen::Matrix2cd::Identity();
            auto im = [&](Generator::Kind kind) {
                const Eigen::Matrix2cd &G = Detail::GeneratorMatrix(kind);
                return (V * G * V.adjoint()).cwiseProduct(M).sum().imag();
            };
            for (size_t idx = block.gates.size(); idx-- > 0;)
            {
                const auto &g = gates[block.gates[idx]];
                const double *gp = p + g.firstParam;
                double *gg = grad + g.firstParam;
                if (g.kind == GateKind::U)
                {
                    // U = e^{i(f + l)/2} Rz(f) Ry(t) Rz(l); the phase cancels in V G V^dag.
                    gg[1] += im(Generator::Kind::Z);
                    V = V * Detail::RzMatrix(gp[1]);
                    gg[0] += im(Generator::Kind::Y);
                    V = V * Detail::RyMatrix(gp[0]);
                    gg[2] += im(Generator::Kind::Z);
                    V = V * Detail::RzMatrix(gp[2]);
                    continue;
                }
                if (g.kind != GateKind::Fixed)
                    gg[0] += im(Detail::RotationGenerator(g.kind));
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
            // Cone() fuses every one-qubit gate into a run, so this gate has
            // a target and at least one control.
            const auto &g = gates[block.gates.front()];
            const double *gp = p + g.firstParam;
            double *gg = grad + g.firstParam;
            const uint64_t target = uint64_t{1} << cone.compact[g.kernelQubits[0]];
            const uint64_t control = uint64_t{1} << cone.compact[g.kernelQubits[1]];
            auto im = [&](Generator::Kind kind, uint64_t on, uint64_t controls) {
                return Detail::GeneratorElement({kind, on, controls}, lambda, phi, parallel).imag();
            };
            // Undo a one-qubit piece of the gate, controlled like the gate.
            auto undoPart = [&](const Eigen::MatrixXcd &m) { Undo(g, Detail::Controlled(m), cone, phi, lambda, parallel); };

            switch (g.kind)
            {
            case GateKind::Fixed:
                Undo(g, g.fixed, cone, phi, lambda, parallel);
                break;
            case GateKind::CP:
            case GateKind::CRx:
            case GateKind::CRy:
            case GateKind::CRz:
                gg[0] += im(Detail::RotationGenerator(g.kind), target, control);
                Undo(g, Detail::GateMatrix(g, gp), cone, phi, lambda, parallel);
                break;
            case GateKind::CU: {
                // CU = controlled [e^{i(f + l)/2} Rz(f) Ry(t) Rz(l)] times the
                // controlled phase P(gamma + (f + l)/2) on the control qubit,
                // which is not global and so is undone explicitly first.
                const double d = im(Generator::Kind::Projector, control, 0);
                gg[3] += d;
                gg[1] += 0.5 * d;
                gg[2] += 0.5 * d;
                const Complex phase = std::polar(1., gp[3] + 0.5 * (gp[1] + gp[2]));
                undoPart(phase * Eigen::MatrixXcd::Identity(2, 2));
                gg[1] += im(Generator::Kind::Z, target, control);
                undoPart(Detail::RzMatrix(gp[1]));
                gg[0] += im(Generator::Kind::Y, target, control);
                undoPart(Detail::RyMatrix(gp[0]));
                gg[2] += im(Generator::Kind::Z, target, control);
                undoPart(Detail::RzMatrix(gp[2]));
                break;
            }
            default:
                throw std::logic_error("One-qubit gates are differentiated as fused runs.");
            }
        }
    }

    std::vector<Detail::GateSpec> gates;
    std::vector<double> defaults;
    size_t numQubits = 0;
};

} // namespace MaestroExecution::Adjoint

#endif // _ADJOINT_DIFFERENTIATION_H_
