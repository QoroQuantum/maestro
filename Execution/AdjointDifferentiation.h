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
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigen>

#ifdef _OPENMP
#include <omp.h>
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

// Up to this many statevector qubits, the rows of a batch run in parallel with
// serial kernels (two states per thread, 32 MiB each at 20 qubits); above it,
// rows run one at a time with parallel kernels.
inline constexpr size_t kRowParallelMaxQubits = 20;

// States whose qubit masks must fit in 64 bits.
inline constexpr size_t kMaxQubits = 63;

namespace Detail
{

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
     * Expectation values for each parameter row: out[row * nObs + o].
     * Observables are evaluated on one statevector (their joint cone).
     */
    std::vector<double> Expectations(const std::vector<Observable> &observables, const std::vector<double> &rows, size_t numRows, bool causalCone) const
    {
        const auto masks = ParseAll(observables);
        uint64_t support = 0;
        for (const auto &obs : masks)
            for (const auto &term : obs)
                support |= term.flip | term.phase;
        const auto cone = Cone(support, causalCone);
        const auto compact = CompactAll(masks, cone);

        std::vector<double> out(numRows * observables.size(), 0.);
        ForRows(numRows, cone.numQubits, [&](size_t row, bool parallel) {
            const double *p = rows.data() + row * NumParams();
            State psi = Forward(cone, p, parallel);
            for (size_t o = 0; o < compact.size(); ++o)
                out[row * observables.size() + o] = Expectation(compact[o], psi, parallel);
        });
        return out;
    }

    /**
     * Jacobian of every observable with respect to every parameter, for each
     * row: jac[(row * nObs + o) * nParams + k]; values as in Expectations.
     * Each observable gets its own causal cone and backward sweep.
     */
    void Jacobian(const std::vector<Observable> &observables, const std::vector<double> &rows, size_t numRows, bool causalCone, std::vector<double> &values,
                  std::vector<double> &jac) const
    {
        const auto masks = ParseAll(observables);
        const size_t nObs = observables.size(), nParams = NumParams();
        values.assign(numRows * nObs, 0.);
        jac.assign(numRows * nObs * nParams, 0.);
        for (size_t o = 0; o < nObs; ++o)
        {
            uint64_t support = 0;
            for (const auto &term : masks[o])
                support |= term.flip | term.phase;
            const auto cone = Cone(support, causalCone);
            const auto obs = Compact(masks[o], cone);
            ForRows(numRows, cone.numQubits, [&](size_t row, bool parallel) {
                const double *p = rows.data() + row * nParams;
                State phi = Forward(cone, p, parallel);
                State lambda = State::Zero(phi.size());
                for (const auto &term : obs)
                    Detail::AddPauli(term, term.coeff, phi, lambda, parallel);
                values[row * nObs + o] = phi.dot(lambda).real();
                Backward(cone, p, phi, lambda, jac.data() + (row * nObs + o) * nParams, parallel);
            });
        }
    }

    /**
     * Vector-Jacobian product for each row: vjp[row * nParams + k] =
     * sum_o cotangents[row * nObs + o] * d<O_o>/dp_k, in one backward sweep.
     */
    void VectorJacobianProduct(const std::vector<Observable> &observables, const std::vector<double> &cotangents, const std::vector<double> &rows,
                               size_t numRows, bool causalCone, std::vector<double> &values, std::vector<double> &vjp) const
    {
        const auto masks = ParseAll(observables);
        const size_t nObs = observables.size(), nParams = NumParams();
        uint64_t support = 0;
        for (const auto &obs : masks)
            for (const auto &term : obs)
                support |= term.flip | term.phase;
        const auto cone = Cone(support, causalCone);
        const auto compact = CompactAll(masks, cone);
        values.assign(numRows * nObs, 0.);
        vjp.assign(numRows * nParams, 0.);
        ForRows(numRows, cone.numQubits, [&](size_t row, bool parallel) {
            const double *p = rows.data() + row * nParams;
            State phi = Forward(cone, p, parallel);
            State lambda = State::Zero(phi.size());
            for (size_t o = 0; o < nObs; ++o)
            {
                values[row * nObs + o] = Expectation(compact[o], phi, parallel);
                const double dy = cotangents[row * nObs + o];
                if (dy != 0.)
                    for (const auto &term : compact[o])
                        Detail::AddPauli(term, dy * term.coeff, phi, lambda, parallel);
            }
            Backward(cone, p, phi, lambda, vjp.data() + row * nParams, parallel);
        });
    }

  private:
    using Masks = std::vector<Detail::PauliMasks>;

    Masks Parse(const Observable &observable) const
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

    std::vector<Masks> ParseAll(const std::vector<Observable> &observables) const
    {
        std::vector<Masks> out;
        out.reserve(observables.size());
        for (const auto &obs : observables)
            out.push_back(Parse(obs));
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

    static Masks Compact(const Masks &obs, const Detail::ConeProgram &cone)
    {
        Masks out = obs;
        for (auto &m : out)
        {
            m.flip = CompactMask(m.flip, cone);
            m.phase = CompactMask(m.phase, cone);
        }
        return out;
    }

    static std::vector<Masks> CompactAll(const std::vector<Masks> &all, const Detail::ConeProgram &cone)
    {
        std::vector<Masks> out;
        out.reserve(all.size());
        for (const auto &obs : all)
            out.push_back(Compact(obs, cone));
        return out;
    }

    static double Expectation(const Masks &obs, const State &psi, bool parallel)
    {
        double value = 0.;
        for (const auto &term : obs)
            value += term.coeff * (term.flip || term.phase ? Detail::PauliExpectation(term, psi, parallel) : psi.squaredNorm());
        return value;
    }

    /** Run rows in parallel for small registers, else kernels in parallel. */
    template <class F> static void ForRows(size_t numRows, size_t coneQubits, F f)
    {
        const bool kernelsParallel = (size_t{1} << coneQubits) >= QC::ParallelMinBasisStatesSetting().load();
        bool rowsParallel = numRows > 1 && coneQubits <= kRowParallelMaxQubits;
#ifdef _OPENMP
        // Registers big enough for parallel kernels only go row-parallel when
        // the rows can keep every thread busy.
        if (kernelsParallel && numRows < static_cast<size_t>(omp_get_max_threads()))
            rowsParallel = false;
#endif
#ifdef _OPENMP
        if (rowsParallel)
        {
#pragma omp parallel for schedule(dynamic)
            for (long long row = 0; row < static_cast<long long>(numRows); ++row)
                f(static_cast<size_t>(row), false);
            return;
        }
#else
        (void)rowsParallel;
#endif
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
