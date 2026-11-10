#pragma once

// Batched extended-stabilizer trajectories that share one Clifford basis.
//
// QC::ExtendedStabilizer represents a pure state as U sum_b a_b |b>, with U a
// signed inverse Clifford map, logical labels b and complex amplitudes. A
// batch of noisy trajectories ("lanes") is held here as
//
//     lane s:  E_s U sum_b a_{b,s} |b>,
//
// with one shared reference map U, one physical Pauli frame E_s per lane
// (bit-packed 64 lanes per word, like ReferenceFrameSampler), one label set
// shared by all lanes (the union of their supports) and a dense amplitude
// matrix amplitudes[label * lanes + lane]. A lane is exactly the scalar state
// whose signed map is E_s U: a Pauli only flips row signs, so every lane
// shares the unsigned tableau, the images' X/Z parts, the coset structure and
// the label index. Lanes differ only in
//   - the sign (-1)^<P_j, E_s> of each compiled Pauli term (frames), and
//   - their amplitudes and Born draws.
// Clifford gates update U once and the frames by XOR; Pauli noise only XORs
// frames. A measurement whose logical outcome is one in some lane shifts that
// lane's labels by a logical X^d and folds the Pauli U X^d U^dagger into its
// frame (the gauge U' X_p U'^dagger for a random outcome), so every lane stays
// on the same labels and the shared set stays as small as one lane's support.
//
// Gates are compiled exactly as QC::ExtendedStabilizer compiles them, so
// each lane follows the scalar simulator's arithmetic. Exact mode only.

#include "ExtendedStabilizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MaestroExecution
{
namespace ExtendedFrames
{
using Word = uint64_t;
using Complex = std::complex<double>;
using LocalGate = QC::detail::LocalGate;
using LocalPauli = QC::detail::LocalPauli;
using LocalPauliSum = QC::detail::LocalPauliSum;
using CliffordOp = QC::detail::LocalGate::CliffordOp;

// The batch outgrew its amplitude budget. The sampler splits it, down to
// single shots; a shot that alone does not fit fails a strict budget, while a
// soft one (execution and the compiled samplers) runs it beyond the budget.
class MemoryLimitExceeded : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

// A gate on at most three local qubits: a Pauli sum acting first (absent for
// a Clifford), then a local Clifford replayed on the basis. qubits maps local
// qubits to physical ones, as QC::ExtendedStabilizer::ApplyLocalGate does.
struct CompiledGate
{
    size_t nrQubits = 0;
    std::array<size_t, LocalPauli::MaxQubits> qubits{};
    std::vector<std::pair<LocalPauli, Complex>> terms;
    std::vector<CliffordOp> cliffords;
};

// The gate compilations of QC::ExtendedStabilizer (QCSim ebc28be),
// reproduced term for term. Arguments use its (target, control) order.
class GateCompiler
{
  public:
    CompiledGate U(size_t qubit, double theta, double phi, double lambda)
    {
        RequireFinite({theta, phi, lambda});
        auto &gate = Begin(1);
        AddU(gate, 0, theta, phi, lambda);
        return Finish(gate, {qubit});
    }

    CompiledGate CU(size_t target, size_t control, double theta, double phi, double lambda, double gamma)
    {
        RequireFinite({theta, phi, lambda, gamma});
        auto &gate = Begin(2);
        gate.Rotate(0, 'Z', gamma).Rotate(1, 'Z', 0.5 * (lambda - phi)).Rotate(0, 'Z', 0.5 * (lambda + phi)).CX(1, 0);
        AddU(gate, 1, -0.5 * theta, 0.0, -0.5 * (phi + lambda));
        gate.CX(1, 0);
        AddU(gate, 1, 0.5 * theta, phi, 0.0);
        return Finish(gate, {control, target});
    }

    CompiledGate CRx(size_t target, size_t control, double angle)
    {
        RequireFinite({angle});
        auto &gate = Begin(2);
        gate.H(1).CX(1, 0).Rotate(1, 'Z', -0.5 * angle).CX(1, 0).Rotate(1, 'Z', 0.5 * angle).H(1);
        return Finish(gate, {control, target});
    }

    CompiledGate CRy(size_t target, size_t control, double angle)
    {
        RequireFinite({angle});
        auto &gate = Begin(2);
        gate.Rotate(1, 'Y', 0.5 * angle).CX(1, 0).Rotate(1, 'Y', -0.5 * angle).CX(1, 0);
        return Finish(gate, {control, target});
    }

    CompiledGate CRz(size_t target, size_t control, double angle)
    {
        RequireFinite({angle});
        auto &gate = Begin(2);
        gate.Rotate(1, 'Z', 0.5 * angle).CX(1, 0).Rotate(1, 'Z', -0.5 * angle).CX(1, 0);
        return Finish(gate, {control, target});
    }

    CompiledGate CH(size_t target, size_t control)
    {
        const double eighthTurn = 0.25 * std::acos(-1.0);
        auto &gate = Begin(2);
        gate.Rotate(1, 'Y', -eighthTurn).CZ(1, 0).Rotate(1, 'Y', eighthTurn);
        return Finish(gate, {control, target});
    }

    // Controlled phase diag(1, 1, 1, e^(i lambda)). The identity at multiples
    // of 2 pi and CZ at odd multiples of pi, as QC::ExtendedStabilizer::ApplyCP.
    CompiledGate CP(size_t target, size_t control, double lambda)
    {
        RequireFinite({lambda});
        Complex phase;
        long long quarterTurns = 0;
        if (QC::detail::TryGetQuarterTurns(lambda, quarterTurns))
        {
            const unsigned turns = unsigned(((quarterTurns % 4) + 4) % 4);
            if (turns == 0)
                return Finish(Begin(2), {control, target});
            if (turns == 2)
            {
                auto &gate = Begin(2);
                gate.CZ(1, 0);
                return Finish(gate, {control, target});
            }
            phase = LocalPauliSum::IPower(turns);
        }
        else
            phase = std::polar(1.0, lambda);
        auto &gate = Begin(2);
        gate.Multiply(Controlled(0, LocalPauliSum::Projector(1, false) + LocalPauliSum::Projector(1, true) * LocalPauliSum::Identity(phase)));
        return Finish(gate, {control, target});
    }

    CompiledGate CS(size_t target, size_t control)
    {
        return CP(target, control, 0.5 * std::acos(-1.0));
    }

    CompiledGate CSdg(size_t target, size_t control)
    {
        return CP(target, control, -0.5 * std::acos(-1.0));
    }

    // Controlled square root of X, ((1 + i) I + (1 - i) X) / 2, and its inverse.
    CompiledGate CSx(size_t target, size_t control, bool inverse)
    {
        const Complex plus(0.5, 0.5), minus(0.5, -0.5);
        auto &gate = Begin(2);
        gate.Multiply(Controlled(0, LocalPauliSum::Identity(inverse ? minus : plus) + LocalPauliSum::Pauli(1, 'X', inverse ? plus : minus)));
        return Finish(gate, {control, target});
    }

    CompiledGate CCX(size_t target, size_t control1, size_t control2)
    {
        auto &gate = Begin(3);
        gate.Multiply(Controlled(0, Controlled(1, LocalPauliSum::Pauli(2, 'X'))));
        return Finish(gate, {control1, control2, target});
    }

    CompiledGate CSwap(size_t target1, size_t target2, size_t control)
    {
        LocalPauliSum swap = LocalPauliSum::Identity(0.5);
        for (const char pauli : {'X', 'Y', 'Z'})
            swap = swap + LocalPauliSum::Pauli(1, pauli, 0.5) * LocalPauliSum::Pauli(2, pauli);
        auto &gate = Begin(3);
        gate.Multiply(Controlled(0, swap));
        return Finish(gate, {control, target1, target2});
    }

  private:
    static void AddU(LocalGate &gate, size_t qubit, double theta, double phi, double lambda)
    {
        gate.Rotate(qubit, 'Z', lambda).Rotate(qubit, 'Y', theta).Rotate(qubit, 'Z', phi);
    }

    static LocalPauliSum Controlled(size_t control, const LocalPauliSum &op)
    {
        return LocalPauliSum::Projector(control, false) + LocalPauliSum::Projector(control, true) * op;
    }

    static void RequireFinite(std::initializer_list<double> angles)
    {
        for (const double angle : angles)
            if (!std::isfinite(angle))
                throw std::invalid_argument("Rotation angle must be finite");
    }

    LocalGate &Begin(size_t qubits)
    {
        gate.Reset(qubits);
        return gate;
    }

    static CompiledGate Finish(const LocalGate &gate, std::initializer_list<size_t> qubits)
    {
        CompiledGate compiled;
        compiled.nrQubits = gate.GetNrQubits();
        std::copy(qubits.begin(), qubits.end(), compiled.qubits.begin());
        if (!gate.Sum().IsScalar())
            gate.Sum().ForEachTerm([&](LocalPauli pauli, Complex coefficient) { compiled.terms.emplace_back(pauli, coefficient); });
        compiled.cliffords = gate.Cliffords();
        return compiled;
    }

    LocalGate gate{1};
};

class LaneBatch
{
    using Tableau = QC::Clifford::detail::PackedTableau;
    static constexpr size_t kNotFound = QC::PackedComponentIndex::NotFound;
    static constexpr size_t kMaxGenerators = 2 * LocalPauli::MaxQubits;
    static constexpr size_t kMaxOffsets = size_t(1) << kMaxGenerators;
    static constexpr size_t kMaxTerms = LocalPauliSum::Keys;

  public:
    LaneBatch(size_t qubits, size_t lanes, size_t maxAmplitudeBytes = std::numeric_limits<size_t>::max())
        : n(qubits), labelWords((qubits + 63) / 64), identity(qubits), reference(qubits), labels(qubits, 0), nextLabels(qubits, 0),
          amplitudeBudget(maxAmplitudeBytes)
    {
        if (!qubits)
            throw std::invalid_argument("A lane batch needs at least one qubit");
        Reset(lanes);
    }

    // Every lane |0...0> with an identity frame.
    void Reset(size_t laneCount)
    {
        if (!laneCount)
            throw std::invalid_argument("A lane batch needs at least one lane");
        lanes = laneCount;
        words = (lanes + 63) / 64;
        reference = identity;
        frameX.assign(n * words, 0);
        frameZ.assign(n * words, 0);
        labels.clear();
        zeroLabel.assign(labelWords, 0);
        labels.Append(zeroLabel.data());
        // Buffers kept from an earlier batch are reused, but the budget counts
        // only what this batch reserves, as a fresh batch would.
        nextAmplitudes.clear();
        laneCoefficients.clear();
        gathered.clear();
        results.clear();
        reserved = Reserved{};
        peakEntries = 0;
        Reserve(amplitudes, lanes);
        amplitudes.assign(lanes, Complex(1.0, 0.0));
        indexValid = false;
    }

    size_t Qubits() const noexcept
    {
        return n;
    }

    size_t Lanes() const noexcept
    {
        return lanes;
    }

    size_t Words() const noexcept
    {
        return words;
    }

    size_t Components() const noexcept
    {
        return labels.size();
    }

    // Bytes this batch reserved for its amplitude buffers: the matrix, the
    // next matrix of a measurement or expansion, and gate workspaces, as a
    // fresh batch would allocate them. This is what the budget limits,
    // including the old buffer while one grows.
    size_t AmplitudeBytes() const noexcept
    {
        return LiveEntries() * sizeof(Complex);
    }

    // Bytes the amplitude buffers actually hold, which reuse across batches
    // can make larger than AmplitudeBytes().
    size_t HeldAmplitudeBytes() const noexcept
    {
        return (amplitudes.capacity() + nextAmplitudes.capacity() + laneCoefficients.capacity() + gathered.capacity() + results.capacity()) * sizeof(Complex);
    }

    void SetAmplitudeBudget(size_t bytes) noexcept
    {
        amplitudeBudget = bytes;
    }

    // The most amplitude memory held at once, growth included.
    size_t PeakAmplitudeBytes() const noexcept
    {
        return peakEntries * sizeof(Complex);
    }

    size_t AmplitudeBudget() const noexcept
    {
        return amplitudeBudget;
    }

    // A new budget after a copy: the copied buffers count as a fresh copy
    // allocates them (their sizes), and the peak restarts there. Throws
    // MemoryLimitExceeded when the copy alone exceeds the new budget.
    void RestartBudget(size_t bytes)
    {
        amplitudeBudget = bytes;
        reserved = Reserved{amplitudes.size(), nextAmplitudes.size(), laneCoefficients.size(), gathered.size(), results.size()};
        peakEntries = LiveEntries();
        if (peakEntries > amplitudeBudget / sizeof(Complex))
            throw MemoryLimitExceeded("Extended-frame batch exceeds its amplitude budget");
    }

    // Releases the workspaces and spare capacity of a state kept only to be
    // copied, such as a cached head, so it holds little more than its
    // amplitude matrix; HeldAmplitudeBytes() is then that matrix alone, or
    // its capacity when shrinking it would not fit the budget.
    void Compact()
    {
        for (auto *buffer : {&nextAmplitudes, &laneCoefficients, &gathered, &results, &pairPhases})
            std::vector<Complex>().swap(*buffer);
        for (auto *buffer : {&gatheredL1, &bounds, &probabilityZero, &probabilityOne, &inverseNorms})
            std::vector<double>().swap(*buffer);
        // Shrinking copies into a new block while the old one is still live,
        // so it happens only when both fit; otherwise the capacity is kept,
        // and charged.
        if (amplitudes.capacity() > amplitudes.size() && amplitudes.capacity() + amplitudes.size() <= amplitudeBudget / sizeof(Complex))
        {
            peakEntries = std::max(peakEntries, amplitudes.capacity() + amplitudes.size());
            amplitudes.shrink_to_fit();
        }
        reserved = Reserved{amplitudes.capacity(), 0, 0, 0, 0};
        peakEntries = std::max(peakEntries, LiveEntries());
    }

    // Physical Pauli frame bits of a qubit, one bit per lane: X^x Z^z.
    Word *FrameX(size_t qubit) noexcept
    {
        return frameX.data() + qubit * words;
    }

    Word *FrameZ(size_t qubit) noexcept
    {
        return frameZ.data() + qubit * words;
    }

    const Word *FrameX(size_t qubit) const noexcept
    {
        return frameX.data() + qubit * words;
    }

    const Word *FrameZ(size_t qubit) const noexcept
    {
        return frameZ.data() + qubit * words;
    }

    const QC::CliffordBasisMap &Reference() const noexcept
    {
        return reference;
    }

    const QC::PackedComponentLabels &Labels() const noexcept
    {
        return labels;
    }

    Complex Amplitude(size_t component, size_t lane) const noexcept
    {
        return amplitudes[component * lanes + lane];
    }

    // ---- Clifford gates (QCSim's target/control order). The reference map
    // follows QC::Clifford::detail::InverseMap; every row product there XORs
    // the rows' sign differences, which are the frame bits: the X_q row's sign
    // difference is the frame's Z bit and the Z_q row's its X bit.
    void ApplyH(size_t q)
    {
        reference.ApplyH(q);
        for (size_t w = 0; w < words; ++w)
            std::swap(FrameX(q)[w], FrameZ(q)[w]);
    }

    void ApplyS(size_t q)
    {
        reference.ApplyS(q);
        XorInto(FrameZ(q), FrameX(q));
    }

    void ApplySdg(size_t q)
    {
        reference.ApplySdg(q);
        XorInto(FrameZ(q), FrameX(q));
    }

    void ApplySx(size_t q)
    {
        reference.ApplySx(q);
        XorInto(FrameX(q), FrameZ(q));
    }

    void ApplySxDag(size_t q)
    {
        reference.ApplySxDag(q);
        XorInto(FrameX(q), FrameZ(q));
    }

    void ApplyK(size_t q)
    {
        reference.ApplyK(q);
        XorInto(FrameX(q), FrameZ(q));
    }

    // Pauli gates only flip reference signs; they commute with every frame.
    void ApplyX(size_t q)
    {
        reference.ApplyX(q);
    }

    void ApplyY(size_t q)
    {
        reference.ApplyY(q);
    }

    void ApplyZ(size_t q)
    {
        reference.ApplyZ(q);
    }

    void ApplyCX(size_t target, size_t control)
    {
        reference.ApplyCX(target, control);
        XorInto(FrameZ(control), FrameZ(target));
        XorInto(FrameX(target), FrameX(control));
    }

    void ApplyCY(size_t target, size_t control)
    {
        ApplySdg(target);
        ApplyCX(target, control);
        ApplyS(target);
    }

    void ApplyCZ(size_t target, size_t control)
    {
        reference.ApplyCZ(target, control);
        XorInto(FrameZ(target), FrameX(control));
        XorInto(FrameZ(control), FrameX(target));
    }

    void ApplySwap(size_t a, size_t b)
    {
        reference.ApplySwap(a, b);
        for (size_t w = 0; w < words; ++w)
        {
            std::swap(FrameX(a)[w], FrameX(b)[w]);
            std::swap(FrameZ(a)[w], FrameZ(b)[w]);
        }
    }

    void ApplyISwap(size_t a, size_t b)
    {
        ApplyS(a);
        ApplyS(b);
        ApplyCZ(a, b);
        ApplySwap(a, b);
    }

    void ApplyISwapDag(size_t a, size_t b)
    {
        ApplySdg(a);
        ApplySdg(b);
        ApplyCZ(a, b);
        ApplySwap(a, b);
    }

    // exp(-+i pi/4 P) on the basis, as CliffordBasisMap::ApplyQuarterTurn.
    void ApplyQuarterTurn(size_t q, char axis, bool inverse)
    {
        reference.ApplyQuarterTurn(q, axis, inverse);
        if (axis == 'X')
            XorInto(FrameX(q), FrameZ(q));
        else if (axis == 'Z')
            XorInto(FrameZ(q), FrameX(q));
        else
            for (size_t w = 0; w < words; ++w)
                std::swap(FrameX(q)[w], FrameZ(q)[w]);
    }

    // ---- Non-Clifford gates.

    // Rx/Ry/Rz exactly as QC::ExtendedStabilizer::ApplyAxisRotation.
    void ApplyAxisRotation(size_t q, double angle, char axis)
    {
        if (!std::isfinite(angle))
            throw std::invalid_argument("Rotation angle must be finite");
        if (angle == 0.0)
            return;
        long long quarterTurns = 0;
        if (QC::detail::TryGetQuarterTurns(angle, quarterTurns))
        {
            ApplyCliffordRotation(q, axis, quarterTurns);
            return;
        }
        const double halfAngle = 0.5 * angle;
        axisTerms.clear();
        axisTerms.emplace_back(LocalPauli{}, Complex(std::cos(halfAngle), 0.0));
        axisTerms.emplace_back(LocalPauli::Single(0, axis), Complex(0.0, -std::sin(halfAngle)));
        const size_t qubits[] = {q};
        ApplyPauliSum(axisTerms, qubits, 1);
    }

    void ApplyGate(const CompiledGate &gate)
    {
        if (!gate.terms.empty())
            ApplyPauliSum(gate.terms, gate.qubits.data(), gate.nrQubits);
        for (const auto &op : gate.cliffords)
            ApplyCliffordOp(op, gate.qubits.data());
    }

    // ---- Lane management.

    // XOR a physical Pauli into the frames of the lanes in mask.
    void XorPauli(size_t q, bool x, bool z, const Word *mask)
    {
        if (x)
            XorInto(FrameX(q), mask);
        if (z)
            XorInto(FrameZ(q), mask);
    }

    // New lane i becomes a copy of old lane source[i].
    void ExpandLanes(const std::vector<size_t> &source)
    {
        const size_t newLanes = source.size();
        if (!newLanes)
            throw std::invalid_argument("A lane batch needs at least one lane");
        const size_t newWords = (newLanes + 63) / 64;
        const size_t components = labels.size();
        Reserve(nextAmplitudes, components * newLanes);
        nextAmplitudes.resize(components * newLanes);
        for (size_t c = 0; c < components; ++c)
            for (size_t lane = 0; lane < newLanes; ++lane)
                nextAmplitudes[c * newLanes + lane] = amplitudes[c * lanes + source[lane]];
        SwapAmplitudes();
        auto expand = [&](std::vector<Word> &frame) {
            std::vector<Word> expanded(n * newWords, 0);
            for (size_t q = 0; q < n; ++q)
                for (size_t lane = 0; lane < newLanes; ++lane)
                    if ((frame[q * words + source[lane] / 64] >> (source[lane] % 64)) & 1U)
                        expanded[q * newWords + lane / 64] |= Word(1) << (lane % 64);
            frame.swap(expanded);
        };
        expand(frameX);
        expand(frameZ);
        lanes = newLanes;
        words = newWords;
    }

    // ---- Measurement.

    // Measures physical Z_q in every lane. choose(lane, probabilityOne)
    // receives the normalized probability that the lane's physical outcome is
    // one and returns that outcome, written to physical[w] (bit = lane). A
    // lane's frame and labels depend on the other lanes of its batch (a mixed
    // measurement folds some of them), its physical state does not, so equal
    // draws give equal outcomes in any batch. The logical outcome (for
    // Q = U^dagger Z_q U of the reference map) is physical xor the lane's X
    // frame bit on q.
    template <class Choose> void MeasureZ(size_t q, Choose &&choose, Word *physical)
    {
        const auto observable = reference.ImageZ(q);
        const Word *flipMask = observable.X.words;
        const Word *phaseMask = observable.Z.words;
        bool hasFlip = false;
        for (size_t w = 0; w < labelWords; ++w)
            hasFlip |= flipMask[w] != 0;
        Complex basePhase = BasePhase(observable);
        logicalOutcomes.assign(words, 0);
        // The outcome is read through the frame before the measurement: the
        // gauge fold below changes the X bit of q in outcome-one lanes.
        measuredX.assign(FrameX(q), FrameX(q) + words);
        if (!hasFlip)
            MeasureDiagonal(phaseMask, basePhase, choose);
        else
            MeasureOffDiagonal(q, observable, basePhase, choose);
        for (size_t w = 0; w < words; ++w)
            physical[w] = (logicalOutcomes[w] ^ measuredX[w]) & ActiveMask(w);
    }

    // Reset to |0> of a circuit qubit: measure, then X on the lanes that read
    // one (a frame flip), as QCSimExtendedStabilizerSimulator::ApplyReset.
    template <class Choose> void ResetQubit(size_t q, Choose &&choose)
    {
        std::vector<Word> outcome(words);
        MeasureZ(q, choose, outcome.data());
        XorInto(FrameX(q), outcome.data());
    }

    // ---- Inspection (tests).

    // One lane as a scalar QC::ExtendedFrame: the reference map with the
    // lane's Pauli applied and the lane's nonzero components.
    QC::ExtendedFrame LaneFrame(size_t lane) const
    {
        QC::ExtendedFrame frame(n);
        frame.cliffordBasis = reference;
        for (size_t q = 0; q < n; ++q)
        {
            if ((FrameX(q)[lane / 64] >> (lane % 64)) & 1U)
                frame.cliffordBasis.ApplyX(q);
            if ((FrameZ(q)[lane / 64] >> (lane % 64)) & 1U)
                frame.cliffordBasis.ApplyZ(q);
        }
        frame.amplitudes.clear();
        frame.signs.clear();
        for (size_t c = 0; c < labels.size(); ++c)
        {
            const Complex a = amplitudes[c * lanes + lane];
            if (a.real() == 0.0 && a.imag() == 0.0)
                continue;
            frame.amplitudes.push_back(a);
            frame.signs.Append(labels.LabelWords(c));
        }
        return frame;
    }

  private:
    struct Term
    {
        Complex coefficient; // reference coefficient, image phase included
        size_t offset;
        bool hasZ;
    };

    Word ActiveMask(size_t w) const noexcept
    {
        const size_t first = 64 * w;
        return first + 64 <= lanes ? ~Word(0) : first < lanes ? (Word(1) << (lanes - first)) - 1 : 0;
    }

    void XorInto(Word *target, const Word *source) noexcept
    {
        for (size_t w = 0; w < words; ++w)
            target[w] ^= source[w];
    }

    bool LaneBit(const Word *mask, size_t lane) const noexcept
    {
        return (mask[lane / 64] >> (lane % 64)) & 1U;
    }

    // What a fresh batch would hold for each amplitude buffer: the largest
    // size reserved so far in this batch (sizes after a copy).
    struct Reserved
    {
        size_t amplitudes = 0, next = 0, coefficients = 0, gathered = 0, results = 0;
    };

    size_t &ReservedFor(const std::vector<Complex> &buffer) noexcept
    {
        if (&buffer == &amplitudes)
            return reserved.amplitudes;
        if (&buffer == &nextAmplitudes)
            return reserved.next;
        if (&buffer == &laneCoefficients)
            return reserved.coefficients;
        if (&buffer == &gathered)
            return reserved.gathered;
        return reserved.results;
    }

    size_t LiveEntries() const noexcept
    {
        return reserved.amplitudes + reserved.next + reserved.coefficients + reserved.gathered + reserved.results;
    }

    // amplitudes and nextAmplitudes exchange contents, and so their reservations.
    void SwapAmplitudes() noexcept
    {
        amplitudes.swap(nextAmplitudes);
        std::swap(reserved.amplitudes, reserved.next);
    }

    // Grow buffer to hold entries, within the budget for every live buffer.
    // Growing keeps the old allocation alive until the copy is made, which
    // LiveEntries() still counts.
    void Reserve(std::vector<Complex> &buffer, size_t entries)
    {
        size_t &held = ReservedFor(buffer);
        if (entries > held)
        {
            const size_t live = LiveEntries() + entries;
            if (live > amplitudeBudget / sizeof(Complex))
                throw MemoryLimitExceeded("Extended-frame batch exceeds its amplitude budget");
            peakEntries = std::max(peakEntries, live);
            held = entries;
        }
        if (entries > buffer.capacity())
            buffer.reserve(entries);
    }

    void EnsureIndex(size_t expected)
    {
        if (!indexValid || index.Capacity() < expected)
        {
            index.Build(labels, expected);
            indexValid = true;
        }
    }

    size_t FindXor(const Word *label, const Word *mask) const noexcept
    {
        return index.FindXor(labels, label, mask);
    }

    static double L1(const Complex &value) noexcept
    {
        return std::abs(value.real()) + std::abs(value.imag());
    }

    static bool OddParity(const Word *left, const Word *right, size_t nrWords) noexcept
    {
        Word parity = 0;
        for (size_t word = 0; word < nrWords; ++word)
            parity ^= left[word] & right[word];
        return (QC::Clifford::detail::Popcount(parity) & 1U) != 0;
    }

    static void EnsureRows(Tableau &rows, size_t count, size_t qubits)
    {
        if (rows.GetNrQubits() == qubits && rows.size() >= count)
            return;
        rows = Tableau(rows.GetNrQubits() == qubits ? std::max(count, rows.size()) : count, qubits);
    }

    template <class Row> Complex BasePhase(const Row &pauli) const
    {
        size_t nrY = 0;
        for (size_t word = 0; word < labelWords; ++word)
            nrY += QC::Clifford::detail::Popcount(pauli.X.words[word] & pauli.Z.words[word]);
        Complex phase = LocalPauliSum::IPower(unsigned(nrY % 4));
        return pauli.PhaseSign ? -phase : phase;
    }

    Complex PauliPhase(const Word *label, const Word *phaseMask, const Complex &basePhase) const noexcept
    {
        return OddParity(label, phaseMask, labelWords) ? -basePhase : basePhase;
    }

    static Complex QuarterTurnGlobalPhase(long long quarterTurns)
    {
        int phase = static_cast<int>(quarterTurns % 8);
        if (phase < 0)
            phase += 8;
        const double inverseSqrtTwo = 1.0 / std::sqrt(2.0);
        switch (phase)
        {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {inverseSqrtTwo, -inverseSqrtTwo};
        case 2:
            return {0.0, -1.0};
        case 3:
            return {-inverseSqrtTwo, -inverseSqrtTwo};
        case 4:
            return {-1.0, 0.0};
        case 5:
            return {-inverseSqrtTwo, inverseSqrtTwo};
        case 6:
            return {0.0, 1.0};
        default:
            return {inverseSqrtTwo, inverseSqrtTwo};
        }
    }

    void ApplyCliffordRotation(size_t q, char axis, long long quarterTurns)
    {
        int mapTurns = static_cast<int>(quarterTurns % 4);
        if (mapTurns < 0)
            mapTurns += 4;
        const auto globalPhase = QuarterTurnGlobalPhase(quarterTurns);
        if (mapTurns == 1)
            ApplyQuarterTurn(q, axis, false);
        else if (mapTurns == 2)
        {
            ApplyQuarterTurn(q, axis, false);
            ApplyQuarterTurn(q, axis, false);
        }
        else if (mapTurns == 3)
            ApplyQuarterTurn(q, axis, true);
        for (auto &amplitude : amplitudes)
            amplitude *= globalPhase;
    }

    void ApplyCliffordOp(const CliffordOp &op, const size_t *qubits)
    {
        const size_t first = qubits[op.first], second = qubits[op.second];
        switch (op.kind)
        {
        case CliffordOp::Kind::H:
            ApplyH(first);
            break;
        case CliffordOp::Kind::S:
            ApplyS(first);
            break;
        case CliffordOp::Kind::Sdg:
            ApplySdg(first);
            break;
        case CliffordOp::Kind::X:
            ApplyX(first);
            break;
        case CliffordOp::Kind::Y:
            ApplyY(first);
            break;
        case CliffordOp::Kind::Z:
            ApplyZ(first);
            break;
        case CliffordOp::Kind::CX:
            ApplyCX(first, second);
            break;
        case CliffordOp::Kind::CZ:
            ApplyCZ(first, second);
            break;
        case CliffordOp::Kind::QuarterTurn:
            ApplyQuarterTurn(first, op.axis, false);
            break;
        case CliffordOp::Kind::InverseQuarterTurn:
            ApplyQuarterTurn(first, op.axis, true);
            break;
        }
    }

    // QC::ExtendedStabilizer::ApplyPauliSum for every lane at once. The
    // images, generators, offsets and coset lookups are computed once from
    // the reference map; lane s multiplies term j by (-1)^<P_j, E_s>. Each
    // lane's sums use the scalar order and cancellation test.
    void ApplyPauliSum(const std::vector<std::pair<LocalPauli, Complex>> &sum, const size_t *qubits, size_t nrLocalQubits)
    {
        const size_t nrWords = labelWords;
        const size_t nrTerms = sum.size();
        const size_t generatorBase = nrTerms;
        EnsureRows(rows, nrTerms + std::min(nrTerms, kMaxGenerators), n);

        terms.clear();
        Reserve(laneCoefficients, nrTerms * lanes);
        laneCoefficients.resize(nrTerms * lanes);
        signMask.assign(words, 0);
        for (const auto &[pauli, coefficient] : sum)
        {
            const size_t term = terms.size();
            auto image = rows[term];
            image.Clear();
            for (size_t qubit = 0; qubit < nrLocalQubits; ++qubit)
            {
                const char factor = pauli.Factor(qubit);
                if (factor != 'I')
                    reference.MultiplyImageInto(image, qubits[qubit], factor);
            }
            unsigned nrY = 0;
            bool hasZ = false;
            for (size_t word = 0; word < nrWords; ++word)
            {
                nrY += QC::Clifford::detail::Popcount(image.X.words[word] & image.Z.words[word]);
                hasZ |= image.Z.words[word] != 0;
            }
            const auto phase = LocalPauliSum::IPower(nrY);
            const Complex value = (image.PhaseSign ? -coefficient : coefficient) * phase;
            terms.push_back({value, 0, hasZ});

            // <P, E> = x(P).z(E) + z(P).x(E) on the gate's qubits.
            std::fill(signMask.begin(), signMask.end(), Word(0));
            for (size_t qubit = 0; qubit < nrLocalQubits; ++qubit)
            {
                if ((pauli.x >> qubit) & 1U)
                    XorInto(signMask.data(), FrameZ(qubits[qubit]));
                if ((pauli.z >> qubit) & 1U)
                    XorInto(signMask.data(), FrameX(qubits[qubit]));
            }
            Complex *laneValues = laneCoefficients.data() + term * lanes;
            const Complex negated = -value;
            for (size_t lane = 0; lane < lanes; ++lane)
                laneValues[lane] = LaneBit(signMask.data(), lane) ? negated : value;
        }

        // Generators of the X parts in reduced echelon form (scalar code).
        std::array<size_t, kMaxGenerators> pivots{}, generatorTerms{};
        std::array<unsigned, kMaxGenerators> combinations{};
        size_t nrGenerators = 0;
        for (size_t term = 0; term < nrTerms; ++term)
        {
            auto reduced = rows[generatorBase + nrGenerators];
            const auto image = rows[term];
            std::copy_n(image.X.words, nrWords, reduced.X.words);
            unsigned combination = 0;
            for (size_t g = 0; g < nrGenerators; ++g)
                if (reduced.X[pivots[g]])
                {
                    const auto other = rows[generatorBase + g];
                    for (size_t word = 0; word < nrWords; ++word)
                        reduced.X.words[word] ^= other.X.words[word];
                    combination ^= combinations[g];
                }
            size_t pivot = 0;
            if (!QC::Clifford::detail::FindPivot(reduced, pivot))
            {
                terms[term].offset = combination;
                continue;
            }
            combination ^= 1U << nrGenerators;
            for (size_t g = 0; g < nrGenerators; ++g)
            {
                auto other = rows[generatorBase + g];
                if (!other.X[pivot])
                    continue;
                for (size_t word = 0; word < nrWords; ++word)
                    other.X.words[word] ^= reduced.X.words[word];
                combinations[g] ^= combination;
            }
            pivots[nrGenerators] = pivot;
            combinations[nrGenerators] = combination;
            generatorTerms[nrGenerators] = term;
            terms[term].offset = size_t(1) << nrGenerators;
            ++nrGenerators;
        }

        const size_t nrOffsets = size_t(1) << nrGenerators;
        EnsureRows(offsets, nrOffsets, n);
        for (size_t u = 0; u < nrOffsets; ++u)
        {
            auto offset = offsets[u];
            std::fill_n(offset.X.words, nrWords, Word(0));
            for (size_t g = 0; g < nrGenerators; ++g)
                if ((u >> g) & 1U)
                {
                    const auto generator = rows[generatorTerms[g]];
                    for (size_t word = 0; word < nrWords; ++word)
                        offset.X.words[word] ^= generator.X.words[word];
                }
        }
        offsetParity.assign(nrTerms * nrOffsets, 0);
        for (size_t term = 0; term < nrTerms; ++term)
            if (terms[term].hasZ)
                for (size_t u = 1; u < nrOffsets; ++u)
                    offsetParity[term * nrOffsets + u] = OddParity(rows[term].Z.words, offsets[u].X.words, nrWords);

        const double tolerance = 8.0 * double(nrTerms + 1) * std::numeric_limits<double>::epsilon();
        std::array<double, kMaxTerms> magnitudes;
        double diagonalBound = 0.0;
        for (size_t term = 0; term < nrTerms; ++term)
        {
            magnitudes[term] = L1(terms[term].coefficient);
            diagonalBound += magnitudes[term];
        }
        const size_t originalSize = labels.size();

        if (nrGenerators == 0)
        {
            // A diagonal sum rescales every amplitude in place.
            std::array<bool, kMaxTerms> odd;
            for (size_t component = 0; component < originalSize; ++component)
            {
                const Word *label = labels.LabelWords(component);
                for (size_t term = 0; term < nrTerms; ++term)
                    odd[term] = terms[term].hasZ && OddParity(rows[term].Z.words, label, nrWords);
                Complex *row = amplitudes.data() + component * lanes;
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    Complex factor(0.0, 0.0);
                    for (size_t term = 0; term < nrTerms; ++term)
                    {
                        const Complex c = laneCoefficients[term * lanes + lane];
                        factor += odd[term] ? -c : c;
                    }
                    row[lane] = L1(factor) <= tolerance * diagonalBound ? Complex(0.0, 0.0) : row[lane] * factor;
                }
            }
            Prune();
            return;
        }

        std::array<size_t, kMaxTerms> termOffsets;
        for (size_t term = 0; term < nrTerms; ++term)
            termOffsets[term] = terms[term].offset;
        Reserve(gathered, nrOffsets * lanes);
        Reserve(results, nrOffsets * lanes);
        gathered.resize(nrOffsets * lanes);
        gatheredL1.resize(nrOffsets * lanes);
        results.resize(nrOffsets * lanes);
        bounds.resize(lanes);
        std::array<size_t, kMaxOffsets> members;
        std::array<bool, kMaxTerms> labelParity;

        EnsureIndex(2 * originalSize);
        visited.assign(originalSize, 0);
        label.resize(nrWords);
        bool appended = false;
        for (size_t component = 0; component < originalSize; ++component)
        {
            if (visited[component])
                continue;
            std::copy_n(labels.LabelWords(component), nrWords, label.data());
            for (size_t u = 0; u < nrOffsets; ++u)
            {
                const size_t member = u == 0 ? component : FindXor(label.data(), offsets[u].X.words);
                members[u] = member;
                Complex *destination = gathered.data() + u * lanes;
                double *destinationL1 = gatheredL1.data() + u * lanes;
                if (member == kNotFound)
                {
                    std::fill_n(destination, lanes, Complex(0.0, 0.0));
                    std::fill_n(destinationL1, lanes, 0.0);
                    continue;
                }
                visited[member] = 1;
                const Complex *source = amplitudes.data() + member * lanes;
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    destination[lane] = source[lane];
                    destinationL1[lane] = L1(source[lane]);
                }
            }
            for (size_t term = 0; term < nrTerms; ++term)
                labelParity[term] = terms[term].hasZ && OddParity(rows[term].Z.words, label.data(), nrWords);

            // out[v] = sum_j c_j (-1)^(z_j . (L xor offset_u)) in[u], u = v xor offset_j.
            for (size_t v = 0; v < nrOffsets; ++v)
            {
                Complex *out = results.data() + v * lanes;
                std::fill_n(out, lanes, Complex(0.0, 0.0));
                std::fill(bounds.begin(), bounds.end(), 0.0);
                for (size_t term = 0; term < nrTerms; ++term)
                {
                    const size_t u = v ^ termOffsets[term];
                    if (members[u] == kNotFound)
                        continue;
                    const bool odd = labelParity[term] != bool(offsetParity[term * nrOffsets + u]);
                    const Complex *c = laneCoefficients.data() + term * lanes;
                    const Complex *in = gathered.data() + u * lanes;
                    const double *inL1 = gatheredL1.data() + u * lanes;
                    const double magnitude = magnitudes[term];
                    if (odd)
                        for (size_t lane = 0; lane < lanes; ++lane)
                            out[lane] += -c[lane] * in[lane];
                    else
                        for (size_t lane = 0; lane < lanes; ++lane)
                            out[lane] += c[lane] * in[lane];
                    for (size_t lane = 0; lane < lanes; ++lane)
                        bounds[lane] += magnitude * inL1[lane];
                }
                for (size_t lane = 0; lane < lanes; ++lane)
                    if (L1(out[lane]) <= tolerance * bounds[lane])
                        out[lane] = Complex(0.0, 0.0);
            }

            for (size_t v = 0; v < nrOffsets; ++v)
            {
                const Complex *out = results.data() + v * lanes;
                if (members[v] != kNotFound)
                {
                    std::copy_n(out, lanes, amplitudes.data() + members[v] * lanes);
                    continue;
                }
                bool nonzero = false;
                for (size_t lane = 0; lane < lanes && !nonzero; ++lane)
                    nonzero = out[lane] != Complex(0.0, 0.0);
                if (!nonzero)
                    continue;
                // Grow by half at a time; the old matrix stays live until
                // copied. Growth follows what this batch counted, never the
                // capacity a reused buffer kept from an earlier batch.
                const size_t held = reserved.amplitudes;
                if (amplitudes.size() + lanes > held)
                    Reserve(amplitudes, std::max(amplitudes.size() + lanes, held + held / 2));
                labels.AppendXor(label.data(), offsets[v].X.words);
                amplitudes.insert(amplitudes.end(), out, out + lanes);
                appended = true;
            }
        }
        if (appended)
            indexValid = false;
        Prune();
    }

    // Exact mode: drop labels that are zero in every lane.
    void Prune()
    {
        const size_t originalSize = labels.size();
        size_t write = 0;
        for (size_t component = 0; component < originalSize; ++component)
        {
            const Complex *row = amplitudes.data() + component * lanes;
            bool live = false;
            for (size_t lane = 0; lane < lanes && !live; ++lane)
                live = row[lane].real() != 0.0 || row[lane].imag() != 0.0;
            if (!live)
                continue;
            if (write != component)
            {
                std::copy_n(row, lanes, amplitudes.data() + write * lanes);
                labels.CopyLabel(write, component);
            }
            ++write;
        }
        if (write == 0)
            throw std::runtime_error("A frame operation cancelled every component");
        if (write != originalSize)
        {
            amplitudes.resize(write * lanes);
            labels.resize(write);
            indexValid = false;
        }
    }

    bool DiagonalOutcome(const Word *componentLabel, const Word *phaseMask, const Complex &basePhase) const noexcept
    {
        bool outcome = basePhase.real() < 0.0;
        for (size_t word = 0; word < labelWords; ++word)
            if ((QC::Clifford::detail::Popcount(phaseMask[word] & componentLabel[word]) & 1U) != 0)
                outcome = !outcome;
        return outcome;
    }

    template <class Choose> void ChooseOutcomes(Choose &choose)
    {
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            const double total = probabilityZero[lane] + probabilityOne[lane];
            if (!(total > 0.0))
                throw std::runtime_error("Cannot measure a frame with zero total probability");
            // The draw decides the physical outcome; the logical one reads it
            // through the X frame bit held before the measurement.
            const bool flipped = LaneBit(measuredX.data(), lane);
            const double physicalOne = std::max(0.0, std::min(1.0, (flipped ? probabilityZero[lane] : probabilityOne[lane]) / total));
            const bool outcome = choose(lane, physicalOne) != flipped;
            const double outcomeProbability = outcome ? probabilityOne[lane] : probabilityZero[lane];
            if (!(outcomeProbability > 0.0))
                throw std::runtime_error("Cannot normalize a zero-probability measurement outcome");
            inverseNorms[lane] = 1.0 / std::sqrt(outcomeProbability);
            if (outcome)
                logicalOutcomes[lane / 64] |= Word(1) << (lane % 64);
        }
    }

    template <class Choose> void MeasureDiagonal(const Word *phaseMask, const Complex &basePhase, Choose &choose)
    {
        const size_t components = labels.size();
        componentOutcome.resize(components);
        probabilityZero.assign(lanes, 0.0);
        probabilityOne.assign(lanes, 0.0);
        inverseNorms.resize(lanes);
        for (size_t c = 0; c < components; ++c)
        {
            const bool one = DiagonalOutcome(labels.LabelWords(c), phaseMask, basePhase);
            componentOutcome[c] = one;
            auto &probability = one ? probabilityOne : probabilityZero;
            const Complex *row = amplitudes.data() + c * lanes;
            for (size_t lane = 0; lane < lanes; ++lane)
                probability[lane] += std::norm(row[lane]);
        }
        ChooseOutcomes(choose);
        bool anyOne = false, anyZero = false;
        for (size_t w = 0; w < words; ++w)
        {
            anyOne = anyOne || logicalOutcomes[w] != 0;
            anyZero = anyZero || (~logicalOutcomes[w] & ActiveMask(w)) != 0;
        }
        if (!anyOne || !anyZero)
        {
            // Every lane has the same outcome: their supports stay aligned.
            for (size_t c = 0; c < components; ++c)
            {
                Complex *row = amplitudes.data() + c * lanes;
                if (componentOutcome[c] != anyOne)
                    std::fill_n(row, lanes, Complex(0.0, 0.0));
                else
                    for (size_t lane = 0; lane < lanes; ++lane)
                        row[lane] *= inverseNorms[lane];
            }
            Prune();
            return;
        }

        // Lanes with logical outcome one keep their amplitudes on labels b with
        // z.b odd (Q = +-Z^z); the others on labels with z.b even. Any d with
        // z.d = 1 maps the first half onto the second: outcome-one lanes move
        // each amplitude from b to b xor d and take U X^d U^dagger into their
        // frames, so every lane ends on outcome-zero labels. d is chosen to land
        // on the labels the outcome-zero lanes occupy, so the union stays as
        // small as one lane's support instead of growing with each random
        // outcome (mid-circuit syndrome measurements, for example).
        live.assign(components, 0);
        for (size_t c = 0; c < components; ++c)
        {
            const Complex *row = amplitudes.data() + c * lanes;
            const bool one = componentOutcome[c];
            for (size_t lane = 0; lane < lanes; ++lane)
                if (LaneBit(logicalOutcomes.data(), lane) == one && (row[lane].real() != 0.0 || row[lane].imag() != 0.0))
                {
                    live[c] = 1;
                    break;
                }
        }
        componentTarget.assign(components, kNotFound);
        nextLabels.clear();
        for (size_t c = 0; c < components; ++c)
            if (live[c] && !componentOutcome[c])
            {
                componentTarget[c] = nextLabels.size();
                nextLabels.Append(labels.LabelWords(c));
            }
        nextIndex.Build(nextLabels, components);
        ChooseFold(phaseMask, components);
        for (size_t c = 0; c < components; ++c)
            if (live[c] && componentOutcome[c])
            {
                size_t target = nextIndex.FindXor(nextLabels, labels.LabelWords(c), foldMask.data());
                if (target == kNotFound)
                {
                    target = nextLabels.size();
                    nextLabels.AppendXor(labels.LabelWords(c), foldMask.data());
                    nextIndex.Insert(nextLabels, target);
                }
                componentTarget[c] = target;
            }
        Reserve(nextAmplitudes, nextLabels.size() * lanes);
        nextAmplitudes.assign(nextLabels.size() * lanes, Complex(0.0, 0.0));
        for (size_t c = 0; c < components; ++c)
        {
            if (componentTarget[c] == kNotFound)
                continue;
            const Complex *row = amplitudes.data() + c * lanes;
            Complex *destination = nextAmplitudes.data() + componentTarget[c] * lanes;
            const bool one = componentOutcome[c];
            for (size_t lane = 0; lane < lanes; ++lane)
                if (LaneBit(logicalOutcomes.data(), lane) == one)
                    destination[lane] = row[lane] * inverseNorms[lane];
        }
        // U X^d U^dagger: X part (U^dagger Z_k U).Z . d, Z part (U^dagger X_k U).Z . d.
        for (size_t k = 0; k < n; ++k)
        {
            if (OddParity(reference.ImageZ(k).Z.words, foldMask.data(), labelWords))
                XorInto(FrameX(k), logicalOutcomes.data());
            if (OddParity(reference.ImageX(k).Z.words, foldMask.data(), labelWords))
                XorInto(FrameZ(k), logicalOutcomes.data());
        }
        labels.swap(nextLabels);
        SwapAmplitudes();
        indexValid = false;
        Prune();
    }

    // The fold d with z.d = 1 that maps the most outcome-one labels (a sample)
    // onto live outcome-zero labels (nextLabels, indexed by nextIndex).
    // Candidates: each single bit of z and the differences between one
    // outcome-one label and a few outcome-zero labels.
    void ChooseFold(const Word *phaseMask, size_t components)
    {
        constexpr size_t kSample = 64, kSingleBits = 64, kDifferences = 16;
        sampleOnes.clear();
        for (size_t c = 0; c < components && sampleOnes.size() < kSample; ++c)
            if (live[c] && componentOutcome[c])
                sampleOnes.push_back(c);
        candidates.clear();
        size_t singles = 0;
        for (size_t bit = 0; bit < n && singles < kSingleBits; ++bit)
            if ((phaseMask[bit / 64] >> (bit % 64)) & 1U)
            {
                const size_t at = candidates.size();
                candidates.resize(at + labelWords, 0);
                candidates[at + bit / 64] = Word(1) << (bit % 64);
                ++singles;
            }
        if (!sampleOnes.empty())
            for (size_t c = 0; c < nextLabels.size() && c < kDifferences; ++c)
            {
                const Word *one = labels.LabelWords(sampleOnes.front());
                const Word *zero = nextLabels.LabelWords(c);
                for (size_t w = 0; w < labelWords; ++w)
                    candidates.push_back(one[w] ^ zero[w]);
            }
        size_t best = 0, bestScore = 0;
        const size_t count = candidates.size() / labelWords;
        for (size_t candidate = 0; candidate < count; ++candidate)
        {
            const Word *d = candidates.data() + candidate * labelWords;
            size_t score = 0;
            for (const size_t c : sampleOnes)
                score += nextIndex.FindXor(nextLabels, labels.LabelWords(c), d) != kNotFound;
            if (candidate == 0 || score > bestScore)
            {
                best = candidate;
                bestScore = score;
                if (score == sampleOnes.size())
                    break;
            }
        }
        foldMask.assign(candidates.begin() + best * labelWords, candidates.begin() + (best + 1) * labelWords);
    }

    template <class Row, class Choose> void MeasureOffDiagonal(size_t q, const Row &observable, const Complex &basePhase, Choose &choose)
    {
        size_t pivot = 0;
        QC::Clifford::detail::FindPivot(observable, pivot);
        const Word *flipMask = observable.X.words;
        const Word *phaseMask = observable.Z.words;
        // The gauge U' X_p U'^dagger on physical qubits, read from the old map:
        // its X part on j is (U^dagger Z_j U).X_p, its Z part (U^dagger X_j U).X_p.
        gaugeX.clear();
        gaugeZ.clear();
        for (size_t j = 0; j < n; ++j)
        {
            if (reference.ImageZ(j).X[pivot])
                gaugeX.push_back(j);
            if (reference.ImageX(j).X[pivot])
                gaugeZ.push_back(j);
        }

        const size_t components = labels.size();
        EnsureIndex(components);
        pairs.clear();
        pairs.reserve(components);
        for (size_t component = 0; component < components; ++component)
        {
            const Word *componentLabel = labels.LabelWords(component);
            const size_t partner = FindXor(componentLabel, flipMask);
            size_t root = component, target = partner;
            if (labels.Get(component, pivot))
            {
                if (partner != kNotFound)
                    continue;
                root = kNotFound;
                target = component;
            }
            pairs.emplace_back(root, target);
        }

        // New-basis coefficient (a_r + (-1)^m phase(r xor x) a_(r xor x)) / sqrt(2),
        // computed in both passes rather than stored: two more full
        // matrices would otherwise be live next to the old and new ones.
        const double inverseSqrtTwo = 1.0 / std::sqrt(2.0);
        const size_t nrPairs = pairs.size();
        const auto pairSums = [&](size_t p, size_t lane, Complex &plus, Complex &minus) {
            const auto [root, target] = pairs[p];
            const Complex a = root == kNotFound ? Complex(0.0, 0.0) : amplitudes[root * lanes + lane];
            const Complex b = target == kNotFound ? Complex(0.0, 0.0) : pairPhases[p] * amplitudes[target * lanes + lane];
            plus = inverseSqrtTwo * (a + b);
            minus = inverseSqrtTwo * (a - b);
        };
        pairPhases.resize(nrPairs);
        for (size_t p = 0; p < nrPairs; ++p)
        {
            const size_t target = pairs[p].second;
            pairPhases[p] = target == kNotFound ? Complex(0.0, 0.0) : PauliPhase(labels.LabelWords(target), phaseMask, basePhase);
        }
        probabilityZero.assign(lanes, 0.0);
        probabilityOne.assign(lanes, 0.0);
        inverseNorms.resize(lanes);
        for (size_t p = 0; p < nrPairs; ++p)
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                Complex plus, minus;
                pairSums(p, lane, plus, minus);
                probabilityZero[lane] += std::norm(plus);
                probabilityOne[lane] += std::norm(minus);
            }
        ChooseOutcomes(choose);

        nextLabels.clear();
        nextAmplitudes.clear();
        nextLabels.reserve(nrPairs);
        Reserve(nextAmplitudes, nrPairs * lanes);
        for (size_t p = 0; p < nrPairs; ++p)
        {
            bool live = false;
            const size_t start = nextAmplitudes.size();
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                Complex plus, minus;
                pairSums(p, lane, plus, minus);
                const Complex a = (LaneBit(logicalOutcomes.data(), lane) ? minus : plus) * inverseNorms[lane];
                live = live || a.real() != 0.0 || a.imag() != 0.0;
                nextAmplitudes.push_back(a);
            }
            if (!live)
            {
                nextAmplitudes.resize(start);
                continue;
            }
            const auto [root, target] = pairs[p];
            // The new label keeps the pivot zero in every lane; outcome-one
            // lanes take the gauge into their frames below.
            if (root != kNotFound)
                nextLabels.Append(labels.LabelWords(root));
            else
                nextLabels.AppendXor(labels.LabelWords(target), flipMask);
        }
        if (nextLabels.size() == 0)
            throw std::runtime_error("Measurement removed every frame component");
        for (const size_t j : gaugeX)
            XorInto(FrameX(j), logicalOutcomes.data());
        for (const size_t j : gaugeZ)
            XorInto(FrameZ(j), logicalOutcomes.data());
        reference.RebaseZ(q, pivot, false);
        labels.swap(nextLabels);
        SwapAmplitudes();
        indexValid = false;
    }

    size_t n;
    size_t labelWords;
    size_t lanes = 0;
    size_t words = 0;
    QC::CliffordBasisMap identity;
    QC::CliffordBasisMap reference;
    std::vector<Word> frameX, frameZ;
    QC::PackedComponentLabels labels, nextLabels;
    std::vector<Complex> amplitudes, nextAmplitudes;
    mutable QC::PackedComponentIndex index;
    bool indexValid = false;
    size_t amplitudeBudget;
    size_t peakEntries = 0;
    Reserved reserved;

    // Workspaces, reused across gates.
    std::vector<Word> zeroLabel, signMask, logicalOutcomes, measuredX, label, foldMask, candidates;
    std::vector<size_t> componentTarget, sampleOnes;
    std::vector<uint8_t> live;
    QC::PackedComponentIndex nextIndex;
    std::vector<std::pair<LocalPauli, Complex>> axisTerms;
    std::vector<Term> terms;
    std::vector<Complex> laneCoefficients, gathered, results, pairPhases;
    std::vector<double> gatheredL1, bounds, probabilityZero, probabilityOne, inverseNorms;
    std::vector<uint8_t> offsetParity, visited, componentOutcome;
    std::vector<std::pair<size_t, size_t>> pairs;
    std::vector<size_t> gaugeX, gaugeZ;
    Tableau rows, offsets;
};
} // namespace ExtendedFrames
} // namespace MaestroExecution
