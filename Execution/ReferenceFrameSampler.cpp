#include "ReferenceFrameSampler.h"

#include "../Circuit/Circuit.h"
#include "../Utils/RandomStream.h"
#include "Clifford.h"
#include "PauliFrameKernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MaestroExecution
{
namespace
{
// Only compilation uses a tableau. A random Z measurement has a stabilizer
// gauge U Z_p U^dagger that exchanges its two equiprobable branches. The
// inverse tableau gives that physical Pauli by a symplectic transpose:
// X_i = (U^dagger Z_i U).X_p, Z_i = (U^dagger X_i U).X_p.
// Global Pauli phases have no effect on any trajectory's observations.
class ReferenceTableau : public QC::Clifford::StabilizerSimulator
{
  public:
    explicit ReferenceTableau(size_t qubits) : StabilizerSimulator(qubits)
    {
        SetMultithreading(false);
    }

    bool MeasureReference(size_t qubit, bool &random, std::vector<size_t> &gaugeX, std::vector<size_t> &gaugeZ)
    {
        ValidateQubit(qubit);
        size_t pivot = 0;
        random = IsRandomResult(qubit, pivot);
        if (!random)
            return MeasureQubit(qubit);
        for (size_t q = 0; q < getNrQubits(); ++q)
        {
            if ((inverseZ[q].X.words[pivot / 64] >> (pivot % 64)) & 1U)
                gaugeX.push_back(q);
            if ((inverseX[q].X.words[pivot / 64] >> (pivot % 64)) & 1U)
                gaugeZ.push_back(q);
        }
        CollapseRandomQubit(qubit, pivot, false);
        return false;
    }
};

using FrameKernels::ForEachError;
using FrameKernels::PauliMixture;
using FrameKernels::Stream;
using FrameKernels::WriteBits;

template <class Rate> Rate MakeRate(double rate)
{
    return FrameKernels::MakeRate<Rate>(rate);
}

template <class Consumer> void TransposeBatch(const std::vector<uint64_t> &classical, size_t width, size_t lanes, Consumer &&consume)
{
    FrameKernels::TransposeBatch(classical, width, lanes, ReferenceFrameSampler::kBatchWords, std::forward<Consumer>(consume));
}
} // namespace

std::optional<ReferenceFrameSampler> CompileReferenceSampler(const std::shared_ptr<Circuits::Circuit<double>> &circuit, size_t numQubits,
                                                             size_t numClassicalBits)
{
    if (!circuit)
        throw std::invalid_argument("Reference-frame sampling requires a circuit");
    // The tableau is quadratic and the frames linear in their qubit count.
    // Keep rows only for the qubits the circuit touches, numbered in
    // increasing order as the network executor compacts circuits. Random
    // streams are keyed by instruction, so seeded rows do not depend on the
    // labels. The operations are never copied: compile() translates each
    // index it uses (row), so unsupported operations still fall back before
    // anything but AffectedQubits() touches them, and null operations and
    // qubits beyond numQubits are rejected in circuit order as before.
    std::set<size_t> used;
    std::function<void(const Circuits::Circuit<double> &)> collect = [&](const Circuits::Circuit<double> &current) {
        for (const auto &op : current.GetOperations())
        {
            if (!op)
                continue;
            if (op->GetType() == Circuits::OperationType::kComposite)
                collect(*std::static_pointer_cast<Circuits::Circuit<double>>(op));
            else
                for (const auto q : op->AffectedQubits())
                    used.insert(q);
        }
    };
    collect(*circuit);
    std::vector<size_t> rows;
    size_t frameQubits = numQubits;
    if (used.size() < numQubits)
    {
        rows.assign(used.begin(), used.end());
        frameQubits = std::max<size_t>(1, rows.size());
    }
    const auto row = [&rows](size_t qubit) {
        if (rows.empty())
            return qubit;
        const auto found = std::lower_bound(rows.begin(), rows.end(), qubit);
        if (found == rows.end() || *found != qubit)
            throw std::logic_error("Reference-frame qubit missing from the compacted register");
        return static_cast<size_t>(found - rows.begin());
    };
    ReferenceFrameSampler plan;
    plan.qubits_ = numQubits;
    plan.frameQubits_ = frameQubits;
    plan.width_ = numClassicalBits;
    ReferenceTableau reference(frameQubits);
    bool hasMeasurement = false;
    using Kind = ReferenceFrameSampler::Kind;
    using Instruction = ReferenceFrameSampler::Instruction;
    using ErrorRate = ReferenceFrameSampler::ErrorRate;
    using PauliTerm = ReferenceFrameSampler::PauliTerm;
    using Type = Circuits::OperationType;
    using Gate = Circuits::QuantumGateType;

    auto measurement = [&](size_t qubit, size_t bit, const Circuits::ReadoutRates &rates, bool reset) {
        Instruction instruction{reset ? Kind::Reset : Kind::Measurement};
        instruction.first = qubit;
        instruction.second = bit;
        instruction.referenceOne = reference.MeasureReference(qubit, instruction.random, instruction.gaugeX, instruction.gaugeZ);
        instruction.readout01 = rates.p_meas1_prep0;
        instruction.readout10 = rates.p_meas0_prep1;
        instruction.readout = MakeRate<ErrorRate>(std::max(instruction.readout01, instruction.readout10));
        if (reset)
        {
            if (instruction.referenceOne)
                reference.ApplyX(qubit);
        }
        else
            hasMeasurement = true;
        plan.instructions_.push_back(std::move(instruction));
    };

    std::function<bool(const Circuits::Circuit<double> &)> compile = [&](const Circuits::Circuit<double> &current) {
        for (const auto &op : current.GetOperations())
        {
            if (!op)
                throw std::invalid_argument("Reference-frame circuit contains a null operation");
            for (const auto q : op->AffectedQubits())
                if (q >= numQubits)
                    throw std::out_of_range("Reference-frame qubit exceeds the quantum register");
            for (const auto b : op->AffectedBits())
                if (b >= numClassicalBits)
                    throw std::out_of_range("Reference-frame bit exceeds the classical register");
            switch (op->GetType())
            {
            case Type::kComposite:
                if (!compile(*std::static_pointer_cast<Circuits::Circuit<double>>(op)))
                    return false;
                break;
            case Type::kNoOp:
            case Type::kDelay:
                break;
            case Type::kMeasurement: {
                const auto measured = std::static_pointer_cast<Circuits::MeasurementOperation<double>>(op);
                const auto &qubits = measured->GetQubits();
                const auto &bits = measured->GetBitsIndices();
                const auto &rates = measured->GetReadout();
                if (qubits.size() != bits.size())
                    throw std::invalid_argument("Reference-frame measurement has mismatched qubits and bits");
                for (size_t i = 0; i < qubits.size(); ++i)
                    measurement(row(qubits[i]), bits[i], i < rates.size() ? rates[i] : Circuits::ReadoutRates{}, false);
                break;
            }
            case Type::kReset: {
                const auto reset = std::static_pointer_cast<Circuits::Reset<double>>(op);
                const auto &targets = reset->GetResetTargets();
                for (size_t i = 0; i < reset->GetQubits().size(); ++i)
                    measurement(row(reset->GetQubits()[i]), 0, {}, true);
                // Reset::Execute first resets the complete list to zero, then
                // applies target flips. Retain that ordering for repeated qubits.
                for (size_t i = 0; i < targets.size() && i < reset->GetQubits().size(); ++i)
                    if (targets[i])
                        reference.ApplyX(row(reset->GetQubits()[i]));
                break;
            }
            case Type::kQuantumChannel: {
                const auto channel = std::static_pointer_cast<Circuits::QuantumChannelOperation<double>>(op);
                // Pauli terms below decode one or two targets (index = first + 4 * second).
                // A Kraus-only channel qualifies when it is exactly a Pauli mixture.
                std::vector<double> mixture;
                const std::vector<double> *found = &channel->GetChannel().GetPauliProbabilities();
                if (found->empty())
                {
                    mixture = PauliMixture(channel->GetChannel(), channel->AffectedQubits().size());
                    found = &mixture;
                }
                const auto &probabilities = *found;
                if (probabilities.empty() || probabilities.size() > 16)
                    return false;
                Instruction instruction{Kind::Channel};
                const auto targets = channel->AffectedQubits();
                instruction.first = row(targets[0]);
                instruction.second = targets.size() == 2 ? row(targets[1]) : instruction.first;
                double rate = 0;
                for (size_t i = 1; i < probabilities.size(); ++i)
                    rate += probabilities[i];
                if (!(rate > 0))
                    break; // The identity channel draws nothing.
                double cumulative = 0;
                for (size_t i = 1; i < probabilities.size(); ++i)
                {
                    if (!(probabilities[i] > 0))
                        continue;
                    cumulative += probabilities[i] / rate;
                    const size_t firstPauli = i % 4, secondPauli = i / 4;
                    PauliTerm term;
                    term.cumulative = cumulative;
                    term.firstX = firstPauli == 1 || firstPauli == 2;
                    term.firstZ = firstPauli == 2 || firstPauli == 3;
                    term.secondX = secondPauli == 1 || secondPauli == 2;
                    term.secondZ = secondPauli == 2 || secondPauli == 3;
                    instruction.terms.push_back(term);
                }
                instruction.terms.back().cumulative = 1; // Absorb rounding in the sum.
                instruction.error = MakeRate<ErrorRate>(rate);
                plan.instructions_.push_back(std::move(instruction));
                break;
            }
            case Type::kGate: {
                const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<double>>(op);
                const size_t a = row(gate->GetQubit(0));
                const size_t b = gate->GetNumQubits() > 1 ? row(gate->GetQubit(1)) : a;
                Instruction instruction{Kind::H};
                instruction.first = a;
                instruction.second = b;
                switch (gate->GetGateType())
                {
                case Gate::kXGateType:
                    reference.ApplyX(a);
                    continue;
                case Gate::kYGateType:
                    reference.ApplyY(a);
                    continue;
                case Gate::kZGateType:
                    reference.ApplyZ(a);
                    continue;
                case Gate::kHadamardGateType:
                    reference.ApplyH(a);
                    instruction.kind = Kind::H;
                    break;
                case Gate::kSGateType:
                    reference.ApplyS(a);
                    instruction.kind = Kind::S;
                    break;
                case Gate::kSdgGateType:
                    reference.ApplySdg(a);
                    instruction.kind = Kind::S;
                    break;
                case Gate::kSxGateType:
                    reference.ApplySx(a);
                    instruction.kind = Kind::Sx;
                    break;
                case Gate::kSxDagGateType:
                    reference.ApplySxDag(a);
                    instruction.kind = Kind::Sx;
                    break;
                case Gate::kKGateType:
                    reference.ApplyK(a);
                    instruction.kind = Kind::Sx;
                    break;
                case Gate::kCXGateType:
                    reference.ApplyCX(b, a);
                    instruction.kind = Kind::CX;
                    break;
                case Gate::kCYGateType:
                    reference.ApplyCY(b, a);
                    instruction.kind = Kind::CY;
                    break;
                case Gate::kCZGateType:
                    reference.ApplyCZ(b, a);
                    instruction.kind = Kind::CZ;
                    break;
                case Gate::kSwapGateType:
                    reference.ApplySwap(a, b);
                    instruction.kind = Kind::Swap;
                    break;
                default:
                    return false;
                }
                plan.instructions_.push_back(std::move(instruction));
                break;
            }
            default:
                return false;
            }
        }
        return true;
    };
    if (!compile(*circuit) || !hasMeasurement)
        return std::nullopt;
    return plan;
}

std::vector<uint8_t> ReferenceFrameSampler::Sample(size_t shots, uint64_t seed) const
{
    return Sample(shots, Seeds{seed, seed});
}

std::vector<uint8_t> ReferenceFrameSampler::Sample(size_t shots, Seeds seeds) const
{
    if (width_ != 0 && shots > std::numeric_limits<size_t>::max() / width_)
        throw std::length_error("Reference-frame sample matrix exceeds addressable memory");
    std::vector<uint8_t> result(shots * width_);
    ForEachBatch(shots, seeds, [&](size_t start, size_t lanes, const std::vector<uint64_t> &classical) {
        TransposeBatch(classical, width_, lanes, [&](size_t first, size_t base, size_t count, size_t laneCount, const auto &block) {
            for (size_t lane = 0; lane < laneCount; ++lane)
                WriteBits(block[lane], result.data() + (start + first + lane) * width_ + base, count);
        });
    });
    return result;
}

FrameOutcomes ReferenceFrameSampler::SampleCounts(size_t shots, uint64_t seed) const
{
    return SampleCounts(shots, Seeds{seed, seed});
}

FrameOutcomes ReferenceFrameSampler::SampleCounts(size_t shots, Seeds seeds) const
{
    FrameOutcomes result;
    if (!shots)
        return result;
    const size_t rowWords = width_ / 64 + (width_ % 64 != 0);
    const size_t batchLanes = std::min(shots, kBatchShots);
    if (rowWords != 0 && batchLanes > std::numeric_limits<size_t>::max() / rowWords)
        throw std::length_error("Reference-frame count scratch exceeds addressable memory");
    std::vector<uint64_t> rows(batchLanes * rowWords), key(rowWords);
    std::unordered_map<std::vector<uint64_t>, size_t, FrameDetail::PackedRowHash> packed;
    ForEachBatch(shots, seeds, [&](size_t, size_t lanes, const std::vector<uint64_t> &classical) {
        TransposeBatch(classical, width_, lanes, [&](size_t first, size_t base, size_t, size_t laneCount, const auto &block) {
            for (size_t lane = 0; lane < laneCount; ++lane)
                rows[(first + lane) * rowWords + base / 64] = block[lane];
        });
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            std::copy_n(rows.begin() + lane * rowWords, rowWords, key.begin());
            ++packed[key];
        }
    });
    result.reserve(packed.size());
    while (!packed.empty())
    {
        auto row = packed.extract(packed.begin());
        std::vector<bool> bits(width_);
        for (size_t bit = 0; bit < width_; ++bit)
            bits[bit] = ((row.key()[bit / 64] >> (bit % 64)) & 1U) != 0;
        result.emplace_back(std::move(bits), row.mapped());
    }
    return result;
}

void ReferenceFrameSampler::ForEachBatch(size_t shots, Seeds seeds, const std::function<void(size_t, size_t, const std::vector<uint64_t> &)> &consume) const
{
    if (!shots)
        return;
    constexpr size_t W = kBatchWords;
    if (frameQubits_ > std::numeric_limits<size_t>::max() / W || width_ > std::numeric_limits<size_t>::max() / W)
        throw std::length_error("Reference-frame batch dimensions exceed addressable memory");
    // Frame rows are qubit-major: x[q * W + w] holds lanes 64w..64w+63 of qubit q.
    std::vector<uint64_t> x(frameQubits_ * W), z(frameQubits_ * W), classical(width_ * W);
    std::array<uint64_t, W> active{}, gauge{};
    for (size_t start = 0, batch = 0; start < shots; ++batch)
    {
        const size_t lanes = std::min(kBatchShots, shots - start);
        const size_t words = (lanes + 63) / 64;
        for (size_t w = 0; w < W; ++w)
        {
            const size_t first = 64 * w;
            active[w] = first + 64 <= lanes ? ~uint64_t(0) : first < lanes ? (uint64_t(1) << (lanes - first)) - 1 : 0;
        }
        std::fill(x.begin(), x.end(), 0);
        std::fill(z.begin(), z.end(), 0);
        std::fill(classical.begin(), classical.end(), 0);
        // Gauges and readout draw from the measurement seed, channel errors
        // from the noise seed, with the streams one seed used for both.
        const uint64_t measurementSeed = Utils::RandomStream::Derive(seeds.measurement, batch);
        const uint64_t noiseSeed = Utils::RandomStream::Derive(seeds.noise, batch);

        for (size_t index = 0; index < instructions_.size(); ++index)
        {
            const auto &instruction = instructions_[index];
            uint64_t *xa = x.data() + instruction.first * W, *za = z.data() + instruction.first * W;
            switch (instruction.kind)
            {
            case Kind::H:
                for (size_t w = 0; w < words; ++w)
                    std::swap(xa[w], za[w]);
                break;
            case Kind::S:
                for (size_t w = 0; w < words; ++w)
                    za[w] ^= xa[w];
                break;
            case Kind::Sx:
                for (size_t w = 0; w < words; ++w)
                    xa[w] ^= za[w];
                break;
            case Kind::CX: {
                uint64_t *xb = x.data() + instruction.second * W, *zb = z.data() + instruction.second * W;
                for (size_t w = 0; w < words; ++w)
                {
                    xb[w] ^= xa[w];
                    za[w] ^= zb[w];
                }
                break;
            }
            case Kind::CY: {
                uint64_t *xb = x.data() + instruction.second * W, *zb = z.data() + instruction.second * W;
                for (size_t w = 0; w < words; ++w)
                {
                    zb[w] ^= xb[w];
                    xb[w] ^= xa[w];
                    za[w] ^= zb[w];
                    zb[w] ^= xb[w];
                }
                break;
            }
            case Kind::CZ: {
                uint64_t *xb = x.data() + instruction.second * W, *zb = z.data() + instruction.second * W;
                for (size_t w = 0; w < words; ++w)
                {
                    zb[w] ^= xa[w];
                    za[w] ^= xb[w];
                }
                break;
            }
            case Kind::Swap: {
                uint64_t *xb = x.data() + instruction.second * W, *zb = z.data() + instruction.second * W;
                for (size_t w = 0; w < words; ++w)
                {
                    std::swap(xa[w], xb[w]);
                    std::swap(za[w], zb[w]);
                }
                break;
            }
            case Kind::Measurement:
            case Kind::Reset: {
                if (instruction.random)
                {
                    Stream stream(Utils::RandomStream::Derive(measurementSeed, 2 * index));
                    for (size_t w = 0; w < words; ++w)
                        gauge[w] = stream.Next() & active[w];
                    for (const auto q : instruction.gaugeX)
                        for (size_t w = 0; w < words; ++w)
                            x[q * W + w] ^= gauge[w];
                    for (const auto q : instruction.gaugeZ)
                        for (size_t w = 0; w < words; ++w)
                            z[q * W + w] ^= gauge[w];
                }
                if (instruction.kind == Kind::Reset)
                {
                    // Both the reference and every trajectory have the requested
                    // reset state. Its local Pauli frame is now immaterial.
                    for (size_t w = 0; w < words; ++w)
                        xa[w] = za[w] = 0;
                    break;
                }
                uint64_t *values = classical.data() + instruction.second * W;
                const uint64_t reference = instruction.referenceOne ? ~uint64_t(0) : 0;
                for (size_t w = 0; w < words; ++w)
                    values[w] = (xa[w] ^ reference) & active[w];
                if (instruction.readout.rate > 0)
                {
                    Stream stream(Utils::RandomStream::Derive(measurementSeed, 2 * index + 1));
                    const double candidate = instruction.readout.rate;
                    ForEachError(stream, candidate, instruction.readout.inverseLogSurvival, lanes, [&](size_t lane) {
                        const uint64_t bit = uint64_t(1) << (lane % 64);
                        uint64_t &word = values[lane / 64];
                        const double keep = (word & bit) ? instruction.readout10 : instruction.readout01;
                        if (keep >= candidate || (keep > 0 && stream.Uniform() * candidate < keep))
                            word ^= bit;
                    });
                }
                break;
            }
            case Kind::Channel: {
                uint64_t *xb = x.data() + instruction.second * W, *zb = z.data() + instruction.second * W;
                Stream stream(Utils::RandomStream::Derive(noiseSeed, 2 * index + 1));
                const PauliTerm *terms = instruction.terms.data();
                const PauliTerm *last = terms + instruction.terms.size() - 1;
                ForEachError(stream, instruction.error.rate, instruction.error.inverseLogSurvival, lanes, [&](size_t lane) {
                    const PauliTerm *term = terms;
                    if (term != last)
                        for (const double draw = stream.Uniform(); term != last && draw >= term->cumulative;)
                            ++term;
                    const size_t w = lane / 64;
                    const uint64_t bit = uint64_t(1) << (lane % 64);
                    if (term->firstX)
                        xa[w] ^= bit;
                    if (term->firstZ)
                        za[w] ^= bit;
                    if (term->secondX)
                        xb[w] ^= bit;
                    if (term->secondZ)
                        zb[w] ^= bit;
                });
                break;
            }
            }
        }

        consume(start, lanes, classical);
        start += lanes;
    }
}
} // namespace MaestroExecution
