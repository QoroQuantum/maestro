#pragma once

// Noisy extended-stabilizer sampling with batched Pauli frames.
//
// CompileExtendedFrameSampler translates a circuit with Pauli noise channels
// (inject_exact_noise output plus attached readout) into a plan that a
// LaneBatch executes for many shots at once. Shots run in batches of lanes;
// every lane is one shot. Noise follows the realization schedule of
// ExecuteNoisy: shots are split into contiguous realizations, all shots of a
// realization see the same circuit errors, and each shot has its own
// measurement outcomes and readout errors. With one realization per shot,
// errors are drawn by geometric skipping, as ReferenceFrameSampler does,
// over fixed blocks of shots that each have their own stream. With shared
// realizations, a realization's draws come from its own stream (so they agree
// in every batch it spans), and the batch evolves one lane per realization
// until the first measurement or reset, then copies it into one lane per
// shot. Every draw belongs to a shot, a block of shots or a realization,
// never to a batch, and a measurement draws the shot's physical outcome (its
// frame depends on the other lanes of the batch, its state does not), so the
// rows do not depend on how shots are batched.
//
// Header-only: Maestro's libraries and Composer's build of them need no new
// translation unit.

#include "ExtendedFrameBatch.h"
#include "PauliFrameKernels.h"
#include "ReferenceFrameSampler.h"

#include "../Circuit/Circuit.h"
#include "../Utils/RandomStream.h"
#include "../Utils/Threads/RealizationWorkers.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace MaestroExecution
{
class ExtendedFrameSampler;
inline std::optional<ExtendedFrameSampler> CompileExtendedFrameSampler(const std::shared_ptr<Circuits::Circuit<double>> &circuit, size_t numQubits,
                                                                       size_t numClassicalBits);

class ExtendedFrameSampler
{
  public:
    static constexpr size_t kDefaultBatchLanes = 256;
    static constexpr size_t kDefaultAmplitudeBudget = size_t(512) << 20;

    size_t Width() const noexcept
    {
        return width_;
    }

    size_t Qubits() const noexcept
    {
        return qubits_;
    }

    size_t FrameQubits() const noexcept
    {
        return frameQubits_;
    }

    // Lanes per batch (a multiple of 64 is most efficient) and the amplitude
    // memory one batch may use. Exceeding it throws
    // ExtendedFrames::MemoryLimitExceeded.
    void SetBatchLanes(size_t lanes)
    {
        if (!lanes)
            throw std::invalid_argument("A batch needs at least one lane");
        batchLanes_ = lanes;
    }

    void SetAmplitudeBudget(size_t bytes) noexcept
    {
        amplitudeBudget_ = bytes;
    }

    // Evolve a shared realization once until its first measurement or reset
    // (default). Disabling it changes no result, only the work done.
    void SetRealizationSharing(bool enable) noexcept
    {
        shareRealizations_ = enable;
    }

    // Evolve the noiseless gates before the first channel, measurement or
    // reset once per call and start every batch from a copy (default), as a
    // checkpointed prefix does. Disabling it changes no result.
    void SetHeadCaching(bool enable) noexcept
    {
        cacheHead_ = enable;
    }

    // Lanes per batch (a multiple of 64 is most efficient) and the amplitude
    // memory one batch may use, which also pays for the cached head. A batch
    // that outgrows it splits into kSplitParts parts, recursively down to
    // single shots; splitting changes no row. A shot that does not fit
    // beside the cached head drops it for the rest of the call, which frees
    // it before the shot retries. A shot that does not fit even then
    // throws ExtendedFrames::MemoryLimitExceeded, unless soft: it then runs
    // alone beyond the budget, as a scalar trajectory would, and keeps its
    // draws. (A retry on another method would draw again, and which shots
    // retry depends on their outcomes.)
    struct Shape
    {
        size_t lanes = kDefaultBatchLanes;
        size_t amplitudeBudget = kDefaultAmplitudeBudget;
        bool soft = false;
    };

    // Circuit errors draw from the noise seed; measurement outcomes and
    // readout errors from the measurement seed, as in scalar execution, so
    // either can change without changing the other's draws.
    struct Seeds
    {
        uint64_t noise = 0;
        uint64_t measurement = 0;
    };

    static constexpr size_t kSplitParts = 4;
    // SampleWithin runs at most this many batches at once, each within its
    // share of the budget, so together they never exceed it.
    static constexpr size_t kMaxWorkers = 16;

    // How a call divides its amplitude budget. The cached head stays live
    // beside every batch, so it is charged once, and kept only when it leaves
    // the batches at least its own size. Shared: at most kMaxWorkers batches
    // in flight each decide within batch, and hold at most twice that with
    // the buffers they keep; a batch that does not fit then runs by itself
    // within alone. Otherwise every batch decides within alone.
    struct Allowance
    {
        bool keepHead = false;
        size_t batch = 0;
        size_t alone = 0;
    };

    static Allowance DivideBudget(size_t budget, size_t headBytes, bool shared) noexcept
    {
        Allowance allowance;
        allowance.keepHead = headBytes <= budget / 2;
        allowance.alone = allowance.keepHead ? budget - headBytes : budget;
        allowance.batch = shared ? allowance.alone / (2 * kMaxWorkers) : allowance.alone;
        return allowance;
    }

    // Row-major bytes in classical-register order. Rows are deterministic for
    // the seeds and realizations, and independent of the batch shape, the
    // budget (when every shot fits) and the number of workers.
    std::vector<uint8_t> Sample(size_t shots, size_t realizations, uint64_t seed, size_t workers = 1) const
    {
        return Sample(shots, realizations, Seeds{seed, seed}, workers, Shape{batchLanes_, amplitudeBudget_});
    }

    // shared: shape.amplitudeBudget is shared by all batches in flight, as
    // SampleWithin describes; otherwise each batch decides within it (and a
    // worker's kept buffers can hold up to twice as much).
    std::vector<uint8_t> Sample(size_t shots, size_t realizations, Seeds seeds, size_t workers, Shape shape, bool shared = false) const
    {
        RequireShape(shape);
        if (width_ != 0 && shots > std::numeric_limits<size_t>::max() / width_)
            throw std::length_error("Extended-frame sample matrix exceeds addressable memory");
        std::vector<uint8_t> result(shots * width_);
        const size_t W = BatchWords(shape);
        Run(shots, realizations, seeds, workers, shape, shared, [&](size_t, size_t start, size_t lanes, const std::vector<uint64_t> &classical) {
            FrameKernels::TransposeBatch(classical, width_, lanes, W, [&](size_t first, size_t base, size_t count, size_t laneCount, const auto &block) {
                for (size_t lane = 0; lane < laneCount; ++lane)
                    FrameKernels::WriteBits(block[lane], result.data() + (start + first + lane) * width_ + base, count);
            });
        });
        return result;
    }

    // The histogram of exactly the rows Sample() returns for the same arguments.
    FrameOutcomes SampleCounts(size_t shots, size_t realizations, uint64_t seed, size_t workers = 1) const
    {
        return SampleCounts(shots, realizations, Seeds{seed, seed}, workers, Shape{batchLanes_, amplitudeBudget_});
    }

    FrameOutcomes SampleCounts(size_t shots, size_t realizations, Seeds seeds, size_t workers, Shape shape, bool shared = false) const
    {
        RequireShape(shape);
        FrameOutcomes result;
        if (!shots)
            return result;
        using Packed = std::unordered_map<std::vector<uint64_t>, size_t, FrameDetail::PackedRowHash>;
        const size_t rowWords = width_ / 64 + (width_ % 64 != 0);
        const size_t W = BatchWords(shape);
        const size_t pool = std::max<size_t>(1, Utils::RealizationWorkerCount(BatchCount(shots, shape), workers));
        std::vector<Packed> packed(pool);
        std::vector<std::vector<uint64_t>> rows(pool);
        Run(shots, realizations, seeds, workers, shape, shared, [&](size_t worker, size_t, size_t lanes, const std::vector<uint64_t> &classical) {
            auto &local = rows[worker];
            local.assign(lanes * std::max<size_t>(1, rowWords), 0);
            FrameKernels::TransposeBatch(classical, width_, lanes, W, [&](size_t first, size_t base, size_t, size_t laneCount, const auto &block) {
                for (size_t lane = 0; lane < laneCount; ++lane)
                    local[(first + lane) * rowWords + base / 64] = block[lane];
            });
            std::vector<uint64_t> key(rowWords);
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                std::copy_n(local.begin() + lane * rowWords, rowWords, key.begin());
                ++packed[worker][key];
            }
        });
        for (size_t worker = 1; worker < pool; ++worker)
            for (const auto &[row, count] : packed[worker])
                packed[0][row] += count;
        result.reserve(packed[0].size());
        for (const auto &[row, count] : packed[0])
        {
            std::vector<bool> bits(width_);
            for (size_t bit = 0; bit < width_; ++bit)
                bits[bit] = ((row[bit / 64] >> (bit % 64)) & 1U) != 0;
            result.emplace_back(std::move(bits), count);
        }
        return result;
    }

    // Rows (rows true) or counts within budgetMB of amplitude memory, cached
    // head included, in batches of kDefaultBatchLanes lanes: at most
    // min(maxWorkers, kMaxWorkers) at once, each within its share (see
    // DivideBudget). A batch that does not fit, even split, runs afterwards
    // alone with the rest of the budget, split down to single shots; a shot
    // that does not fit beside the cached head drops it. The budget and the
    // workers decide only how shots are split, never a row. A single shot
    // that exceeds the whole budget returns false; with soft (a soft budget,
    // as scalar trajectories have) it runs alone beyond the budget instead,
    // and the call never returns false. A const call, safe to run
    // concurrently.
    bool SampleWithin(size_t shots, size_t realizations, Seeds seeds, bool rows, size_t maxWorkers, size_t budgetMB, std::vector<uint8_t> &samples,
                      FrameOutcomes &counts, bool soft = false) const
    {
        budgetMB = std::min<size_t>(budgetMB, size_t(1) << 30);
        const Shape shape{kDefaultBatchLanes, budgetMB << 20, soft};
        const size_t workers = std::max<size_t>(1, std::min(maxWorkers ? maxWorkers : kMaxWorkers, kMaxWorkers));
        try
        {
            if (rows)
                samples = Sample(shots, realizations, seeds, workers, shape, true);
            else
                counts = SampleCounts(shots, realizations, seeds, workers, shape, true);
            return true;
        }
        catch (const ExtendedFrames::MemoryLimitExceeded &)
        {
            return false;
        }
    }

  private:
    enum class Op : uint8_t
    {
        H,
        S,
        Sdg,
        Sx,
        SxDag,
        K,
        X,
        Y,
        Z,
        CX,
        CY,
        CZ,
        Swap,
        Rotation,
        Gate,
        Channel,
        Measurement,
        Reset
    };

    struct ErrorRate
    {
        double rate = 0;
        double inverseLogSurvival = 0;
    };

    struct PauliTerm
    {
        double cumulative = 1;
        bool firstX = false, firstZ = false, secondX = false, secondZ = false;
    };

    struct Instruction
    {
        Op op;
        size_t first = 0;  // the qubit, or the target of a controlled gate
        size_t second = 0; // the control, or a channel's second target
        char axis = 'Z';
        double angle = 0;
        size_t gate = 0;
        size_t bit = 0;
        ErrorRate error;
        std::vector<PauliTerm> terms;
        ErrorRate readout;
        double readout01 = 0;
        double readout10 = 0;
    };

    // Shots split into contiguous realizations exactly as ExecuteNoisy does.
    struct Layout
    {
        size_t shots, realizations, base, leftover;

        Layout(size_t shotCount, size_t realizationCount)
            : shots(shotCount), realizations(std::max<size_t>(1, std::min(shotCount, realizationCount))), base(shots / realizations),
              leftover(shots % realizations)
        {
        }

        size_t RealizationOf(size_t shot) const noexcept
        {
            const size_t large = leftover * (base + 1);
            return shot < large ? shot / (base + 1) : leftover + (shot - large) / base;
        }

        bool PerShot() const noexcept
        {
            return realizations == shots;
        }
    };

    static size_t BatchWords(const Shape &shape) noexcept
    {
        return (shape.lanes + 63) / 64;
    }

    static size_t BatchCount(size_t shots, const Shape &shape) noexcept
    {
        return (shots + shape.lanes - 1) / shape.lanes;
    }

    static void RequireShape(const Shape &shape)
    {
        if (!shape.lanes)
            throw std::invalid_argument("A batch needs at least one lane");
    }

    template <class Consume> void Run(size_t shots, size_t realizations, Seeds seeds, size_t workers, const Shape &shape, bool shared, Consume &&consume) const
    {
        if (!shots)
            return;
        RequireShape(shape);
        const Layout layout(shots, realizations);
        const size_t batches = BatchCount(shots, shape);
        const size_t pool = std::max<size_t>(1, Utils::RealizationWorkerCount(batches, workers));
        const size_t W = BatchWords(shape);
        // The deterministic head: every lane starts in the same state. It
        // stays live beside the batches, so it is compacted and charged. A
        // head that alone exceeds the budget is not cached; the shots decide.
        Head head;
        if (cacheHead_)
        {
            while (head.end < instructions_.size() && IsGate(instructions_[head.end].op))
                ++head.end;
            if (head.end)
                try
                {
                    auto built = std::make_shared<ExtendedFrames::LaneBatch>(frameQubits_, 1, shape.amplitudeBudget);
                    for (size_t index = 0; index < head.end; ++index)
                        Apply(*built, instructions_[index]);
                    built->Compact();
                    head.state = std::move(built);
                }
                catch (const ExtendedFrames::MemoryLimitExceeded &)
                {
                    head = Head{};
                }
        }
        const Allowance allowance = DivideBudget(shape.amplitudeBudget, head.state ? head.state->HeldAmplitudeBytes() : 0, shared);
        if (!allowance.keepHead)
            head = Head{};
        // A shot that does not fit beside the head drops it for the rest of
        // the call: the head and the budgets that charge it change together.
        // Attempts take the head under the lock and release it before they
        // split or retry, so a dropped head is freed once the attempts that
        // were copying it end (at once with one worker).
        std::mutex headMutex;
        size_t aloneBudget = allowance.alone, batchBudget = allowance.batch;
        const auto current = [&](bool alone) {
            const std::lock_guard<std::mutex> lock(headMutex);
            return std::make_pair(head, alone ? aloneBudget : batchBudget);
        };
        const auto dropHead = [&] {
            const std::lock_guard<std::mutex> lock(headMutex);
            const Allowance whole = DivideBudget(shape.amplitudeBudget, 0, shared);
            head = Head{};
            aloneBudget = whole.alone;
            batchBudget = whole.batch;
        };
        std::vector<std::unique_ptr<ExtendedFrames::LaneBatch>> states(pool);
        std::vector<std::vector<uint64_t>> classical(pool);
        // Shots [first, first + count) on a worker's state within budget,
        // their bits left in classical[worker]; false when they do not fit.
        // Workers reuse their buffers, but a batch's decisions count only what
        // a fresh batch would allocate. fresh releases the buffers first, so
        // the batch holds at most the budget; kept buffers can hold twice it.
        const auto fits = [&](size_t worker, size_t first, size_t count, size_t budget, bool fresh, const Head &from) {
            auto &state = states[worker];
            classical[worker].assign(width_ * W, 0);
            try
            {
                if (fresh || !state || state->HeldAmplitudeBytes() > budget)
                {
                    state.reset();
                    state = std::make_unique<ExtendedFrames::LaneBatch>(frameQubits_, count, budget);
                }
                state->SetAmplitudeBudget(budget);
                RunBatch(*state, from, layout, seeds, first, count, W, classical[worker]);
                return true;
            }
            catch (const ExtendedFrames::MemoryLimitExceeded &)
            {
                return false;
            }
        };
        // Every shot has draws of its own, so splitting changes no row, only
        // the memory a part needs; nor does starting without the head. Shots
        // that do not fit split into kSplitParts parts, recursively, each
        // consumed once it fits. alone (one batch at a time) uses aloneBudget
        // and fresh buffers. A shot that does not fit beside the cached head
        // drops the head (dropHead) and retries on fresh buffers; a soft
        // budget finally runs a shot beyond the budget.
        std::function<void(size_t, size_t, size_t, bool)> settle = [&](size_t worker, size_t first, size_t count, bool alone) {
            bool hadHead = false;
            {
                const auto [from, budget] = current(alone);
                hadHead = from.state != nullptr;
                if (fits(worker, first, count, budget, alone, from))
                {
                    consume(worker, first, count, classical[worker]);
                    return;
                }
            }
            if (count > 1)
            {
                const size_t part = (count + kSplitParts - 1) / kSplitParts;
                for (size_t offset = 0; offset < count; offset += part)
                    settle(worker, first + offset, std::min(part, count - offset), alone);
                return;
            }
            if (hadHead)
            {
                dropHead();
                states[worker].reset();
                settle(worker, first, count, alone);
                return;
            }
            if (!shape.soft || !fits(worker, first, 1, std::numeric_limits<size_t>::max(), true, Head{}))
                throw ExtendedFrames::MemoryLimitExceeded("An extended-frame shot exceeds the amplitude budget");
            consume(worker, first, 1, classical[worker]);
        };
        if (!shared)
        {
            Utils::RunRealizations(
                batches, pool, [](size_t batch) { return batch; },
                [&](size_t worker, size_t, size_t batch) {
                    const size_t start = batch * shape.lanes;
                    settle(worker, start, std::min(shape.lanes, shots - start), false);
                });
            return;
        }
        // Shared budget: at most kMaxWorkers batches in flight. A batch that
        // fits neither whole nor in kSplitParts parts runs again afterwards,
        // by itself, so its parts are consumed only once all of them fit.
        std::mutex mutex;
        std::vector<size_t> deferred;
        Utils::RunRealizations(
            batches, std::min(pool, kMaxWorkers), [](size_t batch) { return batch; },
            [&](size_t worker, size_t, size_t batch) {
                const size_t start = batch * shape.lanes, lanes = std::min(shape.lanes, shots - start);
                if (fits(worker, start, lanes, allowance.batch, false, head))
                {
                    consume(worker, start, lanes, classical[worker]);
                    return;
                }
                std::vector<std::pair<size_t, std::vector<uint64_t>>> parts;
                const size_t part = (lanes + kSplitParts - 1) / kSplitParts;
                bool all = lanes > 1;
                for (size_t offset = 0; all && offset < lanes; offset += part)
                {
                    all = fits(worker, start + offset, std::min(part, lanes - offset), allowance.batch, false, head);
                    if (all)
                        parts.emplace_back(offset, classical[worker]);
                }
                if (all)
                {
                    for (const auto &[offset, data] : parts)
                        consume(worker, start + offset, std::min(part, lanes - offset), data);
                    return;
                }
                const std::lock_guard<std::mutex> lock(mutex);
                deferred.push_back(batch);
            });
        for (auto &state : states)
            state.reset();
        std::sort(deferred.begin(), deferred.end());
        for (const size_t batch : deferred)
        {
            const size_t start = batch * shape.lanes;
            settle(0, start, std::min(shape.lanes, shots - start), true);
        }
    }

    // Shared by the attempts that copy it, so dropping it frees it once the
    // last of them ends.
    struct Head
    {
        size_t end = 0;
        std::shared_ptr<const ExtendedFrames::LaneBatch> state;
    };

    static bool IsGate(Op op) noexcept
    {
        return op != Op::Channel && op != Op::Measurement && op != Op::Reset;
    }

    // lanes copies of the head state, or of |0...0>; returns where to continue.
    size_t Start(ExtendedFrames::LaneBatch &state, const Head &head, size_t lanes) const
    {
        if (!head.state)
        {
            state.Reset(lanes);
            return 0;
        }
        // The copy brings the head's budget; keep the batch's own, and count the
        // copied buffers as a fresh copy would allocate them. A head above the
        // batch's budget is refused before it is copied.
        const size_t budget = state.AmplitudeBudget();
        if (head.state->HeldAmplitudeBytes() > budget)
            throw ExtendedFrames::MemoryLimitExceeded("The cached head exceeds the batch's amplitude budget");
        state = *head.state;
        state.RestartBudget(budget);
        if (lanes > 1)
            state.ExpandLanes(std::vector<size_t>(lanes, 0));
        return head.end;
    }

    // Shots [start, start + lanes) of the call, one per lane. Draws are keyed
    // by shot, block of shots or realization, never by the batch.
    void RunBatch(ExtendedFrames::LaneBatch &state, const Head &head, const Layout &layout, const Seeds &seeds, size_t start, size_t lanes, size_t W,
                  std::vector<uint64_t> &classical) const
    {
        using FrameKernels::Stream;
        using Utils::RandomStream::Derive;
        const uint64_t measurementSeed = Derive(seeds.measurement, kMeasurementStreams);
        const uint64_t readoutSeed = Derive(seeds.measurement, kReadoutStreams);
        const uint64_t shotNoiseSeed = Derive(seeds.noise, kShotNoiseStreams);
        const uint64_t noiseSeed = Derive(seeds.noise, kRealizationStreams);

        // Realizations present in the batch; shared ones start with one lane each.
        const size_t firstRealization = layout.RealizationOf(start);
        const size_t lastRealization = layout.RealizationOf(start + lanes - 1);
        const bool perShot = layout.PerShot();
        const bool grouped = shareRealizations_ && !perShot && lastRealization - firstRealization + 1 < lanes;
        size_t index = 0;
        std::vector<uint64_t> mask;
        if (grouped)
        {
            const size_t groups = lastRealization - firstRealization + 1;
            index = Start(state, head, groups);
            mask.assign(state.Words(), 0);
            for (; index < instructions_.size(); ++index)
            {
                const auto &instruction = instructions_[index];
                if (instruction.op == Op::Measurement || instruction.op == Op::Reset)
                    break;
                if (instruction.op == Op::Channel)
                {
                    for (size_t group = 0; group < groups; ++group)
                        RealizationError(state, instruction, noiseSeed, firstRealization + group, index, OneLane(mask, group));
                    continue;
                }
                Apply(state, instruction);
            }
            std::vector<size_t> source(lanes);
            for (size_t lane = 0; lane < lanes; ++lane)
                source[lane] = layout.RealizationOf(start + lane) - firstRealization;
            state.ExpandLanes(source);
        }
        else
            index = Start(state, head, lanes);
        mask.assign(state.Words(), 0);

        // Lanes of each realization, for errors after the expansion.
        std::vector<std::pair<size_t, std::vector<uint64_t>>> realizationMasks;
        if (!perShot)
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                const size_t realization = layout.RealizationOf(start + lane);
                if (realizationMasks.empty() || realizationMasks.back().first != realization)
                    realizationMasks.emplace_back(realization, std::vector<uint64_t>(state.Words(), 0));
                realizationMasks.back().second[lane / 64] |= uint64_t(1) << (lane % 64);
            }

        std::vector<uint64_t> outcome(state.Words());
        for (; index < instructions_.size(); ++index)
        {
            const auto &instruction = instructions_[index];
            switch (instruction.op)
            {
            case Op::Channel:
                if (perShot)
                    ForEachShotError(Derive(shotNoiseSeed, index), instruction.error, start, lanes, [&](Stream &stream, size_t lane) {
                        const PauliTerm &term = DrawTerm(instruction, stream);
                        if (lane < lanes)
                            ApplyTerm(state, instruction, term, OneLane(mask, lane));
                    });
                else
                    for (const auto &[realization, lanesOf] : realizationMasks)
                        RealizationError(state, instruction, noiseSeed, realization, index, lanesOf.data());
                break;
            case Op::Measurement:
            case Op::Reset: {
                const uint64_t outcomes = Derive(measurementSeed, index);
                auto choose = [&](size_t lane, double one) { return one >= 1.0 || (one > 0.0 && ShotUniform(outcomes, start + lane) < one); };
                if (instruction.op == Op::Reset)
                {
                    state.ResetQubit(instruction.first, choose);
                    break;
                }
                state.MeasureZ(instruction.first, choose, outcome.data());
                uint64_t *values = classical.data() + instruction.bit * W;
                for (size_t w = 0; w < state.Words(); ++w)
                    values[w] = outcome[w];
                if (instruction.readout.rate > 0)
                {
                    const uint64_t readout = Derive(readoutSeed, index);
                    const uint64_t accept = Derive(readout, 1);
                    const double candidate = instruction.readout.rate;
                    ForEachShotError(Derive(readout, 0), instruction.readout, start, lanes, [&](Stream &, size_t lane) {
                        if (lane >= lanes)
                            return;
                        const uint64_t bit = uint64_t(1) << (lane % 64);
                        uint64_t &word = values[lane / 64];
                        const double keep = (word & bit) ? instruction.readout10 : instruction.readout01;
                        if (keep >= candidate || (keep > 0 && ShotUniform(accept, start + lane) * candidate < keep))
                            word ^= bit;
                    });
                }
                break;
            }
            default:
                Apply(state, instruction);
                break;
            }
        }
    }

    // Per-shot errors are drawn over fixed blocks of this many shots of the
    // call, each from its own stream, so every shot sees the same errors
    // however the shots are batched. A block equals a default batch, so a
    // default batch needs one stream per channel.
    static constexpr size_t kStreamBlock = kDefaultBatchLanes;

    // Calls hit(stream, lane) for each shot of the blocks that overlap
    // [start, start + lanes) that an error with this rate hits; lane is the
    // shot's lane, or lanes and above for a shot outside the batch. hit must
    // draw the same from stream either way, so a block's draws are fixed.
    template <class Hit> static void ForEachShotError(uint64_t seed, const ErrorRate &error, size_t start, size_t lanes, Hit &&hit)
    {
        const size_t end = start + lanes;
        for (size_t block = start / kStreamBlock; block * kStreamBlock < end; ++block)
        {
            FrameKernels::Stream stream(Utils::RandomStream::Derive(seed, block));
            const size_t first = block * kStreamBlock;
            FrameKernels::ForEachError(stream, error.rate, error.inverseLogSurvival, kStreamBlock, [&](size_t offset) {
                const size_t shot = first + offset;
                hit(stream, shot >= start && shot < end ? shot - start : lanes);
            });
        }
    }

    // A uniform draw in [0, 1) of one shot, from a counter-based stream.
    static double ShotUniform(uint64_t seed, size_t shot) noexcept
    {
        return static_cast<double>(Utils::RandomStream::Derive(seed, shot) >> 11) * 0x1.0p-53;
    }

    static const uint64_t *OneLane(std::vector<uint64_t> &mask, size_t lane)
    {
        std::fill(mask.begin(), mask.end(), 0);
        mask[lane / 64] = uint64_t(1) << (lane % 64);
        return mask.data();
    }

    static const PauliTerm &DrawTerm(const Instruction &instruction, FrameKernels::Stream &stream)
    {
        const PauliTerm *term = instruction.terms.data();
        const PauliTerm *last = term + instruction.terms.size() - 1;
        if (term != last)
            for (const double draw = stream.Uniform(); term != last && draw >= term->cumulative;)
                ++term;
        return *term;
    }

    static void ApplyTerm(ExtendedFrames::LaneBatch &state, const Instruction &instruction, const PauliTerm &term, const uint64_t *mask)
    {
        state.XorPauli(instruction.first, term.firstX, term.firstZ, mask);
        state.XorPauli(instruction.second, term.secondX, term.secondZ, mask);
    }

    // One realization's error at one channel: the same draw in every batch.
    static void RealizationError(ExtendedFrames::LaneBatch &state, const Instruction &instruction, uint64_t noiseSeed, size_t realization, size_t index,
                                 const uint64_t *lanes)
    {
        FrameKernels::Stream stream(Utils::RandomStream::Derive(Utils::RandomStream::Derive(noiseSeed, realization), index));
        if (!(stream.Uniform() < instruction.error.rate))
            return;
        ApplyTerm(state, instruction, DrawTerm(instruction, stream), lanes);
    }

    void Apply(ExtendedFrames::LaneBatch &state, const Instruction &instruction) const
    {
        const size_t a = instruction.first, b = instruction.second;
        switch (instruction.op)
        {
        case Op::H:
            return state.ApplyH(a);
        case Op::S:
            return state.ApplyS(a);
        case Op::Sdg:
            return state.ApplySdg(a);
        case Op::Sx:
            return state.ApplySx(a);
        case Op::SxDag:
            return state.ApplySxDag(a);
        case Op::K:
            return state.ApplyK(a);
        case Op::X:
            return state.ApplyX(a);
        case Op::Y:
            return state.ApplyY(a);
        case Op::Z:
            return state.ApplyZ(a);
        case Op::CX:
            return state.ApplyCX(a, b);
        case Op::CY:
            return state.ApplyCY(a, b);
        case Op::CZ:
            return state.ApplyCZ(a, b);
        case Op::Swap:
            return state.ApplySwap(a, b);
        case Op::Rotation:
            return state.ApplyAxisRotation(a, instruction.angle, instruction.axis);
        case Op::Gate:
            return state.ApplyGate(gates_[instruction.gate]);
        default:
            throw std::logic_error("Extended-frame instruction needs randomness");
        }
    }

    static constexpr uint64_t kMeasurementStreams = 0x6d656173ULL; // "meas"
    static constexpr uint64_t kRealizationStreams = 0x6e6f6973ULL; // "nois"
    static constexpr uint64_t kShotNoiseStreams = 0x73686f74ULL;   // "shot"
    static constexpr uint64_t kReadoutStreams = 0x72656164ULL;     // "read"

    size_t width_ = 0;
    size_t qubits_ = 0;
    size_t frameQubits_ = 0;
    size_t batchLanes_ = kDefaultBatchLanes;
    size_t amplitudeBudget_ = kDefaultAmplitudeBudget;
    bool shareRealizations_ = true;
    bool cacheHead_ = true;
    std::vector<Instruction> instructions_;
    std::vector<ExtendedFrames::CompiledGate> gates_;

    friend std::optional<ExtendedFrameSampler> CompileExtendedFrameSampler(const std::shared_ptr<Circuits::Circuit<double>> &, size_t, size_t);
};

// Gates, Pauli channels (or Kraus maps that are exact Pauli mixtures),
// measurements with readout errors, resets, delays and nested circuits.
// Anything else (classical control, random generators, other channels) and
// measurement-free circuits return nullopt so the caller can use ordinary
// trajectories. Invalid dimensions throw.
inline std::optional<ExtendedFrameSampler> CompileExtendedFrameSampler(const std::shared_ptr<Circuits::Circuit<double>> &circuit, size_t numQubits,
                                                                       size_t numClassicalBits)
{
    using Sampler = ExtendedFrameSampler;
    using Op = Sampler::Op;
    using Type = Circuits::OperationType;
    using Gate = Circuits::QuantumGateType;
    if (!circuit)
        throw std::invalid_argument("Extended-frame sampling requires a circuit");

    // Rows only for the qubits the circuit touches, in increasing order.
    std::set<size_t> used;
    std::function<void(const Circuits::Circuit<double> &)> collect = [&](const Circuits::Circuit<double> &current) {
        for (const auto &op : current.GetOperations())
        {
            if (!op)
                continue;
            if (op->GetType() == Type::kComposite)
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
            throw std::logic_error("Extended-frame qubit missing from the compacted register");
        return static_cast<size_t>(found - rows.begin());
    };

    Sampler plan;
    plan.qubits_ = numQubits;
    plan.frameQubits_ = std::max<size_t>(1, frameQubits);
    plan.width_ = numClassicalBits;
    ExtendedFrames::GateCompiler compiler;
    bool hasMeasurement = false;
    constexpr double kPi = 3.141592653589793238462643383279502884;

    auto add = [&](Op op, size_t first, size_t second = 0) {
        Sampler::Instruction instruction{op};
        instruction.first = first;
        instruction.second = second;
        plan.instructions_.push_back(std::move(instruction));
    };
    auto rotation = [&](size_t qubit, char axis, double angle) {
        Sampler::Instruction instruction{Op::Rotation};
        instruction.first = qubit;
        instruction.axis = axis;
        instruction.angle = angle;
        plan.instructions_.push_back(std::move(instruction));
    };
    auto compiled = [&](ExtendedFrames::CompiledGate gate) {
        if (gate.terms.empty() && gate.cliffords.empty())
            return;
        Sampler::Instruction instruction{Op::Gate};
        instruction.gate = plan.gates_.size();
        plan.gates_.push_back(std::move(gate));
        plan.instructions_.push_back(std::move(instruction));
    };

    std::function<bool(const Circuits::Circuit<double> &)> compile = [&](const Circuits::Circuit<double> &current) {
        for (const auto &op : current.GetOperations())
        {
            if (!op)
                throw std::invalid_argument("Extended-frame circuit contains a null operation");
            for (const auto q : op->AffectedQubits())
                if (q >= numQubits)
                    throw std::out_of_range("Extended-frame qubit exceeds the quantum register");
            for (const auto b : op->AffectedBits())
                if (b >= numClassicalBits)
                    throw std::out_of_range("Extended-frame bit exceeds the classical register");
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
                    throw std::invalid_argument("Extended-frame measurement has mismatched qubits and bits");
                for (size_t i = 0; i < qubits.size(); ++i)
                {
                    Sampler::Instruction instruction{Op::Measurement};
                    instruction.first = row(qubits[i]);
                    instruction.bit = bits[i];
                    const auto readout = i < rates.size() ? rates[i] : Circuits::ReadoutRates{};
                    instruction.readout01 = readout.p_meas1_prep0;
                    instruction.readout10 = readout.p_meas0_prep1;
                    const auto rate = FrameKernels::MakeRate<Sampler::ErrorRate>(std::max(instruction.readout01, instruction.readout10));
                    instruction.readout = rate;
                    plan.instructions_.push_back(std::move(instruction));
                    hasMeasurement = true;
                }
                break;
            }
            case Type::kReset: {
                const auto reset = std::static_pointer_cast<Circuits::Reset<double>>(op);
                const auto &qubits = reset->GetQubits();
                const auto &targets = reset->GetResetTargets();
                for (const auto q : qubits)
                    add(Op::Reset, row(q));
                // Reset::Execute resets the whole list, then flips the targets.
                for (size_t i = 0; i < targets.size() && i < qubits.size(); ++i)
                    if (targets[i])
                        add(Op::X, row(qubits[i]));
                break;
            }
            case Type::kQuantumChannel: {
                const auto channel = std::static_pointer_cast<Circuits::QuantumChannelOperation<double>>(op);
                std::vector<double> mixture;
                const std::vector<double> *found = &channel->GetChannel().GetPauliProbabilities();
                if (found->empty())
                {
                    mixture = FrameKernels::PauliMixture(channel->GetChannel(), channel->AffectedQubits().size());
                    found = &mixture;
                }
                const auto &probabilities = *found;
                if (probabilities.empty() || probabilities.size() > 16)
                    return false;
                const auto targets = channel->AffectedQubits();
                Sampler::Instruction instruction{Op::Channel};
                instruction.first = row(targets[0]);
                instruction.second = targets.size() == 2 ? row(targets[1]) : instruction.first;
                double rate = 0;
                for (size_t i = 1; i < probabilities.size(); ++i)
                    rate += probabilities[i];
                if (!(rate > 0))
                    break;
                double cumulative = 0;
                for (size_t i = 1; i < probabilities.size(); ++i)
                {
                    if (!(probabilities[i] > 0))
                        continue;
                    cumulative += probabilities[i] / rate;
                    const size_t firstPauli = i % 4, secondPauli = i / 4;
                    Sampler::PauliTerm term;
                    term.cumulative = cumulative;
                    term.firstX = firstPauli == 1 || firstPauli == 2;
                    term.firstZ = firstPauli == 2 || firstPauli == 3;
                    term.secondX = secondPauli == 1 || secondPauli == 2;
                    term.secondZ = secondPauli == 2 || secondPauli == 3;
                    instruction.terms.push_back(term);
                }
                instruction.terms.back().cumulative = 1;
                instruction.error = FrameKernels::MakeRate<Sampler::ErrorRate>(rate);
                plan.instructions_.push_back(std::move(instruction));
                break;
            }
            case Type::kGate: {
                const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<double>>(op);
                const size_t arity = gate->GetNumQubits();
                const size_t a = row(gate->GetQubit(0));
                const size_t b = arity > 1 ? row(gate->GetQubit(1)) : a;
                const size_t c = arity > 2 ? row(gate->GetQubit(2)) : a;
                // QC::ExtendedStabilizer rejects repeated operands, and so do frames.
                if (arity == 2 && a == b)
                    throw std::invalid_argument("A two-qubit gate needs two distinct qubits");
                if (arity == 3 && (a == b || a == c || b == c))
                    throw std::invalid_argument("A three-qubit gate needs three distinct qubits");
                const auto params = gate->GetParams();
                const auto param = [&params](size_t i) { return i < params.size() ? params[i] : 0.0; };
                // The mapping of QCSimExtendedStabilizer: (control, target) gate
                // arguments become QCSim's (target, control).
                switch (gate->GetGateType())
                {
                case Gate::kXGateType:
                    add(Op::X, a);
                    break;
                case Gate::kYGateType:
                    add(Op::Y, a);
                    break;
                case Gate::kZGateType:
                    add(Op::Z, a);
                    break;
                case Gate::kHadamardGateType:
                    add(Op::H, a);
                    break;
                case Gate::kSGateType:
                    add(Op::S, a);
                    break;
                case Gate::kSdgGateType:
                    add(Op::Sdg, a);
                    break;
                case Gate::kSxGateType:
                    add(Op::Sx, a);
                    break;
                case Gate::kSxDagGateType:
                    add(Op::SxDag, a);
                    break;
                case Gate::kKGateType:
                    add(Op::K, a);
                    break;
                case Gate::kPhaseGateType:
                case Gate::kRzGateType:
                    rotation(a, 'Z', param(0));
                    break;
                case Gate::kTGateType:
                    rotation(a, 'Z', kPi / 4.0);
                    break;
                case Gate::kTdgGateType:
                    rotation(a, 'Z', -kPi / 4.0);
                    break;
                case Gate::kRxGateType:
                    rotation(a, 'X', param(0));
                    break;
                case Gate::kRyGateType:
                    rotation(a, 'Y', param(0));
                    break;
                case Gate::kUGateType:
                    compiled(compiler.U(a, param(0), param(1), param(2)));
                    break;
                case Gate::kSwapGateType:
                    add(Op::Swap, a, b);
                    break;
                case Gate::kCXGateType:
                    add(Op::CX, b, a);
                    break;
                case Gate::kCYGateType:
                    add(Op::CY, b, a);
                    break;
                case Gate::kCZGateType:
                    add(Op::CZ, b, a);
                    break;
                case Gate::kCPGateType:
                    compiled(compiler.CP(b, a, param(0)));
                    break;
                case Gate::kCRxGateType:
                    compiled(compiler.CRx(b, a, param(0)));
                    break;
                case Gate::kCRyGateType:
                    compiled(compiler.CRy(b, a, param(0)));
                    break;
                case Gate::kCRzGateType:
                    compiled(compiler.CRz(b, a, param(0)));
                    break;
                case Gate::kCHGateType:
                    compiled(compiler.CH(b, a));
                    break;
                case Gate::kCSxGateType:
                    compiled(compiler.CSx(b, a, false));
                    break;
                case Gate::kCSxDagGateType:
                    compiled(compiler.CSx(b, a, true));
                    break;
                case Gate::kCUGateType:
                    compiled(compiler.CU(b, a, param(0), param(1), param(2), param(3)));
                    break;
                case Gate::kCSwapGateType:
                    compiled(compiler.CSwap(b, c, a));
                    break;
                case Gate::kCCXGateType:
                    compiled(compiler.CCX(c, a, b));
                    break;
                default:
                    return false;
                }
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
} // namespace MaestroExecution
