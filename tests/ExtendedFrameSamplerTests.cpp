// Checks of the batched extended-stabilizer Pauli-frame engine and sampler
// against the scalar QCSim ExtendedStabilizer and exact density matrices.

#include "../Execution/ExtendedFrameBatch.h"
#include "../Execution/ExtendedFrameSampler.h"
#include "../Execution/PrefixCheckpoint.h"
#include "../Execution/SampleRecords.h"
#include "../Noise/NoiseModel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The live and peak bytes of large blocks (amplitude matrices and their
// labels) while a test watches; other allocations pass straight through.
namespace HeapPeak
{
constexpr size_t kLarge = 256 * 1024, kSlots = 512;
std::atomic<bool> watching{false};
std::mutex mutex;
std::array<void *, kSlots> blocks{};
std::array<size_t, kSlots> sizes{};
size_t live = 0, peak = 0;

void Start()
{
    const std::lock_guard<std::mutex> lock(mutex);
    blocks.fill(nullptr);
    live = peak = 0;
    watching = true;
}

size_t Stop()
{
    const std::lock_guard<std::mutex> lock(mutex);
    watching = false;
    return peak;
}
} // namespace HeapPeak

void *operator new(size_t size)
{
    void *block = std::malloc(size ? size : 1);
    if (!block)
        throw std::bad_alloc();
    if (size >= HeapPeak::kLarge && HeapPeak::watching)
    {
        const std::lock_guard<std::mutex> lock(HeapPeak::mutex);
        for (size_t slot = 0; slot < HeapPeak::kSlots; ++slot)
            if (!HeapPeak::blocks[slot])
            {
                HeapPeak::blocks[slot] = block;
                HeapPeak::sizes[slot] = size;
                HeapPeak::live += size;
                HeapPeak::peak = std::max(HeapPeak::peak, HeapPeak::live);
                break;
            }
    }
    return block;
}

void operator delete(void *block) noexcept
{
    if (block && HeapPeak::watching)
    {
        const std::lock_guard<std::mutex> lock(HeapPeak::mutex);
        for (size_t slot = 0; slot < HeapPeak::kSlots; ++slot)
            if (HeapPeak::blocks[slot] == block)
            {
                HeapPeak::blocks[slot] = nullptr;
                HeapPeak::live -= HeapPeak::sizes[slot];
                break;
            }
    }
    std::free(block);
}

void operator delete(void *block, size_t) noexcept
{
    operator delete(block);
}

namespace
{
using MaestroExecution::ExtendedFrames::CompiledGate;
using MaestroExecution::ExtendedFrames::GateCompiler;
using MaestroExecution::ExtendedFrames::LaneBatch;
using MaestroExecution::ExtendedFrames::Word;

void Require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

// A scalar simulator holding exactly the state of one lane.
std::unique_ptr<QC::ExtendedStabilizer> LaneAsScalar(const LaneBatch &batch, size_t lane)
{
    auto scalar = std::make_unique<QC::ExtendedStabilizer>(batch.Qubits());
    auto &frames = const_cast<std::vector<QC::ExtendedFrame> &>(scalar->GetFrames());
    frames.front() = batch.LaneFrame(lane);
    return scalar;
}

std::string PauliString(size_t code, size_t qubits)
{
    std::string pauli;
    for (size_t q = 0; q < qubits; ++q, code /= 4)
        pauli += "IXYZ"[code % 4];
    return pauli;
}

// Physical observables: every basis probability and, up to 4 qubits, every
// Pauli expectation; otherwise a fixed sample of Pauli strings.
std::vector<double> Observables(QC::ExtendedStabilizer &state)
{
    const size_t n = state.GetNrQubits();
    std::vector<double> values;
    for (size_t basis = 0; basis < (size_t(1) << n); ++basis)
        values.push_back(state.getBasisStateProbability(basis));
    size_t strings = 1;
    for (size_t q = 0; q < n; ++q)
        strings *= 4;
    std::mt19937_64 pick(12345);
    const size_t count = n <= 4 ? strings : 300;
    for (size_t i = 1; i < count; ++i)
        values.push_back(state.ExpectationValue(PauliString(n <= 4 ? i : 1 + pick() % (strings - 1), n)));
    return values;
}

void RequireSameState(QC::ExtendedStabilizer &expected, const LaneBatch &batch, size_t lane, const std::string &context)
{
    auto actual = LaneAsScalar(batch, lane);
    const auto a = Observables(expected), b = Observables(*actual);
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(a[i] - b[i]));
    Require(worst < 1e-10, context + ": lane " + std::to_string(lane) + " differs from the scalar simulator by " + std::to_string(worst));
}

bool Bit(const Word *words, size_t lane)
{
    return (words[lane / 64] >> (lane % 64)) & 1U;
}

// One random operation on the batch and on every lane's scalar copy.
struct Driver
{
    size_t n;
    LaneBatch &batch;
    std::vector<std::unique_ptr<QC::ExtendedStabilizer>> &scalars;
    std::mt19937_64 &rng;
    GateCompiler compiler;

    double Angle()
    {
        // Mix generic angles with exact Clifford angles, which compile differently.
        const double pi = std::acos(-1.0);
        switch (rng() % 4)
        {
        case 0:
            return 0.5 * pi * double(int(rng() % 9) - 4);
        case 1:
            return pi / 4;
        default:
            return std::uniform_real_distribution<double>(-pi, pi)(rng);
        }
    }

    size_t Qubit()
    {
        return rng() % n;
    }

    void Distinct(size_t count, size_t *out)
    {
        std::vector<size_t> all(n);
        for (size_t q = 0; q < n; ++q)
            all[q] = q;
        std::shuffle(all.begin(), all.end(), rng);
        std::copy_n(all.begin(), count, out);
    }

    void Both(const std::function<void(LaneBatch &)> &onBatch, const std::function<void(QC::ExtendedStabilizer &)> &onScalar)
    {
        onBatch(batch);
        for (auto &scalar : scalars)
            onScalar(*scalar);
    }

    void Gate(const CompiledGate &compiled, const std::function<void(QC::ExtendedStabilizer &)> &onScalar)
    {
        Both([&](LaneBatch &b) { b.ApplyGate(compiled); }, onScalar);
    }

    void RandomGate()
    {
        size_t q[3];
        const double a = Angle(), b = Angle(), c = Angle(), d = Angle();
        const unsigned kind = unsigned(rng() % 30);
        if (kind < 13 || n < 2)
        {
            q[0] = Qubit();
            switch (n < 2 ? kind % 13 : kind)
            {
            case 0:
                return Both([&](LaneBatch &s) { s.ApplyH(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyH(q[0]); });
            case 1:
                return Both([&](LaneBatch &s) { s.ApplyS(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyS(q[0]); });
            case 2:
                return Both([&](LaneBatch &s) { s.ApplySdg(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplySdg(q[0]); });
            case 3:
                return Both([&](LaneBatch &s) { s.ApplySx(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplySx(q[0]); });
            case 4:
                return Both([&](LaneBatch &s) { s.ApplySxDag(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplySxDag(q[0]); });
            case 5:
                return Both([&](LaneBatch &s) { s.ApplyK(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyK(q[0]); });
            case 6:
                return Both([&](LaneBatch &s) { s.ApplyX(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyX(q[0]); });
            case 7:
                return Both([&](LaneBatch &s) { s.ApplyY(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyY(q[0]); });
            case 8:
                return Both([&](LaneBatch &s) { s.ApplyZ(q[0]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyZ(q[0]); });
            case 9:
                return Both([&](LaneBatch &s) { s.ApplyAxisRotation(q[0], a, 'X'); }, [&](QC::ExtendedStabilizer &s) { s.ApplyRx(q[0], a); });
            case 10:
                return Both([&](LaneBatch &s) { s.ApplyAxisRotation(q[0], a, 'Y'); }, [&](QC::ExtendedStabilizer &s) { s.ApplyRy(q[0], a); });
            case 11:
                return Both([&](LaneBatch &s) { s.ApplyAxisRotation(q[0], a, 'Z'); }, [&](QC::ExtendedStabilizer &s) { s.ApplyRz(q[0], a); });
            default:
                return Gate(compiler.U(q[0], a, b, c), [&](QC::ExtendedStabilizer &s) { s.ApplyU(q[0], a, b, c); });
            }
        }
        if (kind < 28 || n < 3)
        {
            Distinct(2, q);
            switch (n < 3 ? 13 + (kind - 13) % 15 : kind)
            {
            case 13:
                return Both([&](LaneBatch &s) { s.ApplyCX(q[0], q[1]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCX(q[0], q[1]); });
            case 14:
                return Both([&](LaneBatch &s) { s.ApplyCY(q[0], q[1]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCY(q[0], q[1]); });
            case 15:
                return Both([&](LaneBatch &s) { s.ApplyCZ(q[0], q[1]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCZ(q[0], q[1]); });
            case 16:
                return Both([&](LaneBatch &s) { s.ApplySwap(q[0], q[1]); }, [&](QC::ExtendedStabilizer &s) { s.ApplySwap(q[0], q[1]); });
            case 17:
                return Both([&](LaneBatch &s) { s.ApplyISwap(q[0], q[1]); }, [&](QC::ExtendedStabilizer &s) { s.ApplyISwap(q[0], q[1]); });
            case 18:
                return Gate(compiler.CU(q[0], q[1], a, b, c, d), [&](QC::ExtendedStabilizer &s) { s.ApplyCU(q[0], q[1], a, b, c, d); });
            case 19:
                return Gate(compiler.CRx(q[0], q[1], a), [&](QC::ExtendedStabilizer &s) { s.ApplyCRx(q[0], q[1], a); });
            case 20:
                return Gate(compiler.CRy(q[0], q[1], a), [&](QC::ExtendedStabilizer &s) { s.ApplyCRy(q[0], q[1], a); });
            case 21:
                return Gate(compiler.CRz(q[0], q[1], a), [&](QC::ExtendedStabilizer &s) { s.ApplyCRz(q[0], q[1], a); });
            case 22:
                return Gate(compiler.CH(q[0], q[1]), [&](QC::ExtendedStabilizer &s) { s.ApplyCH(q[0], q[1]); });
            case 23:
                return Gate(compiler.CP(q[0], q[1], a), [&](QC::ExtendedStabilizer &s) { s.ApplyCP(q[0], q[1], a); });
            case 24:
                return Gate(compiler.CS(q[0], q[1]), [&](QC::ExtendedStabilizer &s) { s.ApplyCS(q[0], q[1]); });
            case 25:
                return Gate(compiler.CSdg(q[0], q[1]), [&](QC::ExtendedStabilizer &s) { s.ApplyCSdg(q[0], q[1]); });
            case 26:
                return Gate(compiler.CSx(q[0], q[1], false), [&](QC::ExtendedStabilizer &s) { s.ApplyCSx(q[0], q[1]); });
            default:
                return Gate(compiler.CSx(q[0], q[1], true), [&](QC::ExtendedStabilizer &s) { s.ApplyCSxDag(q[0], q[1]); });
            }
        }
        Distinct(3, q);
        if (kind == 28)
            return Gate(compiler.CCX(q[0], q[1], q[2]), [&](QC::ExtendedStabilizer &s) { s.ApplyCCX(q[0], q[1], q[2]); });
        return Gate(compiler.CSwap(q[0], q[1], q[2]), [&](QC::ExtendedStabilizer &s) { s.ApplyCSwap(q[0], q[1], q[2]); });
    }

    // A random physical Pauli on one lane: a frame flip in the batch, a gate
    // on that lane's scalar copy.
    void RandomError()
    {
        const size_t lane = rng() % scalars.size(), q = Qubit();
        const unsigned pauli = 1 + unsigned(rng() % 3);
        std::vector<Word> mask(batch.Words(), 0);
        mask[lane / 64] |= Word(1) << (lane % 64);
        batch.XorPauli(q, pauli != 3, pauli != 1, mask.data());
        auto &scalar = *scalars[lane];
        if (pauli == 1)
            scalar.ApplyX(q);
        else if (pauli == 2)
            scalar.ApplyY(q);
        else
            scalar.ApplyZ(q);
    }

    // Measure (or reset) a qubit with a random possible physical outcome per
    // lane, forced identically in the scalar copies.
    void RandomMeasurement(bool reset)
    {
        const size_t q = Qubit();
        std::vector<bool> physical(scalars.size());
        std::vector<double> scalarOne(scalars.size());
        for (size_t lane = 0; lane < scalars.size(); ++lane)
            scalarOne[lane] = scalars[lane]->GetQubitProbability(q);
        auto choose = [&](size_t lane, double physicalOne) {
            Require(std::abs(physicalOne - scalarOne[lane]) < 1e-10, "Born probability differs from the scalar simulator: lane " + std::to_string(lane) +
                                                                         " qubit " + std::to_string(q) + " batch " + std::to_string(physicalOne) + " scalar " +
                                                                         std::to_string(scalarOne[lane]) + " labels " + std::to_string(batch.Components()));
            bool outcome;
            if (physicalOne < 1e-12)
                outcome = false;
            else if (physicalOne > 1 - 1e-12)
                outcome = true;
            else
                outcome = rng() % 2 != 0;
            physical[lane] = outcome;
            return outcome;
        };
        std::vector<Word> result(batch.Words());
        if (reset)
            batch.ResetQubit(q, choose);
        else
        {
            batch.MeasureZ(q, choose, result.data());
            for (size_t lane = 0; lane < scalars.size(); ++lane)
                Require(Bit(result.data(), lane) == physical[lane], "reported physical outcome");
        }
        for (size_t lane = 0; lane < scalars.size(); ++lane)
        {
            Require(scalars[lane]->MeasureConditioned(q, physical[lane]) == physical[lane], "scalar forced outcome");
            if (reset && physical[lane])
                scalars[lane]->ApplyX(q);
        }
    }
};

void RunRandomCircuits(bool measurements)
{
    std::mt19937_64 rng(measurements ? 2024 : 7);
    size_t largest = 1;
    for (size_t trial = 0; trial < 90; ++trial)
    {
        const size_t n = 1 + trial % 5;
        const size_t lanes = trial % 3 == 0 ? 70 : 1 + trial % 9;
        LaneBatch batch(n, lanes);
        std::vector<std::unique_ptr<QC::ExtendedStabilizer>> scalars;
        for (size_t lane = 0; lane < lanes; ++lane)
            scalars.push_back(std::make_unique<QC::ExtendedStabilizer>(n));
        Driver driver{n, batch, scalars, rng, {}};
        const size_t operations = 30 + 15 * n;
        size_t peak = 1;
        for (size_t op = 0; op < operations; ++op)
        {
            const unsigned what = unsigned(rng() % 10);
            if (what < 6)
                driver.RandomGate();
            else if (what < 8 || !measurements)
                driver.RandomError();
            else
                driver.RandomMeasurement(what == 9);
            peak = std::max(peak, batch.Components());
        }
        for (size_t lane = 0; lane < lanes; ++lane)
            RequireSameState(*scalars[lane], batch, lane, std::string(measurements ? "measured" : "unitary") + " trial " + std::to_string(trial));
        largest = std::max(largest, peak);
    }
    std::cout << "  90 trials, up to " << largest << " shared labels\n";
}

void ExpandLanesCopiesStates()
{
    std::mt19937_64 rng(99);
    const size_t n = 3, lanes = 5;
    LaneBatch batch(n, lanes);
    std::vector<std::unique_ptr<QC::ExtendedStabilizer>> scalars;
    for (size_t lane = 0; lane < lanes; ++lane)
        scalars.push_back(std::make_unique<QC::ExtendedStabilizer>(n));
    Driver driver{n, batch, scalars, rng, {}};
    for (size_t op = 0; op < 40; ++op)
        (op % 3 == 2) ? driver.RandomError() : driver.RandomGate();
    const std::vector<size_t> source = {4, 0, 0, 2, 3, 3, 3, 1, 4, 2, 0, 1, 4, 4, 4, 4, 2, 1, 0, 3, 2, 2, 1, 0, 3, 4, 1, 0, 2, 3, 4, 0, 1, 2, 3,
                                        4, 4, 3, 2, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3};
    batch.ExpandLanes(source);
    Require(batch.Lanes() == source.size() && batch.Words() == 2, "expanded batch shape");
    for (size_t lane = 0; lane < source.size(); ++lane)
        RequireSameState(*scalars[source[lane]], batch, lane, "expanded lane");
}

// Random parity outcomes would leave each lane on a different half of the
// labels; folding keeps every lane on the same ones, so the shared set stays
// as small as one lane's support while the lanes still match the scalar runs.
void MeasurementsKeepLabelsShared(bool xChecks)
{
    const size_t data = 6, n = 2 * data - 1, lanes = 80;
    LaneBatch batch(n, lanes);
    std::vector<std::unique_ptr<QC::ExtendedStabilizer>> scalars;
    for (size_t lane = 0; lane < lanes; ++lane)
        scalars.push_back(std::make_unique<QC::ExtendedStabilizer>(n));
    std::mt19937_64 rng(4);
    Driver driver{n, batch, scalars, rng, {}};
    for (size_t d = 0; d < data; ++d)
        driver.Both([&](LaneBatch &s) { s.ApplyAxisRotation(2 * d, 0.6, 'X'); }, [&](QC::ExtendedStabilizer &s) { s.ApplyRx(2 * d, 0.6); });
    const size_t before = batch.Components();
    for (size_t round = 0; round < 3; ++round)
    {
        for (size_t a = 0; a + 1 < data; ++a)
        {
            const size_t ancilla = 2 * a + 1;
            if (xChecks)
            {
                // X-type checks: the ancilla measurement is off-diagonal.
                driver.Both([&](LaneBatch &s) { s.ApplyH(ancilla); }, [&](QC::ExtendedStabilizer &s) { s.ApplyH(ancilla); });
                driver.Both([&](LaneBatch &s) { s.ApplyCX(2 * a, ancilla); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCX(2 * a, ancilla); });
                driver.Both([&](LaneBatch &s) { s.ApplyCX(2 * a + 2, ancilla); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCX(2 * a + 2, ancilla); });
                driver.Both([&](LaneBatch &s) { s.ApplyH(ancilla); }, [&](QC::ExtendedStabilizer &s) { s.ApplyH(ancilla); });
                continue;
            }
            driver.Both([&](LaneBatch &s) { s.ApplyCX(ancilla, 2 * a); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCX(ancilla, 2 * a); });
            driver.Both([&](LaneBatch &s) { s.ApplyCX(ancilla, 2 * a + 2); }, [&](QC::ExtendedStabilizer &s) { s.ApplyCX(ancilla, 2 * a + 2); });
        }
        for (size_t error = 0; error < 10; ++error)
            driver.RandomError();
        for (size_t a = 0; a + 1 < data; ++a)
        {
            // Measure each ancilla with a random possible outcome per lane.
            const size_t q = 2 * a + 1;
            std::vector<bool> physical(lanes);
            std::vector<Word> result(batch.Words());
            batch.MeasureZ(
                q,
                [&](size_t lane, double one) {
                    physical[lane] = one < 1e-12 ? false : one > 1 - 1e-12 ? true : rng() % 2 != 0;
                    return bool(physical[lane]);
                },
                result.data());
            for (size_t lane = 0; lane < lanes; ++lane)
                Require(scalars[lane]->MeasureConditioned(q, physical[lane]) == physical[lane], "scalar parity outcome");
        }
        size_t widest = 0;
        for (size_t lane = 0; lane < lanes; ++lane)
            widest = std::max(widest, batch.LaneFrame(lane).GetFrameSize());
        Require(batch.Components() == widest,
                "lanes no longer share their labels: " + std::to_string(batch.Components()) + " shared, " + std::to_string(widest) + " in the widest lane");
    }
    for (size_t lane = 0; lane < lanes; ++lane)
        RequireSameState(*scalars[lane], batch, lane, xChecks ? "X parity rounds" : "Z parity rounds");
    std::cout << (xChecks ? "  X" : "  Z") << " parity rounds: " << before << " labels before, " << batch.Components() << " shared after (lanes still match)\n";
}

// ---------------------------------------------------------------- sampler

using Circuit = Circuits::Circuit<double>;
using Histogram = std::map<std::string, size_t>;

std::shared_ptr<Circuits::MeasurementOperation<double>> Measure(size_t qubit, size_t bit, double readout01 = 0, double readout10 = 0)
{
    auto measured = std::make_shared<Circuits::MeasurementOperation<double>>(std::vector<std::pair<Types::qubit_t, size_t>>{{qubit, bit}});
    if (readout01 > 0 || readout10 > 0)
        measured->SetReadout({{readout01, readout10}});
    return measured;
}

std::shared_ptr<Circuits::QuantumChannelOperation<double>> Pauli1(size_t q, double px, double py, double pz)
{
    return std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{q}, Simulators::QuantumChannel::Pauli(px, py, pz));
}

// Two-qubit depolarizing: every non-identity Pauli pair with probability p / 15.
std::shared_ptr<Circuits::QuantumChannelOperation<double>> Depolarize2(size_t a, size_t b, double p)
{
    std::vector<double> probabilities(16, p / 15);
    probabilities[0] = 1 - p;
    return std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{a, b}, Simulators::QuantumChannel::Pauli(probabilities));
}

template <class Gate, class... Args> void Add(const std::shared_ptr<Circuit> &circuit, Args... args)
{
    circuit->AddOperation(std::make_shared<Gate>(args...));
}

std::string Key(const std::vector<bool> &bits, size_t width)
{
    std::string key;
    for (size_t bit = 0; bit < width; ++bit)
        key += bit < bits.size() && bits[bit] ? '1' : '0';
    return key;
}

Histogram FrameHistogram(const MaestroExecution::ExtendedFrameSampler &plan, size_t shots, size_t realizations, uint64_t seed)
{
    Histogram histogram;
    for (const auto &[bits, count] : plan.SampleCounts(shots, realizations, seed, 4))
        histogram[Key(bits, plan.Width())] += count;
    return histogram;
}

// The exact noisy distribution, sampled from Maestro's density-matrix backend.
Histogram DensityMatrixHistogram(const std::shared_ptr<Circuit> &circuit, size_t width, size_t shots, uint64_t seed)
{
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = Simulators::SimulatorType::kQCSim;
    config.simulation_type = Simulators::SimulationType::kDensityMatrix;
    config.fixed_backend = true;
    config.seed = seed;
    const auto records =
        MaestroExecution::ExecuteNoisy(circuit, noise::NoiseModel{}, config, shots, std::nullopt, std::nullopt, MaestroExecution::NoisyExecutionOptions{});
    Histogram histogram;
    for (const auto &[bits, count] : records.counts)
        histogram[Key(bits, width)] += count;
    return histogram;
}

// Two-sample chi-square for equal totals, with a generous bound: a wrong
// distribution at these shot counts exceeds it by orders of magnitude.
double ChiSquare(const Histogram &a, const Histogram &b, double &bound)
{
    std::map<std::string, std::pair<size_t, size_t>> joined;
    for (const auto &[key, count] : a)
        joined[key].first = count;
    for (const auto &[key, count] : b)
        joined[key].second = count;
    double chi2 = 0;
    for (const auto &[key, counts] : joined)
    {
        const double x = double(counts.first), y = double(counts.second);
        chi2 += (x - y) * (x - y) / (x + y);
    }
    const double dof = double(joined.size()) - 1;
    bound = dof + 6 * std::sqrt(2 * std::max(1.0, dof)) + 10;
    return chi2;
}

void RequireSameDistribution(const Histogram &a, const Histogram &b, const std::string &context)
{
    double bound = 0;
    const double chi2 = ChiSquare(a, b, bound);
    std::cout << "  " << context << ": " << a.size() << " outcomes, chi2 " << chi2 << " (bound " << bound << ")\n";
    Require(chi2 < bound, context + ": distributions differ, chi2 = " + std::to_string(chi2));
}

// A noisy non-Clifford circuit with mid-circuit measurement, readout error,
// resets (one to |1>), a repeated measurement into one bit and 2q noise.
std::shared_ptr<Circuit> MidCircuitCircuit()
{
    auto c = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(c, 0);
    Add<Circuits::TGate<double>>(c, 0);
    Add<Circuits::CXGate<double>>(c, 0, 1);
    c->AddOperation(Pauli1(0, 0.02, 0.01, 0.03));
    Add<Circuits::RyGate<double>>(c, 1, 0.7);
    Add<Circuits::CCXGate<double>>(c, 0, 1, 2);
    c->AddOperation(Depolarize2(1, 2, 0.08));
    c->AddOperation(Measure(0, 0, 0.02, 0.05));
    Add<Circuits::CHGate<double>>(c, 1, 2);
    c->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{0}));
    Add<Circuits::UGate<double>>(c, 0, 0.3, 0.4, 0.5, 0.2);
    Add<Circuits::CRzGate<double>>(c, 2, 0, 1.1);
    c->AddOperation(Pauli1(2, 0.05, 0.0, 0.05));
    Add<Circuits::TdgGate<double>>(c, 2);
    Add<Circuits::HadamardGate<double>>(c, 2);
    c->AddOperation(Measure(1, 1, 0.01, 0.01));
    c->AddOperation(Measure(2, 2));
    c->AddOperation(Measure(0, 3));
    c->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{1}, 0, std::vector<bool>{true}));
    Add<Circuits::CSxGate<double>>(c, 1, 2);
    c->AddOperation(Measure(2, 1));
    c->AddOperation(Measure(1, 4));
    return c;
}

// Layers of Clifford+T gates with noise after every gate, measured at the end.
std::shared_ptr<Circuit> LayeredCircuit(size_t qubits, size_t layers, uint64_t seed, double noiseScale = 1)
{
    std::mt19937_64 rng(seed);
    auto c = std::make_shared<Circuit>();
    for (size_t layer = 0; layer < layers; ++layer)
    {
        for (size_t q = 0; q < qubits; ++q)
        {
            switch (rng() % 6)
            {
            case 0:
                Add<Circuits::HadamardGate<double>>(c, q);
                break;
            case 1:
                Add<Circuits::TGate<double>>(c, q);
                break;
            case 2:
                Add<Circuits::SGate<double>>(c, q);
                break;
            case 3:
                Add<Circuits::RxGate<double>>(c, q, 0.4 + 0.1 * double(layer));
                break;
            case 4:
                Add<Circuits::HadamardGate<double>>(c, q);
                Add<Circuits::TGate<double>>(c, q);
                break;
            default:
                Add<Circuits::CPGate<double>>(c, q, (q + 1) % qubits, 0.9);
                break;
            }
            c->AddOperation(Pauli1(q, 0.004 * noiseScale, 0.004 * noiseScale, 0.004 * noiseScale));
        }
        for (size_t q = layer % 2; q + 1 < qubits; q += 2)
        {
            Add<Circuits::CXGate<double>>(c, q, q + 1);
            c->AddOperation(Depolarize2(q, q + 1, 0.02 * noiseScale));
        }
    }
    if (qubits >= 3)
    {
        Add<Circuits::CSwapGate<double>>(c, 0, 1, 2);
        Add<Circuits::CUGate<double>>(c, 2, 0, 0.5, 0.6, 0.7, 0.8);
        Add<Circuits::CSxDagGate<double>>(c, 1, 0);
    }
    for (size_t q = 0; q < qubits; ++q)
        c->AddOperation(Measure(q, q, 0.01, 0.02));
    return c;
}

void SamplerMatchesDensityMatrix()
{
    struct Case
    {
        std::string name;
        std::shared_ptr<Circuit> circuit;
        size_t qubits, width, shots;
    };

    const std::vector<Case> cases = {{"mid-circuit", MidCircuitCircuit(), 3, 5, 60000},
                                     {"layered 3q", LayeredCircuit(3, 6, 5), 3, 3, 200000},
                                     {"layered 4q", LayeredCircuit(4, 5, 8), 4, 4, 200000}};
    for (const auto &test : cases)
    {
        const auto plan = MaestroExecution::CompileExtendedFrameSampler(test.circuit, test.qubits, test.width);
        Require(plan.has_value(), test.name + ": supported circuit rejected");
        const auto started = std::chrono::steady_clock::now();
        const auto frames = FrameHistogram(*plan, test.shots, test.shots, 17);
        const double frameSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const auto exact = DensityMatrixHistogram(test.circuit, test.width, test.shots, 23);
        RequireSameDistribution(frames, exact, test.name + " vs density matrix");
        std::cout << "    frames took " << frameSeconds << " s\n";
    }
    // The comparison has power: 1.5x stronger noise is detected.
    const auto plan = MaestroExecution::CompileExtendedFrameSampler(LayeredCircuit(4, 5, 8), 4, 4);
    double bound = 0;
    const double chi2 = ChiSquare(FrameHistogram(*plan, 200000, 200000, 17), DensityMatrixHistogram(LayeredCircuit(4, 5, 8, 1.5), 4, 200000, 23), bound);
    std::cout << "  negative control (1.5x noise): chi2 " << chi2 << " (bound " << bound << ")\n";
    Require(chi2 > 3 * bound, "the distribution test cannot tell 1.5x noise apart");
}

// Shots of one realization share its errors; different realizations do not.
void SharedRealizations()
{
    auto c = std::make_shared<Circuit>();
    c->AddOperation(Pauli1(0, 0.5, 0, 0)); // an X error in half the realizations
    Add<Circuits::HadamardGate<double>>(c, 1);
    Add<Circuits::TGate<double>>(c, 1);
    Add<Circuits::HadamardGate<double>>(c, 1);
    c->AddOperation(Measure(0, 0));
    c->AddOperation(Measure(1, 1));
    auto plan = MaestroExecution::CompileExtendedFrameSampler(c, 2, 2);
    Require(plan.has_value(), "shared-realization circuit rejected");
    for (const size_t batchLanes : {size_t(256), size_t(64), size_t(37)})
    {
        plan->SetBatchLanes(batchLanes);
        const size_t shots = 1000;
        for (const size_t realizations : {size_t(1), size_t(7), size_t(64)})
        {
            const auto rows = plan->Sample(shots, realizations, 5, 3);
            const size_t base = shots / realizations, leftover = shots % realizations;
            size_t flipped = 0, ones = 0;
            for (size_t r = 0; r < realizations; ++r)
            {
                const size_t first = r * base + std::min(r, leftover), count = base + (r < leftover);
                for (size_t s = first; s < first + count; ++s)
                {
                    Require(rows[s * 2] == rows[first * 2], "a realization's shots saw different errors");
                    ones += rows[s * 2 + 1];
                }
                flipped += rows[first * 2];
            }
            if (realizations == 64)
                Require(flipped > 16 && flipped < 48, "realization errors are not independent");
            // P(1) on qubit 1 is sin^2(pi/8) ~ 0.146 for every shot.
            Require(ones > 100 && ones < 200, "shots of a realization share their measurement outcomes");
            // Sharing the evolution until the first measurement changes nothing.
            plan->SetRealizationSharing(false);
            Require(plan->Sample(shots, realizations, 5, 2) == rows, "realization sharing changed the rows");
            plan->SetRealizationSharing(true);
        }
        // Independent noise per shot.
        const auto rows = plan->Sample(shots, shots, 5, 3);
        size_t flipped = 0;
        for (size_t s = 0; s < shots; ++s)
            flipped += rows[s * 2];
        Require(flipped > 430 && flipped < 570, "per-shot errors are not independent");
    }
    std::cout << "  shared realizations keep their errors; sharing the evolution changes no row\n";
}

void RowsCountsAndWorkers()
{
    const auto circuit = MidCircuitCircuit();
    const auto plan = MaestroExecution::CompileExtendedFrameSampler(circuit, 3, 5);
    for (const size_t realizations : {size_t(3000), size_t(64)})
    {
        const auto rows = plan->Sample(3000, realizations, 99, 1);
        Require(rows == plan->Sample(3000, realizations, 99, 8), "rows depend on the worker count");
        Require(rows != plan->Sample(3000, realizations, 100, 8), "seed is ignored");
        Histogram fromRows;
        for (size_t s = 0; s < 3000; ++s)
        {
            std::string key;
            for (size_t bit = 0; bit < 5; ++bit)
                key += rows[s * 5 + bit] ? '1' : '0';
            ++fromRows[key];
        }
        Histogram counts;
        for (const auto &[bits, count] : plan->SampleCounts(3000, realizations, 99, 5))
            counts[Key(bits, 5)] += count;
        Require(fromRows == counts, "counts differ from the rows of the same seed");
        // Draws belong to shots, blocks of shots or realizations: any batch
        // shape gives the same rows.
        for (const size_t lanes : {size_t(100), size_t(64), size_t(7), size_t(1000)})
            Require(plan->Sample(3000, realizations, {99, 99}, 3, {lanes, size_t(1) << 30}) == rows, "the batch shape changed the rows");
    }
    std::cout << "  rows are seeded, independent of workers and batch shape, and match counts\n";
}

void FallbackAndLimits()
{
    auto conditional = std::make_shared<Circuit>();
    Add<Circuits::TGate<double>>(conditional, 0);
    conditional->AddOperation(Measure(0, 0));
    conditional->AddOperation(std::make_shared<Circuits::ConditionalGate<double>>(
        std::make_shared<Circuits::XGate<double>>(0), std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{0}, std::vector<bool>{true})));
    conditional->AddOperation(Measure(0, 0));
    Require(!MaestroExecution::CompileExtendedFrameSampler(conditional, 1, 1), "classical control accepted");
    auto damping = std::make_shared<Circuit>();
    Add<Circuits::TGate<double>>(damping, 0);
    damping->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(0.3)));
    damping->AddOperation(Measure(0, 0));
    Require(!MaestroExecution::CompileExtendedFrameSampler(damping, 1, 1), "non-Pauli channel accepted");
    auto unmeasured = std::make_shared<Circuit>();
    Add<Circuits::TGate<double>>(unmeasured, 0);
    Require(!MaestroExecution::CompileExtendedFrameSampler(unmeasured, 1, 0), "measurement-free circuit accepted");
    auto phaseDamping = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(phaseDamping, 0);
    phaseDamping->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseDamping(0.4)));
    Add<Circuits::HadamardGate<double>>(phaseDamping, 0);
    phaseDamping->AddOperation(Measure(0, 0));
    Require(MaestroExecution::CompileExtendedFrameSampler(phaseDamping, 1, 1).has_value(), "a Kraus Pauli mixture was rejected");

    // A wide superposition: 1,024 amplitudes per shot. 64 KiB splits batches
    // down to a few shots without changing a row; 8 KiB is below one shot.
    auto wide = std::make_shared<Circuit>();
    for (size_t q = 0; q < 10; ++q)
        Add<Circuits::RxGate<double>>(wide, q, 0.3);
    for (size_t q = 0; q < 10; ++q)
        wide->AddOperation(Measure(q, q));
    auto plan = MaestroExecution::CompileExtendedFrameSampler(wide, 10, 10);
    plan->SetAmplitudeBudget(size_t(1) << 16);
    const auto split = plan->Sample(512, 512, 1, 2);
    plan->SetAmplitudeBudget(size_t(1) << 30);
    Require(split == plan->Sample(512, 512, 1, 2), "a split batch changed the rows");
    plan->SetAmplitudeBudget(size_t(1) << 13);
    bool threw = false;
    try
    {
        plan->SampleCounts(512, 512, 1, 2);
    }
    catch (const MaestroExecution::ExtendedFrames::MemoryLimitExceeded &)
    {
        threw = true;
    }
    Require(threw, "the amplitude budget was not enforced");
    std::cout << "  unsupported circuits fall back; the amplitude budget is enforced\n";
}

// ---------------------------------------------------------------- ExecuteNoisy

MaestroExecution::SimulatorConfig MethodConfig(Simulators::SimulationType method, std::optional<uint64_t> seed, const std::string &policy = "auto")
{
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = Simulators::SimulatorType::kQCSim;
    config.simulation_type = method;
    config.fixed_backend = true;
    config.seed = seed;
    config.counts_policy = policy;
    return config;
}

// A noiseless circuit for model-driven noise: non-Clifford gates, a
// mid-circuit measurement and reset, and fewer bits than qubits for one key.
std::shared_ptr<Circuit> ModelCircuit()
{
    auto c = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(c, 0);
    Add<Circuits::TGate<double>>(c, 0);
    Add<Circuits::CXGate<double>>(c, 0, 1);
    Add<Circuits::RyGate<double>>(c, 1, 0.6);
    Add<Circuits::CCXGate<double>>(c, 0, 1, 2);
    c->AddOperation(Measure(0, 0));
    c->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{0}));
    Add<Circuits::HadamardGate<double>>(c, 0);
    Add<Circuits::TGate<double>>(c, 0);
    Add<Circuits::CXGate<double>>(c, 0, 3);
    Add<Circuits::CHGate<double>>(c, 2, 3);
    Add<Circuits::HadamardGate<double>>(c, 0);
    for (size_t q = 0; q < 4; ++q)
        c->AddOperation(Measure(q, q + 1));
    return c;
}

noise::NoiseModel Model()
{
    noise::NoiseModel model;
    for (int q = 0; q < 4; ++q)
    {
        model.set_depolarizing(q, 0.02);
        model.set_readout_error(q, 0.02, 0.04);
    }
    model.set_2q_depolarizing(0, 1, 0.05);
    model.set_2q_depolarizing(0, 3, 0.05);
    return model;
}

Histogram RecordsHistogram(const MaestroExecution::ShotRecords &records, size_t width)
{
    Histogram histogram;
    if (!records.rows.empty())
        for (size_t shot = 0; shot < records.shots; ++shot)
        {
            std::string key;
            for (size_t bit = 0; bit < width; ++bit)
                key += records.rows[shot * records.width + bit] ? '1' : '0';
            ++histogram[key];
        }
    for (const auto &[bits, count] : records.counts)
        histogram[Key(bits, width)] += count;
    return histogram;
}

void ExecuteNoisyDispatch()
{
    using MaestroExecution::ExecuteNoisy;
    using MaestroExecution::NoisyExecutionOptions;
    using Method = Simulators::SimulationType;
    const auto circuit = ModelCircuit();
    const auto model = Model();
    const size_t width = 5, shots = 20000;
    NoisyExecutionOptions counts, rows;
    rows.rows = true;

    // Unseeded auto requests run on frames, counts and rows, any realization schedule.
    for (const std::optional<size_t> realizations : {std::optional<size_t>{}, std::optional<size_t>{shots}, std::optional<size_t>{7}})
        for (const auto &options : {counts, rows})
        {
            const auto records =
                ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 300, realizations, std::nullopt, options);
            Require(records.sampling_strategy == "extended_frame_v1", "unseeded extended_stabilizer did not use frames: " + records.sampling_strategy);
            Require(records.method == Method::kExtendedStabilizer && records.realizations == std::min<size_t>(300, realizations.value_or(64)),
                    "frame records report the method or realizations wrongly");
            size_t total = 0;
            for (const auto &[bits, count] : records.counts)
            {
                total += count;
                Require(bits.size() == 5, "frame count keys are not padded like the executor's");
            }
            Require(options.rows ? records.rows.size() == 300 * width : total == 300, "frame records lost shots");
        }

    // Seeded auto requests use reproducible frames; explicit legacy stays scalar.
    const auto seeded = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, 5), 2000, size_t{2000}, std::nullopt, counts);
    const auto legacy = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, 5, "legacy"), 2000, size_t{2000}, std::nullopt, counts);
    Require(seeded.sampling_strategy == "extended_frame_v1" && legacy.sampling_strategy != "extended_frame_v1", "seeded auto did not select frames");
    const auto replay = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, 5), 2000, size_t{2000}, std::nullopt, counts);
    Require(RecordsHistogram(seeded, width) == RecordsHistogram(replay, width), "seeded auto frames did not replay");
    const auto unseededLegacy =
        ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, std::nullopt, "legacy"), 50, size_t{50}, std::nullopt, counts);
    Require(unseededLegacy.sampling_strategy != "extended_frame_v1", "legacy used frames");
    // frame_v1 opts a seeded request in, reproducibly.
    const auto optedA = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, 9, "frame_v1"), 2000, size_t{2000}, std::nullopt, rows);
    const auto optedB = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, 9, "frame_v1"), 2000, size_t{2000}, std::nullopt, rows);
    Require(optedA.sampling_strategy == "extended_frame_v1" && optedA.rows == optedB.rows, "frame_v1 is not a reproducible opt-in");

    // Same distribution as the scalar trajectories and the exact density matrix.
    const auto frames = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, std::nullopt), shots, shots, std::nullopt, counts);
    const auto scalar = ExecuteNoisy(circuit, model, MethodConfig(Method::kExtendedStabilizer, 31, "legacy"), shots, shots, std::nullopt, counts);
    const auto exact = ExecuteNoisy(circuit, model, MethodConfig(Method::kDensityMatrix, 37), shots, std::nullopt, std::nullopt, counts);
    RequireSameDistribution(RecordsHistogram(frames, width), RecordsHistogram(scalar, width), "ExecuteNoisy frames vs scalar trajectories");
    RequireSameDistribution(RecordsHistogram(frames, width), RecordsHistogram(exact, width), "ExecuteNoisy frames vs density matrix");

    // Noise the frames cannot represent, and classical control, fall back.
    auto amplitude = model;
    amplitude.set_t1(2, 0.05);
    const auto damped = ExecuteNoisy(circuit, amplitude, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 100, size_t{100}, std::nullopt, counts);
    Require(damped.sampling_strategy != "extended_frame_v1", "amplitude damping ran on Pauli frames");
    auto conditional = std::make_shared<Circuit>(*circuit);
    conditional->AddOperation(std::make_shared<Circuits::ConditionalGate<double>>(
        std::make_shared<Circuits::XGate<double>>(1), std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{0}, std::vector<bool>{true})));
    conditional->AddOperation(Measure(1, 0));
    const auto controlled = ExecuteNoisy(conditional, model, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 100, size_t{100}, std::nullopt, counts);
    Require(controlled.sampling_strategy != "extended_frame_v1", "classical control ran on Pauli frames");
    std::cout << "  ExecuteNoisy dispatch, seeds and fallbacks behave as specified\n";
}

// The cached noiseless head and the budgeted entry point change no row.
void HeadCachingAndBudgetedSampling()
{
    auto c = std::make_shared<Circuit>();
    for (size_t q = 0; q < 3; ++q)
    {
        Add<Circuits::HadamardGate<double>>(c, q);
        Add<Circuits::TGate<double>>(c, q);
        Add<Circuits::RxGate<double>>(c, q, 0.3 + 0.2 * double(q));
    }
    Add<Circuits::CCXGate<double>>(c, 0, 1, 2);
    c->AddOperation(Depolarize2(0, 2, 0.1));
    Add<Circuits::CHGate<double>>(c, 2, 1);
    c->AddOperation(Measure(1, 0));
    c->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{1}));
    c->AddOperation(Pauli1(1, 0.1, 0.05, 0.0));
    for (size_t q = 0; q < 3; ++q)
        c->AddOperation(Measure(q, q + 1, 0.01, 0.02));
    auto plan = MaestroExecution::CompileExtendedFrameSampler(c, 3, 4);
    Require(plan.has_value(), "head-caching circuit rejected");
    for (const size_t realizations : {size_t(2000), size_t(7)})
    {
        const auto rows = plan->Sample(2000, realizations, 9, 3);
        plan->SetHeadCaching(false);
        Require(plan->Sample(2000, realizations, 9, 2) == rows, "the cached head changed the rows");
        plan->SetHeadCaching(true);
        std::vector<uint8_t> within;
        MaestroExecution::FrameOutcomes unused;
        // 256 MiB: 16 MiB per batch, so no batch splits; the rows are Sample()'s.
        Require(plan->SampleWithin(2000, realizations, {9, 9}, true, 4, 256, within, unused) && within == rows, "SampleWithin drew other rows");
    }
    std::cout << "  the cached head and SampleWithin change no row\n";
}

// Noisy suffixes after an ideal gate-only prefix run on frames and match the
// exact distribution of that prefix followed by the noisy suffix.
void CheckpointSuffixesOnFrames()
{
    using Method = Simulators::SimulationType;
    auto prefix = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(prefix, 0);
    Add<Circuits::TGate<double>>(prefix, 0);
    Add<Circuits::CXGate<double>>(prefix, 0, 1);
    Add<Circuits::RxGate<double>>(prefix, 1, 0.4);
    auto suffix = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(suffix, 1);
    Add<Circuits::TGate<double>>(suffix, 1);
    Add<Circuits::HadamardGate<double>>(suffix, 1);
    suffix->AddOperation(Measure(0, 0));
    Add<Circuits::CHGate<double>>(suffix, 0, 1);
    suffix->AddOperation(Measure(1, 1));
    noise::NoiseModel model;
    for (int q = 0; q < 2; ++q)
    {
        model.set_depolarizing(q, 0.03);
        model.set_readout_error(q, 0.02, 0.04);
    }
    const size_t shots = 60000;
    MaestroExecution::PrefixCheckpoint frames(prefix, 2, MethodConfig(Method::kExtendedStabilizer, std::nullopt));
    const auto records = frames.ExecuteSuffix(suffix, shots, &model, shots, std::nullopt, 0, false);
    Require(records.sampling_strategy == "extended_frame_v1", "checkpoint suffixes did not use frames: " + records.sampling_strategy);
    Require(frames.ExecuteSuffix(suffix, 100, &model, std::nullopt, std::nullopt, 0, true).sampling_strategy == "extended_frame_v1",
            "checkpoint rows under the default schedule did not use frames");
    // Exact reference: the ideal prefix, then the suffix with exact channels.
    auto noisy = noise::inject_exact_noise(suffix, model);
    noise::attach_readout_error(noisy, model);
    auto exact = std::make_shared<Circuit>(prefix->GetOperations());
    for (const auto &op : noisy->GetOperations())
        exact->AddOperation(op);
    RequireSameDistribution(RecordsHistogram(records, 2), DensityMatrixHistogram(exact, 2, shots, 41), "checkpoint frames vs density matrix");
    // Seeded checkpoints use reproducible frames; a measuring prefix stays scalar.
    MaestroExecution::PrefixCheckpoint seeded(prefix, 2, MethodConfig(Method::kExtendedStabilizer, 5));
    MaestroExecution::PrefixCheckpoint replayed(prefix, 2, MethodConfig(Method::kExtendedStabilizer, 5));
    const auto seededRecords = seeded.ExecuteSuffix(suffix, 100, &model, std::nullopt, std::nullopt, 0, false);
    Require(seededRecords.sampling_strategy == "extended_frame_v1", "a seeded checkpoint did not use frames");
    Require(RecordsHistogram(seededRecords, 2) == RecordsHistogram(replayed.ExecuteSuffix(suffix, 100, &model, std::nullopt, std::nullopt, 0, false), 2),
            "seeded checkpoint frames did not replay");
    // The end measurements of a prefix are removed, since only suffixes are sampled: such a prefix is still
    // gates only and runs on frames.
    auto measuring = std::make_shared<Circuit>(prefix->GetOperations());
    measuring->AddOperation(Measure(0, 0));
    MaestroExecution::PrefixCheckpoint ended(measuring, 2, MethodConfig(Method::kExtendedStabilizer, std::nullopt));
    Require(ended.ExecuteSuffix(suffix, 100, &model, std::nullopt, std::nullopt, 0, false).sampling_strategy == "extended_frame_v1",
            "a prefix that ends with a measurement did not run on frames");
    // A measurement that a later gate of the prefix follows runs, and collapses once for every suffix shot: no frames.
    Add<Circuits::HadamardGate<double>>(measuring, 0);
    MaestroExecution::PrefixCheckpoint collapsed(measuring, 2, MethodConfig(Method::kExtendedStabilizer, std::nullopt));
    Require(collapsed.ExecuteSuffix(suffix, 100, &model, std::nullopt, std::nullopt, 0, false).sampling_strategy != "extended_frame_v1",
            "a measuring prefix ran on frames");
}

// ---------------------------------------------------------------- review regressions

template <class Function> bool Throws(Function &&function)
{
    try
    {
        function();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

// Relaxation that sampled execution draws as resets keeps scalar trajectories
// when shots share a realization; T2 > T1, which sampled execution clamps,
// always does. Per-shot thermal noise with T2 <= T1 matches scalar shots.
void ThermalNoiseKeepsSampledSemantics()
{
    using MaestroExecution::ExecuteNoisy;
    using Method = Simulators::SimulationType;
    auto c = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(c, 0);
    Add<Circuits::TGate<double>>(c, 0);
    c->AddOperation(Measure(0, 0));
    noise::NoiseModel thermal;
    thermal.set_thermal_relaxation(0, 1.0, 0.5, 0.5, 0.5); // a Pauli-diagonal channel average
    const MaestroExecution::NoisyExecutionOptions counts;
    const auto shared = ExecuteNoisy(c, thermal, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 1000, size_t{1}, std::nullopt, counts);
    Require(shared.sampling_strategy != "extended_frame_v1", "a shared thermal realization ran on frames");
    const size_t shots = 40000;
    const auto frames = ExecuteNoisy(c, thermal, MethodConfig(Method::kExtendedStabilizer, std::nullopt), shots, shots, std::nullopt, counts);
    const auto scalar = ExecuteNoisy(c, thermal, MethodConfig(Method::kExtendedStabilizer, 13, "legacy"), shots, shots, std::nullopt, counts);
    Require(frames.sampling_strategy == "extended_frame_v1", "per-shot thermal noise did not use frames");
    RequireSameDistribution(RecordsHistogram(frames, 1), RecordsHistogram(scalar, 1), "per-shot thermal frames vs scalar");
    noise::NoiseModel clamped;
    clamped.set_thermal_relaxation(0, 0.1, 1.0, 1.9, 0.5);
    Require(ExecuteNoisy(c, clamped, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 500, size_t{500}, std::nullopt, counts).sampling_strategy !=
                "extended_frame_v1",
            "T2 > T1 ran on frames");
    noise::NoiseModel idle;
    idle.set_idle_noise(0, 1e-3, 1e-3, 0.5);
    Require(idle.has_sampled_resets() && thermal.has_sampled_resets() && !Model().has_sampled_resets(), "sampled-reset predicate");
    // Infinite T1 draws only phase flips, as frames do: shared realizations
    // keep frames.
    noise::NoiseModel dephasing;
    dephasing.set_thermal_relaxation(0, 1.0, std::numeric_limits<double>::infinity(), 2.0);
    dephasing.set_thermal_relaxation_2q(1, 1.0, std::numeric_limits<double>::infinity(), 3.0);
    Require(!dephasing.has_sampled_resets(), "infinite-T1 relaxation counted as resets");
    Require(ExecuteNoisy(c, dephasing, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 1000, size_t{4}, std::nullopt, counts).sampling_strategy ==
                "extended_frame_v1",
            "shared infinite-T1 realizations left frames");

    // The circuit decides: relaxation that never resets it, on qubits it
    // never gates or delays or with zero probability, keeps frames.
    const double inf = std::numeric_limits<double>::infinity();
    noise::NoiseModel zeroT1, infiniteT1, elsewhere, multiOnly, t1;
    zeroT1.set_t1(0, 0.0);
    infiniteT1.set_t1_from_time(0, 1.0, inf);
    elsewhere.set_thermal_relaxation(5, 1.0, 1.0, 1.0, 0.5);
    multiOnly.set_thermal_relaxation_2q(0, 1.0, 1.0, 1.0, 0.5); // c has one-qubit gates only
    t1.set_t1(0, 0.1);
    Require(zeroT1.has_sampled_resets() && elsewhere.has_sampled_resets(), "model-wide predicate");
    for (const auto *model : {&zeroT1, &infiniteT1, &elsewhere, &multiOnly, &idle})
        Require(!model->has_sampled_resets_on(*c), "relaxation that never resets the circuit counted as resets");
    Require(t1.has_sampled_resets_on(*c) && thermal.has_sampled_resets_on(*c), "resets on gated qubits not counted");
    auto delayed = std::make_shared<Circuit>();
    for (const auto &op : c->GetOperations())
        delayed->AddOperation(op);
    delayed->AddOperation(std::make_shared<Circuits::Delay<double>>(0, 1e-3));
    Require(idle.has_sampled_resets_on(*delayed), "a delay with idle relaxation not counted");
    noise::NoiseModel clampedElsewhere, instant, idleWindow;
    clampedElsewhere.set_thermal_relaxation(5, 0.1, 1.0, 1.9, 0.5);
    instant.set_thermal_relaxation(0, 0.0, 1.0, 1.9, 0.5); // zero duration: the identity
    idleWindow.set_idle_noise(0, 1.0, 1.9);
    Require(clamped.has_thermal_in_sampled_overdephasing_regime_on(*c) && !clampedElsewhere.has_thermal_in_sampled_overdephasing_regime_on(*c) &&
                !instant.has_thermal_in_sampled_overdephasing_regime_on(*c) && !instant.has_thermal_in_sampled_overdephasing_regime(),
            "circuit-aware over-dephasing predicate");
    Require(clamped.thermal_approximation_qubits_on(*c) == std::vector<int>{0} && clampedElsewhere.thermal_approximation_qubits_on(*c).empty() &&
                idleWindow.thermal_approximation_qubits_on(*c).empty() && idleWindow.thermal_approximation_qubits_on(*delayed) == std::vector<int>{0},
            "circuit-aware clamped qubits");
    zeroT1.set_bit_flip(0, 0.05);
    Require(ExecuteNoisy(c, zeroT1, MethodConfig(Method::kExtendedStabilizer, std::nullopt), 1000, size_t{4}, std::nullopt, counts).sampling_strategy ==
                "extended_frame_v1",
            "shared realizations with zero-probability T1 left frames");
    std::cout << "  relaxation keeps sampled semantics; per-shot thermal frames match scalar shots\n";
}

// The checkpoint keeps a snapshot of its prefix, not the caller's circuit.
void CheckpointSnapshotsItsPrefix()
{
    auto prefix = std::make_shared<Circuit>();
    Add<Circuits::XGate<double>>(prefix, 0);
    MaestroExecution::PrefixCheckpoint checkpoint(prefix, 1, MethodConfig(Simulators::SimulationType::kExtendedStabilizer, std::nullopt));
    Add<Circuits::XGate<double>>(prefix, 0); // edited after the state was saved
    auto suffix = std::make_shared<Circuit>();
    suffix->AddOperation(Measure(0, 0));
    noise::NoiseModel flip;
    flip.set_bit_flip(0, 1e-9);
    const auto records = checkpoint.ExecuteSuffix(suffix, 100, &flip, size_t{100}, std::nullopt, 0, false);
    Require(records.sampling_strategy == "extended_frame_v1", "snapshot test did not use frames");
    const auto histogram = RecordsHistogram(records, 1);
    Require(histogram.size() == 1 && histogram.count("1"), "frames replayed the caller's edited prefix");
    std::cout << "  checkpoints replay the prefix they saved\n";
}

// noise_seed alone drives circuit errors; the config seed alone measurements.
void NoiseAndMeasurementSeedsAreIndependent()
{
    using MaestroExecution::ExecuteNoisy;
    auto c = std::make_shared<Circuit>();
    Add<Circuits::XGate<double>>(c, 0);
    Add<Circuits::TGate<double>>(c, 0);
    c->AddOperation(Measure(0, 0)); // deterministic but for the error
    noise::NoiseModel flip;
    flip.set_bit_flip(0, 0.3);
    MaestroExecution::NoisyExecutionOptions rows;
    rows.rows = true;
    const auto run = [&](uint64_t seed, uint64_t noiseSeed) {
        const auto records =
            ExecuteNoisy(c, flip, MethodConfig(Simulators::SimulationType::kExtendedStabilizer, seed, "frame_v1"), 300, size_t{300}, noiseSeed, rows);
        Require(records.sampling_strategy == "extended_frame_v1", "frame_v1 seeds test did not use frames");
        return records.rows;
    };
    Require(run(1, 7) == run(2, 7), "the measurement seed changed the circuit errors");
    Require(run(1, 7) != run(1, 8), "the noise seed did not drive the circuit errors");
    std::cout << "  noise and measurement seeds are independent\n";
}

std::shared_ptr<Circuit> Rotations(size_t qubits)
{
    auto c = std::make_shared<Circuit>();
    for (size_t q = 0; q < qubits; ++q)
        Add<Circuits::RyGate<double>>(c, q, 1.2);
    c->AddOperation(Pauli1(0, 0.05, 0.0, 0.05));
    for (size_t q = 0; q < qubits; ++q)
        c->AddOperation(Measure(q, q));
    return c;
}

// Splitting changes no row: neither the budget nor the workers do.
void RowsIgnoreWorkersAndBudget()
{
    const auto plan = MaestroExecution::CompileExtendedFrameSampler(Rotations(11), 11, 11);
    const MaestroExecution::ExtendedFrameSampler::Seeds seeds{5, 6};
    std::vector<uint8_t> reference, rows;
    MaestroExecution::FrameOutcomes unused;
    Require(plan->SampleWithin(1024, 1024, seeds, true, 1, 1024, reference, unused), "batches did not fit");
    // A 256-lane batch of 2048 labels and its next matrix need 16 MiB: 12 MiB
    // splits every batch, 1 MiB splits down to a few shots, 128 and 1024 MiB
    // leave room for whole batches, alone or several at once.
    for (const size_t budget : {size_t(1), size_t(12), size_t(128), size_t(1024)})
        for (const size_t workers : {size_t(1), size_t(2), size_t(8), size_t(0)})
        {
            Require(plan->SampleWithin(1024, 1024, seeds, true, workers, budget, rows, unused), "batches did not fit");
            Require(rows == reference, "the budget or the worker count changed the rows");
        }
    // No floor: a budget below a single shot falls back, cached head or not.
    auto wide = MaestroExecution::CompileExtendedFrameSampler(Rotations(17), 17, 17);
    Require(!wide->SampleWithin(4, 4, seeds, true, 1, 1, rows, unused), "a shot above 1 MiB was admitted");
    wide->SetHeadCaching(false);
    Require(!wide->SampleWithin(4, 4, seeds, true, 1, 1, rows, unused), "a shot above 1 MiB was admitted without the head");
    // A soft budget, as direct execution has, runs such a shot alone beyond
    // it with the draws it would have had.
    std::vector<uint8_t> ample;
    Require(wide->SampleWithin(4, 4, seeds, true, 1, 1024, ample, unused), "the wide shots did not fit 1 GiB");
    for (const bool cache : {false, true})
    {
        wide->SetHeadCaching(cache);
        Require(wide->SampleWithin(4, 4, seeds, true, 2, 1, rows, unused, true) && rows == ample, "a soft budget refused or redrew a shot");
    }
    std::cout << "  rows ignore the budget and the worker count; small budgets are honored\n";
}

// A retry after a batch outgrows its budget keeps every shot's draws, even
// when the outcome decides how much memory the shot needs.
void MemoryRetriesKeepEveryDraw()
{
    const auto expanding = [](size_t targets) {
        auto c = std::make_shared<Circuit>();
        Add<Circuits::HadamardGate<double>>(c, 0);
        c->AddOperation(Measure(0, 0));
        for (size_t t = 1; t <= targets; ++t)
        {
            // Controlled Ry(0.8): the target's amplitudes split only after a one.
            Add<Circuits::RyGate<double>>(c, t, 0.4);
            Add<Circuits::CXGate<double>>(c, 0, t);
            Add<Circuits::RyGate<double>>(c, t, -0.4);
            Add<Circuits::CXGate<double>>(c, 0, t);
        }
        c->AddOperation(Pauli1(1, 0.02, 0.0, 0.02));
        for (size_t t = 1; t <= targets; ++t)
            c->AddOperation(Measure(t, t, 0.01, 0.03));
        return c;
    };
    std::vector<uint8_t> split, ample;
    MaestroExecution::FrameOutcomes unused;
    const auto one = MaestroExecution::CompileExtendedFrameSampler(expanding(12), 13, 13);
    size_t ones = 0;
    for (uint64_t seed = 0; seed < 200; ++seed)
    {
        Require(one->SampleWithin(1, 1, {seed, seed + 1}, true, 1, 1, split, unused), "a single shot did not fit 1 MiB");
        Require(one->SampleWithin(1, 1, {seed, seed + 1}, true, 1, 128, ample, unused) && split == ample, "a memory retry redrew a shot");
        ones += split[0];
    }
    Require(ones > 60 && ones < 140, "memory retries biased the outcome: " + std::to_string(ones) + " ones in 200");
    const auto many = MaestroExecution::CompileExtendedFrameSampler(expanding(8), 9, 9);
    for (const size_t realizations : {size_t(700), size_t(9)})
    {
        Require(many->SampleWithin(700, realizations, {3, 4}, true, 1, 128, ample, unused), "whole batches did not fit");
        for (const size_t workers : {size_t(1), size_t(4)})
            Require(many->SampleWithin(700, realizations, {3, 4}, true, workers, 1, split, unused) && split == ample, "split batches redrew shots");
    }
    std::cout << "  memory retries keep every shot's draws\n";
}

// A mixed measurement folds some lanes into new frames, so a lane's frame
// depends on the other lanes of its batch; its outcome must not. Here a
// second measurement of the same qubit decides how much memory the shot
// needs, so a split retry would otherwise change it.
void OutcomesIgnoreBatchMates()
{
    using MaestroExecution::ExtendedFrames::LaneBatch;
    const auto second = [](size_t lanes, double draw) {
        LaneBatch batch(1, lanes);
        batch.ApplyAxisRotation(0, 1.5, 'Y');
        std::vector<Word> out(batch.Words());
        batch.MeasureZ(
            0, [](size_t lane, double) { return lane == 0; }, out.data()); // lane 1 reads zero
        batch.ApplyAxisRotation(0, 1.5, 'Y');
        double probability = -1;
        batch.MeasureZ(
            0,
            [&](size_t lane, double one) {
                if (lane == 0)
                    probability = one;
                return draw < one;
            },
            out.data());
        return std::make_pair(probability, Bit(out.data(), 0));
    };
    for (const double draw : {0.05, 0.3, 0.5, 0.7, 0.95})
    {
        const auto alone = second(1, draw), mixed = second(2, draw);
        Require(std::abs(alone.first - mixed.first) < 1e-12 && alone.second == mixed.second, "a batch-mate changed a lane's outcome");
    }

    auto c = std::make_shared<Circuit>();
    Add<Circuits::RyGate<double>>(c, 0, 1.5);
    c->AddOperation(Measure(0, 0));
    Add<Circuits::RyGate<double>>(c, 0, 1.5);
    c->AddOperation(Measure(0, 1));
    for (size_t t = 2; t < 15; ++t)
    {
        // Controlled Ry(0.8) from qubit 0: the shot grows only after a one.
        Add<Circuits::RyGate<double>>(c, t, 0.4);
        Add<Circuits::CXGate<double>>(c, 0, t);
        Add<Circuits::RyGate<double>>(c, t, -0.4);
        Add<Circuits::CXGate<double>>(c, 0, t);
    }
    for (size_t t = 2; t < 15; ++t)
        c->AddOperation(Measure(t, t));
    const auto plan = MaestroExecution::CompileExtendedFrameSampler(c, 15, 15);
    std::vector<uint8_t> split, ample;
    MaestroExecution::FrameOutcomes unused;
    size_t ones = 0;
    for (uint64_t seed = 0; seed < 1000; ++seed)
    {
        Require(plan->SampleWithin(2, 2, {seed, seed}, true, 1, 1, split, unused), "two shots did not fit 1 MiB");
        Require(plan->SampleWithin(2, 2, {seed, seed}, true, 1, 128, ample, unused) && split == ample, "a split changed a measurement");
        ones += split[1] + split[15 + 1];
    }
    // P(one) of the second measurement is sin^2(1.5) / 2 = 0.4975: 995 +- 32.
    Require(ones > 900 && ones < 1090, "split retries biased the outcome: " + std::to_string(ones) + " ones in 2000");
    using Sampler = MaestroExecution::ExtendedFrameSampler;
    const auto rows = [&](size_t lanes) { return plan->Sample(500, 500, Sampler::Seeds{3, 4}, 1, Sampler::Shape{lanes, size_t(1) << 30}); };
    const auto wide = rows(256);
    for (const size_t lanes : {size_t(1), size_t(2), size_t(7), size_t(64)})
        Require(rows(lanes) == wide, "the batch shape changed a row after a mixed measurement");
    std::cout << "  outcomes ignore the other lanes of a batch\n";
}

// A shot that does not fit beside the cached head drops it, on every path,
// and the head is freed before the shot retries: caching never holds more
// than the call without it would, and changes no row.
void RetriesFreeTheHead()
{
    // 16 rotations make a 1 MiB head; two more after a spectator measurement
    // need about 8 MiB, which fits the budget only without the head.
    auto c = std::make_shared<Circuit>();
    for (size_t q = 0; q < 16; ++q)
        Add<Circuits::RyGate<double>>(c, q, 1.2);
    c->AddOperation(Measure(18, 0));
    Add<Circuits::RyGate<double>>(c, 16, 1.2);
    Add<Circuits::RyGate<double>>(c, 17, 1.2);
    c->AddOperation(Measure(17, 1));
    std::vector<uint8_t> rows[2];
    size_t peaks[2];
    for (const bool cache : {false, true})
    {
        auto plan = MaestroExecution::CompileExtendedFrameSampler(c, 19, 2);
        plan->SetAmplitudeBudget(size_t(8) << 20);
        plan->SetHeadCaching(cache);
        HeapPeak::Start();
        rows[cache] = plan->Sample(1, 1, 5, 1);
        peaks[cache] = HeapPeak::Stop();
    }
    Require(rows[0] == rows[1], "dropping the head changed an unshared row");
    // The head and its labels take about 1.5 MiB; the shared path's
    // compaction leaves a small residue.
    Require(peaks[1] < peaks[0] + (size_t(512) << 10),
            "an unshared retry ran beside the cached head: " + std::to_string(peaks[1]) + " bytes against " + std::to_string(peaks[0]));
    std::cout << "  retries free the cached head first\n";
}

// Direct execution's budget is soft, as for scalar trajectories: a shot that
// alone exceeds it stays on frames with its own draws. A scalar retry would
// draw again, and only for the shots whose outcomes made them large.
void OversizeShotsKeepTheirDraws()
{
    using MaestroExecution::ExecuteNoisy;
    using Method = Simulators::SimulationType;
    // A 50% flip of qubit 0, copied to qubit 1, which controls 16 rotations of
    // fresh targets: a one needs 2^16 components (above 1 MiB with the
    // workspaces), a zero one.
    auto c = std::make_shared<Circuit>();
    Add<Circuits::XGate<double>>(c, 0);
    Add<Circuits::CXGate<double>>(c, 0, 1);
    for (size_t t = 2; t < 18; ++t)
    {
        Add<Circuits::RyGate<double>>(c, t, 0.4);
        Add<Circuits::CXGate<double>>(c, 1, t);
        Add<Circuits::RyGate<double>>(c, t, -0.4);
        Add<Circuits::CXGate<double>>(c, 1, t);
    }
    c->AddOperation(Measure(1, 0));
    noise::NoiseModel flip;
    flip.set_bit_flip(0, 0.5);
    MaestroExecution::NoisyExecutionOptions rows;
    rows.rows = true;
    size_t ones = 0;
    for (uint64_t seed = 0; seed < 60; ++seed)
    {
        auto config = MethodConfig(Method::kExtendedStabilizer, seed, "frame_v1");
        config.trajectory_max_memory_mb = 1;
        const auto tight = ExecuteNoisy(c, flip, config, 1, size_t{1}, std::optional<uint64_t>{seed}, rows);
        config.trajectory_max_memory_mb = 128;
        const auto ample = ExecuteNoisy(c, flip, config, 1, size_t{1}, std::optional<uint64_t>{seed}, rows);
        Require(tight.sampling_strategy == "extended_frame_v1" && ample.sampling_strategy == "extended_frame_v1", "an oversize shot left frames");
        Require(tight.rows == ample.rows, "an oversize shot was drawn again");
        ones += tight.rows[0];
    }
    Require(ones > 15 && ones < 45, "oversize shots biased the outcome: " + std::to_string(ones) + " ones in 60");
    std::cout << "  oversize shots keep their draws in direct execution\n";
}

// Noisy Clifford circuits on reference frames follow the scalar rules: the
// noise seed drives circuit errors and the measurement seed the outcomes,
// and thermal noise that trajectories clamp keeps the trajectories.
void ReferenceFramesKeepScalarRules()
{
    using MaestroExecution::ExecuteNoisy;
    using Method = Simulators::SimulationType;
    MaestroExecution::NoisyExecutionOptions rows;
    rows.rows = true;
    const auto run = [&](const std::shared_ptr<Circuit> &circuit, const noise::NoiseModel &model, uint64_t seed, uint64_t noiseSeed) {
        auto records = ExecuteNoisy(circuit, model, MethodConfig(Method::kStabilizer, seed), 200, size_t{200}, std::optional<uint64_t>{noiseSeed}, rows);
        Require(records.sampling_strategy == "reference_frame_v1", "a noisy Clifford circuit left reference frames: " + records.sampling_strategy);
        return records.rows;
    };
    // A deterministic measurement after bit flips: only the noise seed matters.
    auto flipped = std::make_shared<Circuit>();
    Add<Circuits::XGate<double>>(flipped, 0);
    flipped->AddOperation(Measure(0, 0));
    noise::NoiseModel flip;
    flip.set_bit_flip(0, 0.3);
    Require(run(flipped, flip, 1, 7) == run(flipped, flip, 2, 7), "the measurement seed changed reference-frame errors");
    Require(run(flipped, flip, 1, 7) != run(flipped, flip, 1, 8), "the noise seed did not drive reference-frame errors");
    // A random measurement with noise elsewhere: only the measurement seed.
    auto random = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(random, 0);
    Add<Circuits::XGate<double>>(random, 1);
    random->AddOperation(Measure(0, 0));
    noise::NoiseModel aside;
    aside.set_bit_flip(1, 0.3);
    Require(run(random, aside, 1, 7) == run(random, aside, 1, 8), "the noise seed changed reference-frame outcomes");
    Require(run(random, aside, 1, 7) != run(random, aside, 2, 7), "the measurement seed did not drive reference-frame outcomes");
    // One seed for both keeps the compiled sampler's rows.
    auto channels = noise::inject_exact_noise(flipped, flip);
    const auto plan = MaestroExecution::CompileReferenceSampler(channels, 1, 1);
    Require(plan && plan->Sample(500, 42) == plan->Sample(500, MaestroExecution::ReferenceFrameSampler::Seeds{42, 42}), "one seed changed its rows");

    // Trajectories clamp T2 > T1 to T1 (and report it); exact channels would not.
    auto clifford = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(clifford, 0);
    Add<Circuits::SGate<double>>(clifford, 0);
    Add<Circuits::HadamardGate<double>>(clifford, 0);
    clifford->AddOperation(Measure(0, 0));
    noise::NoiseModel exact, window, windowElsewhere;
    exact.set_thermal_relaxation(0, 1.0, 1.0, 1.0, 0.5);
    window.set_thermal_relaxation(0, 1.0, 1.0, 2.0, 0.5);
    windowElsewhere.set_thermal_relaxation(4, 1.0, 1.0, 2.0, 0.5);
    windowElsewhere.set_bit_flip(0, 0.1);
    run(clifford, exact, 3, 3);
    run(clifford, windowElsewhere, 3, 3);
    Require(ExecuteNoisy(clifford, window, MethodConfig(Method::kStabilizer, 3), 200, size_t{200}, std::optional<uint64_t>{3}, rows).sampling_strategy !=
                "reference_frame_v1",
            "clamped thermal noise ran on reference frames");
    std::cout << "  reference frames keep the scalar seed and thermal rules\n";
}

// A buffer kept from an earlier batch decides exactly as a fresh one.
void ReusedBuffersCountAsFresh()
{
    const auto grow = [](LaneBatch &batch, size_t qubits) {
        for (size_t q = 0; q < qubits; ++q)
            batch.ApplyAxisRotation(q, 0.7, 'X');
    };
    LaneBatch fresh(8, 64);
    grow(fresh, 5);
    LaneBatch reused(8, 64);
    grow(reused, 8); // keeps a matrix eight times larger
    reused.Reset(64);
    grow(reused, 5);
    Require(reused.PeakAmplitudeBytes() == fresh.PeakAmplitudeBytes() && reused.AmplitudeBytes() == fresh.AmplitudeBytes(),
            "a reused buffer changed the counted amplitude memory");
    const size_t budget = fresh.PeakAmplitudeBytes() - sizeof(std::complex<double>);
    reused.Reset(64);
    reused.SetAmplitudeBudget(budget);
    bool threw = false;
    try
    {
        grow(reused, 5);
    }
    catch (const MaestroExecution::ExtendedFrames::MemoryLimitExceeded &)
    {
        threw = true;
    }
    Require(threw, "a reused buffer admitted what a fresh batch exceeds");
    std::cout << "  reused buffers decide as fresh ones\n";
}

// The cached head is compacted and charged once, beside batches that share
// the rest; a shot that cannot fit beside it runs without it.
void HeadCountsAgainstTheBudget()
{
    LaneBatch head(6, 1);
    for (size_t q = 0; q < 6; ++q)
        head.ApplyAxisRotation(q, 0.7, 'X');
    std::vector<Word> outcome(head.Words());
    head.MeasureZ(
        0, [](size_t, double one) { return one > 0.5; }, outcome.data());
    Require(head.HeldAmplitudeBytes() > head.Components() * sizeof(std::complex<double>), "a measured batch kept no workspace");
    head.Compact();
    Require(head.HeldAmplitudeBytes() == head.Components() * sizeof(std::complex<double>) && head.AmplitudeBytes() == head.HeldAmplitudeBytes(),
            "a compacted head keeps workspaces");
    using Sampler = MaestroExecution::ExtendedFrameSampler;
    for (const size_t budget : {size_t(1) << 20, size_t(8) << 20, size_t(1) << 30})
        for (const size_t headBytes : {size_t(0), size_t(16), budget / 4, budget / 2, budget / 2 + 16, budget})
            for (const bool shared : {false, true})
            {
                const auto allowance = Sampler::DivideBudget(budget, headBytes, shared);
                const size_t charged = allowance.keepHead ? headBytes : 0;
                Require(charged + allowance.alone <= budget && charged + (shared ? 2 * Sampler::kMaxWorkers : 1) * allowance.batch <= budget,
                        "the head and the batches exceed the budget");
                Require(allowance.keepHead == (headBytes <= budget / 2) && (allowance.keepHead || allowance.alone == budget),
                        "the head was kept without leaving room for its copy");
            }
    // 18 rotations: a 4 MiB head. Beside it a copy and a measurement do not
    // fit 8 MiB, so the shot drops the head; its rows are unchanged.
    auto wide = std::make_shared<Circuit>();
    for (size_t q = 0; q < 18; ++q)
        Add<Circuits::RyGate<double>>(wide, q, 1.2);
    for (size_t q = 0; q < 18; ++q)
        wide->AddOperation(Measure(q, q));
    const auto plan = MaestroExecution::CompileExtendedFrameSampler(wide, 18, 18);
    std::vector<uint8_t> tight, ample;
    MaestroExecution::FrameOutcomes unused;
    Require(plan->SampleWithin(1, 1, {7, 8}, true, 1, 128, ample, unused), "the wide shot did not fit 128 MiB");
    Require(plan->SampleWithin(1, 1, {7, 8}, true, 1, 8, tight, unused) && tight == ample, "dropping the head changed the shot");

    // Compacting copies the matrix while the old block is live: it shrinks
    // only when both fit, and its peak counts the copy.
    for (const size_t budget : {size_t(8) << 20, size_t(64) << 20})
    {
        LaneBatch rotated(18, 1, budget);
        for (size_t q = 0; q < 18; ++q)
            rotated.ApplyAxisRotation(q, 1.2, 'Y');
        rotated.Compact();
        Require(rotated.PeakAmplitudeBytes() <= budget, "compaction exceeded the budget");
        Require(budget < (size_t(64) << 20) || rotated.HeldAmplitudeBytes() == rotated.Components() * sizeof(std::complex<double>),
                "an ample budget did not shrink the head");
    }

    // Unshared batches (Sample) run a shot that fits only without the head
    // without it, as when caching is off, rather than failing.
    auto measured = std::make_shared<Circuit>();
    for (size_t q = 0; q < 18; ++q)
        Add<Circuits::RyGate<double>>(measured, q, 1.2);
    Add<Circuits::HadamardGate<double>>(measured, 0);
    measured->AddOperation(Measure(0, 0));
    std::vector<std::vector<uint8_t>> unshared;
    for (const bool cache : {true, false})
    {
        auto sampler = MaestroExecution::CompileExtendedFrameSampler(measured, 18, 1);
        sampler->SetAmplitudeBudget(size_t(8) << 20);
        sampler->SetHeadCaching(cache);
        unshared.push_back(sampler->Sample(2, 2, 11, 1));
    }
    Require(unshared[0] == unshared[1], "the cached head changed unshared rows");
    std::cout << "  the cached head is compacted and counts against the budget\n";
}

// Measurement buffers count against the budget. The gates run on one lane and
// an exact-size expansion to 64 lanes sets the peak, so the smallest budget
// that admits them leaves no room for an off-diagonal measurement's next
// matrix next to the live one.
void MeasurementBuffersCountAgainstTheBudget()
{
    const size_t n = 6, lanes = 64;
    const auto prepare = [&](size_t budget) {
        auto batch = std::make_unique<LaneBatch>(n, 1, budget);
        for (size_t q = 0; q < n; ++q)
            batch->ApplyAxisRotation(q, 0.7, 'X');
        batch->ApplyH(0);
        batch->ExpandLanes(std::vector<size_t>(lanes, 0));
        return batch;
    };
    size_t low = 0, high = size_t(1) << 26;
    while (high - low > sizeof(std::complex<double>))
    {
        const size_t middle = low + (high - low) / 2;
        try
        {
            prepare(middle);
            high = middle;
        }
        catch (const MaestroExecution::ExtendedFrames::MemoryLimitExceeded &)
        {
            low = middle;
        }
    }
    auto batch = prepare(high);
    std::vector<Word> outcome(batch->Words());
    bool threw = false;
    try
    {
        batch->MeasureZ(
            0, [](size_t, double one) { return one > 0.5; }, outcome.data());
    }
    catch (const MaestroExecution::ExtendedFrames::MemoryLimitExceeded &)
    {
        threw = true;
    }
    Require(threw, "a measurement allocated beyond the budget");
    Require(batch->AmplitudeBytes() <= high, "the live amplitude buffers exceed the budget");
    std::cout << "  measurement buffers count against the budget\n";
}

void InvalidOperandsAndShapes()
{
    auto repeated = std::make_shared<Circuit>();
    Add<Circuits::HadamardGate<double>>(repeated, 0);
    Add<Circuits::TGate<double>>(repeated, 0);
    Add<Circuits::CXGate<double>>(repeated, 0, 0);
    repeated->AddOperation(Measure(0, 0));
    Require(Throws([&] { MaestroExecution::CompileExtendedFrameSampler(repeated, 1, 1); }), "CX(0,0) compiled");
    auto three = std::make_shared<Circuit>();
    Add<Circuits::CCXGate<double>>(three, 0, 1, 0);
    three->AddOperation(Measure(0, 0));
    Require(Throws([&] { MaestroExecution::CompileExtendedFrameSampler(three, 2, 1); }), "CCX(0,1,0) compiled");
    const auto plan = MaestroExecution::CompileExtendedFrameSampler(MidCircuitCircuit(), 3, 5);
    const MaestroExecution::ExtendedFrameSampler::Shape empty{0, size_t(1) << 20};
    Require(Throws([&] {
                plan->SampleCounts(10, 10, {1, 1}, 1, empty);
            }) &&
                Throws([&] {
                    plan->Sample(10, 10, {1, 1}, 1, empty);
                }),
            "a zero-lane shape was accepted");
    std::cout << "  repeated operands and zero-lane shapes are rejected\n";
}
} // namespace

int main()
{
    try
    {
        RunRandomCircuits(false);
        std::cout << "Unitary lanes match the scalar simulator\n";
        RunRandomCircuits(true);
        std::cout << "Measured and reset lanes match the scalar simulator\n";
        ExpandLanesCopiesStates();
        MeasurementsKeepLabelsShared(false);
        MeasurementsKeepLabelsShared(true);
        SharedRealizations();
        RowsCountsAndWorkers();
        FallbackAndLimits();
        SamplerMatchesDensityMatrix();
        ExecuteNoisyDispatch();
        HeadCachingAndBudgetedSampling();
        CheckpointSuffixesOnFrames();
        ThermalNoiseKeepsSampledSemantics();
        CheckpointSnapshotsItsPrefix();
        NoiseAndMeasurementSeedsAreIndependent();
        RowsIgnoreWorkersAndBudget();
        MemoryRetriesKeepEveryDraw();
        OutcomesIgnoreBatchMates();
        OversizeShotsKeepTheirDraws();
        RetriesFreeTheHead();
        ReferenceFramesKeepScalarRules();
        ReusedBuffersCountAsFresh();
        HeadCountsAgainstTheBudget();
        MeasurementBuffersCountAgainstTheBudget();
        InvalidOperandsAndShapes();
        std::cout << "Extended-frame tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
