#include "../Circuit/Circuit.h"
#include "../Execution/ReferenceFrameSampler.h"
#include "Clifford.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace
{
using Circuit = Circuits::Circuit<double>;
using Measurement = Circuits::MeasurementOperation<double>;

void Require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::shared_ptr<Measurement> Measure(size_t qubit, size_t bit)
{
    return std::make_shared<Measurement>(std::vector<std::pair<Types::qubit_t, size_t>>{{qubit, bit}});
}

std::vector<uint8_t> Sample(const std::shared_ptr<Circuit> &circuit, size_t qubits, size_t width, size_t shots = 10000, uint64_t seed = 9)
{
    const auto plan = MaestroExecution::CompileReferenceSampler(circuit, qubits, width);
    Require(plan.has_value(), "supported circuit rejected");
    Require(plan->Width() == width && plan->Qubits() == qubits, "plan dimensions");
    const auto result = plan->Sample(shots, seed);
    Require(result == plan->Sample(shots, seed), "seed reproducibility");
    return result;
}

// Independent full-tableau enumeration of all intrinsic measurement branches.
// Unlike the sampler, every branch owns and evolves its complete quantum state.
class OracleTableau : public QC::Clifford::StabilizerSimulator
{
  public:
    explicit OracleTableau(size_t qubits) : StabilizerSimulator(qubits)
    {
        SetMultithreading(false);
    }

    void Force(size_t qubit, bool outcome)
    {
        size_t pivot = 0;
        if (IsRandomResult(qubit, pivot))
            CollapseRandomQubit(qubit, pivot, outcome);
        else
            Require(MeasureQubit(qubit) == outcome, "impossible oracle branch");
    }
};

struct Branch
{
    OracleTableau state;
    std::string record;
    double weight = 1;

    Branch(size_t qubits, size_t width) : state(qubits), record(width, '0')
    {
    }
};

void ApplyGate(OracleTableau &state, const Circuits::IQuantumGate<double> &gate)
{
    const auto a = gate.GetQubit(0), b = gate.GetNumQubits() > 1 ? gate.GetQubit(1) : a;
    using Gate = Circuits::QuantumGateType;
    switch (gate.GetGateType())
    {
    case Gate::kHadamardGateType:
        state.ApplyH(a);
        break;
    case Gate::kSGateType:
        state.ApplyS(a);
        break;
    case Gate::kSdgGateType:
        state.ApplySdg(a);
        break;
    case Gate::kSxGateType:
        state.ApplySx(a);
        break;
    case Gate::kSxDagGateType:
        state.ApplySxDag(a);
        break;
    case Gate::kKGateType:
        state.ApplyK(a);
        break;
    case Gate::kXGateType:
        state.ApplyX(a);
        break;
    case Gate::kYGateType:
        state.ApplyY(a);
        break;
    case Gate::kZGateType:
        state.ApplyZ(a);
        break;
    case Gate::kCXGateType:
        state.ApplyCX(b, a);
        break;
    case Gate::kCYGateType:
        state.ApplyCY(b, a);
        break;
    case Gate::kCZGateType:
        state.ApplyCZ(b, a);
        break;
    case Gate::kSwapGateType:
        state.ApplySwap(a, b);
        break;
    default:
        throw std::runtime_error("unsupported oracle gate");
    }
}

std::map<std::string, double> Oracle(const Circuit &circuit, size_t qubits, size_t width)
{
    std::vector<Branch> branches;
    branches.emplace_back(qubits, width);
    auto measure = [&](size_t qubit, size_t bit, bool reset, bool resetTarget) {
        std::vector<Branch> next;
        for (const auto &branch : branches)
        {
            const double p = branch.state.GetQubitProbability(qubit);
            for (bool outcome : {false, true})
            {
                const double weight = outcome ? p : 1 - p;
                if (weight == 0)
                    continue;
                Branch copy = branch;
                copy.weight *= weight;
                copy.state.Force(qubit, outcome);
                if (reset)
                {
                    if (outcome != resetTarget)
                        copy.state.ApplyX(qubit);
                }
                else
                    copy.record[bit] = outcome ? '1' : '0';
                next.push_back(std::move(copy));
            }
        }
        branches = std::move(next);
    };
    for (const auto &operation : circuit.GetOperations())
    {
        if (operation->GetType() == Circuits::OperationType::kGate)
            for (auto &branch : branches)
                ApplyGate(branch.state, *std::static_pointer_cast<Circuits::IQuantumGate<double>>(operation));
        else if (operation->GetType() == Circuits::OperationType::kMeasurement)
        {
            const auto measurement = std::static_pointer_cast<Measurement>(operation);
            for (size_t i = 0; i < measurement->GetQubits().size(); ++i)
                measure(measurement->GetQubits()[i], measurement->GetBitsIndices()[i], false, false);
        }
        else if (operation->GetType() == Circuits::OperationType::kReset)
        {
            const auto reset = std::static_pointer_cast<Circuits::Reset<double>>(operation);
            for (size_t i = 0; i < reset->GetQubits().size(); ++i)
                measure(reset->GetQubits()[i], 0, true, false);
            for (size_t i = 0; i < reset->GetQubits().size() && i < reset->GetResetTargets().size(); ++i)
                if (reset->GetResetTargets()[i])
                    for (auto &branch : branches)
                        branch.state.ApplyX(reset->GetQubits()[i]);
        }
        else
            throw std::runtime_error("unsupported oracle operation");
    }
    std::map<std::string, double> result;
    for (const auto &branch : branches)
        result[branch.record] += branch.weight;
    return result;
}

void CheckDistribution(const std::shared_ptr<Circuit> &circuit, size_t qubits, size_t width)
{
    constexpr size_t shots = 8192;
    const auto expected = Oracle(*circuit, qubits, width);
    const auto samples = Sample(circuit, qubits, width, shots);
    std::map<std::string, size_t> observed;
    for (size_t shot = 0; shot < shots; ++shot)
    {
        std::string row(width, '0');
        for (size_t bit = 0; bit < width; ++bit)
            row[bit] += samples[shot * width + bit];
        Require(expected.count(row) != 0, "sample outside exact tableau support");
        ++observed[row];
    }
    for (const auto &[row, probability] : expected)
    {
        const double frequency = double(observed[row]) / shots;
        const double tolerance = 7 * std::sqrt(probability * (1 - probability) / shots) + 2.0 / shots;
        Require(std::abs(frequency - probability) < tolerance, "packed sampler disagrees with exact tableau distribution");
    }
}

void BellAndMeasurements()
{
    auto bell = std::make_shared<Circuit>();
    bell->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    bell->AddOperation(std::make_shared<Circuits::CXGate<double>>(0, 1));
    bell->AddOperation(Measure(0, 0));
    bell->AddOperation(Measure(1, 1));
    bell->AddOperation(Measure(0, 2));
    CheckDistribution(bell, 2, 4);
    const auto plan = MaestroExecution::CompileReferenceSampler(bell, 2, 4);
    Require(plan->Sample(0, 9).empty(), "zero shots");
    const auto first = plan->Sample(64, 9), later = plan->Sample(129, 9);
    Require(std::equal(first.begin(), first.end(), later.begin()), "block stream prefix");
    Require(plan->Sample(129, 10) != later, "distinct seeds reused a shot stream");

    auto incompatible = std::make_shared<Circuit>();
    incompatible->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    incompatible->AddOperation(Measure(0, 0));
    incompatible->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    incompatible->AddOperation(Measure(0, 1));
    incompatible->AddOperation(Measure(0, 2));
    CheckDistribution(incompatible, 1, 3);

    auto wide = std::make_shared<Circuit>();
    wide->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(69));
    wide->AddOperation(std::make_shared<Circuits::CXGate<double>>(69, 0));
    wide->AddOperation(Measure(0, 0));
    wide->AddOperation(Measure(69, 70));
    const auto samples = Sample(wide, 70, 72, 129);
    for (size_t shot = 0; shot < 129; ++shot)
    {
        Require(samples[shot * 72] == samples[shot * 72 + 70], "wide gauge or partial shot word");
        for (size_t bit = 1; bit < 72; ++bit)
            if (bit != 70)
                Require(samples[shot * 72 + bit] == 0, "unused classical bit changed");
    }
}

void RandomCircuits()
{
    std::mt19937 rng(31);
    for (size_t trial = 0; trial < 40; ++trial)
    {
        auto circuit = std::make_shared<Circuit>();
        size_t bit = 0;
        for (size_t step = 0; step < 18; ++step)
        {
            const size_t a = rng() % 3, b = (a + 1 + rng() % 2) % 3;
            switch (rng() % 14)
            {
            case 0:
                circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(a));
                break;
            case 1:
                circuit->AddOperation(std::make_shared<Circuits::SGate<double>>(a));
                break;
            case 2:
                circuit->AddOperation(std::make_shared<Circuits::SdgGate<double>>(a));
                break;
            case 3:
                circuit->AddOperation(std::make_shared<Circuits::SxGate<double>>(a));
                break;
            case 4:
                circuit->AddOperation(std::make_shared<Circuits::SxDagGate<double>>(a));
                break;
            case 5:
                circuit->AddOperation(std::make_shared<Circuits::KGate<double>>(a));
                break;
            case 6:
                circuit->AddOperation(std::make_shared<Circuits::XGate<double>>(a));
                break;
            case 7:
                circuit->AddOperation(std::make_shared<Circuits::YGate<double>>(a));
                break;
            case 8:
                circuit->AddOperation(std::make_shared<Circuits::ZGate<double>>(a));
                break;
            case 9:
                circuit->AddOperation(std::make_shared<Circuits::CXGate<double>>(a, b));
                break;
            case 10:
                circuit->AddOperation(std::make_shared<Circuits::CYGate<double>>(a, b));
                break;
            case 11:
                circuit->AddOperation(std::make_shared<Circuits::CZGate<double>>(a, b));
                break;
            case 12:
                circuit->AddOperation(std::make_shared<Circuits::SwapGate<double>>(a, b));
                break;
            case 13:
                circuit->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{a}, 0, std::vector<bool>{bool(rng() % 2)}));
                break;
            }
            if (step % 3 == 2)
                circuit->AddOperation(Measure(rng() % 3, bit++));
        }
        // Reuse one bit, checking final classical-register semantics.
        circuit->AddOperation(Measure(1, 0));
        CheckDistribution(circuit, 3, bit);
    }
}

void NoiseAndReadout()
{
    auto noise = std::make_shared<Circuit>();
    std::vector<double> probabilities(16, 0);
    probabilities[1] = 0.25; // X on target zero only.
    probabilities[4] = 0.75; // X on target one only.
    noise->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0, 1}, Simulators::QuantumChannel::Pauli(probabilities)));
    noise->AddOperation(Measure(0, 0));
    noise->AddOperation(Measure(1, 1));
    const auto noisy = Sample(noise, 2, 2);
    size_t ones = 0;
    for (size_t shot = 0; shot < noisy.size() / 2; ++shot)
    {
        Require(noisy[2 * shot] != noisy[2 * shot + 1], "correlated two-target Pauli channel lost exclusivity");
        ones += noisy[2 * shot];
    }
    Require(std::abs(double(ones) / 10000 - 0.25) < 0.025, "Pauli categorical rates");

    auto phase = std::make_shared<Circuit>();
    phase->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    phase->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseFlip(1)));
    phase->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    phase->AddOperation(Measure(0, 0));
    const auto phaseSamples = Sample(phase, 1, 1, 129);
    Require(std::all_of(phaseSamples.begin(), phaseSamples.end(), [](uint8_t value) { return value == 1; }), "Z-frame propagation through H");

    auto readout = std::make_shared<Circuit>();
    auto flipped = Measure(0, 0);
    flipped->SetReadout({{1, 0}});
    readout->AddOperation(flipped);
    readout->AddOperation(Measure(0, 1));
    readout->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{0}, 0, std::vector<bool>{true}));
    auto flippedOne = Measure(0, 2);
    flippedOne->SetReadout({{0, 1}});
    readout->AddOperation(flippedOne);
    readout->AddOperation(Measure(0, 3));
    const auto samples = Sample(readout, 1, 4, 129);
    for (size_t shot = 0; shot < 129; ++shot)
        Require(samples[4 * shot] == 1 && samples[4 * shot + 1] == 0 && samples[4 * shot + 2] == 0 && samples[4 * shot + 3] == 1,
                "readout changed quantum state or reset target");

    auto entangledReset = std::make_shared<Circuit>();
    entangledReset->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    entangledReset->AddOperation(std::make_shared<Circuits::CXGate<double>>(0, 1));
    entangledReset->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{0}));
    entangledReset->AddOperation(Measure(0, 0));
    entangledReset->AddOperation(Measure(1, 1));
    CheckDistribution(entangledReset, 2, 2);

    auto repeatedReset = std::make_shared<Circuit>();
    repeatedReset->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    repeatedReset->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{0, 0}, 0, std::vector<bool>{true, true}));
    repeatedReset->AddOperation(Measure(0, 0));
    CheckDistribution(repeatedReset, 1, 1);
}

std::shared_ptr<Circuits::QuantumChannelOperation<double>> Channel(Types::qubits_vector targets, std::vector<double> probabilities)
{
    return std::make_shared<Circuits::QuantumChannelOperation<double>>(std::move(targets), Simulators::QuantumChannel::Pauli(probabilities));
}

// |observed - expected| within five binomial standard deviations.
void RequireRate(size_t observed, size_t trials, double probability, const std::string &message)
{
    const double expected = trials * probability;
    const double sigma = std::sqrt(trials * probability * (1 - probability));
    Require(std::abs(double(observed) - expected) <= 5 * sigma + 1,
            message + ": observed " + std::to_string(observed) + ", expected " + std::to_string(expected));
}

size_t CountOnes(const std::vector<uint8_t> &samples, size_t width, size_t firstBit, size_t endBit)
{
    size_t ones = 0;
    for (size_t row = 0; row < samples.size() / width; ++row)
        for (size_t bit = firstBit; bit < endBit; ++bit)
            ones += samples[row * width + bit];
    return ones;
}

void SparseAndDenseNoiseRates()
{
    // Sparse (geometric) and dense (per-lane) single-Pauli rates, observed directly.
    constexpr size_t qubits = 32, shots = 40000;
    for (const double rate : {0.0007, 0.003, 0.05, 0.3})
    {
        auto circuit = std::make_shared<Circuit>();
        for (size_t q = 0; q < qubits; ++q)
            circuit->AddOperation(Channel({q}, {1 - rate, rate, 0, 0}));
        for (size_t q = 0; q < qubits; ++q)
            circuit->AddOperation(Measure(q, q));
        const auto samples = Sample(circuit, qubits, qubits, shots, 101);
        RequireRate(CountOnes(samples, qubits, 0, qubits), qubits * shots, rate, "single-Pauli error rate");
        // Errors are independent across shots: the per-shot count is binomial.
        double sum = 0, squares = 0;
        size_t firstHalf = 0, secondHalf = 0;
        for (size_t row = 0; row < shots; ++row)
        {
            size_t count = 0;
            for (size_t q = 0; q < qubits; ++q)
                count += samples[row * qubits + q];
            sum += double(count);
            squares += double(count) * double(count);
            ((row % 64) < 32 ? firstHalf : secondHalf) += count;
        }
        const double mean = sum / shots, variance = squares / shots - mean * mean;
        const double expectedVariance = qubits * rate * (1 - rate);
        // Standard error of a binomial sample variance: sqrt((2 + excess kurtosis) / shots).
        const double relativeError = std::sqrt((2 + (1 - 6 * rate * (1 - rate)) / expectedVariance) / shots);
        Require(std::abs(variance - expectedVariance) < 5 * relativeError * expectedVariance, "per-shot error counts are not binomial");
        // Lanes inside a word see the same rate.
        RequireRate(firstHalf, qubits * shots / 2, rate, "lower lanes of a word");
        RequireRate(secondHalf, qubits * shots / 2, rate, "upper lanes of a word");
    }

    // Sparse Pauli choice: X/Y flips a Z measurement; Z/Y flips after H...H.
    auto choice = std::make_shared<Circuit>();
    for (size_t q = 0; q < qubits; ++q)
    {
        if (q % 2)
            choice->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(q));
        choice->AddOperation(Channel({q}, {0.993, 0.002, 0.001, 0.004}));
        if (q % 2)
            choice->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(q));
        choice->AddOperation(Measure(q, q));
    }
    const auto chosen = Sample(choice, qubits, qubits, shots, 202);
    size_t xFlips = 0, zFlips = 0;
    for (size_t row = 0; row < shots; ++row)
        for (size_t q = 0; q < qubits; ++q)
            (q % 2 ? zFlips : xFlips) += chosen[row * qubits + q];
    RequireRate(xFlips, qubits / 2 * shots, 0.003, "X or Y term choice");
    RequireRate(zFlips, qubits / 2 * shots, 0.005, "Z or Y term choice");

    // Sparse two-qubit depolarizing: 8 of 15 terms flip each target, 4 flip both.
    auto pair = std::make_shared<Circuit>();
    std::vector<double> depolarizing(16, 0.01 / 15);
    depolarizing[0] = 0.99;
    for (size_t q = 0; q < qubits; q += 2)
        pair->AddOperation(Channel({q, q + 1}, depolarizing));
    for (size_t q = 0; q < qubits; ++q)
        pair->AddOperation(Measure(q, q));
    const auto pairs = Sample(pair, qubits, qubits, shots, 303);
    size_t firstFlips = 0, secondFlips = 0, bothFlips = 0;
    for (size_t row = 0; row < shots; ++row)
        for (size_t q = 0; q < qubits; q += 2)
        {
            firstFlips += pairs[row * qubits + q];
            secondFlips += pairs[row * qubits + q + 1];
            bothFlips += pairs[row * qubits + q] & pairs[row * qubits + q + 1];
        }
    RequireRate(firstFlips, qubits / 2 * shots, 0.01 * 8 / 15, "first target of a two-qubit channel");
    RequireRate(secondFlips, qubits / 2 * shots, 0.01 * 8 / 15, "second target of a two-qubit channel");
    RequireRate(bothFlips, qubits / 2 * shots, 0.01 * 4 / 15, "correlated flips of a two-qubit channel");
}

void AsymmetricReadoutRates()
{
    constexpr size_t qubits = 32, shots = 40000;
    for (const auto &[p01, p10] : std::vector<std::pair<double, double>>{{0.002, 0.006}, {0.006, 0.002}, {0.1, 0.3}, {0, 0.004}, {0.2, 1}})
    {
        auto circuit = std::make_shared<Circuit>();
        for (size_t q = 0; q < qubits; ++q)
        {
            if (q % 2)
                circuit->AddOperation(std::make_shared<Circuits::XGate<double>>(q));
            auto measured = Measure(q, q);
            measured->SetReadout({{p01, p10}});
            circuit->AddOperation(measured);
        }
        const auto samples = Sample(circuit, qubits, qubits, shots, 404);
        size_t zeroFlips = 0, oneFlips = 0;
        for (size_t row = 0; row < shots; ++row)
            for (size_t q = 0; q < qubits; ++q)
                (q % 2 ? oneFlips : zeroFlips) += q % 2 ? 1 - samples[row * qubits + q] : samples[row * qubits + q];
        RequireRate(zeroFlips, qubits / 2 * shots, p01, "readout of prepared zero");
        RequireRate(oneFlips, qubits / 2 * shots, p10, "readout of prepared one");
    }
}

void StreamsArePrefixStable()
{
    // Random gauges, sparse and dense channels and readout, across batch boundaries.
    auto circuit = std::make_shared<Circuit>();
    for (size_t q = 0; q < 6; ++q)
        circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(q));
    for (size_t q = 0; q + 1 < 6; ++q)
        circuit->AddOperation(std::make_shared<Circuits::CXGate<double>>(q, q + 1));
    circuit->AddOperation(Channel({0}, {0.997, 0.001, 0.001, 0.001}));
    circuit->AddOperation(Channel({1, 2}, std::vector<double>{0.7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02}));
    for (size_t q = 0; q < 6; ++q)
    {
        auto measured = Measure(q, q);
        measured->SetReadout({{0.01, 0.002}});
        circuit->AddOperation(measured);
    }
    circuit->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{3}));
    circuit->AddOperation(Measure(3, 6));
    const auto plan = MaestroExecution::CompileReferenceSampler(circuit, 6, 7);
    Require(plan.has_value(), "prefix circuit rejected");
    const auto longer = plan->Sample(2 * MaestroExecution::ReferenceFrameSampler::kBatchShots + 300, 77);
    for (const size_t shots : {size_t{1}, size_t{63}, size_t{65}, size_t{700}, MaestroExecution::ReferenceFrameSampler::kBatchShots + 1})
    {
        const auto shorter = plan->Sample(shots, 77);
        Require(std::equal(shorter.begin(), shorter.end(), longer.begin()), "a shorter request changed its leading rows");
    }
    Require(CountOnes(longer, 7, 6, 7) == 0, "reset did not clear its frame");
}

// SampleCounts lists every distinct record exactly once.
std::unordered_map<std::vector<bool>, size_t> AsHistogram(const MaestroExecution::FrameOutcomes &outcomes)
{
    std::unordered_map<std::vector<bool>, size_t> histogram;
    for (const auto &[bits, count] : outcomes)
        Require(count > 0 && histogram.emplace(bits, count).second, "frame outcomes repeat a record or list an empty count");
    return histogram;
}

void StreamingCountsMatchRows()
{
    auto circuit = std::make_shared<Circuit>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<double>>(0, 1));
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(2));
    circuit->AddOperation(Channel({0}, {0.997, 0.001, 0.001, 0.001}));
    circuit->AddOperation(Channel({1, 2}, {0.7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02}));
    auto first = Measure(0, 0);
    first->SetReadout({{0.01, 0.01}});
    circuit->AddOperation(first);
    auto second = Measure(1, 1);
    second->SetReadout({{0.2, 0.2}});
    circuit->AddOperation(second);
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    circuit->AddOperation(Measure(0, 2));
    circuit->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{2}));
    circuit->AddOperation(Measure(2, 70));
    const auto plan = MaestroExecution::CompileReferenceSampler(circuit, 3, 71);
    Require(plan.has_value(), "streaming-count fixture rejected");

    // Captured from the previous frame kernel, before sharing batch evolution
    // with counts. Ordinary-rate seeded rows must survive this refactor.
    uint64_t hash = 14695981039346656037ULL;
    for (const auto value : plan->Sample(2050, 77))
        hash = (hash ^ value) * 1099511628211ULL;
    Require(hash == 0xd130f063a6c7525cULL, "existing seeded frame rows changed");

    for (const size_t shots : {size_t{0}, size_t{1}, size_t{63}, size_t{64}, size_t{65}, size_t{1023}, size_t{1024}, size_t{1025}, size_t{2050}})
        for (const uint64_t seed : {uint64_t{0}, uint64_t{77}, std::numeric_limits<uint64_t>::max()})
        {
            const auto rows = plan->Sample(shots, seed);
            std::unordered_map<std::vector<bool>, size_t> expected;
            for (size_t shot = 0; shot < shots; ++shot)
            {
                std::vector<bool> bits(plan->Width());
                for (size_t bit = 0; bit < bits.size(); ++bit)
                    bits[bit] = rows[shot * bits.size() + bit] != 0;
                ++expected[bits];
            }
            Require(AsHistogram(plan->SampleCounts(shots, seed)) == expected, "streamed histogram differs from seeded sample rows");
        }

    // Many shots and one wide outcome: histogram storage should depend on the
    // occupied outcome, not require a dense shots-by-width byte matrix.
    auto wide = std::make_shared<Circuit>();
    wide->AddOperation(std::make_shared<Circuits::XGate<double>>(0));
    wide->AddOperation(Measure(0, 1024));
    const auto widePlan = MaestroExecution::CompileReferenceSampler(wide, 1, 1025);
    const auto counts = AsHistogram(widePlan->SampleCounts(100003, 5));
    std::vector<bool> expected(1025, false);
    expected[1024] = true;
    Require(counts.size() == 1 && counts.at(expected) == 100003, "wide sparse histogram lost shots or classical positions");
}

void SparseQubitsCompact()
{
    // A circuit on qubits {3, 9} of a 12-qubit register compiles to two frame
    // rows and gives exactly the seeded rows of its relabelling onto {0, 1}.
    const auto build = [](size_t a, size_t b) {
        auto circuit = std::make_shared<Circuit>();
        circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(a));
        circuit->AddOperation(std::make_shared<Circuits::CXGate<double>>(a, b));
        circuit->AddOperation(std::make_shared<Circuits::SGate<double>>(b));
        circuit->AddOperation(Channel({a, b}, {0.7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02}));
        auto first = Measure(a, 0);
        first->SetReadout({{0.05, 0.1}});
        circuit->AddOperation(first);
        circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(a));
        circuit->AddOperation(Measure(a, 1));
        circuit->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{b}));
        circuit->AddOperation(Measure(b, 2));
        return circuit;
    };
    const auto sparse = MaestroExecution::CompileReferenceSampler(build(3, 9), 12, 3);
    const auto dense = MaestroExecution::CompileReferenceSampler(build(0, 1), 2, 3);
    Require(sparse.has_value() && dense.has_value(), "sparse frame fixture rejected");
    Require(sparse->Qubits() == 12 && sparse->FrameQubits() == 2 && dense->FrameQubits() == 2, "sparse frame rows were not compacted");
    for (const size_t shots : {size_t{1}, size_t{1025}, size_t{3000}})
        for (const uint64_t seed : {uint64_t{0}, uint64_t{77}})
        {
            Require(sparse->Sample(shots, seed) == dense->Sample(shots, seed), "compaction changed seeded frame rows");
            Require(AsHistogram(sparse->SampleCounts(shots, seed)) == AsHistogram(dense->SampleCounts(shots, seed)), "compaction changed frame counts");
        }

    // One used qubit at an index whose dense tableau could not be allocated.
    const size_t far = size_t{1} << 40;
    auto distant = std::make_shared<Circuit>();
    distant->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(far));
    distant->AddOperation(Measure(far, 0));
    const auto farPlan = MaestroExecution::CompileReferenceSampler(distant, far + 1, 1);
    Require(farPlan.has_value() && farPlan->FrameQubits() == 1, "a distant qubit was not compacted");
    const auto rows = farPlan->Sample(4096, 5);
    RequireRate(size_t(std::count(rows.begin(), rows.end(), uint8_t{1})), rows.size(), 0.5, "distant qubit lost its distribution");
}

void KrausPauliMixtures()
{
    // Phase damping is a Z flip with p = (1 - sqrt(1 - gamma)) / 2: H, damping,
    // H makes that flip a measured X error. Here gamma = 0.36 gives p = 0.1.
    auto damped = std::make_shared<Circuit>();
    damped->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    damped->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseDamping(0.36)));
    damped->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(0));
    damped->AddOperation(Measure(0, 0));
    const auto plan = MaestroExecution::CompileReferenceSampler(damped, 1, 1);
    Require(plan.has_value(), "phase damping did not compile as a Pauli channel");
    const auto rows = plan->Sample(40000, 9);
    RequireRate(size_t(std::count(rows.begin(), rows.end(), uint8_t{1})), rows.size(), 0.1, "phase-damping flip rate");

    // A two-target channel given only as Kraus operators: sqrt(0.7) I and
    // sqrt(0.3) X on target 0 with Z on target 1 (Pauli index 1 + 4 * 3).
    std::vector<double> single(16, 0.0);
    single[1 + 4 * 3] = 1.0;
    const auto xz = Simulators::QuantumChannel::Pauli(single).GetKrausOperators().front();
    Simulators::QuantumChannel::KrausOperators kraus{std::sqrt(0.7) * Simulators::QuantumChannel::Matrix::Identity(4, 4), std::sqrt(0.3) * xz};
    auto paired = std::make_shared<Circuit>();
    paired->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0, 1}, Simulators::QuantumChannel(kraus)));
    paired->AddOperation(Measure(0, 0));
    paired->AddOperation(Measure(1, 1));
    const auto pairPlan = MaestroExecution::CompileReferenceSampler(paired, 2, 2);
    Require(pairPlan.has_value(), "a Kraus-only two-qubit Pauli channel did not compile");
    const auto pairRows = pairPlan->Sample(40000, 4);
    size_t flips = 0;
    for (size_t shot = 0; shot < 40000; ++shot)
    {
        Require(pairRows[2 * shot + 1] == 0, "a Z error flipped a measured |0>");
        flips += pairRows[2 * shot];
    }
    RequireRate(flips, 40000, 0.3, "two-qubit Kraus Pauli rate");
}

void PackedRowHashSpreadsHighBits()
{
    // MSVC's unordered_map picks a bucket by masking the low hash bits. Records
    // that differ only in high bits of a word must still spread across those
    // bits, or counting degrades quadratically. 4096 random keys occupy about
    // 1 - 1/e of 4096 buckets; a word-wise FNV combine leaves them in one.
    constexpr size_t keys = 4096;
    for (const size_t shift : {size_t{40}, size_t{52}})
        for (const size_t word : {size_t{0}, size_t{1}})
        {
            std::vector<bool> occupied(keys, false);
            for (uint64_t key = 0; key < keys; ++key)
            {
                uint64_t row[2] = {0, 0};
                row[word] = key << shift;
                occupied[MaestroExecution::FrameDetail::HashPackedRow(row, 2) & (keys - 1)] = true;
            }
            Require(size_t(std::count(occupied.begin(), occupied.end(), true)) > keys / 2, "packed-row hash ignores high bits in its low bits");
        }

    // The public count maps (Circuit::ExecuteResults) key vector<bool> records.
    // Records that differ only in bit 7 of each byte must spread too: MSVC's
    // byte-wise std::hash leaves them in 32 of these 4096 buckets.
    {
        std::vector<bool> occupied(keys, false);
        for (uint64_t key = 0; key < keys; ++key)
        {
            std::vector<bool> record(320, false);
            for (size_t bit = 0; bit < 12; ++bit)
                record[8 * (3 * bit + 1) + 7] = ((key >> bit) & 1U) != 0;
            occupied[Types::BitsHash{}(record) & (keys - 1)] = true;
        }
        Require(size_t(std::count(occupied.begin(), occupied.end(), true)) > keys / 2, "count-map hash ignores byte-high record bits");
        static_assert(std::is_same_v<Circuits::Circuit<>::ExecuteResults::hasher, Types::BitsHash>, "count maps must use the mixed record hash");
    }

    // End to end: outcomes that differ only in high bits of the record word.
    auto circuit = std::make_shared<Circuit>();
    for (size_t q = 23; q < 63; ++q)
        circuit->AddOperation(std::make_shared<Circuits::HadamardGate<double>>(q));
    for (size_t q = 0; q < 64; ++q)
        circuit->AddOperation(Measure(q, q));
    const auto plan = MaestroExecution::CompileReferenceSampler(circuit, 64, 64);
    Require(plan.has_value(), "high-bit record circuit rejected");
    const auto counts = AsHistogram(plan->SampleCounts(20000, 3));
    size_t total = 0;
    for (const auto &[bits, count] : counts)
    {
        total += count;
        Require(!bits[0] && !bits[22] && !bits[63], "unmeasured-random record bit changed");
    }
    Require(total == 20000 && counts.size() > 19990, "high-bit records lost shots or distinct outcomes");
}

void MicroscopicNoiseRates()
{
    for (const double rate : {std::numeric_limits<double>::denorm_min(), 1e-320, 1e-309, 1e-300})
    {
        auto circuit = std::make_shared<Circuit>();
        circuit->AddOperation(Channel({0}, {1, rate, 0, 0}));
        auto measured = Measure(0, 0);
        measured->SetReadout({{rate, 0}});
        circuit->AddOperation(measured);
        std::feclearexcept(FE_ALL_EXCEPT);
        const auto plan = MaestroExecution::CompileReferenceSampler(circuit, 1, 1);
        const auto counts = AsHistogram(plan->SampleCounts(2050, 77));
        Require((std::fetestexcept(FE_OVERFLOW | FE_INVALID)) == 0, "microscopic rates overflowed a geometric gap");
        Require(counts.size() == 1 && counts.at(std::vector<bool>{false}) == 2050, "microscopic-rate sampling produced an unexpected event");
    }
}

void FallbackAndValidation()
{
    auto unsupported = std::make_shared<Circuit>();
    unsupported->AddOperation(std::make_shared<Circuits::TGate<double>>(0));
    unsupported->AddOperation(Measure(0, 0));
    Require(!MaestroExecution::CompileReferenceSampler(unsupported, 1, 1), "non-Clifford circuit accepted");
    auto channel = std::make_shared<Circuit>();
    channel->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(0.3)));
    channel->AddOperation(Measure(0, 0));
    Require(!MaestroExecution::CompileReferenceSampler(channel, 1, 1), "non-Pauli channel accepted");
    auto conditional = std::make_shared<Circuit>();
    conditional->AddOperation(Measure(0, 0));
    conditional->AddOperation(std::make_shared<Circuits::ConditionalGate<double>>(
        std::make_shared<Circuits::XGate<double>>(0), std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{0}, std::vector<bool>{true})));
    Require(!MaestroExecution::CompileReferenceSampler(conditional, 1, 1), "unsupported classical control accepted");
    // Execution treats a null condition as a no-op, but copying such a gate
    // dereferences it. Compaction must never copy operations: a sparse circuit
    // falls back exactly like its dense equivalent.
    for (const size_t qubit : {size_t{0}, size_t{9}})
    {
        auto unconditioned = std::make_shared<Circuit>();
        unconditioned->AddOperation(Measure(qubit, 0));
        unconditioned->AddOperation(std::make_shared<Circuits::ConditionalGate<double>>(std::make_shared<Circuits::XGate<double>>(qubit), nullptr));
        Require(!MaestroExecution::CompileReferenceSampler(unconditioned, qubit + 1, 1), "a null-condition gate did not fall back");
    }
    auto nested = std::make_shared<Circuit>();
    auto child = std::make_shared<Circuit>();
    child->AddOperation(std::make_shared<Circuits::XGate<double>>(0));
    child->AddOperation(Measure(0, 0));
    nested->AddOperation(child);
    const auto nestedSamples = Sample(nested, 1, 1, 129);
    Require(std::all_of(nestedSamples.begin(), nestedSamples.end(), [](uint8_t value) { return value == 1; }), "nested circuit compilation");
    auto empty = std::make_shared<Circuit>();
    Require(!MaestroExecution::CompileReferenceSampler(empty, 1, 0), "implicit measurements need ordinary execution");
    auto invalid = std::make_shared<Circuit>();
    invalid->AddOperation(Measure(1, 0));
    bool threw = false;
    try
    {
        MaestroExecution::CompileReferenceSampler(invalid, 1, 1);
    }
    catch (const std::out_of_range &)
    {
        threw = true;
    }
    Require(threw, "invalid dimensions were silently downgraded to fallback");
}
} // namespace

int main()
{
    try
    {
        BellAndMeasurements();
        RandomCircuits();
        NoiseAndReadout();
        SparseAndDenseNoiseRates();
        AsymmetricReadoutRates();
        StreamsArePrefixStable();
        StreamingCountsMatchRows();
        SparseQubitsCompact();
        KrausPauliMixtures();
        PackedRowHashSpreadsHighBits();
        MicroscopicNoiseRates();
        FallbackAndValidation();
        std::cout << "Reference-frame sampler tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
