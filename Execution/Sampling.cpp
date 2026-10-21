#include "Sampling.h"
#include "SimulatorConfig.h"
#include "Transforms/ConeAnalysis.h"
#include "Json/Json.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <vector>

namespace MaestroExecution
{
namespace
{
// Invert the hypergeometric PMF around its mode using stable probability
// ratios. No factorial cancellation, shot loop, or per-shot storage. Summation
// costs O(standard deviation); omitted tails are below floating-point precision.
size_t Hypergeometric(size_t population, size_t successes, size_t draws, std::mt19937_64 &rng)
{
    const size_t low = draws > population - successes ? draws - (population - successes) : 0;
    const size_t high = std::min(draws, successes);
    if (low == high)
        return low;
    const size_t mode = std::clamp(
        static_cast<size_t>((static_cast<long double>(successes) + 1) * (static_cast<long double>(draws) + 1) / (static_cast<long double>(population) + 2)),
        low, high);
    const auto down = [&](size_t k) {
        return static_cast<long double>(k) * (population - successes - (draws - k)) /
               ((static_cast<long double>(successes - k) + 1) * (static_cast<long double>(draws - k) + 1));
    };
    const auto up = [&](size_t k) {
        return static_cast<long double>(successes - k) * (draws - k) /
               ((static_cast<long double>(k) + 1) * (static_cast<long double>(population - successes - (draws - k)) + 1));
    };
    long double total = 1, weight = 1;
    size_t lower = mode, upper = mode;
    constexpr auto epsilon = std::numeric_limits<long double>::epsilon();
    while (lower > low)
    {
        weight *= down(lower--);
        total += weight;
        const auto ratio = lower > low ? down(lower) : 0;
        if (weight * ratio <= epsilon * total * (1 - ratio))
            break;
    }
    weight = 1;
    while (upper < high)
    {
        weight *= up(upper++);
        total += weight;
        const auto ratio = upper < high ? up(upper) : 0;
        if (weight * ratio <= epsilon * total * (1 - ratio))
            break;
    }
    auto target = std::generate_canonical<long double, 64>(rng) * total;
    if ((target -= 1) <= 0)
        return mode;
    weight = 1;
    for (size_t k = mode; k > lower;)
    {
        weight *= down(k--);
        if ((target -= weight) <= 0)
            return k;
    }
    weight = 1;
    for (size_t k = mode; k < upper;)
    {
        weight *= up(k++);
        if ((target -= weight) <= 0)
            return k;
    }
    return upper; // Roundoff at the last cumulative bin.
}
} // namespace

Network::INetwork<double>::ExecuteResults CombineIndependentCounts(const Network::INetwork<double>::ExecuteResults &joint,
                                                                   const Network::INetwork<double>::ExecuteResults &component, const std::vector<size_t> &bits,
                                                                   uint64_t seed)
{
    // Sorted buckets make a seed independent of unordered_map iteration order.
    std::map<std::vector<bool>, size_t> rows(joint.begin(), joint.end());
    std::vector<std::pair<std::vector<bool>, size_t>> columns(component.begin(), component.end());
    std::sort(columns.begin(), columns.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    size_t population = 0, other = 0;
    for (const auto &[outcome, count] : rows)
        population += count;
    for (const auto &[outcome, count] : columns)
    {
        if (outcome.size() < bits.size())
            throw std::runtime_error("Backend returned an invalid outcome width");
        other += count;
    }
    if (population != other)
        throw std::runtime_error("Backend returned an invalid shot count");
    Network::INetwork<double>::ExecuteResults result;
    result.reserve(rows.size());
    std::mt19937_64 rng(seed);
    size_t levels = 0;
    for (auto n = columns.size(); n; n >>= 1)
        ++levels;
    // Choose O(count * log(columns)) draws for sparse rows, O(columns)
    // hypergeometric allocations for large ones. Never expand a shot array.
    const size_t sparse_limit = levels ? columns.size() / levels : 0;
    std::vector<size_t> tree;
    bool tree_current = false;
    for (const auto &[prefix, count] : rows)
    {
        if (!count)
            continue;
        const auto append = [&](const std::vector<bool> &outcome, size_t take) {
            auto combined = prefix;
            for (size_t b = 0; b < bits.size(); ++b)
                combined.at(bits[b]) = outcome[b];
            result[std::move(combined)] += take;
        };
        if (count <= sparse_limit)
        {
            if (!tree_current)
            {
                // Build a Fenwick tree in linear time from the remaining urn.
                tree.assign(columns.size() + 1, 0);
                for (size_t i = 1; i < tree.size(); ++i)
                {
                    tree[i] += columns[i - 1].second;
                    const size_t parent = i + (i & (~i + 1));
                    if (parent < tree.size())
                        tree[parent] += tree[i];
                }
                tree_current = true;
            }
            for (size_t draw = 0; draw < count; ++draw)
            {
                // Integer ranks preserve the weights exactly, including empty
                // buckets. Removing each draw gives sampling without replacement.
                auto rank = std::uniform_int_distribution<size_t>(0, population - draw - 1)(rng);
                size_t column = 0;
                for (size_t step = size_t{1} << (levels - 1); step; step >>= 1)
                {
                    const size_t next = column + step;
                    if (next < tree.size() && tree[next] <= rank)
                    {
                        column = next;
                        rank -= tree[next];
                    }
                }
                --columns[column].second;
                for (size_t i = column + 1; i < tree.size(); i += i & (~i + 1))
                    --tree[i];
                append(columns[column].first, 1);
            }
        }
        else
        {
            size_t remaining = count, available = population;
            for (auto &[outcome, capacity] : columns)
            {
                if (!remaining)
                    break;
                if (!capacity)
                    continue;
                const auto take = Hypergeometric(available, capacity, remaining, rng);
                available -= capacity;
                capacity -= take;
                remaining -= take;
                if (take)
                    append(outcome, take);
            }
            // Rebuild lazily if a later row needs the tree. Updating every dense
            // allocation separately would add a logarithmic factor to this path.
            tree_current = false;
        }
        population -= count;
    }
    return result;
}

std::optional<SamplingResult> SampleDisjointCones(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const SimulatorConfig &config, size_t shots)
{
    if (!circuit || !config.enable_sampling_reduction || Simulators::IsDistributedGpuSimulator(config.simulator_type) || !shots)
        return std::nullopt;
    config.Validate();
    const auto start = std::chrono::steady_clock::now();
    auto body = std::make_shared<Circuits::Circuit<double>>();
    std::vector<std::shared_ptr<Circuits::MeasurementOperation<double>>> measurements;
    std::vector<size_t> measured;
    std::unordered_map<size_t, size_t> bit_owner;
    const auto width = circuit->GetMaxQubitIndex() + 1;
    size_t bits = width;
    for (const auto &op : circuit->GetOperations())
    {
        if (!op)
            return std::nullopt;
        if (const auto measurement = std::dynamic_pointer_cast<Circuits::MeasurementOperation<double>>(op))
        {
            measurements.push_back(measurement);
            for (size_t i = 0; i < measurement->GetQubits().size(); ++i)
            {
                const auto q = measurement->GetQubits()[i], bit = measurement->GetBitsIndices()[i];
                // Preserve the existing overwrite/readout semantics by falling
                // back when any classical destination is written more than once.
                if (!bit_owner.emplace(bit, q).second)
                    return std::nullopt;
                bits = std::max(bits, bit + 1);
                measured.push_back(q);
            }
        }
        else
        {
            if (!measurements.empty() && op->GetType() != Circuits::OperationType::kNoOp && op->GetType() != Circuits::OperationType::kDelay)
                return std::nullopt;
            body->AddOperation(op);
        }
    }
    if (measured.empty() || !CanTraceConeSupport(body, measured, HasEnsembleState(config.simulation_type), config.allow_reset_marginalization))
        return std::nullopt;
    const auto cones = TraceMeasurementCones(body, measured);
    // Preserve GPU tensor-network selection, which cannot sample a
    // single-qubit component. Ordinary full-register sampling remains valid.
    if (config.simulator_type == Simulators::SimulatorType::kGpuSim && config.simulation_type == Simulators::SimulationType::kTensorNetwork &&
        std::any_of(cones.begin(), cones.end(), [](const auto &cone) { return cone.mapping.size() == 1; }))
        return std::nullopt;
    if (cones.size() == 1 && cones.front().mapping.size() == width)
        return std::nullopt;

    SamplingResult result;
    result.components = cones.size();
    result.simulator = config.simulator_type;
    result.method = config.simulation_type;
    const auto seed = config.seed ? *config.seed : Simulators::GenerateRandomSeed(config.simulator_type, config.distributed_options);
    Network::INetwork<double>::ExecuteResults joint{{std::vector<bool>(bits, false), shots}};
    for (size_t c = 0; c < cones.size(); ++c)
    {
        const auto &cone = cones[c];
        auto reduced = std::make_shared<Circuits::Circuit<double>>();
        for (auto i : cone.operations)
            reduced->AddOperation(body->GetOperations()[i]->Remap(cone.mapping, {}));
        Circuits::Circuit<double>::BitMapping bit_map;
        std::vector<size_t> original_bits;
        for (const auto &m : measurements)
        {
            std::vector<std::pair<Types::qubit_t, size_t>> pairs;
            std::vector<Circuits::ReadoutRates> rates;
            for (size_t i = 0; i < m->GetQubits().size(); ++i)
                if (cone.mapping.count(m->GetQubits()[i]))
                {
                    const auto bit = m->GetBitsIndices()[i];
                    bit_map.emplace(bit, bit_map.size());
                    original_bits.push_back(bit);
                    pairs.emplace_back(cone.mapping.at(m->GetQubits()[i]), bit_map.at(bit));
                    rates.push_back(m->HasReadout() ? m->GetReadout()[i] : Circuits::ReadoutRates{});
                }
            if (!pairs.empty())
            {
                auto local = std::make_shared<Circuits::MeasurementOperation<double>>(pairs, m->GetDelay());
                local->SetReadout(std::move(rates));
                reduced->AddOperation(local);
            }
        }
        GetMaestroObjectWithMute();
        const auto handle = CreateSimpleSimulator(static_cast<int>(cone.mapping.size()));
        if (!handle)
            throw std::runtime_error("Failed to create sampling simulator");

        struct Cleanup
        {
            unsigned long handle;

            ~Cleanup()
            {
                DestroySimpleSimulator(handle);
            }
        } cleanup{handle};

        auto local_config = config;
        local_config.seed = Simulators::IState::DeriveSeed(seed, 2 * c);
        auto network = ConfigureNetwork(handle, local_config);
        if (!network || !network->GetSimulator())
            throw std::runtime_error("Failed to configure sampling simulator");
        if (config.fixed_backend &&
            (network->GetSimulator()->GetType() != config.simulator_type || network->GetSimulator()->GetSimulationType() != config.simulation_type))
            throw Error("backend_unavailable", "Requested fixed backend/method is unavailable");
        if (config.native_options.count("gpu_device") && network->GetSimulator()->GetGpuDevice() != std::stoi(config.native_options.at("gpu_device")))
            throw Error("backend_unavailable", "Requested GPU device was not selected");
        const auto counts = network->RepeatedExecuteOnHost(reduced, 0, shots);
        // Hypergeometric bucket splits have the same distribution as pairing
        // independent random permutations, while preserving both marginals.
        joint = CombineIndependentCounts(joint, counts, original_bits, Simulators::IState::DeriveSeed(seed, 2 * c + 1));
        if (!c)
        {
            result.simulator = network->GetLastSimulatorType();
            result.method = network->GetLastSimulationType();
            result.gpu_device = network->GetLastGpuDevice();
        }
        result.largest_component = std::max(result.largest_component, cone.mapping.size());
        result.max_bond_dimension = std::max(result.max_bond_dimension, network->GetCurrentMaxBondDimension());
        result.gate_fusion = result.gate_fusion || network->WasGateFusionEnabled();
        result.gate_fusion_max_qubits = std::max(result.gate_fusion_max_qubits, static_cast<size_t>(network->GetLastGateFusionMaxQubits()));
    }
    result.counts = std::move(joint);
    result.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

SamplingResult SampleCircuit(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const SimulatorConfig &config, size_t shots)
{
    if (!circuit)
        throw std::invalid_argument("Sampling requires a non-null circuit");
    config.Validate();
    if (auto reduced = SampleDisjointCones(circuit, config, shots))
        return std::move(*reduced);
    const auto start = std::chrono::steady_clock::now();
    GetMaestroObjectWithMute();
    const auto width = std::max(size_t{1}, circuit->GetMaxQubitIndex() + 1);
    const auto handle = CreateSimpleSimulator(static_cast<int>(width));
    if (!handle)
        throw std::runtime_error("Failed to create sampling simulator");

    struct Cleanup
    {
        unsigned long handle;

        ~Cleanup()
        {
            DestroySimpleSimulator(handle);
        }
    } cleanup{handle};

    auto network = ConfigureNetwork(handle, config);
    if (!network || !network->GetSimulator())
        throw Error("backend_unavailable", "Requested backend could not be created");
    if (config.fixed_backend &&
        (network->GetSimulator()->GetType() != config.simulator_type || network->GetSimulator()->GetSimulationType() != config.simulation_type))
        throw Error("backend_unavailable", "Requested fixed backend/method is unavailable");
    if (!Simulators::IsDistributedGpuSimulator(config.simulator_type) && config.native_options.count("gpu_device") &&
        network->GetSimulator()->GetGpuDevice() != std::stoi(config.native_options.at("gpu_device")))
        throw Error("backend_unavailable", "Requested GPU device was not selected");
    SamplingResult result;
    result.counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
    result.components = 1;
    result.largest_component = width;
    result.simulator = network->GetLastSimulatorType();
    result.method = network->GetLastSimulationType();
    result.gpu_device = network->GetLastGpuDevice();
    result.max_bond_dimension = network->GetCurrentMaxBondDimension();
    result.gate_fusion = network->WasGateFusionEnabled();
    result.gate_fusion_max_qubits = network->GetLastGateFusionMaxQubits();
    result.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}
} // namespace MaestroExecution
