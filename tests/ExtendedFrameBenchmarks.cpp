// Manual before/after benchmark of noisy extended_stabilizer sampling:
// scalar trajectories (counts_policy legacy) against batched Pauli frames
// (extended_frame_v1) on the same requests and noise semantics. Excluded
// from normal builds and CTest because timings depend on the host.
//
//   extended_frame_benchmarks [repeats]

#include "../Execution/SampleRecords.h"
#include "../Noise/NoiseModel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Circuit = Circuits::Circuit<double>;

template <class Gate, class... Args> void Add(const std::shared_ptr<Circuit> &circuit, Args... args)
{
    circuit->AddOperation(std::make_shared<Gate>(args...));
}

void Measure(const std::shared_ptr<Circuit> &circuit, size_t qubit, size_t bit)
{
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<double>>(std::vector<std::pair<Types::qubit_t, size_t>>{{qubit, bit}}));
}

struct Workload
{
    std::string name;
    std::shared_ptr<Circuit> circuit;
    noise::NoiseModel model;
    size_t width;
    size_t shots;
    std::optional<size_t> realizations; // nullopt: the default schedule (64)
};

noise::NoiseModel DeviceModel(size_t qubits, const std::vector<std::pair<int, int>> &pairs)
{
    noise::NoiseModel model;
    for (size_t q = 0; q < qubits; ++q)
    {
        model.set_depolarizing(int(q), 1e-3);
        model.set_readout_error(int(q), 1e-2, 2e-2);
    }
    for (const auto &[a, b] : pairs)
        model.set_2q_depolarizing(a, b, 1e-2);
    return model;
}

// Brickwork Clifford layers with a few T gates in superposition, measured at the end.
Workload CliffordT(size_t qubits, size_t layers, size_t tGates, size_t shots, std::optional<size_t> realizations, const std::string &name)
{
    std::mt19937_64 rng(qubits * 1000 + layers);
    auto c = std::make_shared<Circuit>();
    std::vector<std::pair<int, int>> pairs;
    for (size_t q = 0; q + 1 < qubits; ++q)
        pairs.emplace_back(int(q), int(q + 1));
    size_t placed = 0;
    for (size_t layer = 0; layer < layers; ++layer)
    {
        for (size_t q = 0; q < qubits; ++q)
        {
            switch (rng() % 3)
            {
            case 0:
                Add<Circuits::HadamardGate<double>>(c, q);
                break;
            case 1:
                Add<Circuits::SGate<double>>(c, q);
                break;
            default:
                Add<Circuits::SxGate<double>>(c, q);
                break;
            }
        }
        if (layer % 2 == 1 && placed < tGates)
            for (size_t q = 0; q < qubits && placed < tGates; q += 3, ++placed)
            {
                Add<Circuits::HadamardGate<double>>(c, q);
                Add<Circuits::TGate<double>>(c, q);
            }
        for (size_t q = layer % 2; q + 1 < qubits; q += 2)
            Add<Circuits::CXGate<double>>(c, q, q + 1);
    }
    for (size_t q = 0; q < qubits; ++q)
        Measure(c, q, q);
    return {name, c, DeviceModel(qubits, pairs), qubits, shots, realizations};
}

// Repeated stabilizer-style rounds on a line of data qubits with ancillas
// between them: mid-circuit measurements and resets after T-rotated data.
Workload Syndrome(size_t data, size_t rounds, size_t shots, std::optional<size_t> realizations, const std::string &name)
{
    const size_t qubits = 2 * data - 1;
    auto c = std::make_shared<Circuit>();
    std::vector<std::pair<int, int>> pairs;
    for (size_t q = 0; q + 1 < qubits; ++q)
        pairs.emplace_back(int(q), int(q + 1));
    for (size_t d = 0; d < data; ++d)
    {
        Add<Circuits::HadamardGate<double>>(c, 2 * d);
        Add<Circuits::TGate<double>>(c, 2 * d);
        Add<Circuits::HadamardGate<double>>(c, 2 * d);
    }
    size_t bit = 0;
    for (size_t round = 0; round < rounds; ++round)
    {
        for (size_t a = 0; a + 1 < data; ++a)
        {
            const size_t ancilla = 2 * a + 1;
            Add<Circuits::CXGate<double>>(c, 2 * a, ancilla);
            Add<Circuits::CXGate<double>>(c, 2 * a + 2, ancilla);
        }
        for (size_t a = 0; a + 1 < data; ++a)
        {
            Measure(c, 2 * a + 1, bit++);
            c->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{2 * a + 1}));
        }
    }
    for (size_t d = 0; d < data; ++d)
        Measure(c, 2 * d, bit++);
    return {name, c, DeviceModel(qubits, pairs), bit, shots, realizations};
}

// Many T gates on every qubit: a wide superposition of labels.
Workload HighT(size_t qubits, size_t layers, size_t shots, std::optional<size_t> realizations, const std::string &name)
{
    auto c = std::make_shared<Circuit>();
    std::vector<std::pair<int, int>> pairs;
    for (size_t q = 0; q + 1 < qubits; ++q)
        pairs.emplace_back(int(q), int(q + 1));
    for (size_t layer = 0; layer < layers; ++layer)
    {
        for (size_t q = 0; q < qubits; ++q)
        {
            Add<Circuits::HadamardGate<double>>(c, q);
            Add<Circuits::TGate<double>>(c, q);
        }
        for (size_t q = layer % 2; q + 1 < qubits; q += 2)
            Add<Circuits::CXGate<double>>(c, q, q + 1);
    }
    for (size_t q = 0; q < qubits; ++q)
        Measure(c, q, q);
    return {name, c, DeviceModel(qubits, pairs), qubits, shots, realizations};
}

MaestroExecution::SimulatorConfig Config(const std::string &policy, std::optional<size_t> workers)
{
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = Simulators::SimulatorType::kQCSim;
    config.simulation_type = Simulators::SimulationType::kExtendedStabilizer;
    config.fixed_backend = true;
    config.counts_policy = policy;
    config.max_simulators = workers;
    return config;
}

struct Run
{
    double seconds;
    MaestroExecution::ShotRecords records;
};

Run Time(const Workload &workload, const MaestroExecution::SimulatorConfig &config, size_t repeats)
{
    Run best{1e300, {}};
    for (size_t i = 0; i < repeats; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        // Rows: the legacy path then always runs its fastest executor, the
        // direct sessions that unseeded auto counts used before frames.
        MaestroExecution::NoisyExecutionOptions options;
        options.rows = true;
        auto records = MaestroExecution::ExecuteNoisy(workload.circuit, workload.model, config, workload.shots, workload.realizations, std::nullopt, options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (seconds < best.seconds)
            best = {seconds, std::move(records)};
    }
    return best;
}

std::vector<double> Marginals(const MaestroExecution::ShotRecords &records, size_t width)
{
    std::vector<double> ones(width, 0.0);
    for (size_t shot = 0; shot < records.shots; ++shot)
        for (size_t bit = 0; bit < width; ++bit)
            ones[bit] += records.rows[shot * records.width + bit];
    for (auto &value : ones)
        value /= double(std::max<size_t>(1, records.shots));
    return ones;
}

// Largest per-bit difference in standard errors; independent shots under
// per-shot noise, so a correct pair of samplers stays well below 5.
double MarginalSigmas(const std::vector<double> &a, const std::vector<double> &b, size_t shots)
{
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double p = 0.5 * (a[i] + b[i]);
        const double sigma = std::sqrt(std::max(p * (1 - p), 1e-12) * 2.0 / double(shots));
        worst = std::max(worst, std::abs(a[i] - b[i]) / sigma);
    }
    return worst;
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const size_t repeats = argc > 1 ? std::max(1, std::atoi(argv[1])) : 2;
        const std::vector<Workload> workloads = {
            CliffordT(24, 16, 8, 2000, size_t{2000}, "Clifford+T 24q, 8 T, per-shot noise"),
            CliffordT(24, 16, 8, 20000, std::nullopt, "Clifford+T 24q, 8 T, 64 realizations"),
            CliffordT(48, 24, 10, 2000, size_t{2000}, "Clifford+T 48q, 10 T, per-shot noise"),
            Syndrome(7, 5, 2000, size_t{2000}, "syndrome 13q x5 rounds, per-shot noise"),
            Syndrome(7, 5, 20000, std::nullopt, "syndrome 13q x5 rounds, 64 realizations"),
            HighT(10, 3, 500, size_t{500}, "high-T 10q, 30 T, per-shot noise"),
            HighT(10, 3, 5000, std::nullopt, "high-T 10q, 30 T, 64 realizations"),
        };
        std::cout << std::left << std::setw(44) << "workload" << std::right << std::setw(11) << "legacy s" << std::setw(11) << "frames 1w" << std::setw(11)
                  << "frames" << std::setw(10) << "x (1w)" << std::setw(10) << "x" << std::setw(9) << "marg σ" << '\n';
        bool slower = false;
        for (const auto &workload : workloads)
        {
            const auto legacy = Time(workload, Config("legacy", std::nullopt), repeats);
            const auto single = Time(workload, Config("auto", size_t{1}), repeats);
            const auto frames = Time(workload, Config("auto", std::nullopt), repeats);
            if (frames.records.sampling_strategy != "extended_frame_v1" || single.records.sampling_strategy != "extended_frame_v1")
                throw std::runtime_error(workload.name + ": frames did not run (" + frames.records.sampling_strategy + ")");
            if (legacy.records.sampling_strategy == "extended_frame_v1")
                throw std::runtime_error(workload.name + ": legacy ran on frames");
            const double sigmas = workload.realizations
                                      ? MarginalSigmas(Marginals(legacy.records, workload.width), Marginals(frames.records, workload.width), workload.shots)
                                      : -1;
            std::cout << std::left << std::setw(44) << workload.name << std::right << std::fixed << std::setprecision(3) << std::setw(11) << legacy.seconds
                      << std::setw(11) << single.seconds << std::setw(11) << frames.seconds << std::setprecision(1) << std::setw(10)
                      << legacy.seconds / single.seconds << std::setw(10) << legacy.seconds / frames.seconds << std::setw(9);
            if (sigmas >= 0)
                std::cout << std::setprecision(2) << sigmas;
            else
                std::cout << "-";
            std::cout << '\n';
            slower = slower || single.seconds >= legacy.seconds;
        }
        if (slower)
        {
            std::cout << "A frame run was not faster than the legacy path\n";
            return 2;
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
