/**
 * @file NoiseAdd.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Add noise to a circuit, using the NoiseModel defined in Noise.h.
 */

#pragma once

#ifndef __NOISE_ADD_H_
#define __NOISE_ADD_H_

#include "../Execution/Estimation.h"
#include "../Execution/NoisePolicy.h"
#include "../Execution/ReferenceFrameSampler.h"
#include "NoiseModel.h"

#include "../Network/Network.h"
#include "../Simulators/Core/RandomSeed.h"
#include "../Utils/Threads/RealizationWorkers.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <vector>

#define MAESTRO_NOISE_ADD_VERSION 5

namespace noise
{

class NoiseAdd
{
  public:
    NoiseAdd() : rng(Utils::RandomStream::FreshSeed())
    {
    }

    /**
     * Select the exact-channel (CPTP) realization of the Markovian noise
     * layers instead of sampled trajectories.
     *
     * Enable this when the network executes on a backend that carries a mixed
     * state -- QCSim density_matrix or matrix_product_operator, Aer
     * density_matrix -- so the noise is applied as deterministic Kraus channels
     * in one pass rather than averaged over realizations. It also unlocks the
     * channels that have no stochastic realization (generalized amplitude
     * damping, correlated phase flips, custom Kraus maps).
     *
     * Execute/estimate calls infer this from the backend the network will run
     * unless explicitly overridden: its current simulator, or, when the network
     * selects its backend, every optimization candidate (a network configured
     * for automatic selection holds only a placeholder simulator). An inferred
     * exact backend is pinned on the execution clone. Injection without a
     * network remains sampled until set_exact_channels(true) is called.
     */
    // Bindings may supply a language-level warning (e.g. Python RuntimeWarning).
    void set_warning_handler(std::function<void(const std::string &)> handler)
    {
        warning_handler = std::move(handler);
    }

    void set_exact_channels(bool exact)
    {
        exact_channels_override = exact;
        exact_channels = exact;
    }

    void set_automatic_channels()
    {
        exact_channels_override.reset();
        exact_channels = false;
    }

    const MaestroExecution::NoiseRealizationPolicy &last_realization_policy() const
    {
        return last_policy;
    }

    // The backend the last noisy execution or estimate ran on. Those run on
    // clones of the caller's network (execution clones, realization workers),
    // so the caller's network does not describe them: read this instead.
    struct ExecutionRecord
    {
        bool valid = false; /**< False until a noisy run executed on a network. */
        Simulators::SimulatorType simulator = Simulators::SimulatorType::kQCSim;
        Simulators::SimulationType method = Simulators::SimulationType::kStatevector;
        bool gate_fusion = false;
        size_t gate_fusion_max_qubits = 0;
        int gpu_device = -1;
    };

    const ExecutionRecord &last_execution() const
    {
        return last_record;
    }

    bool uses_exact_channels() const
    {
        return exact_channels;
    }

    std::shared_ptr<Circuits::Circuit<double>> inject(const std::shared_ptr<Circuits::Circuit<double>> &circ, const NoiseModel &nm)
    {
        WarnThermalApproximation(nm);
        return Inject(circ, nm, Layer::Pauli);
    }

    std::shared_ptr<Circuits::Circuit<double>> inject_coherent(const std::shared_ptr<Circuits::Circuit<double>> &circ, const NoiseModel &nm)
    {
        // Coherent noise is a sampled unitary trajectory on every backend: there
        // is no single channel that reproduces a systematic miscalibration.
        // Circuit-level channels are still applied exactly on DM/MPO.
        return Inject(circ, nm, Layer::Coherent);
    }

    std::shared_ptr<Circuits::Circuit<double>> inject_combined(const std::shared_ptr<Circuits::Circuit<double>> &circ, const NoiseModel &nm)
    {
        WarnThermalApproximation(nm);
        return Inject(circ, nm, Layer::Combined);
    }

    // Noise draws and simulator draws have independent streams. Re-seeding
    // restarts both; successive calls on this object advance the streams.
    void seed(unsigned int s)
    {
        rng.seed(s);
        public_seed = s;
        next_simulator_stream = 0;
    }

    Circuits::Circuit<>::ExecuteResults noisy_execute(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                      const std::shared_ptr<Network::INetwork<>> &network, size_t hostId, const NoiseModel &nm, int shots,
                                                      std::optional<int> noise_realizations = std::nullopt)
    {
        return Execute(circuit, network, hostId, nm, shots, noise_realizations, Layer::Pauli);
    }

    Circuits::Circuit<>::ExecuteResults coherent_execute(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                         const std::shared_ptr<Network::INetwork<>> &network, size_t hostId, const NoiseModel &nm, int shots,
                                                         std::optional<int> noise_realizations = std::nullopt)
    {
        return Execute(circuit, network, hostId, nm, shots, noise_realizations, Layer::Coherent);
    }

    Circuits::Circuit<>::ExecuteResults full_noise_execute(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                           const std::shared_ptr<Network::INetwork<>> &network, size_t hostId, const NoiseModel &nm, int shots,
                                                           std::optional<int> noise_realizations = std::nullopt)
    {
        return Execute(circuit, network, hostId, nm, shots, noise_realizations, Layer::Combined);
    }

    Circuits::Circuit<>::ExecuteResults noisy_execute_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                  const std::shared_ptr<Network::INetwork<>> &network, const NoiseModel &nm, int shots,
                                                                  std::optional<int> noise_realizations = std::nullopt)
    {
        return Execute(circuit, network, std::nullopt, nm, shots, noise_realizations, Layer::Pauli);
    }

    Circuits::Circuit<>::ExecuteResults coherent_execute_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                     const std::shared_ptr<Network::INetwork<>> &network, const NoiseModel &nm, int shots,
                                                                     std::optional<int> noise_realizations = std::nullopt)
    {
        return Execute(circuit, network, std::nullopt, nm, shots, noise_realizations, Layer::Coherent);
    }

    Circuits::Circuit<>::ExecuteResults full_noise_execute_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                       const std::shared_ptr<Network::INetwork<>> &network, const NoiseModel &nm, int shots,
                                                                       std::optional<int> noise_realizations = std::nullopt)
    {
        return Execute(circuit, network, std::nullopt, nm, shots, noise_realizations, Layer::Combined);
    }

    std::vector<double> noisy_estimate(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::shared_ptr<Network::INetwork<>> &network,
                                       size_t hostId, const std::vector<std::string> &paulis, const NoiseModel &nm)
    {
        if (!network || !circuit || paulis.empty())
            return {};

        auto ideal = MaestroExecution::EstimateOnNetwork(circuit, paulis, network, hostId, true);
        std::vector<double> noisy;
        noisy.reserve(paulis.size());
        for (size_t i = 0; i < paulis.size(); ++i)
        {
            const double damping = nm.compute_damping(paulis[i]);
            noisy.push_back(damping * ideal[i]);
        }

        return noisy;
    }

    std::vector<double> noisy_estimate_montecarlo(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                  const std::shared_ptr<Network::INetwork<>> &network, size_t hostId, const std::vector<std::string> &paulis,
                                                  const NoiseModel &nm, std::optional<int> noise_realizations = std::nullopt)
    {
        return Estimate(circuit, network, hostId, paulis, nm, noise_realizations, Layer::Pauli);
    }

    std::vector<double> coherent_estimate(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::shared_ptr<Network::INetwork<>> &network,
                                          size_t hostId, const std::vector<std::string> &paulis, const NoiseModel &nm,
                                          std::optional<int> noise_realizations = std::nullopt)
    {
        return Estimate(circuit, network, hostId, paulis, nm, noise_realizations, Layer::Coherent);
    }

    std::vector<double> full_noise_estimate(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::shared_ptr<Network::INetwork<>> &network,
                                            size_t hostId, const std::vector<std::string> &paulis, const NoiseModel &nm,
                                            std::optional<int> noise_realizations = std::nullopt)
    {
        return Estimate(circuit, network, hostId, paulis, nm, noise_realizations, Layer::Combined);
    }

    std::vector<double> noisy_estimate_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                   const std::shared_ptr<Network::INetwork<>> &network, const std::vector<std::string> &paulis,
                                                   const NoiseModel &nm)
    {
        if (!network || !circuit || paulis.empty())
            return {};

        auto ideal = MaestroExecution::EstimateOnNetwork(circuit, paulis, network, std::nullopt, true);
        std::vector<double> noisy;
        noisy.reserve(paulis.size());
        for (size_t i = 0; i < paulis.size(); ++i)
        {
            const double damping = nm.compute_damping(paulis[i]);
            noisy.push_back(damping * ideal[i]);
        }

        return noisy;
    }

    std::vector<double> noisy_estimate_montecarlo_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                              const std::shared_ptr<Network::INetwork<>> &network, const std::vector<std::string> &paulis,
                                                              const NoiseModel &nm, std::optional<int> noise_realizations = std::nullopt)
    {
        return Estimate(circuit, network, std::nullopt, paulis, nm, noise_realizations, Layer::Pauli);
    }

    std::vector<double> coherent_estimate_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                      const std::shared_ptr<Network::INetwork<>> &network, const std::vector<std::string> &paulis,
                                                      const NoiseModel &nm, std::optional<int> noise_realizations = std::nullopt)
    {
        return Estimate(circuit, network, std::nullopt, paulis, nm, noise_realizations, Layer::Coherent);
    }

    std::vector<double> full_noise_estimate_distributed(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                        const std::shared_ptr<Network::INetwork<>> &network, const std::vector<std::string> &paulis,
                                                        const NoiseModel &nm, std::optional<int> noise_realizations = std::nullopt)
    {
        return Estimate(circuit, network, std::nullopt, paulis, nm, noise_realizations, Layer::Combined);
    }

  private:
    enum class Layer
    {
        Pauli,
        Coherent,
        Combined
    };
    using CircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;
    using NetworkPtr = std::shared_ptr<Network::INetwork<>>;

    struct ScopedChannels
    {
        bool &value;
        bool previous;

        ScopedChannels(bool &value, bool exact) : value(value), previous(value)
        {
            value = exact;
        }

        ~ScopedChannels()
        {
            value = previous;
        }
    };

    using Backend = std::pair<Simulators::SimulatorType, Simulators::SimulationType>;

    // The backends automatic selection may run instead of the network's
    // simulator, in the order it tries them: the configured candidates
    // (INetwork::GetOptimizationCandidates). Empty when execution keeps that
    // simulator: selection is off, the network distributes (never replaced),
    // or no list was configured. ConfigureNetwork always configures one; its
    // simulator is then a placeholder of the first listed candidate.
    static std::vector<Backend> SelectableCandidates(const NetworkPtr &network)
    {
        const auto simulator = network->GetSimulator();
        if (!network->GetOptimizeSimulator() || (simulator && Simulators::IsDistributedGpuSimulator(simulator->GetType())))
            return {};
        return network->GetOptimizationCandidates();
    }

    // The backends execution can use, described as a configuration. Exact
    // channels then apply only when every candidate that may run supports
    // them (MaestroExecution::UsesExactQuantumChannels).
    static MaestroExecution::SimulatorConfig ExecutionBackends(const NetworkPtr &network)
    {
        MaestroExecution::SimulatorConfig config;
        config.fixed_backend = true;
        if (const auto simulator = network->GetSimulator())
        {
            config.simulator_type = simulator->GetType();
            config.simulation_type = simulator->GetSimulationType();
        }
        const auto candidates = SelectableCandidates(network);
        if (!candidates.empty())
        {
            config.fixed_backend = false;
            config.optimization_candidates = candidates;
        }
        return config;
    }

    // The backend selection starts from: the network's simulator when it is
    // listed (or no list was configured), else the first listed candidate.
    // Never a placeholder outside the list.
    static std::optional<Backend> PinnedBackend(const NetworkPtr &network)
    {
        std::optional<Backend> current;
        if (const auto simulator = network->GetSimulator())
            current = Backend{simulator->GetType(), simulator->GetSimulationType()};
        const auto candidates = SelectableCandidates(network);
        if (candidates.empty() || (current && std::find(candidates.begin(), candidates.end(), *current) != candidates.end()))
            return current;
        return candidates.front();
    }

    MaestroExecution::NoiseRealizationPolicy ResolvePolicy(const CircuitPtr &circuit, const NetworkPtr &network, const NoiseModel &model,
                                                           std::optional<int> requested, Layer layer, bool estimation)
    {
        if (requested && *requested <= 0)
            throw std::invalid_argument("noise_realizations must be positive");
        const auto config = ExecutionBackends(network);
        const auto selected = layer == Layer::Pauli      ? MaestroExecution::NoiseLayer::kPauli
                              : layer == Layer::Coherent ? MaestroExecution::NoiseLayer::kCoherent
                                                         : MaestroExecution::NoiseLayer::kCombined;
        return MaestroExecution::ResolveNoiseRealizations(requested ? std::optional<size_t>(static_cast<size_t>(*requested)) : std::nullopt, model, config,
                                                          selected, estimation ? 100 : 64, circuit, estimation, exact_channels_override);
    }

    struct ExecutionContext
    {
        NetworkPtr network;
        std::optional<uint64_t> seed;
    };

    ExecutionContext PrepareExecution(const NetworkPtr &network)
    {
        auto base_seed = public_seed;
        const auto simulator = network->GetSimulator();
        if (simulator)
        {
            const auto configured = simulator->GetConfiguration("seed");
            if (!configured.empty())
                base_seed = std::stoull(configured);
            if (simulator->GetType() == Simulators::SimulatorType::kDistMpiGpuSim)
            {
                if (!base_seed)
                    base_seed = Simulators::GenerateRandomSeed(simulator->GetType(), simulator->GetConfigMap());
                if (!public_seed && !mpi_noise_seeded)
                {
                    rng.seed(static_cast<uint32_t>(*base_seed));
                    mpi_noise_seeded = true;
                }
            }
        }
        // Work on a pristine network so derived batch seeds never replace the
        // caller's configuration, including on exception paths.
        auto execution_network = ExecutionClone(*network);
        if (!execution_network)
            throw std::runtime_error("Cannot clone network for noisy execution");
        execution_network->SetMaxSimulators(network->GetMaxSimulators());
        // Timing-based backend selection (a cost model) changes the random
        // sequence between otherwise identical calls, and may rank away the
        // mixed-state simulator exact channels need when no list says
        // otherwise. Without it, selection is deterministic: a configured list
        // is tried in order, falling back to the next candidate that fits, and
        // otherwise the network simulator's backend runs. A Clifford circuit
        // may still move to a stabilizer the list admits, seeded or not (see
        // INetwork::GetOptimizationCandidates). Worker clones of the execution
        // network keep the setting.
        if (base_seed || exact_channels)
            execution_network->SetCostModelEnabled(false);
        return {std::move(execution_network), base_seed};
    }

    struct Realization
    {
        CircuitPtr circuit;
        std::optional<uint64_t> seed;
    };

    // The next injected realization and its simulator stream, drawn in order.
    Realization NextRealization(const NoiseInjectionPlan &plan, Layer layer, const ExecutionContext &execution)
    {
        auto noisy = Inject(plan, layer);
        std::optional<uint64_t> seed;
        if (execution.seed)
            seed = Simulators::IState::DeriveSeed(*execution.seed, next_simulator_stream++);
        return Realization{std::move(noisy), seed};
    }

    static void SeedNetwork(Network::INetwork<> &network, std::optional<uint64_t> seed)
    {
        if (!seed)
            return;
        network.Configure("seed", std::to_string(*seed).c_str());
        if (const auto simulator = network.GetSimulator())
            simulator->SetSeed(*seed);
    }

    // A clone that only executes on the network. Simple networks size their own
    // register for each execution, so the clone's simulator just selects the
    // backend (INetwork::CreateSimulator) and never copies a register the
    // network would discard, whatever width the caller's simulator has
    // (INetwork::CloneForExecution). Distribution keeps its configured
    // register, and other network types (netqasm, discrete events) may run
    // their shots on the network's simulator itself.
    static NetworkPtr ExecutionClone(const Network::INetwork<> &network)
    {
        auto cloned = network.CloneForExecution(Network::INetwork<>::kBackendPlaceholderQubits);
        if (!cloned)
            return cloned;
        // A simple network whose clone still copied a larger register drops it.
        const auto type = cloned->GetType();
        if (type != Network::NetworkType::kSimpleDisconnectedNetwork && type != Network::NetworkType::kSimpleNetwork)
            return cloned;
        auto simulator = cloned->GetSimulator();
        if (!simulator || Simulators::IsDistributedGpuSimulator(simulator->GetType()) ||
            simulator->GetNumberOfQubits() <= Network::INetwork<>::kBackendPlaceholderQubits)
            return cloned;
        const auto backend = simulator->GetType();
        const auto method = simulator->GetSimulationType();
        simulator.reset(); // or the copied register stays allocated
        cloned->CreateSimulator(backend, method, Network::INetwork<>::kBackendPlaceholderQubits);
        return cloned;
    }

    // Single-threaded clones of the execution network, one per worker.
    static std::vector<NetworkPtr> WorkerNetworks(const NetworkPtr &network, size_t workers)
    {
        std::vector<NetworkPtr> networks;
        networks.reserve(workers);
        for (size_t worker = 0; worker < workers; ++worker)
        {
            auto cloned = ExecutionClone(*network);
            if (!cloned)
                throw std::runtime_error("Cannot clone network for noisy realization worker");
            cloned->SetMaxSimulators(1);
            networks.push_back(std::move(cloned));
        }
        return networks;
    }

    // Whether the network samples every shot of a batch from one prepared
    // state: terminal measurements on a dense QCSim method (run with one
    // thread) or a QCSim MPO (whose workers only clone the prepared state, and
    // realization worker networks have one simulator each).
    static bool SamplesFromOnePreparedState(const CircuitPtr &circuit, const NetworkPtr &network)
    {
        const auto backend = PinnedBackend(network);
        if (!backend || backend->first != Simulators::SimulatorType::kQCSim ||
            circuit->HasOpsAfterMeasurements(MaestroExecution::SupportsExactQuantumChannels(backend->first, backend->second)))
            return false;
        const auto method = backend->second;
        return method == Simulators::SimulationType::kStatevector || method == Simulators::SimulationType::kDensityMatrix ||
               method == Simulators::SimulationType::kPathIntegral || method == Simulators::SimulationType::kMatrixProductOperator;
    }

    // The width of the register the caller's network keeps allocated.
    static size_t RetainedQubits(const Network::INetwork<> &network)
    {
        const auto simulator = network.GetSimulator();
        return simulator ? simulator->GetNumberOfQubits() : 0;
    }

    static ExecutionRecord Describe(const Network::INetwork<> &network)
    {
        ExecutionRecord record;
        record.valid = true;
        record.simulator = network.GetLastSimulatorType();
        record.method = network.GetLastSimulationType();
        record.gate_fusion = network.WasGateFusionEnabled();
        record.gate_fusion_max_qubits = network.GetLastGateFusionMaxQubits();
        record.gpu_device = network.GetLastGpuDevice();
        return record;
    }

    // The worker network that ran the highest realization index, the one a
    // serial loop would have run last.
    static const NetworkPtr &LastWorker(const std::vector<NetworkPtr> &networks, const std::vector<size_t> &latest)
    {
        size_t last = 0;
        for (size_t worker = 1; worker < networks.size(); ++worker)
            if (latest[worker] > latest[last])
                last = worker;
        return networks[last];
    }

    void SeedExecution(const ExecutionContext &execution)
    {
        if (!execution.seed)
            return;
        const auto seed = Simulators::IState::DeriveSeed(*execution.seed, next_simulator_stream++);
        execution.network->Configure("seed", std::to_string(seed).c_str());
        // Also cover networks that execute their existing simulator directly.
        if (const auto simulator = execution.network->GetSimulator())
            simulator->SetSeed(seed);
    }

    CircuitPtr Inject(const CircuitPtr &circuit, const NoiseModel &nm, Layer layer)
    {
        if (!circuit)
            return nullptr;
        CircuitPtr noisy;
        if (layer == Layer::Coherent)
            noisy = exact_channels ? inject_coherent_noise_exact(circuit, nm, rng) : inject_coherent_noise(circuit, nm, rng);
        else if (layer == Layer::Pauli)
            noisy = exact_channels ? inject_exact_noise(circuit, nm) : inject_noise(circuit, nm, rng);
        else
            noisy = exact_channels ? inject_combined_noise_exact(circuit, nm, rng) : inject_combined_noise(circuit, nm, rng);
        attach_readout_error(noisy, nm);
        return noisy;
    }

    // Inject(plan.circuit(), plan.model(), layer), without per-realization lookups.
    CircuitPtr Inject(const NoiseInjectionPlan &plan, Layer layer)
    {
        CircuitPtr noisy;
        if (layer == Layer::Coherent)
            noisy = plan.Coherent(rng, exact_channels);
        else if (layer == Layer::Pauli)
            noisy = exact_channels ? plan.ExactPauli() : plan.Pauli(rng);
        else
            noisy = plan.Combined(rng, exact_channels);
        plan.AttachReadout(noisy);
        return noisy;
    }

    void WarnThermalApproximation(const NoiseModel &nm) const
    {
        if (exact_channels || nm.has_additional_quantum_channels())
            return;
        const auto qubits = nm.thermal_approximation_qubits();
        if (qubits.empty())
            return;
        std::ostringstream message;
        message << "Sampled thermal approximation for circuit qubits [";
        for (size_t i = 0; i < qubits.size(); ++i)
        {
            if (i)
                message << ", ";
            message << qubits[i];
        }
        message << "]: effective T2 clamped to T1. Use density matrix or a "
                   "supported MPO backend to preserve calibrated T2.";
        if (warning_handler)
            warning_handler(message.str());
        else
            std::cerr << "Warning: " << message.str() << '\n';
    }

    // network executes (an execution clone); retainedQubits is the width of the
    // caller's register, which stays allocated while workers run.
    // The width of a realization's circuit: injection adds crosstalk rotations
    // on spectator qubits the circuit itself may not use.
    static size_t RealizationWidth(const CircuitPtr &circuit, const NoiseInjectionPlan &plan)
    {
        size_t qubits = std::max<size_t>(1, circuit->GetMaxQubitIndex() + 1);
        for (const auto spectator : plan.Spectators())
            qubits = std::max(qubits, spectator + 1);
        return qubits;
    }

    // The width an expectation run allocates for its observables: the host
    // executor keeps every position of the longest Pauli string, idle ones
    // included.
    static size_t ObservableWidth(const std::vector<std::string> &paulis)
    {
        size_t width = 0;
        for (const auto &pauli : paulis)
            width = std::max(width, pauli.size());
        return width;
    }

    static size_t RealizationWorkers(size_t realizationQubits, const NetworkPtr &network, std::optional<size_t> host, size_t batches, size_t retainedQubits)
    {
        const auto simulator = network->GetSimulator();
        // Accelerator and distributed runtimes retain their calling-thread and
        // collective execution rules. Unknown state sizes also stay serial.
        if (!host || !simulator || simulator->GetType() != Simulators::SimulatorType::kQCSim)
            return 1;
        if (network->GetOptimizeSimulator())
        {
            // This estimate describes the source method. An unseeded automatic
            // choice may select a denser or unbounded candidate instead; do not
            // multiply that unknown allocation across realization workers.
            const auto source = std::make_pair(simulator->GetType(), simulator->GetSimulationType());
            for (const auto &candidate : network->GetSimulatorsSet())
                if (candidate != source)
                    return 1;
        }
        size_t workers = Utils::RealizationWorkerCount(batches, network->GetMaxSimulators());
        if (workers <= 1)
            return workers;
        // A simple network's realization register is the circuit's: each
        // worker's network maps it to the qubits it uses, and its clones hold
        // only placeholders (ExecutionClone). Netqasm and discrete-event
        // networks run on their simulator itself, which their clones copy:
        // each worker's register is the whole layout, entanglement qubits
        // included, and the execution clone keeps one more. The caller's
        // simulator stays allocated while workers run, at whatever width it has.
        const auto type = network->GetType();
        const bool simple = type == Network::NetworkType::kSimpleDisconnectedNetwork || type == Network::NetworkType::kSimpleNetwork;
        const size_t layoutQubits = network->GetNumQubits() + network->GetNumNetworkEntangledQubits();
        const size_t qubits = std::max<size_t>(std::max<size_t>(1, realizationQubits), simple ? 0 : layoutQubits);
        const size_t callerQubits = retainedQubits;
        const size_t cloneQubits = simulator->GetNumberOfQubits();
        Utils::RealizationState state;
        size_t bond = 0, copiesPerWorker = 1;
        switch (simulator->GetSimulationType())
        {
        case Simulators::SimulationType::kStabilizer:
            state = Utils::RealizationState::kStabilizer;
            break;
        case Simulators::SimulationType::kStatevector:
        case Simulators::SimulationType::kDensityMatrix:
            state = simulator->GetSimulationType() == Simulators::SimulationType::kDensityMatrix ? Utils::RealizationState::kDensityMatrix
                                                                                                 : Utils::RealizationState::kStatevector;
            copiesPerWorker = 4; // Live register, checkpoint and gate workspaces.
            break;
        case Simulators::SimulationType::kMatrixProductState:
        case Simulators::SimulationType::kMatrixProductOperator: {
            auto configured = simulator->GetConfiguration("matrix_product_state_max_bond_dimension");
            if (simulator->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator)
            {
                const auto mpoBond = simulator->GetConfiguration("matrix_product_operator_max_bond_dimension");
                if (!mpoBond.empty())
                    configured = mpoBond;
            }
            if (configured.empty() || std::stoll(configured) <= 0)
                return 1;
            bond = static_cast<size_t>(std::stoull(configured));
            state = simulator->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator ? Utils::RealizationState::kMatrixProductOperator
                                                                                                         : Utils::RealizationState::kMatrixProductState;
            break;
        }
        default:
            return 1;
        }
        // Shared with the Python realization executors (RealizationStateBytes).
        const long double bytes = Utils::RealizationStateBytes(state, qubits, bond);
        const long double limit = static_cast<long double>(std::numeric_limits<size_t>::max());
        if (!(bytes > 0) || !std::isfinite(bytes) || bytes >= limit)
            return 1;
        const long double callerBytes = callerQubits ? Utils::RealizationStateBytes(state, callerQubits, bond) : 0.L;
        const long double cloneBytes = cloneQubits ? Utils::RealizationStateBytes(state, cloneQubits, bond) : 0.L;
        if (!std::isfinite(callerBytes) || callerBytes >= limit || !std::isfinite(cloneBytes) || cloneBytes >= limit)
            return 1;
        const auto configuredBudget = simulator->GetConfiguration("trajectory_max_memory_mb");
        const size_t budgetMB = configuredBudget.empty() ? 1024 : std::stoull(configuredBudget);
        // Reserved, in copies of a realization's register: the caller's
        // register and the execution clone's, alive while all worker clones
        // execute, and a spare. Other methods use a conservative workspace
        // estimate for a copy too, rather than treating retained states as free.
        const long double retainedCopies = std::ceil(callerBytes / std::max(1.L, bytes)) + std::ceil(cloneBytes / std::max(1.L, bytes));
        if (retainedCopies >= limit - 1)
            return 1;
        const size_t reserved = 1 + static_cast<size_t>(retainedCopies);
        return Utils::LimitRealizationWorkersByMemory(workers, static_cast<size_t>(std::max(1.L, bytes)), budgetMB, reserved, copiesPerWorker);
    }

    std::optional<Circuits::Circuit<>::ExecuteResults> FrameCounts(const CircuitPtr &circuit, const NetworkPtr &network, std::optional<size_t> host,
                                                                   const NoiseModel &nm, int shots, int batches, Layer layer)
    {
        const auto simulator = network->GetSimulator();
        if (!simulator)
            return std::nullopt;
        auto policy = simulator->GetConfiguration("counts_policy");
        if (policy.empty())
            policy = "auto";
        if (policy != "auto" && policy != "legacy" && policy != "frame_v1")
            throw std::invalid_argument("counts_policy must be auto, legacy or frame_v1");
        auto baseSeed = public_seed;
        const auto configuredSeed = simulator->GetConfiguration("seed");
        if (!configuredSeed.empty())
            baseSeed = std::stoull(configuredSeed);
        // Frames need the stabilizer to be an allowed backend: a fixed
        // Stabilizer simulator, or an optimizing network whose candidate set
        // admits it. A restricted set binds frame_v1 and a Stabilizer request
        // too. Under auto, the ordinary executor must also be able to choose
        // the stabilizer itself: exact channels pin the requested mixed-state
        // backend of another method (PrepareExecution).
        const bool stabilizerMethod = simulator->GetSimulationType() == Simulators::SimulationType::kStabilizer;
        const bool allowsStabilizer = network->GetOptimizeSimulator()
                                          ? network->OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer)
                                          : stabilizerMethod;
        const bool admitsStabilizer = allowsStabilizer && (stabilizerMethod || !exact_channels);
        if (policy == "legacy" || !allowsStabilizer || (policy == "auto" && (baseSeed || !admitsStabilizer)) || !host || *host != 0 ||
            network->GetNumHosts() != 1 || batches != shots || simulator->GetType() != Simulators::SimulatorType::kQCSim || layer == Layer::Coherent ||
            !nm.inject_noise_covers_model())
            return std::nullopt;
        const size_t width = std::max<size_t>(1, circuit->GetMaxQubitIndex() + 1);
        if (width > network->GetNumQubitsForHost(*host))
            return std::nullopt; // Sparse/global IDs still use the host mapper.
        const auto bits = circuit->GetBits();
        if (bits.empty())
            return std::nullopt;
        auto channels = inject_exact_noise(circuit, nm);
        attach_readout_error(channels, nm);
        const size_t channelWidth = std::max<size_t>(1, channels->GetMaxQubitIndex() + 1);
        if (channelWidth > network->GetNumQubitsForHost(*host))
            return std::nullopt;
        const size_t countWidth = std::max(network->GetNumClassicalBits(), *bits.rbegin() + 1);
        const auto plan = MaestroExecution::CompileReferenceSampler(channels, channelWidth, countWidth);
        if (!plan)
            return std::nullopt;
        // Frame sampling is an explicit new seeded policy. Its per-call noise
        // seed advances this object's stream; reserve the same simulator-stream
        // indices as the realization executor before a later policy switch.
        const uint64_t high = rng();
        const uint64_t noiseSeed = (high << 32) | static_cast<uint64_t>(rng());
        const uint64_t seed =
            baseSeed ? Simulators::IState::DeriveSeed(Simulators::IState::DeriveSeed(*baseSeed, next_simulator_stream), noiseSeed) : noiseSeed;
        if (baseSeed)
            next_simulator_stream += static_cast<uint64_t>(batches);
        auto outcomes = plan->SampleCounts(static_cast<size_t>(shots), seed);
        Circuits::Circuit<>::ExecuteResults counts;
        counts.reserve(outcomes.size());
        for (auto &[bits, count] : outcomes)
            counts.emplace(std::move(bits), count);
        return counts;
    }

    Circuits::Circuit<>::ExecuteResults Execute(const CircuitPtr &circuit, const NetworkPtr &network, std::optional<size_t> host, const NoiseModel &nm,
                                                int shots, std::optional<int> requested_realizations, Layer layer)
    {
        last_record = {};
        if (shots < 0)
            throw std::invalid_argument("shots must be nonnegative");
        if (requested_realizations && *requested_realizations <= 0)
            throw std::invalid_argument("noise_realizations must be positive");
        if (!network || !circuit || shots == 0)
            return {};
        last_policy = ResolvePolicy(circuit, network, nm, requested_realizations, layer, false);
        const ScopedChannels channels(exact_channels, last_policy.exact_channels);
        if (layer != Layer::Coherent)
            WarnThermalApproximation(nm);
        const int batches = static_cast<int>(std::min(static_cast<size_t>(shots), last_policy.realizations));
        last_policy.realizations = static_cast<size_t>(batches);
        if (auto counts = FrameCounts(circuit, network, host, nm, shots, batches, layer))
            return std::move(*counts);
        auto execution = PrepareExecution(network);
        Circuits::Circuit<>::ExecuteResults combined;
        const NoiseInjectionPlan plan(circuit, nm);
        // A one-shot realization depends only on its own seed, so it may run on
        // any worker. A seeded multi-shot batch keeps the network's own shot
        // partition, which sets its sequence. Unseeded batches that the network
        // samples from one prepared state run single-threaded there anyway, so
        // realization workers only add parallelism.
        const bool independent = batches == shots || (!execution.seed && SamplesFromOnePreparedState(circuit, execution.network));
        const size_t workers =
            independent ? RealizationWorkers(RealizationWidth(circuit, plan), execution.network, host, batches, RetainedQubits(*network)) : 1;
        if (workers > 1)
        {
            const auto workerNetworks = WorkerNetworks(execution.network, workers);
            std::vector<Circuits::Circuit<>::ExecuteResults> localCounts(workers);
            std::vector<size_t> latest(workers, 0);

            Utils::RunRealizations(
                static_cast<size_t>(batches), workers, [&](size_t) { return NextRealization(plan, layer, execution); },
                [&](size_t worker, size_t batch, Realization realization) {
                    const auto &workerNetwork = workerNetworks[worker];
                    SeedNetwork(*workerNetwork, realization.seed);
                    const size_t batchShots = static_cast<size_t>(shots / batches) + (batch < static_cast<size_t>(shots % batches) ? 1 : 0);
                    const auto counts = workerNetwork->RepeatedExecuteOnHost(realization.circuit, *host, batchShots);
                    for (const auto &entry : counts)
                        localCounts[worker][entry.first] += entry.second;
                    if (!counts.empty())
                        latest[worker] = std::max(latest[worker], batch + 1);
                });
            for (const auto &counts : localCounts)
                for (const auto &entry : counts)
                    combined[entry.first] += entry.second;
            // A network that did not simulate returned no counts.
            if (std::any_of(latest.begin(), latest.end(), [](size_t ran) { return ran > 0; }))
                last_record = Describe(*LastWorker(workerNetworks, latest));
            return combined;
        }
        for (int b = 0; b < batches; ++b)
        {
            const int batch_shots = shots / batches + (b < shots % batches ? 1 : 0);
            auto noisy = Inject(plan, layer);
            SeedExecution(execution);
            const auto counts =
                host ? execution.network->RepeatedExecuteOnHost(noisy, *host, batch_shots) : execution.network->RepeatedExecute(noisy, batch_shots);
            for (const auto &entry : counts)
                combined[entry.first] += entry.second;
        }
        // A network that did not simulate (no simulator) returned no counts.
        if (!combined.empty())
            last_record = Describe(*execution.network);
        return combined;
    }

    std::vector<double> Estimate(const CircuitPtr &circuit, const NetworkPtr &network, std::optional<size_t> host, const std::vector<std::string> &paulis,
                                 const NoiseModel &nm, std::optional<int> requested_realizations, Layer layer)
    {
        last_record = {};
        if (requested_realizations && *requested_realizations <= 0)
            throw std::invalid_argument("noise_realizations must be positive");
        if (!network || !circuit || paulis.empty())
            return {};
        last_policy = ResolvePolicy(circuit, network, nm, requested_realizations, layer, true);
        const int realizations = static_cast<int>(last_policy.realizations);
        const ScopedChannels channels(exact_channels, last_policy.exact_channels);
        if (layer != Layer::Coherent)
            WarnThermalApproximation(nm);
        auto cleanCircuit = circuit->RemoveEndMeasurements();
        auto execution = PrepareExecution(network);
        std::vector<double> values(paulis.size(), 0.0);
        const NoiseInjectionPlan plan(cleanCircuit, nm);
        // Each realization's values depend only on its circuit and seed, so
        // workers may evaluate them in any order. They are summed in realization
        // order, as the serial loop does.
        const size_t workers = RealizationWorkers(std::max(RealizationWidth(cleanCircuit, plan), ObservableWidth(paulis)), execution.network, host,
                                                  static_cast<size_t>(realizations), RetainedQubits(*network));
        if (workers > 1)
        {
            const auto workerNetworks = WorkerNetworks(execution.network, workers);
            std::vector<MaestroExecution::EstimationCache> caches(workers);
            std::vector<std::vector<double>> samples(static_cast<size_t>(realizations));
            std::vector<size_t> latest(workers, 0);
            Utils::RunRealizations(
                static_cast<size_t>(realizations), workers, [&](size_t) { return NextRealization(plan, layer, execution); },
                [&](size_t worker, size_t r, Realization realization) {
                    SeedNetwork(*workerNetworks[worker], realization.seed);
                    bool executed = false;
                    samples[r] = MaestroExecution::EstimateOnNetwork(realization.circuit, paulis, workerNetworks[worker], host, false, &caches[worker],
                                                                     &executed);
                    if (executed)
                        latest[worker] = std::max(latest[worker], r + 1);
                });
            for (const auto &sample : samples)
                for (size_t i = 0; i < values.size() && i < sample.size(); ++i)
                    values[i] += sample[i];
            for (auto &value : values)
                value /= realizations;
            // Planning may resolve every observable without running a network.
            if (std::any_of(latest.begin(), latest.end(), [](size_t ran) { return ran > 0; }))
                last_record = Describe(*LastWorker(workerNetworks, latest));
            return values;
        }
        MaestroExecution::EstimationCache cache;
        bool ran = false;
        for (int r = 0; r < realizations; ++r)
        {
            auto noisy = Inject(plan, layer);
            SeedExecution(execution);
            bool executed = false;
            const auto sample = MaestroExecution::EstimateOnNetwork(noisy, paulis, execution.network, host, false, &cache, &executed);
            ran = ran || executed;
            for (size_t i = 0; i < values.size(); ++i)
                values[i] += sample[i];
        }
        for (auto &value : values)
            value /= realizations;
        // Planning may resolve every observable without running the network.
        if (ran)
            last_record = Describe(*execution.network);
        return values;
    }

    bool exact_channels = false;
    std::optional<bool> exact_channels_override;
    MaestroExecution::NoiseRealizationPolicy last_policy;
    ExecutionRecord last_record;
    std::mt19937 rng;
    std::optional<uint64_t> public_seed;
    uint64_t next_simulator_stream = 0;
    bool mpi_noise_seeded = false;
    std::function<void(const std::string &)> warning_handler;
};

} // namespace noise

#endif
