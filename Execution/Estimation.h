#pragma once

#include "../Circuit/Circuit.h"
#include "Transforms/Export.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Network
{
template <class Time> class INetwork;
}

namespace MaestroExecution
{
struct SimulatorConfig;

// A per-request cache, never shared across threads. The ordered multi-qubit
// topology and observables are checked before reusing groups and qubit maps.
struct EstimationCache
{
    std::vector<size_t> signature;
    std::vector<std::string> observables;
    std::vector<std::vector<size_t>> groups;
    std::vector<Circuits::Circuit<double>::BitMapping> mappings;
    size_t hits = 0;
};

struct EstimationGroup
{
    std::shared_ptr<Circuits::Circuit<double>> circuit;
    std::vector<std::string> observables;
    std::vector<size_t> indices;
    std::vector<double> factors;
    size_t qubits = 0;
};

struct EstimationPlan
{
    std::vector<EstimationGroup> groups;
    std::vector<double> resolved_values;
    std::vector<double> error_bounds;
    size_t bounded_observables = 0;
    bool auto_reduced = false;
    size_t qubits_before = 0;
    size_t qubits_after = 0;
    size_t total_cluster_qubits = 0;
    double reduction_time = 0;
};

struct EstimationResult
{
    std::vector<double> values;
    EstimationPlan plan;
    std::vector<Simulators::SimulationType> cluster_methods;
    Simulators::SimulatorType simulator;
    Simulators::SimulationType method;
    int gpu_device = -1;
    size_t max_bond_dimension = 0;
    bool gate_fusion = false;
    size_t gate_fusion_max_qubits = 0;
    double time_taken = 0;
    std::vector<double> standard_errors;
    size_t measurement_trajectories = 0;
    bool terminal_channels_averaged = false;
};

// No simulator is allocated during preparation. Empty requests have no groups.
MAESTRO_TRANSFORM_API EstimationPlan PrepareEstimation(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                                       const SimulatorConfig &config, EstimationCache *cache = nullptr);
MAESTRO_TRANSFORM_API EstimationResult EstimateCircuit(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                                       const SimulatorConfig &config, EstimationCache *cache = nullptr);

// Execute one realization. Ensemble estimators call this after selecting the circuit and its trajectory seed.
MAESTRO_TRANSFORM_API EstimationResult EstimateCircuitTrajectory(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                 const std::vector<std::string> &observables, const SimulatorConfig &config,
                                                                 EstimationCache *cache = nullptr);

using NetworkExpectationEvaluator = std::function<std::vector<double>(const std::shared_ptr<Circuits::Circuit<double>> &, const std::vector<std::string> &)>;
MAESTRO_TRANSFORM_API SimulatorConfig NetworkEstimateConfig(const Network::INetwork<double> &network);
MAESTRO_TRANSFORM_API std::optional<std::vector<double>> TryNetworkTerminalExpectation(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                                       const std::vector<std::string> &observables,
                                                                                       const Network::INetwork<double> &network,
                                                                                       const NetworkExpectationEvaluator &evaluate);

// Adapt an already-configured native network without changing its configuration.
// Multi-host distributed execution retains its existing placement contract.
// executed, when given, reports whether the network ran: planning may resolve
// every observable (identities, zero factors) without it.
MAESTRO_TRANSFORM_API std::vector<double> EstimateOnNetwork(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                            const std::vector<std::string> &observables,
                                                            const std::shared_ptr<Network::INetwork<double>> &network, std::optional<size_t> host,
                                                            bool auto_reduce = false, EstimationCache *cache = nullptr, bool *executed = nullptr);
} // namespace MaestroExecution
