#pragma once

#include "../Circuit/Circuit.h"
#include "Transforms/Export.h"

#include <optional>

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
};

// No simulator is allocated during preparation. Empty requests have no groups.
MAESTRO_TRANSFORM_API EstimationPlan PrepareEstimation(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                                       const SimulatorConfig &config, EstimationCache *cache = nullptr);
MAESTRO_TRANSFORM_API EstimationResult EstimateCircuit(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                                       const SimulatorConfig &config, EstimationCache *cache = nullptr);

// Adapt an already-configured native network without changing its configuration.
// Multi-host distributed execution retains its existing placement contract.
MAESTRO_TRANSFORM_API std::vector<double> EstimateOnNetwork(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                            const std::vector<std::string> &observables,
                                                            const std::shared_ptr<Network::INetwork<double>> &network, std::optional<size_t> host,
                                                            bool auto_reduce = false, EstimationCache *cache = nullptr);
} // namespace MaestroExecution
