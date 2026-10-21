#pragma once

#include "../Network/Network.h"
#include "Estimation.h"

namespace MaestroExecution
{
struct SamplingResult
{
    Network::INetwork<double>::ExecuteResults counts;
    size_t components = 0;
    size_t largest_component = 0;
    Simulators::SimulatorType simulator;
    Simulators::SimulationType method;
    int gpu_device = -1;
    size_t max_bond_dimension = 0;
    bool gate_fusion = false;
    size_t gate_fusion_max_qubits = 0;
    double time_taken = 0;
};

// Histogram-level random pairing; memory scales with occupied outcomes. Sparse
// rows use weighted draws without replacement, dense rows use hypergeometric allocation.
MAESTRO_TRANSFORM_API Network::INetwork<double>::ExecuteResults CombineIndependentCounts(const Network::INetwork<double>::ExecuteResults &joint,
                                                                                         const Network::INetwork<double>::ExecuteResults &component,
                                                                                         const std::vector<size_t> &bits, uint64_t seed);

// Only disjoint structural past cones factorize. Overlapping/budget-packed
// estimation groups must NEVER be independently sampled. nullopt uses the
// existing execution path (dynamic circuits, distributed layouts, no saving).
MAESTRO_TRANSFORM_API std::optional<SamplingResult> SampleDisjointCones(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                        const SimulatorConfig &config, size_t shots);
MAESTRO_TRANSFORM_API SamplingResult SampleCircuit(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const SimulatorConfig &config, size_t shots);
} // namespace MaestroExecution
