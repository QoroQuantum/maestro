#pragma once

#include "Circuit/Circuit.h"
#include "Export.h"
#include "Simulators/Interfaces/State.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace MaestroExecution
{
// Partitions observables into groups whose backward lightcones are pairwise
// disjoint across groups (Union-Find over shared lightcone qubits). Each group
// is a list of indices into `observables`, ascending, and the groups together
// cover every index exactly once. Observables with an empty lightcone
// (identity-only) share one group.
//
// Falls back to a single group holding every index when the circuit contains
// non-unitary operations or an observable is malformed, matching the
// conditions under which CausalConeTransform declines to reduce.
//
// With max_cluster_qubits, overlapping cones are packed greedily instead: each
// group is seeded by the largest remaining cone and grows by the overlapping
// observable that adds the fewest qubits while the union stays within the
// budget. A cone larger than the budget forms its own group, absorbing only
// cones it contains. Groups never join disjoint cones, so an unbounded budget
// reproduces the default partition. Groups are ordered by first member.
MAESTRO_TRANSFORM_API std::vector<std::vector<size_t>> ClusterObservables(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                          const std::vector<std::string> &observables,
                                                                          std::optional<size_t> max_cluster_qubits = std::nullopt);

// GPU Pauli propagation pays about a second of setup per simulator run and saves
// only a few milliseconds per group, so it keeps every observable in one run.
MAESTRO_TRANSFORM_API bool BenefitsFromObservableClustering(Simulators::SimulatorType simulator_type, Simulators::SimulationType simulation_type);
} // namespace MaestroExecution
