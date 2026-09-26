// Shared native configuration. This header has no Python dependency.
#pragma once
#include <algorithm>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "maestrolib/Interface.h"
#include "maestrolib/Maestro.h"
#include "Simulators/RandomSeed.h"

namespace MaestroExecution {
// Deduplication cadence for PauliPropagator simulations that leave it unset.
inline constexpr int kDefaultPpGatesBetweenDeduplications = 10;

inline const std::vector<std::string>& TruncationModes() {
  static const std::vector<std::string> values{"relative_max",
                                               "discarded_weight"};
  return values;
}
inline const std::vector<std::string>& Precisions() {
  static const std::vector<std::string> values{"single", "double"};
  return values;
}
inline const std::vector<std::string>& SvdSolvers() {
  static const std::vector<std::string> values{"gesvd", "gesvdj", "gesvdp",
                                               "gesvdr"};
  return values;
}
inline const std::vector<std::string>& MpsSamplingModes() {
  static const std::vector<std::string> values{"probabilities",
                                               "apply_measure"};
  return values;
}
inline const std::vector<std::string>& KrausCompletenessChecks() {
  static const std::vector<std::string> values{"ignore", "warn", "strict"};
  return values;
}

inline void RequireOneOf(const std::optional<std::string>& value,
                         const std::vector<std::string>& allowed,
                         const char* name) {
  if (!value ||
      std::find(allowed.begin(), allowed.end(), *value) != allowed.end())
    return;
  std::string message = std::string(name) + " must be one of";
  for (size_t i = 0; i < allowed.size(); ++i)
    message += (i ? ", '" : " '") + allowed[i] + "'";
  throw std::invalid_argument(message + "; got '" + *value + "'.");
}

struct SimulatorConfig {
  // Python exposes these typed fields directly. Native requests also use typed
  // network controls, but store most validated backend options in
  // native_options below. Both representations are active and meet in
  // ConfigureNetwork.
  Simulators::SimulatorType simulator_type = Simulators::SimulatorType::kQCSim;
  Simulators::SimulationType simulation_type =
      Simulators::SimulationType::kStatevector;
  // Unset uses the backend default; GPU MPS/MPO resolve to 128 in the network
  // configuration so initial-layout planning and execution share the cap.
  // The three truncation settings also apply to MPO and GPU tensor networks.
  std::optional<size_t> max_bond_dimension = std::nullopt;
  std::optional<double> singular_value_threshold = std::nullopt;
  // "relative_max" (keep sigma_i >= threshold * sigma_max, the historical
  // QCSim/GPU convention) or "discarded_weight" (discard the smallest singular
  // values until their cumulative squared weight reaches the threshold,
  // matching Qiskit Aer's and ITensor's convention -- the default on every
  // backend unless this is set). The Aer backend only ever implements
  // discarded_weight and raises if relative_max is requested; QCSim and the GPU
  // backend support switching between both.
  std::optional<std::string> truncation_mode = std::nullopt;
  // "single" or "double" for Qiskit Aer and the GPU simulators; unset keeps
  // each backend's default. Other backends ignore it.
  std::optional<std::string> precision = std::nullopt;
  bool disable_optimized_swapping = false;
  int lookahead_depth = -1;
  // "probabilities" (no collapse) or "apply_measure" (measure and restore).
  std::string mps_sampling = "probabilities";
  std::optional<std::string> mpo_kraus_completeness_check = std::nullopt;
  bool mpo_restore_trace_after_truncation = false;
  bool mpo_hermitize_after_truncation = false;
  // GPU SVD solver per backend: "gesvd", "gesvdj", "gesvdp" or "gesvdr"; unset
  // keeps the GPU library's default.
  std::optional<std::string> mps_svd_solver = std::nullopt;
  std::optional<std::string> mpo_svd_solver = std::nullopt;
  std::optional<std::string> tensor_network_svd_solver = std::nullopt;

  // PauliPropagator truncation; the thresholds apply on each trim or
  // deduplication pass.
  std::optional<double> pp_coefficient_threshold = std::nullopt;
  std::optional<size_t> pp_max_pauli_weight = std::nullopt;
  std::optional<int> pp_gates_between_trims = std::nullopt;
  std::optional<int> pp_gates_between_deduplications = std::nullopt;

  // path integral parameters
  std::optional<double> path_integral_threshold = std::nullopt;
  std::optional<uint64_t> seed = std::nullopt;
  std::optional<int> gpu_device = std::nullopt;
  // Values use the same names and syntax as ISimulator::Configure.
  std::unordered_map<std::string, std::string> distributed_options;

  // Native requests use fixed selection; Python's legacy automatic path keeps
  // its historical default. Extra options are validated by the request parser.
  std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>>
      optimization_candidates;
  bool fixed_backend = false;
  bool optimize_circuit = true;
  std::unordered_map<std::string, std::string> native_options;

  // Throws std::invalid_argument for an unsupported combination or value.
  void Validate() const {
    if ((simulator_type == Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
         || simulator_type == Simulators::SimulatorType::kCompositeQiskitAer
#endif
         ) &&
        simulation_type != Simulators::SimulationType::kStatevector)
      throw std::invalid_argument(
          "Composite simulators only support Statevector simulation type.");
    if (simulator_type == Simulators::SimulatorType::kQuestSim &&
        simulation_type != Simulators::SimulationType::kStatevector)
      throw std::invalid_argument(
          "QuestSim only supports Statevector simulation type.");
    if (gpu_device && *gpu_device < 0)
      throw std::invalid_argument("gpu_device must be nonnegative");
    RequireOneOf(truncation_mode, TruncationModes(), "truncation_mode");
    RequireOneOf(precision, Precisions(), "precision");
    RequireOneOf(mps_sampling, MpsSamplingModes(), "mps_sampling");
    RequireOneOf(mpo_kraus_completeness_check, KrausCompletenessChecks(),
                 "mpo_kraus_completeness_check");
    RequireOneOf(mps_svd_solver, SvdSolvers(), "mps_svd_solver");
    RequireOneOf(mpo_svd_solver, SvdSolvers(), "mpo_svd_solver");
    RequireOneOf(tensor_network_svd_solver, SvdSolvers(),
                 "tensor_network_svd_solver");
    // The propagator takes these modulo a gate index.
    for (const auto& [cadence, name] :
         {std::pair{pp_gates_between_trims, "pp_gates_between_trims"},
          std::pair{pp_gates_between_deduplications,
                    "pp_gates_between_deduplications"}})
      if (cadence && *cadence < 1)
        throw std::invalid_argument(std::string(name) + " must be positive");
    for (const auto& entry : distributed_options)
      if (entry.first.compare(0, 12, "distributed_") != 0 &&
          entry.first.compare(0, 4, "mpi_") != 0)
        throw std::invalid_argument(
            "distributed_options accepts only distributed_* and mpi_* keys");
  }
};

// Native Configure keys for typed options without a one-to-one key. Precision
// sets both keys: GpuState reads use_double_precision, the others precision.
inline std::vector<std::pair<std::string, std::string>> TypedNativeOptions(
    const SimulatorConfig& config) {
  std::vector<std::pair<std::string, std::string>> options;
  for (const auto& [backend, solver] :
       {std::pair{"matrix_product_state", config.mps_svd_solver},
        std::pair{"matrix_product_operator", config.mpo_svd_solver},
        std::pair{"tensor_network", config.tensor_network_svd_solver}})
    if (solver) options.emplace_back(std::string(backend) + "_use_" + *solver,
                                     "true");
  if (config.precision) {
    options.emplace_back("precision", *config.precision);
    options.emplace_back("use_double_precision",
                         *config.precision == "double" ? "1" : "0");
  }
  options.emplace_back("mps_sample_measure_algorithm",
                       "mps_" + config.mps_sampling);
  return options;
}

// Helper to configure the simulation network
inline std::shared_ptr<Network::INetwork<double>> ConfigureNetwork(
    unsigned long int handle, const SimulatorConfig& config) {
  config.Validate();
  if (Simulators::IsDistributedGpuSimulator(config.simulator_type) &&
      config.simulation_type != Simulators::SimulationType::kStatevector)
    throw std::invalid_argument(
        "Distributed GPU supports only Statevector simulation");
  // QuEST only supports statevector simulation
  if (config.simulator_type == Simulators::SimulatorType::kQuestSim &&
      config.simulation_type != Simulators::SimulationType::kStatevector) {
    throw std::invalid_argument(
        "QuestSim only supports Statevector simulation type.");
  }

  // Composite only supports statevector simulation
  if ((config.simulator_type == Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
       ||
       config.simulator_type == Simulators::SimulatorType::kCompositeQiskitAer
#endif
       ) &&
      config.simulation_type != Simulators::SimulationType::kStatevector) {
    throw std::invalid_argument(
        "Composite simulators only support Statevector simulation type.");
  }

  if (RemoveAllOptimizationSimulatorsAndAdd(handle, (int)config.simulator_type,
                                            (int)config.simulation_type) == 0) {
    return nullptr;
  }

  if (!config.optimization_candidates.empty()) {
    const auto& first = config.optimization_candidates.front();
    RemoveAllOptimizationSimulatorsAndAdd(handle, static_cast<int>(first.first),
                                          static_cast<int>(first.second));
    for (size_t i = 1; i < config.optimization_candidates.size(); ++i)
      AddOptimizationSimulator(
          handle, static_cast<int>(config.optimization_candidates[i].first),
          static_cast<int>(config.optimization_candidates[i].second));
  }
  auto* maestro = static_cast<Maestro*>(GetMaestroObject());
  auto network = maestro->GetSimpleSimulator(handle);

  if (!network) return nullptr;

  network->SetOptimizeSimulator(!config.fixed_backend);
  network->GetController()->SetOptimizeCircuit(config.optimize_circuit);
  for (const auto& [key, value] : config.native_options)
    network->Configure(key.c_str(), value.c_str());

  for (const auto& [key, value] : config.distributed_options)
    network->Configure(key.c_str(), value.c_str());
  if (config.simulator_type == Simulators::SimulatorType::kDistMpiGpuSim &&
      !config.seed)
    network->Configure("seed", std::to_string(Simulators::GenerateRandomSeed(
                                                  config.simulator_type,
                                                  config.distributed_options))
                                   .c_str());
  if (config.gpu_device) {
    if (*config.gpu_device < 0)
      throw std::invalid_argument("gpu_device must be nonnegative");
    network->Configure("gpu_device",
                       std::to_string(*config.gpu_device).c_str());
  }

  if (config.max_bond_dimension) {
    auto val = std::to_string(*config.max_bond_dimension);
    network->Configure("matrix_product_state_max_bond_dimension", val.c_str());
  } else if (config.simulator_type == Simulators::SimulatorType::kGpuSim &&
             (config.simulation_type ==
                  Simulators::SimulationType::kMatrixProductState ||
              config.simulation_type ==
                  Simulators::SimulationType::kMatrixProductOperator) &&
             !config.native_options.count(
                 "matrix_product_state_max_bond_dimension") &&
             !(config.simulation_type ==
                   Simulators::SimulationType::kMatrixProductOperator &&
               config.native_options.count(
                   "matrix_product_operator_max_bond_dimension"))) {
    // Match GpuState and the GPU library before the network's initial-layout
    // planner runs. Explicit native options already applied above take priority.
    network->Configure("matrix_product_state_max_bond_dimension", "128");
  }
  if (config.singular_value_threshold) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *config.singular_value_threshold;
    auto val = oss.str();
    network->Configure("matrix_product_state_truncation_threshold",
                       val.c_str());
  }
  if (config.truncation_mode) {
    network->Configure("matrix_product_state_truncation_mode",
                       config.truncation_mode->c_str());
  }
  if (config.mpo_kraus_completeness_check)
    network->Configure("matrix_product_operator_kraus_completeness_check",
                       config.mpo_kraus_completeness_check->c_str());
  if (config.mpo_restore_trace_after_truncation)
    network->Configure("matrix_product_operator_restore_trace_after_truncation",
                       "true");
  if (config.mpo_hermitize_after_truncation)
    network->Configure("matrix_product_operator_hermitize_after_truncation",
                       "true");
  for (const auto& [key, value] : TypedNativeOptions(config))
    network->Configure(key.c_str(), value.c_str());
  if (config.seed) {
    const auto value = std::to_string(*config.seed);
    network->Configure("seed", value.c_str());
  }

  // Disable MPS swap optimization if requested
  if (config.disable_optimized_swapping) {
    network->SetInitialQubitsMapOptimization(false);
    network->SetMPSOptimizeSwaps(false);
  }

  // Set the lookahead depth for swap optimization
  network->SetLookaheadDepth(config.lookahead_depth);


  // Create the configured backend. The desired simulator type is specified via
  // RemoveAllOptimizationSimulatorsAndAdd above.
  // PauliPropagator truncation settings are Configured before CreateSimulator;
  // the state replays its config map once the propagator exists, so they are
  // applied then. Both thresholds are only consulted during a trim or
  // deduplication pass; PauliPropagator simulations deduplicate by default.
  if (config.pp_coefficient_threshold) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *config.pp_coefficient_threshold;
    network->Configure("pauli_propagator_coefficient_threshold",
                       oss.str().c_str());
  }
  if (config.pp_max_pauli_weight) {
    network->Configure("pauli_propagator_pauli_weight_threshold",
                       std::to_string(*config.pp_max_pauli_weight).c_str());
  }
  if (config.pp_gates_between_trims) {
    network->Configure("pauli_propagator_steps_between_trims",
                       std::to_string(*config.pp_gates_between_trims).c_str());
  }
  auto pp_gates_between_deduplications = config.pp_gates_between_deduplications;
  if (!pp_gates_between_deduplications &&
      config.simulation_type == Simulators::SimulationType::kPauliPropagator &&
      !config.native_options.count(
          "pauli_propagator_num_gates_between_deduplications"))
    pp_gates_between_deduplications = kDefaultPpGatesBetweenDeduplications;
  if (pp_gates_between_deduplications) {
    network->Configure(
        "pauli_propagator_num_gates_between_deduplications",
        std::to_string(*pp_gates_between_deduplications).c_str());
  }
  if (config.path_integral_threshold) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *config.path_integral_threshold;
    auto val = oss.str();
    network->Configure("path_integral_threshold", val.c_str());
  }

  // Distribution must be selected before circuit mapping: its configured
  // register and MPI control flow must not depend on the CPU optimizer.
  if (config.fixed_backend ||
      Simulators::IsDistributedGpuSimulator(config.simulator_type))
    network->CreateSimulator(config.simulator_type, config.simulation_type);
  else if (config.simulator_type == Simulators::SimulatorType::kGpuSim &&
           (config.simulation_type == Simulators::SimulationType::kDensityMatrix ||
            config.simulation_type ==
                Simulators::SimulationType::kMatrixProductOperator))
    // Automatic selection retains this CPU simulator when the GPU is absent.
    // MPO preserves mixed states and exact channels without a dense allocation.
    network->CreateSimulator(Simulators::SimulatorType::kQCSim,
                             Simulators::SimulationType::kMatrixProductOperator);
  else
    network->CreateSimulator();

  // Verify the simulator was actually created (e.g. GPU library may fail)
  if (!network->GetSimulator()) {
    return nullptr;
  }

  return network;
}
}  // namespace MaestroExecution
