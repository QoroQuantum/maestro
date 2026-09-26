// Shared native configuration. This header has no Python dependency.
#pragma once
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include "maestrolib/Interface.h"
#include "maestrolib/Maestro.h"
#include "Simulators/RandomSeed.h"

namespace MaestroExecution {
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
  std::optional<size_t> max_bond_dimension = std::nullopt;
  std::optional<double> singular_value_threshold = std::nullopt;
  // "relative_max" (keep sigma_i > threshold * sigma_max, the historical
  // QCSim/GPU convention) or "discarded_weight" (discard the smallest singular
  // values until their cumulative squared weight reaches the threshold,
  // matching Qiskit Aer's and ITensor's convention -- the default on every
  // backend unless this is set). The Aer backend only ever implements
  // discarded_weight and raises if relative_max is requested; QCSim and the GPU
  // backend support switching between both.
  std::optional<std::string> truncation_mode = std::nullopt;
  bool use_double_precision = false;
  bool disable_optimized_swapping = false;
  int lookahead_depth = -1;
  bool mps_measure_no_collapse = true;
  std::optional<std::string> mpo_kraus_completeness_check = std::nullopt;
  bool mpo_restore_trace_after_truncation = false;
  bool mpo_hermitize_after_truncation = false;
  // GPU SVD algorithm selection. At most one setting in each backend group
  // should be true; configuring one clears the other choices in the backend.
  bool mps_use_gesvd = false;
  bool mps_use_gesvdj = false;
  bool mps_use_gesvdp = false;
  bool mps_use_gesvdr = false;
  bool mpo_use_gesvd = false;
  bool mpo_use_gesvdj = false;
  bool mpo_use_gesvdp = false;
  bool mpo_use_gesvdr = false;
  bool tensor_network_use_gesvd = false;
  bool tensor_network_use_gesvdj = false;
  bool tensor_network_use_gesvdp = false;
  bool tensor_network_use_gesvdr = false;

  // true for double precision, false for single precision, nullopt for default
  // this is a separate setting for qiskit aer, the use_double_precision above
  // is for gpu mps and tensor network simulators
  std::optional<bool> precision = std::nullopt;

  // PauliPropagator truncation parameters
  std::optional<double> pp_coefficient_threshold = std::nullopt;
  std::optional<size_t> pp_pauli_weight_threshold = std::nullopt;
  std::optional<int> pp_steps_between_trims = std::nullopt;
  std::optional<int> pp_steps_between_deduplications = std::nullopt;

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
  bool enable_causal_cone_reduction = true;
  size_t causal_cone_statevector_threshold = 20;
  std::unordered_map<std::string, std::string> native_options;

  SimulatorConfig() = default;

  SimulatorConfig(
      Simulators::SimulatorType st, Simulators::SimulationType set,
      std::optional<size_t> mb, std::optional<double> sv, bool dp, bool ds,
      int la, bool mnc, std::optional<std::string> tm,
      std::optional<uint64_t> random_seed,
      std::optional<int> device = std::nullopt,
      std::unordered_map<std::string, std::string> distribution = {})
      : simulator_type(st),
        simulation_type(set),
        max_bond_dimension(mb),
        singular_value_threshold(sv),
        truncation_mode(std::move(tm)),
        use_double_precision(dp),
        disable_optimized_swapping(ds),
        lookahead_depth(la),
        mps_measure_no_collapse(mnc),
        seed(random_seed),
        gpu_device(device),
        distributed_options(std::move(distribution)) {
    if (device && *device < 0)
      throw std::invalid_argument("gpu_device must be nonnegative");
    if ((st == Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
         || st == Simulators::SimulatorType::kCompositeQiskitAer
#endif
         ) &&
        set != Simulators::SimulationType::kStatevector) {
      throw std::invalid_argument(
          "Composite simulators only support Statevector simulation type.");
    }
    if (st == Simulators::SimulatorType::kQuestSim &&
        set != Simulators::SimulationType::kStatevector) {
      throw std::invalid_argument(
          "QuestSim only supports Statevector simulation type.");
    }
  }
};

// Helper to configure the simulation network
inline std::shared_ptr<Network::INetwork<double>> ConfigureNetwork(
    unsigned long int handle, const SimulatorConfig& config,
    bool for_expectations = false) {
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
  network->Configure("enable_causal_cone_reduction",
                     config.enable_causal_cone_reduction ? "true" : "false");
  network->Configure(
      "causal_cone_statevector_threshold",
      std::to_string(config.causal_cone_statevector_threshold).c_str());
  for (const auto& [key, value] : config.native_options)
    network->Configure(key.c_str(), value.c_str());

  for (const auto& [key, value] : config.distributed_options) {
    if (key.compare(0, 12, "distributed_") != 0 &&
        key.compare(0, 4, "mpi_") != 0)
      throw std::invalid_argument(
          "distributed_options accepts only distributed_* and mpi_* keys");
    network->Configure(key.c_str(), value.c_str());
  }
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
  const auto configure_svd = [&network](const char* backend, const char* method,
                                        bool enabled) {
    if (!enabled) return;
    const std::string key = std::string(backend) + "_use_" + method;
    network->Configure(key.c_str(), "true");
  };
  configure_svd("matrix_product_state", "gesvd", config.mps_use_gesvd);
  configure_svd("matrix_product_state", "gesvdj", config.mps_use_gesvdj);
  configure_svd("matrix_product_state", "gesvdp", config.mps_use_gesvdp);
  configure_svd("matrix_product_state", "gesvdr", config.mps_use_gesvdr);
  configure_svd("matrix_product_operator", "gesvd", config.mpo_use_gesvd);
  configure_svd("matrix_product_operator", "gesvdj", config.mpo_use_gesvdj);
  configure_svd("matrix_product_operator", "gesvdp", config.mpo_use_gesvdp);
  configure_svd("matrix_product_operator", "gesvdr", config.mpo_use_gesvdr);
  configure_svd("tensor_network", "gesvd", config.tensor_network_use_gesvd);
  configure_svd("tensor_network", "gesvdj", config.tensor_network_use_gesvdj);
  configure_svd("tensor_network", "gesvdp", config.tensor_network_use_gesvdp);
  configure_svd("tensor_network", "gesvdr", config.tensor_network_use_gesvdr);
  if (config.use_double_precision) {
    network->Configure("use_double_precision", "1");
  }

  if (config.precision) {
    network->Configure("precision", *config.precision ? "double" : "single");
  }
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

  if (config.native_options.count("mps_sample_measure_algorithm")) {
    // Already configured above.
  } else if (!config.mps_measure_no_collapse) {
    network->Configure("mps_sample_measure_algorithm", "mps_apply_measure");
  } else {
    network->Configure("mps_sample_measure_algorithm", "mps_probabilities");
  }

  // Create the configured backend. The desired simulator type is specified via
  // RemoveAllOptimizationSimulatorsAndAdd above.
  // PauliPropagator truncation settings are Configured before CreateSimulator;
  // the state replays its config map once the propagator exists, so they are
  // applied then. Note that both thresholds are only consulted during a
  // truncation pass, so a trim or deduplication cadence must also be set.
  if (config.pp_coefficient_threshold) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *config.pp_coefficient_threshold;
    network->Configure("pauli_propagator_coefficient_threshold",
                       oss.str().c_str());
  }
  if (config.pp_pauli_weight_threshold) {
    network->Configure(
        "pauli_propagator_pauli_weight_threshold",
        std::to_string(*config.pp_pauli_weight_threshold).c_str());
  }
  if (config.pp_steps_between_trims) {
    network->Configure("pauli_propagator_steps_between_trims",
                       std::to_string(*config.pp_steps_between_trims).c_str());
  }
  if (config.pp_steps_between_deduplications) {
    network->Configure(
        "pauli_propagator_num_gates_between_deduplications",
        std::to_string(*config.pp_steps_between_deduplications).c_str());
  }
  if (config.path_integral_threshold) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *config.path_integral_threshold;
    auto val = oss.str();
    network->Configure("path_integral_threshold", val.c_str());
  }

  // Expectation execution sizes its register after extracting the cone. Keep
  // the topology intact but initialize only a one-qubit backend here; ordinary
  // execution will resize it if reduction is disabled or unsupported.
  const size_t initial_qubits =
      for_expectations && !Simulators::IsDistributedGpuSimulator(config.simulator_type)
          ? 1 : 0;

  // Distribution must be selected before circuit mapping: its configured
  // register and MPI control flow must not depend on the CPU optimizer.
  if (config.fixed_backend ||
      Simulators::IsDistributedGpuSimulator(config.simulator_type))
    network->CreateSimulator(config.simulator_type, config.simulation_type,
                             initial_qubits);
  else if (config.simulator_type == Simulators::SimulatorType::kGpuSim &&
           (config.simulation_type == Simulators::SimulationType::kDensityMatrix ||
            config.simulation_type ==
                Simulators::SimulationType::kMatrixProductOperator))
    // Automatic selection retains this CPU simulator when the GPU is absent.
    // MPO preserves mixed states and exact channels without a dense allocation.
    network->CreateSimulator(Simulators::SimulatorType::kQCSim,
                             Simulators::SimulationType::kMatrixProductOperator, initial_qubits);
  else
    network->CreateSimulator(Simulators::SimulatorType::kQCSim,
                             Simulators::SimulationType::kMatrixProductState,
                             initial_qubits);

  // Verify the simulator was actually created (e.g. GPU library may fail)
  if (!network->GetSimulator()) {
    return nullptr;
  }

  return network;
}
}  // namespace MaestroExecution
