#pragma once
#include "Json.h"
#include "SimulatorConfig.h"
#include <map>
#include <climits>
#include <set>

namespace MaestroExecution {
using Backend = Simulators::SimulatorType;
using Method = Simulators::SimulationType;
inline const std::map<std::string, Backend>& Backends() {
  static const std::map<std::string, Backend> values{
      {"qcsim", Backend::kQCSim},
      {"composite_qcsim", Backend::kCompositeQCSim},
      {"quest", Backend::kQuestSim},
#ifndef NO_QISKIT_AER
      {"aer", Backend::kQiskitAer},
      {"composite_aer", Backend::kCompositeQiskitAer},
#endif
#ifdef __linux__
      {"gpu", Backend::kGpuSim},
      {"distributed_gpu", Backend::kDistGpuSim},
      {"distributed_mpi_gpu", Backend::kDistMpiGpuSim},
#endif
  };
  return values;
}
inline const std::map<std::string, Method>& Methods() {
  static const std::map<std::string, Method> values{
      {"statevector", Method::kStatevector},
      {"matrix_product_state", Method::kMatrixProductState},
      {"stabilizer", Method::kStabilizer},
      {"tensor_network", Method::kTensorNetwork},
      {"pauli_propagator", Method::kPauliPropagator},
      {"extended_stabilizer", Method::kExtendedStabilizer},
      {"path_integral", Method::kPathIntegral},
      {"density_matrix", Method::kDensityMatrix},
      {"matrix_product_operator", Method::kMatrixProductOperator}};
  return values;
}
inline std::string BackendName(Backend backend) {
  for (const auto& entry : Backends())
    if (entry.second == backend) return entry.first;
  return "unknown";
}
inline std::string MethodName(Method method) {
  for (const auto& entry : Methods())
    if (entry.second == method) return entry.first;
  return "unknown";
}
inline bool Accepts(Backend backend, Method method) {
  if (backend == Backend::kQCSim) return true;
  if (backend == Backend::kGpuSim)
    return method == Method::kStatevector ||
           method == Method::kMatrixProductState ||
           method == Method::kTensorNetwork ||
           method == Method::kPauliPropagator ||
           method == Method::kDensityMatrix ||
           method == Method::kMatrixProductOperator;
#ifndef NO_QISKIT_AER
  if (backend == Backend::kQiskitAer)
    return method != Method::kPauliPropagator &&
           method != Method::kPathIntegral &&
           method != Method::kMatrixProductOperator;
#endif
  return method == Method::kStatevector;
}
inline bool Mixed(const SimulatorConfig& config) {
  return config.simulation_type == Method::kDensityMatrix ||
         config.simulation_type == Method::kMatrixProductOperator;
}

struct Option {
  const char* name;
  const char* native_name;
  const char* type;
  const char* family;
};
inline const std::vector<Option>& Options() {
  static const std::vector<Option> options{
      {"max_bond_dimension", "matrix_product_state_max_bond_dimension",
       "positive_integer", "tensor"},
      {"singular_value_threshold", "matrix_product_state_truncation_threshold",
       "nonnegative", "tensor"},
      {"truncation_mode", "matrix_product_state_truncation_mode", "string",
       "tensor"},
      {"use_double_precision", "use_double_precision", "boolean", "gpu"},
      {"precision", "precision", "string", "precision"},
      {"gpu_device", "gpu_device", "integer", "device"},
      {"seed", "seed", "integer", "all"},
      {"mps_measure_no_collapse", "", "boolean", "mps"},
      {"mps_sample_measure_algorithm", "mps_sample_measure_algorithm", "string",
       "mps"},
      {"disable_optimized_swapping", "", "boolean", "mps"},
      {"lookahead_depth", "", "lookahead", "mps"},
      {"optimize_circuit", "", "boolean", "all"},
      {"max_simulators", "max_simulators", "positive_integer", "all"},
      {"mpo_kraus_completeness_check",
       "matrix_product_operator_kraus_completeness_check", "string", "mpo"},
      {"mpo_restore_trace_after_truncation",
       "matrix_product_operator_restore_trace_after_truncation", "boolean",
       "cpu_mpo"},
      {"mpo_hermitize_after_truncation",
       "matrix_product_operator_hermitize_after_truncation", "boolean",
       "cpu_mpo"},
      {"mps_use_gesvd", "matrix_product_state_use_gesvd", "boolean", "gpu_mps"},
      {"mps_use_gesvdj", "matrix_product_state_use_gesvdj", "boolean",
       "gpu_mps"},
      {"mps_use_gesvdp", "matrix_product_state_use_gesvdp", "boolean",
       "gpu_mps"},
      {"mps_use_gesvdr", "matrix_product_state_use_gesvdr", "boolean",
       "gpu_mps"},
      {"mpo_use_gesvd", "matrix_product_operator_use_gesvd", "boolean",
       "gpu_mpo"},
      {"mpo_use_gesvdj", "matrix_product_operator_use_gesvdj", "boolean",
       "gpu_mpo"},
      {"mpo_use_gesvdp", "matrix_product_operator_use_gesvdp", "boolean",
       "gpu_mpo"},
      {"mpo_use_gesvdr", "matrix_product_operator_use_gesvdr", "boolean",
       "gpu_mpo"},
      {"tensor_network_use_gesvd", "tensor_network_use_gesvd", "boolean",
       "gpu_tn"},
      {"tensor_network_use_gesvdj", "tensor_network_use_gesvdj", "boolean",
       "gpu_tn"},
      {"tensor_network_use_gesvdp", "tensor_network_use_gesvdp", "boolean",
       "gpu_tn"},
      {"tensor_network_use_gesvdr", "tensor_network_use_gesvdr", "boolean",
       "gpu_tn"},
      {"pp_coefficient_threshold", "pauli_propagator_coefficient_threshold",
       "nonnegative", "pp"},
      {"pp_pauli_weight_threshold", "pauli_propagator_pauli_weight_threshold",
       "integer", "pp"},
      {"pp_steps_between_trims", "pauli_propagator_steps_between_trims",
       "positive_integer", "pp"},
      {"pp_steps_between_deduplications",
       "pauli_propagator_num_gates_between_deduplications", "positive_integer",
       "pp"},
      {"path_integral_threshold", "path_integral_threshold", "nonnegative",
       "path"},
  };
  return options;
}
inline bool Applies(const Option& option, const SimulatorConfig& config) {
  const std::string family(option.family);
  const auto backend = config.simulator_type;
  const auto method = config.simulation_type;
  const bool gpu = backend == Backend::kGpuSim;
  const bool distributed = Simulators::IsDistributedGpuSimulator(backend);
  const bool mps = method == Method::kMatrixProductState;
  const bool mpo = method == Method::kMatrixProductOperator;
  if (family == "all") return true;
  if (family == "gpu") return gpu || distributed;
  if (family == "device") return gpu || distributed;
  if (family == "precision") {
#ifndef NO_QISKIT_AER
    if (backend == Backend::kQiskitAer) return true;
#endif
    return distributed;
  }
  if (family == "tensor") return mps || mpo || method == Method::kTensorNetwork;
  if (family == "mps") return mps;
  if (family == "mpo") return mpo;
  if (family == "cpu_mpo") return mpo && backend == Backend::kQCSim;
  if (family == "gpu_mps") return gpu && mps;
  if (family == "gpu_mpo") return gpu && mpo;
  if (family == "gpu_tn") return gpu && method == Method::kTensorNetwork;
  if (family == "pp") return method == Method::kPauliPropagator;
  if (family == "path") return method == Method::kPathIntegral;
  return false;
}

inline SimulatorConfig ParseConfig(const json::object& simulator) {
  Keys(simulator, {"backend", "method", "selection", "options", "distribution",
                   "candidates"});
  SimulatorConfig config;
  const auto backend = Backends().find(String(simulator, "backend", "qcsim"));
  std::string methodName = String(simulator, "method", "statevector");
  if (methodName == "mps") methodName = "matrix_product_state";
  if (methodName == "mpo") methodName = "matrix_product_operator";
  const auto method = Methods().find(methodName);
  Supported(backend != Backends().end(),
            "Backend is unknown or not compiled into this library");
  Supported(method != Methods().end(), "Unknown simulation method");
  config.simulator_type = backend->second;
  config.simulation_type = method->second;
  Supported(Accepts(config.simulator_type, config.simulation_type),
            "Unsupported backend/method combination");
  const auto selection = String(simulator, "selection", "fixed");
  Require(selection == "fixed" || selection == "automatic",
          "selection must be fixed or automatic");
  config.fixed_backend = selection == "fixed";
  Supported(config.fixed_backend ||
                !Simulators::IsDistributedGpuSimulator(config.simulator_type),
            "Distributed backends require fixed selection");
  std::set<std::string> seen;
  std::map<std::string, unsigned> svd;
  for (const auto& entry : Sub(simulator, "options")) {
    const std::string key(entry.key());
    const Option* match = nullptr;
    for (const auto& option : Options())
      if (key == option.name ||
          (option.native_name[0] && key == option.native_name))
        match = &option;
    Require(match != nullptr, "Unknown simulator option: " + key);
    Require(seen.insert(match->name).second, "Duplicate option alias: " + key);
    Supported(Applies(*match, config),
              "Option is unsupported by selected backend/method: " + key);
    const auto& value = entry.value();
    const std::string type(match->type), name(match->name),
        native(match->native_name);
    if (type == "boolean")
      Boolean(value);
    else if (type == "integer")
      UInt(value);
    else if (type == "positive_integer")
      Require(UInt(value) > 0, key + " must be positive");
    else if (type == "nonnegative")
      Require(Number(value) >= 0, key + " must be nonnegative");
    else if (type == "lookahead")
      Require(value.is_int64() && value.as_int64() >= -1 &&
                  value.as_int64() <= 1000000,
              "Invalid lookahead_depth");
    else
      String(value);
    if (name == "gpu_device")
      Require(UInt(value) <= INT_MAX, "gpu_device is too large");
    if (name == "max_simulators")
      Require(UInt(value) <= 1024, "max_simulators exceeds 1024");
    if (name == "truncation_mode") {
      const auto mode = String(value);
      Require(mode == "relative_max" || mode == "discarded_weight",
              "Unknown truncation_mode");
#ifndef NO_QISKIT_AER
      Supported(config.simulator_type != Backend::kQiskitAer ||
                    mode == "discarded_weight",
                "Aer supports only discarded_weight truncation");
#endif
    }
    if (name == "mpo_kraus_completeness_check")
      Require(String(value) == "ignore" || String(value) == "warn" ||
                  String(value) == "strict",
              "Unknown MPO completeness check mode");
    if (name == "precision")
      Require(String(value) == "single" || String(value) == "double",
              "Invalid precision");
    if (name == "mps_sample_measure_algorithm")
      Require(String(value) == "mps_apply_measure" ||
                  String(value) == "mps_probabilities",
              "Invalid MPS measurement algorithm");
    if (native.find("_use_gesvd") != std::string::npos && Boolean(value))
      Require(++svd[native.substr(0, native.find("_use_"))] == 1,
              "Conflicting SVD algorithms");
    // Network controls use typed fields; other settings keep the validated
    // native Configure key/value. Keep both paths in SimulatorConfig.h in sync
    // when changing serialization or an option's meaning.
    if (name == "disable_optimized_swapping")
      config.disable_optimized_swapping = Boolean(value);
    else if (name == "lookahead_depth")
      config.lookahead_depth = static_cast<int>(value.as_int64());
    else if (name == "optimize_circuit")
      config.optimize_circuit = Boolean(value);
    else if (name == "mps_measure_no_collapse")
      config.mps_measure_no_collapse = Boolean(value);
    else if (name == "seed")
      config.seed = UInt(value);
    else
      config.native_options[native] = Scalar(value);
  }
  Require(!(seen.count("mps_measure_no_collapse") &&
            seen.count("mps_sample_measure_algorithm")),
          "Use only one MPS measurement option");
  const auto& distribution = Sub(simulator, "distribution");
  if (distribution.contains("global_qubits"))
    Require(Array(Field(distribution, "global_qubits")).size() <= 5,
            "At most five global qubits are supported");
  Supported(distribution.empty() ||
                Simulators::IsDistributedGpuSimulator(config.simulator_type),
            "distribution requires a distributed backend");
  Keys(distribution,
       {"devices", "global_qubits", "backend", "flags", "max_queued_gates",
        "transfer_workspace_bytes", "snapshot_storage", "host_qubit_indexing",
        "mpi_p2p_bits"});
  for (const auto& entry : distribution) {
    const std::string key(entry.key());
    Supported(
        key != "devices" || config.simulator_type != Backend::kDistMpiGpuSim,
        "MPI device selection uses rank-local visibility or gpu_device");
    Supported(key != "mpi_p2p_bits" ||
                  config.simulator_type == Backend::kDistMpiGpuSim,
              "mpi_p2p_bits requires MPI");
    std::string value;
    if (key == "devices" || key == "global_qubits") {
      std::set<uint64_t> unique;
      for (const auto& item : Array(entry.value())) {
        const auto number = UInt(item);
        Require(number <= INT_MAX &&
                    (key == "devices" || unique.insert(number).second),
                "Invalid or duplicate global qubit index");
        if (!value.empty()) value += ",";
        value += std::to_string(number);
      }
      Require(!value.empty() || key == "global_qubits",
              "Device list must not be empty");
    } else {
      if (key == "flags")
        Require(UInt(entry.value()) <= 31, "flags must be in 0..31");
      else if (key == "max_queued_gates")
        Require(UInt(entry.value()) >= 1 && UInt(entry.value()) <= 65536,
                "max_queued_gates must be in 1..65536");
      else if (key == "transfer_workspace_bytes")
        UInt(entry.value());
      else if (key == "mpi_p2p_bits")
        Require(UInt(entry.value()) <= 5, "mpi_p2p_bits must be in 0..5");
      else if (key == "host_qubit_indexing")
        Require(String(entry.value()) == "auto" ||
                    String(entry.value()) == "local" ||
                    String(entry.value()) == "global",
                "Invalid host_qubit_indexing");
      else if (key == "snapshot_storage")
        Require(
            String(entry.value()) == "host" || String(entry.value()) == "gpu",
            "snapshot_storage must be host or gpu");
      else if (key == "backend") {
        Require(String(entry.value()) == "ex" ||
                    String(entry.value()) == "conventional",
                "Unknown distribution backend");
        Supported(config.simulator_type != Backend::kDistMpiGpuSim ||
                      String(entry.value()) == "ex",
                  "MPI requires the ex backend");
      }
      value = Scalar(entry.value());
    }
    config.distributed_options[key == "mpi_p2p_bits" ? key
                                                     : "distributed_" + key] =
        value;
  }
  const auto flags = UInt(distribution, "flags", 0);
  Require((flags & 10) != 10, "fixed and pinned layouts are mutually exclusive");
  Supported(!(flags & 17) || (config.simulator_type != Backend::kDistMpiGpuSim && String(distribution, "backend", "ex") == "conventional"), "Shared-device and host-staging flags require the conventional local backend");
  Supported(config.simulator_type != Backend::kDistMpiGpuSim || !(flags & 4), "MPI does not support the local full-mesh topology flag");
  if (const auto* devices = distribution.if_contains("devices")) {
    const auto count = Array(*devices).size();
    Require(count > 0 && count <= 32 && !(count & (count - 1)),
            "devices must contain a power of two shards in 1..32");
    std::set<uint64_t> unique;
    for (const auto& device : Array(*devices)) unique.insert(UInt(device));
    Require(!(UInt(distribution, "flags", 0) & 1) ||
                String(distribution, "backend", "ex") == "conventional",
            "allow-shared-device requires the conventional backend");
    // Native flag 1 explicitly permits multiple logical shards on a device.
    Require(unique.size() == count || (UInt(distribution, "flags", 0) & 1),
            "Repeated devices require the allow-shared-device flag (1)");
    if (const auto* globals = distribution.if_contains("global_qubits"))
      Require((size_t{1} << Array(*globals).size()) == count,
              "global_qubits count must equal log2(shards)");
  }
  if (config.simulator_type == Backend::kDistMpiGpuSim && !config.seed)
    config.seed = 0;
  if (config.native_options.count("use_double_precision") &&
      Simulators::IsDistributedGpuSimulator(config.simulator_type)) {
    Require(!config.native_options.count("precision"),
            "Specify only one precision option");
    config.native_options["precision"] =
        config.native_options.at("use_double_precision") == "true" ? "double"
                                                                   : "single";
    config.native_options.erase("use_double_precision");
  }
  if (const auto* candidates = simulator.if_contains("candidates")) {
    Supported(!config.fixed_backend,
              "Candidate lists require automatic selection");
    const auto& values = Array(*candidates);
    Require(!values.empty() && values.size() <= 64, "Invalid candidate count");
    for (const auto& value : values) {
      const auto& candidate = Object(value);
      Keys(candidate, {"backend", "method"});
      const auto parsed = ParseConfig(candidate);
      Supported(!Simulators::IsDistributedGpuSimulator(parsed.simulator_type),
                "Distribution cannot be an automatic candidate");
      config.optimization_candidates.emplace_back(parsed.simulator_type,
                                                  parsed.simulation_type);
    }
  }
  return config;
}
}  // namespace MaestroExecution
