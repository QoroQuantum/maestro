// Additive discovery metadata for the native request contract.
#pragma once
#include "Options.h"
#include "Request.h"
#include "Json/NoiseJson.h"
// Source consumers such as Composer do not run Maestro's CMake configuration.
// Build identity is optional and must not prevent those consumers from
// compiling.
#if __has_include("MaestroBuildInfo.h")
#include "MaestroBuildInfo.h"
#else
#define MAESTRO_BUILD_VERSION "unknown"
#define MAESTRO_SOURCE_REVISION "unknown"
#endif

namespace MaestroExecution
{
inline const std::set<std::string> &DiagnosticNames()
{
    static const std::set<std::string> names{"trace",
                                             "purity",
                                             "trace_of_square",
                                             "hermiticity_residual",
                                             "is_hermitian",
                                             "partial_trace",
                                             "density_matrix",
                                             "unnormalized_density_matrix",
                                             "element",
                                             "expectation_complex",
                                             "unnormalized_expectation",
                                             "expectations_complex",
                                             "unnormalized_expectations",
                                             "operator_expectation"};
    return names;
}

// density_matrix_overlap and fidelity_with_statevector read the prepared mixed
// state directly, which the QCSim and GPU density-matrix/MPO simulators provide.
inline bool MixedStateQueryApplies(const SimulatorConfig &config)
{
    return (config.simulator_type == Backend::kQCSim || config.simulator_type == Backend::kGpuSim) && Mixed(config);
}

// fidelity_with_statevector reads a dense target of 2^n amplitudes, the same
// bound as a full basis output (max_output_elements at most 1,048,576).
constexpr size_t MaxTargetStatevectorQubits = 20;

// These are parser-level capabilities. Optional plugins still require runtime
// probing; discovery never initializes a GPU or loads a user's circuit.
inline bool DiagnosticApplies(const std::string &name, const SimulatorConfig &config)
{
    const bool native = config.simulator_type == Backend::kQCSim || config.simulator_type == Backend::kGpuSim;
    if (name == "operator_expectation")
        return native && config.simulation_type == Method::kMatrixProductState;
    if (name == "trace" || name == "purity" || name == "trace_of_square" || name == "hermiticity_residual" || name == "is_hermitian" || name == "partial_trace")
        return Mixed(config);
    return native && config.simulation_type == Method::kMatrixProductOperator;
}

template <typename Predicate> inline json::array Configurations(Predicate applies)
{
    json::array result;
    for (const auto &backend : Backends())
        for (const auto &method : Methods())
        {
            SimulatorConfig config;
            config.simulator_type = backend.second;
            config.simulation_type = method.second;
            if (Accepts(backend.second, method.second) && applies(config))
                result.emplace_back(json::object{{"backend", backend.first}, {"method", method.first}});
        }
    return result;
}

inline json::object OptionCapability(const Option &option)
{
    const std::string name(option.name), type(option.type);
    json::object result{{"name", name},
                        {"native_name", option.native_name},
                        {"type", type},
                        {"family", option.family},
                        {"supported_configurations", Configurations([&](const auto &config) { return Applies(option, config); })}};
    const std::vector<std::string> *choices = nullptr;
    if (name == "precision")
        choices = &Precisions();
    else if (name == "mps_sampling")
        choices = &MpsSamplingModes();
    else if (name == "counts_policy")
        choices = &CountsPolicies();
    else if (name == "mps_routing_strategy")
        choices = &MpsRoutingStrategies();
    else if (name == "truncation_mode")
        choices = &TruncationModes();
    else if (name == "mpo_kraus_completeness_check")
        choices = &KrausCompletenessChecks();
    else if (name == "mps_svd_solver" || name == "mpo_svd_solver" || name == "tensor_network_svd_solver")
        choices = &SvdSolvers();
    if (choices)
    {
        json::array values;
        for (const auto &choice : *choices)
            values.emplace_back(choice);
        result["enum"] = std::move(values);
    }
    if (name == "mps_routing_strategy")
        result["default"] = "legacy";
    if (name == "trajectory_policy")
    {
        result["enum"] = json::array{"block_v1", "shot_v1"};
        result["default"] = "shot_v1";
    }
    if (name == "counts_policy")
        result["default"] = "auto";
    if (type == "integer" || type == "nonnegative")
        result["minimum"] = 0;
    else if (type == "positive_integer")
        result["minimum"] = 1;
    else if (type == "positive_probability")
    {
        result["exclusiveMinimum"] = 0;
        result["maximum"] = 1;
    }
    else if (type == "lookahead")
    {
        result["minimum"] = -1;
        result["maximum"] = 1000000;
    }
    if (name == "max_simulators" || name == "pp_workers")
        result["maximum"] = 1024;
    if (name == "pp_workers")
        result["constraints"] = json::array{json::object{{"backend", "gpu"}, {"minimum", 1}, {"maximum", 64}}};
    if (name == "pp_max_fused_gates")
        result["maximum"] = 256;
    if (name == "gpu_device")
        result["maximum"] = INT_MAX;
    if (name == "max_bond_dimension")
    {
        result["exceptions"] =
            json::array{json::object{{"supported_configurations", Configurations([](const auto &config) { return DiagnosticApplies("element", config); })},
                                     {"minimum", 0},
                                     {"zero_means", "unlimited"}}};
    }
    if (name == "truncation_mode")
    {
        result["constraints"] = json::array{json::object{{"backend", "aer"}, {"enum", json::array{"discarded_weight"}}}};
    }
    return result;
}

inline json::object NativeCapabilities(const std::set<std::string> &operations)
{
    json::array backends, options, noiseKinds, names, diagnostics;
    for (const auto &entry : Backends())
    {
        json::array methods;
        for (const auto &method : Methods())
            if (Accepts(entry.second, method.second))
                methods.emplace_back(method.first);
        backends.emplace_back(
            json::object{{"name", entry.first},
                         {"legacy_id", static_cast<int>(entry.second)},
                         {"methods", methods},
                         {"compiled", true},
                         {"readiness", entry.first == "qcsim" || entry.first == "composite_qcsim" || entry.first == "aer" || entry.first == "composite_aer"
                                           ? "available"
                                           : "requires_runtime_probe"}});
    }
    for (const auto &option : Options())
        options.emplace_back(OptionCapability(option));
    for (const auto &name : DiagnosticNames())
        diagnostics.emplace_back(
            json::object{{"name", name}, {"supported_configurations", Configurations([&](const auto &config) { return DiagnosticApplies(name, config); })}});
    for (const auto &entry : NoiseKinds())
        noiseKinds.emplace_back(entry.first);
    for (const auto &operation : operations)
        names.emplace_back(operation);
    // Instruction-array circuits: gate names with their parameter counts, the
    // non-gate instructions, and the optional per-instruction fields.
    json::array gates;
    for (const auto &[name, count] : std::vector<std::pair<const char *, int>>{
             {"id", 0}, {"x", 0},   {"y", 0},     {"z", 0},  {"h", 0},   {"s", 0},   {"sdg", 0}, {"t", 0},  {"tdg", 0},  {"sx", 0},  {"sxdg", 0},  {"k", 0},
             {"p", 1},  {"rx", 1},  {"ry", 1},    {"rz", 1}, {"u", 3},   {"u1", 1},  {"u2", 2},  {"u3", 3}, {"swap", 0}, {"cx", 0},  {"cy", 0},    {"cz", 0},
             {"ch", 0}, {"csx", 0}, {"csxdg", 0}, {"cp", 1}, {"crx", 1}, {"cry", 1}, {"crz", 1}, {"cu", 4}, {"rxx", 1},  {"rzz", 1}, {"cswap", 0}, {"ccx", 0}})
        gates.emplace_back(json::object{{"name", name}, {"params", count}});
    const json::object instructions{{"gates", std::move(gates)},
                                    {"operations", json::array{"measure", "reset", "delay", "kraus", "pauli_channel"}},
                                    {"fields", json::object{{"conditional_reg", json::array{"gates"}},
                                                            {"noiseless", json::array{"gates"}},
                                                            {"readout_error", json::array{"measure"}},
                                                            {"clbits", json::array{"measure"}},
                                                            {"duration", json::array{"delay"}},
                                                            {"operators", json::array{"kraus"}},
                                                            {"probabilities", json::array{"pauli_channel"}}}},
                                    {"channel_evaluation",
                                     "Mixed-state methods apply kraus and pauli_channel exactly; pure-state methods sample pauli_channel per noise "
                                     "realization and need a trajectory noise block, whose channels may then be omitted"}};
    const json::object adjoint{
        {"operations", json::array{"adjoint_expectation_values", "adjoint_jacobian", "adjoint_vjp"}},
        {"supported_configurations",
         json::array{json::object{{"backend", "qcsim"}, {"method", "statevector"}}, json::object{{"backend", "composite_qcsim"}, {"method", "statevector"}}}},
        {"readouts", json::array{"pauli", "terms", "probabilities"}},
        {"differentiable_gates", json::object{{"p", 1}, {"rx", 1}, {"ry", 1}, {"rz", 1}, {"cp", 1}, {"crx", 1}, {"cry", 1}, {"crz", 1}, {"u", 3}, {"cu", 4}}},
        {"max_qubits", 63},
        {"max_probability_qubits", 24}};
    const json::object mixedStateQueries{{"operations", json::array{"density_matrix_overlap", "fidelity_with_statevector"}},
                                         {"supported_configurations", Configurations([](const auto &config) { return MixedStateQueryApplies(config); })},
                                         {"max_target_statevector_qubits", MaxTargetStatevectorQubits}};
    const auto mixedConfigurations = Configurations([](const auto &config) { return Mixed(config); });
    // The noise block: how realization counts resolve and what is reported.
    const json::object noise{
        {"modes", json::array{"combined", "pauli", "coherent", "analytical"}},
        {"evaluations", json::array{"auto", "exact", "trajectories"}},
        {"exact_configurations", mixedConfigurations},
        {"realizations",
         json::object{
             {"accepts", json::array{"positive_integer", "null", "auto"}},
             {"maximum", 1000000},
             {"automatic", json::object{{"deterministic_exact_channels", 1},
                                        {"sampled_execute", 64},
                                        {"sampled_estimate", 100},
                                        {"sampled_noisy_fidelity", 100},
                                        {"rule", "one realization when exact channels need no sampled disturbance (coherent, correlated) and, for "
                                                 "estimates, the circuit records no branch; explicit counts are honored, capped to shots for execution"}}}}},
        {"metadata", json::array{"mode", "evaluation", "seed", "realizations", "requested_realizations", "realizations_policy", "stochastic_noise",
                                 "stochastic_execution", "approximations"}},
        {"checkpoint_prefix", json::object{{"field", "prefix"},
                                           {"operations", json::array{"checkpoint_batch"}},
                                           {"requires", "deterministic exact evaluation: no coherent or correlated layers"},
                                           {"supported_configurations", mixedConfigurations},
                                           {"noise_scope", json::array{"suffix_only_ideal_prefix", "prefix_and_suffix"}}}}};
    const json::object estimate{{"evaluations", json::array{"deterministic", "trajectory_average"}},
                                {"dynamic_measurement_trajectories", kMeasurementTrajectories},
                                {"metadata", json::array{"stochastic_execution", "measurement_trajectories", "estimate_evaluation"}},
                                {"single_stochastic_realization_standard_errors", "null"}};
    const json::object noisyFidelity{{"modes", json::array{"combined", "pauli", "coherent"}},
                                     {"supported_configurations", Configurations([](const auto &) { return true; })},
                                     {"exact_configurations", mixedConfigurations},
                                     {"mixed_state_evaluation", "P(0...0) after the ideal circuit's adjoint"},
                                     {"single_stochastic_realization_standard_error", "null"}};
    const json::object automaticSelection{{"candidates", "simulator.candidates with selection automatic"},
                                          {"fallback", "first listed candidate that can hold the register"},
                                          {"stabilizer_shortcut", "Clifford circuits move to a stabilizer only as a listed backend/method pair"},
                                          {"non_clifford_stabilizer", "a listed stabilizer never runs a non-Clifford circuit"},
                                          {"failure", "when no listed candidate runs, the first candidate's error is reported"},
                                          {"unavailable", "error; an unlisted backend never runs"},
                                          {"exact_channels", "only when every candidate applies them"}};
    return {{"schema_version", SchemaVersion},
            {"instructions", instructions},
            {"adjoint", adjoint},
            {"mixed_state_queries", mixedStateQueries},
            {"noise", noise},
            {"estimate", estimate},
            {"noisy_fidelity", noisyFidelity},
            {"automatic_selection", automaticSelection},
            {"sampling_strategies", json::array{"reference_frame_v1", "extended_frame_v1", "terminal_v1", "trajectory_v1", "mixed_v1"}},
            {"api", "maestro.native.request"},
            {"build", json::object{{"version", MAESTRO_BUILD_VERSION}, {"source_revision", MAESTRO_SOURCE_REVISION}}},
            {"capability_scope", "validation"},
            {"diagnostics", diagnostics},
            {"operations", names},
            {"backends", backends},
            {"options", options},
            {"noise_channels", noiseKinds},
            {"circuit_formats", json::array{"openqasm", "instructions"}},
            {"python_required", false},
            {"max_request_bytes", MaxRequestBytes},
            {"max_result_bytes", MaxResultBytes},
            {"max_output_elements", 1048576},
            {"count_order", "classical_bit_0_first"},
            {"basis_order", "qubit_0_least_significant"},
            {"complex_encoding", "[real, imaginary]"},
            {"mpi_lifecycle", "externally_initialized_collective"}};
}
} // namespace MaestroExecution
