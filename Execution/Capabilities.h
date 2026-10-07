// Additive discovery metadata for the native request contract.
#pragma once
#include "Options.h"
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
    if (type == "integer" || type == "nonnegative")
        result["minimum"] = 0;
    else if (type == "positive_integer")
        result["minimum"] = 1;
    else if (type == "lookahead")
    {
        result["minimum"] = -1;
        result["maximum"] = 1000000;
    }
    if (name == "max_simulators" || name == "pp_workers")
        result["maximum"] = 1024;
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
    return {{"schema_version", SchemaVersion},
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
