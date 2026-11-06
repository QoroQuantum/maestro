// Shared native configuration. This header has no Python dependency.
#pragma once
#include "../Simulators/Core/RandomSeed.h"
#include "../Utils/Threads/RealizationWorkers.h"
#include "../maestrolib/Interface.h"
#include "../maestrolib/Maestro.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace MaestroExecution
{
// Deduplication cadence for PauliPropagator simulations that leave it unset.
inline constexpr int kDefaultPpGatesBetweenDeduplications = 10;

inline const std::vector<std::string> &TruncationModes()
{
    static const std::vector<std::string> values{"relative_max", "discarded_weight"};
    return values;
}

inline const std::vector<std::string> &Precisions()
{
    static const std::vector<std::string> values{"single", "double"};
    return values;
}

inline const std::vector<std::string> &SvdSolvers()
{
    static const std::vector<std::string> values{"gesvd", "gesvdj", "gesvdp", "gesvdr"};
    return values;
}

inline const std::vector<std::string> &MpsSamplingModes()
{
    static const std::vector<std::string> values{"probabilities", "apply_measure"};
    return values;
}

inline const std::vector<std::string> &MpsRoutingStrategies()
{
    static const std::vector<std::string> values{"combined", "legacy"};
    return values;
}

inline const std::vector<std::string> &KrausCompletenessChecks()
{
    static const std::vector<std::string> values{"ignore", "warn", "strict"};
    return values;
}

inline void RequireOneOf(const std::optional<std::string> &value, const std::vector<std::string> &allowed, const char *name)
{
    if (!value || std::find(allowed.begin(), allowed.end(), *value) != allowed.end())
        return;
    std::string message = std::string(name) + " must be one of";
    for (size_t i = 0; i < allowed.size(); ++i)
        message += (i ? ", '" : " '") + allowed[i] + "'";
    throw std::invalid_argument(message + "; got '" + *value + "'.");
}

struct SimulatorConfig
{
    // Python exposes these typed fields directly. Native requests also use typed
    // network controls, but store most validated backend options in
    // native_options below. Both representations are active and meet in
    // ConfigureNetwork.
    Simulators::SimulatorType simulator_type = Simulators::SimulatorType::kQCSim;
    Simulators::SimulationType simulation_type = Simulators::SimulationType::kStatevector;
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
    // CPU MPS/MPO automatic routing: "combined" or "legacy". Unset retains native options or the legacy default.
    std::optional<std::string> mps_routing_strategy;
    // Experimental accuracy-aware routing. Unset preserves native request values.
    std::optional<bool> mps_routing_virtual_bonds;
    std::optional<double> mps_routing_truncation_weight;
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

    // GPU thresholds apply at trim boundaries; QCSim also trims at deduplication boundaries.
    std::optional<double> pp_coefficient_threshold = std::nullopt;
    std::optional<size_t> pp_max_pauli_weight = std::nullopt;
    std::optional<int> pp_gates_between_trims = std::nullopt;
    std::optional<int> pp_gates_between_deduplications = std::nullopt;

    // path integral parameters
    std::optional<double> path_integral_threshold = std::nullopt;
    std::optional<uint64_t> seed = std::nullopt;
    std::optional<int> gpu_device = std::nullopt;
    bool enable_causal_cone_reduction = true;
    // Exact independent sampling of disjoint measurement cones. Opt-in because
    // decomposing a seeded sampler changes its random streams.
    bool enable_sampling_reduction = false;
    // Average over unobserved reset outcomes when reducing pure trajectories.
    // Preserves ensemble expectations/distributions, not individual trajectories.
    bool allow_reset_marginalization = false;
    // With causal-cone reduction, split multi-observable estimates into groups
    // with disjoint lightcones and simulate each on its own compact register.
    bool enable_observable_clustering = true;
    // Unset: dense statevectors use a 20-qubit batching budget, density matrices
    // use 10; other methods merge connected cones. Oversized cones stay intact.
    std::optional<size_t> max_cluster_qubits = std::nullopt;
    // Optionally batch disjoint small cones up to this width to amortize setup.
    std::optional<size_t> min_cluster_qubits = std::nullopt;
    // Opt-in absolute error budget per expectation, for bounded backward Pauli
    // evaluation of exact noisy channels. Unset preserves exact reduction.
    std::optional<double> observable_error_tolerance = std::nullopt;
    // Automatically reduce check-generated invariant subspaces for expectation evaluation.
    bool auto_reduce = true;
    // Values use the same names and syntax as ISimulator::Configure.
    std::unordered_map<std::string, std::string> distributed_options;

    // Native requests use fixed selection; Python's legacy automatic path keeps
    // its historical default. Extra options are validated by the request parser.
    std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>> optimization_candidates;
    bool fixed_backend = false;
    bool optimize_circuit = true;
    // Unset: each backend's default, which turns fusion off for registers too
    // small to benefit. true/false force it.
    std::optional<bool> gate_fusion;
    // Unset retains the network defaults and any native request options.
    std::optional<std::string> trajectory_policy;
    std::optional<size_t> trajectory_max_memory_mb;
    std::optional<size_t> max_simulators;
    // auto accelerates unseeded counts that the ordinary executor could run on
    // the stabilizer; legacy retains the existing stream; frame_v1 also opts
    // seeded requests and other automatically selected methods in.
    std::string counts_policy = "auto";
    std::unordered_map<std::string, std::string> native_options;

    // Throws std::invalid_argument for an unsupported combination or value.
    void Validate() const
    {
        if ((simulator_type == Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
             || simulator_type == Simulators::SimulatorType::kCompositeQiskitAer
#endif
             ) &&
            simulation_type != Simulators::SimulationType::kStatevector)
            throw std::invalid_argument("Composite simulators only support Statevector simulation type.");
        if (simulator_type == Simulators::SimulatorType::kQuestSim && simulation_type != Simulators::SimulationType::kStatevector)
            throw std::invalid_argument("QuestSim only supports Statevector simulation type.");
        if (mps_routing_truncation_weight && (!std::isfinite(*mps_routing_truncation_weight) || *mps_routing_truncation_weight < 0.))
            throw std::invalid_argument("mps_routing_truncation_weight must be finite and nonnegative");
        if (gpu_device && *gpu_device < 0)
            throw std::invalid_argument("gpu_device must be nonnegative");
        RequireOneOf(trajectory_policy, {"block_v1", "shot_v1"}, "trajectory_policy");
        RequireOneOf(counts_policy, {"auto", "legacy", "frame_v1"}, "counts_policy");
        if (trajectory_max_memory_mb && (!*trajectory_max_memory_mb || *trajectory_max_memory_mb > (std::numeric_limits<size_t>::max() >> 20)))
            throw std::invalid_argument("trajectory_max_memory_mb must be a positive representable MiB count");
        if (max_cluster_qubits && !*max_cluster_qubits)
            throw std::invalid_argument("max_cluster_qubits must be positive");
        if (min_cluster_qubits && (!*min_cluster_qubits || (max_cluster_qubits && *min_cluster_qubits > *max_cluster_qubits)))
            throw std::invalid_argument("min_cluster_qubits must be positive and no greater than max_cluster_qubits");
        if (observable_error_tolerance && (!std::isfinite(*observable_error_tolerance) || *observable_error_tolerance <= 0 || *observable_error_tolerance > 1))
            throw std::invalid_argument("observable_error_tolerance must be finite and in (0, 1]");
        if (max_simulators && (!*max_simulators || *max_simulators > 1024))
            throw std::invalid_argument("max_simulators must be between 1 and 1024");
        RequireOneOf(truncation_mode, TruncationModes(), "truncation_mode");
        RequireOneOf(precision, Precisions(), "precision");
        RequireOneOf(mps_sampling, MpsSamplingModes(), "mps_sampling");
        RequireOneOf(mps_routing_strategy, MpsRoutingStrategies(), "mps_routing_strategy");
        RequireOneOf(mpo_kraus_completeness_check, KrausCompletenessChecks(), "mpo_kraus_completeness_check");
        RequireOneOf(mps_svd_solver, SvdSolvers(), "mps_svd_solver");
        RequireOneOf(mpo_svd_solver, SvdSolvers(), "mpo_svd_solver");
        RequireOneOf(tensor_network_svd_solver, SvdSolvers(), "tensor_network_svd_solver");
        // The propagator takes these modulo a gate index.
        for (const auto &[cadence, name] :
             {std::pair{pp_gates_between_trims, "pp_gates_between_trims"}, std::pair{pp_gates_between_deduplications, "pp_gates_between_deduplications"}})
            if (cadence && *cadence < 1)
                throw std::invalid_argument(std::string(name) + " must be positive");
        for (const auto &entry : distributed_options)
            if (entry.first.compare(0, 12, "distributed_") != 0 && entry.first.compare(0, 4, "mpi_") != 0)
                throw std::invalid_argument("distributed_options accepts only distributed_* and mpi_* keys");
        // Distribution is an explicit execution choice, never selected automatically.
        for (const auto &candidate : optimization_candidates)
            if (Simulators::IsDistributedGpuSimulator(candidate.first))
                throw std::invalid_argument("Distribution cannot be an automatic candidate");
    }
};

// The realization state a method holds, for realization memory budgets
// (Utils::RealizationStateBytes); none when its size has no known bound.
inline std::optional<Utils::RealizationState> RealizationStateOf(Simulators::SimulationType method)
{
    using Method = Simulators::SimulationType;
    switch (method)
    {
    case Method::kStatevector:
        return Utils::RealizationState::kStatevector;
    case Method::kDensityMatrix:
        return Utils::RealizationState::kDensityMatrix;
    case Method::kStabilizer:
        return Utils::RealizationState::kStabilizer;
    case Method::kMatrixProductState:
        return Utils::RealizationState::kMatrixProductState;
    case Method::kMatrixProductOperator:
        return Utils::RealizationState::kMatrixProductOperator;
    default:
        return std::nullopt;
    }
}

// The configured MPS/MPO bond cap, or 0 when none is set.
inline size_t ConfiguredBondCap(const SimulatorConfig &config)
{
    if (config.max_bond_dimension)
        return *config.max_bond_dimension;
    auto value =
        config.native_options.find(config.simulation_type == Simulators::SimulationType::kMatrixProductOperator ? "matrix_product_operator_max_bond_dimension"
                                                                                                                : "matrix_product_state_max_bond_dimension");
    if (value == config.native_options.end())
        value = config.native_options.find("matrix_product_state_max_bond_dimension");
    return value == config.native_options.end() ? 0 : static_cast<size_t>(std::stoull(value->second));
}

// The soft realization memory budget in MiB (trajectory_max_memory_mb).
inline size_t RealizationBudgetMB(const SimulatorConfig &config)
{
    if (config.trajectory_max_memory_mb)
        return *config.trajectory_max_memory_mb;
    const auto value = config.native_options.find("trajectory_max_memory_mb");
    return value == config.native_options.end() ? 1024 : static_cast<size_t>(std::stoull(value->second));
}

// Bytes of one realization on `qubits` qubits, or 0 when unbounded or unknown.
inline long double RealizationBytes(const SimulatorConfig &config, size_t qubits)
{
    const auto state = RealizationStateOf(config.simulation_type);
    if (!state)
        return 0;
    const bool tensor = *state == Utils::RealizationState::kMatrixProductState || *state == Utils::RealizationState::kMatrixProductOperator;
    const size_t bond = tensor ? ConfiguredBondCap(config) : 0;
    if (tensor && !bond)
        return 0;
    const long double bytes = Utils::RealizationStateBytes(*state, qubits, bond);
    return std::isfinite(bytes) && bytes < static_cast<long double>(std::numeric_limits<size_t>::max()) ? bytes : 0;
}

inline bool HasCandidateList(const SimulatorConfig &config)
{
    return !config.fixed_backend && !config.optimization_candidates.empty();
}

inline bool CandidateListAdmitsStabilizer(const SimulatorConfig &config)
{
    for (const auto &[type, method] : config.optimization_candidates)
        if (type == Simulators::SimulatorType::kQCSim && method == Simulators::SimulationType::kStabilizer)
            return true;
    return false;
}

// Whether the ordinary executor could itself run this request on the QCSim
// stabilizer simulator. ConfigureNetwork restricts automatic selection to the
// requested method, or to optimization_candidates when they are supplied: an
// explicit list replaces the requested method, so a requested Stabilizer
// method is admitted only if the list contains it too.
inline bool AdmitsStabilizerExecution(const SimulatorConfig &config)
{
    return HasCandidateList(config) ? CandidateListAdmitsStabilizer(config) : config.simulation_type == Simulators::SimulationType::kStabilizer;
}

// Whether a request may use reference frames: a QCSim Stabilizer request, or
// automatic selection that permits the stabilizer. An explicit candidate list
// names every backend the request allows and replaces the requested pair
// (ConfigureNetwork), so it decides on its own: frames need the QCSim
// stabilizer in it, whatever the requested backend.
inline bool AdmitsReferenceFrames(const SimulatorConfig &config)
{
    if (HasCandidateList(config))
        return CandidateListAdmitsStabilizer(config);
    return config.simulator_type == Simulators::SimulatorType::kQCSim &&
           (config.simulation_type == Simulators::SimulationType::kStabilizer || !config.fixed_backend);
}

// Seed zero is explicit too. Only an absent seed permits automatic policy changes.
// auto never replaces a method the ordinary executor would honor. frame_v1 is an
// explicit opt-in and may replace an automatically selected method, but an
// explicit candidate list names every backend the request allows: frame_v1
// uses the stabilizer only if that list includes it. Sampling reduction keeps
// the ordinary executor, which splits disjoint measurement cones.
inline bool PreferFrameCounts(const SimulatorConfig &config, std::optional<uint64_t> noise_seed = std::nullopt)
{
    if (!AdmitsReferenceFrames(config) || config.enable_sampling_reduction)
        return false;
    if (config.counts_policy == "frame_v1")
        return true;
    return config.counts_policy == "auto" && AdmitsStabilizerExecution(config) && !config.seed && !noise_seed;
}

// Native Configure keys for typed options without a one-to-one key. Precision
// sets both precision aliases for compatibility with existing callers.
inline std::vector<std::pair<std::string, std::string>> TypedNativeOptions(const SimulatorConfig &config)
{
    std::vector<std::pair<std::string, std::string>> options;
    for (const auto &[backend, solver] : {std::pair{"matrix_product_state", config.mps_svd_solver}, std::pair{"matrix_product_operator", config.mpo_svd_solver},
                                          std::pair{"tensor_network", config.tensor_network_svd_solver}})
        if (solver)
            options.emplace_back(std::string(backend) + "_use_" + *solver, "true");
    if (config.precision)
    {
        options.emplace_back("precision", *config.precision);
        options.emplace_back("use_double_precision", *config.precision == "double" ? "1" : "0");
    }
    options.emplace_back("mps_sample_measure_algorithm", "mps_" + config.mps_sampling);
    return options;
}

// Helper to configure the simulation network. A nonzero allocationQubits sizes a
// fixed CPU backend to exactly that register; zero keeps the network default,
// which also allocates each host's network-entangled qubit.
inline std::shared_ptr<Network::INetwork<double>> ConfigureNetwork(unsigned long int handle, const SimulatorConfig &config, size_t allocationQubits = 0)
{
    config.Validate();
    if (Simulators::IsDistributedGpuSimulator(config.simulator_type) && config.simulation_type != Simulators::SimulationType::kStatevector)
        throw std::invalid_argument("Distributed GPU supports only Statevector simulation");
    // QuEST only supports statevector simulation
    if (config.simulator_type == Simulators::SimulatorType::kQuestSim && config.simulation_type != Simulators::SimulationType::kStatevector)
    {
        throw std::invalid_argument("QuestSim only supports Statevector simulation type.");
    }

    // Composite only supports statevector simulation
    if ((config.simulator_type == Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
         || config.simulator_type == Simulators::SimulatorType::kCompositeQiskitAer
#endif
         ) &&
        config.simulation_type != Simulators::SimulationType::kStatevector)
    {
        throw std::invalid_argument("Composite simulators only support Statevector simulation type.");
    }

    if (RemoveAllOptimizationSimulatorsAndAdd(handle, (int)config.simulator_type, (int)config.simulation_type) == 0)
    {
        return nullptr;
    }

    if (!config.optimization_candidates.empty())
    {
        const auto &first = config.optimization_candidates.front();
        RemoveAllOptimizationSimulatorsAndAdd(handle, static_cast<int>(first.first), static_cast<int>(first.second));
        for (size_t i = 1; i < config.optimization_candidates.size(); ++i)
            AddOptimizationSimulator(handle, static_cast<int>(config.optimization_candidates[i].first),
                                     static_cast<int>(config.optimization_candidates[i].second));
    }
    auto *maestro = static_cast<Maestro *>(GetMaestroObject());
    auto network = maestro->GetSimpleSimulator(handle);

    if (!network)
        return nullptr;

    network->Configure("gate_fusion", !config.gate_fusion ? "auto" : *config.gate_fusion ? "true" : "false");
    network->SetOptimizeSimulator(!config.fixed_backend);
    network->Configure("counts_policy", config.counts_policy.c_str());
    network->GetController()->SetOptimizeCircuit(config.optimize_circuit);
    for (const auto &[key, value] : config.native_options)
        network->Configure(key.c_str(), value.c_str());
    if (config.mps_routing_strategy)
        network->Configure(Simulators::RoutingConfig::kStrategy, config.mps_routing_strategy->c_str());
    if (config.mps_routing_virtual_bonds)
        network->Configure(Simulators::RoutingConfig::kVirtualBonds, *config.mps_routing_virtual_bonds ? "true" : "false");
    if (config.mps_routing_truncation_weight)
    {
        std::ostringstream weight;
        weight << std::setprecision(std::numeric_limits<double>::max_digits10) << *config.mps_routing_truncation_weight;
        network->Configure(Simulators::RoutingConfig::kTruncationWeight, weight.str().c_str());
    }
    if (config.trajectory_policy)
        network->Configure("trajectory_policy", config.trajectory_policy->c_str());
    if (config.trajectory_max_memory_mb)
        network->Configure("trajectory_max_memory_mb", std::to_string(*config.trajectory_max_memory_mb).c_str());
    if (config.max_simulators)
        network->SetMaxSimulators(*config.max_simulators);

    for (const auto &[key, value] : config.distributed_options)
        network->Configure(key.c_str(), value.c_str());
    if (config.simulator_type == Simulators::SimulatorType::kDistMpiGpuSim && !config.seed)
        network->Configure("seed", std::to_string(Simulators::GenerateRandomSeed(config.simulator_type, config.distributed_options)).c_str());
    if (config.gpu_device)
    {
        if (*config.gpu_device < 0)
            throw std::invalid_argument("gpu_device must be nonnegative");
        network->Configure("gpu_device", std::to_string(*config.gpu_device).c_str());
    }

    if (config.max_bond_dimension)
    {
        auto val = std::to_string(*config.max_bond_dimension);
        network->Configure("matrix_product_state_max_bond_dimension", val.c_str());
    }
    else if (config.simulator_type == Simulators::SimulatorType::kGpuSim &&
             (config.simulation_type == Simulators::SimulationType::kMatrixProductState ||
              config.simulation_type == Simulators::SimulationType::kMatrixProductOperator) &&
             !config.native_options.count("matrix_product_state_max_bond_dimension") &&
             !(config.simulation_type == Simulators::SimulationType::kMatrixProductOperator &&
               config.native_options.count("matrix_product_operator_max_bond_dimension")))
    {
        // Match GpuState and the GPU library before the network's initial-layout
        // planner runs. Explicit native options already applied above take priority.
        network->Configure("matrix_product_state_max_bond_dimension", "128");
    }
    if (config.singular_value_threshold)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::max_digits10) << *config.singular_value_threshold;
        auto val = oss.str();
        network->Configure("matrix_product_state_truncation_threshold", val.c_str());
    }
    if (config.truncation_mode)
    {
        network->Configure("matrix_product_state_truncation_mode", config.truncation_mode->c_str());
    }
    if (config.mpo_kraus_completeness_check)
        network->Configure("matrix_product_operator_kraus_completeness_check", config.mpo_kraus_completeness_check->c_str());
    if (config.mpo_restore_trace_after_truncation)
        network->Configure("matrix_product_operator_restore_trace_after_truncation", "true");
    if (config.mpo_hermitize_after_truncation)
        network->Configure("matrix_product_operator_hermitize_after_truncation", "true");
    for (const auto &[key, value] : TypedNativeOptions(config))
        network->Configure(key.c_str(), value.c_str());
    if (config.seed)
    {
        const auto value = std::to_string(*config.seed);
        network->Configure("seed", value.c_str());
    }

    // Disable MPS swap optimization if requested
    if (config.disable_optimized_swapping)
    {
        network->SetInitialQubitsMapOptimization(false);
        network->SetMPSOptimizeSwaps(false);
    }

    // Set the lookahead depth for swap optimization
    network->SetLookaheadDepth(config.lookahead_depth);

    // Create the configured backend. The desired simulator type is specified via
    // RemoveAllOptimizationSimulatorsAndAdd above.
    // PauliPropagator truncation settings are Configured before CreateSimulator;
    // the state replays its config map once the propagator exists, so they are
    // applied then. An omitted trim interval follows the effective deduplication
    // interval when thresholds are supplied. Explicit trim settings take precedence.
    if (config.pp_coefficient_threshold)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::max_digits10) << *config.pp_coefficient_threshold;
        network->Configure("pauli_propagator_coefficient_threshold", oss.str().c_str());
    }
    if (config.pp_max_pauli_weight)
    {
        network->Configure("pauli_propagator_pauli_weight_threshold", std::to_string(*config.pp_max_pauli_weight).c_str());
    }
    auto pp_gates_between_deduplications = config.pp_gates_between_deduplications;
    const auto native_deduplication = config.native_options.find("pauli_propagator_num_gates_between_deduplications");
    if (!pp_gates_between_deduplications && config.simulation_type == Simulators::SimulationType::kPauliPropagator &&
        native_deduplication == config.native_options.end())
        pp_gates_between_deduplications = kDefaultPpGatesBetweenDeduplications;
    if (pp_gates_between_deduplications)
        network->Configure("pauli_propagator_num_gates_between_deduplications", std::to_string(*pp_gates_between_deduplications).c_str());

    if (config.pp_gates_between_trims)
        network->Configure("pauli_propagator_steps_between_trims", std::to_string(*config.pp_gates_between_trims).c_str());
    else if (config.simulation_type == Simulators::SimulationType::kPauliPropagator && !config.native_options.count("pauli_propagator_steps_between_trims") &&
             (config.pp_coefficient_threshold || config.pp_max_pauli_weight || config.native_options.count("pauli_propagator_coefficient_threshold") ||
              config.native_options.count("pauli_propagator_pauli_weight_threshold")))
    {
        const auto interval = pp_gates_between_deduplications ? std::to_string(*pp_gates_between_deduplications) : native_deduplication->second;
        network->Configure("pauli_propagator_steps_between_trims", interval.c_str());
    }
    if (config.path_integral_threshold)
    {
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<double>::max_digits10) << *config.path_integral_threshold;
        auto val = oss.str();
        network->Configure("path_integral_threshold", val.c_str());
    }

    // Distribution must be selected before circuit mapping: its configured
    // register and MPI control flow must not depend on the CPU optimizer.
    if (config.fixed_backend || Simulators::IsDistributedGpuSimulator(config.simulator_type))
        network->CreateSimulator(config.simulator_type, config.simulation_type,
                                 Simulators::IsDistributedGpuSimulator(config.simulator_type) ? 0 : allocationQubits);
    else if (HasCandidateList(config))
    {
        // An explicit candidate list names every backend the request allows,
        // one or several. Without a cost model, selection starts from the
        // network's simulator when it is one of them: allocate the first listed
        // candidate whose register fits (a density matrix of 40 qubits yields
        // to a listed MPO). When none can be created, report that instead of
        // running a placeholder outside the list. (Without a list, a GPU
        // mixed-state request below keeps its CPU fallback.)
        bool created = false;
        std::exception_ptr failure;
        for (const auto &[type, method] : config.optimization_candidates)
        {
            try
            {
                network->CreateSimulator(type, method, allocationQubits);
            }
            catch (const std::exception &)
            {
                failure = std::current_exception();
                continue; // a failed allocation leaves an unusable simulator behind
            }
            const auto simulator = network->GetSimulator();
            if ((created = simulator && simulator->GetType() == type && simulator->GetSimulationType() == method))
                break;
        }
        if (!created)
        {
            if (failure)
                std::rethrow_exception(failure);
            throw std::runtime_error("No optimization candidate backend is available");
        }
    }
    else if (config.simulator_type == Simulators::SimulatorType::kGpuSim && (config.simulation_type == Simulators::SimulationType::kDensityMatrix ||
                                                                             config.simulation_type == Simulators::SimulationType::kMatrixProductOperator))
        // Automatic selection retains this CPU simulator when the GPU is absent.
        // MPO preserves mixed states and exact channels without a dense allocation.
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductOperator);
    else
        network->CreateSimulator();

    // Verify the simulator was actually created (e.g. GPU library may fail)
    if (!network->GetSimulator())
    {
        return nullptr;
    }

    return network;
}

// The one backend that code creating its simulator explicitly (checkpoints,
// noisy fidelity) uses, so that noise policy, allocation and metadata agree:
// the requested pair, the sole candidate of an automatic list, or the first
// listed candidate whose register of `qubits` qubits can be allocated, as
// ConfigureNetwork chooses. The result is fixed to that backend. Several
// candidates are probed by allocating them in turn.
inline SimulatorConfig ResolveCandidateBackend(const SimulatorConfig &config, size_t qubits)
{
    if (!HasCandidateList(config))
        return config;
    auto resolved = config;
    resolved.fixed_backend = true;
    resolved.optimization_candidates.clear();
    if (config.optimization_candidates.size() == 1)
    {
        std::tie(resolved.simulator_type, resolved.simulation_type) = config.optimization_candidates.front();
        return resolved;
    }
    const size_t width = std::max<size_t>(1, qubits);
    if (width > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Circuit width exceeds the simulator limit");
    GetMaestroObjectWithMute();
    std::exception_ptr failure;
    for (const auto &[type, method] : config.optimization_candidates)
    {
        resolved.simulator_type = type;
        resolved.simulation_type = method;
        const auto handle = CreateSimpleSimulator(static_cast<int>(width));
        if (!handle)
            continue;
        bool usable = false;
        try
        {
            const auto network = ConfigureNetwork(handle, resolved, width);
            usable = network && network->GetSimulator();
        }
        catch (...)
        {
            failure = std::current_exception();
        }
        DestroySimpleSimulator(handle);
        if (usable)
            return resolved;
    }
    if (failure)
        std::rethrow_exception(failure);
    throw std::runtime_error("No optimization candidate can hold the register");
}
} // namespace MaestroExecution
