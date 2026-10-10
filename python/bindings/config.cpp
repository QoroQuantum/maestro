#include "core.h"
#include "module.h"

namespace maestro_bindings
{

template <typename T> struct IsOptional : std::false_type
{
};

template <typename T> struct IsOptional<std::optional<T>> : std::true_type
{
};

// A property whose setter validates the whole config before committing, so an
// invalid value leaves the config unchanged.
template <typename T> void BindConfigField(nb::class_<SimulatorConfig> &cls, const char *name, T SimulatorConfig::*member, const char *doc)
{
    cls.def_prop_rw(
        name, [member](const SimulatorConfig &c) -> T { return c.*member; },
        [member](SimulatorConfig &c, T value) {
            SimulatorConfig next = c;
            next.*member = std::move(value);
            next.Validate();
            c = std::move(next);
        },
        nb::for_setter(nb::arg("value").none(IsOptional<T>::value)), doc);
}

// Constructor keywords and __repr__ order.
const char *const kConfigFields[] = {
    "simulator_type",
    "simulation_type",
    "max_bond_dimension",
    "singular_value_threshold",
    "truncation_mode",
    "precision",
    "seed",
    "gpu_device",
    "distributed_options",
    "disable_optimized_swapping",
    "lookahead_depth",
    "mps_sampling",
    "mps_svd_solver",
    "mpo_svd_solver",
    "tensor_network_svd_solver",
    "mpo_kraus_completeness_check",
    "mpo_restore_trace_after_truncation",
    "mpo_hermitize_after_truncation",
    "pp_coefficient_threshold",
    "pp_max_pauli_weight",
    "pp_gates_between_trims",
    "pp_gates_between_deduplications",
    "path_integral_threshold",
    "gate_fusion",
    "enable_causal_cone_reduction",
};

void bind_config(nb::module_ &m)
{
    // --- Enums (must be registered before SimulatorConfig) ---
    nb::enum_<Simulators::SimulatorType>(m, "SimulatorType")
        .value("QCSim", Simulators::SimulatorType::kQCSim)
#ifndef NO_QISKIT_AER
        .value("QiskitAer", Simulators::SimulatorType::kQiskitAer)
        .value("CompositeQiskitAer", Simulators::SimulatorType::kCompositeQiskitAer)
#endif
        .value("CompositeQCSim", Simulators::SimulatorType::kCompositeQCSim)
        .value("Gpu", Simulators::SimulatorType::kGpuSim)
        .value("DistributedGpu", Simulators::SimulatorType::kDistGpuSim)
        .value("DistributedMpiGpu", Simulators::SimulatorType::kDistMpiGpuSim)
        .value("QuestSim", Simulators::SimulatorType::kQuestSim)
        .export_values();

    nb::enum_<Simulators::SimulationType>(m, "SimulationType")
        .value("Statevector", Simulators::SimulationType::kStatevector)
        .value("MatrixProductState", Simulators::SimulationType::kMatrixProductState)
        .value("Stabilizer", Simulators::SimulationType::kStabilizer)
        .value("TensorNetwork", Simulators::SimulationType::kTensorNetwork)
        .value("PauliPropagator", Simulators::SimulationType::kPauliPropagator)
        .value("ExtendedStabilizer", Simulators::SimulationType::kExtendedStabilizer)
        .value("PathIntegral", Simulators::SimulationType::kPathIntegral)
        .value("DensityMatrix", Simulators::SimulationType::kDensityMatrix)
        .value("MatrixProductOperator", Simulators::SimulationType::kMatrixProductOperator)
        .export_values();

    // --- SimulatorConfig ---
    const SimulatorConfig defaults;
    auto config_class = nb::class_<SimulatorConfig>(m, "SimulatorConfig",
                                                    "Configuration for the quantum simulator backend. Create once and "
                                                    "reuse across execute/estimate/statevector calls. Every field is a "
                                                    "keyword argument of the constructor.");
    config_class.def(
        "__init__",
        [](SimulatorConfig *self, Simulators::SimulatorType simulator_type, Simulators::SimulationType simulation_type,
           std::optional<size_t> max_bond_dimension, std::optional<double> singular_value_threshold, std::optional<std::string> truncation_mode,
           std::optional<std::string> precision, std::optional<uint64_t> seed, std::optional<int> gpu_device,
           std::unordered_map<std::string, std::string> distributed_options, bool disable_optimized_swapping, int lookahead_depth, std::string mps_sampling,
           std::optional<std::string> mps_svd_solver, std::optional<std::string> mpo_svd_solver, std::optional<std::string> tensor_network_svd_solver,
           std::optional<std::string> mpo_kraus_completeness_check, bool mpo_restore_trace_after_truncation, bool mpo_hermitize_after_truncation,
           std::optional<double> pp_coefficient_threshold, std::optional<size_t> pp_max_pauli_weight, std::optional<int> pp_gates_between_trims,
           std::optional<int> pp_gates_between_deduplications, std::optional<double> path_integral_threshold, std::optional<bool> gate_fusion,
           bool enable_causal_cone_reduction) {
            SimulatorConfig config;
            config.simulator_type = simulator_type;
            config.simulation_type = simulation_type;
            config.max_bond_dimension = max_bond_dimension;
            config.singular_value_threshold = singular_value_threshold;
            config.truncation_mode = std::move(truncation_mode);
            config.precision = std::move(precision);
            config.seed = seed;
            config.gpu_device = gpu_device;
            config.distributed_options = std::move(distributed_options);
            config.disable_optimized_swapping = disable_optimized_swapping;
            config.lookahead_depth = lookahead_depth;
            config.mps_sampling = std::move(mps_sampling);
            config.mps_svd_solver = std::move(mps_svd_solver);
            config.mpo_svd_solver = std::move(mpo_svd_solver);
            config.tensor_network_svd_solver = std::move(tensor_network_svd_solver);
            config.mpo_kraus_completeness_check = std::move(mpo_kraus_completeness_check);
            config.mpo_restore_trace_after_truncation = mpo_restore_trace_after_truncation;
            config.mpo_hermitize_after_truncation = mpo_hermitize_after_truncation;
            config.pp_coefficient_threshold = pp_coefficient_threshold;
            config.pp_max_pauli_weight = pp_max_pauli_weight;
            config.pp_gates_between_trims = pp_gates_between_trims;
            config.pp_gates_between_deduplications = pp_gates_between_deduplications;
            config.path_integral_threshold = path_integral_threshold;
            config.gate_fusion = gate_fusion;
            config.enable_causal_cone_reduction = enable_causal_cone_reduction;
            config.Validate();
            new (self) SimulatorConfig(std::move(config));
        },
        nb::kw_only(), "simulator_type"_a = defaults.simulator_type, "simulation_type"_a = defaults.simulation_type, "max_bond_dimension"_a = nb::none(),
        "singular_value_threshold"_a = nb::none(), "truncation_mode"_a = nb::none(), "precision"_a = nb::none(), "seed"_a = nb::none(),
        "gpu_device"_a = nb::none(), "distributed_options"_a = defaults.distributed_options,
        "disable_optimized_swapping"_a = defaults.disable_optimized_swapping, "lookahead_depth"_a = defaults.lookahead_depth,
        "mps_sampling"_a = defaults.mps_sampling, "mps_svd_solver"_a = nb::none(), "mpo_svd_solver"_a = nb::none(), "tensor_network_svd_solver"_a = nb::none(),
        "mpo_kraus_completeness_check"_a = nb::none(), "mpo_restore_trace_after_truncation"_a = defaults.mpo_restore_trace_after_truncation,
        "mpo_hermitize_after_truncation"_a = defaults.mpo_hermitize_after_truncation, "pp_coefficient_threshold"_a = nb::none(),
        "pp_max_pauli_weight"_a = nb::none(), "pp_gates_between_trims"_a = nb::none(), "pp_gates_between_deduplications"_a = nb::none(),
        "path_integral_threshold"_a = nb::none(), "gate_fusion"_a = nb::none(), "enable_causal_cone_reduction"_a = defaults.enable_causal_cone_reduction);

    BindConfigField(config_class, "gate_fusion", &SimulatorConfig::gate_fusion,
                    "Fuse compatible gates on supported backends. None (the "
                    "default) uses each backend's default: on, except for CPU "
                    "statevectors below 11 qubits and CPU density matrices "
                    "below 5, where fusion costs more than it saves. True or "
                    "False force it. Truncated MPS/MPO results can change.");
    BindConfigField(config_class, "simulator_type", &SimulatorConfig::simulator_type, "Simulator backend, a SimulatorType.");
    BindConfigField(config_class, "simulation_type", &SimulatorConfig::simulation_type, "Simulation method, a SimulationType.");
    BindConfigField(config_class, "max_bond_dimension", &SimulatorConfig::max_bond_dimension,
                    "Largest bond dimension kept when truncating MPS, MPO and GPU "
                    "tensor-network states. None uses the backend default (128 for GPU MPS "
                    "and MPO).");
    BindConfigField(config_class, "singular_value_threshold", &SimulatorConfig::singular_value_threshold,
                    "SVD truncation threshold for MPS, MPO and GPU tensor-network states, "
                    "read according to truncation_mode. Under 'relative_max' it is a ratio "
                    "of singular values; under 'discarded_weight' it bounds the discarded "
                    "normalised squared weight, so the same number truncates much harder "
                    "(1e-8 drops singular values up to about 1e-4 of the spectrum's norm). "
                    "None uses the backend default.");
    BindConfigField(config_class, "truncation_mode", &SimulatorConfig::truncation_mode,
                    "'relative_max' drops singular values below singular_value_threshold "
                    "times the largest; 'discarded_weight' (the default on every backend) "
                    "drops the smallest until their cumulative normalised squared weight "
                    "reaches the threshold. Only QCSim and the GPU "
                    "backend support 'relative_max'; Aer raises if it is requested.");
    BindConfigField(config_class, "precision", &SimulatorConfig::precision,
                    "'single' or 'double' floating point for Qiskit Aer and the GPU "
                    "simulators. None keeps each backend's default. Other backends ignore "
                    "it; QCSim always computes in double precision.");
    BindConfigField(config_class, "seed", &SimulatorConfig::seed,
                    "Seed for simulation randomness: measurement, readout and reset. The "
                    "noisy functions also seed their injected noise from its low 32 bits "
                    "when their noise_seed is unset. None seeds from system entropy.");
    BindConfigField(config_class, "gpu_device", &SimulatorConfig::gpu_device, "CUDA-visible device ordinal, or None to use the default.");
    BindConfigField(config_class, "distributed_options", &SimulatorConfig::distributed_options,
                    "Distribution settings passed to Configure before allocation. Keys "
                    "start with 'distributed_' or 'mpi_'. Defaults: first global qubits, "
                    "automatic Ex execution, visible GPUs. MPI calls must match across "
                    "ranks; mpi_communicator is mpi4py Comm.py2f().");
    BindConfigField(config_class, "disable_optimized_swapping", &SimulatorConfig::disable_optimized_swapping,
                    "Turn off swap-cost optimisation and the initial qubit-map "
                    "optimisation.");
    BindConfigField(config_class, "lookahead_depth", &SimulatorConfig::lookahead_depth,
                    "Lookahead depth for swap optimisation; -1 uses Maestro's "
                    "default.");
    BindConfigField(config_class, "mps_sampling", &SimulatorConfig::mps_sampling,
                    "How QCSim and Aer MPS simulations sample shots; the GPU MPS simulator "
                    "ignores it. 'probabilities' (the default) samples without collapsing "
                    "the state; 'apply_measure' measures, collapses and restores it for "
                    "every shot. Both draw from the same distribution, but consume the "
                    "random stream differently, so one seed gives different counts.");
    BindConfigField(config_class, "mps_svd_solver", &SimulatorConfig::mps_svd_solver,
                    "GPU SVD solver for MPS truncation: 'gesvd', 'gesvdj' (Jacobi), "
                    "'gesvdp' (polar) or 'gesvdr' (randomised). None keeps the GPU "
                    "library's default.");
    BindConfigField(config_class, "mpo_svd_solver", &SimulatorConfig::mpo_svd_solver,
                    "GPU SVD solver for MPO truncation; the choices of "
                    "mps_svd_solver.");
    BindConfigField(config_class, "tensor_network_svd_solver", &SimulatorConfig::tensor_network_svd_solver,
                    "GPU SVD solver for tensor-network truncation; the choices "
                    "of mps_svd_solver.");
    BindConfigField(config_class, "mpo_kraus_completeness_check", &SimulatorConfig::mpo_kraus_completeness_check,
                    "How the MPO simulator treats Kraus operators that do not sum to the "
                    "identity: 'ignore', 'warn' or 'strict' (raise). None uses the "
                    "default.");
    BindConfigField(config_class, "mpo_restore_trace_after_truncation", &SimulatorConfig::mpo_restore_trace_after_truncation,
                    "Rescale the CPU MPO to unit trace after each truncation.");
    BindConfigField(config_class, "mpo_hermitize_after_truncation", &SimulatorConfig::mpo_hermitize_after_truncation,
                    "Make the CPU MPO Hermitian again after each truncation.");
    BindConfigField(config_class, "pp_coefficient_threshold", &SimulatorConfig::pp_coefficient_threshold,
                    "Pauli propagation: truncation passes drop strings whose |coefficient| "
                    "is at most this value.");
    BindConfigField(config_class, "pp_max_pauli_weight", &SimulatorConfig::pp_max_pauli_weight,
                    "Pauli propagation: truncation passes drop strings acting on more "
                    "qubits than this; a value at or above the qubit count keeps them all.");
    BindConfigField(config_class, "pp_gates_between_trims", &SimulatorConfig::pp_gates_between_trims,
                    "Pauli propagation: apply both thresholds every this many operations, "
                    "counting each primitive operation a gate decomposes into. Must be "
                    "positive.");
    BindConfigField(config_class, "pp_gates_between_deduplications", &SimulatorConfig::pp_gates_between_deduplications,
                    "Pauli propagation: every this many operations, merge repeated strings "
                    "and then apply both thresholds; takes precedence over a trim due on "
                    "the same operation. Must be positive. Unset, PauliPropagator "
                    "simulations use 10.");
    BindConfigField(config_class, "path_integral_threshold", &SimulatorConfig::path_integral_threshold,
                    "Trim threshold for PathIntegral simulation; None disables "
                    "trimming.");
    BindConfigField(config_class, "enable_causal_cone_reduction", &SimulatorConfig::enable_causal_cone_reduction,
                    "Reduce observable causal cones before allocation; distributed GPU backends retain full execution.");
    nb::list config_fields;
    for (const char *name : kConfigFields)
        config_fields.append(name);
    config_class.attr("_fields") = nb::tuple(config_fields);
    config_class.def("__repr__", [](nb::handle self) {
        std::string out = "SimulatorConfig(";
        for (size_t i = 0; i < std::size(kConfigFields); ++i)
        {
            const std::string name = kConfigFields[i];
            nb::object value = nb::getattr(self, name.c_str());
            const bool is_enum = name == "simulator_type" || name == "simulation_type";
            out += (i ? ", " : "") + name + "=" + nb::cast<std::string>(is_enum ? nb::str(value) : nb::repr(value));
        }
        return out + ")";
    });
}

} // namespace maestro_bindings
