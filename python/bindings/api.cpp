#include "Simulators/PathIntegral/PathIntegralSimulator.h"
#include "core.h"
#include "module.h"
#include "qasm/QasmCirc.h"

namespace maestro_bindings
{

void bind_api(nb::module_ &m)
{
    // --- QASM Tools ---
    nb::class_<qasm::QasmToCirc<double>>(m, "QasmToCirc")
        .def(nb::init<>())
        .def(
            "parse_and_translate",
            [](qasm::QasmToCirc<double> &self, const std::string &qasm_str, const std::unordered_map<std::string, double> &params) {
                auto circuit = self.ParseAndTranslateWithParams(qasm_str, params);
                if (self.Failed() || !circuit)
                {
                    throw nb::value_error(("Failed to parse QASM string: " + self.GetErrorMessage()).c_str());
                }
                return circuit;
            },
            "qasm_str"_a, "params"_a = std::unordered_map<std::string, double>{},
            "Parse an OpenQASM string and translate it into a circuit.\n\n"
            "Args:\n"
            "    qasm_str: The QASM program text.\n"
            "    params: Optional dict binding QASM3 `input` declarations "
            "(str -> float) at parse time.")
        .def("failed", &qasm::QasmToCirc<double>::Failed)
        .def("get_error_message", &qasm::QasmToCirc<double>::GetErrorMessage)
        .def("get_inputs", &qasm::QasmToCirc<double>::GetInputs,
             "The names declared by QASM3 `input` statements, in declaration "
             "order.");

    // --- Module Level Convenience Functions ---

    // 1. simple_execute (Overloaded)
    // Variant A: Circuit Object
    m.def("simple_execute", &execute_core, "circuit"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024);

    // Variant B: QASM String
    m.def(
        "simple_execute",
        [](const std::string &qasm, const SimulatorConfig &config, int shots) {
            qasm::QasmToCirc<> parser;
            auto circuit = parser.ParseAndTranslate(qasm);
            if (parser.Failed() || !circuit)
            {
                // IMPROVEMENT: Throw error instead of silent failure
                throw nb::value_error("Failed to parse QASM string.");
            }
            return execute_core(circuit, config, shots);
        },
        "qasm_circuit"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024);

    // 2. simple_estimate (Overloaded)
    // Variant A: Circuit Object
    m.def(
        "simple_estimate",
        [](std::shared_ptr<Circuits::Circuit<double>> circuit, const nb::object &obs, const SimulatorConfig &config) {
            return estimate_core(circuit, ParseObservables(obs), config);
        },
        "circuit"_a, "observables"_a, "config"_a = SimulatorConfig{});

    // Variant B: QASM String
    m.def(
        "simple_estimate",
        [](const std::string &qasm, const nb::object &obs, const SimulatorConfig &config) {
            qasm::QasmToCirc<> parser;
            auto circuit = parser.ParseAndTranslate(qasm);
            if (parser.Failed() || !circuit)
            {
                throw nb::value_error("Failed to parse QASM string.");
            }
            return estimate_core(circuit, ParseObservables(obs), config);
        },
        "qasm_circuit"_a, "observables"_a, "config"_a = SimulatorConfig{});

    // 3. incremental_evolve
    // Runs time evolution incrementally: executes init_circuit once, then
    // applies trotter_step repeatedly, computing expectation values at
    // specified measurement steps. Uses the simulator's persistent state
    // to avoid re-simulating from scratch at each measurement point.
    m.def(
        "incremental_evolve",
        [](std::shared_ptr<Circuits::Circuit<double>> init_circuit, std::shared_ptr<Circuits::Circuit<double>> trotter_step,
           const std::vector<int> &measure_at_steps, const nb::object &observables, const SimulatorConfig &config) {
            return incremental_evolve_core(init_circuit, trotter_step, measure_at_steps, ParseObservables(observables), config);
        },
        "init_circuit"_a, "trotter_step"_a, "measure_at_steps"_a, "observables"_a, "config"_a = SimulatorConfig{},
        "Incremental time evolution with persistent simulator state.\n\n"
        "Creates a single simulator, executes init_circuit once, then applies\n"
        "trotter_step incrementally. At each step in measure_at_steps, computes\n"
        "expectation values for the given observables without re-simulating\n"
        "from scratch. Cost: O(total_steps) instead of O(sum of step "
        "indices).\n\n"
        "Args:\n"
        "    init_circuit: Circuit preparing the initial state.\n"
        "    trotter_step: Circuit for one Trotter step.\n"
        "    measure_at_steps: List of step indices at which to measure.\n"
        "    observables: Pauli strings to measure (list or ';'-separated).\n"
        "    config: SimulatorConfig for backend selection.\n\n"
        "Returns:\n"
        "    dict with 'expectation_values' (list of lists), 'steps', "
        "'time_taken'.");

    // --- QuEST Library Management ---
    m.def(
        "init_quest", []() { return Simulators::SimulatorsFactory::InitQuestLibrary(); }, "Initialize the QuEST simulation library. Returns True on success.");

    m.def(
        "is_quest_available", []() { return Simulators::SimulatorsFactory::IsQuestLibraryAvailable(); },
        "Check whether the QuEST simulation library is loaded and available.");

#ifdef __linux__
    m.def(
        "finalize_distributed_mpi_gpu", []() { Simulators::SimulatorsFactory::FinalizeDistributedMpiGpuBackend(); },
        "Terminal shutdown after all MPI GPU states are destroyed, before "
        "MPI.Finalize().");
    m.def(
        "is_distributed_gpu_available", []() { return Simulators::SimulatorsFactory::IsDistributedGpuAvailable(); },
        "Non-throwing probe of the local distributed plugin and devices, without "
        "license admission or state allocation. Returns False for missing or "
        "incompatible plugins.");
#endif
    // --- GPU Library Management ---
    m.def(
        "init_gpu", []() { return Simulators::SimulatorsFactory::InitGpuLibrary(); }, "Initialize the GPU simulation library. Returns True on success.");

    m.def(
        "is_gpu_available", []() { return Simulators::SimulatorsFactory::IsGpuLibraryAvailable(); },
        "Check availability of the default GPU, initializing it lazily.");

    m.def(
        "select_gpu_device", [](int deviceId) { Simulators::SimulatorsFactory::SelectGpuDevice(deviceId); },
        "Select the default CUDA device for future simulators and init_gpu(). "
        "SimulatorConfig.gpu_device overrides this default; existing simulators "
        "keep their device.");

    m.def(
        "get_gpu_device_count", []() { return Simulators::SimulatorsFactory::GetGpuDeviceCount(); },
        "Number of CUDA-capable devices visible to the process, or 0 if the "
        "GPU library cannot be loaded or none are visible; -1 on CUDA discovery "
        "errors. "
        "Does not initialize a simulator.");

    // --- Probability / Amplitude Access ---
    m.def(
        "get_probabilities",
        [](std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config) -> nb::list {
            const auto amplitudes = statevector_core(circuit, config);
            nb::list probs;
            for (const auto &amp : amplitudes)
                probs.append(std::norm(amp));
            return probs;
        },
        "circuit"_a, "config"_a = SimulatorConfig{}, "Get the full probability distribution after executing a circuit.");

    m.def(
        "get_statevector", [](std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config) { return statevector_core(circuit, config); },
        "circuit"_a, "config"_a = SimulatorConfig{},
        "Get the full statevector (complex amplitudes) after executing a "
        "circuit.");

    m.def(
        "mirror_fidelity",
        [](std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots, bool full_amplitude) {
            return mirror_fidelity_core(circuit, config, shots, full_amplitude);
        },
        "circuit"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "full_amplitude"_a = false,
        "Compute mirror fidelity: run a circuit forward then its adjoint in "
        "reverse, returning P(|0...0>). Uses shot-based sampling by "
        "default. Set full_amplitude=True for exact statevector "
        "computation (small circuits only).");

    m.def(
        "inner_product",
        [](const std::shared_ptr<Circuits::Circuit<double>> &circuit_1, const std::shared_ptr<Circuits::Circuit<double>> &circuit_2,
           const SimulatorConfig &config) { return inner_product_core(circuit_1, circuit_2, config); },
        "circuit_1"_a, "circuit_2"_a, "config"_a = SimulatorConfig{},
        "Compute the inner product <psi_1|psi_2> = <0|U1^dag U2|0> between "
        "two circuits' output states, using ProjectOnZero.");

    // =========================================================================
    // Path Integral: Single-State Probability
    // =========================================================================

    m.def(
        "state_probability",
        [](std::shared_ptr<Circuits::Circuit<double>> circuit, const std::string &target_state) -> nb::dict {
            if (!circuit)
                throw nb::value_error("Circuit is null.");
            if (target_state.empty())
                throw nb::value_error("target_state must be a non-empty bitstring.");

            // Convert bitstring to vector<bool>
            std::vector<bool> end_state(target_state.size());
            for (size_t i = 0; i < target_state.size(); ++i)
            {
                if (target_state[i] == '1')
                    end_state[i] = true;
                else if (target_state[i] == '0')
                    end_state[i] = false;
                else
                    throw nb::value_error("target_state must contain only '0' and '1' characters.");
            }

            Simulators::PathIntegralSimulator sim;
            sim.SetStartZeroState(target_state.size());

            auto start = std::chrono::high_resolution_clock::now();
            bool ok;
            {
                nb::gil_scoped_release release;
                ok = sim.SetCircuit(circuit);
            }
            if (!ok)
                throw std::runtime_error("Circuit contains operations not supported by the path "
                                         "integral simulator.");

            auto amplitude = sim.AmplitudeFromZero(end_state);
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict result;
            result["amplitude"] = amplitude;
            result["probability"] = std::norm(amplitude);
            result["target_state"] = target_state;
            result["time_taken"] = std::chrono::duration<double>(end - start).count();
            return result;
        },
        "circuit"_a, "target_state"_a,
        "Compute the probability of a specific output state using the Pauli "
        "path integral simulator.\n\n"
        "This is the path integral's key advantage: it computes a single "
        "amplitude <target_state|U|0...0> without building the full "
        "statevector, making it efficient for large circuits with few "
        "branching gates.\n\n"
        "Args:\n"
        "    circuit: A QuantumCircuit (no measurements needed).\n"
        "    target_state: A bitstring like '10001001' (qubit 0 is leftmost).\n\n"
        "Returns:\n"
        "    dict with 'probability', 'amplitude', 'target_state', "
        "'time_taken'.");

    // QASM variant
    m.def(
        "state_probability",
        [](const std::string &qasm, const std::string &target_state) -> nb::dict {
            if (target_state.empty())
                throw nb::value_error("target_state must be a non-empty bitstring.");

            qasm::QasmToCirc<> parser;
            auto circuit = parser.ParseAndTranslate(qasm);
            if (parser.Failed() || !circuit)
                throw nb::value_error("Failed to parse QASM string.");

            std::vector<bool> end_state(target_state.size());
            for (size_t i = 0; i < target_state.size(); ++i)
            {
                if (target_state[i] == '1')
                    end_state[i] = true;
                else if (target_state[i] == '0')
                    end_state[i] = false;
                else
                    throw nb::value_error("target_state must contain only '0' and '1' characters.");
            }

            Simulators::PathIntegralSimulator sim;
            sim.SetStartZeroState(target_state.size());

            auto start = std::chrono::high_resolution_clock::now();
            bool ok;
            {
                nb::gil_scoped_release release;
                ok = sim.SetCircuit(circuit);
            }
            if (!ok)
                throw std::runtime_error("Circuit contains operations not supported by the path "
                                         "integral simulator.");

            auto amplitude = sim.AmplitudeFromZero(end_state);
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict result;
            result["amplitude"] = amplitude;
            result["probability"] = std::norm(amplitude);
            result["target_state"] = target_state;
            result["time_taken"] = std::chrono::duration<double>(end - start).count();
            return result;
        },
        "qasm_circuit"_a, "target_state"_a,
        "Compute the probability of a specific output state from a QASM "
        "circuit using the Pauli path integral simulator.");
}

} // namespace maestro_bindings
