#include "core.h"
#include "module.h"
#include "qasm/QasmCirc.h"

namespace maestro_bindings
{

void bind_noise(nb::module_ &m)
{
    // =========================================================================
    // Noise Modeling
    // =========================================================================

    nb::class_<noise::NoiseModel>(m, "NoiseModel")
        .def(nb::init<>(), "Create an empty noise model.")
        .def("set_qubit_noise", &noise::NoiseModel::set_qubit_noise, "qubit"_a, "px"_a, "py"_a, "pz"_a,
             "Add Pauli channel: Λ(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + "
             "pz·ZρZ")
        .def("set_depolarizing", &noise::NoiseModel::set_depolarizing, "qubit"_a, "p"_a, "Add symmetric depolarizing noise (px=py=pz=p/3).")
        .def("set_dephasing", &noise::NoiseModel::set_dephasing, "qubit"_a, "p"_a, "Add pure dephasing (Z) noise.")
        .def("set_bit_flip", &noise::NoiseModel::set_bit_flip, "qubit"_a, "p"_a, "Add bit-flip (X) noise.")
        .def("set_all_depolarizing", &noise::NoiseModel::set_all_depolarizing, "num_qubits"_a, "p"_a,
             "Add uniform depolarizing noise to all qubits [0, num_qubits).")
        .def("set_all_dephasing", &noise::NoiseModel::set_all_dephasing, "num_qubits"_a, "p"_a, "Add uniform dephasing noise to all qubits [0, num_qubits).")
        .def("compute_damping", &noise::NoiseModel::compute_damping, "pauli_string"_a,
             "Damping factor for a Pauli string observable from ONE "
             "application of the configured Pauli channels.\n\n"
             "Exact only when the channel acts once, immediately before "
             "measurement; for a circuit with noise after every gate it "
             "underestimates the noise. Only the all-gates Pauli layer "
             "contributes.")
        // ── Coherent noise setters ──
        .def("set_coherent_rotation", &noise::NoiseModel::set_coherent_rotation, "qubit"_a, "rx"_a, "ry"_a, "rz"_a,
             "Set per-qubit coherent noise as rotation angles (radians). "
             "After every gate on this qubit, Rx(±rx), Ry(±ry), Rz(±rz) "
             "rotations are applied with random ± signs.")
        .def("set_coherent_depolarizing", &noise::NoiseModel::set_coherent_depolarizing, "qubit"_a, "p"_a,
             "Set coherent noise from a depolarizing probability. "
             "Converts p to Rz angle ε = 2·arcsin(√p), matching the "
             "infidelity of DEPOLARIZE1(p).")
        .def("set_coherent_dephasing", &noise::NoiseModel::set_coherent_dephasing, "qubit"_a, "p"_a,
             "Set coherent dephasing noise: Rz rotation from probability p.")
        .def("set_coherent_bit_flip", &noise::NoiseModel::set_coherent_bit_flip, "qubit"_a, "p"_a,
             "Set coherent bit-flip noise: Rx rotation from probability p.")
        .def("set_all_coherent_depolarizing", &noise::NoiseModel::set_all_coherent_depolarizing, "num_qubits"_a, "p"_a,
             "Set uniform coherent depolarizing noise on qubits [0, "
             "num_qubits).")
        .def("set_all_coherent_dephasing", &noise::NoiseModel::set_all_coherent_dephasing, "num_qubits"_a, "p"_a,
             "Set uniform coherent dephasing on qubits [0, num_qubits).")
        .def("set_coherent_strength", &noise::NoiseModel::set_coherent_strength, "num_qubits"_a, "p"_a,
             "Convenience: set uniform coherent noise strength on all qubits. "
             "Equivalent to set_all_coherent_depolarizing(n, p).")
        .def("has_coherent", &noise::NoiseModel::has_coherent, "Return True if any coherent noise parameters have been set.")
        // ── Correlated (time-correlated) noise ──
        .def("set_correlated_ar1", &noise::NoiseModel::set_correlated_ar1, "qubit"_a, "phi"_a, "sigma_eta"_a, "after_1q"_a = true, "after_2q"_a = true,
             "stationary_init"_a = true,
             "Set AR(1) correlated dephasing on a qubit.\n\n"
             "After every gate, Rz(y[k]) is injected where:\n"
             "  y[k] = phi * y[k-1] + eta[k],  eta ~ N(0, sigma_eta^2)\n\n"
             "Args:\n"
             "    qubit: Qubit index.\n"
             "    phi: AR(1) autoregressive coefficient.\n"
             "    sigma_eta: Driving noise standard deviation.\n"
             "    after_1q: If True (default), inject after 1Q gates.\n"
             "    after_2q: If True (default), inject after 2Q gates.\n"
             "    stationary_init: If True (default), sample step 0 from "
             "stationary equilibrium.\n\n"
             "Example: nm.set_correlated_ar1(0, phi=0.135, sigma_eta=2.35e-3)")
        .def("set_correlated_ou", &noise::NoiseModel::set_correlated_ou, "qubit"_a, "sigma"_a, "alpha"_a, "gate_time"_a, "after_1q"_a = true,
             "after_2q"_a = true, "stationary_init"_a = true,
             "Set correlated noise from Ornstein-Uhlenbeck parameters.\n\n"
             "OU: dX = -theta*X*dt + sigma*dW, discretized as AR(1).\n"
             "  theta = 1/(alpha * gate_time)\n"
             "  phi = exp(-theta * gate_time)\n"
             "  sigma_eta^2 = (sigma^2 / 2*theta) * (1 - phi^2)\n\n"
             "Args:\n"
             "    qubit: Qubit index.\n"
             "    sigma: OU diffusion coefficient (noise strength).\n"
             "    alpha: Correlation time in gate-time units.\n"
             "    gate_time: Gate duration in seconds.\n"
             "    after_1q: If True (default), inject after 1Q gates.\n"
             "    after_2q: If True (default), inject after 2Q gates.\n"
             "    stationary_init: If True (default), sample step 0 from "
             "stationary equilibrium.\n\n"
             "Example: nm.set_correlated_ou(0, sigma=15.0, alpha=0.5, "
             "gate_time=100e-9)")
        .def("set_correlated_ou_band", &noise::NoiseModel::set_correlated_ou_band, "qubit"_a, "sigma"_a, "alpha"_a, "gate_time"_a, "after_1q"_a = true,
             "after_2q"_a = true, "stationary_init"_a = true,
             "Append an OU fluctuator band to a qubit's band list, keeping the "
             "bands already set on it.")
        .def("set_multi_correlated_ou", &noise::NoiseModel::set_multi_correlated_ou, "qubit"_a, "bands"_a, "gate_time"_a, "after_1q"_a = true,
             "after_2q"_a = true, "stationary_init"_a = true,
             "Batch multi-OU setter: clears existing bands and populates from a "
             "list of (sigma, alpha) pairs.")
        .def("set_all_multi_correlated_ou", &noise::NoiseModel::set_all_multi_correlated_ou, "num_qubits"_a, "bands"_a, "gate_time"_a, "after_1q"_a = true,
             "after_2q"_a = true, "stationary_init"_a = true, "Uniform multi-OU setter across qubits 0..num_qubits-1.")
        .def("set_1_over_f_noise", &noise::NoiseModel::set_1_over_f_noise, "qubit"_a, "total_power"_a, "f_min"_a, "f_max"_a, "num_bands"_a, "gate_time"_a,
             "after_1q"_a = true, "after_2q"_a = true, "stationary_init"_a = true,
             "Synthesize 1/f noise spectrum via logarithmically spaced OU "
             "fluctuator bands.")
        .def("set_all_correlated_ou", &noise::NoiseModel::set_all_correlated_ou, "num_qubits"_a, "sigma"_a, "alpha"_a, "gate_time"_a, "after_1q"_a = true,
             "after_2q"_a = true, "stationary_init"_a = true,
             "Set identical OU correlated noise on qubits [0, num_qubits).\n\n"
             "Example: nm.set_all_correlated_ou(20, sigma=15.0, alpha=0.5, "
             "gate_time=100e-9)")
        .def("set_all_correlated_from_power", &noise::NoiseModel::set_all_correlated_from_power, "num_qubits"_a, "power"_a, "alpha"_a, "gate_time"_a,
             "after_1q"_a = true, "after_2q"_a = true,
             "Set correlated noise from total noise power.\n\n"
             "P_tot = N * sigma^2 * pi * alpha * gate_time\n"
             "Derives sigma from P_tot and sets OU noise on all qubits.\n\n"
             "Args:\n"
             "    num_qubits: Number of qubits N.\n"
             "    power: Total noise power P_tot.\n"
             "    alpha: Correlation time in gate-time units.\n"
             "    gate_time: Gate duration in seconds.\n"
             "    after_1q: If True (default), inject after 1Q gates too.\n"
             "    after_2q: If True (default), inject after 2Q gates too.\n\n"
             "Example: nm.set_all_correlated_from_power(20, power=1e-3, "
             "alpha=0.5, gate_time=100e-9)")
        .def("has_correlated", &noise::NoiseModel::has_correlated, "Return True if any correlated noise parameters have been set.")
        // ── Idle noise ──
        .def("set_idle_noise", &noise::NoiseModel::set_idle_noise, "qubit"_a, "t1"_a, "t2"_a, "excited_population"_a = 0.0, "detuning_hz"_a = 0.0,
             "Configure idle dephasing/relaxation and detuning for delay "
             "instructions.\n\n"
             "Args:\n"
             "    qubit: Qubit index.\n"
             "    t1: T1 relaxation time in seconds.\n"
             "    t2: T2 dephasing time in seconds (T2 <= 2*T1).\n"
             "    excited_population: Equilibrium |1> state population (default "
             "0.0).\n"
             "    detuning_hz: Coherent detuning frequency in Hz (default 0.0).\n")
        .def("has_idle_noise", &noise::NoiseModel::has_idle_noise, "Return True if any idle noise parameters have been set.")
        // ── T1 amplitude damping ──
        .def("set_t1", &noise::NoiseModel::set_t1, "qubit"_a, "gamma"_a,
             "Set per-gate T1 decay probability. Density-matrix/MPO execution "
             "uses exact amplitude damping; pure-state execution retains the "
             "legacy sampled-reset approximation.")
        .def("set_all_t1", &noise::NoiseModel::set_all_t1, "num_qubits"_a, "gamma"_a, "Set uniform T1 decay probability on qubits [0, num_qubits).")
        .def("set_t1_from_time", &noise::NoiseModel::set_t1_from_time, "qubit"_a, "gate_time_s"_a, "t1_time_s"_a,
             "Set T1 from physical time constants. "
             "gamma = 1 - exp(-gate_time / T1).\n\n"
             "Example: nm.set_t1_from_time(0, gate_time_s=30e-9, "
             "t1_time_s=100e-6)")
        .def("has_t1", &noise::NoiseModel::has_t1, "Return True if any T1 parameters have been set.")
        // ── Additional exact CPTP channels (density matrix / MPO) ──
        .def("set_phase_damping", &noise::NoiseModel::set_phase_damping, "qubit"_a, "gamma"_a,
             "Set phase damping with coherence multiplier sqrt(1-gamma).\n\n"
             "Phase damping and the stochastic phase flip are the same "
             "channel (sqrt(1-gamma) = 1-2p), so this is realized exactly on "
             "every backend, not only density-matrix/MPO.")
        .def("set_phase_damping_from_time", &noise::NoiseModel::set_phase_damping_from_time, "qubit"_a, "gate_time_s"_a, "t_phi_s"_a,
             "Set pure phase damping so coherence decays as "
             "exp(-gate_time/T_phi). Realized exactly on every backend.")
        .def("set_generalized_amplitude_damping", &noise::NoiseModel::set_generalized_amplitude_damping, "qubit"_a, "gamma"_a, "excited_population"_a,
             "Set finite-temperature generalized amplitude damping after each "
             "gate on a qubit. Requires an exact density-matrix or MPO backend.")
        .def("set_thermal_relaxation", &noise::NoiseModel::set_thermal_relaxation, "qubit"_a, "gate_time_s"_a, "t1_s"_a, "t2_s"_a, "excited_population"_a = 0.0,
             "Set hardware-style T1/T2 thermal relaxation after each gate.\n\n"
             "This is the preferred way to specify decoherence. Because T1 and "
             "T2 are given together, every backend reproduces the same "
             "coherence decay exp(-gate_time/T2): density-matrix/MPO use the "
             "exact CPTP channel, pure-state/MPS use the equivalent reset+Z "
             "mixture. Calling set_t1() and set_dephasing() separately cannot "
             "achieve that -- the phase-flip probability that is correct for "
             "the sampled reset model under-dephases by exp(t/2*T1) per gate "
             "on the exact amplitude-damping path.\n\n"
             "The physical constraint T2 <= 2*T1 is enforced.")
        .def("set_thermal_relaxation_2q", &noise::NoiseModel::set_thermal_relaxation_2q, "qubit"_a, "gate_time_s"_a, "t1_s"_a, "t2_s"_a,
             "excited_population"_a = 0.0,
             "Set T1/T2 thermal relaxation applied only after two-qubit "
             "gates, using the (longer) 2Q gate duration. When set, 2Q gates "
             "use this instead of the 'all gates' relaxation.")
        .def("has_thermal_relaxation", &noise::NoiseModel::has_thermal_relaxation, "Return True if any T1/T2 thermal relaxation has been set.")
        .def("set_correlated_phase_flip", &noise::NoiseModel::set_correlated_phase_flip, "q1"_a, "q2"_a, "probability"_a, "correlation"_a = 1.0,
             "Set a correlated two-qubit phase-flip channel after gates on the "
             "pair. correlation=0 is independent; correlation=1 is II/ZZ.")
        .def(
            "set_kraus_channel",
            [](noise::NoiseModel &self, const Types::qubits_vector &targets, const std::vector<std::vector<std::vector<std::complex<double>>>> &operators) {
                Simulators::QuantumChannel::KrausOperators kraus;
                kraus.reserve(operators.size());
                for (const auto &operatorRows : operators)
                {
                    if (operatorRows.empty() || operatorRows.front().empty())
                        throw nb::value_error("Kraus operators must be nonempty matrices.");
                    const size_t rows = operatorRows.size();
                    const size_t columns = operatorRows.front().size();
                    Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns));
                    for (size_t row = 0; row < rows; ++row)
                    {
                        if (operatorRows[row].size() != columns)
                            throw nb::value_error("Kraus operator rows must all have equal length.");
                        for (size_t column = 0; column < columns; ++column)
                            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = operatorRows[row][column];
                    }
                    kraus.emplace_back(std::move(matrix));
                }
                self.set_kraus_channel(targets, kraus);
            },
            "targets"_a, "kraus_operators"_a,
            "Attach an arbitrary one- or two-qubit CPTP Kraus channel after "
            "gates on the same targets. Matrices are nested row-major lists; "
            "completeness (sum_k E_k^dag E_k = I) is validated.\n\n"
            "BASIS ORDER for two-qubit operators: targets[0] is the LEAST "
            "significant bit of the 4x4 matrix index, targets[1] the most "
            "significant. So an operator acting as A on targets[0] and B on "
            "targets[1] must be supplied as the Kronecker product B (x) A. "
            "Getting this backwards silently transposes the channel onto the "
            "wrong qubit.\n\n"
            "Example (X on targets[0], identity on targets[1]):\n"
            "    nm.set_kraus_channel([0, 2], [[[0,1,0,0],[1,0,0,0],"
            "[0,0,0,1],[0,0,1,0]]])")
        .def("has_additional_quantum_channels", &noise::NoiseModel::has_additional_quantum_channels,
             "Return True if a channel that ONLY an exact density-matrix/MPO "
             "backend can run is configured: generalized amplitude damping, "
             "correlated phase flips or custom Kraus maps. Phase damping and "
             "thermal relaxation are excluded -- both have an exact or "
             "well-defined stochastic realization on sampled backends.")
        .def("has_thermal_in_sampled_overdephasing_regime", &noise::NoiseModel::has_thermal_in_sampled_overdephasing_regime,
             "Return True if any thermal-relaxation layer has T2 > T1, where "
             "the sampled reset+Z mixture over-dephases. Use density-matrix "
             "or MPO execution in that regime.")
        .def("requires_exact_quantum_channels", &noise::NoiseModel::requires_exact_quantum_channels,
             "Return True if sampled injection cannot faithfully realize this "
             "model "
             "(exact-only Kraus maps, or T2 > T1 requiring approximation on "
             "sampled paths).")
        .def("compute_damping_covers_model", &noise::NoiseModel::compute_damping_covers_model,
             "Return True iff compute_damping() captures every layer that "
             "affects Pauli expectations. False for thermal, T1, gate-type "
             "Pauli, 2Q depolarizing, coherent, correlated and crosstalk.")
        // ── T1 gate-type-specific overrides ──
        .def("set_t1_2q", &noise::NoiseModel::set_t1_2q, "qubit"_a, "gamma"_a,
             "Set T1 decay probability applied only after two-qubit gates. "
             "When set, 2Q gates use this gamma instead of the 'all gates' value.")
        .def("set_t1_2q_from_time", &noise::NoiseModel::set_t1_2q_from_time, "qubit"_a, "gate_time_s"_a, "t1_time_s"_a,
             "Set T1 for 2Q gates from physical time constants. "
             "gamma = 1 - exp(-gate_time / T1).\n\n"
             "Example: nm.set_t1_2q_from_time(0, gate_time_s=40e-9, "
             "t1_time_s=100e-6)")
        .def("get_t1_2q", &noise::NoiseModel::get_t1_2q, "qubit"_a,
             "Get T1 decay probability for 2Q gates (falls back to get_t1 "
             "if not set).")
        .def("get_t1_for_gate", &noise::NoiseModel::get_t1_for_gate, "qubit"_a, "is_2q"_a,
             "Get gate-type-aware T1: returns t1_2q if is_2q and set, "
             "else t1.")
        // ── Crosstalk ──
        .def("set_crosstalk", &noise::NoiseModel::set_crosstalk, "q1"_a, "q2"_a, "strength"_a,
             "Set spectator-Z crosstalk between two qubits. After a gate on "
             "q1, Rz(strength) is applied on q2, and vice versa. This is not "
             "a genuine two-qubit ZZ interaction.")
        .def("has_crosstalk", &noise::NoiseModel::has_crosstalk, "Return True if any crosstalk couplings have been set.")
        // ── Readout error ──
        .def("set_readout_error", &noise::NoiseModel::set_readout_error, "qubit"_a, "p_meas1_prep0"_a, "p_meas0_prep1"_a,
             "Set asymmetric readout error on a qubit.\n\n"
             "Applied when a measurement of this qubit writes its classical "
             "bit, whatever bit index it targets, so mid-circuit and repeated "
             "measurements are covered and classically-conditioned operations "
             "downstream observe the noisy outcome.\n\n"
             "Args:\n"
             "    qubit: Qubit index.\n"
             "    p_meas1_prep0: P(measure 1 | state was 0) — false positive.\n"
             "    p_meas0_prep1: P(measure 0 | state was 1) — false negative.\n\n"
             "Example: nm.set_readout_error(0, 0.003, 0.06)")
        .def("set_readout_error_symmetric", &noise::NoiseModel::set_readout_error_symmetric, "qubit"_a, "p_error"_a,
             "Set symmetric readout error (same rate for both directions).\n\n"
             "Applied per measured qubit when the measurement writes its "
             "classical bit.\n\n"
             "Example: nm.set_readout_error_symmetric(0, 0.01)")
        .def("set_all_readout_error", &noise::NoiseModel::set_all_readout_error, "num_qubits"_a, "p_error"_a,
             "Set uniform symmetric readout error on qubits [0, num_qubits).\n\n"
             "Applied per measured qubit when the measurement writes its "
             "classical bit.")
        .def("has_readout_error", &noise::NoiseModel::has_readout_error, "Return True if any readout error parameters have been set.")
        // ── Two-qubit depolarizing ──
        .def("set_2q_depolarizing", &noise::NoiseModel::set_2q_depolarizing, "q1"_a, "q2"_a, "p"_a,
             "Set two-qubit depolarizing channel applied after CX/CZ gates "
             "on (q1, q2).\n\n"
             "Channel: Λ(ρ) = (1-p)ρ + p/15 · Σ PρP†  "
             "(15 non-identity two-qubit Paulis).\n"
             "Applied ONLY after 2Q gates, separate from per-qubit noise.\n\n"
             "Example: nm.set_2q_depolarizing(0, 1, 1.8e-3)")
        .def("has_any_2q_depolarizing", &noise::NoiseModel::has_any_2q_depolarizing, "Return True if any two-qubit depolarizing has been set.")
        // ── Gate-type-specific noise ──
        .def("set_1q_gate_depolarizing", &noise::NoiseModel::set_1q_gate_depolarizing, "qubit"_a, "p"_a,
             "Set depolarizing noise applied only after single-qubit gates.\n\n"
             "This is separate from set_depolarizing() which applies after ALL "
             "gates.\n\n"
             "Example: nm.set_1q_gate_depolarizing(0, 2.3e-4)")
        .def("set_2q_gate_depolarizing", &noise::NoiseModel::set_2q_gate_depolarizing, "qubit"_a, "p"_a,
             "Set depolarizing noise applied only after two-qubit gates "
             "involving this qubit.\n\n"
             "Example: nm.set_2q_gate_depolarizing(0, 1.8e-3)")
        .def("set_all_1q_gate_depolarizing", &noise::NoiseModel::set_all_1q_gate_depolarizing, "num_qubits"_a, "p"_a,
             "Set uniform 1Q gate depolarizing on qubits [0, num_qubits).")
        .def("set_all_2q_gate_depolarizing", &noise::NoiseModel::set_all_2q_gate_depolarizing, "num_qubits"_a, "p"_a,
             "Set uniform 2Q gate depolarizing on qubits [0, num_qubits).")
        .def("has_1q_gate_noise", &noise::NoiseModel::has_1q_gate_noise, "Return True if any 1Q gate-specific noise has been set.")
        .def("has_2q_gate_noise", &noise::NoiseModel::has_2q_gate_noise, "Return True if any 2Q gate-specific noise has been set.")
        .def("has_any", &noise::NoiseModel::has_any, "Return True if any noise of any type has been configured.");

    // --- Noisy Estimation (analytical — zero simulation overhead) ---
    m.def(
        "noisy_estimate",
        [](std::shared_ptr<Circuits::Circuit<double>> circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
           const SimulatorConfig &config) {
            auto paulis = ParseObservables(observables);
            nb::dict result = estimate_core(circuit, paulis, config);

            // Apply analytical Pauli noise damping
            nb::list ideal = nb::cast<nb::list>(result["expectation_values"]);
            nb::list noisy_vals;
            for (size_t i = 0; i < paulis.size(); ++i)
            {
                double damping = noise_model.compute_damping(paulis[i]);
                noisy_vals.append(damping * nb::cast<double>(ideal[i]));
            }

            nb::dict out;
            out["expectation_values"] = noisy_vals;
            out["ideal_expectation_values"] = ideal;
            out["time_taken"] = result["time_taken"];
            out["simulator"] = result["simulator"];
            out["method"] = result["method"];
            if (result.contains("gpu_device"))
                out["gpu_device"] = result["gpu_device"];
            return out;
        },
        "circuit"_a, "observables"_a, "noise_model"_a, "config"_a = SimulatorConfig{},
        "Compute expectation values with single-layer analytical Pauli "
        "noise damping. Runs the noiseless simulation then applies the "
        "damping factor -- zero simulation overhead compared to "
        "noiseless.\n\n"
        "WARNING -- this is a SINGLE-LAYER approximation, not an exact "
        "result. The damping factor is exact only for one Pauli layer applied "
        "immediately before measurement. A circuit that carries noise after "
        "every gate is not modelled: Pauli channels do not commute through "
        "non-Clifford gates, and the damping does not compound with the number "
        "of noisy layers acting on each qubit. The returned values therefore "
        "UNDERESTIMATE the noise, increasingly so with circuit depth.\n\n"
        "Only the all-gates Pauli layer contributes; T1, thermal, coherent, "
        "correlated and two-qubit layers are ignored entirely. Use "
        "noisy_estimate_montecarlo(), or noisy_execute() with a density-matrix "
        "or MPO method, when the magnitude of the noise matters.");

    // --- QASM variant ---
    m.def(
        "noisy_estimate",
        [](const std::string &qasm, const nb::object &observables, const noise::NoiseModel &noise_model, const SimulatorConfig &config) {
            qasm::QasmToCirc<> parser;
            auto circuit = parser.ParseAndTranslate(qasm);
            if (parser.Failed() || !circuit)
                throw nb::value_error("Failed to parse QASM string.");

            auto paulis = ParseObservables(observables);
            nb::dict result = estimate_core(circuit, paulis, config);

            nb::list ideal = nb::cast<nb::list>(result["expectation_values"]);
            nb::list noisy_vals;
            for (size_t i = 0; i < paulis.size(); ++i)
            {
                double damping = noise_model.compute_damping(paulis[i]);
                noisy_vals.append(damping * nb::cast<double>(ideal[i]));
            }

            nb::dict out;
            out["expectation_values"] = noisy_vals;
            out["ideal_expectation_values"] = ideal;
            out["time_taken"] = result["time_taken"];
            out["simulator"] = result["simulator"];
            out["method"] = result["method"];
            if (result.contains("gpu_device"))
                out["gpu_device"] = result["gpu_device"];
            return out;
        },
        "qasm_circuit"_a, "observables"_a, "noise_model"_a, "config"_a = SimulatorConfig{},
        "Compute expectation values from a QASM circuit with single-layer "
        "analytical Pauli noise damping. Zero simulation overhead.\n\n"
        "WARNING -- this is a SINGLE-LAYER approximation, not an exact "
        "result. The damping factor is exact only for one Pauli layer applied "
        "immediately before measurement. A circuit that carries noise after "
        "every gate is not modelled: Pauli channels do not commute through "
        "non-Clifford gates, and the damping does not compound with the number "
        "of noisy layers acting on each qubit. The returned values therefore "
        "UNDERESTIMATE the noise, increasingly so with circuit depth.\n\n"
        "Only the all-gates Pauli layer contributes; T1, thermal, coherent, "
        "correlated and two-qubit layers are ignored entirely. Use "
        "noisy_estimate_montecarlo(), or noisy_execute() with a density-matrix "
        "or MPO method, when the magnitude of the noise matters.");

    // --- Gate-by-gate Monte Carlo Noisy Estimation ---
    m.def("noisy_estimate_montecarlo", &NoisyEstimateMonteCarlo, "circuit"_a, "observables"_a, "noise_model"_a, "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{}, "noise_seed"_a = nb::none(),
          "Gate-by-gate Monte Carlo noisy estimation. Injects random Pauli "
          "errors after every gate and averages expectation values over "
          "noise_realizations independent samples. More accurate than "
          "analytical noisy_estimate for deep circuits.");

    m.def("noisy_execute", &NoisyExecute, "circuit"_a, "noise_model"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "noise_realizations"_a = 64,
          "noise_seed"_a = nb::none(),
          "Execute with exact Pauli/T1 channels for density-matrix/MPO methods, "
          "or sampled trajectories for pure-state methods. Shots are distributed "
          "evenly across 'noise_realizations' batches.");

    // =========================================================================
    // Coherent Noise: Execute
    // =========================================================================

    m.def("coherent_execute", &CoherentExecute, "circuit"_a, "noise_model"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "noise_realizations"_a = 64,
          "noise_seed"_a = nb::none(),
          "Execute a circuit with sampled coherent over/under-rotation errors. "
          "After every gate, Rx/Ry/Rz rotations are injected with random ± "
          "signs. Each of 'noise_realizations' batches uses a different sign "
          "pattern. Supported by statevector, MPS, density-matrix, and MPO "
          "methods (not Stabilizer).\n\n"
          "Example:\n"
          "    nm = maestro.NoiseModel()\n"
          "    nm.set_all_coherent_depolarizing(n_qubits, 0.001)\n"
          "    result = maestro.coherent_execute(qc, nm, shots=1000)\n");

    // =========================================================================
    // Coherent Noise: Estimate (Monte Carlo averaged)
    // =========================================================================

    m.def("coherent_estimate", &CoherentEstimate, "circuit"_a, "observables"_a, "noise_model"_a, "noise_realizations"_a = 100, "config"_a = SimulatorConfig{},
          "noise_seed"_a = nb::none(),
          "Estimate expectation values with coherent noise (rotation errors). "
          "Injects systematic Rx/Ry/Rz rotations after every gate and averages "
          "expectation values over noise_realizations independent sign samples. "
          "Unlike Pauli noise, coherent noise preserves phase coherence and "
          "does not commute with the circuit — it can model systematic "
          "calibration errors.\n\n"
          "Example:\n"
          "    nm = maestro.NoiseModel()\n"
          "    nm.set_coherent_strength(n_qubits, 0.001)\n"
          "    result = maestro.coherent_estimate(qc, ['ZZ', 'XX'], nm)\n");

    // =========================================================================
    // Combined Noise: Execute (all layers in one call)
    // =========================================================================

    m.def("full_noise_execute", &FullNoiseExecute, "circuit"_a, "noise_model"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "noise_realizations"_a = 64,
          "noise_seed"_a = nb::none(),
          "Execute a circuit with combined noise (coherent + crosstalk + T1 + "
          "Pauli). Density-matrix/MPO methods apply Markovian T1 and Pauli layers "
          "as exact channels; trajectory-only layers remain sampled.\n\n"
          "Example:\n"
          "    nm = maestro.NoiseModel()\n"
          "    nm.set_all_coherent_depolarizing(n, 0.001)\n"
          "    nm.set_crosstalk(0, 1, 0.005)\n"
          "    nm.set_all_t1(n, 0.0003)\n"
          "    nm.set_all_depolarizing(n, 0.001)\n"
          "    result = maestro.full_noise_execute(qc, nm, shots=1000)\n");

    // =========================================================================
    // Combined Noise: Estimate (all layers in one call)
    // =========================================================================

    m.def("full_noise_estimate", &FullNoiseEstimate, "circuit"_a, "observables"_a, "noise_model"_a, "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{}, "noise_seed"_a = nb::none(),
          "Estimate expectation values with combined noise (coherent + crosstalk "
          "+ T1 + Pauli). All configured noise layers are applied per gate.\n\n"
          "Example:\n"
          "    nm = maestro.NoiseModel()\n"
          "    nm.set_all_coherent_depolarizing(n, 0.001)\n"
          "    nm.set_crosstalk(0, 1, 0.005)\n"
          "    nm.set_all_t1(n, 0.0003)\n"
          "    result = maestro.full_noise_estimate(qc, ['ZZ'], nm)\n");
}

} // namespace maestro_bindings
