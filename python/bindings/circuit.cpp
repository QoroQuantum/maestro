#include "core.h"
#include "module.h"

namespace maestro_bindings {

void bind_circuit(nb::module_& m) {
  // --- Circuits Submodule ---
  auto circuits = m.def_submodule("circuits", "Quantum circuits submodule");

  nb::class_<Circuits::Circuit<double>>(circuits, "QuantumCircuit")
      .def(nb::init<>())
      .def_prop_ro("num_qubits",
                   [](const Circuits::Circuit<double> &c) {
                     return c.GetMaxQubitIndex() + 1;
                   })
      // Standard Gates
      .def("x",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::XGate<>>(q));
           })
      .def("y",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::YGate<>>(q));
           })
      .def("z",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::ZGate<>>(q));
           })
      .def("h",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::HadamardGate<>>(q));
           })
      // Single Qubit Gates (Non-Parametric)
      .def("s",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::SGate<>>(q));
           })
      .def("sdg",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::SdgGate<>>(q));
           })
      .def("t",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::TGate<>>(q));
           })
      .def("tdg",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::TdgGate<>>(q));
           })
      .def("sx",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::SxGate<>>(q));
           })
      .def("sxdg",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::SxDagGate<>>(q));
           })
      .def("k",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(std::make_shared<Circuits::KGate<>>(q));
           })

      // Single Qubit Gates (Parametric)
      .def("p",
           [](Circuits::Circuit<double> &s, Types::qubit_t q, double lambda) {
             s.AddOperation(std::make_shared<Circuits::PhaseGate<>>(q, lambda));
           })
      .def("rx",
           [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta) {
             s.AddOperation(std::make_shared<Circuits::RxGate<>>(q, theta));
           })
      .def("ry",
           [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta) {
             s.AddOperation(std::make_shared<Circuits::RyGate<>>(q, theta));
           })
      .def("rz",
           [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta) {
             s.AddOperation(std::make_shared<Circuits::RzGate<>>(q, theta));
           })
      .def("u",
           [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta,
              double phi, double lambda) {
             s.AddOperation(
                 std::make_shared<Circuits::UGate<>>(q, theta, phi, lambda));
           })
      .def("delay",
           [](Circuits::Circuit<double> &s, Types::qubit_t q, double duration) {
             s.Delay(q, duration);
           }, "qubit"_a, "duration"_a,
           "Append a delay idle operation on a qubit for a given duration in seconds.")
      .def("delay",
           [](Circuits::Circuit<double> &s, double duration, Types::qubit_t q) {
             s.Delay(q, duration);
           }, "duration"_a, "qubit"_a,
           "Append a delay idle operation on a qubit for a given duration in seconds.")

      // Two Qubit Gates
      .def(
          "cx",
          [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) {
            s.AddOperation(std::make_shared<Circuits::CXGate<>>(c, t));
          })
      .def(
          "cy",
          [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) {
            s.AddOperation(std::make_shared<Circuits::CYGate<>>(c, t));
          })
      .def(
          "cz",
          [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) {
            s.AddOperation(std::make_shared<Circuits::CZGate<>>(c, t));
          })
      .def(
          "ch",
          [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) {
            s.AddOperation(std::make_shared<Circuits::CHGate<>>(c, t));
          })
      .def(
          "csx",
          [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) {
            s.AddOperation(std::make_shared<Circuits::CSxGate<>>(c, t));
          })
      .def(
          "csxdg",
          [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) {
            s.AddOperation(std::make_shared<Circuits::CSxDagGate<>>(c, t));
          })
      .def(
          "swap",
          [](Circuits::Circuit<double> &s, Types::qubit_t a, Types::qubit_t b) {
            s.AddOperation(std::make_shared<Circuits::SwapGate<>>(a, b));
          })

      // Controlled Parametric Gates
      .def("cp",
           [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
              double lambda) {
             s.AddOperation(std::make_shared<Circuits::CPGate<>>(c, t, lambda));
           })
      .def("crx",
           [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
              double theta) {
             s.AddOperation(std::make_shared<Circuits::CRxGate<>>(c, t, theta));
           })
      .def("cry",
           [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
              double theta) {
             s.AddOperation(std::make_shared<Circuits::CRyGate<>>(c, t, theta));
           })
      .def("crz",
           [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
              double theta) {
             s.AddOperation(std::make_shared<Circuits::CRzGate<>>(c, t, theta));
           })
      .def("cu",
           [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
              double theta, double phi, double lambda, double gamma) {
             s.AddOperation(std::make_shared<Circuits::CUGate<>>(
                 c, t, theta, phi, lambda, gamma));
           })

      // Three Qubit Gates
      .def("ccx",
           [](Circuits::Circuit<double> &s, Types::qubit_t c1,
              Types::qubit_t c2, Types::qubit_t t) {
             s.AddOperation(std::make_shared<Circuits::CCXGate<>>(c1, c2, t));
           })
      .def("cswap",
           [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t a,
              Types::qubit_t b) {
             s.AddOperation(std::make_shared<Circuits::CSwapGate<>>(c, a, b));
           })
      // Measurement
      .def("measure",
           [](Circuits::Circuit<double> &s,
              const std::vector<std::pair<Types::qubit_t, size_t>> &q) {
             s.AddOperation(
                 std::make_shared<Circuits::MeasurementOperation<>>(q));
           })
      .def("measure_all",
           [](Circuits::Circuit<double> &s) {
             size_t n = s.GetMaxQubitIndex() + 1;
             std::vector<std::pair<Types::qubit_t, size_t>> pairs;
             pairs.reserve(n);
             for (size_t i = 0; i < n; ++i)
               pairs.emplace_back(static_cast<Types::qubit_t>(i), i);
             s.AddOperation(
                 std::make_shared<Circuits::MeasurementOperation<>>(pairs));
           })
      // Reset
      .def("reset",
           [](Circuits::Circuit<double> &s, Types::qubit_t q) {
             s.AddOperation(
                 std::make_shared<Circuits::Reset<>>(Types::qubits_vector{q}));
           },
           "qubit"_a,
           "Reset a qubit to |0>.")
      .def("reset_qubits",
           [](Circuits::Circuit<double> &s,
              const std::vector<Types::qubit_t> &qubits) {
             s.AddOperation(std::make_shared<Circuits::Reset<>>(qubits));
           },
           "qubits"_a,
           "Reset multiple qubits to |0>.")

      // ---- Bound Methods for Direct Execution ----
      .def("execute", &execute_core,
           "config"_a = SimulatorConfig{}, "shots"_a = 1024)
      .def(
          "estimate",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const nb::object &observables,
             const SimulatorConfig &config) {
            return estimate_core(self, ParseObservables(observables), config);
          },
          "observables"_a, "config"_a = SimulatorConfig{})
      .def(
          "get_statevector",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const SimulatorConfig &config) {
            return statevector_core(self, config);
          },
          "config"_a = SimulatorConfig{},
          "Get the full statevector (complex amplitudes) after executing the "
          "circuit.")
      .def(
          "mirror_fidelity",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const SimulatorConfig &config, int shots, bool full_amplitude) {
            return mirror_fidelity_core(self, config, shots, full_amplitude);
          },
          "config"_a = SimulatorConfig{},
          "shots"_a = 1024,
          "full_amplitude"_a = false,
          "Compute mirror fidelity: run circuit forward then its adjoint in "
          "reverse, returning P(|0...0>). Uses shot-based sampling by "
          "default. Set full_amplitude=True for exact statevector "
          "computation (small circuits only).")
      .def(
          "inner_product",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             std::shared_ptr<Circuits::Circuit<double>> other,
             const SimulatorConfig &config) {
            return inner_product_core(self, other, config);
          },
          "other"_a, "config"_a = SimulatorConfig{},
          "Compute the inner product <psi_self|psi_other> = <0|U_self^dag "
          "U_other|0> between this circuit's state and another circuit's "
          "state, using ProjectOnZero.")
      .def(
          "prob",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const std::string &target_state) -> nb::dict {
            if (!self) throw nb::value_error("Circuit is null.");
            if (target_state.empty())
              throw nb::value_error(
                  "target_state must be a non-empty bitstring.");

            std::vector<bool> end_state(target_state.size());
            for (size_t i = 0; i < target_state.size(); ++i) {
              if (target_state[i] == '1') end_state[i] = true;
              else if (target_state[i] == '0') end_state[i] = false;
              else throw nb::value_error(
                  "target_state must contain only '0' and '1' characters.");
            }

            Simulators::PathIntegralSimulator sim;
            sim.SetStartZeroState(target_state.size());

            auto start = std::chrono::high_resolution_clock::now();
            bool ok;
            {
              nb::gil_scoped_release release;
              ok = sim.SetCircuit(self);
            }
            if (!ok)
              throw std::runtime_error(
                  "Circuit contains operations not supported by the path "
                  "integral simulator.");

            auto amplitude = sim.AmplitudeFromZero(end_state);
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict result;
            result["amplitude"] = amplitude;
            result["probability"] = std::norm(amplitude);
            result["target_state"] = target_state;
            result["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            return result;
          },
          "target_state"_a,
          "Compute the probability of a specific output state using the "
          "Pauli path integral simulator.\n\n"
          "Example: qc.prob('111') returns the probability of |111>.\n\n"
          "Args:\n"
          "    target_state: Bitstring like '10001001' (qubit 0 leftmost).\n\n"
          "Returns:\n"
          "    dict with 'probability', 'amplitude', 'target_state', "
          "'time_taken'.")
      .def(
          "noisy_prob",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const std::string &target_state,
             const noise::NoiseModel &noise_model) -> nb::dict {
            if (!self) throw nb::value_error("Circuit is null.");
            if (target_state.empty())
              throw nb::value_error(
                  "target_state must be a non-empty bitstring.");

            const size_t n = target_state.size();
            for (size_t i = 0; i < n; ++i) {
              if (target_state[i] != '0' && target_state[i] != '1')
                throw nb::value_error(
                    "target_state must contain only '0' and '1' characters.");
            }

            // Helper: compute P(bitstring) via path integral
            auto pi_prob = [&](const std::string &bs) -> double {
              std::vector<bool> end_state(bs.size());
              for (size_t i = 0; i < bs.size(); ++i)
                end_state[i] = (bs[i] == '1');

              Simulators::PathIntegralSimulator sim;
              sim.SetStartZeroState(bs.size());
              bool ok;
              {
                nb::gil_scoped_release release;
                ok = sim.SetCircuit(self);
              }
              if (!ok)
                throw std::runtime_error(
                    "Circuit contains operations not supported by the path "
                    "integral simulator.");
              auto amplitude = sim.AmplitudeFromZero(end_state);
              return std::norm(amplitude);
            };

            auto start = std::chrono::high_resolution_clock::now();

            double p_noisy;

            if (noise_model.has_readout_error()) {
              // First-order readout error expansion:
              // P_noisy(b) ≈ Π(1-p_i) * P(b) + Σ_i [p_i * Π_{j≠i}(1-p_j)] * P(b⊕e_i)

              // Compute per-qubit flip probabilities for the target bitstring
              std::vector<double> p_flip(n);
              for (size_t i = 0; i < n; ++i) {
                const auto *re = noise_model.get_readout_error(
                    static_cast<int>(i));
                if (!re) { p_flip[i] = 0.0; continue; }
                p_flip[i] = (target_state[i] == '0')
                    ? re->p_meas1_prep0 : re->p_meas0_prep1;
              }

              // 0-flip term: probability of no readout errors
              double no_flip_prob = 1.0;
              for (size_t i = 0; i < n; ++i)
                no_flip_prob *= (1.0 - p_flip[i]);

              double p_target = pi_prob(target_state);
              p_noisy = no_flip_prob * p_target;

              // 1-flip terms: one readout error on qubit i
              for (size_t i = 0; i < n; ++i) {
                if (p_flip[i] <= 0.0) continue;
                std::string flipped = target_state;
                flipped[i] = (flipped[i] == '0') ? '1' : '0';
                double one_flip_weight =
                    p_flip[i] / (1.0 - p_flip[i]) * no_flip_prob;
                p_noisy += one_flip_weight * pi_prob(flipped);
              }
            } else {
              // No readout error — just compute exact probability
              p_noisy = pi_prob(target_state);
            }

            auto end = std::chrono::high_resolution_clock::now();

            nb::dict result;
            result["probability"] = p_noisy;
            result["target_state"] = target_state;
            result["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            result["has_readout_error"] = noise_model.has_readout_error();
            return result;
          },
          "target_state"_a, "noise_model"_a,
          "Compute readout-corrected probability using path integral.\n\n"
          "When the noise model has readout error, uses first-order expansion:\n"
          "P_noisy(b) ≈ Π(1-p_i)·P(b) + Σ_i p_i·Π_{j≠i}(1-p_j)·P(b⊕eᵢ)\n\n"
          "This requires n+1 path integral evaluations (target + n flipped "
          "variants).\n\n"
          "Args:\n"
          "    target_state: Bitstring like '10001001'.\n"
          "    noise_model: NoiseModel with readout error set.\n\n"
          "Returns:\n"
          "    dict with 'probability', 'target_state', 'time_taken', "
          "'has_readout_error'.")

      // ---- Bound Methods for Noisy Execution ----
      .def("noisy_execute", &NoisyExecute, "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "shots"_a = 1024,
          "noise_realizations"_a = 64, "noise_seed"_a = nb::none(),
          "Execute with exact Pauli/T1 channels for density-matrix/MPO "
          "methods, or sampled trajectories for pure-state methods.\n\n"
          "Example: qc.noisy_execute(nm, shots=1000)")
      .def(
          "noisy_estimate",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const nb::object &observables,
             const noise::NoiseModel &noise_model,
             const SimulatorConfig &config) {
            auto paulis = ParseObservables(observables);
            nb::dict result = estimate_core(self, paulis, config);

            nb::list ideal = nb::cast<nb::list>(result["expectation_values"]);
            nb::list noisy_vals;
            for (size_t i = 0; i < paulis.size(); ++i) {
              double damping = noise_model.compute_damping(paulis[i]);
              noisy_vals.append(damping * nb::cast<double>(ideal[i]));
            }

            nb::dict out;
            out["expectation_values"] = noisy_vals;
            out["ideal_expectation_values"] = ideal;
            out["time_taken"] = result["time_taken"];
            out["simulator"] = result["simulator"];
            out["method"] = result["method"];
            if (result.contains("gpu_device")) out["gpu_device"] = result["gpu_device"];
            return out;
          },
          "observables"_a, "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "Single-layer analytical noisy estimation (zero simulation "
          "overhead). Applies per-qubit Pauli damping to the ideal "
          "expectation values.\n\n"
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
          "or MPO method, when the magnitude of the noise matters."
          "\n\n"
          "Example: qc.noisy_estimate(['ZZ', 'XX'], nm)")
      .def("noisy_estimate_montecarlo", &NoisyEstimateMonteCarlo,
          "observables"_a, "noise_model"_a,
          "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{},
          "noise_seed"_a = nb::none(),
          "Gate-by-gate Monte Carlo noisy estimation.\n\n"
          "Example: qc.noisy_estimate_montecarlo(['ZZ'], nm, "
          "noise_realizations=200)")
      .def("coherent_execute", &CoherentExecute, "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "shots"_a = 1024,
          "noise_realizations"_a = 64, "noise_seed"_a = nb::none(),
          "Execute with sampled coherent over/under-rotation errors.\n\n"
          "Example: qc.coherent_execute(nm, shots=1000)")
      .def("coherent_estimate", &CoherentEstimate,
          "observables"_a, "noise_model"_a,
          "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{},
          "noise_seed"_a = nb::none(),
          "Estimate expectation values with coherent noise.\n\n"
          "Example: qc.coherent_estimate(['ZZ', 'XX'], nm, "
          "noise_realizations=200)")
      // ---- Combined Noise (all layers) ----
      .def("full_noise_execute", &FullNoiseExecute, "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "shots"_a = 1024,
          "noise_realizations"_a = 64, "noise_seed"_a = nb::none(),
          "Execute with combined noise: coherent + crosstalk + T1 + Pauli.\n\n"
          "DM/MPO methods use exact channels for T1 and Pauli layers; "
          "trajectory-only layers remain sampled.\n"
          "Example: qc.full_noise_execute(nm, shots=1000)")
      .def("full_noise_estimate", &FullNoiseEstimate,
          "observables"_a, "noise_model"_a,
          "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{},
          "noise_seed"_a = nb::none(),
          "Estimate with combined noise (coherent + crosstalk + T1 + Pauli).\n\n"
          "Example: qc.full_noise_estimate(['ZZ', 'XX'], nm)")
      // ---- Noisy Fidelity (inner-product) ----
      .def(
          "noisy_fidelity",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const noise::NoiseModel &noise_model, int noise_realizations,
             const SimulatorConfig &config,
             std::optional<uint64_t> noise_seed) {
            require_circuit(self);
            require_realizations(noise_realizations);
            require_any_noise(noise_model);

            warn_thermal_approximation(noise_model, config);
            auto rng = MakeNoiseRng(config, noise_seed);
            double sum_fid = 0.0;
            double sum_fid_sq = 0.0;

            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < noise_realizations; ++r) {
              auto noisy =
                  inject_combined_noise_for_config(self, noise_model, rng, config);
              // Reset collapse must use a fresh seed for each realization.
              auto realization_config =
                  NoiseExecutionConfig(config, noise_seed, r);
              if (!realization_config.seed) realization_config.seed = rng();
              double fid = noisy_fidelity_core(self, noisy, realization_config);
              sum_fid += fid;
              sum_fid_sq += fid * fid;
            }
            auto end = std::chrono::high_resolution_clock::now();

            double mean_fid = sum_fid / noise_realizations;
            double var = (noise_realizations > 1)
                ? (sum_fid_sq - noise_realizations * mean_fid * mean_fid) /
                      (noise_realizations - 1)
                : 0.0;
            double se = std::sqrt(std::max(var, 0.0) / noise_realizations);
            double mean_infid = 1.0 - mean_fid;

            nb::dict out;
            out["fidelity"] = mean_fid;
            out["infidelity"] = mean_infid;
            out["std_error"] = se;
            out["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            out["noise_realizations"] = noise_realizations;
            return out;
          },
          "noise_model"_a,
          "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{},
          "noise_seed"_a = nb::none(),
          "Compute fidelity to the ideal unitary circuit state under noise.\n\n"
          "Injects all configured noise types (correlated, coherent, "
          "crosstalk, T1, Pauli) and averages |<psi_ideal|psi_noisy>|^2 "
          "over noise realizations.\n\n"
          "Returns dict with 'fidelity', 'infidelity', 'std_error', "
          "'time_taken', 'noise_realizations'.\n\n"
          "Example: qc.noisy_fidelity(nm, noise_realizations=200)");
}

}  // namespace maestro_bindings
