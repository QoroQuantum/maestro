#include "core.h"
#include "module.h"

namespace maestro_bindings
{

namespace
{
using Channel = Simulators::QuantumChannel;
using NestedMatrix = std::vector<std::vector<std::complex<double>>>;

// Nested row-major lists (or 2-D arrays) as matrices; the channel constructors validate the values.
Channel::KrausOperators MatricesFromNested(const std::vector<NestedMatrix> &operators, const char *what)
{
    Channel::KrausOperators matrices;
    matrices.reserve(operators.size());
    for (const auto &rows : operators)
    {
        if (rows.empty() || rows.front().empty())
            throw nb::value_error((std::string(what) + " must be nonempty matrices.").c_str());
        const size_t columns = rows.front().size();
        Eigen::MatrixXcd matrix(static_cast<Eigen::Index>(rows.size()), static_cast<Eigen::Index>(columns));
        for (size_t row = 0; row < rows.size(); ++row)
        {
            if (rows[row].size() != columns)
                throw nb::value_error((std::string(what) + " must have rows of equal length.").c_str());
            for (size_t column = 0; column < columns; ++column)
                matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) = rows[row][column];
        }
        matrices.emplace_back(std::move(matrix));
    }
    return matrices;
}

const char *FactorKindName(Channel::FactorKind kind)
{
    switch (kind)
    {
    case Channel::FactorKind::kPauli:
        return "pauli";
    case Channel::FactorKind::kUnitaryMixture:
        return "unitary_mixture";
    case Channel::FactorKind::kThermal:
        return "thermal";
    default:
        return "kraus";
    }
}

nb::list FactorsOf(const Channel &channel)
{
    nb::list factors;
    for (const auto &factor : channel.GetFactors())
    {
        nb::dict entry;
        entry["kind"] = FactorKindName(factor.kind);
        entry["qubits"] = nb::cast(factor.qubits);
        if (!factor.probabilities.empty())
            entry["probabilities"] = nb::cast(factor.probabilities);
        if (factor.kind == Channel::FactorKind::kThermal)
        {
            entry["decay"] = factor.thermal.decay;
            entry["excited_population"] = factor.thermal.excitedStatePopulation;
            entry["coherence"] = factor.thermal.coherence;
        }
        factors.append(entry);
    }
    return factors;
}

void bind_quantum_channel(nb::module_ &m)
{
    nb::class_<Channel>(m, "QuantumChannel",
                        "A one- or two-qubit noise channel (a completely positive, trace-preserving map).\n\n"
                        "Build one once and pass it to QuantumCircuit.channel() at every place it acts: the "
                        "circuit then shares a single object instead of copying matrices per instruction. "
                        "then() composes channels and keeps their parts, so a gate error made of a Pauli "
                        "mixture followed by relaxation stays a Pauli mixture followed by relaxation.\n\n"
                        "BASIS ORDER for two-qubit matrices: qubits[0] is the LEAST significant bit of the "
                        "4x4 index, so A on qubits[0] and B on qubits[1] is the Kronecker product B (x) A.")
        .def_static(
            "from_kraus", [](const std::vector<NestedMatrix> &operators) { return Channel(MatricesFromNested(operators, "Kraus operators")); }, "operators"_a,
            "Channel with the given Kraus operators (2x2 or 4x4 matrices). Completeness, sum_k E_k^dag E_k = I, is validated.")
        .def_static(
            "pauli", [](const std::vector<double> &probabilities) { return Channel::Pauli(probabilities); }, "probabilities"_a,
            "Pauli channel from 4**n probabilities summing to one. Base-4 digit k of an index selects I, X, Y or Z on "
            "qubits[k]: one qubit is [pI, pX, pY, pZ], two are II, XI, YI, ZI, IX, XX, ... ZZ.")
        .def_static(
            "unitary_mixture",
            [](const std::vector<double> &probabilities, const std::vector<NestedMatrix> &unitaries) {
                return Channel::UnitaryMixture(probabilities, MatricesFromNested(unitaries, "Unitaries"));
            },
            "probabilities"_a, "unitaries"_a, "Channel that applies unitaries[i] with probabilities[i]. The probabilities must sum to one.")
        .def_static(
            "depolarizing",
            [](double probability, size_t num_qubits) {
                if (num_qubits == 1)
                    return Channel::Depolarizing(probability);
                if (num_qubits == 2)
                    return Channel::TwoQubitDepolarizing(probability);
                throw nb::value_error("num_qubits must be 1 or 2.");
            },
            "probability"_a, "num_qubits"_a = 1,
            "Depolarizing channel: the identity with probability 1 - probability, otherwise one of the 3 (or 15) "
            "other Paulis with equal probability.")
        .def_static(
            "amplitude_damping", [](double gamma) { return Channel::AmplitudeDamping(gamma); }, "gamma"_a, "Relaxation of |1> to |0> with probability gamma.")
        .def_static(
            "phase_damping", [](double gamma) { return Channel::PhaseDamping(gamma); }, "gamma"_a,
            "Phase damping: coherences are multiplied by sqrt(1 - gamma).")
        .def_static(
            "generalized_amplitude_damping",
            [](double gamma, double excited_population) { return Channel::GeneralizedAmplitudeDamping(gamma, excited_population); }, "gamma"_a,
            "excited_population"_a, "Amplitude damping towards an equilibrium with the given population of |1>.")
        .def_static(
            "thermal_relaxation",
            [](double duration, double t1, double t2, double excited_population) { return Channel::ThermalRelaxation(duration, t1, t2, excited_population); },
            "duration"_a, "t1"_a, "t2"_a, "excited_population"_a = 0.0,
            "Thermal relaxation over a duration: populations relax with exp(-duration/t1), coherences with "
            "exp(-duration/t2). Requires t2 <= 2*t1; float('inf') is accepted for either time.")
        .def_static(
            "on_qubit", [](const Channel &channel, size_t qubit, size_t num_qubits) { return Channel::OnQubit(channel, qubit, num_qubits); }, "channel"_a,
            "qubit"_a, "num_qubits"_a = 2, "A one-qubit channel placed on one target (0 or 1) of a two-qubit channel that leaves the other target alone.")
        .def_static(
            "placed", [](const Channel &channel, const std::vector<size_t> &qubits, size_t num_qubits) { return Channel::Placed(channel, qubits, num_qubits); },
            "channel"_a, "qubits"_a, "num_qubits"_a = 2,
            "A channel placed on targets of a channel of num_qubits targets: qubits[k] is the target that target k of channel "
            "acts on. on_qubit is the one-qubit case; [1, 0] exchanges the targets of a two-qubit channel.")
        .def(
            "then",
            [](const Channel &self, const Channel &next, const std::optional<std::vector<size_t>> &qubits) {
                return qubits ? self.Then(next, *qubits) : self.Then(next);
            },
            "next"_a, "qubits"_a = nb::none(),
            "This channel followed by another. With qubits, next acts on those targets of this channel: "
            "two_qubit_error.then(relaxation, [1]) relaxes the second qubit after the two-qubit error.")
        .def_prop_ro("num_qubits", [](const Channel &self) { return self.GetNumberOfQubits(); })
        .def_prop_ro(
            "kraus_operators", [](const Channel &self) { return self.GetKrausOperators(); },
            "Kraus operators as complex arrays. A composed channel returns its minimal set, largest weight first.")
        .def_prop_ro("factors", &FactorsOf,
                     "The parts of the channel, first applied first: dicts with 'kind' ('pauli', 'unitary_mixture', 'thermal' or "
                     "'kraus'), 'qubits' and, where they apply, 'probabilities' and the relaxation parameters.")
        .def_prop_ro(
            "is_deterministic", [](const Channel &self) { return self.IsDeterministic(); },
            "Whether the channel has one branch only, a unitary. It is then a gate: a backend that samples channels draws "
            "nothing for it, and a circuit whose channels are all such evolves once for any number of shots.")
        .def_prop_ro(
            "is_state_independent", [](const Channel &self) { return self.IsStateIndependent(); },
            "True if every part is a Pauli or unitary mixture, so that a trajectory can draw it without reading its state.")
        .def(
            "superoperator", [](const Channel &self) { return self.Superoperator(); },
            "The channel as a matrix on vectorized density blocks indexed row + d*column: sum_k conj(K) (x) K.")
        .def(
            "choi", [](const Channel &self) { return self.ChoiMatrix(); }, "Choi matrix sum_k vec(K) vec(K)^dag, column-stacking vec.")
        .def(
            "is_approx", [](const Channel &self, const Channel &other, double tolerance) { return self.IsApprox(other, tolerance); }, "other"_a,
            "tolerance"_a = 1e-12, "Whether two channels are the same map (their Choi matrices agree to the tolerance).")
        .def("__repr__", [](const Channel &self) {
            std::string parts;
            for (const auto &factor : self.GetFactors())
                parts += (parts.empty() ? "" : ", ") + std::string(FactorKindName(factor.kind));
            return "QuantumChannel(num_qubits=" + std::to_string(self.GetNumberOfQubits()) + ", factors=[" + parts + "])";
        });
}
} // namespace

void bind_circuit(nb::module_ &m)
{
    bind_quantum_channel(m);

    // --- Circuits Submodule ---
    auto circuits = m.def_submodule("circuits", "Quantum circuits submodule");

    nb::class_<Circuits::Circuit<double>>(circuits, "QuantumCircuit")
        .def(nb::init<>())
        .def_prop_ro("num_qubits", [](const Circuits::Circuit<double> &c) { return c.GetMaxQubitIndex() + 1; })
        // Standard Gates
        .def("x", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::XGate<>>(q)); })
        .def("y", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::YGate<>>(q)); })
        .def("z", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::ZGate<>>(q)); })
        // h, s and sdg are the basis changes of non-Z measurements and resets;
        // noiseless=True keeps noise models from treating them as gates.
        .def(
            "h",
            [](Circuits::Circuit<double> &s, Types::qubit_t q, bool noiseless) {
                auto gate = std::make_shared<Circuits::HadamardGate<>>(q);
                gate->SetNoiseless(noiseless);
                s.AddOperation(gate);
            },
            "qubit"_a, "noiseless"_a = false,
            "Apply a Hadamard gate. With noiseless=True, noise models add no "
            "gate noise after it (for basis changes of a measurement or reset).")
        // Single Qubit Gates (Non-Parametric)
        .def(
            "s",
            [](Circuits::Circuit<double> &s, Types::qubit_t q, bool noiseless) {
                auto gate = std::make_shared<Circuits::SGate<>>(q);
                gate->SetNoiseless(noiseless);
                s.AddOperation(gate);
            },
            "qubit"_a, "noiseless"_a = false,
            "Apply an S gate. With noiseless=True, noise models add no gate "
            "noise after it (for basis changes of a measurement or reset).")
        .def(
            "sdg",
            [](Circuits::Circuit<double> &s, Types::qubit_t q, bool noiseless) {
                auto gate = std::make_shared<Circuits::SdgGate<>>(q);
                gate->SetNoiseless(noiseless);
                s.AddOperation(gate);
            },
            "qubit"_a, "noiseless"_a = false,
            "Apply an S-dagger gate. With noiseless=True, noise models add no "
            "gate noise after it (for basis changes of a measurement or reset).")
        .def("t", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::TGate<>>(q)); })
        .def("tdg", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::TdgGate<>>(q)); })
        .def("sx", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::SxGate<>>(q)); })
        .def("sxdg", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::SxDagGate<>>(q)); })
        .def("k", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::KGate<>>(q)); })

        // Single Qubit Gates (Parametric)
        .def("p", [](Circuits::Circuit<double> &s, Types::qubit_t q, double lambda) { s.AddOperation(std::make_shared<Circuits::PhaseGate<>>(q, lambda)); })
        .def("rx", [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta) { s.AddOperation(std::make_shared<Circuits::RxGate<>>(q, theta)); })
        .def("ry", [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta) { s.AddOperation(std::make_shared<Circuits::RyGate<>>(q, theta)); })
        .def("rz", [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta) { s.AddOperation(std::make_shared<Circuits::RzGate<>>(q, theta)); })
        .def("u", [](Circuits::Circuit<double> &s, Types::qubit_t q, double theta, double phi,
                     double lambda) { s.AddOperation(std::make_shared<Circuits::UGate<>>(q, theta, phi, lambda)); })
        .def(
            "delay", [](Circuits::Circuit<double> &s, Types::qubit_t q, double duration) { s.Delay(q, duration); }, "qubit"_a, "duration"_a,
            "Append a delay idle operation on a qubit for a given duration in seconds.")
        .def(
            "delay", [](Circuits::Circuit<double> &s, double duration, Types::qubit_t q) { s.Delay(q, duration); }, "duration"_a, "qubit"_a,
            "Append a delay idle operation on a qubit for a given duration in seconds.")

        // Two Qubit Gates
        .def("cx", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CXGate<>>(c, t)); })
        .def("cy", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CYGate<>>(c, t)); })
        .def("cz", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CZGate<>>(c, t)); })
        .def("ch", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CHGate<>>(c, t)); })
        .def("csx", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CSxGate<>>(c, t)); })
        .def("csxdg", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CSxDagGate<>>(c, t)); })
        .def("swap", [](Circuits::Circuit<double> &s, Types::qubit_t a, Types::qubit_t b) { s.AddOperation(std::make_shared<Circuits::SwapGate<>>(a, b)); })

        // Controlled Parametric Gates
        .def("cp", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
                      double lambda) { s.AddOperation(std::make_shared<Circuits::CPGate<>>(c, t, lambda)); })
        .def("crx", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
                       double theta) { s.AddOperation(std::make_shared<Circuits::CRxGate<>>(c, t, theta)); })
        .def("cry", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
                       double theta) { s.AddOperation(std::make_shared<Circuits::CRyGate<>>(c, t, theta)); })
        .def("crz", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t,
                       double theta) { s.AddOperation(std::make_shared<Circuits::CRzGate<>>(c, t, theta)); })
        .def("cu", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t t, double theta, double phi, double lambda,
                      double gamma) { s.AddOperation(std::make_shared<Circuits::CUGate<>>(c, t, theta, phi, lambda, gamma)); })

        // Three Qubit Gates
        .def("ccx", [](Circuits::Circuit<double> &s, Types::qubit_t c1, Types::qubit_t c2,
                       Types::qubit_t t) { s.AddOperation(std::make_shared<Circuits::CCXGate<>>(c1, c2, t)); })
        .def("cswap", [](Circuits::Circuit<double> &s, Types::qubit_t c, Types::qubit_t a,
                         Types::qubit_t b) { s.AddOperation(std::make_shared<Circuits::CSwapGate<>>(c, a, b)); })
        // Measurement
        .def(
            "measure",
            [](Circuits::Circuit<double> &s, const std::vector<std::pair<Types::qubit_t, size_t>> &q, double readout_error) {
                auto measurement = std::make_shared<Circuits::MeasurementOperation<>>(q);
                if (!std::isfinite(readout_error) || readout_error < 0.0 || readout_error > 1.0)
                    throw nb::value_error("readout_error must be finite and in [0, 1].");
                if (readout_error > 0.0)
                    measurement->SetReadout(std::vector<Circuits::ReadoutRates>(q.size(), {readout_error, readout_error}));
                s.AddOperation(measurement);
            },
            "qubits_and_bits"_a, "readout_error"_a = 0.0,
            "Measure each (qubit, classical bit) pair.\n\n"
            "readout_error flips each recorded result with that probability, "
            "like Stim's M(p); a NoiseModel's readout error is applied on top.")
        .def("measure_all",
             [](Circuits::Circuit<double> &s) {
                 size_t n = s.GetMaxQubitIndex() + 1;
                 std::vector<std::pair<Types::qubit_t, size_t>> pairs;
                 pairs.reserve(n);
                 for (size_t i = 0; i < n; ++i)
                     pairs.emplace_back(static_cast<Types::qubit_t>(i), i);
                 s.AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(pairs));
             })
        // Reset
        .def(
            "reset", [](Circuits::Circuit<double> &s, Types::qubit_t q) { s.AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{q})); },
            "qubit"_a, "Reset a qubit to ``|0>``.")
        .def(
            "reset_qubits",
            [](Circuits::Circuit<double> &s, const std::vector<Types::qubit_t> &qubits) { s.AddOperation(std::make_shared<Circuits::Reset<>>(qubits)); },
            "qubits"_a, "Reset multiple qubits to ``|0>``.")
        // Noise channel
        .def(
            "pauli_channel",
            [](Circuits::Circuit<double> &s, const Types::qubits_vector &qubits, const std::vector<double> &probabilities) {
                s.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, Simulators::QuantumChannel::Pauli(probabilities)));
            },
            "qubits"_a, "probabilities"_a,
            "Apply a Pauli channel to one or two qubits, e.g. a Stim DEPOLARIZE1.\n\n"
            "probabilities has 4**len(qubits) entries summing to one. Base-4 "
            "digit k of an index selects I, X, Y or Z on qubits[k], so one "
            "qubit is [pI, pX, pY, pZ] and two are II, XI, YI, ZI, IX, XX, ... "
            "ZZ.\n\n"
            "Density-matrix and MPO backends, and the QCSim Pauli propagator, "
            "apply the channel exactly. The statevector and the matrix product "
            "state (QCSim and Qiskit Aer) and the composite statevector draw one "
            "Pauli each time the circuit runs, in a plain execute as in "
            "noisy_execute or full_noise_execute. The other pure-state methods "
            "draw one Pauli per noise realization: run the circuit through "
            "noisy_execute or full_noise_execute with noise_realizations=shots; "
            "plain execute rejects it there.")
        .def(
            "channel",
            [](Circuits::Circuit<double> &s, const Types::qubits_vector &qubits, const Simulators::QuantumChannel &channel) {
                s.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, channel));
            },
            "qubits"_a, "channel"_a,
            "Apply a QuantumChannel to one or two qubits.\n\n"
            "The circuit shares the channel object, so build each distinct "
            "channel once and pass it at every place it acts.\n\n"
            "Density-matrix and MPO backends, and the QCSim Pauli propagator, "
            "apply the channel exactly. The statevector and the matrix product "
            "state (QCSim and Qiskit Aer) and the composite statevector sample "
            "it from their state: every execution draws one branch, so the shots "
            "average to the channel, in a plain execute as in noisy_execute or "
            "full_noise_execute. The other pure-state methods hold and sample no "
            "channel: a noise realization (noisy_execute, full_noise_execute) "
            "draws the channel into gates and resets, which needs every part of "
            "it to be drawn without the state (a Pauli channel, a one-qubit "
            "unitary mixture, relaxation with t2 <= t1); they reject any other "
            "channel.")
        .def("remove_end_measurements", &Circuits::Circuit<double>::RemoveEndMeasurements,
             "Create a new circuit holding shared pointers to the original operations, with terminal measurements removed.")

        // ---- Bound Methods for Direct Execution ----
        .def("execute", &execute_core, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "return_samples"_a = false)
        .def(
            "estimate",
            [](std::shared_ptr<Circuits::Circuit<double>> self, const nb::object &observables,
               const SimulatorConfig &config) { return estimate_core(self, ParseObservables(observables), config); },
            "observables"_a, "config"_a = SimulatorConfig{},
            "Expectation values of Pauli observables after the circuit (terminal measurements are ignored).\n\n"
            "Mid-circuit measurements and classical control select a branch per execution: such circuits average "
            "100 measurement trajectories (seeded ones derive a seed per trajectory) and report "
            "estimate_evaluation='trajectory_average', measurement_trajectories and realization_standard_errors. "
            "Other circuits are evaluated once (estimate_evaluation='deterministic').")
        .def(
            "get_statevector", [](std::shared_ptr<Circuits::Circuit<double>> self, const SimulatorConfig &config) { return statevector_core(self, config); },
            "config"_a = SimulatorConfig{},
            "Get the full statevector (complex amplitudes) after executing the "
            "circuit.")
        .def(
            "mirror_fidelity",
            [](std::shared_ptr<Circuits::Circuit<double>> self, const SimulatorConfig &config, int shots, bool full_amplitude,
               std::optional<nb::object> observables) { return mirror_fidelity_core(self, config, shots, full_amplitude, observables); },
            "config"_a = SimulatorConfig{}, "shots"_a = 1024, "full_amplitude"_a = false, "observables"_a = nb::none(),
            "Compute mirror fidelity: run circuit forward then its adjoint in "
            "reverse, returning ``P(|0...0>)``. Uses shot-based sampling by "
            "default. Set full_amplitude=True for exact statevector "
            "computation (small circuits only). Pass observables (a Pauli "
            "string, ';'-separated string or list) to instead return the "
            "expectation of each observable on the mirror circuit: a float "
            "for a single string, otherwise a list.")
        .def(
            "inner_product",
            [](std::shared_ptr<Circuits::Circuit<double>> self, std::shared_ptr<Circuits::Circuit<double>> other, const SimulatorConfig &config) {
                return inner_product_core(self, other, config);
            },
            "other"_a, "config"_a = SimulatorConfig{},
            "Compute the inner product ``<psi_self|psi_other> = <0|U_self^dag "
            "U_other|0>`` between this circuit's state and another circuit's "
            "state, using ProjectOnZero.")
        .def(
            "prob",
            [](std::shared_ptr<Circuits::Circuit<double>> self,
               const std::string &target_state) -> nb::
                                                    dict {
                                                        if (!self)
                                                            throw nb::value_error("Circuit is null.");
                                                        if (target_state.empty())
                                                            throw nb::value_error("target_state must be a non-empty bitstring.");

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
                                                            ok = sim.SetCircuit(self);
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
            "target_state"_a,
            "Compute the probability of a specific output state using the "
            "Pauli path integral simulator.\n\n"
            "Example: qc.prob('111') returns the probability of ``|111>``.\n\n"
            "Args:\n"
            "    target_state: Bitstring like '10001001' (qubit 0 leftmost).\n\n"
            "Returns:\n"
            "    dict with 'probability', 'amplitude', 'target_state', "
            "'time_taken'.")
        .def(
            "noisy_prob",
            [](std::shared_ptr<Circuits::Circuit<double>> self, const std::string &target_state, const noise::NoiseModel &noise_model) -> nb::dict {
                if (!self)
                    throw nb::value_error("Circuit is null.");
                if (target_state.empty())
                    throw nb::value_error("target_state must be a non-empty bitstring.");
                // The path integral is noiseless; only readout is modelled here.
                if (noise_model.has_non_readout_noise())
                    throw nb::value_error("noisy_prob models only readout error, but this NoiseModel has gate noise. "
                                          "Use noisy_execute or full_noise_execute to include it.");

                const size_t n = target_state.size();
                for (size_t i = 0; i < n; ++i)
                {
                    if (target_state[i] != '0' && target_state[i] != '1')
                        throw nb::value_error("target_state must contain only '0' and '1' characters.");
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
                        throw std::runtime_error("Circuit contains operations not supported by the path "
                                                 "integral simulator.");
                    auto amplitude = sim.AmplitudeFromZero(end_state);
                    return std::norm(amplitude);
                };

                auto start = std::chrono::high_resolution_clock::now();

                double p_noisy;

                if (noise_model.has_readout_error())
                {
                    // First-order readout error expansion, with b the target:
                    // P_noisy(b) ≈ Π_j(1-a_j)·P(b) + Σ_i c_i·Π_{j≠i}(1-a_j)·P(b⊕e_i)
                    // where a_i = P(read ¬b_i | prepared b_i) flips away from b_i,
                    // and c_i = P(read b_i | prepared ¬b_i) flips into it. The two
                    // differ for asymmetric readout error.
                    std::vector<double> p_flip_away(n, 0.0), p_flip_into(n, 0.0);
                    for (size_t i = 0; i < n; ++i)
                    {
                        const auto *re = noise_model.get_readout_error(static_cast<int>(i));
                        if (!re)
                            continue;
                        const bool zero = target_state[i] == '0';
                        p_flip_away[i] = zero ? re->p_meas1_prep0 : re->p_meas0_prep1;
                        p_flip_into[i] = zero ? re->p_meas0_prep1 : re->p_meas1_prep0;
                    }

                    // Π_{j≠skip}(1-a_j), computed directly so that a_i = 1 cannot
                    // produce 0/0 by dividing it back out.
                    auto keep_prob = [&](size_t skip) {
                        double prob = 1.0;
                        for (size_t j = 0; j < n; ++j)
                            if (j != skip)
                                prob *= 1.0 - p_flip_away[j];
                        return prob;
                    };

                    p_noisy = keep_prob(n) * pi_prob(target_state);

                    // 1-flip terms: qubit i was prepared as ¬b_i and read as b_i
                    for (size_t i = 0; i < n; ++i)
                    {
                        if (p_flip_into[i] <= 0.0)
                            continue;
                        std::string flipped = target_state;
                        flipped[i] = (flipped[i] == '0') ? '1' : '0';
                        p_noisy += p_flip_into[i] * keep_prob(i) * pi_prob(flipped);
                    }
                }
                else
                {
                    // No readout error — just compute exact probability
                    p_noisy = pi_prob(target_state);
                }

                auto end = std::chrono::high_resolution_clock::now();

                nb::dict result;
                result["probability"] = p_noisy;
                result["target_state"] = target_state;
                result["time_taken"] = std::chrono::duration<double>(end - start).count();
                result["has_readout_error"] = noise_model.has_readout_error();
                return result;
            },
            "target_state"_a, "noise_model"_a,
            "Compute readout-corrected probability using path integral.\n\n"
            "When the noise model has readout error, uses first-order expansion:\n"
            "P_noisy(b) ≈ Π_j(1-a_j)·P(b) + Σ_i c_i·Π_{j≠i}(1-a_j)·P(b⊕eᵢ)\n"
            "where a_i = P(read ¬b_i | prepared b_i) and "
            "c_i = P(read b_i | prepared ¬b_i).\n\n"
            "This requires n+1 path integral evaluations (target + n flipped "
            "variants).\n\n"
            "Args:\n"
            "    target_state: Bitstring like '10001001'.\n"
            "    noise_model: NoiseModel with readout error set. Any other "
            "(gate) noise raises ValueError, since the path integral is "
            "noiseless.\n\n"
            "Returns:\n"
            "    dict with 'probability', 'target_state', 'time_taken', "
            "'has_readout_error'.")

        // ---- Bound Methods for Noisy Execution ----
        .def("noisy_execute", &NoisyExecute, "noise_model"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "noise_realizations"_a = nb::none(),
             "noise_seed"_a = nb::none(), "return_samples"_a = false,
             "Execute with exact Pauli/T1 channels for density-matrix/MPO "
             "methods, exact Pauli channels on the QCSim Pauli propagator "
             "(which samples T1), or sampled trajectories for pure-state methods.\n\n"
             "Example: qc.noisy_execute(nm, shots=1000)" MAESTRO_NOISY_EXECUTE_DOC)
        .def("noisy_estimate", &NoisyEstimate,
            "observables"_a, "noise_model"_a, "config"_a = SimulatorConfig{},
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
            "Only the all-gates Pauli layer is modelled; a NoiseModel with any "
            "other layer (T1, thermal, coherent, correlated, gate-type or "
            "two-qubit noise) raises ValueError, see "
            "compute_damping_covers_model(). Use noisy_estimate_montecarlo(), "
            "full_noise_estimate(), or noisy_execute() with a density-matrix "
            "or MPO method, when the magnitude of the noise matters."
            "\n\n"
            "Example: qc.noisy_estimate(['ZZ', 'XX'], nm)")
        .def("noisy_estimate_montecarlo", &NoisyEstimateMonteCarlo, "observables"_a, "noise_model"_a, "noise_realizations"_a = nb::none(),
             "config"_a = SimulatorConfig{}, "noise_seed"_a = nb::none(),
             "Gate-by-gate Monte Carlo noisy estimation.\n\n"
             "Example: qc.noisy_estimate_montecarlo(['ZZ'], nm, "
             "noise_realizations=200)")
        .def("coherent_execute", &CoherentExecute, "noise_model"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "noise_realizations"_a = nb::none(),
             "noise_seed"_a = nb::none(), "return_samples"_a = false,
             "Execute with sampled coherent over/under-rotation errors.\n\n"
             "Example: qc.coherent_execute(nm, shots=1000)" MAESTRO_NOISY_EXECUTE_DOC)
        .def("coherent_estimate", &CoherentEstimate, "observables"_a, "noise_model"_a, "noise_realizations"_a = nb::none(), "config"_a = SimulatorConfig{},
             "noise_seed"_a = nb::none(),
             "Estimate expectation values with coherent noise.\n\n"
             "Example: qc.coherent_estimate(['ZZ', 'XX'], nm, "
             "noise_realizations=200)")
        // ---- Combined Noise (all layers) ----
        .def("full_noise_execute", &FullNoiseExecute, "noise_model"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024, "noise_realizations"_a = nb::none(),
             "noise_seed"_a = nb::none(), "return_samples"_a = false,
             "Execute with combined noise: coherent + crosstalk + T1 + Pauli.\n\n"
             "DM/MPO methods use exact channels for T1 and Pauli layers, and "
             "the QCSim Pauli propagator for the Pauli layers (it samples T1); "
             "trajectory-only layers remain sampled.\n"
             "Example: qc.full_noise_execute(nm, shots=1000)" MAESTRO_NOISY_EXECUTE_DOC)
        .def("full_noise_estimate", &FullNoiseEstimate, "observables"_a, "noise_model"_a, "noise_realizations"_a = nb::none(), "config"_a = SimulatorConfig{},
             "noise_seed"_a = nb::none(),
             "Estimate with combined noise (coherent + crosstalk + T1 + Pauli).\n\n"
             "Example: qc.full_noise_estimate(['ZZ', 'XX'], nm)")
        // ---- Noisy Fidelity (inner-product) ----
        .def("noisy_fidelity", &NoisyFidelity,
            "noise_model"_a, "noise_realizations"_a = nb::none(), "config"_a = SimulatorConfig{}, "noise_seed"_a = nb::none(),
            "Compute fidelity to the ideal unitary circuit state under noise.\n\n"
            "Injects all configured noise types (correlated, coherent, "
            "crosstalk, T1, Pauli) and averages ``|<psi_ideal|psi_noisy>|^2`` "
            "over noise realizations.\n\n"
            "Returns dict with 'fidelity', 'infidelity', 'std_error', "
            "'time_taken', 'noise_realizations'.\n\n"
            "Example: qc.noisy_fidelity(nm, noise_realizations=200)");
}

} // namespace maestro_bindings
