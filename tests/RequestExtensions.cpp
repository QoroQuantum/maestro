// Native request contract for instruction-level noise, two-qubit rotations,
// observable mirror fidelity and adjoint differentiation. No Python or GPU.
#include <boost/json.hpp>
#include <cmath>
#include <string>
namespace j = boost::json;
void Check(bool, const char *);
void Near(double, double);
double Real(const j::value &);
j::object Request(const char *, size_t, const std::string &, const char * = "statevector", size_t = 0);
j::object Call(const j::object &, bool = true, bool = false);

namespace
{
j::object Instructions(const char *operation, size_t qubits, j::array source, const char *method = "statevector")
{
    auto request = Request(operation, qubits, "", method);
    request["circuit"] = j::object{{"format", "instructions"}, {"num_qubits", qubits}, {"source", std::move(source)}};
    return request;
}

j::object Gate(const char *name, j::array qubits, j::array params = {})
{
    j::object gate{{"name", name}, {"qubits", std::move(qubits)}};
    if (!params.empty())
        gate["params"] = std::move(params);
    return gate;
}

std::string ErrorCode(const j::object &response)
{
    return std::string(response.at("error").at("code").as_string().c_str());
}

uint64_t Count(const j::object &response, const char *bits)
{
    const auto &counts = response.at("counts").as_object();
    const auto *value = counts.if_contains(bits);
    return value ? value->to_number<uint64_t>() : 0;
}

void TestTwoQubitRotationInstructions()
{
    for (const char *name : {"rzz", "rxx"})
    {
        auto instructions = Instructions("estimate", 2, {Gate("h", {0}), Gate("ry", {1}, {0.4}), Gate(name, {0, 1}, {0.7})});
        auto qasm = Request("estimate", 2, std::string("h q[0]; ry(0.4) q[1]; ") + name + "(0.7) q[0],q[1];");
        const j::array observables{"XX", "YY", "ZZ", "XI", "IZ", "YZ"};
        instructions["observables"] = observables;
        qasm["observables"] = observables;
        const auto left = Call(instructions).at("expectation_values").as_array();
        const auto right = Call(qasm).at("expectation_values").as_array();
        for (size_t i = 0; i < left.size(); ++i)
            Near(Real(left[i]), Real(right[i]));
        // Conditional rotations expand into conditional gates.
        auto conditional = Gate(name, {0, 1}, {0.7});
        conditional["conditional_reg"] = 0;
        Call(Instructions("execute", 2, {conditional, Gate("measure", {0, 1})}));
    }
    auto request = Instructions("estimate", 2, {Gate("rzz", {0, 1})});
    request["observables"] = j::array{"ZZ"};
    Check(ErrorCode(Call(request, false)) == "invalid_input", "rzz without an angle was accepted");
    request = Instructions("estimate", 2, {Gate("rxx", {0}, {0.1})});
    request["observables"] = j::array{"ZZ"};
    Check(ErrorCode(Call(request, false)) == "invalid_input", "One-qubit rxx was accepted");
}

void TestInstructionNoiseFields()
{
    // Per-measurement readout flips recorded results, symmetrically or not.
    auto measure = Gate("measure", {0});
    measure["readout_error"] = 1;
    Check(Count(Call(Instructions("execute", 1, {measure})), "1") == 80, "Symmetric readout_error was not applied");
    measure["readout_error"] = j::object{{"p_meas1_prep0", 0}, {"p_meas0_prep1", 1}};
    Check(Count(Call(Instructions("execute", 1, {Gate("x", {0}), measure})), "0") == 80, "Asymmetric readout_error was not applied");
    measure["readout_error"] = j::object{{"p_meas1_prep0", 0}, {"p_meas0_prep1", 0}};
    Check(Count(Call(Instructions("execute", 1, {Gate("x", {0}), measure})), "1") == 80, "Zero readout_error changed counts");
    // A NoiseModel readout composes with the measurement's own rates.
    measure["readout_error"] = 1;
    auto composed = Instructions("execute", 1, {measure});
    composed["noise"] = j::object{{"channels", j::array{j::object{{"kind", "readout"}, {"targets", j::array{0}}, {"probability", 1}}}}};
    Check(Count(Call(composed), "0") == 80, "Model and measurement readout did not compose");

    measure["readout_error"] = 0.5;
    auto estimate = Instructions("estimate", 1, {measure});
    estimate["observables"] = j::array{"Z"};
    Check(ErrorCode(Call(estimate, false)) == "unsupported_capability", "Readout on an estimate was silently ignored");
    for (const auto &value : {j::value(1.5), j::value(-0.1), j::value("0.1"), j::value(j::object{{"p_meas1_prep0", 0.1}})})
    {
        measure["readout_error"] = value;
        Check(ErrorCode(Call(Instructions("execute", 1, {measure}), false)) == "invalid_input", "Invalid readout_error was accepted");
    }
    auto gate = Gate("x", {0});
    gate["readout_error"] = 0.1;
    Check(ErrorCode(Call(Instructions("execute", 1, {gate}), false)) == "invalid_input", "readout_error on a gate was accepted");

    // Noise models skip noiseless gates, including expanded rotations.
    const j::object flip{{"realizations", 4}, {"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 1}}}}};
    auto noisy = Instructions("execute", 1, {Gate("x", {0}), Gate("measure", {0})});
    noisy["noise"] = flip;
    Check(Count(Call(noisy), "0") == 80, "Gate noise was not applied");
    gate = Gate("x", {0});
    gate["noiseless"] = true;
    noisy = Instructions("execute", 1, {gate, Gate("measure", {0})});
    noisy["noise"] = flip;
    Check(Count(Call(noisy), "1") == 80, "A noiseless gate received gate noise");
    gate["noiseless"] = false;
    noisy = Instructions("execute", 1, {gate, Gate("measure", {0})});
    noisy["noise"] = flip;
    Check(Count(Call(noisy), "0") == 80, "noiseless=false exempted a gate");
    measure = Gate("measure", {0});
    measure["noiseless"] = true;
    Check(ErrorCode(Call(Instructions("execute", 1, {measure}), false)) == "invalid_input", "noiseless on a measurement was accepted");
    gate["noiseless"] = 1;
    Check(ErrorCode(Call(Instructions("execute", 1, {gate}), false)) == "invalid_input", "A non-boolean noiseless was accepted");
}

void TestPauliChannelInstructions()
{
    const auto channel = [](j::array probabilities, j::array qubits = {0}) {
        return j::object{{"name", "pauli_channel"}, {"qubits", std::move(qubits)}, {"probabilities", std::move(probabilities)}};
    };
    // Mixed-state methods apply the channel exactly, without a noise block.
    for (const char *method : {"density_matrix", "matrix_product_operator"})
    {
        auto estimate = Instructions("estimate", 1, {channel({0.25, 0.25, 0.25, 0.25})}, method);
        estimate["observables"] = j::array{"Z"};
        Near(Real(Call(estimate).at("expectation_values").at(0)), 0);
        Check(Count(Call(Instructions("execute", 1, {channel({0, 1, 0, 0}), Gate("measure", {0})}, method)), "1") == 80, "Exact Pauli channel was not applied");
    }
    // Pure-state methods sample one Pauli per realization from a trajectory noise
    // block, which then needs no model channels of its own.
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        auto execute = Instructions("execute", 1, {channel({0.5, 0.5, 0, 0}), Gate("measure", {0})}, method);
        Check(ErrorCode(Call(execute, false)) == "unsupported_capability", "Pure-state channel ran without sampling");
        execute["noise"] = j::object{{"realizations", 80}};
        const auto response = Call(execute);
        Check(Count(response, "0") + Count(response, "1") == 80 && Count(response, "0") > 20 && Count(response, "1") > 20,
              "Sampled Pauli channel did not mix outcomes");
        Check(response.at("noise").at("evaluation") == "trajectories", "Pauli sampling was not reported as trajectories");
    }
    // Two-qubit channel order: base-4 digit k selects the Pauli on qubits[k].
    j::array xi(16, 0.0);
    xi[1] = 1.0; // XI: X on qubits[0]
    Check(Count(Call(Instructions("execute", 2, {channel(xi, {1, 0}), Gate("measure", {0, 1})}, "density_matrix")), "01") == 80,
          "Two-qubit Pauli channel order changed");

    for (const auto &bad : {j::array{0.5, 0.6, 0, 0}, j::array{1, 0, 0}, j::array{-0.1, 1.1, 0, 0}})
        Check(ErrorCode(Call(Instructions("execute", 1, {channel(bad)}, "density_matrix"), false)) == "invalid_input",
              "Invalid Pauli probabilities were accepted");
    auto conditional = channel({1, 0, 0, 0});
    conditional["conditional_reg"] = 0;
    Check(ErrorCode(Call(Instructions("execute", 1, {conditional}, "density_matrix"), false)) == "invalid_input", "Conditional channel was accepted");
    // General Kraus channels still need exact mixed-state evaluation.
    auto kraus = Instructions("execute", 1, {j::object{{"name", "kraus"}, {"qubits", j::array{0}}, {"operators", j::array{j::array{0, 1, 1, 0}}}}});
    kraus["noise"] = j::object{{"realizations", 4}};
    Check(ErrorCode(Call(kraus, false)) == "unsupported_capability", "A Kraus channel was sampled on a pure state");
    // Exact model noise keeps circuit channels: bit flip 0.1 after x, then a
    // Kraus flip with probability 0.36, gives <Z> = -(0.8)(0.28).
    for (const char *method : {"density_matrix", "matrix_product_operator"})
    {
        auto exact =
            Instructions("estimate", 1,
                         {Gate("x", {0}),
                          j::object{{"name", "kraus"}, {"qubits", j::array{0}}, {"operators", j::array{j::array{0.8, 0, 0, 0.8}, j::array{0, 0.6, 0.6, 0}}}}},
                         method);
        exact["observables"] = j::array{"Z"};
        exact["noise"] =
            j::object{{"evaluation", "exact"}, {"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 0.1}}}}};
        Near(Real(Call(exact).at("expectation_values").at(0)), -0.8 * 0.28);
    }
    // An empty noise model still needs channels from somewhere.
    auto empty = Instructions("execute", 1, {Gate("x", {0}), Gate("measure", {0})});
    empty["noise"] = j::object{{"realizations", 4}};
    Check(ErrorCode(Call(empty, false)) == "invalid_input", "An empty noise block without circuit channels was accepted");
    empty["noise"] = j::object{{"channels", j::array{}}};
    Check(ErrorCode(Call(empty, false)) == "invalid_input", "An empty channel list without circuit channels was accepted");
}

void TestObservableMirrorFidelity()
{
    const std::string body = "h q[0]; cx q[0],q[1]; rx(0.3) q[1]; ry(0.8) q[0];";
    auto request = Request("mirror_fidelity", 2, body);
    Near(Real(Call(request).at("fidelity")), 1);
    request["observables"] = j::array{"ZZ", "ZI", "XX", "IZ"};
    auto response = Call(request);
    const auto &values = response.at("expectation_values").as_array();
    Check(values.size() == 4 && !response.contains("fidelity"), "Observable mirror fidelity has the wrong shape");
    Near(Real(values[0]), 1);
    Near(Real(values[1]), 1);
    Near(Real(values[2]), 0);
    Near(Real(values[3]), 1);
    request = Request("mirror_fidelity", 2, body, "matrix_product_state");
    request["observables"] = j::array{"ZZ"};
    Near(Real(Call(request).at("expectation_values").at(0)), 1);
    request["observables"] = j::array{"ZZZ"};
    Check(ErrorCode(Call(request, false)) == "invalid_input", "A mismatched mirror observable was accepted");
}

double Expectation(const std::string &body, const char *observable)
{
    auto request = Request("estimate", 2, body);
    request["observables"] = j::array{observable};
    return Real(Call(request).at("expectation_values").at(0));
}

void TestAdjointDifferentiation()
{
    const auto body = [](double a, double b, double c) {
        return "ry(" + std::to_string(a) + ") q[0]; cx q[0],q[1]; rz(" + std::to_string(b) + ") q[1]; rx(" + std::to_string(c) + ") q[1];";
    };
    const double params[3] = {0.4, 0.2, 0.9};
    auto request = Request("adjoint_jacobian", 2, body(params[0], params[1], params[2]));
    request["observables"] = j::array{"ZI", j::object{{"terms", j::array{j::array{0.5, "ZZ"}, j::array{2, "XI"}}}}, j::object{{"probabilities", j::array{1}}}};
    const auto response = Call(request);
    Check(response.at("num_parameters").to_number<int>() == 3 && response.at("num_values").to_number<int>() == 4, "Adjoint shape metadata changed");
    Check(response.at("circuit_parameters").as_array().size() == 3, "Circuit parameters were not reported");
    const auto &jacobian = response.at("jacobian").at(0).as_array();
    // Central differences of the native estimate agree with the adjoint sweep.
    const double h = 1e-5;
    for (size_t k = 0; k < 3; ++k)
    {
        double plus[3] = {params[0], params[1], params[2]}, minus[3] = {params[0], params[1], params[2]};
        plus[k] += h;
        minus[k] -= h;
        const auto fd = [&](const char *pauli) {
            return (Expectation(body(plus[0], plus[1], plus[2]), pauli) - Expectation(body(minus[0], minus[1], minus[2]), pauli)) / (2 * h);
        };
        Check(std::abs(fd("ZI") - Real(jacobian[0].at(k))) < 1e-6, "Adjoint Jacobian differs from finite differences");
        Check(std::abs(0.5 * fd("ZZ") + 2 * fd("XI") - Real(jacobian[1].at(k))) < 1e-6, "Weighted observable Jacobian differs");
        // P(q1=1) = (1 - <IZ>) / 2.
        Check(std::abs(-0.5 * fd("IZ") - Real(jacobian[3].at(k))) < 1e-6, "Probability Jacobian differs");
    }

    request = Request("adjoint_expectation_values", 2, body(params[0], params[1], params[2]));
    request["observables"] = j::array{"ZI"};
    request["parameter_rows"] = j::array{j::array{0, 0, 0}, j::array{std::acos(-1.0), 0, 0}};
    const auto rows = Call(request).at("expectation_values").as_array();
    Near(Real(rows[0].at(0)), 1);
    Near(Real(rows[1].at(0)), -1);

    request = Request("adjoint_vjp", 2, body(params[0], params[1], params[2]));
    request["observables"] = j::array{"ZI", j::object{{"terms", j::array{j::array{0.5, "ZZ"}, j::array{2, "XI"}}}}};
    request["cotangents"] = j::array{j::array{2, -1}};
    const auto vjp = Call(request).at("vjp").at(0).as_array();
    for (size_t k = 0; k < 3; ++k)
        Near(Real(vjp[k]), 2 * Real(jacobian[0].at(k)) - Real(jacobian[1].at(k)));

    // Validation reports shapes; malformed requests fail before computation.
    request = Request("adjoint_jacobian", 2, body(params[0], params[1], params[2]));
    request["observables"] = j::array{"ZI", j::object{{"probabilities", j::array{0, 1}}}};
    const auto validated = Call(request, true, true);
    Check(validated.at("valid").as_bool() && validated.at("num_values").to_number<int>() == 5, "Adjoint validation lost its shape");
    const auto rejected = [&](j::object document, const char *code, const char *message) {
        Check(ErrorCode(Call(document, false, true)) == code, message);
        Check(ErrorCode(Call(document, false)) == code, message);
    };
    auto bad = request;
    bad["parameter_rows"] = j::array{j::array{1, 2}};
    rejected(bad, "invalid_input", "A short parameter row was accepted");
    bad = request;
    bad["operation"] = "adjoint_vjp";
    rejected(bad, "invalid_input", "A VJP without cotangents was accepted");
    bad["cotangents"] = j::array{j::array{1, 0}};
    rejected(bad, "invalid_input", "Short cotangents were accepted");
    bad = request;
    bad["max_output_elements"] = 4;
    rejected(bad, "invalid_input", "Adjoint output ignored max_output_elements");
    bad = request;
    bad["observables"] = j::array{j::object{{"probabilities", j::array{0, 0}}}};
    rejected(bad, "invalid_input", "Repeated probability qubits were accepted");
    bad["observables"] = j::array{j::object{{"terms", j::array{}}}};
    rejected(bad, "invalid_input", "An empty weighted observable was accepted");
    bad["observables"] = j::array{j::object{{"terms", j::array{j::array{1, "Z"}}}}};
    rejected(bad, "invalid_input", "A short Pauli term was accepted");
    bad = request;
    bad["cotangents"] = j::array{j::array{1}};
    rejected(bad, "invalid_input", "cotangents were accepted by a Jacobian");
    bad = request;
    bad["simulator"].as_object()["method"] = "matrix_product_state";
    rejected(bad, "unsupported_capability", "Adjoint differentiation accepted MPS");
    bad = request;
    bad["noise"] = j::object{{"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 0.1}}}}};
    rejected(bad, "unsupported_capability", "Adjoint differentiation accepted noise");
    bad = Request("adjoint_jacobian", 2, "measure q[0] -> c[0]; rx(0.1) q[0];");
    bad["observables"] = j::array{"ZI"};
    rejected(bad, "invalid_input", "Adjoint differentiation accepted a mid-circuit measurement");
    bad = request;
    bad["operation"] = "estimate";
    bad["parameter_rows"] = j::array{j::array{1, 2, 3}};
    bad["observables"] = j::array{"ZI"};
    rejected(bad, "invalid_input", "parameter_rows was accepted by estimate");

    // Terminal measurements are ignored, as for estimate, and batches work.
    request = Request("adjoint_jacobian", 1, "rx(0.1) q[0]; measure q[0] -> c[0];");
    request["observables"] = j::array{"Z"};
    Near(Real(Call(request).at("jacobian").at(0).at(0).at(0)), -std::sin(0.1));
    const j::object batch{{"schema_version", 2}, {"operation", "batch"}, {"requests", j::array{request}}};
    Near(Real(Call(batch).at("results").at(0).at("expectation_values").at(0).at(0)), std::cos(0.1));
}
} // namespace

void TestNativeExtensions(const j::object &capabilities)
{
    bool adjoint = false;
    for (const auto &operation : capabilities.at("operations").as_array())
        adjoint = adjoint || operation == "adjoint_jacobian";
    Check(adjoint && capabilities.at("adjoint").at("max_qubits").to_number<int>() == 63, "Adjoint operations were not advertised");
    bool rzz = false;
    for (const auto &gate : capabilities.at("instructions").at("gates").as_array())
        rzz = rzz || (gate.at("name") == "rzz" && gate.at("params").to_number<int>() == 1);
    Check(rzz && capabilities.at("instructions").at("fields").at("readout_error") == j::array{"measure"}, "Instruction catalog is incomplete");
    TestTwoQubitRotationInstructions();
    TestInstructionNoiseFields();
    TestPauliChannelInstructions();
    TestObservableMirrorFidelity();
    TestAdjointDifferentiation();
}
