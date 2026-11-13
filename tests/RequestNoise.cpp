// Numerical contract tests through the public request API, without Python/GPU.
#include "../maestrolib/Interface.h"
#include <array>
#include <boost/json.hpp>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace j = boost::json;
void Check(bool, const char *);
void Near(double, double);
double Real(const j::value &);
j::object Request(const char *, size_t, const std::string &, const char * = "statevector", size_t = 0);
j::object Call(const j::object &, bool = true, bool = false);

namespace
{
j::object Estimate(const std::string &body, j::array observables, j::object channel, const char *method = "density_matrix", size_t qubits = 1)
{
    auto request = Request("estimate", qubits, body, method);
    request["observables"] = observables;
    request["noise"] = j::object{{"channels", j::array{channel}}, {"seed", 23}};
    return request;
}

j::object Channel(const char *kind, j::object parameters, j::array targets = {0})
{
    parameters["kind"] = kind;
    parameters["targets"] = targets;
    return parameters;
}

void SameExpectations(const j::object &a, const j::object &b)
{
    const auto &left = a.at("expectation_values").as_array();
    const auto &right = b.at("expectation_values").as_array();
    Check(left.size() == right.size(), "Expectation shape changed");
    for (size_t i = 0; i < left.size(); ++i)
        Near(Real(left[i]), Real(right[i]));
}

j::object ThermalCall(const j::object &request, bool sampled, bool validate = false)
{
    struct Capture
    {
        std::ostringstream output;
        std::streambuf *previous = std::cerr.rdbuf(output.rdbuf());

        ~Capture()
        {
            std::cerr.rdbuf(previous);
        }
    } capture;

    const auto result = Call(request, true, validate);
    const auto warning = capture.output.str();
    if (validate || !sampled)
    {
        Check(warning.empty(), "Validation/exact thermal execution warned");
    }
    else
    {
        const std::string expected = "Sampled thermal approximation for circuit qubits [0]: effective T2 "
                                     "clamped to T1.";
        const auto first = warning.find(expected);
        Check(first != std::string::npos, "Missing sampled thermal warning");
        Check(warning.find(expected, first + expected.size()) == std::string::npos, "Thermal warning repeated for each realization");
    }
    if (!validate)
    {
        bool marked = false;
        for (const auto &value : result.at("noise").at("approximations").as_array())
            marked |= value == "thermal_T2_clamped_to_T1";
        Check(marked == sampled, "Incorrect thermal approximation metadata");
    }
    return result;
}

void TestThermalApproximation()
{
    // Relaxation with T2 > T1 has no branch drawn without the state. A method
    // that neither holds nor samples channels draws it as an explicit T2=T1
    // model does, and says so. The statevector and the matrix product state
    // sample the channel itself and the mixed-state methods apply it: those
    // keep the supplied T2, including the physical limit.
    for (const char *method : {"tensor_network", "path_integral", "statevector", "matrix_product_state", "density_matrix", "matrix_product_operator"})
    {
        const std::string name(method);
        const bool clamped = name == "tensor_network" || name == "path_integral";
        const bool trajectories = name == "statevector" || name == "matrix_product_state";
        for (const char *kind : {"thermal_relaxation", "thermal_relaxation_2q", "idle"})
        {
            for (const double t2 : {1.5, 2.0})
            {
                auto channel = Channel(kind, {{"t1", 1.0}, {"t2", t2}});
                if (std::string(kind) != "idle")
                    channel["duration"] = 0.3;
                const std::string body = std::string(kind) == "thermal_relaxation_2q" ? "h q[0]; cx q[1],q[0];" : "h q[0];";
                auto request = Estimate(body, {"XI"}, channel, method, 2);
                if (std::string(kind) == "idle")
                    request["circuit"] = j::object{{"format", "instructions"},
                                                   {"num_qubits", 2},
                                                   {"source", j::array{j::object{{"name", "h"}, {"qubits", j::array{0}}},
                                                                       j::object{{"name", "delay"}, {"qubits", j::array{0}}, {"duration", 0.3}}}}};
                request["noise"].as_object()["realizations"] = clamped ? 256 : trajectories ? 1024 : 1;
                ThermalCall(request, clamped, true);
                const auto result = ThermalCall(request, clamped);
                if (clamped)
                {
                    SameExpectations(result, ThermalCall(request, true));
                    auto explicitT2 = request;
                    explicitT2["noise"].as_object()["channels"].as_array()[0].as_object()["t2"] = 1.0;
                    SameExpectations(result, ThermalCall(explicitT2, false));
                    Check(std::abs(Real(result.at("expectation_values").at(0)) - std::exp(-0.3)) < 0.1,
                          "Sampled thermal coherence did not follow effective T2=T1");
                }
                else if (trajectories)
                {
                    // Trajectories of the channel: the calibrated coherence
                    // (0.82 or 0.86), not that of the clamp (0.74).
                    SameExpectations(result, ThermalCall(request, false));
                    Check(result.at("noise").at("evaluation") == "trajectories", "A pure-state method did not sample its relaxation");
                    Check(std::abs(Real(result.at("expectation_values").at(0)) - std::exp(-0.3 / t2)) < 0.05,
                          "Sampled thermal channels did not keep the calibrated T2");
                }
                else
                {
                    Near(Real(result.at("expectation_values").at(0)), std::exp(-0.3 / t2));
                }
            }
        }
    }

    // The clamp is reported only where the circuit draws it: idle relaxation
    // without delays, and a zero-duration gate relaxation (the identity), are
    // not approximated.
    {
        auto idle = Estimate("h q[0];", {"XI"}, Channel("idle", {{"t1", 1.0}, {"t2", 2.0}}), "tensor_network", 2);
        idle["noise"].as_object()["realizations"] = 256;
        ThermalCall(idle, false);
        auto instant = Estimate("h q[0];", {"XI"}, Channel("thermal_relaxation", {{"duration", 0.0}, {"t1", 1.0}, {"t2", 2.0}}), "tensor_network", 2);
        instant["noise"].as_object()["realizations"] = 256;
        ThermalCall(instant, false);
    }

    // No pure-state method evaluates noise exactly. Maps that have no branch
    // drawn without the state are refused where they cannot be sampled, and
    // accepted by the methods that sample channels.
    auto thermal = Channel("thermal_relaxation", {{"duration", 0.3}, {"t1", 1.0}, {"t2", 1.5}});
    auto request = Estimate("h q[0];", {"XI"}, thermal, "statevector", 2);
    request["noise"].as_object()["evaluation"] = "exact";
    Check(Call(request, false, true).at("error").at("code") == "unsupported_capability", "Exact thermal evolution accepted on a pure-state method");
    request["noise"].as_object().erase("evaluation");
    for (const auto &channel : j::array{Channel("generalized_amplitude_damping", {{"gamma", 0.2}, {"excited_population", 0.1}}),
                                        Channel("correlated_phase_flip", {{"probability", 0.2}}, {0, 1}),
                                        Channel("kraus", {{"operators", j::array{j::value(j::array{1, 0, 0, 1})}}})})
    {
        request["noise"].as_object()["channels"] = j::array{thermal, channel};
        for (const char *method : {"tensor_network", "path_integral", "extended_stabilizer"})
        {
            request["simulator"].as_object()["method"] = method;
            Check(Call(request, false, true).at("error").at("code") == "unsupported_capability",
                  "A channel that needs the state was accepted by a method that cannot sample it");
        }
        for (const char *method : {"statevector", "matrix_product_state"})
        {
            request["simulator"].as_object()["method"] = method;
            Check(Call(request, true, true).at("valid").as_bool(), "A method that samples channels refused one");
        }
    }
}

// The statevector and the matrix product state sample channels from their
// state. Noise that has no branch drawn without it (relaxation with T2 > T1,
// amplitude damping, generalized amplitude damping, Kraus maps) runs there as
// trajectories of the channels themselves, in a model or as an instruction:
// nothing is clamped, drawn as a reset or refused.
void TestSampledChannelRequests()
{
    const auto approximated = [](const j::object &result, const char *name) {
        const auto &list = result.at("noise").at("approximations").as_array();
        return std::find(list.begin(), list.end(), j::value(name)) != list.end();
    };
    const double keep = std::sqrt(0.7), decay = std::sqrt(0.3);
    const j::array damp{j::array{1, 0, 0, keep}, j::array{0, decay, 0, 0}};

    // A model, against the density matrix.
    const j::array model{Channel("thermal_relaxation", {{"duration", 0.3}, {"t1", 1.0}, {"t2", 1.6}, {"excited_population", 0.1}}, {0}),
                         Channel("generalized_amplitude_damping", {{"gamma", 0.15}, {"excited_population", 0.2}}, {1}), Channel("t1", {{"gamma", 0.1}}, {2}),
                         Channel("kraus", {{"operators", damp}}, {1}), Channel("depolarizing", {{"probability", 0.02}}, {0, 1, 2})};
    const j::array observables{"XII", "ZZI", "IZZ", "IIZ", "XXI", "ZIZ"};
    const auto estimate = [&](const char *method, size_t realizations) {
        auto request = Request("estimate", 3, "h q[0]; cx q[0],q[1]; ry(0.8) q[2]; cx q[1],q[2]; h q[1];", method);
        request["observables"] = observables;
        request["noise"] = j::object{{"seed", 29}, {"channels", model}};
        if (realizations)
            request["noise"].as_object()["realizations"] = realizations;
        return request;
    };
    const auto exact = Call(estimate("density_matrix", 0));
    Check(exact.at("noise").at("evaluation") == "exact_channels", "The reference did not apply its channels exactly");
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        Check(Call(estimate(method, 2000), true, true).at("valid").as_bool(), "A sampled model did not validate");
        const auto result = Call(estimate(method, 2000));
        Check(result.at("noise").at("evaluation") == "trajectories" && !approximated(result, "thermal_T2_clamped_to_T1") &&
                  !approximated(result, "sampled_T1_reset") && approximated(result, "finite_noise_realizations"),
              "Channels sampled from the state were reported as approximated");
        for (size_t i = 0; i < observables.size(); ++i)
            Check(std::abs(Real(result.at("expectation_values").at(i)) - Real(exact.at("expectation_values").at(i))) < 0.1,
                  "Sampled channels do not average to the density matrix");
        SameExpectations(result, Call(estimate(method, 2000)));
    }

    // Counts of that model. The backend draws all the noise as each shot runs,
    // so the shots share one circuit, built once, and each is a realization of
    // its own whatever number was asked for. Both count paths, against the
    // density matrix: the probability of each bit, and of the first two agreeing.
    const auto counted = [&](const char *method, bool reduction, uint64_t seed, uint64_t noiseSeed, j::value realizations) {
        auto request = Request("execute", 3, "h q[0]; cx q[0],q[1]; ry(0.8) q[2]; cx q[1],q[2]; h q[1]; measure q->c;", method);
        request["execution"].as_object()["shots"] = 8192;
        request["execution"].as_object()["seed"] = seed;
        request["noise"] = j::object{{"seed", noiseSeed}, {"channels", model}};
        if (!realizations.is_null())
            request["noise"].as_object()["realizations"] = realizations;
        if (reduction)
            request["simulator"].as_object()["options"] = j::object{{"enable_sampling_reduction", true}};
        return Call(request);
    };
    const auto marginals = [](const j::object &result) {
        std::array<double, 4> sums{};
        double shots = 0;
        for (const auto &entry : result.at("counts").as_object())
        {
            const std::string bits(entry.key());
            const double count = Real(entry.value());
            shots += count;
            for (size_t bit = 0; bit < 3; ++bit)
                sums[bit] += bits[bit] == '1' ? count : 0.0;
            sums[3] += bits[0] == bits[1] ? count : 0.0;
        }
        for (double &value : sums)
            value /= shots;
        return sums;
    };
    const auto exactCounts = marginals(counted("density_matrix", false, 123, 29, nullptr));
    for (const char *method : {"statevector", "matrix_product_state"})
        for (const bool reduction : {false, true})
        {
            const auto result = counted(method, reduction, 123, 29, nullptr);
            Check(result.at("noise").at("realizations").to_number<size_t>() == 8192 && result.at("noise").at("evaluation") == "trajectories",
                  "A count request that draws its noise as it runs did not report one realization per shot");
            const auto sampled = marginals(result);
            for (size_t i = 0; i < sampled.size(); ++i)
                Check(std::abs(sampled[i] - exactCounts[i]) < 0.03, "Counts drawn as the circuit runs do not follow the density matrix");
            // The number asked for changes nothing; either seed changes the draws.
            Check(result.at("counts") == counted(method, reduction, 123, 29, nullptr).at("counts"), "Seeded run-time noise is not reproducible");
            Check(result.at("counts") == counted(method, reduction, 123, 29, 7).at("counts"), "The realization count changed noise drawn per shot");
            Check(result.at("counts") != counted(method, reduction, 123, 30, nullptr).at("counts"), "The noise seed does not reach noise drawn per shot");
            Check(result.at("counts") != counted(method, reduction, 124, 29, nullptr).at("counts"), "The seed does not reach noise drawn per shot");
        }
    // A Kraus instruction: amplitude damping of qubit 0 in |11>. Each shot
    // draws its branch, with or without a noise block.
    const j::object damping{{"name", "kraus"}, {"qubits", j::array{0}}, {"operators", damp}};
    const j::object measure{{"name", "measure"}, {"qubits", j::array{0, 1}}, {"clbits", j::array{0, 1}}};
    const j::array prepare{j::object{{"name", "x"}, {"qubits", j::array{0}}}, j::object{{"name", "cx"}, {"qubits", j::array{0, 1}}}};
    const auto circuit = [&](j::array source) {
        return j::object{{"format", "instructions"}, {"num_qubits", 2}, {"num_clbits", 2}, {"source", std::move(source)}};
    };
    const auto damped = [&](const j::value &counts) {
        const auto &outcomes = counts.as_object();
        return outcomes.size() == 2 && outcomes.contains("11") && outcomes.contains("01") && std::abs(Real(outcomes.at("11")) / 4096 - 0.7) < 0.04;
    };
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        auto request = Request("execute", 2, "", method);
        request["circuit"] = circuit({prepare[0], prepare[1], damping, measure});
        request["execution"].as_object()["shots"] = 4096;
        auto result = Call(request);
        Check(damped(result.at("counts")), "A Kraus instruction was not drawn per shot");
        Check(result.at("counts") == Call(request).at("counts"), "Seeded shots of a sampled Kraus instruction are not reproducible");
        // One realization of an empty model: the instruction is still drawn per shot.
        request["noise"] = j::object{{"seed", 5}, {"realizations", 1}, {"channels", j::array{}}};
        result = Call(request);
        Check(damped(result.at("counts")), "One noise realization froze a Kraus instruction");

        auto mean = Request("estimate", 2, "", method);
        mean["circuit"] = circuit({prepare[0], prepare[1], damping});
        mean["observables"] = j::array{"ZI", "IZ"};
        mean["noise"] = j::object{{"seed", 5}, {"realizations", 2000}, {"channels", j::array{}}};
        result = Call(mean);
        Check(std::abs(Real(result.at("expectation_values").at(0)) + 0.4) < 0.1 && std::abs(Real(result.at("expectation_values").at(1)) + 1.0) < 1e-9,
              "A sampled Kraus instruction gave the wrong expectation values");
        // Without a noise block an estimate averages the branches as it averages measurement branches.
        mean.erase("noise");
        result = Call(mean);
        Check(result.at("execution_metadata").at("stochastic_execution").as_bool() && std::abs(Real(result.at("expectation_values").at(0)) + 0.4) < 0.35,
              "A plain estimate did not average the branches of a Kraus instruction");

        // In a checkpoint suffix, after a saved prefix.
        auto saved = Request("checkpoint_batch", 2, "x q[0]; cx q[0],q[1];", method);
        saved["execution"].as_object()["shots"] = 4096;
        saved["suffixes"] = j::array{circuit({damping, measure})};
        result = Call(saved);
        Check(damped(result.at("results").at(0).at("counts")), "A checkpoint suffix froze a Kraus instruction");
    }
    // A method that neither holds nor samples channels has no form for it.
    for (const char *method : {"tensor_network", "path_integral", "extended_stabilizer"})
    {
        auto request = Request("execute", 2, "", method);
        request["circuit"] = circuit({prepare[0], prepare[1], damping, measure});
        Check(Call(request, false, true).at("error").at("code") == "unsupported_capability", "A Kraus instruction ran without a noise block");
        request["noise"] = j::object{{"seed", 5}, {"channels", j::array{}}};
        Check(Call(request, false, true).at("error").at("code") == "unsupported_capability",
              "A Kraus instruction that needs the state was accepted by a method that cannot sample it");
    }
    // A query of one state runs the circuit once: it returns the state of one
    // branch, which the seed selects and the metadata reports, as Python's
    // get_statevector and get_probabilities do. |11> stays (index 3) with
    // probability 0.7 and decays to qubit 0 in |0> (index 2) otherwise.
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        auto request = Request("probabilities", 2, "", method);
        request["circuit"] = circuit({prepare[0], prepare[1], damping});
        size_t kept = 0;
        for (uint64_t seed = 1; seed <= 60; ++seed)
        {
            request["execution"].as_object()["seed"] = seed;
            const auto result = Call(request);
            const double stay = Real(result.at("probabilities").at(3)), decayed = Real(result.at("probabilities").at(2));
            Check(std::abs(stay + decayed - 1) < 1e-9 && (stay > 1 - 1e-9 || decayed > 1 - 1e-9),
                  "A state query did not return one branch of a Kraus instruction");
            Check(result.at("execution_metadata").at("stochastic_execution").as_bool(), "A state query of one branch does not say so");
            if (seed <= 3)
                Check(result.at("probabilities") == Call(request).at("probabilities"), "A seeded state query is not reproducible");
            kept += stay > 0.5;
        }
        Check(kept > 29 && kept < 55, "The branches of a state query do not follow the channel");
        request["operation"] = "state_probability";
        request["target_state"] = "11";
        const double one = Real(Call(request).at("probability"));
        Check(one < 1e-9 || one > 1 - 1e-9, "state_probability did not read one branch");
    }
    // On the density matrix the same query is the ensemble, and no branch is reported.
    {
        auto request = Request("probabilities", 2, "", "density_matrix");
        request["circuit"] = circuit({prepare[0], prepare[1], damping});
        const auto result = Call(request);
        Near(Real(result.at("probabilities").at(3)), 0.7);
        Check(!result.at("execution_metadata").as_object().contains("stochastic_execution"), "An exact state query reported a branch");
    }
    // A candidate list that holds channels in one of its backends runs the
    // circuit there, as Python's execute does: the network tries its
    // candidates in turn. A noise block has to know before it selects.
    {
        auto request = Request("execute", 2, "", "statevector");
        request["circuit"] = circuit({prepare[0], prepare[1], damping, measure});
        request["execution"].as_object()["shots"] = 4096;
        for (const char *other : {"stabilizer", "tensor_network"})
        {
            request["simulator"] = j::object{
                {"selection", "automatic"},
                {"candidates", j::array{j::object{{"backend", "qcsim"}, {"method", "statevector"}}, j::object{{"backend", "qcsim"}, {"method", other}}}}};
            request.erase("noise");
            const auto result = Call(request);
            Check(damped(result.at("counts")) && result.at("execution_metadata").at("method") == "statevector",
                  "A candidate list did not run channel instructions on the backend that holds them");
        }
        request["simulator"] = j::object{
            {"selection", "automatic"},
            {"candidates", j::array{j::object{{"backend", "qcsim"}, {"method", "stabilizer"}}, j::object{{"backend", "qcsim"}, {"method", "tensor_network"}}}}};
        Check(Call(request, false, true).at("error").at("code") == "unsupported_capability", "Candidates that hold no channels accepted a Kraus instruction");
    }
    // A method that neither holds nor samples channels still refuses the query.
    {
        auto request = Request("probabilities", 2, "", "tensor_network");
        request["circuit"] = circuit({prepare[0], prepare[1], damping});
        Check(Call(request, false, true).at("error").at("code") == "unsupported_capability", "A state query ran a Kraus instruction it cannot hold");
    }

    // The structured form of a channel instruction: factors applied in order
    // and kept apart, each on its positions among the instruction's qubits
    // (Python's QuantumChannel constructors composed with then()).
    const j::object hadamard{{"name", "h"}, {"qubits", j::array{0}}};
    const j::object entangle{{"name", "cx"}, {"qubits", j::array{0, 1}}};
    const j::object depolarize{{"kind", "depolarizing"}, {"probability", 0.1}};
    const j::object relax{{"kind", "thermal_relaxation"}, {"qubits", j::array{0}}, {"duration", 0.3}, {"t1", 1.0}, {"t2", 1.6}, {"excited_population", 0.1}};
    const j::object structured{{"name", "channel"},
                               {"qubits", j::array{0, 1}},
                               {"factors", j::array{depolarize, j::object{{"kind", "amplitude_damping"}, {"qubits", j::array{1}}, {"gamma", 0.3}}, relax}}};
    const auto values = [&](const char *method, j::array source, size_t realizations) {
        auto request = Request("estimate", 2, "", method);
        request["circuit"] = circuit(std::move(source));
        request["observables"] = j::array{"ZI", "IZ", "XX", "ZZ", "YY"};
        if (realizations)
            request["noise"] = j::object{{"seed", 9}, {"realizations", realizations}, {"channels", j::array{}}};
        return Call(request);
    };
    {
        // On the density matrix it is the product of its factors, given one by one.
        auto relaxAlone = relax;
        relaxAlone.erase("qubits");
        const auto whole = values("density_matrix", {hadamard, entangle, structured}, 0);
        const auto parts = values("density_matrix",
                                  {hadamard, entangle, j::object{{"name", "channel"}, {"qubits", j::array{0, 1}}, {"factors", j::array{depolarize}}},
                                   j::object{{"name", "kraus"}, {"qubits", j::array{1}}, {"operators", damp}},
                                   j::object{{"name", "channel"}, {"qubits", j::array{0}}, {"factors", j::array{relaxAlone}}}},
                                  0);
        SameExpectations(whole, parts);
        Check(std::abs(Real(whole.at("expectation_values").at(3)) - 1) > 0.1, "The structured channel did nothing");
        // The statevector and the matrix product state sample it from their state.
        for (const char *method : {"statevector", "matrix_product_state"})
        {
            const auto sampled = values(method, {hadamard, entangle, structured}, 3000);
            for (size_t i = 0; i < 5; ++i)
                Check(std::abs(Real(sampled.at("expectation_values").at(i)) - Real(whole.at("expectation_values").at(i))) < 0.1,
                      "A structured channel sampled from the state does not average to the density matrix");
        }
        // Positions [1, 0] exchange the qubits of a factor: damping of the
        // factor's first qubit, on the instruction's second.
        const j::array pairDamp{j::array{1, 0, 0, 0, 0, keep, 0, 0, 0, 0, 1, 0, 0, 0, 0, keep},
                                j::array{0, decay, 0, 0, 0, 0, 0, 0, 0, 0, 0, decay, 0, 0, 0, 0}};
        const auto exchanged = values("density_matrix",
                                      {hadamard, entangle,
                                       j::object{{"name", "channel"},
                                                 {"qubits", j::array{0, 1}},
                                                 {"factors", j::array{j::object{{"kind", "kraus"}, {"qubits", j::array{1, 0}}, {"operators", pairDamp}}}}}},
                                      0);
        SameExpectations(exchanged,
                         values("density_matrix", {hadamard, entangle, j::object{{"name", "kraus"}, {"qubits", j::array{1}}, {"operators", damp}}}, 0));
    }
    {
        // A factor keeps what a flat Kraus list loses: relaxation with T2 <= T1
        // can be drawn into resets by a method that holds no channels.
        const j::object flip{{"name", "x"}, {"qubits", j::array{0}}};
        const j::object slow{{"kind", "thermal_relaxation"}, {"duration", 0.3}, {"t1", 1.0}, {"t2", 0.8}};
        const auto drawn = [&](j::object instruction) {
            auto request = Request("estimate", 1, "", "tensor_network");
            request["circuit"] =
                j::object{{"format", "instructions"}, {"num_qubits", 1}, {"num_clbits", 1}, {"source", j::array{flip, std::move(instruction)}}};
            request["observables"] = j::array{"Z"};
            request["noise"] = j::object{{"seed", 4}, {"realizations", 3000}, {"channels", j::array{}}};
            return request;
        };
        const auto result = Call(drawn(j::object{{"name", "channel"}, {"qubits", j::array{0}}, {"factors", j::array{slow}}}));
        Check(std::abs(Real(result.at("expectation_values").at(0)) - (1 - 2 * std::exp(-0.3))) < 0.08, "Relaxation drawn from a channel factor is wrong");
        Check(Call(drawn(damping), false, true).at("error").at("code") == "unsupported_capability",
              "A flat Kraus list was drawn by a method that holds no channels");
    }
    {
        // A damping of zero has one branch, the identity: the instruction is a gate, and a statevector evolves the
        // circuit once for all its shots, as with a mixture of one unitary (TestDeterministicChannelInstructions).
        const j::object flip{{"name", "x"}, {"qubits", j::array{0}}};
        const auto counted = [&](const char *method, j::object factor) {
            auto request = Request("execute", 1, "", method);
            const j::object channel{{"name", "channel"}, {"qubits", j::array{0}}, {"factors", j::array{std::move(factor)}}};
            const j::object read{{"name", "measure"}, {"qubits", j::array{0}}, {"clbits", j::array{0}}};
            request["circuit"] = j::object{{"format", "instructions"}, {"num_qubits", 1}, {"num_clbits", 1}, {"source", j::array{flip, channel, read}}};
            request["execution"].as_object()["shots"] = 200;
            request["noise"] = j::object{{"seed", 5}, {"realizations", 1}, {"channels", j::array{}}};
            return Call(request);
        };
        const j::object still{{"kind", "amplitude_damping"}, {"gamma", 0.0}};
        const j::object stillWarm{{"kind", "generalized_amplitude_damping"}, {"gamma", 0.0}, {"excited_population", 0.3}};
        for (const char *method : {"statevector", "matrix_product_state"})
        {
            for (const auto &factor : {still, stillWarm})
            {
                const auto result = counted(method, factor);
                Check(result.at("counts") == j::object{{"1", 200}}, "A damping of zero changed the state");
                Check(result.at("noise").at("realizations") == 1, "A damping of zero made every shot a realization");
            }
            Check(counted(method, j::object{{"kind", "amplitude_damping"}, {"gamma", 0.3}}).at("noise").at("realizations") == 200,
                  "A damping that is drawn no longer makes every shot a realization");
        }
    }
    {
        // Malformed factors are input errors.
        const auto malformed = [&](j::array qubits, j::value factors) {
            auto request = Request("estimate", 2, "", "density_matrix");
            request["circuit"] = circuit({j::object{{"name", "channel"}, {"qubits", std::move(qubits)}, {"factors", std::move(factors)}}});
            request["observables"] = j::array{"ZI"};
            return Call(request, false, true).at("error").at("code") == "invalid_input";
        };
        Check(malformed({0, 1}, j::array{}), "A channel without factors was accepted");
        Check(malformed({0}, j::array{j::object{{"kind", "squeezing"}, {"gamma", 0.1}}}), "An unknown factor kind was accepted");
        Check(malformed({0, 1}, j::array{j::object{{"kind", "amplitude_damping"}, {"gamma", 0.1}}}), "A one-qubit factor without a position was accepted");
        Check(malformed({0, 1}, j::array{j::object{{"kind", "amplitude_damping"}, {"qubits", j::array{2}}, {"gamma", 0.1}}}),
              "A factor position outside the instruction was accepted");
        Check(malformed({0, 1}, j::array{j::object{{"kind", "kraus"}, {"qubits", j::array{1, 1}}, {"operators", j::array{}}}}),
              "A repeated factor position was accepted");
        Check(malformed({0}, j::array{j::object{{"kind", "amplitude_damping"}, {"gamma", 1.5}}}), "A damping probability above one was accepted");
        Check(malformed({0}, j::array{j::object{{"kind", "pauli"}, {"probabilities", j::array{0.5, 0.2}}}}), "Two Pauli probabilities were accepted");
        Check(malformed({0}, j::array{j::object{{"kind", "kraus"}, {"operators", j::array{j::array{1, 0, 0, 0.5}}}}}),
              "Kraus operators that are no channel were accepted");
    }

    // noisy_fidelity takes channel instructions as noise the ideal circuit does
    // not have, as Python's noisy_fidelity does.
    {
        const auto fidelityOf = [&](const char *method, size_t realizations) {
            auto request = Request("noisy_fidelity", 2, "", method);
            request["circuit"] = circuit({hadamard, entangle, structured});
            request["noise"] = j::object{{"seed", 31}, {"channels", j::array{}}};
            if (realizations)
                request["noise"].as_object()["realizations"] = realizations;
            return Real(Call(request).at("fidelity"));
        };
        const double exactFidelity = fidelityOf("density_matrix", 0);
        Check(exactFidelity > 0.5 && exactFidelity < 0.95, "Channel instructions left the fidelity alone");
        for (const char *method : {"statevector", "matrix_product_state"})
            Check(std::abs(fidelityOf(method, 3000) - exactFidelity) < 0.05, "Channel instructions sampled in noisy_fidelity give the wrong fidelity");

        // A reset before any other operation on its qubit only prepares it, as
        // in Python's noisy_fidelity: the fidelity is the same with and without
        // it. A reset after a gate on its qubit has no unitary ideal.
        const j::object reset{{"name", "reset"}, {"qubits", j::array{1}}};
        const auto prepared = [&](const char *method, j::array source, size_t realizations, bool success) {
            auto request = Request("noisy_fidelity", 2, "", method);
            request["circuit"] = circuit(std::move(source));
            request["noise"] = j::object{{"seed", 31}, {"channels", j::array{}}};
            if (realizations)
                request["noise"].as_object()["realizations"] = realizations;
            return Call(request, success, !success);
        };
        Near(Real(prepared("density_matrix", {hadamard, reset, entangle, structured}, 0, true).at("fidelity")), exactFidelity);
        for (const char *method : {"statevector", "matrix_product_state"})
            Check(std::abs(Real(prepared(method, {hadamard, reset, entangle, structured}, 3000, true).at("fidelity")) - exactFidelity) < 0.05,
                  "A leading reset changed the fidelity of sampled channel instructions");
        for (const char *method : {"density_matrix", "statevector"})
            Check(prepared(method, {hadamard, entangle, reset, structured}, 0, false).at("error").at("code") == "unsupported_capability",
                  "noisy_fidelity accepted a reset after a gate on its qubit");
    }

    // An estimate whose model the backend draws entirely as it runs: one
    // circuit for every realization, so the noise seed joins the execution
    // seed, as in the count requests above and in Python.
    {
        const j::array drawnAsRun{Channel("thermal_relaxation", {{"duration", 0.3}, {"t1", 1.0}, {"t2", 1.6}, {"excited_population", 0.1}}, {0}),
                                  Channel("generalized_amplitude_damping", {{"gamma", 0.15}, {"excited_population", 0.2}}, {1})};
        const auto run = [&](const char *method, uint64_t noiseSeed, size_t realizations) {
            auto request = Request("estimate", 2, "h q[0]; cx q[0],q[1]; ry(0.8) q[1];", method);
            request["observables"] = j::array{"ZI", "IZ", "XX"};
            request["noise"] = j::object{{"seed", noiseSeed}, {"channels", drawnAsRun}};
            if (realizations)
                request["noise"].as_object()["realizations"] = realizations;
            return Call(request);
        };
        const auto exactValues = run("density_matrix", 29, 0);
        for (const char *method : {"statevector", "matrix_product_state"})
        {
            const auto first = run(method, 29, 2000);
            for (size_t i = 0; i < 3; ++i)
                Check(std::abs(Real(first.at("expectation_values").at(i)) - Real(exactValues.at("expectation_values").at(i))) < 0.1,
                      "An estimate that draws its noise as it runs does not average to the density matrix");
            SameExpectations(first, run(method, 29, 2000));
            Check(first.at("expectation_values") != run(method, 30, 2000).at("expectation_values"),
                  "The noise seed does not reach an estimate that draws its noise as it runs");
        }
    }

    // The fidelity to the ideal state, with relaxation that dephases more
    // slowly than it decays: the clamp would lose a tenth of it.
    const auto fidelity = [&](const char *method, size_t realizations) {
        auto request = Request("noisy_fidelity", 2, "h q[0]; cx q[0],q[1];", method);
        request["noise"] = j::object{{"seed", 31},
                                     {"channels", j::array{Channel("thermal_relaxation", {{"duration", 0.3}, {"t1", 1.0}, {"t2", 1.8}}, {0, 1}),
                                                           Channel("thermal_relaxation_2q", {{"duration", 0.4}, {"t1", 1.0}, {"t2", 1.8}}, {0, 1})}}};
        if (realizations)
            request["noise"].as_object()["realizations"] = realizations;
        return Call(request);
    };
    const double reference = Real(fidelity("density_matrix", 0).at("fidelity"));
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        const auto result = fidelity(method, 2000);
        Check(!approximated(result, "thermal_T2_clamped_to_T1") && std::abs(Real(result.at("fidelity")) - reference) < 0.05,
              "Sampled relaxation gave the wrong fidelity");
    }
    Check(std::abs(Real(fidelity("path_integral", 2000).at("fidelity")) - reference) > 0.05, "The clamp no longer shows in this fidelity");
}

// Where the backend draws channel branches as the circuit runs, noise.seed has
// no injection to drive for them and joins the request seed: either seed alone
// changes the errors drawn, in noisy_fidelity and in checkpoint suffixes as in
// execute and estimate, and the same two seeds repeat the result. That holds
// whether all the noise is drawn that way, so that the realizations share one
// circuit, or a layer of its own per realization keeps them apart: a coherent
// rotation, whose signs are drawn for each realization on a qubit of the
// circuit, and which injects nothing on an idle one, where the channel branches
// are all that noise.seed can change.
void TestRuntimeNoiseSeeds()
{
    const std::string gates = "h q[0]; h q[1]; h q[2]; cx q[0],q[1]; cx q[1],q[2]; x q[0]; h q[0]; x q[1]; h q[1]; x q[2]; h q[2];";
    const std::string measure = " measure q[0] -> c[0]; measure q[1] -> c[1]; measure q[2] -> c[2];";
    for (const char *method : {"statevector", "matrix_product_state"})
        for (const int coherent : {-1, 0, 3}) // no coherent noise, on a qubit of the circuit, on the idle fourth qubit
        {
            j::array channels{Channel("generalized_amplitude_damping", {{"gamma", 0.2}, {"excited_population", 0.0}}, {0}),
                              Channel("generalized_amplitude_damping", {{"gamma", 0.2}, {"excited_population", 0.0}}, {1}),
                              Channel("generalized_amplitude_damping", {{"gamma", 0.2}, {"excited_population", 0.0}}, {2})};
            if (coherent >= 0)
                channels.push_back(Channel("coherent_rotation", {{"rx", 0.2}, {"ry", 0.1}, {"rz", 0.3}}, {coherent}));
            const auto run = [&](const char *operation, int seed, int noiseSeed) {
                const bool counts = std::string(operation) == "execute" || std::string(operation) == "checkpoint_batch";
                auto request = Request(operation, 4, std::string(operation) == "checkpoint_batch" ? "" : counts ? gates + measure : gates, method);
                request["execution"].as_object()["seed"] = seed;
                request["noise"] = j::object{{"seed", noiseSeed}, {"channels", channels}};
                if (counts)
                    request["execution"].as_object()["shots"] = 60;
                else
                    request["noise"].as_object()["realizations"] = 20;
                if (std::string(operation) == "checkpoint_batch")
                {
                    auto suffix = request.at("circuit").as_object();
                    suffix["source"] = "OPENQASM 2.0;\nqreg q[4]; creg c[4];\n" + gates + measure;
                    request["suffixes"] = j::array{suffix};
                }
                else if (std::string(operation) == "estimate")
                    request["observables"] = j::array{"ZZII", "IXXI"};
                const auto result = Call(request);
                return std::string(operation) == "noisy_fidelity"     ? j::serialize(result.at("fidelity"))
                       : std::string(operation) == "estimate"         ? j::serialize(result.at("expectation_values"))
                       : std::string(operation) == "checkpoint_batch" ? j::serialize(result.at("results").at(0).at("counts"))
                                                                      : j::serialize(result.at("counts"));
            };
            for (const char *operation : {"noisy_fidelity", "checkpoint_batch", "estimate", "execute"})
            {
                const auto base = run(operation, 42, 1);
                Check(run(operation, 42, 1) == base, "A runtime-noise request does not repeat with its two seeds");
                // Of three other noise seeds at least two give another result: counts of 60 shots can coincide once.
                int changed = 0;
                for (const int noiseSeed : {2, 3, 77})
                    changed += run(operation, 42, noiseSeed) != base;
                Check(changed >= 2, "noise.seed has no effect on channel branches the backend draws as the circuit runs");
                changed = 0;
                for (const int seed : {43, 44, 45})
                    changed += run(operation, seed, 1) != base;
                Check(changed >= 2, "The request seed has no effect on channel branches the backend draws as the circuit runs");
            }
        }
}

// A channel instruction with one branch only, a unitary, is a gate: nothing is
// drawn for it, so a backend that samples channels evolves the circuit once for
// all its shots, as it does the circuit with the gate in its place, and under a
// noise block a realization holds as many shots as were asked for. A mixture of
// two unitaries is still drawn shot by shot.
void TestDeterministicChannelInstructions()
{
    const j::object hadamard{{"name", "h"}, {"qubits", j::array{0}}}, entangle{{"name", "cx"}, {"qubits", j::array{0, 1}}};
    const j::object flip{{"name", "x"}, {"qubits", j::array{0}}}, turn{{"name", "h"}, {"qubits", j::array{1}}};
    const j::object measure{{"name", "measure"}, {"qubits", j::array{0, 1}}, {"clbits", j::array{0, 1}}};
    const j::array x{0, 1, 1, 0}, identity{1, 0, 0, 1};
    const auto mixture = [&](j::array probabilities, j::array unitaries) {
        return j::object{
            {"name", "channel"},
            {"qubits", j::array{0}},
            {"factors", j::array{j::object{{"kind", "unitary_mixture"}, {"probabilities", std::move(probabilities)}, {"unitaries", std::move(unitaries)}}}}};
    };
    // A list of one matrix has to be built from a value: a braced array of one array is a copy of it.
    const j::object asChannel = mixture({1.0}, j::array{j::value(x)}), drawn = mixture({0.5, 0.5}, {x, identity});
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        const auto run = [&](const char *operation, const j::object &third, bool noise) {
            auto request = Request(operation, 2, "", method);
            const bool counts = std::string(operation) == "execute";
            j::array source{hadamard, entangle, third, turn};
            if (counts)
            {
                source.push_back(measure);
                request["execution"].as_object()["shots"] = 4096;
            }
            else
                request["observables"] = j::array{"ZZ", "XI", "IX"};
            request["circuit"] = j::object{{"format", "instructions"}, {"num_qubits", 2}, {"num_clbits", 2}, {"source", std::move(source)}};
            if (noise)
                request["noise"] = j::object{{"seed", 5}, {"realizations", 1}, {"channels", j::array{}}};
            return Call(request);
        };
        const auto gate = run("execute", flip, false), channel = run("execute", asChannel, false);
        Check(channel.at("counts") == gate.at("counts"), "A channel with one unitary branch did not sample as the gate it is");
        Check(run("execute", drawn, false).at("counts") != gate.at("counts"), "A mixture of two unitaries sampled as one of them");
        const auto expected = run("estimate", flip, false), estimated = run("estimate", asChannel, false);
        for (size_t i = 0; i < 3; ++i)
            Check(std::abs(Real(estimated.at("expectation_values").at(i)) - Real(expected.at("expectation_values").at(i))) < 1e-12,
                  "A channel with one unitary branch changed an expectation value of the gate it is");
        // Under a noise block that asks for one realization, the shots share it; a channel that is drawn makes every shot one.
        Check(run("execute", asChannel, true).at("noise").at("realizations") == 1, "A channel with one unitary branch made every shot a realization");
        Check(run("execute", drawn, true).at("noise").at("realizations") == 4096, "A drawn channel no longer makes every shot a realization");
    }
}

void TestNoisyEstimateResetStreams()
{
    for (const char *method : {"statevector", "matrix_product_state"})
        for (const char *kind : {"t1", "t1_2q"})
            for (int seedMode = 0; seedMode < 6; ++seedMode)
            {
                auto request = Estimate("h q[0]; cx q[0],q[1];", {"ZI", "IZ"}, Channel(kind, {{"gamma", 0.5}}, {1}), method, 2);
                auto &execution = request["execution"].as_object();
                auto &noise = request["noise"].as_object();
                execution.erase("seed");
                noise.erase("seed");
                noise["realizations"] = 2000;
                if (seedMode == 1 || seedMode == 2 || seedMode == 5)
                    execution["seed"] = seedMode == 1 ? 0 : 5;
                if (seedMode == 3 || seedMode == 4 || seedMode == 5)
                    noise["seed"] = seedMode == 3 ? 0 : 23;

                const auto result = Call(request);
                const auto &values = result.at("expectation_values").as_array();
                Check(std::abs(Real(values[0])) < 0.1 && std::abs(Real(values[1]) - 0.5) < 0.1,
                      "Native noise estimates reused injected errors or reset outcomes");
                if (seedMode == 0)
                {
                    Check(result.at("seed") != Call(request).at("seed"), "Unseeded native estimates reused the same seed");
                    execution["seed"] = result.at("seed");
                }
                SameExpectations(result, Call(request));
            }
}
} // namespace

namespace
{
j::object ReadoutRequest(const char *operation, const char *method, const std::string &body, size_t bits = 2)
{
    auto request = Request("execute", 2, body, method, bits);
    if (std::string(operation) == "checkpoint_batch")
    {
        request["suffixes"] = j::array{request.at("circuit")};
        request["circuit"] = Request("execute", 2, "", method, bits).at("circuit");
        request["operation"] = operation;
    }
    request["noise"] =
        j::object{{"seed", 456}, {"realizations", 1}, {"channels", j::array{Channel("readout", {{"p_meas1_prep0", 1.0}, {"p_meas0_prep1", 0.0}})}}};
    return request;
}

j::value ReadoutCounts(const j::object &result)
{
    return result.at("operation") == "checkpoint_batch" ? result.at("results").at(0).at("counts") : result.at("counts");
}

void TestReadoutExecution()
{
    for (const char *method : {"statevector", "matrix_product_state", "density_matrix", "matrix_product_operator"})
    {
        for (const char *operation : {"execute", "checkpoint_batch"})
        {
            for (bool optimize : {false, true})
            {
                const std::pair<const char *, const char *> cases[] = {{"measure q[0]->c[2]; measure q[1]->c[0];", "0010"},
                                                                       {"measure q[0]->c[0]; measure q[0]->c[1];", "1100"},
                                                                       {"measure q[0]->c[0]; measure q[1]->c[0];", "0000"},
                                                                       {"measure q[0]->c[0]; if(c==1) x q[1]; measure q[1]->c[1];", "1100"},
                                                                       {"x q[1]; measure q[1]->c[0]; if(c==1) measure q[0]->c[1];", "1100"},
                                                                       {"if(c==1) measure q[0]->c[0];", "0000"}};
                for (const auto &[body, expected] : cases)
                {
                    auto request = ReadoutRequest(operation, method, body, 4);
                    request["simulator"].as_object()["options"] = j::object{{"optimize_circuit", optimize}};
                    Call(request, true, true);
                    Check(ReadoutCounts(Call(request)) == j::object{{expected, 80}}, "Measurement-time readout or classical control is incorrect");
                }
            }
            for (const int realizations : {1, 128})
            {
                auto request = ReadoutRequest(operation, method, "measure q[0]->c[0];");
                request["execution"].as_object()["shots"] = 128;
                request["noise"].as_object()["realizations"] = realizations;
                request["noise"].as_object()["channels"] = j::array{Channel("readout", {{"probability", 0.5}})};
                const auto counts = ReadoutCounts(Call(request));
                Check(counts == ReadoutCounts(Call(request)), "Seeded readout changed");
                Check(counts.as_object().size() == 2, "Readout draws repeated across shots");
                const double zeros = Real(counts.at("00"));
                Check(zeros > 24 && zeros < 104 && zeros + Real(counts.at("10")) == 128, "Readout flips are not independent per shot");
                request["noise"].as_object()["seed"] = 999;
                Check(counts == ReadoutCounts(Call(request)), "Noise injection seed changed the explicit readout stream");
                request["execution"].as_object().erase("seed");
                const auto fallback = Call(request);
                Check(fallback.at("seed") == 999, "Public noise seed was not inherited");
                request["execution"].as_object()["seed"] = 999;
                Check(ReadoutCounts(fallback) == ReadoutCounts(Call(request)), "Implicit and explicit simulator seeds disagree");
                request["execution"].as_object().erase("seed");
                request["simulator"].as_object()["options"] = j::object{{"seed", 0}};
                const auto zero = Call(request);
                Check(zero.at("seed") == 0, "Explicit zero seed lost precedence");
                request["noise"].as_object()["seed"] = 456;
                Check(ReadoutCounts(zero) == ReadoutCounts(Call(request)), "Simulator option seed did not control readout");
            }
        }
    }
}

void TestFixedBackendNoisyShots()
{
    // A sampled noise realization is chosen at injection time. Multiplying its
    // shots must preserve that choice, while different realizations remain
    // independent. Basis-state circuits make this comparison exact, with the
    // single-shot execution path as the reference.
    // Amplitude damping (t1) has no branch drawn without the state: only a
    // method that does not sample channels draws it at injection, as a reset.
    for (const char *method : {"statevector", "matrix_product_state", "path_integral"})
    {
        j::array channels{Channel("bit_flip", {{"probability", 0.5}}), Channel("thermal_relaxation", {{"duration", std::log(2.0)}, {"t1", 1.0}, {"t2", 1.0}})};
        if (std::string(method) == "path_integral")
            channels.emplace_back(Channel("t1", {{"gamma", 0.5}}));
        for (const auto &channel : channels)
        {
            for (size_t realizations : {size_t{1}, size_t{64}})
            {
                auto request = Request("execute", 1, "x q[0]; measure q->c;", method);
                request["noise"] = j::object{{"seed", 23}, {"realizations", realizations}, {"channels", j::array{channel}}};
                request["execution"].as_object()["shots"] = realizations;
                const auto reference = Call(request).at("counts").as_object();
                request["execution"].as_object()["shots"] = 32 * realizations;
                const auto result = Call(request);
                const auto &counts = result.at("counts").as_object();
                Check(counts.size() == reference.size(), "Shot reuse changed the sampled noise realizations");
                for (const auto &entry : reference)
                    Near(Real(counts.at(entry.key())), 32 * Real(entry.value()));
                Check(counts.size() == (realizations == 1 ? 1 : 2), "Distinct noise realizations reused one injected error");
                Check(result.at("noise").at("realizations").to_number<size_t>() == realizations, "Shot reuse changed the realization count");
                Check(result.at("counts") == Call(request).at("counts"), "Seeded noisy shots are not reproducible");
            }
        }
    }

    // Where channels are sampled from the state, amplitude damping is drawn as
    // each shot runs: one realization does not freeze it.
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        auto request = Request("execute", 1, "x q[0]; measure q->c;", method);
        request["noise"] = j::object{{"seed", 23}, {"realizations", 1}, {"channels", j::array{Channel("t1", {{"gamma", 0.5}})}}};
        request["execution"].as_object()["shots"] = 4096;
        const auto result = Call(request);
        const auto &counts = result.at("counts").as_object();
        Check(counts.size() == 2 && std::abs(Real(counts.at("0")) / 4096 - 0.5) < 0.05, "Amplitude damping was not drawn per shot");
        Check(result.at("counts") == Call(request).at("counts"), "Seeded shots of sampled amplitude damping are not reproducible");
    }

    // A sampled relaxation reset on one half of a Bell pair must not freeze the
    // other half's outcome across shots. Exact channels must retain the same
    // marginal. Exercise both gate relaxation and delay/idle injection.
    for (const char *method : {"statevector", "matrix_product_state", "density_matrix", "matrix_product_operator"})
    {
        for (bool idle : {false, true})
        {
            auto request = Request("execute", 2, "h q[0]; cx q[0],q[1]; measure q->c;", method);
            auto channel = Channel("t1_2q", {{"gamma", 1.0}});
            if (idle)
            {
                channel = Channel("idle", {{"t1", 1.0}, {"t2", 1.0}});
                request["circuit"] =
                    j::object{{"format", "instructions"},
                              {"num_qubits", 2},
                              {"source", j::array{j::object{{"name", "h"}, {"qubits", j::array{0}}}, j::object{{"name", "cx"}, {"qubits", j::array{0, 1}}},
                                                  j::object{{"name", "delay"}, {"qubits", j::array{0}}, {"duration", 1000.0}},
                                                  j::object{{"name", "measure"}, {"qubits", j::array{0, 1}}, {"clbits", j::array{0, 1}}}}}};
            }
            request["execution"].as_object()["shots"] = 4096;
            request["noise"] = j::object{{"seed", 23}, {"realizations", 1}, {"channels", j::array{channel}}};
            const auto result = Call(request);
            const auto &counts = result.at("counts").as_object();
            Check(counts.size() == 2 && counts.contains("00") && counts.contains("01"), "Relaxation reset froze or changed an entangled shot outcome");
            Check(std::abs(Real(counts.at("00")) / 4096 - 0.5) < 0.05, "Relaxation reset outcomes were not independent per shot");
            Near(Real(counts.at("00")) + Real(counts.at("01")), 4096);
        }
    }

    // Exact Kraus evolution may be reused, but its sampled measurement still
    // has to drive the conditional separately on every shot.
    for (const char *method : {"density_matrix", "matrix_product_operator"})
    {
        auto request = Request("execute", 2, "x q[0]; measure q[0]->c[0]; if(c==1) x q[1]; measure q[1]->c[1];", method);
        request["execution"].as_object()["shots"] = 4096;
        request["noise"] = j::object{{"evaluation", "exact"}, {"realizations", 1}, {"channels", j::array{Channel("t1", {{"gamma", 0.25}})}}};
        const auto result = Call(request);
        const auto &counts = result.at("counts").as_object();
        Check(counts.size() == 2 && counts.contains("00") && counts.contains("11"), "Exact noisy measurements lost their classical dependency");
        Check(std::abs(Real(counts.at("00")) / 4096 - 0.25) < 0.05, "Exact channel shot distribution changed");
        Near(Real(counts.at("00")) + Real(counts.at("11")), 4096);
    }
}
} // namespace

std::vector<std::string> Rows(const j::object &result)
{
    std::vector<std::string> rows;
    for (const auto &row : result.at("samples").as_array())
        rows.emplace_back(row.as_string());
    return rows;
}

j::object Native(char *result)
{
    Check(result != nullptr, "Null native result");
    const std::string text(result);
    FreeResult(result);
    return j::parse(text).as_object();
}

size_t Mismatches(const std::vector<std::string> &rows, size_t a, size_t b)
{
    size_t count = 0;
    for (const auto &row : rows)
        count += row[a] != row[b];
    return count;
}

// execute with return_samples: each sampler, noise, limits and replay.
void TestShotRows()
{
    struct Case
    {
        const char *method, *body, *strategy;
    };

    const char *bell = "h q[0]; cx q[0],q[1]; measure q -> c;";
    // The third classical bit stays unwritten.
    const char *wideBell = "h q[0]; cx q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];";
    for (const auto &item : {Case{"stabilizer", wideBell, "reference_frame_v1"}, Case{"statevector", wideBell, "terminal_v1"},
                             // The CX uses measured qubit 0, so every shot replays.
                             Case{"statevector", "h q[0]; measure q[0] -> c[0]; cx q[0],q[1]; measure q[1] -> c[1];", "trajectory_v1"}})
    {
        auto request = Request("execute", 2, item.body, item.method, 3);
        request["execution"].as_object()["shots"] = 200;
        request["execution"].as_object()["return_samples"] = true;
        const auto result = Call(request);
        const auto reported = j::serialize(result.at("execution_metadata").at("sampling_strategy"));
        Check(result.at("execution_metadata").at("sampling_strategy") == item.strategy,
              ("Native rows reported " + reported + " for " + item.method + ": " + item.body).c_str());
        Check(!result.contains("counts") && result.at("shots") == 200, "Native rows changed the result shape");
        const auto rows = Rows(result);
        Check(rows.size() == 200, "Native rows lost shots");
        size_t ones = 0;
        for (const auto &row : rows)
        {
            Check(row.size() == 3 && row[2] == '0' && row[0] == row[1], "Native Bell rows lost their width or correlation");
            ones += row[0] == '1';
        }
        Check(ones > 60 && ones < 140, "Native Bell rows are not balanced");
        Check(Rows(Call(request)) == rows, "Seeded native rows did not replay");
    }

    for (const char *method : {"statevector", "stabilizer"})
    {
        auto noisy = Request("execute", 2, bell, method);
        noisy["execution"].as_object()["shots"] = 150;
        noisy["execution"].as_object()["return_samples"] = true;
        noisy["noise"] = j::object{{"mode", "pauli"},
                                   {"evaluation", "trajectories"},
                                   {"realizations", 150},
                                   {"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 0.2}}}}};
        const auto result = Call(noisy);
        const bool frames = std::string(method) == "stabilizer";
        Check(result.at("execution_metadata").at("sampling_strategy") == (frames ? "reference_frame_v1" : "trajectory_v1"),
              "Noisy native rows reported the wrong sampler");
        Check(result.at("noise").at("realizations") == 150, "Noisy native rows lost their realizations");
        const auto rows = Rows(result);
        // Only the flip after the CX separates the qubits: about 30 of 150 rows.
        const auto split = Mismatches(rows, 0, 1);
        Check(rows.size() == 150 && split > 10 && split < 60, "Noisy native rows have the wrong error rate");
        Check(Rows(Call(noisy)) == rows, "Seeded noisy native rows did not replay");
    }

    auto estimate = Request("estimate", 1, "h q[0];");
    estimate["observables"] = j::array{"Z"};
    estimate["execution"].as_object()["return_samples"] = true;
    Check(Call(estimate, false).at("error").at("code") == "invalid_input", "return_samples applied to an estimate");
    auto huge = Request("execute", 2, bell);
    huge["execution"].as_object()["shots"] = 1000000000;
    huge["execution"].as_object()["return_samples"] = true;
    Check(Call(huge, false, true).at("error").at("code") == "invalid_input", "JSON rows exceeded the result limit");

    // Packed rows: 70 classical bits take two words per row.
    auto wide = Request("execute", 2, "h q[0]; cx q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[69];", "statevector", 70);
    wide["execution"].as_object()["shots"] = 64;
    const auto text = j::serialize(wide);
    const auto query = Native(MaestroRunRequestSamples(text.c_str(), nullptr, 0));
    Check(query.at("ok") == true && query.at("num_clbits") == 70 && query.at("row_words") == 2 && query.at("required_words") == 128,
          "Packed row query reported the wrong size");
    std::vector<unsigned long long> words(128, ~0ULL);
    Check(Native(MaestroRunRequestSamples(text.c_str(), words.data(), 127)).at("ok") == false, "Packed rows accepted a short buffer");
    const auto packed = Native(MaestroRunRequestSamples(text.c_str(), words.data(), words.size()));
    Check(packed.at("ok") == true && !packed.contains("samples") && packed.at("row_words") == 2, "Packed rows changed the result shape");
    Check(packed.at("execution_metadata").at("sampling_strategy") == "terminal_v1", "Packed rows reported the wrong sampler");
    wide["execution"].as_object()["return_samples"] = true;
    const auto rows = Rows(Call(wide));
    for (size_t shot = 0; shot < rows.size(); ++shot)
    {
        Check((words[2 * shot + 1] >> 6) == 0, "Packed rows set bits beyond the register");
        for (size_t bit = 0; bit < 70; ++bit)
            Check(((words[2 * shot + bit / 64] >> (bit % 64)) & 1ULL) == (rows[shot][bit] == '1' ? 1ULL : 0ULL), "Packed rows differ from JSON rows");
    }
    auto unknownOperation = wide;
    unknownOperation["operation"] = "estimate";
    Check(Native(MaestroRunRequestSamples(j::serialize(unknownOperation).c_str(), nullptr, 0)).at("ok") == false, "Packed rows accepted an estimate");
}

// Reference frames compiled once and sampled through the C handle.
void TestFrameSamplerHandle()
{
    auto circuit = Request("execute", 2, "h q[0]; cx q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[69];", "stabilizer", 70).at("circuit");
    const auto text = j::serialize(j::object{
        {"schema_version", 2},
        {"circuit", circuit},
        {"simulator", j::object{{"backend", "qcsim"}, {"method", "stabilizer"}}},
        {"noise", j::object{{"mode", "pauli"}, {"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 0.2}}}}}}});
    char *error = nullptr;
    void *sampler = MaestroCreateReferenceFrameSampler(text.c_str(), &error);
    Check(sampler != nullptr && error == nullptr, "Frame sampler handle did not compile");
    size_t qubits = 0, clbits = 0;
    Check(MaestroReferenceFrameSamplerShape(sampler, &qubits, &clbits) == 1 && qubits == 2 && clbits == 70, "Frame sampler reported the wrong shape");
    const size_t shots = 500;
    std::vector<unsigned long long> first(2 * shots), again(2 * shots), other(2 * shots);
    Check(MaestroReferenceFrameSampleBits(sampler, shots, 9, first.data(), first.size()) == 1, "Frame sampler rows failed");
    Check(MaestroReferenceFrameSampleBits(sampler, shots, 9, again.data(), again.size()) == 1 && again == first, "Seeded frame rows did not replay");
    Check(MaestroReferenceFrameSampleBits(sampler, shots, 10, other.data(), other.size()) == 1 && other != first, "Frame seeds reused one stream");
    Check(MaestroReferenceFrameSampleBits(sampler, shots, 9, other.data(), other.size() - 1) == 0, "Frame rows accepted a short buffer");
    std::map<std::pair<unsigned long long, unsigned long long>, unsigned long long> histogram;
    size_t split = 0;
    for (size_t shot = 0; shot < shots; ++shot)
    {
        ++histogram[{first[2 * shot], first[2 * shot + 1]}];
        split += (first[2 * shot] & 1ULL) != ((first[2 * shot + 1] >> 5) & 1ULL);
    }
    Check(split > 60 && split < 140, "Frame rows have the wrong error rate");
    std::vector<unsigned long long> distinct(2 * shots), counts(shots);
    size_t written = 0;
    Check(MaestroReferenceFrameSampleCountsBits(sampler, shots, 9, distinct.data(), counts.data(), shots, &written) == 1, "Frame counts failed");
    Check(written == histogram.size(), "Frame counts and rows disagree on distinct outcomes");
    for (size_t entry = 0; entry < written; ++entry)
        Check(histogram[{distinct[2 * entry], distinct[2 * entry + 1]}] == counts[entry], "Frame counts are not the histogram of the same rows");
    Check(MaestroReferenceFrameSampleCountsBits(sampler, shots, 9, distinct.data(), counts.data(), shots - 1, &written) == 0,
          "Frame counts accepted a short buffer");
    MaestroDestroyReferenceFrameSampler(sampler);
    MaestroDestroyReferenceFrameSampler(nullptr);

    // An omitted simulator is the QCSim stabilizer, so a Clifford circuit
    // compiles and a non-Clifford one fails for its gates.
    const auto plain = j::serialize(j::object{{"schema_version", 2}, {"circuit", circuit}});
    sampler = MaestroCreateReferenceFrameSampler(plain.c_str(), nullptr);
    Check(sampler != nullptr, "An omitted simulator did not compile a Clifford circuit to frames");
    MaestroDestroyReferenceFrameSampler(sampler);
    auto t = Request("execute", 1, "t q[0]; measure q -> c;", "stabilizer").at("circuit");
    const auto nonClifford = j::serialize(j::object{{"schema_version", 2}, {"circuit", t}});
    Check(MaestroCreateReferenceFrameSampler(nonClifford.c_str(), nullptr) == nullptr, "A non-Clifford circuit compiled to frames");
    error = nullptr;
    Check(MaestroCreateReferenceFrameSampler(nonClifford.c_str(), &error) == nullptr && error != nullptr, "Frame sampler failure lost its error");
    const auto failure = Native(error).at("error").as_object();
    Check(failure.at("code") == "unsupported_capability", "Frame sampler failure has the wrong code");
    Check(std::string(failure.at("message").as_string()).find("Clifford gates") != std::string::npos,
          "A non-Clifford circuit was rejected for its simulator, not its gates");
    size_t ignored = 0;
    Check(MaestroReferenceFrameSamplerShape(nullptr, &ignored, &ignored) == 0 && MaestroReferenceFrameSampleBits(nullptr, 1, 0, nullptr, 0) == 0,
          "Null frame sampler accepted");

    // The handle accepts what an execute request accepts, and a candidate list
    // admits frames only when it includes the QCSim stabilizer.
    const auto compile = [&](const j::object &simulator, const j::object &noise, const char *code) {
        j::object request{{"schema_version", 2}, {"circuit", circuit}, {"simulator", simulator}};
        if (!noise.empty())
            request["noise"] = noise;
        char *failure = nullptr;
        void *handle = MaestroCreateReferenceFrameSampler(j::serialize(request).c_str(), &failure);
        const bool compiled = handle != nullptr;
        MaestroDestroyReferenceFrameSampler(handle);
        if (failure)
            Check(code && Native(failure).at("error").at("code") == code, "Frame sampler rejected a request with the wrong code");
        return compiled;
    };
    const auto automatic = [](const char *method) {
        return j::object{{"backend", "qcsim"},
                         {"method", "statevector"},
                         {"selection", "automatic"},
                         {"candidates", j::array{j::object{{"backend", "qcsim"}, {"method", method}}}}};
    };
    const j::object fixedStabilizer{{"backend", "qcsim"}, {"method", "stabilizer"}};
    Check(compile(automatic("stabilizer"), {}, nullptr), "Candidates with the stabilizer did not admit frames");
    Check(!compile(automatic("statevector"), {}, "unsupported_capability"), "Frames ignored a candidate list without the stabilizer");
    Check(!compile(fixedStabilizer, j::object{{"mode", "pauli"}}, "invalid_input"), "Frames accepted noise without channels");
    const j::object flip{{"mode", "pauli"}, {"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 0.2}}}}};
    Check(!compile(automatic("stabilizer"), flip, "unsupported_capability"), "Frames accepted noise without fixed selection");
}

// Rows honor automatic selection and candidate lists as counts do, and packed
// noisy rows are the JSON rows of the same request.
void TestRowSelection()
{
    const auto automatic = [](j::object request, const char *method) {
        auto &simulator = request["simulator"].as_object();
        simulator["selection"] = "automatic";
        simulator["candidates"] = j::array{j::object{{"backend", "qcsim"}, {"method", method}}};
        request["execution"].as_object()["shots"] = 100;
        request["execution"].as_object()["return_samples"] = true;
        return request;
    };
    const char *bell = "h q[0]; cx q[0],q[1]; measure q -> c;";
    auto result = Call(automatic(Request("execute", 2, bell), "statevector"));
    Check(result.at("execution_metadata").at("sampling_strategy") != "reference_frame_v1", "Rows used frames that the candidate list excludes");
    Check(result.at("execution_metadata").at("method") == "statevector" && Rows(result).size() == 100, "Rows left the candidate list");
    result = Call(automatic(Request("execute", 2, bell), "stabilizer"));
    Check(result.at("execution_metadata").at("sampling_strategy") == "reference_frame_v1", "Rows skipped frames that the candidate list admits");
    // A non-Clifford circuit runs on the candidate, not on the requested method.
    result = Call(automatic(Request("execute", 2, "t q[0]; h q[0]; cx q[0],q[1]; measure q -> c;"), "matrix_product_state"));
    Check(result.at("execution_metadata").at("method") == "matrix_product_state", "Rows ran outside the candidate list");
    const auto rows = Rows(result);
    Check(rows.size() == 100 && Mismatches(rows, 0, 1) == 0, "Candidate rows lost their correlation");
    // The list replaces the requested backend, so a QCSim stabilizer candidate
    // admits frames whatever that backend is.
    auto composite = automatic(Request("execute", 2, bell), "stabilizer");
    composite["simulator"].as_object()["backend"] = "composite_qcsim";
    for (const bool samples : {true, false})
    {
        composite["execution"].as_object()["return_samples"] = samples;
        // Seeded counts take frames only on request.
        composite["simulator"].as_object()["options"] = j::object{{"counts_policy", samples ? "auto" : "frame_v1"}};
        const auto metadata = Call(composite).at("execution_metadata").as_object();
        Check(metadata.at("sampling_strategy") == "reference_frame_v1" && metadata.at("backend") == "qcsim" && metadata.at("method") == "stabilizer",
              "A stabilizer candidate under another requested backend did not admit frames");
    }
    // Sampling reduction splits rows into disjoint measurement cones as it does
    // counts, so it keeps the network executor instead of frames or one session.
    for (const char *method : {"stabilizer", "statevector"})
    {
        auto pairs = Request("execute", 4, "h q[0]; cx q[0],q[1]; h q[2]; cx q[2],q[3]; measure q -> c;", method);
        pairs["execution"].as_object()["shots"] = 64;
        pairs["execution"].as_object()["return_samples"] = true;
        pairs["simulator"].as_object()["options"] = j::object{{"enable_sampling_reduction", true}};
        result = Call(pairs);
        Check(result.at("sampling_components") == 2 && result.at("execution_metadata").at("sampling_strategy") != "reference_frame_v1",
              "Rows ignored sampling reduction");
        const auto split = Rows(result);
        Check(split.size() == 64 && Mismatches(split, 0, 1) == 0 && Mismatches(split, 2, 3) == 0, "Reduced rows lost their correlations");
    }

    // Packed noisy rows, from frames (a realization per shot) and from the
    // trajectory fallback (shared realizations).
    for (const size_t realizations : {120, 30})
    {
        auto noisy = Request("execute", 2, bell, "stabilizer");
        noisy["execution"].as_object()["shots"] = 120;
        noisy["noise"] = j::object{{"mode", "pauli"},
                                   {"evaluation", "trajectories"},
                                   {"realizations", realizations},
                                   {"channels", j::array{j::object{{"kind", "bit_flip"}, {"targets", j::array{0}}, {"probability", 0.2}}}}};
        const auto text = j::serialize(noisy);
        std::vector<unsigned long long> words(120);
        const auto packed = Native(MaestroRunRequestSamples(text.c_str(), words.data(), words.size()));
        Check(packed.at("ok") == true && packed.at("noise").at("realizations") == realizations, "Packed noisy rows failed");
        Check((packed.at("execution_metadata").at("sampling_strategy") == "reference_frame_v1") == (realizations == 120),
              "Packed noisy rows reported the wrong sampler");
        noisy["execution"].as_object()["return_samples"] = true;
        const auto json = Rows(Call(noisy));
        for (size_t shot = 0; shot < json.size(); ++shot)
            Check(((words[shot] & 1ULL) != 0) == (json[shot][0] == '1') && ((words[shot] & 2ULL) != 0) == (json[shot][1] == '1'),
                  "Packed noisy rows differ from their JSON rows");
    }
}

void TestMultinomialCountReplay()
{
    for (const char *method : {"statevector", "density_matrix", "path_integral", "matrix_product_state", "matrix_product_operator", "extended_stabilizer"})
    {
        auto request = Request("execute", 3, "h q[0]; t q[0]; h q[0]; cx q[0],q[1]; measure q[0] -> c[0];", method);
        auto &execution = request["execution"].as_object();
        execution["shots"] = 4096;
        execution.erase("seed");
        const auto fresh = Call(request);
        const auto policy = fresh.at("execution_metadata").at("configured_options").at("counts_policy");
        Check(policy == "multinomial_v1", "Unseeded terminal count policy is missing from replay metadata");
        auto &simulator = request["simulator"].as_object();
        execution["seed"] = fresh.at("seed");
        simulator["options"] = j::object{{"counts_policy", policy}};
        Check(Call(request).at("counts") == fresh.at("counts"), "Native terminal counts failed seed and policy replay");
        execution["seed"] = 0;
        simulator["options"] = j::object{{"counts_policy", "auto"}};
        const auto seeded = Call(request);
        Check(seeded.at("execution_metadata").at("configured_options").at("counts_policy") == "multinomial_v1",
              "Seed zero prevented automatic terminal count sampling");
        simulator["options"] = j::object{{"counts_policy", "multinomial_v1"}};
        Check(Call(request).at("counts") == seeded.at("counts"), "Seeded automatic counts differ from the explicit multinomial policy");
        if (std::string_view(method) == "statevector")
        {
            // The explicitly requested ordinary sampler is reproducible too.
            simulator["options"] = j::object{{"counts_policy", "legacy"}, {"max_simulators", 1}};
            const auto ordinary = Call(request).at("counts");
            Check(Call(request).at("counts") == ordinary, "Explicit ordinary statevector counts did not replay");
        }
    }
}

// SimpleExecute's counts_policy: frames for eligible Clifford counts under
// auto with or without a seed, on request under frame_v1, never under legacy.
void TestLegacyCountsPolicy()
{
    GetMaestroObjectWithMute();
    const auto catalog = Native(MaestroGetCapabilitiesJson());
    int backend = -1;
    for (const auto &entry : catalog.at("backends").as_array())
        if (entry.at("name") == "qcsim")
            backend = entry.at("legacy_id").to_number<int>();
    Check(backend >= 0, "Missing QCSim backend");

    struct Simulator
    {
        unsigned long handle = CreateSimpleSimulator(2);

        ~Simulator()
        {
            DestroySimpleSimulator(handle);
        }
    } simulator;

    Check(simulator.handle != 0 && RemoveAllOptimizationSimulatorsAndAdd(simulator.handle, backend, 2) == 1, "Cannot configure a legacy stabilizer");
    const char *circuit = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q->c;";
    const auto run = [&](const j::object &options) {
        auto *raw = SimpleExecute(simulator.handle, circuit, j::serialize(options).c_str());
        Check(raw != nullptr, "Legacy execution failed");
        const auto result = j::parse(raw).as_object();
        FreeResult(raw);
        uint64_t total = 0;
        for (const auto &entry : result.at("counts").as_object())
        {
            Check(entry.key() == "00" || entry.key() == "11", "Legacy Bell counts lost their correlation");
            total += entry.value().to_number<uint64_t>();
        }
        Check(total == 300, "Legacy counts lost shots");
        return result;
    };
    const auto framed = [](const j::object &result) {
        const auto *strategy = result.if_contains("sampling_strategy");
        return strategy && *strategy == "reference_frame_v1";
    };
    Check(framed(run({{"shots", 300}})), "Unseeded legacy Clifford counts did not use frames");
    Check(!framed(run({{"shots", 300}, {"counts_policy", "legacy"}})), "Legacy policy used frames");
    const auto seededFrames = run({{"shots", 300}, {"seed", 5}});
    Check(framed(seededFrames), "Seeded automatic counts did not select frames");
    Check(run({{"shots", 300}, {"seed", 5}}).at("counts") == seededFrames.at("counts"), "Seeded automatic frame counts did not replay");
    const auto explicitFrames = run({{"shots", 300}, {"seed", 5}, {"counts_policy", "frame_v1"}});
    Check(framed(explicitFrames) && explicitFrames.at("method") == "stabilizer", "frame_v1 did not select frames");
    Check(run({{"shots", 300}, {"seed", 5}, {"counts_policy", "frame_v1"}}).at("counts") == explicitFrames.at("counts"), "Seeded legacy frames did not replay");
    for (const auto &policy : j::array{"frames", 1, nullptr})
    {
        auto *raw = SimpleExecute(simulator.handle, circuit, j::serialize(j::object{{"shots", 10}, {"counts_policy", policy}}).c_str());
        const bool rejected = raw == nullptr;
        if (raw)
            FreeResult(raw);
        Check(rejected, "Malformed legacy counts_policy was accepted");
    }

    // Dynamic stabilizer execution defaults to logical shot streams. An
    // explicit earlier policy must not leak into the next call on a handle.
    const char *dynamic = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[0] -> c[0]; h q[0]; cx q[0],q[1]; measure q[1] -> c[1];";
    const auto counts = [&](unsigned long handle, const j::object &options) {
        auto *raw = SimpleExecute(handle, dynamic, j::serialize(options).c_str());
        Check(raw != nullptr, "Legacy dynamic execution failed");
        const auto result = j::parse(raw).as_object();
        FreeResult(raw);
        return result.at("counts");
    };
    const j::object omitted{{"shots", 400}, {"seed", 7}, {"max_simulators", 1}, {"counts_policy", "legacy"}};
    auto chosen = omitted;
    chosen["trajectory_policy"] = "shot_v1";
    Simulator fresh;
    Check(fresh.handle != 0 && RemoveAllOptimizationSimulatorsAndAdd(fresh.handle, backend, 2) == 1, "Cannot configure a legacy stabilizer");
    const auto defaults = counts(fresh.handle, omitted);
    const auto logical = counts(simulator.handle, chosen);
    Check(logical == defaults, "Omitting trajectory_policy did not select logical shot streams");
    chosen["trajectory_policy"] = "block_v1";
    counts(simulator.handle, chosen);
    Check(counts(simulator.handle, omitted) == defaults, "A reused handle kept an earlier explicit trajectory_policy");
}

void TestAutomaticMixedStateNoise()
{
    for (const char *method : {"density_matrix", "matrix_product_operator"})
    {
        auto request = Estimate("x q[0];", {"Z"}, Channel("bit_flip", {{"probability", 0.25}}), method);
        auto result = Call(request);
        Near(Real(result.at("expectation_values").at(0)), -0.5);
        Check(result.at("noise").at("realizations") == 1 && result.at("noise").at("requested_realizations").is_null(),
              "Deterministic mixed-state noise did not choose one automatic realization");
        request["noise"].as_object()["realizations"] = 3;
        result = Call(request);
        Check(result.at("noise").at("realizations") == 3 && result.at("noise").at("realizations_policy") == "explicit",
              "Explicit mixed-state realizations were discarded");
        for (const auto &automatic : j::array{nullptr, "auto"})
        {
            request["noise"].as_object()["realizations"] = automatic;
            Check(Call(request).at("noise").at("realizations") == 1, "Automatic JSON realization spelling changed behavior");
        }

        request = Estimate("h q[0];", {"Z"}, Channel("coherent_rotation", {{"rx", 0.0}, {"ry", std::acos(-1.0) / 2}, {"rz", 0.0}}), method);
        result = Call(request);
        Check(result.at("noise").at("realizations") == 100 && result.at("noise").at("stochastic_noise").as_bool(),
              "Exact channels suppressed coherent disorder averaging");
        Check(std::abs(Real(result.at("expectation_values").at(0))) < 0.4, "Coherent default retained a single disorder state");
        Check(!result.at("noise").at("approximations").as_array().empty(), "Mixed coherent sampling approximation was hidden");
        request["noise"].as_object()["realizations"] = 1;
        result = Call(request);
        Near(std::abs(Real(result.at("expectation_values").at(0))), 1.0);
        Check(result.at("realization_standard_errors").is_null(), "A single sampled disturbance claimed zero uncertainty");

        request = Request("estimate", 2, "h q[0]; measure q[0]->c[0]; if(c==1) x q[1];", method);
        request["observables"] = j::array{"IZ"};
        result = Call(request);
        Check(std::abs(Real(result.at("expectation_values").at(0))) < 0.4, "Dynamic mixed-state estimate retained one measurement branch");
        Check(result.at("execution_metadata").at("measurement_trajectories") == 100, "Dynamic branch averaging metadata missing");

        request = Request("checkpoint_batch", 2, "h q[0]; cx q[0],q[1];", method);
        request["suffixes"] = j::array{Request("execute", 2, "reset q[0]; measure q->c;", method).at("circuit"),
                                       Request("execute", 2, "cx q[0],q[1]; h q[0]; measure q->c;", method).at("circuit")};
        request["execution"].as_object()["shots"] = 512;
        result = Call(request);
        const auto &outputs = result.at("results").as_array();
        Check(outputs.at(0).at("sampling_strategy") == "terminal_v1" && outputs.at(1).at("sampling_strategy") == "terminal_v1",
              "Mixed-state checkpoint suffixes did not use terminal sampling");
        Check(outputs.at(0).at("counts").as_object().size() == 2 && outputs.at(1).at("counts") == j::object{{"00", 512}},
              "Checkpoint suffix sampling lost its mixed state or overwrote the saved prefix");
        Check(result.at("results") == Call(request).at("results"), "Checkpoint terminal sampling did not replay its seed");

        request["suffixes"] = j::array{Request("execute", 2, "measure q[0]->c[0]; if(c==1) x q[1]; measure q->c;", method).at("circuit")};
        request["noise"] = j::object{{"channels", j::array{Channel("bit_flip", {{"probability", 0.0}})}}};
        result = Call(request);
        // Each shot samples its branch exactly: stochastic, but no approximation.
        const auto &approximations = result.at("noise").at("approximations").as_array();
        Check(result.at("noise").at("stochastic_execution").as_bool() && result.at("results").at(0).at("stochastic_execution").as_bool() &&
                  std::find(approximations.begin(), approximations.end(), j::value("finite_measurement_trajectories")) == approximations.end(),
              "Checkpoint dynamic suffix misreported its measurement branches");
        request["operation"] = "execute";
        request["circuit"] = Request("execute", 2, "h q[0]; measure q[0]->c[0]; if(c==1) x q[1]; measure q->c;", method).at("circuit");
        request.erase("suffixes");
        result = Call(request);
        const auto &execute_approximations = result.at("noise").at("approximations").as_array();
        Check(result.at("noise").at("stochastic_execution").as_bool() && std::find(execute_approximations.begin(), execute_approximations.end(),
                                                                                   j::value("finite_measurement_trajectories")) == execute_approximations.end(),
              "Dynamic execute labelled exact branch sampling an approximation");
    }

    // Pure-state estimates: a reset before any other operation on its wire
    // leaves |0>, so the estimate stays a single deterministic evaluation.
    auto leading = Request("estimate", 2, "reset q[0]; reset q[1]; h q[0]; cx q[0],q[1];", "statevector");
    leading["observables"] = j::array{"XX"};
    auto result = Call(leading);
    Near(Real(result.at("expectation_values").at(0)), 1.0);
    Check(result.at("execution_metadata").at("estimate_evaluation") == "deterministic" && result.at("execution_metadata").at("measurement_trajectories") == 0,
          "A leading pure-state reset turned the estimate into trajectory averaging");
    auto later = Request("estimate", 2, "h q[0]; cx q[0],q[1]; reset q[0];", "statevector");
    later["observables"] = j::array{"IZ"};
    result = Call(later);
    Check(result.at("execution_metadata").at("estimate_evaluation") == "trajectory_average" &&
              result.at("execution_metadata").at("measurement_trajectories") == 100,
          "A collapsing pure-state reset did not average its branches");
    Check(std::abs(Real(result.at("expectation_values").at(0))) < 0.4, "Pure-state reset branches were not averaged");

    // One sampled coherent disturbance has no estimable fidelity uncertainty.
    auto fidelity = Request("noisy_fidelity", 1, "h q[0];");
    fidelity["noise"] =
        j::object{{"mode", "coherent"}, {"realizations", 1}, {"channels", j::array{Channel("coherent_rotation", {{"rx", 0.1}, {"ry", 0.0}, {"rz", 0.0}})}}};
    result = Call(fidelity);
    Check(result.at("standard_error").is_null(), "A single sampled fidelity claimed zero uncertainty");
    fidelity["noise"].as_object()["realizations"] = 4;
    Check(!Call(fidelity).at("standard_error").is_null(), "Sampled fidelity lost its standard error");

    // Pure-state checkpoints batch terminal suffixes too, keeping the prefix.
    auto checkpoint = Request("checkpoint_batch", 2, "h q[0]; cx q[0],q[1];");
    checkpoint["suffixes"] =
        j::array{Request("execute", 2, "measure q->c;").at("circuit"), Request("execute", 2, "cx q[0],q[1]; h q[0]; measure q->c;").at("circuit"),
                 Request("execute", 2, "reset q[0]; measure q->c;").at("circuit")};
    checkpoint["execution"].as_object()["shots"] = 400;
    result = Call(checkpoint);
    const auto &outputs = result.at("results").as_array();
    Check(outputs.at(0).at("sampling_strategy") == "terminal_v1" && outputs.at(0).at("counts").as_object().size() == 2,
          "Pure-state terminal checkpoint suffix was not batch sampled");
    Check(outputs.at(1).at("counts") == j::object{{"00", 400}}, "Pure-state terminal checkpoint sampling overwrote the prefix");
    Check(outputs.at(2).at("sampling_strategy") == "trajectory_v1" && outputs.at(2).at("counts").as_object().size() == 2,
          "A pure-state suffix reset shared one branch");

    // A dynamic suffix evolves its deterministic head once and still samples
    // its branch per shot: H H on q1 is the identity, then feedback clears q1.
    for (const char *method : {"statevector", "density_matrix", "matrix_product_operator"})
    {
        auto dynamic = Request("checkpoint_batch", 2, "h q[0]; cx q[0],q[1];", method);
        dynamic["suffixes"] = j::array{Request("execute", 2, "h q[1]; h q[1]; measure q[0]->c[0]; if(c==1) x q[1]; measure q->c;", method).at("circuit")};
        dynamic["execution"].as_object()["shots"] = 600;
        // Keep the result alive: at() returns a reference into it.
        result = Call(dynamic);
        const auto &counts = result.at("results").at(0).at("counts").as_object();
        Check(counts.size() == 2 && counts.at("00").as_int64() > 200 && counts.at("10").as_int64() > 200, "A shared dynamic suffix head changed its branches");
    }

    // JSON noisy_fidelity covers mixed states and every sampled layer.
    for (const char *method : {"density_matrix", "matrix_product_operator"})
    {
        auto mixed = Request("noisy_fidelity", 1, "x q[0];", method);
        mixed["noise"] = j::object{{"channels", j::array{Channel("bit_flip", {{"probability", 0.25}})}}};
        result = Call(mixed);
        Near(Real(result.at("fidelity")), 0.75);
        Check(result.at("noise").at("realizations") == 1 && result.at("noise").at("evaluation") == "exact_channels",
              "Mixed-state noisy fidelity did not evolve exact channels once");
    }
    auto sampled = Request("noisy_fidelity", 1, "x q[0];");
    sampled["noise"] = j::object{{"realizations", 2000}, {"channels", j::array{Channel("bit_flip", {{"probability", 0.25}})}}};
    result = Call(sampled);
    Check(std::abs(Real(result.at("fidelity")) - 0.75) < 0.04 && !result.at("standard_error").is_null(), "Sampled Pauli noisy fidelity is wrong");

    // Automatic candidate lists stay inside the list. Without a cost model the
    // first listed candidate that can hold the register runs; Composer builds
    // have one and may pick any listed candidate.
    const auto automatic = [](size_t qubits, const std::string &body, std::initializer_list<const char *> methods) {
        auto request = Request("execute", qubits, body);
        auto &simulator = request["simulator"].as_object();
        simulator["selection"] = "automatic";
        j::array candidates;
        for (const char *method : methods)
            candidates.push_back(j::object{{"backend", "qcsim"}, {"method", method}});
        simulator["candidates"] = std::move(candidates);
        return request;
    };
    for (const char *body : {"h q[0]; t q[0]; measure q -> c;", "h q[0]; measure q -> c;"})
    {
        const auto chosen = Call(automatic(1, body, {"density_matrix", "statevector"})).at("execution_metadata").at("method");
#ifdef MAESTRO_ENABLE_COMPOSER
        Check(chosen == "density_matrix" || chosen == "statevector", "Automatic selection ran a backend outside its candidate list");
#else
        Check(chosen == "density_matrix", "Automatic selection without a cost model skipped the first listed candidate");
#endif
    }
    std::string wide;
    for (size_t qubit = 0; qubit < 40; qubit += 2)
        wide += "x q[" + std::to_string(qubit) + "]; ";
    wide += "measure q -> c;";
    result = Call(automatic(40, wide, {"density_matrix", "matrix_product_operator"}));
    Check(result.at("execution_metadata").at("method") == "matrix_product_operator" && result.at("counts").as_object().size() == 1,
          "An impossible density matrix was chosen over a listed MPO");

    // The Clifford shortcut moves only to a listed (backend, stabilizer) pair.
    const auto listed = [](size_t qubits, const std::string &body, std::initializer_list<std::pair<const char *, const char *>> pairs) {
        auto request = Request("execute", qubits, body);
        auto &simulator = request["simulator"].as_object();
        simulator["selection"] = "automatic";
        j::array candidates;
        for (const auto &[backend, method] : pairs)
            candidates.push_back(j::object{{"backend", backend}, {"method", method}});
        simulator["candidates"] = std::move(candidates);
        return request;
    };
    const std::string clifford = "h q[0]; cx q[0],q[1]; measure q -> c;";
    result = Call(listed(2, clifford, {{"qcsim", "density_matrix"}, {"qcsim", "stabilizer"}}));
    Check(result.at("execution_metadata").at("backend") == "qcsim" && result.at("execution_metadata").at("method") == "stabilizer",
          "A listed QCSim stabilizer was not used for a Clifford circuit");
    bool aer = true;
    try
    {
        auto probe = Request("execute", 2, clifford, "stabilizer");
        probe["simulator"].as_object()["backend"] = "aer";
        Call(probe);
    }
    catch (const std::exception &)
    {
        aer = false; // Aer is not built in
    }
    if (aer)
    {
        result = Call(listed(2, clifford, {{"qcsim", "density_matrix"}, {"aer", "stabilizer"}}));
        Check(result.at("execution_metadata").at("backend") == "aer" && result.at("execution_metadata").at("method") == "stabilizer",
              "The Clifford shortcut ran an unlisted stabilizer pair");
    }

    // When no listed candidate can hold the register, report it instead of
    // running an unlisted placeholder.
    Call(listed(40, wide, {{"qcsim", "density_matrix"}, {"quest", "statevector"}}), false);
    // A one-entry list too: it runs its sole candidate or reports failure.
    result = Call(listed(1, "h q[0]; t q[0]; measure q -> c;", {{"qcsim", "density_matrix"}}));
    Check(result.at("execution_metadata").at("method") == "density_matrix", "A sole listed candidate did not run");
    bool quest = true;
    try
    {
        auto probe = Request("execute", 1, "h q[0]; measure q -> c;");
        probe["simulator"].as_object()["backend"] = "quest";
        Call(probe);
    }
    catch (const std::exception &)
    {
        quest = false; // QuEST is not available here
    }
    if (!quest)
        Call(listed(1, "h q[0]; t q[0]; measure q -> c;", {{"quest", "statevector"}}), false);

    // noise.prefix applies deterministic exact noise to the checkpoint prefix once.
    for (const char *method : {"density_matrix", "matrix_product_operator"})
    {
        auto prefixed = Request("checkpoint_batch", 1, "x q[0];", method);
        prefixed["suffixes"] = j::array{Request("execute", 1, "measure q -> c;", method).at("circuit")};
        prefixed["execution"].as_object()["shots"] = 4000;
        prefixed["noise"] = j::object{{"prefix", true}, {"channels", j::array{Channel("bit_flip", {{"probability", 0.25}})}}};
        result = Call(prefixed);
        Check(result.at("noise_scope") == "prefix_and_suffix", "Prefix noise scope missing");
        const auto &counts = result.at("results").at(0).at("counts").as_object();
        Check(std::abs(static_cast<double>(counts.at("1").as_int64()) / 4000 - 0.75) < 0.03, "Prefix noise was not applied once");
        prefixed["noise"].as_object()["prefix"] = false;
        Check(Call(prefixed).at("results").at(0).at("counts") == j::object{{"1", 4000}}, "An ideal prefix gained noise");
    }
    auto sampledPrefix = Request("checkpoint_batch", 1, "x q[0];");
    sampledPrefix["suffixes"] = j::array{Request("execute", 1, "measure q -> c;").at("circuit")};
    sampledPrefix["noise"] = j::object{{"prefix", true}, {"channels", j::array{Channel("bit_flip", {{"probability", 0.25}})}}};
    Call(sampledPrefix, false); // a pure-state prefix would freeze one sampled draw
    auto coherentPrefix = Request("checkpoint_batch", 1, "x q[0];", "density_matrix");
    coherentPrefix["suffixes"] = sampledPrefix.at("suffixes");
    coherentPrefix["noise"] = j::object{{"prefix", true}, {"channels", j::array{Channel("coherent_rotation", {{"rx", 0.1}, {"ry", 0.0}, {"rz", 0.0}})}}};
    Call(coherentPrefix, false);
    auto prefixElsewhere = Request("execute", 1, "x q[0]; measure q -> c;", "density_matrix");
    prefixElsewhere["noise"] = j::object{{"prefix", true}, {"channels", j::array{Channel("bit_flip", {{"probability", 0.25}})}}};
    Call(prefixElsewhere, false);

    // Capabilities describe the realization policy, prefix noise and selection.
    char *catalogText = MaestroGetCapabilitiesJson();
    const auto catalog = j::parse(catalogText).as_object();
    FreeResult(catalogText);
    Check(catalog.at("noise").at("realizations").at("accepts") == j::array{"positive_integer", "null", "auto"} &&
              catalog.at("noise").at("checkpoint_prefix").at("field") == "prefix" && !catalog.at("noise").at("exact_configurations").as_array().empty(),
          "Capabilities omit the noise realization policy");
    const auto &strategies = catalog.at("sampling_strategies").as_array();
    Check(std::find(strategies.begin(), strategies.end(), j::value("mixed_v1")) != strategies.end() &&
              catalog.at("estimate").at("dynamic_measurement_trajectories") == 100 &&
              !catalog.at("noisy_fidelity").at("exact_configurations").as_array().empty() && catalog.at("automatic_selection").is_object(),
          "Capabilities omit estimate, fidelity or selection rules");
}

// Batched extended-stabilizer frames support automatic counts, rows and
// checkpoint suffixes with supplied or generated seeds. Metadata retains the
// dispatch policy needed to replay a mixture of frames and fallback suffixes.
void TestExtendedFrameRequests()
{
    const char *body = "h q[0]; t q[0]; cx q[0],q[1]; h q[1]; t q[1]; h q[1]; measure q -> c;";
    const j::object noise{{"mode", "pauli"},
                          {"evaluation", "trajectories"},
                          {"channels", j::array{Channel("depolarizing", {{"probability", 0.05}}), Channel("bit_flip", {{"probability", 0.1}}, j::array{1})}}};
    const auto prepare = [&](const char *operation, const char *source, bool seeded, const char *policy, bool rows) {
        auto request = Request(operation, 2, source, "extended_stabilizer");
        request["noise"] = noise;
        auto &execution = request["execution"].as_object();
        execution["shots"] = 400;
        if (!seeded)
            execution.erase("seed");
        if (rows)
            execution["return_samples"] = true;
        if (policy)
            request["simulator"].as_object()["options"] = j::object{{"counts_policy", policy}};
        return request;
    };
    // The ordinary counts executor reports no strategy.
    const auto strategy = [](const j::object &result) {
        const auto *value = result.at("execution_metadata").as_object().if_contains("sampling_strategy");
        return value ? std::string(value->as_string()) : std::string();
    };
    const auto replay = [](j::object request, const j::object &result) {
        request["execution"].as_object()["seed"] = result.at("seed");
        request["simulator"].as_object()["options"] =
            j::object{{"counts_policy", result.at("execution_metadata").at("configured_options").at("counts_policy")}};
        return Call(request);
    };
    for (const bool rows : {false, true})
    {
        const auto request = prepare("execute", body, false, nullptr, rows);
        const auto result = Call(request);
        Check(strategy(result) == "extended_frame_v1", "Unseeded native extended_stabilizer shots did not use frames");
        Check(result.at("execution_metadata").at("configured_options").at("counts_policy") == "frame_v1", "Frame results do not report their policy");
        const auto again = replay(request, result);
        Check(strategy(again) == "extended_frame_v1" && (rows ? Rows(again) == Rows(result) : again.at("counts") == result.at("counts")),
              "Native frame shots did not replay from their seed and policy");
        if (rows)
            Check(Rows(result).size() == 400, "Native frame rows lost shots");
        else
        {
            uint64_t total = 0;
            for (const auto &entry : result.at("counts").as_object())
                total += entry.value().as_int64();
            Check(total == 400, "Native frame counts lost shots");
        }
        const auto seeded = prepare("execute", body, true, nullptr, rows);
        const auto seededResult = Call(seeded);
        Check(strategy(seededResult) == "extended_frame_v1", "A seeded auto request did not select frames");
        Check(rows ? Rows(Call(seeded)) == Rows(seededResult) : Call(seeded).at("counts") == seededResult.at("counts"),
              "Seeded automatic extended frames did not replay");
        const auto opted = Call(prepare("execute", body, true, "frame_v1", rows));
        Check(strategy(opted) == "extended_frame_v1", "frame_v1 did not opt a seeded request in");
    }

    // The trajectory memory budget applies to extended_stabilizer frames.
    auto budgeted = prepare("execute", body, false, nullptr, false);
    budgeted["simulator"].as_object()["options"] = j::object{{"trajectory_max_memory_mb", 128}};
    Check(strategy(Call(budgeted)) == "extended_frame_v1", "A budgeted extended_stabilizer request did not use frames");

    // Checkpoint suffixes after an ideal gate-only prefix.
    auto checkpoint = prepare("checkpoint_batch", "h q[0]; t q[0]; cx q[0],q[1];", false, nullptr, false);
    checkpoint["suffixes"] =
        j::array{Request("execute", 2, "h q[1]; t q[1]; h q[1]; measure q -> c;").at("circuit"), Request("execute", 2, "measure q -> c;").at("circuit")};
    auto batch = Call(checkpoint);
    for (const auto &suffix : batch.at("results").as_array())
    {
        Check(suffix.at("sampling_strategy") == "extended_frame_v1", "Checkpoint suffixes did not use frames");
        uint64_t total = 0;
        for (const auto &entry : suffix.at("counts").as_object())
            total += entry.value().as_int64();
        Check(total == 400, "Checkpoint frame counts lost shots");
    }
    Check(batch.at("execution_metadata").at("configured_options").at("counts_policy") == "auto", "Checkpoint frames lost their dispatch policy");
    const auto again = replay(checkpoint, batch);
    Check(again.at("results") == batch.at("results"), "Checkpoint frame suffixes did not replay");

    // An ineligible suffix follows the ordinary path even when another suffix
    // uses frames. The single returned policy must replay both distributions.
    auto mixed = checkpoint;
    mixed["noise"].as_object()["realizations"] = 2;
    mixed["suffixes"].as_array().push_back(Request("execute", 2, "measure q[0] -> c[0]; if(c==1) x q[1]; measure q -> c;").at("circuit"));
    const auto mixedBatch = Call(mixed);
    Check(mixedBatch.at("execution_metadata").at("configured_options").at("counts_policy") == "auto" &&
              mixedBatch.at("results").at(0).at("sampling_strategy") == "extended_frame_v1" &&
              mixedBatch.at("results").as_array().back().at("sampling_strategy") != "extended_frame_v1",
          "Mixed checkpoint suffixes did not preserve their frame and fallback choices");
    Check(replay(mixed, mixedBatch).at("results") == mixedBatch.at("results"), "Mixed checkpoint frame and ordinary suffixes did not replay");

    // Coherent noise excludes frames for every suffix. Automatic requests
    // still use the fast terminal fallback, independent of seed presence.
    auto noFrames = checkpoint;
    noFrames["noise"] =
        j::object{{"mode", "coherent"}, {"realizations", 1}, {"channels", j::array{Channel("coherent_rotation", {{"rx", 0.1}, {"ry", 0.0}, {"rz", 0.0}})}}};
    const auto ordinaryBatch = Call(noFrames);
    Check(ordinaryBatch.at("execution_metadata").at("configured_options").at("counts_policy") == "multinomial_v1",
          "An unframed checkpoint did not record its fallback count policy");
    for (const auto &suffix : ordinaryBatch.at("results").as_array())
        Check(suffix.at("sampling_strategy") == "terminal_v1", "An unframed terminal checkpoint replayed its suffix");
    Check(replay(noFrames, ordinaryBatch).at("results") == ordinaryBatch.at("results"), "Unframed checkpoint suffixes did not replay");

    // The frame handle: any extended_stabilizer circuit, independent noise per shot.
    auto circuit = Request("execute", 2, "h q[0]; t q[0]; cx q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[69];", "extended_stabilizer", 70).at("circuit");
    const auto text = j::serialize(j::object{
        {"schema_version", 2}, {"circuit", circuit}, {"simulator", j::object{{"backend", "qcsim"}, {"method", "extended_stabilizer"}}}, {"noise", noise}});
    char *error = nullptr;
    void *sampler = MaestroCreateReferenceFrameSampler(text.c_str(), &error);
    Check(sampler != nullptr && error == nullptr, "An extended_stabilizer frame handle did not compile");
    size_t qubits = 0, clbits = 0;
    Check(MaestroReferenceFrameSamplerShape(sampler, &qubits, &clbits) == 1 && qubits == 2 && clbits == 70, "Extended frame handle shape");
    const size_t shots = 600;
    std::vector<unsigned long long> first(2 * shots), second(2 * shots);
    Check(MaestroReferenceFrameSampleBits(sampler, shots, 5, first.data(), first.size()) == 1 &&
              MaestroReferenceFrameSampleBits(sampler, shots, 5, second.data(), second.size()) == 1 && first == second,
          "Extended frame handle rows did not replay");
    std::map<std::pair<unsigned long long, unsigned long long>, unsigned long long> histogram;
    size_t ones = 0;
    for (size_t shot = 0; shot < shots; ++shot)
    {
        ++histogram[{first[2 * shot], first[2 * shot + 1]}];
        ones += first[2 * shot] & 1ULL;
    }
    // H T on |0> gives P(1) = 1/2, moved little by the noise.
    Check(ones > 220 && ones < 380, "Extended frame handle rows have the wrong distribution");
    std::vector<unsigned long long> distinct(2 * shots), counts(shots);
    size_t written = 0;
    Check(MaestroReferenceFrameSampleCountsBits(sampler, shots, 5, distinct.data(), counts.data(), shots, &written) == 1 && written == histogram.size(),
          "Extended frame handle counts failed");
    for (size_t entry = 0; entry < written; ++entry)
        Check(histogram[{distinct[2 * entry], distinct[2 * entry + 1]}] == counts[entry], "Extended frame counts are not the histogram of its rows");
    MaestroDestroyReferenceFrameSampler(sampler);
    const auto budgetedHandle = j::serialize(
        j::object{{"schema_version", 2},
                  {"circuit", circuit},
                  {"simulator", j::object{{"backend", "qcsim"}, {"method", "extended_stabilizer"}, {"options", j::object{{"trajectory_max_memory_mb", 128}}}}},
                  {"noise", noise}});
    sampler = MaestroCreateReferenceFrameSampler(budgetedHandle.c_str(), nullptr);
    Check(sampler != nullptr, "An extended frame handle rejected its memory budget");
    MaestroDestroyReferenceFrameSampler(sampler);
    // The handle's budget is soft, as execute's: a shot that alone exceeds
    // it runs beyond it with the rows of an ample budget.
    std::string rotations;
    for (int q = 0; q < 17; ++q)
        rotations += "ry(1.2) q[" + std::to_string(q) + "]; ";
    const auto wide = Request("execute", 17, rotations + "measure q -> c;", "extended_stabilizer").at("circuit");
    std::vector<unsigned long long> tight(2), ample(2);
    for (const auto &[megabytes, rows] : {std::pair<int, std::vector<unsigned long long> *>{1, &tight}, {128, &ample}})
    {
        const auto handle = j::serialize(
            j::object{{"schema_version", 2},
                      {"circuit", wide},
                      {"simulator",
                       j::object{{"backend", "qcsim"}, {"method", "extended_stabilizer"}, {"options", j::object{{"trajectory_max_memory_mb", megabytes}}}}}});
        sampler = MaestroCreateReferenceFrameSampler(handle.c_str(), nullptr);
        Check(sampler != nullptr && MaestroReferenceFrameSampleBits(sampler, 2, 7, rows->data(), rows->size()) == 1,
              "An extended frame handle failed a shot above its budget");
        MaestroDestroyReferenceFrameSampler(sampler);
    }
    Check(tight == ample, "The extended frame handle's budget changed its rows");
    // Classical control keeps scalar trajectories: no frame handle.
    auto controlled =
        Request("execute", 1, "h q[0]; t q[0]; measure q[0] -> c[0]; if(c==1) x q[0]; measure q[0] -> c[0];", "extended_stabilizer").at("circuit");
    error = nullptr;
    Check(MaestroCreateReferenceFrameSampler(
              j::serialize(
                  j::object{{"schema_version", 2}, {"circuit", controlled}, {"simulator", j::object{{"backend", "qcsim"}, {"method", "extended_stabilizer"}}}})
                  .c_str(),
              &error) == nullptr &&
              error != nullptr,
          "Classical control compiled to extended frames");
    FreeResult(error);
}

// The QCSim Pauli propagator applies channels to the observable it propagates.
// One execution then gives the ensemble a density matrix gives: Pauli-type
// noise by default, relaxation on request, and channels with no sampled form.
void TestPauliPropagatorExactNoise()
{
    const char *propagator = "pauli_propagator";
    const std::string body = "h q[0]; cx q[0],q[1]; ry(0.7) q[1]; rx(0.4) q[0];";
    const j::array observables{"ZZ", "XI", "IY", "ZI", "XX"};
    const auto run = [&](const j::object &channel, const char *evaluation, const char *expected, int realizations, bool compare) {
        auto request = Estimate(body, observables, channel, propagator, 2);
        if (evaluation)
            request["noise"].as_object()["evaluation"] = evaluation;
        const auto result = Call(request);
        Check(result.at("noise").at("evaluation") == expected && result.at("noise").at("realizations") == realizations,
              "Pauli propagator noise resolved to the wrong evaluation");
        if (compare)
        {
            auto reference = Estimate(body, observables, channel, "density_matrix", 2);
            SameExpectations(result, Call(reference));
            Check(result.at("execution_metadata").at("estimate_evaluation") == "deterministic", "Exact propagator noise averaged trajectories");
        }
        return result;
    };
    // Diagonal in the Pauli basis: exact by default, one realization.
    run(Channel("depolarizing", {{"probability", 0.2}}, {0, 1}), nullptr, "exact_channels", 1, true);
    run(Channel("pauli", {{"px", 0.05}, {"py", 0.1}, {"pz", 0.2}}, {1}), nullptr, "exact_channels", 1, true);
    run(Channel("dephasing", {{"probability", 0.15}}, {0}), nullptr, "exact_channels", 1, true);
    run(Channel("phase_damping", {{"gamma", 0.3}}, {1}), nullptr, "exact_channels", 1, true);
    run(Channel("gate_depolarizing_2q", {{"probability", 0.1}}, {0, 1}), nullptr, "exact_channels", 1, true);
    // The caller can still sample them.
    run(Channel("depolarizing", {{"probability", 0.2}}, {0, 1}), "trajectories", "trajectories", 100, false);

    // Relaxation keeps its trajectories unless exact evaluation is requested;
    // exact evaluation keeps the calibrated T2 above T1.
    const auto thermal = Channel("thermal_relaxation", {{"duration", 1.0}, {"t1", 3.0}, {"t2", 5.0}, {"excited_population", 0.1}}, {0, 1});
    auto result = run(thermal, nullptr, "trajectories", 100, false);
    auto approximations = result.at("noise").at("approximations").as_array();
    Check(std::find(approximations.begin(), approximations.end(), j::value("thermal_T2_clamped_to_T1")) != approximations.end(),
          "Sampled propagator relaxation hid its T2 clamp");
    result = run(thermal, "exact", "exact_channels", 1, true);
    approximations = result.at("noise").at("approximations").as_array();
    Check(std::find(approximations.begin(), approximations.end(), j::value("thermal_T2_clamped_to_T1")) == approximations.end(),
          "Exact propagator relaxation reported a T2 clamp");
    run(Channel("t1", {{"gamma", 0.3}}, {0}), nullptr, "trajectories", 100, false);
    run(Channel("t1", {{"gamma", 0.3}}, {0}), "exact", "exact_channels", 1, true);
    // No sampled form exists for this one: exact, where it was rejected before.
    run(Channel("generalized_amplitude_damping", {{"gamma", 0.3}, {"excited_population", 0.2}}, {1}), nullptr, "exact_channels", 1, true);
    // Coherent disorder is still sampled.
    result = run(Channel("coherent_rotation", {{"rx", 0.0}, {"ry", 0.3}, {"rz", 0.0}}, {0}), nullptr, "exact_channels", 100, false);
    Check(result.at("noise").at("stochastic_noise").as_bool(), "Exact propagator channels suppressed coherent averaging");

    // Circuit channel instructions run in place, without a noise block.
    const auto instructions = [&](const char *method) {
        auto request = Request("estimate", 2, "", method);
        const double a = std::sqrt(0.7), b = std::sqrt(0.3);
        request["circuit"].as_object()["format"] = "instructions";
        request["circuit"].as_object()["source"] = j::array{
            j::object{{"name", "h"}, {"qubits", j::array{0}}}, j::object{{"name", "cx"}, {"qubits", j::array{0, 1}}},
            j::object{{"name", "kraus"}, {"qubits", j::array{1}}, {"operators", j::array{j::array{1, 0, 0, a}, j::array{0, b, 0, 0}}}},
            j::object{
                {"name", "pauli_channel"}, {"qubits", j::array{0, 1}}, {"probabilities", j::array{0.7, 0.12, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.08, 0, 0, 0}}},
            j::object{{"name", "ry"}, {"qubits", j::array{0}}, {"params", j::array{0.6}}}};
        request["observables"] = observables;
        return Call(request);
    };
    SameExpectations(instructions(propagator), instructions("density_matrix"));

    // An unrecorded reset is a deterministic channel: one evaluation, no
    // measurement trajectories, and terminal counts from one recorded circuit.
    auto reset = Request("estimate", 2, "h q[0]; cx q[0],q[1]; reset q[0]; h q[0];", propagator);
    reset["observables"] = j::array{"XI", "IZ"};
    result = Call(reset);
    Near(Real(result.at("expectation_values").at(0)), 1.0);
    Near(Real(result.at("expectation_values").at(1)), 0.0);
    Check(result.at("execution_metadata").at("estimate_evaluation") == "deterministic" && result.at("execution_metadata").at("measurement_trajectories") == 0,
          "A reset on the Pauli propagator averaged measurement trajectories");
    auto counts = Request("execute", 2, "h q[0]; cx q[0],q[1]; reset q[0]; measure q->c;", propagator);
    counts["execution"].as_object()["shots"] = 2000;
    result = Call(counts);
    const auto &histogram = result.at("counts").as_object();
    Check(histogram.size() == 2 && histogram.contains("00") && histogram.contains("01") && std::abs(histogram.at("00").as_int64() - 1000) < 150,
          "Reset counts on the Pauli propagator lost the mixed partner");

    // Checkpoint suffixes sample from the saved prefix without replaying it,
    // a reset included, and a noisy prefix is evolved once as exact channels.
    auto checkpoint = Request("checkpoint_batch", 2, "h q[0]; cx q[0],q[1];", propagator);
    checkpoint["suffixes"] = j::array{Request("execute", 2, "reset q[0]; measure q->c;", propagator).at("circuit"),
                                      Request("execute", 2, "cx q[0],q[1]; h q[0]; measure q->c;", propagator).at("circuit")};
    checkpoint["execution"].as_object()["shots"] = 512;
    result = Call(checkpoint);
    const auto &outputs = result.at("results").as_array();
    Check(outputs.at(0).at("sampling_strategy") == "terminal_v1" && outputs.at(1).at("sampling_strategy") == "terminal_v1",
          "Pauli propagator checkpoint suffixes did not use terminal sampling");
    Check(outputs.at(0).at("counts").as_object().size() == 2 && outputs.at(1).at("counts") == j::object{{"00", 512}},
          "Pauli propagator checkpoint sampling lost its mixed state or overwrote the saved prefix");
    checkpoint["noise"] = j::object{{"prefix", true}, {"channels", j::array{Channel("depolarizing", {{"probability", 0.75}}, {1})}}};
    result = Call(checkpoint);
    Check(result.at("noise").at("evaluation") == "exact_channels" && result.at("results").at(1).at("counts").as_object().size() > 1,
          "A noisy Pauli propagator prefix was not evolved as exact channels");
    // Relaxation on the prefix is sampled by default there, which a prefix
    // evolved once cannot be; exact evaluation applies it.
    checkpoint["noise"] = j::object{{"prefix", true}, {"channels", j::array{Channel("t1", {{"gamma", 0.2}}, {0})}}};
    Check(Call(checkpoint, false).at("error").at("code") == "unsupported_capability", "A sampled relaxation prefix was accepted on the propagator");
    checkpoint["noise"].as_object()["evaluation"] = "exact";
    Check(Call(checkpoint).at("noise").at("evaluation") == "exact_channels", "An exact relaxation prefix was refused on the propagator");

    // Fidelity reads P(0...0) of the mixed state, up to the documented width.
    const auto fidelity = [&](const char *method, size_t qubits) {
        auto request = Request("noisy_fidelity", qubits, "h q[0]; cx q[0],q[1]; ry(0.7) q[1];", method);
        request["noise"] = j::object{{"channels", j::array{Channel("depolarizing", {{"probability", 0.2}}, {0, 1})}}};
        return request;
    };
    result = Call(fidelity(propagator, 2));
    Near(Real(result.at("fidelity")), Real(Call(fidelity("density_matrix", 2)).at("fidelity")));
    Check(result.at("noise").at("evaluation") == "exact_channels" && result.at("noise").at("realizations") == 1 && Real(result.at("standard_error")) == 0.0,
          "Exact propagator fidelity was sampled");
    Check(Call(fidelity(propagator, 25), false).at("error").at("code") == "unsupported_capability", "A wide propagator fidelity was not refused");

    // Automatic evaluation follows what the channels do to the circuit.
    const auto approximated = [](const j::object &output, const char *name) {
        const auto &list = output.at("noise").at("approximations").as_array();
        return std::find(list.begin(), list.end(), j::value(name)) != list.end();
    };
    // Thermal relaxation towards equal populations only scales terms: exact,
    // with the calibrated T2 that sampled relaxation would clamp to T1.
    const auto unital = Channel("thermal_relaxation", {{"duration", 1.0}, {"t1", 1.0}, {"t2", 2.0}, {"excited_population", 0.5}}, {0, 1});
    result = run(unital, nullptr, "exact_channels", 1, true);
    Check(result.at("noise").at("approximations").as_array().empty(), "Unital relaxation on the propagator reported an approximation");
    const auto spectators = [&](const std::string &gates, const char *method) {
        auto request = Request("estimate", 3, gates, method);
        request["observables"] = j::array{"XII", "ZII", "YII"};
        // A zero T1 probability on the gated qubit, T1 on a qubit no gate
        // touches, and idle relaxation in a circuit without delays.
        request["noise"] = j::object{{"seed", 23},
                                     {"channels", j::array{Channel("depolarizing", {{"probability", 0.1}}, {0}), Channel("t1", {{"gamma", 0.0}}, {0}),
                                                           Channel("t1", {{"gamma", 0.3}}, {1}), Channel("idle", {{"t1", 50.0}, {"t2", 60.0}}, {2})}}};
        return Call(request);
    };
    result = spectators("h q[0]; ry(0.7) q[0];", propagator);
    Check(result.at("noise").at("evaluation") == "exact_channels" && result.at("noise").at("realizations") == 1 &&
              result.at("noise").at("approximations").as_array().empty(),
          "Relaxation that never acts on the circuit kept its trajectories");
    SameExpectations(result, spectators("h q[0]; ry(0.7) q[0];", "density_matrix"));
    result = spectators("h q[0]; ry(0.7) q[0]; h q[1];", propagator);
    Check(result.at("noise").at("evaluation") == "trajectories" && approximated(result, "sampled_T1_reset"), "Relaxation on a gated qubit was not sampled");

    // A Kraus instruction cannot be sampled, so a relaxation model beside it
    // is evaluated exactly too instead of refusing the request.
    const double keep = std::sqrt(0.7), decay = std::sqrt(0.3);
    const j::object damping{{"name", "kraus"}, {"qubits", j::array{0}}, {"operators", j::array{j::array{1, 0, 0, keep}, j::array{0, decay, 0, 0}}}};
    const auto instructed = [&](const char *method, const char *evaluation) {
        auto request = Request("estimate", 1, "", method);
        request["circuit"].as_object()["format"] = "instructions";
        request["circuit"].as_object()["source"] =
            j::array{j::object{{"name", "h"}, {"qubits", j::array{0}}}, j::object{{"name", "ry"}, {"qubits", j::array{0}}, {"params", j::array{0.6}}}, damping};
        request["observables"] = j::array{"Z", "X"};
        request["noise"] = j::object{{"channels", j::array{Channel("t1", {{"gamma", 0.2}}, {0})}}};
        if (evaluation)
            request["noise"].as_object()["evaluation"] = evaluation;
        return request;
    };
    result = Call(instructed(propagator, nullptr));
    Check(result.at("noise").at("evaluation") == "exact_channels" && result.at("noise").at("realizations") == 1 && !approximated(result, "sampled_T1_reset"),
          "A Kraus instruction beside relaxation was not evaluated exactly");
    SameExpectations(result, Call(instructed("density_matrix", nullptr)));
    Check(Call(instructed(propagator, nullptr), true, true).at("valid").as_bool(), "A Kraus instruction beside relaxation did not validate");
    Check(Call(instructed(propagator, "trajectories"), false).at("error").at("code") == "unsupported_capability", "A Kraus instruction was sampled");
    // The same in a checkpoint, where the instruction sits in a suffix and
    // only the suffixes say what the relaxation acts on.
    const auto measured = [&](j::array source, size_t qubits) {
        j::array clbits, targets;
        for (size_t q = 0; q < qubits; ++q)
        {
            targets.emplace_back(q);
            clbits.emplace_back(q);
        }
        source.emplace_back(j::object{{"name", "measure"}, {"qubits", targets}, {"clbits", clbits}});
        return j::object{{"format", "instructions"}, {"num_qubits", qubits}, {"num_clbits", qubits}, {"source", source}};
    };
    auto suffixed = Request("checkpoint_batch", 1, "h q[0];", propagator);
    suffixed["suffixes"] = j::array{measured(j::array{damping}, 1)};
    suffixed["noise"] = j::object{{"channels", j::array{Channel("t1", {{"gamma", 0.2}}, {0})}}};
    result = Call(suffixed);
    Check(result.at("noise").at("evaluation") == "exact_channels" && result.at("results").at(0).at("sampling_strategy") == "terminal_v1" &&
              result.at("results").at(0).at("realizations") == 1,
          "A Kraus instruction in a checkpoint suffix was not evaluated exactly");
    auto idle = Request("checkpoint_batch", 2, "h q[0]; h q[1];", propagator);
    idle["suffixes"] = j::array{Request("execute", 2, "ry(0.3) q[0]; measure q->c;", propagator).at("circuit")};
    idle["noise"] = j::object{{"channels", j::array{Channel("depolarizing", {{"probability", 0.1}}, {0}), Channel("t1", {{"gamma", 0.3}}, {1})}}};
    result = Call(idle);
    Check(result.at("noise").at("evaluation") == "exact_channels" && !result.at("noise").at("stochastic_noise").as_bool() &&
              result.at("noise").at("approximations").as_array().empty() && result.at("results").at(0).at("realizations") == 1,
          "Relaxation that no checkpoint suffix draws kept its trajectories");
    idle["suffixes"].as_array().emplace_back(Request("execute", 2, "h q[1]; measure q->c;", propagator).at("circuit"));
    result = Call(idle);
    Check(result.at("noise").at("evaluation") == "trajectories" && approximated(result, "sampled_T1_reset") &&
              approximated(result, "finite_noise_realizations"),
          "Relaxation that a checkpoint suffix draws was not sampled");

    // The propagator holds no state: amplitude queries are refused when the
    // request is validated, with or without noise, not when it runs.
    for (const char *operation : {"amplitudes", "statevector", "inner_product", "mirror_fidelity"})
        for (const bool noisy : {false, true})
        {
            auto request = Request(operation, 1, "h q[0];", propagator);
            if (std::string(operation) == "inner_product")
            {
                const auto other = request.at("circuit");
                request["other_circuit"] = other;
            }
            if (noisy)
                request["noise"] = j::object{{"channels", j::array{Channel("depolarizing", {{"probability", 0.1}}, {0})}}};
            for (const bool validate : {true, false})
                Check(Call(request, false, validate).at("error").at("code") == "unsupported_capability",
                      "An amplitude query on the propagator was not refused");
        }
    auto probabilities = Request("probabilities", 1, "h q[0];", propagator);
    probabilities["noise"] = j::object{{"channels", j::array{Channel("depolarizing", {{"probability", 0.1}}, {0})}}};
    Check(Call(probabilities, true, true).at("valid").as_bool(), "Noisy probabilities on the propagator did not validate");
    Near(Real(Call(probabilities).at("probabilities").at(0)), 0.5);

    // A truncation threshold in effect is reported beside exact channels.
    const auto truncates = [&](const j::object &options) {
        auto request = Estimate(body, observables, Channel("depolarizing", {{"probability", 0.05}}, {0, 1}), propagator, 2);
        if (!options.empty())
            request["simulator"].as_object()["options"] = options;
        const auto output = Call(request);
        Check(output.at("noise").at("evaluation") == "exact_channels", "A truncated propagator request changed its evaluation");
        return approximated(output, "Pauli_truncation");
    };
    Check(!truncates({}) && !truncates({{"pp_coefficient_threshold", 0}}) && !truncates({{"pp_max_pauli_weight", 2}}),
          "An untruncated propagator request reported truncation");
    Check(truncates({{"pp_coefficient_threshold", 1e-3}}) && truncates({{"pp_max_pauli_weight", 1}}), "Active Pauli truncation was not reported");

    // Checkpoint counts under the legacy policy keep the per-shot replay of
    // earlier releases, which draws readout errors shot by shot; auto samples
    // the recorded state once.
    const auto replayed = [&](const char *policy, bool readout) {
        auto request = Request("checkpoint_batch", 2, "h q[0]; cx q[0],q[1];", propagator);
        auto suffix = measured(j::array{}, 2);
        if (readout)
            suffix["source"].as_array().back().as_object()["readout_error"] = 0.1;
        request["suffixes"] = j::array{suffix};
        if (policy)
            request["simulator"].as_object()["options"] = j::object{{"counts_policy", policy}};
        request["execution"].as_object()["shots"] = 256;
        const auto output = Call(request).at("results").at(0).as_object();
        int64_t total = 0;
        for (const auto &entry : output.at("counts").as_object())
            total += entry.value().as_int64();
        Check(total == 256 && (readout || output.at("counts").as_object().size() == 2), "Checkpoint counts on the propagator are wrong");
        return output.at("sampling_strategy") == "trajectory_v1";
    };
    Check(!replayed(nullptr, false) && !replayed(nullptr, true) && !replayed("multinomial_v1", true), "Automatic propagator checkpoints replayed their shots");
    Check(replayed("legacy", false) && replayed("legacy", true), "Legacy propagator checkpoints did not replay their shots");

    // Discovery lists the method for exact noise, on the CPU backend only.
    char *text = MaestroGetCapabilitiesJson();
    const auto catalog = j::parse(text).as_object();
    FreeResult(text);
    const auto lists = [&](const j::value &configurations, const char *backend) {
        for (const auto &entry : configurations.as_array())
            if (entry.at("backend") == backend && entry.at("method") == propagator)
                return true;
        return false;
    };
    const auto &capabilities = catalog.contains("capabilities") ? catalog.at("capabilities").as_object() : catalog;
    Check(lists(capabilities.at("noise").at("exact_configurations"), "qcsim") && !lists(capabilities.at("noise").at("exact_configurations"), "gpu") &&
              lists(capabilities.at("noisy_fidelity").at("exact_configurations"), "qcsim") &&
              lists(capabilities.at("noise").at("checkpoint_prefix").at("supported_configurations"), "qcsim"),
          "Capabilities do not list the Pauli propagator for exact noise");
}

void TestRequestNoiseAndOptions()
{
    TestExtendedFrameRequests();
    TestAutomaticMixedStateNoise();
    TestPauliPropagatorExactNoise();
    TestShotRows();
    TestFrameSamplerHandle();
    TestRowSelection();
    TestLegacyCountsPolicy();
    TestMultinomialCountReplay();
    TestReadoutExecution();
    TestThermalApproximation();
    TestSampledChannelRequests();
    TestRuntimeNoiseSeeds();
    TestDeterministicChannelInstructions();
    TestFixedBackendNoisyShots();
    TestNoisyEstimateResetStreams();
    const double duration = 0.3, t1 = 1, t2 = 0.6, excited = 0.1;
    auto thermal = Channel("thermal_relaxation", {{"duration", duration}, {"t1", t1}, {"t2", t2}, {"excited_population", excited}});
    auto request = Estimate("h q[0];", {"X", "Z"}, thermal);
    auto result = Call(request);
    Near(Real(result.at("expectation_values").at(0)), std::exp(-duration / t2));
    Near(Real(result.at("expectation_values").at(1)), (1 - 2 * excited) * (1 - std::exp(-duration / t1)));
    request["simulator"].as_object()["method"] = "matrix_product_operator";
    SameExpectations(result, Call(request));

    auto two = thermal;
    two["kind"] = "thermal_relaxation_2q";
    request = Estimate("x q[0]; cx q[0],q[1];", {"ZI", "IZ"}, two, "density_matrix", 2);
    result = Call(request);
    Near(Real(result.at("expectation_values").at(0)), -1 + 2 * (1 - excited) * (1 - std::exp(-duration / t1)));
    Near(Real(result.at("expectation_values").at(1)), -1);
    request = Estimate("h q[0];", {"X"}, Channel("phase_damping", {{"gamma", 0.36}}));
    Near(Real(Call(request).at("expectation_values").at(0)), 0.8);
    request = Estimate("x q[0];", {"Z"}, Channel("generalized_amplitude_damping", {{"gamma", 0.4}, {"excited_population", 0.25}}));
    Near(Real(Call(request).at("expectation_values").at(0)), -0.4);

    // Idle relaxation is applied to delay duration, not to each ordinary gate.
    request = Estimate("", {"X", "Y", "Z"}, Channel("idle", {{"t1", t1}, {"t2", t2}, {"excited_population", excited}, {"detuning_hz", 0.25}}));
    auto gate = j::object{{"name", "h"}, {"qubits", j::array{0}}};
    auto delay = j::object{{"name", "delay"}, {"qubits", j::array{0}}, {"duration", duration}};
    request["circuit"] = j::object{{"format", "instructions"}, {"num_qubits", 1}, {"source", j::array{gate, delay}}};
    result = Call(request);
    Near(Real(result.at("expectation_values").at(0)), std::exp(-duration / t2) * std::cos(2 * std::acos(-1.0) * 0.25 * duration));
    Near(Real(result.at("expectation_values").at(2)), (1 - 2 * excited) * (1 - std::exp(-duration / t1)));
    delay["duration"] = duration / 2;
    request["circuit"].as_object()["source"] = j::array{gate, delay, delay};
    SameExpectations(result, Call(request));

    // A spectator rotation has a known effect; selecting the wrong qubit fails.
    request = Estimate("h q[1]; x q[0];", {"IX"}, Channel("crosstalk", {{"strength", 0.3}}, {0, 1}), "statevector", 2);
    Near(Real(Call(request).at("expectation_values").at(0)), std::cos(0.3));

    // One-shot noise realizations may run on several networks; each keeps its
    // own seed, so seeded counts must not depend on the worker count. Stabilizer
    // and MPS requests also accept the realization memory budget.
    for (const char *method : {"statevector", "density_matrix", "stabilizer", "matrix_product_state"})
    {
        auto noisy = Request("execute", 2, "h q[0]; cx q[0],q[1]; h q[1]; measure q->c;", method);
        noisy["execution"].as_object()["shots"] = 96;
        noisy["noise"] = j::object{{"channels", j::array{Channel("depolarizing", {{"probability", 0.2}})}}, {"seed", 31}, {"realizations", 96}};
        noisy["simulator"].as_object()["options"] = j::object{{"max_simulators", 1}, {"trajectory_max_memory_mb", 64}};
        if (std::string(method) == "matrix_product_state")
            noisy["simulator"].as_object()["options"].as_object()["max_bond_dimension"] = 4;
        const auto serial = Call(noisy).at("counts");
        noisy["simulator"].as_object()["options"].as_object()["max_simulators"] = 4;
        Check(Call(noisy).at("counts") == serial, "Parallel native noise realizations changed seeded counts");
    }
    // An explicit trajectory policy is accepted for the non-dense methods it now opts in.
    for (const char *method : {"stabilizer", "extended_stabilizer", "matrix_product_state", "matrix_product_operator"})
    {
        auto opted = Request("execute", 1, "h q[0]; measure q->c; reset q[0]; h q[0]; measure q->c;", method);
        opted["simulator"].as_object()["options"] = j::object{{"trajectory_policy", "shot_v1"}, {"max_simulators", 1}};
        const auto serial = Call(opted).at("counts");
        opted["simulator"].as_object()["options"].as_object()["max_simulators"] = 4;
        Check(Call(opted).at("counts") == serial, "An explicit trajectory policy left non-dense counts worker-dependent");
    }

    // Noise targets lie in the declared register, so a crosstalk spectator can
    // never fall outside a fixed native register, checkpoint suffixes included.
    request = Request("checkpoint_batch", 1, "h q[0];");
    request["suffixes"] = j::array{Request("execute", 1, "h q[0]; measure q->c;").at("circuit")};
    request["noise"] = j::object{{"channels", j::array{Channel("crosstalk", {{"strength", 0.3}}, {0, 1})}}, {"seed", 23}};
    for (bool validate : {false, true})
    {
        const auto error = Call(request, false, validate).at("error").as_object();
        Check(std::string(error.at("message").as_string()).find("Noise target is invalid") != std::string::npos,
              "A noise target outside the declared register was accepted");
    }

    const j::array correlated{Channel("correlated_ar1", {{"phi", 0.5}, {"sigma_eta", 0.4}}),
                              Channel("correlated_ou", {{"sigma", 0.4}, {"alpha", 2}, {"gate_time", 0.1}}),
                              Channel("correlated_ou_band", {{"sigma", 0.4}, {"alpha", 2}, {"gate_time", 0.1}}),
                              Channel("multi_correlated_ou", {{"bands", j::array{j::array{0.4, 2}, j::array{0.2, 5}}}, {"gate_time", 0.1}}),
                              Channel("one_over_f", {{"total_power", 0.2}, {"f_min", 0.1}, {"f_max", 10}, {"num_bands", 8}, {"gate_time", 0.1}})};
    for (const auto &value : correlated)
    {
        auto channel = value.as_object();
        request = Estimate("h q[0];", {"X"}, channel, "statevector");
        request["noise"].as_object()["realizations"] = 16;
        result = Call(request);
        SameExpectations(result, Call(request));
        Check(Real(result.at("expectation_values").at(0)) < 0.999, "Correlated channel had no effect");
        request["noise"].as_object()["seed"] = 24;
        Check(std::abs(Real(result.at("expectation_values").at(0)) - Real(Call(request).at("expectation_values").at(0))) > 1e-8,
              "Correlated noise seed had no effect");
        channel["after_1q"] = false;
        channel["after_2q"] = false;
        request["noise"].as_object()["channels"] = j::array{channel};
        Near(Real(Call(request).at("expectation_values").at(0)), 1);
        channel["unused_parameter"] = 1;
        request["noise"].as_object()["channels"] = j::array{channel};
        Check(Call(request, false, true).at("error").at("code") == "invalid_input", "Unknown correlated parameter accepted");
    }
    for (auto bad : j::array{Channel("thermal_relaxation", {{"duration", 1}, {"t1", 1}, {"t2", 3}}), Channel("idle", {{"t1", 0}, {"t2", 1}}),
                             Channel("correlated_ou", {{"sigma", -1}, {"alpha", 2}, {"gate_time", 0.1}}),
                             Channel("multi_correlated_ou", {{"bands", j::array{}}, {"gate_time", 0.1}}),
                             Channel("one_over_f", {{"total_power", 1}, {"f_min", 0.1}, {"f_max", 10}, {"num_bands", 0}, {"gate_time", 0.1}}),
                             Channel("crosstalk", {{"strength", 0.3}})})
    {
        request = Estimate("h q[0];", {"X"}, bad.as_object());
        Check(Call(request, false, true).at("error").at("code") == "invalid_input", "Unphysical noise accepted");
    }

    request = Estimate("h q[0];", {"X", "Z"}, Channel("pauli", {{"px", 0}, {"py", 0}, {"pz", 0.25}}), "statevector");
    request["noise"].as_object()["mode"] = "analytical";
    result = Call(request);
    Near(Real(result.at("expectation_values").at(0)), 0.5);
    Near(Real(result.at("expectation_values").at(1)), 0);
    Check(result.at("noise").at("evaluation") == "analytical_approximation", "Analytical approximation not identified");
    request["operation"] = "execute";
    request.erase("observables");
    Call(request, false, true);

    // Optimize only after channels have been placed on the original gate stream.
    for (const char *method : {"density_matrix", "matrix_product_operator", "statevector", "matrix_product_state"})
    {
        for (const auto &channel : j::array{Channel("t1", {{"gamma", 0.36}}), Channel("pauli", {{"px", 0}, {"py", 0}, {"pz", 0.25}}),
                                            Channel("coherent_rotation", {{"rx", 0.2}, {"ry", 0.1}, {"rz", 0.3}})})
        {
            request = Estimate("h q[0]; h q[0]; x q[0]; x q[0];", {"X", "Y", "Z"}, channel.as_object(), method);
            request["noise"].as_object()["realizations"] = 16;
            request["simulator"].as_object()["options"] = j::object{{"optimize_circuit", false}};
            result = Call(request);
            request["simulator"].as_object()["options"] = j::object{{"optimize_circuit", true}};
            SameExpectations(result, Call(request));
        }
    }

    // A real option effect: bond dimension two retains a Bell pair, one cannot.
    request = Request("probabilities", 2, "h q[0]; cx q[0],q[1];", "matrix_product_state");
    request["simulator"].as_object()["options"] = j::object{{"max_bond_dimension", 2}};
    result = Call(request);
    Near(Real(result.at("probabilities").at(0)), 0.5);
    Near(Real(result.at("probabilities").at(3)), 0.5);
    request["simulator"].as_object()["options"] = j::object{{"max_bond_dimension", 1}};
    Check(std::abs(Real(Call(request).at("probabilities").at(0)) - 0.5) > 0.2, "MPS bond bound had no effect");
    auto aliased = request;
    aliased["simulator"].as_object()["options"] = j::object{{"matrix_product_state_max_bond_dimension", 1}};
    Check(Call(aliased).at("probabilities") == Call(request).at("probabilities"), "Option alias changed semantics");
}
