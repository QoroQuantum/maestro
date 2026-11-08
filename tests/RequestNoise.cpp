// Numerical contract tests through the public request API, without Python/GPU.
#include "../maestrolib/Interface.h"
#include <boost/json.hpp>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
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
    // The same seeded sampled channel must behave as an explicit T2=T1 model;
    // exact channels must retain the supplied T2, including the physical limit.
    for (const char *method : {"statevector", "matrix_product_state", "density_matrix", "matrix_product_operator"})
    {
        const bool sampled = std::string(method) == "statevector" || std::string(method) == "matrix_product_state";
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
                request["noise"].as_object()["realizations"] = sampled ? 256 : 1;
                ThermalCall(request, sampled, true);
                const auto result = ThermalCall(request, sampled);
                if (sampled)
                {
                    SameExpectations(result, ThermalCall(request, true));
                    auto clamped = request;
                    clamped["noise"].as_object()["channels"].as_array()[0].as_object()["t2"] = 1.0;
                    SameExpectations(result, ThermalCall(clamped, false));
                    Check(std::abs(Real(result.at("expectation_values").at(0)) - std::exp(-0.3)) < 0.1,
                          "Sampled thermal coherence did not follow effective T2=T1");
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
        auto idle = Estimate("h q[0];", {"XI"}, Channel("idle", {{"t1", 1.0}, {"t2", 2.0}}), "statevector", 2);
        idle["noise"].as_object()["realizations"] = 256;
        ThermalCall(idle, false);
        auto instant = Estimate("h q[0];", {"XI"}, Channel("thermal_relaxation", {{"duration", 0.0}, {"t1", 1.0}, {"t2", 2.0}}), "statevector", 2);
        instant["noise"].as_object()["realizations"] = 256;
        ThermalCall(instant, false);
    }

    // Accepting sampled thermal approximations must not admit exact-only maps
    // or an explicit exact evaluation on a pure-state method.
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
        Check(Call(request, false, true).at("error").at("code") == "unsupported_capability",
              "Exact-only channel accepted with a sampled thermal approximation");
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
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        for (const auto &channel : j::array{Channel("bit_flip", {{"probability", 0.5}}), Channel("t1", {{"gamma", 0.5}}),
                                            Channel("thermal_relaxation", {{"duration", std::log(2.0)}, {"t1", 1.0}, {"t2", 1.0}})})
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

// SimpleExecute's counts_policy: frames for unseeded Clifford counts under auto,
// on request under frame_v1, never under legacy or for seeded auto calls.
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
    Check(!framed(run({{"shots", 300}, {"seed", 5}})), "Seeded auto legacy counts changed their sequence");
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

    // Only a trajectory_policy in a call's configuration opts a dynamic
    // stabilizer circuit into logical shot streams; omitting it keeps the 0.3.13
    // sequence, on a fresh handle and on one an earlier call configured.
    const char *dynamic = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[0] -> c[0]; h q[0]; cx q[0],q[1]; measure q[1] -> c[1];";
    const auto counts = [&](unsigned long handle, const j::object &options) {
        auto *raw = SimpleExecute(handle, dynamic, j::serialize(options).c_str());
        Check(raw != nullptr, "Legacy dynamic execution failed");
        const auto result = j::parse(raw).as_object();
        FreeResult(raw);
        return result.at("counts");
    };
    const j::object omitted{{"shots", 400}, {"seed", 7}, {"max_simulators", 1}};
    auto chosen = omitted;
    chosen["trajectory_policy"] = "shot_v1";
    Simulator fresh;
    Check(fresh.handle != 0 && RemoveAllOptimizationSimulatorsAndAdd(fresh.handle, backend, 2) == 1, "Cannot configure a legacy stabilizer");
    const auto earlier = counts(fresh.handle, omitted);
    const auto logical = counts(simulator.handle, chosen);
    Check(logical != earlier, "Omitting trajectory_policy opted a stabilizer into logical shot streams");
    Check(counts(simulator.handle, omitted) == earlier, "A reused handle kept an earlier explicit trajectory_policy");
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

// Batched extended-stabilizer frames: unseeded auto requests (counts, rows and
// checkpoint suffixes) opt in before receiving their seed and report the policy
// that replays them; seeded requests keep scalar trajectories unless
// counts_policy is frame_v1. The frame handle compiles extended_stabilizer
// requests too.
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
        Check(strategy(Call(prepare("execute", body, true, nullptr, rows))) != "extended_frame_v1", "A seeded auto request changed its sampler");
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
    Check(batch.at("execution_metadata").at("configured_options").at("counts_policy") == "frame_v1", "Checkpoint frames do not report their policy");
    const auto again = replay(checkpoint, batch);
    Check(again.at("results") == batch.at("results"), "Checkpoint frame suffixes did not replay");

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

void TestRequestNoiseAndOptions()
{
    TestExtendedFrameRequests();
    TestAutomaticMixedStateNoise();
    TestShotRows();
    TestFrameSamplerHandle();
    TestRowSelection();
    TestLegacyCountsPolicy();
    TestReadoutExecution();
    TestThermalApproximation();
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
