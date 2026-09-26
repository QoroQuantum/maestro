// Numerical contract tests through the public request API, without Python/GPU.
#include <boost/json.hpp>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
namespace j = boost::json;
void Check(bool, const char*);
void Near(double, double);
double Real(const j::value&);
j::object Request(const char*, size_t, const std::string&,
                  const char* = "statevector", size_t = 0);
j::object Call(const j::object&, bool = true, bool = false);

namespace {
j::object Estimate(const std::string& body, j::array observables,
                   j::object channel, const char* method = "density_matrix",
                   size_t qubits = 1) {
  auto request = Request("estimate", qubits, body, method);
  request["observables"] = observables;
  request["noise"] = j::object{{"channels", j::array{channel}}, {"seed", 23}};
  return request;
}
j::object Channel(const char* kind, j::object parameters,
                  j::array targets = {0}) {
  parameters["kind"] = kind;
  parameters["targets"] = targets;
  return parameters;
}
void SameExpectations(const j::object& a, const j::object& b) {
  const auto& left = a.at("expectation_values").as_array();
  const auto& right = b.at("expectation_values").as_array();
  Check(left.size() == right.size(), "Expectation shape changed");
  for (size_t i = 0; i < left.size(); ++i) Near(Real(left[i]), Real(right[i]));
}
j::object ThermalCall(const j::object& request, bool sampled,
                      bool validate = false) {
  struct Capture {
    std::ostringstream output;
    std::streambuf* previous = std::cerr.rdbuf(output.rdbuf());
    ~Capture() { std::cerr.rdbuf(previous); }
  } capture;
  const auto result = Call(request, true, validate);
  const auto warning = capture.output.str();
  if (validate || !sampled) {
    Check(warning.empty(), "Validation/exact thermal execution warned");
  } else {
    const std::string expected =
        "Sampled thermal approximation for circuit qubits [0]: effective T2 "
        "clamped to T1.";
    const auto first = warning.find(expected);
    Check(first != std::string::npos, "Missing sampled thermal warning");
    Check(warning.find(expected, first + expected.size()) == std::string::npos,
          "Thermal warning repeated for each realization");
  }
  if (!validate) {
    bool marked = false;
    for (const auto& value : result.at("noise").at("approximations").as_array())
      marked |= value == "thermal_T2_clamped_to_T1";
    Check(marked == sampled, "Incorrect thermal approximation metadata");
  }
  return result;
}

void TestThermalApproximation() {
  // The same seeded sampled channel must behave as an explicit T2=T1 model;
  // exact channels must retain the supplied T2, including the physical limit.
  for (const char* method : {"statevector", "matrix_product_state",
                             "density_matrix", "matrix_product_operator"}) {
    const bool sampled = std::string(method) == "statevector" ||
                         std::string(method) == "matrix_product_state";
    for (const char* kind :
         {"thermal_relaxation", "thermal_relaxation_2q", "idle"}) {
      for (const double t2 : {1.5, 2.0}) {
        auto channel = Channel(kind, {{"t1", 1.0}, {"t2", t2}});
        if (std::string(kind) != "idle") channel["duration"] = 0.3;
        const std::string body = std::string(kind) == "thermal_relaxation_2q"
                                     ? "h q[0]; cx q[1],q[0];"
                                     : "h q[0];";
        auto request = Estimate(body, {"XI"}, channel, method, 2);
        if (std::string(kind) == "idle")
          request["circuit"] = j::object{
              {"format", "instructions"},
              {"num_qubits", 2},
              {"source",
               j::array{j::object{{"name", "h"}, {"qubits", j::array{0}}},
                        j::object{{"name", "delay"},
                                  {"qubits", j::array{0}},
                                  {"duration", 0.3}}}}};
        request["noise"].as_object()["realizations"] = sampled ? 256 : 1;
        ThermalCall(request, sampled, true);
        const auto result = ThermalCall(request, sampled);
        if (sampled) {
          SameExpectations(result, ThermalCall(request, true));
          auto clamped = request;
          clamped["noise"]
              .as_object()["channels"]
              .as_array()[0]
              .as_object()["t2"] = 1.0;
          SameExpectations(result, ThermalCall(clamped, false));
          Check(std::abs(Real(result.at("expectation_values").at(0)) -
                         std::exp(-0.3)) < 0.1,
                "Sampled thermal coherence did not follow effective T2=T1");
        } else {
          Near(Real(result.at("expectation_values").at(0)),
               std::exp(-0.3 / t2));
        }
      }
    }
  }

  // Accepting sampled thermal approximations must not admit exact-only maps
  // or an explicit exact evaluation on a pure-state method.
  auto thermal = Channel("thermal_relaxation",
                         {{"duration", 0.3}, {"t1", 1.0}, {"t2", 1.5}});
  auto request = Estimate("h q[0];", {"XI"}, thermal, "statevector", 2);
  request["noise"].as_object()["evaluation"] = "exact";
  Check(Call(request, false, true).at("error").at("code") ==
            "unsupported_capability",
        "Exact thermal evolution accepted on a pure-state method");
  request["noise"].as_object().erase("evaluation");
  for (const auto& channel : j::array{
           Channel("generalized_amplitude_damping",
                   {{"gamma", 0.2}, {"excited_population", 0.1}}),
           Channel("correlated_phase_flip", {{"probability", 0.2}}, {0, 1}),
           Channel("kraus", {{"operators",
                              j::array{j::value(j::array{1, 0, 0, 1})}}})}) {
    request["noise"].as_object()["channels"] = j::array{thermal, channel};
    Check(Call(request, false, true).at("error").at("code") ==
              "unsupported_capability",
          "Exact-only channel accepted with a sampled thermal approximation");
  }
}
void TestNoisyEstimateResetStreams() {
  for (const char* method : {"statevector", "matrix_product_state"})
    for (const char* kind : {"t1", "t1_2q"})
      for (int seedMode = 0; seedMode < 6; ++seedMode) {
        auto request =
            Estimate("h q[0]; cx q[0],q[1];", {"ZI", "IZ"},
                     Channel(kind, {{"gamma", 0.5}}, {1}), method, 2);
        auto& execution = request["execution"].as_object();
        auto& noise = request["noise"].as_object();
        execution.erase("seed");
        noise.erase("seed");
        noise["realizations"] = 2000;
        if (seedMode == 1 || seedMode == 2 || seedMode == 5)
          execution["seed"] = seedMode == 1 ? 0 : 5;
        if (seedMode == 3 || seedMode == 4 || seedMode == 5)
          noise["seed"] = seedMode == 3 ? 0 : 23;

        const auto result = Call(request);
        const auto& values = result.at("expectation_values").as_array();
        Check(
            std::abs(Real(values[0])) < 0.1 &&
                std::abs(Real(values[1]) - 0.5) < 0.1,
            "Native noise estimates reused injected errors or reset outcomes");
        if (seedMode == 0) {
          Check(result.at("seed") != Call(request).at("seed"),
                "Unseeded native estimates reused the same seed");
          execution["seed"] = result.at("seed");
        }
        SameExpectations(result, Call(request));
      }
}
}  // namespace

namespace {
j::object ReadoutRequest(const char* operation, const char* method,
                         const std::string& body, size_t bits = 2) {
  auto request = Request("execute", 2, body, method, bits);
  if (std::string(operation) == "checkpoint_batch") {
    request["suffixes"] = j::array{request.at("circuit")};
    request["circuit"] = Request("execute", 2, "", method, bits).at("circuit");
    request["operation"] = operation;
  }
  request["noise"] = j::object{
      {"seed", 456},
      {"realizations", 1},
      {"channels", j::array{Channel("readout", {{"p_meas1_prep0", 1.0},
                                                {"p_meas0_prep1", 0.0}})}}};
  return request;
}
j::value ReadoutCounts(const j::object& result) {
  return result.at("operation") == "checkpoint_batch"
             ? result.at("results").at(0).at("counts")
             : result.at("counts");
}
void TestReadoutExecution() {
  for (const char* method : {"statevector", "matrix_product_state",
                             "density_matrix", "matrix_product_operator"}) {
    for (const char* operation : {"execute", "checkpoint_batch"}) {
      for (bool optimize : {false, true}) {
        const std::pair<const char*, const char*> cases[] = {
            {"measure q[0]->c[2]; measure q[1]->c[0];", "0010"},
            {"measure q[0]->c[0]; measure q[0]->c[1];", "1100"},
            {"measure q[0]->c[0]; measure q[1]->c[0];", "0000"},
            {"measure q[0]->c[0]; if(c==1) x q[1]; measure q[1]->c[1];",
             "1100"},
            {"x q[1]; measure q[1]->c[0]; if(c==1) measure q[0]->c[1];",
             "1100"},
            {"if(c==1) measure q[0]->c[0];", "0000"}};
        for (const auto& [body, expected] : cases) {
          auto request = ReadoutRequest(operation, method, body, 4);
          request["simulator"].as_object()["options"] =
              j::object{{"optimize_circuit", optimize}};
          Call(request, true, true);
          Check(ReadoutCounts(Call(request)) == j::object{{expected, 80}},
                "Measurement-time readout or classical control is incorrect");
        }
      }
      for (const int realizations : {1, 128}) {
        auto request = ReadoutRequest(operation, method, "measure q[0]->c[0];");
        request["execution"].as_object()["shots"] = 128;
        request["noise"].as_object()["realizations"] = realizations;
        request["noise"].as_object()["channels"] =
            j::array{Channel("readout", {{"probability", 0.5}})};
        const auto counts = ReadoutCounts(Call(request));
        Check(counts == ReadoutCounts(Call(request)), "Seeded readout changed");
        Check(counts.as_object().size() == 2,
              "Readout draws repeated across shots");
        const double zeros = Real(counts.at("00"));
        Check(zeros > 24 && zeros < 104 && zeros + Real(counts.at("10")) == 128,
              "Readout flips are not independent per shot");
        request["noise"].as_object()["seed"] = 999;
        Check(counts == ReadoutCounts(Call(request)),
              "Noise injection seed changed the explicit readout stream");
        request["execution"].as_object().erase("seed");
        const auto fallback = Call(request);
        Check(fallback.at("seed") == 999,
              "Public noise seed was not inherited");
        request["execution"].as_object()["seed"] = 999;
        Check(ReadoutCounts(fallback) == ReadoutCounts(Call(request)),
              "Implicit and explicit simulator seeds disagree");
        request["execution"].as_object().erase("seed");
        request["simulator"].as_object()["options"] = j::object{{"seed", 0}};
        const auto zero = Call(request);
        Check(zero.at("seed") == 0, "Explicit zero seed lost precedence");
        request["noise"].as_object()["seed"] = 456;
        Check(ReadoutCounts(zero) == ReadoutCounts(Call(request)),
              "Simulator option seed did not control readout");
      }
    }
  }
}

void TestFixedBackendNoisyShots() {
  // A sampled noise realization is chosen at injection time. Multiplying its
  // shots must preserve that choice, while different realizations remain
  // independent. Basis-state circuits make this comparison exact, with the
  // single-shot execution path as the reference.
  for (const char* method : {"statevector", "matrix_product_state"}) {
    for (const auto& channel : j::array{
             Channel("bit_flip", {{"probability", 0.5}}),
             Channel("t1", {{"gamma", 0.5}}),
             Channel(
                 "thermal_relaxation",
                 {{"duration", std::log(2.0)}, {"t1", 1.0}, {"t2", 1.0}})}) {
      for (size_t realizations : {size_t{1}, size_t{64}}) {
        auto request = Request("execute", 1, "x q[0]; measure q->c;", method);
        request["noise"] = j::object{{"seed", 23},
                                     {"realizations", realizations},
                                     {"channels", j::array{channel}}};
        request["execution"].as_object()["shots"] = realizations;
        const auto reference = Call(request).at("counts").as_object();
        request["execution"].as_object()["shots"] = 32 * realizations;
        const auto result = Call(request);
        const auto& counts = result.at("counts").as_object();
        Check(counts.size() == reference.size(),
              "Shot reuse changed the sampled noise realizations");
        for (const auto& entry : reference)
          Near(Real(counts.at(entry.key())), 32 * Real(entry.value()));
        Check(counts.size() == (realizations == 1 ? 1 : 2),
              "Distinct noise realizations reused one injected error");
        Check(result.at("noise").at("realizations").to_number<size_t>() ==
                  realizations,
              "Shot reuse changed the realization count");
        Check(result.at("counts") == Call(request).at("counts"),
              "Seeded noisy shots are not reproducible");
      }
    }
  }

  // A sampled relaxation reset on one half of a Bell pair must not freeze the
  // other half's outcome across shots. Exact channels must retain the same
  // marginal. Exercise both gate relaxation and delay/idle injection.
  for (const char* method : {"statevector", "matrix_product_state",
                             "density_matrix", "matrix_product_operator"}) {
    for (bool idle : {false, true}) {
      auto request =
          Request("execute", 2, "h q[0]; cx q[0],q[1]; measure q->c;", method);
      auto channel = Channel("t1_2q", {{"gamma", 1.0}});
      if (idle) {
        channel = Channel("idle", {{"t1", 1.0}, {"t2", 1.0}});
        request["circuit"] = j::object{
            {"format", "instructions"},
            {"num_qubits", 2},
            {"source",
             j::array{j::object{{"name", "h"}, {"qubits", j::array{0}}},
                      j::object{{"name", "cx"}, {"qubits", j::array{0, 1}}},
                      j::object{{"name", "delay"},
                                {"qubits", j::array{0}},
                                {"duration", 1000.0}},
                      j::object{{"name", "measure"},
                                {"qubits", j::array{0, 1}},
                                {"clbits", j::array{0, 1}}}}}};
      }
      request["execution"].as_object()["shots"] = 4096;
      request["noise"] = j::object{
          {"seed", 23}, {"realizations", 1}, {"channels", j::array{channel}}};
      const auto result = Call(request);
      const auto& counts = result.at("counts").as_object();
      Check(
          counts.size() == 2 && counts.contains("00") && counts.contains("01"),
          "Relaxation reset froze or changed an entangled shot outcome");
      Check(std::abs(Real(counts.at("00")) / 4096 - 0.5) < 0.05,
            "Relaxation reset outcomes were not independent per shot");
      Near(Real(counts.at("00")) + Real(counts.at("01")), 4096);
    }
  }

  // Exact Kraus evolution may be reused, but its sampled measurement still
  // has to drive the conditional separately on every shot.
  for (const char* method : {"density_matrix", "matrix_product_operator"}) {
    auto request = Request(
        "execute", 2,
        "x q[0]; measure q[0]->c[0]; if(c==1) x q[1]; measure q[1]->c[1];",
        method);
    request["execution"].as_object()["shots"] = 4096;
    request["noise"] =
        j::object{{"evaluation", "exact"},
                  {"realizations", 1},
                  {"channels", j::array{Channel("t1", {{"gamma", 0.25}})}}};
    const auto result = Call(request);
    const auto& counts = result.at("counts").as_object();
    Check(counts.size() == 2 && counts.contains("00") && counts.contains("11"),
          "Exact noisy measurements lost their classical dependency");
    Check(std::abs(Real(counts.at("00")) / 4096 - 0.25) < 0.05,
          "Exact channel shot distribution changed");
    Near(Real(counts.at("00")) + Real(counts.at("11")), 4096);
  }
}
}  // namespace

void TestRequestNoiseAndOptions() {
  TestReadoutExecution();
  TestThermalApproximation();
  TestFixedBackendNoisyShots();
  TestNoisyEstimateResetStreams();
  const double duration = 0.3, t1 = 1, t2 = 0.6, excited = 0.1;
  auto thermal =
      Channel("thermal_relaxation", {{"duration", duration},
                                     {"t1", t1},
                                     {"t2", t2},
                                     {"excited_population", excited}});
  auto request = Estimate("h q[0];", {"X", "Z"}, thermal);
  auto result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), std::exp(-duration / t2));
  Near(Real(result.at("expectation_values").at(1)),
       (1 - 2 * excited) * (1 - std::exp(-duration / t1)));
  request["simulator"].as_object()["method"] = "matrix_product_operator";
  SameExpectations(result, Call(request));

  auto two = thermal;
  two["kind"] = "thermal_relaxation_2q";
  request =
      Estimate("x q[0]; cx q[0],q[1];", {"ZI", "IZ"}, two, "density_matrix", 2);
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)),
       -1 + 2 * (1 - excited) * (1 - std::exp(-duration / t1)));
  Near(Real(result.at("expectation_values").at(1)), -1);
  request =
      Estimate("h q[0];", {"X"}, Channel("phase_damping", {{"gamma", 0.36}}));
  Near(Real(Call(request).at("expectation_values").at(0)), 0.8);
  request = Estimate("x q[0];", {"Z"},
                     Channel("generalized_amplitude_damping",
                             {{"gamma", 0.4}, {"excited_population", 0.25}}));
  Near(Real(Call(request).at("expectation_values").at(0)), -0.4);

  // Idle relaxation is applied to delay duration, not to each ordinary gate.
  request = Estimate("", {"X", "Y", "Z"},
                     Channel("idle", {{"t1", t1},
                                      {"t2", t2},
                                      {"excited_population", excited},
                                      {"detuning_hz", 0.25}}));
  auto gate = j::object{{"name", "h"}, {"qubits", j::array{0}}};
  auto delay = j::object{
      {"name", "delay"}, {"qubits", j::array{0}}, {"duration", duration}};
  request["circuit"] = j::object{{"format", "instructions"},
                                 {"num_qubits", 1},
                                 {"source", j::array{gate, delay}}};
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)),
       std::exp(-duration / t2) *
           std::cos(2 * std::acos(-1.0) * 0.25 * duration));
  Near(Real(result.at("expectation_values").at(2)),
       (1 - 2 * excited) * (1 - std::exp(-duration / t1)));
  delay["duration"] = duration / 2;
  request["circuit"].as_object()["source"] = j::array{gate, delay, delay};
  SameExpectations(result, Call(request));

  // A spectator rotation has a known effect; selecting the wrong qubit fails.
  request = Estimate("h q[1]; x q[0];", {"IX"},
                     Channel("crosstalk", {{"strength", 0.3}}, {0, 1}),
                     "statevector", 2);
  Near(Real(Call(request).at("expectation_values").at(0)), std::cos(0.3));

  const j::array correlated{
      Channel("correlated_ar1", {{"phi", 0.5}, {"sigma_eta", 0.4}}),
      Channel("correlated_ou",
              {{"sigma", 0.4}, {"alpha", 2}, {"gate_time", 0.1}}),
      Channel("correlated_ou_band",
              {{"sigma", 0.4}, {"alpha", 2}, {"gate_time", 0.1}}),
      Channel("multi_correlated_ou",
              {{"bands", j::array{j::array{0.4, 2}, j::array{0.2, 5}}},
               {"gate_time", 0.1}}),
      Channel("one_over_f", {{"total_power", 0.2},
                             {"f_min", 0.1},
                             {"f_max", 10},
                             {"num_bands", 8},
                             {"gate_time", 0.1}})};
  for (const auto& value : correlated) {
    auto channel = value.as_object();
    request = Estimate("h q[0];", {"X"}, channel, "statevector");
    request["noise"].as_object()["realizations"] = 16;
    result = Call(request);
    SameExpectations(result, Call(request));
    Check(Real(result.at("expectation_values").at(0)) < 0.999,
          "Correlated channel had no effect");
    request["noise"].as_object()["seed"] = 24;
    Check(std::abs(Real(result.at("expectation_values").at(0)) -
                   Real(Call(request).at("expectation_values").at(0))) > 1e-8,
          "Correlated noise seed had no effect");
    channel["after_1q"] = false;
    channel["after_2q"] = false;
    request["noise"].as_object()["channels"] = j::array{channel};
    Near(Real(Call(request).at("expectation_values").at(0)), 1);
    channel["unused_parameter"] = 1;
    request["noise"].as_object()["channels"] = j::array{channel};
    Check(Call(request, false, true).at("error").at("code") == "invalid_input",
          "Unknown correlated parameter accepted");
  }
  for (auto bad :
       j::array{Channel("thermal_relaxation",
                        {{"duration", 1}, {"t1", 1}, {"t2", 3}}),
                Channel("idle", {{"t1", 0}, {"t2", 1}}),
                Channel("correlated_ou",
                        {{"sigma", -1}, {"alpha", 2}, {"gate_time", 0.1}}),
                Channel("multi_correlated_ou",
                        {{"bands", j::array{}}, {"gate_time", 0.1}}),
                Channel("one_over_f", {{"total_power", 1},
                                       {"f_min", 0.1},
                                       {"f_max", 10},
                                       {"num_bands", 0},
                                       {"gate_time", 0.1}}),
                Channel("crosstalk", {{"strength", 0.3}})}) {
    request = Estimate("h q[0];", {"X"}, bad.as_object());
    Check(Call(request, false, true).at("error").at("code") == "invalid_input",
          "Unphysical noise accepted");
  }

  request = Estimate("h q[0];", {"X", "Z"},
                     Channel("pauli", {{"px", 0}, {"py", 0}, {"pz", 0.25}}),
                     "statevector");
  request["noise"].as_object()["mode"] = "analytical";
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), 0.5);
  Near(Real(result.at("expectation_values").at(1)), 0);
  Check(result.at("noise").at("evaluation") == "analytical_approximation",
        "Analytical approximation not identified");
  request["operation"] = "execute";
  request.erase("observables");
  Call(request, false, true);

  // Optimize only after channels have been placed on the original gate stream.
  for (const char* method : {"density_matrix", "matrix_product_operator",
                             "statevector", "matrix_product_state"}) {
    for (const auto& channel :
         j::array{Channel("t1", {{"gamma", 0.36}}),
                  Channel("pauli", {{"px", 0}, {"py", 0}, {"pz", 0.25}}),
                  Channel("coherent_rotation",
                          {{"rx", 0.2}, {"ry", 0.1}, {"rz", 0.3}})}) {
      request = Estimate("h q[0]; h q[0]; x q[0]; x q[0];", {"X", "Y", "Z"},
                         channel.as_object(), method);
      request["noise"].as_object()["realizations"] = 16;
      request["simulator"].as_object()["options"] =
          j::object{{"optimize_circuit", false}};
      result = Call(request);
      request["simulator"].as_object()["options"] =
          j::object{{"optimize_circuit", true}};
      SameExpectations(result, Call(request));
    }
  }

  // A real option effect: bond dimension two retains a Bell pair, one cannot.
  request = Request("probabilities", 2, "h q[0]; cx q[0],q[1];",
                    "matrix_product_state");
  request["simulator"].as_object()["options"] =
      j::object{{"max_bond_dimension", 2}};
  result = Call(request);
  Near(Real(result.at("probabilities").at(0)), 0.5);
  Near(Real(result.at("probabilities").at(3)), 0.5);
  request["simulator"].as_object()["options"] =
      j::object{{"max_bond_dimension", 1}};
  Check(std::abs(Real(Call(request).at("probabilities").at(0)) - 0.5) > 0.2,
        "MPS bond bound had no effect");
  auto aliased = request;
  aliased["simulator"].as_object()["options"] =
      j::object{{"matrix_product_state_max_bond_dimension", 1}};
  Check(Call(aliased).at("probabilities") == Call(request).at("probabilities"),
        "Option alias changed semantics");
}
