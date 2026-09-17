// Numerical contract tests through the public request API, without Python/GPU.
#include <boost/json.hpp>
#include <cmath>
#include <string>
namespace j = boost::json;
void Check(bool, const char*);
void Near(double, double);
double Real(const j::value&);
j::object Request(const char*, size_t, const std::string&, const char* = "statevector", size_t = 0);
j::object Call(const j::object&, bool = true, bool = false);

namespace {
j::object Estimate(const std::string& body, j::array observables, j::object channel,
                   const char* method = "density_matrix", size_t qubits = 1) {
  auto request = Request("estimate", qubits, body, method);
  request["observables"] = observables;
  request["noise"] = j::object{{"channels", j::array{channel}}, {"seed", 23}};
  return request;
}
j::object Channel(const char* kind, j::object parameters, j::array targets = {0}) {
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
}

void TestRequestNoiseAndOptions() {
  const double duration = 0.3, t1 = 1, t2 = 0.6, excited = 0.1;
  auto thermal = Channel("thermal_relaxation", {{"duration", duration}, {"t1", t1},
      {"t2", t2}, {"excited_population", excited}});
  auto request = Estimate("h q[0];", {"X", "Z"}, thermal);
  auto result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), std::exp(-duration/t2));
  Near(Real(result.at("expectation_values").at(1)), (1-2*excited)*(1-std::exp(-duration/t1)));
  request["simulator"].as_object()["method"] = "matrix_product_operator";
  SameExpectations(result, Call(request));

  auto two = thermal;
  two["kind"] = "thermal_relaxation_2q";
  request = Estimate("x q[0]; cx q[0],q[1];", {"ZI", "IZ"}, two, "density_matrix", 2);
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), -1+2*(1-excited)*(1-std::exp(-duration/t1)));
  Near(Real(result.at("expectation_values").at(1)), -1);
  request = Estimate("h q[0];", {"X"}, Channel("phase_damping", {{"gamma", 0.36}}));
  Near(Real(Call(request).at("expectation_values").at(0)), 0.8);
  request = Estimate("x q[0];", {"Z"}, Channel("generalized_amplitude_damping", {{"gamma", 0.4}, {"excited_population", 0.25}}));
  Near(Real(Call(request).at("expectation_values").at(0)), -0.4);

  // Idle relaxation is applied to delay duration, not to each ordinary gate.
  request = Estimate("", {"X", "Y", "Z"}, Channel("idle", {{"t1", t1}, {"t2", t2},
      {"excited_population", excited}, {"detuning_hz", 0.25}}));
  auto gate = j::object{{"name", "h"}, {"qubits", j::array{0}}};
  auto delay = j::object{{"name", "delay"}, {"qubits", j::array{0}}, {"duration", duration}};
  request["circuit"] = j::object{{"format", "instructions"}, {"num_qubits", 1}, {"source", j::array{gate, delay}}};
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), std::exp(-duration/t2)*std::cos(2*std::acos(-1.0)*0.25*duration));
  Near(Real(result.at("expectation_values").at(2)), (1-2*excited)*(1-std::exp(-duration/t1)));
  delay["duration"] = duration/2;
  request["circuit"].as_object()["source"] = j::array{gate, delay, delay};
  SameExpectations(result, Call(request));

  // A spectator rotation has a known effect; selecting the wrong qubit fails.
  request = Estimate("h q[1]; x q[0];", {"IX"}, Channel("crosstalk", {{"strength", 0.3}}, {0,1}), "statevector", 2);
  Near(Real(Call(request).at("expectation_values").at(0)), std::cos(0.3));

  const j::array correlated{
      Channel("correlated_ar1", {{"phi", 0.5}, {"sigma_eta", 0.4}}),
      Channel("correlated_ou", {{"sigma", 0.4}, {"alpha", 2}, {"gate_time", 0.1}}),
      Channel("correlated_ou_band", {{"sigma", 0.4}, {"alpha", 2}, {"gate_time", 0.1}}),
      Channel("multi_correlated_ou", {{"bands", j::array{j::array{0.4,2},j::array{0.2,5}}}, {"gate_time", 0.1}}),
      Channel("one_over_f", {{"total_power", 0.2}, {"f_min", 0.1}, {"f_max", 10}, {"num_bands", 8}, {"gate_time", 0.1}})};
  for (const auto& value : correlated) {
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
  for (auto bad : j::array{
      Channel("thermal_relaxation", {{"duration", 1}, {"t1", 1}, {"t2", 3}}),
      Channel("idle", {{"t1", 0}, {"t2", 1}}),
      Channel("correlated_ou", {{"sigma", -1}, {"alpha", 2}, {"gate_time", 0.1}}),
      Channel("multi_correlated_ou", {{"bands", j::array{}}, {"gate_time", 0.1}}),
      Channel("one_over_f", {{"total_power", 1}, {"f_min", 0.1}, {"f_max", 10}, {"num_bands", 0}, {"gate_time", 0.1}}),
      Channel("crosstalk", {{"strength", 0.3}})}) {
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
  for (const char* method : {"density_matrix", "matrix_product_operator", "statevector", "matrix_product_state"}) {
    for (const auto& channel : j::array{
        Channel("t1", {{"gamma", 0.36}}),
        Channel("pauli", {{"px", 0}, {"py", 0}, {"pz", 0.25}}),
        Channel("coherent_rotation", {{"rx", 0.2}, {"ry", 0.1}, {"rz", 0.3}})}) {
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
