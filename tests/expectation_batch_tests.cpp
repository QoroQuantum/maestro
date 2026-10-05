#include "Simulators/Factory.h"
#include "Simulators/FusionSimulator.h"
#include "Network/SimpleDisconnectedNetwork.h"
#include "maestrolib/Interface.h"
#include <boost/json.hpp>
#include <cmath>
#include <iostream>

namespace {
using Method = Simulators::SimulationType;
using Backend = Simulators::SimulatorType;
using CF = Circuits::CircuitFactory<>;
using Gate = Circuits::QuantumGateType;
using Circuit = Circuits::Circuit<>;
using Net = Network::SimpleDisconnectedNetwork<>;
namespace j = boost::json;
size_t checks = 0;
void Check(bool ok, const char* message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
void Near(std::complex<double> a, std::complex<double> b) {
  Check(std::abs(a - b) < 1e-9, "Expectation mismatch");
}
auto Make(Method method, size_t n = 4) {
  auto sim = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQCSim, method);
  sim->SetMultithreading(false);
  sim->AllocateQubits(n);
  sim->Initialize();
  return sim;
}

void RealBatches() {
  const std::vector<std::string> paulis{"XIZY", "ZZZZ", "IIII", "XIZY", "YIIY", "ZIIZ"};
  for (auto method : {Method::kStatevector, Method::kMatrixProductState,
                      Method::kMatrixProductOperator, Method::kDensityMatrix})
    for (bool fusion : {false, true}) {
      auto sim = Make(method);
      sim->Configure("gate_fusion", fusion ? "true" : "false");
      if (method == Method::kMatrixProductState || method == Method::kMatrixProductOperator)
        sim->SetInitialQubitsMap({3, 1, 0, 2});
      sim->ApplyH(0);
      sim->ApplyRy(2, .31);
      sim->ApplyCX(0, 3);
      sim->ApplyRz(3, .17);
      const auto values = sim->ExpectationValues(paulis); // flush pending gates
      Check(values.size() == paulis.size(), "Batch changed length");
      for (size_t i = 0; i < paulis.size(); ++i) Near(values[i], sim->ExpectationValue(paulis[i]));
      Check(sim->ExpectationValues({}).empty(), "Empty real batch");
      const std::vector<std::string> padded{"", "Z", "zi", "IIIIZ", "IIIIX"};
      const auto padding = sim->ExpectationValues(padded);
      for (size_t i = 0; i < padded.size(); ++i) Near(padding[i], sim->ExpectationValue(padded[i]));
      sim->SaveState();
      sim->ApplyX(0);
      const auto changed = sim->ExpectationValues(paulis);
      for (size_t i = 0; i < paulis.size(); ++i) Near(changed[i], sim->ExpectationValue(paulis[i]));
      sim->RestoreState();
      const auto restored = sim->ExpectationValues(paulis);
      for (size_t i = 0; i < values.size(); ++i) Near(values[i], restored[i]);
    }
}

void ComplexAndC() {
  auto sim = Make(Method::kMatrixProductOperator, 2);
  sim->ApplyH(0);
  sim->ApplyCX(0, 1);
  Eigen::Matrix2cd scale = 2. * Eigen::Matrix2cd::Identity();
  sim->ApplyOperator({0}, scale, false); // trace four, not one
  sim->MoveAtBeginningOfChain({1});
  const std::vector<std::string> paulis{"XX", "YY", "ZZ", "II", "XX"};
  for (bool normalized : {false, true}) {
    const auto values = sim->ExpectationValuesComplex(paulis, normalized);
    for (size_t i = 0; i < paulis.size(); ++i)
      Near(values[i], sim->ExpectationValueComplex(paulis[i], normalized));
    Near(values[0], normalized ? 1. : 4.);
    Near(values[1], normalized ? -1. : -4.);
  }
  Check(sim->ExpectationValuesComplex({}).empty(), "Empty complex batch");
  const char* input[]{"XX", "YY", "II"};
  double real[4]{9, 9, 9, 9}, imag[4]{9, 9, 9, 9};
  Check(MaestroExpectationValues(sim.get(), input, 3, real, 4), "C real batch failed");
  Near(real[0], 1); Near(real[1], -1); Near(real[2], 1); Near(real[3], 9);
  Check(MaestroExpectationValuesComplex(sim.get(), input, 3, 0, real, imag, 4), "C complex batch failed");
  Near(real[0], 4); Near(real[1], -4); Near(imag[0], 0); Near(imag[3], 9);
  const char* invalid[]{"XX", "bad"};
  Check(!MaestroExpectationValuesComplex(sim.get(), invalid, 2, 1, real, imag, 4), "C invalid Pauli accepted");
  Near(real[0], 4); Near(imag[0], 0);
  Check(!MaestroExpectationValues(sim.get(), input, 3, real, 2), "C capacity ignored");
  Near(real[0], 4);
  const char* nullString[]{"XX", nullptr};
  Check(!MaestroExpectationValues(sim.get(), nullString, 2, real, 4), "C null string accepted");
  Check(!MaestroExpectationValues(sim.get(), nullptr, 1, real, 4), "C null input accepted");
  Check(!MaestroExpectationValues(sim.get(), input, 1, nullptr, 4), "C null output accepted");
  Check(MaestroExpectationValues(sim.get(), nullptr, 0, nullptr, 0), "C empty real batch failed");
  Check(MaestroExpectationValuesComplex(sim.get(), nullptr, 0, 1, nullptr, nullptr, 0), "C empty complex batch failed");
  auto unsupported = Make(Method::kStatevector, 2);
  Check(!MaestroExpectationValuesComplex(unsupported.get(), input, 3, 1, real, imag, 4), "Unsupported complex query accepted");
}

struct Recording : Simulators::Private::FusionSimulator<Simulators::Private::FusionState> {
  using Base = Simulators::Private::FusionSimulator<Simulators::Private::FusionState>;
  explicit Recording(std::shared_ptr<Simulators::ISimulator> sim) : Base(std::move(sim)) {}
  size_t batches = 0, singles = 0;
  std::vector<double> ExpectationValues(const std::vector<std::string>& p) override {
    ++batches;
    return Base::ExpectationValues(p);
  }
  double ExpectationValue(const std::string& p) override {
    ++singles;
    return Base::ExpectationValue(p);
  }
  std::unique_ptr<Simulators::ISimulator> Clone() override { throw std::logic_error("Unexpected clone"); }
};
struct QueryNetwork : Net {
  explicit QueryNetwork(const std::shared_ptr<Recording>& sim) : Net({4}, {4}) {
    simulator = sim;
    recreateIfNeeded = false;
  }
  ExecuteResults RepeatedExecute(const std::shared_ptr<Circuit>&, size_t) override { return {}; }
  ExecuteResults RepeatedExecuteOnHost(const std::shared_ptr<Circuit>&, size_t, size_t) override { return {}; }
};
void NetworkDispatch() {
  auto record = std::make_shared<Recording>(Make(Method::kMatrixProductState));
  record->ApplyX(3);
  QueryNetwork query(record);
  auto circuit = CF::CreateCircuit();
  for (auto values : {query.ExecuteExpectations(circuit, {"IIIZ", "ZIII"}),
                      query.ExecuteOnHostExpectations(circuit, 0, {"IIIZ", "ZIII"})}) {
    Near(values[0], -1); Near(values[1], 1);
  }
  Check(record->batches == 2 && record->singles == 0, "Network bypassed batch dispatch");
  circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 3));
  circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
  auto network = std::make_shared<Net>(Types::qubits_vector{4}, std::vector<size_t>{4});
  network->SetOptimizeSimulator(false);
  network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kMatrixProductState);
  network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState);
  const auto mapped = network->ExecuteOnHostExpectations(circuit, 0, {"IIIZ", "XIII", "IXII", "XIII"});
  Near(mapped[0], -1); Near(mapped[1], 1); Near(mapped[2], 0); Near(mapped[3], 1);
}

j::object Request(const char* operation, const char* method) {
  return {{"schema_version", 2}, {"operation", operation},
          {"circuit", j::object{{"num_qubits", 2}, {"source", "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"}}},
          {"simulator", j::object{{"backend", "qcsim"}, {"method", method}}}};
}
j::object Call(const j::object& request, bool valid = true, bool validate = false) {
  const auto input = j::serialize(request);
  auto* raw = validate ? MaestroValidateRequestJson(input.c_str()) : MaestroRunRequestJson(input.c_str());
  Check(raw != nullptr, "Null JSON response");
  const auto result = j::parse(raw).as_object();
  FreeResult(raw);
  if (result.at("ok").as_bool() != valid) throw std::runtime_error(j::serialize(result));
  return result;
}
void JsonBatches() {
  for (const char* method : {"matrix_product_state", "matrix_product_operator"}) {
    auto request = Request("estimate", method);
    request["observables"] = j::array{"XX", "YY", "ZZ", "XX"};
    const auto values = Call(request).at("expectation_values").as_array();
    Near(values[0].to_number<double>(), 1); Near(values[1].to_number<double>(), -1);
    Near(values[2].to_number<double>(), 1); Near(values[3].to_number<double>(), 1);
    request["operation"] = "incremental_evolve";
    request["step_circuit"] = j::object{{"num_qubits", 2}, {"source", "OPENQASM 2.0; qreg q[2]; x q[0];"}};
    request["steps"] = j::array{0, 1, 2};
    const auto rows = Call(request).at("expectation_values").as_array();
    for (size_t r = 0; r < 3; ++r) {
      Near(rows[r].at(0).to_number<double>(), 1);
      Near(rows[r].at(2).to_number<double>(), r == 1 ? -1. : 1.);
    }
  }
  auto request = Request("diagnostics", "matrix_product_operator");
  request["observables"] = j::array{"XX", "YY", "XX"};
  request["diagnostics"] = j::array{"expectations_complex", "unnormalized_expectations"};
  request["operators"] = j::array{j::object{{"qubits", j::array{0}},
      {"matrix", j::array{j::array{2, 0}, j::array{0, 0}, j::array{0, 0}, j::array{2, 0}}}}};
  const auto result = Call(request);
  Near(result.at("expectations_complex").at(0).at(0).to_number<double>(), 1);
  Near(result.at("unnormalized_expectations").at(1).at(0).to_number<double>(), -4);
  request["max_output_elements"] = 2;
  Call(request, false, true);
  request.erase("max_output_elements");
  request["observables"] = j::array{"wrong"};
  Call(request, false, true);
  request["observables"] = j::array{};
  Check(Call(request).at("expectations_complex").as_array().empty(), "JSON empty batch");
  request.erase("observables");
  Call(request, false, true);
}
}  // namespace

int main() {
  try {
    RealBatches(); ComplexAndC(); NetworkDispatch(); JsonBatches();
    std::cout << checks << " expectation batch checks passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
