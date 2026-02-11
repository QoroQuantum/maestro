#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

// Domain Headers
#include "Circuit/Circuit.h"
#include "Interface.h"
#include "Maestro.h"
#include "Simulators/Factory.h"
#include "Simulators/Simulator.h"
#include "qasm/QasmCirc.h"
#include "Network/SimpleDisconnectedNetwork.h"

namespace nb = nanobind;
using namespace nb::literals;

// ============================================================================
// Internal Implementation Helpers (Hidden from Python)
// ============================================================================

namespace {

// RAII Wrapper to ensure the simulator handle is destroyed strictly
struct ScopedSimulator {
  unsigned long int handle;

  explicit ScopedSimulator(int num_qubits) {
    GetMaestroObjectWithMute();
    handle = CreateSimpleSimulator(num_qubits);
  }

  ~ScopedSimulator() {
    if (handle != 0) DestroySimpleSimulator(handle);
  }

  // Disable copying to prevent double-free
  ScopedSimulator(const ScopedSimulator &) = delete;
  ScopedSimulator &operator=(const ScopedSimulator &) = delete;
};

// Helper to configure the simulation network
std::shared_ptr<Network::INetwork<double>> ConfigureNetwork(
    unsigned long int handle, Simulators::SimulatorType sim_type,
    Simulators::SimulationType sim_exec_type, std::optional<size_t> max_bond,
    std::optional<double> sv_threshold) {
  if (RemoveAllOptimizationSimulatorsAndAdd(handle, (int)sim_type,
                                            (int)sim_exec_type) == 0) {
    return nullptr;
  }

  auto *maestro = static_cast<Maestro *>(GetMaestroObject());
  auto network = maestro->GetSimpleSimulator(handle);

  if (!network) return nullptr;

  if (max_bond) {
    auto val = std::to_string(*max_bond);
    network->Configure("matrix_product_state_max_bond_dimension", val.c_str());
  }
  if (sv_threshold) {
    auto val = std::to_string(*sv_threshold);
    network->Configure("matrix_product_state_truncation_threshold",
                       val.c_str());
  }

  network->CreateSimulator();
  return network;
}

// Helper to parse observables from String (";" sep) or List[str]
std::vector<std::string> ParseObservables(const nb::object &observables) {
  std::vector<std::string> paulis;

  if (nb::isinstance<nb::str>(observables)) {
    std::string obsStr = nb::cast<std::string>(observables);
    std::stringstream ss(obsStr);
    std::string item;
    while (std::getline(ss, item, ';')) {
      // Trim whitespace if necessary, usually safe to skip empty
      if (!item.empty()) paulis.push_back(item);
    }
  } else if (nb::isinstance<nb::list>(observables)) {
    paulis = nb::cast<std::vector<std::string>>(observables);
  } else {
    throw nb::type_error(
        "Observables must be a ';'-separated string or a list of strings.");
  }
  return paulis;
}

// Core Execution Logic
nb::dict execute_core(std::shared_ptr<Circuits::Circuit<double>> circuit,
                      Simulators::SimulatorType sim_type,
                      Simulators::SimulationType sim_exec_type, int shots,
                      std::optional<size_t> max_bond,
                      std::optional<double> sv_threshold) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, sim_type, sim_exec_type, max_bond,
                                  sv_threshold);
  if (!network) throw std::runtime_error("Failed to configure network.");

  Network::INetwork<double>::ExecuteResults raw_results;

  // Release GIL for heavy computation
  auto start = std::chrono::high_resolution_clock::now();
  {
    nb::gil_scoped_release release;
    raw_results = network->RepeatedExecuteOnHost(circuit, 0, (size_t)shots);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Process results back in Python land
  nb::dict counts;
  for (const auto &pair : raw_results) {
    // Optimization: Pre-allocate string to avoid repeated reallocation
    const auto &bool_vec = pair.first;
    std::string bitstring(bool_vec.size(), '0');
    for (size_t i = 0; i < bool_vec.size(); ++i) {
      if (bool_vec[i]) bitstring[i] = '1';
    }
    counts[bitstring.c_str()] = pair.second;
  }

  nb::dict py_result;
  py_result["counts"] = counts;
  py_result["time_taken"] = std::chrono::duration<double>(end - start).count();
  py_result["simulator"] = (int)network->GetLastSimulatorType();
  py_result["method"] = (int)network->GetLastSimulationType();

  return py_result;
}

// Core Estimation Logic
nb::dict estimate_core(std::shared_ptr<Circuits::Circuit<double>> circuit,
                       const std::vector<std::string> &paulis,
                       Simulators::SimulatorType sim_type,
                       Simulators::SimulationType sim_exec_type,
                       std::optional<size_t> max_bond,
                       std::optional<double> sv_threshold) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits = static_cast<int>(circuit->GetMaxQubitIndex()) + 1;
  for (const auto &p : paulis)
    num_qubits = std::max(num_qubits, (int)p.length());

  ScopedSimulator sim(std::max(1, num_qubits));
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, sim_type, sim_exec_type, max_bond,
                                  sv_threshold);
  if (!network) throw std::runtime_error("Failed to configure network.");

  std::vector<double> expectations;

  // Release GIL
  auto start = std::chrono::high_resolution_clock::now();
  {
    nb::gil_scoped_release release;
    expectations = network->ExecuteOnHostExpectations(circuit, 0, paulis);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Convert to Python list
  nb::list exp_vals;
  for (double val : expectations) exp_vals.append(val);

  nb::dict py_result;
  py_result["expectation_values"] = exp_vals;
  py_result["time_taken"] = std::chrono::duration<double>(end - start).count();
  py_result["simulator"] = (int)network->GetLastSimulatorType();
  py_result["method"] = (int)network->GetLastSimulationType();

  return py_result;
}
}  // namespace

// ============================================================================
// Module Definition
// ============================================================================

NB_MODULE(maestro, m) {
  m.doc() = "Python bindings for Maestro Quantum Simulator";

  // --- Enums ---
  nb::enum_<Simulators::SimulatorType>(m, "SimulatorType")
      .value("QCSim", Simulators::SimulatorType::kQCSim)
#ifndef NO_QISKIT_AER
      .value("QiskitAer", Simulators::SimulatorType::kQiskitAer)
      .value("CompositeQiskitAer",
             Simulators::SimulatorType::kCompositeQiskitAer)
#endif
      .value("CompositeQCSim", Simulators::SimulatorType::kCompositeQCSim)
      .value("Gpu", Simulators::SimulatorType::kGpuSim)
      .export_values();

  nb::enum_<Simulators::SimulationType>(m, "SimulationType")
      .value("Statevector", Simulators::SimulationType::kStatevector)
      .value("MatrixProductState",
             Simulators::SimulationType::kMatrixProductState)
      .value("Stabilizer", Simulators::SimulationType::kStabilizer)
      .value("TensorNetwork", Simulators::SimulationType::kTensorNetwork)
      .export_values();

  // --- Maestro Class ---
  nb::class_<Maestro>(m, "Maestro")
      .def(nb::init<>())
      .def("CreateSimulator", &Maestro::CreateSimulator,
           "simType"_a = Simulators::SimulatorType::kQCSim,
           "simExecType"_a = Simulators::SimulationType::kMatrixProductState)
      .def(
          "GetSimulator",
          [](Maestro &self, unsigned long int h) {
            return static_cast<Simulators::ISimulator *>(self.GetSimulator(h));
          },
          nb::rv_policy::reference_internal)
      .def("DestroySimulator", &Maestro::DestroySimulator);

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
      // Measurement
      .def("measure",
           [](Circuits::Circuit<double> &s,
              const std::vector<std::pair<Types::qubit_t, size_t>> &q) {
             s.AddOperation(
                 std::make_shared<Circuits::MeasurementOperation<>>(q));
           })
      // Bound Methods for Direct Execution
      .def("execute", &execute_core,
           "simulator_type"_a = Simulators::SimulatorType::kQCSim,
           "simulation_type"_a = Simulators::SimulationType::kStatevector,
           "shots"_a = 1024, "max_bond_dimension"_a = 2,
           "singular_value_threshold"_a = 1e-8)
      .def(
          "estimate",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const nb::object &observables, Simulators::SimulatorType st,
             Simulators::SimulationType set, std::optional<size_t> mb,
             std::optional<double> sv) {
            return estimate_core(self, ParseObservables(observables), st, set,
                                 mb, sv);
          },
          "observables"_a,
          "simulator_type"_a = Simulators::SimulatorType::kQCSim,
          "simulation_type"_a = Simulators::SimulationType::kStatevector,
          "max_bond_dimension"_a = 2, "singular_value_threshold"_a = 1e-8);

  // --- QASM Tools ---
  nb::class_<qasm::QasmToCirc<double>>(m, "QasmToCirc")
      .def(nb::init<>())
      .def("ParseAndTranslate", &qasm::QasmToCirc<double>::ParseAndTranslate);

  // --- Module Level Convenience Functions ---

  // 1. simple_execute (Overloaded)
  // Variant A: Circuit Object
  m.def("simple_execute", &execute_core, "circuit"_a,
        "simulator_type"_a = Simulators::SimulatorType::kQCSim,
        "simulation_type"_a = Simulators::SimulationType::kStatevector,
        "shots"_a = 1024, "max_bond_dimension"_a = 2,
        "singular_value_threshold"_a = 1e-8);

  // Variant B: QASM String
  m.def(
      "simple_execute",
      [](const std::string &qasm, Simulators::SimulatorType st,
         Simulators::SimulationType set, int shots, std::optional<size_t> mb,
         std::optional<double> sv) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit) {
          // IMPROVEMENT: Throw error instead of silent failure
          throw nb::value_error("Failed to parse QASM string.");
        }
        return execute_core(circuit, st, set, shots, mb, sv);
      },
      "qasm_circuit"_a, "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "shots"_a = 1024, "max_bond_dimension"_a = 2,
      "singular_value_threshold"_a = 1e-8);

  // 2. simple_estimate (Overloaded)
  // Variant A: Circuit Object
  m.def(
      "simple_estimate",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const nb::object &obs, Simulators::SimulatorType st,
         Simulators::SimulationType set, std::optional<size_t> mb,
         std::optional<double> sv) {
        return estimate_core(circuit, ParseObservables(obs), st, set, mb, sv);
      },
      "circuit"_a, "observables"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = 2, "singular_value_threshold"_a = 1e-8);

  // Variant B: QASM String
  m.def(
      "simple_estimate",
      [](const std::string &qasm, const nb::object &obs,
         Simulators::SimulatorType st, Simulators::SimulationType set,
         std::optional<size_t> mb, std::optional<double> sv) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit) {
          throw nb::value_error("Failed to parse QASM string.");
        }
        return estimate_core(circuit, ParseObservables(obs), st, set, mb, sv);
      },
      "qasm_circuit"_a, "observables"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = 2, "singular_value_threshold"_a = 1e-8);
}
