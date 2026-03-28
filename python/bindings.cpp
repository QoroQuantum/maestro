#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
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
    std::optional<double> sv_threshold, bool use_double_precision = false) {
  // QuEST only supports statevector simulation
  if (sim_type == Simulators::SimulatorType::kQuestSim &&
      sim_exec_type != Simulators::SimulationType::kStatevector) {
    throw std::invalid_argument(
        "QuestSim only supports Statevector simulation type.");
  }

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
  if (use_double_precision) {
    network->Configure("use_double_precision", "1");
  }

  // Always create the default simulator (no parameters = QCSim MPS).
  // The desired simulator type is specified via
  // RemoveAllOptimizationSimulatorsAndAdd above.
  network->CreateSimulator();

  // Verify the simulator was actually created (e.g. GPU library may fail)
  if (!network->GetSimulator()) {
    return nullptr;
  }

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
                      std::optional<double> sv_threshold,
                      bool use_double_precision = false) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, sim_type, sim_exec_type, max_bond,
                                  sv_threshold, use_double_precision);
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
                       std::optional<double> sv_threshold,
                       bool use_double_precision = false) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits = static_cast<int>(circuit->GetMaxQubitIndex()) + 1;
  for (const auto &p : paulis)
    num_qubits = std::max(num_qubits, (int)p.length());

  ScopedSimulator sim(std::max(1, num_qubits));
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, sim_type, sim_exec_type, max_bond,
                                  sv_threshold, use_double_precision);
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
// Core Statevector Logic
std::vector<std::complex<double>> statevector_core(
    std::shared_ptr<Circuits::Circuit<double>> circuit,
    Simulators::SimulatorType sim_type,
    Simulators::SimulationType sim_exec_type, std::optional<size_t> max_bond,
    std::optional<double> sv_threshold, bool use_double_precision = false) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, sim_type, sim_exec_type, max_bond,
                                  sv_threshold, use_double_precision);
  if (!network) throw std::runtime_error("Failed to configure network.");

  std::vector<std::complex<double>> amplitudes;
  {
    nb::gil_scoped_release release;
    amplitudes = network->ExecuteOnHostAmplitudes(circuit, 0);
  }
  return amplitudes;
}

// Helper: Create the adjoint (inverse) of a single quantum gate operation.
// Non-gate operations (measurements, resets, etc.) return nullptr and are
// skipped when building the mirror circuit.
using OperationPtr = std::shared_ptr<Circuits::IOperation<double>>;

OperationPtr adjoint_gate(const OperationPtr &op) {
  if (op->GetType() != Circuits::OperationType::kGate) return nullptr;

  auto gate =
      std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
  if (!gate) return nullptr;

  const auto gt = gate->GetGateType();
  const auto params = gate->GetParams();

  switch (gt) {
    // ---- Self-inverse (Hermitian) gates ----
    case Circuits::QuantumGateType::kXGateType:
    case Circuits::QuantumGateType::kYGateType:
    case Circuits::QuantumGateType::kZGateType:
    case Circuits::QuantumGateType::kHadamardGateType:
    case Circuits::QuantumGateType::kKGateType:
    case Circuits::QuantumGateType::kCXGateType:
    case Circuits::QuantumGateType::kCYGateType:
    case Circuits::QuantumGateType::kCZGateType:
    case Circuits::QuantumGateType::kCHGateType:
    case Circuits::QuantumGateType::kSwapGateType:
    case Circuits::QuantumGateType::kCCXGateType:
    case Circuits::QuantumGateType::kCSwapGateType:
      return op->Clone();

    // ---- Paired gates ----
    case Circuits::QuantumGateType::kSGateType:
      return std::make_shared<Circuits::SdgGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSdgGateType:
      return std::make_shared<Circuits::SGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kTGateType:
      return std::make_shared<Circuits::TdgGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kTdgGateType:
      return std::make_shared<Circuits::TGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSxGateType:
      return std::make_shared<Circuits::SxDagGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSxDagGateType:
      return std::make_shared<Circuits::SxGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kCSxGateType:
      return std::make_shared<Circuits::CSxDagGate<>>(gate->GetQubit(0),
                                                       gate->GetQubit(1));
    case Circuits::QuantumGateType::kCSxDagGateType:
      return std::make_shared<Circuits::CSxGate<>>(gate->GetQubit(0),
                                                    gate->GetQubit(1));

    // ---- Parametric single-qubit: negate angle ----
    case Circuits::QuantumGateType::kPhaseGateType:
      return std::make_shared<Circuits::PhaseGate<>>(gate->GetQubit(),
                                                      -params[0]);
    case Circuits::QuantumGateType::kRxGateType:
      return std::make_shared<Circuits::RxGate<>>(gate->GetQubit(),
                                                   -params[0]);
    case Circuits::QuantumGateType::kRyGateType:
      return std::make_shared<Circuits::RyGate<>>(gate->GetQubit(),
                                                   -params[0]);
    case Circuits::QuantumGateType::kRzGateType:
      return std::make_shared<Circuits::RzGate<>>(gate->GetQubit(),
                                                   -params[0]);

    // ---- U gate: U†(θ,φ,λ,γ) = U(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kUGateType:
      return std::make_shared<Circuits::UGate<>>(
          gate->GetQubit(), -params[0], -params[2], -params[1], -params[3]);

    // ---- Controlled parametric: negate angle ----
    case Circuits::QuantumGateType::kCPGateType:
      return std::make_shared<Circuits::CPGate<>>(gate->GetQubit(0),
                                                   gate->GetQubit(1),
                                                   -params[0]);
    case Circuits::QuantumGateType::kCRxGateType:
      return std::make_shared<Circuits::CRxGate<>>(gate->GetQubit(0),
                                                    gate->GetQubit(1),
                                                    -params[0]);
    case Circuits::QuantumGateType::kCRyGateType:
      return std::make_shared<Circuits::CRyGate<>>(gate->GetQubit(0),
                                                    gate->GetQubit(1),
                                                    -params[0]);
    case Circuits::QuantumGateType::kCRzGateType:
      return std::make_shared<Circuits::CRzGate<>>(gate->GetQubit(0),
                                                    gate->GetQubit(1),
                                                    -params[0]);

    // ---- CU gate: CU†(θ,φ,λ,γ) = CU(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kCUGateType:
      return std::make_shared<Circuits::CUGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0], -params[2],
          -params[1], -params[3]);

    default:
      return op->Clone();  // Fallback: clone as-is
  }
}

// Core Mirror Fidelity Logic
// Builds circuit + adjoint(circuit) in reverse, returns P(|0...0>).
// By default uses shot-based sampling. Set full_amplitude=true for exact
// statevector computation (only feasible for small qubit counts).
double mirror_fidelity_core(
    std::shared_ptr<Circuits::Circuit<double>> circuit,
    Simulators::SimulatorType sim_type,
    Simulators::SimulationType sim_exec_type, int shots,
    std::optional<size_t> max_bond, std::optional<double> sv_threshold,
    bool use_double_precision = false, bool full_amplitude = false) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  // Build the mirror circuit: forward gates + adjoint gates in reverse
  auto mirror = std::make_shared<Circuits::Circuit<double>>();
  const auto &ops = circuit->GetOperations();

  // Forward pass: add only gate operations (skip measurements)
  for (const auto &op : ops) {
    if (op->GetType() == Circuits::OperationType::kGate) {
      mirror->AddOperation(op->Clone());
    }
  }

  // Reverse pass: iterate backward and add adjoint of each gate
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    auto adj = adjoint_gate(*it);
    if (adj) mirror->AddOperation(adj);
  }

  // Helper lambda for the shot-based path
  auto run_shot_based = [&]() -> double {
    // Need a fresh mirror circuit since measurements mutate it
    auto mirror_copy = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : mirror->GetOperations()) {
      mirror_copy->AddOperation(op->Clone());
    }

    size_t n = std::max(1, static_cast<int>(mirror_copy->GetMaxQubitIndex()) + 1);
    std::vector<std::pair<Types::qubit_t, size_t>> pairs;
    pairs.reserve(n);
    for (size_t i = 0; i < n; ++i)
      pairs.emplace_back(static_cast<Types::qubit_t>(i), i);
    mirror_copy->AddOperation(
        std::make_shared<Circuits::MeasurementOperation<>>(pairs));

    nb::dict result = execute_core(mirror_copy, sim_type, sim_exec_type, shots,
                                   max_bond, sv_threshold, use_double_precision);

    nb::dict counts = nb::cast<nb::dict>(result["counts"]);
    std::string zeros(n, '0');
    size_t zero_count = 0;
    if (counts.contains(zeros.c_str())) {
      zero_count = nb::cast<size_t>(counts[zeros.c_str()]);
    }
    return static_cast<double>(zero_count) / static_cast<double>(shots);
  };

  if (full_amplitude) {
    // Try exact statevector; fall back to shots if unsupported
    try {
      const auto amplitudes = statevector_core(mirror, sim_type, sim_exec_type,
                                               max_bond, sv_threshold,
                                               use_double_precision);
      if (!amplitudes.empty()) return std::norm(amplitudes[0]);
    } catch (...) {
      // Statevector not available for this backend — fall back to shots
    }
    // Issue a Python warning so the user knows we fell back
    PyErr_WarnEx(PyExc_RuntimeWarning,
        "full_amplitude mode not supported by this simulator/simulation "
        "type. Falling back to shot-based sampling.", 1);
    return run_shot_based();
  } else {
    return run_shot_based();
  }
}

// Core Inner Product Logic
// Computes <psi_1|psi_2> = <0|U1† U2|0> via ProjectOnZero.
std::complex<double> inner_product_core(
    const std::shared_ptr<Circuits::Circuit<double>> &circuit_1,
    const std::shared_ptr<Circuits::Circuit<double>> &circuit_2,
    Simulators::SimulatorType sim_type,
    Simulators::SimulationType sim_exec_type,
    std::optional<size_t> max_bond,
    std::optional<double> sv_threshold,
    bool use_double_precision = false) {
  if (!circuit_1) throw nb::value_error("circuit_1 is null.");
  if (!circuit_2) throw nb::value_error("circuit_2 is null.");

  // Build combined circuit for <0| U1† U2 |0>.
  // Circuit gates are applied left-to-right, so we place U2's gates first
  // (they act on |0> first), then U1†'s gates (applied last = leftmost in
  // the matrix product).
  auto combined = std::make_shared<Circuits::Circuit<double>>();
  const auto &ops1 = circuit_1->GetOperations();
  const auto &ops2 = circuit_2->GetOperations();

  // Forward pass of circuit_2: gate operations only
  for (const auto &op : ops2) {
    if (op->GetType() == Circuits::OperationType::kGate) {
      combined->AddOperation(op->Clone());
    }
  }

  // Adjoint of circuit_1: reverse order, each gate adjointed
  for (auto it = ops1.rbegin(); it != ops1.rend(); ++it) {
    auto adj = adjoint_gate(*it);
    if (adj) combined->AddOperation(adj);
  }

  int num_qubits = std::max(
      1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, sim_type, sim_exec_type,
                                  max_bond, sv_threshold, use_double_precision);
  if (!network) throw std::runtime_error("Failed to configure network.");

  std::complex<double> result;
  {
    nb::gil_scoped_release release;
    result = network->ExecuteOnHostProjectOnZero(combined, 0);
  }
  return result;
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
      .value("QuestSim", Simulators::SimulatorType::kQuestSim)
      .export_values();

  nb::enum_<Simulators::SimulationType>(m, "SimulationType")
      .value("Statevector", Simulators::SimulationType::kStatevector)
      .value("MatrixProductState",
             Simulators::SimulationType::kMatrixProductState)
      .value("Stabilizer", Simulators::SimulationType::kStabilizer)
      .value("TensorNetwork", Simulators::SimulationType::kTensorNetwork)
      .value("PauliPropagator", Simulators::SimulationType::kPauliPropagator)
      .value("ExtendedStabilizer",
             Simulators::SimulationType::kExtendedStabilizer)
      .export_values();

  // --- Maestro Class ---
  nb::class_<Maestro>(m, "Maestro")
      .def(nb::init<>())
      .def("create_simulator", &Maestro::CreateSimulator,
           "sim_type"_a = Simulators::SimulatorType::kQCSim,
           "sim_exec_type"_a = Simulators::SimulationType::kMatrixProductState)
      .def(
          "get_simulator",
          [](Maestro &self, unsigned long int h) {
            return static_cast<Simulators::ISimulator *>(self.GetSimulator(h));
          },
          nb::rv_policy::reference_internal)
      .def("destroy_simulator", &Maestro::DestroySimulator);

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
      // Bound Methods for Direct Execution
      .def("execute", &execute_core,
           "simulator_type"_a = Simulators::SimulatorType::kQCSim,
           "simulation_type"_a = Simulators::SimulationType::kStatevector,
           "shots"_a = 1024, "max_bond_dimension"_a = 2,
           "singular_value_threshold"_a = 1e-8,
           "use_double_precision"_a = false)
      .def(
          "estimate",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const nb::object &observables, Simulators::SimulatorType st,
             Simulators::SimulationType set, std::optional<size_t> mb,
             std::optional<double> sv, bool use_dp) {
            return estimate_core(self, ParseObservables(observables), st, set,
                                 mb, sv, use_dp);
          },
          "observables"_a,
          "simulator_type"_a = Simulators::SimulatorType::kQCSim,
          "simulation_type"_a = Simulators::SimulationType::kStatevector,
          "max_bond_dimension"_a = 2, "singular_value_threshold"_a = 1e-8,
          "use_double_precision"_a = false)
      .def("get_statevector", &statevector_core,
          "simulator_type"_a = Simulators::SimulatorType::kQCSim,
          "simulation_type"_a = Simulators::SimulationType::kStatevector,
          "max_bond_dimension"_a = nb::none(),
          "singular_value_threshold"_a = nb::none(),
          "use_double_precision"_a = false,
          "Get the full statevector (complex amplitudes) after executing the "
          "circuit.")
      .def("mirror_fidelity", &mirror_fidelity_core,
          "simulator_type"_a = Simulators::SimulatorType::kQCSim,
          "simulation_type"_a = Simulators::SimulationType::kStatevector,
          "shots"_a = 1024,
          "max_bond_dimension"_a = nb::none(),
          "singular_value_threshold"_a = nb::none(),
          "use_double_precision"_a = false,
          "full_amplitude"_a = false,
          "Compute mirror fidelity: run circuit forward then its adjoint in "
          "reverse, returning P(|0...0>). Uses shot-based sampling by "
          "default. Set full_amplitude=True for exact statevector "
          "computation (small circuits only).")
      .def("inner_product",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             std::shared_ptr<Circuits::Circuit<double>> other,
             Simulators::SimulatorType st,
             Simulators::SimulationType set,
             std::optional<size_t> mb,
             std::optional<double> sv, bool use_dp) {
            return inner_product_core(self, other, st, set, mb, sv, use_dp);
          },
          "other"_a,
          "simulator_type"_a = Simulators::SimulatorType::kQCSim,
          "simulation_type"_a = Simulators::SimulationType::kStatevector,
          "max_bond_dimension"_a = nb::none(),
          "singular_value_threshold"_a = nb::none(),
          "use_double_precision"_a = false,
          "Compute the inner product <psi_self|psi_other> = <0|U_self^dag "
          "U_other|0> between this circuit's state and another circuit's "
          "state, using ProjectOnZero.");

  // --- QASM Tools ---
  nb::class_<qasm::QasmToCirc<double>>(m, "QasmToCirc")
      .def(nb::init<>())
      .def("parse_and_translate", &qasm::QasmToCirc<double>::ParseAndTranslate);

  // --- Module Level Convenience Functions ---

  // 1. simple_execute (Overloaded)
  // Variant A: Circuit Object
  m.def("simple_execute", &execute_core, "circuit"_a,
        "simulator_type"_a = Simulators::SimulatorType::kQCSim,
        "simulation_type"_a = Simulators::SimulationType::kStatevector,
        "shots"_a = 1024, "max_bond_dimension"_a = 2,
        "singular_value_threshold"_a = 1e-8, "use_double_precision"_a = false);

  // Variant B: QASM String
  m.def(
      "simple_execute",
      [](const std::string &qasm, Simulators::SimulatorType st,
         Simulators::SimulationType set, int shots, std::optional<size_t> mb,
         std::optional<double> sv, bool use_dp) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit) {
          // IMPROVEMENT: Throw error instead of silent failure
          throw nb::value_error("Failed to parse QASM string.");
        }
        return execute_core(circuit, st, set, shots, mb, sv, use_dp);
      },
      "qasm_circuit"_a, "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "shots"_a = 1024, "max_bond_dimension"_a = 2,
      "singular_value_threshold"_a = 1e-8, "use_double_precision"_a = false);

  // 2. simple_estimate (Overloaded)
  // Variant A: Circuit Object
  m.def(
      "simple_estimate",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const nb::object &obs, Simulators::SimulatorType st,
         Simulators::SimulationType set, std::optional<size_t> mb,
         std::optional<double> sv, bool use_dp) {
        return estimate_core(circuit, ParseObservables(obs), st, set, mb, sv,
                             use_dp);
      },
      "circuit"_a, "observables"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = 2, "singular_value_threshold"_a = 1e-8,
      "use_double_precision"_a = false);

  // Variant B: QASM String
  m.def(
      "simple_estimate",
      [](const std::string &qasm, const nb::object &obs,
         Simulators::SimulatorType st, Simulators::SimulationType set,
         std::optional<size_t> mb, std::optional<double> sv, bool use_dp) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit) {
          throw nb::value_error("Failed to parse QASM string.");
        }
        return estimate_core(circuit, ParseObservables(obs), st, set, mb, sv,
                             use_dp);
      },
      "qasm_circuit"_a, "observables"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = 2, "singular_value_threshold"_a = 1e-8,
      "use_double_precision"_a = false);

  // --- QuEST Library Management ---
  m.def(
      "init_quest",
      []() { return Simulators::SimulatorsFactory::InitQuestLibrary(); },
      "Initialize the QuEST simulation library. Returns True on success.");

  m.def(
      "is_quest_available",
      []() { return Simulators::SimulatorsFactory::IsQuestLibraryAvailable(); },
      "Check whether the QuEST simulation library is loaded and available.");

  // --- GPU Library Management ---
  m.def(
      "init_gpu",
      []() { return Simulators::SimulatorsFactory::InitGpuLibrary(); },
      "Initialize the GPU simulation library. Returns True on success.");

  m.def(
      "is_gpu_available",
      []() { return Simulators::SimulatorsFactory::IsGpuLibraryAvailable(); },
      "Check whether the GPU simulation library is loaded and available.");

  // --- Probability / Amplitude Access ---
  m.def(
      "get_probabilities",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         Simulators::SimulatorType sim_type,
         Simulators::SimulationType sim_exec_type,
         std::optional<size_t> max_bond, std::optional<double> sv_threshold,
         bool use_double_precision) -> nb::list {
        const auto amplitudes = statevector_core(circuit, sim_type,
                                                 sim_exec_type, max_bond,
                                                 sv_threshold,
                                                 use_double_precision);
        nb::list probs;
        for (const auto &amp : amplitudes) probs.append(std::norm(amp));
        return probs;
      },
      "circuit"_a, "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = nb::none(),
      "singular_value_threshold"_a = nb::none(),
      "use_double_precision"_a = false,
      "Get the full probability distribution after executing a circuit.");

  m.def("get_statevector", &statevector_core, "circuit"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = nb::none(),
      "singular_value_threshold"_a = nb::none(),
      "use_double_precision"_a = false,
      "Get the full statevector (complex amplitudes) after executing a "
      "circuit.");

  m.def("mirror_fidelity", &mirror_fidelity_core, "circuit"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "shots"_a = 1024,
      "max_bond_dimension"_a = nb::none(),
      "singular_value_threshold"_a = nb::none(),
      "use_double_precision"_a = false,
      "full_amplitude"_a = false,
      "Compute mirror fidelity: run a circuit forward then its adjoint in "
      "reverse, returning P(|0...0>). Uses shot-based sampling by "
      "default. Set full_amplitude=True for exact statevector "
       "computation (small circuits only).");

  m.def("inner_product", &inner_product_core, "circuit_1"_a, "circuit_2"_a,
      "simulator_type"_a = Simulators::SimulatorType::kQCSim,
      "simulation_type"_a = Simulators::SimulationType::kStatevector,
      "max_bond_dimension"_a = nb::none(),
      "singular_value_threshold"_a = nb::none(),
      "use_double_precision"_a = false,
      "Compute the inner product <psi_1|psi_2> = <0|U1^dag U2|0> between "
      "two circuits' output states, using ProjectOnZero.");
}
