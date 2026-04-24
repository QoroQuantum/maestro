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
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "noise.h"

// Domain Headers
#include "Circuit/Circuit.h"
#include "Interface.h"
#include "Maestro.h"
#include "Simulators/Factory.h"
#include "Simulators/Simulator.h"
#include "Simulators/PathIntegralSimulator.h"
#include "qasm/QasmCirc.h"
#include "Network/SimpleDisconnectedNetwork.h"

namespace nb = nanobind;
using namespace nb::literals;

// ============================================================================
// Simulator Configuration (shared across all executor paths)
// ============================================================================

/// Bundles every knob that controls *how* the simulator runs a circuit.
/// Adding a new simulator parameter requires only:
///   1. Add the field here (with a default).
///   2. Wire it in ConfigureNetwork().
///   3. Expose it in the nanobind class binding below.
struct SimulatorConfig {
  Simulators::SimulatorType simulator_type = Simulators::SimulatorType::kQCSim;
  Simulators::SimulationType simulation_type =
      Simulators::SimulationType::kStatevector;
  std::optional<size_t> max_bond_dimension = std::nullopt;
  std::optional<double> singular_value_threshold = std::nullopt;
  bool use_double_precision = false;
  bool disable_optimized_swapping = false;
  int lookahead_depth = -1;
  bool mps_measure_no_collapse = true;

  SimulatorConfig() = default;

  SimulatorConfig(Simulators::SimulatorType st, Simulators::SimulationType set,
                  std::optional<size_t> mb, std::optional<double> sv, bool dp,
                  bool ds, int la, bool mnc)
      : simulator_type(st),
        simulation_type(set),
        max_bond_dimension(mb),
        singular_value_threshold(sv),
        use_double_precision(dp),
        disable_optimized_swapping(ds),
        lookahead_depth(la),
        mps_measure_no_collapse(mnc) {}
};

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
  ScopedSimulator(const ScopedSimulator&) = delete;
  ScopedSimulator& operator=(const ScopedSimulator&) = delete;
};

// Helper to configure the simulation network
std::shared_ptr<Network::INetwork<double>> ConfigureNetwork(
    unsigned long int handle, const SimulatorConfig& config) {
  // QuEST only supports statevector simulation
  if (config.simulator_type == Simulators::SimulatorType::kQuestSim &&
      config.simulation_type != Simulators::SimulationType::kStatevector) {
    throw std::invalid_argument(
        "QuestSim only supports Statevector simulation type.");
  }

  if (RemoveAllOptimizationSimulatorsAndAdd(handle, (int)config.simulator_type,
                                            (int)config.simulation_type) == 0) {
    return nullptr;
  }

  auto* maestro = static_cast<Maestro*>(GetMaestroObject());
  auto network = maestro->GetSimpleSimulator(handle);

  if (!network) return nullptr;

  if (config.max_bond_dimension) {
    auto val = std::to_string(*config.max_bond_dimension);
    network->Configure("matrix_product_state_max_bond_dimension", val.c_str());
  }
  if (config.singular_value_threshold) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10)
        << *config.singular_value_threshold;
    auto val = oss.str();
    network->Configure("matrix_product_state_truncation_threshold",
                       val.c_str());
  }
  if (config.use_double_precision) {
    network->Configure("use_double_precision", "1");
  }

  // Disable MPS swap optimization if requested
  if (config.disable_optimized_swapping) {
    network->SetInitialQubitsMapOptimization(false);
    network->SetMPSOptimizeSwaps(false);
  }

  // Set the lookahead depth for swap optimization
  network->SetLookaheadDepth(config.lookahead_depth);

  if (!config.mps_measure_no_collapse) {
    network->Configure("mps_sample_measure_algorithm", "mps_apply_measure");
  } else {
    network->Configure("mps_sample_measure_algorithm", "mps_probabilities");
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
std::vector<std::string> ParseObservables(const nb::object& observables) {
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
                      const SimulatorConfig& config, int shots) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
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
  for (const auto& pair : raw_results) {
    // Optimization: Pre-allocate string to avoid repeated reallocation
    const auto& bool_vec = pair.first;
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
                       const std::vector<std::string>& paulis,
                       const SimulatorConfig& config) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits = static_cast<int>(circuit->GetMaxQubitIndex()) + 1;
  for (const auto& p : paulis)
    num_qubits = std::max(num_qubits, (int)p.length());

  ScopedSimulator sim(std::max(1, num_qubits));
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
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
    const SimulatorConfig& config) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
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

OperationPtr adjoint_gate(const OperationPtr& op) {
  if (op->GetType() != Circuits::OperationType::kGate) return nullptr;

  auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
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
      return std::make_shared<Circuits::RxGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRyGateType:
      return std::make_shared<Circuits::RyGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRzGateType:
      return std::make_shared<Circuits::RzGate<>>(gate->GetQubit(), -params[0]);

    // ---- U gate: U†(θ,φ,λ,γ) = U(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kUGateType:
      return std::make_shared<Circuits::UGate<>>(
          gate->GetQubit(), -params[0], -params[2], -params[1], -params[3]);

    // ---- Controlled parametric: negate angle ----
    case Circuits::QuantumGateType::kCPGateType:
      return std::make_shared<Circuits::CPGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRxGateType:
      return std::make_shared<Circuits::CRxGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRyGateType:
      return std::make_shared<Circuits::CRyGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRzGateType:
      return std::make_shared<Circuits::CRzGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);

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
double mirror_fidelity_core(std::shared_ptr<Circuits::Circuit<double>> circuit,
                            const SimulatorConfig& config, int shots,
                            bool full_amplitude) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  // Build the mirror circuit: forward gates + adjoint gates in reverse
  auto mirror = std::make_shared<Circuits::Circuit<double>>();
  const auto& ops = circuit->GetOperations();

  // Forward pass: add only gate operations (skip measurements)
  for (const auto& op : ops) {
    if (op->GetType() == Circuits::OperationType::kGate) {
      mirror->AddOperation(op->Clone());
    }
  }

  // Reverse pass: iterate backward and add adjoint of each gate operation only
  // (skip measurements and other non-gate ops — they have no adjoint)
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    if ((*it)->GetType() != Circuits::OperationType::kGate) continue;
    auto adj = adjoint_gate(*it);
    if (adj) mirror->AddOperation(adj);
  }

  // Helper lambda for the shot-based path
  auto run_shot_based = [&]() -> double {
    // Need a fresh mirror circuit since measurements mutate it
    auto mirror_copy = std::make_shared<Circuits::Circuit<double>>();
    for (const auto& op : mirror->GetOperations()) {
      mirror_copy->AddOperation(op->Clone());
    }

    size_t n =
        std::max(1, static_cast<int>(mirror_copy->GetMaxQubitIndex()) + 1);
    std::vector<std::pair<Types::qubit_t, size_t>> pairs;
    pairs.reserve(n);
    for (size_t i = 0; i < n; ++i)
      pairs.emplace_back(static_cast<Types::qubit_t>(i), i);
    mirror_copy->AddOperation(
        std::make_shared<Circuits::MeasurementOperation<>>(pairs));

    // Build network directly (like execute_core) but with circuit optimization
    // disabled. The mirror circuit's paired gate/adjoint sequences must not be
    // cancelled by the optimizer: e.g. ry(-θ) + ry(θ) → ry(0), followed by
    // s + ry(0) + sdg, which the optimizer would incorrectly simplify further.
    int num_qubits =
        std::max(1, static_cast<int>(mirror_copy->GetMaxQubitIndex()) + 1);
    ScopedSimulator sim(num_qubits);
    if (sim.handle == 0)
      throw std::runtime_error("mirror_fidelity: failed to create simulator.");
    auto network = ConfigureNetwork(sim.handle, config);
    if (!network)
      throw std::runtime_error("mirror_fidelity: failed to configure network.");
    // Disable circuit optimization: the mirror's gate/adjoint pairs must not
    // be cancelled or merged by the optimizer.
    network->GetController()->SetOptimizeCircuit(false);
    // Disable MPS swap optimization: MPSDummySimulator used for swap-cost
    // estimation throws on multi-qubit measurement operations.
    network->SetInitialQubitsMapOptimization(false);
    network->SetMPSOptimizeSwaps(false);

    Network::INetwork<double>::ExecuteResults raw_results;
    {
      nb::gil_scoped_release release;
      raw_results =
          network->RepeatedExecuteOnHost(mirror_copy, 0, (size_t)shots);
    }

    // Convert results to counts dict and look up all-zeros bitstring
    std::string zeros(n, '0');
    size_t zero_count = 0;
    for (const auto& pair : raw_results) {
      const auto& bool_vec = pair.first;
      std::string bitstring(bool_vec.size(), '0');
      for (size_t i = 0; i < bool_vec.size(); ++i)
        if (bool_vec[i]) bitstring[i] = '1';
      if (bitstring == zeros) zero_count += pair.second;
    }
    return static_cast<double>(zero_count) / static_cast<double>(shots);
  };

  if (full_amplitude) {
    // Try exact statevector with circuit optimization disabled
    try {
      int num_qubits =
          std::max(1, static_cast<int>(mirror->GetMaxQubitIndex()) + 1);
      ScopedSimulator sim(num_qubits);
      if (sim.handle != 0) {
        auto network = ConfigureNetwork(sim.handle, config);
        if (network) {
          network->GetController()->SetOptimizeCircuit(false);
          network->SetInitialQubitsMapOptimization(false);
          network->SetMPSOptimizeSwaps(false);
          std::vector<std::complex<double>> amplitudes;
          {
            nb::gil_scoped_release release;
            amplitudes = network->ExecuteOnHostAmplitudes(mirror, 0);
          }
          if (!amplitudes.empty()) return std::norm(amplitudes[0]);
        }
      }
    } catch (...) {
      // Statevector not available for this backend — fall back to shots
    }
    // Issue a Python warning so the user knows we fell back
    PyErr_WarnEx(
        PyExc_RuntimeWarning,
        "full_amplitude mode not supported by this simulator/simulation "
        "type. Falling back to shot-based sampling.",
        1);
    return run_shot_based();
  } else {
    return run_shot_based();
  }
}

// Core Inner Product Logic
// Computes <psi_1|psi_2> = <0|U1† U2|0> via ProjectOnZero.
std::complex<double> inner_product_core(
    const std::shared_ptr<Circuits::Circuit<double>>& circuit_1,
    const std::shared_ptr<Circuits::Circuit<double>>& circuit_2,
    const SimulatorConfig& config) {
  if (!circuit_1) throw nb::value_error("circuit_1 is null.");
  if (!circuit_2) throw nb::value_error("circuit_2 is null.");

  // Build combined circuit for <0| U1† U2 |0>.
  // Circuit gates are applied left-to-right, so we place U2's gates first
  // (they act on |0> first), then U1†'s gates (applied last = leftmost in
  // the matrix product).
  auto combined = std::make_shared<Circuits::Circuit<double>>();
  const auto& ops1 = circuit_1->GetOperations();
  const auto& ops2 = circuit_2->GetOperations();

  // Forward pass of circuit_2: gate operations only
  for (const auto& op : ops2) {
    if (op->GetType() == Circuits::OperationType::kGate) {
      combined->AddOperation(op->Clone());
    }
  }

  // Adjoint of circuit_1: reverse order, each gate adjointed
  for (auto it = ops1.rbegin(); it != ops1.rend(); ++it) {
    auto adj = adjoint_gate(*it);
    if (adj) combined->AddOperation(adj);
  }

  int num_qubits =
      std::max(1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
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

  // --- Enums (must be registered before SimulatorConfig) ---
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
      .value("PathIntegral", Simulators::SimulationType::kPathIntegral)
      .export_values();

  // --- SimulatorConfig ---
  nb::class_<SimulatorConfig>(
      m, "SimulatorConfig",
      "Configuration for the quantum simulator backend. Create once and "
      "reuse across execute/estimate/statevector calls.")
      .def(nb::init<Simulators::SimulatorType, Simulators::SimulationType,
                    std::optional<size_t>, std::optional<double>, bool, bool,
                    int, bool>(),
           "simulator_type"_a = Simulators::SimulatorType::kQCSim,
           "simulation_type"_a = Simulators::SimulationType::kStatevector,
           "max_bond_dimension"_a = nb::none(),
           "singular_value_threshold"_a = nb::none(),
           "use_double_precision"_a = false,
           "disable_optimized_swapping"_a = false, "lookahead_depth"_a = -1,
           "mps_measure_no_collapse"_a = true)
      .def_rw("simulator_type", &SimulatorConfig::simulator_type)
      .def_rw("simulation_type", &SimulatorConfig::simulation_type)
      .def_rw("max_bond_dimension", &SimulatorConfig::max_bond_dimension)
      .def_rw("singular_value_threshold",
              &SimulatorConfig::singular_value_threshold)
      .def_rw("use_double_precision", &SimulatorConfig::use_double_precision)
      .def_rw("disable_optimized_swapping",
              &SimulatorConfig::disable_optimized_swapping)
      .def_rw("lookahead_depth", &SimulatorConfig::lookahead_depth)
      .def_rw("mps_measure_no_collapse",
              &SimulatorConfig::mps_measure_no_collapse)
      .def("__repr__", [](const SimulatorConfig& c) {
        std::ostringstream oss;
        oss << "SimulatorConfig("
            << "simulator_type=" << (int)c.simulator_type
            << ", simulation_type=" << (int)c.simulation_type
            << ", max_bond_dimension="
            << (c.max_bond_dimension ? std::to_string(*c.max_bond_dimension)
                                     : "None")
            << ", singular_value_threshold="
            << (c.singular_value_threshold
                    ? std::to_string(*c.singular_value_threshold)
                    : "None")
            << ", use_double_precision="
            << (c.use_double_precision ? "True" : "False")
            << ", disable_optimized_swapping="
            << (c.disable_optimized_swapping ? "True" : "False")
            << ", lookahead_depth=" << c.lookahead_depth
            << ", mps_measure_no_collapse="
            << (c.mps_measure_no_collapse ? "True" : "False") << ")";
        return oss.str();
      });

  // --- Maestro Class ---
  nb::class_<Maestro>(m, "Maestro")
      .def(nb::init<>())
      .def("create_simulator", &Maestro::CreateSimulator,
           "sim_type"_a = Simulators::SimulatorType::kQCSim,
           "sim_exec_type"_a = Simulators::SimulationType::kMatrixProductState)
      .def(
          "get_simulator",
          [](Maestro& self, unsigned long int h) {
            return static_cast<Simulators::ISimulator*>(self.GetSimulator(h));
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

      // ---- Bound Methods for Noisy Execution ----
      .def(
          "noisy_execute",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const noise::NoiseModel &noise_model,
             const SimulatorConfig &config,
             int shots, int noise_realizations,
             std::optional<unsigned int> seed) {
            if (!self) throw nb::value_error("Circuit is null.");
            std::mt19937 rng(seed.value_or(std::random_device{}()));
            const int batches =
                std::min(shots, std::max(1, noise_realizations));
            const int base_batch = shots / batches;
            int leftover = shots % batches;

            std::unordered_map<std::string, size_t> combined;

            auto start = std::chrono::high_resolution_clock::now();
            for (int b = 0; b < batches; ++b) {
              int batch_shots = base_batch + (b < leftover ? 1 : 0);
              if (batch_shots <= 0) continue;

              auto noisy = noise::inject_noise(self, noise_model, rng);
              nb::dict r = execute_core(noisy, config, batch_shots);
              nb::dict counts = nb::cast<nb::dict>(r["counts"]);
              for (auto item : counts)
                combined[nb::cast<std::string>(nb::str(item.first))] +=
                    nb::cast<size_t>(item.second);
            }
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict py_counts;
            for (const auto &[k, v] : combined) py_counts[k.c_str()] = v;

            nb::dict out;
            out["counts"] = py_counts;
            out["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            out["simulator"] = (int)config.simulator_type;
            out["method"] = (int)config.simulation_type;
            out["noise_realizations"] = batches;
            return out;
          },
          "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "shots"_a = 1024,
          "noise_realizations"_a = 64, "seed"_a = nb::none(),
          "Execute this circuit with Monte Carlo Pauli noise.\n\n"
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
            return out;
          },
          "observables"_a, "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "Analytical noisy estimation (zero overhead). "
          "Applies per-qubit Pauli damping to ideal expectation values.\n\n"
          "Example: qc.noisy_estimate(['ZZ', 'XX'], nm)")
      .def(
          "noisy_estimate_montecarlo",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const nb::object &observables,
             const noise::NoiseModel &noise_model, int noise_realizations,
             const SimulatorConfig &config,
             std::optional<unsigned int> seed) {
            if (!self) throw nb::value_error("Circuit is null.");
            auto paulis = ParseObservables(observables);

            std::mt19937 rng(seed.value_or(std::random_device{}()));
            const size_t n_obs = paulis.size();
            std::vector<double> sum_vals(n_obs, 0.0);

            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < noise_realizations; ++r) {
              auto noisy = noise::inject_noise(self, noise_model, rng);
              nb::dict result = estimate_core(noisy, paulis, config);
              nb::list ev = nb::cast<nb::list>(result["expectation_values"]);
              for (size_t i = 0; i < n_obs; ++i)
                sum_vals[i] += nb::cast<double>(ev[i]);
            }
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict ideal_result = estimate_core(self, paulis, config);
            nb::list noisy_vals, ideal_vals;
            nb::list ideal_ev =
                nb::cast<nb::list>(ideal_result["expectation_values"]);
            for (size_t i = 0; i < n_obs; ++i) {
              noisy_vals.append(sum_vals[i] / noise_realizations);
              ideal_vals.append(nb::cast<double>(ideal_ev[i]));
            }

            nb::dict out;
            out["expectation_values"] = noisy_vals;
            out["ideal_expectation_values"] = ideal_vals;
            out["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            out["simulator"] = ideal_result["simulator"];
            out["method"] = ideal_result["method"];
            out["noise_realizations"] = noise_realizations;
            return out;
          },
          "observables"_a, "noise_model"_a,
          "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{},
          "seed"_a = nb::none(),
          "Gate-by-gate Monte Carlo noisy estimation.\n\n"
          "Example: qc.noisy_estimate_montecarlo(['ZZ'], nm, "
          "noise_realizations=200)")
      .def(
          "coherent_execute",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const noise::NoiseModel &noise_model,
             const SimulatorConfig &config,
             int shots, int noise_realizations,
             std::optional<unsigned int> seed) {
            if (!self) throw nb::value_error("Circuit is null.");
            if (!noise_model.has_coherent())
              throw nb::value_error(
                  "NoiseModel has no coherent noise set. Use "
                  "set_coherent_depolarizing(), set_coherent_rotation(), "
                  "etc.");

            std::mt19937 rng(seed.value_or(std::random_device{}()));
            const int batches =
                std::min(shots, std::max(1, noise_realizations));
            const int base_batch = shots / batches;
            int leftover = shots % batches;

            std::unordered_map<std::string, size_t> combined;

            auto start = std::chrono::high_resolution_clock::now();
            for (int b = 0; b < batches; ++b) {
              int batch_shots = base_batch + (b < leftover ? 1 : 0);
              if (batch_shots <= 0) continue;

              auto noisy =
                  noise::inject_coherent_noise(self, noise_model, rng);
              nb::dict r = execute_core(noisy, config, batch_shots);
              nb::dict counts = nb::cast<nb::dict>(r["counts"]);
              for (auto item : counts)
                combined[nb::cast<std::string>(nb::str(item.first))] +=
                    nb::cast<size_t>(item.second);
            }
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict py_counts;
            for (const auto &[k, v] : combined) py_counts[k.c_str()] = v;

            nb::dict out;
            out["counts"] = py_counts;
            out["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            out["simulator"] = (int)config.simulator_type;
            out["method"] = (int)config.simulation_type;
            out["noise_realizations"] = batches;
            out["noise_type"] = "coherent";
            return out;
          },
          "noise_model"_a,
          "config"_a = SimulatorConfig{},
          "shots"_a = 1024,
          "noise_realizations"_a = 64, "seed"_a = nb::none(),
          "Execute this circuit with coherent noise (rotation errors).\n\n"
          "Example: qc.coherent_execute(nm, shots=1000)")
      .def(
          "coherent_estimate",
          [](std::shared_ptr<Circuits::Circuit<double>> self,
             const nb::object &observables,
             const noise::NoiseModel &noise_model, int noise_realizations,
             const SimulatorConfig &config,
             std::optional<unsigned int> seed) {
            if (!self) throw nb::value_error("Circuit is null.");
            if (!noise_model.has_coherent())
              throw nb::value_error(
                  "NoiseModel has no coherent noise set. Use "
                  "set_coherent_depolarizing(), set_coherent_rotation(), "
                  "etc.");

            auto paulis = ParseObservables(observables);
            std::mt19937 rng(seed.value_or(std::random_device{}()));
            const size_t n_obs = paulis.size();
            std::vector<double> sum_vals(n_obs, 0.0);

            auto start = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < noise_realizations; ++r) {
              auto noisy =
                  noise::inject_coherent_noise(self, noise_model, rng);
              nb::dict result = estimate_core(noisy, paulis, config);
              nb::list ev = nb::cast<nb::list>(result["expectation_values"]);
              for (size_t i = 0; i < n_obs; ++i)
                sum_vals[i] += nb::cast<double>(ev[i]);
            }
            auto end = std::chrono::high_resolution_clock::now();

            nb::dict ideal_result = estimate_core(self, paulis, config);
            nb::list noisy_vals, ideal_vals;
            nb::list ideal_ev =
                nb::cast<nb::list>(ideal_result["expectation_values"]);
            for (size_t i = 0; i < n_obs; ++i) {
              noisy_vals.append(sum_vals[i] / noise_realizations);
              ideal_vals.append(nb::cast<double>(ideal_ev[i]));
            }

            nb::dict out;
            out["expectation_values"] = noisy_vals;
            out["ideal_expectation_values"] = ideal_vals;
            out["time_taken"] =
                std::chrono::duration<double>(end - start).count();
            out["simulator"] = ideal_result["simulator"];
            out["method"] = ideal_result["method"];
            out["noise_realizations"] = noise_realizations;
            out["noise_type"] = "coherent";
            return out;
          },
          "observables"_a, "noise_model"_a,
          "noise_realizations"_a = 100,
          "config"_a = SimulatorConfig{},
          "seed"_a = nb::none(),
          "Estimate expectation values with coherent noise.\n\n"
          "Example: qc.coherent_estimate(['ZZ', 'XX'], nm, "
          "noise_realizations=200)");

  // --- QASM Tools ---
  nb::class_<qasm::QasmToCirc<double>>(m, "QasmToCirc")
      .def(nb::init<>())
      .def("parse_and_translate",
           [](qasm::QasmToCirc<double>& self, const std::string& qasm_str) {
             auto circuit = self.ParseAndTranslate(qasm_str);
             if (self.Failed() || !circuit) {
               throw nb::value_error(
                   ("Failed to parse QASM string: " + self.GetErrorMessage())
                       .c_str());
             }
             return circuit;
           })
      .def("failed", &qasm::QasmToCirc<double>::Failed)
      .def("get_error_message", &qasm::QasmToCirc<double>::GetErrorMessage);

  // --- Module Level Convenience Functions ---

  // 1. simple_execute (Overloaded)
  // Variant A: Circuit Object
  m.def("simple_execute", &execute_core, "circuit"_a,
        "config"_a = SimulatorConfig{}, "shots"_a = 1024);

  // Variant B: QASM String
  m.def(
      "simple_execute",
      [](const std::string& qasm, const SimulatorConfig& config, int shots) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit) {
          // IMPROVEMENT: Throw error instead of silent failure
          throw nb::value_error("Failed to parse QASM string.");
        }
        return execute_core(circuit, config, shots);
      },
      "qasm_circuit"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024);

  // 2. simple_estimate (Overloaded)
  // Variant A: Circuit Object
  m.def(
      "simple_estimate",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const nb::object& obs, const SimulatorConfig& config) {
        return estimate_core(circuit, ParseObservables(obs), config);
      },
      "circuit"_a, "observables"_a, "config"_a = SimulatorConfig{});

  // Variant B: QASM String
  m.def(
      "simple_estimate",
      [](const std::string& qasm, const nb::object& obs,
         const SimulatorConfig& config) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit) {
          throw nb::value_error("Failed to parse QASM string.");
        }
        return estimate_core(circuit, ParseObservables(obs), config);
      },
      "qasm_circuit"_a, "observables"_a, "config"_a = SimulatorConfig{});

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
         const SimulatorConfig& config) -> nb::list {
        const auto amplitudes = statevector_core(circuit, config);
        nb::list probs;
        for (const auto& amp : amplitudes) probs.append(std::norm(amp));
        return probs;
      },
      "circuit"_a, "config"_a = SimulatorConfig{},
      "Get the full probability distribution after executing a circuit.");

  m.def(
      "get_statevector",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const SimulatorConfig& config) {
        return statevector_core(circuit, config);
      },
      "circuit"_a, "config"_a = SimulatorConfig{},
      "Get the full statevector (complex amplitudes) after executing a "
      "circuit.");

  m.def(
      "mirror_fidelity",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const SimulatorConfig& config, int shots, bool full_amplitude) {
        return mirror_fidelity_core(circuit, config, shots, full_amplitude);
      },
      "circuit"_a, "config"_a = SimulatorConfig{}, "shots"_a = 1024,
      "full_amplitude"_a = false,
      "Compute mirror fidelity: run a circuit forward then its adjoint in "
      "reverse, returning P(|0...0>). Uses shot-based sampling by "
      "default. Set full_amplitude=True for exact statevector "
      "computation (small circuits only).");

  m.def(
      "inner_product",
      [](const std::shared_ptr<Circuits::Circuit<double>>& circuit_1,
         const std::shared_ptr<Circuits::Circuit<double>>& circuit_2,
         const SimulatorConfig& config) {
        return inner_product_core(circuit_1, circuit_2, config);
      },
      "circuit_1"_a, "circuit_2"_a, "config"_a = SimulatorConfig{},
      "Compute the inner product <psi_1|psi_2> = <0|U1^dag U2|0> between "
      "two circuits' output states, using ProjectOnZero.");

  // =========================================================================
  // Path Integral: Single-State Probability
  // =========================================================================

  m.def(
      "state_probability",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const std::string& target_state) -> nb::dict {
        if (!circuit) throw nb::value_error("Circuit is null.");
        if (target_state.empty())
          throw nb::value_error("target_state must be a non-empty bitstring.");

        // Convert bitstring to vector<bool>
        std::vector<bool> end_state(target_state.size());
        for (size_t i = 0; i < target_state.size(); ++i) {
          if (target_state[i] == '1')
            end_state[i] = true;
          else if (target_state[i] == '0')
            end_state[i] = false;
          else
            throw nb::value_error(
                "target_state must contain only '0' and '1' characters.");
        }

        Simulators::PathIntegralSimulator sim;
        sim.SetStartZeroState(target_state.size());

        auto start = std::chrono::high_resolution_clock::now();
        bool ok;
        {
          nb::gil_scoped_release release;
          ok = sim.SetCircuit(circuit);
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
      "circuit"_a, "target_state"_a,
      "Compute the probability of a specific output state using the Pauli "
      "path integral simulator.\n\n"
      "This is the path integral's key advantage: it computes a single "
      "amplitude <target_state|U|0...0> without building the full "
      "statevector, making it efficient for large circuits with few "
      "branching gates.\n\n"
      "Args:\n"
      "    circuit: A QuantumCircuit (no measurements needed).\n"
      "    target_state: A bitstring like '10001001' (qubit 0 is leftmost).\n\n"
      "Returns:\n"
      "    dict with 'probability', 'amplitude', 'target_state', "
      "'time_taken'.");

  // QASM variant
  m.def(
      "state_probability",
      [](const std::string& qasm, const std::string& target_state) -> nb::dict {
        if (target_state.empty())
          throw nb::value_error("target_state must be a non-empty bitstring.");

        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit)
          throw nb::value_error("Failed to parse QASM string.");

        std::vector<bool> end_state(target_state.size());
        for (size_t i = 0; i < target_state.size(); ++i) {
          if (target_state[i] == '1')
            end_state[i] = true;
          else if (target_state[i] == '0')
            end_state[i] = false;
          else
            throw nb::value_error(
                "target_state must contain only '0' and '1' characters.");
        }

        Simulators::PathIntegralSimulator sim;
        sim.SetStartZeroState(target_state.size());

        auto start = std::chrono::high_resolution_clock::now();
        bool ok;
        {
          nb::gil_scoped_release release;
          ok = sim.SetCircuit(circuit);
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
      "qasm_circuit"_a, "target_state"_a,
      "Compute the probability of a specific output state from a QASM "
      "circuit using the Pauli path integral simulator.");

  // =========================================================================
  // Noise Modeling
  // =========================================================================

  nb::class_<noise::NoiseModel>(m, "NoiseModel")
      .def(nb::init<>(), "Create an empty noise model.")
      .def("set_qubit_noise", &noise::NoiseModel::set_qubit_noise, "qubit"_a,
           "px"_a, "py"_a, "pz"_a,
           "Add Pauli channel: Λ(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + "
           "pz·ZρZ")
      .def("set_depolarizing", &noise::NoiseModel::set_depolarizing, "qubit"_a,
           "p"_a, "Add symmetric depolarizing noise (px=py=pz=p/3).")
      .def("set_dephasing", &noise::NoiseModel::set_dephasing, "qubit"_a, "p"_a,
           "Add pure dephasing (Z) noise.")
      .def("set_bit_flip", &noise::NoiseModel::set_bit_flip, "qubit"_a, "p"_a,
           "Add bit-flip (X) noise.")
      .def("set_all_depolarizing", &noise::NoiseModel::set_all_depolarizing,
           "num_qubits"_a, "p"_a,
           "Add uniform depolarizing noise to all qubits [0, num_qubits).")
      .def("set_all_dephasing", &noise::NoiseModel::set_all_dephasing,
           "num_qubits"_a, "p"_a,
           "Add uniform dephasing noise to all qubits [0, num_qubits).")
      .def("compute_damping", &noise::NoiseModel::compute_damping,
           "pauli_string"_a,
           "Compute the noise damping factor for a Pauli string observable.")
      // ── Coherent noise setters ──
      .def("set_coherent_rotation", &noise::NoiseModel::set_coherent_rotation,
           "qubit"_a, "rx"_a, "ry"_a, "rz"_a,
           "Set per-qubit coherent noise as rotation angles (radians). "
           "After every gate on this qubit, Rx(±rx), Ry(±ry), Rz(±rz) "
           "rotations are applied with random ± signs.")
      .def("set_coherent_depolarizing",
           &noise::NoiseModel::set_coherent_depolarizing, "qubit"_a, "p"_a,
           "Set coherent noise from a depolarizing probability. "
           "Converts p to Rz angle ε = 2·arcsin(√p), matching the "
           "infidelity of DEPOLARIZE1(p).")
      .def("set_coherent_dephasing", &noise::NoiseModel::set_coherent_dephasing,
           "qubit"_a, "p"_a,
           "Set coherent dephasing noise: Rz rotation from probability p.")
      .def("set_coherent_bit_flip", &noise::NoiseModel::set_coherent_bit_flip,
           "qubit"_a, "p"_a,
           "Set coherent bit-flip noise: Rx rotation from probability p.")
      .def("set_all_coherent_depolarizing",
           &noise::NoiseModel::set_all_coherent_depolarizing, "num_qubits"_a,
           "p"_a,
           "Set uniform coherent depolarizing noise on qubits [0, "
           "num_qubits).")
      .def("set_all_coherent_dephasing",
           &noise::NoiseModel::set_all_coherent_dephasing, "num_qubits"_a,
           "p"_a, "Set uniform coherent dephasing on qubits [0, num_qubits).")
      .def("set_coherent_strength", &noise::NoiseModel::set_coherent_strength,
           "num_qubits"_a, "p"_a,
           "Convenience: set uniform coherent noise strength on all qubits. "
           "Equivalent to set_all_coherent_depolarizing(n, p).")
      .def("has_coherent", &noise::NoiseModel::has_coherent,
           "Return True if any coherent noise parameters have been set.");

  // --- Noisy Estimation (analytical — zero simulation overhead) ---
  m.def(
      "noisy_estimate",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const nb::object& observables, const noise::NoiseModel& noise_model,
         const SimulatorConfig& config) {
        auto paulis = ParseObservables(observables);
        nb::dict result = estimate_core(circuit, paulis, config);

        // Apply analytical Pauli noise damping
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
        return out;
      },
      "circuit"_a, "observables"_a, "noise_model"_a,
      "config"_a = SimulatorConfig{},
      "Compute expectation values with analytical Pauli noise damping. "
      "Runs noiseless simulation then applies exact noise attenuation — "
      "zero simulation overhead compared to noiseless.");

  // --- QASM variant ---
  m.def(
      "noisy_estimate",
      [](const std::string& qasm, const nb::object& observables,
         const noise::NoiseModel& noise_model, const SimulatorConfig& config) {
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm);
        if (parser.Failed() || !circuit)
          throw nb::value_error("Failed to parse QASM string.");

        auto paulis = ParseObservables(observables);
        nb::dict result = estimate_core(circuit, paulis, config);

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
        return out;
      },
      "qasm_circuit"_a, "observables"_a, "noise_model"_a,
      "config"_a = SimulatorConfig{},
      "Compute expectation values from a QASM circuit with analytical Pauli "
      "noise. Zero simulation overhead.");

  // --- Gate-by-gate Monte Carlo Noisy Estimation ---
  m.def(
      "noisy_estimate_montecarlo",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const nb::object& observables, const noise::NoiseModel& noise_model,
         int noise_realizations, const SimulatorConfig& config,
         std::optional<unsigned int> seed) {
        if (!circuit) throw nb::value_error("Circuit is null.");
        auto paulis = ParseObservables(observables);

        std::mt19937 rng(seed.value_or(std::random_device{}()));
        const size_t n_obs = paulis.size();

        // Accumulate expectation values across realizations
        std::vector<double> sum_vals(n_obs, 0.0);

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < noise_realizations; ++r) {
          auto noisy = noise::inject_noise(circuit, noise_model, rng);
          nb::dict result = estimate_core(noisy, paulis, config);
          nb::list ev = nb::cast<nb::list>(result["expectation_values"]);
          for (size_t i = 0; i < n_obs; ++i)
            sum_vals[i] += nb::cast<double>(ev[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        // Also run noiseless for reference
        nb::dict ideal_result = estimate_core(circuit, paulis, config);

        nb::list noisy_vals, ideal_vals;
        nb::list ideal_ev =
            nb::cast<nb::list>(ideal_result["expectation_values"]);
        for (size_t i = 0; i < n_obs; ++i) {
          noisy_vals.append(sum_vals[i] / noise_realizations);
          ideal_vals.append(nb::cast<double>(ideal_ev[i]));
        }

        nb::dict out;
        out["expectation_values"] = noisy_vals;
        out["ideal_expectation_values"] = ideal_vals;
        out["time_taken"] = std::chrono::duration<double>(end - start).count();
        out["simulator"] = ideal_result["simulator"];
        out["method"] = ideal_result["method"];
        out["noise_realizations"] = noise_realizations;
        return out;
      },
      "circuit"_a, "observables"_a, "noise_model"_a,
      "noise_realizations"_a = 100, "config"_a = SimulatorConfig{},
      "seed"_a = nb::none(),
      "Gate-by-gate Monte Carlo noisy estimation. Injects random Pauli "
      "errors after every gate and averages expectation values over "
      "noise_realizations independent samples. More accurate than "
      "analytical noisy_estimate for deep circuits.");

  m.def(
      "noisy_execute",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const noise::NoiseModel& noise_model, const SimulatorConfig& config,
         int shots, int noise_realizations, std::optional<unsigned int> seed) {
        if (!circuit) throw nb::value_error("Circuit is null.");

        std::mt19937 rng(seed.value_or(std::random_device{}()));
        const int batches = std::min(shots, std::max(1, noise_realizations));
        const int base_batch = shots / batches;
        int leftover = shots % batches;

        std::unordered_map<std::string, size_t> combined;

        auto start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < batches; ++b) {
          int batch_shots = base_batch + (b < leftover ? 1 : 0);
          if (batch_shots <= 0) continue;

          auto noisy = noise::inject_noise(circuit, noise_model, rng);
          nb::dict r = execute_core(noisy, config, batch_shots);
          nb::dict counts = nb::cast<nb::dict>(r["counts"]);
          for (auto item : counts)
            combined[nb::cast<std::string>(nb::str(item.first))] +=
                nb::cast<size_t>(item.second);
        }
        auto end = std::chrono::high_resolution_clock::now();

        nb::dict py_counts;
        for (const auto& [k, v] : combined) py_counts[k.c_str()] = v;

        nb::dict out;
        out["counts"] = py_counts;
        out["time_taken"] = std::chrono::duration<double>(end - start).count();
        out["simulator"] = (int)config.simulator_type;
        out["method"] = (int)config.simulation_type;
        out["noise_realizations"] = batches;
        return out;
      },
      "circuit"_a, "noise_model"_a, "config"_a = SimulatorConfig{},
      "shots"_a = 1024, "noise_realizations"_a = 64, "seed"_a = nb::none(),
      "Execute a circuit with Monte Carlo Pauli noise. "
      "Each of 'noise_realizations' batches uses a different random noise "
      "pattern, with shots distributed evenly across batches.");

  // =========================================================================
  // Coherent Noise: Execute
  // =========================================================================

  m.def(
      "coherent_execute",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const noise::NoiseModel& noise_model, const SimulatorConfig& config,
         int shots, int noise_realizations, std::optional<unsigned int> seed) {
        if (!circuit) throw nb::value_error("Circuit is null.");
        if (!noise_model.has_coherent())
          throw nb::value_error(
              "NoiseModel has no coherent noise set. Use "
              "set_coherent_depolarizing(), set_coherent_rotation(), etc.");

        std::mt19937 rng(seed.value_or(std::random_device{}()));
        const int batches = std::min(shots, std::max(1, noise_realizations));
        const int base_batch = shots / batches;
        int leftover = shots % batches;

        std::unordered_map<std::string, size_t> combined;

        auto start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < batches; ++b) {
          int batch_shots = base_batch + (b < leftover ? 1 : 0);
          if (batch_shots <= 0) continue;

          auto noisy = noise::inject_coherent_noise(circuit, noise_model, rng);
          nb::dict r = execute_core(noisy, config, batch_shots);
          nb::dict counts = nb::cast<nb::dict>(r["counts"]);
          for (auto item : counts)
            combined[nb::cast<std::string>(nb::str(item.first))] +=
                nb::cast<size_t>(item.second);
        }
        auto end = std::chrono::high_resolution_clock::now();

        nb::dict py_counts;
        for (const auto& [k, v] : combined) py_counts[k.c_str()] = v;

        nb::dict out;
        out["counts"] = py_counts;
        out["time_taken"] = std::chrono::duration<double>(end - start).count();
        out["simulator"] = (int)config.simulator_type;
        out["method"] = (int)config.simulation_type;
        out["noise_realizations"] = batches;
        out["noise_type"] = "coherent";
        return out;
      },
      "circuit"_a, "noise_model"_a, "config"_a = SimulatorConfig{},
      "shots"_a = 1024, "noise_realizations"_a = 64, "seed"_a = nb::none(),
      "Execute a circuit with coherent noise (systematic rotation errors). "
      "After every gate, Rx/Ry/Rz rotations are injected with random ± "
      "signs. Each of 'noise_realizations' batches uses a different sign "
      "pattern. Requires MPS/Statevector simulation (not Stabilizer).\n\n"
      "Example:\n"
      "    nm = maestro.NoiseModel()\n"
      "    nm.set_all_coherent_depolarizing(n_qubits, 0.001)\n"
      "    result = maestro.coherent_execute(qc, nm, shots=1000)\n");

  // =========================================================================
  // Coherent Noise: Estimate (Monte Carlo averaged)
  // =========================================================================

  m.def(
      "coherent_estimate",
      [](std::shared_ptr<Circuits::Circuit<double>> circuit,
         const nb::object& observables, const noise::NoiseModel& noise_model,
         int noise_realizations, const SimulatorConfig& config,
         std::optional<unsigned int> seed) {
        if (!circuit) throw nb::value_error("Circuit is null.");
        if (!noise_model.has_coherent())
          throw nb::value_error(
              "NoiseModel has no coherent noise set. Use "
              "set_coherent_depolarizing(), set_coherent_rotation(), etc.");

        auto paulis = ParseObservables(observables);
        std::mt19937 rng(seed.value_or(std::random_device{}()));
        const size_t n_obs = paulis.size();

        std::vector<double> sum_vals(n_obs, 0.0);

        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < noise_realizations; ++r) {
          auto noisy = noise::inject_coherent_noise(circuit, noise_model, rng);
          nb::dict result = estimate_core(noisy, paulis, config);
          nb::list ev = nb::cast<nb::list>(result["expectation_values"]);
          for (size_t i = 0; i < n_obs; ++i)
            sum_vals[i] += nb::cast<double>(ev[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();

        // Also run noiseless for reference
        nb::dict ideal_result = estimate_core(circuit, paulis, config);

        nb::list noisy_vals, ideal_vals;
        nb::list ideal_ev =
            nb::cast<nb::list>(ideal_result["expectation_values"]);
        for (size_t i = 0; i < n_obs; ++i) {
          noisy_vals.append(sum_vals[i] / noise_realizations);
          ideal_vals.append(nb::cast<double>(ideal_ev[i]));
        }

        nb::dict out;
        out["expectation_values"] = noisy_vals;
        out["ideal_expectation_values"] = ideal_vals;
        out["time_taken"] = std::chrono::duration<double>(end - start).count();
        out["simulator"] = ideal_result["simulator"];
        out["method"] = ideal_result["method"];
        out["noise_realizations"] = noise_realizations;
        out["noise_type"] = "coherent";
        return out;
      },
      "circuit"_a, "observables"_a, "noise_model"_a,
      "noise_realizations"_a = 100, "config"_a = SimulatorConfig{},
      "seed"_a = nb::none(),
      "Estimate expectation values with coherent noise (rotation errors). "
      "Injects systematic Rx/Ry/Rz rotations after every gate and averages "
      "expectation values over noise_realizations independent sign samples. "
      "Unlike Pauli noise, coherent noise preserves phase coherence and "
      "does not commute with the circuit — it can model systematic "
      "calibration errors.\n\n"
      "Example:\n"
      "    nm = maestro.NoiseModel()\n"
      "    nm.set_coherent_strength(n_qubits, 0.001)\n"
      "    result = maestro.coherent_estimate(qc, ['ZZ', 'XX'], nm)\n");
}
