// Exercise the production distributed-host mapping on CPU, without a plugin.
// Gate targets and full complex states are checked independently of measurement
// remapping: applying the same wrong permutation to both must not pass.
#include "../../Network/SimpleDisconnectedNetwork.h"
#include "../../Circuit/Factory.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>

using namespace Simulators;
using Factory = SimulatorsFactory;
using CF = Circuits::CircuitFactory<>;
using Circuit = Circuits::Circuit<>;
using Gate = Circuits::QuantumGateType;
using Type = SimulatorType;
using Method = SimulationType;

void Require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
template<class F> void Reject(F action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("Expected rejection");
}

class TestNetwork : public Network::SimpleDisconnectedNetwork<> {
 public:
  using Network::SimpleDisconnectedNetwork<>::SimpleDisconnectedNetwork;
  using Network::SimpleDisconnectedNetwork<>::ExecutionConfiguration;
  using Network::SimpleDisconnectedNetwork<>::CaptureSimulatorConfiguration;
  using Network::SimpleDisconnectedNetwork<>::configuration;
  using Network::SimpleDisconnectedNetwork<>::resolvedDistributedDevices;
  using Network::SimpleDisconnectedNetwork<>::simulator;
  struct Mapping {
    std::shared_ptr<Circuit> circuit;
    size_t qubits = 0, bits = 0;
    std::unordered_map<Types::qubit_t, Types::qubit_t> reverseBits;
  };
  Mapping Map(const std::shared_ptr<Circuit>& circuit, size_t host) {
    Mapping result;
    result.reverseBits = MapCircuitOnHost(circuit, host, result.qubits, result.bits, true);
    result.circuit = distCirc;
    return result;
  }
  // The distributed factory is deliberately lazy; this selects the real mapping
  // branch without requiring CUDA. Execution jobs below use a CPU statevector.
  void CreateSimulator(Type type = Type::kDistGpuSim,
                       Method method = Method::kStatevector, size_t = 0) override {
    simulator = Factory::CreateSimulator(type, method);
  }
  std::shared_ptr<ISimulator> ChooseBestSimulator(
      std::shared_ptr<Circuit>&, size_t&, size_t, size_t, size_t,
      Type& type, Method& method, std::vector<bool>&, bool = false,
      bool = false) override {
    type = Type::kQCSim;
    method = Method::kStatevector;
    return nullptr;
  }
};

std::shared_ptr<TestNetwork> NetworkFor(size_t start, size_t width,
                                      const char* mode = "auto") {
  auto net = std::make_shared<TestNetwork>(
      std::vector<Types::qubit_t>{start, width}, std::vector<size_t>{2, 20});
  net->CreateSimulator();
  net->Configure("distributed_host_qubit_indexing", mode);
  net->SetOptimizeSimulator(false);
  return net;
}
std::shared_ptr<ISimulator> Run(const std::shared_ptr<Circuit>& circuit, size_t n,
                              size_t bits = 32) {
  auto sim = Factory::CreateSimulator(Type::kQCSim, Method::kStatevector);
  sim->AllocateQubits(n);
  sim->Initialize();
  Circuits::OperationState state;
  state.AllocateBits(bits);
  circuit->Execute(sim, state);
  return sim;
}
void Compare(const std::vector<std::complex<double>>& actual, ISimulator& expected) {
  Require(actual.size() == (size_t{1} << expected.GetNumberOfQubits()), "statevector width");
  for (size_t i = 0; i < actual.size(); ++i)
    Require(std::abs(actual[i] - expected.Amplitude(i)) < 1e-10,
            "complex amplitude at basis " + std::to_string(i));
}

void ExhaustiveMapping() {
  // Every subset of every small host register, in both encounter orders,
  // across overlapping, disjoint and zero-offset ranges and both plugin types.
  size_t cases = 0;
  for (auto type : {Type::kDistGpuSim, Type::kDistMpiGpuSim})
    for (size_t width = 1; width <= 5; ++width)
      for (size_t start = 0; start <= 6; ++start)
        for (const char* mode : {"local", "global", "auto"})
          for (bool globalInput : {false, true})
            for (size_t mask = 0; mask < (size_t{1} << width); ++mask)
              for (bool reverse : {false, true}) {
                auto net = NetworkFor(start, width, mode);
                net->CreateSimulator(type);
                auto circuit = CF::CreateCircuit();
                std::vector<size_t> targets;
                for (size_t q = 0; q < width; ++q)
                  if ((mask >> q) & 1) targets.push_back(q + (globalInput ? start : 0));
                if (reverse) std::reverse(targets.begin(), targets.end());
                for (auto q : targets) circuit->AddOperation(CF::CreateGate(Gate::kXGateType, q));
                bool fitsLocal = true, fitsGlobal = true;
                for (auto q : targets) {
                  fitsLocal &= q < width;
                  fitsGlobal &= q >= start && q < start + width;
                }
                bool global = std::string(mode) == "global" ||
                    (std::string(mode) == "auto" && fitsGlobal);
                if (!(global ? fitsGlobal : fitsLocal)) {
                  Reject([&] { net->Map(circuit, 1); });
                  continue;
                }
                const auto mapped = net->Map(circuit, 1);
                Require(mapped.qubits == width, "idle host wires were lost");
                Require(mapped.circuit->GetOperations().size() == targets.size(), "operation count");
                size_t expectedBasis = 0;
                for (size_t i = 0; i < targets.size(); ++i) {
                  const size_t q = targets[i] - (global ? start : 0);
                  Require(mapped.circuit->GetOperations()[i]->AffectedQubits() ==
                              Types::qubits_vector{q}, "wrong gate target");
                  Require(circuit->GetOperations()[i]->AffectedQubits() ==
                              Types::qubits_vector{targets[i]}, "input circuit mutated");
                  expectedBasis |= size_t{1} << q;
                }
                auto state = Run(mapped.circuit, width);
                Require(std::abs(state->Amplitude(expectedBasis) - 1.) < 1e-12, "basis state");
                ++cases;
              }
  std::cout << "Checked " << cases << " exhaustive mapping cases\n";
}

void InvalidAndClassicalMapping() {
  auto net = NetworkFor(3, 5);
  for (const auto& targets : {Types::qubits_vector{0, 7}, {2, 8},
                              {8}, {std::numeric_limits<Types::qubit_t>::max()}}) {
    auto circuit = CF::CreateCircuit();
    for (auto q : targets) circuit->AddOperation(CF::CreateGate(Gate::kXGateType, q));
    Reject([&] { net->Map(circuit, 1); });
  }
  Reject([&] { net->Configure("distributed_host_qubit_indexing", "guess"); });
  auto emptyHost = NetworkFor(3, 0);
  Reject([&] { emptyHost->Map(CF::CreateCircuit(), 1); });

  auto circuit = CF::CreateCircuit();
  circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 4));
  circuit->AddOperation(CF::CreateMeasurement({{4, 11}, {3, 2}}));
  circuit->AddOperation(CF::CreateSimpleConditionalGate(
      CF::CreateGate(Gate::kXGateType, 7), 11));
  circuit->AddOperation(CF::CreateReset({4}));
  auto mapped = net->Map(circuit, 1);
  Require(mapped.reverseBits.at(0) == 11 && mapped.reverseBits.at(1) == 2,
          "classical IDs were treated as qubit IDs");
  Require(mapped.circuit->GetOperations()[1]->AffectedQubits() == Types::qubits_vector({1, 0}),
          "measurement qubit order");
  auto state = Run(mapped.circuit, 5);
  Require(std::abs(state->Amplitude(16) - 1.) < 1e-12, "conditional/reset mapping");

  // No quantum wires: classical IDs must not influence index classification.
  auto classical = CF::CreateCircuit();
  classical->AddOperation(CF::CreateRandom({11}, 17));
  auto cm = net->Map(classical, 1);
  Require(cm.qubits == 5 && cm.bits == 1 && cm.reverseBits.at(0) == 11,
          "classical-only circuit");
  net->Map(CF::CreateCircuit(), 1);
  Require(net->Map(circuit, 1).qubits == 5, "mapping state leaked between calls");
}

void PublicExecution() {
  for (bool optimize : {false, true}) {
    for (const char* mode : {"local", "global", "auto"}) {
      auto net = NetworkFor(3, 5, mode);
      net->GetController()->SetOptimizeCircuit(optimize);
      const bool local = std::string(mode) == "local";
      const size_t offset = local ? 0 : 3;
      auto circuit = CF::CreateCircuit();
      auto reference = CF::CreateCircuit();
      for (auto* c : {circuit.get(), reference.get()}) {
        const size_t d = c == circuit.get() ? offset : 0;
        c->AddOperation(CF::CreateGate(Gate::kHadamardGateType, d));
        c->AddOperation(CF::CreateGate(Gate::kRzGateType, d, 0, 0, .37));
        c->AddOperation(CF::CreateGate(Gate::kCXGateType, d, d + 1));
        c->AddOperation(CF::CreateGate(Gate::kRyGateType, d + 1, 0, 0, -.21));
      }
      auto expected = Run(reference, 5);
      Compare(net->ExecuteOnHostAmplitudes(circuit, 1), *expected);
      const std::vector<std::string> paulis{"ZIIII", "IZIII", "IIZII", "XXIII", "XYIII"};
      const auto values = net->ExecuteOnHostExpectations(circuit, 1, paulis);
      for (size_t i = 0; i < values.size(); ++i)
        Require(std::abs(values[i] - expected->ExpectationValue(paulis[i])) < 1e-10,
                "host-local observable mapping");
      Reject([&] { net->ExecuteOnHostExpectations(circuit, 1, {"IIIIIZ"}); });
      Compare(net->ExecuteOnHostAmplitudes(circuit, 1), *expected);
    }
    // Cancelling gates on q0 establish local indexing before optimization.
    auto net = NetworkFor(3, 5);
    net->GetController()->SetOptimizeCircuit(optimize);
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 4));
    auto amps = net->ExecuteOnHostAmplitudes(circuit, 1);
    Require(amps.size() == 32 && std::abs(amps[16] - 1.) < 1e-12,
            "optimization changed automatic indexing");
    // Ambiguous already-global X4 belongs on host-local q1.
    auto global = CF::CreateCircuit();
    global->AddOperation(CF::CreateGate(Gate::kXGateType, 4));
    amps = net->ExecuteOnHostAmplitudes(global, 1);
    Require(std::abs(amps[2] - 1.) < 1e-12, "reported overlapping-global regression");
    global->AddOperation(CF::CreateMeasurement({{4, 11}, {3, 2}}));
    const auto counts = net->RepeatedExecuteOnHost(global, 1, 7);
    Require(counts.size() == 1 && counts.begin()->second == 7, "measurement counts");
    const auto& bits = counts.begin()->first;
    Require(bits.size() > 11 && bits[11] && !bits[2], "restored classical result IDs");
  }
}

void RandomComplexStates() {
  std::mt19937 random(19781);
  for (size_t width = 1; width <= 5; ++width)
    for (size_t trial = 0; trial < 24; ++trial) {
      const size_t start = trial % 7;
      const bool global = trial % 2;
      auto net = NetworkFor(start, width, global ? "global" : "local");
      auto original = CF::CreateCircuit(), reference = CF::CreateCircuit();
      for (size_t gate = 0; gate < 24; ++gate) {
        const size_t q = random() % width;
        const size_t other = width > 1 ? (q + 1 + random() % (width - 1)) % width : q;
        const auto kind = random() % 5;
        const auto type = kind == 0 ? Gate::kHadamardGateType : kind == 1 ? Gate::kRxGateType :
            kind == 2 ? Gate::kRzGateType : kind == 3 && width > 1 ? Gate::kCXGateType :
            width > 1 ? Gate::kSwapGateType : Gate::kRyGateType;
        const double angle = (int(random() % 1000) - 500) / 113.;
        const size_t offset = global ? start : 0;
        original->AddOperation(CF::CreateGate(type, q + offset, other + offset, 0, angle));
        reference->AddOperation(CF::CreateGate(type, q, other, 0, angle));
      }
      auto mapped = net->Map(original, 1);
      auto expected = Run(reference, width);
      auto actual = Run(mapped.circuit, width);
      for (size_t i = 0; i < (size_t{1} << width); ++i)
        Require(std::abs(actual->Amplitude(i) - expected->Amplitude(i)) < 1e-10,
                "random complex state mapping");
    }
}

void AllGateMappings() {
  size_t cases = 0;
  for (bool global : {false, true})
    for (size_t q1 = 0; q1 < 5; ++q1)
      for (size_t q2 = 0; q2 < 5; ++q2)
        for (size_t q3 = 0; q3 < 5; ++q3) {
          if (q1 == q2 || q1 == q3 || q2 == q3) continue;
          for (int kind = 0; kind < int(Gate::kNone); ++kind) {
            auto net = NetworkFor(3, 5, global ? "global" : "local");
            auto original = CF::CreateCircuit(), reference = CF::CreateCircuit();
            const size_t offset = global ? 3 : 0;
            // Different rotations on every wire expose misplaced controls and
            // targets even when the gate's computational-basis action is sparse.
            for (size_t q = 0; q < 5; ++q) {
              original->AddOperation(CF::CreateGate(Gate::kRyGateType, q + offset, 0, 0, .13 + .27 * q));
              original->AddOperation(CF::CreateGate(Gate::kRzGateType, q + offset, 0, 0, -.31 + .19 * q));
              reference->AddOperation(CF::CreateGate(Gate::kRyGateType, q, 0, 0, .13 + .27 * q));
              reference->AddOperation(CF::CreateGate(Gate::kRzGateType, q, 0, 0, -.31 + .19 * q));
            }
            original->AddOperation(CF::CreateGate(Gate(kind), q1 + offset, q2 + offset, q3 + offset,
                                                  .19, -.27, .33, .41));
            reference->AddOperation(CF::CreateGate(Gate(kind), q1, q2, q3, .19, -.27, .33, .41));
            auto outer = CF::CreateCircuit();
            outer->AddOperation(original);
            auto mapped = net->Map(outer, 1);
            auto inner = std::static_pointer_cast<Circuit>(mapped.circuit->GetOperations().front());
            Require(inner->GetOperations().back()->AffectedQubits() ==
                        reference->GetOperations().back()->AffectedQubits(), "nested gate target order");
            auto actual = Run(mapped.circuit, 5), expected = Run(reference, 5);
            for (size_t i = 0; i < 32; ++i)
              Require(std::abs(actual->Amplitude(i) - expected->Amplitude(i)) < 1e-10,
                      "built-in gate mapping: " + std::to_string(kind));
            ++cases;
          }
        }
  std::cout << "Checked " << cases << " nested built-in gate mappings\n";
}

void Placement() {
  auto net = NetworkFor(3, 5);
  net->resolvedDistributedDevices[Type::kDistGpuSim] = "2,3,4,5";
  for (size_t n : {8, 2, 1, 8}) {
    const auto cfg = net->ExecutionConfiguration(Type::kDistGpuSim, n);
    const auto want = n == 1 ? "2" : n == 2 ? "2,3" : "2,3,4,5";
    Require(cfg.GetConfiguration("distributed_devices") == want, "automatic shard resizing");
    net->simulator->Configure("distributed_devices", want);
    net->CaptureSimulatorConfiguration();
    Require(net->configuration.GetConfiguration("distributed_devices").empty(),
            "resolved devices became user settings");
  }
  net->resolvedDistributedDevices[Type::kDistMpiGpuSim] = "0,0,0,0";
  Require(net->ExecutionConfiguration(Type::kDistMpiGpuSim, 1)
              .GetConfiguration("distributed_devices") == "0,0,0,0", "MPI ranks were shrunk");
  net->configuration.SetConfiguration("distributed_global_qubits", "0,1");
  Require(net->ExecutionConfiguration(Type::kDistGpuSim, 1)
              .GetConfiguration("distributed_devices") == "2,3,4,5", "explicit globals overridden");
  net->Configure("distributed_devices", "6,7");
  Require(net->ExecutionConfiguration(Type::kDistGpuSim, 1)
              .GetConfiguration("distributed_devices") == "6,7", "explicit shards overridden");
  Require(net->resolvedDistributedDevices.empty(), "new device selection retained old placement");
}
int main() {
  try {
    ExhaustiveMapping();
    InvalidAndClassicalMapping();
    PublicExecution();
    RandomComplexStates();
    AllGateMappings();
    Placement();
    std::cout << "Distributed host mapping and placement passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
