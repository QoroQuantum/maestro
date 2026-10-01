/**
 * @file fusionnetwtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Gate fusion through the network interface.
 *
 * Random circuits are executed with gate fusion explicitly enabled on every
 * simulator that supports it, and compared with a statevector reference that
 * runs without any fusion (Qiskit Aer with its own fusion pass disabled, or the
 * unfused QCSim statevector when Aer is not built in).
 *
 * Covered paths, for circuits with final measurements only and for circuits
 * with mid-circuit measurements, conditional gates and resets:
 * - sampling: many shots in one network call (sampling of the final state, or
 *   the executed prefix restored for every shot)
 * - repeated measurements: one shot per network call on a retained simulator,
 *   the whole circuit, measurements included, executed every time
 * - expectation values of random Pauli strings
 */

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Simulators/Factory.h"

namespace {

using CF = Circuits::CircuitFactory<>;
using SimType = Simulators::SimulatorType;
using Method = Simulators::SimulationType;
using HostNetwork = Network::SimpleDisconnectedNetwork<>;
using Counts = Circuits::Circuit<>::ExecuteResults;

// Mid-circuit results go to the first kMidBits classical bits, final ones
// start at the number of qubits. Few measured bits keep the joint outcome
// space small enough for meaningful per-outcome statistics.
constexpr size_t kMidBits = 2;

constexpr size_t kReferenceShots = 20000;
constexpr size_t kSamplingShots = 8000;
constexpr size_t kRepeatedShots = 1000;
// A GPU backend restoring or recreating its device state on every shot costs
// milliseconds per shot.
constexpr size_t kGpuMidCircuitShots = 500;
constexpr size_t kGpuRepeatedShots = 300;

// Mid-circuit measurements execute every shot past the first measurement, so
// those circuits are kept small.
struct CircuitShape {
  const char* name;
  size_t qubits;
  int gates;
  size_t finalMeasured;
  bool midCircuit;
};
const CircuitShape kFinalMeasurement{"final measurement", 6, 40, 4, false};
const CircuitShape kMidCircuitMeasurement{"mid-circuit measurement", 4, 16, 2,
                                          true};
// Repeated measurements execute the whole circuit for every shot.
const CircuitShape kRepeatedFinalMeasurement{"final measurement", 5, 24, 3,
                                             false};

struct Target {
  const char* name;
  SimType type;
  Method method;
  bool routed = false;  // MPS swap optimization with look-ahead routing
};

std::ostream& operator<<(std::ostream& os, const Target& target) {
  return os << target.name;
}

// Every simulator and method that implements gate fusion.
const std::vector<Target> kTargets{
    {"qcsim_statevector", SimType::kQCSim, Method::kStatevector},
    {"qcsim_density_matrix", SimType::kQCSim, Method::kDensityMatrix},
    {"qcsim_mps", SimType::kQCSim, Method::kMatrixProductState},
    {"qcsim_mps_routed", SimType::kQCSim, Method::kMatrixProductState, true},
    {"qcsim_mpo", SimType::kQCSim, Method::kMatrixProductOperator},
    {"qcsim_mpo_routed", SimType::kQCSim, Method::kMatrixProductOperator,
     true},
    {"qcsim_tensor_network", SimType::kQCSim, Method::kTensorNetwork},
    {"qcsim_composite", SimType::kCompositeQCSim, Method::kStatevector},
#ifdef __linux__
    {"gpu_statevector", SimType::kGpuSim, Method::kStatevector},
    {"gpu_density_matrix", SimType::kGpuSim, Method::kDensityMatrix},
    {"gpu_mps", SimType::kGpuSim, Method::kMatrixProductState},
    {"gpu_mps_routed", SimType::kGpuSim, Method::kMatrixProductState, true},
    {"gpu_mpo", SimType::kGpuSim, Method::kMatrixProductOperator},
    // Only the conventional distributed backend uses Maestro's fusion.
    {"distributed_gpu_conventional", SimType::kDistGpuSim,
     Method::kStatevector},
#endif
};

// GPU backends are optional; CPU ones must always be available.
bool Available(const Target& target) {
#ifdef __linux__
  if (Simulators::IsDistributedGpuSimulator(target.type))
    return Simulators::SimulatorsFactory::IsDistributedGpuAvailable();
  if (target.type == SimType::kGpuSim) {
    static const bool gpu =
        Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
    return gpu && Simulators::SimulatorsFactory::IsGpuLibraryAvailable();
  }
#endif
  return true;
}

std::shared_ptr<HostNetwork> NewNetwork(size_t qubits) {
  return std::make_shared<HostNetwork>(
      std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(qubits)},
      std::vector<size_t>{2 * qubits});
}

std::shared_ptr<HostNetwork> ConfigureFusedNetwork(const Target& target,
                                                   bool optimizeSimulator,
                                                   size_t qubits) {
  auto network = NewNetwork(qubits);
  // Allocation settings must precede the simulator creation.
  network->Configure("gate_fusion", "true");
  if (Simulators::IsGpuSimulator(target.type))
    network->Configure("use_double_precision", "true");
  if (Simulators::IsDistributedGpuSimulator(target.type)) {
    network->Configure("distributed_backend", "conventional");
    // Two shards, sharing a device when only one is present.
    network->Configure("distributed_devices", "0,0");
    network->Configure("distributed_flags", "1");
  }
  if (target.routed) {
    network->SetMPSOptimizeSwaps(true);
    network->SetInitialQubitsMapOptimization(true);
    network->SetMPSOptimizationQubitsNumberThreshold(0);
    network->SetMPSOptimizationBondDimensionThreshold(0);
    network->SetLookaheadDepth(4);
  }
  network->SetOptimizeSimulator(optimizeSimulator);
  network->RemoveAllOptimizationSimulatorsAndAdd(target.type, target.method);
  network->CreateSimulator(target.type, target.method);
  return network;
}

// A GPU backend that is missing, cannot be created here or does not fuse on
// this plugin is skipped (null is returned); a CPU backend must work.
std::shared_ptr<HostNetwork> MakeFusedNetwork(const Target& target,
                                              bool optimizeSimulator,
                                              size_t qubits) {
  const bool optional = Simulators::IsGpuSimulator(target.type);
  if (optional && !Available(target)) {
    BOOST_TEST_MESSAGE(target << " is unavailable; skipping");
    return nullptr;
  }

  std::shared_ptr<HostNetwork> network;
  try {
    network = ConfigureFusedNetwork(target, optimizeSimulator, qubits);
  } catch (const std::exception& e) {
    if (!optional) throw;
    BOOST_TEST_MESSAGE(target << " cannot be created (" << e.what()
                              << "); skipping");
    return nullptr;
  }

  const auto sim = network->GetSimulator();
  const bool fusing =
      sim && sim->GetGateFusionMaxQubits() > 0 && sim->IsGateFusionEnabled();
  if (optional && !fusing) {
    BOOST_TEST_MESSAGE(target << (sim ? " does not support gate fusion"
                                      : " cannot be created")
                              << " here; skipping");
    return nullptr;
  }
  BOOST_REQUIRE_MESSAGE(sim, target << ": the simulator cannot be created");
  BOOST_REQUIRE_MESSAGE(fusing, target << ": gate fusion is not enabled");
  return network;
}

// The known-good reference: a statevector without any gate fusion. Without
// Qiskit Aer in the build, the QCSim statevector with fusion disabled.
std::shared_ptr<HostNetwork> MakeReferenceNetwork(size_t qubits) {
  auto network = NewNetwork(qubits);
  network->Configure("gate_fusion", "false");
#ifndef NO_QISKIT_AER
  // Aer runs its own fusion pass unless it is disabled.
  network->Configure("fusion_enable", "false");
  const auto type = SimType::kQiskitAer;
#else
  const auto type = SimType::kQCSim;
#endif
  network->RemoveAllOptimizationSimulatorsAndAdd(type, Method::kStatevector);
  network->CreateSimulator(type, Method::kStatevector);
  BOOST_REQUIRE(network->GetSimulator());
  return network;
}

void AddRandomGate(Circuits::Circuit<>& circuit, size_t nrQubits,
                   std::mt19937& g) {
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
  std::uniform_real_distribution<double> paramDist(-2. * M_PI, 2. * M_PI);
  std::bernoulli_distribution local(0.5);

  // Half the gates act inside a window of three neighbouring qubits, so there
  // are runs of overlapping gates for fusion to merge.
  Types::qubits_vector qubits;
  if (local(g)) {
    std::uniform_int_distribution<Types::qubit_t> startDist(0, nrQubits - 3);
    const auto start = startDist(g);
    qubits = {start, start + 1, start + 2};
  } else {
    qubits.resize(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
  }
  std::shuffle(qubits.begin(), qubits.end(), g);

  const auto type = static_cast<Circuits::QuantumGateType>(gateDist(g));
  const double p1 = paramDist(g), p2 = paramDist(g), p3 = paramDist(g),
               p4 = paramDist(g);
  circuit.AddOperation(
      CF::CreateGate(type, qubits[0], qubits[1], qubits[2], p1, p2, p3, p4));
}

std::shared_ptr<Circuits::Circuit<>> RandomUnitaryCircuit(
    const CircuitShape& shape, std::mt19937& g) {
  auto circuit = CF::CreateCircuit();
  // A generic U gate first: an all-Clifford circuit would be moved to the
  // stabilizer simulator, which does not fuse.
  std::uniform_real_distribution<double> paramDist(-2. * M_PI, 2. * M_PI);
  const double p1 = paramDist(g), p2 = paramDist(g), p3 = paramDist(g);
  circuit->AddOperation(
      CF::CreateGate(Circuits::QuantumGateType::kUGateType, 0, 0, 0, p1, p2,
                     p3));
  for (int i = 0; i < shape.gates; ++i)
    AddRandomGate(*circuit, shape.qubits, g);
  return circuit;
}

void AddFinalMeasurement(Circuits::Circuit<>& circuit,
                         const CircuitShape& shape, std::mt19937& g) {
  Types::qubits_vector qubits(shape.qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  std::shuffle(qubits.begin(), qubits.end(), g);

  std::vector<std::pair<Types::qubit_t, size_t>> pairs;
  for (size_t k = 0; k < shape.finalMeasured; ++k)
    pairs.emplace_back(qubits[k], shape.qubits + k);
  circuit.AddOperation(CF::CreateMeasurement(pairs));
}

// Gates interleaved with measurements, conditional gates and resets, which are
// all boundaries for the fusion cache.
std::shared_ptr<Circuits::Circuit<>> RandomMidCircuitMeasurementCircuit(
    const CircuitShape& shape, std::mt19937& g) {
  auto circuit = RandomUnitaryCircuit(shape, g);
  // Rebuild with the boundaries interleaved, keeping the first U gate.
  const auto unitary = circuit->GetOperations();
  circuit = CF::CreateCircuit({unitary.front()});

  std::uniform_int_distribution<Types::qubit_t> qubitDist(0, shape.qubits - 1);
  std::uniform_int_distribution<size_t> cbitDist(0, kMidBits - 1);
  std::uniform_real_distribution<double> angleDist(-M_PI, M_PI);
  std::bernoulli_distribution measureNow(0.15), conditionalNow(0.5),
      resetNow(0.05);

  std::vector<size_t> measured;
  for (size_t i = 1; i < unitary.size(); ++i) {
    circuit->AddOperation(unitary[i]);

    // At least two measurements, at a third and two thirds of the circuit.
    const bool forced = i == unitary.size() / 3 || i == 2 * unitary.size() / 3;
    if (forced || measureNow(g)) {
      const auto cbit = measured.size() < kMidBits ? measured.size()
                                                   : cbitDist(g);
      circuit->AddOperation(CF::CreateMeasurement({{qubitDist(g), cbit}}));
      measured.push_back(cbit);
    }
    if (!measured.empty() && conditionalNow(g)) {
      std::uniform_int_distribution<size_t> pick(0, measured.size() - 1);
      const auto gate = std::static_pointer_cast<Circuits::IGateOperation<>>(
          CF::CreateGate(Circuits::QuantumGateType::kRyGateType, qubitDist(g),
                         0, 0, angleDist(g)));
      circuit->AddOperation(
          CF::CreateSimpleConditionalGate(gate, measured[pick(g)]));
    }
    if (resetNow(g)) circuit->AddOperation(CF::CreateReset({qubitDist(g)}));
  }

  AddFinalMeasurement(*circuit, shape, g);
  return circuit;
}

std::shared_ptr<Circuits::Circuit<>> RandomCircuit(const CircuitShape& shape,
                                                   std::mt19937& g) {
  if (shape.midCircuit) return RandomMidCircuitMeasurementCircuit(shape, g);
  auto circuit = RandomUnitaryCircuit(shape, g);
  AddFinalMeasurement(*circuit, shape, g);
  return circuit;
}

std::string Bits(const std::vector<bool>& bits) {
  std::string result;
  for (const bool bit : bits) result += bit ? '1' : '0';
  return result;
}

// Upper quantile of the chi-square distribution (Wilson-Hilferty), for the
// standard normal quantile z.
double ChiSquareQuantile(size_t df, double z) {
  const double k = static_cast<double>(df);
  const double a = 2. / (9. * k);
  return k * std::pow(1. - a + z * std::sqrt(a), 3);
}

// Two-sample check that the tested counts follow the reference distribution:
// every outcome within 5 standard deviations, and a chi-square homogeneity
// test at p = 1e-6 over the whole distribution.
void CheckSameDistribution(const Counts& tested, size_t testedShots,
                           const Counts& reference, size_t referenceShots,
                           const std::string& label) {
  size_t total = 0;
  for (const auto& [outcome, count] : tested) total += count;
  BOOST_CHECK_MESSAGE(total == testedShots,
                      label << ": " << total << " results for " << testedShots
                            << " shots");

  Counts all = reference;
  for (const auto& [outcome, count] : tested) all.emplace(outcome, 0);

  const double nA = static_cast<double>(testedShots);
  const double nB = static_cast<double>(referenceShots);
  const double kA = std::sqrt(nB / nA), kB = std::sqrt(nA / nB);
  double chi2 = 0;
  size_t bins = 0;
  double rareA = 0, rareB = 0;

  for (const auto& [outcome, unused] : all) {
    const auto itA = tested.find(outcome);
    const auto itB = reference.find(outcome);
    const double a = itA == tested.end() ? 0. : itA->second;
    const double b = itB == reference.end() ? 0. : itB->second;

    const double pA = a / nA, pB = b / nB;
    const double pooled = (a + b) / (nA + nB);
    const double sigma = std::sqrt(pooled * (1. - pooled) * (1. / nA + 1. / nB));
    const double tolerance = 5. * sigma + 0.5 / nA + 0.5 / nB;
    BOOST_CHECK_MESSAGE(std::abs(pA - pB) <= tolerance,
                        label << ": outcome " << Bits(outcome) << " has "
                              << "probability " << pA << ", reference " << pB
                              << " (tolerance " << tolerance << ")");

    // Sparse outcomes are pooled to keep the chi-square approximation valid.
    if (a + b < 10) {
      rareA += a;
      rareB += b;
      continue;
    }
    chi2 += std::pow(kA * a - kB * b, 2) / (a + b);
    ++bins;
  }
  if (rareA + rareB > 0) {
    chi2 += std::pow(kA * rareA - kB * rareB, 2) / (rareA + rareB);
    ++bins;
  }
  if (bins > 1) {
    const double critical = ChiSquareQuantile(bins - 1, 4.753);
    BOOST_CHECK_MESSAGE(chi2 <= critical,
                        label << ": chi-square " << chi2 << " exceeds "
                              << critical << " for " << bins - 1
                              << " degrees of freedom");
  }
}

std::string RandomPauliString(size_t nrQubits, std::mt19937& g) {
  static const char paulis[] = {'I', 'X', 'Y', 'Z'};
  std::uniform_int_distribution<int> dist(0, 3);
  std::string pauli(nrQubits, 'I');
  for (auto& p : pauli) p = paulis[dist(g)];
  return pauli;
}

size_t Shots(const Target& target, const CircuitShape& shape, size_t shots) {
  return shape.midCircuit && Simulators::IsGpuSimulator(target.type)
             ? std::min(shots, kGpuMidCircuitShots)
             : shots;
}

std::string Label(const Target& target, const char* kind, int trial) {
  std::ostringstream os;
  os << target << ' ' << kind << " circuit " << trial;
  return os.str();
}

// Many shots in one call: the final state is sampled, or for mid-circuit
// measurements the executed prefix is restored for every shot.
void CheckSampling(const Target& target, const CircuitShape& shape,
                   uint32_t seed) {
  auto fused = MakeFusedNetwork(target, true, shape.qubits);
  if (!fused) return;
  auto reference = MakeReferenceNetwork(shape.qubits);
  const auto shots = Shots(target, shape, kSamplingShots);

  for (int trial = 0; trial < 4; ++trial) {
    std::mt19937 g(seed + trial);
    const auto circuit = RandomCircuit(shape, g);
    reference->Configure("seed", std::to_string(seed + 1000 + trial).c_str());
    fused->Configure("seed", std::to_string(seed + 2000 + trial).c_str());

    const auto expected =
        reference->RepeatedExecuteOnHost(circuit, 0, kReferenceShots);
    BOOST_REQUIRE(!reference->WasGateFusionEnabled());
    const auto counts =
        fused->RepeatedExecuteOnHost(circuit, 0, shots);
    BOOST_CHECK_MESSAGE(fused->WasGateFusionEnabled(),
                        Label(target, shape.name, trial) << ": ran without fusion");
    CheckSameDistribution(counts, shots, expected, kReferenceShots,
                          Label(target, shape.name, trial));
  }
}

// One shot per call, on the simulator the network retains between calls: the
// whole circuit runs every time and measurements collapse the state.
void CheckRepeatedMeasurements(const Target& target,
                               const CircuitShape& shape, uint32_t seed) {
  // Without the simulator optimization a single shot is not sampled.
  auto fused = MakeFusedNetwork(target, false, shape.qubits);
  if (!fused) return;
  auto reference = MakeReferenceNetwork(shape.qubits);
  const auto shots = Simulators::IsGpuSimulator(target.type)
                         ? kGpuRepeatedShots
                         : kRepeatedShots;

  for (int trial = 0; trial < 2; ++trial) {
    std::mt19937 g(seed + trial);
    const auto circuit = RandomCircuit(shape, g);
    reference->Configure("seed", std::to_string(seed + 1000 + trial).c_str());
    const auto expected =
        reference->RepeatedExecuteOnHost(circuit, 0, kReferenceShots);

    Counts counts;
    bool fusedEveryShot = true;
    for (size_t shot = 0; shot < shots; ++shot) {
      // Every call reseeds the simulator, so each shot needs its own seed.
      const auto shotSeed = (uint64_t(seed + trial) << 20) + shot;
      fused->Configure("seed", std::to_string(shotSeed).c_str());
      fused->ExecuteOnHost(circuit, 0);
      fusedEveryShot = fusedEveryShot && fused->WasGateFusionEnabled();

      auto bits = fused->GetState().GetAllBits();
      bits.resize(2 * shape.qubits, false);
      ++counts[bits];
    }
    BOOST_CHECK_MESSAGE(fusedEveryShot,
                        Label(target, shape.name, trial) << ": ran without fusion");
    CheckSameDistribution(counts, shots, expected, kReferenceShots,
                          Label(target, shape.name, trial));
  }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(FusionNetwTests)

BOOST_DATA_TEST_CASE(FusedSamplingMatchesReference, bdata::make(kTargets),
                     target) {
  CheckSampling(target, kFinalMeasurement, 0xF5A0u);
}

BOOST_DATA_TEST_CASE(FusedMidCircuitSamplingMatchesReference,
                     bdata::make(kTargets), target) {
  CheckSampling(target, kMidCircuitMeasurement, 0xF5B0u);
}

BOOST_DATA_TEST_CASE(FusedRepeatedMeasurementsMatchReference,
                     bdata::make(kTargets), target) {
  CheckRepeatedMeasurements(target, kRepeatedFinalMeasurement, 0xF5C0u);
}

BOOST_DATA_TEST_CASE(FusedMidCircuitRepeatedMeasurementsMatchReference,
                     bdata::make(kTargets), target) {
  CheckRepeatedMeasurements(target, kMidCircuitMeasurement, 0xF5D0u);
}

BOOST_DATA_TEST_CASE(FusedExpectationValuesMatchReference,
                     bdata::make(kTargets), target) {
  const auto& shape = kFinalMeasurement;
  auto fused = MakeFusedNetwork(target, true, shape.qubits);
  if (!fused) return;
  auto reference = MakeReferenceNetwork(shape.qubits);

  for (int trial = 0; trial < 6; ++trial) {
    std::mt19937 g(0xF5E0u + trial);
    const auto circuit = RandomUnitaryCircuit(shape, g);
    std::vector<std::string> paulis;
    for (int i = 0; i < 20; ++i)
      paulis.push_back(RandomPauliString(shape.qubits, g));

    const auto expected =
        reference->ExecuteOnHostExpectations(circuit, 0, paulis);
    BOOST_REQUIRE(!reference->WasGateFusionEnabled());
    const auto values = fused->ExecuteOnHostExpectations(circuit, 0, paulis);
    const auto label = Label(target, "expectation", trial);
    BOOST_CHECK_MESSAGE(fused->WasGateFusionEnabled(),
                        label << ": ran without fusion");

    BOOST_REQUIRE_EQUAL(expected.size(), paulis.size());
    BOOST_REQUIRE_EQUAL(values.size(), paulis.size());
    for (size_t i = 0; i < paulis.size(); ++i)
      BOOST_CHECK_MESSAGE(std::abs(values[i] - expected[i]) < 1e-6,
                          label << ": <" << paulis[i] << "> = " << values[i]
                                << ", reference " << expected[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
