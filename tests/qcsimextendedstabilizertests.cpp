/**
 * @file qcsimextendedstabilizertests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the Maestro QCSim extended stabilizer wrapper.
 */

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Simulators/Factory.h"
#include "../Simulators/QCSimExtendedStabilizer.h"

namespace {

constexpr size_t kNrQubits = 3;
constexpr double kTolerance = 1e-9;
constexpr double kPi = 3.141592653589793238462643383279502884;

enum class Gate {
  kX,
  kY,
  kZ,
  kH,
  kS,
  kSDG,
  kSX,
  kSXDG,
  kK,
  kP,
  kT,
  kTDG,
  kRX,
  kRY,
  kRZ,
  kU,
  kCX,
  kCY,
  kCZ,
  kSwap,
  kISwap,
  kISwapDG,
  kCH,
  kCRX,
  kCRY,
  kCRZ,
  kCP,
  kCS,
  kCSDAG,
  kCSX,
  kCSXDAG,
  kCU,
  kCCX,
  kCSwap,
};

struct GateCase {
  Gate gate;
  const char* name;
};

const std::array<GateCase, 34> kGateCases{{
    {Gate::kX, "X"},           {Gate::kY, "Y"},
    {Gate::kZ, "Z"},           {Gate::kH, "H"},
    {Gate::kS, "S"},           {Gate::kSDG, "SDG"},
    {Gate::kSX, "SX"},         {Gate::kSXDG, "SXDG"},
    {Gate::kK, "K"},           {Gate::kP, "P"},
    {Gate::kT, "T"},           {Gate::kTDG, "TDG"},
    {Gate::kRX, "RX"},         {Gate::kRY, "RY"},
    {Gate::kRZ, "RZ"},         {Gate::kU, "U"},
    {Gate::kCX, "CX"},         {Gate::kCY, "CY"},
    {Gate::kCZ, "CZ"},         {Gate::kSwap, "SWAP"},
    {Gate::kISwap, "ISWAP"},   {Gate::kISwapDG, "ISWAPDG"},
    {Gate::kCH, "CH"},         {Gate::kCRX, "CRX"},
    {Gate::kCRY, "CRY"},       {Gate::kCRZ, "CRZ"},
    {Gate::kCP, "CP"},         {Gate::kCS, "CS"},
    {Gate::kCSDAG, "CSDAG"},   {Gate::kCSX, "CSX"},
    {Gate::kCSXDAG, "CSXDAG"}, {Gate::kCU, "CU"},
    {Gate::kCCX, "CCX"},       {Gate::kCSwap, "CSWAP"},
}};

std::shared_ptr<Simulators::ISimulator> CreateStatevector() {
  auto statevector = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  statevector->AllocateQubits(kNrQubits);
  statevector->Initialize();
  return statevector;
}

void PrepareReferenceState(
    Simulators::QCSimExtendedStabilizer& extendedStabilizer,
    const std::shared_ptr<Simulators::ISimulator>& statevector) {
  extendedStabilizer.ApplyH(0);
  statevector->ApplyH(0);

  extendedStabilizer.ApplyRY(1, 0.37);
  statevector->ApplyRy(1, 0.37);

  extendedStabilizer.ApplyRX(2, -0.29);
  statevector->ApplyRx(2, -0.29);

  extendedStabilizer.ApplyCX(0, 1);
  statevector->ApplyCX(0, 1);

  extendedStabilizer.ApplyRZ(0, 0.21);
  statevector->ApplyRz(0, 0.21);

  extendedStabilizer.ApplyCX(1, 2);
  statevector->ApplyCX(1, 2);
}

void ApplyGate(Simulators::QCSimExtendedStabilizer& simulator, Gate gate) {
  switch (gate) {
    case Gate::kX:
      simulator.ApplyX(0);
      break;
    case Gate::kY:
      simulator.ApplyY(0);
      break;
    case Gate::kZ:
      simulator.ApplyZ(0);
      break;
    case Gate::kH:
      simulator.ApplyH(0);
      break;
    case Gate::kS:
      simulator.ApplyS(0);
      break;
    case Gate::kSDG:
      simulator.ApplySDG(0);
      break;
    case Gate::kSX:
      simulator.ApplySX(0);
      break;
    case Gate::kSXDG:
      simulator.ApplySXDG(0);
      break;
    case Gate::kK:
      simulator.ApplyK(0);
      break;
    case Gate::kP:
      simulator.ApplyP(0, 0.43);
      break;
    case Gate::kT:
      simulator.ApplyT(0);
      break;
    case Gate::kTDG:
      simulator.ApplyTDG(0);
      break;
    case Gate::kRX:
      simulator.ApplyRX(0, -0.31);
      break;
    case Gate::kRY:
      simulator.ApplyRY(0, 0.47);
      break;
    case Gate::kRZ:
      simulator.ApplyRZ(0, -0.53);
      break;
    case Gate::kU:
      simulator.ApplyU(0, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCX:
      simulator.ApplyCX(0, 1);
      break;
    case Gate::kCY:
      simulator.ApplyCY(0, 1);
      break;
    case Gate::kCZ:
      simulator.ApplyCZ(0, 1);
      break;
    case Gate::kSwap:
      simulator.ApplySWAP(0, 1);
      break;
    case Gate::kISwap:
      simulator.ApplyISWAP(0, 1);
      break;
    case Gate::kISwapDG:
      simulator.ApplyISWAPDG(0, 1);
      break;
    case Gate::kCH:
      simulator.ApplyCH(0, 1);
      break;
    case Gate::kCRX:
      simulator.ApplyCRX(0, 1, -0.31);
      break;
    case Gate::kCRY:
      simulator.ApplyCRY(0, 1, 0.47);
      break;
    case Gate::kCRZ:
      simulator.ApplyCRZ(0, 1, -0.53);
      break;
    case Gate::kCP:
      simulator.ApplyCP(0, 1, 0.43);
      break;
    case Gate::kCS:
      simulator.ApplyCS(0, 1);
      break;
    case Gate::kCSDAG:
      simulator.ApplyCSDAG(0, 1);
      break;
    case Gate::kCSX:
      simulator.ApplyCSX(0, 1);
      break;
    case Gate::kCSXDAG:
      simulator.ApplyCSXDAG(0, 1);
      break;
    case Gate::kCU:
      simulator.ApplyCU(0, 1, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCCX:
      simulator.ApplyCCX(0, 1, 2);
      break;
    case Gate::kCSwap:
      simulator.ApplyCSwap(0, 1, 2);
      break;
  }
}

void ApplyGate(const std::shared_ptr<Simulators::ISimulator>& simulator,
               Gate gate) {
  switch (gate) {
    case Gate::kX:
      simulator->ApplyX(0);
      break;
    case Gate::kY:
      simulator->ApplyY(0);
      break;
    case Gate::kZ:
      simulator->ApplyZ(0);
      break;
    case Gate::kH:
      simulator->ApplyH(0);
      break;
    case Gate::kS:
      simulator->ApplyS(0);
      break;
    case Gate::kSDG:
      simulator->ApplySDG(0);
      break;
    case Gate::kSX:
      simulator->ApplySx(0);
      break;
    case Gate::kSXDG:
      simulator->ApplySxDAG(0);
      break;
    case Gate::kK:
      simulator->ApplyK(0);
      break;
    case Gate::kP:
      simulator->ApplyP(0, 0.43);
      break;
    case Gate::kT:
      simulator->ApplyT(0);
      break;
    case Gate::kTDG:
      simulator->ApplyTDG(0);
      break;
    case Gate::kRX:
      simulator->ApplyRx(0, -0.31);
      break;
    case Gate::kRY:
      simulator->ApplyRy(0, 0.47);
      break;
    case Gate::kRZ:
      simulator->ApplyRz(0, -0.53);
      break;
    case Gate::kU:
      simulator->ApplyU(0, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCX:
      simulator->ApplyCX(0, 1);
      break;
    case Gate::kCY:
      simulator->ApplyCY(0, 1);
      break;
    case Gate::kCZ:
      simulator->ApplyCZ(0, 1);
      break;
    case Gate::kSwap:
      simulator->ApplySwap(0, 1);
      break;
    case Gate::kISwap: {
      const Eigen::Matrix4cd matrix =
          QC::Gates::iSwapGate<>().getRawOperatorMatrix();
      simulator->ApplyGenericTwoQubitGate(0, 1, matrix);
      break;
    }
    case Gate::kISwapDG: {
      const Eigen::Matrix4cd matrix =
          QC::Gates::iSwapDagGate<>().getRawOperatorMatrix();
      simulator->ApplyGenericTwoQubitGate(0, 1, matrix);
      break;
    }
    case Gate::kCH:
      simulator->ApplyCH(0, 1);
      break;
    case Gate::kCRX:
      simulator->ApplyCRx(0, 1, -0.31);
      break;
    case Gate::kCRY:
      simulator->ApplyCRy(0, 1, 0.47);
      break;
    case Gate::kCRZ:
      simulator->ApplyCRz(0, 1, -0.53);
      break;
    case Gate::kCP:
      simulator->ApplyCP(0, 1, 0.43);
      break;
    case Gate::kCS:
      simulator->ApplyCP(0, 1, kPi / 2.0);
      break;
    case Gate::kCSDAG:
      simulator->ApplyCP(0, 1, -kPi / 2.0);
      break;
    case Gate::kCSX:
      simulator->ApplyCSx(0, 1);
      break;
    case Gate::kCSXDAG:
      simulator->ApplyCSxDAG(0, 1);
      break;
    case Gate::kCU:
      simulator->ApplyCU(0, 1, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCCX:
      simulator->ApplyCCX(0, 1, 2);
      break;
    case Gate::kCSwap:
      simulator->ApplyCSwap(0, 1, 2);
      break;
  }
}

std::string DecodePauliString(size_t encoded) {
  static constexpr std::array<char, 4> paulis{'I', 'X', 'Y', 'Z'};
  std::string pauli(kNrQubits, 'I');
  for (size_t qubit = 0; qubit < kNrQubits; ++qubit)
    pauli[qubit] = paulis[(encoded >> (2 * qubit)) & 3ULL];
  return pauli;
}

double ExtendedBasisProbability(
    const Simulators::QCSimExtendedStabilizer& simulator, size_t outcome) {
  double probability = 0.0;
  const size_t nrTerms = 1ULL << kNrQubits;
  for (size_t mask = 0; mask < nrTerms; ++mask) {
    std::string pauli(kNrQubits, 'I');
    double sign = 1.0;
    for (size_t qubit = 0; qubit < kNrQubits; ++qubit) {
      if (((mask >> qubit) & 1ULL) == 0) continue;
      pauli[qubit] = 'Z';
      if (((outcome >> qubit) & 1ULL) != 0) sign = -sign;
    }
    probability += sign * simulator.ExpectationValue(pauli);
  }
  return probability / static_cast<double>(nrTerms);
}

void CheckStatesMatch(
    const Simulators::QCSimExtendedStabilizer& extendedStabilizer,
    const std::shared_ptr<Simulators::ISimulator>& statevector,
    const std::string& context) {
  BOOST_TEST_CONTEXT(context) {
    const size_t nrPauliStrings = 1ULL << (2 * kNrQubits);
    for (size_t encoded = 0; encoded < nrPauliStrings; ++encoded) {
      const std::string pauli = DecodePauliString(encoded);
      const double expected = statevector->ExpectationValue(pauli);
      const double actual = extendedStabilizer.ExpectationValue(pauli);
      BOOST_TEST(std::abs(expected - actual) < kTolerance,
                 "Expectation mismatch for "
                     << pauli << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }

    const size_t nrStates = 1ULL << kNrQubits;
    for (size_t outcome = 0; outcome < nrStates; ++outcome) {
      const double expected = statevector->Probability(outcome);
      const double actual =
          ExtendedBasisProbability(extendedStabilizer, outcome);
      BOOST_TEST(std::abs(expected - actual) < kTolerance,
                 "Probability mismatch for outcome "
                     << outcome << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }

    for (size_t qubit = 0; qubit < kNrQubits; ++qubit) {
      double expected = 0.0;
      for (size_t outcome = 0; outcome < nrStates; ++outcome)
        if (((outcome >> qubit) & 1ULL) != 0)
          expected += statevector->Probability(outcome);
      const double actual = extendedStabilizer.GetQubitProbability(qubit);
      BOOST_TEST(std::abs(expected - actual) < kTolerance,
                 "Qubit probability mismatch for qubit "
                     << qubit << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }
  }
}

std::unordered_map<size_t, size_t> SampleExtendedStabilizer(
    Simulators::QCSimExtendedStabilizer& simulator, size_t shots) {
  const std::array<std::array<size_t, kNrQubits>, 3> measurementOrders{{
      {0, 1, 2},
      {2, 0, 1},
      {1, 2, 0},
  }};

  std::unordered_map<size_t, size_t> counts;
  simulator.SaveState();
  for (size_t shot = 0; shot < shots; ++shot) {
    simulator.RestoreState();
    size_t outcome = 0;
    for (const size_t qubit :
         measurementOrders[shot % measurementOrders.size()])
      if (simulator.Measure(qubit)) outcome |= 1ULL << qubit;
    ++counts[outcome];
  }
  simulator.RestoreState();
  return counts;
}

constexpr size_t kInterfaceNrQubits = 4;

struct InterfaceOperation {
  unsigned int gate;
  Types::qubit_t qubit0;
  Types::qubit_t qubit1;
  Types::qubit_t qubit2;
  double theta;
  double phi;
  double lambda;
  double gamma;
};

std::shared_ptr<Simulators::ISimulator> CreateQCSimInterfaceSimulator(
    Simulators::SimulationType simulationType,
    size_t nrQubits = kInterfaceNrQubits) {
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim, simulationType);
  simulator->AllocateQubits(nrQubits);
  simulator->Initialize();
  return simulator;
}

void ApplyInterfaceOperation(Simulators::ISimulator& simulator,
                             const InterfaceOperation& operation) {
  switch (operation.gate) {
    case 0: simulator.ApplyX(operation.qubit0); break;
    case 1: simulator.ApplyY(operation.qubit0); break;
    case 2: simulator.ApplyZ(operation.qubit0); break;
    case 3: simulator.ApplyH(operation.qubit0); break;
    case 4: simulator.ApplyS(operation.qubit0); break;
    case 5: simulator.ApplySDG(operation.qubit0); break;
    case 6: simulator.ApplyT(operation.qubit0); break;
    case 7: simulator.ApplyTDG(operation.qubit0); break;
    case 8: simulator.ApplySx(operation.qubit0); break;
    case 9: simulator.ApplySxDAG(operation.qubit0); break;
    case 10: simulator.ApplyK(operation.qubit0); break;
    case 11: simulator.ApplyP(operation.qubit0, operation.theta); break;
    case 12: simulator.ApplyRx(operation.qubit0, operation.theta); break;
    case 13: simulator.ApplyRy(operation.qubit0, operation.theta); break;
    case 14: simulator.ApplyRz(operation.qubit0, operation.theta); break;
    case 15:
      simulator.ApplyU(operation.qubit0, operation.theta, operation.phi,
                       operation.lambda, operation.gamma);
      break;
    case 16: simulator.ApplyCX(operation.qubit0, operation.qubit1); break;
    case 17: simulator.ApplyCY(operation.qubit0, operation.qubit1); break;
    case 18: simulator.ApplyCZ(operation.qubit0, operation.qubit1); break;
    case 19:
      simulator.ApplyCP(operation.qubit0, operation.qubit1, operation.theta);
      break;
    case 20:
      simulator.ApplyCRx(operation.qubit0, operation.qubit1, operation.theta);
      break;
    case 21:
      simulator.ApplyCRy(operation.qubit0, operation.qubit1, operation.theta);
      break;
    case 22:
      simulator.ApplyCRz(operation.qubit0, operation.qubit1, operation.theta);
      break;
    case 23: simulator.ApplyCH(operation.qubit0, operation.qubit1); break;
    case 24: simulator.ApplyCSx(operation.qubit0, operation.qubit1); break;
    case 25: simulator.ApplyCSxDAG(operation.qubit0, operation.qubit1); break;
    case 26: simulator.ApplySwap(operation.qubit0, operation.qubit1); break;
    case 27:
      simulator.ApplyCU(operation.qubit0, operation.qubit1, operation.theta,
                        operation.phi, operation.lambda, operation.gamma);
      break;
    case 28:
      simulator.ApplyCCX(operation.qubit0, operation.qubit1,
                         operation.qubit2);
      break;
    case 29:
      simulator.ApplyCSwap(operation.qubit0, operation.qubit1,
                           operation.qubit2);
      break;
    default: BOOST_FAIL("Unknown interface gate " << operation.gate);
  }
}

std::vector<InterfaceOperation> GenerateInterfaceCircuit(uint64_t seed,
                                                         size_t gateCount) {
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<unsigned int> gateDistribution(0, 29);
  std::uniform_real_distribution<double> angleDistribution(-kPi, kPi);
  std::array<Types::qubit_t, kInterfaceNrQubits> qubits = {0, 1, 2, 3};
  std::vector<InterfaceOperation> circuit;
  circuit.reserve(gateCount);

  for (size_t gate = 0; gate < gateCount; ++gate) {
    std::shuffle(qubits.begin(), qubits.end(), generator);
    circuit.push_back({gateDistribution(generator), qubits[0], qubits[1],
                       qubits[2], angleDistribution(generator),
                       angleDistribution(generator),
                       angleDistribution(generator),
                       angleDistribution(generator)});
  }
  return circuit;
}

void CheckInterfaceProbabilities(Simulators::IState& expected,
                                 Simulators::IState& actual,
                                 double tolerance = 1e-9) {
  const auto expectedProbabilities = expected.AllProbabilities();
  const auto actualProbabilities = actual.AllProbabilities();
  BOOST_REQUIRE_EQUAL(actualProbabilities.size(), expectedProbabilities.size());
  for (size_t outcome = 0; outcome < expectedProbabilities.size(); ++outcome) {
    BOOST_TEST(std::abs(actualProbabilities[outcome] -
                        expectedProbabilities[outcome]) < tolerance,
               "Probability mismatch for outcome " << outcome);
    BOOST_TEST(std::abs(actual.Probability(outcome) -
                        expectedProbabilities[outcome]) < tolerance);
  }
}

std::vector<double> InterfaceMarginals(
    const std::vector<double>& fullProbabilities,
    const Types::qubits_vector& qubits) {
  std::vector<double> probabilities(1ULL << qubits.size(), 0.0);
  for (size_t state = 0; state < fullProbabilities.size(); ++state) {
    size_t outcome = 0;
    for (size_t i = 0; i < qubits.size(); ++i)
      outcome |= ((state >> qubits[i]) & 1ULL) << i;
    probabilities[outcome] += fullProbabilities[state];
  }
  return probabilities;
}

void CheckInterfaceSamples(
    const std::unordered_map<Types::qubit_t, Types::qubit_t>& counts,
    const std::vector<double>& expected, size_t shots,
    double tolerance = 0.05) {
  for (size_t outcome = 0; outcome < expected.size(); ++outcome) {
    const auto found = counts.find(outcome);
    const double sampled = static_cast<double>(
        found == counts.end() ? 0 : found->second) / shots;
    BOOST_TEST(std::abs(sampled - expected[outcome]) < tolerance,
               "Sampling mismatch for outcome " << outcome);
  }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(qcsim_extended_stabilizer_wrapper_tests)

BOOST_AUTO_TEST_CASE(control_target_order_matches_maestro) {
  Simulators::QCSimExtendedStabilizer simulator(2);

  simulator.ApplyX(0);
  simulator.ApplyCX(0, 1);

  BOOST_CHECK_SMALL(1.0 - simulator.GetQubitProbability(0), 1e-12);
  BOOST_CHECK_SMALL(1.0 - simulator.GetQubitProbability(1), 1e-12);
}

BOOST_AUTO_TEST_CASE(all_gates_match_qcsim_statevector) {
  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);

  for (const auto& gateCase : kGateCases) {
    statevector->Reset();
    extendedStabilizer.Reset(kNrQubits);
    PrepareReferenceState(extendedStabilizer, statevector);

    ApplyGate(extendedStabilizer, gateCase.gate);
    ApplyGate(statevector, gateCase.gate);

    CheckStatesMatch(extendedStabilizer, statevector,
                     std::string("Gate ") + gateCase.name);
  }
}

BOOST_AUTO_TEST_CASE(mixed_circuit_matches_qcsim_statevector) {
  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);

  PrepareReferenceState(extendedStabilizer, statevector);
  const std::array<Gate, 18> circuit{
      Gate::kT,   Gate::kCRY, Gate::kH,  Gate::kCU,  Gate::kSDG, Gate::kCP,
      Gate::kRX,  Gate::kCY,  Gate::kU,  Gate::kCSX, Gate::kRZ,  Gate::kCCX,
      Gate::kTDG, Gate::kCH,  Gate::kRY, Gate::kCRZ, Gate::kCZ,  Gate::kCSwap,
  };

  for (const Gate gate : circuit) {
    ApplyGate(extendedStabilizer, gate);
    ApplyGate(statevector, gate);
  }

  CheckStatesMatch(extendedStabilizer, statevector, "Mixed circuit");
}

BOOST_AUTO_TEST_CASE(measurement_sampling_matches_qcsim_statevector) {
  constexpr size_t shots = 8192;
  constexpr double sampleTolerance = 0.04;

  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);
  extendedStabilizer.SetRandomSeed(0x5A17B1E3U);
  PrepareReferenceState(extendedStabilizer, statevector);
  ApplyGate(extendedStabilizer, Gate::kT);
  ApplyGate(statevector, Gate::kT);
  ApplyGate(extendedStabilizer, Gate::kCRY);
  ApplyGate(statevector, Gate::kCRY);
  ApplyGate(extendedStabilizer, Gate::kCP);
  ApplyGate(statevector, Gate::kCP);

  Types::qubits_vector qubits(kNrQubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  const auto statevectorCounts = statevector->SampleCounts(qubits, shots);
  const auto extendedCounts =
      SampleExtendedStabilizer(extendedStabilizer, shots);

  for (size_t outcome = 0; outcome < (1ULL << kNrQubits); ++outcome) {
    const double expected = statevector->Probability(outcome);
    const auto statevectorCount = statevectorCounts.find(outcome);
    const auto extendedCount = extendedCounts.find(outcome);
    const double statevectorFrequency =
        statevectorCount == statevectorCounts.end()
            ? 0.0
            : static_cast<double>(statevectorCount->second) / shots;
    const double extendedFrequency =
        extendedCount == extendedCounts.end()
            ? 0.0
            : static_cast<double>(extendedCount->second) / shots;

    BOOST_TEST(std::abs(statevectorFrequency - expected) < sampleTolerance,
               "Statevector sample mismatch for outcome "
                   << outcome << ": exact " << expected << ", sampled "
                   << statevectorFrequency);
    BOOST_TEST(std::abs(extendedFrequency - expected) < sampleTolerance,
               "Extended stabilizer sample mismatch for outcome "
                   << outcome << ": statevector exact " << expected
                   << ", sampled " << extendedFrequency);
    BOOST_TEST(std::abs(extendedFrequency - statevectorFrequency) <
                   2.0 * sampleTolerance,
               "Sample distribution mismatch for outcome "
                   << outcome << ": statevector " << statevectorFrequency
                   << ", extended stabilizer " << extendedFrequency);
  }

  // Sampling restores the state after every shot and leaves it unchanged.
  CheckStatesMatch(extendedStabilizer, statevector,
                   "State after measurement sampling");
}

BOOST_AUTO_TEST_CASE(save_restore_and_reset_match_qcsim_statevector) {
  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);
  PrepareReferenceState(extendedStabilizer, statevector);

  extendedStabilizer.SaveState();
  statevector->SaveState();

  ApplyGate(extendedStabilizer, Gate::kCU);
  ApplyGate(statevector, Gate::kCU);
  ApplyGate(extendedStabilizer, Gate::kCCX);
  ApplyGate(statevector, Gate::kCCX);
  CheckStatesMatch(extendedStabilizer, statevector, "Mutated saved state");

  extendedStabilizer.RestoreState();
  statevector->RestoreState();
  CheckStatesMatch(extendedStabilizer, statevector, "Restored state");

  extendedStabilizer.Reset(kNrQubits);
  statevector->Reset();
  CheckStatesMatch(extendedStabilizer, statevector, "Reset state");
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(qcsim_extended_stabilizer_interface_tests)

BOOST_AUTO_TEST_CASE(factory_configuration_and_unsupported_operations) {
  auto simulator = CreateQCSimInterfaceSimulator(
      Simulators::SimulationType::kExtendedStabilizer);
  BOOST_REQUIRE(simulator);
  BOOST_TEST(static_cast<int>(simulator->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kExtendedStabilizer));
  BOOST_TEST(simulator->GetConfiguration("method") == "extended_stabilizer");

  auto unique = Simulators::SimulatorsFactory::CreateSimulatorUnique(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kExtendedStabilizer);
  BOOST_REQUIRE(unique);
  BOOST_TEST(unique->GetConfiguration("method") == "extended_stabilizer");

  BOOST_CHECK_THROW(simulator->Amplitude(0), std::runtime_error);
  BOOST_CHECK_THROW(simulator->AmplitudeRaw(0), std::runtime_error);
  BOOST_CHECK_THROW(simulator->ProjectOnZero(), std::runtime_error);

  const Eigen::Matrix2cd oneQubit = Eigen::Matrix2cd::Identity();
  const Eigen::Matrix4cd twoQubit = Eigen::Matrix4cd::Identity();
  BOOST_CHECK_THROW(simulator->ApplyGenericOneQubitGate(0, oneQubit),
                    std::runtime_error);
  BOOST_CHECK_THROW(simulator->ApplyGenericTwoQubitGate(0, 1, twoQubit),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(all_gates_and_random_circuits_match_statevector) {
  auto extended = CreateQCSimInterfaceSimulator(
      Simulators::SimulationType::kExtendedStabilizer);
  auto statevector = CreateQCSimInterfaceSimulator(
      Simulators::SimulationType::kStatevector);

  for (unsigned int gate = 0; gate < 30; ++gate) {
    const InterfaceOperation operation = {
        gate, 0, 1, 2, 0.31, -0.47, 0.83, -0.19};
    ApplyInterfaceOperation(*statevector, operation);
    ApplyInterfaceOperation(*extended, operation);
  }
  CheckInterfaceProbabilities(*statevector, *extended);

  const std::array<std::string, 8> paulis = {
      "XIII", "IYZI", "ZZZZ", "XYIZ", "IXXX", "YZYX", "ZIIY", "IXYZ"};
  for (const auto& pauli : paulis)
    BOOST_TEST(std::abs(extended->ExpectationValue(pauli) -
                        statevector->ExpectationValue(pauli)) < 1e-9,
               "Expectation mismatch for " << pauli);

  for (uint64_t seed = 1; seed <= 6; ++seed) {
    extended->Reset();
    statevector->Reset();
    for (const auto& operation : GenerateInterfaceCircuit(seed, 35)) {
      ApplyInterfaceOperation(*statevector, operation);
      ApplyInterfaceOperation(*extended, operation);
    }

    CheckInterfaceProbabilities(*statevector, *extended);
    const Types::qubits_vector outcomes = {0, 3, 7, 12, 15};
    const auto expectedOutcomes = statevector->Probabilities(outcomes);
    const auto actualOutcomes = extended->Probabilities(outcomes);
    BOOST_REQUIRE_EQUAL(actualOutcomes.size(), expectedOutcomes.size());
    for (size_t i = 0; i < expectedOutcomes.size(); ++i)
      BOOST_TEST(std::abs(actualOutcomes[i] - expectedOutcomes[i]) < 1e-9);

    for (const auto& pauli : paulis)
      BOOST_TEST(std::abs(extended->ExpectationValue(pauli) -
                          statevector->ExpectationValue(pauli)) < 1e-9,
                 "Expectation mismatch for " << pauli << " at seed " << seed);
  }
}

BOOST_AUTO_TEST_CASE(sampling_matches_marginals_and_does_not_collapse) {
  auto extended = CreateQCSimInterfaceSimulator(
      Simulators::SimulationType::kExtendedStabilizer);
  extended->ApplyH(0);
  extended->ApplyRy(1, 0.63);
  extended->ApplyCX(0, 2);
  extended->ApplyT(2);
  extended->ApplyCU(1, 3, 0.41, -0.27, 0.52, 0.13);

  const auto before = extended->AllProbabilities();
  const Types::qubits_vector qubits = {2, 0, 3};
  const auto expected = InterfaceMarginals(before, qubits);
  constexpr size_t shots = 8000;

  CheckInterfaceSamples(extended->SampleCounts(qubits, shots), expected, shots);
  const auto manyCounts = extended->SampleCountsMany(qubits, shots);
  std::unordered_map<Types::qubit_t, Types::qubit_t> packedCounts;
  for (const auto& [bits, count] : manyCounts) {
    Types::qubit_t outcome = 0;
    for (size_t i = 0; i < bits.size(); ++i)
      if (bits[i]) outcome |= 1ULL << i;
    packedCounts[outcome] += count;
  }
  CheckInterfaceSamples(packedCounts, expected, shots);

  const auto sampledOutcome = extended->MeasureNoCollapse();
  BOOST_TEST(sampledOutcome < (1ULL << kInterfaceNrQubits));
  const auto sampledBits = extended->MeasureNoCollapseMany();
  BOOST_REQUIRE_EQUAL(sampledBits.size(), kInterfaceNrQubits);

  const auto after = extended->AllProbabilities();
  BOOST_REQUIRE_EQUAL(after.size(), before.size());
  for (size_t outcome = 0; outcome < before.size(); ++outcome)
    BOOST_TEST(std::abs(after[outcome] - before[outcome]) < 1e-9);
}

BOOST_AUTO_TEST_CASE(measurement_reset_snapshots_and_clone) {
  auto extended = CreateQCSimInterfaceSimulator(
      Simulators::SimulationType::kExtendedStabilizer, 2);
  extended->ApplyH(0);
  extended->ApplyCX(0, 1);
  extended->SaveState();

  const auto first = extended->Measure({0});
  const auto second = extended->Measure({1});
  BOOST_TEST(first == second);
  BOOST_CHECK_CLOSE(extended->Probability(first == 0 ? 0 : 3), 1.0, 1e-7);

  extended->RestoreState();
  const auto restored = extended->AllProbabilities();
  BOOST_CHECK_SMALL(restored[0] - 0.5, 1e-9);
  BOOST_CHECK_SMALL(restored[3] - 0.5, 1e-9);

  auto clone = extended->Clone();
  BOOST_REQUIRE(clone);
  BOOST_TEST(static_cast<int>(clone->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kExtendedStabilizer));
  CheckInterfaceProbabilities(*extended, *clone);

  clone->ApplyX(0);
  const auto changedClone = clone->AllProbabilities();
  const auto unchangedOriginal = extended->AllProbabilities();
  BOOST_TEST(!std::equal(changedClone.begin(), changedClone.end(),
                         unchangedOriginal.begin()));

  clone->RestoreState();
  CheckInterfaceProbabilities(*extended, *clone);

  extended->ApplyReset({0});
  BOOST_CHECK_SMALL(extended->ExpectationValue("ZI") - 1.0, 1e-9);
  extended->Reset();
  BOOST_CHECK_CLOSE(extended->Probability(0), 1.0, 1e-7);
}

BOOST_AUTO_TEST_SUITE_END()
