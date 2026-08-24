/**
 * @file qcsimmpotests.cpp
 * @version 1.0
 *
 * Tests for QCSim's matrix product operator simulator through Maestro.
 */

#include <boost/test/unit_test.hpp>

#undef min
#undef max

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Simulators/Factory.h"

namespace {

constexpr size_t kMPONumQubits = 4;
constexpr double kMPOTolerance = 1e-9;

struct MPOOperation {
  unsigned int gate;
  Types::qubit_t qubit0;
  Types::qubit_t qubit1;
  Types::qubit_t qubit2;
  double theta;
  double phi;
  double lambda;
  double gamma;
};

std::shared_ptr<Simulators::ISimulator> MakeQCSimMPOTestSimulator(
    Simulators::SimulationType simulationType,
    size_t numQubits = kMPONumQubits) {
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim, simulationType);
  simulator->AllocateQubits(numQubits);
  simulator->Initialize();
  return simulator;
}

void ApplyMPOOperation(Simulators::ISimulator& simulator,
                       const MPOOperation& operation) {
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
    default: BOOST_FAIL("Unknown MPO test gate " << operation.gate);
  }
}

std::vector<MPOOperation> GenerateMPOCircuit(uint64_t seed,
                                            size_t gateCount) {
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<unsigned int> gateDistribution(0, 29);
  const double pi = std::acos(-1.0);
  std::uniform_real_distribution<double> angleDistribution(-pi, pi);
  std::array<Types::qubit_t, kMPONumQubits> qubits = {0, 1, 2, 3};
  std::vector<MPOOperation> circuit;
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

void CheckMPOProbabilities(Simulators::IState& expected,
                           Simulators::IState& actual,
                           double tolerance = kMPOTolerance) {
  const auto expectedProbabilities = expected.AllProbabilities();
  const auto actualProbabilities = actual.AllProbabilities();
  BOOST_REQUIRE_EQUAL(actualProbabilities.size(),
                      expectedProbabilities.size());
  for (size_t outcome = 0; outcome < expectedProbabilities.size(); ++outcome) {
    BOOST_TEST(std::abs(actualProbabilities[outcome] -
                        expectedProbabilities[outcome]) < tolerance,
               "Probability mismatch for outcome " << outcome);
    BOOST_TEST(std::abs(actual.Probability(outcome) -
                        expectedProbabilities[outcome]) < tolerance);
  }
}

std::vector<double> MPOMarginals(const std::vector<double>& fullProbabilities,
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

void CheckMPOSamples(
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

BOOST_AUTO_TEST_SUITE(qcsim_matrix_product_operator_tests)

BOOST_AUTO_TEST_CASE(factory_configuration_and_unsupported_amplitudes) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator);
  BOOST_REQUIRE(mpo);
  BOOST_TEST(static_cast<int>(mpo->GetSimulationType()) ==
             static_cast<int>(
                 Simulators::SimulationType::kMatrixProductOperator));
  BOOST_TEST(mpo->GetConfiguration("method") == "matrix_product_operator");
  BOOST_CHECK_CLOSE(mpo->Probability(0), 1.0, 1e-10);
  BOOST_CHECK_THROW(mpo->Amplitude(0), std::runtime_error);
  BOOST_CHECK_THROW(mpo->AmplitudeRaw(0), std::runtime_error);
  BOOST_CHECK_THROW(mpo->ProjectOnZero(), std::runtime_error);

  mpo->Configure("matrix_product_operator_max_bond_dimension", "32");
  mpo->Configure("matrix_product_operator_truncation_threshold", "1e-12");
  BOOST_TEST(mpo->GetConfiguration(
                 "matrix_product_operator_max_bond_dimension") == "32");

  auto unique = Simulators::SimulatorsFactory::CreateSimulatorUnique(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductOperator);
  BOOST_REQUIRE(unique);
  BOOST_TEST(unique->GetConfiguration("method") ==
             "matrix_product_operator");
}

BOOST_AUTO_TEST_CASE(all_gates_random_circuits_and_expectations_match) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator);
  auto statevector = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kStatevector);
  const std::array<std::string, 6> paulis = {
      "XIII", "IYZI", "ZZZZ", "XYIZ", "YZYX", "IXYZ"};

  for (unsigned int gate = 0; gate < 30; ++gate) {
    const MPOOperation operation = {
        gate, 0, 1, 2, 0.31, -0.47, 0.83, -0.19};
    ApplyMPOOperation(*statevector, operation);
    ApplyMPOOperation(*mpo, operation);
  }
  CheckMPOProbabilities(*statevector, *mpo);

  for (uint64_t seed = 1; seed <= 5; ++seed) {
    mpo->Reset();
    statevector->Reset();
    for (const auto& operation : GenerateMPOCircuit(seed, 30)) {
      ApplyMPOOperation(*statevector, operation);
      ApplyMPOOperation(*mpo, operation);
    }

    CheckMPOProbabilities(*statevector, *mpo);
    const Types::qubits_vector outcomes = {0, 3, 7, 12, 15};
    const auto expectedOutcomes = statevector->Probabilities(outcomes);
    const auto actualOutcomes = mpo->Probabilities(outcomes);
    BOOST_REQUIRE_EQUAL(actualOutcomes.size(), expectedOutcomes.size());
    for (size_t i = 0; i < expectedOutcomes.size(); ++i)
      BOOST_TEST(std::abs(actualOutcomes[i] - expectedOutcomes[i]) <
                 kMPOTolerance);

    for (const auto& pauli : paulis)
      BOOST_TEST(std::abs(mpo->ExpectationValue(pauli) -
                          statevector->ExpectationValue(pauli)) <
                     kMPOTolerance,
                 "Expectation mismatch for " << pauli << " at seed " << seed);
  }
}

BOOST_AUTO_TEST_CASE(generic_gates_match_statevector) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator);
  auto statevector = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kStatevector);

  const double theta = 0.731;
  Eigen::Matrix2cd oneQubit;
  oneQubit << std::cos(theta), -std::sin(theta), std::sin(theta),
      std::cos(theta);

  Eigen::Matrix4cd twoQubit = Eigen::Matrix4cd::Zero();
  const std::complex<double> imaginary(0.0, 1.0);
  twoQubit(0, 0) = 1.0;
  twoQubit(1, 2) = imaginary;
  twoQubit(2, 1) = imaginary;
  twoQubit(3, 3) = 1.0;

  statevector->ApplyH(0);
  mpo->ApplyH(0);
  statevector->ApplyGenericOneQubitGate(2, oneQubit);
  mpo->ApplyGenericOneQubitGate(2, oneQubit);
  statevector->ApplyGenericTwoQubitGate(0, 2, twoQubit);
  mpo->ApplyGenericTwoQubitGate(0, 2, twoQubit);
  CheckMPOProbabilities(*statevector, *mpo);
}

BOOST_AUTO_TEST_CASE(sampling_matches_marginals_without_collapsing) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator);
  mpo->ApplyH(0);
  mpo->ApplyRy(1, 0.7);
  mpo->ApplyCX(0, 2);
  mpo->ApplyRx(3, -0.4);

  const auto before = mpo->AllProbabilities();
  const Types::qubits_vector qubits = {2, 0, 3};
  const auto expected = MPOMarginals(before, qubits);
  constexpr size_t shots = 8000;
  CheckMPOSamples(mpo->SampleCounts(qubits, shots), expected, shots);

  const auto many = mpo->SampleCountsMany(qubits, shots);
  std::unordered_map<Types::qubit_t, Types::qubit_t> packed;
  for (const auto& [bits, count] : many) {
    Types::qubit_t outcome = 0;
    for (size_t i = 0; i < bits.size(); ++i)
      if (bits[i]) outcome |= 1ULL << i;
    packed[outcome] += count;
  }
  CheckMPOSamples(packed, expected, shots);

  BOOST_TEST(mpo->MeasureNoCollapse() < (1ULL << kMPONumQubits));
  BOOST_REQUIRE_EQUAL(mpo->MeasureNoCollapseMany().size(), kMPONumQubits);
  const auto after = mpo->AllProbabilities();
  BOOST_REQUIRE_EQUAL(after.size(), before.size());
  for (size_t outcome = 0; outcome < before.size(); ++outcome)
    BOOST_TEST(std::abs(after[outcome] - before[outcome]) < kMPOTolerance);
}

BOOST_AUTO_TEST_CASE(measurement_reset_snapshots_and_clone) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator, 2);
  mpo->ApplyH(0);
  mpo->ApplyCX(0, 1);
  BOOST_TEST(mpo->GetCurrentMaxBondDimension() > 1);
  mpo->SaveState();

  const auto measured = mpo->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(measured.size(), 2);
  BOOST_TEST(measured[0] == measured[1]);
  BOOST_CHECK_CLOSE(mpo->Probability(measured[0] ? 3 : 0), 1.0, 1e-7);

  mpo->RestoreState();
  BOOST_CHECK_SMALL(mpo->Probability(0) - 0.5, kMPOTolerance);
  BOOST_CHECK_SMALL(mpo->Probability(3) - 0.5, kMPOTolerance);

  auto clone = mpo->Clone();
  BOOST_REQUIRE(clone);
  BOOST_TEST(static_cast<int>(clone->GetSimulationType()) ==
             static_cast<int>(
                 Simulators::SimulationType::kMatrixProductOperator));
  BOOST_TEST(clone->GetConfiguration("method") == "matrix_product_operator");
  CheckMPOProbabilities(*mpo, *clone);

  clone->ApplyX(0);
  const auto changedClone = clone->AllProbabilities();
  const auto unchangedOriginal = mpo->AllProbabilities();
  BOOST_TEST(!std::equal(changedClone.begin(), changedClone.end(),
                         unchangedOriginal.begin()));
  clone->RestoreState();
  CheckMPOProbabilities(*mpo, *clone);

  auto densityMatrix = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kDensityMatrix, 2);
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyCX(0, 1);
  densityMatrix->ApplyReset({0});
  mpo->ApplyReset({0});
  CheckMPOProbabilities(*densityMatrix, *mpo);
  const auto resetProbabilities = mpo->AllProbabilities();
  BOOST_CHECK_SMALL(resetProbabilities[0] - 0.5, kMPOTolerance);
  BOOST_CHECK_SMALL(resetProbabilities[1], kMPOTolerance);
  BOOST_CHECK_SMALL(resetProbabilities[2] - 0.5, kMPOTolerance);
  BOOST_CHECK_SMALL(resetProbabilities[3], kMPOTolerance);

  mpo->Reset();
  BOOST_CHECK_CLOSE(mpo->Probability(0), 1.0, 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
