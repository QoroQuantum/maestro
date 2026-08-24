/**
 * @file densitymatrixtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the Qiskit Aer density matrix simulator.
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
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Simulators/Factory.h"

namespace {

constexpr size_t kNumQubits = 4;
constexpr double kProbabilityTolerance = 1e-10;

struct Operation {
  unsigned int gate;
  Types::qubit_t qubit0;
  Types::qubit_t qubit1;
  Types::qubit_t qubit2;
  double theta;
  double phi;
  double lambda;
  double gamma;
};

std::shared_ptr<Simulators::ISimulator> MakeSimulator(
    Simulators::SimulatorType simulatorType,
    Simulators::SimulationType simulationType,
    size_t numQubits = kNumQubits) {
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      simulatorType, simulationType);
  simulator->AllocateQubits(numQubits);
  simulator->Initialize();
  return simulator;
}

void ApplyOperation(Simulators::ISimulator& simulator,
                    const Operation& operation) {
  switch (operation.gate) {
    case 0:
      simulator.ApplyX(operation.qubit0);
      break;
    case 1:
      simulator.ApplyY(operation.qubit0);
      break;
    case 2:
      simulator.ApplyZ(operation.qubit0);
      break;
    case 3:
      simulator.ApplyH(operation.qubit0);
      break;
    case 4:
      simulator.ApplyS(operation.qubit0);
      break;
    case 5:
      simulator.ApplySDG(operation.qubit0);
      break;
    case 6:
      simulator.ApplyT(operation.qubit0);
      break;
    case 7:
      simulator.ApplyTDG(operation.qubit0);
      break;
    case 8:
      simulator.ApplySx(operation.qubit0);
      break;
    case 9:
      simulator.ApplySxDAG(operation.qubit0);
      break;
    case 10:
      simulator.ApplyK(operation.qubit0);
      break;
    case 11:
      simulator.ApplyP(operation.qubit0, operation.theta);
      break;
    case 12:
      simulator.ApplyRx(operation.qubit0, operation.theta);
      break;
    case 13:
      simulator.ApplyRy(operation.qubit0, operation.theta);
      break;
    case 14:
      simulator.ApplyRz(operation.qubit0, operation.theta);
      break;
    case 15:
      simulator.ApplyU(operation.qubit0, operation.theta, operation.phi,
                       operation.lambda, operation.gamma);
      break;
    case 16:
      simulator.ApplyCX(operation.qubit0, operation.qubit1);
      break;
    case 17:
      simulator.ApplyCY(operation.qubit0, operation.qubit1);
      break;
    case 18:
      simulator.ApplyCZ(operation.qubit0, operation.qubit1);
      break;
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
    case 23:
      simulator.ApplyCH(operation.qubit0, operation.qubit1);
      break;
    case 24:
      simulator.ApplyCSx(operation.qubit0, operation.qubit1);
      break;
    case 25:
      simulator.ApplyCSxDAG(operation.qubit0, operation.qubit1);
      break;
    case 26:
      simulator.ApplySwap(operation.qubit0, operation.qubit1);
      break;
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
    default:
      BOOST_FAIL("Unknown random gate " << operation.gate);
  }
}

std::vector<Operation> GenerateCircuit(uint64_t seed, size_t gateCount) {
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<unsigned int> gateDistribution(0, 29);
  const double pi = std::acos(-1.0);
  std::uniform_real_distribution<double> angleDistribution(-2.0 * pi,
                                                            2.0 * pi);

  std::array<Types::qubit_t, kNumQubits> qubits = {0, 1, 2, 3};
  std::vector<Operation> circuit;
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

void CheckProbabilities(Simulators::IState& expectedState,
                        Simulators::IState& actualState,
                        double tolerance = kProbabilityTolerance) {
  const auto expected = expectedState.AllProbabilities();
  const auto actual = actualState.AllProbabilities();

  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t outcome = 0; outcome < expected.size(); ++outcome) {
    BOOST_TEST(std::abs(actual[outcome] - expected[outcome]) < tolerance,
               "Probability mismatch for outcome "
                   << outcome << ": expected " << expected[outcome]
                   << ", got " << actual[outcome]);
    BOOST_TEST(std::abs(actualState.Probability(outcome) - expected[outcome]) <
                   tolerance,
               "Single-outcome probability mismatch for outcome " << outcome);
  }
}

std::vector<double> MarginalProbabilities(
    const std::vector<double>& fullProbabilities,
    const Types::qubits_vector& qubits) {
  std::vector<double> probabilities(1ULL << qubits.size(), 0.0);
  for (size_t basisState = 0; basisState < fullProbabilities.size();
       ++basisState) {
    size_t outcome = 0;
    for (size_t index = 0; index < qubits.size(); ++index)
      outcome |= ((basisState >> qubits[index]) & 1ULL) << index;
    probabilities[outcome] += fullProbabilities[basisState];
  }
  return probabilities;
}

void CheckSampledDistribution(
    const std::unordered_map<Types::qubit_t, Types::qubit_t>& counts,
    const std::vector<double>& expected, size_t shots,
    double tolerance = 0.035) {
  for (size_t outcome = 0; outcome < expected.size(); ++outcome) {
    const auto found = counts.find(outcome);
    const double sampled =
        static_cast<double>(found == counts.end() ? 0 : found->second) / shots;
    BOOST_TEST(std::abs(sampled - expected[outcome]) < tolerance,
               "Sampling mismatch for outcome "
                   << outcome << ": expected " << expected[outcome]
                   << ", got " << sampled);
  }
}

}  // namespace

struct DensityMatrixTestFixture {
  DensityMatrixTestFixture()
      : densityMatrix(MakeSimulator(
            Simulators::SimulatorType::kQiskitAer,
            Simulators::SimulationType::kDensityMatrix)),
        statevector(MakeSimulator(Simulators::SimulatorType::kQCSim,
                                  Simulators::SimulationType::kStatevector)) {}

  std::shared_ptr<Simulators::ISimulator> densityMatrix;
  std::shared_ptr<Simulators::ISimulator> statevector;
};

BOOST_AUTO_TEST_SUITE(aer_density_matrix_tests)

BOOST_FIXTURE_TEST_CASE(FactoryConfigurationAndUnsupportedAmplitudes,
                        DensityMatrixTestFixture) {
  BOOST_REQUIRE(densityMatrix);
  BOOST_TEST(static_cast<int>(densityMatrix->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kDensityMatrix));
  BOOST_TEST(densityMatrix->GetConfiguration("method") == "density_matrix");
  BOOST_CHECK_CLOSE(densityMatrix->Probability(0), 1.0, 1e-10);
  BOOST_CHECK_THROW(densityMatrix->Amplitude(0), std::runtime_error);
  BOOST_CHECK_THROW(densityMatrix->ProjectOnZero(), std::runtime_error);
}

BOOST_FIXTURE_TEST_CASE(RandomUnitaryCircuitsMatchStatevector,
                        DensityMatrixTestFixture) {
  const std::array<std::string, 8> paulis = {
      "XIII", "IYZI", "ZZZZ", "XYIZ", "IXXX", "YZYX", "ZIIY", "IXYZ"};

  for (uint64_t seed = 1; seed <= 10; ++seed) {
    if (seed != 1) {
      densityMatrix->Reset();
      statevector->Reset();
    }

    for (const auto& operation : GenerateCircuit(seed, 40)) {
      ApplyOperation(*statevector, operation);
      ApplyOperation(*densityMatrix, operation);
    }

    CheckProbabilities(*statevector, *densityMatrix);

    const Types::qubits_vector outcomes = {3, 1};
    const auto expectedOutcomes = statevector->Probabilities(outcomes);
    const auto actualOutcomes = densityMatrix->Probabilities(outcomes);
    BOOST_REQUIRE_EQUAL(actualOutcomes.size(), expectedOutcomes.size());
    for (size_t outcome = 0; outcome < expectedOutcomes.size(); ++outcome)
      BOOST_TEST(std::abs(actualOutcomes[outcome] - expectedOutcomes[outcome]) <
                 kProbabilityTolerance);

    for (const auto& pauli : paulis)
      BOOST_TEST(std::abs(densityMatrix->ExpectationValue(pauli) -
                          statevector->ExpectationValue(pauli)) <
                 kProbabilityTolerance,
                 "Expectation value mismatch for " << pauli);
  }
}

BOOST_FIXTURE_TEST_CASE(GenericOperationsMatchStatevector,
                        DensityMatrixTestFixture) {
  const double theta = 0.731;
  Eigen::Matrix2cd singleQubit;
  singleQubit << std::cos(theta), -std::sin(theta), std::sin(theta),
      std::cos(theta);

  const std::complex<double> imaginary(0.0, 1.0);
  Eigen::Matrix4cd twoQubit = Eigen::Matrix4cd::Zero();
  twoQubit(0, 0) = 1.0;
  twoQubit(1, 2) = imaginary;
  twoQubit(2, 1) = imaginary;
  twoQubit(3, 3) = 1.0;

  statevector->ApplyH(0);
  densityMatrix->ApplyH(0);
  statevector->ApplyGenericOneQubitGate(2, singleQubit);
  densityMatrix->ApplyGenericOneQubitGate(2, singleQubit);
  statevector->ApplyGenericTwoQubitGate(0, 2, twoQubit);
  densityMatrix->ApplyGenericTwoQubitGate(0, 2, twoQubit);
  statevector->ApplyNop();
  densityMatrix->ApplyNop();

  CheckProbabilities(*statevector, *densityMatrix);
}

BOOST_AUTO_TEST_CASE(PureStateInitializationCreatesMatchingDensityMatrix) {
  auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix);
  auto statevector = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);

  std::vector<std::complex<double>> amplitudes = {
      {0.2, 0.1}, {-0.3, 0.4}, {0.1, -0.5}, {0.6, 0.2}};
  double norm = 0.0;
  for (const auto amplitude : amplitudes) norm += std::norm(amplitude);
  for (auto& amplitude : amplitudes) amplitude /= std::sqrt(norm);

  auto densityAmplitudes = amplitudes;
  densityMatrix->InitializeState(2, densityAmplitudes);
  statevector->InitializeState(2, amplitudes);

  CheckProbabilities(*statevector, *densityMatrix);
  for (const auto& pauli : {"XI", "YX", "ZZ", "IY"})
    BOOST_TEST(std::abs(densityMatrix->ExpectationValue(pauli) -
                        statevector->ExpectationValue(pauli)) <
               kProbabilityTolerance);
}

BOOST_FIXTURE_TEST_CASE(SamplingMatchesExactMarginalsAndDoesNotCollapse,
                        DensityMatrixTestFixture) {
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyRy(1, 0.7);
  densityMatrix->ApplyCX(0, 2);
  densityMatrix->ApplyRx(3, -0.4);

  const auto fullProbabilities = densityMatrix->AllProbabilities();
  const Types::qubits_vector qubits = {2, 0, 3};
  const auto expected = MarginalProbabilities(fullProbabilities, qubits);
  constexpr size_t shots = 20000;

  const auto counts = densityMatrix->SampleCounts(qubits, shots);
  CheckSampledDistribution(counts, expected, shots);

  const auto manyCounts = densityMatrix->SampleCountsMany(qubits, shots);
  std::unordered_map<Types::qubit_t, Types::qubit_t> packedCounts;
  for (const auto& [bits, count] : manyCounts) {
    Types::qubit_t outcome = 0;
    for (size_t bit = 0; bit < bits.size(); ++bit)
      if (bits[bit]) outcome |= 1ULL << bit;
    packedCounts[outcome] += count;
  }
  CheckSampledDistribution(packedCounts, expected, shots);

  const auto probabilitiesAfterSampling = densityMatrix->AllProbabilities();
  BOOST_REQUIRE_EQUAL(probabilitiesAfterSampling.size(),
                      fullProbabilities.size());
  for (size_t outcome = 0; outcome < fullProbabilities.size(); ++outcome)
    BOOST_TEST(std::abs(probabilitiesAfterSampling[outcome] -
                        fullProbabilities[outcome]) < kProbabilityTolerance);
}

BOOST_AUTO_TEST_CASE(MeasurementsCollapseCorrelatedQubits) {
  auto densityMatrix = MakeSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix, 2);
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyCX(0, 1);

  const auto first = densityMatrix->Measure({0});
  const auto second = densityMatrix->Measure({1});
  BOOST_TEST(first == second);
  BOOST_CHECK_CLOSE(densityMatrix->Probability(first == 0 ? 0 : 3), 1.0,
                    1e-8);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyCX(0, 1);
  const auto many = densityMatrix->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(many.size(), 2);
  BOOST_TEST(many[0] == many[1]);
}

BOOST_AUTO_TEST_CASE(SaveRestoreSupportsRepeatedMeasurements) {
  auto densityMatrix = MakeSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix, 2);
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyRy(1, 0.8);

  const auto exact = densityMatrix->AllProbabilities();
  densityMatrix->SaveState();

  constexpr size_t shots = 2000;
  std::unordered_map<Types::qubit_t, Types::qubit_t> counts;
  for (size_t shot = 0; shot < shots; ++shot) {
    ++counts[densityMatrix->Measure({0, 1})];
    densityMatrix->RestoreState();
  }

  CheckSampledDistribution(counts, exact, shots, 0.055);
  const auto restored = densityMatrix->AllProbabilities();
  for (size_t outcome = 0; outcome < exact.size(); ++outcome)
    BOOST_TEST(std::abs(restored[outcome] - exact[outcome]) <
               kProbabilityTolerance);
}

BOOST_AUTO_TEST_CASE(ResetQubitCanProduceAMixedState) {
  auto densityMatrix = MakeSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix, 2);
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyCX(0, 1);
  densityMatrix->ApplyReset({0});

  const auto probabilities = densityMatrix->AllProbabilities();
  BOOST_REQUIRE_EQUAL(probabilities.size(), 4);
  BOOST_CHECK_SMALL(probabilities[0] - 0.5, kProbabilityTolerance);
  BOOST_CHECK_SMALL(probabilities[1], kProbabilityTolerance);
  BOOST_CHECK_SMALL(probabilities[2] - 0.5, kProbabilityTolerance);
  BOOST_CHECK_SMALL(probabilities[3], kProbabilityTolerance);
}

BOOST_FIXTURE_TEST_CASE(ClonePreservesDensityMatrixAndIsIndependent,
                        DensityMatrixTestFixture) {
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyRy(1, 0.42);
  densityMatrix->ApplyCX(0, 3);
  densityMatrix->ApplyReset({2});

  const auto originalProbabilities = densityMatrix->AllProbabilities();
  auto clone = densityMatrix->Clone();
  BOOST_REQUIRE(clone);
  BOOST_TEST(static_cast<int>(clone->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kDensityMatrix));

  const auto cloneProbabilities = clone->AllProbabilities();
  BOOST_REQUIRE_EQUAL(cloneProbabilities.size(), originalProbabilities.size());
  for (size_t outcome = 0; outcome < originalProbabilities.size(); ++outcome)
    BOOST_TEST(std::abs(cloneProbabilities[outcome] -
                        originalProbabilities[outcome]) <
               kProbabilityTolerance);

  clone->ApplyX(1);
  const auto modifiedClone = clone->AllProbabilities();
  const auto unchangedOriginal = densityMatrix->AllProbabilities();
  BOOST_TEST(!std::equal(modifiedClone.begin(), modifiedClone.end(),
                         unchangedOriginal.begin()));
  BOOST_TEST(std::equal(unchangedOriginal.begin(), unchangedOriginal.end(),
                        originalProbabilities.begin()));
}

BOOST_AUTO_TEST_SUITE_END()
