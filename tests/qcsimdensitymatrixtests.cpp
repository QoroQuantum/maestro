/**
 * @file qcsimdensitymatrixtests.cpp
 * @version 1.0
 *
 * Tests for QCSim's density matrix simulator through Maestro.
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

#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Simulators/Factory.h"
#include "../python/noise.h"

namespace {

constexpr size_t kQCSimDensityNumQubits = 4;
constexpr double kQCSimDensityTolerance = 1e-10;

struct QCSimDensityOperation {
  unsigned int gate;
  Types::qubit_t qubit0;
  Types::qubit_t qubit1;
  Types::qubit_t qubit2;
  double theta;
  double phi;
  double lambda;
  double gamma;
};

std::shared_ptr<Simulators::ISimulator> MakeQCSim(
    Simulators::SimulationType simulationType,
    size_t numQubits = kQCSimDensityNumQubits) {
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim, simulationType);
  simulator->AllocateQubits(numQubits);
  simulator->Initialize();
  return simulator;
}

void ApplyQCSimDensityOperation(Simulators::ISimulator& simulator,
                                const QCSimDensityOperation& operation) {
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
    default: BOOST_FAIL("Unknown random gate " << operation.gate);
  }
}

std::vector<QCSimDensityOperation> GenerateQCSimDensityCircuit(
    uint64_t seed, size_t gateCount) {
  std::mt19937_64 generator(seed);
  std::uniform_int_distribution<unsigned int> gateDistribution(0, 29);
  const double pi = std::acos(-1.0);
  std::uniform_real_distribution<double> angleDistribution(-2.0 * pi,
                                                            2.0 * pi);
  std::array<Types::qubit_t, kQCSimDensityNumQubits> qubits = {0, 1, 2, 3};
  std::vector<QCSimDensityOperation> circuit;
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

void CheckQCSimDensityProbabilities(Simulators::IState& expectedState,
                                    Simulators::IState& actualState) {
  const auto expected = expectedState.AllProbabilities();
  const auto actual = actualState.AllProbabilities();
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t outcome = 0; outcome < expected.size(); ++outcome) {
    BOOST_TEST(std::abs(actual[outcome] - expected[outcome]) <
               kQCSimDensityTolerance,
               "Probability mismatch for outcome " << outcome);
    BOOST_TEST(std::abs(actualState.Probability(outcome) - expected[outcome]) <
               kQCSimDensityTolerance);
  }
}

std::vector<double> QCSimDensityMarginals(
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

void CheckQCSimDensitySamples(
    const std::unordered_map<Types::qubit_t, Types::qubit_t>& counts,
    const std::vector<double>& expected, size_t shots,
    double tolerance = 0.04) {
  for (size_t outcome = 0; outcome < expected.size(); ++outcome) {
    const auto found = counts.find(outcome);
    const double sampled = static_cast<double>(
        found == counts.end() ? 0 : found->second) / shots;
    BOOST_TEST(std::abs(sampled - expected[outcome]) < tolerance,
               "Sampling mismatch for outcome " << outcome);
  }
}

}  // namespace

struct QCSimDensityMatrixFixture {
  QCSimDensityMatrixFixture()
      : densityMatrix(MakeQCSim(Simulators::SimulationType::kDensityMatrix)),
        statevector(MakeQCSim(Simulators::SimulationType::kStatevector)) {}

  std::shared_ptr<Simulators::ISimulator> densityMatrix;
  std::shared_ptr<Simulators::ISimulator> statevector;
};

BOOST_AUTO_TEST_SUITE(qcsim_density_matrix_tests)

BOOST_FIXTURE_TEST_CASE(FactoryConfigurationAndUnsupportedAmplitudes,
                        QCSimDensityMatrixFixture) {
  BOOST_REQUIRE(densityMatrix);
  BOOST_TEST(static_cast<int>(densityMatrix->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kDensityMatrix));
  BOOST_TEST(densityMatrix->GetConfiguration("method") == "density_matrix");
  BOOST_CHECK_CLOSE(densityMatrix->Probability(0), 1.0, 1e-10);
  BOOST_CHECK_THROW(densityMatrix->Amplitude(0), std::runtime_error);
  BOOST_CHECK_THROW(densityMatrix->AmplitudeRaw(0), std::runtime_error);
  BOOST_CHECK_THROW(densityMatrix->ProjectOnZero(), std::runtime_error);

  auto unique = Simulators::SimulatorsFactory::CreateSimulatorUnique(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);
  BOOST_REQUIRE(unique);
  BOOST_TEST(unique->GetConfiguration("method") == "density_matrix");
}

BOOST_FIXTURE_TEST_CASE(RandomCircuitsAndExpectationsMatchStatevector,
                        QCSimDensityMatrixFixture) {
  const std::array<std::string, 6> paulis = {
      "XIII", "IYZI", "ZZZZ", "XYIZ", "YZYX", "IXYZ"};

  // Exercise every exposed gate family at least once before the randomized
  // circuits add varied qubits and parameters.
  for (unsigned int gate = 0; gate < 30; ++gate) {
    const QCSimDensityOperation operation = {
        gate, 0, 1, 2, 0.31, -0.47, 0.83, -0.19};
    ApplyQCSimDensityOperation(*statevector, operation);
    ApplyQCSimDensityOperation(*densityMatrix, operation);
  }
  CheckQCSimDensityProbabilities(*statevector, *densityMatrix);

  for (uint64_t seed = 1; seed <= 8; ++seed) {
    if (seed != 1) {
      densityMatrix->Reset();
      statevector->Reset();
    }
    for (const auto& operation : GenerateQCSimDensityCircuit(seed, 40)) {
      ApplyQCSimDensityOperation(*statevector, operation);
      ApplyQCSimDensityOperation(*densityMatrix, operation);
    }

    CheckQCSimDensityProbabilities(*statevector, *densityMatrix);
    const Types::qubits_vector outcomes = {3, 1};
    const auto expectedOutcomes = statevector->Probabilities(outcomes);
    const auto actualOutcomes = densityMatrix->Probabilities(outcomes);
    BOOST_REQUIRE_EQUAL(actualOutcomes.size(), expectedOutcomes.size());
    for (size_t i = 0; i < expectedOutcomes.size(); ++i)
      BOOST_TEST(std::abs(actualOutcomes[i] - expectedOutcomes[i]) <
                 kQCSimDensityTolerance);

    for (const auto& pauli : paulis)
      BOOST_TEST(std::abs(densityMatrix->ExpectationValue(pauli) -
                          statevector->ExpectationValue(pauli)) <
                 kQCSimDensityTolerance,
                 "Expectation value mismatch for " << pauli);
  }
}

BOOST_FIXTURE_TEST_CASE(GenericGatesMatchStatevector,
                        QCSimDensityMatrixFixture) {
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
  densityMatrix->ApplyH(0);
  statevector->ApplyGenericOneQubitGate(2, oneQubit);
  densityMatrix->ApplyGenericOneQubitGate(2, oneQubit);
  statevector->ApplyGenericTwoQubitGate(0, 2, twoQubit);
  densityMatrix->ApplyGenericTwoQubitGate(0, 2, twoQubit);
  statevector->ApplyNop();
  densityMatrix->ApplyNop();
  CheckQCSimDensityProbabilities(*statevector, *densityMatrix);
}

BOOST_AUTO_TEST_CASE(PureStateInitializationMatchesStatevector) {
  auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);
  auto statevector = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  std::vector<std::complex<double>> amplitudes = {
      {0.2, 0.1}, {-0.3, 0.4}, {0.1, -0.5}, {0.6, 0.2}};
  double norm = 0.0;
  for (const auto& amplitude : amplitudes) norm += std::norm(amplitude);
  for (auto& amplitude : amplitudes) amplitude /= std::sqrt(norm);

  auto densityAmplitudes = amplitudes;
  densityMatrix->InitializeState(2, densityAmplitudes);
  statevector->InitializeState(2, amplitudes);
  CheckQCSimDensityProbabilities(*statevector, *densityMatrix);
  BOOST_TEST(std::abs(densityMatrix->ExpectationValue("X") -
                      statevector->ExpectationValue("X")) <
             kQCSimDensityTolerance);
}

BOOST_FIXTURE_TEST_CASE(SamplingMatchesMarginalsWithoutCollapsing,
                        QCSimDensityMatrixFixture) {
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyRy(1, 0.7);
  densityMatrix->ApplyCX(0, 2);
  densityMatrix->ApplyRx(3, -0.4);
  const auto before = densityMatrix->AllProbabilities();
  const Types::qubits_vector qubits = {2, 0, 3};
  const auto expected = QCSimDensityMarginals(before, qubits);
  constexpr size_t shots = 12000;

  CheckQCSimDensitySamples(densityMatrix->SampleCounts(qubits, shots),
                           expected, shots);
  const auto many = densityMatrix->SampleCountsMany(qubits, shots);
  std::unordered_map<Types::qubit_t, Types::qubit_t> packed;
  for (const auto& [bits, count] : many) {
    Types::qubit_t outcome = 0;
    for (size_t i = 0; i < bits.size(); ++i)
      if (bits[i]) outcome |= 1ULL << i;
    packed[outcome] += count;
  }
  CheckQCSimDensitySamples(packed, expected, shots);

  const auto after = densityMatrix->AllProbabilities();
  BOOST_REQUIRE_EQUAL(after.size(), before.size());
  for (size_t outcome = 0; outcome < before.size(); ++outcome)
    BOOST_TEST(std::abs(after[outcome] - before[outcome]) <
               kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(MeasurementCollapseSaveAndRestore) {
  auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyCX(0, 1);
  densityMatrix->SaveState();

  const auto first = densityMatrix->Measure({0});
  const auto second = densityMatrix->Measure({1});
  BOOST_TEST(first == second);
  BOOST_CHECK_CLOSE(densityMatrix->Probability(first == 0 ? 0 : 3), 1.0,
                    1e-8);

  densityMatrix->RestoreState();
  const auto restored = densityMatrix->AllProbabilities();
  BOOST_CHECK_SMALL(restored[0] - 0.5, kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(restored[3] - 0.5, kQCSimDensityTolerance);
  const auto many = densityMatrix->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(many.size(), 2);
  BOOST_TEST(many[0] == many[1]);
}

BOOST_AUTO_TEST_CASE(ResetEntangledQubitProducesMixedState) {
  auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyCX(0, 1);
  densityMatrix->ApplyReset({0});
  const auto probabilities = densityMatrix->AllProbabilities();
  BOOST_REQUIRE_EQUAL(probabilities.size(), 4);
  BOOST_CHECK_SMALL(probabilities[0] - 0.5, kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(probabilities[1], kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(probabilities[2] - 0.5, kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(probabilities[3], kQCSimDensityTolerance);
}

BOOST_FIXTURE_TEST_CASE(ClonePreservesStateAndIsIndependent,
                        QCSimDensityMatrixFixture) {
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyRy(1, 0.42);
  densityMatrix->ApplyCX(0, 3);
  densityMatrix->ApplyReset({2});
  densityMatrix->SaveState();
  const auto original = densityMatrix->AllProbabilities();
  auto clone = densityMatrix->Clone();
  BOOST_REQUIRE(clone);
  BOOST_TEST(static_cast<int>(clone->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kDensityMatrix));
  BOOST_TEST(clone->GetConfiguration("method") == "density_matrix");

  const auto cloned = clone->AllProbabilities();
  BOOST_REQUIRE_EQUAL(cloned.size(), original.size());
  for (size_t outcome = 0; outcome < original.size(); ++outcome)
    BOOST_TEST(std::abs(cloned[outcome] - original[outcome]) <
               kQCSimDensityTolerance);

  clone->ApplyX(1);
  const auto modifiedClone = clone->AllProbabilities();
  BOOST_TEST(!std::equal(modifiedClone.begin(), modifiedClone.end(),
                         original.begin()));
  const auto unchangedOriginal = densityMatrix->AllProbabilities();
  BOOST_TEST(std::equal(unchangedOriginal.begin(), unchangedOriginal.end(),
                        original.begin()));
  clone->RestoreState();
  CheckQCSimDensityProbabilities(*densityMatrix, *clone);
}

BOOST_AUTO_TEST_CASE(ExactSingleQubitNoiseChannelsAndConventions) {
  auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix,
                                 1);
  BOOST_TEST(densityMatrix->SupportsQuantumChannels());

  densityMatrix->ApplyH(0);
  densityMatrix->ApplyPhaseFlipNoise(0, 0.2);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.6,
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyPhaseDamping(0, 0.36);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.8,
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyPhaseDampingFromTime(0, 0.4, 0.8);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - std::exp(-0.5),
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyX(0);
  densityMatrix->ApplyAmplitudeDamping(0, 0.25);
  BOOST_CHECK_SMALL(densityMatrix->Probability(0) - 0.25,
                    kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.75,
                    kQCSimDensityTolerance);

  // QCSim/noise.h use total nonidentity-Pauli probability. Consequently
  // p=3/4, rather than p=1, is the fully mixed single-qubit channel.
  densityMatrix->Reset();
  densityMatrix->ApplyDepolarizingNoise(0, 0.75);
  BOOST_CHECK_SMALL(densityMatrix->Probability(0) - 0.5,
                    kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.5,
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyDepolarizingMixingNoise(0, 1.0);
  BOOST_CHECK_SMALL(densityMatrix->Probability(0) - 0.5,
                    kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.5,
                    kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(AdvancedAndCorrelatedNoiseChannels) {
  auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix,
                                 2);

  densityMatrix->ApplyGeneralizedAmplitudeDamping(0, 0.4, 0.25);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.1,
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyX(0);
  constexpr double duration = 0.7;
  constexpr double t1 = 2.0;
  constexpr double t2 = 1.5;
  constexpr double excitedPopulation = 0.1;
  densityMatrix->ApplyThermalRelaxation(
      0, duration, t1, t2, excitedPopulation);
  const double expectedExcited =
      excitedPopulation + (1.0 - excitedPopulation) *
                                  std::exp(-duration / t1);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - expectedExcited,
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyThermalRelaxation(
      0, duration, t1, t2, excitedPopulation);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XI") -
                        std::exp(-duration / t2),
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyH(1);
  densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 1, 0.5);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XI"),
                    kQCSimDensityTolerance);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XX") - 1.0,
                    kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyH(1);
  densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 1, 0.25, 0.4);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XX") - 0.55,
                    kQCSimDensityTolerance);

  // p=15/16 makes all sixteen two-qubit Paulis equiprobable and hence sends
  // every input to I/4 in the total-Pauli-error convention.
  densityMatrix->Reset();
  densityMatrix->ApplyTwoQubitDepolarizingNoise(0, 1, 15.0 / 16.0);
  const auto probabilities = densityMatrix->AllProbabilities();
  for (const double probability : probabilities)
    BOOST_CHECK_SMALL(probability - 0.25, kQCSimDensityTolerance);

  densityMatrix->Reset();
  densityMatrix->ApplyTwoQubitDepolarizingMixingNoise(0, 1, 1.0);
  for (const double probability : densityMatrix->AllProbabilities())
    BOOST_CHECK_SMALL(probability - 0.25, kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(ArbitraryPauliAndKrausValidation) {
  auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix,
                                 2);

  // Base-4 digit zero selects the Pauli on targets[0].
  std::vector<double> pauliProbabilities(16, 0.0);
  pauliProbabilities[1] = 1.0;  // XI in target-list order: X on qubit 0.
  densityMatrix->ApplyPauliChannel({0, 1}, pauliProbabilities);
  BOOST_CHECK_CLOSE(densityMatrix->Probability(1), 1.0, 1e-10);

  densityMatrix->Reset();
  pauliProbabilities[1] = 0.0;
  pauliProbabilities[4] = 1.0;  // IX: X on qubit 1.
  densityMatrix->ApplyPauliChannel({0, 1}, pauliProbabilities);
  BOOST_CHECK_CLOSE(densityMatrix->Probability(2), 1.0, 1e-10);

  Eigen::MatrixXcd nonTracePreserving =
      0.5 * Eigen::MatrixXcd::Identity(2, 2);
  BOOST_CHECK_THROW(
      densityMatrix->ApplyKrausChannel({0}, {nonTracePreserving}),
      std::invalid_argument);
  BOOST_CHECK_THROW(densityMatrix->ApplyBitFlipNoise(0, -0.1),
                    std::invalid_argument);
  BOOST_CHECK_THROW(densityMatrix->ApplyPauliChannel(0, 0.6, 0.5, 0.0),
                    std::invalid_argument);
  BOOST_CHECK_THROW(
      densityMatrix->ApplyQuantumChannel(
          {0, 1}, Simulators::QuantumChannel::BitFlip(0.1)),
      std::invalid_argument);
  BOOST_CHECK_THROW(densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 0, 0.1),
                    std::invalid_argument);
  BOOST_CHECK_THROW(
      densityMatrix->ApplyThermalRelaxation(0, 1.0, 1.0, 2.1),
      std::invalid_argument);

  auto statevector = MakeQCSim(Simulators::SimulationType::kStatevector, 1);
  BOOST_TEST(!statevector->SupportsQuantumChannels());
  BOOST_CHECK_THROW(statevector->ApplyBitFlipNoise(0, 0.1),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(NoiseModelUsesExactT1CircuitOperation) {
  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));

  noise::NoiseModel noiseModel;
  noiseModel.set_t1(0, 0.75);
  const auto noisyCircuit = noise::inject_exact_noise(circuit, noiseModel);
  BOOST_REQUIRE_EQUAL(noisyCircuit->GetOperations().size(), 2);
  BOOST_TEST(static_cast<int>(noisyCircuit->GetOperations()[1]->GetType()) ==
             static_cast<int>(Circuits::OperationType::kQuantumChannel));

  auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix,
                                 1);
  Circuits::OperationState classicalState;
  noisyCircuit->Execute(densityMatrix, classicalState);

  // Amplitude damping multiplies |0><1| by sqrt(1-gamma). The old
  // state-independent probabilistic Reset path would incorrectly give 0.25.
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.5,
                    kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(ExactChannelRunsInOptimizedMultiShotPrefix) {
  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
  circuit->AddOperation(
      std::make_shared<Circuits::QuantumChannelOperation<double>>(
          Types::qubits_vector{0},
          Simulators::QuantumChannel::AmplitudeDamping(0.75)));
  circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));
  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement({{0, 0}}));

  auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
      std::vector<Types::qubit_t>{1}, std::vector<size_t>{1});
  network->RemoveAllOptimizationSimulatorsAndAdd(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);
  network->SetMaxSimulators(1);
  network->CreateSimulator();

  constexpr size_t shots = 4096;
  const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
  size_t ones = 0;
  for (const auto& [bits, count] : counts)
    if (!bits.empty() && bits[0]) ones += count;

  // H, AD(0.75), X gives P(1)=0.875. If a channel incorrectly stops the
  // optimized prefix, the sampler observes the post-H state and returns 0.5.
  const double measuredProbability = static_cast<double>(ones) / shots;
  BOOST_CHECK_SMALL(measuredProbability - 0.875, 0.04);
}

BOOST_AUTO_TEST_CASE(QuantumChannelCircuitComparisonUsesChannelContents) {
  Circuits::ComparableCircuit<> left;
  Circuits::ComparableCircuit<> equal;
  Circuits::ComparableCircuit<> different;
  left.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(
      Types::qubits_vector{0},
      Simulators::QuantumChannel::PhaseDamping(0.2)));
  equal.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(
      Types::qubits_vector{0},
      Simulators::QuantumChannel::PhaseDamping(0.2)));
  different.AddOperation(
      std::make_shared<Circuits::QuantumChannelOperation<>>(
          Types::qubits_vector{0},
          Simulators::QuantumChannel::PhaseDamping(0.3)));

  BOOST_TEST(left == equal);
  BOOST_TEST(left != different);
}

BOOST_AUTO_TEST_SUITE_END()
