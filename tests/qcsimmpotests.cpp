/**
 * @file qcsimmpotests.cpp
 * @version 1.0
 *
 * Tests for QCSim's matrix product operator simulator through Maestro.
 */

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace bdata = boost::unit_test::data;

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
#include "../Simulators/Factory.h"
#include "../python/noise.h"

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
  mpo->Configure("matrix_product_operator_truncation_mode", "relative_max");
  BOOST_TEST(mpo->GetConfiguration(
                 "matrix_product_operator_max_bond_dimension") == "32");
  BOOST_TEST(mpo->GetConfiguration(
                 "matrix_product_operator_truncation_mode") == "relative_max");
  BOOST_CHECK_THROW(mpo->Configure("matrix_product_operator_truncation_mode",
                                   "typo"),
                    std::invalid_argument);
  BOOST_TEST(mpo->GetConfiguration(
                 "matrix_product_operator_truncation_mode") == "relative_max");

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
  statevector->ApplyNop();
  mpo->ApplyNop();
  CheckMPOProbabilities(*statevector, *mpo);
}

BOOST_AUTO_TEST_CASE(swap_optimization_and_bond_tracking_survive_clone) {
  constexpr size_t numQubits = 5;
  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(4));
  circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 4));
  circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(4, 1));
  circuit->AddOperation(std::make_shared<Circuits::SwapGate<>>(1, 3));
  circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(2, 0.37));
  circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(3, 0));
  circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(2, 4));
  circuit->AddOperation(std::make_shared<Circuits::CCXGate<>>(0, 2, 4));
  circuit->AddOperation(std::make_shared<Circuits::CSwapGate<>>(1, 0, 3));

  auto mpo = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductOperator);
  mpo->Configure("matrix_product_operator_max_bond_dimension", "16");
  mpo->AllocateQubits(numQubits);
  mpo->Initialize();

  BOOST_CHECK_THROW(mpo->SetInitialQubitsMap({0, 1, 2, 3, 3}),
                    std::invalid_argument);
  const std::vector<long long int> initialMap = {0, 4, 1, 3, 2};
  mpo->SetInitialQubitsMap(initialMap);
  mpo->SetUseOptimalMeetingPosition(true);
  mpo->SetLookaheadDepth(4);
  mpo->SetLookaheadDepthWithHeuristic(2);
  mpo->SetUpcomingGates(circuit->GetOperations());

  std::shared_ptr<Simulators::ISimulator> clone(mpo->Clone());
  auto statevector = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kStatevector, numQubits);
  Circuits::OperationState mpoState;
  Circuits::OperationState cloneState;
  Circuits::OperationState statevectorState;
  circuit->Execute(mpo, mpoState);
  circuit->Execute(clone, cloneState);
  circuit->Execute(statevector, statevectorState);

  CheckMPOProbabilities(*statevector, *mpo);
  CheckMPOProbabilities(*statevector, *clone);
  BOOST_TEST(mpo->GetGatesCounter() ==
             static_cast<long long int>(circuit->GetOperations().size()));
  BOOST_TEST(clone->GetGatesCounter() ==
             static_cast<long long int>(circuit->GetOperations().size()));
  BOOST_TEST(mpo->GetCurrentMaxBondDimension() > 1);
  BOOST_TEST(clone->GetCurrentMaxBondDimension() ==
             mpo->GetCurrentMaxBondDimension());
  BOOST_TEST(mpo->GetCurrentMaxBondDimension() <= 16);

  // Reset restores both the real and dummy chain mappings to identity. With
  // zero explicit lookahead, MPO still uses one-gate bond-aware optimization
  // when optimal meeting positions are enabled.
  mpo->Reset();
  statevector->Reset();
  mpo->SetLookaheadDepth(0);
  mpo->SetUpcomingGates(circuit->GetOperations());
  Circuits::OperationState resetMpoState;
  Circuits::OperationState resetStatevectorState;
  circuit->Execute(mpo, resetMpoState);
  circuit->Execute(statevector, resetStatevectorState);
  CheckMPOProbabilities(*statevector, *mpo);
  BOOST_TEST(mpo->GetCurrentMaxBondDimension() > 1);

  // The native MPO switch also supports the MPS-compatible routing heuristic
  // when immediate bond-aware optimization is disabled.
  mpo->Reset();
  statevector->Reset();
  mpo->SetUseOptimalMeetingPosition(false);
  mpo->SetUpcomingGates(circuit->GetOperations());
  Circuits::OperationState heuristicMpoState;
  Circuits::OperationState heuristicStatevectorState;
  circuit->Execute(mpo, heuristicMpoState);
  circuit->Execute(statevector, heuristicStatevectorState);
  CheckMPOProbabilities(*statevector, *mpo);
}

BOOST_AUTO_TEST_CASE(pure_state_initialization_is_rejected) {
  auto mpo = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductOperator);
  std::vector<std::complex<double>> amplitudes = {
      {0.5, 0.0}, {0.0, 0.5}, {-0.5, 0.0}, {0.0, -0.5}};

  // Unlike the dense density-matrix backend, the MPO backend currently only
  // supports the canonical |0...0> initialization path.
  BOOST_CHECK_THROW(mpo->InitializeState(2, amplitudes), std::runtime_error);
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

BOOST_AUTO_TEST_CASE(measurements_collapse_correlated_qubits) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator, 2);
  mpo->ApplyH(0);
  mpo->ApplyCX(0, 1);

  const auto first = mpo->Measure({0});
  const auto second = mpo->Measure({1});
  BOOST_TEST(first == second);
  BOOST_CHECK_CLOSE(mpo->Probability(first == 0 ? 0 : 3), 1.0, 1e-8);

  mpo->Reset();
  mpo->ApplyH(0);
  mpo->ApplyCX(0, 1);
  const auto many = mpo->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(many.size(), 2);
  BOOST_TEST(many[0] == many[1]);
}

BOOST_AUTO_TEST_CASE(save_restore_supports_repeated_measurements) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator, 2);
  mpo->ApplyH(0);
  mpo->ApplyRy(1, 0.8);

  const auto exact = mpo->AllProbabilities();
  mpo->SaveState();

  constexpr size_t shots = 2000;
  std::unordered_map<Types::qubit_t, Types::qubit_t> counts;
  for (size_t shot = 0; shot < shots; ++shot) {
    ++counts[mpo->Measure({0, 1})];
    mpo->RestoreState();
  }

  CheckMPOSamples(counts, exact, shots, 0.055);
  const auto restored = mpo->AllProbabilities();
  BOOST_REQUIRE_EQUAL(restored.size(), exact.size());
  for (size_t outcome = 0; outcome < exact.size(); ++outcome)
    BOOST_TEST(std::abs(restored[outcome] - exact[outcome]) < kMPOTolerance);
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

BOOST_AUTO_TEST_CASE(exact_noise_channels_match_dense_density_matrix) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator, 3);
  auto densityMatrix = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kDensityMatrix, 3);
  BOOST_TEST(mpo->SupportsQuantumChannels());
  BOOST_TEST(densityMatrix->SupportsQuantumChannels());

  auto prepare = [](Simulators::ISimulator &simulator) {
    simulator.ApplyH(0);
    simulator.ApplyRy(1, 0.43);
    simulator.ApplyCX(0, 2);
  };
  prepare(*mpo);
  prepare(*densityMatrix);

  mpo->ApplyPauliChannel(0, 0.07, 0.03, 0.11);
  densityMatrix->ApplyPauliChannel(0, 0.07, 0.03, 0.11);
  mpo->ApplyAmplitudeDamping(2, 0.23);
  densityMatrix->ApplyAmplitudeDamping(2, 0.23);
  mpo->ApplyPhaseDamping(1, 0.31);
  densityMatrix->ApplyPhaseDamping(1, 0.31);
  mpo->ApplyGeneralizedAmplitudeDamping(0, 0.19, 0.08);
  densityMatrix->ApplyGeneralizedAmplitudeDamping(0, 0.19, 0.08);
  mpo->ApplyThermalRelaxation(2, 0.4, 1.7, 1.2, 0.06);
  densityMatrix->ApplyThermalRelaxation(2, 0.4, 1.7, 1.2, 0.06);
  mpo->ApplyCorrelatedPhaseFlipNoise(0, 2, 0.17);
  densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 2, 0.17);
  mpo->ApplyTwoQubitDepolarizingNoise(0, 2, 0.13);
  densityMatrix->ApplyTwoQubitDepolarizingNoise(0, 2, 0.13);

  CheckMPOProbabilities(*densityMatrix, *mpo, 2e-8);
  for (const std::string &pauli : {"XII", "IZZ", "XYZ", "ZZZ"})
    BOOST_TEST(std::abs(mpo->ExpectationValue(pauli) -
                        densityMatrix->ExpectationValue(pauli)) < 2e-8,
               "Expectation mismatch after exact channels for " << pauli);
}

BOOST_AUTO_TEST_CASE(two_qubit_pauli_target_order_is_preserved) {
  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator, 3);

  std::vector<double> probabilities(16, 0.0);
  probabilities[1] = 1.0;  // X on the first target, identity on the second.
  mpo->ApplyPauliChannel({0, 2}, probabilities);
  BOOST_CHECK_CLOSE(mpo->Probability(1), 1.0, 1e-8);

  mpo->Reset();
  probabilities[1] = 0.0;
  probabilities[4] = 1.0;  // Identity on first target, X on second.
  mpo->ApplyPauliChannel({0, 2}, probabilities);
  BOOST_CHECK_CLOSE(mpo->Probability(4), 1.0, 1e-8);
}

BOOST_AUTO_TEST_CASE(noise_model_exact_path_matches_dense_density_matrix) {
  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
  circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(1, 0.37));
  circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 2));

  noise::NoiseModel noiseModel;
  noiseModel.set_qubit_noise(0, 0.02, 0.03, 0.04);
  // T1 and thermal relaxation both implement T1 decay, so they cannot
  // share a qubit. Keep amplitude damping on 0 and thermal on 2.
  noiseModel.set_t1(0, 0.17);
  noiseModel.set_1q_gate_depolarizing(1, 0.09);
  noiseModel.set_2q_gate_depolarizing(0, 0.07);
  noiseModel.set_2q_depolarizing(0, 2, 0.11);
  noiseModel.set_phase_damping(0, 0.08);
  noiseModel.set_generalized_amplitude_damping(1, 0.06, 0.2);
  noiseModel.set_thermal_relaxation(2, 0.3, 2.0, 1.5, 0.1);
  noiseModel.set_correlated_phase_flip(0, 2, 0.05, 0.4);
  const auto customOneQubit = Simulators::QuantumChannel::BitFlip(0.03);
  noiseModel.set_kraus_channel(
      {1}, customOneQubit.GetKrausOperators());
  const auto customTwoQubit =
      Simulators::QuantumChannel::TwoQubitDepolarizing(0.02);
  noiseModel.set_kraus_channel(
      {2, 0}, customTwoQubit.GetKrausOperators());

  std::mt19937 generator(1234);
  const auto noisyCircuit =
      noise::inject_combined_noise_exact(circuit, noiseModel, generator);
  size_t channelOperations = 0;
  for (const auto &operation : noisyCircuit->GetOperations())
    if (operation->GetType() == Circuits::OperationType::kQuantumChannel)
      ++channelOperations;
  BOOST_TEST(channelOperations == 14);

  std::mt19937 legacyGenerator(1234);
  BOOST_CHECK_THROW(
      noise::inject_combined_noise(circuit, noiseModel, legacyGenerator),
      std::invalid_argument);

  auto mpo = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kMatrixProductOperator, 3);
  auto densityMatrix = MakeQCSimMPOTestSimulator(
      Simulators::SimulationType::kDensityMatrix, 3);
  mpo->SetUpcomingGates(noisyCircuit->GetOperations());
  Circuits::OperationState mpoClassicalState;
  Circuits::OperationState densityClassicalState;
  noisyCircuit->Execute(mpo, mpoClassicalState);
  noisyCircuit->Execute(densityMatrix, densityClassicalState);

  BOOST_TEST(mpo->GetGatesCounter() ==
             static_cast<long long>(noisyCircuit->GetOperations().size()));

  CheckMPOProbabilities(*densityMatrix, *mpo, 2e-8);
  for (const std::string &pauli : {"XII", "IZZ", "XYZ", "ZZZ"})
    BOOST_TEST(std::abs(mpo->ExpectationValue(pauli) -
                        densityMatrix->ExpectationValue(pauli)) < 2e-8,
               "Expectation mismatch for exact NoiseModel path: " << pauli);
}

struct QCSimMPORandomCircuitsFixture {
  QCSimMPORandomCircuitsFixture() {
    qcsimSV = MakeQCSimMPOTestSimulator(Simulators::SimulationType::kStatevector,
                                       nrQubitsForRandomCirc);
    qcsimMPO = MakeQCSimMPOTestSimulator(
        Simulators::SimulationType::kMatrixProductOperator,
        nrQubitsForRandomCirc);

    circ = std::make_shared<Circuits::Circuit<>>();
    state.AllocateBits(nrQubitsForRandomCirc);

    resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
    Types::qubits_vector qubits(nrQubitsForRandomCirc);
    std::iota(qubits.begin(), qubits.end(), 0);
    resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));
  }

  void GenerateCircuit(int nrGates) {
    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
    auto gateGenIter = gateGen.begin();

    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      Types::qubits_vector qubits(nrQubitsForRandomCirc);
      std::iota(qubits.begin(), qubits.end(), 0);
      std::shuffle(qubits.begin(), qubits.end(), g);
      auto q1 = qubits[0];
      auto q2 = qubits[1];
      auto q3 = qubits[2];

      const double param1 = *dblGenIter; ++dblGenIter;
      const double param2 = *dblGenIter; ++dblGenIter;
      const double param3 = *dblGenIter; ++dblGenIter;
      const double param4 = *dblGenIter; ++dblGenIter;

      Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);

      auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3, param4);
      circ->AddOperation(theGate);
    }
  }

  const unsigned int nrQubitsForRandomCirc = 4;
  std::shared_ptr<Simulators::ISimulator> qcsimSV;
  std::shared_ptr<Simulators::ISimulator> qcsimMPO;

  std::shared_ptr<Circuits::Circuit<>> circ;
  std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
  Circuits::OperationState state;
};

BOOST_DATA_TEST_CASE_F(QCSimMPORandomCircuitsFixture,
                       RandomCircuitsTest, bdata::xrange(1, 20), nrGates) {
  size_t nrStates = 1ULL << nrQubitsForRandomCirc;

  GenerateCircuit(nrGates);

  circ->Execute(qcsimSV, state);
  qcsimMPO->SetUpcomingGates(circ->GetOperations());
  circ->Execute(qcsimMPO, state);

  auto svProbs = qcsimSV->AllProbabilities();
  BOOST_REQUIRE_EQUAL(svProbs.size(), nrStates);

  auto mpoProbs = qcsimMPO->AllProbabilities();
  BOOST_REQUIRE_EQUAL(mpoProbs.size(), nrStates);

  for (size_t stateIdx = 0; stateIdx < nrStates; ++stateIdx) {
    const auto psv = svProbs[stateIdx];
    const auto pmpo = mpoProbs[stateIdx];

    BOOST_TEST(std::abs(pmpo - psv) < kMPOTolerance,
               "Probability mismatch for outcome " << stateIdx << ": expected "
                                                << psv << ", got " << pmpo);
  }

  resetRandomCirc->Execute(qcsimMPO, state);
  resetRandomCirc->Execute(qcsimSV, state);

  circ->Clear();
  state.Reset();
}

BOOST_DATA_TEST_CASE_F(QCSimMPORandomCircuitsFixture,
                       SampleCountsManyTest, bdata::xrange(15, 30), nrGates) {
  GenerateCircuit(nrGates);

  circ->Execute(qcsimSV, state);
  qcsimMPO->SetUpcomingGates(circ->GetOperations());
  circ->Execute(qcsimMPO, state);

  std::random_device rd;
  std::mt19937 g(rd());

  Types::qubits_vector allQubits(nrQubitsForRandomCirc);
  std::iota(allQubits.begin(), allQubits.end(), 0);
  std::shuffle(allQubits.begin(), allQubits.end(), g);

  std::uniform_int_distribution<unsigned int> subsetSizeDist(
      1, nrQubitsForRandomCirc - 1);
  const unsigned int subsetSize = subsetSizeDist(g);

  Types::qubits_vector sampledQubits(allQubits.begin(),
                                     allQubits.begin() + subsetSize);

  const size_t shots = 10000;

  auto svCounts = qcsimSV->SampleCountsMany(sampledQubits, shots);
  auto mpoCounts = qcsimMPO->SampleCountsMany(sampledQubits, shots);

  for (const auto& [outcome, cnt] : svCounts) {
    double svProb = static_cast<double>(cnt) / static_cast<double>(shots);
    if (svProb < 0.02) continue;

    double mpoProb = 0;
    if (mpoCounts.find(outcome) != mpoCounts.end())
      mpoProb = static_cast<double>(mpoCounts[outcome]) /
                static_cast<double>(shots);

    BOOST_CHECK_CLOSE(svProb, mpoProb, mpoProb < 0.1 ? 66 : 33);
  }

  for (const auto& [outcome, cnt] : mpoCounts) {
    double mpoProb = static_cast<double>(cnt) / static_cast<double>(shots);
    if (mpoProb < 0.02) continue;

    double svProb = 0;
    if (svCounts.find(outcome) != svCounts.end())
      svProb = static_cast<double>(svCounts[outcome]) /
               static_cast<double>(shots);

    BOOST_CHECK_CLOSE(mpoProb, svProb, svProb < 0.1 ? 66 : 33);
  }

  resetRandomCirc->Execute(qcsimMPO, state);
  resetRandomCirc->Execute(qcsimSV, state);

  circ->Clear();
  state.Reset();
}

BOOST_AUTO_TEST_SUITE_END()
