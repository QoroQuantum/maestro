/**
 * @file aertests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the Aer simulator.
 *
 * Just tests for simulator creation, qubits allocation, initialization, a
 * single X gate and measurements of all qubits.
 */

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include <framework/avx2_detect.hpp>

#include <cmath>
#include <vector>

#undef min
#undef max

#include "../Simulators/Factory.h"  // project being tested
#include "../python/noise.h"

struct AerSimulatorTestFixture {
  AerSimulatorTestFixture() {
    aer = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kStatevector);
    aer->AllocateQubits(3);
    aer->Initialize();
  }

  ~AerSimulatorTestFixture() {}

  std::shared_ptr<Simulators::ISimulator> aer;
};

BOOST_AUTO_TEST_SUITE(aer_simulator_tests)

BOOST_FIXTURE_TEST_CASE(test, AerSimulatorTestFixture) {
  BOOST_REQUIRE(AER::is_avx2_supported());

  BOOST_TEST(aer);

  aer->ApplyX(0);

  auto res = aer->Measure({0, 1, 2});

  BOOST_TEST(res == 1ULL);

  BOOST_CHECK_CLOSE(aer->Probability(res), 1.0, 0.000001);
}

BOOST_AUTO_TEST_CASE(density_matrix_exact_quantum_channels) {
  auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix);
  densityMatrix->AllocateQubits(3);
  densityMatrix->Initialize();
  BOOST_TEST(densityMatrix->SupportsQuantumChannels());

  densityMatrix->ApplyH(0);
  densityMatrix->ApplyPhaseFlipNoise(0, 0.2);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XII") - 0.6, 1e-10);

  densityMatrix->Reset();
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyAmplitudeDamping(0, 0.36);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.32, 1e-10);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XII") - 0.8, 1e-10);

  densityMatrix->Reset();
  densityMatrix->ApplyGeneralizedAmplitudeDamping(0, 0.4, 0.25);
  BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.1, 1e-10);

  densityMatrix->Reset();
  constexpr double duration = 0.7;
  constexpr double t1 = 2.0;
  constexpr double t2 = 1.5;
  constexpr double excitedPopulation = 0.1;
  densityMatrix->ApplyH(0);
  densityMatrix->ApplyThermalRelaxation(
      0, duration, t1, t2, excitedPopulation);
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XII") -
                        std::exp(-duration / t2),
                    1e-10);

  densityMatrix->Reset();
  densityMatrix->ApplyTwoQubitDepolarizingNoise(0, 2, 15.0 / 16.0);
  const auto fullyMixed = densityMatrix->AllProbabilities();
  for (size_t outcome = 0; outcome < fullyMixed.size(); ++outcome) {
    const double expected = (outcome & 2U) == 0 ? 0.25 : 0.0;
    BOOST_CHECK_SMALL(fullyMixed[outcome] - expected, 1e-10);
  }
}

BOOST_AUTO_TEST_CASE(density_matrix_kraus_target_order_matches_maestro) {
  auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix);
  densityMatrix->AllocateQubits(3);
  densityMatrix->Initialize();

  // I (x) X flips local bit zero, which is targets[0] in Maestro and Aer.
  Eigen::MatrixXcd identityTensorX = Eigen::MatrixXcd::Zero(4, 4);
  identityTensorX(1, 0) = 1.0;
  identityTensorX(0, 1) = 1.0;
  identityTensorX(3, 2) = 1.0;
  identityTensorX(2, 3) = 1.0;
  densityMatrix->ApplyKrausChannel({0, 2}, {identityTensorX});
  BOOST_CHECK_CLOSE(densityMatrix->Probability(1), 1.0, 1e-8);

  // The generic-unitary and Kraus APIs use the same target convention.
  densityMatrix->Reset();
  densityMatrix->ApplyGenericTwoQubitGate(0, 2, identityTensorX);
  BOOST_CHECK_CLOSE(densityMatrix->Probability(1), 1.0, 1e-8);

  auto statevector = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kStatevector);
  statevector->AllocateQubits(1);
  statevector->Initialize();
  BOOST_TEST(!statevector->SupportsQuantumChannels());
  BOOST_CHECK_THROW(statevector->ApplyBitFlipNoise(0, 0.1),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(density_matrix_executes_exact_noise_model_circuit) {
  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
  noise::NoiseModel noiseModel;
  noiseModel.set_t1(0, 0.75);
  noiseModel.set_phase_damping(0, 0.36);
  const auto noisyCircuit = noise::inject_exact_noise(circuit, noiseModel);

  auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kDensityMatrix);
  densityMatrix->AllocateQubits(1);
  densityMatrix->Initialize();
  Circuits::OperationState classicalState;
  noisyCircuit->Execute(densityMatrix, classicalState);
  // AD contributes sqrt(1-0.75)=0.5 and phase damping contributes
  // sqrt(1-0.36)=0.8.
  BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.4, 1e-10);
}

BOOST_AUTO_TEST_CASE(matrix_product_state_truncation_mode_only_accepts_discarded_weight) {
  auto mps = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kMatrixProductState);
  mps->AllocateQubits(1);

  // Aer's own MPS/MPO truncation always implements the discarded-weight
  // (Aer/iTensor) convention natively -- unlike the QCSim and GPU backends,
  // there is no relative-to-max mode to switch to, so it should be a no-op
  // to (re)confirm the only mode it has...
  BOOST_CHECK_NO_THROW(
      mps->Configure("matrix_product_state_truncation_mode", "discarded_weight"));
  BOOST_CHECK_NO_THROW(
      mps->Configure("matrix_product_operator_truncation_mode", "discarded_weight"));

  // ...but should reject a request to switch to the relative-to-max mode
  // that the QCSim/GPU backends support, rather than silently ignoring it.
  BOOST_CHECK_THROW(
      mps->Configure("matrix_product_state_truncation_mode", "relative_max"),
      std::invalid_argument);
  BOOST_CHECK_THROW(
      mps->Configure("matrix_product_operator_truncation_mode", "relative_max"),
      std::invalid_argument);
}

// Regression test for a real bug found during review: Configure() used to store the
// requested value into `configuration` BEFORE validating it, so a rejected
// "relative_max" request would throw as intended but still leave "relative_max"
// sitting in the configuration map -- visible via GetConfiguration(), and liable to
// throw again, unexpectedly, from an unrelated later call site (e.g. Clone()'s
// generic configuration-replay loop). Fixed by validating before storing.
BOOST_AUTO_TEST_CASE(rejected_truncation_mode_does_not_linger_in_configuration) {
  auto mps = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQiskitAer,
      Simulators::SimulationType::kMatrixProductState);
  mps->AllocateQubits(1);

  BOOST_CHECK_THROW(
      mps->Configure("matrix_product_state_truncation_mode", "relative_max"),
      std::invalid_argument);

  // The rejected value must not have been persisted -- GetConfiguration should report
  // the unset/default state, not the value that was just thrown out.
  BOOST_TEST(mps->GetConfiguration("matrix_product_state_truncation_mode") !=
             std::string("relative_max"));

  // A rejected call must not corrupt Configure()'s ability to accept the one value
  // Aer actually supports right afterward.
  BOOST_CHECK_NO_THROW(
      mps->Configure("matrix_product_state_truncation_mode", "discarded_weight"));
}

BOOST_AUTO_TEST_SUITE_END()
