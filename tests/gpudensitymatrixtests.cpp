/** Tests for the optional GPU dense density-matrix backend. */
#ifdef __linux__

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <complex>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Simulators/Factory.h"
#include "../Simulators/QuantumChannel.h"

namespace {

std::shared_ptr<Simulators::ISimulator> MakeGpuDensity(size_t qubits) {
  Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  if (simulator) {
    simulator->Configure("use_double_precision", "true");
    simulator->AllocateQubits(qubits);
    simulator->Initialize();
  }
  return simulator;
}

void CheckClose(const std::vector<double>& actual,
                const std::vector<double>& expected) {
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i)
    BOOST_CHECK_SMALL(actual[i] - expected[i], 1e-9);
}

}  // namespace

BOOST_AUTO_TEST_SUITE(gpu_density_matrix_tests)

BOOST_AUTO_TEST_CASE(factory_and_unitary_evolution) {
  auto density = MakeGpuDensity(3);
  if (!density) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  BOOST_TEST(static_cast<int>(density->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kDensityMatrix));
  BOOST_TEST(density->GetConfiguration("method") == "density_matrix");
  BOOST_CHECK_THROW(density->Amplitude(0), std::runtime_error);

  auto unique = Simulators::SimulatorsFactory::CreateSimulatorUnique(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  BOOST_REQUIRE(unique);
  BOOST_TEST(unique->GetConfiguration("method") == "density_matrix");

  density->ApplyH(0);
  density->ApplyCX(0, 1);
  density->ApplyRy(2, 0.63);
  density->ApplyCCX(0, 1, 2);
  Eigen::Matrix2cd generic;
  generic << 0.0, 1.0, 1.0, 0.0;
  density->ApplyGenericOneQubitGate(2, generic);

  auto reference = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);
  reference->AllocateQubits(3);
  reference->Initialize();
  reference->ApplyH(0);
  reference->ApplyCX(0, 1);
  reference->ApplyRy(2, 0.63);
  reference->ApplyCCX(0, 1, 2);
  reference->ApplyGenericOneQubitGate(2, generic);
  CheckClose(density->AllProbabilities(), reference->AllProbabilities());
  BOOST_CHECK_SMALL(density->ExpectationValue("XYZ") -
                        reference->ExpectationValue("XYZ"),
                    1e-9);
}

BOOST_AUTO_TEST_CASE(pure_state_initialization_sampling_and_measurement) {
  Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
  auto density = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  if (!density) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  density->Configure("use_double_precision", "true");

  const double inverseSqrtTwo = 1.0 / std::sqrt(2.0);
  std::vector<std::complex<double>> amplitudes = {
      inverseSqrtTwo, 0.0, 0.0, inverseSqrtTwo};
  density->InitializeState(2, amplitudes);
  CheckClose(density->AllProbabilities(), {0.5, 0.0, 0.0, 0.5});

  constexpr size_t shots = 4096;
  const auto counts = density->SampleCounts({0, 1}, shots);
  BOOST_CHECK_EQUAL(counts.at(0) + counts.at(3), shots);
  BOOST_CHECK_SMALL(static_cast<double>(counts.at(0)) / shots - 0.5, 0.04);
  CheckClose(density->AllProbabilities(), {0.5, 0.0, 0.0, 0.5});

  const auto measured = density->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(measured.size(), 2);
  BOOST_TEST(measured[0] == measured[1]);
  CheckClose(density->AllProbabilities(),
             measured[0] ? std::vector<double>{0.0, 0.0, 0.0, 1.0}
                         : std::vector<double>{1.0, 0.0, 0.0, 0.0});
}

BOOST_AUTO_TEST_CASE(channel_state_management_and_clone) {
  auto density = MakeGpuDensity(2);
  if (!density) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  BOOST_TEST(density->SupportsQuantumChannels());
  BOOST_CHECK_THROW(density->Configure("use_double_precision", "false"),
                    std::runtime_error);
  density->ApplyX(0);
  density->ApplyQuantumChannel(
      {0}, Simulators::QuantumChannel::AmplitudeDamping(0.25));
  CheckClose(density->AllProbabilities(), {0.25, 0.75, 0.0, 0.0});

  const auto beforeNonCollapsingMeasurement = density->AllProbabilities();
  const auto measured = density->MeasureNoCollapse();
  BOOST_CHECK(measured == 0 || measured == 1);
  BOOST_CHECK_EQUAL(density->MeasureNoCollapseMany().size(), 2);
  CheckClose(density->AllProbabilities(), beforeNonCollapsingMeasurement);

  density->SaveState();
  density->ApplyReset({0});
  CheckClose(density->AllProbabilities(), {1.0, 0.0, 0.0, 0.0});
  density->RestoreState();
  CheckClose(density->AllProbabilities(), {0.25, 0.75, 0.0, 0.0});

  auto clone = density->Clone();
  BOOST_REQUIRE(clone);
  BOOST_TEST(clone->GetConfiguration("use_double_precision") == "true");
  CheckClose(clone->AllProbabilities(), density->AllProbabilities());
  clone->ApplyX(1);
  BOOST_CHECK(clone->AllProbabilities() != density->AllProbabilities());
}

BOOST_AUTO_TEST_CASE(network_execution_uses_density_matrix_backend) {
  auto availabilityCheck = MakeGpuDensity(1);
  if (!availabilityCheck) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }

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
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  network->Configure("use_double_precision", "true");
  network->SetMaxSimulators(1);
  network->CreateSimulator();

  constexpr size_t shots = 4096;
  const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
  size_t ones = 0;
  for (const auto& [bits, count] : counts)
    if (!bits.empty() && bits[0]) ones += count;

  BOOST_TEST(static_cast<int>(network->GetLastSimulatorType()) ==
             static_cast<int>(Simulators::SimulatorType::kGpuSim));
  BOOST_TEST(static_cast<int>(network->GetLastSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kDensityMatrix));
  BOOST_CHECK_SMALL(static_cast<double>(ones) / shots - 0.875, 0.04);
}

BOOST_AUTO_TEST_SUITE_END()
#endif
