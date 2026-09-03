/** Tests for the optional GPU dense density-matrix backend. */
#ifdef __linux__

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace bdata = boost::unit_test::data;

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Simulators/Factory.h"
#include "../Simulators/QuantumChannel.h"

namespace {

constexpr double kGpuDensityRandomCircuitTolerance = 1e-6;

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

BOOST_AUTO_TEST_CASE(updated_diagnostics_and_measurement_api) {
  auto density = MakeGpuDensity(2);
  if (!density) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  density->ApplyH(0);
  density->ApplyCX(0, 1);
  BOOST_CHECK_CLOSE(density->DensityMatrixTrace().real(), 1., 1e-7);
  BOOST_CHECK_CLOSE(density->DensityMatrixPurity(), 1., 1e-6);
  BOOST_CHECK(density->IsDensityMatrixHermitian());
  Eigen::VectorXcd bell(4);
  bell << 1. / std::sqrt(2.), 0., 0., 1. / std::sqrt(2.);
  BOOST_CHECK_CLOSE(density->FidelityWithStatevector(bell), 1., 1e-6);
  const auto reduced = density->PartialTrace(Types::qubits_vector{0});
  BOOST_REQUIRE_EQUAL(reduced.rows(), 2);
  BOOST_CHECK_SMALL(std::abs(reduced(0, 0) - 0.5), 1e-6);
  BOOST_CHECK_SMALL(std::abs(reduced(1, 1) - 0.5), 1e-6);
  BOOST_CHECK_SMALL(std::abs(density->DensityMatrixOverlap(*density).real() - 1.), 1e-6);
}

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

struct GpuDensityMatrixRandomCircuitsFixture {
  GpuDensityMatrixRandomCircuitsFixture() {
    qcsimSV = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qcsimSV->AllocateQubits(nrQubitsForRandomCirc);
    qcsimSV->Initialize();

    gpuDensity = MakeGpuDensity(nrQubitsForRandomCirc);

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
  std::shared_ptr<Simulators::ISimulator> gpuDensity;

  std::shared_ptr<Circuits::Circuit<>> circ;
  std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
  Circuits::OperationState state;
};

BOOST_DATA_TEST_CASE_F(GpuDensityMatrixRandomCircuitsFixture,
                       RandomCircuitsTest, bdata::xrange(1, 20), nrGates) {
  if (!gpuDensity) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }

  size_t nrStates = 1ULL << nrQubitsForRandomCirc;

  GenerateCircuit(nrGates);

  circ->Execute(qcsimSV, state);
  circ->Execute(gpuDensity, state);

  auto svProbs = qcsimSV->AllProbabilities();
  BOOST_REQUIRE_EQUAL(svProbs.size(), nrStates);

  auto dmProbs = gpuDensity->AllProbabilities();
  BOOST_REQUIRE_EQUAL(dmProbs.size(), nrStates);

  for (size_t stateIdx = 0; stateIdx < nrStates; ++stateIdx) {
    const auto psv = svProbs[stateIdx];
    const auto pdm = dmProbs[stateIdx];

    BOOST_TEST(std::abs(pdm - psv) < kGpuDensityRandomCircuitTolerance,
               "Probability mismatch for outcome " << stateIdx << ": expected "
                                                << psv << ", got " << pdm);
  }

  resetRandomCirc->Execute(gpuDensity, state);
  resetRandomCirc->Execute(qcsimSV, state);

  circ->Clear();
  state.Reset();
}

BOOST_DATA_TEST_CASE_F(GpuDensityMatrixRandomCircuitsFixture,
                       SampleCountsManyTest, bdata::xrange(15, 30), nrGates) {
  if (!gpuDensity) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }

  GenerateCircuit(nrGates);

  circ->Execute(qcsimSV, state);
  circ->Execute(gpuDensity, state);

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
  auto dmCounts = gpuDensity->SampleCountsMany(sampledQubits, shots);

  for (const auto& [outcome, cnt] : svCounts) {
    double svProb = static_cast<double>(cnt) / static_cast<double>(shots);
    if (svProb < 0.02) continue;

    double dmProb = 0;
    if (dmCounts.find(outcome) != dmCounts.end())
      dmProb = static_cast<double>(dmCounts[outcome]) /
               static_cast<double>(shots);

    BOOST_CHECK_CLOSE(svProb, dmProb, dmProb < 0.1 ? 66 : 33);
  }

  for (const auto& [outcome, cnt] : dmCounts) {
    double dmProb = static_cast<double>(cnt) / static_cast<double>(shots);
    if (dmProb < 0.02) continue;

    double svProb = 0;
    if (svCounts.find(outcome) != svCounts.end())
      svProb = static_cast<double>(svCounts[outcome]) /
               static_cast<double>(shots);

    BOOST_CHECK_CLOSE(dmProb, svProb, svProb < 0.1 ? 66 : 33);
  }

  resetRandomCirc->Execute(gpuDensity, state);
  resetRandomCirc->Execute(qcsimSV, state);

  circ->Clear();
  state.Reset();
}

BOOST_AUTO_TEST_SUITE_END()
#endif
