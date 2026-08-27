/** Tests for the optional GPU matrix-product-operator (MPO) backend. */
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

constexpr double kGpuMPORandomCircuitTolerance = 1e-6;

std::shared_ptr<Simulators::ISimulator> MakeGpuMPO(size_t qubits) {
  Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kMatrixProductOperator);
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

BOOST_AUTO_TEST_SUITE(gpu_mpo_tests)

BOOST_AUTO_TEST_CASE(factory_and_unitary_evolution) {
  auto mpo = MakeGpuMPO(3);
  if (!mpo) {
    BOOST_TEST_MESSAGE("GPU matrix-product-operator library is unavailable; "
                       "skipping");
    return;
  }
  BOOST_TEST(static_cast<int>(mpo->GetSimulationType()) ==
             static_cast<int>(Simulators::SimulationType::kMatrixProductOperator));
  BOOST_TEST(mpo->GetConfiguration("method") == "matrix_product_operator");
  BOOST_CHECK_THROW(mpo->Amplitude(0), std::runtime_error);
  BOOST_CHECK_THROW(mpo->AmplitudeRaw(0), std::runtime_error);

  mpo->Configure("matrix_product_operator_max_bond_dimension", "64");
  BOOST_TEST(mpo->GetConfiguration(
                 "matrix_product_operator_max_bond_dimension") == "64");

  auto unique = Simulators::SimulatorsFactory::CreateSimulatorUnique(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kMatrixProductOperator);
  BOOST_REQUIRE(unique);
  BOOST_TEST(unique->GetConfiguration("method") == "matrix_product_operator");

  mpo->ApplyH(0);
  mpo->ApplyCX(0, 1);
  mpo->ApplyRy(2, 0.63);
  mpo->ApplyCCX(0, 1, 2);
  mpo->ApplyCSwap(0, 1, 2);
  Eigen::Matrix2cd generic;
  generic << 0.0, 1.0, 1.0, 0.0;
  mpo->ApplyGenericOneQubitGate(2, generic);

  auto reference = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  reference->AllocateQubits(3);
  reference->Initialize();
  reference->ApplyH(0);
  reference->ApplyCX(0, 1);
  reference->ApplyRy(2, 0.63);
  reference->ApplyCCX(0, 1, 2);
  reference->ApplyCSwap(0, 1, 2);
  reference->ApplyGenericOneQubitGate(2, generic);
  CheckClose(mpo->AllProbabilities(), reference->AllProbabilities());
  BOOST_CHECK_SMALL(mpo->ExpectationValue("XYZ") -
                        reference->ExpectationValue("XYZ"),
                    1e-9);
}

BOOST_AUTO_TEST_CASE(pure_state_initialization_sampling_and_measurement) {
  Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
  auto mpo = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kMatrixProductOperator);
  if (!mpo) {
    BOOST_TEST_MESSAGE("GPU matrix-product-operator library is unavailable; "
                       "skipping");
    return;
  }
  mpo->Configure("use_double_precision", "true");

  const double inverseSqrtTwo = 1.0 / std::sqrt(2.0);
  std::vector<std::complex<double>> amplitudes = {
      inverseSqrtTwo, 0.0, 0.0, inverseSqrtTwo};
  mpo->InitializeState(2, amplitudes);
  CheckClose(mpo->AllProbabilities(), {0.5, 0.0, 0.0, 0.5});

  constexpr size_t shots = 4096;
  const auto counts = mpo->SampleCounts({0, 1}, shots);
  BOOST_CHECK_EQUAL(counts.at(0) + counts.at(3), shots);
  BOOST_CHECK_SMALL(static_cast<double>(counts.at(0)) / shots - 0.5, 0.04);
  CheckClose(mpo->AllProbabilities(), {0.5, 0.0, 0.0, 0.5});

  const auto measured = mpo->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(measured.size(), 2);
  BOOST_TEST(measured[0] == measured[1]);
  CheckClose(mpo->AllProbabilities(),
             measured[0] ? std::vector<double>{0.0, 0.0, 0.0, 1.0}
                         : std::vector<double>{1.0, 0.0, 0.0, 0.0});
}

BOOST_AUTO_TEST_CASE(channel_bond_dimension_state_management_and_clone) {
  auto mpo = MakeGpuMPO(2);
  if (!mpo) {
    BOOST_TEST_MESSAGE("GPU matrix-product-operator library is unavailable; "
                       "skipping");
    return;
  }
  BOOST_TEST(mpo->SupportsQuantumChannels());
  BOOST_CHECK_THROW(mpo->Configure("use_double_precision", "false"),
                    std::runtime_error);

  mpo->ApplyH(0);
  mpo->ApplyCX(0, 1);
  // Entangling the two qubits should have grown the bond dimension beyond
  // the initial product-state value of 1.
  BOOST_TEST(mpo->GetCurrentMaxBondDimension() > 1);
  mpo->SaveState();

  const auto measured = mpo->MeasureMany({0, 1});
  BOOST_REQUIRE_EQUAL(measured.size(), 2);
  BOOST_TEST(measured[0] == measured[1]);
  mpo->RestoreState();
  CheckClose(mpo->AllProbabilities(), {0.5, 0.0, 0.0, 0.5});
  // The historical maximum bond dimension must survive the collapsing
  // measurement and the restore that undoes it.
  BOOST_TEST(mpo->GetCurrentMaxBondDimension() > 1);

  mpo->ApplyQuantumChannel({0},
                          Simulators::QuantumChannel::AmplitudeDamping(0.25));
  const auto afterChannel = mpo->AllProbabilities();
  BOOST_CHECK_SMALL(afterChannel[0] + afterChannel[2] - 0.625, 1e-9);

  mpo->SaveState();
  mpo->ApplyReset({0, 1});
  CheckClose(mpo->AllProbabilities(), {1.0, 0.0, 0.0, 0.0});
  mpo->RestoreState();
  CheckClose(mpo->AllProbabilities(), afterChannel);

  auto clone = mpo->Clone();
  BOOST_REQUIRE(clone);
  BOOST_TEST(clone->GetConfiguration("use_double_precision") == "true");
  BOOST_TEST(clone->GetCurrentMaxBondDimension() ==
             mpo->GetCurrentMaxBondDimension());
  CheckClose(clone->AllProbabilities(), mpo->AllProbabilities());
  clone->ApplyX(1);
  BOOST_CHECK(clone->AllProbabilities() != mpo->AllProbabilities());
}

BOOST_AUTO_TEST_CASE(network_execution_uses_mpo_backend) {
  auto availabilityCheck = MakeGpuMPO(1);
  if (!availabilityCheck) {
    BOOST_TEST_MESSAGE("GPU matrix-product-operator library is unavailable; "
                       "skipping");
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
      Simulators::SimulationType::kMatrixProductOperator);
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
             static_cast<int>(Simulators::SimulationType::kMatrixProductOperator));
  BOOST_CHECK_SMALL(static_cast<double>(ones) / shots - 0.875, 0.04);
}

struct GpuMPORandomCircuitsFixture {
  GpuMPORandomCircuitsFixture() {
    qcsimSV = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qcsimSV->AllocateQubits(nrQubitsForRandomCirc);
    qcsimSV->Initialize();

    gpuMPO = MakeGpuMPO(nrQubitsForRandomCirc);

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
  std::shared_ptr<Simulators::ISimulator> gpuMPO;

  std::shared_ptr<Circuits::Circuit<>> circ;
  std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
  Circuits::OperationState state;
};

BOOST_DATA_TEST_CASE_F(GpuMPORandomCircuitsFixture,
                       RandomCircuitsTest, bdata::xrange(1, 20), nrGates) {
  if (!gpuMPO) {
    BOOST_TEST_MESSAGE("GPU matrix-product-operator library is unavailable; "
                       "skipping");
    return;
  }

  size_t nrStates = 1ULL << nrQubitsForRandomCirc;

  GenerateCircuit(nrGates);

  circ->Execute(qcsimSV, state);
  circ->Execute(gpuMPO, state);

  auto svProbs = qcsimSV->AllProbabilities();
  BOOST_REQUIRE_EQUAL(svProbs.size(), nrStates);

  auto mpoProbs = gpuMPO->AllProbabilities();
  BOOST_REQUIRE_EQUAL(mpoProbs.size(), nrStates);

  for (size_t stateIdx = 0; stateIdx < nrStates; ++stateIdx) {
    const auto psv = svProbs[stateIdx];
    const auto pmpo = mpoProbs[stateIdx];

    BOOST_TEST(std::abs(pmpo - psv) < kGpuMPORandomCircuitTolerance,
               "Probability mismatch for outcome " << stateIdx << ": expected "
                                                << psv << ", got " << pmpo);
  }

  resetRandomCirc->Execute(gpuMPO, state);
  resetRandomCirc->Execute(qcsimSV, state);

  circ->Clear();
  state.Reset();
}

BOOST_DATA_TEST_CASE_F(GpuMPORandomCircuitsFixture,
                       SampleCountsManyTest, bdata::xrange(15, 30), nrGates) {
  if (!gpuMPO) {
    BOOST_TEST_MESSAGE("GPU matrix-product-operator library is unavailable; "
                       "skipping");
    return;
  }

  GenerateCircuit(nrGates);

  circ->Execute(qcsimSV, state);
  circ->Execute(gpuMPO, state);

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
  auto mpoCounts = gpuMPO->SampleCountsMany(sampledQubits, shots);

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

  resetRandomCirc->Execute(gpuMPO, state);
  resetRandomCirc->Execute(qcsimSV, state);

  circ->Clear();
  state.Reset();
}

BOOST_AUTO_TEST_SUITE_END()
#endif
