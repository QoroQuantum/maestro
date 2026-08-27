/** Extensive GPU-vs-CPU (QCSim) cross-checks for the density-matrix backend. */
#ifdef __linux__

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace bdata = boost::unit_test::data;

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../Circuit/Factory.h"
#include "../Simulators/Factory.h"
#include "../Simulators/QuantumChannel.h"

namespace {

constexpr unsigned int kNumQubits = 4;
constexpr double kTolerance = 1e-7;

std::shared_ptr<Simulators::ISimulator> MakeQCSimDM(
    size_t qubits = kNumQubits) {
  auto sim = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);
  sim->AllocateQubits(qubits);
  sim->Initialize();
  return sim;
}

std::shared_ptr<Simulators::ISimulator> MakeGpuDM(size_t qubits = kNumQubits) {
  Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
  auto sim = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  if (sim) {
    sim->Configure("use_double_precision", "true");
    sim->AllocateQubits(qubits);
    sim->Initialize();
  }
  return sim;
}

void CheckAllClose(const std::vector<double>& a, const std::vector<double>& b,
                   double tol = kTolerance) {
  BOOST_REQUIRE_EQUAL(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i)
    BOOST_TEST(std::abs(a[i] - b[i]) < tol,
               "mismatch at outcome " << i << ": " << a[i] << " vs " << b[i]);
}

// Applies one instance of every supported gate directly, in an order that
// mixes single- and multi-qubit operators, including the 3-qubit ones (CCX,
// CSwap) which on the gpu backend are decomposed internally.
void ApplyAllGatesOnce(Simulators::ISimulator& sim) {
  sim.ApplyH(0);
  sim.ApplyX(1);
  sim.ApplyY(2);
  sim.ApplyZ(3);
  sim.ApplyS(0);
  sim.ApplySDG(1);
  sim.ApplyT(2);
  sim.ApplyTDG(3);
  sim.ApplySx(0);
  sim.ApplySxDAG(1);
  sim.ApplyK(2);
  sim.ApplyP(3, 0.37);
  sim.ApplyRx(0, 0.51);
  sim.ApplyRy(1, -0.62);
  sim.ApplyRz(2, 1.13);
  sim.ApplyU(3, 0.2, -0.4, 0.6, -0.8);
  sim.ApplyCX(0, 1);
  sim.ApplyCY(1, 2);
  sim.ApplyCZ(2, 3);
  sim.ApplyCP(3, 0, 0.44);
  sim.ApplyCRx(0, 2, 0.71);
  sim.ApplyCRy(1, 3, -0.33);
  sim.ApplyCRz(2, 0, 0.19);
  sim.ApplyCH(3, 1);
  sim.ApplyCSx(0, 3);
  sim.ApplyCSxDAG(1, 0);
  sim.ApplySwap(2, 3);
  sim.ApplyCU(0, 1, 0.15, -0.25, 0.35, -0.45);
  sim.ApplyCCX(0, 1, 2);
  sim.ApplyCSwap(3, 0, 1);
  Eigen::Matrix2cd oneQubit;
  oneQubit << 0.0, 1.0, 1.0, 0.0;
  sim.ApplyGenericOneQubitGate(2, oneQubit);
  Eigen::Matrix4cd twoQubit = Eigen::Matrix4cd::Zero();
  const std::complex<double> imaginary(0.0, 1.0);
  twoQubit(0, 0) = 1.0;
  twoQubit(1, 2) = imaginary;
  twoQubit(2, 1) = imaginary;
  twoQubit(3, 3) = 1.0;
  sim.ApplyGenericTwoQubitGate(1, 3, twoQubit);
}

struct QCSimVsGpuDMFixture {
  QCSimVsGpuDMFixture() {
    qcsimDM = MakeQCSimDM();
    gpuDM = MakeGpuDM();

    resetCirc = std::make_shared<Circuits::Circuit<>>();
    Types::qubits_vector qubits(kNumQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    resetCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));
  }

  std::shared_ptr<Circuits::Circuit<>> GenerateCircuit(int nrGates) {
    auto circ = std::make_shared<Circuits::Circuit<>>();
    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
    auto gateGenIter = gateGen.begin();

    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      Types::qubits_vector qubits(kNumQubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      std::shuffle(qubits.begin(), qubits.end(), g);
      auto q1 = qubits[0];
      auto q2 = qubits[1];
      auto q3 = qubits[2];

      const double param1 = *dblGenIter; ++dblGenIter;
      const double param2 = *dblGenIter; ++dblGenIter;
      const double param3 = *dblGenIter; ++dblGenIter;
      const double param4 = *dblGenIter; ++dblGenIter;

      const auto gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);
      circ->AddOperation(Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3, param4));
    }
    return circ;
  }

  std::shared_ptr<Simulators::ISimulator> qcsimDM;
  std::shared_ptr<Simulators::ISimulator> gpuDM;
  std::shared_ptr<Circuits::Circuit<>> resetCirc;
  Circuits::OperationState state{kNumQubits};
};

}  // namespace

BOOST_AUTO_TEST_SUITE(gpu_vs_cpu_density_matrix_tests)

BOOST_AUTO_TEST_CASE(configuration_matches) {
  auto gpuDM = MakeGpuDM();
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = MakeQCSimDM();
  BOOST_TEST(gpuDM->GetConfiguration("method") ==
             qcsimDM->GetConfiguration("method"));
  BOOST_TEST(static_cast<int>(gpuDM->GetSimulationType()) ==
             static_cast<int>(qcsimDM->GetSimulationType()));
  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());
}

BOOST_AUTO_TEST_CASE(basis_state_initialization_matches) {
  auto gpuDM = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);

  const std::array<std::string, 3> paulis = {"XIII", "ZZZZ", "XYZI"};
  for (Types::qubit_t basisState = 0; basisState < (1ULL << kNumQubits);
       ++basisState) {
    gpuDM->InitializeToBasisState(kNumQubits, basisState);
    qcsimDM->InitializeToBasisState(kNumQubits, basisState);

    CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());
    for (const auto& pauli : paulis)
      BOOST_TEST(std::abs(gpuDM->ExpectationValue(pauli) -
                          qcsimDM->ExpectationValue(pauli)) < kTolerance,
                 "Pauli " << pauli << " mismatch at basis state "
                          << basisState);
  }
}

BOOST_AUTO_TEST_CASE(mixture_of_basis_states_initialization_matches) {
  auto gpuDM = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix);
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix);

  const std::vector<std::pair<Types::qubit_t, double>> mixture = {
      {0, 0.5}, {3, 0.2}, {5, 0.2}, {12, 0.1}};

  gpuDM->InitializeToMixtureOfBasisStates(kNumQubits, mixture);
  qcsimDM->InitializeToMixtureOfBasisStates(kNumQubits, mixture);

  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());
  BOOST_CHECK_CLOSE(gpuDM->Probability(0), 0.5, 1e-4);
  BOOST_CHECK_CLOSE(gpuDM->Probability(3), 0.2, 1e-4);

  // Applying identical unitary evolution afterwards should keep both
  // backends in lockstep.
  gpuDM->ApplyH(0);
  gpuDM->ApplyCX(0, 1);
  qcsimDM->ApplyH(0);
  qcsimDM->ApplyCX(0, 1);
  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());

  const std::array<std::string, 2> paulis = {"ZZZI", "XXII"};
  for (const auto& pauli : paulis)
    BOOST_TEST(std::abs(gpuDM->ExpectationValue(pauli) -
                        qcsimDM->ExpectationValue(pauli)) < kTolerance);
}

BOOST_AUTO_TEST_CASE(all_gates_and_channels_direct_application_matches) {
  auto gpuDM = MakeGpuDM();
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = MakeQCSimDM();

  ApplyAllGatesOnce(*gpuDM);
  ApplyAllGatesOnce(*qcsimDM);

  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());

  // Apply an amplitude-damping channel identically and check again.
  gpuDM->ApplyQuantumChannel({0},
                             Simulators::QuantumChannel::AmplitudeDamping(0.3));
  qcsimDM->ApplyQuantumChannel(
      {0}, Simulators::QuantumChannel::AmplitudeDamping(0.3));
  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());

  // Unlike a pure-state (MPS/statevector) reset, DensityMatrix::ApplyReset is
  // a deterministic quantum channel (trace out the qubit, reinitialize to
  // |0><0|) rather than a stochastic single-shot measurement, so comparing
  // AllProbabilities() directly after it is meaningful here.
  gpuDM->ApplyReset({1, 2});
  qcsimDM->ApplyReset({1, 2});
  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());
}

BOOST_AUTO_TEST_CASE(all_gates_pauli_expectation_values_match) {
  auto gpuDM = MakeGpuDM();
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = MakeQCSimDM();

  ApplyAllGatesOnce(*gpuDM);
  ApplyAllGatesOnce(*qcsimDM);

  const std::array<std::string, 18> paulis = {
      "ZIII", "IZII", "IIZI", "IIIZ", "XIII", "IXII", "IIXI", "IIIX",
      "YIII", "IYII", "IIYI", "IIIY", "XYIZ", "YZYX", "IXYZ", "IYZI",
      "ZZZZ", "XXXX"};
  for (const auto& pauli : paulis)
    BOOST_TEST(std::abs(gpuDM->ExpectationValue(pauli) -
                        qcsimDM->ExpectationValue(pauli)) < kTolerance,
               "Pauli " << pauli << " mismatch");
}

BOOST_AUTO_TEST_CASE(all_gates_sampling_matches) {
  auto gpuDM = MakeGpuDM();
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = MakeQCSimDM();

  ApplyAllGatesOnce(*gpuDM);
  ApplyAllGatesOnce(*qcsimDM);

  constexpr size_t shots = 20000;
  Types::qubits_vector allQubits(kNumQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);
  auto qcsimCounts = qcsimDM->SampleCounts(allQubits, shots);
  auto gpuCounts = gpuDM->SampleCounts(allQubits, shots);
  for (Types::qubit_t outcome = 0; outcome < (1ULL << kNumQubits); ++outcome) {
    const double qcsimProb = static_cast<double>(qcsimCounts[outcome]) / shots;
    const double gpuProb = static_cast<double>(gpuCounts[outcome]) / shots;
    if (qcsimProb < 0.02 && gpuProb < 0.02) continue;
    BOOST_CHECK_CLOSE(qcsimProb, gpuProb, 40);
  }
}

// Repeated-measurement counts, mirroring the same save/measure/restore
// pattern used for the MPS backend (see gpuvscpumpstests.cpp and
// mpssimtests.cpp's RandomCircuitsTest50): apply the gates and save the
// state once, then repeatedly measure all qubits and restore the
// pre-measurement state before the next trial, accumulating an empirical
// outcome distribution.
BOOST_AUTO_TEST_CASE(all_gates_repeated_measurement_matches) {
  auto gpuDM = MakeGpuDM();
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = MakeQCSimDM();

  ApplyAllGatesOnce(*gpuDM);
  ApplyAllGatesOnce(*qcsimDM);

  Types::qubits_vector allQubits(kNumQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);

  constexpr size_t trials = 2000;
  std::unordered_map<std::vector<bool>, size_t> gpuCounts, qcsimCounts;

  gpuDM->SaveState();
  qcsimDM->SaveState();
  for (size_t trial = 0; trial < trials; ++trial) {
    if (trial > 0) {
      gpuDM->RestoreState();
      qcsimDM->RestoreState();
    }
    ++gpuCounts[gpuDM->MeasureMany(allQubits)];
    ++qcsimCounts[qcsimDM->MeasureMany(allQubits)];
  }

  for (const auto& [key, cnt] : gpuCounts) {
    const double val = static_cast<double>(cnt) / trials;
    if (val < 0.03) continue;
    double val2 = 0;
    if (qcsimCounts.find(key) != qcsimCounts.end())
      val2 = static_cast<double>(qcsimCounts[key]) / trials;
    BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
  }
  for (const auto& [key, cnt] : qcsimCounts) {
    const double val = static_cast<double>(cnt) / trials;
    if (val < 0.03) continue;
    double val2 = 0;
    if (gpuCounts.find(key) != gpuCounts.end())
      val2 = static_cast<double>(gpuCounts[key]) / trials;
    BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
  }
}

BOOST_DATA_TEST_CASE_F(QCSimVsGpuDMFixture, RandomCircuitsMatch,
                       bdata::xrange(1, 25), nrGates) {
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }

  auto circ = GenerateCircuit(nrGates);
  circ->Execute(qcsimDM, state);
  circ->Execute(gpuDM, state);

  CheckAllClose(gpuDM->AllProbabilities(), qcsimDM->AllProbabilities());

  const std::array<std::string, 4> paulis = {"ZZZZ", "XXXX", "YIXZ", "ZXYI"};
  for (const auto& pauli : paulis)
    BOOST_TEST(std::abs(gpuDM->ExpectationValue(pauli) -
                        qcsimDM->ExpectationValue(pauli)) < 1e-6,
               "Pauli " << pauli << " mismatch at " << nrGates << " gates");

  resetCirc->Execute(gpuDM, state);
  resetCirc->Execute(qcsimDM, state);
  state.Reset();
}

BOOST_DATA_TEST_CASE_F(QCSimVsGpuDMFixture, SamplingMatches,
                       bdata::xrange(10, 20), nrGates) {
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }

  auto circ = GenerateCircuit(nrGates);
  circ->Execute(qcsimDM, state);
  circ->Execute(gpuDM, state);

  constexpr size_t shots = 10000;
  Types::qubits_vector allQubits(kNumQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);

  auto qcsimCounts = qcsimDM->SampleCounts(allQubits, shots);
  auto gpuCounts = gpuDM->SampleCounts(allQubits, shots);

  for (Types::qubit_t outcome = 0; outcome < (1ULL << kNumQubits); ++outcome) {
    const double qcsimProb =
        static_cast<double>(qcsimCounts[outcome]) / shots;
    const double gpuProb = static_cast<double>(gpuCounts[outcome]) / shots;
    if (qcsimProb < 0.02 && gpuProb < 0.02) continue;
    BOOST_CHECK_CLOSE(qcsimProb, gpuProb, 40);
  }

  resetCirc->Execute(gpuDM, state);
  resetCirc->Execute(qcsimDM, state);
  state.Reset();
}

BOOST_AUTO_TEST_CASE(measurement_collapse_marginals_match) {
  auto gpuDM = MakeGpuDM(2);
  if (!gpuDM) {
    BOOST_TEST_MESSAGE("GPU density-matrix library is unavailable; skipping");
    return;
  }
  auto qcsimDM = MakeQCSimDM(2);

  constexpr int trials = 400;
  size_t gpuOnes = 0;
  size_t qcsimOnes = 0;
  for (int i = 0; i < trials; ++i) {
    gpuDM->Reset();
    gpuDM->ApplyH(0);
    gpuDM->ApplyCX(0, 1);
    const auto gpuMeasured = gpuDM->MeasureMany({0, 1});
    BOOST_REQUIRE_EQUAL(gpuMeasured.size(), 2);
    BOOST_TEST(gpuMeasured[0] == gpuMeasured[1]);
    if (gpuMeasured[0]) ++gpuOnes;

    qcsimDM->Reset();
    qcsimDM->ApplyH(0);
    qcsimDM->ApplyCX(0, 1);
    const auto qcsimMeasured = qcsimDM->MeasureMany({0, 1});
    BOOST_REQUIRE_EQUAL(qcsimMeasured.size(), 2);
    BOOST_TEST(qcsimMeasured[0] == qcsimMeasured[1]);
    if (qcsimMeasured[0]) ++qcsimOnes;
  }

  BOOST_CHECK_CLOSE(static_cast<double>(gpuOnes) / trials,
                    static_cast<double>(qcsimOnes) / trials, 40);
}

BOOST_AUTO_TEST_SUITE_END()
#endif
