/**
 * @file pauliproptests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for pauli propagation simulators.
 */

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Simulators/Factory.h"
#include "../Circuit/Factory.h"
#include "../Simulators/QcsimPauliPropagator.h"

struct Operation {
  int gate = 0;  // gate id, first codes for clifford gates, then for
          // non-clifford gates, ordered by number of qubits
  int qubit1 = 0;
  int qubit2 = 0;  // unused for single qubit operations
  int qubit3 = 0;  // unused for single and two qubit operations
  // params, used only for some gates
  double theta = 0;
  double phi = 0;
  double lambda = 0;
  double gamma = 0;
};

struct PauliSimTestFixture {
  PauliSimTestFixture() {
    statevectorSim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    statevectorSim->AllocateQubits(nrQubitsForRandomCirc);
    statevectorSim->Initialize();

    qcsimPauliSim.EnableParallel();
    qcsimPauliSim.SetNrQubits(nrQubitsForRandomCirc);

#ifdef __linux__
    if (Simulators::SimulatorsFactory::IsGpuLibraryAvailable()) {
      gpuPauliSim =
          Simulators::SimulatorsFactory::CreateGpuPauliPropagatorSimulator();
      gpuPauliSim->CreateSimulator(nrQubitsForRandomCirc);
      gpuPauliSim->SetWillUseSampling(true);
      gpuPauliSim->AllocateMemory(0.8);
    }
#endif
  }

  std::string GeneratePauliString(int nrQubits) {
    std::string pauli;
    pauli.resize(nrQubits);
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dist(0, 3);

    for (int i = 0; i < nrQubits; ++i) {
      const int v = dist(g);
      switch (v) {
        case 0:
          pauli[i] = 'I';
          break;
        case 1:
          pauli[i] = 'X';
          break;
        case 2:
          pauli[i] = 'Y';
          break;
        case 3:
          pauli[i] = 'Z';
          break;
      }
    }

    return pauli;
  }

  std::vector<Operation> GenerateCircuit(int nrQubits, int nrGates, int maxGate = 12)
  {
    std::random_device rd;
    std::mt19937 g(rd());

    std::uniform_int_distribution<int> dist(0, maxGate);

    std::uniform_real_distribution<double> param_dist(0.0, 2. * M_PI);
    std::bernoulli_distribution bool_dist(0.8);  // high chance to make a clifford gate from a non-clifford one

    std::vector<Operation> circuit;
    std::vector<int> qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);

    for (int i = 0; i < nrGates; ++i) {
      Operation op;

      std::shuffle(qubits.begin(), qubits.end(), g);

      int gate = dist(g);  // random gate id

      if (gate > 12 && bool_dist(g))
        gate %= 13;  // make it clifford, to have less non-clifford gates in
                     // the circuit, which are more expensive to simulate

      op.gate = gate;
      op.qubit1 = qubits[0];
      op.qubit2 = qubits[1];
      op.qubit3 = qubits[2];

      op.theta = param_dist(g);
      op.phi = param_dist(g);
      op.lambda = param_dist(g);
      op.gamma = param_dist(g);

      circuit.push_back(std::move(op));
    }

    // check for 'dangerous' non-clifford gates, which can make the simulation
    // too expensive, and replace them with clifford ones - except the first one
    // for now only three qubit non-clifford gates, which are the most
    // expensive - for example cswap does 12 doublings!
    bool first = true;
    for (auto& op : circuit) {
      if (op.gate >= 28 && op.gate <= 29) {
        if (first) first = false;
        else op.gate = 10;  // cx... a clifford gate
      }
    }

    return circuit;
  }



  void ExecuteGate(const Operation& op,
                   std::shared_ptr<Simulators::ISimulator>& state_vector) {
    switch (op.gate) {
      case 0:
        state_vector->ApplyX(op.qubit1);
        break;
      case 1:
        state_vector->ApplyY(op.qubit1);
        break;
      case 2:
        state_vector->ApplyZ(op.qubit1);
        break;
      case 3:
        state_vector->ApplyH(op.qubit1);
        break;
      case 4:
        state_vector->ApplyS(op.qubit1);
        break;
      case 5:
        state_vector->ApplySDG(op.qubit1);
        break;

      case 6:
        state_vector->ApplySx(op.qubit1);
        break;
      case 7:
        state_vector->ApplySxDAG(op.qubit1);
        break;
      case 8:
        state_vector->ApplyK(op.qubit1);
        break;

      // two qubit gates
      case 9:
        state_vector->ApplySwap(op.qubit2, op.qubit1);
        break;
      case 10:
        state_vector->ApplyCX(op.qubit2, op.qubit1);
        break;
      case 11:
        state_vector->ApplyCY(op.qubit2, op.qubit1);
        break;
      case 12:
        state_vector->ApplyCZ(op.qubit2, op.qubit1);
        break;

// non-clifford single qubit gates

      case 13:
        state_vector->ApplyP(op.qubit1, op.theta);
        break;
      case 14:
        state_vector->ApplyRx(op.qubit1, op.theta);
        break;
      case 15:
        state_vector->ApplyRy(op.qubit1, op.theta);
        break;
      case 16:
        state_vector->ApplyRz(op.qubit1, op.theta);
        break;
      case 17:
        state_vector->ApplyU(op.qubit1, op.theta, op.phi, op.lambda, op.gamma);
        break;

      case 18:
        state_vector->ApplyT(op.qubit1);
        break;
      case 19:
        state_vector->ApplyTDG(op.qubit1);
        break;

// non-clifford two qubit gates
      case 20:
        state_vector->ApplyCH(op.qubit2, op.qubit1);
        break;


      case 21:
        state_vector->ApplyCRz(op.qubit2, op.qubit1, op.theta);
        break;
      case 22:
        state_vector->ApplyCRy(op.qubit2, op.qubit1, op.theta);
        break;
      case 23:
        state_vector->ApplyCRx(op.qubit2, op.qubit1, op.theta);
        break;


      case 24:
        state_vector->ApplyCP(op.qubit2, op.qubit1, op.theta);
        break;
      case 25:
        state_vector->ApplyCSx(op.qubit2, op.qubit1);
        break;
      case 26:
        state_vector->ApplyCSxDAG(op.qubit2, op.qubit1);
        break;


      case 27:
        state_vector->ApplyCU(op.qubit2, op.qubit1, op.theta, op.phi, op.lambda,
                             op.gamma);
        break;

      // non-clifford three qubit gates
      case 28:
        state_vector->ApplyCCX(op.qubit3, op.qubit2, op.qubit1);
        break;
      case 29:
        state_vector->ApplyCSwap(op.qubit3, op.qubit2, op.qubit1);
        break;
      default:
        std::cerr << "Unknown gate id: " << op.gate << std::endl;
    }
  }


   void ExecuteGate(const Operation& op,
                   Simulators::QcsimPauliPropagator& pauliSim) {
    switch (op.gate) {
      case 0:
        pauliSim.ApplyX(op.qubit1);
        break;
      case 1:
        pauliSim.ApplyY(op.qubit1);
        break;
      case 2:
        pauliSim.ApplyZ(op.qubit1);
        break;
      case 3:
        pauliSim.ApplyH(op.qubit1);
        break;
      case 4:
        pauliSim.ApplyS(op.qubit1);
        break;
      case 5:
        pauliSim.ApplySDG(op.qubit1);
        break;

      case 6:
        pauliSim.ApplySX(op.qubit1);
        break;
      case 7:
        pauliSim.ApplySXDG(op.qubit1);
        break;
      case 8:
        pauliSim.ApplyK(op.qubit1);
        break;

      // two qubit gates
      case 9:
        pauliSim.ApplySWAP(op.qubit1, op.qubit2);
        break;
      case 10:
        pauliSim.ApplyCX(op.qubit2, op.qubit1);
        break;
      case 11:
        pauliSim.ApplyCY(op.qubit2, op.qubit1);
        break;
      case 12:
        pauliSim.ApplyCZ(op.qubit2, op.qubit1);
        break;

        // non-clifford single qubit gates

      case 13:
        pauliSim.ApplyP(op.qubit1, op.theta);
        break;
      case 14:
        pauliSim.ApplyRX(op.qubit1, op.theta);
        break;
      case 15:
        pauliSim.ApplyRY(op.qubit1, op.theta);
        break;
      case 16:
        pauliSim.ApplyRZ(op.qubit1, op.theta);
        break;
      case 17:
        pauliSim.ApplyU(op.qubit1, op.theta, op.phi, op.lambda, op.gamma);
        break;

      case 18:
        pauliSim.ApplyT(op.qubit1);
        break;
      case 19:
        pauliSim.ApplyTDG(op.qubit1);
        break;

        // non-clifford two qubit gates

      case 20:
        pauliSim.ApplyCH(op.qubit2, op.qubit1);
        break;

      case 21:
        pauliSim.ApplyCRZ(op.qubit2, op.qubit1, op.theta);
        break;
      case 22:
        pauliSim.ApplyCRY(op.qubit2, op.qubit1, op.theta);
        break;
      case 23:
        pauliSim.ApplyCRX(op.qubit2, op.qubit1, op.theta);
        break;



      case 24:
        pauliSim.ApplyCP(op.qubit2, op.qubit1, op.theta);
        break;

      case 25:
        pauliSim.ApplyCSX(op.qubit2, op.qubit1);
        break;
      case 26:
        pauliSim.ApplyCSXDAG(op.qubit2, op.qubit1);
        break;
      case 27:
        pauliSim.ApplyCU(op.qubit2, op.qubit1, op.theta, op.phi, op.lambda, op.gamma);
        break;


      // non-clifford three qubit gates
      case 28:
        pauliSim.ApplyCCX(op.qubit3, op.qubit2, op.qubit1);
        break;
      case 29:
        pauliSim.ApplyCSwap(op.qubit3, op.qubit2, op.qubit1);
        break;
      default:
        std::cerr << "Unknown gate id: " << op.gate << std::endl;
    }
  }

#ifdef __linux__
   void ExecuteGate(const Operation& op,
                   Simulators::GpuPauliPropagator& pauliSim) {
    switch (op.gate) {
      case 0:
        pauliSim.ApplyX(op.qubit1);
        break;
      case 1:
        pauliSim.ApplyY(op.qubit1);
        break;
      case 2:
        pauliSim.ApplyZ(op.qubit1);
        break;
      case 3:
        pauliSim.ApplyH(op.qubit1);
        break;
      case 4:
        pauliSim.ApplyS(op.qubit1);
        break;
      case 5:
        pauliSim.ApplySDG(op.qubit1);
        break;

      case 6:
        pauliSim.ApplySQRTX(op.qubit1);
        break;
      case 7:
        pauliSim.ApplySxDAG(op.qubit1);
        break;
      case 8:
        pauliSim.ApplyK(op.qubit1);
        break;

      // two qubit gates
      case 9:
        pauliSim.ApplySWAP(op.qubit1, op.qubit2);
        break;
      case 10:
        pauliSim.ApplyCX(op.qubit2, op.qubit1);
        break;
      case 11:
        pauliSim.ApplyCY(op.qubit2, op.qubit1);
        break;
      case 12:
        pauliSim.ApplyCZ(op.qubit2, op.qubit1);
        break;

        // non-clifford single qubit gates

      case 13:
        pauliSim.ApplyP(op.qubit1, op.theta);
        break;
      case 14:
        pauliSim.ApplyRX(op.qubit1, op.theta);
        break;
      case 15:
        pauliSim.ApplyRY(op.qubit1, op.theta);
        break;
      case 16:
        pauliSim.ApplyRZ(op.qubit1, op.theta);
        break;
      case 17:
        pauliSim.ApplyU(op.qubit1, op.theta, op.phi, op.lambda, op.gamma);
        break;

      case 18:
        pauliSim.ApplyT(op.qubit1);
        break;
      case 19:
        pauliSim.ApplyTDG(op.qubit1);
        break;

        // non-clifford two qubit gates

      case 20:
        pauliSim.ApplyCH(op.qubit2, op.qubit1);
        break;

      case 21:
        pauliSim.ApplyCRZ(op.qubit2, op.qubit1, op.theta);
        break;
      case 22:
        pauliSim.ApplyCRY(op.qubit2, op.qubit1, op.theta);
        break;

      case 23:
        pauliSim.ApplyCRX(op.qubit2, op.qubit1, op.theta);
        break;



      case 24:
        pauliSim.ApplyCP(op.qubit2, op.qubit1, op.theta);
        break;
      case 25:
        pauliSim.ApplyCSX(op.qubit2, op.qubit1);
        break;
      case 26:
        pauliSim.ApplyCSXDAG(op.qubit2, op.qubit1);
        break;

      case 27:
        pauliSim.ApplyCU(op.qubit2, op.qubit1, op.theta, op.phi, op.lambda, op.gamma);
        break;

      // non-clifford three qubit gates
      case 28:
        pauliSim.ApplyCCX(op.qubit3, op.qubit2, op.qubit1);
        break;
      case 29:
        pauliSim.ApplyCSwap(op.qubit3, op.qubit2, op.qubit1);
        break;
      default:
        std::cerr << "Unknown gate id: " << op.gate << std::endl;
    }
   }
#endif


  const unsigned int nrQubitsForRandomCirc = 4;

  std::shared_ptr<Simulators::ISimulator> statevectorSim;
  Simulators::QcsimPauliPropagator qcsimPauliSim;
#ifdef __linux__
  std::shared_ptr<Simulators::GpuPauliPropagator> gpuPauliSim;
#endif
  Circuits::OperationState state;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(pauli_propagation_tests)

BOOST_FIXTURE_TEST_CASE(PauliInitTests, PauliSimTestFixture) {
  BOOST_TEST(statevectorSim);
  BOOST_TEST(qcsimPauliSim.GetNrQubits() == nrQubitsForRandomCirc);
}


BOOST_DATA_TEST_CASE_F(PauliSimTestFixture, RandomCliffordCircuitsTest, bdata::xrange(1, 20), nrGates) {
  auto circuit = GenerateCircuit(nrQubitsForRandomCirc, nrGates);

  // execute
  for (const auto& op : circuit) {
    ExecuteGate(op, statevectorSim);
    ExecuteGate(op, qcsimPauliSim);
#ifdef __linux__
    if (gpuPauliSim) {
      ExecuteGate(op, *gpuPauliSim);
    }
#endif
  }

  // check, first some random pauli strings
  const int nrChecks = 100;
  for (int i = 0; i < nrChecks; ++i) {
    const std::string pauliStr = GeneratePauliString(nrQubitsForRandomCirc);
    const double expValStateVec = statevectorSim->ExpectationValue(pauliStr);
    const double expValPauliSim = qcsimPauliSim.ExpectationValue(pauliStr);
    BOOST_TEST(std::abs(expValStateVec - expValPauliSim) < 1e-7,
               "Expectation value mismatch for pauli string "
                   << pauliStr << ": statevector " << expValStateVec
                   << ", pauli sim " << expValPauliSim);
#ifdef __linux__
    if (gpuPauliSim) {
      const double expValGpuPauliSim = gpuPauliSim->ExpectationValue(pauliStr);
      BOOST_TEST(std::abs(expValStateVec - expValGpuPauliSim) < 1e-7,
                 "Expectation value mismatch for pauli string "
                     << pauliStr << ": statevector " << expValStateVec
                     << ", gpu pauli sim " << expValGpuPauliSim);
    }
#endif
  }

  // now sampling
  const int nrSamples = 1000;
  Types::qubits_vector qubitsToMeasure(nrQubitsForRandomCirc);
  std::iota(qubitsToMeasure.begin(), qubitsToMeasure.end(), 0);

  // perform sampling
  auto svRes = statevectorSim->SampleCounts(qubitsToMeasure, nrSamples);

  std::vector<int> pq(qubitsToMeasure.begin(), qubitsToMeasure.end());

  std::unordered_map<Types::qubit_t, Types::qubit_t> qcsimRes;

  std::random_device rd;
  std::mt19937 g(rd());

  for (int i = 0; i < nrSamples; ++i) {
    std::shuffle(pq.begin(), pq.end(), g);
    auto res = qcsimPauliSim.Sample(pq);

    Types::qubit_t result = 0;
    for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
      if (res[q])
         result |= (1ULL << pq[q]);
    }
    ++qcsimRes[result];
  }

  // compare results
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t psCount =
        qcsimRes.find(key) != qcsimRes.end() ? qcsimRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double psProb = static_cast<double>(psCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - psProb) < 0.1,
               "Sampling probability mismatch for outcome "
                   << key << ": statevector " << svProb
                   << ", pauli sim " << psProb);
  }

  // now the same for gpu sim
#ifdef __linux__
  if (gpuPauliSim) {
    std::unordered_map<Types::qubit_t, Types::qubit_t> gpuRes;
    for (int i = 0; i < nrSamples; ++i) {
      std::shuffle(pq.begin(), pq.end(), g);
      auto res = gpuPauliSim->SampleQubits(pq);
      Types::qubit_t result = 0;
      for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
        if (res[q])
          result |= (1ULL << pq[q]);
      }
      ++gpuRes[result];
    }
    // compare results
    for (const auto& kv : svRes) {
      const Types::qubit_t key = kv.first;
      const Types::qubit_t svCount = kv.second;
      const Types::qubit_t psCount =
          gpuRes.find(key) != gpuRes.end() ? gpuRes[key] : 0;
      const double svProb = static_cast<double>(svCount) / nrSamples;
      const double psProb = static_cast<double>(psCount) / nrSamples;
      BOOST_TEST(std::abs(svProb - psProb) < 0.1,
                 "Sampling probability mismatch for outcome "
                     << key << ": statevector " << svProb
                     << ", gpu pauli sim " << psProb);
    }
  }
#endif

  // similar, but using measurements instead of sampling, to check that as well
  // first, qcsim
  qcsimRes.clear();
  qcsimPauliSim.SaveState();

  for (int i = 0; i < nrSamples; ++i) {
    qcsimPauliSim.RestoreState();
    std::shuffle(pq.begin(), pq.end(), g);
    auto res = qcsimPauliSim.Measure(pq);
    Types::qubit_t result = 0;
    for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
      if (res[q])
        result |= (1ULL << pq[q]);
    }
    ++qcsimRes[result];
  }

  // compare results
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t psCount =
        qcsimRes.find(key) != qcsimRes.end() ? qcsimRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double psProb = static_cast<double>(psCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - psProb) < 0.1,
               "Measurement probability mismatch for outcome "
                   << key << ": statevector " << svProb
                   << ", pauli sim " << psProb);
  }

  // the same for gpu pauli sim  

 #ifdef __linux__
  if (gpuPauliSim) {
    std::unordered_map<Types::qubit_t, Types::qubit_t> gpuRes;
    gpuPauliSim->SaveState();
    for (int i = 0; i < nrSamples; ++i) {
      gpuPauliSim->RestoreState();
      std::shuffle(pq.begin(), pq.end(), g);
     
      Types::qubit_t result = 0;
      for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
        auto res = gpuPauliSim->MeasureQubit(pq[q]);
        if (res)
          result |= (1ULL << pq[q]);
      }
      ++gpuRes[result];
    }
    // compare results
    for (const auto& kv : svRes) {
      const Types::qubit_t key = kv.first;
      const Types::qubit_t svCount = kv.second;
      const Types::qubit_t psCount =
          gpuRes.find(key) != gpuRes.end() ? gpuRes[key] : 0;
      const double svProb = static_cast<double>(svCount) / nrSamples;
      const double psProb = static_cast<double>(psCount) / nrSamples;
      BOOST_TEST(std::abs(svProb - psProb) < 0.1,
                 "Measurement probability mismatch for outcome "
                     << key << ": statevector " << svProb
                     << ", gpu pauli sim " << psProb);
    }
  }
#endif
}


BOOST_DATA_TEST_CASE_F(PauliSimTestFixture, RandomNonCliffordCircuitsTest,
                       bdata::xrange(1, 20), nrGates) {
  auto circuit = GenerateCircuit(nrQubitsForRandomCirc, nrGates, 29);

  // execute
  for (const auto& op : circuit) {
    ExecuteGate(op, statevectorSim);
    ExecuteGate(op, qcsimPauliSim);
#ifdef __linux__
    if (gpuPauliSim) {
      ExecuteGate(op, *gpuPauliSim);
    }
#endif
  }

  // check, first some random pauli strings
  const int nrChecks = 100;
  for (int i = 0; i < nrChecks; ++i) {
    const std::string pauliStr = GeneratePauliString(nrQubitsForRandomCirc);
    const double expValStateVec = statevectorSim->ExpectationValue(pauliStr);
    const double expValPauliSim = qcsimPauliSim.ExpectationValue(pauliStr);
    BOOST_TEST(std::abs(expValStateVec - expValPauliSim) < 1e-7,
               "Expectation value mismatch for pauli string "
                   << pauliStr << ": statevector " << expValStateVec
                   << ", pauli sim " << expValPauliSim);
#ifdef __linux__
    if (gpuPauliSim) {
      const double expValGpuPauliSim = gpuPauliSim->ExpectationValue(pauliStr);
      BOOST_TEST(std::abs(expValStateVec - expValGpuPauliSim) < 1e-7,
                 "Expectation value mismatch for pauli string "
                     << pauliStr << ": statevector " << expValStateVec
                     << ", gpu pauli sim " << expValGpuPauliSim);
    }
#endif
  }

  // now sampling
  const int nrSamples = 1000;
  Types::qubits_vector qubitsToMeasure(nrQubitsForRandomCirc);
  std::iota(qubitsToMeasure.begin(), qubitsToMeasure.end(), 0);

  // perform sampling
  auto svRes = statevectorSim->SampleCounts(qubitsToMeasure, nrSamples);

  std::vector<int> pq(qubitsToMeasure.begin(), qubitsToMeasure.end());

  std::unordered_map<Types::qubit_t, Types::qubit_t> qcsimRes;

  std::random_device rd;
  std::mt19937 g(rd());

  for (int i = 0; i < nrSamples; ++i) {
    std::shuffle(pq.begin(), pq.end(), g);
    auto res = qcsimPauliSim.Sample(pq);

    Types::qubit_t result = 0;
    for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
      if (res[q]) result |= (1ULL << pq[q]);
    }
    ++qcsimRes[result];
  }

  // compare results
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t psCount =
        qcsimRes.find(key) != qcsimRes.end() ? qcsimRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double psProb = static_cast<double>(psCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - psProb) < 0.1,
               "Sampling probability mismatch for outcome "
                   << key << ": statevector " << svProb << ", pauli sim "
                   << psProb);
  }

  // now the same for gpu sim
#ifdef __linux__
  if (gpuPauliSim) {
    std::unordered_map<Types::qubit_t, Types::qubit_t> gpuRes;
    for (int i = 0; i < nrSamples; ++i) {
      std::shuffle(pq.begin(), pq.end(), g);
      auto res = gpuPauliSim->SampleQubits(pq);
      Types::qubit_t result = 0;
      for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
        if (res[q]) result |= (1ULL << pq[q]);
      }
      ++gpuRes[result];
    }
    // compare results
    for (const auto& kv : svRes) {
      const Types::qubit_t key = kv.first;
      const Types::qubit_t svCount = kv.second;
      const Types::qubit_t psCount =
          gpuRes.find(key) != gpuRes.end() ? gpuRes[key] : 0;
      const double svProb = static_cast<double>(svCount) / nrSamples;
      const double psProb = static_cast<double>(psCount) / nrSamples;
      BOOST_TEST(std::abs(svProb - psProb) < 0.1,
                 "Sampling probability mismatch for outcome "
                     << key << ": statevector " << svProb << ", gpu pauli sim "
                     << psProb);
    }
  }
#endif

  // similar, but using measurements instead of sampling, to check that as well
  // first, qcsim
  qcsimRes.clear();
  qcsimPauliSim.SaveState();

  for (int i = 0; i < nrSamples; ++i) {
    qcsimPauliSim.RestoreState();
    std::shuffle(pq.begin(), pq.end(), g);
    auto res = qcsimPauliSim.Measure(pq);
    Types::qubit_t result = 0;
    for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
      if (res[q]) result |= (1ULL << pq[q]);
    }
    ++qcsimRes[result];
  }

  // compare results
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t psCount =
        qcsimRes.find(key) != qcsimRes.end() ? qcsimRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double psProb = static_cast<double>(psCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - psProb) < 0.1,
               "Measurement probability mismatch for outcome "
                   << key << ": statevector " << svProb << ", pauli sim "
                   << psProb);
  }

  // the same for gpu pauli sim

#ifdef __linux__
  if (gpuPauliSim) {
    std::unordered_map<Types::qubit_t, Types::qubit_t> gpuRes;
    gpuPauliSim->SaveState();
    for (int i = 0; i < nrSamples; ++i) {
      gpuPauliSim->RestoreState();
      std::shuffle(pq.begin(), pq.end(), g);

      Types::qubit_t result = 0;
      for (int q = 0; q < static_cast<int>(pq.size()); ++q) {
        auto res = gpuPauliSim->MeasureQubit(pq[q]);
        if (res) result |= (1ULL << pq[q]);
      }
      ++gpuRes[result];
    }
    // compare results
    for (const auto& kv : svRes) {
      const Types::qubit_t key = kv.first;
      const Types::qubit_t svCount = kv.second;
      const Types::qubit_t psCount =
          gpuRes.find(key) != gpuRes.end() ? gpuRes[key] : 0;
      const double svProb = static_cast<double>(svCount) / nrSamples;
      const double psProb = static_cast<double>(psCount) / nrSamples;
      BOOST_TEST(std::abs(svProb - psProb) < 0.1,
                 "Measurement probability mismatch for outcome "
                     << key << ": statevector " << svProb << ", gpu pauli sim "
                     << psProb);
    }
  }
#endif
}


BOOST_AUTO_TEST_SUITE_END()
