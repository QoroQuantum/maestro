/**
 * @file expvals.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for expectation values.
 *
 * Randomly generated Pauli strings, computing expectation value for various
 * simulators and comparison among them.
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

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Circuit/CausalCone.h"
#include "../Simulators/Factory.h"  // project being tested

#include "../Network/SimpleDisconnectedNetwork.h"

struct ExpvalTestFixture {
  ExpvalTestFixture() {
#ifdef __linux__
    Simulators::SimulatorsFactory::InitGpuLibrary();

    gpuStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kStatevector);
    if (gpuStatevector) {
      gpuStatevector->AllocateQubits(nrQubits);
      gpuStatevector->Initialize();
    }

    gpuMPS = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kMatrixProductState);
    if (gpuMPS) {
      gpuMPS->AllocateQubits(nrQubits);
      gpuMPS->Initialize();
    }
    gpuTN = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kTensorNetwork);
    if (gpuTN) {
      gpuTN->AllocateQubits(nrQubits);
      gpuTN->Initialize();
    }
#endif

    Simulators::SimulatorsFactory::InitQuestLibrary();

    questStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQuestSim,
        Simulators::SimulationType::kStatevector);
    if (questStatevector) {
      questStatevector->AllocateQubits(nrQubits);
      questStatevector->Initialize();
    }

    state.AllocateBits(nrQubits);

#ifndef NO_QISKIT_AER
    aerStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kStatevector);
    aerStatevector->AllocateQubits(nrQubits);
    aerStatevector->Initialize();

    aerComposite = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kCompositeQiskitAer,
        Simulators::SimulationType::kStatevector);
    aerComposite->AllocateQubits(nrQubits);
    aerComposite->Initialize();

    aerMPS = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kMatrixProductState);
    aerMPS->AllocateQubits(nrQubits);
    aerMPS->Initialize();
#endif

    qcsimStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qcsimStatevector->AllocateQubits(nrQubits);
    qcsimStatevector->Initialize();

    qcsimStatevectorBig = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kCompositeQCSim,
        Simulators::SimulationType::
            kStatevector);  // runs distributed circuits, it should be faster
                            // than the regular one
    qcsimStatevectorBig->AllocateQubits(nrQubits + 5);
    qcsimStatevectorBig->Initialize();

    qcsimComposite = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kCompositeQCSim,
        Simulators::SimulationType::kStatevector);
    qcsimComposite->AllocateQubits(nrQubits);
    qcsimComposite->Initialize();

    qcsimMPS = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kMatrixProductState);
    qcsimMPS->AllocateQubits(nrQubits);
    qcsimMPS->Initialize();

#ifndef NO_QISKIT_AER
    aerClifford = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kStabilizer);
    aerClifford->AllocateQubits(nrQubits);
    aerClifford->Initialize();
#endif

    qcsimClifford = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStabilizer);
    qcsimClifford->AllocateQubits(nrQubits);
    qcsimClifford->Initialize();

    qcTensor = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kTensorNetwork);
    qcTensor->AllocateQubits(nrQubits);
    qcTensor->Initialize();

    randomCirc = Circuits::CircuitFactory<>::CreateCircuit();

    resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

    resetRandomCircBig = std::make_shared<Circuits::Circuit<>>();
    Types::qubits_vector qubitsBig(nrQubits + 5);
    std::iota(qubitsBig.begin(), qubitsBig.end(), 0);
    resetRandomCircBig->AddOperation(
        std::make_shared<Circuits::Reset<>>(qubitsBig));

    // networks part, test only statevector, the other ones are tested against
    // statevector so they should work with the networks, too
    std::vector<Types::qubit_t> networkBits{
        3, (Types::qubit_t)nrQubits,
        2};  // 3 hosts = 10 qubits + 3 entangled qubits = 13 qubits
    std::vector<size_t> networkCbits(networkBits.begin(), networkBits.end());

    networkSim1 = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
        networkBits, networkCbits);
    networkSim1->CreateSimulator(/*Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector*/);
    // networkSim1->SetOptimizeSimulator(false);
  }

  ~ExpvalTestFixture() {}

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
          pauli[i] = 'X';
          break;
        case 1:
          pauli[i] = 'Y';
          break;
        case 2:
          pauli[i] = 'Z';
          break;
        case 3:
          pauli[i] = 'I';
          break;
        default:
          pauli[i] = 'I';
          break;
      }
    }

    return pauli;
  }

  void GenerateCircuit(int nrGates, int nrQubits) {
    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
    auto gateGenIter = gateGen.begin();

    // TODO: Maybe insert from time to time a random number generating 'gate'
    // and a conditioned random one, those should not affect results?
    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      // create a random gate and add it to the circuit

      // first, pick randomly three qubits (depending on the randomly chosen
      // gate type, not all of them will be used)
      Types::qubits_vector qubits(nrQubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      std::shuffle(qubits.begin(), qubits.end(), g);
      auto q1 = qubits[0];
      auto q2 = qubits[1];
      auto q3 = qubits[2];

      // now some random parameters, again, they might be ignored
      const double param1 = *dblGenIter;
      ++dblGenIter;
      const double param2 = *dblGenIter;
      ++dblGenIter;
      const double param3 = *dblGenIter;
      ++dblGenIter;
      const double param4 = *dblGenIter;
      ++dblGenIter;

      const Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);

      auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3, param4);
      randomCirc->AddOperation(theGate);
    }
  }

  void GenerateCliffordCircuit(int nrGates, int nrQubits) {
    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(Circuits::QuantumGateType::kCZGateType));
    auto gateGenIter = gateGen.begin();

    // TODO: Maybe insert from time to time a random number generating 'gate'
    // and a conditioned random one, those should not affect results?
    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      // create a random gate and add it to the circuit

      // first, pick randomly three qubits (depending on the randomly chosen
      // gate type, not all of them will be used)
      Types::qubits_vector qubits(nrQubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      std::shuffle(qubits.begin(), qubits.end(), g);
      auto q1 = qubits[0];
      auto q2 = qubits[1];

      // now some random parameters, again, they might be ignored
      const double param1 = *dblGenIter;
      ++dblGenIter;

      Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);

      auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0,
                                                            param1, 0, 0, 0);
      if (!theGate->IsClifford()) {
        --gateNr;
        continue;
      }

      randomCirc->AddOperation(theGate);
    }
  }

  int nrQubits = 4;
  int nrCircuitsLimit = 30;
  int nrPauliLimit = 30;

  std::shared_ptr<Circuits::Circuit<>> randomCirc;
  std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
  std::shared_ptr<Circuits::Circuit<>> resetRandomCircBig;
  Circuits::OperationState state;

  std::shared_ptr<Simulators::ISimulator> aerStatevector;
  std::shared_ptr<Simulators::ISimulator> qcsimStatevector;
  std::shared_ptr<Simulators::ISimulator> qcsimStatevectorBig;

  std::shared_ptr<Simulators::ISimulator> aerComposite;
  std::shared_ptr<Simulators::ISimulator> qcsimComposite;

#ifdef __linux__
  std::shared_ptr<Simulators::ISimulator> gpuStatevector;
  std::shared_ptr<Simulators::ISimulator> gpuMPS;
  std::shared_ptr<Simulators::ISimulator> gpuTN;
#endif

  std::shared_ptr<Simulators::ISimulator> questStatevector;

  std::shared_ptr<Simulators::ISimulator> aerMPS;
  std::shared_ptr<Simulators::ISimulator> qcsimMPS;

  std::shared_ptr<Simulators::ISimulator> qcTensor;

  std::shared_ptr<Simulators::ISimulator> aerClifford;
  std::shared_ptr<Simulators::ISimulator> qcsimClifford;

  std::shared_ptr<Network::SimpleDisconnectedNetwork<>> networkSim1;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

BOOST_AUTO_TEST_SUITE(expval_tests)

BOOST_FIXTURE_TEST_CASE(ExpvalInitializationTests, ExpvalTestFixture) {
#ifndef NO_QISKIT_AER
  BOOST_TEST(aerStatevector);
  BOOST_TEST(aerComposite);
  BOOST_TEST(aerMPS);
  BOOST_TEST(aerClifford);
#endif
  BOOST_TEST(qcsimStatevector);
  BOOST_TEST(qcsimStatevectorBig);
  BOOST_TEST(qcsimComposite);
  BOOST_TEST(qcsimMPS);
  BOOST_TEST(qcsimClifford);

  BOOST_TEST(qcTensor);

  BOOST_TEST(randomCirc);
  BOOST_TEST(resetRandomCirc);
  BOOST_TEST(resetRandomCircBig);

  BOOST_TEST(networkSim1);
}

BOOST_DATA_TEST_CASE_F(ExpvalTestFixture, NormalSimulatorsTest,
                       bdata::xrange(1, 20), nrGates) {
  const double precision = 0.00001;
  const double precisionGPU = 0.01;
  const double precisionMPS = 0.001;

  for (int i = 0; i < nrCircuitsLimit; ++i) {
    GenerateCircuit(nrGates, nrQubits);

#ifndef NO_QISKIT_AER
    if (aerStatevector) randomCirc->Execute(aerStatevector, state);
    if (aerComposite) randomCirc->Execute(aerComposite, state);
    if (aerMPS) randomCirc->Execute(aerMPS, state);
#endif
    randomCirc->Execute(qcsimStatevector, state);

    randomCirc->Execute(qcsimComposite, state);

    randomCirc->Execute(qcsimMPS, state);

#ifdef __linux__
    if (gpuStatevector) randomCirc->Execute(gpuStatevector, state);
    if (gpuMPS) randomCirc->Execute(gpuMPS, state);
    if (gpuTN) randomCirc->Execute(gpuTN, state);
#endif

    if (questStatevector) randomCirc->Execute(questStatevector, state);

    randomCirc->Execute(qcTensor, state);

    for (int j = 0; j < nrPauliLimit; ++j) {
      const std::string pauli = GeneratePauliString(nrQubits);

      const double qcsimStatevectorVal =
          qcsimStatevector->ExpectationValue(pauli);

#ifndef NO_QISKIT_AER
      if (aerStatevector) {
        const double aerStatevectorVal = aerStatevector->ExpectationValue(pauli);
        BOOST_CHECK_PREDICATE(
            checkClose, (aerStatevectorVal)(qcsimStatevectorVal)(precision));
      }

      if (aerComposite) {
        const double aerCompVal = aerComposite->ExpectationValue(pauli);
        BOOST_CHECK_PREDICATE(checkClose,
                              (aerCompVal)(qcsimStatevectorVal)(precision));
      }

      if (aerMPS) {
        const double aerMPSVal = aerMPS->ExpectationValue(pauli);
        BOOST_CHECK_PREDICATE(checkClose,
                              (aerMPSVal)(qcsimStatevectorVal)(precisionMPS));
      }
#endif

      const double qcsimCompVal = qcsimComposite->ExpectationValue(pauli);
      BOOST_CHECK_PREDICATE(checkClose,
                            (qcsimCompVal)(qcsimStatevectorVal)(precision));

      const double qcsimMPSVal = qcsimMPS->ExpectationValue(pauli);
      BOOST_CHECK_PREDICATE(checkClose,
                            (qcsimMPSVal)(qcsimStatevectorVal)(precisionMPS));

      const double qcTensorVal = qcTensor->ExpectationValue(pauli);
      BOOST_CHECK_PREDICATE(checkClose,
                            (qcTensorVal)(qcsimStatevectorVal)(precision));

#ifdef __linux__
      if (gpuStatevector) {
        const double gpuStatevectorVal =
            gpuStatevector->ExpectationValue(pauli);
        BOOST_REQUIRE_PREDICATE(
            checkClose, (gpuStatevectorVal)(qcsimStatevectorVal)(precisionGPU));
      }

      if (gpuMPS) {
        const double gpuMPSVal = gpuMPS->ExpectationValue(pauli);
        BOOST_REQUIRE_PREDICATE(checkClose,
                                (gpuMPSVal)(qcsimStatevectorVal)(precisionMPS));
      }
      if (gpuTN) {
        const double gpuTNVal = gpuTN->ExpectationValue(pauli);
        BOOST_REQUIRE_PREDICATE(checkClose,
                                (gpuTNVal)(qcsimStatevectorVal)(precisionMPS));
      }
#endif

      if (questStatevector) {
        const double questStatevectorVal =
            questStatevector->ExpectationValue(pauli);
        BOOST_CHECK_PREDICATE(
            checkClose, (questStatevectorVal)(qcsimStatevectorVal)(precision));
      }
    }

#ifndef NO_QISKIT_AER
    if (aerStatevector) resetRandomCirc->Execute(aerStatevector, state);
    if (aerComposite) resetRandomCirc->Execute(aerComposite, state);
    if (aerMPS) resetRandomCirc->Execute(aerMPS, state);
#endif
    resetRandomCirc->Execute(qcsimStatevector, state);

    resetRandomCirc->Execute(qcsimComposite, state);

    resetRandomCirc->Execute(qcsimMPS, state);

#ifdef __linux__
    if (gpuStatevector) resetRandomCirc->Execute(gpuStatevector, state);
    if (gpuMPS) resetRandomCirc->Execute(gpuMPS, state);
    if (gpuTN) resetRandomCirc->Execute(gpuTN, state);
#endif

    if (questStatevector) resetRandomCirc->Execute(questStatevector, state);

    qcTensor->Clear();
    qcTensor->AllocateQubits(nrQubits);
    qcTensor->Initialize();

    randomCirc->Clear();
  }
}

BOOST_DATA_TEST_CASE_F(ExpvalTestFixture, CliffordSimulatorsTest,
                       bdata::xrange(1, 20), nrGates) {
  const double precision = 0.00000001;

  for (int i = 0; i < 30; ++i) {
    GenerateCliffordCircuit(nrGates, nrQubits);

    randomCirc->Execute(qcsimStatevector, state);
#ifndef NO_QISKIT_AER
    if (aerClifford) randomCirc->Execute(aerClifford, state);
#endif
    randomCirc->Execute(qcsimClifford, state);

    for (int j = 0; j < 30; ++j) {
      const std::string pauli = GeneratePauliString(nrQubits);

      const double qcsimStatevectorVal =
          qcsimStatevector->ExpectationValue(pauli);
      const double qcsimCliffordVal = qcsimClifford->ExpectationValue(pauli);

      BOOST_CHECK_PREDICATE(checkClose,
                            (qcsimCliffordVal)(qcsimStatevectorVal)(precision));

#ifndef NO_QISKIT_AER
      if (aerClifford) {
        const double aerCliffordVal = aerClifford->ExpectationValue(pauli);
        BOOST_CHECK_PREDICATE(checkClose,
                              (aerCliffordVal)(qcsimStatevectorVal)(precision));
      }
#endif
    }

    resetRandomCirc->Execute(qcsimStatevector, state);
#ifndef NO_QISKIT_AER
    if (aerClifford) resetRandomCirc->Execute(aerClifford, state);
#endif
    resetRandomCirc->Execute(qcsimClifford, state);

    randomCirc->Clear();
  }
}

BOOST_DATA_TEST_CASE_F(ExpvalTestFixture, NetworkExpectationTest,
                       bdata::xrange(1, 20), nrGates) {
  const double precision = 0.00001;

  for (int i = 0; i < 30; ++i) {
    GenerateCircuit(nrGates, nrQubits);

    randomCirc->Execute(qcsimStatevector, state);

    for (int j = 0; j < 30; ++j) {
      std::vector<std::string> paulis;
      for (int k = 0; k < 10; ++k)
        paulis.push_back(GeneratePauliString(nrQubits));

      const auto vals =
          networkSim1->ExecuteOnHostExpectations(randomCirc, 1, paulis);

      // now check them against the statevector simulator
      for (int k = 0; k < 10; ++k) {
        const double qcsimStatevectorVal =
            qcsimStatevector->ExpectationValue(paulis[k]);

        BOOST_CHECK_PREDICATE(checkClose,
                              (vals[k])(qcsimStatevectorVal)(precision));
      }

      networkSim1->ExecuteOnHost(resetRandomCirc, 1);
    }

    resetRandomCirc->Execute(qcsimStatevector, state);
    randomCirc->Clear();
  }
}

BOOST_AUTO_TEST_CASE(CausalConeExtractionTest) {
  auto circuit = Circuits::CircuitFactory<>::CreateCircuit();
  // Sub-cluster 1 on qubits 0 and 1
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kHadamardGateType, 0));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kCXGateType, 0, 1));

  // Sub-cluster 2 on qubits 8 and 9 (disjoint)
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kHadamardGateType, 8));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kCXGateType, 8, 9));

  // Observable 1: "ZZIIIIIIII" (only qubits 0 and 1 active)
  std::string pauli1 = "ZZIIIIIIII";
  auto cone1 = Circuits::ExtractObservableCone(circuit, pauli1);

  BOOST_CHECK_EQUAL(cone1.GetNumberOfQubits(), 2);
  BOOST_CHECK_EQUAL(cone1.active_qubits_map[0], 0);
  BOOST_CHECK_EQUAL(cone1.active_qubits_map[1], 1);
  BOOST_CHECK_EQUAL(cone1.active_qubits_map[8],
                    Circuits::ReducedObservableCone<>::inactive_qubit);
  BOOST_CHECK_EQUAL(cone1.reduced_pauli_string, "ZZ");
  BOOST_CHECK_EQUAL(cone1.reduced_circuit->GetOperations().size(), 2);
  BOOST_CHECK_EQUAL(cone1.reduced_circuit->GetMaxQubitIndex(), 1);

  // Observable 2: "IIIIIIIIZZ" (only qubits 8 and 9 active)
  std::string pauli2 = "IIIIIIIIZZ";
  auto cone2 = Circuits::ExtractObservableCone(circuit, pauli2);

  BOOST_CHECK_EQUAL(cone2.GetNumberOfQubits(), 2);
  BOOST_CHECK_EQUAL(cone2.active_qubits_map[8], 0);
  BOOST_CHECK_EQUAL(cone2.active_qubits_map[9], 1);
  BOOST_CHECK_EQUAL(cone2.active_qubits_map[0],
                    Circuits::ReducedObservableCone<>::inactive_qubit);
  BOOST_CHECK_EQUAL(cone2.reduced_pauli_string, "ZZ");
  BOOST_CHECK_EQUAL(cone2.reduced_circuit->GetOperations().size(), 2);
  BOOST_CHECK_EQUAL(cone2.reduced_circuit->GetMaxQubitIndex(), 1);
}

BOOST_AUTO_TEST_CASE(NetworkCausalConeTest) {
  std::vector<Types::qubit_t> hostQubits = {50};
  std::vector<size_t> hostCbits = {50};
  auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(hostQubits, hostCbits);
  network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);

  auto circuit = Circuits::CircuitFactory<>::CreateCircuit();
  // Cluster on 0, 1: creates (|00> + |11>) / sqrt(2)
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kHadamardGateType, 0));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kCXGateType, 0, 1));

  // Cluster on 20, 21: creates (|00> + |11>) / sqrt(2)
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kHadamardGateType, 20));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kCXGateType, 20, 21));

  std::string pauli1 = "ZZ" + std::string(48, 'I');
  std::string pauli2 = std::string(20, 'I') + "XX" + std::string(28, 'I');
  std::string pauli3 = "XX" + std::string(48, 'I');

  auto vals = network->ExecuteOnHostExpectations(circuit, 0, {pauli1, pauli2, pauli3});
  BOOST_CHECK_SMALL(vals[0] - 1.0, 1e-12);
  BOOST_CHECK_SMALL(vals[1] - 1.0, 1e-12);
  BOOST_CHECK_SMALL(vals[2] - 1.0, 1e-12);

  // Verify disabling causal cone reduction yields the same expectations
  network->Configure("enable_causal_cone_reduction", "false");
  auto valsDisabled = network->ExecuteOnHostExpectations(circuit, 0, {pauli1, pauli2, pauli3});
  BOOST_CHECK_CLOSE(valsDisabled[0], 1.0, 1e-6);
  BOOST_CHECK_CLOSE(valsDisabled[1], 1.0, 1e-6);
  BOOST_CHECK_CLOSE(valsDisabled[2], 1.0, 1e-6);
}

BOOST_AUTO_TEST_CASE(CausalConeHundredQubitsTest) {
  using F = Circuits::CircuitFactory<>;
  using G = Circuits::QuantumGateType;
  auto circuit = F::CreateCircuit();
  for (size_t q = 0; q < 100; q += 2) {
    circuit->AddOperation(F::CreateGate(G::kHadamardGateType, q));
    circuit->AddOperation(F::CreateGate(G::kCXGateType, q, q + 1));
  }
  auto cone = Circuits::ExtractObservableCone(circuit, "ZZ" + std::string(98, 'I'));
  BOOST_CHECK_EQUAL(cone.GetNumberOfQubits(), 2);
  BOOST_CHECK_EQUAL(cone.reduced_circuit->GetMaxQubitIndex(), 1);
  BOOST_CHECK_EQUAL(cone.reduced_circuit->size(), 2);
  BOOST_CHECK_EQUAL(cone.active_qubits_map[99], decltype(cone)::inactive_qubit);
}

BOOST_AUTO_TEST_CASE(CausalConeBackendEquivalenceTest) {
  using F = Circuits::CircuitFactory<>;
  using G = Circuits::QuantumGateType;
  using S = Simulators::SimulationType;
  for (auto method : {S::kStabilizer, S::kPauliPropagator, S::kStatevector}) {
    auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
        std::vector<Types::qubit_t>{6}, std::vector<size_t>{6});
    network->SetOptimizeSimulator(false); // Exercise the requested backend.
    network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
    for (int trial = 0; trial < 10; ++trial) {
      auto circuit = F::CreateCircuit();
      for (size_t q = 0; q < 6; ++q) {
        circuit->AddOperation(F::CreateGate(G::kHadamardGateType, q));
        circuit->AddOperation(F::CreateGate(G::kSGateType, q));
        if (method != S::kStabilizer)
          circuit->AddOperation(F::CreateGate(G::kRyGateType, q, 0, 0,
                                             0.13 * (trial + q)));
      }
      circuit->AddOperation(F::CreateGate(G::kCXGateType, 4, 2));
      circuit->AddOperation(F::CreateGate(G::kCXGateType, 2, 0));
      // This later gate is outside Z0's backward cone, despite touching q4.
      circuit->AddOperation(F::CreateGate(G::kCXGateType, 4, 5));
      auto cone = Circuits::ExtractObservableCone(circuit, "ZIIIII");
      BOOST_CHECK_EQUAL(cone.GetNumberOfQubits(), 3);
      BOOST_CHECK_EQUAL(cone.active_qubits_map[0], 0);
      BOOST_CHECK_EQUAL(cone.active_qubits_map[2], 1);
      BOOST_CHECK_EQUAL(cone.active_qubits_map[4], 2);
      BOOST_CHECK_EQUAL(cone.active_qubits_map[5], decltype(cone)::inactive_qubit);
      const std::vector<std::string> observables{
          "ZIIIII", "YIIIII", "IIXIII", "IIIIII", "ZZZZZZ"};
      network->Configure("enable_causal_cone_reduction", "false");
      auto full = network->ExecuteOnHostExpectations(circuit, 0, observables);
      network->Configure("enable_causal_cone_reduction", "true");
      auto reduced = network->ExecuteOnHostExpectations(circuit, 0, observables);
      for (size_t i = 0; i < full.size(); ++i)
        BOOST_CHECK_SMALL(full[i] - reduced[i], 1e-12);
      BOOST_CHECK(network->GetLastSimulationType() == method);
    }
  }
}

BOOST_AUTO_TEST_CASE(CausalConeNonunitaryFallbackTest) {
  using F = Circuits::CircuitFactory<>;
  using G = Circuits::QuantumGateType;
  for (bool noise : {false, true}) {
    auto circuit = F::CreateCircuit();
    if (noise) {
      circuit->AddOperation(F::CreateGate(G::kXGateType, 0));
      circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(
          Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(0.25)));
    } else {
      circuit->AddOperation(F::CreateGate(G::kXGateType, 1));
      circuit->AddOperation(F::CreateMeasurement({{1, 2}}));
      circuit->AddOperation(F::CreateSimpleConditionalGate(
          F::CreateGate(G::kXGateType, 0), 2));
    }
    BOOST_CHECK(!Circuits::SupportsObservableCone(circuit));
    auto cone = Circuits::ExtractObservableCone(circuit, "ZII");
    BOOST_CHECK(cone.reduced_circuit == circuit);
    BOOST_CHECK_EQUAL(cone.GetNumberOfQubits(), 3);
    auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
        std::vector<Types::qubit_t>{3}, std::vector<size_t>{3});
    network->SetOptimizeSimulator(false);
    network->CreateSimulator(Simulators::SimulatorType::kQCSim,
        noise ? Simulators::SimulationType::kDensityMatrix
              : Simulators::SimulationType::kStatevector);
    for (bool enabled : {false, true}) {
      network->Configure("enable_causal_cone_reduction", enabled ? "true" : "false");
      auto result = network->ExecuteOnHostExpectations(circuit, 0, {"ZII"});
      BOOST_CHECK_SMALL(result[0] - (noise ? -0.5 : -1.), 1e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(CausalConeIdentityAndIdleQubitTest) {
  auto circuit = Circuits::CircuitFactory<>::CreateCircuit();
  auto identity = Circuits::ExtractObservableCone(circuit, "IIII");
  BOOST_CHECK_EQUAL(identity.GetNumberOfQubits(), 0);
  BOOST_CHECK(identity.reduced_circuit->GetOperations().empty());
  auto idle = Circuits::ExtractObservableCone(circuit, "IIIZ");
  BOOST_CHECK_EQUAL(idle.GetNumberOfQubits(), 1);
  BOOST_CHECK_EQUAL(idle.active_qubits_map[3], 0);
  BOOST_CHECK_EQUAL(idle.reduced_pauli_string, "Z");
}

BOOST_AUTO_TEST_SUITE_END()
