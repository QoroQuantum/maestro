/**
 * @file tensortests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for tensors/tensor networks.
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

#include "../Circuit/Circuit.h"

#include "../Circuit/Conditional.h"
#include "../Circuit/Measurements.h"
#include "../Circuit/QuantumGates.h"
#include "../Circuit/RandomOp.h"
#include "../Circuit/Reset.h"
#include "../Circuit/Factory.h"

#include "../TensorNetworks/TensorNetwork.h"
#include "../TensorNetworks/StochasticContractor.h"
#include "../TensorNetworks/VerticalContractor.h"
#include "../TensorNetworks/ForestContractor.h"
#include "../TensorNetworks/DumbContractor.h"
#include "../TensorNetworks/LookaheadContractor.h"

struct TensorsTestFixture {
  TensorsTestFixture() : g(std::random_device{}()), boolDist(0.5) {
    // setup
    randomCirc = std::make_shared<Circuits::Circuit<>>();

    qc = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qc->AllocateQubits(nrQubits);
    qc->Initialize();
    qcForest = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qcForest->AllocateQubits(nrQubitsForest);
    qcForest->Initialize();

    qcTensor = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kTensorNetwork);

    tensorNetworkOneQubit = std::make_shared<TensorNetworks::TensorNetwork>(1);

    tensorNetwork = std::make_shared<TensorNetworks::TensorNetwork>(nrQubits);
    // const auto tensorContractor =
    // std::make_shared<TensorNetworks::StochasticContractor>(); const auto
    // tensorContractor =
    // std::make_shared<TensorNetworks::VerticalContractor>();
    const auto tensorContractor =
        std::make_shared<TensorNetworks::ForestContractor>();
    // const auto tensorContractor =
    // std::make_shared<TensorNetworks::DumbContractor>(); const auto
    // tensorContractor =
    // std::make_shared<TensorNetworks::LookaheadContractor>();
    tensorNetwork->SetContractor(tensorContractor);
    tensorNetworkOneQubit->SetContractor(tensorContractor);

    tensorNetworkForest =
        std::make_shared<TensorNetworks::TensorNetwork>(nrQubitsForest);
    const auto tensorContractorForest =
        std::make_shared<TensorNetworks::ForestContractor>();
    tensorNetworkForest->SetContractor(tensorContractorForest);

    qcsimClifford = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStabilizer);
    qcsimClifford->AllocateQubits(nrQubitsForest);
    qcsimClifford->SetMultithreading(false);
    qcsimClifford->Initialize();

    qcsimMPS = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kMatrixProductState);
    qcsimMPS->AllocateQubits(nrQubitsForest);
    qcsimMPS->SetMultithreading(
        false);  // doesn't implement multithreading yet, anyway
    qcsimMPS->Initialize();
  }

  ~TensorsTestFixture() {
    // teardown
  }

  // fills randomly the circuit with gates
  void GenerateCircuit(int nrGates, bool avoidParamGates = false) {
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

      Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);

      if (avoidParamGates &&
          (gateType == Circuits::QuantumGateType::kCUGateType ||
           gateType == Circuits::QuantumGateType::kCPGateType ||
           gateType == Circuits::QuantumGateType::kPhaseGateType ||
           gateType == Circuits::QuantumGateType::kRxGateType ||
           gateType == Circuits::QuantumGateType::kRyGateType ||
           gateType == Circuits::QuantumGateType::kRzGateType ||
           gateType == Circuits::QuantumGateType::kUGateType ||
           gateType == Circuits::QuantumGateType::kCRxGateType ||
           gateType == Circuits::QuantumGateType::kCRyGateType ||
           gateType == Circuits::QuantumGateType::kCRzGateType)) {
        --gateNr;
        continue;
      }

      auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3, param4);
      randomCirc->AddOperation(theGate);
    }

    // get rid of three qubit gates
    randomCirc->ConvertForCutting();
  }

  void GenerateCircuits(int nrGates, bool genThreeQubitGates = false,
                        bool avoidParamGates = false) {
    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(genThreeQubitGates
                                ? Circuits::QuantumGateType::kCCXGateType
                                : Circuits::QuantumGateType::kCUGateType));
    auto gateGenIter = gateGen.begin();

    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);

    // TODO: Maybe insert from time to time a random number generating 'gate'
    // and a conditioned random one, those should not affect results?
    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      // create a random gate and add it to the circuit

      const Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);

      if (avoidParamGates &&
          (gateType == Circuits::QuantumGateType::kCUGateType ||
           gateType == Circuits::QuantumGateType::kCPGateType ||
           gateType == Circuits::QuantumGateType::kPhaseGateType ||
           gateType == Circuits::QuantumGateType::kRxGateType ||
           gateType == Circuits::QuantumGateType::kRyGateType ||
           gateType == Circuits::QuantumGateType::kRzGateType ||
           gateType == Circuits::QuantumGateType::kUGateType ||
           gateType == Circuits::QuantumGateType::kCRxGateType ||
           gateType == Circuits::QuantumGateType::kCRyGateType ||
           gateType == Circuits::QuantumGateType::kCRzGateType)) {
        --gateNr;
        continue;
      }

      // first, pick randomly two qubits (depending on the randomly chosen gate
      // type, not all of them will be used)
      std::shuffle(qubits.begin(), qubits.end(), g);
      const auto q1 = qubits[0];
      const auto q2 = qubits[1];
      const auto q3 = qubits[2];

      // now some random parameters, again, they might be ignored
      const double param1 = *dblGenIter;
      ++dblGenIter;
      const double param2 = *dblGenIter;
      ++dblGenIter;
      const double param3 = *dblGenIter;
      ++dblGenIter;
      const double param4 = *dblGenIter;
      ++dblGenIter;

      const auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3, param4);

      randomCirc->AddOperation(theGate);

      // when three qubit gates are requested, the simulation won't happen
      // directly with the tensor network, but through a simulator, which does
      // transpiling for the three qubit gates this is only used for tests with
      // the tensor network directly
      if (!genThreeQubitGates)
        AddQcSimGate(gateType, q1, q2, param1, param2, param3, param4);
    }
  }

  void GenerateForestCircuits(int nrGates, bool genOnlyClifforGates = false,
                              bool avoidParamGates = false) {
    std::bernoulli_distribution newForestDist(
        0.1);  // produce a new root with a low probability

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
    // auto singleQubitGateGen = bdata::random(0,
    // static_cast<int>(Circuits::QuantumGateType::kUGateType));
    auto gateGenIter = gateGen.begin();
    // auto singleQubitGateGenIter = singleQubitGateGen.begin();

    std::uniform_int_distribution<int> qubitDist(0, nrQubitsForest - 1);

    Types::qubits_vector freequbits(nrQubitsForest);
    std::iota(freequbits.begin(), freequbits.end(), 0);

    Types::qubits_vector usedOnceQubits;

    Types::qubits_vector lastQubits;

    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      // create a random gate and add it to the circuit

      // now some random parameters, again, they might be ignored
      const double param1 = *dblGenIter;
      ++dblGenIter;
      const double param2 = *dblGenIter;
      ++dblGenIter;
      const double param3 = *dblGenIter;
      ++dblGenIter;
      const double param4 = *dblGenIter;
      ++dblGenIter;

      Circuits::QuantumGateType gateType;
      gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

      if (avoidParamGates &&
          (gateType == Circuits::QuantumGateType::kCUGateType ||
           gateType == Circuits::QuantumGateType::kCPGateType ||
           gateType == Circuits::QuantumGateType::kPhaseGateType ||
           gateType == Circuits::QuantumGateType::kRxGateType ||
           gateType == Circuits::QuantumGateType::kRyGateType ||
           gateType == Circuits::QuantumGateType::kRzGateType ||
           gateType == Circuits::QuantumGateType::kUGateType ||
           gateType == Circuits::QuantumGateType::kCRxGateType ||
           gateType == Circuits::QuantumGateType::kCRyGateType ||
           gateType == Circuits::QuantumGateType::kCRzGateType)) {
        --gateNr;
        continue;
      }

      if (genOnlyClifforGates) {
        const auto dummyGate = Circuits::CircuitFactory<>::CreateGate(
            gateType, 0, 1, 0, param1, param2, param3, param4);
        if (!dummyGate->IsClifford()) {
          --gateNr;
          continue;
        }
      }

      // first, pick randomly two qubits (depending on the randomly chosen gate
      // type, not all of them will be used)

      Types::qubit_t q1 = 0;
      Types::qubit_t q2 = 0;

      // can put one qubit gates on any qubit
      if (static_cast<int>(gateType) <=
          static_cast<int>(Circuits::QuantumGateType::kUGateType))
        q1 = qubitDist(g);
      else {
        std::shuffle(freequbits.begin(), freequbits.end(), g);
        std::shuffle(usedOnceQubits.begin(), usedOnceQubits.end(), g);

        const bool first = freequbits.size() > 1 ? newForestDist(g) : false;
        if (first) {
          q1 = freequbits.back();
          freequbits.pop_back();
          q2 = freequbits.back();
          freequbits.pop_back();

          usedOnceQubits.push_back(q1);
          usedOnceQubits.push_back(q2);
        } else {
          if (freequbits.empty()) {
            // pick a pair from 'last qubits'
            q1 = lastQubits[0];
            q2 = lastQubits[1];
          } else {
            q1 = freequbits.back();
            freequbits.pop_back();

            if (!usedOnceQubits.empty()) {
              q2 = usedOnceQubits.back();
              usedOnceQubits.back() = q1;
            } else if (!freequbits.empty()) {
              q2 = freequbits.back();
              freequbits.pop_back();
              usedOnceQubits.push_back(q1);
              usedOnceQubits.push_back(q2);
            } else {
              q2 = (q1 == lastQubits[0]) ? lastQubits[1] : lastQubits[0];
              // usedOnceQubits.push_back(q1);
            }
          }
        }

        /*
        if (freequbits.size() == 1 && usedOnceQubits.empty())
        {
                usedOnceQubits.push_back(freequbits.back());
                freequbits.clear();
        }
        */

        lastQubits = {q1, q2};

        if (boolDist(g)) std::swap(q1, q2);
      }

      const auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, 0, param1, param2, param3, param4);
      randomCirc->AddOperation(theGate);

      AddQcSimGate(gateType, q1, q2, param1, param2, param3, param4);
    }
  }

  void AddQcSimGate(Circuits::QuantumGateType gateType, Types::qubit_t q1,
                    Types::qubit_t q2, double param1, double param2,
                    double param3, double param4) {
    switch (gateType) {
      case Circuits::QuantumGateType::kPhaseGateType: {
        QC::Gates::PhaseShiftGate<> pgate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                pgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kXGateType: {
        QC::Gates::PauliXGate<> xgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                xgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kYGateType: {
        QC::Gates::PauliYGate<> ygate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                ygate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kZGateType: {
        QC::Gates::PauliZGate<> zgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                zgate.getRawOperatorMatrix(), q1));

      } break;
      case Circuits::QuantumGateType::kHadamardGateType: {
        QC::Gates::HadamardGate<> h;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                h.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kSGateType: {
        QC::Gates::SGate<> sgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                sgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kSdgGateType: {
        QC::Gates::SDGGate<> sdggate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                sdggate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kTGateType: {
        QC::Gates::TGate<> tgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                tgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kTdgGateType: {
        QC::Gates::TDGGate<> tdggate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                tdggate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kSxGateType: {
        QC::Gates::SquareRootNOTGate<> sxgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                sxgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kSxDagGateType: {
        QC::Gates::SquareRootNOTDagGate<> sxdaggate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                sxdaggate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kKGateType: {
        QC::Gates::HyGate<> k;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                k.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kRxGateType: {
        QC::Gates::RxGate<> rxgate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                rxgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kRyGateType: {
        QC::Gates::RyGate<> rygate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                rygate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kRzGateType: {
        QC::Gates::RzGate<> rzgate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                rzgate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kUGateType: {
        QC::Gates::UGate<> ugate(param1, param2, param3, param4);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                ugate.getRawOperatorMatrix(), q1));
      } break;
      case Circuits::QuantumGateType::kSwapGateType: {
        QC::Gates::SwapGate<> swapgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                swapgate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCXGateType: {
        QC::Gates::CNOTGate<> cxgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                cxgate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCYGateType: {
        QC::Gates::ControlledYGate<> cygate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                cygate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCZGateType: {
        QC::Gates::ControlledZGate<> czgate;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                czgate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCPGateType: {
        QC::Gates::ControlledPhaseShiftGate<> cpgate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                cpgate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCRxGateType: {
        QC::Gates::ControlledRxGate<> crxgate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                crxgate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCRyGateType: {
        QC::Gates::ControlledRyGate<> crygate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                crygate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCRzGateType: {
        QC::Gates::ControlledRzGate<> crzgate(param1);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                crzgate.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCHGateType: {
        QC::Gates::ControlledHadamardGate<> ch;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                ch.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCSxGateType: {
        QC::Gates::ControlledSquareRootNOTGate<> csx;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                csx.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCSxDagGateType: {
        QC::Gates::ControlledSquareRootNOTDagGate<> csxdag;
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                csxdag.getRawOperatorMatrix(), q1, q2));
      } break;
      case Circuits::QuantumGateType::kCUGateType: {
        QC::Gates::ControlledUGate<> cugate(param1, param2, param3, param4);
        randomQcSimCirc.emplace_back(
            std::make_shared<QC::Gates::AppliedGate<
                TensorNetworks::TensorNode::MatrixClass>>(
                cugate.getRawOperatorMatrix(), q1, q2));
      } break;
      default:
        throw std::runtime_error("Unknown gate type");
        break;
    }
  }

  const unsigned int nrQubits = 5;
  // TODO: To actually be able to test the difference in speed, add
  // paralelization to the speed test for clifford and forest tensor network
  // simulator tests!!!!
  const unsigned int nrQubitsForest =
      12;  // for 16 - sometimes even 15 - the forest tensor network simulator
           // beats the statevector one even with no paralelization in tensor
           // networks but with paralelization in statevector!!!!
  // for no parallelization, the forest tensor network simulator is faster even
  // for 13 qubits
  const unsigned int nrShots = 5000;

  std::shared_ptr<Circuits::Circuit<>> randomCirc;
  std::vector<std::shared_ptr<
      QC::Gates::AppliedGate<TensorNetworks::TensorNode::MatrixClass>>>
      randomQcSimCirc;

  std::shared_ptr<Simulators::ISimulator> qc;
  std::shared_ptr<Simulators::ISimulator> qcForest;

  std::shared_ptr<Simulators::ISimulator> qcTensor;

  std::shared_ptr<TensorNetworks::TensorNetwork> tensorNetworkOneQubit;
  std::shared_ptr<TensorNetworks::TensorNetwork> tensorNetwork;

  std::shared_ptr<TensorNetworks::TensorNetwork> tensorNetworkForest;
  std::shared_ptr<Simulators::ISimulator> qcsimClifford;

  std::shared_ptr<Simulators::ISimulator> qcsimMPS;  // for speed comparison

  std::mt19937 g;
  std::bernoulli_distribution boolDist;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

BOOST_AUTO_TEST_SUITE(tensor_tests)

BOOST_FIXTURE_TEST_CASE(TensorsInitializationTest, TensorsTestFixture) {
  BOOST_TEST(randomCirc);
  BOOST_TEST(qc);
  BOOST_TEST(qcForest);
  BOOST_TEST(qcTensor);
  BOOST_TEST(tensorNetwork);
  BOOST_TEST(tensorNetworkOneQubit);
  BOOST_TEST(tensorNetworkForest);
  BOOST_TEST(qcsimClifford);
  BOOST_TEST(qcsimMPS);
}

BOOST_FIXTURE_TEST_CASE(TensorsOneQubitEmptyCircuitTest, TensorsTestFixture) {
  const double prob = tensorNetworkOneQubit->Probability(0);

  BOOST_CHECK_PREDICATE(checkClose, (prob)(1.)(0.000001));

  qcTensor->AllocateQubits(1);
  qcTensor->Initialize();

  double prob1 = qcTensor->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(1.)(0.000001));
  prob1 = qcTensor->Probability(1);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(0.)(0.000001));

  double res = qcTensor->Measure({0}) ? 1 : 0;
  BOOST_CHECK_PREDICATE(checkClose, (res)(0.)(0.000001));

  qcTensor->Clear();
}

BOOST_FIXTURE_TEST_CASE(TensorsOneQubitNotCircuitTest, TensorsTestFixture) {
  QC::Gates::PauliXGate xgate;
  tensorNetworkOneQubit->AddGate(xgate, 0);

  double prob = tensorNetworkOneQubit->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = tensorNetworkOneQubit->Probability(0, false);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(1.)(0.000001));

  qcTensor->AllocateQubits(1);
  qcTensor->Initialize();
  qcTensor->ApplyX(0);

  double prob1 = qcTensor->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(0.)(0.000001));
  prob1 = qcTensor->Probability(1);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(1.)(0.000001));

  double res = qcTensor->Measure({0}) ? 1 : 0;
  BOOST_CHECK_PREDICATE(checkClose, (res)(1.)(0.000001));

  qcTensor->Clear();
}

BOOST_FIXTURE_TEST_CASE(TensorsSimpleOneQubitCircuitTest, TensorsTestFixture) {
  QC::Gates::HadamardGate<> hGate;
  tensorNetworkOneQubit->AddGate(hGate, 0);

  const double prob = tensorNetworkOneQubit->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));

  qcTensor->AllocateQubits(1);
  qcTensor->Initialize();
  qcTensor->ApplyH(0);

  double prob1 = qcTensor->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(0.5)(0.000001));
  prob1 = qcTensor->Probability(1);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(0.5)(0.000001));
}

BOOST_FIXTURE_TEST_CASE(TensorsEmptyCircuitTest, TensorsTestFixture) {
  const double prob = tensorNetwork->Probability(0);

  BOOST_CHECK_PREDICATE(checkClose, (prob)(1.)(0.000001));

  qcTensor->AllocateQubits(nrQubits);
  qcTensor->Initialize();

  double prob1 = qcTensor->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(1.)(0.000001));
  prob1 = qcTensor->Probability(1);
  BOOST_CHECK_PREDICATE(checkClose, (prob1)(0.)(0.000001));

  double res = qcTensor->Measure({0}) ? 1 : 0;
  BOOST_CHECK_PREDICATE(checkClose, (res)(0.)(0.000001));

  qcTensor->Clear();
}

BOOST_FIXTURE_TEST_CASE(TensorsSimpleCircuitTest, TensorsTestFixture) {
  QC::Gates::HadamardGate hGate;
  QC::Gates::PauliXGate xGate;

  tensorNetwork->AddGate(hGate, 0);
  tensorNetwork->AddGate(xGate, 1);
  tensorNetwork->AddGate(xGate, 2);
  tensorNetwork->AddGate(hGate, 2);

  double prob = tensorNetwork->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));
  prob = tensorNetwork->Probability(0, false);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));

  prob = tensorNetwork->Probability(1);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0)(0.000001));
  prob = tensorNetwork->Probability(1, false);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(1)(0.000001));

  prob = tensorNetwork->Probability(2);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));
  prob = tensorNetwork->Probability(2, false);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));

  prob = tensorNetwork->Probability(3);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(1)(0.000001));
  prob = tensorNetwork->Probability(3, false);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0)(0.000001));

  qcTensor->AllocateQubits(nrQubits);
  qcTensor->Initialize();

  qcTensor->ApplyH(0);  // 0 qubit: either value probability 0.5
  qcTensor->ApplyX(1);  // 1 qubit: 0 with probability 0, 1 with probability 1
  qcTensor->ApplyX(2);
  qcTensor->ApplyH(2);  // 2 qubit: either value probability 0.5

  // so all states with qubit 1 = 1 should have probability 0.25
  // the others, probability 0

  prob = qcTensor->Probability(0b010);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.25)(0.000001));

  prob = qcTensor->Probability(0b110);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.25)(0.000001));

  prob = qcTensor->Probability(0b011);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.25)(0.000001));

  prob = qcTensor->Probability(0b111);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.25)(0.000001));

  prob = qcTensor->Probability(0b000);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b100);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b001);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b101);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  qcTensor->Clear();
}

BOOST_FIXTURE_TEST_CASE(TensorsSimpleTwoQubitGatesTest, TensorsTestFixture) {
  QC::Gates::HadamardGate hGate;
  QC::Gates::PauliXGate xGate;
  QC::Gates::CNOTGate cnotGate;
  QC::Gates::SwapGate swapGate;

  tensorNetwork->AddGate(hGate, 0);
  tensorNetwork->AddGate(cnotGate, 0, 1);
  tensorNetwork->AddGate(xGate, 2);
  tensorNetwork->AddGate(cnotGate, 2, 3);
  tensorNetwork->AddGate(swapGate, 3, 4);

  double prob = tensorNetwork->Probability(0);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));

  prob = tensorNetwork->Probability(1);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));

  prob = tensorNetwork->Probability(2);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0)(0.000001));

  prob = tensorNetwork->Probability(3);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(1)(0.000001));

  prob = tensorNetwork->Probability(4);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0)(0.000001));

  qcTensor->AllocateQubits(nrQubits);
  qcTensor->Initialize();

  qcTensor->ApplyH(0);
  qcTensor->ApplyCX(0, 1);
  qcTensor->ApplyX(2);
  qcTensor->ApplyCX(2, 3);
  qcTensor->ApplySwap(3, 4);

  prob = qcTensor->Probability(0b10100);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));
  prob = qcTensor->Probability(0b10111);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.5)(0.000001));
  prob = qcTensor->Probability(0b10110);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b10101);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b00100);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b00111);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b00110);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b00101);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b01100);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b01111);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b01110);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b01101);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b11100);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b11111);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b11110);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b11101);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b10000);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b10011);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b10010);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b10001);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b00000);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b00011);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b00010);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b00001);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b01000);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b01011);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b01010);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b01001);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  prob = qcTensor->Probability(0b11000);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b11011);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b11010);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));
  prob = qcTensor->Probability(0b11001);
  BOOST_CHECK_PREDICATE(checkClose, (prob)(0.)(0.000001));

  qcTensor->Clear();
}

BOOST_DATA_TEST_CASE_F(TensorsTestFixture, TensorsSimpleRandomCircsTest,
                       bdata::xrange(1, 20), nrGates) {
  const size_t nrStates = 1ULL << nrQubits;
  Circuits::OperationState stateDummy(nrQubits);

  for (int t = 0; t < 30; ++t) {
    GenerateCircuits(nrGates);

    randomCirc->Execute(qc, stateDummy);

    for (int g = 0; g < nrGates; ++g) {
      const QC::Gates::QuantumGateWithOp<
          TensorNetworks::TensorNode::MatrixClass>& gate = *randomQcSimCirc[g];
      tensorNetwork->AddGate(gate, randomQcSimCirc[g]->getQubit1(),
                             randomQcSimCirc[g]->getQubit2());
    }

    size_t maxTensorRank = 0;
    for (size_t q = 0; q < nrQubits; ++q) {
      const double prob1 = tensorNetwork->Probability(q);
      maxTensorRank = std::max(
          maxTensorRank, tensorNetwork->GetContractor()->GetMaxTensorRank());

      // sum up all probabilities where the qubit is zero
      double prob2 = 0.;
      const size_t qubitMask = 1ULL << q;
      for (size_t state = 0; state < nrStates; ++state)
        if ((state & qubitMask) == 0) prob2 += qc->Probability(state);

      BOOST_CHECK_PREDICATE(checkClose, (prob1)(prob2)(0.000001));
    }
    BOOST_TEST_MESSAGE("Max tensor rank: " << maxTensorRank);

    qc->Clear();
    qc->AllocateQubits(nrQubits);
    qc->Initialize();

    randomCirc->Clear();
    randomQcSimCirc.clear();

    tensorNetwork->Clear();
  }
}

BOOST_DATA_TEST_CASE_F(TensorsTestFixture,
                       TensorsSimpleRandomCircsTestSimulator,
                       bdata::xrange(1, 20), nrGates) {
  const size_t nrStates = 1ULL << nrQubits;
  Circuits::OperationState stateDummy(nrQubits);

  for (int t = 0; t < 30; ++t) {
    GenerateCircuits(nrGates, true);

    randomCirc->Execute(qc, stateDummy);

    qcTensor->AllocateQubits(nrQubits);
    qcTensor->Initialize();
    randomCirc->Execute(qcTensor, stateDummy);
    for (size_t state = 0; state < nrStates; ++state) {
      const double prob1 = qc->Probability(state);
      const double prob2 = qcTensor->Probability(state);
      BOOST_CHECK_PREDICATE(checkClose, (prob1)(prob2)(0.000001));
    }
    qcTensor->Clear();

    qc->Clear();
    qc->AllocateQubits(nrQubits);
    qc->Initialize();

    randomCirc->Clear();
  }
}

BOOST_DATA_TEST_CASE_F(TensorsTestFixture, TensorsSimpleRandomCircsMeasTest,
                       bdata::xrange(1, 15), nrGates) {
  std::vector<std::pair<Types::qubit_t, size_t>> measQubits(nrQubits);
  for (size_t q = 0; q < nrQubits; ++q) measQubits[q] = std::make_pair(q, q);

  Circuits::MeasurementOperation measurements(measQubits);

  for (int t = 0; t < 3; ++t) {
    GenerateCircuits(nrGates);

    std::unordered_map<Types::qubit_t, size_t> measResultsQcSim;
    std::unordered_map<Types::qubit_t, size_t> measResultsTensorNetwork;

    size_t maxTensorRank = 0;
    for (size_t shot = 0; shot < nrShots; ++shot) {
      Circuits::OperationState state(nrQubits);
      randomCirc->Execute(qc, state);

      for (int g = 0; g < nrGates; ++g) {
        const QC::Gates::QuantumGateWithOp<
            TensorNetworks::TensorNode::MatrixClass>& gate =
            *randomQcSimCirc[g];
        tensorNetwork->AddGate(gate, randomQcSimCirc[g]->getQubit1(),
                               randomQcSimCirc[g]->getQubit2());
      }

      measurements.Execute(qc, state);

      for (size_t q = 0; q < nrQubits; ++q) {
        if (state.GetBit(q)) ++measResultsQcSim[q];
        if (tensorNetwork->Measure(q)) ++measResultsTensorNetwork[q];

        maxTensorRank = std::max(
            maxTensorRank, tensorNetwork->GetContractor()->GetMaxTensorRank());
      }

      qc->Clear();
      qc->AllocateQubits(nrQubits);
      qc->Initialize();

      tensorNetwork->Clear();
    }

    BOOST_TEST_MESSAGE("Max tensor rank: " << maxTensorRank);

    randomCirc->Clear();
    randomQcSimCirc.clear();

    for (size_t q = 0; q < nrQubits; ++q) {
      const double prob1 = static_cast<double>(measResultsQcSim[q]) /
                           static_cast<double>(nrShots);
      const double prob2 = static_cast<double>(measResultsTensorNetwork[q]) /
                           static_cast<double>(nrShots);

      BOOST_CHECK_PREDICATE(checkClose, (prob1)(prob2)(0.05));
    }
  }
}

BOOST_DATA_TEST_CASE_F(TensorsTestFixture, TensorsSimpleRandomForestCircsTest,
                       bdata::xrange(1, 30), nrGates) {
  const size_t nrStates = 1ULL << nrQubitsForest;
  Circuits::OperationState stateDummy(nrQubitsForest);

  for (int t = 0; t < 3; ++t) {
    GenerateForestCircuits(nrGates);

    BOOST_TEST(randomCirc->IsForest());

    if (!randomCirc->IsForest()) {
      for (auto& op : randomCirc->GetOperations()) {
        const auto qs = op->AffectedQubits();

        if (qs.size() <= 1) continue;

        std::cout << "Gate on qubits: " << qs[0] << ", " << qs[1] << std::endl;
      }
    }

    randomCirc->Execute(qcForest, stateDummy);

    for (int g = 0; g < nrGates; ++g) {
      const QC::Gates::QuantumGateWithOp<
          TensorNetworks::TensorNode::MatrixClass>& gate = *randomQcSimCirc[g];
      tensorNetworkForest->AddGate(gate, randomQcSimCirc[g]->getQubit1(),
                                   randomQcSimCirc[g]->getQubit2());
    }

    size_t maxTensorRank = 0;
    for (size_t q = 0; q < nrQubitsForest; ++q) {
      const double prob1 = tensorNetworkForest->Probability(q);
      maxTensorRank =
          std::max(maxTensorRank,
                   tensorNetworkForest->GetContractor()->GetMaxTensorRank());

      // sum up all probabilities where the qubit is zero
      double prob2 = 0.;
      const size_t qubitMask = 1ULL << q;
      for (size_t state = 0; state < nrStates; ++state)
        if ((state & qubitMask) == 0) prob2 += qcForest->Probability(state);

      BOOST_CHECK_PREDICATE(checkClose, (prob1)(prob2)(0.000001));
    }
    BOOST_CHECK(maxTensorRank <= 4);

    qcForest->Clear();
    qcForest->AllocateQubits(nrQubitsForest);
    qcForest->Initialize();

    randomCirc->Clear();
    randomQcSimCirc.clear();

    tensorNetworkForest->Clear();
  }
}

BOOST_DATA_TEST_CASE_F(TensorsTestFixture,
                       TensorsSimpleRandomForestCircsSimulatorTest,
                       bdata::xrange(1, 30), nrGates) {
  const size_t nrStates = 1ULL << nrQubitsForest;
  Circuits::OperationState stateDummy(nrQubitsForest);

  for (int t = 0; t < 3; ++t) {
    GenerateForestCircuits(nrGates);
    BOOST_TEST(randomCirc->IsForest());
    if (!randomCirc->IsForest()) {
      for (auto& op : randomCirc->GetOperations()) {
        const auto qs = op->AffectedQubits();

        if (qs.size() <= 1) continue;

        std::cout << "Gate on qubits: " << qs[0] << ", " << qs[1] << std::endl;
      }
    }

    randomCirc->Execute(qcForest, stateDummy);

    qcTensor->AllocateQubits(nrQubitsForest);
    qcTensor->Initialize();
    randomCirc->Execute(qcTensor, stateDummy);
    for (size_t state = 0; state < nrStates; ++state) {
      const double prob1 = qcForest->Probability(state);
      const double prob2 = qcTensor->Probability(state);
      BOOST_CHECK_PREDICATE(checkClose, (prob1)(prob2)(0.000001));
    }
    qcTensor->Clear();

    qcForest->Clear();
    qcForest->AllocateQubits(nrQubitsForest);
    qcForest->Initialize();

    randomCirc->Clear();
  }
}

/*
BOOST_DATA_TEST_CASE_F(TensorsTestFixture,
TensorsRandomCliffordForestCircsSimulatorTest, bdata::xrange(1, 30), nrGates)
{
        const size_t nrShots = 10000;
        const size_t nrStates = 1ULL << nrQubitsForest;
        Circuits::OperationState stateDummy(nrQubitsForest);

        for (int t = 0; t < 3; ++t)
        {
                BOOST_TEST_MESSAGE("Generating circuit with " << nrGates << "
gates for test #" << t); GenerateForestCircuits(nrGates, true);
                BOOST_TEST_MESSAGE("Done, executing...");

                BOOST_TEST(randomCirc->IsForest());
                if (!randomCirc->IsForest()) {
                        for (auto& op : randomCirc->GetOperations())
                        {
                                const auto qs = op->AffectedQubits();

                                if (qs.size() <= 1) continue;

                                std::cout << "Gate on qubits: " << qs[0] << ", "
<< qs[1] << std::endl;
                        }
                }

                for (size_t i = 0; i < nrQubitsForest; ++i)
                        randomCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement({
{i, i} }));

                qcForest->SetMultithreading(false);
                std::chrono::high_resolution_clock::time_point t1 =
std::chrono::high_resolution_clock::now();

                for (size_t shot = 0; shot < nrShots; ++shot)
                {
                        randomCirc->Execute(qcForest, stateDummy);
                        qcForest->Clear();
                        qcForest->AllocateQubits(nrQubitsForest);
                        qcForest->Initialize();
                        stateDummy.Reset();
                }
                qcForest->SetMultithreading(true);

                std::chrono::high_resolution_clock::time_point t2 =
std::chrono::high_resolution_clock::now();

                auto dif =
std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                BOOST_TEST_MESSAGE("\nExecution in qcsim statevector took: " <<
dif / 1000. << " seconds!");


                t1 = t2;

                qcTensor->SetMultithreading(false);

                for (size_t shot = 0; shot < nrShots; ++shot)
                {
                        qcTensor->AllocateQubits(nrQubitsForest);
                        qcTensor->Initialize();
                        randomCirc->Execute(qcTensor, stateDummy);
                        qcTensor->Clear();
                        stateDummy.Reset();
                }
                qcTensor->SetMultithreading(true);

                t2 = std::chrono::high_resolution_clock::now();

                dif = std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
t1).count(); BOOST_TEST_MESSAGE("\nExecution in qcsim tensor network took: " <<
dif / 1000. << " seconds!");

                t1 = t2;

                for (size_t shot = 0; shot < nrShots; ++shot)
                {
                        randomCirc->Execute(qcsimMPS, stateDummy);
                        qcsimMPS->Clear();
                        qcsimMPS->AllocateQubits(nrQubitsForest);
                        qcsimMPS->Initialize();
                        stateDummy.Reset();
                }

                t2 = std::chrono::high_resolution_clock::now();

                dif = std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
t1).count(); BOOST_TEST_MESSAGE("\nExecution in qcsim MPS took: " << dif / 1000.
<< " seconds!");


                t1 = t2;

                for (size_t shot = 0; shot < nrShots; ++shot)
                {
                        randomCirc->Execute(qcsimClifford, stateDummy);
                        qcsimClifford->Clear();
                        qcsimClifford->AllocateQubits(nrQubitsForest);
                        qcsimClifford->Initialize();
                        stateDummy.Reset();
                }

                t2 = std::chrono::high_resolution_clock::now();

                dif = std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
t1).count(); BOOST_TEST_MESSAGE("\nExecution in qcsim stabilizer took: " << dif
/ 1000. << " seconds!");

                randomCirc->Clear();
        }
}
*/

BOOST_AUTO_TEST_SUITE_END()
