#ifndef MAESTRO_TESTS_QASM_TEST_FIXTURE_H_
#define MAESTRO_TESTS_QASM_TEST_FIXTURE_H_

#include <boost/test/data/monomorphic.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "../Circuit/Measurements.h"
#include "../Circuit/RandomOp.h"
#include "../Circuit/Reset.h"
#include "qasm_test_utils.h"

namespace bdata = boost::unit_test::data;

struct QasmTestFixture {
  QasmTestFixture() {
    state.AllocateBits(nrQubitsForRandomCirc);
    resetCirc = std::make_shared<Circuits::Circuit<>>();
    Types::qubits_vector qubits(nrQubitsForRandomCirc);
    std::iota(qubits.begin(), qubits.end(), 0);
    resetCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

    randomCirc = std::make_shared<Circuits::Circuit<>>();

    qc = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qc->AllocateQubits(nrQubitsForRandomCirc);
    qc->Initialize();

    qc2 = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qc2->AllocateQubits(nrQubitsForRandomCirc);
    qc2->Initialize();
  }

  // fills randomly the circuit with gates
  void GenerateCircuit(int nrGates, int nrQubits, double probReset = 0.,
                       double probMeasurement = 0.) {
    std::uniform_real_distribution<double> angle(-2. * M_PI, 2. * M_PI);
    std::uniform_real_distribution<double> operationType(0., 1.);

    const double cumProb = probReset + probMeasurement;

    // TODO: Maybe insert from time to time a random number generating 'gate'
    // and a conditioned random one, those should not affect results?
    for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
      const double typeOpProb = operationType(randomGenerator);

      Types::qubits_vector qubits(nrQubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      std::shuffle(qubits.begin(), qubits.end(), randomGenerator);

      if (typeOpProb < probReset) {
        // reset operation
        auto resetOp = Circuits::CircuitFactory<>::CreateReset({qubits[0]});
        randomCirc->AddOperation(resetOp);
        continue;
      } else if (typeOpProb < cumProb) {
        // measurement operation
        auto measOp = Circuits::CircuitFactory<>::CreateMeasurement(
            {{qubits[0], qubits[0]}});
        randomCirc->AddOperation(measOp);
        continue;
      }

      // create a random gate and add it to the circuit

      // first, pick randomly three qubits (depending on the randomly chosen
      // gate type, not all of them will be used)
      auto q1 = qubits[0];
      auto q2 = qubits[1];
      auto q3 = qubits[2];

      // now some random parameters, again, they might be ignored
      const double param1 = angle(randomGenerator);
      const double param2 = angle(randomGenerator);
      const double param3 = angle(randomGenerator);

      const Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(
              gateTypeDistribution(randomGenerator));

      auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3);
      randomCirc->AddOperation(theGate);
    }
  }

  Circuits::OperationState state;
  std::shared_ptr<Simulators::ISimulator> qc;
  std::shared_ptr<Simulators::ISimulator> qc2;

  // for testing randomly generated circuits
  const unsigned int nrQubitsForRandomCirc = 5;

  std::shared_ptr<Circuits::Circuit<>> randomCirc;
  std::shared_ptr<Circuits::Circuit<>> resetCirc;

 private:
  std::mt19937 randomGenerator{0x4d414553U};
  std::uniform_int_distribution<int> gateTypeDistribution{
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType)};
};

#endif  // MAESTRO_TESTS_QASM_TEST_FIXTURE_H_
