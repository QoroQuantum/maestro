/**
 * @file PathIntegralTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the path integral simulator wrapper (Simulators::PathIntegralSimulator).
 *
 * Random Maestro circuits are generated using the circuit factory, executed on
 * the Qiskit Aer statevector simulator, and the resulting per basis-state
 * amplitudes are compared against the ones produced by propagating the same
 * circuit (after conversion to QCSim's AppliedGate format) through the path
 * integral simulator wrapper.
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
#include <complex>
#define _USE_MATH_DEFINES
#include <math.h>

// project being tested
#include "../Simulators/PathIntegralSimulator.h"

#include "../Simulators/Factory.h"

#include "../Circuit/Circuit.h"
#include "../Circuit/QuantumGates.h"
#include "../Circuit/Factory.h"

namespace {

// Build a random circuit using the gate kinds the path integral wrapper knows
// how to convert. Measurements / resets are intentionally excluded.
std::shared_ptr<Circuits::Circuit<>> GenerateRandomCircuit(
    int nrGates, int nrQubits, std::mt19937& g) {
  auto circuit = Circuits::CircuitFactory<>::CreateCircuit();

  std::uniform_real_distribution<double> paramDist(-2.0 * M_PI, 2.0 * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));

  for (int i = 0; i < nrGates; ++i) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);

    const auto q1 = qubits[0];
    const auto q2 = qubits[1];
    const auto q3 = qubits[2];

    const double p1 = paramDist(g);
    const double p2 = paramDist(g);
    const double p3 = paramDist(g);
    const double p4 = paramDist(g);

    const auto gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

    circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, q3, p1, p2, p3, p4));
  }

  return circuit;
}

}  // namespace

struct PathIntegralFixture {
  PathIntegralFixture() {
    aer = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kStatevector);
    aer->AllocateQubits(nrQubits);
    aer->Initialize();
  }

  ~PathIntegralFixture() {}

  static constexpr unsigned int nrQubits = 8;
  static constexpr int nrGates = 20;

  std::shared_ptr<Simulators::ISimulator> aer;
  Simulators::PathIntegralSimulator pi;
};

BOOST_AUTO_TEST_SUITE(path_integral_tests)

// Generate several random Maestro circuits and check that the per basis-state
// amplitudes computed by the path integral wrapper match those of the Qiskit
// Aer statevector simulator.
BOOST_DATA_TEST_CASE_F(PathIntegralFixture,
                       RandomCircuitAmplitudesMatchAer,
                       bdata::xrange(10),
                       trial) {
  BOOST_TEST(aer);
  aer->Reset();

  // Use a deterministic seed per trial so failures can be reproduced.
  std::mt19937 g(static_cast<std::mt19937::result_type>(0xC0FFEEu + trial));

  auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
  BOOST_REQUIRE(circuit);

  Circuits::OperationState state;
  state.AllocateBits(nrQubits);
  circuit->Execute(aer, state);

  // Feed the same circuit to the path integral wrapper.
  BOOST_REQUIRE(pi.SetCircuit(circuit));

  const size_t nrBasisStates = 1ULL << nrQubits;
  std::vector<bool> endState(nrQubits);

  for (size_t s = 0; s < nrBasisStates; ++s) {
    for (size_t q = 0; q < nrQubits; ++q)
      endState[q] = ((s >> q) & 1ULL) == 1ULL;

    const std::complex<double> aerAmp = aer->Amplitude(s);
    const std::complex<double> piAmp = pi.AmplitudeFromZero(endState);

    BOOST_CHECK_SMALL(std::abs(aerAmp - piAmp), 1e-7);
  }
}

BOOST_AUTO_TEST_SUITE_END()
