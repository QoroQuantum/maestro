/**
 * @file PathIntegralTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the path integral simulator wrapper and the qcsim path integral
 * implementation exposed through the common simulator interface.
 *
 * Random Maestro circuits are generated using the circuit factory and their
 * per basis-state amplitudes are compared against reference simulators.
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
#include <unordered_map>
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
std::shared_ptr<Circuits::Circuit<>> GenerateRandomCircuit(int nrGates,
                                                           int nrQubits,
                                                           std::mt19937& g) {
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

std::string GenerateRandomPauliString(int nrQubits, std::mt19937& g) {
  std::string pauli(nrQubits, 'I');
  std::uniform_int_distribution<int> pauliDist(0, 3);

  for (int q = 0; q < nrQubits; ++q) {
    switch (pauliDist(g)) {
      case 0:
        pauli[q] = 'X';
        break;
      case 1:
        pauli[q] = 'Y';
        break;
      case 2:
        pauli[q] = 'Z';
        break;
      default:
        pauli[q] = 'I';
        break;
    }
  }

  return pauli;
}

Types::qubit_t OutcomeFromBits(const std::vector<bool>& bits) {
  Types::qubit_t outcome = 0;
  Types::qubit_t mask = 1ULL;

  for (const bool bit : bits) {
    if (bit) outcome |= mask;
    mask <<= 1ULL;
  }

  return outcome;
}

std::unordered_map<Types::qubit_t, Types::qubit_t> ConvertManyCounts(
    const std::unordered_map<std::vector<bool>, Types::qubit_t>& manyCounts) {
  std::unordered_map<Types::qubit_t, Types::qubit_t> counts;

  for (const auto& [outcomeBits, count] : manyCounts)
    counts[OutcomeFromBits(outcomeBits)] += count;

  return counts;
}

std::vector<double> ComputeMarginalProbabilities(
    const std::shared_ptr<Simulators::ISimulator>& simulator,
    const Types::qubits_vector& qubits, int nrQubits) {
  const size_t nrOutcomes = 1ULL << qubits.size();
  const size_t nrStates = 1ULL << nrQubits;
  std::vector<double> probabilities(nrOutcomes, 0.0);

  for (size_t state = 0; state < nrStates; ++state) {
    Types::qubit_t outcome = 0;
    Types::qubit_t mask = 1ULL;

    for (const auto qubit : qubits) {
      if ((state & (1ULL << qubit)) != 0) outcome |= mask;
      mask <<= 1ULL;
    }

    probabilities[outcome] += std::norm(simulator->Amplitude(state));
  }

  return probabilities;
}

void CheckCountsAgainstProbabilities(
    const std::unordered_map<Types::qubit_t, Types::qubit_t>& counts,
    const std::vector<double>& probabilities, size_t shots) {
  Types::qubit_t totalCounts = 0;
  for (const auto& [outcome, count] : counts) totalCounts += count;

  BOOST_TEST(totalCounts == shots);

  for (size_t outcome = 0; outcome < probabilities.size(); ++outcome) {
    const auto it = counts.find(static_cast<Types::qubit_t>(outcome));
    const double observedProbability =
        it == counts.end() ? 0.0 : static_cast<double>(it->second) / shots;

    const double expectedProbability = probabilities[outcome];
    const double tolerance =
        0.01 + 6.0 * std::sqrt(expectedProbability *
                               (1.0 - expectedProbability) / shots);

    BOOST_CHECK_SMALL(std::abs(observedProbability - expectedProbability),
                      tolerance);
  }
}

// Build a random circuit that interleaves unitary gates with mid-circuit
// measurements and classically controlled gates, followed by a final
// measurement on a randomly chosen subset of qubits.
//
// Returns the circuit together with:
//   - midCbits: classical-bit indices used for mid-circuit measurements
//   - finalMeasPairs: (qubit, cbit) pairs for the final measurements
// The caller should allocate at least (nrQubits + midCbits.size() +
// finalMeasPairs.size()) classical bits in the OperationState.
struct MixedCircuitInfo {
  std::shared_ptr<Circuits::Circuit<>> circuit;
  // classical bit indices written by the final measurement
  std::vector<size_t> finalCbits;
  // number of classical bits needed in total
  size_t totalCbits;
};

MixedCircuitInfo GenerateRandomMixedCircuit(int nrGates, int nrQubits,
                                            std::mt19937& g) {
  auto circuit = Circuits::CircuitFactory<>::CreateCircuit();

  std::uniform_real_distribution<double> paramDist(-2.0 * M_PI, 2.0 * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
  std::uniform_int_distribution<int> qubitDist(0, nrQubits - 1);
  std::bernoulli_distribution measureNow(0.15);  // ~15% chance per gate slot

  // Classical bits: reserve nrQubits slots for mid-circuit measurements (one
  // per qubit) and nrQubits more for the final measurements.
  const size_t midCbitBase = 0;
  const size_t finalCbitBase = static_cast<size_t>(nrQubits);
  const size_t totalCbits = 2 * static_cast<size_t>(nrQubits);

  // Track which qubits have been measured mid-circuit (and thus have a valid
  // classical bit we can condition on).
  std::vector<bool> measuredMidCircuit(nrQubits, false);

  auto addRandomGate = [&]() {
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
  };

  for (int i = 0; i < nrGates; ++i) {
    addRandomGate();

    // Occasionally insert a mid-circuit measurement on a random qubit.
    if (measureNow(g)) {
      const int mQubit = qubitDist(g);
      const size_t mCbit = midCbitBase + static_cast<size_t>(mQubit);
      circuit->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(
          {{static_cast<Types::qubit_t>(mQubit), mCbit}}));
      measuredMidCircuit[mQubit] = true;

      // If we have measured at least one qubit, optionally add a classically
      // controlled gate on another qubit conditioned on this bit.
      if (measureNow(g)) {
        const int tQubit = qubitDist(g);
        const auto innerGate =
            std::dynamic_pointer_cast<Circuits::IGateOperation<>>(
                Circuits::CircuitFactory<>::CreateGate(
                    Circuits::QuantumGateType::kXGateType,
                    static_cast<size_t>(tQubit)));
        if (innerGate) {
          circuit->AddOperation(
              Circuits::CircuitFactory<>::CreateSimpleConditionalGate(innerGate,
                                                                      mCbit));
        }
      }
    }
  }

  // Final measurement on a random non-empty subset of qubits.
  Types::qubits_vector allQubits(nrQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);
  std::shuffle(allQubits.begin(), allQubits.end(), g);

  std::uniform_int_distribution<int> subsetSizeDist(1, nrQubits);
  const int finalMeasCount = subsetSizeDist(g);

  std::vector<std::pair<Types::qubit_t, size_t>> finalPairs;
  std::vector<size_t> finalCbits;
  for (int k = 0; k < finalMeasCount; ++k) {
    const size_t cbit = finalCbitBase + static_cast<size_t>(k);
    finalPairs.push_back({static_cast<Types::qubit_t>(allQubits[k]), cbit});
    finalCbits.push_back(cbit);
  }

  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement(finalPairs));

  return {circuit, finalCbits, totalCbits};
}

// Execute a circuit once and read back the values of the specified classical
// bits, returning them as a packed integer (bit k = cbits[k]).
Types::qubit_t RunOneShot(
    const std::shared_ptr<Simulators::ISimulator>& simulator,
    const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t totalCbits,
    const std::vector<size_t>& readbackCbits) {
  simulator->Reset();

  Circuits::OperationState state;
  state.AllocateBits(totalCbits);
  circuit->Execute(simulator, state);

  Types::qubit_t outcome = 0;
  for (size_t k = 0; k < readbackCbits.size(); ++k) {
    if (state.GetBit(readbackCbits[k])) outcome |= (1ULL << k);
  }
  return outcome;
}

}  // namespace

struct PathIntegralFixture {
  PathIntegralFixture() {
    aer = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kStatevector);
    aer->AllocateQubits(nrQubits);
    aer->Initialize();

    qcsimStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qcsimStatevector->AllocateQubits(nrQubits);
    qcsimStatevector->Initialize();

    qcsimPathIntegral = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kPathIntegral);
    qcsimPathIntegral->AllocateQubits(nrQubits);
    qcsimPathIntegral->Initialize();
  }

  ~PathIntegralFixture() {}

  static constexpr unsigned int nrQubits = 8;
  static constexpr int nrGates = 20;

  std::shared_ptr<Simulators::ISimulator> aer;
  std::shared_ptr<Simulators::ISimulator> qcsimStatevector;
  std::shared_ptr<Simulators::ISimulator> qcsimPathIntegral;
  Simulators::PathIntegralSimulator pi;
};

BOOST_AUTO_TEST_SUITE(path_integral_tests)

// Generate several random Maestro circuits and check that the per basis-state
// amplitudes computed by the path integral wrapper match those of the Qiskit
// Aer statevector simulator.
BOOST_DATA_TEST_CASE_F(PathIntegralFixture, RandomCircuitAmplitudesMatchAer,
                       bdata::xrange(10), trial) {
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

// Generate several random Maestro circuits and execute them through the common
// simulator interface using qcsim's path integral implementation. Check that
// the resulting per basis-state amplitudes match qcsim's statevector
// simulation.
BOOST_DATA_TEST_CASE_F(
    PathIntegralFixture,
    RandomCircuitCommonInterfaceAmplitudesMatchQCSimStatevector,
    bdata::xrange(10), trial) {
  BOOST_TEST(qcsimStatevector);
  BOOST_TEST(qcsimPathIntegral);

  qcsimStatevector->Reset();
  qcsimPathIntegral->Reset();

  std::mt19937 g(static_cast<std::mt19937::result_type>(0xFACEB00Cu + trial));

  auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
  BOOST_REQUIRE(circuit);

  Circuits::OperationState statevectorState;
  statevectorState.AllocateBits(nrQubits);
  circuit->Execute(qcsimStatevector, statevectorState);

  Circuits::OperationState pathIntegralState;
  pathIntegralState.AllocateBits(nrQubits);
  circuit->Execute(qcsimPathIntegral, pathIntegralState);

  const size_t nrBasisStates = 1ULL << nrQubits;

  for (size_t s = 0; s < nrBasisStates; ++s) {
    const std::complex<double> svAmp = qcsimStatevector->Amplitude(s);
    const std::complex<double> piAmp = qcsimPathIntegral->Amplitude(s);

    BOOST_CHECK_SMALL(std::abs(svAmp - piAmp), 1e-7);
  }
}

// Generate random Maestro circuits and random Pauli strings, then check that
// expectation values computed by qcsim's path integral simulator match qcsim's
// statevector simulator.
BOOST_DATA_TEST_CASE_F(PathIntegralFixture,
                       RandomPauliExpectationValuesMatchQCSimStatevector,
                       bdata::xrange(10), trial) {
  BOOST_TEST(qcsimStatevector);
  BOOST_TEST(qcsimPathIntegral);

  qcsimStatevector->Reset();
  qcsimPathIntegral->Reset();

  std::mt19937 g(static_cast<std::mt19937::result_type>(0xBADC0DEu + trial));

  auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
  BOOST_REQUIRE(circuit);

  Circuits::OperationState statevectorState;
  statevectorState.AllocateBits(nrQubits);
  circuit->Execute(qcsimStatevector, statevectorState);

  Circuits::OperationState pathIntegralState;
  pathIntegralState.AllocateBits(nrQubits);
  circuit->Execute(qcsimPathIntegral, pathIntegralState);

  for (int i = 0; i < 20; ++i) {
    const std::string pauliString = GenerateRandomPauliString(nrQubits, g);

    const double svExpval = qcsimStatevector->ExpectationValue(pauliString);
    const double piExpval = qcsimPathIntegral->ExpectationValue(pauliString);

    BOOST_CHECK_SMALL(std::abs(svExpval - piExpval), 1e-7);
  }
}

// Generate random Maestro circuits and verify that sampling through
// SampleCounts matches the exact marginal probabilities of the statevector for
// both qcsim's statevector and path integral simulators.
BOOST_DATA_TEST_CASE_F(PathIntegralFixture,
                       RandomCircuitSampleCountsMatchReferenceProbabilities,
                       bdata::xrange(5), trial) {
  BOOST_TEST(qcsimStatevector);
  BOOST_TEST(qcsimPathIntegral);

  qcsimStatevector->Reset();
  qcsimPathIntegral->Reset();

  std::mt19937 g(static_cast<std::mt19937::result_type>(0x13579BDFu + trial));

  auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
  BOOST_REQUIRE(circuit);

  Circuits::OperationState statevectorState;
  statevectorState.AllocateBits(nrQubits);
  circuit->Execute(qcsimStatevector, statevectorState);

  Circuits::OperationState pathIntegralState;
  pathIntegralState.AllocateBits(nrQubits);
  circuit->Execute(qcsimPathIntegral, pathIntegralState);

  Types::qubits_vector allQubits(nrQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);

  const Types::qubits_vector sampledQubits{allQubits[0], allQubits[1],
                                           allQubits[2], allQubits[3]};
  constexpr size_t shots = 4000;

  const auto probabilities =
      ComputeMarginalProbabilities(qcsimStatevector, sampledQubits, nrQubits);

  const auto statevectorCounts =
      qcsimStatevector->SampleCounts(sampledQubits, shots);
  const auto pathIntegralCounts =
      qcsimPathIntegral->SampleCounts(sampledQubits, shots);

  CheckCountsAgainstProbabilities(statevectorCounts, probabilities, shots);
  CheckCountsAgainstProbabilities(pathIntegralCounts, probabilities, shots);
}

// Generate random Maestro circuits and verify that sampling through
// SampleCountsMany matches the exact marginal probabilities of the statevector
// for both qcsim's statevector and path integral simulators.
BOOST_DATA_TEST_CASE_F(PathIntegralFixture,
                       RandomCircuitSampleCountsManyMatchReferenceProbabilities,
                       bdata::xrange(5), trial) {
  BOOST_TEST(qcsimStatevector);
  BOOST_TEST(qcsimPathIntegral);

  qcsimStatevector->Reset();
  qcsimPathIntegral->Reset();

  std::mt19937 g(static_cast<std::mt19937::result_type>(0x2468ACE0u + trial));

  auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
  BOOST_REQUIRE(circuit);

  Circuits::OperationState statevectorState;
  statevectorState.AllocateBits(nrQubits);
  circuit->Execute(qcsimStatevector, statevectorState);

  Circuits::OperationState pathIntegralState;
  pathIntegralState.AllocateBits(nrQubits);
  circuit->Execute(qcsimPathIntegral, pathIntegralState);

  const Types::qubits_vector sampledQubits{5, 1, 7, 2};
  constexpr size_t shots = 4000;

  const auto probabilities =
      ComputeMarginalProbabilities(qcsimStatevector, sampledQubits, nrQubits);

  const auto statevectorCountsMany =
      qcsimStatevector->SampleCountsMany(sampledQubits, shots);
  const auto pathIntegralCountsMany =
      qcsimPathIntegral->SampleCountsMany(sampledQubits, shots);

  for (const auto& [outcomeBits, count] : statevectorCountsMany) {
    BOOST_TEST(outcomeBits.size() == sampledQubits.size());
    BOOST_TEST(count > 0);
  }

  for (const auto& [outcomeBits, count] : pathIntegralCountsMany) {
    BOOST_TEST(outcomeBits.size() == sampledQubits.size());
    BOOST_TEST(count > 0);
  }

  CheckCountsAgainstProbabilities(ConvertManyCounts(statevectorCountsMany),
                                  probabilities, shots);
  CheckCountsAgainstProbabilities(ConvertManyCounts(pathIntegralCountsMany),
                                  probabilities, shots);
}

// Generate random Maestro circuits that include mid-circuit measurements,
// classically controlled gates, and a final measurement on a random subset of
// qubits. Execute the circuit many times with both qcsim's statevector and
// path integral simulators, accumulate the classical-bit outcome counts, and
// verify that the two distributions are statistically consistent with each
// other.
BOOST_DATA_TEST_CASE_F(PathIntegralFixture,
                       RandomMixedCircuitMeasurementCountsMatch,
                       bdata::xrange(5), trial) {
  BOOST_TEST(qcsimStatevector);
  BOOST_TEST(qcsimPathIntegral);

  std::mt19937 g(static_cast<std::mt19937::result_type>(0xDEADBEEFu + trial));

  const auto info = GenerateRandomMixedCircuit(nrGates, nrQubits, g);
  BOOST_REQUIRE(info.circuit);
  BOOST_REQUIRE(!info.finalCbits.empty());

  constexpr size_t shots = 2000;
  const size_t nrOutcomes = 1ULL << info.finalCbits.size();

  std::unordered_map<Types::qubit_t, Types::qubit_t> svCounts;
  std::unordered_map<Types::qubit_t, Types::qubit_t> piCounts;

  for (size_t shot = 0; shot < shots; ++shot) {
    const auto svOutcome = RunOneShot(qcsimStatevector, info.circuit,
                                      info.totalCbits, info.finalCbits);
    const auto piOutcome = RunOneShot(qcsimPathIntegral, info.circuit,
                                      info.totalCbits, info.finalCbits);
    ++svCounts[svOutcome];
    ++piCounts[piOutcome];
  }

  // Compare the two empirical distributions against each other: treat the
  // statevector counts as the reference and check that the path integral counts
  // are within a generous statistical tolerance.
  Types::qubit_t totalSv = 0;
  for (const auto& [outcome, count] : svCounts) totalSv += count;
  BOOST_TEST(totalSv == shots);

  Types::qubit_t totalPi = 0;
  for (const auto& [outcome, count] : piCounts) totalPi += count;
  BOOST_TEST(totalPi == shots);

  for (size_t outcome = 0; outcome < nrOutcomes; ++outcome) {
    const auto svIt = svCounts.find(static_cast<Types::qubit_t>(outcome));
    const auto piIt = piCounts.find(static_cast<Types::qubit_t>(outcome));

    const double svProb = svIt == svCounts.end()
                              ? 0.0
                              : static_cast<double>(svIt->second) / shots;
    const double piProb = piIt == piCounts.end()
                              ? 0.0
                              : static_cast<double>(piIt->second) / shots;

    // Tolerance: allow for statistical fluctuations in both samples.
    const double refProb = (svProb + piProb) / 2.0;
    const double tolerance =
        0.05 + 6.0 * std::sqrt(refProb * (1.0 - refProb) / shots);

    BOOST_CHECK_SMALL(std::abs(svProb - piProb), tolerance);
  }
}

BOOST_AUTO_TEST_SUITE_END()
