/**
 * @file mpsnetwtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the MPS simulator running inside a SimpleDisconnectedNetwork
 * with a single host. Uses a variable number of qubits (8..14) and random
 * circuits that may contain mid-circuit measurements plus final
 * measurements.
 *
 * Execution is performed via RepeatedExecuteOnHost. The resulting statistics
 * are compared against the same circuit executed on a statevector based
 * SimpleDisconnectedNetwork.
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
#include <unordered_map>
#include <vector>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"

#include "../Network/SimpleDisconnectedNetwork.h"

namespace {

// Build a random circuit on `nrQubits` qubits with `nrGates` random gates,
// optionally inserting mid-circuit measurements (and classically conditioned
// gates) and always ending with a final measurement on a non-empty random
// subset of the qubits.
std::shared_ptr<Circuits::Circuit<>> GenerateRandomCircuitWithMeasurements(
    int nrGates, int nrQubits, std::mt19937& g) {
  auto circuit = Circuits::CircuitFactory<>::CreateCircuit();

  std::uniform_real_distribution<double> paramDist(-2.0 * M_PI, 2.0 * M_PI);
  // avoid 3-qubit gates so the circuits stay reasonably cheap to simulate
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
  std::uniform_int_distribution<int> qubitDist(0, nrQubits - 1);
  std::bernoulli_distribution measureNow(0.15);

  // classical bits: nrQubits for the mid-circuit ones, nrQubits more for the
  // final measurement
  const size_t midCbitBase = 0;
  const size_t finalCbitBase = static_cast<size_t>(nrQubits);

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

    if (measureNow(g)) {
      const int mQubit = qubitDist(g);
      const size_t mCbit = midCbitBase + static_cast<size_t>(mQubit);
      circuit->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(
          {{static_cast<Types::qubit_t>(mQubit), mCbit}}));

      // Sometimes add a classically-conditioned gate based on the just
      // measured bit.
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

  // final measurement on a random non-empty subset of qubits
  Types::qubits_vector allQubits(nrQubits);
  std::iota(allQubits.begin(), allQubits.end(), 0);
  std::shuffle(allQubits.begin(), allQubits.end(), g);

  std::uniform_int_distribution<int> subsetSizeDist(1, nrQubits);
  const int finalMeasCount = subsetSizeDist(g);

  std::vector<std::pair<Types::qubit_t, size_t>> finalPairs;
  for (int k = 0; k < finalMeasCount; ++k) {
    const size_t cbit = finalCbitBase + static_cast<size_t>(k);
    finalPairs.push_back({static_cast<Types::qubit_t>(allQubits[k]), cbit});
  }

  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement(finalPairs));

  return circuit;
}

void CheckCountsAgainstStatevector(
    const std::unordered_map<std::vector<bool>, Types::qubit_t>& counts,
    const std::unordered_map<std::vector<bool>, Types::qubit_t>&
        statevectorCounts,
    size_t shots) {
  Types::qubit_t totalCounts = 0;
  for (const auto& [outcome, count] : counts) totalCounts += count;

  BOOST_TEST(totalCounts == shots);
  BOOST_REQUIRE(!counts.empty());

  for (const auto& meas : counts) {
    const auto it = statevectorCounts.find(meas.first);
    const double svProbability =
        it == statevectorCounts.end()
            ? 0.0
            : static_cast<double>(it->second) / shots;

    const double mpsProbability = static_cast<double>(meas.second) / shots;
    // 6-sigma binomial tolerance plus a small absolute margin
    const double tolerance =
        0.01 + 6.0 * std::sqrt(mpsProbability * (1.0 - mpsProbability) /
                               static_cast<double>(shots));

    BOOST_CHECK_SMALL(std::abs(svProbability - mpsProbability), tolerance);
  }
}

std::shared_ptr<Network::SimpleDisconnectedNetwork<>> MakeNetwork(
    int nrQubits, Simulators::SimulationType method) {
  // single host with `nrQubits` qubits and matching number of classical bits
  // (twice the number, to leave room for mid-circuit + final measurements)
  const size_t cbits = static_cast<size_t>(2 * nrQubits);
  auto net = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
      std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
      std::vector<size_t>{cbits});
  net->RemoveAllOptimizationSimulatorsAndAdd(
      Simulators::SimulatorType::kQCSim, method);
  net->CreateSimulator();

  if (method == Simulators::SimulationType::kMatrixProductState) {
    net->GetSimulator()->Configure("matrix_product_state_max_bond_dimension", "50");
  }

  return net;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(MpsNetwTests)

// nrQubits varies from 8 to 14 inclusive
BOOST_DATA_TEST_CASE(
    MpsNetworkRandomCircuitWithMeasurementsMatchesStatevector,
    bdata::xrange(8, 15),
    nrQubits) {
  std::mt19937 g(static_cast<std::mt19937::result_type>(0xD15CD15Cu +
                                                        nrQubits));

  for (int trial = 0; trial < 3; ++trial) {
    // a few gates per qubit makes a non-trivial but still tractable circuit
    const int nrGates = 3 * nrQubits;

    auto circuit = GenerateRandomCircuitWithMeasurements(nrGates, nrQubits, g);
    BOOST_REQUIRE(circuit);

    auto mpsNet =
        MakeNetwork(nrQubits, Simulators::SimulationType::kMatrixProductState);
    auto svNet =
        MakeNetwork(nrQubits, Simulators::SimulationType::kStatevector);

    constexpr size_t shots = 4000;

    const auto statevectorCounts =
        svNet->RepeatedExecuteOnHost(circuit, 0, shots);
    const auto mpsCounts = mpsNet->RepeatedExecuteOnHost(circuit, 0, shots);

    CheckCountsAgainstStatevector(mpsCounts, statevectorCounts, shots);
  }
}

BOOST_AUTO_TEST_SUITE_END()
