/**
 * @file qasm.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Round-trip tests for the QASM parser and generator.
 */

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <complex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../qasm/CircQasm.h"
#include "qasm_test_fixture.h"

namespace bdata = boost::unit_test::data;

namespace {

using QasmVersion = qasm::CircToQasm<>::QasmVersion;

void CheckRandomRoundTrip(QasmTestFixture &fixture, int nrGates,
                          QasmVersion version) {
  const size_t nrStates = 1ULL << fixture.nrQubitsForRandomCirc;

  for (int trial = 0; trial < 5; ++trial) {
    fixture.GenerateCircuit(nrGates, fixture.nrQubitsForRandomCirc);
    fixture.randomCirc->Execute(fixture.qc, fixture.state);

    const std::string qasmStr =
        qasm::CircToQasm<>::Generate(fixture.randomCirc, version);
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    circuit->Execute(fixture.qc2, fixture.state);
    for (size_t basisState = 0; basisState < nrStates; ++basisState) {
      const double expected = fixture.qc->Probability(basisState);
      const double actual = fixture.qc2->Probability(basisState);
      BOOST_TEST(checkClose(std::complex<double>(expected, 0.),
                            std::complex<double>(actual, 0.), 0.0001),
                 "Probability mismatch for state |"
                     << basisState << ">: " << expected << " vs " << actual
                     << ", qasm: " << qasmStr);
    }

    fixture.randomCirc->Clear();
    fixture.resetCirc->Execute(fixture.qc, fixture.state);
    fixture.resetCirc->Execute(fixture.qc2, fixture.state);
    fixture.state.Reset();
  }
}

void CheckRandomRoundTripWithMeasurement(QasmTestFixture &fixture, int nrGates,
                                         QasmVersion version) {
  constexpr int nrShots = 5000;

  for (int iteration = 0; iteration < 5; ++iteration) {
    fixture.GenerateCircuit(nrGates, fixture.nrQubitsForRandomCirc, 0.025,
                            0.15);
    const std::string qasmStr =
        qasm::CircToQasm<>::Generate(fixture.randomCirc, version);

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    std::unordered_map<std::vector<bool>, size_t> expectedResults;
    std::unordered_map<std::vector<bool>, size_t> actualResults;
    for (int trial = 0; trial < nrShots; ++trial) {
      fixture.randomCirc->Execute(fixture.qc, fixture.state);
      ++expectedResults[fixture.state.GetAllBits()];
      fixture.state.Reset();

      circuit->Execute(fixture.qc2, fixture.state);
      ++actualResults[fixture.state.GetAllBits()];

      fixture.resetCirc->Execute(fixture.qc, fixture.state);
      fixture.resetCirc->Execute(fixture.qc2, fixture.state);
      fixture.state.Reset();
    }

    const auto checkDistribution = [nrShots](const auto &observed,
                                             const auto &reference) {
      for (const auto &[outcome, count] : observed) {
        const double probability =
            static_cast<double>(count) / static_cast<double>(nrShots);
        if (probability < 0.03) continue;

        double referenceProbability = 0.;
        const auto referenceIt = reference.find(outcome);
        if (referenceIt != reference.end()) {
          referenceProbability = static_cast<double>(referenceIt->second) /
                                 static_cast<double>(nrShots);
        }
        BOOST_CHECK_CLOSE(probability, referenceProbability,
                          referenceProbability < 0.1 ? 66 : 33);
      }
    };

    checkDistribution(expectedResults, actualResults);
    checkDistribution(actualResults, expectedResults);
    fixture.randomCirc->Clear();
  }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(qasm_tests)

BOOST_FIXTURE_TEST_CASE(InitializationTests, QasmTestFixture) {
  BOOST_TEST(qc);
  BOOST_TEST(qc2);
  BOOST_TEST(randomCirc);
  BOOST_TEST(resetCirc);
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsTest,
                       bdata::xrange(1, 30), nrGates) {
  CheckRandomRoundTrip(*this, nrGates, QasmVersion::V2);
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsWithMeasAndResetTest,
                       bdata::xrange(20, 40), nrGates) {
  CheckRandomRoundTripWithMeasurement(*this, nrGates, QasmVersion::V2);
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsV3Test,
                       bdata::xrange(1, 30), nrGates) {
  CheckRandomRoundTrip(*this, nrGates, QasmVersion::V3);
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsWithMeasAndResetV3Test,
                       bdata::xrange(20, 40), nrGates) {
  CheckRandomRoundTripWithMeasurement(*this, nrGates, QasmVersion::V3);
}

BOOST_AUTO_TEST_SUITE_END()
