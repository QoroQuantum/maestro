/**
 * @file qcsimtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the qcsim simulator.
 *
 * Just tests for simulator creation, qubits allocation, initialization, a
 * single X gate and measurements of all qubits.
 */

#include <boost/test/unit_test.hpp>

#undef min
#undef max

#include "../Simulators/Factory.h"  // project being tested

BOOST_AUTO_TEST_SUITE(qcsim_simulator_tests)

BOOST_AUTO_TEST_CASE(test) {
  auto qcsim = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  BOOST_TEST(qcsim);

  qcsim->AllocateQubits(3);
  qcsim->Initialize();

  qcsim->ApplyX(0);

  auto res = qcsim->Measure({0, 1, 2});

  BOOST_TEST(res == 1ULL);
  BOOST_CHECK_CLOSE(qcsim->Probability(res), 1.0, 0.000001);
}

BOOST_AUTO_TEST_SUITE_END()
