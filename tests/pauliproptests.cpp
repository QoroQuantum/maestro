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

struct PauliSimTestFixture {
  PauliSimTestFixture() {
    statevectorSim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    statevectorSim->AllocateQubits(nrQubitsForRandomCirc);
    statevectorSim->Initialize();

    qcsimPauliSim.SetNrQubits(nrQubitsForRandomCirc);

#ifdef __linux__
    if (Simulators::SimulatorsFactory::IsGpuLibraryAvailable()) {
      gpuPauliSim =
          Simulators::SimulatorsFactory::CreateGpuPauliPropagatorSimulator();
      gpuPauliSim->CreateSimulator(nrQubitsForRandomCirc);
    }
#endif
  }

  const unsigned int nrQubitsForRandomCirc = 8;
  const unsigned int maxNrGatesForRandomCirc = 100;
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


BOOST_AUTO_TEST_SUITE_END()
