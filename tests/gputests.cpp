/**
 * @file gputests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the gpu simulator.
 *
 * Just tests for simulator creation, qubits allocation, initialization, a
 * single X gate and measurements of all qubits.
 */

#ifdef __linux__

#include <boost/test/unit_test.hpp>

#undef min
#undef max

#include "../Simulators/Factory.h"  // project being tested

BOOST_AUTO_TEST_SUITE(gpu_simulator_tests)

BOOST_AUTO_TEST_CASE(test) {
  Simulators::SimulatorsFactory::InitGpuLibrary();

  std::shared_ptr<Simulators::ISimulator> gpusim;

  try {
    gpusim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kStatevector);
    if (gpusim) {
      gpusim->AllocateQubits(3);
      gpusim->Initialize();

      gpusim->ApplyX(0);

      auto res = gpusim->Measure({0, 1, 2});

      BOOST_TEST(res == 1ULL);
      BOOST_CHECK_CLOSE(gpusim->Probability(res), 1.0, 0.000001);
    }
    gpusim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kMatrixProductState);
    if (gpusim) {
      gpusim->AllocateQubits(3);
      gpusim->Initialize();

      gpusim->ApplyX(0);

      auto res = gpusim->Measure({0, 1, 2});

      BOOST_TEST(res == 1ULL);
      BOOST_CHECK_CLOSE(gpusim->Probability(res), 1.0, 0.000001);
    }
    gpusim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kTensorNetwork);
    if (gpusim) {
      gpusim->AllocateQubits(3);
      gpusim->Initialize();

      gpusim->ApplyX(0);

      auto res = gpusim->Measure({0, 1, 2});

      BOOST_TEST(res == 1ULL);
      BOOST_CHECK_CLOSE(gpusim->Probability(res), 1.0, 0.000001);
    }
  } catch (const std::exception& e) {
    BOOST_TEST_MESSAGE(
        "Exception: "
        << e.what() << ". Please ensure the proper gpu library is available.");
  } catch (...) {
    BOOST_TEST_MESSAGE("Unknown exception");
  }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
