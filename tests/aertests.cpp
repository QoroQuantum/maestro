/**
 * @file aertests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the Aer simulator.
 * 
 * Just tests for simulator creation, qubits allocation, initialization, a single X gate and measurements of all qubits.
 */

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include <framework/avx2_detect.hpp>

#undef min
#undef max

#include "../Simulators/Factory.h" // project being tested


struct AerSimulatorTestFixture
{
	AerSimulatorTestFixture()
	{
		aer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);
		aer->AllocateQubits(3);
		aer->Initialize();
	}

	~AerSimulatorTestFixture()
	{
	}

	std::shared_ptr<Simulators::ISimulator> aer;
};


BOOST_AUTO_TEST_SUITE(aer_simulator_tests)

BOOST_FIXTURE_TEST_CASE(test, AerSimulatorTestFixture)
{
	BOOST_REQUIRE(AER::is_avx2_supported());

	BOOST_TEST(aer);

	aer->ApplyX(0);

	auto res = aer->Measure({ 0, 1, 2 });

	BOOST_TEST(res == 1ULL);

	BOOST_CHECK_CLOSE(aer->Probability(res), 1.0, 0.000001);
}

BOOST_AUTO_TEST_SUITE_END()


