/**
 * @file tests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The main file for the tests project.
 *
 * Doesn't do much, just avoids spurious memory leaks.
 * With the hack here, the tests won't report a memory leak anymore, but then VS reports one (it's not reported without the hack).
 * The reported leak is in qiskit aer and probably not real.
 */

#define BOOST_TEST_MODULE Conductor Test Module
#include <boost/test/included/unit_test.hpp> //single-header
#include <fstream>

// this is just to fix some bug with detecting memory leaks in qiskit-aer
#ifdef _WIN32
#ifdef _DEBUG

#define _CRTDBG_MAP_ALLOC

#include <stdlib.h>
#include <crtdbg.h>
#endif
#endif
//************************************************************

#undef min
#undef max

#define _TESTS_NO_EXCLUDE 1
#include "Tests.h"

#include "../Simulators/Factory.cpp"


bool checkClose(std::complex<double> a, std::complex<double> b, double dif)
{
	return std::abs(a.real() - b.real()) < dif && std::abs(a.imag() - b.imag()) < dif;
}


struct MyConfig {
	MyConfig() : test_log("tests.log") { boost::unit_test::unit_test_log.set_stream(test_log); }
	~MyConfig() { boost::unit_test::unit_test_log.set_stream(std::cout); }

	std::ofstream test_log;
};

// for logging to a file, uncomment this
//BOOST_GLOBAL_FIXTURE(MyConfig);

BOOST_AUTO_TEST_SUITE(main_tests)

BOOST_AUTO_TEST_CASE(avoid_spurious_memory_leaks)
{
	BOOST_TEST(true);
	// this is just to fix some bug with detecting memory leaks in qiskit-aer
	// on the other hand, turning this on will make Visual Studio to report memory leaks in the IDE if debugging
	// I must find better way
#ifdef _WIN32
#ifdef _DEBUG 
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
	_CrtDumpMemoryLeaks();
#endif
#endif
}

BOOST_AUTO_TEST_SUITE_END()