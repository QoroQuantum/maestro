/**
 * @file cliffordtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for stabilizer simulators.
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

struct CliffordSimTestFixture
{
	CliffordSimTestFixture()
	{
		aerClifford = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStabilizer);
		aerClifford->AllocateQubits(nrQubitsForRandomCirc);
		aerClifford->Initialize();

		qcsimClifford = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer);
		qcsimClifford->AllocateQubits(nrQubitsForRandomCirc);
		qcsimClifford->Initialize();

		circ = std::make_shared<Circuits::Circuit<>>();
		state.AllocateBits(nrQubitsForRandomCirc);

		resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubits(nrQubitsForRandomCirc);
		std::iota(qubits.begin(), qubits.end(), 0);
		resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));
	}


	// fills randomly the circuit with gates
	void GenerateCircuit(int nrGates)
	{
		std::random_device rd;
		std::mt19937 g(rd());

		auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
		auto dblGenIter = dblGen.begin();

		auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCZGateType));
		auto gateGenIter = gateGen.begin();

		// TODO: Maybe insert from time to time a random number generating 'gate' and a conditioned random one, those should not affect results?
		for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
		{
			// create a random gate and add it to the circuit

			// first, pick randomly three qubits (depending on the randomly chosen gate type, not all of them will be used)
			Types::qubits_vector qubits(nrQubitsForRandomCirc);
			std::iota(qubits.begin(), qubits.end(), 0);
			std::shuffle(qubits.begin(), qubits.end(), g);
			auto q1 = qubits[0];
			auto q2 = qubits[1];

			// now some random parameters, again, they might be ignored
			const double param1 = *dblGenIter;
			++dblGenIter;

			Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

			auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, param1, 0, 0, 0);
			if (!theGate->IsClifford())
			{
				--gateNr;
				continue;
			}

			circ->AddOperation(theGate);
		}
	}

	const unsigned int nrQubitsForRandomCirc = 8;
	std::shared_ptr<Simulators::ISimulator> aerClifford;
	std::shared_ptr<Simulators::ISimulator> qcsimClifford;
	std::shared_ptr<Circuits::Circuit<>> circ;
	std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
	Circuits::OperationState state;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(clifford_tests)

BOOST_FIXTURE_TEST_CASE(CliffordSimInitializationTest, CliffordSimTestFixture)
{
	BOOST_TEST(aerClifford);
	BOOST_TEST(qcsimClifford);
	BOOST_TEST(circ);
	BOOST_TEST(resetRandomCirc);
}


BOOST_DATA_TEST_CASE_F(CliffordSimTestFixture, RandomCircuitsTest, bdata::xrange(1, 20), nrGates)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	GenerateCircuit(nrGates);

	auto start = std::chrono::system_clock::now();
	circ->Execute(aerClifford, state);
	auto end = std::chrono::system_clock::now();
	double qiskitTime = std::chrono::duration<double>(end - start).count() * 1000.;

	start = std::chrono::system_clock::now();
	circ->Execute(qcsimClifford, state);
	end = std::chrono::system_clock::now();
	double qcsimTime = std::chrono::duration<double>(end - start).count() * 1000.;

	BOOST_TEST_MESSAGE("Time for qiskit aer Clifford: " << qiskitTime << " ms, time for qcsim Clifford: " << qcsimTime << " ms, qcsim is " << qiskitTime / qcsimTime << " faster");

	auto qcsimProbs = qcsimClifford->AllProbabilities();
	BOOST_TEST(qcsimProbs.size() == nrStates);

	auto aerProbs = aerClifford->AllProbabilities();
	BOOST_TEST(qcsimProbs.size() == aerProbs.size());

	// now check the results, they should be the same!

	for (size_t state = 0; state < nrStates; ++state)
	{
		// does not seem to work with the aer simulator (well, I fixed it for now, let's see if it ends in the qiskit aer as well)
		const auto aaer = aerClifford->Probability(state);
		const auto aqc = qcsimClifford->Probability(state);

		//BOOST_TEST_MESSAGE("State: " << state << ", Value qcsim: " << pqc << ", Value aer: " << paer);

		BOOST_CHECK_PREDICATE(checkClose, (aaer)(aqc)(0.001));

		const auto paer = aerProbs[state];
		const auto pqc = qcsimProbs[state];

		if (pqc < 1e-5 && paer < 1e-5)
			continue;

		BOOST_CHECK_CLOSE(paer, pqc, 0.001);
	}
	
	resetRandomCirc->Execute(aerClifford, state);
	resetRandomCirc->Execute(qcsimClifford, state);

	circ->Clear();
	state.Reset();
}

BOOST_AUTO_TEST_SUITE_END()
