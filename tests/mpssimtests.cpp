/**
 * @file mpssimtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for MPS simulators.
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

struct MPSSimTestFixture
{
	MPSSimTestFixture()
	{
#ifdef __linux__
		Simulators::SimulatorsFactory::InitGpuLibrary();
#endif

		aerMPS = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
		aerMPS->AllocateQubits(nrQubitsForRandomCirc);
		aerMPS->Initialize();

		qcsimMPS = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
		qcsimMPS->AllocateQubits(nrQubitsForRandomCirc);
		qcsimMPS->Initialize();

		aerMPS50 = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
		aerMPS50->AllocateQubits(50);
		aerMPS50->Configure("matrix_product_state_max_bond_dimension", "20");
		aerMPS50->Configure("matrix_product_state_truncation_threshold", "0.0001");
		aerMPS50->Initialize();

		qcsimMPS50 = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
		qcsimMPS50->AllocateQubits(50);
		qcsimMPS50->Configure("matrix_product_state_max_bond_dimension", "20");
		qcsimMPS50->Configure("matrix_product_state_truncation_threshold", "0.0001");
		qcsimMPS50->Initialize();

#ifdef __linux__
		gpusimMPS = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
		if (gpusimMPS)
		{
			gpusimMPS->AllocateQubits(nrQubitsForRandomCirc);
			gpusimMPS->Initialize();
		}

		gpusimMPS50 = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
		if (gpusimMPS50)
		{
			gpusimMPS50->AllocateQubits(50);
			gpusimMPS50->Configure("matrix_product_state_max_bond_dimension", "20");
			gpusimMPS50->Configure("matrix_product_state_truncation_threshold", "0.0001");
			gpusimMPS50->Initialize();
		}
#endif

		circ = std::make_shared<Circuits::Circuit<>>();
		state.AllocateBits(nrQubitsForRandomCirc);

		circ50 = std::make_shared<Circuits::Circuit<>>();
		state50.AllocateBits(50);

		resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubits(nrQubitsForRandomCirc);
		std::iota(qubits.begin(), qubits.end(), 0);
		resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

		resetRandomCirc50 = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubits50(50);
		std::iota(qubits50.begin(), qubits50.end(), 0);
		resetRandomCirc50->AddOperation(std::make_shared<Circuits::Reset<>>(qubits50));
	}


	// fills randomly the circuit with gates
	void GenerateCircuit(int nrGates)
	{
		std::random_device rd;
		std::mt19937 g(rd());

		auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
		auto dblGenIter = dblGen.begin();

		auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
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
			auto q3 = qubits[2];

			// now some random parameters, again, they might be ignored
			const double param1 = *dblGenIter;
			++dblGenIter;
			const double param2 = *dblGenIter;
			++dblGenIter;
			const double param3 = *dblGenIter;
			++dblGenIter;
			const double param4 = *dblGenIter;
			++dblGenIter;

			Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

			auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, param1, param2, param3, param4);
			circ->AddOperation(theGate);
		}
	}


	void GenerateCircuit50(int nrGates)
	{
		std::random_device rd;
		std::mt19937 g(rd());

		auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
		auto dblGenIter = dblGen.begin();

		auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
		auto gateGenIter = gateGen.begin();

		// TODO: Maybe insert from time to time a random number generating 'gate' and a conditioned random one, those should not affect results?
		for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
		{
			// create a random gate and add it to the circuit

			// first, pick randomly three qubits (depending on the randomly chosen gate type, not all of them will be used)
			Types::qubits_vector qubits(50);
			std::iota(qubits.begin(), qubits.end(), 0);
			std::shuffle(qubits.begin(), qubits.end(), g);
			auto q1 = qubits[0];
			auto q2 = qubits[1];
			auto q3 = qubits[2];

			// now some random parameters, again, they might be ignored
			const double param1 = *dblGenIter;
			++dblGenIter;
			const double param2 = *dblGenIter;
			++dblGenIter;
			const double param3 = *dblGenIter;
			++dblGenIter;
			const double param4 = *dblGenIter;
			++dblGenIter;

			Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

			auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, param1, param2, param3, param4);
			circ50->AddOperation(theGate);
		}

		Types::qubits_vector mqubits(50);
		std::iota(mqubits.begin(), mqubits.end(), 0);
		std::shuffle(mqubits.begin(), mqubits.end(), g);

		// only 3 measurements on randomly chosen qubits to avoid having too many results
		for (int q = 0; q < 5; ++q)
			circ50->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement({ {mqubits[q], mqubits[q]} }));
	}

	const unsigned int nrQubitsForRandomCirc = 5;
	std::shared_ptr<Simulators::ISimulator> aerMPS;
	std::shared_ptr<Simulators::ISimulator> qcsimMPS;
	std::shared_ptr<Simulators::ISimulator> aerMPS50;
	std::shared_ptr<Simulators::ISimulator> qcsimMPS50;

#ifdef __linux__
	std::shared_ptr<Simulators::ISimulator> gpusimMPS;
	std::shared_ptr<Simulators::ISimulator> gpusimMPS50;
#endif

	std::shared_ptr<Circuits::Circuit<>> circ;
	std::shared_ptr<Circuits::Circuit<>> circ50;
	std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
	std::shared_ptr<Circuits::Circuit<>> resetRandomCirc50;
	Circuits::OperationState state;
	Circuits::OperationState state50;

	const unsigned int nrShots = 5000;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(mps_tests)

BOOST_FIXTURE_TEST_CASE(MPSSimInitializationTest, MPSSimTestFixture)
{
	BOOST_TEST(aerMPS);
	BOOST_TEST(qcsimMPS);
	BOOST_TEST(aerMPS50);
	BOOST_TEST(qcsimMPS50);
	BOOST_TEST(circ);
	BOOST_TEST(circ50);
	BOOST_TEST(resetRandomCirc);
	BOOST_TEST(resetRandomCirc50);
}


BOOST_DATA_TEST_CASE_F(MPSSimTestFixture, RandomCircuitsTest, bdata::xrange(1, 20), nrGates)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	GenerateCircuit(nrGates);

	auto start = std::chrono::system_clock::now();
	circ->Execute(aerMPS, state);
	auto end = std::chrono::system_clock::now();
	double qiskitTime = std::chrono::duration<double>(end - start).count() * 1000.;

	start = std::chrono::system_clock::now();
	circ->Execute(qcsimMPS, state);
	end = std::chrono::system_clock::now();
	double qcsimTime = std::chrono::duration<double>(end - start).count() * 1000.;

	BOOST_TEST_MESSAGE("Time for qiskit aer MPS: " << qiskitTime << " ms, time for qcsim MPS: " << qcsimTime << " ms, qcsim is " <<  qiskitTime / qcsimTime << " faster");

#ifdef __linux__
	if (gpusimMPS)
	{
		start = std::chrono::system_clock::now();
		circ->Execute(gpusimMPS, state);
		end = std::chrono::system_clock::now();
		double gpusimTime = std::chrono::duration<double>(end - start).count() * 1000.;

		BOOST_TEST_MESSAGE("Time for gpu MPS: " << gpusimTime << " ms, gpu is " << qiskitTime / gpusimTime << " faster than qiskit aer mps");
	}
#endif


	auto qcsimProbs = qcsimMPS->AllProbabilities();
	BOOST_TEST(qcsimProbs.size() == nrStates);

	auto aerProbs = aerMPS->AllProbabilities();
	BOOST_TEST(aerProbs.size() == nrStates);

#ifdef __linux__
	std::vector<double> gpusimProbs;
	if (gpusimMPS)
	{
		gpusimProbs = gpusimMPS->AllProbabilities();
		BOOST_TEST(gpusimProbs.size() == nrStates);
	}
#endif

	// now check the results, they should be the same!
	
	for (size_t state = 0; state < nrStates; ++state)
	{		
		const auto aaer = aerMPS->Amplitude(state);
		const auto aqc = qcsimMPS->Amplitude(state);

		BOOST_CHECK_PREDICATE(checkClose, (aaer)(aqc)(0.0001));

#ifdef __linux__
		if (gpusimMPS)
		{
			const auto agpusim = gpusimMPS->Amplitude(state);
			BOOST_CHECK_PREDICATE(checkClose, (aaer)(agpusim)(0.0001));
		}
#endif

		const auto paer = aerProbs[state];
		const auto pqc = qcsimProbs[state];

		if (paer < 1e-4 && pqc < 1e-4)
			continue;

		BOOST_CHECK_CLOSE(paer, pqc, 0.1);

#ifdef __linux__
		if (gpusimMPS)
		{
			const auto pgpusim = gpusimProbs[state];
			if (pgpusim < 1e-4)
				continue;
			BOOST_CHECK_PREDICATE(checkClose, (paer)(pgpusim)(0.1));
		}
#endif
	}

	
	resetRandomCirc->Execute(aerMPS, state);
	resetRandomCirc->Execute(qcsimMPS, state);

#ifdef __linux__
	if (gpusimMPS)
		resetRandomCirc->Execute(gpusimMPS, state);
#endif

	circ->Clear();
	state.Reset();
}

// this is quite slow, I'll leave it here with the number of tests/gates reduced (originally it started from 100)

BOOST_DATA_TEST_CASE_F(MPSSimTestFixture, RandomCircuitsTest50, bdata::xrange(30, 33), nrGates)
{
	GenerateCircuit50(nrGates);

	std::unordered_map<std::vector<bool>, size_t> measResultsQcSim;
	std::unordered_map<std::vector<bool>, size_t> measResultsQiskit;

	auto start = std::chrono::system_clock::now();
	
	for (size_t shot = 0; shot < nrShots; ++shot)
	{
		circ50->Execute(aerMPS50, state50);

		measResultsQiskit[state50.GetAllBits()]++;

		resetRandomCirc50->Execute(aerMPS50, state50);
		state50.Reset();
	}
	
	auto end = std::chrono::system_clock::now();
	double qiskitTime = std::chrono::duration<double>(end - start).count() * 1000.;

	start = std::chrono::system_clock::now();
	std::vector<bool> executed;

	for (size_t shot = 0; shot < nrShots; ++shot)
	{
		//circ50->Execute(qcsimMPS50, state50);
		if (shot == 0)
		{
			executed = circ50->ExecuteNonMeasurements(qcsimMPS50, state50);
			qcsimMPS50->SaveState();
		}
		else
			qcsimMPS50->RestoreState();
			
		circ50->ExecuteMeasurements(qcsimMPS50, state50, executed);

		measResultsQcSim[state50.GetAllBits()]++;

		//resetRandomCirc50->Execute(qcsimMPS50, state50);
		//state50.Reset();
	}

	resetRandomCirc50->Execute(qcsimMPS50, state50);
	state50.Reset();
	
	end = std::chrono::system_clock::now();
	double qcsimTime = std::chrono::duration<double>(end - start).count() * 1000.;

	BOOST_TEST_MESSAGE("Time for qiskit aer MPS: " << qiskitTime << " ms, time for qcsim MPS: " << qcsimTime << " ms, qcsim is " << qiskitTime / qcsimTime << " faster");

#ifdef __linux__
	std::unordered_map<std::vector<bool>, size_t> measResultsGpuSim;

	if (gpusimMPS50)
	{
		start = std::chrono::system_clock::now();

		for (size_t shot = 0; shot < nrShots; ++shot)
		{
			//circ50->Execute(gpusimMPS50, state50);
			if (shot == 0)
			{
				executed = circ50->ExecuteNonMeasurements(gpusimMPS50, state50);
				gpusimMPS50->SaveState();
			}
			else
				gpusimMPS50->RestoreState();

			circ50->ExecuteMeasurements(gpusimMPS50, state50, executed);

			measResultsGpuSim[state50.GetAllBits()]++;

			//resetRandomCirc50->Execute(gpusimMPS50, state50);
			//state50.Reset();
		}

		resetRandomCirc50->Execute(gpusimMPS50, state50);
		state50.Reset();

		end = std::chrono::system_clock::now();
		double gpusimTime = std::chrono::duration<double>(end - start).count() * 1000.;

		BOOST_TEST_MESSAGE("Time for gpu MPS: " << gpusimTime << " ms, gpu is " << qiskitTime / gpusimTime << " faster than qiskit aer mps");
	}
#endif


	for (const auto& [key, cnt] : measResultsQcSim)
	{
		double val = static_cast<double>(cnt) / static_cast<double>(nrShots);

		if (val < 0.03) continue;

		double val2 = 0;
		if (measResultsQiskit.find(key) != measResultsQiskit.end())
			val2 = static_cast<double>(measResultsQiskit[key]) / static_cast<double>(nrShots);

		BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
	}

	for (const auto& [key, cnt] : measResultsQiskit)
	{
		double val = static_cast<double>(cnt) / static_cast<double>(nrShots);
		if (val < 0.03) continue;
		
		double val2 = 0;
		if (measResultsQcSim.find(key) != measResultsQcSim.end())
			val2 = static_cast<double>(measResultsQcSim[key]) / static_cast<double>(nrShots);
		BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
	}

#ifdef __linux__
	if (gpusimMPS50)
	{
		for (const auto& [key, cnt] : measResultsGpuSim)
		{
			double val = static_cast<double>(cnt) / static_cast<double>(nrShots);

			if (val < 0.03) continue;

			double val2 = 0;
			if (measResultsQiskit.find(key) != measResultsQiskit.end())
				val2 = static_cast<double>(measResultsQiskit[key]) / static_cast<double>(nrShots);

			BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
		}

		for (const auto& [key, cnt] : measResultsQiskit)
		{
			double val = static_cast<double>(cnt) / static_cast<double>(nrShots);
			if (val < 0.03) continue;

			double val2 = 0;
			if (measResultsGpuSim.find(key) != measResultsGpuSim.end())
				val2 = static_cast<double>(measResultsGpuSim[key]) / static_cast<double>(nrShots);
			BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
		}
	}
#endif

	circ50->Clear();
}



BOOST_AUTO_TEST_SUITE_END()
