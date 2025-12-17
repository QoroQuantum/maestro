/**
 * @file compsimtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for composite quantum computing simulators.
 *
 * The tests check the execution of circtuits on the composite simulators against the execution of the same circuits on the normal simulators.
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

#include <fstream>

// project being tested
#include "Simulators/Factory.h"

#include "../Circuit/Circuit.h"
#include "../Circuit/Conditional.h"
#include "../Circuit/Measurements.h"
#include "../Circuit/QuantumGates.h"
#include "../Circuit/RandomOp.h"
#include "../Circuit/Reset.h"
#include "../Circuit/Factory.h"

struct CompositeSimulatorsTestFixture
{
	CompositeSimulatorsTestFixture()
	{
		state.AllocateBits(nrQubitsForRandomCirc);
		tstate.AllocateBits(3);

		aer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);
		aer->AllocateQubits(nrQubitsForRandomCirc);
		aer->Initialize();

		qc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
		qc->AllocateQubits(nrQubitsForRandomCirc);
		qc->Initialize();

		setCirc = std::make_shared<Circuits::Circuit<>>();
		resetCirc = std::make_shared<Circuits::Circuit<>>();
		measureCirc = std::make_shared<Circuits::Circuit<>>();
		teleportationCirc = std::make_shared<Circuits::Circuit<>>();
		genTeleportationCirc = std::make_shared<Circuits::Circuit<>>();

		randomCirc = Circuits::CircuitFactory<>::CreateCircuit();

		resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubits(nrQubitsForRandomCirc);
		std::iota(qubits.begin(), qubits.end(), 0);
		resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));


		setCirc->AddOperation(std::make_shared<Circuits::XGate<>>(0));
		resetCirc->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{ 0, 1, 2 }));

		measureCirc->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{ {0, 0}, { 1, 1 }, { 2, 2 } }));


		// Teleportation circuit
		// ************************************************************************************************

		// set the qubit to be teleported to 1

		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, 0));

		// entangle qubit 1 and 2
		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 1));
		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCXGateType, 1, 2));

		// teleport
		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCXGateType, 0, 1));
		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 0));


		// the order of measurements does not matter
		// qubit 0 measured in classical bit 0, qubit 1 measured in bit 1

		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(std::vector<std::pair<Types::qubit_t, size_t>>{ {0, 0}, { 1, 1 } }));

		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, 2),
			Circuits::CircuitFactory<>::CreateEqualCondition(std::vector<size_t>{ 1 }, std::vector<bool>{ true })));

		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kZGateType, 2),
			Circuits::CircuitFactory<>::CreateEqualCondition(std::vector<size_t>{ 0 }, std::vector<bool>{ true })));

		teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(std::vector<std::pair<Types::qubit_t, size_t>>{{2, 2}}));

		// ************************************************************************************************

		// without setting the qubit at the beginning to 1 and without the last measurement
		// instead set a generic state for the qubit at the beginning and check it to be the same in the teleported qubit at the end
		//genTeleportationCirc->SetOperations(std::vector<std::shared_ptr<Circuits::IOperation<>>>(teleportationCirc->GetOperations().begin() + 1, teleportationCirc->GetOperations().end() - 1));
		// some other way, for testing the factory generated one:
		genTeleportationCirc->AddOperations(Circuits::CircuitFactory<>::CreateTeleportationCircuit(1, 2, 0, 0, 1));
	}

	~CompositeSimulatorsTestFixture()
	{
	}

	// fills randomly the circuit with gates
	void GenerateCircuit(int nrGates)
	{
		std::random_device rd;
		std::mt19937 g(rd());

		auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
		auto dblGenIter = dblGen.begin();

		auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType/*Circuits::QuantumGateType::kSwapGateType*//*Circuits::QuantumGateType::kUGateType)*/));
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
			randomCirc->AddOperation(theGate);
		}
	}

	std::shared_ptr<Simulators::ISimulator> aer;
	std::shared_ptr<Simulators::ISimulator> qc;

	std::shared_ptr<Circuits::Circuit<>> setCirc;
	std::shared_ptr<Circuits::Circuit<>> resetCirc;
	std::shared_ptr<Circuits::Circuit<>> measureCirc;
	std::shared_ptr<Circuits::Circuit<>> teleportationCirc;
	std::shared_ptr<Circuits::Circuit<>> genTeleportationCirc;

	Circuits::OperationState tstate;
	Circuits::OperationState state;

	// for testing randomly generated circuits

	//they can handle even 28 qubits, even the non-composite ones
#ifdef _DEBUG
	const unsigned int nrQubitsForRandomCirc = 12;
#elif defined(_FAST_TESTS)
	const unsigned int nrQubitsForRandomCirc = 12;
#else
	const unsigned int nrQubitsForRandomCirc = 18;
#endif
	
	std::shared_ptr<Circuits::Circuit<>> randomCirc;
	std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(convsim_tests)

BOOST_FIXTURE_TEST_CASE(CircuitsInitializationTests, CompositeSimulatorsTestFixture)
{
	BOOST_TEST(aer);
	BOOST_TEST(qc);

	BOOST_TEST(setCirc);
	BOOST_TEST(resetCirc);
	BOOST_TEST(measureCirc);
	BOOST_TEST(teleportationCirc);
	BOOST_TEST(genTeleportationCirc);

	BOOST_TEST(randomCirc);
	BOOST_TEST(resetRandomCirc);
}

BOOST_FIXTURE_TEST_CASE(SimpleTest, CompositeSimulatorsTestFixture)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	qc->ApplyX(0);
	qc->ApplySwap(0, 1);

	std::shared_ptr<Simulators::ISimulator> compqc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim);
	compqc->AllocateQubits(nrQubitsForRandomCirc);
	compqc->Initialize();

	compqc->ApplyX(0);
	compqc->ApplySwap(0, 1);

	// now check the results, they should be the same!
	for (size_t state = 0; state < nrStates; ++state)
	{
		std::complex<double> aaqc = qc->Amplitude(state);
		std::complex<double> aacompqc = compqc->Amplitude(state);

		// TODO: if this fails, do something to identify the offending gate!
		// rebuild the circuit one by one with the already exising gates from the failed circuit
		// check the results until it fails, then print the last gate in the first circuit that fails
		BOOST_CHECK_PREDICATE(checkClose, (aaqc)(aacompqc)(0.000001));
	}

	resetRandomCirc->Execute(qc, state);
	resetRandomCirc->Execute(compqc, state);

	// check if reset is ok
	BOOST_TEST(compqc->Probability(0), 1.);
	for (size_t state = 1; state < nrStates; ++state)
	{
		std::complex<double> aaqc = compqc->Amplitude(state);

		BOOST_CHECK_PREDICATE(checkClose, (aaqc)(0.)(0.000001));
	}
}


BOOST_FIXTURE_TEST_CASE(SimpleTest2, CompositeSimulatorsTestFixture)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	qc->ApplyX(0);
	qc->ApplySwap(0, 1);
	qc->ApplySwap(1, 2);

	std::shared_ptr<Simulators::ISimulator> compqc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim);
	compqc->AllocateQubits(nrQubitsForRandomCirc);
	compqc->Initialize();

	compqc->ApplyX(0);
	compqc->ApplySwap(0, 1);
	compqc->ApplySwap(1, 2);

	// now check the results, they should be the same!
	for (size_t state = 0; state < nrStates; ++state)
	{
		std::complex<double> aaqc = qc->Amplitude(state);
		std::complex<double> aacompqc = compqc->Amplitude(state);

		// TODO: if this fails, do something to identify the offending gate!
		// rebuild the circuit one by one with the already exising gates from the failed circuit
		// check the results until it fails, then print the last gate in the first circuit that fails
		BOOST_CHECK_PREDICATE(checkClose, (aaqc)(aacompqc)(0.000001));
	}

	resetRandomCirc->Execute(qc, state);
	resetRandomCirc->Execute(compqc, state);

	// check if reset is ok
	BOOST_TEST(compqc->Probability(0), 1.);
	for (size_t state = 1; state < nrStates; ++state)
	{
		std::complex<double> aaqc = compqc->Amplitude(state);

		BOOST_CHECK_PREDICATE(checkClose, (aaqc)(0.)(0.000001));
	}
}




BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, TeleportationCompAerTest, bdata::xrange(5), ind)
{
	std::shared_ptr<Simulators::ISimulator> compaer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQiskitAer);
	compaer->AllocateQubits(3);
	compaer->Initialize();

	if (ind % 2)
	{
		// start with a different state to teleport (the teleportation circuit applies a not on the first qubit in the beginning)
		compaer->ApplyX(0);
	}

	teleportationCirc->Execute(compaer, tstate);

	BOOST_TEST(tstate.GetAllBits()[2] == ((ind % 2) ? false : true));

	// leave it in the 0 state
	resetCirc->Execute(compaer, tstate);
	measureCirc->Execute(compaer, tstate);

	BOOST_TEST(tstate.GetAllBits() == std::vector<bool>({ false, false, false }));
}


BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, TeleportationCompQcSimTest, bdata::xrange(5), ind)
{
	std::shared_ptr<Simulators::ISimulator> compqc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim);
	compqc->AllocateQubits(3);
	compqc->Initialize();

	if (ind % 2)
	{
		// start with a different state to teleport (the teleportation circuit applies a not on the first qubit in the beginning)
		compqc->ApplyX(0);
	}

	teleportationCirc->Execute(compqc, tstate);

	BOOST_TEST(tstate.GetAllBits()[2] == ((ind % 2) ? false : true));

	// leave it in the 0 state
	resetCirc->Execute(compqc, tstate);
	measureCirc->Execute(compqc, tstate);

	BOOST_TEST(tstate.GetAllBits() == std::vector<bool>({ false, false, false }));
}

// methods:
// bdata::make({ 0., 0.2, 0.4, 0.6, 0.8, 1. })
// equivalent: bdata::xrange(0.0, 1.0, 0.2)
// below is using random value
// ^ is for 'zip', * is for 'cartesian product'

BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, GenTeleportCompAerTest, (bdata::random() ^ bdata::xrange(5))* bdata::xrange(5), theta, index1, index2)
{
	std::shared_ptr<Simulators::ISimulator> compaer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQiskitAer);
	compaer->AllocateQubits(3);
	compaer->Initialize();

	compaer->ApplyRx(0, theta);

	// now get the values to be compared with the teleported one
	std::complex<double> a = compaer->Amplitude(0);
	std::complex<double> b = compaer->Amplitude(1);

	genTeleportationCirc->Execute(compaer, tstate);

	bool b1 = tstate.GetAllBits()[0];
	bool b2 = tstate.GetAllBits()[1];

	size_t outcome = 0;
	if (b1) outcome |= 1;
	if (b2) outcome |= 2;

	std::complex<double> ta = compaer->Amplitude(outcome);
	std::complex<double> tb = compaer->Amplitude(outcome | 4);

	// check the teleported state
	BOOST_CHECK_PREDICATE(checkClose, (a)(ta)(0.000001));
	BOOST_CHECK_PREDICATE(checkClose, (b)(tb)(0.000001));

	// leave it in the 0 state
	resetCirc->Execute(compaer, tstate);
	measureCirc->Execute(compaer, tstate);

	BOOST_TEST(tstate.GetAllBits() == std::vector<bool>({ false, false, false }));
}

BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, GenTeleportCompQcSimTest, (bdata::random() ^ bdata::xrange(5))* bdata::xrange(5), theta, index1, index2)
{
	std::shared_ptr<Simulators::ISimulator> compqc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim);
	compqc->AllocateQubits(3);
	compqc->Initialize();

	compqc->ApplyRx(0, theta);

	// now get the values to be compared with the teleported one
	std::complex<double> a = compqc->Amplitude(0);
	std::complex<double> b = compqc->Amplitude(1);

	genTeleportationCirc->Execute(compqc, tstate);

	bool b1 = tstate.GetAllBits()[0];
	bool b2 = tstate.GetAllBits()[1];

	size_t outcome = 0;
	if (b1) outcome |= 1;
	if (b2) outcome |= 2;

	std::complex<double> ta = compqc->Amplitude(outcome);
	std::complex<double> tb = compqc->Amplitude(outcome | 4);

	// check the teleported state
	BOOST_CHECK_PREDICATE(checkClose, (a)(ta)(0.000001));
	BOOST_CHECK_PREDICATE(checkClose, (b)(tb)(0.000001));

	// leave it in the 0 state
	resetCirc->Execute(compqc, tstate);
	measureCirc->Execute(compqc, tstate);

	BOOST_TEST(tstate.GetAllBits() == std::vector<bool>({ false, false, false }));
}




BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, RandomAerCircuitsTest, bdata::xrange(5, 20), nrGates)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	GenerateCircuit(nrGates);

	auto start = std::chrono::system_clock::now();
	randomCirc->Execute(aer, state);
	auto end = std::chrono::system_clock::now();
	double qiskitTime = std::chrono::duration<double>(end - start).count() * 1000.;
	
	std::shared_ptr<Simulators::ISimulator> compaer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQiskitAer);
	compaer->AllocateQubits(nrQubitsForRandomCirc);
	compaer->Initialize();

	BOOST_TEST(compaer);

	start = std::chrono::system_clock::now();
	randomCirc->Execute(compaer, state);
	end = std::chrono::system_clock::now();
	double compTime = std::chrono::duration<double>(end - start).count() * 1000.;

	BOOST_TEST_MESSAGE("Time for simple qiskit aer: " << qiskitTime << " ms, time for composite aer: " << compTime << " ms, composite is " << qiskitTime / compTime << " faster");

	//std::ofstream outfile;
	//outfile.open("qiskitaer.csv", std::ios_base::app);
	//outfile << nrQubitsForRandomCirc << "," << nrGates << "," << qiskitTime << "," << compTime << "," << qiskitTime / compTime << std::endl;

	aer->SaveStateToInternalDestructive();
	compaer->SaveStateToInternalDestructive();

	// now check the results, they should be the same!
	for (size_t state = 0; state < nrStates; ++state)
	{
		std::complex<double> aaer = aer->AmplitudeRaw(state);
		std::complex<double> aaercomp = compaer->AmplitudeRaw(state);

		// TODO: if this fails, do something to identify the offending gate!
		// rebuild the circuit one by one with the already exising gates from the failed circuit
		// check the results until it fails, then print the last gate in the first circuit that fails
		BOOST_CHECK_PREDICATE(checkClose, (aaer)(aaercomp)(0.000001));
	}

	aer->RestoreInternalDestructiveSavedState();
	compaer->RestoreInternalDestructiveSavedState();

	resetRandomCirc->Execute(aer, state);
	resetRandomCirc->Execute(compaer, state);
	// check if reset is ok
	BOOST_TEST(compaer->Probability(0), 1.);

	compaer->SaveStateToInternalDestructive();
	for (size_t state = 1; state < nrStates; ++state)
	{
		std::complex<double> aaer = compaer->AmplitudeRaw(state);
		BOOST_CHECK_PREDICATE(checkClose, (aaer)(0.)(0.000001));
	}

	randomCirc->Clear();
}


BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, RandomQcsimCircuitsTest, bdata::xrange(5, 20), nrGates)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	GenerateCircuit(nrGates);

	auto start = std::chrono::system_clock::now();
	randomCirc->Execute(qc, state);
	auto end = std::chrono::system_clock::now();
	double qcsimTime = std::chrono::duration<double>(end - start).count() * 1000.;

	std::shared_ptr<Simulators::ISimulator> compqc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim);
	compqc->AllocateQubits(nrQubitsForRandomCirc);
	compqc->Initialize();

	BOOST_TEST(compqc);

	start = std::chrono::system_clock::now();
	randomCirc->Execute(compqc, state);
	end = std::chrono::system_clock::now();
	double compTime = std::chrono::duration<double>(end - start).count() * 1000.;

	BOOST_TEST_MESSAGE("Time for simple qcsim: " << qcsimTime << " ms, time for composite qcsim: " << compTime << " ms, composite is " << qcsimTime / compTime << " faster");

	//std::ofstream outfile;
	//outfile.open("qcsim.csv", std::ios_base::app);
	//outfile << nrQubitsForRandomCirc << "," << nrGates << "," << qcsimTime << "," << compTime << "," << qcsimTime / compTime << std::endl;

	// now check the results, they should be the same!
	qc->SaveStateToInternalDestructive();
	compqc->SaveStateToInternalDestructive();
	for (size_t state = 0; state < nrStates; ++state)
	{
		std::complex<double> aqcsim = qc->AmplitudeRaw(state);
		std::complex<double> aaqcsimcomp = compqc->AmplitudeRaw(state);

		// TODO: if this fails, do something to identify the offending gate!
		// rebuild the circuit one by one with the already exising gates from the failed circuit
		// check the results until it fails, then print the last gate in the first circuit that fails
		BOOST_CHECK_PREDICATE(checkClose, (aqcsim)(aaqcsimcomp)(0.000001));
	}
	qc->RestoreInternalDestructiveSavedState();
	compqc->RestoreInternalDestructiveSavedState();

	resetRandomCirc->Execute(qc, state);
	resetRandomCirc->Execute(compqc, state);

	// check if reset is ok
	BOOST_TEST(compqc->Probability(0), 1.);
	for (size_t state = 1; state < nrStates; ++state)
	{
		std::complex<double> aqc = compqc->Amplitude(state);

		BOOST_CHECK_PREDICATE(checkClose, (aqc)(0.)(0.000001));
	}

	randomCirc->Clear();
}



BOOST_DATA_TEST_CASE_F(CompositeSimulatorsTestFixture, RandomAerQcsimCircuitsTest, bdata::xrange(5, 20), nrGates)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	GenerateCircuit(nrGates);

	auto start = std::chrono::system_clock::now();
	randomCirc->Execute(aer, state);
	auto end = std::chrono::system_clock::now();
	double aerTime = std::chrono::duration<double>(end - start).count() * 1000.;

	std::shared_ptr<Simulators::ISimulator> compqc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim);
	compqc->AllocateQubits(nrQubitsForRandomCirc);
	compqc->Initialize();

	BOOST_TEST(compqc);

	start = std::chrono::system_clock::now();
	randomCirc->Execute(compqc, state);
	end = std::chrono::system_clock::now();
	double compTime = std::chrono::duration<double>(end - start).count() * 1000.;

	BOOST_TEST_MESSAGE("Time for simple qiskit aer: " << aerTime << " ms, time for composite qcsim: " << compTime << " ms, composite is " << aerTime / compTime << " faster");

	aer->SaveStateToInternalDestructive();
	compqc->SaveStateToInternalDestructive();
	// now check the results, they should be the same!
	for (size_t state = 0; state < nrStates; ++state)
	{
		std::complex<double> aaer = aer->AmplitudeRaw(state);
		std::complex<double> aaqcsimcomp = compqc->AmplitudeRaw(state);

		// TODO: if this fails, do something to identify the offending gate!
		// rebuild the circuit one by one with the already exising gates from the failed circuit
		// check the results until it fails, then print the last gate in the first circuit that fails
		BOOST_CHECK_PREDICATE(checkClose, (aaer)(aaqcsimcomp)(0.000001));
	}

	aer->RestoreInternalDestructiveSavedState();
	compqc->RestoreInternalDestructiveSavedState();

	resetRandomCirc->Execute(aer, state);
	resetRandomCirc->Execute(compqc, state);

	// check if reset is ok
	BOOST_TEST(compqc->Probability(0), 1.);
	for (size_t state = 1; state < nrStates; ++state)
	{
		std::complex<double> aqc = compqc->Amplitude(state);

		BOOST_CHECK_PREDICATE(checkClose, (aqc)(0.)(0.000001));
	}

	randomCirc->Clear();
}


BOOST_AUTO_TEST_SUITE_END()

