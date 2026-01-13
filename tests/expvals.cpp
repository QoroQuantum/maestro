/**
 * @file expvals.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for expectation values.
 *
 * Randomly generated Pauli strings, computing expectation value for various simulators and comparison among them.
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


#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Simulators/Factory.h" // project being tested

#include "../Network/SimpleDisconnectedNetwork.h"

struct ExpvalTestFixture
{
	ExpvalTestFixture()
	{
#ifdef __linux__
		Simulators::SimulatorsFactory::InitGpuLibrary();
		
		gpuStatevector = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector);
		if (gpuStatevector)
		{
			gpuStatevector->AllocateQubits(nrQubits);
			gpuStatevector->Initialize();
		}

		gpuMPS = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
		if (gpuMPS)
		{
			gpuMPS->AllocateQubits(nrQubits);
			gpuMPS->Initialize();
		}
        gpuTN = Simulators::SimulatorsFactory::CreateSimulator(
                    Simulators::SimulatorType::kGpuSim,
                    Simulators::SimulationType::kTensorNetwork);
        if (gpuTN) 
		{
			gpuTN->AllocateQubits(nrQubits);
            gpuTN->Initialize();
        }
#endif

		state.AllocateBits(nrQubits);

		aerStatevector = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);
		aerStatevector->AllocateQubits(nrQubits);
		aerStatevector->Initialize();

		aerComposite = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQiskitAer, Simulators::SimulationType::kStatevector);
		aerComposite->AllocateQubits(nrQubits);
		aerComposite->Initialize();

		aerMPS = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
		aerMPS->AllocateQubits(nrQubits);
		aerMPS->Initialize();

		qcsimStatevector = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
		qcsimStatevector->AllocateQubits(nrQubits);
		qcsimStatevector->Initialize();

		qcsimStatevectorBig = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim, Simulators::SimulationType::kStatevector); // runs distributed circuits, it should be faster than the regular one
		qcsimStatevectorBig->AllocateQubits(nrQubits + 5);
		qcsimStatevectorBig->Initialize();

		qcsimComposite = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kCompositeQCSim, Simulators::SimulationType::kStatevector);
		qcsimComposite->AllocateQubits(nrQubits);
		qcsimComposite->Initialize();

		qcsimMPS = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
		qcsimMPS->AllocateQubits(nrQubits);
		qcsimMPS->Initialize();

		aerClifford = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStabilizer);
		aerClifford->AllocateQubits(nrQubits);
		aerClifford->Initialize();

		qcsimClifford = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer);
		qcsimClifford->AllocateQubits(nrQubits);
		qcsimClifford->Initialize();

		qcTensor = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kTensorNetwork);
		qcTensor->AllocateQubits(nrQubits);
		qcTensor->Initialize();

		randomCirc = Circuits::CircuitFactory<>::CreateCircuit();

		resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubits(nrQubits);
		std::iota(qubits.begin(), qubits.end(), 0);
		resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

		resetRandomCircBig = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubitsBig(nrQubits + 5);
		std::iota(qubitsBig.begin(), qubitsBig.end(), 0);
		resetRandomCircBig->AddOperation(std::make_shared<Circuits::Reset<>>(qubitsBig));

		// networks part, test only statevector, the other ones are tested against statevector so they should work with the networks, too
		std::vector<Types::qubit_t> networkBits{ 3, (Types::qubit_t)nrQubits, 2 }; // 3 hosts = 10 qubits + 3 entangled qubits = 13 qubits
		std::vector<size_t> networkCbits(networkBits.begin(), networkBits.end());

		networkSim1 = std::make_shared<Network::SimpleDisconnectedNetwork<>>(networkBits, networkCbits);
		networkSim1->CreateSimulator(/*Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector*/);
		//networkSim1->SetOptimizeSimulator(false);
	}

	~ExpvalTestFixture()
	{
	}

	std::string GeneratePauliString(int nrQubits)
	{
		std::string pauli;
		pauli.resize(nrQubits);
		std::random_device rd;
		std::mt19937 g(rd());
		std::uniform_int_distribution<int> dist(0, 3);

		for (int i = 0; i < nrQubits; ++i)
		{
			const int v = dist(g);
			switch (v)
			{
			case 0:
				pauli[i] = 'X';
				break;
			case 1:
				pauli[i] = 'Y';
				break;
			case 2:
				pauli[i] = 'Z';
				break;
			case 3:
				pauli[i] = 'I';
				break;
			default:
				pauli[i] = 'I';
				break;
			}
		}

		return pauli;
	}

	void GenerateCircuit(int nrGates, int nrQubits)
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
			Types::qubits_vector qubits(nrQubits);
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

			const Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

			auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, param1, param2, param3, param4);
			randomCirc->AddOperation(theGate);
		}
	}

	void GenerateCliffordCircuit(int nrGates, int nrQubits)
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
			Types::qubits_vector qubits(nrQubits);
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

			randomCirc->AddOperation(theGate);
		}
	}

	int nrQubits = 4;
	int nrCircuitsLimit = 30;
	int nrPauliLimit = 30;

	std::shared_ptr<Circuits::Circuit<>> randomCirc;
	std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
	std::shared_ptr<Circuits::Circuit<>> resetRandomCircBig;
	Circuits::OperationState state;

	std::shared_ptr<Simulators::ISimulator> aerStatevector;
	std::shared_ptr<Simulators::ISimulator> qcsimStatevector;
	std::shared_ptr<Simulators::ISimulator> qcsimStatevectorBig;

	std::shared_ptr<Simulators::ISimulator> aerComposite;
	std::shared_ptr<Simulators::ISimulator> qcsimComposite;

#ifdef __linux__
	std::shared_ptr<Simulators::ISimulator> gpuStatevector;
	std::shared_ptr<Simulators::ISimulator> gpuMPS;
    std::shared_ptr<Simulators::ISimulator> gpuTN;
#endif

	std::shared_ptr<Simulators::ISimulator> aerMPS;
	std::shared_ptr<Simulators::ISimulator> qcsimMPS;

	std::shared_ptr<Simulators::ISimulator> qcTensor;

	std::shared_ptr<Simulators::ISimulator> aerClifford;
	std::shared_ptr<Simulators::ISimulator> qcsimClifford;

	std::shared_ptr<Network::SimpleDisconnectedNetwork<>> networkSim1;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);


BOOST_AUTO_TEST_SUITE(expval_tests)

BOOST_FIXTURE_TEST_CASE(ExpvalInitializationTests, ExpvalTestFixture)
{
	BOOST_TEST(aerStatevector);
	BOOST_TEST(qcsimStatevector);
	BOOST_TEST(qcsimStatevectorBig);
	BOOST_TEST(aerComposite);
	BOOST_TEST(qcsimComposite);
	BOOST_TEST(aerMPS);
	BOOST_TEST(qcsimMPS);
	BOOST_TEST(aerClifford);
	BOOST_TEST(qcsimClifford);

	BOOST_TEST(qcTensor);

	BOOST_TEST(randomCirc);
	BOOST_TEST(resetRandomCirc);
	BOOST_TEST(resetRandomCircBig);

	BOOST_TEST(networkSim1);
}


BOOST_DATA_TEST_CASE_F(ExpvalTestFixture, NormalSimulatorsTest, bdata::xrange(1, 20), nrGates)
{
	const double precision = 0.000001;
	const double precisionGPU = 0.01;
	const double precisionMPS = 0.001;

	for (int i = 0; i < nrCircuitsLimit; ++i)
	{
		GenerateCircuit(nrGates, nrQubits);

		randomCirc->Execute(aerStatevector, state);
		randomCirc->Execute(qcsimStatevector, state);

		randomCirc->Execute(aerComposite, state);
		randomCirc->Execute(qcsimComposite, state);

		randomCirc->Execute(aerMPS, state);
		randomCirc->Execute(qcsimMPS, state);

#ifdef __linux__
		if (gpuStatevector) randomCirc->Execute(gpuStatevector, state);
		if (gpuMPS) randomCirc->Execute(gpuMPS, state);
        if (gpuTN) randomCirc->Execute(gpuTN, state);
#endif
	
		randomCirc->Execute(qcTensor, state);

		for (int j = 0; j < nrPauliLimit; ++j)
		{
			const std::string pauli = GeneratePauliString(nrQubits);

			const double aerStatevectorVal = aerStatevector->ExpectationValue(pauli);			
			const double qcsimStatevectorVal = qcsimStatevector->ExpectationValue(pauli);
			
			BOOST_CHECK_PREDICATE(checkClose, (aerStatevectorVal)(qcsimStatevectorVal)(precision));

			const double aerCompVal = aerComposite->ExpectationValue(pauli);
			const double qcsimCompVal = qcsimComposite->ExpectationValue(pauli);
			
			BOOST_CHECK_PREDICATE(checkClose, (aerCompVal)(qcsimStatevectorVal)(precision));
			BOOST_CHECK_PREDICATE(checkClose, (qcsimCompVal)(qcsimStatevectorVal)(precision));
		

			const double aerMPSVal = aerMPS->ExpectationValue(pauli);
			const double qcsimMPSVal = qcsimMPS->ExpectationValue(pauli);

			BOOST_CHECK_PREDICATE(checkClose, (aerMPSVal)(qcsimStatevectorVal)(precisionMPS));
			BOOST_CHECK_PREDICATE(checkClose, (qcsimMPSVal)(qcsimStatevectorVal)(precisionMPS));

			const double qcTensorVal = qcTensor->ExpectationValue(pauli);
			BOOST_CHECK_PREDICATE(checkClose, (qcTensorVal)(qcsimStatevectorVal)(precision));
			
#ifdef __linux__
			if (gpuStatevector)
			{
				const double gpuStatevectorVal = gpuStatevector->ExpectationValue(pauli);
				BOOST_REQUIRE_PREDICATE(checkClose, (gpuStatevectorVal)(qcsimStatevectorVal)(precisionGPU));
			}

			if (gpuMPS)
			{
				const double gpuMPSVal = gpuMPS->ExpectationValue(pauli);
				BOOST_REQUIRE_PREDICATE(checkClose, (gpuMPSVal)(qcsimStatevectorVal)(precisionMPS));
			}
            if (gpuTN) 
			{
                const double gpuTNVal = gpuTN->ExpectationValue(pauli);
                BOOST_REQUIRE_PREDICATE(checkClose, (gpuTNVal)(qcsimStatevectorVal)(precisionMPS));
            }
#endif
		}

		resetRandomCirc->Execute(aerStatevector, state);
		resetRandomCirc->Execute(qcsimStatevector, state);

		resetRandomCirc->Execute(aerComposite, state);
		resetRandomCirc->Execute(qcsimComposite, state);

		resetRandomCirc->Execute(aerMPS, state);
		resetRandomCirc->Execute(qcsimMPS, state);

#ifdef __linux__
		if (gpuStatevector) resetRandomCirc->Execute(gpuStatevector, state);
		if (gpuMPS) resetRandomCirc->Execute(gpuMPS, state);
        if (gpuTN) resetRandomCirc->Execute(gpuTN, state);
#endif

		qcTensor->Clear();
		qcTensor->AllocateQubits(nrQubits);
		qcTensor->Initialize();

		randomCirc->Clear();
	}
}

BOOST_DATA_TEST_CASE_F(ExpvalTestFixture, CliffordSimulatorsTest, bdata::xrange(1, 20), nrGates)
{
	const double precision = 0.00000001;

	for (int i = 0; i < 30; ++i)
	{
		GenerateCliffordCircuit(nrGates, nrQubits);

		randomCirc->Execute(qcsimStatevector, state);
		randomCirc->Execute(aerClifford, state);
		randomCirc->Execute(qcsimClifford, state);

		for (int j = 0; j < 30; ++j)
		{
			const std::string pauli = GeneratePauliString(nrQubits);

			const double qcsimStatevectorVal = qcsimStatevector->ExpectationValue(pauli);
			const double qcsimCliffordVal = qcsimClifford->ExpectationValue(pauli);

			//aerClifford->SaveState();
			const double aerCliffordVal = aerClifford->ExpectationValue(pauli);
			//aerClifford->RestoreState();

			BOOST_CHECK_PREDICATE(checkClose, (qcsimCliffordVal)(qcsimStatevectorVal)(precision));
			BOOST_CHECK_PREDICATE(checkClose, (aerCliffordVal)(qcsimStatevectorVal)(precision));
		}

		resetRandomCirc->Execute(qcsimStatevector, state);
		resetRandomCirc->Execute(aerClifford, state);
		resetRandomCirc->Execute(qcsimClifford, state);

		randomCirc->Clear();
	}
}

BOOST_DATA_TEST_CASE_F(ExpvalTestFixture, NetworkExpectationTest, bdata::xrange(1, 20), nrGates)
{
	const double precision = 0.000001;

	for (int i = 0; i < 30; ++i)
	{
		GenerateCircuit(nrGates, nrQubits);

		randomCirc->Execute(qcsimStatevector, state);

		for (int j = 0; j < 30; ++j)
		{
			std::vector<std::string> paulis;
			for (int k = 0; k < 10; ++k)
				paulis.push_back(GeneratePauliString(nrQubits));
			
			const auto vals = networkSim1->ExecuteOnHostExpectations(randomCirc, 1, paulis);

			// now check them against the statevector simulator
			for (int k = 0; k < 10; ++k)
			{
				const double qcsimStatevectorVal = qcsimStatevector->ExpectationValue(paulis[k]);

				BOOST_CHECK_PREDICATE(checkClose, (vals[k])(qcsimStatevectorVal)(precision));
			}

			networkSim1->ExecuteOnHost(resetRandomCirc, 1);	
		}

		resetRandomCirc->Execute(qcsimStatevector, state);
		randomCirc->Clear();
	}
}

BOOST_AUTO_TEST_SUITE_END()
