/**
 * @file qasm.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the qasm parser and generator.
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

// project being tested
#include "../Simulators/Factory.h" 

#include "../Circuit/Circuit.h"
#include "../Circuit/Conditional.h"
#include "../Circuit/Measurements.h"
#include "../Circuit/QuantumGates.h"
#include "../Circuit/RandomOp.h"
#include "../Circuit/Reset.h"
#include "../Circuit/Factory.h"

#include "../qasm/QasmCirc.h"
#include "../qasm/CircQasm.h"

struct QasmTestFixture
{
	QasmTestFixture()
	{
		state.AllocateBits(nrQubitsForRandomCirc);
		resetCirc = std::make_shared<Circuits::Circuit<>>();
		Types::qubits_vector qubits(nrQubitsForRandomCirc);
		std::iota(qubits.begin(), qubits.end(), 0);
		resetCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

		randomCirc = std::make_shared<Circuits::Circuit<>>();

		qc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
		qc->AllocateQubits(nrQubitsForRandomCirc);
		qc->Initialize();

		qc2 = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
		qc2->AllocateQubits(nrQubitsForRandomCirc);
		qc2->Initialize();
	}

	// fills randomly the circuit with gates
	void GenerateCircuit(int nrGates, int nrQubits, double probReset = 0., double probMeasurement = 0.)
	{
		std::random_device rd;
		std::mt19937 g(rd());

		auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
		auto dblGenIter = dblGen.begin();

		auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
		auto gateGenIter = gateGen.begin();

		auto typeGen = bdata::random();
		auto typeGenIter = typeGen.begin();

		const double cumProb = probReset + probMeasurement;

		// TODO: Maybe insert from time to time a random number generating 'gate' and a conditioned random one, those should not affect results?
		for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
		{
			const double typeOpProb = *typeGenIter;
			++typeGenIter;

			Types::qubits_vector qubits(nrQubits);
			std::iota(qubits.begin(), qubits.end(), 0);
			std::shuffle(qubits.begin(), qubits.end(), g);

			if (typeOpProb < probReset)
			{
				// reset operation
				auto resetOp = Circuits::CircuitFactory<>::CreateReset({qubits[0]});
				randomCirc->AddOperation(resetOp);
				continue;
			}
			else if (typeOpProb < cumProb)
			{
				// measurement operation
				auto measOp = Circuits::CircuitFactory<>::CreateMeasurement({ {qubits[0], qubits[0]} });
				randomCirc->AddOperation(measOp);
				continue;
			}

			// create a random gate and add it to the circuit

			// first, pick randomly three qubits (depending on the randomly chosen gate type, not all of them will be used)
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

			const Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

			auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, param1, param2, param3);
			randomCirc->AddOperation(theGate);
		}
	}

	void PrintCircuit(const std::shared_ptr<Circuits::Circuit<>>& circuit)
	{
		std::cout << "Circuit with " << circuit->size() << " operations:" << std::endl;
		for (const auto& op : circuit->GetOperations())
		{
			std::cout << "Operation type: ";
			switch (op->GetType())
			{
			case Circuits::OperationType::kGate:
				std::cout << "Gate ";
				{
					auto gatePtr = static_cast<Circuits::IQuantumGate<>*>(op.get());
					PrintGateName(gatePtr->GetGateType());

					auto params = gatePtr->GetParams();
					if (!params.empty())
					{
						std::cout << " (";
						for (size_t i = 0; i < params.size(); ++i)
						{
							std::cout << params[i];
							if (i < params.size() - 1) std::cout << ", ";
						}
						std::cout << ")";
					}

					auto qubits = op->AffectedQubits();
					std::cout << " ";
					for (size_t i = 0; i < qubits.size(); ++i)
					{
						std::cout << qubits[i];
						if (i < qubits.size() - 1) std::cout << ", ";
					}
				}
				break;
			case Circuits::OperationType::kMeasurement:
				std::cout << "Measurement ";
				{
					auto qubits = op->AffectedQubits();
					std::cout << " (";
					for (size_t i = 0; i < qubits.size(); ++i)
					{
						std::cout << qubits[i];
						if (i < qubits.size() - 1) std::cout << ", ";
					}
					std::cout << ")";

					std::cout << " -> ";

					auto bits = op->AffectedBits();
					std::cout << " (";
					for (size_t i = 0; i < bits.size(); ++i)
					{
						std::cout << bits[i];
						if (i < bits.size() - 1) std::cout << ", ";
					}
					std::cout << ")";
				}
				break;
			case Circuits::OperationType::kReset:
				std::cout << "Reset";
				{
					auto qubits = op->AffectedQubits();
					std::cout << " (";
					for (size_t i = 0; i < qubits.size(); ++i)
					{
						std::cout << qubits[i];
						if (i < qubits.size() - 1) std::cout << ", ";
					}
					std::cout << ")";
				}
				break;
			case Circuits::OperationType::kConditionalGate:
				std::cout << "Conditional";
				break;
			default:
				std::cout << "Other";
				break;
			}
			std::cout << std::endl;
		}
	}

	void PrintGateName(Circuits::QuantumGateType gateType)
	{
		std::string name;

		switch (gateType)
		{
		case Circuits::QuantumGateType::kPhaseGateType:
			name = "p";
			break;
		case Circuits::QuantumGateType::kXGateType:
			name = "x";
			break;
		case Circuits::QuantumGateType::kYGateType:
			name = "y";
			break;
		case Circuits::QuantumGateType::kZGateType:
			name = "z";
			break;
		case Circuits::QuantumGateType::kHadamardGateType:
			name = "h";
			break;
		case Circuits::QuantumGateType::kSGateType:
			name = "s";
			break;
		case Circuits::QuantumGateType::kSdgGateType:
			name = "sdg";
			break;
		case Circuits::QuantumGateType::kTGateType:
			name = "t";
			break;
		case Circuits::QuantumGateType::kTdgGateType:
			name = "tdg";
			break;

		case Circuits::QuantumGateType::kSxGateType:
			name = "sx";
			break;
		case Circuits::QuantumGateType::kSxDagGateType:
			name = "sxdg";
			break;
		case Circuits::QuantumGateType::kKGateType:
			name = "k";
			break;
		case Circuits::QuantumGateType::kRxGateType:
			name = "rx";
			break;
		case Circuits::QuantumGateType::kRyGateType:
			name = "ry";
			break;
		case Circuits::QuantumGateType::kRzGateType:
			name = "rz";
			break;

		case Circuits::QuantumGateType::kUGateType:
			name = "u";
			break;

		case Circuits::QuantumGateType::kCXGateType:
			name = "cx";
			// standard gate
			break;
		case Circuits::QuantumGateType::kCYGateType:
			name = "cy";
			break;
		case Circuits::QuantumGateType::kCZGateType:
			name = "cz";
			break;
		case Circuits::QuantumGateType::kCPGateType:
			name = "cp";
			break;

		case Circuits::QuantumGateType::kCRxGateType:
			name = "crx";
			break;
		case Circuits::QuantumGateType::kCRyGateType:
			name = "cry";
			break;

		case Circuits::QuantumGateType::kCRzGateType:
			name = "crz";
			break;
		case Circuits::QuantumGateType::kCHGateType:
			name = "ch";
			break;

		case Circuits::QuantumGateType::kCSxGateType:
			name = "csx";
			break;
		case Circuits::QuantumGateType::kCSxDagGateType:
			name = "csxdg";
			break;

		case Circuits::QuantumGateType::kCUGateType:
			name = "cu";
			break;
		case Circuits::QuantumGateType::kSwapGateType:
			name = "swap";
			break;
		case Circuits::QuantumGateType::kCSwapGateType:
			name = "cswap";
			break;
		case Circuits::QuantumGateType::kCCXGateType:
			name = "ccx";
			break;
		default:
			name = "unknown";
			break;
		}

		std::cout << " " << name << " ";
	}

	Circuits::OperationState state;
	std::shared_ptr<Simulators::ISimulator> qc;
	std::shared_ptr<Simulators::ISimulator> qc2;

	// for testing randomly generated circuits
	const unsigned int nrQubitsForRandomCirc = 5;

	std::shared_ptr<Circuits::Circuit<>> randomCirc;
	std::shared_ptr<Circuits::Circuit<>> resetCirc;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(qasm_tests)

BOOST_FIXTURE_TEST_CASE(InitializationTests, QasmTestFixture)
{
	BOOST_TEST(qc);
	BOOST_TEST(qc2);
	BOOST_TEST(randomCirc);
	BOOST_TEST(resetCirc);
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsTest, bdata::xrange(1, 30), nrGates)
{
	size_t nrStates = 1ULL << nrQubitsForRandomCirc;

	for (int i = 0; i < 5; ++i)
	{
		GenerateCircuit(nrGates, nrQubitsForRandomCirc);

		randomCirc->Execute(qc, state);

		// export to qasm
		std::string qasmStr = qasm::CircToQasm<>::Generate(randomCirc);

		// import from qasm
		qasm::QasmToCirc<> parser;
		auto circuit = parser.ParseAndTranslate(qasmStr);
		BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

		if (!parser.Failed())
		{
			// execute imported circuit
			circuit->Execute(qc2, state);

			// compare results
			for (size_t i = 0; i < nrStates; ++i)
			{
				double prob1 = qc->Probability(i);
				double prob2 = qc2->Probability(i);

				BOOST_TEST(checkClose(std::complex<double>(prob1, 0.), std::complex<double>(prob2, 0.), 0.0001), "Probability mismatch for state |" << i << ">: " << prob1 << " vs " << prob2 << ", qasm: " << qasmStr);

				if (!checkClose(std::complex<double>(prob1, 0.), std::complex<double>(prob2, 0.), 0.0001))
				{
					std::cout << "Original circuit:" << std::endl;
					PrintCircuit(randomCirc);

					std::cout << "\nConverted circuit:" << std::endl;
					PrintCircuit(circuit);
				}
			}
		}

		randomCirc->Clear();

		resetCirc->Execute(qc, state);
		resetCirc->Execute(qc2, state);
		state.Reset();
	}
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsWithMeasAndResetTest, bdata::xrange(20, 40), nrGates)
{
	static const int nrShots = 5000;

	for (int i = 0; i < 5; ++i)
	{
		GenerateCircuit(nrGates, nrQubitsForRandomCirc, 0.025, 0.15);

		// export to qasm
		std::string qasmStr = qasm::CircToQasm<>::Generate(randomCirc);

		// import from qasm
		qasm::QasmToCirc<> parser;
		auto circuit = parser.ParseAndTranslate(qasmStr);
		BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

		/*
		std::cout << "Original circuit:" << std::endl;
		PrintCircuit(randomCirc);
		std::cout << "\nConverted circuit:" << std::endl;
		PrintCircuit(circuit);

		std::cout << "QASM:" << std::endl;
		std::cout << qasmStr << std::endl;

		exit(0);
		*/

		if (!parser.Failed())
		{
			std::unordered_map<std::vector<bool>, size_t> results1;
			std::unordered_map<std::vector<bool>, size_t> results2;

			for (int trial = 0; trial < nrShots; ++trial)
			{
				randomCirc->Execute(qc, state);

				results1[state.GetAllBits()]++;

				state.Reset();

				// execute imported circuit
				circuit->Execute(qc2, state);

				results2[state.GetAllBits()]++;
				
				resetCirc->Execute(qc, state);
				resetCirc->Execute(qc2, state);
				state.Reset();
			}

			for (const auto& [key, cnt] : results1)
			{
				double val = static_cast<double>(cnt) / static_cast<double>(nrShots);

				if (val < 0.03) continue;

				double val2 = 0;
				if (results2.find(key) != results2.end())
					val2 = static_cast<double>(results2[key]) / static_cast<double>(nrShots);

				BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
			}

			for (const auto& [key, cnt] : results2)
			{
				double val = static_cast<double>(cnt) / static_cast<double>(nrShots);
				if (val < 0.03) continue;

				double val2 = 0;
				if (results1.find(key) != results1.end())
					val2 = static_cast<double>(results1[key]) / static_cast<double>(nrShots);
				BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
			}
		}

		randomCirc->Clear();
	}
}

BOOST_AUTO_TEST_SUITE_END()
