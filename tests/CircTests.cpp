/**
 * @file CircTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for circuits execution in both aer and qcsim simulators.
 *
 * The tests use simple circuits (x gate on a qubit or teleportation) and
 * randomly generated circuits. They either check against expected results (the
 * result of the teleportation or x gate) or against the same circuit executed
 * in the other simulator.
 */

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

// project being tested
#include "../Simulators/Core/Factory.h"

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Circuit/Operations/Conditional.h"
#include "../Circuit/Operations/Measurements.h"
#include "../Circuit/Operations/QuantumGates.h"
#include "../Circuit/Operations/RandomOp.h"
#include "../Circuit/Operations/Reset.h"

struct SimulatorsTestFixture
{
    SimulatorsTestFixture()
    {
#ifdef __linux__
        Simulators::SimulatorsFactory::InitGpuLibrary();
#endif
        Simulators::SimulatorsFactory::InitQuestLibrary();

        state.AllocateBits(3);

        aer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);
        aer->AllocateQubits(3);
        aer->Initialize();

        qc = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        qc->AllocateQubits(3);
        qc->Initialize();

        setCirc = std::make_shared<Circuits::Circuit<>>();
        resetCirc = std::make_shared<Circuits::Circuit<>>();
        measureCirc = std::make_shared<Circuits::Circuit<>>();
        teleportationCirc = std::make_shared<Circuits::Circuit<>>();
        genTeleportationCirc = std::make_shared<Circuits::Circuit<>>();

        randomCirc = Circuits::CircuitFactory<>::CreateCircuit();
        aerRandom = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);
        qcRandom = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        aerRandom->AllocateQubits(nrQubitsForRandomCirc);
        qcRandom->AllocateQubits(nrQubitsForRandomCirc);
        aerRandom->Initialize();
        qcRandom->Initialize();

#ifdef __linux__
        gpusim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector);
        if (gpusim)
        {
            gpusim->AllocateQubits(3);
            gpusim->Initialize();
        }

        gpuRandom = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector);
        if (gpuRandom)
        {
            gpuRandom->AllocateQubits(nrQubitsForRandomCirc);
            gpuRandom->Initialize();
        }
#endif

        questsim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQuestSim, Simulators::SimulationType::kStatevector);
        if (questsim)
        {
            questsim->AllocateQubits(3);
            questsim->Initialize();
        }

        questRandom = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQuestSim, Simulators::SimulationType::kStatevector);
        if (questRandom)
        {
            questRandom->AllocateQubits(nrQubitsForRandomCirc);
            questRandom->Initialize();
        }

        resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
        Types::qubits_vector qubits(nrQubitsForRandomCirc);
        std::iota(qubits.begin(), qubits.end(), 0);
        resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

        setCirc->AddOperation(std::make_shared<Circuits::XGate<>>(0));

        resetCirc->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0, 1, 2}));

        measureCirc->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}, {2, 2}}));

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

        teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));

        teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(
            Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, 2),
            Circuits::CircuitFactory<>::CreateEqualCondition(std::vector<size_t>{1}, std::vector<bool>{true})));

        teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(
            Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kZGateType, 2),
            Circuits::CircuitFactory<>::CreateEqualCondition(std::vector<size_t>{0}, std::vector<bool>{true})));

        teleportationCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(std::vector<std::pair<Types::qubit_t, size_t>>{{2, 2}}));

        // ************************************************************************************************

        // without setting the qubit at the beginning to 1 and without the last
        // measurement instead set a generic state for the qubit at the beginning
        // and check it to be the same in the teleported qubit at the end
        // genTeleportationCirc->SetOperations(std::vector<std::shared_ptr<Circuits::IOperation<>>>(teleportationCirc->GetOperations().begin()
        // + 1, teleportationCirc->GetOperations().end() - 1));
        // some other way, for testing the factory generated one:
        genTeleportationCirc->AddOperations(Circuits::CircuitFactory<>::CreateTeleportationCircuit(1, 2, 0, 0, 1));
    }

    ~SimulatorsTestFixture()
    {
    }

    // fills randomly the circuit with gates
    void GenerateCircuit(int nrGates, int nrQubits)
    {
        std::random_device rd;
        std::mt19937 g(rd());

        auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
        auto dblGenIter = dblGen.begin();

        auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
        auto gateGenIter = gateGen.begin();

        // TODO: Maybe insert from time to time a random number generating 'gate'
        // and a conditioned random one, those should not affect results?
        for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
        {
            // create a random gate and add it to the circuit

            // first, pick randomly three qubits (depending on the randomly chosen
            // gate type, not all of them will be used)
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

    std::shared_ptr<Simulators::ISimulator> aer;
    std::shared_ptr<Simulators::ISimulator> qc;

    std::shared_ptr<Simulators::ISimulator> aerRandom;
    std::shared_ptr<Simulators::ISimulator> qcRandom;

#ifdef __linux__
    std::shared_ptr<Simulators::ISimulator> gpusim;
    std::shared_ptr<Simulators::ISimulator> gpuRandom;
#endif

    std::shared_ptr<Simulators::ISimulator> questsim;
    std::shared_ptr<Simulators::ISimulator> questRandom;

    std::shared_ptr<Circuits::Circuit<>> setCirc;
    std::shared_ptr<Circuits::Circuit<>> resetCirc;
    std::shared_ptr<Circuits::Circuit<>> measureCirc;
    std::shared_ptr<Circuits::Circuit<>> teleportationCirc;
    std::shared_ptr<Circuits::Circuit<>> genTeleportationCirc;

    Circuits::OperationState state;

    // for testing randomly generated circuits
    const unsigned int nrQubitsForRandomCirc = 5;

    std::shared_ptr<Circuits::Circuit<>> randomCirc;
    std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(circuit_tests)

BOOST_FIXTURE_TEST_CASE(CircuitsInitializationTests, SimulatorsTestFixture)
{
    BOOST_TEST(aer);
    BOOST_TEST(qc);
    BOOST_TEST(setCirc);
    BOOST_TEST(resetCirc);
    BOOST_TEST(measureCirc);
    BOOST_TEST(teleportationCirc);

    auto depth = teleportationCirc->GetMaxDepth();
    BOOST_TEST(depth.first == 8ULL);

    BOOST_TEST(genTeleportationCirc);
    depth = genTeleportationCirc->GetMaxDepth();
    BOOST_TEST(depth.first == 6ULL);

    BOOST_TEST(randomCirc);
    BOOST_TEST(aerRandom);
    BOOST_TEST(qcRandom);
    BOOST_TEST(resetRandomCirc);

    // TODO: pass maps and check outcome?
    auto teleportationCircRemapped = teleportationCirc->Remap({});
    BOOST_TEST(teleportationCircRemapped);
}

BOOST_FIXTURE_TEST_CASE(SimpleCircuitAerTests, SimulatorsTestFixture)
{
    setCirc->Execute(aer, state);
    measureCirc->Execute(aer, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({true, false, false}));

    resetCirc->Execute(aer, state);
    measureCirc->Execute(aer, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
}

BOOST_FIXTURE_TEST_CASE(SimpleCircuitQcSimTests, SimulatorsTestFixture)
{
    setCirc->Execute(qc, state);
    measureCirc->Execute(qc, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({true, false, false}));

    resetCirc->Execute(qc, state);
    measureCirc->Execute(qc, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
}

#ifdef __linux__

BOOST_FIXTURE_TEST_CASE(SimpleCircuitGpuTests, SimulatorsTestFixture)
{
    if (gpusim)
    {
        setCirc->Execute(gpusim, state);
        measureCirc->Execute(gpusim, state);

        BOOST_TEST(state.GetAllBits() == std::vector<bool>({true, false, false}));

        resetCirc->Execute(gpusim, state);
        measureCirc->Execute(gpusim, state);
        BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
    }
}

#endif

BOOST_FIXTURE_TEST_CASE(SimpleCircuitQuestTests, SimulatorsTestFixture)
{
    if (questsim)
    {
        setCirc->Execute(questsim, state);
        measureCirc->Execute(questsim, state);

        BOOST_TEST(state.GetAllBits() == std::vector<bool>({true, false, false}));

        resetCirc->Execute(questsim, state);
        measureCirc->Execute(questsim, state);
        BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
    }
}

// BOOST_FIXTURE_TEST_CASE(TeleportationAerTest, SimulatorsTestFixture)
BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, TeleportationAerTest, bdata::xrange(5), ind)
{
    if (ind % 2)
    {
        // start with a different state to teleport (the teleportation circuit
        // applies a not on the first qubit in the beginning)
        aer->ApplyX(0);
    }

    teleportationCirc->Execute(aer, state);

    BOOST_TEST(state.GetAllBits()[2] == ((ind % 2) ? false : true));

    // leave it in the 0 state
    resetCirc->Execute(aer, state);
    measureCirc->Execute(aer, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
}

// BOOST_FIXTURE_TEST_CASE(TeleportationQcSimTest, SimulatorsTestFixture)
BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, TeleportationQcSimTest, bdata::xrange(5), ind)
{
    if (ind % 2)
    {
        // start with a different state to teleport (the teleportation circuit
        // applies a not on the first qubit in the beginning)
        qc->ApplyX(0);
    }

    teleportationCirc->Execute(qc, state);

    BOOST_TEST(state.GetAllBits()[2] == ((ind % 2) ? false : true));

    // leave it in the 0 state
    resetCirc->Execute(qc, state);
    measureCirc->Execute(qc, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
}

#ifdef __linux__

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, TeleportationGpuTest, bdata::xrange(5), ind)
{
    if (gpusim)
    {
        if (ind % 2)
        {
            // start with a different state to teleport (the teleportation circuit
            // applies a not on the first qubit in the beginning)
            gpusim->ApplyX(0);
        }
        teleportationCirc->Execute(gpusim, state);
        BOOST_TEST(state.GetAllBits()[2] == ((ind % 2) ? false : true));

        // leave it in the 0 state
        resetCirc->Execute(gpusim, state);
        measureCirc->Execute(gpusim, state);
        BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
    }
}

#endif

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, TeleportationQuestTest, bdata::xrange(5), ind)
{
    if (questsim)
    {
        if (ind % 2)
        {
            // start with a different state to teleport (the teleportation circuit
            // applies a not on the first qubit in the beginning)
            questsim->ApplyX(0);
        }
        teleportationCirc->Execute(questsim, state);
        BOOST_TEST(state.GetAllBits()[2] == ((ind % 2) ? false : true));

        // leave it in the 0 state
        resetCirc->Execute(questsim, state);
        measureCirc->Execute(questsim, state);
        BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
    }
}

// methods:
// bdata::make({ 0., 0.2, 0.4, 0.6, 0.8, 1. })
// equivalent: bdata::xrange(0.0, 1.0, 0.2)
// below is using random value
// ^ is for 'zip', * is for 'cartesian product'

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, GenTeleportAerTest, (bdata::random() ^ bdata::xrange(5)) * bdata::xrange(5), theta, index1, index2)
{
    aer->ApplyRx(0, theta);

    // now get the values to be compared with the teleported one
    std::complex<double> a = aer->Amplitude(0);
    std::complex<double> b = aer->Amplitude(1);

    genTeleportationCirc->Execute(aer, state);

    bool b1 = state.GetAllBits()[0];
    bool b2 = state.GetAllBits()[1];

    size_t outcome = 0;
    if (b1)
        outcome |= 1;
    if (b2)
        outcome |= 2;

    std::complex<double> ta = aer->Amplitude(outcome);
    std::complex<double> tb = aer->Amplitude(outcome | 4);

    // check the teleported state
    BOOST_CHECK_PREDICATE(checkClose, (a)(ta)(0.000001));
    BOOST_CHECK_PREDICATE(checkClose, (b)(tb)(0.000001));

    // leave it in the 0 state
    resetCirc->Execute(aer, state);
    measureCirc->Execute(aer, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
}

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, GenTeleportQcSimTest, (bdata::random() ^ bdata::xrange(5)) * bdata::xrange(5), theta, index1, index2)
{
    qc->ApplyRx(0, theta);

    // now get the values to be compared with the teleported one
    std::complex<double> a = qc->Amplitude(0);
    std::complex<double> b = qc->Amplitude(1);

    genTeleportationCirc->Execute(qc, state);

    bool b1 = state.GetAllBits()[0];
    bool b2 = state.GetAllBits()[1];

    size_t outcome = 0;
    if (b1)
        outcome |= 1;
    if (b2)
        outcome |= 2;

    std::complex<double> ta = qc->Amplitude(outcome);
    std::complex<double> tb = qc->Amplitude(outcome | 4);

    // check the teleported state
    BOOST_CHECK_PREDICATE(checkClose, (a)(ta)(0.000001));
    BOOST_CHECK_PREDICATE(checkClose, (b)(tb)(0.000001));

    // leave it in the 0 state
    resetCirc->Execute(qc, state);
    measureCirc->Execute(qc, state);

    BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
}

#ifdef __linux__

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, GenTeleportGpuTest, (bdata::random() ^ bdata::xrange(5)) * bdata::xrange(5), theta, index1, index2)
{
    if (gpusim)
    {
        gpusim->ApplyRx(0, theta);
        // now get the values to be compared with the teleported one
        std::complex<double> a = gpusim->Amplitude(0);
        std::complex<double> b = gpusim->Amplitude(1);
        genTeleportationCirc->Execute(gpusim, state);
        bool b1 = state.GetAllBits()[0];
        bool b2 = state.GetAllBits()[1];
        size_t outcome = 0;
        if (b1)
            outcome |= 1;
        if (b2)
            outcome |= 2;
        std::complex<double> ta = gpusim->Amplitude(outcome);
        std::complex<double> tb = gpusim->Amplitude(outcome | 4);
        // check the teleported state
        BOOST_CHECK_PREDICATE(checkClose, (a)(ta)(0.000001));
        BOOST_CHECK_PREDICATE(checkClose, (b)(tb)(0.000001));
        // leave it in the 0 state
        resetCirc->Execute(gpusim, state);
        measureCirc->Execute(gpusim, state);
        BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
    }
}

#endif

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, GenTeleportQuestTest, (bdata::random() ^ bdata::xrange(5)) * bdata::xrange(5), theta, index1, index2)
{
    if (questsim)
    {
        questsim->ApplyRx(0, theta);
        // now get the values to be compared with the teleported one
        std::complex<double> a = questsim->Amplitude(0);
        std::complex<double> b = questsim->Amplitude(1);
        genTeleportationCirc->Execute(questsim, state);
        bool b1 = state.GetAllBits()[0];
        bool b2 = state.GetAllBits()[1];
        size_t outcome = 0;
        if (b1)
            outcome |= 1;
        if (b2)
            outcome |= 2;
        std::complex<double> ta = questsim->Amplitude(outcome);
        std::complex<double> tb = questsim->Amplitude(outcome | 4);
        // check the teleported state
        BOOST_CHECK_PREDICATE(checkClose, (a)(ta)(0.000001));
        BOOST_CHECK_PREDICATE(checkClose, (b)(tb)(0.000001));
        // leave it in the 0 state
        resetCirc->Execute(questsim, state);
        measureCirc->Execute(questsim, state);
        BOOST_TEST(state.GetAllBits() == std::vector<bool>({false, false, false}));
    }
}

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, RandomCircuitsTest, bdata::xrange(1, 20), nrGates)
{
    size_t nrStates = 1ULL << nrQubitsForRandomCirc;

    GenerateCircuit(nrGates, nrQubitsForRandomCirc);

    auto start = std::chrono::system_clock::now();
    randomCirc->Execute(aerRandom, state);
    auto end = std::chrono::system_clock::now();
    double qiskitTime = std::chrono::duration<double>(end - start).count() * 1000.;

    start = std::chrono::system_clock::now();
    randomCirc->Execute(qcRandom, state);
    end = std::chrono::system_clock::now();
    double qcsimTime = std::chrono::duration<double>(end - start).count() * 1000.;

    BOOST_TEST_MESSAGE("Time for qiskit aer: " << qiskitTime << " ms, time for qcsim: " << qcsimTime << " ms, qcsim is " << qiskitTime / qcsimTime
                                               << " faster");

#ifdef __linux__
    if (gpuRandom)
    {
        start = std::chrono::system_clock::now();
        randomCirc->Execute(gpuRandom, state);
        end = std::chrono::system_clock::now();
        double gpuTime = std::chrono::duration<double>(end - start).count() * 1000.;

        BOOST_TEST_MESSAGE("Time for qiskit aer: " << qiskitTime << " ms, time for gpu sim: " << gpuTime << " ms, gpu sim is " << qiskitTime / gpuTime
                                                   << " faster");
    }
#endif

    if (questRandom)
    {
        start = std::chrono::system_clock::now();
        randomCirc->Execute(questRandom, state);
        end = std::chrono::system_clock::now();
        double questTime = std::chrono::duration<double>(end - start).count() * 1000.;

        BOOST_TEST_MESSAGE("Time for qiskit aer: " << qiskitTime << " ms, time for quest sim: " << questTime << " ms, quest sim is " << qiskitTime / questTime
                                                   << " faster");
    }

    // now check the results, they should be the same!
    for (size_t state = 0; state < nrStates; ++state)
    {
        std::complex<double> aaer = aerRandom->Amplitude(state);
        std::complex<double> aqc = qcRandom->Amplitude(state);

        // TODO: if this fails, do something to identify the offending gate!
        // rebuild the circuit one by one with the already exising gates from the
        // failed circuit check the results until it fails, then print the last gate
        // in the first circuit that fails
        BOOST_CHECK_PREDICATE(checkClose, (aaer)(aqc)(0.000001));

#ifdef __linux__
        if (gpuRandom)
        {
            std::complex<double> agpu = gpuRandom->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (aaer)(agpu)(0.000001));
        }
#endif

        if (questRandom)
        {
            std::complex<double> aquest = questRandom->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (aaer)(aquest)(0.000001));
        }
    }

    resetRandomCirc->Execute(aerRandom, state);

    // check if reset is ok
    BOOST_TEST(aerRandom->Probability(0), 1.);
    for (size_t state = 1; state < nrStates; ++state)
    {
        std::complex<double> aaer = aerRandom->Amplitude(state);

        BOOST_CHECK_PREDICATE(checkClose, (aaer)(0.)(0.000001));
    }

    resetRandomCirc->Execute(qcRandom, state);

    // check if reset is ok
    BOOST_TEST(qcRandom->Probability(0), 1.);
    for (size_t state = 1; state < nrStates; ++state)
    {
        std::complex<double> aaer = qcRandom->Amplitude(state);

        BOOST_CHECK_PREDICATE(checkClose, (aaer)(0.)(0.000001));
    }

#ifdef __linux__
    if (gpuRandom)
    {
        resetRandomCirc->Execute(gpuRandom, state);
        // check if reset is ok
        BOOST_TEST(gpuRandom->Probability(0), 1.);
        for (size_t state = 1; state < nrStates; ++state)
        {
            std::complex<double> agpu = gpuRandom->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (agpu)(0.)(0.000001));
        }
    }
#endif

    if (questRandom)
    {
        resetRandomCirc->Execute(questRandom, state);
        // check if reset is ok
        BOOST_TEST(questRandom->Probability(0), 1.);
        for (size_t state = 1; state < nrStates; ++state)
        {
            std::complex<double> aquest = questRandom->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (aquest)(0.)(0.000001));
        }
    }

    randomCirc->Clear();
}

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, RandomCircuitsOptimizationTest, bdata::xrange(20, 30), nrGates)
{
    size_t nrStates = 1ULL << 3;

    GenerateCircuit(nrGates, 3);

    auto start = std::chrono::system_clock::now();
    randomCirc->Execute(aer, state);
    auto end = std::chrono::system_clock::now();
    double qiskitTime = std::chrono::duration<double>(end - start).count() * 1000.;

    start = std::chrono::system_clock::now();
    randomCirc->Optimize();
    randomCirc->Execute(qc, state);
    end = std::chrono::system_clock::now();
    double qcsimTime = std::chrono::duration<double>(end - start).count() * 1000.;

    BOOST_TEST_MESSAGE("Time for qiskit aer: " << qiskitTime << " ms, time for qcsim: " << qcsimTime << " ms, qcsim is " << qiskitTime / qcsimTime
                                               << " faster");

#ifdef __linux__
    if (gpuRandom)
    {
        start = std::chrono::system_clock::now();
        randomCirc->Optimize();
        randomCirc->Execute(gpuRandom, state);
        end = std::chrono::system_clock::now();
        double gpuTime = std::chrono::duration<double>(end - start).count() * 1000.;
        BOOST_TEST_MESSAGE("Time for qiskit aer: " << qiskitTime << " ms, time for gpu sim: " << gpuTime << " ms, gpu sim is " << qiskitTime / gpuTime
                                                   << " faster");
    }
#endif

    if (questsim)
    {
        start = std::chrono::system_clock::now();
        randomCirc->Optimize();
        randomCirc->Execute(questsim, state);
        end = std::chrono::system_clock::now();
        double questTime = std::chrono::duration<double>(end - start).count() * 1000.;
        BOOST_TEST_MESSAGE("Time for qiskit aer: " << qiskitTime << " ms, time for quest sim: " << questTime << " ms, quest sim is " << qiskitTime / questTime
                                                   << " faster");
    }

    // now check the results, they should be the same!
    for (size_t state = 0; state < nrStates; ++state)
    {
        std::complex<double> aaer = aer->Amplitude(state);
        std::complex<double> aqc = qc->Amplitude(state);

        // TODO: if this fails, do something to identify the offending gate!
        // rebuild the circuit one by one with the already exising gates from the
        // failed circuit check the results until it fails, then print the last gate
        // in the first circuit that fails
        BOOST_CHECK_PREDICATE(checkClose, (aaer)(aqc)(0.000001));

#ifdef __linux__
        if (gpuRandom)
        {
            std::complex<double> agpu = gpuRandom->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (aaer)(agpu)(0.000001));
        }
#endif

        if (questsim)
        {
            std::complex<double> aquest = questsim->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (aaer)(aquest)(0.000001));
        }
    }

    resetCirc->Execute(aer, state);

    // check if reset is ok
    BOOST_TEST(aer->Probability(0), 1.);
    for (size_t state = 1; state < nrStates; ++state)
    {
        std::complex<double> aaer = aer->Amplitude(state);

        BOOST_CHECK_PREDICATE(checkClose, (aaer)(0.)(0.000001));
    }

    resetCirc->Execute(qc, state);

    // check if reset is ok
    BOOST_TEST(qc->Probability(0), 1.);
    for (size_t state = 1; state < nrStates; ++state)
    {
        std::complex<double> aaer = qc->Amplitude(state);

        BOOST_CHECK_PREDICATE(checkClose, (aaer)(0.)(0.000001));
    }

#ifdef __linux__
    if (gpuRandom)
    {
        resetCirc->Execute(gpuRandom, state);
        // check if reset is ok
        BOOST_TEST(gpuRandom->Probability(0), 1.);
        for (size_t state = 1; state < nrStates; ++state)
        {
            std::complex<double> agpu = gpuRandom->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (agpu)(0.)(0.000001));
        }
    }
#endif

    if (questsim)
    {
        resetCirc->Execute(questsim, state);
        // check if reset is ok
        BOOST_TEST(questsim->Probability(0), 1.);
        for (size_t state = 1; state < nrStates; ++state)
        {
            std::complex<double> aquest = questsim->Amplitude(state);
            BOOST_CHECK_PREDICATE(checkClose, (aquest)(0.)(0.000001));
        }
    }

    randomCirc->Clear();
}

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, RandomCircuitsToLayersTest, bdata::xrange(30, 50), nrGates)
{
    size_t nrStates = 1ULL << nrQubitsForRandomCirc;

    GenerateCircuit(nrGates, nrQubitsForRandomCirc);

    // execute the original circuit on aer
    randomCirc->Execute(aerRandom, state);

    // convert to layers and back to a circuit
    auto layers = randomCirc->ToLayers();
    auto convertedCirc = Circuits::Circuit<>::LayersToCircuit(layers);

    // execute the converted circuit on qcsim
    convertedCirc->Execute(qcRandom, state);

    // now check the results, they should be the same!
    for (size_t state = 0; state < nrStates; ++state)
    {
        std::complex<double> aaer = aerRandom->Amplitude(state);
        std::complex<double> aqc = qcRandom->Amplitude(state);

        BOOST_CHECK_PREDICATE(checkClose, (aaer)(aqc)(0.000001));
    }

    resetRandomCirc->Execute(aerRandom, state);
    resetRandomCirc->Execute(qcRandom, state);

    randomCirc->Clear();
}

BOOST_DATA_TEST_CASE_F(SimulatorsTestFixture, RandomCircuitsToMultipleQubitsLayersTest, bdata::xrange(30, 50), nrGates)
{
    size_t nrStates = 1ULL << nrQubitsForRandomCirc;

    GenerateCircuit(nrGates, nrQubitsForRandomCirc);

    // execute the original circuit on aer
    randomCirc->Execute(aerRandom, state);

    // convert to multiple qubits layers and back to a circuit
    auto layers = randomCirc->ToMultipleQubitsLayers();
    auto convertedCirc = Circuits::Circuit<>::LayersToCircuit(layers);

    // execute the converted circuit on qcsim
    convertedCirc->Execute(qcRandom, state);

    // now check the results, they should be the same!
    for (size_t state = 0; state < nrStates; ++state)
    {
        std::complex<double> aaer = aerRandom->Amplitude(state);
        std::complex<double> aqc = qcRandom->Amplitude(state);

        BOOST_CHECK_PREDICATE(checkClose, (aaer)(aqc)(0.000001));
    }

    resetRandomCirc->Execute(aerRandom, state);
    resetRandomCirc->Execute(qcRandom, state);

    randomCirc->Clear();
}

BOOST_AUTO_TEST_CASE(RemoveEndMeasurementsTests)
{
    using CF = Circuits::CircuitFactory<>;
    using Gate = Circuits::QuantumGateType;

    // 1. Terminal single and multiple consecutive measurements on the same qubit
    // are all removed, while mid-circuit measurements and resets are kept, and
    // unmodified operations share the same pointers.
    auto circ = CF::CreateCircuit();
    auto h0 = CF::CreateGate(Gate::kHadamardGateType, 0);
    auto midMeas0 = CF::CreateMeasurement({{0, 0}});
    auto x0 = CF::CreateGate(Gate::kXGateType, 0);
    auto reset1 = CF::CreateReset({1});
    auto endMeas0a = CF::CreateMeasurement({{0, 0}});
    auto endMeas0b = CF::CreateMeasurement({{0, 1}});
    auto endMeas1 = CF::CreateMeasurement({{1, 2}});

    circ->AddOperation(h0);
    circ->AddOperation(midMeas0);
    circ->AddOperation(x0);
    circ->AddOperation(reset1);
    circ->AddOperation(endMeas0a);
    circ->AddOperation(endMeas0b);
    circ->AddOperation(endMeas1);

    auto cleaned = circ->RemoveEndMeasurements();
    BOOST_REQUIRE(cleaned);
    BOOST_REQUIRE_EQUAL(cleaned->size(), 4ULL);
    BOOST_CHECK_EQUAL((*cleaned)[0].get(), h0.get());
    BOOST_CHECK_EQUAL((*cleaned)[1].get(), midMeas0.get());
    BOOST_CHECK_EQUAL((*cleaned)[2].get(), x0.get());
    BOOST_CHECK_EQUAL((*cleaned)[3].get(), reset1.get());

    // 2. Teleportation circuit: measurements on q0 and q1 have no subsequent
    // gates on q0 or q1, but their classical bits control conditional gates on
    // q2, so they must be kept (with shared pointer identity), while the
    // terminal measurement on q2 is removed.
    auto tele = CF::CreateCircuit();
    auto tx0 = CF::CreateGate(Gate::kXGateType, 0);
    auto th1 = CF::CreateGate(Gate::kHadamardGateType, 1);
    auto tcx12 = CF::CreateGate(Gate::kCXGateType, 1, 2);
    auto tcx01 = CF::CreateGate(Gate::kCXGateType, 0, 1);
    auto th0 = CF::CreateGate(Gate::kHadamardGateType, 0);
    auto tBellMeas = CF::CreateMeasurement({{0, 0}, {1, 1}});
    auto tCondX2 = CF::CreateConditionalGate(CF::CreateGate(Gate::kXGateType, 2), CF::CreateEqualCondition({1}, {true}));
    auto tCondZ2 = CF::CreateConditionalGate(CF::CreateGate(Gate::kZGateType, 2), CF::CreateEqualCondition({0}, {true}));
    auto tEndMeas2 = CF::CreateMeasurement({{2, 2}});

    tele->AddOperation(tx0);
    tele->AddOperation(th1);
    tele->AddOperation(tcx12);
    tele->AddOperation(tcx01);
    tele->AddOperation(th0);
    tele->AddOperation(tBellMeas);
    tele->AddOperation(tCondX2);
    tele->AddOperation(tCondZ2);
    tele->AddOperation(tEndMeas2);

    auto cleanTele = tele->RemoveEndMeasurements();
    BOOST_REQUIRE(cleanTele);
    BOOST_REQUIRE_EQUAL(cleanTele->size(), 8ULL);
    for (size_t i = 0; i < 8; ++i)
        BOOST_CHECK_EQUAL((*cleanTele)[i].get(), (*tele)[i].get());

    // 3. Multi-qubit MeasurementOperation where one qubit has a subsequent gate
    // and another qubit is terminal: only the non-terminal pair is retained,
    // preserving delay and corresponding readout rates.
    auto multi = CF::CreateCircuit();
    auto multiMeas = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}, 2.5);
    multiMeas->SetReadout({Circuits::ReadoutRates{0.1, 0.2}, Circuits::ReadoutRates{0.3, 0.4}});
    auto postGate0 = CF::CreateGate(Gate::kHadamardGateType, 0);
    multi->AddOperation(multiMeas);
    multi->AddOperation(postGate0);

    auto cleanMulti = multi->RemoveEndMeasurements();
    BOOST_REQUIRE(cleanMulti);
    BOOST_REQUIRE_EQUAL(cleanMulti->size(), 2ULL);
    BOOST_CHECK_EQUAL((*cleanMulti)[1].get(), postGate0.get());

    auto filteredMeas = std::dynamic_pointer_cast<Circuits::MeasurementOperation<>>((*cleanMulti)[0]);
    BOOST_REQUIRE(filteredMeas);
    BOOST_CHECK_EQUAL(filteredMeas->GetDelay(), 2.5);
    BOOST_CHECK(filteredMeas->GetQubits() == Types::qubits_vector({0}));
    BOOST_CHECK(filteredMeas->GetBitsIndices() == std::vector<size_t>({0}));
    BOOST_REQUIRE(filteredMeas->HasReadout());
    BOOST_REQUIRE_EQUAL(filteredMeas->GetReadout().size(), 1ULL);
    BOOST_CHECK_CLOSE(filteredMeas->GetReadout()[0].p_meas1_prep0, 0.1, 1e-12);
    BOOST_CHECK_CLOSE(filteredMeas->GetReadout()[0].p_meas0_prep1, 0.2, 1e-12);
}

BOOST_AUTO_TEST_SUITE_END()
