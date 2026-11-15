/**
 * @file NetwSimTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for various simulators in a simple disconnected network host.
 * Just a check of various functionalities, similar with the direct simulator tests elsewhere.
 *
 * Random Maestro circuits are generated using the circuit factory.
 */

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <algorithm>
#include <complex>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"

#include "../Network/SimpleDisconnectedNetwork.h"

namespace
{

std::shared_ptr<Circuits::Circuit<>> GenerateRandomCircuit(int nrGates, int nrQubits, std::mt19937 &g)
{
    auto circuit = Circuits::CircuitFactory<>::CreateCircuit();

    std::uniform_real_distribution<double> paramDist(-2.0 * M_PI, 2.0 * M_PI);
    // the path integral simulation is quite costly for the three qubit gates so avoid them
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType /*kCCXGateType*/));

    for (int i = 0; i < nrGates; ++i)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);

        const auto q1 = qubits[0];
        const auto q2 = qubits[1];
        const auto q3 = qubits[2];

        const double p1 = paramDist(g);
        const double p2 = paramDist(g);
        const double p3 = paramDist(g);
        const double p4 = paramDist(g);

        const auto gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, p1, p2, p3, p4));
    }

    return circuit;
}

std::string GenerateRandomPauliString(int nrQubits, std::mt19937 &g)
{
    std::string pauli(nrQubits, 'I');
    std::uniform_int_distribution<int> pauliDist(0, 3);

    for (int q = 0; q < nrQubits; ++q)
    {
        switch (pauliDist(g))
        {
        case 0:
            pauli[q] = 'X';
            break;
        case 1:
            pauli[q] = 'Y';
            break;
        case 2:
            pauli[q] = 'Z';
            break;
        default:
            pauli[q] = 'I';
            break;
        }
    }

    return pauli;
}

void CheckCountsAgainstStatevector(const Circuits::Circuit<>::ExecuteResults &counts, const Circuits::Circuit<>::ExecuteResults &statevectorCounts,
                                   size_t shots)
{
    size_t totalCounts = 0;
    for (const auto &[outcome, count] : counts)
        totalCounts += count;

    BOOST_TEST(totalCounts == shots);
    BOOST_REQUIRE(!counts.empty());

    for (const auto &meas : counts)
    {
        const auto it = statevectorCounts.find(meas.first);
        const double observedProbability = it == statevectorCounts.end() ? 0.0 : static_cast<double>(it->second) / shots;

        const double expectedProbability = static_cast<double>(meas.second) / shots;
        const double tolerance = 0.01 + 6.0 * std::sqrt(expectedProbability * (1.0 - expectedProbability) / shots);

        BOOST_CHECK_SMALL(std::abs(observedProbability - expectedProbability), tolerance);
    }
}

std::shared_ptr<Circuits::Circuit<>> GenerateRandomMixedCircuit(int nrGates, int nrQubits, std::mt19937 &g)
{
    auto circuit = Circuits::CircuitFactory<>::CreateCircuit();

    std::uniform_real_distribution<double> paramDist(-2.0 * M_PI, 2.0 * M_PI);
    // the path integral simulation is quite costly for the three qubit gates so
    // avoid them
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType /*kCCXGateType*/));
    std::uniform_int_distribution<int> qubitDist(0, nrQubits - 1);
    std::bernoulli_distribution measureNow(0.15); // ~15% chance per gate slot

    // Classical bits: reserve nrQubits slots for mid-circuit measurements (one
    // per qubit) and nrQubits more for the final measurements.
    const size_t midCbitBase = 0;
    const size_t finalCbitBase = static_cast<size_t>(nrQubits);

    // Track which qubits have been measured mid-circuit (and thus have a valid
    // classical bit we can condition on).
    std::vector<bool> measuredMidCircuit(nrQubits, false);

    auto addRandomGate = [&]() {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);

        const auto q1 = qubits[0];
        const auto q2 = qubits[1];
        const auto q3 = qubits[2];

        const double p1 = paramDist(g);
        const double p2 = paramDist(g);
        const double p3 = paramDist(g);
        const double p4 = paramDist(g);

        const auto gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));
        circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, p1, p2, p3, p4));
    };

    for (int i = 0; i < nrGates; ++i)
    {
        addRandomGate();

        // Occasionally insert a mid-circuit measurement on a random qubit.
        if (measureNow(g))
        {
            const int mQubit = qubitDist(g);
            const size_t mCbit = midCbitBase + static_cast<size_t>(mQubit);
            circuit->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement({{static_cast<Types::qubit_t>(mQubit), mCbit}}));
            measuredMidCircuit[mQubit] = true;

            // If we have measured at least one qubit, optionally add a classically
            // controlled gate on another qubit conditioned on this bit.
            if (measureNow(g))
            {
                const int tQubit = qubitDist(g);
                const auto innerGate = std::dynamic_pointer_cast<Circuits::IGateOperation<>>(
                    Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, static_cast<size_t>(tQubit)));
                if (innerGate)
                {
                    circuit->AddOperation(Circuits::CircuitFactory<>::CreateSimpleConditionalGate(innerGate, mCbit));
                }
            }
        }
    }

    // Final measurement on a random non-empty subset of qubits.
    Types::qubits_vector allQubits(nrQubits);
    std::iota(allQubits.begin(), allQubits.end(), 0);
    std::shuffle(allQubits.begin(), allQubits.end(), g);

    std::uniform_int_distribution<int> subsetSizeDist(1, nrQubits);
    const int finalMeasCount = subsetSizeDist(g);

    std::vector<std::pair<Types::qubit_t, size_t>> finalPairs;
    std::vector<size_t> finalCbits;
    for (int k = 0; k < finalMeasCount; ++k)
    {
        const size_t cbit = finalCbitBase + static_cast<size_t>(k);
        finalPairs.push_back({static_cast<Types::qubit_t>(allQubits[k]), cbit});
        finalCbits.push_back(cbit);
    }

    circuit->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(finalPairs));

    return circuit;
}

} // namespace

struct NetwSimTestFixture
{
    NetwSimTestFixture()
    {
        networkSimStatevector = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                                       std::vector<size_t>{static_cast<size_t>(nrQubits)});
        networkSimStatevector->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        networkSimStatevector->CreateSimulator();

        networkSimPauliProp = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                                     std::vector<size_t>{static_cast<size_t>(nrQubits)});
        networkSimPauliProp->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPauliPropagator);
        networkSimPauliProp->CreateSimulator();

        networkSimPathIntegral = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                                        std::vector<size_t>{static_cast<size_t>(nrQubits)});
        networkSimPathIntegral->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPathIntegral);
        networkSimPathIntegral->CreateSimulator();

        networkSimMPS = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                               std::vector<size_t>{static_cast<size_t>(nrQubits)});
        networkSimMPS->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
        networkSimMPS->CreateSimulator();
    }

    ~NetwSimTestFixture()
    {
    }

    static constexpr unsigned int nrQubits = 8;
    static constexpr int nrGates = 15;

    std::shared_ptr<Network::SimpleDisconnectedNetwork<>> networkSimStatevector;
    std::shared_ptr<Network::SimpleDisconnectedNetwork<>> networkSimPauliProp;
    std::shared_ptr<Network::SimpleDisconnectedNetwork<>> networkSimPathIntegral;
    std::shared_ptr<Network::SimpleDisconnectedNetwork<>> networkSimMPS;
};

BOOST_AUTO_TEST_SUITE(NetwSimTests)

BOOST_FIXTURE_TEST_CASE(NetwSimInitializationTest, NetwSimTestFixture)
{
    BOOST_TEST(networkSimStatevector);
    BOOST_TEST(networkSimPauliProp);
    BOOST_TEST(networkSimPathIntegral);
    BOOST_TEST(networkSimMPS);
}

BOOST_DATA_TEST_CASE_F(NetwSimTestFixture, RandomCircuitAmplitudesMatchStatevector, bdata::xrange(10), trial)
{
    std::mt19937 g(static_cast<std::mt19937::result_type>(0xA11CEu + trial));

    auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
    BOOST_REQUIRE(circuit);

    const auto statevectorAmplitudes = networkSimStatevector->ExecuteOnHostAmplitudes(circuit, 0);
    const auto pathIntegralAmplitudes = networkSimPathIntegral->ExecuteOnHostAmplitudes(circuit, 0);
    const auto mpsAmplitudes = networkSimMPS->ExecuteOnHostAmplitudes(circuit, 0);

    // BOOST_REQUIRE(statevectorAmplitudes.size() == (1ULL << nrQubits)); // this can fail if the circuit does not cover all the qubits
    BOOST_REQUIRE(pathIntegralAmplitudes.size() == statevectorAmplitudes.size());
    BOOST_REQUIRE(mpsAmplitudes.size() == statevectorAmplitudes.size());

    for (size_t state = 0; state < statevectorAmplitudes.size(); ++state)
    {
        BOOST_CHECK_SMALL(std::abs(statevectorAmplitudes[state] - pathIntegralAmplitudes[state]), 1e-7);
        BOOST_CHECK_SMALL(std::abs(statevectorAmplitudes[state] - mpsAmplitudes[state]), 1e-6);
    }
}

// Every state has probabilities, also where it has no statevector to read: a
// mixed one, and one that is held without amplitudes.
BOOST_DATA_TEST_CASE_F(NetwSimTestFixture, RandomCircuitProbabilitiesMatchStatevector, bdata::xrange(5), trial)
{
    std::mt19937 g(static_cast<std::mt19937::result_type>(0xD1CEu + trial));

    auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
    BOOST_REQUIRE(circuit);

    std::vector<std::shared_ptr<Network::SimpleDisconnectedNetwork<>>> networks{networkSimStatevector, networkSimMPS, networkSimPathIntegral,
                                                                                networkSimPauliProp};
    for (const auto method :
         {Simulators::SimulationType::kDensityMatrix, Simulators::SimulationType::kMatrixProductOperator, Simulators::SimulationType::kTensorNetwork})
    {
        networks.push_back(std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                                  std::vector<size_t>{static_cast<size_t>(nrQubits)}));
        networks.back()->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, method);
        networks.back()->CreateSimulator();
    }

    // The reference: the gates on a plain statevector of all the qubits, with
    // no network between, so that nothing is optimized away and no qubit is
    // left out. The network leaves out a qubit that nothing executed acts on,
    // one whose gates cancel too, unless it is measured.
    const auto direct = [](const std::shared_ptr<Circuits::Circuit<>> &gates) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        sim->AllocateQubits(nrQubits);
        sim->Initialize();
        Circuits::OperationState state(nrQubits);
        gates->Execute(sim, state);
        return sim->AllProbabilities();
    };
    // That distribution on some of the qubits, in ascending order.
    const auto reduced = [](const std::vector<double> &all, const std::set<size_t> &qubits) {
        std::vector<double> some(size_t{1} << qubits.size(), 0.0);
        for (size_t state = 0; state < all.size(); ++state)
        {
            size_t index = 0, bit = 0;
            for (const auto qubit : qubits)
                index |= ((state >> qubit) & 1) << bit++;
            some[index] += all[state];
        }
        return some;
    };

    // Without measurements the register is the one of the amplitudes.
    const auto amplitudes = networkSimStatevector->ExecuteOnHostAmplitudes(circuit, 0);
    for (const auto &network : networks)
    {
        const auto probabilities = network->ExecuteOnHostProbabilities(circuit, 0);
        BOOST_REQUIRE(probabilities.size() == amplitudes.size());
        for (size_t state = 0; state < amplitudes.size(); ++state)
            BOOST_CHECK_SMALL(probabilities[state] - std::norm(amplitudes[state]), 1e-6);
    }

    // The end measurements of a whole program would collapse the state that is
    // read: they are not executed. Here they measure every qubit the gates
    // use, and each of those has its place in the result.
    const auto gated = circuit->GetQubits();
    const auto expectedGated = reduced(direct(circuit), gated);
    auto measured = std::make_shared<Circuits::Circuit<>>();
    for (const auto &op : circuit->GetOperations())
        measured->AddOperation(op->Clone());
    std::vector<std::pair<Types::qubit_t, size_t>> pairs;
    for (const auto qubit : gated)
        pairs.push_back({qubit, pairs.size()});
    measured->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(pairs));
    for (const auto &network : networks)
    {
        const auto probabilities = network->ExecuteOnHostProbabilities(measured, 0);
        BOOST_REQUIRE(probabilities.size() == expectedGated.size());
        for (size_t state = 0; state < expectedGated.size(); ++state)
            BOOST_CHECK_SMALL(probabilities[state] - expectedGated[state], 1e-6);
    }

    // A qubit that only the end measurements use is still one of the circuit's
    // qubits: it keeps its place in the result, in |0>. Gates on the first six
    // qubits, measurements on all eight.
    auto narrow = GenerateRandomCircuit(nrGates, nrQubits - 2, g);
    BOOST_REQUIRE(narrow);
    const auto expected = direct(narrow);
    std::vector<std::pair<Types::qubit_t, size_t>> everyQubit;
    for (size_t qubit = 0; qubit < nrQubits; ++qubit)
        everyQubit.push_back({qubit, qubit});
    narrow->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(everyQubit));
    for (const auto &network : networks)
    {
        const auto probabilities = network->ExecuteOnHostProbabilities(narrow, 0);
        BOOST_REQUIRE(probabilities.size() == expected.size());
        for (size_t state = 0; state < expected.size(); ++state)
            BOOST_CHECK_SMALL(probabilities[state] - expected[state], 1e-6);
    }

    // The same for a measured qubit whose gates cancel, which an optimizer
    // removes: X twice on qubit 0, a Hadamard on qubit 1, both measured.
    auto cancelled = std::make_shared<Circuits::Circuit<>>();
    cancelled->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
    cancelled->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
    cancelled->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 1));
    cancelled->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement({{0, 0}, {1, 1}}));
    for (const auto &network : networks)
    {
        const auto probabilities = network->ExecuteOnHostProbabilities(cancelled, 0);
        BOOST_REQUIRE(probabilities.size() == 4);
        const double wanted[]{0.5, 0.0, 0.5, 0.0};
        for (size_t state = 0; state < 4; ++state)
            BOOST_CHECK_SMALL(probabilities[state] - wanted[state], 1e-6);
    }
}

BOOST_DATA_TEST_CASE_F(NetwSimTestFixture, RandomPauliExpectationValuesMatchStatevector, bdata::xrange(10), trial)
{
    std::mt19937 g(static_cast<std::mt19937::result_type>(0xBEEFu + trial));

    auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
    BOOST_REQUIRE(circuit);

    std::vector<std::string> paulis;
    for (int i = 0; i < 20; ++i)
        paulis.push_back(GenerateRandomPauliString(nrQubits, g));

    const auto statevectorExpectations = networkSimStatevector->ExecuteExpectations(circuit, paulis);
    const auto pauliPropExpectations = networkSimPauliProp->ExecuteExpectations(circuit, paulis);
    const auto pathIntegralExpectations = networkSimPathIntegral->ExecuteExpectations(circuit, paulis);
    const auto mpsExpectations = networkSimMPS->ExecuteExpectations(circuit, paulis);

    BOOST_REQUIRE(statevectorExpectations.size() == paulis.size());
    BOOST_REQUIRE(pauliPropExpectations.size() == paulis.size());
    BOOST_REQUIRE(pathIntegralExpectations.size() == paulis.size());
    BOOST_REQUIRE(mpsExpectations.size() == paulis.size());

    for (size_t i = 0; i < paulis.size(); ++i)
    {
        BOOST_CHECK_SMALL(std::abs(statevectorExpectations[i] - pauliPropExpectations[i]), 1e-7);
        BOOST_CHECK_SMALL(std::abs(statevectorExpectations[i] - pathIntegralExpectations[i]), 1e-7);
        BOOST_CHECK_SMALL(std::abs(statevectorExpectations[i] - mpsExpectations[i]), 1e-6);
    }
}

BOOST_DATA_TEST_CASE_F(NetwSimTestFixture, RandomCircuitSampleCountsMatchStatevectorProbabilities, bdata::xrange(5), trial)
{
    std::mt19937 g(static_cast<std::mt19937::result_type>(0xC0DEC0DEu + trial));

    auto circuit = GenerateRandomCircuit(nrGates, nrQubits, g);
    BOOST_REQUIRE(circuit);

    Types::qubits_vector sQubits(nrQubits);
    std::iota(sQubits.begin(), sQubits.end(), 0);
    std::shuffle(sQubits.begin(), sQubits.end(), g);

    Types::qubits_vector sampledQubits(sQubits.begin(), sQubits.begin() + (nrQubits / 2));

    constexpr size_t shots = 4000;

    std::vector<std::pair<Types::qubit_t, size_t>> measurementQubits;
    for (size_t i = 0; i < sampledQubits.size(); ++i)
        measurementQubits.emplace_back(sampledQubits[i], sampledQubits[i]);

    circuit->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(measurementQubits));

    const auto statevectorCounts = networkSimStatevector->RepeatedExecuteOnHost(circuit, 0, shots);

    const auto pauliPropCounts = networkSimPauliProp->RepeatedExecuteOnHost(circuit, 0, shots);
    const auto pathIntegralCounts = networkSimPathIntegral->RepeatedExecuteOnHost(circuit, 0, shots);
    const auto mpsCounts = networkSimMPS->RepeatedExecuteOnHost(circuit, 0, shots);

    CheckCountsAgainstStatevector(pauliPropCounts, statevectorCounts, shots);
    CheckCountsAgainstStatevector(pathIntegralCounts, statevectorCounts, shots);
    CheckCountsAgainstStatevector(mpsCounts, statevectorCounts, shots);
}

// A host circuit narrower than the network gets its own simulator, and the
// network simulator is recreated afterwards. That must keep the results the
// execution stored in the classical state.
BOOST_AUTO_TEST_CASE(ExecuteOnHostKeepsMeasurementResults)
{
    for (auto method : {Simulators::SimulationType::kStatevector, Simulators::SimulationType::kMatrixProductState})
    {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{4}, std::vector<size_t>{8});
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        auto circuit = Circuits::CircuitFactory<>::CreateCircuit({Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, 1),
                                                                  Circuits::CircuitFactory<>::CreateMeasurement({{1, 4}, {0, 5}})});

        network->ExecuteOnHost(circuit, 0);
        const auto bits = network->GetState().GetAllBits();
        BOOST_REQUIRE_GE(bits.size(), 6);
        BOOST_TEST(bits[4]);
        BOOST_TEST(!bits[5]);
    }
}

BOOST_DATA_TEST_CASE_F(NetwSimTestFixture, RandomMixedCircuitSampleCountsMatchStatevectorProbabilities, bdata::xrange(5), trial)
{
    std::mt19937 g(static_cast<std::mt19937::result_type>(0xC0DEC0DEu + trial));

    auto circuit = GenerateRandomMixedCircuit(nrGates, nrQubits, g);
    BOOST_REQUIRE(circuit);

    constexpr size_t shots = 2000;

    const auto statevectorCounts = networkSimStatevector->RepeatedExecuteOnHost(circuit, 0, shots);

    const auto pauliPropCounts = networkSimPauliProp->RepeatedExecuteOnHost(circuit, 0, shots);
    const auto pathIntegralCounts = networkSimPathIntegral->RepeatedExecuteOnHost(circuit, 0, shots);
    const auto mpsCounts = networkSimMPS->RepeatedExecuteOnHost(circuit, 0, shots);

    CheckCountsAgainstStatevector(pauliPropCounts, statevectorCounts, shots);
    CheckCountsAgainstStatevector(pathIntegralCounts, statevectorCounts, shots);
    CheckCountsAgainstStatevector(mpsCounts, statevectorCounts, shots);
}

BOOST_AUTO_TEST_SUITE_END()
