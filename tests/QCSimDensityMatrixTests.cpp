/**
 * @file QCSimDensityMatrixTests.cpp
 * @version 1.0
 *
 * Tests for QCSim's density matrix simulator through Maestro.
 */

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Circuit/ChainOrder.h"
#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Simulators/Core/Factory.h"
#include "../python/Noise.h"

namespace
{

constexpr size_t kQCSimDensityNumQubits = 4;
constexpr double kQCSimDensityTolerance = 1e-10;

struct QCSimDensityOperation
{
    unsigned int gate;
    Types::qubit_t qubit0;
    Types::qubit_t qubit1;
    Types::qubit_t qubit2;
    double theta;
    double phi;
    double lambda;
    double gamma;
};

std::shared_ptr<Simulators::ISimulator> MakeQCSim(Simulators::SimulationType simulationType, size_t numQubits = kQCSimDensityNumQubits)
{
    auto simulator = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simulationType);
    simulator->AllocateQubits(numQubits);
    simulator->Initialize();
    return simulator;
}

void ApplyQCSimDensityOperation(Simulators::ISimulator &simulator, const QCSimDensityOperation &operation)
{
    switch (operation.gate)
    {
    case 0:
        simulator.ApplyX(operation.qubit0);
        break;
    case 1:
        simulator.ApplyY(operation.qubit0);
        break;
    case 2:
        simulator.ApplyZ(operation.qubit0);
        break;
    case 3:
        simulator.ApplyH(operation.qubit0);
        break;
    case 4:
        simulator.ApplyS(operation.qubit0);
        break;
    case 5:
        simulator.ApplySDG(operation.qubit0);
        break;
    case 6:
        simulator.ApplyT(operation.qubit0);
        break;
    case 7:
        simulator.ApplyTDG(operation.qubit0);
        break;
    case 8:
        simulator.ApplySx(operation.qubit0);
        break;
    case 9:
        simulator.ApplySxDAG(operation.qubit0);
        break;
    case 10:
        simulator.ApplyK(operation.qubit0);
        break;
    case 11:
        simulator.ApplyP(operation.qubit0, operation.theta);
        break;
    case 12:
        simulator.ApplyRx(operation.qubit0, operation.theta);
        break;
    case 13:
        simulator.ApplyRy(operation.qubit0, operation.theta);
        break;
    case 14:
        simulator.ApplyRz(operation.qubit0, operation.theta);
        break;
    case 15:
        simulator.ApplyU(operation.qubit0, operation.theta, operation.phi, operation.lambda, operation.gamma);
        break;
    case 16:
        simulator.ApplyCX(operation.qubit0, operation.qubit1);
        break;
    case 17:
        simulator.ApplyCY(operation.qubit0, operation.qubit1);
        break;
    case 18:
        simulator.ApplyCZ(operation.qubit0, operation.qubit1);
        break;
    case 19:
        simulator.ApplyCP(operation.qubit0, operation.qubit1, operation.theta);
        break;
    case 20:
        simulator.ApplyCRx(operation.qubit0, operation.qubit1, operation.theta);
        break;
    case 21:
        simulator.ApplyCRy(operation.qubit0, operation.qubit1, operation.theta);
        break;
    case 22:
        simulator.ApplyCRz(operation.qubit0, operation.qubit1, operation.theta);
        break;
    case 23:
        simulator.ApplyCH(operation.qubit0, operation.qubit1);
        break;
    case 24:
        simulator.ApplyCSx(operation.qubit0, operation.qubit1);
        break;
    case 25:
        simulator.ApplyCSxDAG(operation.qubit0, operation.qubit1);
        break;
    case 26:
        simulator.ApplySwap(operation.qubit0, operation.qubit1);
        break;
    case 27:
        simulator.ApplyCU(operation.qubit0, operation.qubit1, operation.theta, operation.phi, operation.lambda, operation.gamma);
        break;
    case 28:
        simulator.ApplyCCX(operation.qubit0, operation.qubit1, operation.qubit2);
        break;
    case 29:
        simulator.ApplyCSwap(operation.qubit0, operation.qubit1, operation.qubit2);
        break;
    default:
        BOOST_FAIL("Unknown random gate " << operation.gate);
    }
}

std::vector<QCSimDensityOperation> GenerateQCSimDensityCircuit(uint64_t seed, size_t gateCount)
{
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<unsigned int> gateDistribution(0, 29);
    const double pi = std::acos(-1.0);
    std::uniform_real_distribution<double> angleDistribution(-2.0 * pi, 2.0 * pi);
    std::array<Types::qubit_t, kQCSimDensityNumQubits> qubits = {0, 1, 2, 3};
    std::vector<QCSimDensityOperation> circuit;
    circuit.reserve(gateCount);

    for (size_t gate = 0; gate < gateCount; ++gate)
    {
        std::shuffle(qubits.begin(), qubits.end(), generator);
        circuit.push_back({gateDistribution(generator), qubits[0], qubits[1], qubits[2], angleDistribution(generator), angleDistribution(generator),
                           angleDistribution(generator), angleDistribution(generator)});
    }
    return circuit;
}

void CheckQCSimDensityProbabilities(Simulators::IState &expectedState, Simulators::IState &actualState)
{
    const auto expected = expectedState.AllProbabilities();
    const auto actual = actualState.AllProbabilities();
    BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
    for (size_t outcome = 0; outcome < expected.size(); ++outcome)
    {
        BOOST_TEST(std::abs(actual[outcome] - expected[outcome]) < kQCSimDensityTolerance, "Probability mismatch for outcome " << outcome);
        BOOST_TEST(std::abs(actualState.Probability(outcome) - expected[outcome]) < kQCSimDensityTolerance);
    }
}

std::vector<double> QCSimDensityMarginals(const std::vector<double> &fullProbabilities, const Types::qubits_vector &qubits)
{
    std::vector<double> probabilities(1ULL << qubits.size(), 0.0);
    for (size_t state = 0; state < fullProbabilities.size(); ++state)
    {
        size_t outcome = 0;
        for (size_t i = 0; i < qubits.size(); ++i)
            outcome |= ((state >> qubits[i]) & 1ULL) << i;
        probabilities[outcome] += fullProbabilities[state];
    }
    return probabilities;
}

void CheckQCSimDensitySamples(const std::unordered_map<Types::qubit_t, Types::qubit_t> &counts, const std::vector<double> &expected, size_t shots,
                              double tolerance = 0.04)
{
    for (size_t outcome = 0; outcome < expected.size(); ++outcome)
    {
        const auto found = counts.find(outcome);
        const double sampled = static_cast<double>(found == counts.end() ? 0 : found->second) / shots;
        BOOST_TEST(std::abs(sampled - expected[outcome]) < tolerance, "Sampling mismatch for outcome " << outcome);
    }
}

} // namespace

struct QCSimDensityMatrixFixture
{
    QCSimDensityMatrixFixture()
        : densityMatrix(MakeQCSim(Simulators::SimulationType::kDensityMatrix)), statevector(MakeQCSim(Simulators::SimulationType::kStatevector))
    {
    }

    std::shared_ptr<Simulators::ISimulator> densityMatrix;
    std::shared_ptr<Simulators::ISimulator> statevector;
};

BOOST_AUTO_TEST_SUITE(qcsim_density_matrix_tests)

BOOST_FIXTURE_TEST_CASE(FactoryConfigurationAndUnsupportedAmplitudes, QCSimDensityMatrixFixture)
{
    BOOST_REQUIRE(densityMatrix);
    BOOST_TEST(static_cast<int>(densityMatrix->GetSimulationType()) == static_cast<int>(Simulators::SimulationType::kDensityMatrix));
    BOOST_TEST(densityMatrix->GetConfiguration("method") == "density_matrix");
    BOOST_CHECK_CLOSE(densityMatrix->Probability(0), 1.0, 1e-10);
    BOOST_CHECK_THROW(densityMatrix->Amplitude(0), std::runtime_error);
    BOOST_CHECK_THROW(densityMatrix->AmplitudeRaw(0), std::runtime_error);
    BOOST_CHECK_THROW(densityMatrix->ProjectOnZero(), std::runtime_error);

    auto unique = Simulators::SimulatorsFactory::CreateSimulatorUnique(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kDensityMatrix);
    BOOST_REQUIRE(unique);
    BOOST_TEST(unique->GetConfiguration("method") == "density_matrix");
}

BOOST_AUTO_TEST_CASE(MixedStateDiagnosticsAndReductions)
{
    auto dm = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);
    auto sv = MakeQCSim(Simulators::SimulationType::kStatevector, 2);
    dm->ApplyH(0);
    dm->ApplyCX(0, 1);
    sv->ApplyH(0);
    sv->ApplyCX(0, 1);

    BOOST_CHECK_SMALL(std::abs(dm->DensityMatrixTrace() - std::complex<double>(1., 0.)), kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(std::abs(dm->DensityMatrixTraceOfSquare() - std::complex<double>(1., 0.)), kQCSimDensityTolerance);
    BOOST_CHECK_CLOSE(dm->DensityMatrixPurity(), 1., 1e-8);
    BOOST_CHECK(dm->IsDensityMatrixHermitian());
    BOOST_CHECK_SMALL(dm->DensityMatrixHermiticityResidual(), kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(std::abs(dm->DensityMatrixOverlap(*dm) - std::complex<double>(1., 0.)), kQCSimDensityTolerance);

    Eigen::VectorXcd bell(4);
    bell << 1. / std::sqrt(2.), 0., 0., 1. / std::sqrt(2.);
    BOOST_CHECK_CLOSE(dm->FidelityWithStatevector(bell), 1., 1e-8);
    const auto reduced = dm->PartialTrace(Types::qubits_vector{0});
    BOOST_REQUIRE_EQUAL(reduced.rows(), 2);
    BOOST_REQUIRE_EQUAL(reduced.cols(), 2);
    BOOST_CHECK_SMALL(std::abs(reduced(0, 0) - 0.5), kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(std::abs(reduced(1, 1) - 0.5), kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(std::abs(reduced(0, 1)), kQCSimDensityTolerance);
}

BOOST_FIXTURE_TEST_CASE(RandomCircuitsAndExpectationsMatchStatevector, QCSimDensityMatrixFixture)
{
    const std::array<std::string, 6> paulis = {"XIII", "IYZI", "ZZZZ", "XYIZ", "YZYX", "IXYZ"};

    // Exercise every exposed gate family at least once before the randomized
    // circuits add varied qubits and parameters.
    for (unsigned int gate = 0; gate < 30; ++gate)
    {
        const QCSimDensityOperation operation = {gate, 0, 1, 2, 0.31, -0.47, 0.83, -0.19};
        ApplyQCSimDensityOperation(*statevector, operation);
        ApplyQCSimDensityOperation(*densityMatrix, operation);
    }
    CheckQCSimDensityProbabilities(*statevector, *densityMatrix);

    for (uint64_t seed = 1; seed <= 8; ++seed)
    {
        if (seed != 1)
        {
            densityMatrix->Reset();
            statevector->Reset();
        }
        for (const auto &operation : GenerateQCSimDensityCircuit(seed, 40))
        {
            ApplyQCSimDensityOperation(*statevector, operation);
            ApplyQCSimDensityOperation(*densityMatrix, operation);
        }

        CheckQCSimDensityProbabilities(*statevector, *densityMatrix);
        const Types::qubits_vector outcomes = {3, 1};
        const auto expectedOutcomes = statevector->Probabilities(outcomes);
        const auto actualOutcomes = densityMatrix->Probabilities(outcomes);
        BOOST_REQUIRE_EQUAL(actualOutcomes.size(), expectedOutcomes.size());
        for (size_t i = 0; i < expectedOutcomes.size(); ++i)
            BOOST_TEST(std::abs(actualOutcomes[i] - expectedOutcomes[i]) < kQCSimDensityTolerance);

        for (const auto &pauli : paulis)
            BOOST_TEST(std::abs(densityMatrix->ExpectationValue(pauli) - statevector->ExpectationValue(pauli)) < kQCSimDensityTolerance,
                       "Expectation value mismatch for " << pauli);
    }
}

BOOST_FIXTURE_TEST_CASE(GenericGatesMatchStatevector, QCSimDensityMatrixFixture)
{
    const double theta = 0.731;
    Eigen::Matrix2cd oneQubit;
    oneQubit << std::cos(theta), -std::sin(theta), std::sin(theta), std::cos(theta);

    Eigen::Matrix4cd twoQubit = Eigen::Matrix4cd::Zero();
    const std::complex<double> imaginary(0.0, 1.0);
    twoQubit(0, 0) = 1.0;
    twoQubit(1, 2) = imaginary;
    twoQubit(2, 1) = imaginary;
    twoQubit(3, 3) = 1.0;

    statevector->ApplyH(0);
    densityMatrix->ApplyH(0);
    statevector->ApplyGenericOneQubitGate(2, oneQubit);
    densityMatrix->ApplyGenericOneQubitGate(2, oneQubit);
    statevector->ApplyGenericTwoQubitGate(0, 2, twoQubit);
    densityMatrix->ApplyGenericTwoQubitGate(0, 2, twoQubit);
    statevector->ApplyNop();
    densityMatrix->ApplyNop();
    CheckQCSimDensityProbabilities(*statevector, *densityMatrix);
}

BOOST_AUTO_TEST_CASE(PureStateInitializationMatchesStatevector)
{
    auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kDensityMatrix);
    auto statevector = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
    std::vector<std::complex<double>> amplitudes = {{0.2, 0.1}, {-0.3, 0.4}, {0.1, -0.5}, {0.6, 0.2}};
    double norm = 0.0;
    for (const auto &amplitude : amplitudes)
        norm += std::norm(amplitude);
    for (auto &amplitude : amplitudes)
        amplitude /= std::sqrt(norm);

    auto densityAmplitudes = amplitudes;
    densityMatrix->InitializeState(2, densityAmplitudes);
    statevector->InitializeState(2, amplitudes);
    CheckQCSimDensityProbabilities(*statevector, *densityMatrix);
    BOOST_TEST(std::abs(densityMatrix->ExpectationValue("X") - statevector->ExpectationValue("X")) < kQCSimDensityTolerance);
}

BOOST_FIXTURE_TEST_CASE(SamplingMatchesMarginalsWithoutCollapsing, QCSimDensityMatrixFixture)
{
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyRy(1, 0.7);
    densityMatrix->ApplyCX(0, 2);
    densityMatrix->ApplyRx(3, -0.4);
    const auto before = densityMatrix->AllProbabilities();
    const Types::qubits_vector qubits = {2, 0, 3};
    const auto expected = QCSimDensityMarginals(before, qubits);
    constexpr size_t shots = 12000;

    CheckQCSimDensitySamples(densityMatrix->SampleCounts(qubits, shots), expected, shots);
    const auto many = densityMatrix->SampleCountsMany(qubits, shots);
    std::unordered_map<Types::qubit_t, Types::qubit_t> packed;
    for (const auto &[bits, count] : many)
    {
        Types::qubit_t outcome = 0;
        for (size_t i = 0; i < bits.size(); ++i)
            if (bits[i])
                outcome |= 1ULL << i;
        packed[outcome] += count;
    }
    CheckQCSimDensitySamples(packed, expected, shots);

    const auto after = densityMatrix->AllProbabilities();
    BOOST_REQUIRE_EQUAL(after.size(), before.size());
    for (size_t outcome = 0; outcome < before.size(); ++outcome)
        BOOST_TEST(std::abs(after[outcome] - before[outcome]) < kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(MeasurementCollapseSaveAndRestore)
{
    auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyCX(0, 1);
    densityMatrix->SaveState();

    const auto first = densityMatrix->Measure({0});
    const auto second = densityMatrix->Measure({1});
    BOOST_TEST(first == second);
    BOOST_CHECK_CLOSE(densityMatrix->Probability(first == 0 ? 0 : 3), 1.0, 1e-8);

    densityMatrix->RestoreState();
    const auto restored = densityMatrix->AllProbabilities();
    BOOST_CHECK_SMALL(restored[0] - 0.5, kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(restored[3] - 0.5, kQCSimDensityTolerance);
    const auto many = densityMatrix->MeasureMany({0, 1});
    BOOST_REQUIRE_EQUAL(many.size(), 2);
    BOOST_TEST(many[0] == many[1]);
}

BOOST_AUTO_TEST_CASE(ResetEntangledQubitProducesMixedState)
{
    auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyCX(0, 1);
    densityMatrix->ApplyReset({0});
    const auto probabilities = densityMatrix->AllProbabilities();
    BOOST_REQUIRE_EQUAL(probabilities.size(), 4);
    BOOST_CHECK_SMALL(probabilities[0] - 0.5, kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(probabilities[1], kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(probabilities[2] - 0.5, kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(probabilities[3], kQCSimDensityTolerance);
}

BOOST_FIXTURE_TEST_CASE(ClonePreservesStateAndIsIndependent, QCSimDensityMatrixFixture)
{
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyRy(1, 0.42);
    densityMatrix->ApplyCX(0, 3);
    densityMatrix->ApplyReset({2});
    densityMatrix->SaveState();
    const auto original = densityMatrix->AllProbabilities();
    auto clone = densityMatrix->Clone();
    BOOST_REQUIRE(clone);
    BOOST_TEST(static_cast<int>(clone->GetSimulationType()) == static_cast<int>(Simulators::SimulationType::kDensityMatrix));
    BOOST_TEST(clone->GetConfiguration("method") == "density_matrix");

    const auto cloned = clone->AllProbabilities();
    BOOST_REQUIRE_EQUAL(cloned.size(), original.size());
    for (size_t outcome = 0; outcome < original.size(); ++outcome)
        BOOST_TEST(std::abs(cloned[outcome] - original[outcome]) < kQCSimDensityTolerance);

    clone->ApplyX(1);
    const auto modifiedClone = clone->AllProbabilities();
    BOOST_TEST(!std::equal(modifiedClone.begin(), modifiedClone.end(), original.begin()));
    const auto unchangedOriginal = densityMatrix->AllProbabilities();
    BOOST_TEST(std::equal(unchangedOriginal.begin(), unchangedOriginal.end(), original.begin()));
    clone->RestoreState();
    CheckQCSimDensityProbabilities(*densityMatrix, *clone);
}

BOOST_AUTO_TEST_CASE(ExactSingleQubitNoiseChannelsAndConventions)
{
    auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 1);
    BOOST_TEST(densityMatrix->SupportsQuantumChannels());

    densityMatrix->ApplyH(0);
    densityMatrix->ApplyPhaseFlipNoise(0, 0.2);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.6, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyPhaseDamping(0, 0.36);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.8, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyPhaseDampingFromTime(0, 0.4, 0.8);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - std::exp(-0.5), kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyX(0);
    densityMatrix->ApplyAmplitudeDamping(0, 0.25);
    BOOST_CHECK_SMALL(densityMatrix->Probability(0) - 0.25, kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.75, kQCSimDensityTolerance);

    // QCSim/Noise.h use total nonidentity-Pauli probability. Consequently
    // p=3/4, rather than p=1, is the fully mixed single-qubit channel.
    densityMatrix->Reset();
    densityMatrix->ApplyDepolarizingNoise(0, 0.75);
    BOOST_CHECK_SMALL(densityMatrix->Probability(0) - 0.5, kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.5, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyDepolarizingMixingNoise(0, 1.0);
    BOOST_CHECK_SMALL(densityMatrix->Probability(0) - 0.5, kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.5, kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(AdvancedAndCorrelatedNoiseChannels)
{
    auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);

    densityMatrix->ApplyGeneralizedAmplitudeDamping(0, 0.4, 0.25);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.1, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyX(0);
    constexpr double duration = 0.7;
    constexpr double t1 = 2.0;
    constexpr double t2 = 1.5;
    constexpr double excitedPopulation = 0.1;
    densityMatrix->ApplyThermalRelaxation(0, duration, t1, t2, excitedPopulation);
    const double expectedExcited = excitedPopulation + (1.0 - excitedPopulation) * std::exp(-duration / t1);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - expectedExcited, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyThermalRelaxation(0, duration, t1, t2, excitedPopulation);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XI") - std::exp(-duration / t2), kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyH(1);
    densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 1, 0.5);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XI"), kQCSimDensityTolerance);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XX") - 1.0, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyH(1);
    densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 1, 0.25, 0.4);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XX") - 0.55, kQCSimDensityTolerance);

    // p=15/16 makes all sixteen two-qubit Paulis equiprobable and hence sends
    // every input to I/4 in the total-Pauli-error convention.
    densityMatrix->Reset();
    densityMatrix->ApplyTwoQubitDepolarizingNoise(0, 1, 15.0 / 16.0);
    const auto probabilities = densityMatrix->AllProbabilities();
    for (const double probability : probabilities)
        BOOST_CHECK_SMALL(probability - 0.25, kQCSimDensityTolerance);

    densityMatrix->Reset();
    densityMatrix->ApplyTwoQubitDepolarizingMixingNoise(0, 1, 1.0);
    for (const double probability : densityMatrix->AllProbabilities())
        BOOST_CHECK_SMALL(probability - 0.25, kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(ArbitraryPauliAndKrausValidation)
{
    auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 2);

    // Base-4 digit zero selects the Pauli on targets[0].
    std::vector<double> pauliProbabilities(16, 0.0);
    pauliProbabilities[1] = 1.0; // XI in target-list order: X on qubit 0.
    densityMatrix->ApplyPauliChannel({0, 1}, pauliProbabilities);
    BOOST_CHECK_CLOSE(densityMatrix->Probability(1), 1.0, 1e-10);

    densityMatrix->Reset();
    pauliProbabilities[1] = 0.0;
    pauliProbabilities[4] = 1.0; // IX: X on qubit 1.
    densityMatrix->ApplyPauliChannel({0, 1}, pauliProbabilities);
    BOOST_CHECK_CLOSE(densityMatrix->Probability(2), 1.0, 1e-10);

    Eigen::MatrixXcd nonTracePreserving = 0.5 * Eigen::MatrixXcd::Identity(2, 2);
    BOOST_CHECK_THROW(densityMatrix->ApplyKrausChannel({0}, {nonTracePreserving}), std::invalid_argument);
    BOOST_CHECK_THROW(densityMatrix->ApplyBitFlipNoise(0, -0.1), std::invalid_argument);
    BOOST_CHECK_THROW(densityMatrix->ApplyPauliChannel(0, 0.6, 0.5, 0.0), std::invalid_argument);
    BOOST_CHECK_THROW(densityMatrix->ApplyQuantumChannel({0, 1}, Simulators::QuantumChannel::BitFlip(0.1)), std::invalid_argument);
    BOOST_CHECK_THROW(densityMatrix->ApplyCorrelatedPhaseFlipNoise(0, 0, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(densityMatrix->ApplyThermalRelaxation(0, 1.0, 1.0, 2.1), std::invalid_argument);

    // A statevector holds no ensemble: it takes one sampled branch of a channel.
    auto statevector = MakeQCSim(Simulators::SimulationType::kStatevector, 1);
    BOOST_TEST(!statevector->SupportsQuantumChannels());
    BOOST_TEST(statevector->SamplesQuantumChannels());
    statevector->ApplyBitFlipNoise(0, 1.0);
    BOOST_CHECK_CLOSE(statevector->Probability(1), 1.0, 1e-10);
    // A backend that neither holds nor samples channels still rejects them.
    auto stabilizer = MakeQCSim(Simulators::SimulationType::kStabilizer, 1);
    BOOST_TEST(!stabilizer->SamplesQuantumChannels());
    BOOST_CHECK_THROW(stabilizer->ApplyBitFlipNoise(0, 0.1), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(NoiseModelUsesExactT1CircuitOperation)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));

    noise::NoiseModel noiseModel;
    noiseModel.set_t1(0, 0.75);
    const auto noisyCircuit = noise::inject_exact_noise(circuit, noiseModel);
    BOOST_REQUIRE_EQUAL(noisyCircuit->GetOperations().size(), 2);
    BOOST_TEST(static_cast<int>(noisyCircuit->GetOperations()[1]->GetType()) == static_cast<int>(Circuits::OperationType::kQuantumChannel));

    auto densityMatrix = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 1);
    Circuits::OperationState classicalState;
    noisyCircuit->Execute(densityMatrix, classicalState);

    // Amplitude damping multiplies |0><1| by sqrt(1-gamma). The old
    // state-independent probabilistic Reset path would incorrectly give 0.25.
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.5, kQCSimDensityTolerance);
}

BOOST_AUTO_TEST_CASE(ExactChannelRunsInOptimizedMultiShotPrefix)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(0.75)));
    circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));
    circuit->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement({{0, 0}}));

    auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{1}, std::vector<size_t>{1});
    network->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kDensityMatrix);
    network->SetMaxSimulators(1);
    network->CreateSimulator();

    constexpr size_t shots = 4096;
    const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
    size_t ones = 0;
    for (const auto &[bits, count] : counts)
        if (!bits.empty() && bits[0])
            ones += count;

    // H, AD(0.75), X gives P(1)=0.875. If a channel incorrectly stops the
    // optimized prefix, the sampler observes the post-H state and returns 0.5.
    const double measuredProbability = static_cast<double>(ones) / shots;
    BOOST_CHECK_SMALL(measuredProbability - 0.875, 0.04);
}

BOOST_AUTO_TEST_CASE(QuantumChannelCircuitComparisonUsesChannelContents)
{
    Circuits::ComparableCircuit<> left;
    Circuits::ComparableCircuit<> equal;
    Circuits::ComparableCircuit<> different;
    left.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseDamping(0.2)));
    equal.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseDamping(0.2)));
    different.AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseDamping(0.3)));

    BOOST_TEST(left == equal);
    BOOST_TEST(left != different);
}

BOOST_AUTO_TEST_CASE(QuantumChannelRecordsHowItCanBeRealized)
{
    using Channel = Simulators::QuantumChannel;
    const auto kind = [](const Channel &channel) { return static_cast<int>(channel.GetFactors().front().kind); };

    BOOST_TEST(kind(Channel::Depolarizing(0.1)) == static_cast<int>(Channel::FactorKind::kPauli));
    BOOST_TEST(Channel::Depolarizing(0.1).IsStateIndependent());

    const Channel damping = Channel::AmplitudeDamping(0.2);
    BOOST_REQUIRE_EQUAL(damping.GetFactors().size(), 1);
    BOOST_TEST(kind(damping) == static_cast<int>(Channel::FactorKind::kThermal));
    BOOST_CHECK_SMALL(damping.GetFactors().front().thermal.decay - 0.2, 1e-15);
    BOOST_CHECK_SMALL(damping.GetFactors().front().thermal.coherence - std::sqrt(0.8), 1e-15);
    BOOST_TEST(!damping.IsStateIndependent());

    // Phase damping is a phase flip whatever Kraus form describes it; the
    // Pauli() probabilities keep their meaning and stay empty.
    const Channel phase = Channel::PhaseDamping(0.36);
    BOOST_TEST(kind(phase) == static_cast<int>(Channel::FactorKind::kPauli));
    BOOST_CHECK_SMALL(phase.GetFactors().front().probabilities[3] - 0.1, 1e-15);
    BOOST_TEST(phase.GetPauliProbabilities().empty());
    BOOST_TEST(phase.IsApprox(Channel::PhaseFlip(0.1)));

    // Kraus operators given without their structure are classified on first use.
    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1.0, 1.0, 1.0, -1.0;
    hadamard /= std::sqrt(2.0);
    const Channel mixture(Channel::KrausOperators{std::sqrt(0.7) * Eigen::MatrixXcd::Identity(2, 2), std::sqrt(0.3) * hadamard});
    BOOST_TEST(kind(mixture) == static_cast<int>(Channel::FactorKind::kUnitaryMixture));
    BOOST_CHECK_SMALL(mixture.GetFactors().front().probabilities[1] - 0.3, 1e-12);
    BOOST_TEST(mixture.IsStateIndependent());
    BOOST_TEST(mixture.IsApprox(Channel::UnitaryMixture({0.7, 0.3}, {Eigen::MatrixXcd::Identity(2, 2), hadamard})));

    const Channel rawFlip(Channel::BitFlip(0.25).GetKrausOperators());
    BOOST_TEST(kind(rawFlip) == static_cast<int>(Channel::FactorKind::kPauli));
    BOOST_CHECK_SMALL(rawFlip.GetFactors().front().probabilities[1] - 0.25, 1e-12);

    const Channel rawDamping(damping.GetKrausOperators());
    BOOST_TEST(kind(rawDamping) == static_cast<int>(Channel::FactorKind::kKraus));

    BOOST_CHECK_THROW(Channel::UnitaryMixture({0.5, 0.4}, {Eigen::MatrixXcd::Identity(2, 2), hadamard}), std::invalid_argument);
    BOOST_CHECK_THROW(Channel::UnitaryMixture({0.5, 0.5}, {Eigen::MatrixXcd::Identity(2, 2), 2.0 * hadamard}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ComposedQuantumChannelKeepsItsPartsAndIsTheirProduct)
{
    using Channel = Simulators::QuantumChannel;
    const Channel depolarizing = Channel::TwoQubitDepolarizing(0.03);
    const Channel relax0 = Channel::ThermalRelaxation(1.0, 50.0, 70.0);
    const Channel relax1 = Channel::ThermalRelaxation(1.5, 40.0, 30.0, 0.1);
    const Channel composed = depolarizing.Then(relax0, {0}).Then(relax1, {1});

    BOOST_REQUIRE_EQUAL(composed.GetNumberOfQubits(), 2);
    BOOST_REQUIRE_EQUAL(composed.GetFactors().size(), 3);
    BOOST_TEST(static_cast<int>(composed.GetFactors()[0].kind) == static_cast<int>(Channel::FactorKind::kPauli));
    BOOST_TEST(static_cast<int>(composed.GetFactors()[1].kind) == static_cast<int>(Channel::FactorKind::kThermal));
    BOOST_REQUIRE_EQUAL(composed.GetFactors()[2].qubits.size(), 1);
    BOOST_TEST(composed.GetFactors()[2].qubits[0] == 1);
    BOOST_TEST(!composed.IsStateIndependent());
    BOOST_TEST(composed.GetPauliProbabilities().empty());

    // The derived Kraus set is minimal and trace preserving.
    const auto &kraus = composed.GetKrausOperators();
    BOOST_TEST(kraus.size() <= 16);
    Eigen::MatrixXcd completeness = Eigen::MatrixXcd::Zero(4, 4);
    for (const auto &op : kraus)
        completeness += op.adjoint() * op;
    BOOST_CHECK_SMALL((completeness - Eigen::MatrixXcd::Identity(4, 4)).norm(), 1e-12);

    // Applying the composed channel equals applying its parts in order, on an
    // entangled state and with the targets reversed and non-adjacent.
    const std::vector<std::string> paulis = {"I", "X", "Y", "Z"};
    for (const auto &targets : {Types::qubits_vector{0, 1}, Types::qubits_vector{2, 0}})
    {
        auto parts = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 3);
        auto whole = MakeQCSim(Simulators::SimulationType::kDensityMatrix, 3);
        for (auto &simulator : {parts, whole})
        {
            simulator->ApplyH(0);
            simulator->ApplyCX(0, 1);
            simulator->ApplyRy(1, 0.7);
            simulator->ApplyCX(1, 2);
            simulator->ApplyRx(2, 1.3);
            simulator->ApplyRy(0, 0.4);
        }
        parts->ApplyQuantumChannel(targets, depolarizing);
        parts->ApplyQuantumChannel({targets[0]}, relax0);
        parts->ApplyQuantumChannel({targets[1]}, relax1);
        whole->ApplyQuantumChannel(targets, composed);
        for (const auto &a : paulis)
            for (const auto &b : paulis)
                for (const auto &c : paulis)
                    BOOST_TEST(std::abs(parts->ExpectationValue(a + b + c) - whole->ExpectationValue(a + b + c)) < kQCSimDensityTolerance,
                               "Composed channel differs from its parts on " << a + b + c);
    }

    // Pauli mixtures compose to a Pauli mixture, by convolution over the Pauli group.
    const Channel twoPaulis = Channel::Depolarizing(0.12).Then(Channel::PhaseFlip(0.2));
    BOOST_REQUIRE_EQUAL(twoPaulis.GetPauliProbabilities().size(), 4);
    BOOST_CHECK_SMALL(twoPaulis.GetPauliProbabilities()[0] - (0.88 * 0.8 + 0.04 * 0.2), 1e-14);
    BOOST_CHECK_SMALL(twoPaulis.GetPauliProbabilities()[1] - (0.04 * 0.8 + 0.04 * 0.2), 1e-14);
    BOOST_CHECK_SMALL(twoPaulis.GetPauliProbabilities()[3] - (0.04 * 0.8 + 0.88 * 0.2), 1e-14);
    const Channel depolarizing1 = Channel::Depolarizing(0.12), phaseFlip = Channel::PhaseFlip(0.2);
    Channel::KrausOperators products;
    for (const auto &first : depolarizing1.GetKrausOperators())
        for (const auto &second : phaseFlip.GetKrausOperators())
            products.emplace_back(second * first);
    BOOST_TEST(twoPaulis.IsApprox(Channel(products), 1e-12));

    // A one-qubit channel placed on the second target of a pair.
    std::vector<double> flipSecond(16, 0.0);
    flipSecond[4] = 1.0; // IX
    BOOST_TEST(Channel::OnQubit(Channel::BitFlip(1.0), 1).IsApprox(Channel::Pauli(flipSecond)));

    BOOST_CHECK_THROW(relax0.Then(depolarizing), std::invalid_argument);
    BOOST_CHECK_THROW(depolarizing.Then(relax0, {2}), std::invalid_argument);
    BOOST_CHECK_THROW(depolarizing.Then(depolarizing, {0, 0}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(QuantumChannelDataIsSharedNotCopied)
{
    using Channel = Simulators::QuantumChannel;
    const Channel channel = Channel::GeneralizedAmplitudeDamping(0.3, 0.1);
    const auto *storage = &channel.GetKrausOperators();

    const auto operation = std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{2}, channel);
    BOOST_TEST(&operation->GetChannel().GetKrausOperators() == storage);
    const auto clone = std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(operation->Clone());
    BOOST_TEST(&clone->GetChannel().GetKrausOperators() == storage);
    const auto remapped = std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(operation->Remap({{2, 0}}));
    BOOST_TEST(&remapped->GetChannel().GetKrausOperators() == storage);
    BOOST_TEST(remapped->AffectedQubits()[0] == 0);

    // Exact noise injection builds each distinct channel once per circuit.
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(1));
    noise::NoiseModel noiseModel;
    noiseModel.set_t1(0, 0.25);
    noiseModel.set_t1(1, 0.5);
    const auto noisy = noise::inject_exact_noise(circuit, noiseModel);
    std::vector<const Channel::KrausOperators *> injected;
    for (const auto &op : noisy->GetOperations())
        if (op->GetType() == Circuits::OperationType::kQuantumChannel)
            injected.push_back(&std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(op)->GetChannel().GetKrausOperators());
    BOOST_REQUIRE_EQUAL(injected.size(), 3);
    BOOST_TEST(injected[0] == injected[1]);
    BOOST_TEST(injected[0] != injected[2]);
}

// A pure-state backend that samples channels runs one trajectory per execution;
// the trajectories must average to the density-matrix result. The gate error is
// a Pauli mixture followed by relaxation with T2 > T1 on one qubit and T2 < T1
// at finite temperature on the other, on reversed targets.
BOOST_AUTO_TEST_CASE(StatevectorSamplesChannelsAsTrajectories)
{
    using Channel = Simulators::QuantumChannel;
    const Channel gateError =
        Channel::TwoQubitDepolarizing(0.05).Then(Channel::ThermalRelaxation(1.0, 4.0, 6.0), {0}).Then(Channel::ThermalRelaxation(1.0, 3.0, 2.0, 0.2), {1});
    const Channel damping = Channel::AmplitudeDamping(0.3);
    const Channel generalized = Channel::GeneralizedAmplitudeDamping(0.25, 0.3);
    const Channel phase = Channel::PhaseDamping(0.2);
    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1.0, 1.0, 1.0, -1.0;
    hadamard /= std::sqrt(2.0);
    const Channel mixture = Channel::UnitaryMixture({0.8, 0.2}, {Eigen::MatrixXcd::Identity(2, 2), hadamard});
    // A two-qubit Kraus set that is no product of one-qubit channels and no
    // mixture of unitaries: the high qubit decays at a rate set by the low one.
    Eigen::MatrixXcd keep = Eigen::MatrixXcd::Identity(4, 4), decayLow = Eigen::MatrixXcd::Zero(4, 4), decayHigh = Eigen::MatrixXcd::Zero(4, 4);
    keep(2, 2) = std::sqrt(0.8);
    keep(3, 3) = std::sqrt(0.6);
    decayLow(0, 2) = std::sqrt(0.2);
    decayHigh(1, 3) = std::sqrt(0.4);
    const Channel correlated(Channel::KrausOperators{keep, decayLow, decayHigh});
    BOOST_REQUIRE_EQUAL(correlated.GetFactors().size(), 1);
    BOOST_TEST(correlated.GetFactors().front().qubits.size() == 2u);
    BOOST_TEST(!correlated.GetFactors().front().IsStateIndependent());

    const auto evolve = [&](Simulators::ISimulator &sim, size_t last) {
        sim.ApplyH(0);
        sim.ApplyCX(0, 1);
        sim.ApplyRy(1, 0.7);
        sim.ApplyCX(1, last);
        sim.ApplyRx(last, 1.3);
        sim.ApplyQuantumChannel({1, 0}, gateError);
        sim.ApplyRy(0, 0.4);
        sim.ApplyQuantumChannel({last}, damping);
        sim.ApplyCX(last, 0);
        sim.ApplyQuantumChannel({0}, generalized);
        sim.ApplyQuantumChannel({1}, phase);
        sim.ApplyQuantumChannel({last}, mixture);
        sim.ApplyQuantumChannel({last, 1}, gateError);
        sim.ApplyRy(1, 1.1);
        sim.ApplyQuantumChannel({0, last}, correlated);
        sim.ApplyQuantumChannel({1, 0}, correlated);
    };

    using Method = Simulators::SimulationType;

    struct Case
    {
        Method method;
        size_t qubits, trajectories;
    };

    // Three statevector qubits run unfused; eleven are above its fusion
    // threshold. The matrix product state samples the same channels.
    for (const auto &c : {Case{Method::kStatevector, 3, 40000}, Case{Method::kStatevector, 11, 3000}, Case{Method::kMatrixProductState, 3, 20000},
                          Case{Method::kMatrixProductState, 11, 2000}})
    {
        const size_t last = c.qubits - 1;
        std::vector<std::string> observables;
        for (const std::string local : {"ZII", "IZI", "IIZ", "XXI", "IXX", "YIY", "ZZZ", "XYZ"})
        {
            // Pauli strings name qubit 0 first: put the third letter on the last qubit.
            std::string full(c.qubits, 'I');
            full[0] = local[0];
            full[1] = local[1];
            full[last] = local[2];
            observables.push_back(full);
        }

        auto exact = MakeQCSim(Simulators::SimulationType::kDensityMatrix, c.qubits);
        evolve(*exact, last);
        auto statevector = MakeQCSim(c.method, c.qubits);
        BOOST_TEST(statevector->SamplesQuantumChannels());
        BOOST_TEST(!statevector->SupportsQuantumChannels());
        BOOST_TEST(!exact->SamplesQuantumChannels());

        statevector->SetSeed(2027);
        std::vector<double> sums(observables.size(), 0.0), squares(observables.size(), 0.0);
        for (size_t trajectory = 0; trajectory < c.trajectories; ++trajectory)
        {
            statevector->Reset();
            evolve(*statevector, last);
            for (size_t i = 0; i < observables.size(); ++i)
            {
                const double value = statevector->ExpectationValue(observables[i]);
                sums[i] += value;
                squares[i] += value * value;
            }
            if (trajectory % 997 == 0)
                BOOST_CHECK_SMALL(statevector->ExpectationValue(std::string(c.qubits, 'I')) - 1.0, 1e-10);
        }
        for (size_t i = 0; i < observables.size(); ++i)
        {
            const double mean = sums[i] / c.trajectories;
            const double error = std::sqrt(std::max(squares[i] / c.trajectories - mean * mean, 0.0) / c.trajectories);
            BOOST_TEST(std::abs(mean - exact->ExpectationValue(observables[i])) < 5.0 * error + 1e-9,
                       observables[i] << " on " << c.qubits << " qubits: trajectories give " << mean << " +- " << error << ", the density matrix "
                                      << exact->ExpectationValue(observables[i]));
        }

        // The same seed replays the same trajectory.
        const auto replay = [&] {
            statevector->SetSeed(77);
            statevector->Reset();
            evolve(*statevector, last);
            return statevector->ExpectationValue(observables[3]);
        };
        const double first = replay();
        BOOST_TEST(replay() == first);
    }

    for (const Method method : {Method::kStatevector, Method::kMatrixProductState})
    {
        auto pure = MakeQCSim(method, 2);
        BOOST_CHECK_THROW(pure->ApplyQuantumChannel({0, 1}, damping), std::invalid_argument);
        BOOST_CHECK_THROW(pure->ApplyQuantumChannel({0, 0}, gateError), std::invalid_argument);
        BOOST_CHECK_THROW(pure->ApplyQuantumChannel({2}, damping), std::invalid_argument);
    }
}

// The matrix product state draws the branch the statevector draws, from the
// same channel stream, so under one seed the two follow one trajectory: the
// explicit environments the chain keeps after a Kraus step give the
// probabilities the full state gives. Queries in between restore the canonical
// form and leave the trajectory where it was.
BOOST_AUTO_TEST_CASE(MatrixProductStateFollowsTheStatevectorTrajectory)
{
    using Channel = Simulators::QuantumChannel;
    using Method = Simulators::SimulationType;
    const Channel relaxSlow = Channel::ThermalRelaxation(1.0, 4.0, 6.0), relaxFast = Channel::ThermalRelaxation(1.0, 3.0, 2.0, 0.2);
    Channel gateError = Channel::TwoQubitDepolarizing(0.05).Then(relaxSlow, {0}).Then(relaxFast, {1});
    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1.0, 1.0, 1.0, -1.0;
    hadamard /= std::sqrt(2.0);
    std::vector<Channel> oneQubit{relaxSlow,
                                  relaxFast,
                                  Channel::AmplitudeDamping(0.2),
                                  Channel::GeneralizedAmplitudeDamping(0.25, 0.3),
                                  Channel::PhaseDamping(0.2),
                                  Channel::UnitaryMixture({0.8, 0.2}, {Eigen::MatrixXcd::Identity(2, 2), hadamard})};
    // Two-qubit Kraus sets: a decay of one qubit at a rate set by the other, three
    // operators cut from a random isometry, and a reset of the pair, whose
    // operators are projections (they leave an environment without full rank).
    std::vector<Channel> twoQubit;
    {
        Eigen::MatrixXcd keep = Eigen::MatrixXcd::Identity(4, 4), decayLow = Eigen::MatrixXcd::Zero(4, 4), decayHigh = Eigen::MatrixXcd::Zero(4, 4);
        keep(2, 2) = std::sqrt(0.8);
        keep(3, 3) = std::sqrt(0.6);
        decayLow(0, 2) = std::sqrt(0.2);
        decayHigh(1, 3) = std::sqrt(0.4);
        twoQubit.emplace_back(Channel::KrausOperators{keep, decayLow, decayHigh});

        std::mt19937 generator(31);
        std::normal_distribution<double> normal;
        Eigen::MatrixXcd random(12, 4);
        for (Eigen::Index k = 0; k < random.size(); ++k)
            random.data()[k] = {normal(generator), normal(generator)};
        const Eigen::MatrixXcd isometry = Eigen::HouseholderQR<Eigen::MatrixXcd>(random).householderQ() * Eigen::MatrixXcd::Identity(12, 4);
        Channel::KrausOperators generic;
        for (Eigen::Index k = 0; k < 3; ++k)
            generic.push_back(isometry.middleRows(4 * k, 4));
        twoQubit.emplace_back(generic);

        Channel::KrausOperators collapse;
        for (Eigen::Index k = 0; k < 4; ++k)
        {
            Eigen::MatrixXcd e = Eigen::MatrixXcd::Zero(4, 4);
            e(k == 3 ? 2 : 0, k) = 1.0;
            collapse.push_back(e);
        }
        twoQubit.emplace_back(collapse);
        for (const auto &channel : twoQubit)
        {
            BOOST_REQUIRE_EQUAL(channel.GetFactors().size(), 1);
            BOOST_TEST(channel.GetFactors().front().qubits.size() == 2u);
            BOOST_TEST(!channel.GetFactors().front().IsStateIndependent());
        }
    }

    struct Step
    {
        int kind;
        size_t a, b;
        double angle;
        size_t channel;
    };

    const auto make = [](Method method, size_t qubits, const char *bond) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        if (method == Method::kMatrixProductState)
        {
            sim->Configure("matrix_product_state_max_bond_dimension", bond);
            sim->Configure("matrix_product_state_truncation_threshold", "0");
        }
        sim->AllocateQubits(qubits);
        sim->Initialize();
        return sim;
    };
    // Applies the steps; `canonical` sweeps the chain back into its canonical form after every one.
    const auto run = [&](Simulators::ISimulator &sim, const std::vector<Step> &steps, uint64_t seed, bool canonical, std::vector<double> &queries) {
        const size_t qubits = sim.GetNumberOfQubits();
        sim.SetSeed(seed);
        for (const auto &step : steps)
        {
            switch (step.kind)
            {
            case 0:
                sim.ApplyRy(step.a, step.angle);
                break;
            case 1:
                sim.ApplyH(step.a);
                break;
            case 2:
                sim.ApplyRz(step.a, step.angle);
                break;
            case 3:
                sim.ApplyQuantumChannel({step.a}, oneQubit[step.channel]);
                break;
            case 4:
                sim.ApplyCX(step.a, step.b);
                sim.ApplyQuantumChannel({step.b, step.a}, gateError);
                break;
            case 5:
                sim.ApplyCZ(step.a, step.b);
                sim.ApplyQuantumChannel({step.a}, oneQubit[step.channel]);
                break;
            case 7:
                // The reset of the pair one time in six.
                sim.ApplyQuantumChannel({step.a, step.b}, twoQubit[step.channel == 5 ? 2 : step.channel % 2]);
                break;
            default: {
                std::string pauli(qubits, 'I');
                pauli[step.a] = 'Z';
                if (qubits > 1)
                    pauli[step.b] = 'X';
                queries.push_back(sim.ExpectationValue(pauli));
                break;
            }
            }
            if (canonical)
                sim.ReCanonicalize();
        }
    };
    const auto distance = [](const std::vector<std::complex<double>> &x, const std::vector<std::complex<double>> &y) {
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
            sum += std::norm(x[i] - y[i]);
        return x.size() == y.size() ? std::sqrt(sum) : 1.0;
    };

    std::mt19937_64 rng(97);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    const auto circuit = [&](size_t qubits, size_t length) {
        std::vector<Step> steps;
        for (size_t i = 0; i < length; ++i)
        {
            const size_t a = rng() % qubits, b = qubits > 1 ? (a + 1 + rng() % (qubits - 1)) % qubits : 0;
            steps.push_back({static_cast<int>(rng() % (qubits > 1 ? 7 : 4)), a, b, angle(rng), static_cast<size_t>(rng() % oneQubit.size())});
            if (steps.back().kind == 6 && rng() % 3) // fewer queries than operations
                steps.back().kind = 0;
        }
        return steps;
    };

    BOOST_TEST(make(Method::kMatrixProductState, 2, "8")->SamplesQuantumChannels());
    BOOST_TEST(Simulators::SamplesQuantumChannels(Simulators::SimulatorType::kQCSim, Method::kMatrixProductState));

    for (const size_t qubits : {size_t{1}, size_t{2}, size_t{5}, size_t{9}})
        for (int trial = 0; trial < 12; ++trial)
        {
            const auto steps = circuit(qubits, 70);
            const uint64_t seed = 4000 + trial;
            std::vector<double> expected, measured;
            auto statevector = make(Method::kStatevector, qubits, "");
            auto chain = make(Method::kMatrixProductState, qubits, "512");
            run(*statevector, steps, seed, false, expected);
            run(*chain, steps, seed, false, measured);
            const double error = distance(chain->GetStateVector(), statevector->GetStateVector());
            BOOST_TEST(error < 1e-6, qubits << " qubits, trial " << trial << ": the chain left the statevector's trajectory by " << error);
            BOOST_REQUIRE(measured.size() == expected.size());
            for (size_t i = 0; i < expected.size(); ++i)
                BOOST_TEST(std::abs(measured[i] - expected[i]) < 1e-8, qubits << " qubits, trial " << trial << ", query " << i);

            // A copy taken after a channel holds the same state and goes on as the original does.
            const auto more = circuit(qubits, 12);
            auto longer = make(Method::kStatevector, qubits, ""), original = make(Method::kMatrixProductState, qubits, "512");
            std::vector<double> unused;
            run(*longer, steps, seed, false, unused);
            run(*original, steps, seed, false, unused);
            original->ApplyQuantumChannel({0}, oneQubit[2]);
            longer->ApplyQuantumChannel({0}, oneQubit[2]);
            const auto copy = original->Clone();
            BOOST_TEST(distance(copy->GetStateVector(), longer->GetStateVector()) < 1e-6, qubits << " qubits, trial " << trial << ": copied chain");
            run(*longer, more, seed + 77, false, unused);
            run(*copy, more, seed + 77, false, unused);
            BOOST_TEST(distance(copy->GetStateVector(), longer->GetStateVector()) < 1e-6,
                       qubits << " qubits, trial " << trial << ": copied chain after more operations");
        }

    // With a bond limit the environments weigh a truncation as the canonical form
    // does. After a noisy evolution without truncation, one gate that has to cut
    // its bond loses the same fidelity whether the chain kept its explicit
    // environments or was first swept back into the canonical form: both cuts
    // are the optimal one. (The fidelity, not the state, is compared: among
    // equal Schmidt values either cut may keep a different one.)
    const auto fidelity = [](const std::vector<std::complex<double>> &x, const std::vector<std::complex<double>> &y) {
        std::complex<double> overlap = 0.0;
        double normX = 0.0, normY = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            overlap += std::conj(x[i]) * y[i];
            normX += std::norm(x[i]);
            normY += std::norm(y[i]);
        }
        return std::norm(overlap) / (normX * normY);
    };
    // Weaker noise here: strong damping resets the qubits so often that no bond grows enough to be cut.
    const Channel mildSlow = Channel::ThermalRelaxation(0.1, 4.0, 6.0), mildFast = Channel::ThermalRelaxation(0.1, 3.0, 2.0, 0.2);
    gateError = Channel::TwoQubitDepolarizing(0.01).Then(mildSlow, {0}).Then(mildFast, {1});
    oneQubit = {mildSlow,
                mildFast,
                Channel::AmplitudeDamping(0.03),
                Channel::GeneralizedAmplitudeDamping(0.04, 0.3),
                Channel::PhaseDamping(0.05),
                Channel::UnitaryMixture({0.9, 0.1}, {Eigen::MatrixXcd::Identity(2, 2), hadamard})};
    size_t truncations = 0;
    for (int trial = 0; trial < 24; ++trial)
    {
        const size_t qubits = 8;
        // Neighbouring pairs only: the chain then keeps its qubits in place, and one gate cuts one bond.
        auto steps = circuit(qubits, 80);
        for (auto &step : steps)
        {
            if (step.kind == 6)
                step.kind = 2;
            step.a = std::min(step.a, qubits - 2);
            step.b = step.a + 1;
        }
        steps.insert(steps.begin(), Step{3, 0, 0, 0.0, 2});
        const size_t a = 2 + trial % 4;
        const char *limit = trial % 3 == 1 ? "3" : "2";
        std::vector<double> unused;
        auto exact = make(Method::kStatevector, qubits, "");
        auto lazy = make(Method::kMatrixProductState, qubits, "512"), swept = make(Method::kMatrixProductState, qubits, "512");
        run(*exact, steps, 6000 + trial, false, unused);
        run(*lazy, steps, 6000 + trial, false, unused);
        run(*swept, steps, 6000 + trial, false, unused);
        swept->ReCanonicalize();
        for (auto *sim : {exact.get(), lazy.get(), swept.get()})
        {
            if (sim != exact.get())
                sim->Configure("matrix_product_state_max_bond_dimension", limit);
            sim->ApplyRy(a, 0.9);
            sim->ApplyCX(a, a + 1);
        }
        const auto reference = exact->GetStateVector();
        const double kept = fidelity(lazy->GetStateVector(), reference), expected = fidelity(swept->GetStateVector(), reference);
        BOOST_TEST(std::abs(kept - expected) < 1e-9,
                   "trial " << trial << ": the cut with explicit environments keeps fidelity " << kept << ", the canonical cut " << expected);
        truncations += expected < 1.0 - 1e-6;
    }
    BOOST_TEST(truncations >= 8, "only " << truncations << " of the gates had anything to cut");

    // Two-qubit Kraus sets. The chain draws the operator from the reduced density
    // matrix of the pair, built from the two-site tensor and its environments,
    // and splits the tensor the operator gives in one decomposition; pairs that
    // are not neighbours are brought together first. It follows the statevector
    // here too, in either order of the two targets.
    oneQubit = {Channel::ThermalRelaxation(1.0, 4.0, 6.0),
                Channel::ThermalRelaxation(1.0, 3.0, 2.0, 0.2),
                Channel::AmplitudeDamping(0.2),
                Channel::GeneralizedAmplitudeDamping(0.25, 0.3),
                Channel::PhaseDamping(0.2),
                Channel::UnitaryMixture({0.8, 0.2}, {Eigen::MatrixXcd::Identity(2, 2), hadamard})};
    size_t pairSteps = 0;
    for (const size_t qubits : {size_t{2}, size_t{3}, size_t{6}, size_t{9}})
        for (int trial = 0; trial < 12; ++trial)
        {
            auto steps = circuit(qubits, 70);
            for (auto &step : steps)
                if ((step.kind == 4 || step.kind == 5) && rng() % 3)
                {
                    step.kind = 7;
                    ++pairSteps;
                }
            const uint64_t seed = 8000 + trial;
            std::vector<double> expected, measured;
            auto statevector = make(Method::kStatevector, qubits, "");
            auto chain = make(Method::kMatrixProductState, qubits, "512");
            run(*statevector, steps, seed, false, expected);
            run(*chain, steps, seed, false, measured);
            const double error = distance(chain->GetStateVector(), statevector->GetStateVector());
            BOOST_TEST(error < 1e-6,
                       qubits << " qubits, trial " << trial << ": through two-qubit Kraus sets the chain left the statevector's trajectory by " << error);
            BOOST_REQUIRE(measured.size() == expected.size());
            for (size_t i = 0; i < expected.size(); ++i)
                BOOST_TEST(std::abs(measured[i] - expected[i]) < 1e-8, qubits << " qubits, trial " << trial << ", query " << i);
        }
    BOOST_TEST(pairSteps > 300u);
}

// A composite simulator keeps the qubits that share no entanglement in separate
// statevectors. It samples a channel in the component that holds the channel's
// qubit; a two-qubit Pauli error joins nothing.
BOOST_AUTO_TEST_CASE(CompositeSamplesChannelsInItsComponents)
{
    using Channel = Simulators::QuantumChannel;
    using Method = Simulators::SimulationType;
    const Channel relaxSlow = Channel::ThermalRelaxation(1.0, 4.0, 6.0), relaxFast = Channel::ThermalRelaxation(1.0, 3.0, 2.0, 0.2);
    const Channel gateError = Channel::TwoQubitDepolarizing(0.05).Then(relaxSlow, {0}).Then(relaxFast, {1});
    const Channel damping = Channel::AmplitudeDamping(0.3), generalized = Channel::GeneralizedAmplitudeDamping(0.25, 0.3);
    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1.0, 1.0, 1.0, -1.0;
    hadamard /= std::sqrt(2.0);
    const Channel mixture = Channel::UnitaryMixture({0.8, 0.2}, {Eigen::MatrixXcd::Identity(2, 2), hadamard});
    // Both qubits decay together or not at all: a two-qubit set that is no product and no mixture of unitaries.
    Eigen::MatrixXcd keep = Eigen::MatrixXcd::Identity(4, 4), decay = Eigen::MatrixXcd::Zero(4, 4);
    keep(3, 3) = std::sqrt(0.7);
    decay(0, 3) = std::sqrt(0.3);
    const Channel pairDecay(Channel::KrausOperators{keep, decay});

    constexpr size_t qubits = 6;
    const auto evolve = [&](Simulators::ISimulator &sim) {
        sim.ApplyH(0);
        sim.ApplyCX(0, 1);
        sim.ApplyRy(2, 0.9);
        sim.ApplyRx(3, 1.3);
        sim.ApplyH(4);
        sim.ApplyX(5);
        sim.ApplyQuantumChannel({1, 0}, gateError); // inside one component
        sim.ApplyQuantumChannel({2}, damping);      // a lone qubit
        sim.ApplyQuantumChannel({3, 2}, gateError); // across two components: the Pauli part joins nothing
        sim.ApplyQuantumChannel({4}, generalized);
        sim.ApplyQuantumChannel({3}, mixture);
        sim.ApplyQuantumChannel({5, 4}, pairDecay); // joins its two qubits
        sim.ApplyCX(2, 3);
        sim.ApplyQuantumChannel({3}, relaxFast);
        sim.ApplyQuantumChannel({0}, relaxSlow);
    };
    const auto make = [&](Simulators::SimulatorType type, Method method) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, method);
        sim->AllocateQubits(qubits);
        sim->Initialize();
        return sim;
    };
    const std::vector<std::string> observables{"ZIIIII", "IZIIII", "XXIIII", "IIZIII", "IIIZII", "IIXXII", "IIIIZI", "IIIIIZ", "IIIIZZ", "ZIZIZI", "IIYYII"};

    auto exact = make(Simulators::SimulatorType::kQCSim, Method::kDensityMatrix);
    evolve(*exact);
    auto composite = make(Simulators::SimulatorType::kCompositeQCSim, Method::kStatevector);
    BOOST_TEST(composite->SamplesQuantumChannels());
    BOOST_TEST(!composite->SupportsQuantumChannels());
    BOOST_TEST(Simulators::SamplesQuantumChannels(Simulators::SimulatorType::kCompositeQCSim, Method::kStatevector));

    constexpr size_t trajectories = 30000;
    composite->SetSeed(909);
    std::vector<double> sums(observables.size(), 0.0), squares(observables.size(), 0.0);
    for (size_t trajectory = 0; trajectory < trajectories; ++trajectory)
    {
        composite->Reset();
        evolve(*composite);
        for (size_t i = 0; i < observables.size(); ++i)
        {
            const double value = composite->ExpectationValue(observables[i]);
            sums[i] += value;
            squares[i] += value * value;
        }
    }
    for (size_t i = 0; i < observables.size(); ++i)
    {
        const double mean = sums[i] / trajectories;
        const double error = std::sqrt(std::max(squares[i] / trajectories - mean * mean, 0.0) / trajectories);
        BOOST_TEST(std::abs(mean - exact->ExpectationValue(observables[i])) < 5.0 * error + 1e-9, observables[i] << ": composite trajectories give " << mean
                                                                                                                 << " +- " << error << ", the density matrix "
                                                                                                                 << exact->ExpectationValue(observables[i]));
    }

    // The same seed replays the same trajectory.
    const auto replay = [&] {
        composite->SetSeed(31);
        composite->Reset();
        evolve(*composite);
        return composite->ExpectationValue(observables[2]) + composite->ExpectationValue(observables[8]);
    };
    const double first = replay();
    BOOST_CHECK_SMALL(replay() - first, 1e-12);

    BOOST_CHECK_THROW(composite->ApplyQuantumChannel({0, 1}, damping), std::invalid_argument);
    BOOST_CHECK_THROW(composite->ApplyQuantumChannel({0, 0}, gateError), std::invalid_argument);
    BOOST_CHECK_THROW(composite->ApplyQuantumChannel({qubits}, damping), std::invalid_argument);
}

// A matrix product state that samples channels keeps the environments of its
// bonds, and pays for every site between two operations that read them. The
// chain order runs the operations of a circuit so that they stay close: any
// order that keeps the order on every qubit is the same circuit.
BOOST_AUTO_TEST_CASE(ChainOrderKeepsTheCircuitAndShortensTheWalk)
{
    using Channel = Simulators::QuantumChannel;
    using Type = Circuits::OperationType;
    using CircuitPtr = std::shared_ptr<Circuits::Circuit<>>;
    const Channel damping = Channel::AmplitudeDamping(0.1), relax = Channel::ThermalRelaxation(1.0, 4.0, 6.0);
    const Channel drawn = Channel::Depolarizing(0.05);
    const Channel gateError = Channel::TwoQubitDepolarizing(0.04).Then(relax, {0}).Then(damping, {1});
    const auto channel = [](const Types::qubits_vector &qubits, const Channel &c) { return std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, c); };

    std::mt19937_64 rng(4711);
    std::uniform_real_distribution<double> angle(-3.0, 3.0);
    // barriers: resets and measurements among the operations.
    const auto build = [&](size_t qubits, size_t length, bool barriers) {
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        for (size_t i = 0; i < length; ++i)
        {
            const size_t a = rng() % qubits, b = a + 1 < qubits ? a + 1 : a - 1;
            switch (rng() % (barriers ? 9 : 7))
            {
            case 0:
                circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(a, angle(rng)));
                circuit->AddOperation(channel({a}, rng() % 2 ? damping : relax));
                break;
            case 1:
                circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(a));
                circuit->AddOperation(channel({a}, drawn));
                break;
            case 2:
                circuit->AddOperation(std::make_shared<Circuits::RzGate<>>(a, angle(rng)));
                break;
            case 3:
            case 4:
                circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(a, b));
                circuit->AddOperation(channel(rng() % 2 ? Types::qubits_vector{a, b} : Types::qubits_vector{b, a}, gateError));
                break;
            case 5:
                circuit->AddOperation(std::make_shared<Circuits::CZGate<>>(a, b));
                break;
            case 6:
                circuit->AddOperation(std::make_shared<Circuits::Delay<>>(a, 1e-7));
                circuit->AddOperation(channel({a}, relax));
                break;
            case 7:
                circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{a}));
                break;
            default:
                circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{a, a}}));
                break;
            }
        }
        return circuit;
    };
    const auto barrier = [](const std::shared_ptr<Circuits::IOperation<>> &op) {
        return op->GetType() != Type::kGate && op->GetType() != Type::kQuantumChannel && op->GetType() != Type::kDelay;
    };
    // The environment products of an order, counted as the chain does: an operation that reads the state needs
    // those of the blocks on both its sides, and leaves current only the ones on its own side of each bond.
    const auto walk = [&](const CircuitPtr &circuit, size_t qubits) {
        size_t left = qubits, right = 0, products = 0;
        bool reading = false;
        for (const auto &op : circuit->GetOperations())
        {
            if (barrier(op))
            {
                left = qubits, right = 0, reading = false;
                continue;
            }
            const auto targets = op->AffectedQubits();
            const size_t low = *std::min_element(targets.begin(), targets.end()), high = *std::max_element(targets.begin(), targets.end());
            const bool sampled =
                op->GetType() == Type::kQuantumChannel && !std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(op)->GetChannel().IsStateIndependent();
            reading = reading || sampled;
            if (!reading || (targets.size() == 1 && !sampled))
                continue;
            products += (low > left ? low - left : 0) + (right > high ? right - high : 0);
            left = targets.size() == 2 ? low + 1 : low;
            right = low;
        }
        return products;
    };

    size_t before = 0, after = 0;
    for (int trial = 0; trial < 30; ++trial)
    {
        const size_t qubits = 4 + trial % 9;
        const bool barriers = trial % 3 == 0;
        const auto circuit = build(qubits, 200, barriers);
        BOOST_TEST(Circuits::ChainOrdered(circuit, Circuits::ChainOrder::kNone) == circuit);
        // For a matrix product state that samples channels, and for a matrix product operator.
        for (const auto chain : {Circuits::ChainOrder::kSampledChannels, Circuits::ChainOrder::kMixedState})
        {
            const bool sampling = chain == Circuits::ChainOrder::kSampledChannels;
            const auto ordered = Circuits::ChainOrdered(circuit, chain);
            const auto &original = circuit->GetOperations();
            const auto &moved = ordered->GetOperations();
            BOOST_REQUIRE_EQUAL(moved.size(), original.size());
            BOOST_TEST(ordered != circuit, "trial " << trial << " was not reordered");

            // The same operations, in the same order on every qubit, with everything that came before a
            // measurement or a reset still before it.
            for (size_t q = 0; q < qubits; ++q)
            {
                const auto on = [&](const std::vector<std::shared_ptr<Circuits::IOperation<>>> &ops) {
                    std::vector<const Circuits::IOperation<> *> wire;
                    for (const auto &op : ops)
                    {
                        const auto targets = op->AffectedQubits();
                        if (barrier(op) || std::find(targets.begin(), targets.end(), q) != targets.end())
                            wire.push_back(op.get());
                    }
                    return wire;
                };
                BOOST_TEST((on(moved) == on(original)), "trial " << trial << ": the order on qubit " << q << " changed");
            }
            for (size_t i = 0; i < original.size(); ++i)
                if (barrier(original[i]))
                {
                    const size_t j = static_cast<size_t>(std::find(moved.begin(), moved.end(), original[i]) - moved.begin());
                    BOOST_REQUIRE(j < moved.size());
                    BOOST_TEST(j == i, "trial " << trial << ": a measurement or reset moved from " << i << " to " << j);
                }
            if (sampling)
            {
                before += walk(circuit, qubits);
                after += walk(ordered, qubits);
            }

            // The same map: on the density matrix, which applies the channels exactly, both orders end in the same state.
            if (!barriers && qubits <= 7)
            {
                Types::qubits_vector all(qubits);
                std::iota(all.begin(), all.end(), Types::qubit_t{0});
                Eigen::MatrixXcd states[2];
                int index = 0;
                for (const auto &run : {circuit, ordered})
                {
                    auto exact = MakeQCSim(Simulators::SimulationType::kDensityMatrix, qubits);
                    Circuits::OperationState state;
                    state.AllocateBits(qubits);
                    run->Execute(exact, state);
                    states[index++] = exact->PartialTrace(all);
                }
                BOOST_TEST((states[0] - states[1]).norm() < 1e-10, "trial " << trial << ": the reordered circuit is another map");
            }
        }
    }
    BOOST_TEST(after * 3 < before, "the chain order saved too little: " << after << " environment products against " << before);

    // Left as it is: nothing reads the state; a gate needs routing; a gate on three qubits.
    auto plain = std::make_shared<Circuits::Circuit<>>();
    plain->AddOperation(std::make_shared<Circuits::CXGate<>>(2, 3));
    plain->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    plain->AddOperation(channel({0}, drawn));
    plain->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    BOOST_TEST(Circuits::ChainOrdered(plain) == plain);
    // On a matrix product operator the drawn channel is a map like any other.
    BOOST_TEST(Circuits::ChainOrdered(plain, Circuits::ChainOrder::kMixedState) != plain);
    auto routed = std::make_shared<Circuits::Circuit<>>(plain->GetOperations());
    routed->AddOperation(channel({3}, damping));
    BOOST_TEST(Circuits::ChainOrdered(routed) != routed);
    routed->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 2));
    BOOST_TEST(Circuits::ChainOrdered(routed) == routed);
    auto wide = std::make_shared<Circuits::Circuit<>>(plain->GetOperations());
    wide->AddOperation(channel({3}, damping));
    wide->AddOperation(std::make_shared<Circuits::CCXGate<>>(0, 1, 2));
    BOOST_TEST(Circuits::ChainOrdered(wide) == wide);
}

// A circuit laid out on a device acts on a line of qubits whose indices are
// not in order. Placed along the line it needs no swap, and its chain order is
// that of the same circuit written on index neighbours.
BOOST_AUTO_TEST_CASE(ChainLinePlacesQubitsThatFormALineOutOfIndexOrder)
{
    using Channel = Simulators::QuantumChannel;
    using CircuitPtr = std::shared_ptr<Circuits::Circuit<>>;
    const Channel damping = Channel::AmplitudeDamping(0.1), relax = Channel::ThermalRelaxation(1.0, 4.0, 6.0);
    const Channel gateError = Channel::TwoQubitDepolarizing(0.04).Then(relax, {0}).Then(damping, {1});
    const auto channel = [](const Types::qubits_vector &qubits, const Channel &c) { return std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, c); };
    // Gates and channels between neighbours of the line at[0], at[1], ..., every pair of them used at least once.
    const auto build = [&](const std::vector<Types::qubit_t> &at, size_t length, uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> angle(-3.0, 3.0);
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        const size_t qubits = at.size();
        for (size_t i = 0; i + 1 < qubits; ++i)
            circuit->AddOperation(std::make_shared<Circuits::CZGate<>>(at[i], at[i + 1]));
        for (size_t i = 0; i < length; ++i)
        {
            const size_t a = rng() % qubits, b = a + 1 < qubits ? a + 1 : a - 1;
            switch (rng() % 6)
            {
            case 0:
                circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(at[a], angle(rng)));
                circuit->AddOperation(channel({at[a]}, damping));
                break;
            case 1:
                circuit->AddOperation(std::make_shared<Circuits::RzGate<>>(at[a], angle(rng)));
                break;
            case 2:
            case 3:
                circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(at[a], at[b]));
                circuit->AddOperation(channel(rng() % 2 ? Types::qubits_vector{at[a], at[b]} : Types::qubits_vector{at[b], at[a]}, gateError));
                break;
            case 4:
                circuit->AddOperation(std::make_shared<Circuits::CZGate<>>(at[a], at[b]));
                break;
            default:
                circuit->AddOperation(std::make_shared<Circuits::Delay<>>(at[a], 1e-7));
                circuit->AddOperation(channel({at[a]}, relax));
                break;
            }
        }
        return circuit;
    };
    // The place, in the circuit as given, of each operation of the circuit as ordered.
    const auto order = [](const CircuitPtr &given, const CircuitPtr &ordered) {
        std::vector<size_t> places;
        const auto &original = given->GetOperations();
        for (const auto &op : ordered->GetOperations())
            places.push_back(static_cast<size_t>(std::find(original.begin(), original.end(), op) - original.begin()));
        return places;
    };

    std::mt19937_64 shuffle(2718);
    for (int trial = 0; trial < 24; ++trial)
    {
        const size_t qubits = 4 + trial % 9;
        std::vector<Types::qubit_t> straight(qubits), at(qubits);
        std::iota(straight.begin(), straight.end(), Types::qubit_t{0});
        do
        {
            at = straight;
            std::shuffle(at.begin(), at.end(), shuffle);
            // the line is told from its end with the smaller index
            if (at.front() > at.back())
                std::reverse(at.begin(), at.end());
        } while (at == straight);
        const auto onLine = build(at, 150, 500 + trial), inOrder = build(straight, 150, 500 + trial);

        BOOST_TEST(Circuits::ChainLine(inOrder, qubits).empty(), "trial " << trial << ": a chain in index order was given a layout");
        const auto line = Circuits::ChainLine(onLine, qubits);
        BOOST_REQUIRE_EQUAL(line.size(), qubits);
        for (size_t place = 0; place < qubits; ++place)
            BOOST_TEST(line[at[place]] == static_cast<long long int>(place), "trial " << trial << ": qubit " << at[place] << " is not at place " << place);

        for (const auto chain : {Circuits::ChainOrder::kSampledChannels, Circuits::ChainOrder::kMixedState})
        {
            // By index the circuit needs routing and is left as it is; by its line it is the chain in index order.
            BOOST_TEST(Circuits::ChainOrdered(onLine, chain) == onLine);
            const auto ordered = Circuits::ChainOrdered(onLine, chain, &line);
            BOOST_TEST(ordered != onLine, "trial " << trial << " was not reordered along its line");
            BOOST_TEST((order(onLine, ordered) == order(inOrder, Circuits::ChainOrdered(inOrder, chain))),
                       "trial " << trial << ": the order along the line is not the order of the chain");
        }

        // The same map: both orders end in the same state on the density matrix, which runs them as listed.
        if (qubits <= 7)
        {
            Types::qubits_vector all(qubits);
            std::iota(all.begin(), all.end(), Types::qubit_t{0});
            Eigen::MatrixXcd states[2];
            int index = 0;
            for (const auto &run : {onLine, Circuits::ChainOrdered(onLine, Circuits::ChainOrder::kSampledChannels, &line)})
            {
                auto exact = MakeQCSim(Simulators::SimulationType::kDensityMatrix, qubits);
                Circuits::OperationState state;
                state.AllocateBits(qubits);
                run->Execute(exact, state);
                states[index++] = exact->PartialTrace(all);
            }
            BOOST_TEST((states[0] - states[1]).norm() < 1e-10, "trial " << trial << ": the circuit ordered along its line is another map");
        }
    }

    // Several lines and a qubit on its own: each line in turn, by its smallest qubit, from its end with the smaller index.
    auto pieces = std::make_shared<Circuits::Circuit<>>();
    pieces->AddOperation(std::make_shared<Circuits::CXGate<>>(6, 2));
    pieces->AddOperation(std::make_shared<Circuits::CXGate<>>(2, 4));
    pieces->AddOperation(std::make_shared<Circuits::HadamardGate<>>(3));
    pieces->AddOperation(std::make_shared<Circuits::CZGate<>>(5, 0));
    pieces->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {6, 1}, {3, 2}}));
    BOOST_TEST((Circuits::ChainLine(pieces, 7) == std::vector<long long int>{0, 2, 4, 6, 3, 1, 5}));

    // No layout without swaps, or none wanted: a third partner, a ring, three qubits at once, and a SWAP gate on a
    // backend that relabels its qubits there.
    auto star = std::make_shared<Circuits::Circuit<>>(pieces->GetOperations());
    star->AddOperation(std::make_shared<Circuits::CXGate<>>(2, 0));
    BOOST_TEST(Circuits::ChainLine(star, 7).empty());
    auto ring = std::make_shared<Circuits::Circuit<>>(pieces->GetOperations());
    ring->AddOperation(std::make_shared<Circuits::CXGate<>>(4, 6));
    BOOST_TEST(Circuits::ChainLine(ring, 7).empty());
    auto wide = std::make_shared<Circuits::Circuit<>>(pieces->GetOperations());
    wide->AddOperation(std::make_shared<Circuits::CCXGate<>>(0, 1, 3));
    BOOST_TEST(Circuits::ChainLine(wide, 7).empty());
    auto swapped = std::make_shared<Circuits::Circuit<>>(pieces->GetOperations());
    swapped->AddOperation(std::make_shared<Circuits::SwapGate<>>(4, 3));
    BOOST_TEST(!Circuits::ChainLine(swapped, 7).empty());
    BOOST_TEST(Circuits::ChainLine(swapped, 7, true).empty());
    // A register too small for the circuit, and one with nothing to place.
    BOOST_TEST(Circuits::ChainLine(pieces, 6).empty());
    BOOST_TEST(Circuits::ChainLine(std::make_shared<Circuits::Circuit<>>(), 7).empty());
}

// A backend that neither holds nor samples channels runs realizations in which
// every channel instruction has been drawn. That works for any channel whose
// branches need no state: a Pauli, a one-qubit unitary mixture, a relaxation
// with T2 <= T1 (nothing, a phase flip or a reset).
BOOST_AUTO_TEST_CASE(SampledInjectionDrawsEveryChannelThatNeedsNoState)
{
    using Channel = Simulators::QuantumChannel;
    using Type = Circuits::OperationType;

    // The U gate that stands for a drawn unitary is that unitary.
    std::mt19937 generator(5);
    std::normal_distribution<double> normal;
    std::vector<Eigen::MatrixXcd> unitaries;
    for (int i = 0; i < 40; ++i)
    {
        Eigen::MatrixXcd m(2, 2);
        for (Eigen::Index k = 0; k < 4; ++k)
            m.data()[k] = {normal(generator), normal(generator)};
        unitaries.push_back(Eigen::HouseholderQR<Eigen::MatrixXcd>(m).householderQ() * Eigen::MatrixXcd::Identity(2, 2));
    }
    Eigen::MatrixXcd special(2, 2);
    special << 1.0, 0.0, 0.0, std::complex<double>(0.0, 1.0);
    unitaries.push_back(special); // diagonal
    special << 0.0, std::complex<double>(0.0, -1.0), std::complex<double>(0.6, 0.8), 0.0;
    unitaries.push_back(special); // anti-diagonal
    unitaries.push_back(-Eigen::MatrixXcd::Identity(2, 2));
    for (const auto &unitary : unitaries)
    {
        const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(noise::unitary_gate_(0, unitary));
        BOOST_REQUIRE(gate);
        BOOST_TEST((gate->GetMatrix() - unitary).norm() < 1e-12);
    }

    // A Clifford is drawn as H and S gates, which a stabilizer backend runs:
    // the 24 of them, each equal to its word up to a global phase.
    BOOST_TEST(noise::one_qubit_cliffords_().size() == 24u);
    for (const auto &word : noise::one_qubit_cliffords_())
    {
        auto drawn = std::make_shared<Circuits::Circuit<>>();
        // With another global phase, which the word need not reproduce.
        noise::append_unitary_(drawn, 0, word.matrix * std::polar(1.0, 0.7));
        Eigen::MatrixXcd product = Eigen::MatrixXcd::Identity(2, 2);
        for (const auto &op : drawn->GetOperations())
        {
            const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(op);
            BOOST_REQUIRE(gate);
            BOOST_TEST((gate->GetGateType() == Circuits::QuantumGateType::kHadamardGateType || gate->GetGateType() == Circuits::QuantumGateType::kSGateType));
            product = gate->GetMatrix() * product;
        }
        BOOST_TEST(drawn->size() == word.gates.size());
        BOOST_TEST((noise::phase_free_(product) - word.matrix).norm() < 1e-9);
    }

    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1.0, 1.0, 1.0, -1.0;
    hadamard /= std::sqrt(2.0);
    const Channel relaxFast = Channel::ThermalRelaxation(1.0, 3.0, 2.0, 0.2), relaxEqual = Channel::ThermalRelaxation(0.8, 4.0, 4.0);
    const Channel gateError = Channel::TwoQubitDepolarizing(0.06).Then(relaxFast, {0}).Then(relaxEqual, {1});
    const Channel mixture = Channel::UnitaryMixture({0.7, 0.2, 0.1}, {Eigen::MatrixXcd::Identity(2, 2), hadamard, unitaries[0]});
    const auto channel = [](const Types::qubits_vector &qubits, const Channel &c) { return std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, c); };
    constexpr size_t qubits = 4;
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(1, 0.8));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    circuit->AddOperation(channel({1, 0}, gateError));
    circuit->AddOperation(std::make_shared<Circuits::RxGate<>>(2, 1.1));
    circuit->AddOperation(channel({2}, Channel::PhaseDamping(0.3)));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(1, 2));
    circuit->AddOperation(channel({2}, mixture));
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(3));
    circuit->AddOperation(channel({3}, relaxFast));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(2, 3));
    circuit->AddOperation(channel({3, 2}, gateError));
    circuit->AddOperation(channel({0}, Channel::Depolarizing(0.1)));
    const noise::NoiseModel empty;

    const Types::qubits_vector all{0, 1, 2, 3};
    auto exact = MakeQCSim(Simulators::SimulationType::kDensityMatrix, qubits);
    {
        Circuits::OperationState state;
        state.AllocateBits(qubits);
        noise::inject_exact_noise(circuit, empty)->Execute(exact, state);
    }
    const Eigen::MatrixXcd expected = exact->PartialTrace(all);

    constexpr size_t realizations = 20000;
    std::mt19937 rng(77);
    Eigen::MatrixXcd mean = Eigen::MatrixXcd::Zero(expected.rows(), expected.cols());
    auto statevector = MakeQCSim(Simulators::SimulationType::kStatevector, qubits);
    statevector->SetSeed(12);
    for (size_t r = 0; r < realizations; ++r)
    {
        const auto realization = noise::inject_noise(circuit, empty, rng);
        for (const auto &op : realization->GetOperations())
            BOOST_REQUIRE(op->GetType() == Type::kGate || op->GetType() == Type::kReset);
        statevector->Reset();
        Circuits::OperationState state;
        state.AllocateBits(qubits);
        realization->Execute(statevector, state);
        const auto amplitudes = statevector->GetStateVector();
        const Eigen::Map<const Eigen::VectorXcd> psi(amplitudes.data(), static_cast<Eigen::Index>(amplitudes.size()));
        mean += psi * psi.adjoint();
    }
    mean /= static_cast<double>(realizations);
    const double purity = (expected * expected).trace().real();
    const double error = (mean - expected).norm(), sampling = std::sqrt((1.0 - purity) / realizations);
    BOOST_TEST(purity < 0.8);
    BOOST_TEST(error < 2.0 * sampling, "sampled realizations are " << error << " from the channels, about " << sampling << " expected");

    // A part that needs the state has no such form.
    for (const Channel &needsState :
         {Channel::AmplitudeDamping(0.2), Channel::ThermalRelaxation(1.0, 4.0, 6.0), Channel::GeneralizedAmplitudeDamping(0.2, 0.1)})
    {
        auto rejected = std::make_shared<Circuits::Circuit<>>();
        rejected->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
        rejected->AddOperation(channel({0}, needsState));
        BOOST_CHECK_THROW(noise::inject_noise(rejected, empty, rng), std::invalid_argument);
    }
}

// Sampled realizations for a backend that samples channels keep the noise that
// has no exact sampled lowering as channel operations, and nothing else.
BOOST_AUTO_TEST_CASE(RuntimeChannelInjectionKeepsWhatCannotBeSampledExactly)
{
    using Channel = Simulators::QuantumChannel;
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    circuit->AddOperation(std::make_shared<Circuits::Delay<>>(2, 1.0));
    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Channel::AmplitudeDamping(0.4)));
    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{1}, Channel::Depolarizing(0.3)));

    noise::NoiseModel model;
    model.set_t1(0, 0.2);
    model.set_thermal_relaxation(1, 1.0, 4.0, 6.0); // T2 > T1: no exact reset mixture
    model.set_idle_noise(2, 4.0, 6.0);

    const auto channels = [](const std::shared_ptr<Circuits::Circuit<double>> &noisy) {
        std::vector<Simulators::QuantumChannel> found;
        for (const auto &op : noisy->GetOperations())
            if (op->GetType() == Circuits::OperationType::kQuantumChannel)
                found.push_back(std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(op)->GetChannel());
        return found;
    };

    std::mt19937 rng(5);
    const noise::NoiseInjectionPlan runtime(circuit, model, true);
    BOOST_TEST(runtime.RuntimeChannels());
    for (int realization = 0; realization < 20; ++realization)
    {
        // Damping after H and after CX on qubit 0, relaxation after CX on qubit 1
        // and on the delay, and the circuit's own damping. The circuit's
        // depolarizing channel is drawn, not kept.
        const auto kept = channels(runtime.Combined(rng, false));
        BOOST_REQUIRE_EQUAL(kept.size(), 5);
        for (const auto &channel : kept)
            BOOST_TEST(channel.GetPauliProbabilities().empty());
    }

    // Without such a backend the circuit's damping cannot be sampled at all.
    const noise::NoiseInjectionPlan sampled(circuit, model);
    BOOST_TEST(!sampled.RuntimeChannels());
    BOOST_CHECK_THROW(sampled.Combined(rng, false), std::invalid_argument);

    // Relaxation with T2 <= T1 keeps its exact reset mixture: no channel operation.
    auto gates = std::make_shared<Circuits::Circuit<double>>();
    gates->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    gates->AddOperation(std::make_shared<Circuits::XGate<>>(0));
    noise::NoiseModel slowDephasing;
    slowDephasing.set_thermal_relaxation(0, 1.0, 4.0, 3.0);
    const noise::NoiseInjectionPlan mixture(gates, slowDephasing, true);
    for (int realization = 0; realization < 20; ++realization)
        BOOST_TEST(channels(mixture.Combined(rng, false)).empty());

    // Exact injection is unchanged by the mode.
    BOOST_REQUIRE_EQUAL(channels(runtime.Combined(rng, true)).size(), 6);
}

struct QCSimDensityMatrixRandomCircuitsFixture
{
    QCSimDensityMatrixRandomCircuitsFixture()
    {
        qcsimSV = MakeQCSim(Simulators::SimulationType::kStatevector, nrQubitsForRandomCirc);
        qcsimDensity = MakeQCSim(Simulators::SimulationType::kDensityMatrix, nrQubitsForRandomCirc);

        circ = std::make_shared<Circuits::Circuit<>>();
        state.AllocateBits(nrQubitsForRandomCirc);

        resetRandomCirc = std::make_shared<Circuits::Circuit<>>();
        Types::qubits_vector qubits(nrQubitsForRandomCirc);
        std::iota(qubits.begin(), qubits.end(), 0);
        resetRandomCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));
    }

    void GenerateCircuit(int nrGates)
    {
        std::random_device rd;
        std::mt19937 g(rd());

        auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
        auto dblGenIter = dblGen.begin();

        auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
        auto gateGenIter = gateGen.begin();

        for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
        {
            Types::qubits_vector qubits(nrQubitsForRandomCirc);
            std::iota(qubits.begin(), qubits.end(), 0);
            std::shuffle(qubits.begin(), qubits.end(), g);
            auto q1 = qubits[0];
            auto q2 = qubits[1];
            auto q3 = qubits[2];

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

    const unsigned int nrQubitsForRandomCirc = 4;
    std::shared_ptr<Simulators::ISimulator> qcsimSV;
    std::shared_ptr<Simulators::ISimulator> qcsimDensity;

    std::shared_ptr<Circuits::Circuit<>> circ;
    std::shared_ptr<Circuits::Circuit<>> resetRandomCirc;
    Circuits::OperationState state;
};

BOOST_DATA_TEST_CASE_F(QCSimDensityMatrixRandomCircuitsFixture, RandomCircuitsTest, bdata::xrange(1, 20), nrGates)
{
    size_t nrStates = 1ULL << nrQubitsForRandomCirc;

    GenerateCircuit(nrGates);

    circ->Execute(qcsimSV, state);
    circ->Execute(qcsimDensity, state);

    auto svProbs = qcsimSV->AllProbabilities();
    BOOST_REQUIRE_EQUAL(svProbs.size(), nrStates);

    auto dmProbs = qcsimDensity->AllProbabilities();
    BOOST_REQUIRE_EQUAL(dmProbs.size(), nrStates);

    for (size_t stateIdx = 0; stateIdx < nrStates; ++stateIdx)
    {
        const auto psv = svProbs[stateIdx];
        const auto pdm = dmProbs[stateIdx];

        BOOST_TEST(std::abs(pdm - psv) < kQCSimDensityTolerance, "Probability mismatch for outcome " << stateIdx << ": expected " << psv << ", got " << pdm);
    }

    resetRandomCirc->Execute(qcsimDensity, state);
    resetRandomCirc->Execute(qcsimSV, state);

    circ->Clear();
    state.Reset();
}

BOOST_DATA_TEST_CASE_F(QCSimDensityMatrixRandomCircuitsFixture, SampleCountsManyTest, bdata::xrange(15, 30), nrGates)
{
    GenerateCircuit(nrGates);

    circ->Execute(qcsimSV, state);
    circ->Execute(qcsimDensity, state);

    std::random_device rd;
    std::mt19937 g(rd());

    Types::qubits_vector allQubits(nrQubitsForRandomCirc);
    std::iota(allQubits.begin(), allQubits.end(), 0);
    std::shuffle(allQubits.begin(), allQubits.end(), g);

    std::uniform_int_distribution<unsigned int> subsetSizeDist(1, nrQubitsForRandomCirc - 1);
    const unsigned int subsetSize = subsetSizeDist(g);

    Types::qubits_vector sampledQubits(allQubits.begin(), allQubits.begin() + subsetSize);

    const size_t shots = 10000;

    auto svCounts = qcsimSV->SampleCountsMany(sampledQubits, shots);
    auto dmCounts = qcsimDensity->SampleCountsMany(sampledQubits, shots);

    for (const auto &[outcome, cnt] : svCounts)
    {
        double svProb = static_cast<double>(cnt) / static_cast<double>(shots);
        if (svProb < 0.02)
            continue;

        double dmProb = 0;
        if (dmCounts.find(outcome) != dmCounts.end())
            dmProb = static_cast<double>(dmCounts[outcome]) / static_cast<double>(shots);

        BOOST_CHECK_CLOSE(svProb, dmProb, dmProb < 0.1 ? 66 : 33);
    }

    for (const auto &[outcome, cnt] : dmCounts)
    {
        double dmProb = static_cast<double>(cnt) / static_cast<double>(shots);
        if (dmProb < 0.02)
            continue;

        double svProb = 0;
        if (svCounts.find(outcome) != svCounts.end())
            svProb = static_cast<double>(svCounts[outcome]) / static_cast<double>(shots);

        BOOST_CHECK_CLOSE(dmProb, svProb, svProb < 0.1 ? 66 : 33);
    }

    resetRandomCirc->Execute(qcsimDensity, state);
    resetRandomCirc->Execute(qcsimSV, state);

    circ->Clear();
    state.Reset();
}

BOOST_AUTO_TEST_SUITE_END()
