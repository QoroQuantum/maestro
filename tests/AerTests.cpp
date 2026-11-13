/**
 * @file AerTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the Aer simulator.
 *
 * Just tests for simulator creation, qubits allocation, initialization, a
 * single X gate and measurements of all qubits.
 */

#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;

#include <framework/avx2_detect.hpp>

#include <cmath>
#include <vector>

#undef min
#undef max

#include "../Simulators/Core/Configuration.h"
#include "../Simulators/Core/Factory.h" // project being tested
#include "../python/Noise.h"

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

    auto res = aer->Measure({0, 1, 2});

    BOOST_TEST(res == 1ULL);

    BOOST_CHECK_CLOSE(aer->Probability(res), 1.0, 0.000001);
}

BOOST_AUTO_TEST_CASE(density_matrix_exact_quantum_channels)
{
    auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kDensityMatrix);
    densityMatrix->AllocateQubits(3);
    densityMatrix->Initialize();
    BOOST_TEST(densityMatrix->SupportsQuantumChannels());

    densityMatrix->ApplyH(0);
    densityMatrix->ApplyPhaseFlipNoise(0, 0.2);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XII") - 0.6, 1e-10);

    densityMatrix->Reset();
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyAmplitudeDamping(0, 0.36);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.32, 1e-10);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XII") - 0.8, 1e-10);

    densityMatrix->Reset();
    densityMatrix->ApplyGeneralizedAmplitudeDamping(0, 0.4, 0.25);
    BOOST_CHECK_SMALL(densityMatrix->Probability(1) - 0.1, 1e-10);

    densityMatrix->Reset();
    constexpr double duration = 0.7;
    constexpr double t1 = 2.0;
    constexpr double t2 = 1.5;
    constexpr double excitedPopulation = 0.1;
    densityMatrix->ApplyH(0);
    densityMatrix->ApplyThermalRelaxation(0, duration, t1, t2, excitedPopulation);
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("XII") - std::exp(-duration / t2), 1e-10);

    densityMatrix->Reset();
    densityMatrix->ApplyTwoQubitDepolarizingNoise(0, 2, 15.0 / 16.0);
    const auto fullyMixed = densityMatrix->AllProbabilities();
    for (size_t outcome = 0; outcome < fullyMixed.size(); ++outcome)
    {
        const double expected = (outcome & 2U) == 0 ? 0.25 : 0.0;
        BOOST_CHECK_SMALL(fullyMixed[outcome] - expected, 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(density_matrix_kraus_target_order_matches_maestro)
{
    auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kDensityMatrix);
    densityMatrix->AllocateQubits(3);
    densityMatrix->Initialize();

    // I (x) X flips local bit zero, which is targets[0] in Maestro and Aer.
    Eigen::MatrixXcd identityTensorX = Eigen::MatrixXcd::Zero(4, 4);
    identityTensorX(1, 0) = 1.0;
    identityTensorX(0, 1) = 1.0;
    identityTensorX(3, 2) = 1.0;
    identityTensorX(2, 3) = 1.0;
    densityMatrix->ApplyKrausChannel({0, 2}, {identityTensorX});
    BOOST_CHECK_CLOSE(densityMatrix->Probability(1), 1.0, 1e-8);

    // The generic-unitary and Kraus APIs use the same target convention.
    densityMatrix->Reset();
    densityMatrix->ApplyGenericTwoQubitGate(0, 2, identityTensorX);
    BOOST_CHECK_CLOSE(densityMatrix->Probability(1), 1.0, 1e-8);

    auto statevector = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);
    statevector->AllocateQubits(1);
    statevector->Initialize();
    // The statevector holds no ensemble: it takes one sampled branch of a channel.
    BOOST_TEST(!statevector->SupportsQuantumChannels());
    BOOST_TEST(statevector->SamplesQuantumChannels());
    statevector->ApplyBitFlipNoise(0, 1.0);
    BOOST_CHECK_CLOSE(statevector->Probability(1), 1.0, 1e-8);
}

// Aer's statevector and matrix product state take a channel as one sampled
// Kraus branch; averaged over trajectories they reproduce Aer's density matrix.
BOOST_AUTO_TEST_CASE(statevector_and_matrix_product_state_sample_channels_as_trajectories)
{
    using Channel = Simulators::QuantumChannel;
    using Method = Simulators::SimulationType;
    const Channel gateError =
        Channel::TwoQubitDepolarizing(0.05).Then(Channel::ThermalRelaxation(1.0, 4.0, 6.0), {0}).Then(Channel::ThermalRelaxation(1.0, 3.0, 2.0, 0.2), {1});
    const Channel damping = Channel::AmplitudeDamping(0.3);
    const Channel generalized = Channel::GeneralizedAmplitudeDamping(0.25, 0.3);
    const Channel phase = Channel::PhaseDamping(0.2);
    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1.0, 1.0, 1.0, -1.0;
    hadamard /= std::sqrt(2.0);
    const Channel mixture = Channel::UnitaryMixture({0.8, 0.2}, {Eigen::MatrixXcd::Identity(2, 2), hadamard});

    const auto evolve = [&](Simulators::ISimulator &sim) {
        sim.ApplyH(0);
        sim.ApplyCX(0, 1);
        sim.ApplyRy(1, 0.7);
        sim.ApplyCX(1, 2);
        sim.ApplyRx(2, 1.3);
        sim.ApplyQuantumChannel({1, 0}, gateError);
        sim.ApplyRy(0, 0.4);
        sim.ApplyQuantumChannel({2}, damping);
        sim.ApplyCX(2, 0);
        sim.ApplyQuantumChannel({0}, generalized);
        sim.ApplyQuantumChannel({1}, phase);
        sim.ApplyQuantumChannel({2}, mixture);
        sim.ApplyQuantumChannel({2, 1}, gateError);
    };
    const auto make = [](Method method) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
        sim->AllocateQubits(3);
        sim->Initialize();
        return sim;
    };
    const std::vector<std::string> observables{"ZII", "IZI", "IIZ", "XXI", "IXX", "YIY", "ZZZ", "XYZ"};

    auto exact = make(Method::kDensityMatrix);
    BOOST_TEST(!exact->SamplesQuantumChannels());
    evolve(*exact);

    for (const Method method : {Method::kStatevector, Method::kMatrixProductState})
    {
        constexpr size_t trajectories = 20000;
        auto sim = make(method);
        BOOST_TEST(sim->SamplesQuantumChannels());
        BOOST_TEST(!sim->SupportsQuantumChannels());
        BOOST_TEST(Simulators::SamplesQuantumChannels(Simulators::SimulatorType::kQiskitAer, method));

        sim->SetSeed(2027);
        std::vector<double> sums(observables.size(), 0.0), squares(observables.size(), 0.0);
        for (size_t trajectory = 0; trajectory < trajectories; ++trajectory)
        {
            sim->Reset();
            evolve(*sim);
            for (size_t i = 0; i < observables.size(); ++i)
            {
                const double value = sim->ExpectationValue(observables[i]);
                sums[i] += value;
                squares[i] += value * value;
            }
        }
        for (size_t i = 0; i < observables.size(); ++i)
        {
            const double mean = sums[i] / trajectories;
            const double error = std::sqrt(std::max(squares[i] / trajectories - mean * mean, 0.0) / trajectories);
            BOOST_TEST(std::abs(mean - exact->ExpectationValue(observables[i])) < 5.0 * error + 1e-9,
                       observables[i] << " on Aer method " << static_cast<int>(method) << ": trajectories give " << mean << " +- " << error
                                      << ", the density matrix " << exact->ExpectationValue(observables[i]));
        }

        // The same seed replays the same trajectory.
        const auto replay = [&] {
            sim->SetSeed(77);
            sim->Reset();
            evolve(*sim);
            return sim->ExpectationValue(observables[3]);
        };
        const double first = replay();
        BOOST_CHECK_SMALL(replay() - first, 1e-12);

        BOOST_CHECK_THROW(sim->ApplyQuantumChannel({0, 1}, damping), std::invalid_argument);
        BOOST_CHECK_THROW(sim->ApplyQuantumChannel({0, 0}, gateError), std::invalid_argument);
        BOOST_CHECK_THROW(sim->ApplyQuantumChannel({3}, damping), std::invalid_argument);
    }

    // Aer's stabilizer has no Kraus instruction: it neither holds nor samples channels.
    auto stabilizer = make(Method::kStabilizer);
    BOOST_TEST(!stabilizer->SamplesQuantumChannels());
    BOOST_TEST(!Simulators::SamplesQuantumChannels(Simulators::SimulatorType::kQiskitAer, Method::kStabilizer));
    BOOST_CHECK_THROW(stabilizer->ApplyBitFlipNoise(0, 0.1), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(density_matrix_executes_exact_noise_model_circuit)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    noise::NoiseModel noiseModel;
    noiseModel.set_t1(0, 0.75);
    noiseModel.set_phase_damping(0, 0.36);
    const auto noisyCircuit = noise::inject_exact_noise(circuit, noiseModel);

    auto densityMatrix = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kDensityMatrix);
    densityMatrix->AllocateQubits(1);
    densityMatrix->Initialize();
    Circuits::OperationState classicalState;
    noisyCircuit->Execute(densityMatrix, classicalState);
    // AD contributes sqrt(1-0.75)=0.5 and phase damping contributes
    // sqrt(1-0.36)=0.8.
    BOOST_CHECK_SMALL(densityMatrix->ExpectationValue("X") - 0.4, 1e-10);
}

BOOST_AUTO_TEST_CASE(matrix_product_state_truncation_mode_only_accepts_discarded_weight)
{
    auto mps = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
    mps->AllocateQubits(1);

    // Aer's own MPS/MPO truncation always implements the discarded-weight
    // (Aer/iTensor) convention natively -- unlike the QCSim and GPU backends,
    // there is no relative-to-max mode to switch to, so it should be a no-op
    // to (re)confirm the only mode it has...
    BOOST_CHECK_NO_THROW(mps->Configure("matrix_product_state_truncation_mode", "discarded_weight"));
    BOOST_CHECK_NO_THROW(mps->Configure("matrix_product_operator_truncation_mode", "discarded_weight"));

    // ...but should reject a request to switch to the relative-to-max mode
    // that the QCSim/GPU backends support, rather than silently ignoring it.
    BOOST_CHECK_THROW(mps->Configure("matrix_product_state_truncation_mode", "relative_max"), std::invalid_argument);
    BOOST_CHECK_THROW(mps->Configure("matrix_product_operator_truncation_mode", "relative_max"), std::invalid_argument);
}

// Regression test for a real bug found during review: Configure() used to store the
// requested value into `configuration` BEFORE validating it, so a rejected
// "relative_max" request would throw as intended but still leave "relative_max"
// sitting in the configuration map -- visible via GetConfiguration(), and liable to
// throw again, unexpectedly, from an unrelated later call site (e.g. Clone()'s
// generic configuration-replay loop). Fixed by validating before storing.
BOOST_AUTO_TEST_CASE(rejected_truncation_mode_does_not_linger_in_configuration)
{
    auto mps = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
    mps->AllocateQubits(1);

    BOOST_CHECK_THROW(mps->Configure("matrix_product_state_truncation_mode", "relative_max"), std::invalid_argument);

    // The rejected value must not have been persisted -- GetConfiguration should report
    // the unset/default state, not the value that was just thrown out.
    BOOST_TEST(mps->GetConfiguration("matrix_product_state_truncation_mode") != std::string("relative_max"));

    // A rejected call must not corrupt Configure()'s ability to accept the one value
    // Aer actually supports right afterward.
    BOOST_CHECK_NO_THROW(mps->Configure("matrix_product_state_truncation_mode", "discarded_weight"));
}

BOOST_AUTO_TEST_CASE(unsupported_shared_settings_are_ignored)
{
    for (const auto method : {Simulators::SimulationType::kStatevector, Simulators::SimulationType::kMatrixProductState,
                              Simulators::SimulationType::kStabilizer, Simulators::SimulationType::kDensityMatrix})
    {
        auto aer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
        BOOST_TEST_CONTEXT("method=" << aer->GetConfiguration("method"))
        {
            aer->AllocateQubits(2);
            for (const bool initialized : {false, true})
            {
                if (initialized)
                    aer->Initialize();
                for (const char *enabled : {"true", "false"})
                {
                    Simulators::Configuration config;
                    config.SetConfiguration("enable_causal_cone_reduction", enabled);
                    config.SetConfiguration("causal_cone_statevector_threshold", "20");
                    config.SetConfiguration("fusion_enable", "false");
                    BOOST_CHECK_NO_THROW(config.ApplyConfigurationToSimulator(aer));
                    BOOST_TEST(aer->GetConfiguration("fusion_enable") == "false");
                    BOOST_TEST(aer->GetConfigMap().count("enable_causal_cone_reduction") == 0U);
                    BOOST_TEST(aer->GetConfigMap().count("causal_cone_statevector_threshold") == 0U);
                }
            }

            aer->ApplyH(0);
            aer->ApplyCX(0, 1);
            BOOST_CHECK_SMALL(aer->Probability(0) - 0.5, 1e-10);
            BOOST_CHECK_SMALL(aer->Probability(3) - 0.5, 1e-10);
            auto clone = aer->Clone();
            BOOST_CHECK_SMALL(clone->Probability(0) - 0.5, 1e-10);
            BOOST_CHECK_SMALL(clone->Probability(3) - 0.5, 1e-10);
        }
    }
}

BOOST_AUTO_TEST_CASE(aer_rejected_values_preserve_accepted_configuration)
{
    auto aer = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
    aer->Configure("matrix_product_state_max_bond_dimension", "8");
    aer->Configure("matrix_product_state_truncation_threshold", "0.01");
    aer->Configure("precision", "double");

    for (const char *value : {"invalid", "999999999999999999999999999999"})
    {
        BOOST_CHECK_NO_THROW(aer->Configure("matrix_product_state_max_bond_dimension", value));
        BOOST_TEST(aer->GetConfiguration("matrix_product_state_max_bond_dimension") == "8");
    }
    BOOST_CHECK_NO_THROW(aer->Configure("matrix_product_state_truncation_threshold", "invalid"));
    BOOST_TEST(aer->GetConfiguration("matrix_product_state_truncation_threshold") == "0.01");
    BOOST_CHECK_THROW(aer->Configure("precision", "invalid"), std::invalid_argument);
    BOOST_TEST(aer->GetConfiguration("precision") == "double");
    BOOST_CHECK_NO_THROW(aer->Configure("method", "invalid"));
    BOOST_TEST(aer->GetConfiguration("method") == "matrix_product_state");

    aer->Configure("matrix_product_state_max_bond_dimension", "4");
    aer->AllocateQubits(1);
    aer->Initialize();
    // Aer also rejects otherwise valid method changes after initialization.
    BOOST_CHECK_NO_THROW(aer->Configure("method", "statevector"));
    BOOST_TEST(aer->GetConfiguration("method") == "matrix_product_state");
    aer->ApplyX(0);
    auto clone = aer->Clone();
    BOOST_TEST(clone->GetConfiguration("matrix_product_state_max_bond_dimension") == "4");
    BOOST_TEST(clone->GetConfiguration("method") == "matrix_product_state");
    BOOST_CHECK_SMALL(clone->Probability(1) - 1.0, 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
