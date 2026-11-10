/**
 * @file NoiseModelTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for noise::NoiseModel: that a single configuration produces the same
 * physics on the exact (density-matrix / MPO) and sampled (pure-state)
 * backends, and that invalid parameters are rejected at configuration time
 * on both paths.
 */

#include <boost/test/unit_test.hpp>

#undef min
#undef max

#include <atomic>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Noise/NoiseAdd.h"
#include "../Simulators/Core/Configuration.h"
#include "../Simulators/Core/Factory.h"
#include "../python/Noise.h"

namespace
{

constexpr double kTolerance = 1e-10;

std::shared_ptr<Simulators::ISimulator> MakeSimulator(Simulators::SimulationType simulationType, size_t numQubits)
{
    auto simulator = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simulationType);
    simulator->AllocateQubits(numQubits);
    simulator->Initialize();
    return simulator;
}

/** A one-qubit circuit that puts the qubit on the equator, so <X> == 1. */
std::shared_ptr<Circuits::Circuit<double>> HadamardCircuit()
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    return circuit;
}

/**
 * <X> after running the noisy circuit on the density-matrix backend.
 * With the H first, this is exactly the coherence multiplier of whatever
 * noise the model injected after it.
 */
double ExactCoherence(const noise::NoiseModel &noiseModel)
{
    const auto noisy = noise::inject_exact_noise(HadamardCircuit(), noiseModel);
    auto simulator = MakeSimulator(Simulators::SimulationType::kDensityMatrix, 1);
    Circuits::OperationState classicalState;
    noisy->Execute(simulator, classicalState);
    return simulator->ExpectationValue("X");
}

/**
 * <X> averaged over sampled trajectories on the statevector backend.
 *
 * Each realization is a separate circuit rewrite, so the ensemble average
 * over many realizations reconstructs the channel the sampled path
 * represents.
 */
double SampledCoherence(const noise::NoiseModel &noiseModel, size_t realizations, unsigned int seed)
{
    std::mt19937 rng(seed);
    const auto circuit = HadamardCircuit();
    double total = 0.0;
    for (size_t realization = 0; realization < realizations; ++realization)
    {
        const auto noisy = noise::inject_noise(circuit, noiseModel, rng);
        auto simulator = MakeSimulator(Simulators::SimulationType::kStatevector, 1);
        Circuits::OperationState classicalState;
        noisy->Execute(simulator, classicalState);
        total += simulator->ExpectationValue("X");
    }
    return total / static_cast<double>(realizations);
}

} // namespace

BOOST_AUTO_TEST_SUITE(noise_model_tests)

/**
 * The regression this file exists for.
 *
 * set_thermal_relaxation() must give the same coherence decay exp(-t/T2) on
 * both backend families. The exact path applies the CPTP channel; the sampled
 * path applies the reset+Z mixture. Configuring T1 and T2 separately cannot
 * do this, which is what the second half of the test pins down.
 */
BOOST_AUTO_TEST_CASE(ThermalRelaxationAgreesOnExactAndSampledBackends)
{
    constexpr double duration = 4e-6;
    constexpr double t1 = 40e-6;
    constexpr double t2 = 25e-6; // T2 <= T1, where the reset mixture is exact
    const double expected = std::exp(-duration / t2);

    noise::NoiseModel noiseModel;
    noiseModel.set_thermal_relaxation(0, duration, t1, t2);

    BOOST_TEST(noiseModel.has_thermal_relaxation());
    // Thermal relaxation must no longer force an exact backend.
    BOOST_TEST(!noiseModel.has_additional_quantum_channels());

    BOOST_CHECK_SMALL(ExactCoherence(noiseModel) - expected, kTolerance);

    // The sampled path is stochastic, so it only converges to the same channel.
    const double sampled = SampledCoherence(noiseModel, 4000, 12345);
    BOOST_CHECK_SMALL(sampled - expected, 0.02);

    // Contrast: T1 and dephasing configured separately, with the phase-flip
    // probability calibrated for the reset model. The exact backend applies
    // amplitude damping instead, so it under-dephases by exp(t/2*T1).
    const double gamma = -std::expm1(-duration / t1);
    const double pz = 0.5 * (1.0 - std::exp(-duration * (1.0 / t2 - 1.0 / t1)));
    noise::NoiseModel legacy;
    legacy.set_t1(0, gamma);
    legacy.set_dephasing(0, pz);

    const double legacyExact = ExactCoherence(legacy);
    BOOST_TEST(legacyExact > expected, "the legacy split configuration under-dephases on the exact "
                                       "backend, which is why set_thermal_relaxation exists");
    BOOST_CHECK_SMALL(legacyExact - std::sqrt(1.0 - gamma) * (1.0 - 2.0 * pz), kTolerance);
}

/** Two-qubit gates use the 2Q relaxation when one is configured. */
BOOST_AUTO_TEST_CASE(ThermalRelaxation2QOverridesTheAllGatesChannel)
{
    constexpr double t1 = 50e-6;
    constexpr double t2 = 30e-6;

    noise::NoiseModel noiseModel;
    noiseModel.set_thermal_relaxation(0, 2e-8, t1, t2);
    noiseModel.set_thermal_relaxation_2q(0, 4e-7, t1, t2);

    const auto *oneQubit = noiseModel.get_thermal_relaxation_params(0, /*is_2q=*/false);
    const auto *twoQubit = noiseModel.get_thermal_relaxation_params(0, /*is_2q=*/true);
    BOOST_REQUIRE(oneQubit);
    BOOST_REQUIRE(twoQubit);
    BOOST_CHECK_CLOSE(oneQubit->duration, 2e-8, 1e-9);
    BOOST_CHECK_CLOSE(twoQubit->duration, 4e-7, 1e-9);

    // A qubit with no 2Q override falls back to the "all gates" relaxation.
    noise::NoiseModel fallback;
    fallback.set_thermal_relaxation(1, 2e-8, t1, t2);
    const auto *fallbackParams = fallback.get_thermal_relaxation_params(1, /*is_2q=*/true);
    BOOST_REQUIRE(fallbackParams);
    BOOST_CHECK_CLOSE(fallbackParams->duration, 2e-8, 1e-9);
}

/**
 * Phase damping and the stochastic phase flip are the same channel, so the
 * sampled backends realize it exactly instead of rejecting the model.
 */
BOOST_AUTO_TEST_CASE(PhaseDampingIsExactOnBothBackends)
{
    constexpr double gamma = 0.36;
    const double expected = std::sqrt(1.0 - gamma); // 0.8

    noise::NoiseModel noiseModel;
    noiseModel.set_phase_damping(0, gamma);
    BOOST_TEST(!noiseModel.has_additional_quantum_channels());
    BOOST_CHECK_SMALL(noiseModel.get_phase_damping_flip_probability(0) - 0.1, kTolerance);

    BOOST_CHECK_SMALL(ExactCoherence(noiseModel) - expected, kTolerance);
    BOOST_CHECK_SMALL(SampledCoherence(noiseModel, 4000, 999) - expected, 0.02);
}

/** Channels with no stochastic realization still require an exact backend. */
BOOST_AUTO_TEST_CASE(ExactOnlyChannelsStillRejectSampledBackends)
{
    noise::NoiseModel noiseModel;
    noiseModel.set_generalized_amplitude_damping(0, 0.2, 0.1);
    BOOST_TEST(noiseModel.has_additional_quantum_channels());

    std::mt19937 rng(1);
    BOOST_CHECK_THROW(noise::inject_noise(HadamardCircuit(), noiseModel, rng), std::invalid_argument);
    BOOST_CHECK_NO_THROW(noise::inject_exact_noise(HadamardCircuit(), noiseModel));
}

/**
 * Invalid probabilities must be rejected where they are configured. The
 * sampled injectors cannot detect them, so without this they would silently
 * skew the sampling while the exact path threw.
 */
BOOST_AUTO_TEST_CASE(SettersRejectUnphysicalParameters)
{
    noise::NoiseModel noiseModel;

    BOOST_CHECK_THROW(noiseModel.set_depolarizing(0, 1.5), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_depolarizing(0, -0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_dephasing(0, 2.0), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_bit_flip(0, -1.0), std::invalid_argument);
    // Pauli probabilities must also sum to at most one.
    BOOST_CHECK_THROW(noiseModel.set_qubit_noise(0, 0.5, 0.4, 0.3), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_t1(0, 1.2), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_2q_depolarizing(0, 1, 1.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_2q_depolarizing(0, 0, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_readout_error(0, 0.1, 1.4), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_coherent_depolarizing(0, 1.3), std::invalid_argument);
    // T2 > 2*T1 is not a completely positive map.
    BOOST_CHECK_THROW(noiseModel.set_thermal_relaxation(0, 1e-6, 10e-6, 30e-6), std::invalid_argument);
    // Degenerate OU parameters used to produce silent inf/nan.
    BOOST_CHECK_THROW(noiseModel.set_correlated_ou(0, 1.0, 0.0, 1e-7), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_all_correlated_from_power(0, 1e-3, 0.5, 1e-7), std::invalid_argument);

    // Nothing above should have been recorded.
    BOOST_TEST(!noiseModel.has_any());

    // Valid values still work.
    BOOST_CHECK_NO_THROW(noiseModel.set_depolarizing(0, 0.01));
    BOOST_TEST(noiseModel.has_any());
}

/**
 * The coherent over/under-rotation sign is a property of the device, so it
 * must be the same for every gate within one realization -- that is what
 * makes the error systematic rather than a random walk.
 */
BOOST_AUTO_TEST_CASE(CoherentNoiseSignIsSystematicWithinARealization)
{
    constexpr double angle = 0.2;
    noise::NoiseModel noiseModel;
    noiseModel.set_coherent_rotation(0, 0.0, 0.0, angle);

    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    constexpr size_t gateCount = 8;
    for (size_t i = 0; i < gateCount; ++i)
        circuit->AddOperation(std::make_shared<Circuits::ZGate<>>(0));

    std::mt19937 rng(7);
    for (size_t trial = 0; trial < 20; ++trial)
    {
        const auto noisy = noise::inject_coherent_noise(circuit, noiseModel, rng);

        std::vector<double> angles;
        for (const auto &op : noisy->GetOperations())
        {
            const auto rz = std::dynamic_pointer_cast<Circuits::RzGate<>>(op);
            if (rz)
                angles.push_back(rz->GetTheta());
        }

        BOOST_REQUIRE_EQUAL(angles.size(), gateCount);
        for (const double injected : angles)
            BOOST_TEST(injected == angles.front(), "every injected rotation in a realization must share the "
                                                   "same sign");
        BOOST_CHECK_SMALL(std::abs(angles.front()) - angle, kTolerance);
    }
}

/** Parameters the injectors cannot use are rejected rather than ignored. */
BOOST_AUTO_TEST_CASE(SettersRejectInvalidQubitsAndNonFiniteParameters)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    noise::NoiseModel noiseModel;

    // AR(1) is stationary only for |phi| < 1, as the JSON API requires.
    BOOST_CHECK_THROW(noiseModel.set_correlated_ar1(0, 1.0, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_ar1(0, -1.5, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_ar1(0, 0.5, -0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_ar1(0, nan, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_ar1(0, 0.5, inf), std::invalid_argument);

    BOOST_CHECK_THROW(noiseModel.set_coherent_rotation(0, nan, 0.0, 0.0), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_coherent_rotation(0, 0.0, inf, 0.0), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_coherent_rotation(0, 0.0, 0.0, -inf), std::invalid_argument);

    // A qubit is never its own spectator, so self-crosstalk would do nothing.
    BOOST_CHECK_THROW(noiseModel.set_crosstalk(1, 1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_crosstalk(0, 1, nan), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_crosstalk(0, 1, inf), std::invalid_argument);

    BOOST_CHECK_THROW(noiseModel.set_multi_correlated_ou(0, {}, 1e-7), std::invalid_argument);

    // Negative qubit keys never match a circuit qubit.
    BOOST_CHECK_THROW(noiseModel.set_depolarizing(-1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_qubit_noise(-1, 0.1, 0.0, 0.0), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_coherent_depolarizing(-1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_coherent_rotation(-1, 0.1, 0.0, 0.0), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_ou(-1, 1.0, 0.5, 1e-7), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_ar1(-1, 0.5, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_t1(-1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_t1_2q_from_time(-1, 1e-7, 1e-5), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_phase_damping(-1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_generalized_amplitude_damping(-1, 0.1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_thermal_relaxation(-1, 1e-6, 40e-6, 25e-6), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_thermal_relaxation_2q(-1, 1e-6, 40e-6, 25e-6), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_idle_noise(-1, 40e-6, 25e-6), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_readout_error(-1, 0.1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_1q_gate_depolarizing(-1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_2q_gate_depolarizing(-1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_2q_depolarizing(-1, 0, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_correlated_phase_flip(0, -1, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(noiseModel.set_crosstalk(-1, 0, 0.1), std::invalid_argument);

    // A nonpositive count would silently configure nothing.
    for (const int count : {0, -2})
    {
        BOOST_CHECK_THROW(noiseModel.set_all_depolarizing(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_dephasing(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_coherent_depolarizing(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_coherent_dephasing(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_correlated_ou(count, 1.0, 0.5, 1e-7), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_t1(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_readout_error(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_1q_gate_depolarizing(count, 0.1), std::invalid_argument);
        BOOST_CHECK_THROW(noiseModel.set_all_2q_gate_depolarizing(count, 0.1), std::invalid_argument);
    }

    BOOST_TEST(!noiseModel.has_any());
}

BOOST_AUTO_TEST_CASE(T1AndThermalRelaxationCannotStack)
{
    noise::NoiseModel noiseModel;
    noiseModel.set_t1(0, 0.1);
    BOOST_CHECK_THROW(noiseModel.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6), std::invalid_argument);

    noise::NoiseModel thermalFirst;
    thermalFirst.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6);
    BOOST_CHECK_THROW(thermalFirst.set_t1(0, 0.1), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(RejectedT1SettersLeaveTheModelUnchanged)
{
    struct Setter
    {
        const char *name;
        void (*apply)(noise::NoiseModel &);
    };

    const Setter setters[] = {
        {"set_t1", [](noise::NoiseModel &model) { model.set_t1(2, 0.1); }},
        {"set_t1_from_time", [](noise::NoiseModel &model) { model.set_t1_from_time(2, 1e-6, 40e-6); }},
        {"set_t1_2q", [](noise::NoiseModel &model) { model.set_t1_2q(2, 0.1); }},
        {"set_t1_2q_from_time", [](noise::NoiseModel &model) { model.set_t1_2q_from_time(2, 1e-6, 40e-6); }},
        {"set_all_t1", [](noise::NoiseModel &model) { model.set_all_t1(3, 0.1); }},
    };

    for (const auto &setter : setters)
    {
        for (const bool existing_t1 : {false, true})
        {
            BOOST_TEST_CONTEXT(setter.name << ", existing T1=" << existing_t1)
            {
                noise::NoiseModel model;
                model.set_thermal_relaxation(2, 1e-6, 40e-6, 25e-6);
                if (existing_t1)
                {
                    model.set_t1(0, 0.2);
                    model.set_t1(1, 0.3);
                    model.set_t1(2, 0.0);
                    model.set_t1_2q(2, 0.0);
                }

                BOOST_CHECK_THROW(setter.apply(model), std::invalid_argument);

                BOOST_TEST(model.has_t1() == existing_t1);
                // The batch setter encounters the conflict last, after qubits 0 and 1.
                BOOST_TEST(model.get_t1(0) == (existing_t1 ? 0.2 : 0.0));
                BOOST_TEST(model.get_t1(1) == (existing_t1 ? 0.3 : 0.0));
                BOOST_TEST(model.get_t1(2) == 0.0);
                BOOST_TEST(model.get_t1_2q(2) == 0.0);
                BOOST_REQUIRE(model.get_thermal_relaxation_params(2, false));
                BOOST_TEST(model.get_thermal_relaxation_params(2, false)->duration == 1e-6);
                BOOST_CHECK_NO_THROW(model.EnsureNoT1ThermalStack());
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(RejectedThermalSettersLeaveTheModelUnchanged)
{
    for (const bool t1_only_after_2q : {false, true})
    {
        for (const bool thermal_only_after_2q : {false, true})
        {
            BOOST_TEST_CONTEXT("2Q T1=" << t1_only_after_2q << ", 2Q thermal=" << thermal_only_after_2q)
            {
                noise::NoiseModel model;
                model.set_thermal_relaxation(0, 2e-6, 40e-6, 25e-6);
                if (t1_only_after_2q)
                    model.set_t1_2q(1, 0.2);
                else
                    model.set_t1(1, 0.2);

                if (thermal_only_after_2q)
                    BOOST_CHECK_THROW(model.set_thermal_relaxation_2q(1, 1e-6, 40e-6, 25e-6), std::invalid_argument);
                else
                    BOOST_CHECK_THROW(model.set_thermal_relaxation(1, 1e-6, 40e-6, 25e-6), std::invalid_argument);

                BOOST_TEST(model.get_t1(1) == (t1_only_after_2q ? 0.0 : 0.2));
                BOOST_TEST(model.get_t1_2q(1) == 0.2);
                BOOST_CHECK(model.get_thermal_relaxation_for_gate(1, false) == nullptr);
                BOOST_CHECK(model.get_thermal_relaxation_for_gate(1, true) == nullptr);
                BOOST_CHECK(model.get_thermal_relaxation_params(1, false) == nullptr);
                BOOST_CHECK(model.get_thermal_relaxation_params(1, true) == nullptr);
                BOOST_REQUIRE(model.get_thermal_relaxation_params(0, false));
                BOOST_TEST(model.get_thermal_relaxation_params(0, false)->duration == 2e-6);
                BOOST_CHECK_NO_THROW(model.EnsureNoT1ThermalStack());
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(T1ThermalValidationPreservesZeroGammaAndGateOverrides)
{
    // Without a 2Q override, all-gates T1 also conflicts with 2Q-only thermal noise.
    noise::NoiseModel fallback;
    fallback.set_thermal_relaxation_2q(0, 1e-6, 40e-6, 25e-6);
    BOOST_CHECK_THROW(fallback.set_t1(0, 0.1), std::invalid_argument);
    BOOST_CHECK_THROW(fallback.set_t1_from_time(0, 1e-6, 40e-6), std::invalid_argument);
    BOOST_CHECK_THROW(fallback.set_all_t1(1, 0.1), std::invalid_argument);
    BOOST_TEST(!fallback.has_t1());

    // Explicit zero T1 channels can coexist with thermal relaxation in either order.
    for (const bool thermal_first : {false, true})
    {
        noise::NoiseModel model;
        if (thermal_first)
            model.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6);
        BOOST_CHECK_NO_THROW(model.set_t1(0, 0.0));
        BOOST_CHECK_NO_THROW(model.set_t1_from_time(0, 0.0, 40e-6));
        BOOST_CHECK_NO_THROW(model.set_t1_2q(0, 0.0));
        BOOST_CHECK_NO_THROW(model.set_t1_2q_from_time(0, 1e-6, std::numeric_limits<double>::infinity()));
        BOOST_CHECK_NO_THROW(model.set_all_t1(1, 0.0));
        if (!thermal_first)
            BOOST_CHECK_NO_THROW(model.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6));
        BOOST_CHECK_NO_THROW(model.set_thermal_relaxation_2q(0, 2e-6, 40e-6, 25e-6));
        BOOST_CHECK_NO_THROW(model.EnsureNoT1ThermalStack());
    }

    // Zero 2Q T1 overrides allow nonzero 1Q T1 alongside a 2Q thermal channel.
    for (const bool thermal_first : {false, true})
    {
        noise::NoiseModel model;
        model.set_t1_2q(0, 0.0);
        if (thermal_first)
            model.set_thermal_relaxation_2q(0, 1e-6, 40e-6, 25e-6);
        BOOST_CHECK_NO_THROW(model.set_t1(0, 0.1));
        BOOST_CHECK_NO_THROW(model.set_t1_from_time(0, 1e-6, 40e-6));
        BOOST_CHECK_NO_THROW(model.set_all_t1(1, 0.2));
        if (!thermal_first)
            BOOST_CHECK_NO_THROW(model.set_thermal_relaxation_2q(0, 1e-6, 40e-6, 25e-6));
        BOOST_TEST(model.get_t1_for_gate(0, false) == 0.2);
        BOOST_TEST(model.get_t1_for_gate(0, true) == 0.0);
        BOOST_CHECK_THROW(model.set_t1_2q(0, 0.3), std::invalid_argument);
        BOOST_TEST(model.get_t1_for_gate(0, true) == 0.0);
        BOOST_CHECK_NO_THROW(model.EnsureNoT1ThermalStack());
    }
}

/**
 * The body-diagonal (coherent depolarizing) rotation and the per-axis
 * rotations are separate miscalibrations, so their signs are drawn
 * independently. With shared draws the Rx sign would always equal the sign
 * of the diagonal's x component.
 */
BOOST_AUTO_TEST_CASE(DiagonalAndAxisCoherentSignsAreIndependent)
{
    noise::NoiseModel noiseModel;
    noiseModel.set_coherent_depolarizing(0, 0.05);
    noiseModel.set_coherent_rotation(0, 0.1, 0.0, 0.0);

    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));

    std::mt19937 rng(11);
    constexpr int realizations = 2000;
    int agreements = 0;
    for (int realization = 0; realization < realizations; ++realization)
    {
        const auto noisy = noise::inject_coherent_noise(circuit, noiseModel, rng);
        BOOST_REQUIRE_EQUAL(noisy->GetOperations().size(), 3u);
        const auto diagonal = std::dynamic_pointer_cast<Circuits::UGate<>>(noisy->GetOperations()[1]);
        const auto rx = std::dynamic_pointer_cast<Circuits::RxGate<>>(noisy->GetOperations()[2]);
        BOOST_REQUIRE(diagonal && rx);
        // diagonal_rotation_gate_ sets lambda + gamma = atan2(sx, sy), so its
        // sine carries the sign of the diagonal's x component.
        const bool diagonalX = std::sin(diagonal->GetLambda() + diagonal->GetGamma()) > 0.0;
        const bool rotationX = rx->GetTheta() > 0.0;
        agreements += diagonalX == rotationX;
    }
    BOOST_CHECK_GT(agreements, 0.4 * realizations);
    BOOST_CHECK_LT(agreements, 0.6 * realizations);
}

/**
 * Three-qubit gates take the per-qubit 2Q gate-type layers (Pauli, T1,
 * thermal), as correlated noise's after_2q does, and never the 1Q-gate ones.
 */
BOOST_AUTO_TEST_CASE(ThreeQubitGatesUseTheMultiQubitGateLayers)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::CCXGate<>>(0, 1, 2));

    noise::NoiseModel noiseModel;
    noiseModel.set_t1_2q(0, 1.0);
    noiseModel.set_2q_gate_depolarizing(1, 1.0);
    noiseModel.set_thermal_relaxation_2q(2, 1e-6, 40e-6, 25e-6);
    // 1Q-gate noise and the all-gates T1 must not apply after a CCX.
    noiseModel.set_1q_gate_depolarizing(0, 1.0);
    noiseModel.set_1q_gate_depolarizing(2, 1.0);
    noiseModel.set_t1(1, 1.0);
    noiseModel.set_t1_2q(1, 0.0);

    const auto channelsOn = [](const std::shared_ptr<Circuits::Circuit<double>> &noisy, Types::qubit_t qubit) {
        size_t count = 0;
        for (const auto &op : noisy->GetOperations())
            if (op->GetType() == Circuits::OperationType::kQuantumChannel && op->AffectedQubits() == Types::qubits_vector{qubit})
                ++count;
        return count;
    };

    std::mt19937 rng(5);
    for (const auto &exact : {noise::inject_exact_noise(circuit, noiseModel), noise::inject_combined_noise_exact(circuit, noiseModel, rng)})
    {
        BOOST_CHECK_EQUAL(channelsOn(exact, 0), 1u); // 2Q T1 only
        BOOST_CHECK_EQUAL(channelsOn(exact, 1), 1u); // 2Q-gate Pauli only
        BOOST_CHECK_EQUAL(channelsOn(exact, 2), 1u); // 2Q thermal only
    }

    // On the sampled path the 2Q T1 of gamma=1 always resets qubit 0.
    const auto sampled = noise::inject_noise(circuit, noiseModel, rng);
    bool resetQubit0 = false;
    for (const auto &op : sampled->GetOperations())
        resetQubit0 = resetQubit0 || (op->GetType() == Circuits::OperationType::kReset && op->AffectedQubits() == Types::qubits_vector{0});
    BOOST_TEST(resetQubit0);
}

BOOST_AUTO_TEST_CASE(ThermalT2GreaterThanT1UsesSampledApproximation)
{
    constexpr double duration = 1e-6;
    constexpr double t1 = 10e-6;
    constexpr double t2 = 15e-6; // T1 < T2 <= 2*T1
    noise::NoiseModel noiseModel;
    noiseModel.set_thermal_relaxation(0, duration, t1, t2);

    BOOST_TEST(!noiseModel.has_additional_quantum_channels());
    BOOST_TEST(noiseModel.has_thermal_in_sampled_overdephasing_regime());
    BOOST_TEST(noiseModel.requires_exact_quantum_channels());
    BOOST_TEST(!noiseModel.compute_damping_covers_model());

    std::mt19937 rng(1);
    BOOST_CHECK_NO_THROW(noise::inject_noise(HadamardCircuit(), noiseModel, rng));
    noise::NoiseModel clamped;
    clamped.set_thermal_relaxation(0, duration, t1, t1);
    BOOST_CHECK_SMALL(SampledCoherence(noiseModel, 2000, 42) - SampledCoherence(clamped, 2000, 42), kTolerance);
    BOOST_TEST(noiseModel.get_thermal_relaxation_params(0, false)->t2 == t2);
    BOOST_CHECK_NO_THROW(noise::inject_exact_noise(HadamardCircuit(), noiseModel));

    const double expected = std::exp(-duration / t2);
    BOOST_CHECK_SMALL(ExactCoherence(noiseModel) - expected, kTolerance);
}

BOOST_AUTO_TEST_CASE(ComputeDampingDoesNotCoverThermalModels)
{
    noise::NoiseModel pauliOnly;
    pauliOnly.set_depolarizing(0, 0.01);
    BOOST_TEST(pauliOnly.compute_damping_covers_model());

    noise::NoiseModel thermal;
    thermal.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6);
    BOOST_TEST(!thermal.compute_damping_covers_model());
}

BOOST_AUTO_TEST_CASE(DelayExactChannelDivisibility)
{
    constexpr double t1 = 50e-6;
    constexpr double t2 = 30e-6;
    constexpr double excited_population = 0.05;
    constexpr double detuning_hz = 250e3;
    constexpr double total_delay = 4e-6;

    noise::NoiseModel noiseModel;
    noiseModel.set_idle_noise(0, t1, t2, excited_population, detuning_hz);

    // Circuit 1: single delay of 4us
    auto c1 = HadamardCircuit();
    c1->Delay(0, total_delay);
    auto noisy1 = noise::inject_exact_noise(c1, noiseModel);
    auto sim1 = MakeSimulator(Simulators::SimulationType::kDensityMatrix, 1);
    Circuits::OperationState state1;
    noisy1->Execute(sim1, state1);

    // Circuit 2: two delays of 2us
    auto c2 = HadamardCircuit();
    c2->Delay(0, total_delay / 2.0);
    c2->Delay(0, total_delay / 2.0);
    auto noisy2 = noise::inject_exact_noise(c2, noiseModel);
    auto sim2 = MakeSimulator(Simulators::SimulationType::kDensityMatrix, 1);
    Circuits::OperationState state2;
    noisy2->Execute(sim2, state2);

    BOOST_CHECK_CLOSE(sim1->ExpectationValue("X"), sim2->ExpectationValue("X"), 1e-5);
    BOOST_CHECK_CLOSE(sim1->ExpectationValue("Y"), sim2->ExpectationValue("Y"), 1e-5);
    BOOST_CHECK_CLOSE(sim1->ExpectationValue("Z"), sim2->ExpectationValue("Z"), 1e-5);
}

BOOST_AUTO_TEST_CASE(DelaySampledCoherenceMatchesExact)
{
    constexpr double duration = 3e-6;
    constexpr double t1 = 40e-6;
    constexpr double t2 = 25e-6;
    noise::NoiseModel noiseModel;
    noiseModel.set_idle_noise(0, t1, t2);

    auto c = HadamardCircuit();
    c->Delay(0, duration);

    auto noisy_exact = noise::inject_exact_noise(c, noiseModel);
    auto sim_exact = MakeSimulator(Simulators::SimulationType::kDensityMatrix, 1);
    Circuits::OperationState state_exact;
    noisy_exact->Execute(sim_exact, state_exact);
    double exact_x = sim_exact->ExpectationValue("X");

    std::mt19937 rng(42);
    size_t realizations = 10000;
    double sampled_x_sum = 0.0;
    for (size_t r = 0; r < realizations; ++r)
    {
        auto noisy_sampled = noise::inject_noise(c, noiseModel, rng);
        auto sim = MakeSimulator(Simulators::SimulationType::kStatevector, 1);
        Circuits::OperationState state;
        noisy_sampled->Execute(sim, state);
        sampled_x_sum += sim->ExpectationValue("X");
    }
    double sampled_x = sampled_x_sum / static_cast<double>(realizations);

    BOOST_CHECK_SMALL(std::abs(sampled_x - exact_x), 0.02);
    BOOST_CHECK_SMALL(std::abs(exact_x - std::exp(-duration / t2)), kTolerance);
}

BOOST_AUTO_TEST_CASE(DelayDurationValidation)
{
    // Valid construction
    Circuits::Delay<double> delay(0, 1e-6);
    BOOST_CHECK_CLOSE(delay.GetDuration(), 1e-6, 1e-12);

    // Valid mutation via SetDuration
    delay.SetDuration(2e-6);
    BOOST_CHECK_CLOSE(delay.GetDuration(), 2e-6, 1e-12);
    delay.SetDuration(0.0);
    BOOST_CHECK_EQUAL(delay.GetDuration(), 0.0);

    // Invalid mutation via SetDuration
    BOOST_CHECK_THROW(delay.SetDuration(-1e-6), std::invalid_argument);
    BOOST_CHECK_THROW(delay.SetDuration(std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_THROW(delay.SetDuration(-std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_THROW(delay.SetDuration(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);

    // Valid mutation via SetDelay
    delay.SetDelay(3e-6);
    BOOST_CHECK_CLOSE(delay.GetDuration(), 3e-6, 1e-12);

    // Invalid mutation via SetDelay
    BOOST_CHECK_THROW(delay.SetDelay(-1e-6), std::invalid_argument);
    BOOST_CHECK_THROW(delay.SetDelay(std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_THROW(delay.SetDelay(-std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_THROW(delay.SetDelay(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);

    // Polymorphic validation via IOperation pointer
    std::shared_ptr<Circuits::IOperation<double>> op = std::make_shared<Circuits::Delay<double>>(0, 1e-6);
    BOOST_CHECK_THROW(op->SetDelay(-1e-6), std::invalid_argument);
    BOOST_CHECK_THROW(op->SetDelay(std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_THROW(op->SetDelay(-std::numeric_limits<double>::infinity()), std::invalid_argument);
    BOOST_CHECK_THROW(op->SetDelay(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(CorrelatedNoiseStationaryInitVariance)
{
    constexpr double sigma = 10.0;
    constexpr double alpha = 4.0;
    constexpr double gate_time = 100e-9;
    double theta = 1.0 / (alpha * gate_time);
    double expected_var = sigma * sigma / (2.0 * theta);

    noise::NoiseModel nm_stationary;
    nm_stationary.set_correlated_ou(0, sigma, alpha, gate_time, true, true, true);

    noise::NoiseModel nm_cold;
    nm_cold.set_correlated_ou(0, sigma, alpha, gate_time, true, true, false);

    auto circ = std::make_shared<Circuits::Circuit<double>>();
    circ->AddOperation(std::make_shared<Circuits::XGate<>>(0));

    std::mt19937 rng(123);
    constexpr size_t N = 5000;
    double sum_stat = 0.0, sum_sq_stat = 0.0;
    double sum_cold = 0.0, sum_sq_cold = 0.0;

    for (size_t i = 0; i < N; ++i)
    {
        auto out_stat = noise::inject_correlated_noise(circ, nm_stationary, rng);
        auto out_cold = noise::inject_correlated_noise(circ, nm_cold, rng);

        double angle_stat = 0.0;
        for (const auto &op : out_stat->GetOperations())
        {
            if (auto rz = std::dynamic_pointer_cast<Circuits::RzGate<double>>(op))
                angle_stat = rz->GetTheta();
        }
        sum_stat += angle_stat;
        sum_sq_stat += angle_stat * angle_stat;

        double angle_cold = 0.0;
        for (const auto &op : out_cold->GetOperations())
        {
            if (auto rz = std::dynamic_pointer_cast<Circuits::RzGate<double>>(op))
                angle_cold = rz->GetTheta();
        }
        sum_cold += angle_cold;
        sum_sq_cold += angle_cold * angle_cold;
    }

    double var_stat = (sum_sq_stat - sum_stat * sum_stat / N) / (N - 1);
    double var_cold = (sum_sq_cold - sum_cold * sum_cold / N) / (N - 1);

    BOOST_CHECK_CLOSE(var_stat, expected_var, 10.0);

    double phi = std::exp(-theta * gate_time);
    double expected_cold_var = expected_var * (1.0 - phi * phi);
    BOOST_CHECK_CLOSE(var_cold, expected_cold_var, 10.0);
}

BOOST_AUTO_TEST_CASE(MultiBandOUConfigurationAndSuperposition)
{
    constexpr double gate_time = 100e-9;
    std::vector<std::pair<double, double>> bands = {{10.0, 2.0}, {20.0, 5.0}};

    noise::NoiseModel nm;
    nm.set_multi_correlated_ou(0, bands, gate_time, true, true, true);

    const auto *crn = nm.get_correlated(0);
    BOOST_REQUIRE(crn != nullptr);
    BOOST_CHECK_EQUAL(crn->bands.size(), 2);

    double var_b0 = 10.0 * 10.0 / (2.0 * (1.0 / (2.0 * gate_time)));
    double var_b1 = 20.0 * 20.0 / (2.0 * (1.0 / (5.0 * gate_time)));
    double total_expected_var = var_b0 + var_b1;

    auto circ = std::make_shared<Circuits::Circuit<double>>();
    circ->AddOperation(std::make_shared<Circuits::XGate<>>(0));

    std::mt19937 rng(999);
    constexpr size_t N = 5000;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        auto out = noise::inject_correlated_noise(circ, nm, rng);
        double angle = 0.0;
        for (const auto &op : out->GetOperations())
        {
            if (auto rz = std::dynamic_pointer_cast<Circuits::RzGate<double>>(op))
                angle = rz->GetTheta();
        }
        sum += angle;
        sum_sq += angle * angle;
    }
    double measured_var = (sum_sq - sum * sum / N) / (N - 1);
    BOOST_CHECK_CLOSE(measured_var, total_expected_var, 10.0);
}

BOOST_AUTO_TEST_CASE(OneOverFNoiseSynthesizer)
{
    noise::NoiseModel nm;
    nm.set_1_over_f_noise(0, 1e-4, 1e3, 1e7, 4, 100e-9);

    const auto *crn = nm.get_correlated(0);
    BOOST_REQUIRE(crn != nullptr);
    BOOST_CHECK_EQUAL(crn->bands.size(), 4);

    for (size_t b = 1; b < crn->bands.size(); ++b)
    {
        BOOST_CHECK_LT(crn->bands[b].phi, crn->bands[b - 1].phi);
        BOOST_CHECK_GT(crn->bands[b].theta, crn->bands[b - 1].theta);
    }
}

/**
 * The auxiliary RNG on IState is what measurement-time readout flips draw
 * from. It must be in [0,1), reproducible under SetSeed, and per-instance so
 * the per-job clones in the multi-shot thread pool never share a stream.
 */
BOOST_AUTO_TEST_CASE(AuxiliaryRngIsSeededAndPerInstance)
{
    auto a = MakeSimulator(Simulators::SimulationType::kStatevector, 1);
    auto b = MakeSimulator(Simulators::SimulationType::kStatevector, 1);
    a->SetSeed(5);
    b->SetSeed(5);

    std::vector<double> va, vb;
    for (int i = 0; i < 8; ++i)
    {
        va.push_back(a->RandomUniform());
        vb.push_back(b->RandomUniform());
    }
    for (const double v : va)
    {
        BOOST_CHECK_GE(v, 0.0);
        BOOST_CHECK_LT(v, 1.0);
    }
    BOOST_CHECK(va == vb);

    auto c = MakeSimulator(Simulators::SimulationType::kStatevector, 1);
    c->SetSeed(6);
    std::vector<double> vc;
    for (int i = 0; i < 8; ++i)
        vc.push_back(c->RandomUniform());
    BOOST_CHECK(va != vc);
}

BOOST_AUTO_TEST_CASE(DirectConfigurationReplaysReadout)
{
    std::vector<Simulators::SimulatorType> backends{Simulators::SimulatorType::kQCSim, Simulators::SimulatorType::kCompositeQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Simulators::SimulatorType::kQiskitAer);
    backends.push_back(Simulators::SimulatorType::kCompositeQiskitAer);
#endif
#ifdef __linux__
    // Configuring the wrapper requires no MPI runtime or GPU allocation.
    backends.push_back(Simulators::SimulatorType::kDistMpiGpuSim);
#endif
    Circuits::MeasurementOperation<> measurement({{0, 0}});
    measurement.SetReadout({{0.4, 0.6}});
    for (const auto backend : backends)
    {
        auto simulator = Simulators::SimulatorsFactory::CreateSimulator(backend, Simulators::SimulationType::kStatevector);
        BOOST_REQUIRE(simulator);
        const auto sample = [&] {
            Circuits::OperationState bits(1);
            std::vector<bool> values;
            for (size_t shot = 0; shot < 128; ++shot)
            {
                measurement.SetStateFromSample({shot % 2 != 0}, bits, simulator.get());
                values.push_back(bits.GetBit(0));
            }
            return values;
        };
        for (uint64_t seed : {uint64_t{0}, uint64_t{11}, UINT64_MAX})
        {
            simulator->SetSeed(seed);
            const auto expected = sample();
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                simulator->Configure("seed", std::to_string(seed).c_str());
                BOOST_CHECK(sample() == expected);
            }
            simulator->Configure("seed", std::to_string(seed ^ 1).c_str());
            BOOST_CHECK(sample() != expected);
        }
    }
}

/**
 * Readout error is a property of the QUBIT being read, applied when the
 * measurement writes its bit -- not a post-pass over the counts string indexed
 * by classical-bit position. Swap the qubit->bit map and only the bit holding
 * qubit 0's outcome may flip.
 */
BOOST_AUTO_TEST_CASE(ReadoutFlipsFollowTheMeasuredQubitNotTheBitIndex)
{
    // x q0; measure q0 -> c1, q1 -> c0. q0 reads 1, q1 reads 0.
    auto meas = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 1}, {1, 0}});
    // index 0 is qubit 0: always report a measured 1 as 0. index 1 (qubit 1):
    // none.
    meas->SetReadoutAt(0, Circuits::ReadoutRates{0.0, 1.0});
    BOOST_CHECK(meas->HasReadout());

    auto simulator = MakeSimulator(Simulators::SimulationType::kStatevector, 2);
    simulator->SetSeed(3);
    simulator->ApplyX(0);

    Circuits::OperationState state;
    state.AllocateBits(2);
    meas->Execute(simulator, state);

    const auto &bits = state.GetAllBits();
    BOOST_REQUIRE_EQUAL(bits.size(), 2u);
    BOOST_CHECK_EQUAL(bits[1], false); // c1 held q0's 1 -> flipped to 0
    BOOST_CHECK_EQUAL(bits[0], false); // c0 held q1's 0, no readout on q1
}

/** Without a simulator handle (legacy callers) readout stays inert. */
BOOST_AUTO_TEST_CASE(SetStateFromSampleWithoutRngIsUnchanged)
{
    auto meas = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}});
    meas->SetReadoutAt(0, Circuits::ReadoutRates{1.0, 1.0});

    Circuits::OperationState state;
    state.AllocateBits(1);
    meas->SetStateFromSample(std::vector<bool>{true}, state);

    BOOST_CHECK_EQUAL(state.GetAllBits()[0], true);
}

/** Clone must carry the readout rates, index for index. */
BOOST_AUTO_TEST_CASE(ReadoutRatesSurviveClone)
{
    auto meas = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{2, 0}, {5, 1}});
    meas->SetReadoutAt(1, Circuits::ReadoutRates{0.1, 0.2});

    auto copy = std::static_pointer_cast<Circuits::MeasurementOperation<>>(meas->Clone());

    BOOST_REQUIRE(copy->HasReadout());
    BOOST_REQUIRE_EQUAL(copy->GetReadout().size(), 2u);
    BOOST_CHECK_CLOSE(copy->GetReadout()[1].p_meas1_prep0, 0.1, 1e-9);
    BOOST_CHECK_CLOSE(copy->GetReadout()[1].p_meas0_prep1, 0.2, 1e-9);
    BOOST_CHECK_SMALL(copy->GetReadout()[0].p_meas1_prep0, 1e-12);
}

/**
 * GetLastMeasurements merges trailing measurements into one op and sorts them
 * by qubit. The readout rates must be permuted with their (qubit, bit) pair.
 */
BOOST_AUTO_TEST_CASE(ReadoutRatesSurviveGetLastMeasurements)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));

    auto m1 = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{3, 0}});
    m1->SetReadoutAt(0, Circuits::ReadoutRates{0.3, 0.03});

    auto m2 = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{1, 1}, {2, 2}});
    m2->SetReadoutAt(0, Circuits::ReadoutRates{0.1, 0.01});
    // index 1 (qubit 2) deliberately left at zero rates

    circuit->AddOperation(m1);
    circuit->AddOperation(m2);

    const std::vector<bool> executed(circuit->GetOperations().size(), false);
    auto merged = circuit->GetLastMeasurements(executed, /*sort=*/true);

    const auto &qs = merged->GetQubits();
    const auto &bs = merged->GetBitsIndices();
    BOOST_REQUIRE_EQUAL(qs.size(), 3u);
    BOOST_REQUIRE(merged->HasReadout());
    const auto &r = merged->GetReadout();
    BOOST_REQUIRE_EQUAL(r.size(), 3u);

    // sorted by qubit: (1,c1,0.1/0.01), (2,c2,0/0), (3,c0,0.3/0.03)
    BOOST_CHECK_EQUAL(qs[0], 1u);
    BOOST_CHECK_EQUAL(bs[0], 1u);
    BOOST_CHECK_CLOSE(r[0].p_meas1_prep0, 0.1, 1e-9);
    BOOST_CHECK_EQUAL(qs[1], 2u);
    BOOST_CHECK_EQUAL(bs[1], 2u);
    BOOST_CHECK_SMALL(r[1].p_meas1_prep0, 1e-12);
    BOOST_CHECK_EQUAL(qs[2], 3u);
    BOOST_CHECK_EQUAL(bs[2], 0u);
    BOOST_CHECK_CLOSE(r[2].p_meas1_prep0, 0.3, 1e-9);
    BOOST_CHECK_CLOSE(r[2].p_meas0_prep1, 0.03, 1e-9);
}

/** No readout on any source op -> the merged op carries none either. */
BOOST_AUTO_TEST_CASE(GetLastMeasurementsWithoutReadoutStaysClean)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));

    const std::vector<bool> executed(1, false);
    auto merged = circuit->GetLastMeasurements(executed, true);

    BOOST_CHECK(!merged->HasReadout());
}

/**
 * EnsureProperOrderForMeasurements splits a multi-qubit measurement into
 * single-pair ops. Each rebuilt op must carry the rates of its own qubit.
 */
BOOST_AUTO_TEST_CASE(ReadoutRatesSurviveEnsureProperOrder)
{
    Circuits::Circuit<double> circuit;
    auto m = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}});
    m->SetReadoutAt(0, Circuits::ReadoutRates{0.05, 0.5});
    m->SetReadoutAt(1, Circuits::ReadoutRates{0.06, 0.6});
    circuit.AddOperation(m);

    circuit.EnsureProperOrderForMeasurements();

    size_t seen = 0;
    for (const auto &op : circuit.GetOperations())
    {
        if (op->GetType() != Circuits::OperationType::kMeasurement)
            continue;
        auto meas = std::static_pointer_cast<Circuits::MeasurementOperation<>>(op);
        BOOST_REQUIRE_EQUAL(meas->GetQubits().size(), 1u);
        BOOST_REQUIRE(meas->HasReadout());
        const auto q = meas->GetQubits()[0];
        const auto &r = meas->GetReadout()[0];
        if (q == 0)
            BOOST_CHECK_CLOSE(r.p_meas0_prep1, 0.5, 1e-9);
        if (q == 1)
            BOOST_CHECK_CLOSE(r.p_meas0_prep1, 0.6, 1e-9);
        ++seen;
    }
    BOOST_CHECK_EQUAL(seen, 2u);
}

BOOST_AUTO_TEST_CASE(ReplacingNoisyMeasurementClearsReadoutRates)
{
    Circuits::Circuit<double> circuit;
    auto noisy = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}});
    noisy->SetReadout({{1.0, 1.0}});
    circuit.AddOperation(noisy);
    circuit.AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{1, 0}}));

    circuit.EnsureProperOrderForMeasurements();

    BOOST_REQUIRE_EQUAL(circuit.GetOperations().size(), 1u);
    auto rebuilt = std::static_pointer_cast<Circuits::MeasurementOperation<>>(circuit.GetOperations()[0]);
    BOOST_CHECK_EQUAL(rebuilt->GetQubits()[0], 1u);
    BOOST_CHECK(!rebuilt->HasReadout());
    auto simulator = MakeSimulator(Simulators::SimulationType::kStatevector, 2);
    Circuits::OperationState state(1);
    rebuilt->Execute(simulator, state);
    BOOST_CHECK_EQUAL(state.GetAllBits()[0], false);
}

BOOST_AUTO_TEST_CASE(ConfigurationSeedsReadoutWithoutChangingBackendSequence)
{
    for (const auto method :
         {Simulators::SimulationType::kStatevector, Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kStabilizer,
          Simulators::SimulationType::kDensityMatrix, Simulators::SimulationType::kMatrixProductOperator})
    {
        auto direct = MakeSimulator(method, 2);
        auto configured = MakeSimulator(method, 2);
        direct->Configure("seed", "11");
        Simulators::Configuration config;
        config.SetConfiguration("seed", "11");
        config.ApplyConfigurationToSimulator(configured);
        for (auto simulator : {direct, configured})
        {
            simulator->ApplyH(0);
            simulator->ApplyH(1);
        }
        BOOST_CHECK(direct->SampleCountsMany({0, 1}, 200) == configured->SampleCountsMany({0, 1}, 200));

        auto again = MakeSimulator(method, 2);
        config.ApplyConfigurationToSimulator(again);
        for (int i = 0; i < 20; ++i)
        {
            const auto expected = configured->RandomUniform();
            BOOST_CHECK_EQUAL(direct->RandomUniform(), expected);
            BOOST_CHECK_EQUAL(again->RandomUniform(), expected);
        }
    }
}

BOOST_AUTO_TEST_CASE(ConfigurationKeepsGpuPlacementOutOfCpuBackends)
{
    std::vector<Simulators::SimulatorType> backends{Simulators::SimulatorType::kQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Simulators::SimulatorType::kQiskitAer);
#endif
    for (const auto backend : backends)
    {
        auto simulator = Simulators::SimulatorsFactory::CreateSimulator(backend, Simulators::SimulationType::kMatrixProductState);
        Simulators::Configuration configuration;
        configuration.SetConfiguration("gpu_device", "0");
        configuration.SetConfiguration("seed", "42");
        configuration.ApplyConfigurationToSimulator(simulator);
        simulator->AllocateQubits(1);
        simulator->Initialize();
        simulator->ApplyX(0);
        const auto counts = simulator->SampleCountsMany({0}, 16);
        BOOST_CHECK_EQUAL(counts.at(std::vector<bool>{true}), 16u);
        BOOST_CHECK_EQUAL(configuration.GetConfiguration("gpu_device"), "0");
        BOOST_CHECK(simulator->GetConfiguration("gpu_device").empty());
    }
}

BOOST_AUTO_TEST_CASE(NoiseAddReadoutUsesMeasurementQubits)
{
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 1}, {1, 0}}));
    auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{2}, std::vector<size_t>{2});
    network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
    noise::NoiseModel model;
    model.set_readout_error_symmetric(0, 1.0);
    noise::NoiseAdd add;
    add.seed(42);
    const auto counts = add.noisy_execute(circuit, network, 0, model, 128, 16);
    BOOST_REQUIRE_EQUAL(counts.size(), 1u);
    BOOST_CHECK_EQUAL(counts.at(std::vector<bool>{false, true}), 128u);
    BOOST_CHECK(!std::static_pointer_cast<Circuits::MeasurementOperation<>>(circuit->GetOperations()[0])->HasReadout());
}

BOOST_AUTO_TEST_CASE(NoiseAddSeedReproducesRunsAndAdvancesRealizations)
{
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    for (Types::qubit_t q = 0; q < 4; ++q)
        circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(q));
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}));
    for (const std::string configured : {"", "0", "18446744073709551615"})
    {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{4}, std::vector<size_t>{4});
        network->SetMaxSimulators(1);
        if (!configured.empty())
            network->Configure("seed", configured.c_str());
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
        noise::NoiseModel model;
        noise::NoiseAdd add;
        add.seed(123);
        const auto first = add.full_noise_execute(circuit, network, 0, model, 128, 128);
        BOOST_CHECK_GE(first.size(), 12u);
        const auto second = add.full_noise_execute(circuit, network, 0, model, 128, 128);
        BOOST_CHECK(first != second);
        add.seed(123);
        BOOST_CHECK(first == add.full_noise_execute(circuit, network, 0, model, 128, 128));
        BOOST_CHECK_EQUAL(network->GetSimulator()->GetConfiguration("seed"), configured);
    }
}

BOOST_AUTO_TEST_CASE(NoiseAddReportsThermalApproximationOncePerCall)
{
    auto circuit = HadamardCircuit();
    auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{1}, std::vector<size_t>{1});
    network->CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
    noise::NoiseModel model;
    model.set_thermal_relaxation(0, 1.0, 10.0, 15.0);
    noise::NoiseAdd add;
    int warnings = 0;
    add.set_warning_handler([&](const std::string &message) {
        BOOST_CHECK(message.find("T2 clamped to T1") != std::string::npos);
        ++warnings;
    });
    add.noisy_estimate_montecarlo(circuit, network, 0, {"X"}, model, 16);
    BOOST_CHECK_EQUAL(warnings, 1);
    add.set_exact_channels(true);
    add.inject(circuit, model);
    BOOST_CHECK_EQUAL(warnings, 1);
}

/** Each coherent setter replaces only its own component, so axes compose. */
BOOST_AUTO_TEST_CASE(CoherentSettersComposeAcrossAxes)
{
    noise::NoiseModel noiseModel;
    noiseModel.set_coherent_bit_flip(0, 0.2);
    noiseModel.set_coherent_dephasing(0, 0.2);
    noiseModel.set_coherent_depolarizing(0, 0.1);

    const auto *coherent = noiseModel.get_coherent(0);
    BOOST_REQUIRE(coherent);
    BOOST_CHECK_CLOSE(coherent->rx, 2.0 * std::asin(std::sqrt(0.2)), 1e-9);
    BOOST_CHECK_SMALL(coherent->ry, kTolerance);
    BOOST_CHECK_CLOSE(coherent->rz, 2.0 * std::asin(std::sqrt(0.2)), 1e-9);
    BOOST_CHECK_CLOSE(coherent->diag, 2.0 * std::asin(std::sqrt(0.1)), 1e-9);
}

/**
 * A circuit-level Pauli channel is replaced by the Pauli it draws on the
 * sampled path and kept as a channel on the exact path. Probability one on
 * index 6 (digit 0 = Y, digit 1 = X) makes the draw certain.
 */
BOOST_AUTO_TEST_CASE(CircuitPauliChannelIsSampledOrKept)
{
    std::vector<double> probabilities(16, 0.0);
    probabilities[6] = 1.0;
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{3, 5}, Simulators::QuantumChannel::Pauli(probabilities)));
    const noise::NoiseModel noNoise;

    std::mt19937 rng(1);
    const auto sampled = noise::inject_combined_noise(circuit, noNoise, rng);
    BOOST_REQUIRE_EQUAL(sampled->GetOperations().size(), 2u);
    const auto y = std::dynamic_pointer_cast<Circuits::YGate<>>(sampled->GetOperations()[0]);
    const auto x = std::dynamic_pointer_cast<Circuits::XGate<>>(sampled->GetOperations()[1]);
    BOOST_REQUIRE(y && x);
    BOOST_CHECK_EQUAL(y->GetQubit(), 3u);
    BOOST_CHECK_EQUAL(x->GetQubit(), 5u);

    const auto exact = noise::inject_combined_noise_exact(circuit, noNoise, rng);
    BOOST_REQUIRE_EQUAL(exact->GetOperations().size(), 1u);
    BOOST_CHECK(exact->GetOperations()[0]->GetType() == Circuits::OperationType::kQuantumChannel);
}

/**
 * An identity draw adds no gate, but a realization must still reach the
 * channel's qubit: count strings are as wide as the highest qubit touched, so
 * dropping it would shorten some realizations' counts and not others.
 */
BOOST_AUTO_TEST_CASE(SampledIdentityKeepsCircuitWidth)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{3}, Simulators::QuantumChannel::Pauli({1.0, 0.0, 0.0, 0.0})));
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{1, 0}}));
    const noise::NoiseModel noNoise;

    std::mt19937 rng(1);
    for (const auto &noisy : {noise::inject_noise(circuit, noNoise, rng), noise::inject_combined_noise(circuit, noNoise, rng),
                              noise::inject_coherent_noise(circuit, noNoise, rng), noise::inject_correlated_noise(circuit, noNoise, rng)})
    {
        BOOST_CHECK_EQUAL(noisy->GetMaxQubitIndex(), 3u);
        BOOST_REQUIRE_EQUAL(noisy->GetOperations().size(), 2u);
        BOOST_CHECK(noisy->GetOperations()[1]->GetType() == Circuits::OperationType::kDelay);
    }

    // A drawn Pauli already touches the qubit, so nothing is added.
    auto flipped = std::make_shared<Circuits::Circuit<double>>();
    flipped->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{3}, Simulators::QuantumChannel::Pauli({0.0, 1.0, 0.0, 0.0})));
    const auto noisy = noise::inject_noise(flipped, noNoise, rng);
    BOOST_REQUIRE_EQUAL(noisy->GetOperations().size(), 1u);
    BOOST_CHECK(std::dynamic_pointer_cast<Circuits::XGate<>>(noisy->GetOperations()[0]));
}

/**
 * Coherent injection samples its rotation signs on every backend but, like the
 * other injectors, keeps circuit-level channels exact for density-matrix/MPO
 * execution. NoiseAdd's coherent layer follows its exact flag and, as its other
 * layers do, attaches the model's readout error.
 */
BOOST_AUTO_TEST_CASE(CoherentInjectionKeepsCircuitChannelsExact)
{
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::Pauli({0.5, 0.5, 0.0, 0.0})));
    circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
    noise::NoiseModel model;
    model.set_coherent_rotation(0, 0.1, 0.0, 0.0);
    model.set_readout_error(0, 0.2, 0.3);

    const auto count = [](const std::shared_ptr<Circuits::Circuit<double>> &noisy, Circuits::OperationType type) {
        size_t n = 0;
        for (const auto &op : noisy->GetOperations())
            n += op->GetType() == type;
        return n;
    };
    const auto rotated = [](const std::shared_ptr<Circuits::Circuit<double>> &noisy) {
        for (const auto &op : noisy->GetOperations())
            if (std::dynamic_pointer_cast<Circuits::RxGate<>>(op))
                return true;
        return false;
    };
    const auto hasReadout = [](const std::shared_ptr<Circuits::Circuit<double>> &noisy) {
        for (const auto &op : noisy->GetOperations())
            if (op->GetType() == Circuits::OperationType::kMeasurement)
                return std::static_pointer_cast<Circuits::MeasurementOperation<>>(op)->HasReadout();
        return false;
    };

    std::mt19937 rng(3);
    const auto exact = noise::inject_coherent_noise_exact(circuit, model, rng);
    BOOST_CHECK_EQUAL(count(exact, Circuits::OperationType::kQuantumChannel), 1u);
    BOOST_CHECK(rotated(exact));
    const auto sampled = noise::inject_coherent_noise(circuit, model, rng);
    BOOST_CHECK_EQUAL(count(sampled, Circuits::OperationType::kQuantumChannel), 0u);
    BOOST_CHECK(rotated(sampled));

    noise::NoiseAdd add;
    add.seed(9);
    for (const bool exactChannels : {true, false})
    {
        add.set_exact_channels(exactChannels);
        const auto noisy = add.inject_coherent(circuit, model);
        BOOST_CHECK_EQUAL(count(noisy, Circuits::OperationType::kQuantumChannel), exactChannels ? 1u : 0u);
        BOOST_CHECK(rotated(noisy));
        BOOST_CHECK(hasReadout(noisy));
    }
    BOOST_CHECK(!std::static_pointer_cast<Circuits::MeasurementOperation<>>(circuit->GetOperations()[2])->HasReadout());
}

/** A noiseless gate gets no injected noise, and gate and circuit copies keep the mark. */
BOOST_AUTO_TEST_CASE(NoiselessGateGetsNoInjectedNoise)
{
    auto basisChange = std::make_shared<Circuits::HadamardGate<>>(0);
    basisChange->SetNoiseless();
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(basisChange);
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    noise::NoiseModel noiseModel;
    noiseModel.set_bit_flip(0, 1.0);

    std::mt19937 rng(1);
    const auto noisy = noise::inject_noise(circuit, noiseModel, rng);
    BOOST_REQUIRE_EQUAL(noisy->GetOperations().size(), 3u);
    BOOST_CHECK(std::dynamic_pointer_cast<Circuits::HadamardGate<>>(noisy->GetOperations()[1]));
    BOOST_CHECK(std::dynamic_pointer_cast<Circuits::XGate<>>(noisy->GetOperations()[2]));

    const auto copy = std::static_pointer_cast<Circuits::Circuit<double>>(circuit->Clone());
    BOOST_CHECK(copy->GetOperations()[0]->IsNoiseless());
    BOOST_CHECK(!copy->GetOperations()[1]->IsNoiseless());
    BOOST_CHECK(basisChange->Clone()->IsNoiseless());
    BOOST_CHECK(basisChange->Remap({{0, 4}})->IsNoiseless());

    // Remapping before injection must not turn the basis change noisy.
    const auto remapped = std::static_pointer_cast<Circuits::Circuit<double>>(circuit->Remap({{0, 4}}));
    noise::NoiseModel remappedModel;
    remappedModel.set_bit_flip(4, 1.0);
    const auto remappedNoisy = noise::inject_noise(remapped, remappedModel, rng);
    BOOST_REQUIRE_EQUAL(remappedNoisy->GetOperations().size(), 3u);
    BOOST_CHECK(std::dynamic_pointer_cast<Circuits::HadamardGate<>>(remappedNoisy->GetOperations()[1]));
}

/** Every gate's Clone and Remap carry its type, qubits, parameters, delay and noiseless mark. */
BOOST_AUTO_TEST_CASE(EveryGateCopyKeepsAllFields)
{
    for (int t = static_cast<int>(Circuits::QuantumGateType::kPhaseGateType); t <= static_cast<int>(Circuits::QuantumGateType::kCCXGateType); ++t)
    {
        const auto type = static_cast<Circuits::QuantumGateType>(t);
        for (const bool noiseless : {false, true})
        {
            const auto gate = Circuits::CircuitFactory<>::CreateGate(type, 3, 5, 7, 0.1, 0.2, 0.3, 0.4);
            BOOST_REQUIRE(gate);
            gate->SetDelay(2.5);
            gate->SetNoiseless(noiseless);

            const auto copy = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(gate->Clone());
            BOOST_REQUIRE(copy);
            BOOST_CHECK(copy != gate);
            BOOST_CHECK(copy->GetGateType() == type);
            BOOST_CHECK(copy->AffectedQubits() == gate->AffectedQubits());
            BOOST_CHECK(copy->GetParams() == gate->GetParams());
            BOOST_CHECK_EQUAL(copy->GetDelay(), 2.5);
            BOOST_CHECK_EQUAL(copy->IsNoiseless(), noiseless);

            const auto remapped = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(gate->Remap({{3, 13}, {5, 15}, {7, 17}}));
            BOOST_REQUIRE(remapped);
            BOOST_CHECK(remapped->GetGateType() == type);
            BOOST_CHECK(remapped->GetParams() == gate->GetParams());
            BOOST_CHECK_EQUAL(remapped->IsNoiseless(), noiseless);
            // The original is untouched by copying.
            BOOST_CHECK_EQUAL(gate->AffectedQubits().front(), 3u);
        }
    }
}

/** The model's readout flip composes with a measurement's own, e.g. Stim's M(p). */
BOOST_AUTO_TEST_CASE(ModelReadoutComposesWithMeasurementReadout)
{
    auto measurement = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}});
    measurement->SetReadoutAt(0, Circuits::ReadoutRates{0.1, 0.2});
    auto circuit = std::make_shared<Circuits::Circuit<double>>();
    circuit->AddOperation(measurement);
    noise::NoiseModel noiseModel;
    noiseModel.set_readout_error(0, 0.3, 0.4);

    noise::attach_readout_error(circuit, noiseModel);

    const auto &rates = measurement->GetReadout();
    BOOST_REQUIRE_EQUAL(rates.size(), 1u);
    BOOST_CHECK_CLOSE(rates[0].p_meas1_prep0, 0.1 * (1 - 0.4) + 0.9 * 0.3, 1e-9);
    BOOST_CHECK_CLOSE(rates[0].p_meas0_prep1, 0.2 * (1 - 0.3) + 0.8 * 0.4, 1e-9);
}

BOOST_AUTO_TEST_CASE(NoiseAddInfersExactChannelsAndAutomaticRealizations)
{
    for (const auto method : {Simulators::SimulationType::kDensityMatrix, Simulators::SimulationType::kMatrixProductOperator})
    {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{1}, std::vector<size_t>{1});
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        network->SetMaxSimulators(1);
        auto circuit = std::make_shared<Circuits::Circuit<>>();
        circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));
        circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
        noise::NoiseModel model;
        model.set_bit_flip(0, 0.5);
        noise::NoiseAdd add;
        add.seed(31);
        const auto counts = add.noisy_execute(circuit, network, 0, model, 512);
        BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
        BOOST_CHECK(add.last_realization_policy().exact_channels);
        BOOST_CHECK(!add.last_realization_policy().requested);
        BOOST_REQUIRE_EQUAL(counts.size(), 2u);
        for (const auto &[bits, count] : counts)
            BOOST_CHECK_GT(count, 150u);
        const auto exact = add.noisy_estimate_montecarlo(circuit, network, 0, {"Z"}, model);
        BOOST_REQUIRE_EQUAL(exact.size(), 1u);
        BOOST_CHECK_SMALL(exact[0], kTolerance);
        BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
        add.noisy_estimate_montecarlo(circuit, network, 0, {"Z"}, model, 3);
        BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 3u);
        BOOST_REQUIRE(add.last_realization_policy().requested);
        BOOST_CHECK_EQUAL(*add.last_realization_policy().requested, 3u);
        add.set_exact_channels(false);
        const auto trajectory = add.noisy_estimate_montecarlo(circuit, network, 0, {"Z"}, model, 1);
        BOOST_CHECK_SMALL(std::abs(trajectory[0]) - 1.0, kTolerance);
        BOOST_CHECK(!add.last_realization_policy().exact_channels);
        add.set_automatic_channels();
        add.noisy_estimate_montecarlo(circuit, network, 0, {"Z"}, model);
        BOOST_CHECK(add.last_realization_policy().exact_channels);
        BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
        BOOST_CHECK_THROW(add.noisy_execute(circuit, network, 0, model, 8, 0), std::invalid_argument);
        // Inference is scoped to execution; injection alone retains its explicit policy.
        BOOST_CHECK(!add.uses_exact_channels());
    }
}

BOOST_AUTO_TEST_CASE(AutomaticNoisePolicyRetainsStochasticLayersAndRecognizesDeterministicOperations)
{
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = Simulators::SimulatorType::kQCSim;
    config.simulation_type = Simulators::SimulationType::kDensityMatrix;
    auto circuit = HadamardCircuit();
    circuit->AddOperation(std::make_shared<Circuits::NoOperation<>>());
    auto nested = std::make_shared<Circuits::Circuit<>>();
    nested->AddOperation(std::make_shared<Circuits::XGate<>>(0));
    circuit->AddOperation(nested);
    circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
    noise::NoiseModel model;
    auto policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 100, circuit, true);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    BOOST_CHECK(!policy.stochastic_execution);
    // Forcing sampled noise does not turn a native density-matrix reset into collapse.
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 100, circuit, true, false);
    BOOST_CHECK_EQUAL(policy.realizations, 100u);
    BOOST_CHECK(!policy.stochastic_execution);
    model.set_coherent_rotation(0, 0.1, 0.0, 0.0);
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 100, circuit, true);
    BOOST_CHECK_EQUAL(policy.realizations, 100u);
    BOOST_CHECK(policy.stochastic_noise);
    model = noise::NoiseModel{};
    model.set_correlated_ou(0, 0.2, 1.0, 0.1);
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 64, circuit);
    BOOST_CHECK_EQUAL(policy.realizations, 64u);
    BOOST_CHECK(policy.stochastic_noise);
    auto measured = HadamardCircuit();
    measured->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
    measured->AddOperation(std::make_shared<Circuits::XGate<>>(0));
    model = noise::NoiseModel{};
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kPauli, 64, measured);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    BOOST_CHECK(policy.stochastic_execution);
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kPauli, 100, measured, true);
    BOOST_CHECK_EQUAL(policy.realizations, 100u);
}

BOOST_AUTO_TEST_CASE(PureStateLeadingResetsAndNestedChannelsFollowTheEvolution)
{
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = Simulators::SimulatorType::kQCSim;
    config.simulation_type = Simulators::SimulationType::kStatevector;
    noise::NoiseModel model;
    // A reset before any other operation on its wire leaves a fresh |0>.
    auto leading = std::make_shared<Circuits::Circuit<>>();
    leading->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0, 1}));
    leading->AddOperation(std::make_shared<Circuits::NoOperation<>>());
    auto nested = std::make_shared<Circuits::Circuit<>>();
    nested->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    leading->AddOperation(nested);
    leading->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    BOOST_CHECK(!MaestroExecution::HasStochasticEvolution(leading, false));
    auto policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 100, leading, true);
    BOOST_CHECK(!policy.stochastic_execution);
    // A checkpoint suffix continues an arbitrary saved state.
    BOOST_CHECK(MaestroExecution::HasStochasticEvolution(leading, false, false));
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 64, leading, false, std::nullopt,
                                                        false);
    BOOST_CHECK(policy.stochastic_execution);
    // After a nested gate on its wire, a reset may collapse a superposition.
    auto later = std::make_shared<Circuits::Circuit<>>();
    later->AddOperation(nested);
    later->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
    BOOST_CHECK(MaestroExecution::HasStochasticEvolution(later, false));
    BOOST_CHECK(!MaestroExecution::HasStochasticEvolution(later, true));

    // Nested channels are sampled noise on pure-state backends too.
    auto channels = std::make_shared<Circuits::Circuit<>>();
    auto inner = std::make_shared<Circuits::Circuit<>>();
    inner->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(.1)));
    channels->AddOperation(inner);
    BOOST_CHECK(channels->HasQuantumChannels());
    policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, MaestroExecution::NoiseLayer::kCombined, 64, channels);
    BOOST_CHECK(policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 64u);
}

BOOST_AUTO_TEST_CASE(ExactChannelsRequireEveryCandidateBackend)
{
    using Backend = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = Backend::kQCSim;
    config.simulation_type = Method::kStatevector;
    config.optimization_candidates = {{Backend::kQCSim, Method::kDensityMatrix}};
    BOOST_CHECK(MaestroExecution::UsesExactQuantumChannels(config));
    config.optimization_candidates.emplace_back(Backend::kQCSim, Method::kMatrixProductOperator);
    BOOST_CHECK(MaestroExecution::UsesExactQuantumChannels(config));
    config.optimization_candidates.emplace_back(Backend::kQCSim, Method::kStatevector);
    BOOST_CHECK(!MaestroExecution::UsesExactQuantumChannels(config));
    config.simulation_type = Method::kDensityMatrix;
    config.optimization_candidates = {{Backend::kQCSim, Method::kStatevector}};
    BOOST_CHECK(!MaestroExecution::UsesExactQuantumChannels(config));
    // A fixed backend ignores the candidate list, as ConfigureNetwork does.
    config.fixed_backend = true;
    BOOST_CHECK(MaestroExecution::UsesExactQuantumChannels(config));
}

BOOST_AUTO_TEST_CASE(NoiseAddFollowsAutomaticallySelectedMixedStateBackend)
{
    // ConfigureNetwork's automatic selection leaves an MPS placeholder beside
    // the one requested candidate. NoiseAdd must infer exact channels from the
    // candidate and, when seeded, pin it rather than the placeholder.
    for (const auto method : {Simulators::SimulationType::kDensityMatrix, Simulators::SimulationType::kMatrixProductOperator})
        for (const bool seeded : {false, true})
        {
            auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{1}, std::vector<size_t>{1});
            network->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, method);
            network->CreateSimulator();
            network->SetOptimizeSimulator(true);
            network->SetMaxSimulators(1);
            auto circuit = std::make_shared<Circuits::Circuit<>>();
            circuit->AddOperation(std::make_shared<Circuits::XGate<>>(0));
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}}));
            noise::NoiseModel model;
            model.set_bit_flip(0, 0.25);
            noise::NoiseAdd add;
            if (seeded)
                add.seed(17);
            const auto counts = add.noisy_execute(circuit, network, 0, model, 2000);
            BOOST_CHECK(add.last_realization_policy().exact_channels);
            BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
            size_t zeros = 0, total = 0;
            for (const auto &[bits, count] : counts)
            {
                total += count;
                if (!bits.empty() && !bits[0])
                    zeros += count;
            }
            BOOST_CHECK_EQUAL(total, 2000u);
            BOOST_CHECK_GT(zeros, 380u);
            BOOST_CHECK_LT(zeros, 620u);
            const auto estimate = add.noisy_estimate_montecarlo(circuit, network, 0, {"Z"}, model);
            BOOST_REQUIRE_EQUAL(estimate.size(), 1u);
            BOOST_CHECK_SMALL(estimate[0] + 0.5, kTolerance);
            BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
        }
}

// The QCSim Pauli propagator applies channels to the observable it propagates,
// so it evaluates them exactly without holding a density matrix. Whether noise
// injection uses that by default depends on the model: channels diagonal in
// the Pauli basis only scale terms, while relaxation also sends Z to the
// identity and, without a truncation threshold, fills the operator space of
// the observable's light cone.
BOOST_AUTO_TEST_CASE(PauliPropagatorExactChannelPolicyFollowsTheModel)
{
    using Backend = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    BOOST_CHECK(Simulators::EvolvesExactQuantumChannels(Backend::kQCSim, Method::kPauliPropagator));
    BOOST_CHECK(Simulators::EvolvesExactQuantumChannels(Backend::kQCSim, Method::kDensityMatrix));
    BOOST_CHECK(Simulators::EvolvesExactQuantumChannels(Backend::kGpuSim, Method::kMatrixProductOperator));
    // The GPU propagator still measures and flips on reset.
    BOOST_CHECK(!Simulators::EvolvesExactQuantumChannels(Backend::kGpuSim, Method::kPauliPropagator));
    BOOST_CHECK(!Simulators::EvolvesExactQuantumChannels(Backend::kQCSim, Method::kStatevector));
    BOOST_CHECK(!Simulators::EvolvesExactQuantumChannels(Backend::kCompositeQCSim, Method::kStatevector));

    MaestroExecution::SimulatorConfig propagator;
    propagator.simulator_type = Backend::kQCSim;
    propagator.simulation_type = Method::kPauliPropagator;
    propagator.fixed_backend = true;
    auto density = propagator, gpu = propagator;
    density.simulation_type = Method::kDensityMatrix;
    gpu.simulator_type = Backend::kGpuSim;
    BOOST_CHECK(MaestroExecution::UsesExactQuantumChannels(propagator));
    BOOST_CHECK(MaestroExecution::HasEnsembleState(propagator));
    BOOST_CHECK(MaestroExecution::MayRunPauliPropagator(propagator));
    BOOST_CHECK(!MaestroExecution::MayRunPauliPropagator(density));
    BOOST_CHECK(!MaestroExecution::UsesExactQuantumChannels(gpu));
    BOOST_CHECK(!MaestroExecution::HasEnsembleState(gpu));
    auto mixed = propagator;
    mixed.fixed_backend = false;
    mixed.optimization_candidates = {{Backend::kQCSim, Method::kPauliPropagator}, {Backend::kQCSim, Method::kDensityMatrix}};
    BOOST_CHECK(MaestroExecution::UsesExactQuantumChannels(mixed));
    BOOST_CHECK(MaestroExecution::MayRunPauliPropagator(mixed));
    mixed.optimization_candidates.emplace_back(Backend::kQCSim, Method::kStatevector);
    BOOST_CHECK(!MaestroExecution::UsesExactQuantumChannels(mixed));

    noise::NoiseModel pauli;
    pauli.set_all_depolarizing(2, 0.1);
    pauli.set_dephasing(0, 0.05);
    pauli.set_qubit_noise(1, 0.01, 0.02, 0.03);
    pauli.set_2q_depolarizing(0, 1, 0.02);
    pauli.set_phase_damping(1, 0.1);
    pauli.set_readout_error(0, 0.02, 0.01);
    BOOST_CHECK(pauli.exact_channels_are_pauli_diagonal());
    BOOST_CHECK(MaestroExecution::InjectsExactChannels(propagator, pauli));
    BOOST_CHECK(MaestroExecution::InjectsExactChannels(density, pauli));
    BOOST_CHECK(!MaestroExecution::InjectsExactChannels(gpu, pauli));

    // Relaxation with a finite T1 is not diagonal, however it is configured.
    noise::NoiseModel thermal, t1, idle;
    thermal.set_thermal_relaxation(0, 1.0, 20.0, 15.0);
    t1.set_t1(0, 0.1);
    idle.set_idle_noise(0, 25.0, 25.0, 0.1);
    for (const auto *model : {&thermal, &t1, &idle})
    {
        BOOST_CHECK(!model->exact_channels_are_pauli_diagonal());
        BOOST_CHECK(!MaestroExecution::InjectsExactChannels(propagator, *model));
        BOOST_CHECK(!MaestroExecution::InjectsExactChannels(mixed, *model));
        // A mixed-state backend holds the state and is unaffected.
        BOOST_CHECK(MaestroExecution::InjectsExactChannels(density, *model));
    }
    // An infinite T1 leaves pure dephasing, which is diagonal.
    noise::NoiseModel dephasing;
    dephasing.set_thermal_relaxation(0, 1.0, std::numeric_limits<double>::infinity(), 15.0);
    BOOST_CHECK(dephasing.exact_channels_are_pauli_diagonal());
    BOOST_CHECK(MaestroExecution::InjectsExactChannels(propagator, dephasing));
    // Channels that cannot be sampled run exactly, with the whole model.
    noise::NoiseModel damping = thermal;
    damping.set_generalized_amplitude_damping(1, 0.1, 0.05);
    BOOST_CHECK(!damping.exact_channels_are_pauli_diagonal());
    BOOST_CHECK(MaestroExecution::InjectsExactChannels(propagator, damping));
    noise::NoiseModel correlated;
    correlated.set_correlated_phase_flip(0, 1, 0.05, 0.5);
    BOOST_CHECK(correlated.exact_channels_are_pauli_diagonal());
    BOOST_CHECK(MaestroExecution::InjectsExactChannels(propagator, correlated));

    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    const auto resolve = [&](const noise::NoiseModel &model, std::optional<bool> exact = std::nullopt) {
        return MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, propagator, MaestroExecution::NoiseLayer::kCombined, 100, circuit, true, exact);
    };
    auto policy = resolve(pauli);
    BOOST_CHECK(policy.exact_channels && !policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    policy = resolve(thermal);
    BOOST_CHECK(!policy.exact_channels && policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 100u);
    policy = resolve(thermal, true);
    BOOST_CHECK(policy.exact_channels && !policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    policy = resolve(damping);
    BOOST_CHECK(policy.exact_channels);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    // A caller can still sample a Pauli model.
    policy = resolve(pauli, false);
    BOOST_CHECK(!policy.exact_channels);
    BOOST_CHECK_EQUAL(policy.realizations, 100u);
    // Coherent disorder is sampled around the exact channels.
    auto coherent = pauli;
    coherent.set_all_coherent_depolarizing(2, 0.02);
    policy = resolve(coherent);
    BOOST_CHECK(policy.exact_channels && policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 100u);

    // An unrecorded reset is a channel there: the evolution stays deterministic
    // and terminal measurements can be sampled from one recorded circuit.
    circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}}));
    BOOST_CHECK(!MaestroExecution::HasStochasticEvolution(circuit, MaestroExecution::UsesExactQuantumChannels(propagator)));
    BOOST_CHECK(MaestroExecution::HasStochasticEvolution(circuit, MaestroExecution::UsesExactQuantumChannels(gpu)));
    BOOST_CHECK(!circuit->HasOpsAfterMeasurements(MakeSimulator(Method::kPauliPropagator, 2)->SupportsQuantumChannels()));
}

// Automatic evaluation judges the channels a circuit receives by what they do.
// Relaxation that is the identity or unital only scales terms, relaxation on
// qubits the circuit never gates is not injected at all, and a Kraus
// instruction cannot be sampled.
BOOST_AUTO_TEST_CASE(PauliPropagatorExactChannelPolicyFollowsTheCircuit)
{
    using Channel = Simulators::QuantumChannel;
    using MaestroExecution::InjectsExactChannels;
    MaestroExecution::SimulatorConfig propagator;
    propagator.simulator_type = Simulators::SimulatorType::kQCSim;
    propagator.simulation_type = Simulators::SimulationType::kPauliPropagator;
    propagator.fixed_backend = true;
    auto statevector = propagator;
    statevector.simulation_type = Simulators::SimulationType::kStatevector;

    // Equal populations make thermal and idle relaxation unital at any T2, and
    // a zero T1 probability (an infinite T1) is the identity.
    noise::NoiseModel unital;
    unital.set_thermal_relaxation(0, 1.0, 1.0, 2.0, 0.5);
    unital.set_thermal_relaxation_2q(1, 1.0, 1.0, 1.5, 0.5);
    unital.set_idle_noise(0, 25.0, 30.0, 0.5);
    unital.set_t1(2, 0.0);
    unital.set_t1_from_time(3, 1.0, std::numeric_limits<double>::infinity());
    BOOST_CHECK(unital.has_sampled_resets());
    BOOST_CHECK(unital.exact_channels_are_pauli_diagonal());
    BOOST_CHECK(InjectsExactChannels(propagator, unital));
    for (const double population : {0.0, 0.4, 0.6, 1.0})
    {
        auto biased = unital;
        biased.set_thermal_relaxation(0, 1.0, 1.0, 2.0, population);
        BOOST_CHECK(!biased.exact_channels_are_pauli_diagonal());
        BOOST_CHECK(!InjectsExactChannels(propagator, biased));
        biased = unital;
        biased.set_idle_noise(0, 25.0, 30.0, population);
        BOOST_CHECK(!biased.exact_channels_are_pauli_diagonal());
    }
    auto damped = unital;
    damped.set_t1(2, 1e-6);
    BOOST_CHECK(!damped.exact_channels_are_pauli_diagonal());

    // Relaxation is injected where the circuit gates or delays a qubit.
    const auto extended = [](const std::shared_ptr<Circuits::Circuit<>> &base, const std::shared_ptr<Circuits::IOperation<>> &op) {
        auto result = std::make_shared<Circuits::Circuit<>>(base->GetOperations());
        result->AddOperation(op);
        return result;
    };
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(0, 0.4));
    noise::NoiseModel elsewhere;
    elsewhere.set_all_depolarizing(3, 0.1);
    elsewhere.set_t1(1, 0.2);
    elsewhere.set_idle_noise(2, 25.0, 25.0, 0.1);
    BOOST_CHECK(!elsewhere.exact_channels_are_pauli_diagonal());
    BOOST_CHECK(elsewhere.exact_channels_are_pauli_diagonal_on(*circuit));
    BOOST_CHECK(!InjectsExactChannels(propagator, elsewhere));
    BOOST_CHECK(InjectsExactChannels(propagator, elsewhere, *circuit));
    // Only a backend that applies channels exactly is asked.
    BOOST_CHECK(!InjectsExactChannels(statevector, elsewhere, *circuit));
    const auto resolve = [&](const std::shared_ptr<Circuits::Circuit<>> &target) {
        return MaestroExecution::ResolveNoiseRealizations(std::nullopt, elsewhere, propagator, MaestroExecution::NoiseLayer::kCombined, 100, target, true);
    };
    auto policy = resolve(circuit);
    BOOST_CHECK(policy.exact_channels && !policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    const auto gated = extended(circuit, std::make_shared<Circuits::CXGate<>>(0, 1));
    const auto delayed = extended(circuit, std::make_shared<Circuits::Delay<>>(2, 1e-3));
    for (const auto &relaxing : {gated, delayed})
    {
        BOOST_CHECK(!elsewhere.exact_channels_are_pauli_diagonal_on(*relaxing));
        BOOST_CHECK(!InjectsExactChannels(propagator, elsewhere, *relaxing));
        policy = resolve(relaxing);
        BOOST_CHECK(!policy.exact_channels && policy.stochastic_noise);
        BOOST_CHECK_EQUAL(policy.realizations, 100u);
    }
    // A delay of no duration relaxes nothing.
    BOOST_CHECK(elsewhere.exact_channels_are_pauli_diagonal_on(*extended(circuit, std::make_shared<Circuits::Delay<>>(2, 0.0))));

    // A Kraus instruction has no sampled form: the relaxation beside it is
    // then exact too, as with a Kraus channel of the model. A Pauli channel
    // instruction can be sampled and decides nothing.
    const auto instruction = [](const Channel &channel) { return std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, channel); };
    const auto kraus = extended(gated, instruction(Channel::AmplitudeDamping(0.2)));
    const auto pauli = extended(gated, instruction(Channel::Pauli(0.1, 0.0, 0.1)));
    BOOST_CHECK(MaestroExecution::HasKrausInstruction(*kraus));
    BOOST_CHECK(!MaestroExecution::HasKrausInstruction(*pauli));
    BOOST_CHECK(InjectsExactChannels(propagator, elsewhere, *kraus));
    BOOST_CHECK(!InjectsExactChannels(propagator, elsewhere, *pauli));
    BOOST_CHECK(!InjectsExactChannels(statevector, elsewhere, *kraus));
    policy = resolve(kraus);
    BOOST_CHECK(policy.exact_channels && !policy.stochastic_noise);
    BOOST_CHECK_EQUAL(policy.realizations, 1u);
    // A nested instruction counts, and so does any circuit of several (the
    // suffixes of a checkpoint): one that draws relaxation samples them all,
    // unless one holds a Kraus instruction.
    auto nested = std::make_shared<Circuits::Circuit<>>();
    nested->AddOperation(kraus);
    BOOST_CHECK(MaestroExecution::HasKrausInstruction(*nested));
    using MaestroExecution::NoisyCircuits;
    BOOST_CHECK(InjectsExactChannels(propagator, elsewhere, NoisyCircuits{circuit.get(), circuit.get()}));
    BOOST_CHECK(!InjectsExactChannels(propagator, elsewhere, NoisyCircuits{circuit.get(), gated.get()}));
    BOOST_CHECK(!InjectsExactChannels(propagator, elsewhere, NoisyCircuits{gated.get(), pauli.get()}));
    BOOST_CHECK(InjectsExactChannels(propagator, elsewhere, NoisyCircuits{gated.get(), nested.get()}));
}

BOOST_AUTO_TEST_CASE(NoiseAddInfersExactChannelsOnThePauliPropagator)
{
    using Method = Simulators::SimulationType;
    const auto network = [](Method method) {
        auto result = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{2}, std::vector<size_t>{2});
        result->Configure("seed", "9");
        result->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        result->SetMaxSimulators(1);
        return result;
    };
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    circuit->AddOperation(std::make_shared<Circuits::RyGate<>>(1, 0.7));
    const std::vector<std::string> observables{"ZZ", "XX", "ZI", "IY"};

    noise::NoiseModel pauli;
    pauli.set_all_depolarizing(2, 0.1);
    pauli.set_qubit_noise(0, 0.01, 0.03, 0.05);
    pauli.set_2q_depolarizing(0, 1, 0.04);
    pauli.set_phase_damping(1, 0.2);
    noise::NoiseAdd add, reference;
    add.seed(3);
    reference.seed(3);
    auto values = add.full_noise_estimate(circuit, network(Method::kPauliPropagator), 0, observables, pauli);
    BOOST_CHECK(add.last_realization_policy().exact_channels);
    BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
    auto expected = reference.full_noise_estimate(circuit, network(Method::kDensityMatrix), 0, observables, pauli);
    BOOST_REQUIRE_EQUAL(values.size(), expected.size());
    for (size_t i = 0; i < values.size(); ++i)
        BOOST_CHECK_SMALL(values[i] - expected[i], 1e-9);

    // Relaxation is sampled by default and exact on request, with T1 < T2 kept.
    noise::NoiseModel thermal;
    thermal.set_thermal_relaxation(0, 1.0, 3.0, 5.0, 0.1);
    thermal.set_thermal_relaxation(1, 1.0, 4.0, 2.0);
    thermal.set_thermal_relaxation_2q(0, 2.0, 3.0, 5.0, 0.1);
    thermal.set_thermal_relaxation_2q(1, 2.0, 4.0, 2.0);
    int warnings = 0;
    add.set_warning_handler([&](const std::string &) { ++warnings; });
    add.full_noise_estimate(circuit, network(Method::kPauliPropagator), 0, observables, thermal, 4);
    BOOST_CHECK(!add.last_realization_policy().exact_channels);
    BOOST_CHECK_EQUAL(warnings, 1); // sampled: T2 is clamped to T1 on qubit 0
    add.set_exact_channels(true);
    values = add.full_noise_estimate(circuit, network(Method::kPauliPropagator), 0, observables, thermal);
    BOOST_CHECK(add.last_realization_policy().exact_channels);
    BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
    BOOST_CHECK_EQUAL(warnings, 1);
    expected = reference.full_noise_estimate(circuit, network(Method::kDensityMatrix), 0, observables, thermal);
    for (size_t i = 0; i < values.size(); ++i)
        BOOST_CHECK_SMALL(values[i] - expected[i], 1e-9);
    add.set_automatic_channels();

    // Channels with no sampled form were rejected on this method before.
    noise::NoiseModel damping;
    damping.set_generalized_amplitude_damping(0, 0.2, 0.1);
    damping.set_correlated_phase_flip(0, 1, 0.05, 0.5);
    values = add.full_noise_estimate(circuit, network(Method::kPauliPropagator), 0, observables, damping);
    BOOST_CHECK(add.last_realization_policy().exact_channels);
    BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
    expected = reference.full_noise_estimate(circuit, network(Method::kDensityMatrix), 0, observables, damping);
    for (size_t i = 0; i < values.size(); ++i)
        BOOST_CHECK_SMALL(values[i] - expected[i], 1e-9);
}

BOOST_AUTO_TEST_SUITE_END()

// ---------------------------------------------------------------- method agreement
//
// NoiseAdd, the network path Composer's bindings call, realizes one noise model
// as trajectories on pure-state methods, as exact channels on the density
// matrix and MPO, and as Pauli frames on the stabilizer (counts_policy
// frame_v1). For one circuit and one model every method must give the same
// count distribution and expectation values, up to sampling error; the density
// matrix is the reference. Runs are seeded, so the comparisons are
// deterministic. Their bounds sit five standard errors out, and negative
// controls show that the same checks reject noise 1.5 times stronger.

namespace
{
using Method = Simulators::SimulationType;
using Histogram = std::map<std::string, double>;
using CircuitPtr = std::shared_ptr<Circuits::Circuit<>>;

constexpr double kSigmas = 5.0;
constexpr int kAgreementShots = 20000;
constexpr int kReferenceShots = 40000;
const std::vector<std::string> kAgreementObservables{"XIIX", "ZXIZ", "XYZI", "XXXX", "YXIY", "IIZI", "IZZX", "YYZZ"};

template <class Gate, class... Args> void Append(const CircuitPtr &circuit, Args... args)
{
    circuit->AddOperation(std::make_shared<Gate>(args...));
}

void Measure(const CircuitPtr &circuit, size_t qubit, size_t bit)
{
    circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{qubit, bit}}));
}

// A Hadamard that counts its executions. Pauli frames never execute gates, so
// a zero count shows that the frame sampler, not a fallback, produced counts.
class CountedHadamard : public Circuits::HadamardGate<>
{
  public:
    CountedHadamard(Types::qubit_t qubit, std::shared_ptr<std::atomic<size_t>> calls) : HadamardGate(qubit), calls(std::move(calls))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
    {
        ++*calls;
        HadamardGate::Execute(simulator, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<CountedHadamard>(*this);
    }

    std::shared_ptr<Circuits::IOperation<>> Remap(const std::unordered_map<Types::qubit_t, Types::qubit_t> &qubits,
                                                  const std::unordered_map<Types::qubit_t, Types::qubit_t> & = {}) const override
    {
        const auto mapped = qubits.find(GetQubit());
        return std::make_shared<CountedHadamard>(mapped == qubits.end() ? GetQubit() : mapped->second, calls);
    }

  private:
    std::shared_ptr<std::atomic<size_t>> calls;
};

// A generic four-qubit circuit: rotations, T gates and entanglers.
CircuitPtr GenericCircuit(bool measure, bool delays = false)
{
    auto c = std::make_shared<Circuits::Circuit<>>();
    Append<Circuits::HadamardGate<>>(c, 0);
    Append<Circuits::RyGate<>>(c, 1, 0.7);
    Append<Circuits::RxGate<>>(c, 2, 1.1);
    Append<Circuits::HadamardGate<>>(c, 3);
    Append<Circuits::CXGate<>>(c, 0, 1);
    Append<Circuits::TGate<>>(c, 1);
    Append<Circuits::CZGate<>>(c, 1, 2);
    Append<Circuits::SGate<>>(c, 2);
    Append<Circuits::RzGate<>>(c, 3, 0.5);
    if (delays)
    {
        Append<Circuits::Delay<>>(c, 3, 3.0);
        Append<Circuits::Delay<>>(c, 0, 1.5);
    }
    Append<Circuits::CXGate<>>(c, 2, 3);
    Append<Circuits::RyGate<>>(c, 0, 0.4);
    Append<Circuits::CXGate<>>(c, 3, 0);
    Append<Circuits::HadamardGate<>>(c, 1);
    Append<Circuits::TGate<>>(c, 2);
    if (delays)
        Append<Circuits::Delay<>>(c, 2, 2.0);
    if (measure)
        for (size_t q = 0; q < 4; ++q)
            Measure(c, q, q);
    return c;
}

// Noiseless records are (a, 1-a, 1-a, 1) for a random bit a, or with dynamic
// (b, 1-a, 1-a, 1-b, a). Correlated, partly deterministic records make every
// error visible; the uniform output of a generic Clifford circuit would hide
// the noise strength.
CircuitPtr CliffordCircuit(bool dynamic, const std::shared_ptr<std::atomic<size_t>> &calls)
{
    auto c = std::make_shared<Circuits::Circuit<>>();
    c->AddOperation(std::make_shared<CountedHadamard>(0, calls));
    Append<Circuits::CXGate<>>(c, 0, 1);
    Append<Circuits::CXGate<>>(c, 1, 2);
    Append<Circuits::XGate<>>(c, 3);
    Append<Circuits::SGate<>>(c, 0);
    Append<Circuits::CZGate<>>(c, 0, 3);
    Append<Circuits::SdgGate<>>(c, 0);
    Append<Circuits::CYGate<>>(c, 3, 1);
    Append<Circuits::YGate<>>(c, 2);
    Append<Circuits::SGate<>>(c, 1);
    if (dynamic)
    {
        Measure(c, 0, 4);
        c->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
        Append<Circuits::HadamardGate<>>(c, 0);
        Append<Circuits::CXGate<>>(c, 0, 3);
    }
    for (size_t q = 0; q < 4; ++q)
        Measure(c, q, q);
    return c;
}

// Mid-circuit measurement, classical control and a reused qubit.
CircuitPtr DynamicCircuit()
{
    auto c = std::make_shared<Circuits::Circuit<>>();
    Append<Circuits::HadamardGate<>>(c, 0);
    Append<Circuits::TGate<>>(c, 0);
    Append<Circuits::CXGate<>>(c, 0, 1);
    Append<Circuits::RyGate<>>(c, 2, 0.8);
    Measure(c, 0, 0);
    c->AddOperation(std::make_shared<Circuits::ConditionalGate<>>(std::make_shared<Circuits::XGate<>>(2),
                                                                  std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{0}, std::vector<bool>{true})));
    c->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
    Append<Circuits::HadamardGate<>>(c, 0);
    Append<Circuits::CZGate<>>(c, 0, 2);
    Append<Circuits::TGate<>>(c, 2);
    Append<Circuits::HadamardGate<>>(c, 2);
    Append<Circuits::CXGate<>>(c, 2, 1);
    Measure(c, 0, 1);
    Measure(c, 1, 2);
    Measure(c, 2, 3);
    return c;
}

// Pauli channels of every kind the model offers, plus readout error.
noise::NoiseModel PauliModel(double scale, int qubits = 4)
{
    noise::NoiseModel m;
    m.set_all_depolarizing(qubits, 0.02 * scale);
    m.set_dephasing(1, 0.03 * scale);
    m.set_bit_flip(2, 0.02 * scale);
    m.set_qubit_noise(0, 0.01 * scale, 0.005 * scale, 0.02 * scale);
    m.set_2q_depolarizing(0, 1, 0.05 * scale);
    m.set_all_2q_gate_depolarizing(qubits, 0.01 * scale);
    m.set_phase_damping(2, 0.1 * scale);
    m.set_readout_error(qubits - 1, 0.03 * scale, 0.06 * scale);
    m.set_readout_error(0, 0.02 * scale, 0.01 * scale);
    return m;
}

// Thermal relaxation with T2 <= T1 (where sampled resets are exact), idle
// noise on delays, crosstalk, depolarizing and readout error. set_t1 is left
// out: its sampled reset damps coherence by 1 - gamma, not sqrt(1 - gamma).
noise::NoiseModel RelaxationModel(double scale)
{
    noise::NoiseModel m;
    const double times[4][3] = {{20.0, 15.0, 0.0}, {30.0, 30.0, 0.1}, {25.0, 12.0, 0.05}, {40.0, 20.0, 0.0}};
    for (int q = 0; q < 4; ++q)
    {
        m.set_thermal_relaxation(q, 1.0 * scale, times[q][0], times[q][1], times[q][2]);
        m.set_thermal_relaxation_2q(q, 2.5 * scale, times[q][0], times[q][1], times[q][2]);
    }
    m.set_idle_noise(3, 30.0 / scale, 20.0 / scale, 0.0, 0.01);
    m.set_idle_noise(0, 25.0 / scale, 25.0 / scale, 0.1);
    m.set_crosstalk(0, 1, 0.08 * scale);
    m.set_all_depolarizing(4, 0.005 * scale);
    m.set_readout_error(1, 0.02 * scale, 0.05 * scale);
    return m;
}

noise::NoiseModel DynamicModel(double scale)
{
    auto m = PauliModel(scale, 3);
    const double t1[3] = {20.0, 30.0, 25.0};
    for (int q = 0; q < 3; ++q)
        m.set_thermal_relaxation(q, 1.0 * scale, t1[q], 0.8 * t1[q], 0.05);
    return m;
}

// A four-qubit QCSim network that keeps the requested method.
std::shared_ptr<Network::SimpleDisconnectedNetwork<>> AgreementNetwork(Method method, size_t bits, uint64_t seed, const char *policy = "legacy")
{
    auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{4}, std::vector<size_t>{bits});
    network->Configure("seed", std::to_string(seed).c_str());
    network->Configure("counts_policy", policy);
    network->SetOptimizeSimulator(false);
    network->GetController()->SetOptimizeCircuit(false);
    network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
    return network;
}

// Counts from one method: an independent realization per shot on sampled
// paths, the inferred exact channels on the density matrix and MPO.
Histogram NoisyCounts(const CircuitPtr &circuit, const noise::NoiseModel &model, bool combined, Method method, bool exact, int shots, size_t bits,
                      uint64_t seed, const char *policy = "legacy", bool frames = false)
{
    auto network = AgreementNetwork(method, bits, seed, policy);
    noise::NoiseAdd add;
    add.seed(static_cast<unsigned>(seed * 7 + 3));
    const std::optional<int> realizations = exact ? std::nullopt : std::optional<int>(shots);
    const auto results =
        combined ? add.full_noise_execute(circuit, network, 0, model, shots, realizations) : add.noisy_execute(circuit, network, 0, model, shots, realizations);
    BOOST_CHECK_EQUAL(add.last_realization_policy().exact_channels, exact);
    BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, exact ? 1u : static_cast<size_t>(shots));
    // Frames run no network, so they record no backend.
    BOOST_CHECK_EQUAL(add.last_execution().valid, !frames);
    if (add.last_execution().valid)
        BOOST_CHECK(add.last_execution().method == method);
    Histogram histogram;
    double total = 0;
    for (const auto &[outcome, count] : results)
    {
        std::string key(bits, '0');
        for (size_t bit = 0; bit < bits && bit < outcome.size(); ++bit)
            key[bit] = outcome[bit] ? '1' : '0';
        histogram[key] += static_cast<double>(count);
        total += static_cast<double>(count);
    }
    BOOST_CHECK_EQUAL(total, static_cast<double>(shots));
    return histogram;
}

// Two-sample chi-square homogeneity statistic and its degrees of freedom.
// Outcomes seen fewer than ten times in both samples together share one bin.
std::pair<double, size_t> ChiSquare(const Histogram &a, const Histogram &b)
{
    double na = 0, nb = 0;
    std::map<std::string, std::pair<double, double>> joined;
    for (const auto &[key, count] : a)
    {
        na += count;
        joined[key].first = count;
    }
    for (const auto &[key, count] : b)
    {
        nb += count;
        joined[key].second = count;
    }
    const double ka = std::sqrt(nb / na), kb = std::sqrt(na / nb);
    double statistic = 0, rareA = 0, rareB = 0;
    size_t bins = 0;
    const auto add = [&](double x, double y) {
        statistic += (ka * x - kb * y) * (ka * x - kb * y) / (x + y);
        ++bins;
    };
    for (const auto &[key, counts] : joined)
    {
        if (counts.first + counts.second < 10)
        {
            rareA += counts.first;
            rareB += counts.second;
        }
        else
            add(counts.first, counts.second);
    }
    if (rareA + rareB > 0)
        add(rareA, rareB);
    return {statistic, bins > 1 ? bins - 1 : 1};
}

// The chi-square quantile kSigmas standard normal deviations out (Wilson-Hilferty).
double ChiSquareBound(size_t dof)
{
    const double h = 2.0 / (9.0 * static_cast<double>(dof));
    return static_cast<double>(dof) * std::pow(1.0 - h + kSigmas * std::sqrt(h), 3);
}

struct Statistic
{
    std::string name;
    double value, error;
};

// Averages, correlations and spread of a histogram: each bit's probability of
// one, every pair's <Z_i Z_j>, and the mean and variance of the Hamming weight.
std::vector<Statistic> Summarize(const Histogram &histogram)
{
    const size_t width = histogram.begin()->first.size();
    double shots = 0;
    for (const auto &[key, count] : histogram)
        shots += count;
    const auto average = [&](const auto &f) {
        double sum = 0;
        for (const auto &[key, count] : histogram)
            sum += count * f(key);
        return sum / shots;
    };
    const auto bit = [](const std::string &key, size_t i) { return key[i] == '1' ? 1.0 : 0.0; };
    std::vector<Statistic> stats;
    for (size_t i = 0; i < width; ++i)
    {
        const double p = average([&](const std::string &key) { return bit(key, i); });
        stats.push_back({"P(bit " + std::to_string(i) + " = 1)", p, std::sqrt(p * (1 - p) / shots)});
    }
    for (size_t i = 0; i < width; ++i)
        for (size_t j = i + 1; j < width; ++j)
        {
            const double c = average([&](const std::string &key) { return (1 - 2 * bit(key, i)) * (1 - 2 * bit(key, j)); });
            stats.push_back({"<Z" + std::to_string(i) + "Z" + std::to_string(j) + ">", c, std::sqrt(std::max(1 - c * c, 0.0) / shots)});
        }
    const auto weight = [&](const std::string &key) {
        double w = 0;
        for (size_t i = 0; i < width; ++i)
            w += bit(key, i);
        return w;
    };
    const double mean = average(weight);
    const double variance = average([&](const std::string &key) { return std::pow(weight(key) - mean, 2); });
    const double fourth = average([&](const std::string &key) { return std::pow(weight(key) - mean, 4); });
    stats.push_back({"weight mean", mean, std::sqrt(variance / shots)});
    stats.push_back({"weight variance", variance, std::sqrt(std::max(fourth - variance * variance, 0.0) / shots)});
    return stats;
}

// Every way two histograms differ by more than sampling explains.
std::vector<std::string> CountDifferences(const Histogram &a, const Histogram &b, std::ostringstream &report)
{
    std::vector<std::string> failures;
    const auto [statistic, dof] = ChiSquare(a, b);
    const double bound = ChiSquareBound(dof);
    report << "chi2 " << statistic << " (dof " << dof << ", bound " << bound << ")";
    if (statistic > bound)
        failures.push_back("distributions differ: chi2 " + std::to_string(statistic) + " > " + std::to_string(bound));
    if (a.begin()->first.size() != b.begin()->first.size())
    {
        failures.push_back("different record widths");
        return failures;
    }
    const auto sa = Summarize(a), sb = Summarize(b);
    for (size_t i = 0; i < sa.size(); ++i)
    {
        const double error = std::hypot(sa[i].error, sb[i].error);
        const double z = error > 0 ? std::abs(sa[i].value - sb[i].value) / error : (sa[i].value == sb[i].value ? 0.0 : INFINITY);
        report << "; " << sa[i].name << " " << sa[i].value << " vs " << sb[i].value << " (z " << z << ")";
        if (z > kSigmas)
            failures.push_back(sa[i].name + ": " + std::to_string(sa[i].value) + " vs " + std::to_string(sb[i].value) + ", z " + std::to_string(z));
    }
    return failures;
}

void CheckSameCounts(const Histogram &a, const Histogram &b, const std::string &label)
{
    std::ostringstream report;
    const auto failures = CountDifferences(a, b, report);
    std::ostringstream message;
    for (const auto &failure : failures)
        message << failure << "; ";
    BOOST_CHECK_MESSAGE(failures.empty(), label << ": " << message.str() << "statistics: " << report.str());
}

void CheckDifferentCounts(const Histogram &a, const Histogram &b, const std::string &label)
{
    std::ostringstream report;
    BOOST_CHECK_MESSAGE(!CountDifferences(a, b, report).empty(),
                        label << ": stronger noise passed every check, so the comparison has no power: " << report.str());
}

std::string Name(Method method)
{
    switch (method)
    {
    case Method::kStatevector:
        return "statevector";
    case Method::kMatrixProductState:
        return "MPS";
    case Method::kStabilizer:
        return "stabilizer";
    case Method::kTensorNetwork:
        return "tensor network";
    case Method::kPauliPropagator:
        return "Pauli propagator";
    case Method::kExtendedStabilizer:
        return "extended stabilizer";
    case Method::kPathIntegral:
        return "path integral";
    case Method::kDensityMatrix:
        return "density matrix";
    case Method::kMatrixProductOperator:
        return "MPO";
    default:
        return "other";
    }
}

// The exact expectation values: one realization of the exact channels, which
// are inferred from the backend unless the caller asks for them.
std::vector<double> ExactEstimate(const CircuitPtr &circuit, const noise::NoiseModel &model, bool combined, Method method, bool request = false)
{
    auto network = AgreementNetwork(method, 4, 5);
    noise::NoiseAdd add;
    add.seed(5);
    if (request)
        add.set_exact_channels(true);
    const auto values = combined ? add.full_noise_estimate(circuit, network, 0, kAgreementObservables, model)
                                 : add.noisy_estimate_montecarlo(circuit, network, 0, kAgreementObservables, model);
    BOOST_CHECK(add.last_realization_policy().exact_channels);
    BOOST_CHECK_EQUAL(add.last_realization_policy().realizations, 1u);
    BOOST_REQUIRE_EQUAL(values.size(), kAgreementObservables.size());
    return values;
}

struct BatchedEstimate
{
    std::vector<double> mean, error;
};

// NoiseAdd returns only realization averages, so the standard error comes
// from independently seeded batches.
BatchedEstimate MonteCarloEstimate(const CircuitPtr &circuit, const noise::NoiseModel &model, bool combined, Method method, size_t batches, int realizations,
                                   uint64_t seed)
{
    const size_t n = kAgreementObservables.size();
    std::vector<double> sum(n, 0.0), squares(n, 0.0);
    for (size_t batch = 0; batch < batches; ++batch)
    {
        auto network = AgreementNetwork(method, 4, seed + batch);
        noise::NoiseAdd add;
        add.seed(static_cast<unsigned>(seed * 1009 + batch));
        // These are the sampled trajectories. The Pauli propagator would
        // otherwise evaluate the Pauli model exactly: see ExactEstimate.
        add.set_exact_channels(false);
        const auto values = combined ? add.full_noise_estimate(circuit, network, 0, kAgreementObservables, model, realizations)
                                     : add.noisy_estimate_montecarlo(circuit, network, 0, kAgreementObservables, model, realizations);
        BOOST_CHECK(!add.last_realization_policy().exact_channels);
        BOOST_REQUIRE_EQUAL(values.size(), n);
        for (size_t i = 0; i < n; ++i)
        {
            sum[i] += values[i];
            squares[i] += values[i] * values[i];
        }
    }
    BatchedEstimate result;
    const double count = static_cast<double>(batches);
    for (size_t i = 0; i < n; ++i)
    {
        const double mean = sum[i] / count;
        result.mean.push_back(mean);
        result.error.push_back(std::sqrt(std::max(squares[i] - count * mean * mean, 0.0) / (count - 1) / count));
    }
    return result;
}

// The observables whose Monte Carlo mean lies more than kSigmas standard
// errors from the exact value.
std::vector<std::string> EstimateDeviations(const BatchedEstimate &estimate, const std::vector<double> &exact)
{
    std::vector<std::string> deviations;
    for (size_t i = 0; i < exact.size(); ++i)
        if (std::abs(estimate.mean[i] - exact[i]) > kSigmas * estimate.error[i] + 1e-9)
        {
            std::ostringstream line;
            line << "<" << kAgreementObservables[i] << "> " << estimate.mean[i] << " +- " << estimate.error[i] << " vs exact " << exact[i];
            deviations.push_back(line.str());
        }
    return deviations;
}
} // namespace

BOOST_AUTO_TEST_SUITE(noise_method_agreement_tests)

BOOST_AUTO_TEST_CASE(NoiseAddCountsAgreeAcrossMethods)
{
    struct Case
    {
        Method method;
        bool exact;
        int shots;
    };

    const std::vector<Case> cases{{Method::kStatevector, false, kAgreementShots},
                                  {Method::kMatrixProductState, false, kAgreementShots},
                                  {Method::kMatrixProductOperator, true, kAgreementShots},
                                  {Method::kTensorNetwork, false, 5000},
                                  {Method::kPauliPropagator, false, 2000},
                                  {Method::kPathIntegral, false, kAgreementShots},
                                  {Method::kExtendedStabilizer, false, kAgreementShots}};
    for (const bool relaxation : {false, true})
    {
        const auto circuit = GenericCircuit(true, relaxation);
        const auto model = relaxation ? RelaxationModel(1.0) : PauliModel(1.0);
        const auto reference = NoisyCounts(circuit, model, relaxation, Method::kDensityMatrix, true, kReferenceShots, 4, 4242);
        uint64_t seed = relaxation ? 200 : 100;
        for (const auto &test : cases)
        {
            // The Pauli propagator applies the Pauli model's channels, all
            // diagonal in the Pauli basis, to the observable: exact, in one
            // realization. Relaxation keeps its sampled trajectories there.
            const bool exact = test.exact || (test.method == Method::kPauliPropagator && !relaxation);
            CheckSameCounts(NoisyCounts(circuit, model, relaxation, test.method, exact, test.shots, 4, seed += 17), reference,
                            Name(test.method) + (relaxation ? " relaxation" : " Pauli") + " counts vs density matrix");
        }
        // Negative control: 1.5x noise on the trajectories is detected.
        const auto stronger = relaxation ? RelaxationModel(1.5) : PauliModel(1.5);
        CheckDifferentCounts(NoisyCounts(circuit, stronger, relaxation, Method::kStatevector, false, kAgreementShots, 4, seed + 1), reference,
                             std::string("statevector with 1.5x ") + (relaxation ? "relaxation" : "Pauli noise"));
    }
}

BOOST_AUTO_TEST_CASE(NoiseAddDynamicCircuitCountsAgreeAcrossMethods)
{
    // Each shot follows its own measured branch, sampled on trajectories and
    // exactly on mixed states.
    const auto circuit = DynamicCircuit();
    const auto model = DynamicModel(1.0);
    const auto reference = NoisyCounts(circuit, model, true, Method::kDensityMatrix, true, kReferenceShots, 4, 4244);
    uint64_t seed = 400;
    for (const auto &[method, exact] : std::vector<std::pair<Method, bool>>{
             {Method::kStatevector, false}, {Method::kMatrixProductState, false}, {Method::kMatrixProductOperator, true}, {Method::kExtendedStabilizer, false}})
        CheckSameCounts(NoisyCounts(circuit, model, true, method, exact, kAgreementShots, 4, seed += 17), reference,
                        Name(method) + " dynamic circuit vs density matrix");
    CheckDifferentCounts(NoisyCounts(circuit, DynamicModel(1.5), true, Method::kStatevector, false, kAgreementShots, 4, seed + 1), reference,
                         "statevector dynamic circuit with 1.5x noise");
}

BOOST_AUTO_TEST_CASE(NoiseAddCliffordFramesAgreeWithDensityMatrix)
{
    for (const bool dynamic : {false, true})
    {
        const size_t bits = dynamic ? 5 : 4;
        const std::string kind = dynamic ? "dynamic Clifford" : "Clifford";
        auto calls = std::make_shared<std::atomic<size_t>>(0);
        const auto circuit = CliffordCircuit(dynamic, calls);
        const auto model = PauliModel(1.0);
        const auto reference = NoisyCounts(circuit, model, false, Method::kDensityMatrix, true, kReferenceShots, bits, 4243);
        calls->store(0);
        const auto frames = NoisyCounts(circuit, model, false, Method::kStabilizer, false, kAgreementShots, bits, 301, "frame_v1", true);
        BOOST_CHECK_MESSAGE(calls->load() == 0, kind << ": frame_v1 executed gates instead of sampling frames");
        const auto trajectories = NoisyCounts(circuit, model, false, Method::kStabilizer, false, kAgreementShots, bits, 302);
        BOOST_CHECK_MESSAGE(calls->load() > 0, kind << ": legacy stabilizer counts did not execute the circuit");
        CheckSameCounts(frames, reference, kind + " stabilizer frames vs density matrix");
        CheckSameCounts(trajectories, reference, kind + " stabilizer trajectories vs density matrix");
        CheckSameCounts(frames, trajectories, kind + " stabilizer frames vs stabilizer trajectories");
        CheckSameCounts(NoisyCounts(circuit, model, false, Method::kStatevector, false, kAgreementShots, bits, 303), reference,
                        kind + " statevector vs density matrix");
        CheckSameCounts(NoisyCounts(circuit, model, false, Method::kMatrixProductOperator, true, kAgreementShots, bits, 304), reference,
                        kind + " MPO vs density matrix");
        calls->store(0);
        CheckDifferentCounts(NoisyCounts(circuit, PauliModel(1.5), false, Method::kStabilizer, false, kAgreementShots, bits, 305, "frame_v1", true), reference,
                             kind + " stabilizer frames with 1.5x noise");
        BOOST_CHECK_EQUAL(calls->load(), 0u);
    }
}

BOOST_AUTO_TEST_CASE(NoiseAddEstimatesAgreeAcrossMethods)
{
    struct Case
    {
        Method method;
        size_t batches;
        int realizations;
    };

    const std::vector<Case> cases{{Method::kStatevector, 20, 100},    {Method::kMatrixProductState, 20, 100}, {Method::kTensorNetwork, 20, 100},
                                  {Method::kPauliPropagator, 10, 40}, {Method::kPathIntegral, 20, 100},       {Method::kExtendedStabilizer, 20, 100}};
    for (const bool relaxation : {false, true})
    {
        const auto circuit = GenericCircuit(false, relaxation);
        const auto model = relaxation ? RelaxationModel(1.0) : PauliModel(1.0);
        const std::string kind = relaxation ? " relaxation" : " Pauli";
        const auto exact = ExactEstimate(circuit, model, relaxation, Method::kDensityMatrix);
        const auto mpo = ExactEstimate(circuit, model, relaxation, Method::kMatrixProductOperator);
        for (size_t i = 0; i < exact.size(); ++i)
            BOOST_CHECK_MESSAGE(std::abs(mpo[i] - exact[i]) < 1e-8, "MPO" << kind << " <" << kAgreementObservables[i] << ">: " << mpo[i] << " vs " << exact[i]);
        // The Pauli propagator holds no mixed state, yet one realization of the
        // channels applied to the observable gives the same ensemble average:
        // inferred for the Pauli model, on request for relaxation.
        const auto propagator = ExactEstimate(circuit, model, relaxation, Method::kPauliPropagator, relaxation);
        for (size_t i = 0; i < exact.size(); ++i)
            BOOST_CHECK_MESSAGE(std::abs(propagator[i] - exact[i]) < 1e-8,
                                "Pauli propagator exact" << kind << " <" << kAgreementObservables[i] << ">: " << propagator[i] << " vs " << exact[i]);
        uint64_t seed = relaxation ? 700 : 600;
        for (const auto &test : cases)
        {
            const auto deviations =
                EstimateDeviations(MonteCarloEstimate(circuit, model, relaxation, test.method, test.batches, test.realizations, seed += 31), exact);
            std::ostringstream message;
            for (const auto &line : deviations)
                message << line << "; ";
            BOOST_CHECK_MESSAGE(deviations.empty(), Name(test.method) << kind << " estimates vs density matrix: " << message.str());
        }
    }
    // Negative control: 1.5x Pauli noise moves some estimate past the bound.
    const auto stronger = MonteCarloEstimate(GenericCircuit(false), PauliModel(1.5), false, Method::kStatevector, 20, 100, 900);
    BOOST_CHECK_MESSAGE(!EstimateDeviations(stronger, ExactEstimate(GenericCircuit(false), PauliModel(1.0), false, Method::kDensityMatrix)).empty(),
                        "statevector estimates with 1.5x Pauli noise passed, so the comparison has no power");
}

BOOST_AUTO_TEST_CASE(NoiseAddTrajectoryMethodsAgreeRealizationByRealization)
{
    // Under one seed, Pauli noise injects the same circuits for every
    // pure-state method. Each realization is then a noiseless simulation, so
    // the averages agree to rounding: a difference is a simulator error on an
    // injected circuit, not sampling noise.
    const auto circuit = GenericCircuit(false);
    const auto model = PauliModel(1.0);
    const auto run = [&](Method method) {
        auto network = AgreementNetwork(method, 4, 77);
        noise::NoiseAdd add;
        add.seed(78);
        // Trajectories on every method: the Pauli propagator would otherwise
        // apply this Pauli model exactly instead of sampling it.
        add.set_exact_channels(false);
        return add.noisy_estimate_montecarlo(circuit, network, 0, kAgreementObservables, model, 200);
    };
    const auto reference = run(Method::kStatevector);
    for (const auto method :
         {Method::kMatrixProductState, Method::kTensorNetwork, Method::kPauliPropagator, Method::kPathIntegral, Method::kExtendedStabilizer})
    {
        const auto values = run(method);
        BOOST_REQUIRE_EQUAL(values.size(), reference.size());
        for (size_t i = 0; i < values.size(); ++i)
            BOOST_CHECK_MESSAGE(std::abs(values[i] - reference[i]) < 1e-9,
                                Name(method) << " <" << kAgreementObservables[i] << ">: " << values[i] << " vs statevector " << reference[i]);
    }
}

BOOST_AUTO_TEST_SUITE_END()
