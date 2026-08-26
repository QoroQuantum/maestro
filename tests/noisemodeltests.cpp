/**
 * @file noisemodeltests.cpp
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

#include <cmath>
#include <complex>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../Circuit/Circuit.h"
#include "../Simulators/Factory.h"
#include "../python/noise.h"

namespace {

constexpr double kTolerance = 1e-10;

std::shared_ptr<Simulators::ISimulator> MakeSimulator(
    Simulators::SimulationType simulationType, size_t numQubits) {
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim, simulationType);
  simulator->AllocateQubits(numQubits);
  simulator->Initialize();
  return simulator;
}

/** A one-qubit circuit that puts the qubit on the equator, so <X> == 1. */
std::shared_ptr<Circuits::Circuit<double>> HadamardCircuit() {
  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
  return circuit;
}

/**
 * <X> after running the noisy circuit on the density-matrix backend.
 * With the H first, this is exactly the coherence multiplier of whatever
 * noise the model injected after it.
 */
double ExactCoherence(const noise::NoiseModel& noiseModel) {
  const auto noisy = noise::inject_exact_noise(HadamardCircuit(), noiseModel);
  auto simulator =
      MakeSimulator(Simulators::SimulationType::kDensityMatrix, 1);
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
double SampledCoherence(const noise::NoiseModel& noiseModel,
                        size_t realizations, unsigned int seed) {
  std::mt19937 rng(seed);
  const auto circuit = HadamardCircuit();
  double total = 0.0;
  for (size_t realization = 0; realization < realizations; ++realization) {
    const auto noisy = noise::inject_noise(circuit, noiseModel, rng);
    auto simulator =
        MakeSimulator(Simulators::SimulationType::kStatevector, 1);
    Circuits::OperationState classicalState;
    noisy->Execute(simulator, classicalState);
    total += simulator->ExpectationValue("X");
  }
  return total / static_cast<double>(realizations);
}

}  // namespace

BOOST_AUTO_TEST_SUITE(noise_model_tests)

/**
 * The regression this file exists for.
 *
 * set_thermal_relaxation() must give the same coherence decay exp(-t/T2) on
 * both backend families. The exact path applies the CPTP channel; the sampled
 * path applies the reset+Z mixture. Configuring T1 and T2 separately cannot
 * do this, which is what the second half of the test pins down.
 */
BOOST_AUTO_TEST_CASE(ThermalRelaxationAgreesOnExactAndSampledBackends) {
  constexpr double duration = 4e-6;
  constexpr double t1 = 40e-6;
  constexpr double t2 = 25e-6;  // T2 <= T1, where the reset mixture is exact
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
  const double pz =
      0.5 * (1.0 - std::exp(-duration * (1.0 / t2 - 1.0 / t1)));
  noise::NoiseModel legacy;
  legacy.set_t1(0, gamma);
  legacy.set_dephasing(0, pz);

  const double legacyExact = ExactCoherence(legacy);
  BOOST_TEST(legacyExact > expected,
             "the legacy split configuration under-dephases on the exact "
             "backend, which is why set_thermal_relaxation exists");
  BOOST_CHECK_SMALL(
      legacyExact - std::sqrt(1.0 - gamma) * (1.0 - 2.0 * pz), kTolerance);
}

/** Two-qubit gates use the 2Q relaxation when one is configured. */
BOOST_AUTO_TEST_CASE(ThermalRelaxation2QOverridesTheAllGatesChannel) {
  constexpr double t1 = 50e-6;
  constexpr double t2 = 30e-6;

  noise::NoiseModel noiseModel;
  noiseModel.set_thermal_relaxation(0, 2e-8, t1, t2);
  noiseModel.set_thermal_relaxation_2q(0, 4e-7, t1, t2);

  const auto* oneQubit =
      noiseModel.get_thermal_relaxation_params(0, /*is_2q=*/false);
  const auto* twoQubit =
      noiseModel.get_thermal_relaxation_params(0, /*is_2q=*/true);
  BOOST_REQUIRE(oneQubit);
  BOOST_REQUIRE(twoQubit);
  BOOST_CHECK_CLOSE(oneQubit->duration, 2e-8, 1e-9);
  BOOST_CHECK_CLOSE(twoQubit->duration, 4e-7, 1e-9);

  // A qubit with no 2Q override falls back to the "all gates" relaxation.
  noise::NoiseModel fallback;
  fallback.set_thermal_relaxation(1, 2e-8, t1, t2);
  const auto* fallbackParams =
      fallback.get_thermal_relaxation_params(1, /*is_2q=*/true);
  BOOST_REQUIRE(fallbackParams);
  BOOST_CHECK_CLOSE(fallbackParams->duration, 2e-8, 1e-9);
}

/**
 * Phase damping and the stochastic phase flip are the same channel, so the
 * sampled backends realize it exactly instead of rejecting the model.
 */
BOOST_AUTO_TEST_CASE(PhaseDampingIsExactOnBothBackends) {
  constexpr double gamma = 0.36;
  const double expected = std::sqrt(1.0 - gamma);  // 0.8

  noise::NoiseModel noiseModel;
  noiseModel.set_phase_damping(0, gamma);
  BOOST_TEST(!noiseModel.has_additional_quantum_channels());
  BOOST_CHECK_SMALL(
      noiseModel.get_phase_damping_flip_probability(0) - 0.1, kTolerance);

  BOOST_CHECK_SMALL(ExactCoherence(noiseModel) - expected, kTolerance);
  BOOST_CHECK_SMALL(SampledCoherence(noiseModel, 4000, 999) - expected, 0.02);
}

/** Channels with no stochastic realization still require an exact backend. */
BOOST_AUTO_TEST_CASE(ExactOnlyChannelsStillRejectSampledBackends) {
  noise::NoiseModel noiseModel;
  noiseModel.set_generalized_amplitude_damping(0, 0.2, 0.1);
  BOOST_TEST(noiseModel.has_additional_quantum_channels());

  std::mt19937 rng(1);
  BOOST_CHECK_THROW(noise::inject_noise(HadamardCircuit(), noiseModel, rng),
                    std::invalid_argument);
  BOOST_CHECK_NO_THROW(
      noise::inject_exact_noise(HadamardCircuit(), noiseModel));
}

/**
 * Invalid probabilities must be rejected where they are configured. The
 * sampled injectors cannot detect them, so without this they would silently
 * skew the sampling while the exact path threw.
 */
BOOST_AUTO_TEST_CASE(SettersRejectUnphysicalParameters) {
  noise::NoiseModel noiseModel;

  BOOST_CHECK_THROW(noiseModel.set_depolarizing(0, 1.5),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_depolarizing(0, -0.1),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_dephasing(0, 2.0), std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_bit_flip(0, -1.0), std::invalid_argument);
  // Pauli probabilities must also sum to at most one.
  BOOST_CHECK_THROW(noiseModel.set_qubit_noise(0, 0.5, 0.4, 0.3),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_t1(0, 1.2), std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_2q_depolarizing(0, 1, 1.1),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_2q_depolarizing(0, 0, 0.1),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_readout_error(0, 0.1, 1.4),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_coherent_depolarizing(0, 1.3),
                    std::invalid_argument);
  // T2 > 2*T1 is not a completely positive map.
  BOOST_CHECK_THROW(noiseModel.set_thermal_relaxation(0, 1e-6, 10e-6, 30e-6),
                    std::invalid_argument);
  // Degenerate OU parameters used to produce silent inf/nan.
  BOOST_CHECK_THROW(noiseModel.set_correlated_ou(0, 1.0, 0.0, 1e-7),
                    std::invalid_argument);
  BOOST_CHECK_THROW(noiseModel.set_all_correlated_from_power(0, 1e-3, 0.5,
                                                             1e-7),
                    std::invalid_argument);

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
BOOST_AUTO_TEST_CASE(CoherentNoiseSignIsSystematicWithinARealization) {
  constexpr double angle = 0.2;
  noise::NoiseModel noiseModel;
  noiseModel.set_coherent_rotation(0, 0.0, 0.0, angle);

  auto circuit = std::make_shared<Circuits::Circuit<double>>();
  constexpr size_t gateCount = 8;
  for (size_t i = 0; i < gateCount; ++i)
    circuit->AddOperation(std::make_shared<Circuits::ZGate<>>(0));

  std::mt19937 rng(7);
  for (size_t trial = 0; trial < 20; ++trial) {
    const auto noisy = noise::inject_coherent_noise(circuit, noiseModel, rng);

    std::vector<double> angles;
    for (const auto& op : noisy->GetOperations()) {
      const auto rz = std::dynamic_pointer_cast<Circuits::RzGate<>>(op);
      if (rz) angles.push_back(rz->GetTheta());
    }

    BOOST_REQUIRE_EQUAL(angles.size(), gateCount);
    for (const double injected : angles)
      BOOST_TEST(injected == angles.front(),
                 "every injected rotation in a realization must share the "
                 "same sign");
    BOOST_CHECK_SMALL(std::abs(angles.front()) - angle, kTolerance);
  }
}

BOOST_AUTO_TEST_CASE(T1AndThermalRelaxationCannotStack) {
  noise::NoiseModel noiseModel;
  noiseModel.set_t1(0, 0.1);
  BOOST_CHECK_THROW(noiseModel.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6),
                    std::invalid_argument);

  noise::NoiseModel thermalFirst;
  thermalFirst.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6);
  BOOST_CHECK_THROW(thermalFirst.set_t1(0, 0.1), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(ThermalT2GreaterThanT1RequiresExactBackend) {
  constexpr double duration = 1e-6;
  constexpr double t1 = 10e-6;
  constexpr double t2 = 15e-6;  // T1 < T2 <= 2*T1
  noise::NoiseModel noiseModel;
  noiseModel.set_thermal_relaxation(0, duration, t1, t2);

  BOOST_TEST(!noiseModel.has_additional_quantum_channels());
  BOOST_TEST(noiseModel.has_thermal_in_sampled_overdephasing_regime());
  BOOST_TEST(noiseModel.requires_exact_quantum_channels());
  BOOST_TEST(!noiseModel.compute_damping_covers_model());

  std::mt19937 rng(1);
  BOOST_CHECK_THROW(noise::inject_noise(HadamardCircuit(), noiseModel, rng),
                    std::invalid_argument);
  BOOST_CHECK_NO_THROW(
      noise::inject_exact_noise(HadamardCircuit(), noiseModel));

  const double expected = std::exp(-duration / t2);
  BOOST_CHECK_SMALL(ExactCoherence(noiseModel) - expected, kTolerance);
}

BOOST_AUTO_TEST_CASE(ComputeDampingDoesNotCoverThermalModels) {
  noise::NoiseModel pauliOnly;
  pauliOnly.set_depolarizing(0, 0.01);
  BOOST_TEST(pauliOnly.compute_damping_covers_model());

  noise::NoiseModel thermal;
  thermal.set_thermal_relaxation(0, 1e-6, 40e-6, 25e-6);
  BOOST_TEST(!thermal.compute_damping_covers_model());
}

BOOST_AUTO_TEST_SUITE_END()
