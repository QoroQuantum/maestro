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

BOOST_AUTO_TEST_CASE(DelayExactChannelDivisibility) {
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

BOOST_AUTO_TEST_CASE(DelaySampledCoherenceMatchesExact) {
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
  for (size_t r = 0; r < realizations; ++r) {
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

BOOST_AUTO_TEST_CASE(CorrelatedNoiseStationaryInitVariance) {
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

  for (size_t i = 0; i < N; ++i) {
    auto out_stat = noise::inject_correlated_noise(circ, nm_stationary, rng);
    auto out_cold = noise::inject_correlated_noise(circ, nm_cold, rng);

    double angle_stat = 0.0;
    for (const auto &op : out_stat->GetOperations()) {
      if (auto rz = std::dynamic_pointer_cast<Circuits::RzGate<double>>(op))
        angle_stat = rz->GetTheta();
    }
    sum_stat += angle_stat;
    sum_sq_stat += angle_stat * angle_stat;

    double angle_cold = 0.0;
    for (const auto &op : out_cold->GetOperations()) {
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

BOOST_AUTO_TEST_CASE(MultiBandOUConfigurationAndSuperposition) {
  constexpr double gate_time = 100e-9;
  std::vector<std::pair<double, double>> bands = {
      {10.0, 2.0},
      {20.0, 5.0}
  };

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
  for (size_t i = 0; i < N; ++i) {
    auto out = noise::inject_correlated_noise(circ, nm, rng);
    double angle = 0.0;
    for (const auto &op : out->GetOperations()) {
      if (auto rz = std::dynamic_pointer_cast<Circuits::RzGate<double>>(op))
        angle = rz->GetTheta();
    }
    sum += angle;
    sum_sq += angle * angle;
  }
  double measured_var = (sum_sq - sum * sum / N) / (N - 1);
  BOOST_CHECK_CLOSE(measured_var, total_expected_var, 10.0);
}

BOOST_AUTO_TEST_CASE(OneOverFNoiseSynthesizer) {
  noise::NoiseModel nm;
  nm.set_1_over_f_noise(0, 1e-4, 1e3, 1e7, 4, 100e-9);

  const auto *crn = nm.get_correlated(0);
  BOOST_REQUIRE(crn != nullptr);
  BOOST_CHECK_EQUAL(crn->bands.size(), 4);

  for (size_t b = 1; b < crn->bands.size(); ++b) {
    BOOST_CHECK_LT(crn->bands[b].phi, crn->bands[b - 1].phi);
    BOOST_CHECK_GT(crn->bands[b].theta, crn->bands[b - 1].theta);
  }
}

BOOST_AUTO_TEST_SUITE_END()
