// Compare Maestro's automatic noisy execution with unsampled QCSim density-matrix probabilities.
#include <boost/test/unit_test.hpp>

#undef min
#undef max

#include <boost/math/distributions/binomial.hpp>

#include "../Circuit/Factory.h"
#include "../Execution/CircuitHelpers.h"
#include "../Execution/NoisyEstimation.h"
#include "../Execution/Options.h"
#include "../Simulators/Core/Factory.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace
{
using MaestroExecution::Backend;
using MaestroExecution::Method;
using MaestroExecution::NoiseLayer;
using MaestroExecution::SimulatorConfig;

using Circuit = std::shared_ptr<Circuits::Circuit<>>;

Circuit Bell()
{
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    return circuit;
}

Circuit FourQubitGHZ()
{
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(2));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(2, 0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 3));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(3, 1));
    // The two peaks, 1000 and 0111, also expose a reversal of the output bits.
    circuit->AddOperation(std::make_shared<Circuits::XGate<>>(3));
    return circuit;
}

Circuit Mirror(const Circuits::Circuit<> &forward, Types::qubit_t preparedQubit)
{
    // The inverse returns the noiseless circuit to this asymmetric basis state.
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::XGate<>>(preparedQubit));
    const auto &gates = forward.GetOperations();
    circuit->AddOperations(gates);
    circuit->Delay(0, 1.5);
    for (auto it = gates.rbegin(); it != gates.rend(); ++it)
        circuit->AddOperation(MaestroExecution::adjoint_gate(*it));
    return circuit;
}

Circuit RandomMirror(uint32_t seed, bool clifford)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle(-2.7, 2.7);
    Circuits::Circuit<> forward;
    for (int layer = 0; layer < 2; ++layer)
    {
        const Types::qubit_t q = rng() % 2;
        if (clifford)
        {
            forward.AddOperation(std::make_shared<Circuits::HadamardGate<>>(q));
            forward.AddOperation(std::make_shared<Circuits::SGate<>>(1 - q));
        }
        else
        {
            forward.AddOperation(std::make_shared<Circuits::RyGate<>>(q, angle(rng)));
            forward.AddOperation(std::make_shared<Circuits::RxGate<>>(1 - q, angle(rng)));
        }
        const Types::qubit_t control = rng() % 2;
        forward.AddOperation(std::make_shared<Circuits::CXGate<>>(control, 1 - control));
    }

    return Mirror(forward, 0);
}

Circuit FourQubitRandomMirror(uint32_t seed, bool clifford)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> angle(-2.7, 2.7);
    Types::qubits_vector path{0, 1, 2, 3};
    std::shuffle(path.begin(), path.end(), rng);
    Circuits::Circuit<> forward;
    forward.AddOperation(std::make_shared<Circuits::HadamardGate<>>(path.front()));
    for (size_t i = 1; i < path.size(); ++i)
    {
        forward.AddOperation(std::make_shared<Circuits::CXGate<>>(path[i - 1], path[i]));
        if (!clifford)
            forward.AddOperation(std::make_shared<Circuits::RyGate<>>(path[i], angle(rng)));
        else if (rng() % 2)
            forward.AddOperation(std::make_shared<Circuits::SGate<>>(path[i]));
        else
            forward.AddOperation(std::make_shared<Circuits::HadamardGate<>>(path[i]));
    }
    return Mirror(forward, 2);
}

// Three phase kicks accumulate between the two Hadamards. A fourth kick,
// after the final Hadamard, leaves the measurement probabilities unchanged.
Circuit Ramsey(double equivalentPhaseFlip = 0.0)
{
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::SGate<>>(0));
    circuit->AddOperation(std::make_shared<Circuits::SdgGate<>>(0));
    if (equivalentPhaseFlip > 0.0)
        circuit->AddOperation(
            std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseFlip(equivalentPhaseFlip)));
    circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(0));
    return circuit;
}

noise::NoiseModel PauliModel()
{
    noise::NoiseModel model;
    model.set_qubit_noise(0, 0.04, 0.075, 0.025);
    model.set_phase_damping(1, 0.15);
    model.set_1q_gate_depolarizing(1, 0.06);
    model.set_2q_gate_depolarizing(0, 0.05);
    model.set_2q_depolarizing(0, 1, 0.12);
    model.set_readout_error(1, 0.16, 0.035);
    return model;
}

noise::NoiseModel ThermalModel()
{
    noise::NoiseModel model;
    // T2 <= T1 also admits an exact mixture of resets and phase flips on trajectory backends.
    model.set_thermal_relaxation(0, 0.4, 4.0, 3.0, 0.12);
    model.set_thermal_relaxation_2q(1, 0.8, 6.0, 4.0, 0.05);
    model.set_idle_noise(0, 3.5, 2.5, 0.08, 0.08);
    model.set_crosstalk(0, 1, 0.25);
    return model;
}

noise::NoiseModel KrausModel(Types::qubit_t partner = 1)
{
    noise::NoiseModel model;
    model.set_t1(0, 0.14);
    model.set_t1_2q(0, 0.2);
    model.set_generalized_amplitude_damping(static_cast<int>(partner), 0.15, 0.24);
    model.set_correlated_phase_flip(0, static_cast<int>(partner), 0.13, 0.6);
    // A stochastic CNOT on reversed targets makes the custom Kraus map order-sensitive.
    Eigen::Matrix4cd cnot = Eigen::Matrix4cd::Identity();
    cnot.row(1).swap(cnot.row(3));
    model.set_kraus_channel({partner, 0}, {std::sqrt(0.85) * Eigen::Matrix4cd::Identity(), std::sqrt(0.15) * cnot});
    return model;
}

noise::NoiseModel CoherentModel()
{
    noise::NoiseModel model;
    // Both systematic signs give the same Ramsey return probability, cos²(3 * 0.38 / 2).
    model.set_coherent_rotation(0, 0.0, 0.0, 0.38);
    return model;
}

constexpr double kPhaseCorrelation = 0.65;
constexpr double kPhaseInnovation = 0.35;

noise::NoiseModel CorrelatedModel()
{
    noise::NoiseModel model;
    model.set_correlated_ar1(0, kPhaseCorrelation, kPhaseInnovation);
    return model;
}

Circuit CorrelatedRamseyReference()
{
    // For three stationary AR(1) Gaussian kicks, cov(y_i,y_j) = variance * phi^|i-j|.
    // Averaging exp(i * sum(y)) multiplies coherence by exp(-var(sum(y))/2).
    const double phi = kPhaseCorrelation;
    const double variance = kPhaseInnovation * kPhaseInnovation / (1.0 - phi * phi);
    const double phaseSumVariance = variance * (3.0 + 4.0 * phi + 2.0 * phi * phi);
    return Ramsey(-0.5 * std::expm1(-0.5 * phaseSumVariance));
}

enum class Coverage
{
    All,
    QCSimOnly
};

std::vector<SimulatorConfig> Targets(Coverage coverage = Coverage::All)
{
    std::vector<SimulatorConfig> targets;
    constexpr Backend backends[] = {
        Backend::kQCSim,
        Backend::kCompositeQCSim,
#ifndef NO_QISKIT_AER
        Backend::kQiskitAer,
        Backend::kCompositeQiskitAer,
#endif
    };
    for (const auto backend : backends)
    {
        if (coverage == Coverage::QCSimOnly && backend != Backend::kQCSim && backend != Backend::kCompositeQCSim)
            continue;
        for (const auto &[name, method] : MaestroExecution::Methods())
        {
            if (!MaestroExecution::Accepts(backend, method) || (backend == Backend::kQCSim && method == Method::kDensityMatrix))
                continue;
#ifndef NO_QISKIT_AER
            // The pinned CPU Aer controller leaves the simulator on statevector
            // when its build does not implement the requested tensor network.
            if (backend == Backend::kQiskitAer && method == Method::kTensorNetwork)
            {
                const auto simulator = Simulators::SimulatorsFactory::CreateSimulator(backend, method);
                BOOST_REQUIRE(simulator);
                if (simulator->GetSimulationType() == Method::kStatevector)
                {
                    BOOST_TEST_MESSAGE("Aer tensor_network is unavailable in this build (factory selected statevector)");
                    continue;
                }
                BOOST_REQUIRE(simulator->GetSimulationType() == method);
            }
#endif
            SimulatorConfig config;
            config.simulator_type = backend;
            config.simulation_type = method;
            config.fixed_backend = true;
            config.max_simulators = 2;
            config.seed = 3719;
            targets.push_back(config);
        }
    }
    return targets;
}

bool CliffordCasesOnly(const SimulatorConfig &config)
{
#ifndef NO_QISKIT_AER
    // Aer's extended stabilizer samples Clifford circuits exactly. Its non-Clifford
    // decomposition and Metropolis sampler have an additional approximation error.
    if (config.simulator_type == Backend::kQiskitAer && config.simulation_type == Method::kExtendedStabilizer)
        return true;
#endif
    return config.simulation_type == Method::kStabilizer;
}

std::shared_ptr<Simulators::ISimulator> DensityState(const Circuit &circuit)
{
    const auto simulator = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQCSim, Method::kDensityMatrix);
    simulator->AllocateQubits(circuit->GetMaxQubitIndex() + 1);
    simulator->Initialize();
    Circuits::OperationState classical;
    circuit->Execute(simulator, classical);
    return simulator;
}

Circuit ExactCircuit(const Circuit &body, const noise::NoiseModel &model)
{
    std::mt19937 rng(42);
    return noise::inject_combined_noise_exact(body, model, rng);
}

// Fold each qubit's asymmetric classical readout channel into the DM diagonal.
// Every terminal measurement writes qubit q to bit q, which is bit q of the bin index.
std::vector<double> ReadoutProbabilities(const Circuit &reference, const noise::NoiseModel &model)
{
    auto probabilities = DensityState(reference)->AllProbabilities();
    for (size_t q = 0; q <= reference->GetMaxQubitIndex(); ++q)
        if (const auto *error = model.get_readout_error(static_cast<int>(q)))
        {
            const size_t mask = size_t{1} << q;
            for (size_t zero = 0; zero < probabilities.size(); ++zero)
                if ((zero & mask) == 0)
                {
                    const double p0 = probabilities[zero], p1 = probabilities[zero | mask];
                    probabilities[zero] = p0 * (1 - error->p_meas1_prep0) + p1 * error->p_meas0_prep1;
                    probabilities[zero | mask] = p0 * error->p_meas1_prep0 + p1 * (1 - error->p_meas0_prep1);
                }
        }
    BOOST_REQUIRE_SMALL(std::accumulate(probabilities.begin(), probabilities.end(), 0.0) - 1.0, 1e-10);
    for (auto &p : probabilities)
    {
        BOOST_REQUIRE(p >= -1e-12 && p <= 1.0 + 1e-12);
        p = std::clamp(p, 0.0, 1.0);
    }
    return probabilities;
}

constexpr size_t kShots = 1536;
// Each bin (and the four-qubit parity) is marginally binomial. Fewer than 1000
// comparisons with two tails each give a family-wise false failure rate < 1e-6.
constexpr double kTailProbability = 5e-10;

double BinomialTail(size_t observed, double expected)
{
    const boost::math::binomial_distribution<double> binomial(kShots, expected);
    const double lower = boost::math::cdf(binomial, observed);
    const double upper = observed == 0 ? 1.0 : boost::math::cdf(boost::math::complement(binomial, observed - 1));
    return std::min(lower, upper);
}

void CompareCounts(const Circuit &body, const noise::NoiseModel &model, bool clifford, Coverage coverage = Coverage::All, const Circuit &reference = nullptr)
{
    const size_t qubits = body->GetMaxQubitIndex() + 1;
    const auto expected = ReadoutProbabilities(reference ? reference : ExactCircuit(body, model), model);
    BOOST_REQUIRE_EQUAL(expected.size(), size_t{1} << qubits);

    // Every fixture has a noiseless forbidden (or certain) outcome. Verify that
    // dropping the entire model would fail the same test, even on a Bell circuit.
    const auto clean = DensityState(body)->AllProbabilities();
    BOOST_REQUIRE_EQUAL(clean.size(), expected.size());
    bool detectsMissingNoise = false;
    for (size_t i = 0; i < expected.size(); ++i)
        if ((clean[i] < 1e-12 && BinomialTail(0, expected[i]) < kTailProbability) ||
            (clean[i] > 1.0 - 1e-12 && BinomialTail(kShots, expected[i]) < kTailProbability))
            detectsMissingNoise = true;
    BOOST_REQUIRE_MESSAGE(detectsMissingNoise, "The chosen noise is invisible at this shot budget");

    auto measured = std::static_pointer_cast<Circuits::Circuit<>>(body->Clone());
    std::vector<std::pair<Types::qubit_t, size_t>> bits;
    for (size_t q = 0; q < qubits; ++q)
        bits.emplace_back(q, q);
    measured->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(bits));

    MaestroExecution::NoisyExecutionOptions options;
    options.layer = NoiseLayer::kCombined;
    for (const auto &config : Targets(coverage))
    {
        if (!clifford && CliffordCasesOnly(config))
            continue;
        if (model.has_additional_quantum_channels() && !MaestroExecution::UsesExactQuantumChannels(config) && !MaestroExecution::SamplesQuantumChannels(config))
            continue;

        BOOST_TEST_CONTEXT(MaestroExecution::BackendName(config.simulator_type) << "/" << MaestroExecution::MethodName(config.simulation_type))
        {
            // Use the default channel and counts policies. Independent trajectories
            // need one noise draw per shot; an exact ensemble only needs one preparation.
            const auto policy = MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, options.layer, kShots, measured);
            const auto result = MaestroExecution::ExecuteNoisy(measured, model, config, kShots, policy.realizations, 8501, options);
            BOOST_REQUIRE(result.simulator == config.simulator_type);
            BOOST_REQUIRE(result.method == config.simulation_type);
            BOOST_REQUIRE_EQUAL(result.width, qubits);
            std::vector<size_t> counts(expected.size());
            for (const auto &[record, n] : result.counts)
            {
                BOOST_REQUIRE_EQUAL(record.size(), qubits);
                size_t outcome = 0;
                for (size_t q = 0; q < qubits; ++q)
                    outcome |= size_t{record[q]} << q;
                counts[outcome] += n;
            }
            BOOST_REQUIRE_EQUAL(std::accumulate(counts.begin(), counts.end(), size_t{0}), kShots);
            for (size_t i = 0; i < counts.size(); ++i)
            {
                const double tail = BinomialTail(counts[i], expected[i]);
                BOOST_CHECK_MESSAGE(tail >= kTailProbability, "outcome " << i << ": observed " << counts[i] << "/" << kShots << ", QCSim DM probability "
                                                                         << expected[i] << ", binomial tail " << tail);
            }
            if (qubits == 4)
            {
                // In the X basis this parity measures <XXXX> of the GHZ state.
                // Pooling the eight even bins keeps sensitivity to loss of coherence.
                size_t evenCount = 0;
                double evenProbability = 0;
                for (size_t i = 0; i < counts.size(); ++i)
                    if (std::bitset<4>(i).count() % 2 == 0)
                    {
                        evenCount += counts[i];
                        evenProbability += expected[i];
                    }
                BOOST_CHECK_MESSAGE(BinomialTail(evenCount, std::clamp(evenProbability, 0.0, 1.0)) >= kTailProbability,
                                    "even parity: observed " << evenCount << "/" << kShots << ", QCSim DM probability " << evenProbability);
            }
        }
    }
}
} // namespace

BOOST_AUTO_TEST_SUITE(noise_sims_vs_dm)

BOOST_AUTO_TEST_CASE(bell_pauli_and_readout)
{
    CompareCounts(Bell(), PauliModel(), true);
}

BOOST_AUTO_TEST_CASE(random_clifford_pauli_and_readout)
{
    CompareCounts(RandomMirror(307, true), PauliModel(), true);
}

BOOST_AUTO_TEST_CASE(random_rotations_thermal_idle_and_crosstalk)
{
    CompareCounts(RandomMirror(119, false), ThermalModel(), false);
}

BOOST_AUTO_TEST_CASE(random_rotations_damping_and_kraus)
{
    CompareCounts(RandomMirror(829, false), KrausModel(), false);
}

BOOST_AUTO_TEST_CASE(ramsey_systematic_rotation)
{
    CompareCounts(Ramsey(), CoherentModel(), false);
}

BOOST_AUTO_TEST_CASE(ramsey_time_correlated_dephasing)
{
    CompareCounts(Ramsey(), CorrelatedModel(), false, Coverage::All, CorrelatedRamseyReference());
}

BOOST_AUTO_TEST_CASE(qcsim_four_qubit_ghz_phase_noise_in_x_basis)
{
    auto circuit = FourQubitGHZ();
    for (Types::qubit_t q = 0; q < 4; ++q)
        circuit->AddOperation(std::make_shared<Circuits::HadamardGate<>>(q));
    noise::NoiseModel model;
    model.set_dephasing(3, 0.08);
    CompareCounts(circuit, model, true, Coverage::QCSimOnly);
}

BOOST_AUTO_TEST_CASE(qcsim_four_qubit_random_clifford_pauli_and_readout)
{
    auto model = PauliModel();
    model.set_bit_flip(2, 0.07);
    model.set_dephasing(3, 0.1);
    // Every shuffled path couples qubit 3 to at least one of these neighbors.
    for (int neighbor = 0; neighbor < 3; ++neighbor)
        model.set_2q_depolarizing(neighbor, 3, 0.14);
    model.set_readout_error(3, 0.04, 0.19);
    CompareCounts(FourQubitRandomMirror(541, true), model, true, Coverage::QCSimOnly);
}

BOOST_AUTO_TEST_CASE(qcsim_four_qubit_random_rotations_thermal_and_idle)
{
    auto model = ThermalModel();
    model.set_thermal_relaxation(2, 0.5, 4.5, 3.0, 0.04);
    model.set_thermal_relaxation_2q(3, 0.8, 4.0, 3.5, 0.2);
    CompareCounts(FourQubitRandomMirror(947, false), model, false, Coverage::QCSimOnly);
}

BOOST_AUTO_TEST_CASE(qcsim_four_qubit_ghz_damping_and_nonadjacent_kraus)
{
    CompareCounts(FourQubitGHZ(), KrausModel(3), false, Coverage::QCSimOnly);
}

BOOST_AUTO_TEST_CASE(exact_channel_expectations)
{
    const auto circuit = RandomMirror(153, false);
    // All 15 non-identity Pauli expectations determine this two-qubit density matrix.
    std::vector<std::string> observables;
    for (const char a : std::string("IXYZ"))
        for (const char b : std::string("IXYZ"))
            if (a != 'I' || b != 'I')
                observables.push_back({a, b});
    const auto targets = Targets();
    for (const auto &[label, model] : {std::pair{"Pauli", PauliModel()}, {"thermal", ThermalModel()}, {"Kraus", KrausModel()}})
    {
        const auto expected = DensityState(ExactCircuit(circuit, model))->ExpectationValues(observables);
        BOOST_REQUIRE_EQUAL(expected.size(), observables.size());
        for (const auto &config : targets)
        {
            if (!MaestroExecution::InjectsExactChannels(config, model, *circuit))
                continue;
            BOOST_TEST_CONTEXT(label << " " << MaestroExecution::BackendName(config.simulator_type) << "/"
                                     << MaestroExecution::MethodName(config.simulation_type))
            {
                const auto result = MaestroExecution::EstimateNoisy(circuit, observables, model, std::nullopt, config, 8501, NoiseLayer::kCombined);
                BOOST_REQUIRE_EQUAL(result.realizations, 1u);
                BOOST_REQUIRE_EQUAL(result.values.size(), expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                    BOOST_CHECK_MESSAGE(std::abs(result.values[i] - expected[i]) < 1e-9,
                                        observables[i] << ": observed " << result.values[i] << ", QCSim DM expectation " << expected[i]);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
