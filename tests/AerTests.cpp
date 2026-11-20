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
#include <complex>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
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
    // Both relaxations are Kraus branches (T2 > T1): Aer applies them as one
    // two-qubit Kraus operation, whose qubit order these targets check.
    const Channel bothRelax = Channel::OnQubit(Channel::ThermalRelaxation(1.0, 2.0, 3.5), 0).Then(Channel::ThermalRelaxation(0.8, 1.5, 2.5, 0.15), {1});
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
        sim.ApplyQuantumChannel({1, 0}, bothRelax);
        sim.ApplyRx(1, 0.9);
        sim.ApplyQuantumChannel({0, 2}, bothRelax);
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

// A Kraus channel on two qubits that are apart in the chain: Aer brings them together to apply it,
// and has to renormalize the tensors around where they are then. Renormalizing around where they
// were, as Aer did before the revision pinned here, left the tensors between them stale:
// the probabilities stayed right, and an expectation value on a qubit in between was wrong.
BOOST_AUTO_TEST_CASE(matrix_product_state_channel_on_distant_qubits_keeps_spectators_consistent)
{
    using Channel = Simulators::QuantumChannel;
    constexpr size_t qubits = 6;
    const Channel damping = Channel::OnQubit(Channel::AmplitudeDamping(0.3), 0).Then(Channel::AmplitudeDamping(0.3), {1});
    const std::vector<Types::qubits_vector> placements{{0, 5}, {5, 0}, {1, 4}, {4, 2}, {2, 3}};
    for (const auto &targets : placements)
        for (uint64_t seed = 1; seed <= 4; ++seed)
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
            sim->AllocateQubits(qubits);
            sim->Initialize();
            sim->SetSeed(seed);
            sim->ApplyH(0);
            for (size_t q = 0; q + 1 < qubits; ++q)
                sim->ApplyCX(q, q + 1);
            for (size_t q = 0; q < qubits; ++q)
                sim->ApplyRy(q, 0.2 + 0.1 * q);
            sim->ApplyQuantumChannel(targets, damping);

            // The Z expectation of each qubit against its marginal from the probabilities of the same state.
            std::vector<double> expectations(qubits);
            for (size_t q = 0; q < qubits; ++q)
            {
                std::string pauli(qubits, 'I');
                pauli[q] = 'Z';
                expectations[q] = sim->ExpectationValue(pauli);
            }
            const auto probabilities = sim->AllProbabilities();
            for (size_t q = 0; q < qubits; ++q)
            {
                double marginal = 0.;
                for (size_t i = 0; i < probabilities.size(); ++i)
                    marginal += probabilities[i] * ((i >> q) & 1 ? -1. : 1.);
                BOOST_TEST(std::abs(expectations[q] - marginal) < 1e-9, "Z on qubit " << q << " after a channel on qubits " << targets[0] << " and "
                                                                                      << targets[1] << " (seed " << seed << "): expectation " << expectations[q]
                                                                                      << ", marginal " << marginal);
            }
        }
}

// A bond limit describes the simulated state, not the temporary copy Aer makes of it to answer a
// query: moving the queried qubits together can need larger bonds for a while, and a temporary that
// truncated them (qiskit-aer b4fddaa, the previous pin) changed the answer. Bell pairs on neighbouring
// qubits fit a bond limit of 2 exactly; every marginal across pairs is uniform.
BOOST_AUTO_TEST_CASE(matrix_product_state_bond_limit_does_not_truncate_marginal_queries)
{
    constexpr size_t qubits = 6;
    const std::vector<Types::qubits_vector> queries{{0, 2}, {2, 0}, {1, 4}, {0, 3, 5}, {5, 2, 1}};
    for (const char *bond : {"2", "4", ""})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
        if (std::string(bond) != "")
            sim->Configure("matrix_product_state_max_bond_dimension", bond);
        sim->AllocateQubits(qubits);
        sim->Initialize();
        for (size_t q = 0; q + 1 < qubits; q += 2)
        {
            sim->ApplyH(q);
            sim->ApplyCX(q, q + 1);
        }
        for (const auto &query : queries)
        {
            const auto marginal = sim->MarginalProbabilities(query);
            BOOST_REQUIRE_EQUAL(marginal.size(), size_t{1} << query.size());
            for (size_t outcome = 0; outcome < marginal.size(); ++outcome)
                BOOST_TEST(std::abs(marginal[outcome] - 1. / marginal.size()) < 1e-9, "bond limit '" << bond << "', marginal of " << query.size()
                                                                                                     << " qubits starting at " << query[0] << ", outcome "
                                                                                                     << outcome << ": " << marginal[outcome]);
        }
        // The queries left the state as it was.
        const auto all = sim->AllProbabilities();
        for (size_t i = 0; i < all.size(); ++i)
        {
            const bool paired = ((i ^ (i >> 1)) & 0b010101) == 0;
            BOOST_TEST(std::abs(all[i] - (paired ? 0.125 : 0.)) < 1e-9, "bond limit '" << bond << "': probability of outcome " << i << " is " << all[i]);
        }
    }
}

// A channel on qubits apart in the chain, under a bond limit: bringing the qubits together truncates
// the state. Aer has to weigh the Kraus operators on the state it then applies one to. Weighed on the
// state before the move (qiskit-aer ba4c60a, the previous pin), an operator could be chosen that the
// moved state gives no weight: the trajectory became a zero state and its renormalization crashed.
BOOST_AUTO_TEST_CASE(matrix_product_state_channel_under_a_bond_limit_keeps_the_trajectory_normalized)
{
    using Channel = Simulators::QuantumChannel;
    const Channel damping = Channel::OnQubit(Channel::AmplitudeDamping(0.3), 0).Then(Channel::AmplitudeDamping(0.3), {1});
    for (const char *bond : {"2", "4"})
        for (uint64_t seed = 1; seed <= 32; ++seed) // every branch of the channel is drawn among them
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
            sim->Configure("matrix_product_state_max_bond_dimension", bond);
            sim->AllocateQubits(4);
            sim->Initialize();
            sim->SetSeed(seed);
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            sim->ApplyH(2);
            sim->ApplyCX(2, 3);
            sim->ApplyQuantumChannel({0, 2}, damping);
            double norm = 0.;
            for (const double probability : sim->AllProbabilities())
                norm += probability;
            BOOST_TEST(std::abs(norm - 1.) < 1e-9, "bond limit " << bond << ", seed " << seed << ": the trajectory has norm " << norm);
        }
}

// A gate on three qubits is applied on a small MPS that is written back into the state, within the
// state's bond limit (qiskit-aer ba4c60a wrote it back without one). Under a limit of 1 the state
// stays a product state whatever the gate: its probabilities are the product of its marginals.
BOOST_AUTO_TEST_CASE(matrix_product_state_three_qubit_gate_respects_the_bond_limit)
{
    constexpr size_t qubits = 4;
    auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
    sim->Configure("matrix_product_state_max_bond_dimension", "1");
    sim->AllocateQubits(qubits);
    sim->Initialize();
    sim->ApplyH(0);
    sim->ApplyH(1);
    sim->ApplyRy(3, 0.7);
    sim->ApplyCCX(0, 1, 2); // entangles the three qubits, unless truncated
    const auto probabilities = sim->AllProbabilities();
    std::vector<double> one(qubits, 0.);
    double norm = 0.;
    for (size_t i = 0; i < probabilities.size(); ++i)
    {
        norm += probabilities[i];
        for (size_t q = 0; q < qubits; ++q)
            if ((i >> q) & 1)
                one[q] += probabilities[i];
    }
    BOOST_TEST(std::abs(norm - 1.) < 1e-9);
    for (size_t i = 0; i < probabilities.size(); ++i)
    {
        double product = 1.;
        for (size_t q = 0; q < qubits; ++q)
            product *= (i >> q) & 1 ? one[q] : 1. - one[q];
        BOOST_TEST(std::abs(probabilities[i] - product) < 1e-9,
                   "outcome " << i << ": probability " << probabilities[i] << " is not the product of the marginals, " << product);
    }
}

// A channel given on two targets that acts on one of them only reaches Aer's matrix product state on
// that one. Sent on both, Aer moved the idle target next to the other, and under a bond limit the
// move truncated the state: damping qubit 2 as a channel on qubits 0 and 2 took P(q1 = 1) from 0.45
// to 0 on two independent pairs, where damping qubit 2 directly leaves it alone.
BOOST_AUTO_TEST_CASE(matrix_product_state_channel_does_not_move_targets_it_leaves_alone)
{
    using Channel = Simulators::QuantumChannel;
    const Channel damping = Channel::AmplitudeDamping(0.3);
    BOOST_TEST((Channel::OnQubit(damping, 1).ActiveQubits() == std::vector<size_t>{1}));
    BOOST_TEST((Channel::OnQubit(damping, 0).Then(damping, {1}).ActiveQubits() == std::vector<size_t>{0, 1}));
    BOOST_TEST(Channel::OnQubit(damping, 1).OnActiveQubits().GetNumberOfQubits() == 1u);

    // A Pauli part placed on one target is kept as a mixture over both; it still acts on that one only.
    const Channel flip = Channel::BitFlip(0.2), strong = Channel::AmplitudeDamping(0.9);
    BOOST_TEST((Channel::OnQubit(flip, 1).ActiveQubits() == std::vector<size_t>{1}));
    BOOST_TEST((Channel::OnQubit(flip, 1).Then(strong, {1}).ActiveQubits() == std::vector<size_t>{1}));
    BOOST_TEST((Channel::OnQubit(flip, 0).Then(strong, {1}).ActiveQubits() == std::vector<size_t>{0, 1}));
    BOOST_TEST(Channel::OnQubit(flip, 1).Then(strong, {1}).OnActiveQubits().GetNumberOfQubits() == 1u);
    BOOST_TEST(Channel::OnQubit(Channel::BitFlip(0.), 1).ActiveQubits().empty());
    // A part that is the identity, however it is written, acts on no target.
    const Channel noDamping = Channel::AmplitudeDamping(0.), noRelaxation = Channel::ThermalRelaxation(0., 1e-4, 1e-4),
                  noMixture = Channel::UnitaryMixture({1.}, {Eigen::MatrixXcd::Identity(2, 2)});
    for (const Channel &idle : {noDamping, noRelaxation, noMixture})
    {
        BOOST_TEST(idle.ActiveQubits().empty());
        BOOST_TEST((Channel::OnQubit(idle, 0).Then(damping, {1}).ActiveQubits() == std::vector<size_t>{1}));
        // Nothing is dropped: the part that changes nothing stays in the reduced channel, where its
        // branch is still drawn.
        BOOST_TEST(Channel::OnQubit(idle, 0).Then(damping, {1}).OnActiveQubits().GetFactors().size() == 2u);
    }
    // The list is kept with the channel: a reference for a channel that stays, a copy for a temporary.
    static_assert(std::is_same_v<decltype(damping.ActiveQubits()), const std::vector<size_t> &>);
    static_assert(std::is_same_v<decltype(Channel::BitFlip(0.2).ActiveQubits()), std::vector<size_t>>);
    {
        const auto &kept = Channel::OnQubit(flip, 1).ActiveQubits(); // extends the copy's lifetime
        BOOST_TEST((kept == std::vector<size_t>{1}));
    }
    // Operators on two targets that are the identity on one of them act on the other only: a bit flip
    // written as a mixture of 4x4 unitaries, a damping written as 4x4 Kraus operators.
    const auto onTargets = [](const Eigen::MatrixXcd &second, const Eigen::MatrixXcd &first) {
        Eigen::MatrixXcd u(4, 4); // second (x) first, with the first target as local matrix bit zero
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                u(i, j) = second(i / 2, j / 2) * first(i % 2, j % 2);
        return u;
    };
    const Eigen::MatrixXcd one = Eigen::MatrixXcd::Identity(2, 2);
    Eigen::MatrixXcd pauliX = Eigen::MatrixXcd::Zero(2, 2);
    pauliX(0, 1) = pauliX(1, 0) = 1.;
    const Channel wideFlip = Channel::UnitaryMixture({0.8, 0.2}, {Eigen::MatrixXcd::Identity(4, 4), onTargets(pauliX, one)});
    Channel::KrausOperators wideDampingOperators;
    for (const auto &op : strong.GetKrausOperators())
        wideDampingOperators.push_back(onTargets(op, one));
    const Channel wideDamping(wideDampingOperators);
    BOOST_TEST((wideFlip.ActiveQubits() == std::vector<size_t>{1}));
    BOOST_TEST((wideFlip.Then(strong, {1}).ActiveQubits() == std::vector<size_t>{1}));
    BOOST_TEST(wideFlip.Then(strong, {1}).OnActiveQubits().GetNumberOfQubits() == 1u);
    BOOST_TEST((Channel::UnitaryMixture({0.5, 0.5}, {onTargets(pauliX, one), onTargets(one, pauliX)}).ActiveQubits() == std::vector<size_t>{0, 1}));
    BOOST_TEST((Channel::UnitaryMixture({1.}, {onTargets(one, pauliX)}).ActiveQubits() == std::vector<size_t>{0}));
    BOOST_TEST((wideFlip.Then(wideDamping).ActiveQubits() == std::vector<size_t>{1}));
    // The reduced channel is built once and kept with the channel and its copies: applying the channel
    // again does not derive its Kraus operators again.
    {
        const Channel placed = Channel::OnQubit(damping, 1), copy = placed;
        BOOST_TEST(&placed.OnActiveQubits().GetKrausOperators() == &placed.OnActiveQubits().GetKrausOperators());
        BOOST_TEST(&placed.OnActiveQubits().GetKrausOperators() == &copy.OnActiveQubits().GetKrausOperators());
    }

    // A channel given by its Kraus operators keeps them on the target they act on: the trajectories
    // are the ones the caller wrote. A 30% damping of |1> written as (K0 + K1) / sqrt(2) and
    // (K0 - K1) / sqrt(2) leaves P(1) = 0.7 in either branch; its canonical operators would give 0 or 1.
    {
        Eigen::MatrixXcd k0 = Eigen::MatrixXcd::Zero(2, 2), k1 = Eigen::MatrixXcd::Zero(2, 2);
        k0(0, 0) = 1.;
        k0(1, 1) = std::sqrt(0.7);
        k1(0, 1) = std::sqrt(0.3);
        const Eigen::MatrixXcd plus = (k0 + k1) / std::sqrt(2.), minus = (k0 - k1) / std::sqrt(2.);

        struct Given
        {
            Types::qubits_vector targets;
            Channel channel;
        };

        const Channel narrow(Channel::KrausOperators{plus, minus}), wide(Channel::KrausOperators{onTargets(one, plus), onTargets(one, minus)});
        // The same operation on qubit 0 every time: given on it, on two targets in either order, with
        // the targets of the instruction and of the channel both reversed, and placed on a target.
        const std::vector<Given> given{{{0}, narrow},
                                       {{0, 1}, wide},
                                       {{1, 0}, Channel(Channel::KrausOperators{onTargets(plus, one), onTargets(minus, one)})},
                                       {{1, 0}, Channel::Placed(wide, {1, 0})},
                                       {{1, 0}, Channel::OnQubit(narrow, 1)},
                                       {{0, 1}, Channel::OnQubit(narrow, 0)}};
        // A set of Kraus operators on all the targets of a channel stays that set when the channel is
        // placed on other targets: here with the two targets exchanged.
        {
            // Operators on both targets: (K0, K1) on the first, (plus, minus) on the second.
            const Channel::KrausOperators given{onTargets(plus, k0), onTargets(minus, k0), onTargets(plus, k1), onTargets(minus, k1)};
            const Channel exchanged = Channel::Placed(Channel(given), {1, 0});
            BOOST_REQUIRE_EQUAL(exchanged.GetKrausOperators().size(), 4u);
            BOOST_TEST((exchanged.GetKrausOperators()[0] - onTargets(k0, plus)).isZero(1e-15));
            BOOST_TEST((exchanged.GetKrausOperators()[1] - onTargets(k0, minus)).isZero(1e-15));
            BOOST_TEST((exchanged.GetKrausOperators()[3] - onTargets(k1, minus)).isZero(1e-15));
        }
        BOOST_TEST(&given[1].channel.OnActiveQubits().GetKrausOperators() == &given[1].channel.OnActiveQubits().GetKrausOperators());
        BOOST_TEST(given[1].channel.OnActiveQubits().GetKrausOperators().size() == 2u);
        BOOST_TEST((given[1].channel.OnActiveQubits().GetKrausOperators()[0] - plus).isZero(1e-15));
        BOOST_TEST((given[2].channel.OnActiveQubits().GetKrausOperators()[1] - minus).isZero(1e-15));
        BOOST_TEST((given[3].channel.OnActiveQubits().GetKrausOperators()[0] - plus).isZero(1e-15));
        BOOST_TEST((given[4].channel.OnActiveQubits().GetKrausOperators()[1] - minus).isZero(1e-15));
        for (const auto method : {Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kStatevector})
            for (const Given &g : given)
                for (uint64_t seed = 1; seed <= 6; ++seed)
                {
                    auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
                    sim->AllocateQubits(2);
                    sim->Initialize();
                    sim->SetSeed(seed);
                    sim->ApplyX(0);
                    sim->ApplyQuantumChannel(g.targets, g.channel);
                    const auto probabilities = sim->AllProbabilities();
                    BOOST_TEST(std::abs(probabilities[1] + probabilities[3] - 0.7) < 1e-9, "a damping given by its own Kraus operators on "
                                                                                               << g.targets.size() << " targets, seed " << seed
                                                                                               << ": P(1) = " << probabilities[1] + probabilities[3]);
                }
    }

    struct Form
    {
        Types::qubits_vector targets;
        Channel channel;
    };

    // The same operation on qubit 2, given several ways: a damping, and a bit flip followed by a damping.
    const std::vector<std::vector<Form>> operations{{{{2}, damping},
                                                     {{0, 2}, Channel::OnQubit(damping, 1)},
                                                     {{2, 0}, Channel::OnQubit(damping, 0)},
                                                     {{0, 2}, Channel::OnQubit(noDamping, 0).Then(damping, {1})},
                                                     {{0, 2}, Channel::OnQubit(noRelaxation, 0).Then(damping, {1})},
                                                     {{0, 2}, Channel::OnQubit(noMixture, 0).Then(damping, {1})}},
                                                    {{{2}, flip.Then(strong)},
                                                     {{0, 2}, Channel::OnQubit(flip, 1).Then(strong, {1})},
                                                     {{2, 0}, Channel::OnQubit(flip, 0).Then(strong, {0})},
                                                     {{0, 2}, Channel::OnQubit(flip.Then(strong), 1)},
                                                     {{0, 2}, wideFlip.Then(strong, {1})},
                                                     {{0, 2}, wideFlip.Then(wideDamping)}}};
    for (const auto &forms : operations)
        for (const char *bond : {"2", "4", ""})
            for (uint64_t seed = 1; seed <= 6; ++seed)
            {
                std::vector<std::vector<double>> distributions;
                for (const Form &form : forms)
                {
                    auto sim =
                        Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
                    if (std::string(bond) != "")
                        sim->Configure("matrix_product_state_max_bond_dimension", bond);
                    sim->AllocateQubits(4);
                    sim->Initialize();
                    sim->SetSeed(seed);
                    // Independent pairs (0,1) and (2,3), with P(11) of 0.45 and 0.49.
                    sim->ApplyRy(0, 2. * std::asin(std::sqrt(0.45)));
                    sim->ApplyCX(0, 1);
                    sim->ApplyRy(2, 2. * std::asin(std::sqrt(0.49)));
                    sim->ApplyCX(2, 3);
                    sim->ApplyQuantumChannel(form.targets, form.channel);
                    distributions.push_back(sim->AllProbabilities());
                    double q1 = 0.;
                    for (size_t i = 0; i < distributions.back().size(); ++i)
                        if ((i >> 1) & 1)
                            q1 += distributions.back()[i];
                    BOOST_TEST(std::abs(q1 - 0.45) < 1e-9, "bond limit '" << bond << "', seed " << seed << ", channel on " << form.targets.size()
                                                                          << " targets: damping qubit 2 left P(q1 = 1) = " << q1);
                }
                // The same operation however it is given: the same trajectory.
                for (size_t form = 1; form < distributions.size(); ++form)
                    for (size_t i = 0; i < distributions[0].size(); ++i)
                        BOOST_TEST(std::abs(distributions[form][i] - distributions[0][i]) < 1e-9,
                                   "bond limit '" << bond << "', seed " << seed << ": form " << form << " differs from the one-qubit channel at outcome " << i);
            }
}

// A unitary mixture draws a branch and applies it as a gate. A drawn unitary that is the identity on
// a target is not applied there: as a two-qubit gate, also the identity made the matrix product state
// bring the two qubits together, and under a bond limit that truncated the state (P(q1 = 1) from 0.45
// to 0 on two independent pairs, for a mixture of the 4x4 identity alone).
BOOST_AUTO_TEST_CASE(unitary_mixture_branches_reach_only_the_targets_they_act_on)
{
    using Channel = Simulators::QuantumChannel;
    using Matrix = Eigen::MatrixXcd;
    // U = second (x) first, with the first target as local matrix bit zero.
    const auto onTargets = [](const Matrix &second, const Matrix &first) {
        Matrix u(4, 4);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                u(i, j) = second(i / 2, j / 2) * first(i % 2, j % 2);
        return u;
    };
    const Matrix identity = Matrix::Identity(2, 2);
    Matrix x = Matrix::Zero(2, 2), rotation(2, 2);
    x(0, 1) = x(1, 0) = 1.;
    const double angle = 0.7;
    rotation << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2); // Ry(angle)

    const auto prepare = [](Simulators::SimulationType method, const char *bond, uint64_t seed) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
        if (std::string(bond) != "")
            sim->Configure("matrix_product_state_max_bond_dimension", bond);
        sim->AllocateQubits(4);
        sim->Initialize();
        sim->SetSeed(seed);
        // Independent pairs (0,1) and (2,3), with P(11) of 0.45 and 0.49.
        sim->ApplyRy(0, 2. * std::asin(std::sqrt(0.45)));
        sim->ApplyCX(0, 1);
        sim->ApplyRy(2, 2. * std::asin(std::sqrt(0.49)));
        sim->ApplyCX(2, 3);
        return sim;
    };
    const auto marginal = [](const std::vector<double> &probabilities, size_t qubit) {
        double one = 0.;
        for (size_t i = 0; i < probabilities.size(); ++i)
            if ((i >> qubit) & 1)
                one += probabilities[i];
        return one;
    };
    const auto mps = Simulators::SimulationType::kMatrixProductState;

    // Channels on qubits 0 and 2 that leave qubit 0 alone in every branch: the pair (0,1) keeps its
    // marginals under any bond limit, and qubit 3 keeps its own (an X on qubit 2 does not change it).
    const std::vector<Channel> idleOnFirst{Channel::UnitaryMixture({1.}, {Matrix::Identity(4, 4)}),
                                           Channel::UnitaryMixture({1.}, {Matrix::Identity(4, 4)}).Then(Channel::BitFlip(0.2), {1}),
                                           Channel::UnitaryMixture({0.7, 0.3}, {Matrix::Identity(4, 4), onTargets(x, identity)})};
    for (const Channel &channel : idleOnFirst)
        for (const char *bond : {"2", "4", ""})
            for (uint64_t seed = 1; seed <= 8; ++seed)
            {
                auto sim = prepare(mps, bond, seed);
                sim->ApplyQuantumChannel({0, 2}, channel);
                const auto probabilities = sim->AllProbabilities();
                BOOST_TEST(std::abs(marginal(probabilities, 1) - 0.45) < 1e-9,
                           "bond limit '" << bond << "', seed " << seed << ": P(q1 = 1) = " << marginal(probabilities, 1));
                BOOST_TEST(std::abs(marginal(probabilities, 0) - 0.45) < 1e-9);
                BOOST_TEST(std::abs(marginal(probabilities, 3) - 0.49) < 1e-9);
            }

    // A branch that is a gate on one target is that gate there, on the statevector too: a rotation
    // given as a two-qubit unitary on either target against the rotation applied directly.
    for (const auto method : {mps, Simulators::SimulationType::kStatevector})
        for (const bool onSecond : {false, true})
            for (const char *bond : {"2", ""})
            {
                if (method != mps && std::string(bond) != "")
                    continue;
                auto sim = prepare(method, bond, 3), reference = prepare(method, bond, 3);
                sim->ApplyQuantumChannel({0, 2}, Channel::UnitaryMixture({1.}, {onSecond ? onTargets(rotation, identity) : onTargets(identity, rotation)}));
                reference->ApplyRy(onSecond ? 2 : 0, angle);
                const auto probabilities = sim->AllProbabilities(), expected = reference->AllProbabilities();
                for (size_t i = 0; i < probabilities.size(); ++i)
                    BOOST_TEST(std::abs(probabilities[i] - expected[i]) < 1e-9, "a rotation on the " << (onSecond ? "second" : "first")
                                                                                                     << " target, bond limit '" << bond << "', outcome " << i
                                                                                                     << ": " << probabilities[i] << " for " << expected[i]);
            }

    // A branch that is a phase times the identity keeps its phase, which the amplitudes show, and
    // still moves nothing: it is applied on one qubit.
    {
        const std::complex<double> phase(0., 1.);
        for (const auto method : {mps, Simulators::SimulationType::kStatevector})
            for (const bool two : {false, true})
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
                sim->AllocateQubits(2);
                sim->Initialize();
                sim->SetSeed(1);
                if (two)
                    sim->ApplyQuantumChannel({0, 1}, Channel::UnitaryMixture({1.}, {Matrix(phase * Matrix::Identity(4, 4))}));
                else
                    sim->ApplyQuantumChannel({0}, Channel::UnitaryMixture({1.}, {Matrix(phase * identity)}));
                BOOST_TEST(std::abs(sim->Amplitude(0) - phase) < 1e-12,
                           "a deterministic i I branch on " << (two ? "two targets" : "one target") << " left the amplitude " << sim->Amplitude(0));
            }
        for (uint64_t seed = 1; seed <= 4; ++seed)
        {
            auto sim = prepare(mps, "2", seed);
            sim->ApplyQuantumChannel({0, 2}, Channel::UnitaryMixture({1.}, {Matrix(phase * Matrix::Identity(4, 4))}));
            BOOST_TEST(std::abs(marginal(sim->AllProbabilities(), 1) - 0.45) < 1e-9);
        }
    }

    // Reducing a channel to the target it acts on keeps each part as it is: a mixture stays a mixture
    // of the same unitaries with the same probabilities, so its trajectories and its phases stay.
    {
        const std::complex<double> phase(0., 1.);
        const double c = std::cos(std::acos(-1.) / 4), s = std::sin(std::acos(-1.) / 4);
        Matrix up(2, 2), down(2, 2);
        up << c, -s, s, c;   // Ry(pi/2)
        down << c, s, -s, c; // Ry(-pi/2)
        const Channel phased = Channel::UnitaryMixture({1.}, {Matrix(phase * onTargets(identity, x))}).OnActiveQubits(),
                      rotations = Channel::UnitaryMixture({0.5, 0.5}, {onTargets(identity, up), onTargets(identity, down)}).OnActiveQubits();
        BOOST_REQUIRE_EQUAL(phased.GetFactors().size(), 1u);
        BOOST_TEST((phased.GetFactors()[0].kind == Channel::FactorKind::kUnitaryMixture));
        BOOST_TEST((phased.GetFactors()[0].operators[0] - Matrix(phase * x)).isZero(0.));
        BOOST_REQUIRE_EQUAL(rotations.GetFactors().size(), 1u);
        BOOST_TEST((rotations.GetFactors()[0].kind == Channel::FactorKind::kUnitaryMixture));
        BOOST_TEST((rotations.GetFactors()[0].probabilities == std::vector<double>{0.5, 0.5}));
        for (const auto method : {mps, Simulators::SimulationType::kStatevector})
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
            sim->AllocateQubits(2);
            sim->Initialize();
            sim->SetSeed(1);
            sim->ApplyQuantumChannel({0}, phased);
            BOOST_TEST(std::abs(sim->Amplitude(1) - phase) < 1e-12, "a reduced i X branch left the amplitude " << sim->Amplitude(1));
            for (uint64_t seed = 1; seed <= 6; ++seed)
            {
                auto rotated = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
                rotated->AllocateQubits(2);
                rotated->Initialize();
                rotated->SetSeed(seed);
                rotated->ApplyQuantumChannel({0}, rotations);
                BOOST_TEST(std::abs(marginal(rotated->AllProbabilities(), 0) - 0.5) < 1e-9);
            }
        }
    }

    // A unitary on both targets is still applied as one: a CX given as a mixture against the gate.
    {
        Matrix cx = Matrix::Zero(4, 4); // control: first target (bit zero), target: second
        cx(0, 0) = cx(2, 2) = cx(3, 1) = cx(1, 3) = 1.;
        auto sim = prepare(mps, "", 3), reference = prepare(mps, "", 3);
        sim->ApplyQuantumChannel({0, 2}, Channel::UnitaryMixture({1.}, {cx}));
        reference->ApplyCX(0, 2);
        const auto probabilities = sim->AllProbabilities(), expected = reference->AllProbabilities();
        for (size_t i = 0; i < probabilities.size(); ++i)
            BOOST_TEST(std::abs(probabilities[i] - expected[i]) < 1e-9);
    }
}

// One operation on one qubit, however it is given: on that qubit, on two targets with the other left
// alone (in either order, placed, with the targets exchanged), reduced by hand, or rebuilt from its
// exported Kraus operators. A target a channel leaves alone must change nothing: not the branch a
// seed draws, not the operators applied, not a phase, on any backend. The comparison is of amplitudes.
// (This is about idle targets. Exchanging the targets of a channel that acts on both keeps its
// distribution, not necessarily the trajectory of a seed on the Aer MPS.)
//
// What makes it hold (QuantumChannel::OnActiveQubits, GetKrausOperators): the reduced channel keeps
// every part of the channel, with its kind, operators, order and probabilities, so the same draws
// select the same branches; and the Kraus operators of a channel are those of its reduced form with
// the identity on the removed target, so the two descriptions cannot disagree.
BOOST_AUTO_TEST_CASE(a_target_a_channel_leaves_alone_does_not_change_the_operation)
{
    using Channel = Simulators::QuantumChannel;
    using Matrix = Eigen::MatrixXcd;
    using Type = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    const std::complex<double> phase(0., 1.);
    const Matrix one = Matrix::Identity(2, 2);
    Matrix k0 = Matrix::Zero(2, 2), k1 = Matrix::Zero(2, 2), x = Matrix::Zero(2, 2), p0 = Matrix::Zero(2, 2), p1 = Matrix::Zero(2, 2), up(2, 2), down(2, 2);
    k0(0, 0) = 1.;
    k0(1, 1) = std::sqrt(0.7);
    k1(0, 1) = std::sqrt(0.3);
    x(0, 1) = x(1, 0) = 1.;
    p0(0, 0) = p1(1, 1) = 1.;
    const double c = std::cos(std::acos(-1.) / 4), s = std::sin(std::acos(-1.) / 4);
    up << c, -s, s, c;
    down << c, s, -s, c;
    // A 2x2 operator on one target of two, the identity on the other.
    const auto pad = [](const Matrix &op, size_t home) {
        Matrix full = Matrix::Zero(4, 4);
        for (int idle = 0; idle < 2; ++idle)
            for (int row = 0; row < 2; ++row)
                for (int column = 0; column < 2; ++column)
                    full(home == 0 ? 2 * idle + row : 2 * row + idle, home == 0 ? 2 * idle + column : 2 * column + idle) = op(row, column);
        return full;
    };
    const auto sameOperators = [](const Channel::KrausOperators &a, const Channel::KrausOperators &b) {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i)
            if ((a[i] - b[i]).cwiseAbs().maxCoeff() > 1e-12)
                return false;
        return true;
    };

    struct Backend
    {
        Type type;
        Method method;
        const char *name;
    };

    const std::vector<Backend> backends{{Type::kQiskitAer, Method::kStatevector, "Aer statevector"},
                                        {Type::kQiskitAer, Method::kMatrixProductState, "Aer MPS"},
                                        {Type::kQCSim, Method::kStatevector, "QCSim statevector"},
                                        {Type::kQCSim, Method::kMatrixProductState, "QCSim MPS"}};
    const Types::qubit_t target = 1, other = 2;
    const auto amplitudes = [](const Backend &backend, uint64_t seed, const Types::qubits_vector &targets, const Channel &channel) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(backend.type, backend.method);
        sim->AllocateQubits(3);
        sim->Initialize();
        sim->SetSeed(seed);
        sim->ApplyRy(0, 0.7);
        sim->ApplyRy(1, 1.1);
        sim->ApplyRy(2, 0.4);
        sim->ApplyCX(0, 1);
        sim->ApplyCX(1, 2);
        sim->ApplyRy(1, 0.3);
        sim->ApplyRz(1, 0.5);
        sim->ApplyQuantumChannel(targets, channel);
        std::vector<std::complex<double>> result;
        for (size_t i = 0; i < 8; ++i)
            result.push_back(sim->Amplitude(i));
        return result;
    };
    const auto distance = [](const std::vector<std::complex<double>> &a, const std::vector<std::complex<double>> &b) {
        double d = 0.;
        for (size_t i = 0; i < a.size(); ++i)
            d = std::max(d, std::abs(a[i] - b[i]));
        return d;
    };

    const Channel given(Channel::KrausOperators{(k0 + k1) / std::sqrt(2.), (k0 - k1) / std::sqrt(2.)}); // a 30% damping, in another unraveling
    const Channel identityPauli = Channel::Pauli(std::vector<double>{1., 0., 0., 0.});
    // Whether a channel rebuilt from the exported Kraus operators is the same sampling program: on
    // every backend when the channel is one set of Kraus operators; only where the Kraus operators are
    // what is sampled (the Aer MPS, for a channel with a part that reads the state) when it has
    // several parts; not at all when the operators are recognized as a Pauli or unitary mixture
    // and drawn as that (see QuantumChannel::GetFactors), or carry relaxation parameters.
    enum class Reload
    {
        kEverywhere,
        kAerMps,
        kNo
    };

    struct Base
    {
        const char *name;
        Channel channel;
        Reload reload;
    };

    const std::vector<Base> bases{
        {"given Kraus operators", given, Reload::kEverywhere},
        {"given Kraus operators, then an identity Pauli", given.Then(identityPauli), Reload::kAerMps},
        {"an identity Pauli, then given Kraus operators", identityPauli.Then(given), Reload::kAerMps},
        {"amplitude damping", Channel::AmplitudeDamping(0.3), Reload::kEverywhere},
        {"bit flip", Channel::BitFlip(0.3), Reload::kNo},
        {"bit flip, then amplitude damping", Channel::BitFlip(0.3).Then(Channel::AmplitudeDamping(0.9)), Reload::kAerMps},
        {"a mixture of i X alone", Channel::UnitaryMixture({1.}, {Matrix(phase * x)}), Reload::kNo},
        {"a mixture of two rotations", Channel::UnitaryMixture({0.5, 0.5}, {up, down}), Reload::kNo},
        {"a mixture of i I alone, then a certain bit flip", Channel::UnitaryMixture({1.}, {Matrix(phase * one)}).Then(Channel::BitFlip(1.)), Reload::kNo},
        {"dephasing given as two projectors", Channel(Channel::KrausOperators{p0, p1}), Reload::kNo},
        {"relaxation with T2 <= T1", Channel::ThermalRelaxation(3e-5, 1e-4, 0.8e-4), Reload::kNo},
        {"relaxation with T2 > T1", Channel::ThermalRelaxation(3e-5, 1e-4, 1.5e-4), Reload::kEverywhere},
        {"an identity mixture, then amplitude damping", Channel::UnitaryMixture({1.}, {one}).Then(Channel::AmplitudeDamping(0.3)), Reload::kAerMps},
        {"a damping of strength zero, then given Kraus operators", Channel::AmplitudeDamping(0.).Then(given), Reload::kAerMps},
        {"depolarizing, then relaxation", Channel::Depolarizing(0.2).Then(Channel::ThermalRelaxation(3e-5, 1e-4, 1.5e-4)), Reload::kAerMps}};

    for (const Base &base : bases)
    {
        const Channel &channel = base.channel;

        struct Presentation
        {
            const char *name;
            Types::qubits_vector targets;
            Channel channel;
        };

        const Channel onSecond = Channel::OnQubit(channel, 1), onFirst = Channel::OnQubit(channel, 0);
        const std::vector<Presentation> presentations{{"on the second of two targets", {other, target}, onSecond},
                                                      {"on the first of two targets", {target, other}, onFirst},
                                                      {"with the two targets exchanged", {target, other}, Channel::Placed(onSecond, {1, 0})},
                                                      {"reduced to its active target", {target}, onSecond.OnActiveQubits()}};
        // The two descriptions of a channel agree, in every presentation: its parts are those of the
        // one-qubit channel, and its Kraus operators are that channel's with the identity added.
        for (size_t i = 0; i < 3; ++i)
        {
            const Channel &placed = presentations[i].channel;
            const Channel reduced = placed.OnActiveQubits();
            BOOST_TEST(reduced.GetNumberOfQubits() == 1u, base.name << ", " << presentations[i].name);
            BOOST_TEST(reduced.GetFactors().size() == channel.GetFactors().size(), base.name << ", " << presentations[i].name << ": parts were lost");
            BOOST_TEST(sameOperators(reduced.GetKrausOperators(), channel.GetKrausOperators()),
                       base.name << ", " << presentations[i].name << ": the reduced channel has other Kraus operators");
            Channel::KrausOperators padded;
            for (const Matrix &op : reduced.GetKrausOperators())
                padded.push_back(pad(op, placed.HomeQubit()));
            BOOST_TEST(sameOperators(padded, placed.GetKrausOperators()),
                       base.name << ", " << presentations[i].name << ": the Kraus operators are not the reduced ones with the identity");
        }
        const Channel reloaded(onSecond.GetKrausOperators());
        BOOST_TEST(sameOperators(reloaded.OnActiveQubits().GetKrausOperators(), channel.GetKrausOperators()),
                   base.name << ": the exported Kraus operators do not reduce to the channel's");
        for (const Backend &backend : backends)
            for (uint64_t seed = 1; seed <= 6; ++seed)
            {
                const auto reference = amplitudes(backend, seed, {target}, channel);
                for (const Presentation &presentation : presentations)
                    BOOST_TEST(distance(amplitudes(backend, seed, presentation.targets, presentation.channel), reference) < 1e-9,
                               base.name << " " << presentation.name << " on " << backend.name << ", seed " << seed << ": another trajectory");
                const bool reload = base.reload == Reload::kEverywhere || (base.reload == Reload::kAerMps && std::string(backend.name) == "Aer MPS");
                if (reload)
                    BOOST_TEST(distance(amplitudes(backend, seed, {other, target}, reloaded), reference) < 1e-9,
                               base.name << " rebuilt from its Kraus operators on " << backend.name << ", seed " << seed << ": another trajectory");
            }
    }

    // A part on the target the channel otherwise leaves alone can only be a phase times the identity.
    // It stays in the reduced channel, on the remaining target: its phase is applied, its branch drawn.
    const std::vector<Base> withIdlePart{
        {"i I on the idle target, then X on the other",
         Channel::OnQubit(Channel::UnitaryMixture({1.}, {Matrix(phase * one)}), 0).Then(Channel::UnitaryMixture({1.}, {x}), {1}), Reload::kNo},
        {"a draw between I and -I on the idle target, then a bit flip",
         Channel::OnQubit(Channel::UnitaryMixture({0.5, 0.5}, {one, Matrix(-one)}), 0).Then(Channel::BitFlip(0.5), {1}), Reload::kNo},
        {"i I on the idle target, then amplitude damping",
         Channel::OnQubit(Channel::UnitaryMixture({1.}, {Matrix(phase * one)}), 0).Then(Channel::AmplitudeDamping(0.3), {1}), Reload::kNo},
        {"a damping of strength zero on the idle target, then given Kraus operators", Channel::OnQubit(Channel::AmplitudeDamping(0.), 0).Then(given, {1}),
         Reload::kNo},
        {"an identity Pauli on the idle target, then given Kraus operators", Channel::OnQubit(identityPauli, 0).Then(given, {1}), Reload::kNo},
        {"a bit flip, then i I on the idle target", Channel::OnQubit(Channel::BitFlip(0.5), 1).Then(Channel::UnitaryMixture({1.}, {Matrix(phase * one)}), {0}),
         Reload::kNo}};
    for (const Base &base : withIdlePart)
    {
        const Channel reduced = base.channel.OnActiveQubits();
        BOOST_TEST(reduced.GetNumberOfQubits() == 1u, base.name);
        BOOST_TEST(reduced.GetFactors().size() == base.channel.GetFactors().size(), base.name << ": a part was dropped");
        for (const Backend &backend : backends)
            for (uint64_t seed = 1; seed <= 6; ++seed)
                BOOST_TEST(distance(amplitudes(backend, seed, {target}, reduced), amplitudes(backend, seed, {other, target}, base.channel)) < 1e-9,
                           base.name << " on " << backend.name << ", seed " << seed << ": the reduced channel is another trajectory");
    }

    // A channel placed on the one target of a one-qubit channel is the channel, in either spelling.
    {
        const Channel projectors(Channel::KrausOperators{p0, p1});
        BOOST_TEST(sameOperators(Channel::OnQubit(projectors, 0, 1).GetKrausOperators(), Channel::KrausOperators{p0, p1}));
        BOOST_TEST(sameOperators(Channel::Placed(projectors, {0}, 1).GetKrausOperators(), Channel::KrausOperators{p0, p1}));
        BOOST_TEST(sameOperators(Channel::OnQubit(given, 0, 1).GetKrausOperators(), given.GetKrausOperators()));
    }

    // Operators that were accepted are not checked again when the channel is reduced or placed. The
    // tolerance of the check grows with the dimension: a 1% damping with sqrt(0.99) rounded to nine
    // digits is accepted on two targets and would be rejected on one, so a request the statevector
    // runs failed on the matrix product state.
    {
        Matrix r0 = Matrix::Zero(2, 2), r1 = Matrix::Zero(2, 2);
        r0(0, 0) = 1.;
        r0(1, 1) = 0.994987437;
        r1(0, 1) = 0.1;
        BOOST_CHECK_THROW(Channel(Channel::KrausOperators{r0, r1}), std::invalid_argument);
        const Channel rounded(Channel::KrausOperators{pad(r0, 1), pad(r1, 1)});
        BOOST_TEST((rounded.ActiveQubits() == std::vector<size_t>{1}));
        BOOST_TEST(sameOperators(rounded.OnActiveQubits().GetKrausOperators(), Channel::KrausOperators{r0, r1}));
        BOOST_TEST(sameOperators(Channel::Placed(rounded, {1, 0}).GetKrausOperators(), Channel::KrausOperators{pad(r0, 0), pad(r1, 0)}));
        // Applied on every backend, and the same operation however the idle target is given.
        for (const Backend &backend : backends)
            for (uint64_t seed = 1; seed <= 6; ++seed)
            {
                const auto reference = amplitudes(backend, seed, {other, target}, rounded);
                BOOST_TEST(distance(amplitudes(backend, seed, {target, other}, Channel::Placed(rounded, {1, 0})), reference) < 1e-9,
                           "rounded Kraus operators with the targets exchanged on " << backend.name << ", seed " << seed);
                BOOST_TEST(distance(amplitudes(backend, seed, {target}, rounded.OnActiveQubits()), reference) < 1e-9,
                           "rounded Kraus operators reduced on " << backend.name << ", seed " << seed);
            }
    }

    // Kraus operators given on two targets, the identity on one: the reduced channel has those
    // operators on the other, also when they are recognized as a Pauli mixture and drawn as one.
    {
        const Channel dephasing(Channel::KrausOperators{pad(p0, 1), pad(p1, 1)});
        BOOST_TEST((dephasing.ActiveQubits() == std::vector<size_t>{1}));
        BOOST_TEST(sameOperators(dephasing.OnActiveQubits().GetKrausOperators(), Channel::KrausOperators{p0, p1}));
        BOOST_TEST((dephasing.OnActiveQubits().GetFactors()[0].kind == dephasing.GetFactors()[0].kind));
    }
}

// Probability, Amplitude and Probabilities take basis-state indices, and MarginalProbabilities takes
// qubits, the same on every backend. The Aer statevector, MPS and stabilizer used to answer
// Probabilities with Aer's marginal over the outcomes read as qubits (2^k values; a read past the
// state or a crash for an outcome of the qubit count or more), and an outcome past the register was
// 0 on some backends, the probability of its low bits on others, and a read past the state on the
// QCSim statevector. It is std::out_of_range everywhere now.
BOOST_AUTO_TEST_CASE(basis_state_queries_agree_on_every_backend)
{
    using Type = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;

    struct Backend
    {
        Type type;
        Method method;
        const char *name;
        bool amplitudes;       // Amplitude is defined
        bool allProbabilities; // AllProbabilities is defined
    };

    const std::vector<Backend> backends{{Type::kQiskitAer, Method::kStatevector, "Aer statevector", true, true},
                                        {Type::kQiskitAer, Method::kMatrixProductState, "Aer MPS", true, true},
                                        {Type::kQiskitAer, Method::kStabilizer, "Aer stabilizer", false, true},
                                        {Type::kQiskitAer, Method::kExtendedStabilizer, "Aer extended stabilizer", true, true},
                                        {Type::kQiskitAer, Method::kDensityMatrix, "Aer density matrix", false, true},
                                        {Type::kQCSim, Method::kStatevector, "QCSim statevector", true, true},
                                        {Type::kQCSim, Method::kMatrixProductState, "QCSim MPS", true, true},
                                        {Type::kQCSim, Method::kMatrixProductOperator, "QCSim MPO", false, true},
                                        {Type::kQCSim, Method::kDensityMatrix, "QCSim density matrix", false, true},
                                        {Type::kQCSim, Method::kStabilizer, "QCSim stabilizer", false, true},
                                        {Type::kQCSim, Method::kExtendedStabilizer, "QCSim extended stabilizer", false, true},
                                        {Type::kQCSim, Method::kPauliPropagator, "QCSim Pauli propagator", false, true},
                                        {Type::kQCSim, Method::kPathIntegral, "QCSim path integral", true, true},
                                        {Type::kQCSim, Method::kTensorNetwork, "QCSim tensor network", false, false},
                                        {Type::kCompositeQCSim, Method::kStatevector, "QCSim composite", true, true},
                                        {Type::kCompositeQiskitAer, Method::kStatevector, "Aer composite", true, true}};
    constexpr size_t qubits = 3;
    for (const Backend &backend : backends)
    {
        BOOST_TEST_CONTEXT(backend.name)
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(backend.type, backend.method);
            BOOST_REQUIRE(sim);
            sim->AllocateQubits(qubits);
            sim->Initialize();
            sim->SetSeed(1);
            // Clifford, so that every method holds it exactly: (|000> + i|011>)/sqrt(2), then H on qubit 2.
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            sim->ApplyS(0);
            sim->ApplyH(2);
            // The tensor network has no AllProbabilities: one Probability per basis state instead.
            const std::vector<double> all = backend.allProbabilities ? sim->AllProbabilities() : Simulators::BasisProbabilities(*sim);
            BOOST_REQUIRE_EQUAL(all.size(), size_t{1} << qubits);
            BOOST_TEST(std::abs(all[0] - 0.25) < 1e-9);
            BOOST_TEST(std::abs(all[3] - 0.25) < 1e-9);

            // One value per outcome, in the order given, each what Probability returns.
            const Types::qubits_vector outcomes{7, 0, 3, 4, 6, 1, 3};
            const std::vector<double> probabilities = sim->Probabilities(outcomes);
            BOOST_REQUIRE_EQUAL(probabilities.size(), outcomes.size());
            for (size_t i = 0; i < outcomes.size(); ++i)
            {
                BOOST_TEST(std::abs(probabilities[i] - all[outcomes[i]]) < 1e-9, "outcome " << outcomes[i]);
                BOOST_TEST(std::abs(sim->Probability(outcomes[i]) - all[outcomes[i]]) < 1e-9, "outcome " << outcomes[i]);
                if (backend.amplitudes)
                    BOOST_TEST(std::abs(std::norm(sim->Amplitude(outcomes[i])) - all[outcomes[i]]) < 1e-9, "outcome " << outcomes[i]);
            }

            // Past the register: the same error everywhere, for one outcome and inside a list.
            for (const Types::qubit_t outside : {Types::qubit_t{8}, Types::qubit_t{11}, Types::qubit_t{64}, Types::qubit_t{1} << 40})
            {
                BOOST_CHECK_THROW(sim->Probability(outside), std::out_of_range);
                BOOST_CHECK_THROW(sim->Probabilities({0, outside}), std::out_of_range);
                if (backend.amplitudes)
                    BOOST_CHECK_THROW(sim->Amplitude(outside), std::out_of_range);
                else // a method without amplitudes says so first, whatever the index
                    BOOST_CHECK_THROW(sim->Amplitude(outside), std::runtime_error);
            }
            if (!backend.amplitudes)
                BOOST_CHECK_THROW(sim->Amplitude(0), std::runtime_error);

            // The distribution of some qubits, the first listed the least significant bit.
            for (const Types::qubits_vector &marginalQubits : std::vector<Types::qubits_vector>{{2, 0}, {0, 1}, {1}, {2, 1, 0}, {}})
            {
                std::vector<double> expected(size_t{1} << marginalQubits.size(), 0.);
                for (size_t index = 0; index < all.size(); ++index)
                {
                    size_t outcome = 0;
                    for (size_t k = 0; k < marginalQubits.size(); ++k)
                        if ((index >> marginalQubits[k]) & 1)
                            outcome |= size_t{1} << k;
                    expected[outcome] += all[index];
                }
                const std::vector<double> marginal = sim->MarginalProbabilities(marginalQubits);
                BOOST_REQUIRE_EQUAL(marginal.size(), expected.size());
                for (size_t i = 0; i < expected.size(); ++i)
                    BOOST_TEST(std::abs(marginal[i] - expected[i]) < 1e-9, "marginal of " << marginalQubits.size() << " qubits, outcome " << i);
            }
            BOOST_CHECK_THROW(sim->MarginalProbabilities({0, 0}), std::invalid_argument);
            BOOST_CHECK_THROW(sim->MarginalProbabilities({0, 3}), std::out_of_range);
        }
    }
}

// The edges of the basis-state queries: a register without qubits, registers too wide to list every
// basis state, an empty marginal on a wide register, and indices past 32 bits.
BOOST_AUTO_TEST_CASE(basis_state_queries_at_the_edges_of_the_register)
{
    using Type = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    const std::vector<std::pair<Type, Method>> backends{{Type::kQiskitAer, Method::kStatevector},      {Type::kQiskitAer, Method::kMatrixProductState},
                                                        {Type::kQiskitAer, Method::kStabilizer},       {Type::kQiskitAer, Method::kExtendedStabilizer},
                                                        {Type::kQiskitAer, Method::kDensityMatrix},    {Type::kQCSim, Method::kStatevector},
                                                        {Type::kQCSim, Method::kMatrixProductState},   {Type::kQCSim, Method::kMatrixProductOperator},
                                                        {Type::kQCSim, Method::kDensityMatrix},        {Type::kQCSim, Method::kStabilizer},
                                                        {Type::kQCSim, Method::kExtendedStabilizer},   {Type::kQCSim, Method::kPauliPropagator},
                                                        {Type::kQCSim, Method::kPathIntegral},         {Type::kQCSim, Method::kTensorNetwork},
                                                        {Type::kCompositeQCSim, Method::kStatevector}, {Type::kCompositeQiskitAer, Method::kStatevector}};
    // No qubits: no state to query, and an error rather than a read through a missing state.
    for (const auto &[type, method] : backends)
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, method);
        sim->AllocateQubits(0);
        sim->Initialize();
        BOOST_CHECK_THROW(sim->Probability(0), std::runtime_error);
        BOOST_CHECK_THROW(sim->Probabilities({0}), std::runtime_error);
        BOOST_CHECK_THROW(sim->Probabilities({}), std::runtime_error); // no outcomes: still no state
        BOOST_CHECK_THROW(sim->Amplitude(0), std::runtime_error);
        BOOST_CHECK_THROW(sim->MarginalProbabilities({}), std::runtime_error);
        BOOST_CHECK_THROW(sim->AllProbabilities(), std::runtime_error);

        // Cleared, a register has no qubits either.
        auto cleared = Simulators::SimulatorsFactory::CreateSimulator(type, method);
        cleared->AllocateQubits(3);
        cleared->Initialize();
        cleared->Clear();
        BOOST_CHECK_THROW(cleared->Probabilities({}), std::runtime_error);
        BOOST_CHECK_THROW(cleared->Probability(0), std::runtime_error);
        BOOST_CHECK_THROW(cleared->AllProbabilities(), std::runtime_error);
    }

    // Past 64 qubits an index has no bit for qubit 64 or above: basis state 1 is qubit 0 alone,
    // and a state with qubit 64 set has no weight on any index. (Aer's MPS, before qiskit-aer
    // 0acd866, and the path integral shifted by 64, which wrapped to qubit 0 on x86, and read basis
    // state 1 as 0.)
    struct Past64
    {
        Type type;
        Method method;
        bool amplitudes;
    };

    for (const Past64 &backend : {Past64{Type::kQiskitAer, Method::kMatrixProductState, true}, Past64{Type::kQiskitAer, Method::kStabilizer, false},
                                  Past64{Type::kQCSim, Method::kMatrixProductState, true}, Past64{Type::kQCSim, Method::kMatrixProductOperator, false},
                                  Past64{Type::kQCSim, Method::kStabilizer, false}, Past64{Type::kQCSim, Method::kPauliPropagator, false},
                                  Past64{Type::kQCSim, Method::kPathIntegral, true}, Past64{Type::kQCSim, Method::kTensorNetwork, false},
                                  Past64{Type::kCompositeQCSim, Method::kStatevector, true}, Past64{Type::kCompositeQiskitAer, Method::kStatevector, true}})
    {
        for (const size_t flipped : {size_t{0}, size_t{64}})
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(backend.type, backend.method);
            sim->AllocateQubits(65);
            sim->Initialize();
            sim->ApplyX(flipped);
            const double expected = flipped == 0 ? 1. : 0.;
            BOOST_TEST(std::abs(sim->Probability(1) - expected) < 1e-9, "65 qubits, X on " << flipped);
            BOOST_TEST(std::abs(sim->Probabilities({1})[0] - expected) < 1e-9, "65 qubits, X on " << flipped);
            BOOST_TEST(sim->Probability(0) < 1e-9, "65 qubits, X on " << flipped);
            if (backend.amplitudes)
                BOOST_TEST(std::abs(std::norm(sim->Amplitude(1)) - expected) < 1e-9, "65 qubits, X on " << flipped);
        }
    }

    // Too wide to list every basis state: the marginal throws rather than summing nothing.
    struct Wide
    {
        Type type;
        Method method;
        size_t qubits;
    };

    // The composite at 63 qubits: 2^63 values cannot be held, and the list grew without end. QCSim's
    // MPS at 40 threw QCSim's own error, which Python raised as RuntimeError, not ValueError.
    for (const Wide &wide : {Wide{Type::kQCSim, Method::kMatrixProductState, 32}, Wide{Type::kQCSim, Method::kMatrixProductState, 40},
                             Wide{Type::kQCSim, Method::kMatrixProductOperator, 64}, Wide{Type::kQCSim, Method::kPathIntegral, 64},
                             Wide{Type::kQCSim, Method::kPauliPropagator, 64}, Wide{Type::kCompositeQCSim, Method::kStatevector, 64},
                             Wide{Type::kCompositeQCSim, Method::kStatevector, 63}})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(wide.type, wide.method);
        sim->AllocateQubits(wide.qubits);
        sim->Initialize();
        sim->ApplyX(wide.qubits - 1);
        BOOST_CHECK_THROW(sim->MarginalProbabilities({wide.qubits - 1}), std::length_error);
        BOOST_CHECK_THROW(sim->AllProbabilities(), std::length_error);
        BOOST_TEST(std::abs(sim->Probability(Types::qubit_t{1} << (wide.qubits - 1)) - 1.) < 1e-9);
    }

    // QCSim's stabilizer lists its distribution up to 32 qubits; Maestro refuses above, before QCSim
    // throws its own error.
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kStabilizer);
        sim->AllocateQubits(40);
        sim->Initialize();
        sim->ApplyX(39);
        BOOST_CHECK_THROW(sim->MarginalProbabilities({39}), std::length_error);
        BOOST_TEST(std::abs(sim->Probability(Types::qubit_t{1} << 39) - 1.) < 1e-12);
    }

    // Aer's stabilizer takes 64 qubits for its distribution and sized it with 1 << 64, which listed
    // one value: Maestro refuses before asking. Its marginal is direct, so a qubit's still works.
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQiskitAer, Method::kStabilizer);
        sim->AllocateQubits(64);
        sim->Initialize();
        sim->ApplyX(63);
        BOOST_CHECK_THROW(sim->AllProbabilities(), std::length_error);
        const auto last = sim->MarginalProbabilities({63});
        BOOST_REQUIRE_EQUAL(last.size(), 2u);
        BOOST_TEST(std::abs(last[1] - 1.) < 1e-12);
    }

    // An empty marginal is the total probability, read at any width where the backend has it, so it
    // holds on registers too wide to list (QCSim MPS at 32, path integral at 64, ...).
    for (const Wide &wide :
         {Wide{Type::kQCSim, Method::kMatrixProductState, 32}, Wide{Type::kQCSim, Method::kPathIntegral, 64}, Wide{Type::kQCSim, Method::kPauliPropagator, 64},
          Wide{Type::kCompositeQCSim, Method::kStatevector, 64}, Wide{Type::kQCSim, Method::kStabilizer, 100}, Wide{Type::kQiskitAer, Method::kStabilizer, 65},
          Wide{Type::kQiskitAer, Method::kMatrixProductState, 70}})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(wide.type, wide.method);
        sim->AllocateQubits(wide.qubits);
        sim->Initialize();
        sim->ApplyH(0);
        sim->ApplyCX(0, wide.qubits - 1);
        const auto total = sim->MarginalProbabilities({});
        BOOST_REQUIRE_EQUAL(total.size(), 1u);
        BOOST_TEST(std::abs(total[0] - 1.) < 1e-12, wide.qubits << " qubits");
    }

    // Aer's extended stabilizer has no single amplitude, only the statevector, which Aer sizes in
    // bytes: from 60 qubits the count wrapped and Aer wrote past its buffer. Refused first.
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQiskitAer, Method::kExtendedStabilizer);
        sim->AllocateQubits(60);
        sim->Initialize();
        BOOST_CHECK_THROW(sim->Amplitude(0), std::length_error);
        BOOST_CHECK_THROW(sim->AllProbabilities(), std::length_error);
    }

    // Indices past 32 bits on the QCSim MPS, read through the bits of the index.
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kMatrixProductState);
        sim->AllocateQubits(40);
        sim->Initialize();
        sim->ApplyX(0);
        sim->ApplyX(35);
        const Types::qubit_t state = (Types::qubit_t{1} << 35) | 1;
        BOOST_TEST(std::abs(sim->Probability(state) - 1.) < 1e-12);
        BOOST_TEST(std::abs(std::norm(sim->Amplitude(state)) - 1.) < 1e-12);
        BOOST_TEST(std::abs(sim->Probability(1)) < 1e-12);
    }
}

// The empty marginal is the total probability of the state, as every other query reports it: a
// generic gate with a non-unitary matrix can leave it below 1 on the dense backends, the tensor
// network and the composite, where nothing renormalizes the state (the matrix product states do:
// see the test after this one). A constant 1 disagreed with the distribution after a projection.
BOOST_AUTO_TEST_CASE(empty_marginal_is_the_total_probability)
{
    using Type = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    Eigen::Matrix2cd projector = Eigen::Matrix2cd::Zero();
    projector(0, 0) = 1.; // onto |0>
    const auto sumOf = [](const std::vector<double> &values) {
        double sum = 0.;
        for (const double p : values)
            sum += p;
        return sum;
    };

    // The backends that take the matrix: the total and every one-qubit marginal agree with the whole
    // distribution after a projection on qubit 0, on a qubit entangled with it, on a qubit apart
    // from it (the tensor network contracted qubit 0's group only), and after a two-qubit matrix.
    // (The matrix product states renormalize instead: see
    // matrix_product_state_stays_normalized_after_a_matrix_that_is_not_unitary.)
    for (const auto &[type, method] : std::vector<std::pair<Type, Method>>{{Type::kQiskitAer, Method::kStatevector},
                                                                           {Type::kQiskitAer, Method::kDensityMatrix},
                                                                           {Type::kQCSim, Method::kStatevector},
                                                                           {Type::kQCSim, Method::kDensityMatrix},
                                                                           {Type::kQCSim, Method::kTensorNetwork},
                                                                           {Type::kCompositeQCSim, Method::kStatevector},
                                                                           {Type::kCompositeQiskitAer, Method::kStatevector}})
    {
        for (int operation = 0; operation < 4; ++operation)
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, method);
            sim->AllocateQubits(3);
            sim->Initialize();
            sim->ApplyH(2);
            sim->ApplyCX(2, 0);
            if (operation == 0)
                sim->ApplyGenericOneQubitGate(0, projector);
            else if (operation == 1)
                sim->ApplyGenericOneQubitGate(2, projector);
            else if (operation == 2)
            {
                sim->ApplyH(1); // qubit 1 is apart from qubits 0 and 2
                sim->ApplyGenericOneQubitGate(1, projector);
            }
            else
                sim->ApplyGenericTwoQubitGate(1, 2, Eigen::Matrix4cd::Identity() * .5);
            const std::vector<double> all = Simulators::BasisProbabilities(*sim);
            const double whole = sumOf(all);
            const auto total = sim->MarginalProbabilities({});
            BOOST_REQUIRE_EQUAL(total.size(), 1u);
            BOOST_TEST(std::abs(total[0] - whole) < 1e-9, "operation " << operation);
            BOOST_TEST(std::abs(sim->TotalProbability() - whole) < 1e-9, "operation " << operation);
            if (operation < 3)
                BOOST_TEST(std::abs(whole - .5) < 1e-9, "operation " << operation);
            for (const size_t qubit : {0, 1, 2})
            {
                const auto marginal = sim->MarginalProbabilities({qubit});
                double zero = 0., one = 0.;
                for (size_t index = 0; index < all.size(); ++index)
                    ((index >> qubit) & 1 ? one : zero) += all[index];
                BOOST_TEST(std::abs(marginal[0] - zero) < 1e-9, "operation " << operation << ", qubit " << qubit);
                BOOST_TEST(std::abs(marginal[1] - one) < 1e-9, "operation " << operation << ", qubit " << qubit);
            }
        }
    }

    struct Wide
    {
        Type type;
        Method method;
        size_t qubits;
    };

    // Past what the distribution can list: the composite has the product of its components and the
    // tensor network contracts each weight of a qubit, so they keep answering.
    for (const Wide &wide : {Wide{Type::kQCSim, Method::kTensorNetwork, 70}, Wide{Type::kCompositeQCSim, Method::kStatevector, 65},
                             Wide{Type::kCompositeQiskitAer, Method::kStatevector, 65}})
    {
        for (const bool apart : {false, true}) // the projected qubit entangled with qubit 0, or apart from it
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(wide.type, wide.method);
            sim->AllocateQubits(wide.qubits);
            sim->Initialize();
            sim->ApplyH(wide.qubits - 1);
            if (!apart)
                sim->ApplyCX(wide.qubits - 1, 0);
            sim->ApplyGenericOneQubitGate(wide.qubits - 1, projector);
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - .5) < 1e-9, wide.qubits << " qubits, apart " << apart);
        }
    }
    // The path integral drops the paths below its threshold and keeps the rest as they are.
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kPathIntegral);
        sim->Configure("path_integral_threshold", "0.3");
        sim->AllocateQubits(1);
        sim->Initialize();
        sim->ApplyRy(0, M_PI / 3.);
        const double whole = sim->Probability(0) + sim->Probability(1);
        BOOST_TEST(std::abs(whole - .75) < 1e-9);
        BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - whole) < 1e-12);
    }
    // A matrix product operator normalizes its probabilities by its trace, which ApplyOperator
    // without normalize leaves where it falls: they total 1, and where the trace cannot normalize
    // them the empty marginal throws as they do.
    for (const bool zeroTrace : {false, true})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kMatrixProductOperator);
        sim->AllocateQubits(2);
        sim->Initialize();
        Eigen::MatrixXcd onto = Eigen::MatrixXcd::Zero(2, 2);
        if (zeroTrace)
            onto(1, 1) = 1.; // |0> onto |1>: trace 0
        else
        {
            sim->ApplyH(0);
            onto(0, 0) = 1.; // trace 0.5
        }
        sim->ApplyOperator({0}, onto, false);
        if (zeroTrace)
        {
            BOOST_CHECK_THROW(sim->Probability(0), std::runtime_error);
            BOOST_CHECK_THROW(sim->MarginalProbabilities({}), std::runtime_error);
        }
        else
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - sumOf(sim->AllProbabilities())) < 1e-12);
    }
    // ApplyLocalMapNorm leaves the state as the matrix leaves it.
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kStatevector);
        sim->AllocateQubits(1);
        sim->Initialize();
        BOOST_TEST(std::abs(sim->ApplyLocalMapNorm({0}, Eigen::MatrixXcd::Identity(2, 2) * .5) - .25) < 1e-12);
        BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - .25) < 1e-12);
    }
}

// A matrix product state stays normalized. A generic gate whose matrix is not unitary left more
// than a weight on the chain: the Schmidt values of its other bonds were stale, so that reads of
// one qubit and samples came from a state it no longer held (a qubit of a Bell pair whose other
// qubit had been projected onto zero was sampled as one half the time), Aer's save, which sorts
// the qubits with swaps, renormalized the live state, and the total could only be had from the
// whole distribution, which a wide register cannot list. Such a matrix is now applied as the one
// operator of a channel, which leaves the chain normalized and its reads valid; one that leaves
// nothing of the state is refused.
BOOST_AUTO_TEST_CASE(matrix_product_state_stays_normalized_after_a_matrix_that_is_not_unitary)
{
    using Type = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    Eigen::Matrix2cd projector = Eigen::Matrix2cd::Zero();
    projector(0, 0) = 1.; // onto |0>
    Eigen::Matrix2cd damping;
    damping << 1., 0., 0., .5;
    Eigen::Matrix4cd pairFilter = Eigen::Matrix4cd::Zero();
    pairFilter.diagonal() << 1., .5, .5, .25;
    pairFilter(0, 3) = std::complex<double>(.2, .1);
    const auto sumOf = [](const std::vector<double> &values) {
        double sum = 0.;
        for (const double p : values)
            sum += p;
        return sum;
    };
    const auto prepare = [](Simulators::ISimulator &sim) {
        sim.AllocateQubits(4);
        sim.Initialize();
        sim.ApplyH(0);
        sim.ApplyCX(0, 1);
        sim.ApplyCX(1, 2);
        sim.ApplyRy(2, .7);
        sim.ApplyCX(2, 3);
        sim.ApplyRx(1, .4);
    };
    // The operations: a filter on a qubit at an end and on one inside, a projection, a two-qubit
    // filter on neighbours and on qubits apart, each followed by gates.
    const auto operate = [&](Simulators::ISimulator &sim, int operation) {
        if (operation == 0)
            sim.ApplyGenericOneQubitGate(0, damping);
        else if (operation == 1)
            sim.ApplyGenericOneQubitGate(2, damping);
        else if (operation == 2)
            sim.ApplyGenericOneQubitGate(3, projector);
        else if (operation == 3)
            sim.ApplyGenericTwoQubitGate(1, 2, pairFilter);
        else
            sim.ApplyGenericTwoQubitGate(3, 0, pairFilter);
        sim.ApplyH(1);
        sim.ApplyCX(3, 1);
    };

    for (const Type type : {Type::kQCSim, Type::kQiskitAer})
    {
        for (int operation = 0; operation < 5; ++operation)
        {
            auto reference = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kStatevector);
            prepare(*reference);
            operate(*reference, operation);
            std::vector<double> expected = reference->AllProbabilities();
            const double weight = sumOf(expected);
            BOOST_REQUIRE(weight < .95); // the matrix does move the norm
            for (double &p : expected)
                p /= weight;

            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            prepare(*sim);
            operate(*sim, operation);
            const std::string where = "backend " + std::to_string(static_cast<int>(type)) + ", operation " + std::to_string(operation);
            const std::vector<double> all = sim->AllProbabilities();
            for (size_t index = 0; index < all.size(); ++index)
                BOOST_TEST(std::abs(all[index] - expected[index]) < 1e-9, where << ", basis state " << index);
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - 1.) < 1e-12, where);
            BOOST_TEST(std::abs(sim->TotalProbability() - 1.) < 1e-12, where);
            for (const size_t qubit : {0, 1, 2, 3})
            {
                double one = 0.;
                for (size_t index = 0; index < expected.size(); ++index)
                    if ((index >> qubit) & 1)
                        one += expected[index];
                const auto marginal = sim->MarginalProbabilities({qubit});
                BOOST_TEST(std::abs(marginal[1] - one) < 1e-9, where << ", qubit " << qubit);
                BOOST_TEST(std::abs(marginal[0] - (1. - one)) < 1e-9, where << ", qubit " << qubit);
                // Samples of one qubit follow the state (they came from the stale Schmidt values).
                const size_t shots = 20000;
                auto counts = sim->SampleCounts({qubit}, shots);
                const double frequency = static_cast<double>(counts[1]) / shots;
                BOOST_TEST(std::abs(frequency - one) < 5. * std::sqrt(.25 / shots), where << ", sampled qubit " << qubit);
            }
            // Saving, restoring and cloning leave the state as it is.
            sim->SaveState();
            const std::vector<double> saved = sim->AllProbabilities();
            sim->ApplyX(0);
            sim->RestoreState();
            const std::vector<double> restored = sim->AllProbabilities();
            const auto clone = sim->Clone();
            const std::vector<double> cloned = clone->AllProbabilities(), original = sim->AllProbabilities();
            for (size_t index = 0; index < all.size(); ++index)
            {
                BOOST_TEST(std::abs(saved[index] - all[index]) < 1e-12, where << ", after a save");
                BOOST_TEST(std::abs(restored[index] - all[index]) < 1e-12, where << ", after a restore");
                BOOST_TEST(std::abs(cloned[index] - all[index]) < 1e-12, where << ", in a clone");
                BOOST_TEST(std::abs(original[index] - all[index]) < 1e-12, where << ", after a clone");
            }
        }

        // The partner of a projected qubit of a Bell pair: zero in every read and in every shot.
        // (The matrix leaves the norm at 1, and still left the partner's Schmidt values stale.)
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(2);
            sim->Initialize();
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            sim->ApplyGenericOneQubitGate(0, Eigen::Matrix2cd(projector * std::sqrt(2.)));
            const auto marginal = sim->MarginalProbabilities({1});
            BOOST_TEST(std::abs(marginal[0] - 1.) < 1e-12);
            auto counts = sim->SampleCounts({1}, 128);
            BOOST_TEST(counts[0] == 128u);
            BOOST_TEST(counts[1] == 0u);
        }

        // A save does not change what a read returns (Aer's renormalized a chain off its norm).
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(3);
            sim->Initialize();
            sim->ApplyH(2);
            sim->ApplyCX(2, 0);
            sim->ApplyGenericOneQubitGate(2, projector);
            const double before = sim->Probability(0);
            BOOST_TEST(std::abs(before - 1.) < 1e-12);
            sim->SaveState();
            BOOST_TEST(std::abs(sim->Probability(0) - before) < 1e-12);
            sim->RestoreState();
            BOOST_TEST(std::abs(sim->Probability(0) - before) < 1e-12);
            const auto clone = sim->Clone();
            BOOST_TEST(std::abs(sim->Probability(0) - before) < 1e-12);
            BOOST_TEST(std::abs(clone->Probability(0) - before) < 1e-12);
        }

        // At a width the distribution cannot list: the total, and the reads of basis states.
        {
            const size_t qubits = type == Type::kQCSim ? 40 : 70;
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(qubits);
            sim->Initialize();
            sim->ApplyX(5);
            sim->ApplyH(qubits - 1);
            sim->ApplyCX(qubits - 1, 1);
            sim->ApplyGenericOneQubitGate(0, projector);          // qubit 0 is zero already
            sim->ApplyGenericOneQubitGate(qubits - 1, projector); // half of the state
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - 1.) < 1e-12, qubits << " qubits");
            BOOST_TEST(std::abs(sim->Probability(32) - 1.) < 1e-12, qubits << " qubits");
            if (type == Type::kQiskitAer) // Aer reads a qubit of the chain at any width
            {
                const auto set = sim->MarginalProbabilities({5}), partner = sim->MarginalProbabilities({1});
                BOOST_TEST(std::abs(set[1] - 1.) < 1e-12);
                BOOST_TEST(std::abs(partner[0] - 1.) < 1e-12);
            }
        }

        // Matrices each close to unitary move the norm by their sum: 65536 of them, each 6e-13
        // short and below what one matrix is taken for unitary at, left 0.99999996.
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(2);
            sim->Initialize();
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            const Eigen::Matrix2cd almost = Eigen::Matrix2cd::Identity() * (1. - 3e-13);
            for (int gate = 0; gate < 65536; ++gate)
                sim->ApplyGenericOneQubitGate(gate % 2, almost);
            BOOST_TEST(std::abs(sumOf(sim->AllProbabilities()) - 1.) < 1e-9);
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - 1.) < 1e-12);
        }

        // A matrix that leaves nothing of the state is refused, and the state is as it was.
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(3);
            sim->Initialize();
            sim->ApplyX(0);
            sim->ApplyH(1);
            sim->ApplyCX(1, 2);
            const std::vector<double> before = sim->AllProbabilities();
            Eigen::Matrix4cd both = Eigen::Matrix4cd::Zero();
            both(0, 0) = 1.;
            BOOST_CHECK_THROW(sim->ApplyGenericOneQubitGate(0, projector), std::runtime_error);
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(0, 2, both), std::runtime_error);
            const std::vector<double> after = sim->AllProbabilities();
            for (size_t index = 0; index < before.size(); ++index)
                BOOST_TEST(std::abs(after[index] - before[index]) < 1e-12);
        }

        // A small weight is a weight: P(1) = 9e-14 projected onto |1> is |1>, and an invertible matrix
        // leaves a state whatever it meets (Aer refused both, below a cutoff of 1e-13). One qubit,
        // and one entangled with another; a projection along the basis and one turned from it.
        for (const double probability : {1e-3, 9e-14, 1e-14})
            for (const size_t qubits : {size_t(1), size_t(2)})
                for (const bool turned : {false, true})
                {
                    auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
                    auto reference = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kStatevector);
                    Eigen::Matrix2cd onto = Eigen::Matrix2cd::Zero();
                    onto(1, 1) = 1.;
                    if (turned) // onto cos(a)|0> + sin(a)|1>, a just short of a quarter turn
                    {
                        const double c = std::cos(M_PI / 2. - 1e-3), s = std::sin(M_PI / 2. - 1e-3);
                        onto << c * c, c * s, c * s, s * s;
                    }
                    for (auto *each : {sim.get(), reference.get()})
                    {
                        each->AllocateQubits(qubits);
                        each->Initialize();
                        each->ApplyRy(0, 2. * std::asin(std::sqrt(probability)));
                        if (qubits == 2)
                            each->ApplyCX(0, 1);
                        each->ApplyGenericOneQubitGate(0, onto);
                    }
                    std::vector<double> expected = reference->AllProbabilities();
                    const double weight = sumOf(expected);
                    const std::vector<double> all = sim->AllProbabilities();
                    for (size_t index = 0; index < all.size(); ++index)
                        BOOST_TEST(std::abs(all[index] - expected[index] / weight) < 1e-8,
                                   "P(1) " << probability << ", " << qubits << " qubits, turned " << turned << ", basis state " << index);
                }
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(2);
            sim->Initialize();
            sim->ApplyX(0);
            Eigen::Matrix2cd invertible;
            invertible << 1., 0., 0., 1e-7;
            sim->ApplyGenericOneQubitGate(0, invertible);
            BOOST_TEST(std::abs(sim->Probability(1) - 1.) < 1e-12);
            sim->ApplyH(1);
            sim->ApplyGenericTwoQubitGate(0, 1, Eigen::Matrix4cd(Eigen::Matrix4cd::Identity() * 1e-7));
            BOOST_TEST(std::abs(sim->Probability(1) - .5) < 1e-9);
            BOOST_TEST(std::abs(sim->Probability(3) - .5) < 1e-9);
            // A two-qubit projection that is not along the basis, onto (|00> + |11>) / sqrt(2), of
            // a state that holds 1e-12 of it.
            auto pair = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            pair->AllocateQubits(3);
            pair->Initialize();
            pair->ApplyH(0);
            pair->ApplyCX(0, 2);
            pair->ApplyZ(0);       // (|00> - |11>) / sqrt(2) on qubits 0 and 2: nothing of it
            pair->ApplyP(0, 2e-6); // (|00> - e^(i phi)|11>) / sqrt(2): sin^2(phi / 2) = 1e-12 of it
            Eigen::Matrix4cd bell = Eigen::Matrix4cd::Zero();
            bell(0, 0) = bell(0, 3) = bell(3, 0) = bell(3, 3) = .5;
            pair->ApplyGenericTwoQubitGate(0, 2, bell);
            BOOST_TEST(std::abs(pair->Probability(0) - .5) < 1e-6);
            BOOST_TEST(std::abs(pair->Probability(5) - .5) < 1e-6);
        }

        // A weight the chain cannot tell from none is none. |Phi-> with one qubit turned has no
        // weight on |Phi+>, whatever the angle; but a chain holds its state only to what its
        // truncation cuts (Aer's cuts Schmidt weights below 1e-16), and after a turn of 2e-4 Aer's
        // held 2.5e-17 of |Phi+>, far above rounding: the projection is refused, not returned as
        // a state.
        for (const double angle : {2e-2, 2e-4, 2e-5, 2e-6, 2e-7})
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(3);
            sim->Initialize();
            sim->ApplyH(0);
            sim->ApplyCX(0, 2);
            sim->ApplyZ(0);
            sim->ApplyRy(2, angle);
            Eigen::Matrix4cd bell = Eigen::Matrix4cd::Zero();
            bell(0, 0) = bell(0, 3) = bell(3, 0) = bell(3, 3) = .5;
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(0, 2, bell), std::runtime_error);
        }

        // A matrix can shift the probabilities of its qubit and leave its weight where it was:
        // diag(1 + e, 1 - e) on a qubit of a Bell pair, 15000 times with e = 2e-11, moves the
        // partner's zero to 0.5000003. Aer read the partner from Schmidt values that were never
        // brought up to date (0.5), since the qubit the matrices met still weighed 1.
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(3);
            sim->Initialize();
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            sim->ApplyCX(1, 2);
            Eigen::Matrix2cd weak;
            weak << 1. + 2e-11, 0., 0., 1. - 2e-11;
            for (int gate = 0; gate < 15000; ++gate)
                sim->ApplyGenericOneQubitGate(gate % 2 ? 2 : 0, weak);
            const std::vector<double> all = sim->AllProbabilities();
            BOOST_TEST(std::abs(all[0] - .5000003) < 2e-9);
            for (const size_t qubit : {0, 1, 2})
            {
                double zero = 0.;
                for (size_t index = 0; index < all.size(); ++index)
                    if (!((index >> qubit) & 1))
                        zero += all[index];
                BOOST_TEST(std::abs(sim->MarginalProbabilities({qubit})[0] - zero) < 1e-9, "weak filters, qubit " << qubit);
            }
        }

        // Refused matrices that are not along the basis: the state is as it was, to rounding (Aer
        // weighs them with the qubits turned, and turns them back).
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(3);
            sim->Initialize();
            sim->ApplyH(0); // |+>
            sim->ApplyH(1);
            sim->ApplyCX(1, 2);
            sim->ApplyZ(1); // (|00> - |11>) / sqrt(2) on qubits 1 and 2
            const std::vector<double> before = sim->AllProbabilities();
            Eigen::Matrix2cd minus;
            minus << .5, -.5, -.5, .5; // onto |->
            Eigen::Matrix4cd bell = Eigen::Matrix4cd::Zero();
            bell(0, 0) = bell(0, 3) = bell(3, 0) = bell(3, 3) = .5;
            BOOST_CHECK_THROW(sim->ApplyGenericOneQubitGate(0, minus), std::runtime_error);
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(1, 2, bell), std::runtime_error);
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(0, 1, Eigen::Matrix4cd(Eigen::Matrix4cd::Zero())), std::runtime_error);
            const std::vector<double> after = sim->AllProbabilities();
            for (size_t index = 0; index < before.size(); ++index)
                BOOST_TEST(std::abs(after[index] - before[index]) < 1e-12);
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - sumOf(after)) < 1e-12);
        }

        // Under a bond limit too a refused matrix leaves the state as it was. With a limit of 2,
        // Bell pairs on qubits (0, 2) and (1, 3) are a state that any further update of a bond
        // between them can cut: Aer weighed a two-qubit matrix that is not along the basis with
        // the pair turned on the live chain, an update that turning back did not undo.
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->Configure("matrix_product_state_max_bond_dimension", "2");
            sim->AllocateQubits(4);
            sim->Initialize();
            sim->ApplyH(0);
            sim->ApplyCX(0, 2);
            sim->ApplyZ(0); // (|00> - |11>) / sqrt(2) on qubits 0 and 2
            sim->ApplyH(1);
            sim->ApplyCX(1, 3);
            const std::vector<double> before = sim->AllProbabilities();
            BOOST_REQUIRE(std::abs(before[0] - .25) < 1e-12);
            Eigen::Matrix4cd bell = Eigen::Matrix4cd::Zero();
            bell(0, 0) = bell(0, 3) = bell(3, 0) = bell(3, 3) = .5; // onto (|00> + |11>) / sqrt(2): nothing of it
            Eigen::Matrix4cd both = Eigen::Matrix4cd::Zero();
            both(1, 1) = 1.; // onto |01>, along the basis: nothing of it either
            Eigen::Matrix2cd minus;
            minus << .5, -.5, -.5, .5;
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(0, 2, bell), std::runtime_error);
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(2, 0, both), std::runtime_error);
            BOOST_CHECK_THROW(sim->ApplyGenericTwoQubitGate(0, 1, Eigen::Matrix4cd(Eigen::Matrix4cd::Zero())), std::runtime_error);
            const std::vector<double> after = sim->AllProbabilities();
            for (size_t index = 0; index < before.size(); ++index)
                BOOST_TEST(std::abs(after[index] - before[index]) < 1e-12, "bond limit 2, basis state " << index);
            // And a matrix that is applied there leaves the chain normalized.
            sim->ApplyGenericTwoQubitGate(1, 3, bell);
            BOOST_TEST(std::abs(sumOf(sim->AllProbabilities()) - 1.) < 1e-9);
        }

        // A register of one qubit has no bond to renormalize at.
        {
            auto sim = Simulators::SimulatorsFactory::CreateSimulator(type, Method::kMatrixProductState);
            sim->AllocateQubits(1);
            sim->Initialize();
            sim->ApplyH(0);
            sim->ApplyGenericOneQubitGate(0, Eigen::Matrix2cd(projector * .25));
            BOOST_TEST(std::abs(sim->Probability(0) - 1.) < 1e-12);
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - 1.) < 1e-12);
        }
    }

    // Matrices unitary to rounding, however many: gate fusion hands over products of gates (here
    // of 40 each, about 3e-15 from unitary), and a long circuit of them leaves a wide register
    // readable (120000 of them add up to more than the tolerance, and the chain is then rebuilt).
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQCSim, Method::kMatrixProductState);
        sim->AllocateQubits(40);
        sim->Initialize();
        std::mt19937 generator(9);
        std::uniform_real_distribution<double> angle(0., 2. * M_PI);
        for (int gate = 0; gate < 120000; ++gate)
        {
            Eigen::Matrix2cd product = Eigen::Matrix2cd::Identity();
            for (int factor = 0; factor < 40; ++factor)
            {
                const double theta = angle(generator), phi = angle(generator), lambda = angle(generator);
                Eigen::Matrix2cd unitary;
                unitary << std::cos(theta / 2.), -std::polar(1., lambda) * std::sin(theta / 2.), std::polar(1., phi) * std::sin(theta / 2.),
                    std::polar(1., phi + lambda) * std::cos(theta / 2.);
                product = unitary * product;
            }
            sim->ApplyGenericOneQubitGate(gate % 40, product);
        }
        BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - 1.) < 1e-12);
        const double zero = sim->Probability(0);
        BOOST_TEST((std::isfinite(zero) && zero >= 0. && zero <= 1.));
    }

    // A list of amplitudes is a state of the statevector method. Aer's initializer builds a dense
    // statevector whatever the simulator was made for, so a matrix product state given amplitudes
    // went on as a dense state under its own name (a bond limit of 1 no longer cut a Bell pair made
    // on it, and a filter was applied to it as to a statevector): it is refused, as on QCSim and
    // the GPU, and so are the stabilizers, before anything of the state they hold is cleared.
    for (const Method method : {Method::kMatrixProductState, Method::kStabilizer, Method::kExtendedStabilizer})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Type::kQiskitAer, method);
        sim->AllocateQubits(2);
        sim->Initialize();
        sim->ApplyH(0);
        sim->ApplyCX(0, 1);
        const double before = sim->Probability(3);
        std::vector<std::complex<double>> amplitudes{1., 0., 0., 0.};
        BOOST_CHECK_THROW(sim->InitializeState(2, amplitudes), std::runtime_error);
        Eigen::VectorXcd eigenAmplitudes = Eigen::VectorXcd::Unit(4, 0);
        BOOST_CHECK_THROW(sim->InitializeState(2, eigenAmplitudes), std::runtime_error);
        BOOST_TEST(std::abs(sim->Probability(3) - before) < 1e-12, "method " << static_cast<int>(method));
    }
}

// The tensor network's total is one contraction of each of its independent groups of qubits, after
// one pass over the network. It went through every tensor of the register once per group, and so
// did the contractor's cache of plans, which made it quadratic in the number of qubits (seconds for
// a few thousand qubits that no gate connects; milliseconds now, measured apart from the tests:
// nothing here depends on the time it takes). The total is right whatever the number of groups,
// for a group apart from qubit 0's, and again when the plans are found cached, also past what the
// cache holds, where it drops the oldest.
BOOST_AUTO_TEST_CASE(tensor_network_total_probability_over_independent_groups)
{
    Eigen::Matrix2cd projector = Eigen::Matrix2cd::Zero();
    projector(0, 0) = 1.;
    for (const size_t qubits : {size_t(3), size_t(64), size_t(4096), size_t(20000)})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kTensorNetwork);
        sim->AllocateQubits(qubits);
        sim->Initialize();
        sim->ApplyH(qubits - 1);
        sim->ApplyCX(qubits - 1, 0);
        sim->ApplyH(qubits / 2);
        sim->ApplyGenericOneQubitGate(qubits / 2, projector); // a group apart from qubit 0's: half the weight
        for (int repeat = 0; repeat < 3; ++repeat)            // the second and third find the plans cached
            BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - .5) < 1e-9, qubits << " qubits, pass " << repeat);
        // A second matrix, on the group of qubit 0: a quarter is left, and the plans of the other groups still hold.
        sim->ApplyH(0);
        sim->ApplyGenericOneQubitGate(0, projector);
        BOOST_TEST(std::abs(sim->MarginalProbabilities({})[0] - .25) < 1e-9, qubits << " qubits, two groups");
    }
}

// A statevector given to Aer must hold the amplitudes of the register: a shorter one was read past
// its end (and 1 << n sized it, undefined from 64 qubits).
BOOST_AUTO_TEST_CASE(aer_initial_statevector_length_is_checked)
{
    for (const auto method : {Simulators::SimulationType::kStatevector, Simulators::SimulationType::kDensityMatrix})
    {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, method);
        std::vector<std::complex<double>> shortVector(4, std::complex<double>(.5, 0.));
        BOOST_CHECK_THROW(sim->InitializeState(3, shortVector), std::invalid_argument);
        std::vector<std::complex<double>> state(8, 0.);
        state[5] = 1.;
        sim->InitializeState(3, state);
        BOOST_TEST(std::abs(sim->Probability(5) - 1.) < 1e-12);
    }
}

// Aer's extended stabilizer Probability is an estimate: the exact amplitude of the decomposition
// over a sampled estimate of its norm, the same on every call for a given state. Probabilities is
// one Probability per index, so two simulators seeded alike, on one thread, and given the same
// calls agree exactly; AllProbabilities divides by the exact norm instead.
BOOST_AUTO_TEST_CASE(extended_stabilizer_probabilities_are_its_probability)
{
    const auto prepare = [](uint64_t seed) {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kExtendedStabilizer);
        // On several threads the norm is summed in parallel and agrees only to rounding: one thread, set before
        // Initialize.
        sim->SetMultithreading(false);
        sim->AllocateQubits(2);
        sim->Initialize();
        sim->SetSeed(seed);
        sim->ApplyH(0);
        sim->ApplyT(0);
        sim->ApplyH(0);
        sim->ApplyCX(0, 1);
        return sim;
    };
    for (uint64_t seed : {1, 6, 11})
    {
        auto batch = prepare(seed), single = prepare(seed);
        const auto values = batch->Probabilities({0, 3, 1});
        BOOST_REQUIRE_EQUAL(values.size(), 3u);
        BOOST_TEST(values[0] == single->Probability(0), "seed " << seed);
        BOOST_TEST(values[1] == single->Probability(3), "seed " << seed);
        BOOST_TEST(values[2] == single->Probability(1), "seed " << seed);
    }
}

// Qubit 63 on Aer's matrix product state. Aer maps an index between its internal qubit order and
// the user's; it held the bit of a qubit as a signed integer, so the bit of qubit 63 was negative,
// its right shift spread the sign, and a basis state with qubit 63 set read as probability 0
// (qiskit-aer 23faa5b, the previous pin).
BOOST_AUTO_TEST_CASE(matrix_product_state_reads_basis_states_with_qubit_63)
{
    auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
    sim->AllocateQubits(64);
    sim->Initialize();
    sim->ApplyX(0);
    sim->ApplyCX(0, 63); // reorders the chain to bring qubit 63 next to qubit 0
    sim->ApplyCX(63, 31);
    const Types::qubit_t state = (Types::qubit_t{1} << 63) | (Types::qubit_t{1} << 31) | 1;
    BOOST_TEST(std::abs(sim->Probability(state) - 1.) < 1e-12);
    BOOST_TEST(std::abs(sim->Probabilities({state})[0] - 1.) < 1e-12);
    BOOST_TEST(std::abs(std::norm(sim->Amplitude(state)) - 1.) < 1e-12);
    BOOST_TEST(std::abs(sim->Probability(state ^ (Types::qubit_t{1} << 63))) < 1e-12);
    const auto marginal = sim->MarginalProbabilities({0, 63});
    BOOST_TEST(std::abs(marginal[3] - 1.) < 1e-12);
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
