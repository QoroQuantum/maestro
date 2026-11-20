/**
 * @file AerState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The aer state class.
 *
 * Should not be used directly, create an instance with the factory and use the
 * generic simulator interface.
 */

#pragma once

#ifndef _AER_STATE_H_
#define _AER_STATE_H_

#ifndef NO_QISKIT_AER

#ifdef INCLUDED_BY_FACTORY

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "../Interfaces/Simulator.h"
#include "QubitRegister.h"

#include "../Core/Configuration.h"
#include "QiskitAerState.h"

namespace Simulators
{
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private
{

class IndividualSimulator;

/**
 * @class AerState
 * @brief Class for the qiskit aer simulator state.
 *
 * Implements the qiskit aer state.
 * Do not use this class directly, use the factory to create an instance.
 * @sa ISimulator
 * @sa IState
 * @sa AerSimulator
 */
class AerState : public ISimulator
{
    friend class IndividualSimulator; /**< Allows the IndividualSimulator to
                                         access the private members of AerState */
  public:
    /**
     * @brief The constructor.
     *
     * The constructor for the qiskit aer simulator state.
     * Seeds the random number generator.
     */
    AerState()
    {
        rng.seed(Utils::RandomStream::FreshSeed());
        Configure("method", "statevector");
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it after the qubits allocation.
     * @sa AerState::AllocateQubits
     */
    void Initialize() override
    {
        pendingDeviation.clear();
        SetMultithreading(enableMultithreading);
        if (simulationType == SimulationType::kMatrixProductState && !configuration.IsSet("mps_sample_measure_algorithm"))
            Configure("mps_sample_measure_algorithm", "mps_probabilities");

        // Gate fusion must be off for MPS. Aer intends this itself -- AerState::transpile_ops()
        // has `case Method::matrix_product_state: fusion_pass_.active = false;` -- but the very
        // next statement is `fusion_pass_.set_config(configs_)`, and Fusion::set_config does an
        // unconditional `active = config.fusion_enable`, whose default is true. The MPS disable is
        // therefore dead code and fusion runs anyway (see qiskit-aer/src/transpile/fusion.hpp and
        // src/controllers/state_controller.hpp). Aer's other execution path, transpile_fusion() in
        // simulators/circuit_executor.hpp, guards against exactly this with an early return.
        // Fixed in our fork as well, but kept here so a build against an unpatched Aer behaves.
        // It is not a small effect: fusing gates into wider unitaries forces the MPS to bring more
        // qubits together and do bigger SVDs, and on a 16 qubit brickwork circuit at bond dimension
        // 128 it costs about 7x (1.5s -> 10.7s). Only set it when the caller has not, so an explicit
        // "fusion_enable" from a configuration still wins.
        if (simulationType == SimulationType::kMatrixProductState && !configuration.IsSet("fusion_enable"))
            Configure("fusion_enable", "false");

        state->initialize();
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This works only for the 'statevector' and 'density_matrix' methods: the
     * others throw std::runtime_error and keep their state.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        if (simulationType != SimulationType::kDensityMatrix)
            RequireStatevectorMethod(); // before anything is cleared: the state is as it was when this is refused
        RequireStatevectorLength(num_qubits, static_cast<size_t>(amplitudes.size()));
        Clear();
        if (simulationType == SimulationType::kDensityMatrix)
            InitializeDensityMatrixFromStatevector(num_qubits, amplitudes.data());
        else
            InitializeStatevectorBuffer(num_qubits, amplitudes.data());
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This works only for the 'statevector' and 'density_matrix' methods: the
     * others throw std::runtime_error and keep their state.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    /*
    void InitializeState(size_t num_qubits, std::vector<std::complex<double>,
    avoid_init_allocator<std::complex<double>>>& amplitudes) override
    {
            Clear();
            InitializeStatevectorBuffer(num_qubits, amplitudes.data());
    }
    */

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This works only for the 'statevector' and 'density_matrix' methods: the
     * others throw std::runtime_error and keep their state.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        if (simulationType != SimulationType::kDensityMatrix)
            RequireStatevectorMethod(); // before anything is cleared: the state is as it was when this is refused
        RequireStatevectorLength(num_qubits, static_cast<size_t>(amplitudes.size()));
        Clear();
        if (simulationType == SimulationType::kDensityMatrix)
            InitializeDensityMatrixFromStatevector(num_qubits, amplitudes.data());
        else
            InitializeStatevectorBuffer(num_qubits, amplitudes.data());
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This works only for the 'statevector' and 'density_matrix' methods: the
     * others throw std::runtime_error and keep their state.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (simulationType != SimulationType::kDensityMatrix)
            RequireStatevectorMethod(); // before anything is cleared: the state is as it was when this is refused
        RequireStatevectorLength(num_qubits, static_cast<size_t>(amplitudes.size()));
        Clear();
        if (simulationType == SimulationType::kDensityMatrix)
            InitializeDensityMatrixFromStatevector(num_qubits, amplitudes.data());
        else
            InitializeStatevectorBuffer(num_qubits, amplitudes.data());
    }

    /**
     * @brief Just resets the state to 0.
     *
     * Does not destroy the internal state, just resets it to zero (as a 'reset'
     * op on each qubit would do).
     */
    void Reset() override
    {
        const auto numQubits = GetNumberOfQubits();
        Clear();
        AllocateQubits(numQubits);
        state->initialize();
    }

    /**
     * @brief Configures the state.
     *
     * This function is called to configure the simulator.
     *
     * @param key The key of the configuration option.
     * @param value The value of the configuration.
     */
    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            throw std::invalid_argument("Null Aer configuration");
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = Configuration::ParsePrecision(key, value);
            const std::string precision = useDouble ? "double" : "single";
            // Fixed once initialized; a reapplied precision is ignored until Clear.
            if (state->is_initialized())
                return;
            state->configure("precision", precision);
            configuration.SetConfiguration("precision", precision);
            return;
        }
        if (std::string("seed") != key && configuration.WasApplied(key, value) && state->is_initialized())
            return;

        // Validate BEFORE storing below: Qiskit Aer's own MPS/MPO truncation always
        // implements the discarded-weight (Aer/iTensor) convention natively -- see
        // reduce_zeros() in qiskit-aer/src/simulators/matrix_product_state/svd.cpp.
        // Unlike the QCSim and GPU backends (see Simulators/QCSimState.h,
        // Simulators/GpuState.h), there is no relative-to-max mode to switch to
        // here, so requesting anything else is rejected outright. Checked here,
        // ahead of the store below, so a rejected value never lingers in
        // `configuration` -- otherwise it would still show up via
        // GetConfiguration(), or get silently replayed later by Clone()'s
        // generic configuration-replay loop, throwing again from an unrelated
        // call site.
        if ((std::string("matrix_product_state_truncation_mode") == key || std::string("matrix_product_operator_truncation_mode") == key) &&
            std::string("discarded_weight") != value)
            throw std::invalid_argument("Aer backend only supports the discarded_weight truncation mode");

        // Shared configurations can contain settings Aer does not support. Only
        // record settings after Aer accepts them, so rejected values are not
        // replayed by Clone() and cannot change the reported simulation type.
        if (std::string("seed") != key && std::string("use_double_precision") != key && std::string("matrix_product_state_truncation_mode") != key &&
            std::string("matrix_product_operator_truncation_mode") != key)
        {
            try
            {
                state->configure(key, value);
            }
            catch (const std::exception &)
            {
                return;
            }
        }

        if (!configuration.WasApplied(key, value))
            configuration.SetConfiguration(key, value);

        if (std::string("seed") == key)
        {
            const uint64_t seed = std::stoull(value);
            SeedAuxiliaryRng(seed);
            nextSeedStream = 0;
            rng.seed(seed);
            state->set_seed(seed);
            return;
        }

        if (std::string("method") == key)
        {
            if (std::string("statevector") == value)
                simulationType = SimulationType::kStatevector;
            else if (std::string("matrix_product_state") == value)
                simulationType = SimulationType::kMatrixProductState;
            else if (std::string("stabilizer") == value)
                simulationType = SimulationType::kStabilizer;
            else if (std::string("tensor_network") == value)
                simulationType = SimulationType::kTensorNetwork;
            else if (std::string("extended_stabilizer") == value)
                simulationType = SimulationType::kExtendedStabilizer;
            else if (std::string("density_matrix") == value)
                simulationType = SimulationType::kDensityMatrix;
            else
                simulationType = SimulationType::kOther;
        }
    }

    /**
     * @brief Returns configuration value.
     *
     * This function is called get a configuration value.
     * @param key The key of the configuration value.
     * @return The configuration value as a string.
     */
    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool fixedDouble = simulationType == SimulationType::kMatrixProductState;
            const bool useDouble = fixedDouble || !configuration.IsSet("precision") || configuration.GetConfiguration("precision") == "double";
            return std::string(key) == "precision" ? (useDouble ? "double" : "single") : (useDouble ? "1" : "0");
        }
        if (std::string("method") == key)
        {
            switch (simulationType)
            {
            case SimulationType::kStatevector:
                return "statevector";
            case SimulationType::kMatrixProductState:
                return "matrix_product_state";
            case SimulationType::kStabilizer:
                return "stabilizer";
            case SimulationType::kTensorNetwork:
                return "tensor_network";
            case SimulationType::kExtendedStabilizer:
                return "extended_stabilizer";
            case SimulationType::kDensityMatrix:
                return "density_matrix";
            default:
                return "other";
            }
        }

        return configuration.GetConfiguration(key);
    }

    /**
     * @brief Allocates qubits.
     *
     * This function is called to allocate qubits.
     * @param num_qubits The number of qubits to allocate.
     * @return The index of the first qubit allocated.
     */
    size_t AllocateQubits(size_t num_qubits) override
    {
        const auto ids = state->allocate_qubits(num_qubits);
        // No qubits allocates none: return the index the next qubit would get, as QCSim does (ids[0] read past an
        // empty list).
        return ids.empty() ? static_cast<size_t>(state->num_of_qubits()) : static_cast<size_t>(ids[0]);
    }

    /**
     * @brief Returns the number of qubits.
     *
     * This function is called to obtain the number of the allocated qubits.
     * @return The number of qubits.
     */
    size_t GetNumberOfQubits() const override
    {
        if (state->is_initialized())
            return state->num_of_qubits();

        return 0;
    }

    /**
     * @brief Clears the state.
     *
     * Sets the number of allocated qubits to 0 and clears the state.
     * After this qubits allocation is required then calling
     * IState::AllocateQubits in order to use the simulator.
     */
    void Clear() override
    {
        pendingDeviation.clear();
        state->clear();
        SetMultithreading(enableMultithreading);

        if (simulationType == SimulationType::kMatrixProductState && !configuration.IsSet("mps_sample_measure_algorithm"))
            Configure("mps_sample_measure_algorithm", "mps_probabilities");
    }

    /**
     * @brief Performs a measurement on the specified qubits.
     *
     * Don't use it if the number of qubits is larger than the number of bits in
     * the size_t type (usually 64), as the outcome will be undefined
     *
     * @param qubits A vector with the qubits to be measured.
     * @return The outcome of the measurements, the first qubit result is the
     * least significant bit.
     */
    size_t Measure(const Types::qubits_vector &qubits) override
    {
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;

        const size_t res = state->apply_measure(qubits);

        NotifyObservers(qubits);

        return res;
    }

    /**
     * @brief Performs a measurement on the specified qubits.
     *
     * @param qubits A vector with the qubits to be measured.
     * @return The outcome of the measurements
     */
    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        auto res = state->apply_measure_many(qubits);

        NotifyObservers(qubits);

        return res;
    }

    /**
     * @brief Performs a reset of the specified qubits.
     *
     * Measures the qubits and for those that are 1, applies X on them
     * @param qubits A vector with the qubits to be reset.
     */
    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        state->apply_reset(qubits);

        NotifyObservers(qubits);
    }

    bool SupportsQuantumChannels() const override
    {
        return simulationType == SimulationType::kDensityMatrix;
    }

    // Aer's statevector and matrix product state take a Kraus set as one
    // sampled operator: each is tried with the probability the state gives
    // it, and the one drawn is applied and the state renormalized.
    bool SamplesQuantumChannels() const override
    {
        return simulationType == SimulationType::kStatevector || simulationType == SimulationType::kMatrixProductState;
    }

    /**
     * @brief Applies a one- or two-qubit CPTP channel to an Aer state.
     *
     * On the density-matrix backend the Kraus operators are fused into a
     * superoperator and applied deterministically. The statevector and the
     * matrix product state apply one sampled branch, a trajectory of the
     * channel (SamplesQuantumChannels): the Pauli and unitary-mixture parts
     * of the channel are drawn from Maestro's channel stream and applied as
     * gates, and the parts that depend on the state go to Aer's apply_kraus,
     * which draws from the state's own seeded stream. The other Aer methods
     * have no Kraus instruction and reject channels.
     */
    void ApplyQuantumChannel(const Types::qubits_vector &targets, const QuantumChannel &channel) override
    {
        const bool sampled = SamplesQuantumChannels();
        if (!sampled && !SupportsQuantumChannels())
            throw std::runtime_error("Aer quantum channels require density_matrix, statevector or matrix_product_state simulation");
        if (targets.size() != channel.GetNumberOfQubits())
            throw std::invalid_argument("The number of channel targets does not match its Kraus operators");
        if (targets.empty() || targets.size() > 2)
            throw std::invalid_argument("Maestro supports only one- and two-qubit local channels");

        std::unordered_set<Types::qubit_t> uniqueTargets;
        for (const Types::qubit_t target : targets)
        {
            if (target >= GetNumberOfQubits())
                throw std::invalid_argument("Quantum-channel qubit is out of range");
            if (!uniqueTargets.insert(target).second)
                throw std::invalid_argument("Quantum-channel target qubits must be distinct");
        }

        const auto toAer = [](const QuantumChannel::KrausOperators &operators) {
            std::vector<AER::cmatrix_t> aerOperators;
            aerOperators.reserve(operators.size());
            for (const Eigen::MatrixXcd &krausOperator : operators)
            {
                AER::cmatrix_t aerOperator(static_cast<size_t>(krausOperator.rows()), static_cast<size_t>(krausOperator.cols()));
                for (Eigen::Index row = 0; row < krausOperator.rows(); ++row)
                    for (Eigen::Index column = 0; column < krausOperator.cols(); ++column)
                        aerOperator(static_cast<size_t>(row), static_cast<size_t>(column)) = krausOperator(row, column);
                aerOperators.emplace_back(std::move(aerOperator));
            }
            return aerOperators;
        };

        // Both QCSim and Aer treat targets[0] as local matrix bit zero (LSB), so
        // channel target order is preserved and matrices need no permutation.
        if (!sampled)
        {
            state->apply_kraus(AER::reg_t(targets.begin(), targets.end()), toAer(channel.GetKrausOperators()));
            NotifyObservers(targets);
            return;
        }
        const auto &factors = channel.GetFactors();
        // The matrix product state renormalizes the chain after every Kraus
        // operation, and how a channel is split into branches changes what its
        // trajectories cost there: drawn as Pauli gates followed by each
        // qubit's damping, QFT 16 took 1.7x Aer's time per shot, with 22% more
        // SVDs on larger bonds. Aer applies a gate's whole error as one Kraus
        // operation; so does this, with the channel's canonical Kraus set
        // (largest branch first), when any part of it would read the state
        // through Kraus operators anyway. A channel drawn without them (Pauli,
        // unitary mixture, T2 <= T1 relaxation as nothing, a phase flip or a
        // rare reset) keeps its cheaper draws below.
        //
        // Only the target the channel acts on goes to Aer when it leaves the
        // other alone. To apply an operation on two qubits, the matrix product
        // state first moves them next to each other, and under a bond limit
        // that move truncates the state: a channel given on two targets that
        // damps only one of them changed the other, and the qubits entangled
        // with it. The reduced channel is the same operation (its Kraus
        // operators are the channel's without the identity on the other target,
        // in the same order), so the trajectory does not depend on how the
        // channel was given.
        if (simulationType == SimulationType::kMatrixProductState &&
            std::any_of(factors.begin(), factors.end(), [](const QuantumChannel::Factor &factor) { return AppliesAsKraus(factor); }))
        {
            const QuantumChannel reduced = channel.OnActiveQubits();
            if (reduced.GetNumberOfQubits() == targets.size())
                state->apply_kraus(AER::reg_t(targets.begin(), targets.end()), toAer(channel.GetKrausOperators()));
            else
                state->apply_kraus(AER::reg_t{targets[channel.HomeQubit()]}, toAer(reduced.GetKrausOperators()));
            NotifyObservers(targets);
            return;
        }
        for (size_t index = 0; index < factors.size(); ++index)
        {
            const auto &factor = factors[index];
            Types::qubits_vector qubits;
            for (const size_t local : factor.qubits)
                qubits.push_back(targets[local]);
            if (factor.kind == QuantumChannel::FactorKind::kPauli)
            {
                const size_t index = DrawChannelBranch(factor.probabilities, ChannelUniform());
                for (size_t k = 0; k < qubits.size(); ++k)
                    switch ((index >> (2 * k)) & 3)
                    {
                    case 1:
                        ApplyX(qubits[k]);
                        break;
                    case 2:
                        ApplyY(qubits[k]);
                        break;
                    case 3:
                        ApplyZ(qubits[k]);
                        break;
                    default:
                        break;
                    }
            }
            else if (factor.kind == QuantumChannel::FactorKind::kUnitaryMixture)
            {
                // The branch is drawn whatever it is, so the draws that follow do not
                // depend on it. A drawn unitary that is the identity on a target is
                // not applied there: a two-qubit gate makes the matrix product state
                // bring its qubits together, which truncates under a bond limit, also
                // for the identity.
                const auto &unitary = factor.operators[DrawChannelBranch(factor.probabilities, ChannelUniform())];
                if (qubits.size() == 1)
                {
                    // The literal identity is not applied; a phase is kept, as the amplitudes show it.
                    if (!QuantumChannel::IsMultipleOfIdentity(unitary) || unitary(0, 0) != 1.)
                        ApplyGenericOneQubitGate(qubits[0], Eigen::Matrix2cd(unitary));
                }
                else
                {
                    Eigen::Matrix2cd local;
                    switch (QuantumChannel::LocalPartOfTwoQubitUnitary(unitary, local))
                    {
                    case QuantumChannel::LocalPart::kNone:
                        // A phase other than one is applied, on one qubit: it shows in the amplitudes and needs no routing.
                        if (unitary(0, 0) != 1.)
                            ApplyGenericOneQubitGate(qubits[0], Eigen::Matrix2cd(unitary(0, 0) * Eigen::Matrix2cd::Identity()));
                        break;
                    case QuantumChannel::LocalPart::kFirst:
                        ApplyGenericOneQubitGate(qubits[0], local);
                        break;
                    case QuantumChannel::LocalPart::kSecond:
                        ApplyGenericOneQubitGate(qubits[1], local);
                        break;
                    case QuantumChannel::LocalPart::kBoth:
                        ApplyGenericTwoQubitGate(qubits[0], qubits[1], Eigen::Matrix4cd(unitary));
                        break;
                    }
                }
            }
            else
            {
                QuantumChannel::ThermalMixture mixture;
                if (factor.kind != QuantumChannel::FactorKind::kThermal || !QuantumChannel::MixtureOf(factor.thermal, mixture))
                {
                    // Aer's statevector renormalizes after every Kraus operation,
                    // with a pass over the state for each branch it weighs (the
                    // matrix product state took its whole channel above). One-qubit
                    // factors on two qubits (the relaxation of each qubit of a
                    // two-qubit gate) commute, so they go to Aer as their tensor
                    // product: one renormalization instead of two, as Aer's own
                    // noise sampling applies a gate's error as one Kraus operation.
                    if (index + 1 < factors.size() && MergesIntoTwoQubitKraus(factor, factors[index + 1]))
                    {
                        const auto &next = factors[index + 1];
                        state->apply_kraus(AER::reg_t{qubits[0], targets[next.qubits[0]]}, toAer(TensorKraus(factor.operators, next.operators)));
                        ++index;
                        continue;
                    }
                    state->apply_kraus(AER::reg_t(qubits.begin(), qubits.end()), toAer(factor.operators));
                    continue;
                }
                // A relaxation with T2 <= T1: nothing, a phase flip, or a reset, which alone reads the state.
                const double random = ChannelUniform();
                if (random >= mixture.identity + mixture.phaseFlip)
                    state->apply_kraus(AER::reg_t(qubits.begin(), qubits.end()), toAer(QuantumChannel::ResetOperators(factor.thermal.excitedStatePopulation)));
                else if (random >= mixture.identity)
                    ApplyZ(qubits[0]);
            }
        }
        NotifyObservers(targets);
    }

    // Whether a factor draws its branch from the state through Aer's Kraus
    // operation: neither a Pauli channel nor a mixture of unitaries, nor a
    // relaxation with T2 <= T1, which is drawn as nothing, a phase flip or a reset.
    static bool AppliesAsKraus(const QuantumChannel::Factor &factor)
    {
        if (factor.IsStateIndependent())
            return false;
        QuantumChannel::ThermalMixture mixture;
        return factor.kind != QuantumChannel::FactorKind::kThermal || !QuantumChannel::MixtureOf(factor.thermal, mixture);
    }

    // Two one-qubit Kraus factors on different qubits, which commute, with at
    // most 16 products of their operators: the number a two-qubit channel needs
    // at most, and what two relaxations give (four operators each). Factors
    // given with more operators (a channel may come with any number) would
    // multiply into as many matrices, and into as many branches for Aer to
    // weigh, as the product of their counts; they are applied one by one.
    static bool MergesIntoTwoQubitKraus(const QuantumChannel::Factor &first, const QuantumChannel::Factor &second)
    {
        return first.qubits.size() == 1 && second.qubits.size() == 1 && first.qubits[0] != second.qubits[0] && AppliesAsKraus(first) &&
               AppliesAsKraus(second) && first.operators.size() * second.operators.size() <= kMaxMergedKrausOperators;
    }

    static constexpr size_t kMaxMergedKrausOperators = 16;

    // The Kraus operators of two one-qubit channels side by side, the first on
    // the local bit zero (LSB). The first operators lead, so the likeliest
    // branch, both qubits' leading operator, is weighed first.
    static QuantumChannel::KrausOperators TensorKraus(const QuantumChannel::KrausOperators &low, const QuantumChannel::KrausOperators &high)
    {
        QuantumChannel::KrausOperators product;
        product.reserve(low.size() * high.size());
        for (const auto &a : low)
            for (const auto &b : high)
            {
                QuantumChannel::Matrix op(4, 4);
                for (int row = 0; row < 4; ++row)
                    for (int column = 0; column < 4; ++column)
                        op(row, column) = a(row & 1, column & 1) * b(row >> 1, column >> 1);
                product.emplace_back(std::move(op));
            }
        return product;
    }

    static size_t DrawChannelBranch(const std::vector<double> &probabilities, double random)
    {
        size_t chosen = 0;
        double accumulated = 0.;
        for (size_t index = 0; index < probabilities.size(); ++index)
        {
            if (probabilities[index] <= 0.)
                continue;
            chosen = index; // also the fallback if rounding leaves the draw above the last sum
            accumulated += probabilities[index];
            if (random < accumulated)
                break;
        }
        return chosen;
    }

    /**
     * @brief Returns the probability of the specified outcome.
     *
     * Use it to obtain the probability to obtain the specified outcome, if all
     * qubits are measured.
     * @sa AerState::Amplitude
     * @sa AerState::Probabilities
     *
     * @param outcome The outcome to obtain the probability for.
     * @return The probability of the specified outcome.
     */
    double Probability(Types::qubit_t outcome) override
    {
        RequireBasisState(outcome);
        return state->probability(outcome);
    }

    /**
     * @brief Returns the amplitude of the specified state.
     *
     * Use it to obtain the amplitude of the specified state.
     * @sa AerState::Probability
     * @sa AerState::Probabilities
     *
     * @param outcome The outcome to obtain the amplitude for.
     * @return The amplitude of the specified outcome.
     */
    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        // A method without amplitudes says so whatever the index, as every backend does.
        if (simulationType == SimulationType::kDensityMatrix)
            throw std::runtime_error("AerState::Amplitude is not defined for density matrix simulation");
        if (simulationType == SimulationType::kStabilizer)
            throw std::runtime_error("AerState::Amplitude is not defined for stabilizer simulation");
        RequireBasisState(outcome);

        if (simulationType == SimulationType::kExtendedStabilizer)
        {
            // Aer has no single amplitude there, only the statevector of the decomposition. Its size is checked first:
            // Aer sizes it in bytes, which wraps from 2^60 amplitudes (60 to 63 qubits initialize), and it wrote past
            // the allocation.
            ListableBasisStateCount<std::complex<double>>(GetNumberOfQubits());
            const auto amplitudes = state->statevector();
            return outcome < amplitudes.size() ? amplitudes[outcome] : complex_t{};
        }

        return state->amplitude(outcome);
    }

    /**
     * @brief Projects the state onto the zero state.
     *
     * Use it to project the state onto the zero state.
     * For most simulator is the same as calling Amplitude(0), but for some
     * simulators it can be optimized to be faster than calling Amplitude(0).
     * This for now is done for qcsim mps and gpu mps.
     *
     * @sa IState::Amplitude
     * @sa IState::Probability
     *
     * @return The inner product result as a complex number.
     */
    std::complex<double> ProjectOnZero() override
    {
        return Amplitude(0);
    }

    /**
     * @brief Returns the probabilities of all possible outcomes.
     *
     * Use it to obtain the probabilities of all possible outcomes.
     * @sa AerState::Probability
     * @sa AerState::Amplitude
     * @sa AerState::AllProbabilities
     *
     * @return A vector with the probabilities of all possible outcomes.
     */
    std::vector<double> AllProbabilities() override
    {
        RequireQubits(GetNumberOfQubits()); // a register without qubits has no state to list
        // Before Aer sizes its output: its stabilizer takes 64 qubits and sizes it with 1 << 64, which is undefined
        // (one value came back, which passed for the distribution).
        ListableBasisStateCount(GetNumberOfQubits());
        if (simulationType == SimulationType::kDensityMatrix)
        {
            Types::qubits_vector qubits(GetNumberOfQubits());
            std::iota(qubits.begin(), qubits.end(), 0);
            return state->probabilities(qubits);
        }

        if (simulationType == SimulationType::kExtendedStabilizer)
        {
            ListableBasisStateCount<std::complex<double>>(GetNumberOfQubits()); // the statevector, as in Amplitude
            const auto amplitudes = state->statevector();
            std::vector<double> probabilities(amplitudes.size());
            for (size_t outcome = 0; outcome < amplitudes.size(); ++outcome)
                probabilities[outcome] = std::norm(amplitudes[outcome]);
            return probabilities;
        }

        return state->probabilities();
    }

    /**
     * @brief Returns the probabilities of the specified outcomes.
     *
     * Use it to obtain the probabilities of the specified outcomes.
     * @sa AerState::Probability
     * @sa AerState::Amplitude
     *
     * @param qubits A vector with the qubits configuration outcomes.
     * @return A vector with the probabilities for the specified qubit
     * configurations.
     */
    std::vector<double> Probabilities(const Types::qubits_vector &outcomes) override
    {
        // One value per basis-state index, each what Probability returns. (Aer's own probabilities(qubits) is the
        // marginal distribution of qubits, MarginalProbabilities; this called it on the statevector, MPS and
        // stabilizer, which read the outcomes as qubits.) On the extended stabilizer Probability is Aer's estimate:
        // the exact amplitude of the decomposition over a sampled estimate of its norm, the same on every call for
        // a given state; AllProbabilities divides by the exact norm, so the two differ by the error of the norm.
        RequireQubits(GetNumberOfQubits()); // also for no outcomes
        for (const Types::qubit_t outcome : outcomes)
            RequireBasisState(outcome);
        return IState::Probabilities(outcomes);
    }

    /**
     * @brief The total probability of the state, from Aer's own marginal of qubit 0, at any width.
     *
     * The stabilizers take no generic gate (Aer rejects the matrix), so their
     * state stays normalized and its total is 1; it also could not be summed
     * from the extended stabilizer's statevector past a few dozen qubits. The
     * matrix product state stays normalized too: a generic gate whose matrix
     * moves its norm is applied so that it is restored (ApplyGenericToChain).
     */
    double TotalProbability() override
    {
        RequireQubits(GetNumberOfQubits());
        if (simulationType == SimulationType::kStabilizer || simulationType == SimulationType::kExtendedStabilizer ||
            simulationType == SimulationType::kMatrixProductState)
            return 1.;
        const std::vector<double> first = state->probabilities(AER::reg_t{0});
        return first.at(0) + first.at(1);
    }

    /**
     * @brief Returns the probability distribution of some of the qubits.
     *
     * Aer computes it directly (save_probs on the qubits) on the statevector,
     * the density matrix, the matrix product state and the stabilizer; the
     * extended stabilizer sums its statevector. The first listed qubit is the
     * least significant bit, as Aer orders it.
     * @sa IState::MarginalProbabilities
     */
    std::vector<double> MarginalProbabilities(const Types::qubits_vector &qubits) override
    {
        RequireMarginalQubits(qubits);
        // No qubits: the total probability (TotalProbability above, through IState::MarginalProbabilities). Aer reads
        // the qubits of a matrix product state from its canonical form, at any width: ApplyGenericToChain keeps it valid
        // after a generic gate whose matrix moves the norm (a qubit of a projected Bell pair read [0.5, 0.5]).
        if (qubits.empty() || simulationType == SimulationType::kExtendedStabilizer)
            return IState::MarginalProbabilities(qubits);
        return state->probabilities(AER::reg_t(qubits.begin(), qubits.end()));
    }

    /**
     * @brief Returns the counts of the outcomes of measurement of the specified
     * qubits, for repeated measurements.
     *
     * Use it to obtain the counts of the outcomes of the specified qubits
     * measurements. The state is not collapsed, so the measurement can be
     * repeated 'shots' times.
     *
     * Don't use it if the number of qubits is larger than the number of bits in
     * the Types::qubit_t type (usually 64), as the outcome will be undefined.
     *
     * @param qubits A vector with the qubits to be measured.
     * @param shots The number of shots to perform.
     * @return A map with the counts for the outcomes of measurements of the
     * specified qubits.
     */
    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};

        if (qubits.size() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;

        // Aer MPS can return native samples sorted by qubit index. Use the
        // ordered sampling adapter for both public result formats.
        std::unordered_map<Types::qubit_t, Types::qubit_t> res;
        for (const auto &[bits, count] : state->sample_counts_many(qubits, shots))
        {
            Types::qubit_t packed = 0;
            const size_t width = std::min(bits.size(), sizeof(Types::qubit_t) * 8);
            for (size_t i = 0; i < width; ++i)
                if (bits[i])
                    packed |= Types::qubit_t(1) << i;
            res[packed] += count;
        }

        NotifyObservers(qubits);

        return res;
    }

    /**
     * @brief Returns the counts of the outcomes of measurement of the specified
     * qubits, for repeated measurements.
     *
     * Use it to obtain the counts of the outcomes of the specified qubits
     * measurements. The state is not collapsed, so the measurement can be
     * repeated 'shots' times.
     *
     * @param qubits A vector with the qubits to be measured.
     * @param shots The number of shots to perform.
     * @return A map with the counts for the otcomes of measurements of the
     * specified qubits.
     */
    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> res = state->sample_counts_many(qubits, shots);
        NotifyObservers(qubits);
        return res;
    }

    /**
     * @brief Returns the expected value of a Pauli string.
     *
     * Use it to obtain the expected value of a Pauli string.
     * The Pauli string is a string of characters representing the Pauli
     * operators, e.g. "XIZY". The length of the string should be less or equal to
     * the number of qubits (if it's less, it's completed with I).
     *
     * @param pauliString The Pauli string to obtain the expected value for.
     * @return The expected value of the specified Pauli string.
     */
    double ExpectationValue(const std::string &pauliStringOrig) override
    {
        if (pauliStringOrig.empty())
            return 1.0;

        std::string pauliString = pauliStringOrig;
        if (pauliString.size() > GetNumberOfQubits())
        {
            for (size_t i = GetNumberOfQubits(); i < pauliString.size(); ++i)
            {
                const auto pauliOp = toupper(pauliString[i]);
                if (pauliOp != 'I' && pauliOp != 'Z')
                    return 0.0;
            }

            pauliString.resize(GetNumberOfQubits());
        }

        AER::reg_t qubits;
        std::string pauli;

        pauli.reserve(pauliString.size());
        qubits.reserve(pauliString.size());

        for (size_t q = 0; q < pauliString.size(); ++q)
        {
            const char p = toupper(pauliString[q]);
            if (p == 'I')
                continue;

            pauli.push_back(p);
            qubits.push_back(q);
        }

        if (qubits.empty())
            return 1.0;

        // qiskit aer expects the pauli string in reverse order
        std::reverse(pauli.begin(), pauli.end());

        return state->expval_pauli(qubits, pauli);
    }

    /**
     * @brief Returns the type of simulator.
     *
     * Returns the type of simulator.
     * @return The type of simulator.
     * @sa SimulatorType
     */
    SimulatorType GetType() const override
    {
        return SimulatorType::kQiskitAer;
    }

    /**
     * @brief Returns the type of simulation.
     *
     * Returns the type of simulation.
     *
     * @return The type of simulation.
     * @sa SimulationType
     */
    SimulationType GetSimulationType() const override
    {
        return simulationType;
    }

    /**
     * @brief Flushes the applied operations
     *
     * This function is called to flush the applied operations.
     * It is used to flush the operations that were applied to the state.
     * qcsim applies them right away, so this has no effect on it, but qiskit aer
     * does not.
     */
    void Flush() override
    {
        state->flush_ops();
        // state->set_random_seed(); // avoid reusing the old seed
    }

    /**
     * @brief Saves the state to internal storage.
     *
     * Saves the state to internal storage, if needed.
     * Calling this should consider as the simulator is gone to uninitialized.
     * Either do not use it except for getting amplitudes, or reinitialize the
     * simulator after calling it. This is needed only for the composite
     * simulator, for an optimization for qiskit aer. For qcsim it does nothing.
     */
    void SaveStateToInternalDestructive() override
    {
        if (simulationType == SimulationType::kDensityMatrix)
            savedDensityMatrix = state->move_to_matrix();
        else
            savedAmplitudes = state->move_to_vector();
    }

    /**
     * @brief Restores the state from the internally saved state
     *
     * Restores the state from the internally saved state, if needed.
     * This does something only for qiskit aer.
     */
    void RestoreInternalDestructiveSavedState() override
    {
        if (simulationType == SimulationType::kDensityMatrix)
        {
            const size_t numQubits = static_cast<size_t>(log2(savedDensityMatrix.GetRows()));
            InitializeDensityMatrixBuffer(numQubits, savedDensityMatrix.data());
            return;
        }

        const size_t numQubits = static_cast<size_t>(log2(savedAmplitudes.size()));
        InitializeStatevectorBuffer(numQubits, savedAmplitudes.data());
    }

    /**
     * @brief Saves the state to internal storage.
     *
     * Saves the state to internal storage, if needed.
     * Calling this will not destroy the internal state, unlike the 'Destructive'
     * variant. To be used in order to recover the state after doing measurements,
     * for multiple shots executions.
     */
    void SaveState() override
    {
        if (!state)
            return;
        savedPendingDeviation = pendingDeviation;
        normSaved = true;

        const auto numQubits = GetNumberOfQubits();

        if (simulationType == SimulationType::kExtendedStabilizer)
        {
            savedExtendedStabilizerState = state->clone_extended_stabilizer_state();
            return;
        }

        if (simulationType == SimulationType::kStatevector || simulationType == SimulationType::kDensityMatrix)
        {
            SaveStateToInternalDestructive();
            if (simulationType == SimulationType::kDensityMatrix)
                InitializeDensityMatrixBuffer(numQubits, savedDensityMatrix.data());
            else
                InitializeStatevectorBuffer(numQubits, savedAmplitudes.data());
            return;
        }

        bool saved = false;

        if (state->is_initialized())
        {
            AER::Operations::Op op;

            op.type = AER::Operations::OpType::save_state;
            op.name = "save_state";
            op.save_type = AER::Operations::DataSubType::single;
            op.string_params.push_back("s");

            for (size_t q = 0; q < numQubits; ++q)
                op.qubits.push_back(q);

            state->buffer_op(std::move(op));
            Flush();

            // get the state from the last result
            AER::ExperimentResult &last_result = state->last_result();
            // state should be in last_result.data
            if (last_result.status == AER::ExperimentResult::Status::completed)
            {
                savedState = std::move(last_result.data);
                saved = true;
            }
        }
        else
        {
            // try get the state from the last result
            AER::ExperimentResult &last_result_prev = state->last_result();
            // state should be in last_result.data
            if (last_result_prev.status == AER::ExperimentResult::Status::completed)
            {
                savedState = std::move(last_result_prev.data);
                saved = true;
            }
        }

        // this is a hack, for statevector and matrix product state if the last op
        // is executed, it can destroy the state! see also the workaround for
        // statevector for the stabilizer at least for now it doesn't seem to do
        // that
        // TODO: check everything!!!!
        if (saved && simulationType == SimulationType::kMatrixProductState)
            RestoreState();
    }

    /**
     * @brief Restores the state from the internally saved state
     *
     * Restores the state from the internally saved state, if needed.
     * To be used in order to recover the state after doing measurements, for
     * multiple shots executions. In the first phase, only qcsim will implement
     * this.
     */
    void RestoreState() override
    {
        // Restoring may reinitialize the state, which clears the mark: it is set from the saved one after.
        RestoreSavedStateOfMethod();
        if (normSaved) // a restore with nothing saved leaves the mark as it is
            pendingDeviation = savedPendingDeviation;
    }

    void RestoreSavedStateOfMethod()
    {
        auto numQubits = GetNumberOfQubits();

        AER::Operations::Op op;

        switch (simulationType)
        {
        case SimulationType::kStatevector: {
            // op.type = AER::Operations::OpType::set_statevec;
            // op.name = "set_statevec";

            // const auto& vec = static_cast<AER::DataMap<AER::SingleData,
            // AER::Vector<complex_t>>>(savedState).value()["s"].value();

            // this is a hack until I figure it out
            Clear();
            numQubits = static_cast<size_t>(log2(savedAmplitudes.size()));
            InitializeStatevectorBuffer(numQubits, savedAmplitudes.data());

            return;
        }
        break;
        case SimulationType::kDensityMatrix: {
            Clear();
            numQubits = static_cast<size_t>(log2(savedDensityMatrix.GetRows()));
            InitializeDensityMatrixBuffer(numQubits, savedDensityMatrix.data());

            return;
        }
        break;
        case SimulationType::kExtendedStabilizer:
            if (!savedExtendedStabilizerState)
                throw std::runtime_error("AerState::RestoreState: no extended stabilizer state was saved");
            state->restore_extended_stabilizer_state(savedExtendedStabilizerState);
            return;
        case SimulationType::kMatrixProductState:
            op.type = AER::Operations::OpType::set_mps;
            op.name = "set_mps";
            op.mps = static_cast<AER::DataMap<AER::SingleData, AER::mps_container_t>>(savedState).value()["s"].value();

            /*
            {
                    auto& value = static_cast<AER::DataMap<AER::SingleData,
            AER::mps_container_t>>(savedState).value(); if (!value.empty())
                    {
                            if (value.find("s") != value.end())
                                    op.mps = value["s"].value();
                            else if (value.find("matrix_product_state") !=
            value.end()) op.mps = value["matrix_product_state"].value();
                    }
            }
            */

            numQubits = op.mps.first.size();
            break;
        case SimulationType::kStabilizer:
            op.type = AER::Operations::OpType::set_stabilizer;
            op.name = "set_stabilizer";
            op.clifford = static_cast<AER::DataMap<AER::SingleData, json_t>>(savedState).value()["s"].value();

            /*
            {
                    auto& value = static_cast<AER::DataMap<AER::SingleData,
            json_t>>(savedState).value(); if (!value.empty())
                    {
                            if (value.find("s") != value.end())
                                    op.clifford = value["s"].value();
                            else if (value.find("stabilizer") != value.end())
                                    op.clifford = value["stabilizer"].value();
                    }
            }
            */

            numQubits = op.clifford.num_qubits();
            break;
        case SimulationType::kTensorNetwork:
        default:
            throw std::runtime_error("AerState::RestoreState: not implemented yet "
                                     "for this type of simulator.");
        }

        op.save_type = AER::Operations::DataSubType::single;
        op.string_params.push_back("s");

        for (size_t q = 0; q < numQubits; ++q)
            op.qubits.push_back(q);

        // WHY?
        if (!state->is_initialized())
        {
            Clear();
            AllocateQubits(numQubits);
            state->initialize();
        }

        state->buffer_op(std::move(op));
        Flush();
    }

    /**
     * @brief Gets the amplitude.
     *
     * Gets the amplitude, from the internal storage if needed.
     * This is needed only for the composite simulator, for an optimization for
     * qiskit aer. For qcsim it does the same thing as Amplitude.
     */
    std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override
    {
        if (simulationType == SimulationType::kDensityMatrix)
            throw std::runtime_error("AerState::AmplitudeRaw is not defined for density matrix "
                                     "simulation");

        // Read from the destructively saved amplitudes: the state itself, and so its qubit count, is gone.
        if (outcome >= savedAmplitudes.size())
            throw std::out_of_range("AerState::AmplitudeRaw: basis state " + std::to_string(outcome) + " is not in the saved state");
        return savedAmplitudes[outcome];
    }

    /**
     * @brief Enable/disable multithreading.
     *
     * Enable/disable multithreading. Default is enabled.
     *
     * @param multithreading A flag to indicate if multithreading should be
     * enabled.
     */
    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (state && !state->is_initialized())
        {
            const std::string nrThreads = std::to_string(enableMultithreading ? 0 : 1); // 0 means auto/all available, 1 limits to 1
            state->configure("max_parallel_threads", nrThreads);
            state->configure("parallel_state_update", nrThreads);
            const std::string threadsLimit = std::to_string(12); // set one less, multithreading is started if the
                                                                 // value is bigger than this
            state->configure("statevector_parallel_threshold", threadsLimit);

            configuration.SetConfiguration(std::string("max_parallel_threads"), nrThreads);
            configuration.SetConfiguration(std::string("parallel_state_update"), nrThreads);
            configuration.SetConfiguration(std::string("statevector_parallel_threshold"), threadsLimit);
        }
    }

    /**
     * @brief Get the multithreading flag.
     *
     * Returns the multithreading flag.
     *
     * @return The multithreading flag.
     */
    bool GetMultithreading() const override
    {
        return enableMultithreading;
    }

    /**
     * @brief Returns if the simulator is a qcsim simulator.
     *
     * Returns if the simulator is a qcsim simulator.
     * This is just a helper function to ease things up: qcsim has different
     * functionality exposed sometimes so it's good to know if we deal with qcsim
     * or with qiskit aer.
     *
     * @return True if the simulator is a qcsim simulator, false otherwise.
     */
    bool IsQcsim() const override
    {
        return false;
    }

    /**
     * @brief Measures all the qubits without collapsing the state.
     *
     * Measures all the qubits without collapsing the state, allowing to perform
     * multiple shots. This is to be used only internally, only for the
     * statevector simulators (or those based on them, as the composite ones). For
     * the qiskit aer case, SaveStateToInternalDestructive is needed to be called
     * before this. If one wants to use the simulator after such measurement(s),
     * RestoreInternalDestructiveSavedState should be called at the end.
     *
     * Don't use this for more qubits than the size of Types::qubit_t, as the
     * result is packed in a limited number of bits (e.g. 64 bits for uint64_t)
     *
     * @return The result of the measurements, the first qubit result is the least
     * significant bit.
     */
    Types::qubit_t MeasureNoCollapse() override
    {
        if (simulationType == SimulationType::kStatevector)
        {
            const double prob = 1. - uniformZeroOne(rng); // this excludes 0 as probabiliy
            double accum = 0;
            Types::qubit_t state = 0;
            for (Types::qubit_t i = 0; i < savedAmplitudes.size(); ++i)
            {
                accum += std::norm(savedAmplitudes[i]);
                if (prob <= accum)
                {
                    state = i;
                    break;
                }
            }

            return state;
        }

        throw std::runtime_error("AerState::MeasureNoCollapse: Invalid simulation type for measuring "
                                 "all the qubits without collapsing the state.");

        return 0;
    }

    /**
     * @brief Measures all the qubits without collapsing the state.
     *
     * Measures all the qubits without collapsing the state, allowing to perform
     * multiple shots. This is to be used only internally, only for the
     * statevector simulators (or those based on them, as the composite ones). For
     * the qiskit aer case, SaveStateToInternalDestructive is needed to be called
     * before this. If one wants to use the simulator after such measurement(s),
     * RestoreInternalDestructiveSavedState should be called at the end.
     *
     * Use this for more qubits than the size of Types::qubit_t
     *
     * @return The result of the measurements
     */
    std::vector<bool> MeasureNoCollapseMany() override
    {
        if (simulationType == SimulationType::kStatevector)
        {
            const size_t numQubits = static_cast<size_t>(log2(savedAmplitudes.size()));
            std::vector<bool> res(numQubits, false);

            Types::qubit_t state = MeasureNoCollapse();

            for (size_t i = 0; i < numQubits; ++i)
            {
                if ((state & 1) == 1)
                    res[i] = true;
                state >>= 1;
            }

            return res;
        }
        throw std::runtime_error("AerState::MeasureNoCollapseMany: Invalid simulation type for "
                                 "measuring "
                                 "all the qubits without collapsing the state.");

        return {};
    }

    const Configuration &GetConfiguration() const
    {
        return configuration;
    }

    const std::unordered_map<std::string, std::string> &GetConfigMap() const override
    {
        return configuration.GetConfigMap();
    }

  protected:
    void InitializeStatevectorBuffer(size_t numQubits, std::complex<double> *amplitudes)
    {
        RequireStatevectorMethod();
        const size_t dimension = ListableBasisStateCount<std::complex<double>>(numQubits);
        if (GetConfiguration("precision") == "double")
        {
            state->initialize_statevector(numQubits, amplitudes, true);
            return;
        }
        AllocateQubits(numQubits);
        Initialize();
        AER::Operations::Op op;
        op.type = AER::Operations::OpType::set_statevec;
        op.name = "set_statevector";
        op.params.assign(amplitudes, amplitudes + dimension);
        for (size_t q = 0; q < numQubits; ++q)
            op.qubits.push_back(q);
        state->buffer_op(std::move(op));
        Flush();
    }

    /**
     * A list of amplitudes is a state of the statevector method only (the
     * density matrix takes one through its own loader). Aer's initializer
     * builds a dense statevector whatever method the simulator was made for:
     * a matrix product state or a stabilizer given amplitudes went on as a
     * dense state under its own name, without its bond limit or its scaling.
     * They are refused, as QCSim's and the GPU's are.
     */
    void RequireStatevectorMethod() const
    {
        if (simulationType != SimulationType::kStatevector)
            throw std::runtime_error("AerState::InitializeState: Invalid simulation type for initializing the state.");
    }

    template <class Derived> static AER::cmatrix_t ToAerMatrix(const Eigen::MatrixBase<Derived> &matrix)
    {
        AER::cmatrix_t result(static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols()));
        for (Eigen::Index row = 0; row < matrix.rows(); ++row)
            for (Eigen::Index column = 0; column < matrix.cols(); ++column)
                result(static_cast<size_t>(row), static_cast<size_t>(column)) = matrix(row, column);
        return result;
    }

    /**
     * A generic gate on the matrix product state, which stays normalized and
     * in canonical form (see IState::NormDeviation).
     *
     * A matrix that moves the norm by more than the tolerance is a filter:
     * ApplyFilterToChain. Smaller deviations add up, since many matrices each
     * close to unitary move the state by their sum, on whatever qubits they
     * act: (1 - 3e-11) I once on each of 1024 qubits left 0.99999994 of the
     * state. They add up over the whole chain, and once they pass the
     * tolerance the tensors are brought up to date from every qubit a matrix
     * has met since the last time (RenormalizeFromSite), whatever weight it
     * has. A matrix can shift the probabilities of its qubit and leave the
     * weight where it was, and the other qubits were then read from Schmidt
     * values it had left stale. Until then the count costs an addition and a
     * mark for each gate.
     */
    template <class Matrix> void ApplyGenericToChain(const AER::reg_t &qubits, const Matrix &gate)
    {
        RequireGateTargets(qubits);
        const double deviation = NormDeviation(gate);
        if (deviation > kNormDeviationTolerance)
            return ApplyFilterToChain(qubits, gate);
        state->apply_unitary(qubits, ToAerMatrix(gate));
        if (deviation == 0.)
            return;
        if (pendingDeviation.touched.size() != GetNumberOfQubits())
            pendingDeviation.touched.assign(GetNumberOfQubits(), 0);
        for (const auto qubit : qubits)
            pendingDeviation.touched[qubit] = 1;
        if ((pendingDeviation.total += deviation) <= kNormDeviationTolerance)
            return;
        // Every qubit a matrix has met since the last time: the scaled identity is exact where only that qubit has
        // drifted, and what the others have drifted by is below the tolerance.
        const std::vector<char> touched = std::move(pendingDeviation.touched);
        pendingDeviation.clear();
        for (size_t qubit = 0; qubit < touched.size(); ++qubit)
            if (touched[qubit])
                RenormalizeFromSite(qubit);
    }

    /**
     * The targets of a generic gate are qubits of the register, each once.
     * Aer does not check them (it reads its tables at whatever index it is
     * given), and neither did the count kept here for each qubit of a chain.
     * The bindings check before they call; a C++ caller is checked here.
     */
    void RequireGateTargets(const AER::reg_t &qubits) const
    {
        const size_t count = GetNumberOfQubits();
        for (size_t k = 0; k < qubits.size(); ++k)
        {
            if (qubits[k] >= count)
                throw std::invalid_argument("Generic gate target is out of range");
            for (size_t j = 0; j < k; ++j)
                if (qubits[j] == qubits[k])
                    throw std::invalid_argument("Generic gate targets must be distinct");
        }
    }

    /**
     * Scales a site back to weight 1 and has Aer renormalize the tensors
     * outwards from it, as it does after a measurement: the scaled identity
     * goes to the chain as a channel of one operator, which Aer applies as it
     * is and follows with that sweep. (It draws once from Aer's stream, as
     * every channel does.)
     */
    void RenormalizeFromSite(Types::qubit_t qubit)
    {
        const std::vector<double> both = state->probabilities(AER::reg_t{qubit});
        const double weight = both.at(0) + both.at(1);
        if (!(weight > 0.) || !std::isfinite(weight))
            throw std::runtime_error("AerState: the matrix product state has no norm left to restore");
        state->apply_kraus(AER::reg_t{qubit}, std::vector<AER::cmatrix_t>{ToAerMatrix(Eigen::Matrix2cd(Eigen::Matrix2cd::Identity() / std::sqrt(weight)))});
    }

    /**
     * A one- or two-qubit matrix M that is not unitary, on the matrix product
     * state: applied as M / sqrt(w), with w = <psi|M^dagger M|psi> the weight
     * it leaves, as the one operator of a channel, so that Aer renormalizes
     * the tensors outside its qubits afterwards. A matrix that does not keep
     * the norm leaves those stale: Aer went on reading the other qubits and
     * sampling from them (a qubit of a projected Bell pair was sampled as 1
     * half the time), and its save, which sorts the qubits with swaps,
     * renormalized the live state.
     *
     * The matrix has to be scaled before Aer applies it: Aer cannot split a
     * tensor that holds nothing, and it cuts Schmidt values by their size as
     * they are, before it normalizes them. So the weight is read first, and
     * read without changing the chain: a refused matrix leaves the state as
     * it was, under a bond limit too.
     *
     * Where the columns of M are orthogonal, the weight is their squared
     * norms times the probabilities Aer has of the qubits, a sum of positive
     * terms that is as exact for a small weight as for a large one. For one
     * qubit otherwise, M = U S V^dagger and the qubit is turned by V^dagger,
     * a unitary on one site, which cuts nothing; the probabilities are read
     * there, weighed by the squared singular values, and U S / sqrt(w) is
     * what remains to apply (or V, to turn back).
     * For two qubits otherwise, the weight is the sum of the Pauli expectation
     * values of the pair that M^dagger M is made of (PauliWeight).
     *
     * A weight the chain cannot tell from none counts as none (NoWeightBelow,
     * and for the two-qubit sum the rounding of its terms, 64 epsilon of what
     * the matrix could leave): the matrix is refused with std::runtime_error.
     */
    template <class Matrix> void ApplyFilterToChain(const AER::reg_t &qubits, const Matrix &gate)
    {
        static constexpr char noState[] = "The matrix of the generic gate leaves no state: a matrix product state cannot hold a state of zero norm";
        // M = U S V^dagger. The singular values are taken from M itself: those of M^dagger M are their squares, and
        // one below 1e-8 of the largest is lost there in the rounding of the product, so that what came out for it
        // was noise of either sign (diag(1e-8, 1) H on |+> was scaled by a weight 27% off, and a projection onto a
        // state orthogonal to the chain's was weighed 2.7e-17 and applied).
        const Eigen::JacobiSVD<Matrix> decomposition(gate, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Index dimension = gate.rows();
        const double largest = decomposition.singularValues()(0) * decomposition.singularValues()(0);
        if (!(largest > 0.) || !std::isfinite(largest))
            throw std::runtime_error(noState); // the zero matrix, or one that is not finite

        // Columns that are orthogonal to rounding: M^dagger M is diagonal, the squared norms of the columns weigh
        // the probabilities Aer has, and nothing is turned.
        using Hermitian = std::decay_t<decltype((gate.adjoint() * gate).eval())>;
        const Hermitian squared = gate.adjoint() * gate;
        Hermitian offDiagonal = squared;
        offDiagonal.diagonal().setZero();
        const bool diagonal = offDiagonal.cwiseAbs().maxCoeff() <= std::numeric_limits<double>::epsilon() * largest;

        if (!diagonal && qubits.size() == 2)
        {
            // Two qubits are not turned to be weighed: that is an update of their bond, which a bond limit can cut
            // on the live chain, and turning back would not restore what was cut. The Pauli expectation values of
            // the pair give the weight without changing anything, to the rounding of the largest of them: enough
            // to scale the matrix, since Aer then normalizes the bond it splits itself, and to tell a weight from
            // none down to that rounding, 64 epsilon of what the matrix could leave.
            const double estimate = PauliWeight(squared, qubits);
            const double rounding = 64. * std::numeric_limits<double>::epsilon();
            if (!(estimate > std::max(rounding, NoWeightBelow()) * largest) || !std::isfinite(estimate))
                throw std::runtime_error(noState);
            state->apply_kraus(qubits, std::vector<AER::cmatrix_t>{ToAerMatrix(gate / std::sqrt(estimate))});
            return;
        }

        // One qubit is turned where it has to be: a unitary on one site changes no bond and cuts nothing, and
        // turning back restores the tensor to rounding.
        const Matrix basis = diagonal ? Matrix(Matrix::Identity()) : Matrix(decomposition.matrixV());
        if (!diagonal)
            state->apply_unitary(qubits, ToAerMatrix(basis.adjoint()));
        const std::vector<double> probabilities = state->probabilities(qubits);
        double weight = 0., total = 0.;
        for (Eigen::Index k = 0; k < dimension; ++k)
        {
            // The squared norm of a column, or the squared singular value: sums of squares and squares, exact for a
            // small one as for a large one.
            const double value = diagonal ? gate.col(k).squaredNorm() : decomposition.singularValues()(k) * decomposition.singularValues()(k);
            weight += value * probabilities.at(static_cast<size_t>(k));
            total += probabilities[static_cast<size_t>(k)];
        }
        if (!(weight > NoWeightBelow() * largest * total) || !std::isfinite(weight))
        {
            if (!diagonal)
                state->apply_unitary(qubits, ToAerMatrix(basis)); // back to the basis the state was in
            throw std::runtime_error(noState);
        }
        // M V = U S, taken from the factors.
        const Matrix remaining = diagonal ? Matrix(gate) : Matrix(decomposition.matrixU() * decomposition.singularValues().asDiagonal());
        state->apply_kraus(qubits, std::vector<AER::cmatrix_t>{ToAerMatrix(remaining / std::sqrt(weight))});
    }

    /**
     * The fraction of what a matrix could leave below which a weight counts
     * as none. Two things bound what a chain knows of a small weight: the
     * rounding of amplitudes that cancel, (64 epsilon)^2 as on QCSim, and the
     * truncation of the chain itself. At every update Aer drops the Schmidt
     * values whose squares are below 1e-16, and those that add up to less
     * than a larger truncation threshold when one is configured, so the
     * chain holds its state to that weight only: |Phi-> with one qubit turned
     * by 2e-4 has no weight on |Phi+>, and the chain, cut, held 2.5e-17 of
     * it, which a projection onto |Phi+> then returned as a state. A margin
     * of 16 covers a few cuts. A register of one qubit has no bond to cut.
     */
    double NoWeightBelow() const
    {
        constexpr double amplitudeRounding = 64. * std::numeric_limits<double>::epsilon();
        if (GetNumberOfQubits() < 2)
            return amplitudeRounding * amplitudeRounding;
        double truncation = 1e-16; // what Aer drops whatever is configured
        if (configuration.IsSet("matrix_product_state_truncation_threshold"))
        {
            try
            {
                const double configured = std::stod(configuration.GetConfiguration("matrix_product_state_truncation_threshold"));
                if (std::isfinite(configured))
                    truncation = std::max(truncation, configured);
            }
            catch (const std::exception &)
            {
            }
        }
        return 16. * truncation;
    }

    /**
     * <psi|H|psi> for a Hermitian 4x4 H on two qubits of a normalized matrix
     * product state, from the Pauli expectation values of the pair (H is a
     * sum of Pauli products), which do not change the state. Its terms
     * cancel: it is exact to the rounding of the largest, not of the result.
     */
    template <class Hermitian> double PauliWeight(const Hermitian &squared, const AER::reg_t &qubits)
    {
        using Complex = std::complex<double>;
        static const std::array<Eigen::Matrix2cd, 4> paulis = [] {
            std::array<Eigen::Matrix2cd, 4> p;
            p[0] << 1., 0., 0., 1.;
            p[1] << 0., 1., 1., 0.;
            p[2] << 0., Complex(0., -1.), Complex(0., 1.), 0.;
            p[3] << 1., 0., 0., -1.;
            return p;
        }();
        double weight = 0.;
        for (size_t first = 0; first < 4; ++first) // on qubits[0], the low bit of the matrix index
            for (size_t second = 0; second < 4; ++second)
            {
                Complex coefficient = 0.;
                for (Eigen::Index row = 0; row < 4; ++row)
                    for (Eigen::Index column = 0; column < 4; ++column)
                        coefficient += squared(row, column) * paulis[first](column % 2, row % 2) * paulis[second](column / 2, row / 2);
                const double part = coefficient.real() / 4.;
                if (part == 0.)
                    continue;
                // The qubits in ascending order, and Aer reads the Pauli string from its last character.
                std::vector<std::pair<Types::qubit_t, char>> factors;
                if (first)
                    factors.emplace_back(qubits[0], "IXYZ"[first]);
                if (second)
                    factors.emplace_back(qubits[1], "IXYZ"[second]);
                std::sort(factors.begin(), factors.end());
                AER::reg_t targets;
                std::string pauli;
                for (const auto &factor : factors)
                {
                    targets.push_back(factor.first);
                    pauli.insert(pauli.begin(), factor.second);
                }
                weight += part * (targets.empty() ? 1. : state->expval_pauli(targets, pauli));
            }
        return weight;
    }

    void InitializeDensityMatrixBuffer(size_t numQubits, std::complex<double> *values)
    {
        if (GetConfiguration("precision") == "double")
        {
            state->initialize_density_matrix(numQubits, values, true, true);
            return;
        }
        AllocateQubits(numQubits);
        Initialize();
        const size_t dimension = size_t{1} << numQubits;
        AER::Operations::Op op;
        op.type = AER::Operations::OpType::set_densmat;
        op.name = "set_density_matrix";
        op.mats.emplace_back(dimension, dimension);
        std::copy(values, values + dimension * dimension, op.mats.back().data());
        for (size_t q = 0; q < numQubits; ++q)
            op.qubits.push_back(q);
        state->buffer_op(std::move(op));
        Flush();
    }

    // The amplitudes of a register: 2^numQubits of them, or the reads that follow run past the vector.
    static void RequireStatevectorLength(size_t numQubits, size_t length)
    {
        const size_t dimension = ListableBasisStateCount<std::complex<double>>(numQubits);
        if (length != dimension)
            throw std::invalid_argument("A statevector of " + std::to_string(numQubits) + " qubits holds " + std::to_string(dimension) + " amplitudes, not " +
                                        std::to_string(length));
    }

    void InitializeDensityMatrixFromStatevector(size_t numQubits, const std::complex<double> *amplitudes)
    {
        const size_t dimension = ListableBasisStateCount<std::complex<double>>(numQubits);
        ListableBasisStateCount<std::complex<double>>(2 * numQubits); // the dimension^2 entries of the matrix
        AER::cmatrix_t densityMatrix(dimension, dimension);
        for (size_t row = 0; row < dimension; ++row)
            for (size_t column = 0; column < dimension; ++column)
                densityMatrix(row, column) = amplitudes[row] * std::conj(amplitudes[column]);

        InitializeDensityMatrixBuffer(numQubits, densityMatrix.data());
    }

    SimulationType simulationType = SimulationType::kStatevector;               /**< The simulation type. */
    std::unique_ptr<QiskitAerState> state = std::make_unique<QiskitAerState>(); /**< The qiskit aer state. */
    AER::Vector<complex_t> savedAmplitudes;                                     /**< The amplitudes, saved. */

    /** How far the generic gates may have moved a matrix product state since its tensors were last brought up to date, and the qubits they met. */
    struct PendingDeviation
    {
        double total = 0.;
        std::vector<char> touched;

        void clear()
        {
            total = 0.;
            touched.clear();
        }
    };

    PendingDeviation pendingDeviation;
    PendingDeviation savedPendingDeviation; /**< Its value at SaveState. */
    bool normSaved = false;                 /**< Whether SaveState has run. */
    AER::cmatrix_t savedDensityMatrix;      /**< The density matrix, saved. */
    std::shared_ptr<AER::QuantumState::Base> savedExtendedStabilizerState;

    bool enableMultithreading = true; /**< The multithreading flag. */
    AER::Data savedState;             /**< The saved data - here there will be the saved state
                                         of the simulator */
    std::mt19937_64 rng;
    uint64_t nextSeedStream = 0;
    std::uniform_real_distribution<double> uniformZeroOne{0., 1.};

    Configuration configuration; /**< The configuration of the simulator. */
};

} // namespace Private
} // namespace Simulators

#endif

#endif

#endif // !_AER_STATE_H_
