/**
 * @file Composite.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The composite simulator class.
 *
 * Uses multiple simulators to simulate quantum computations.
 * The simulators are joined together when qubits in different simulators are
 * entangled. The simulators are then split apart when qubits in different
 * simulators are disentangled (eg measurements or resets).
 */

#pragma once

#ifndef _IMMEDIATE_COMPOSITE_H_
#define _IMMEDIATE_COMPOSITE_H_

#ifdef INCLUDED_BY_FACTORY

#include "Individual.h"

#include <vector>

#include "../Core/Configuration.h"

namespace Simulators
{

// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private
{

/**
 * @class ImmediateCompositeSimulator
 * @brief The composite simulator class.
 *
 * Uses multiple simulators to simulate quantum computations.
 * The simulators are joined together when qubits in different simulators are
 * entangled. The simulators are split apart when qubits are disentangled (eg
 * measurements or resets).
 */
class ImmediateCompositeSimulator : public ISimulator
{
  public:
    /**
     * @brief The constructor.
     *
     * The constructor for the composite simulator.
     *
     * @param type The type of simulators to create and use.
     */
    ImmediateCompositeSimulator(SimulatorType type =
#ifdef NO_QISKIT_AER
                                    Simulators::SimulatorType::kQCSim
#else
                                    Simulators::SimulatorType::kQiskitAer
#endif
                                )
        : type(type)
    {
        SeedAuxiliaryRng(samplingSeed);
        // just in case somebody tries to use a composite simulator composed of
        // composite simulators or a gpu simulator - this one would work, but until
        // we implement some supporting functionality with cuda, especially for
        // composite probably it shouldn't be used splitting and merging is slow and
        // it should be done in the videocard memory instead of using the cpu, if
        // the gpu simulators are used otherwise the transfer from videocard memory
        // and host memory back and forth is going to slow down the simulation quite
        // a bit
        if (
#ifndef NO_QISKIT_AER
            type != SimulatorType::kQiskitAer &&
#endif
            type != SimulatorType::kQCSim)
            type =
#ifdef NO_QISKIT_AER
                Simulators::SimulatorType::kQCSim;
#else
                Simulators::SimulatorType::kQiskitAer;
#endif
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it after the qubits allocation.
     *
     * @sa QCSimState::AllocateQubits
     */
    void Initialize() override
    {
        simulators.clear();
        qubitsMap.resize(nrQubits);
        const uint64_t initializationSeed = nextInitialization++ == 0 ? samplingSeed : DeriveSeed(samplingSeed, 0x494E4954ULL + nextInitialization);

        // start with one qubit simulators:
        for (size_t q = 0; q < nrQubits; ++q)
        {
            qubitsMap[q] = q;
            auto sim = std::make_unique<IndividualSimulator>(type);
            sim->AllocateQubits(1);

            // this is for tests when we default to matrix product state simulator
#if 0
					sim->Configure("method", "statevector");
#endif

            sim->SetMultithreading(enableMultithreading);
            if (config.IsSet("precision"))
                sim->Configure("precision", config.GetConfiguration("precision").c_str());
            sim->Initialize();
            for (const auto &[key, value] : config.GetConfigMap())
                if (key != "seed" && key != "method")
                    sim->Configure(key.c_str(), value.c_str());
            sim->SetSeed(DeriveSeed(initializationSeed, q));
            sim->GetQubitsMap()[q] = 0;
            simulators[q] = std::move(sim);
        }
        nextId = nrQubits;
    }

    /**
     * @brief Just resets the state to 0.
     *
     * Does not destroy the internal state, just resets it to zero (as a 'reset'
     * op on each qubit would do).
     */
    void Reset() override
    {
        Initialize();
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This is good only for a statevector simulator and should be used only by
     * calling from a composite simulator.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        throw std::runtime_error("CompositeSimulator::InitializeState not supported");
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This is good only for a statevector simulator and should be used only by
     * calling from a composite simulator.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    /*
    void InitializeState(size_t num_qubits, std::vector<std::complex<double>,
    avoid_init_allocator<std::complex<double>>>& amplitudes) override
    {
            throw std::runtime_error("CompositeSimulator::InitializeState not
    supported");
    }
    */

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This is good only for a statevector simulator and should be used only by
     * calling from a composite simulator.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
#ifndef NO_QISKIT_AER
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        throw std::runtime_error("CompositeSimulator::InitializeState not supported");
    }
#endif

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it only on a non-initialized state.
     * This is good only for a statevector simulator and should be used only by
     * calling from a composite simulator.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param amplitudes A vector with the amplitudes to initialize the state
     * with.
     */
    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        throw std::runtime_error("CompositeSimulator::InitializeState not supported");
    }

    /**
     * @brief Configures the state.
     *
     * This function is called to configure the simulator.
     * Currently only the composite simulator does not support this, in the future
     * it might be implemented for qiskit aer.
     *
     * @param key The key of the configuration option.
     * @param value The value of the configuration.
     */
    void Configure(const char *key, const char *value) override
    {
        Utils::Sampling::ValidateSetting(key, value);
        // don't allow chaning the method, it should stay statevector
        if (std::string("method") == key)
            return;

        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            Configuration::ParsePrecision(key, value);
            // Fixed once the children exist; a reapplied precision is ignored until
            // Clear, so later children cannot end up with a different one.
            if (!simulators.empty())
                return;
            config.SetConfiguration(key, value);
            return;
        }

        config.SetConfiguration(key, value);

        if (std::string("seed") == key)
        {
            const uint64_t seed = std::stoull(value);
            samplingSeed = seed;
            nextCloneStream = nextRestoreStream = nextInitialization = nextSplitStream = 0;
            SeedAuxiliaryRng(seed);
            for (auto &[id, simulator] : simulators)
                simulator->SetSeed(DeriveSeed(seed, id));
        }
        else
        {
            for (auto &[id, simulator] : simulators)
                simulator->Configure(key, value);
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
        if (std::string(key) == "sampling_seed")
            return std::to_string(samplingSeed);
        if (std::string(key) == "sampling_policy" && type == SimulatorType::kQCSim && !config.IsSet(key))
            return "reproducible_v1";
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            if (!simulators.empty())
                return simulators.begin()->second->GetConfiguration(key);
            const bool useDouble = type == SimulatorType::kQCSim || !config.IsSet("precision") || config.GetConfiguration("precision") == "double";
            return std::string(key) == "precision" ? (useDouble ? "double" : "single") : (useDouble ? "1" : "0");
        }
        if (simulators.empty())
            return "";

        return config.GetConfiguration(key);
    }

    /**
     * @brief Allocates qubits.
     *
     * This function is called to allocate qubits.
     *
     * @param num_qubits The number of qubits to allocate.
     * @return The index of the first qubit allocated.
     */
    size_t AllocateQubits(size_t num_qubits) override
    {
        if (!simulators.empty())
            return 0;

        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    /**
     * @brief Returns the number of qubits.
     *
     * This function is called to obtain the number of the allocated qubits.
     * @return The number of qubits.
     */
    size_t GetNumberOfQubits() const override
    {
        return nrQubits;
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
        simulators.clear();
        qubitsMap.clear();
        nrQubits = 0;
    }

    /**
     * @brief Saves the state to internal storage.
     *
     * Saves the state to internal storage, if needed.
     */
    void SaveStateToInternalDestructive() override
    {
        for (auto &[id, simulator] : simulators)
            simulator->SaveStateToInternalDestructive();
    }

    /**
     * @brief Restores the state from the internally saved state
     *
     * Restores the state from the internally saved state, if needed.
     * This does something only for qiskit aer.
     */
    void RestoreInternalDestructiveSavedState() override
    {
        for (auto &[id, simulator] : simulators)
            simulator->RestoreInternalDestructiveSavedState();
    }

    /**
     * @brief Gets the amplitude.
     *
     * Gets the amplitude, from the internal storage if needed.
     * This is needed only for the composite simulator, for an optimization for
     * qiskit aer. For qcsim it does the same thing as Amplitude. For the
     * composite simulator it should not be called.
     */
    std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override
    {
        std::complex<double> res = 1.0;

        for (auto &[id, simulator] : simulators)
            res *= simulator->AmplitudeRaw(simulator->ConvertOutcomeFromGlobal(outcome));

        return res;
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

        size_t res = 0;
        size_t mask = 1ULL;

        DontNotify();
        for (Types::qubit_t qubit : qubits)
        {
            const bool outcome = GetSimulator(qubit)->Measure({qubit}) != 0;
            if (outcome)
                res |= mask;
            mask <<= 1;

            Split(qubit, outcome);
        }
        Notify();

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
        std::vector<bool> res(qubits.size(), false);

        DontNotify();
        for (size_t q = 0; q < qubits.size(); ++q)
        {
            Types::qubit_t qubit = qubits[q];
            const bool outcome = GetSimulator(qubit)->Measure({qubit}) != 0;
            if (outcome)
                res[q] = true;

            Split(qubit, outcome);
        }
        Notify();

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
        DontNotify();
        for (Types::qubit_t qubit : qubits)
        {
            GetSimulator(qubit)->ApplyReset({qubit});
            Split(qubit, false);
        }
        Notify();

        NotifyObservers(qubits);
    }

    /**
     * @brief Returns the probability of the specified outcome.
     *
     * Use it to obtain the probability to obtain the specified outcome, if all
     * qubits are measured.
     *
     * @param outcome The outcome to obtain the probability for.
     * @return The probability of the specified outcome.
     */
    double Probability(Types::qubit_t outcome) override
    {
        return std::norm(Amplitude(outcome));
    }

    /**
     * @brief Returns the amplitude of the specified state.
     *
     * Use it to obtain the amplitude of the specified state.
     *
     * @param outcome The outcome to obtain the amplitude for.
     * @return The amplitude of the specified outcome.
     */
    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        std::complex<double> res = 1.0;

        for (auto &[id, simulator] : simulators)
            res *= simulator->Amplitude(outcome);

        return res;
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
        std::complex<double> res = 1.0;

        for (auto &[id, simulator] : simulators)
            res *= simulator->ProjectOnZero();

        return res;
    }

    /**
     * @brief Returns the probabilities of all possible outcomes.
     *
     * Use it to obtain the probabilities of all possible outcomes.
     *
     * @return A vector with the probabilities of all possible outcomes.
     */
    std::vector<double> AllProbabilities() override
    {
        const size_t nrBasisStates = 1ULL << nrQubits;
        std::vector<double> result;

        for (size_t i = 0; i < nrBasisStates; ++i)
            result.emplace_back(Probability(i));

        return result;
    }

    /**
     * @brief Returns the probabilities of the specified outcomes.
     *
     * Use it to obtain the probabilities of the specified outcomes.
     *
     * @param qubits A vector with the qubits configuration outcomes.
     * @return A vector with the probabilities for the specified qubit
     * configurations.
     */
    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result;

        for (size_t i = 0; i < qubits.size(); ++i)
            result.emplace_back(Probability(qubits[i]));

        return result;
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
     * @return A map with the counts for the otcomes of measurements of the
     * specified qubits.
     */
    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (type == SimulatorType::kQCSim && config.GetConfiguration("sampling_policy") != "legacy")
            return SamplePrepared<false>(qubits, shots);
        if (qubits.empty() || !shots)
            return {};
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        // TODO: improve it as for the qcsim statevector simulator case!
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();

        SaveStateToInternalDestructive();

        if (shots > 1)
        {
            InitializeAlias();

            for (size_t shot = 0; shot < shots; ++shot)
            {
                size_t measRaw = 0;
                for (auto &[id, simulator] : simulators)
                    measRaw |= simulator->SampleFromAlias();

                size_t meas = 0;
                size_t mask = 1ULL;
                for (auto q : qubits)
                {
                    const size_t qubitMask = 1ULL << q;
                    if ((measRaw & qubitMask) != 0)
                        meas |= mask;
                    mask <<= 1ULL;
                }

                ++result[meas];
            }

            ClearAlias();
        }
        else
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto measRaw = MeasureNoCollapse();

                size_t meas = 0;
                size_t mask = 1ULL;
                for (auto q : qubits)
                {
                    const size_t qubitMask = 1ULL << q;
                    if ((measRaw & qubitMask) != 0)
                        meas |= mask;
                    mask <<= 1ULL;
                }

                ++result[meas];
            }

        RestoreInternalDestructiveSavedState();

        Notify();
        NotifyObservers(qubits);

        return result;
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
        if (type == SimulatorType::kQCSim && config.GetConfiguration("sampling_policy") != "legacy")
            return SamplePrepared<true>(qubits, shots);
        if (qubits.empty() || !shots)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();

        SaveStateToInternalDestructive();
        std::vector<bool> meas(qubits.size());
        if (shots > 1)
        {
            InitializeAlias();

            for (size_t shot = 0; shot < shots; ++shot)
            {
                size_t measRaw = 0;
                for (auto &[id, simulator] : simulators)
                    measRaw |= simulator->SampleFromAlias();

                for (size_t i = 0; i < qubits.size(); ++i)
                    meas[i] = ((measRaw >> qubits[i]) & 1) == 1;

                ++result[meas];
            }

            ClearAlias();
        }
        else
        {
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto measRaw = MeasureNoCollapseMany();

                for (size_t i = 0; i < qubits.size(); ++i)
                    meas[i] = measRaw[qubits[i]];

                ++result[meas];
            }
        }

        RestoreInternalDestructiveSavedState();

        Notify();
        NotifyObservers(qubits);

        return result;
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
    double ExpectationValue(const std::string &pauliString) override
    {
        std::unordered_map<size_t, std::string> pauliStrings;

        for (size_t q = 0; q < pauliString.size(); ++q)
        {
            const char op = toupper(pauliString[q]);
            if (op == 'I')
                continue;

            const size_t simId = qubitsMap[q];
            const size_t localQubit = simulators[simId]->GetQubitsMap().at(q);

            if (pauliStrings[simId].size() <= localQubit)
                pauliStrings[simId].resize(localQubit + 1, 'I');

            pauliStrings[simId][localQubit] = op;
        }

        double result = 1.0;
        for (auto &[id, localPauliString] : pauliStrings)
            result *= simulators[id]->ExpectationValue(localPauliString);

        return result;
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
#ifdef NO_QISKIT_AER
        return SimulatorType::kCompositeQCSim;
#else
        if (type != SimulatorType::kQiskitAer)
            return SimulatorType::kCompositeQCSim;

        return SimulatorType::kCompositeQiskitAer;
#endif
    }

    /**
     * @brief Returns the simulation type.
     *
     * Returns the simulation type.
     *
     * @return The simulation type.
     * @sa SimulationType
     */
    SimulationType GetSimulationType() const override
    {
        return SimulationType::kStatevector;
    }

    /**
     * @brief Flushes the applied operations
     *
     * This function is called to flush the applied operations.
     * It is used to flush the operations that were applied to the state, qcsim
     * applies them right away, but qiskit aer does not.
     */
    void Flush() override
    {
        for (auto &[id, simulator] : simulators)
            simulator->Flush();
    }

    /**
     * @brief Apply a generic one-qubit gate to the specified qubit.
     * @param qubit The qubit to apply the gate to.
     * @param gate The 2x2 matrix representing the gate.
     */
    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        GetSimulator(qubit)->ApplyGenericOneQubitGate(qubit, gate);
        NotifyObservers({qubit});
    }

    /**
     * @brief Apply a generic two-qubit gate to the specified qubits.
     * @param qubit0 The first qubit to apply the gate to.
     * @param qubit1 The second qubit to apply the gate to.
     * @param gate The 4x4 matrix representing the gate.
     */
    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        JoinIfNeeded(qubit0, qubit1);
        GetSimulator(qubit0)->ApplyGenericTwoQubitGate(qubit0, qubit1, gate);
        NotifyObservers({qubit0, qubit1});
    }

    void ApplyGenericThreeQubitGate(Types::qubit_t q0, Types::qubit_t q1, Types::qubit_t q2, const Matrix8cd &gate) override
    {
        JoinIfNeeded(q0, q1);
        JoinIfNeeded(q0, q2);
        GetSimulator(q0)->ApplyGenericThreeQubitGate(q0, q1, q2, gate);
        NotifyObservers({q0, q1, q2});
    }

    // YES, all one qubit gates are that easy:

    /**
     * @brief Applies a phase shift gate to the qubit
     *
     * Applies a specified phase shift gate to the qubit
     * @param qubit The qubit to apply the gate to.
     * @param lambda The phase shift angle.
     */
    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        GetSimulator(qubit)->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a not gate to the qubit
     *
     * Applies a not (X) gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyX(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Y gate to the qubit
     *
     * Applies a not (Y) gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyY(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Z gate to the qubit
     *
     * Applies a not (Z) gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyZ(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Hadamard gate to the qubit
     *
     * Applies a Hadamard gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyH(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a S gate to the qubit
     *
     * Applies a S gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyS(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a S dagger gate to the qubit
     *
     * Applies a S dagger gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplySDG(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a T gate to the qubit
     *
     * Applies a T gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyT(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a T dagger gate to the qubit
     *
     * Applies a T dagger gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyTDG(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Sx gate to the qubit
     *
     * Applies a Sx gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplySx(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplySx(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Sx dagger gate to the qubit
     *
     * Applies a Sx dagger gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplySxDAG(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplySxDAG(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a K gate to the qubit
     *
     * Applies a K (Hy) gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     */
    void ApplyK(Types::qubit_t qubit) override
    {
        GetSimulator(qubit)->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Rx gate to the qubit
     *
     * Applies an x rotation gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     * @param theta The rotation angle.
     */
    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        GetSimulator(qubit)->ApplyRx(qubit, theta);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Ry gate to the qubit
     *
     * Applies a y rotation gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     * @param theta The rotation angle.
     */
    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        GetSimulator(qubit)->ApplyRy(qubit, theta);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a Rz gate to the qubit
     *
     * Applies a z rotation gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     * @param theta The rotation angle.
     */
    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        GetSimulator(qubit)->ApplyRz(qubit, theta);
        NotifyObservers({qubit});
    }

    /**
     * @brief Applies a U gate to the qubit
     *
     * Applies a z rotation gate to the specified qubit
     * @param qubit The qubit to apply the gate to.
     * @param theta The first parameter.
     * @param phi The second parameter.
     * @param lambda The third parameter.
     * @param gamma The fourth parameter.
     */
    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        GetSimulator(qubit)->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    // the gates that operate on more than one qubit need joining of the
    // simulators, if the qubits are in different simulators

    /**
     * @brief Applies a CX gate to the qubits
     *
     * Applies a controlled X gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     */
    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CY gate to the qubits
     *
     * Applies a controlled Y gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     */
    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CZ gate to the qubits
     *
     * Applies a controlled Z gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     */
    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CP gate to the qubits
     *
     * Applies a controlled phase gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     * @param lambda The phase shift angle.
     */
    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CRx gate to the qubits
     *
     * Applies a controlled x rotation gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     * @param theta The rotation angle.
     */
    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCRx(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CRy gate to the qubits
     *
     * Applies a controlled y rotation gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     * @param theta The rotation angle.
     */
    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCRy(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CRz gate to the qubits
     *
     * Applies a controlled z rotation gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     * @param theta The rotation angle.
     */
    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCRz(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CH gate to the qubits
     *
     * Applies a controlled Hadamard gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     */
    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CSx gate to the qubits
     *
     * Applies a controlled squared root not gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     */
    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCSx(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a CSx dagger gate to the qubits
     *
     * Applies a controlled squared root not dagger gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     */
    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCSxDAG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    /**
     * @brief Applies a swap gate to the qubits
     *
     * Applies a swap gate to the specified qubits
     * @param qubit0 The first qubit
     * @param qubit1 The second qubit
     */
    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        JoinIfNeeded(qubit0, qubit1);
        GetSimulator(qubit0)->ApplySwap(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    /**
     * @brief Applies a controlled controlled not gate to the qubits
     *
     * Applies a controlled controlled not gate to the specified qubits
     * @param qubit0 The first control qubit
     * @param qubit1 The second control qubit
     * @param qubit2 The target qubit
     */
    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        // TODO: See if it's worth optimizing to joing all three simulators at once,
        // if needed (that is, there is a different one for each qubit)
        JoinIfNeeded(qubit0, qubit1);
        JoinIfNeeded(qubit0, qubit2);
        GetSimulator(qubit0)->ApplyCCX(qubit0, qubit1, qubit2);
        NotifyObservers({qubit2, qubit1, qubit0});
    }

    /**
     * @brief Applies a controlled swap gate to the qubits
     *
     * Applies a controlled swap gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param qubit0 The first qubit
     * @param qubit1 The second qubit
     */
    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        // TODO: See if it's worth optimizing to joing all three simulators at once,
        // if needed (that is, there is a different one for each qubit)
        JoinIfNeeded(ctrl_qubit, qubit0);
        JoinIfNeeded(ctrl_qubit, qubit1);
        GetSimulator(ctrl_qubit)->ApplyCSwap(ctrl_qubit, qubit0, qubit1);
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    /**
     * @brief Applies a controlled U gate to the qubits
     *
     * Applies a controlled U gate to the specified qubits
     * @param ctrl_qubit The control qubit
     * @param tgt_qubit The target qubit
     * @param theta Theta parameter for the U gate
     * @param phi Phi parameter for the U gate
     * @param lambda Lambda parameter for the U gate
     * @param gamma Gamma parameter for the U gate
     */
    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        JoinIfNeeded(ctrl_qubit, tgt_qubit);
        GetSimulator(ctrl_qubit)->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyNop() override
    {
        GetSimulator(0)->ApplyNop();
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
        for (auto &[id, simulator] : simulators)
            simulator->SetMultithreading(multithreading);
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
        return type == Simulators::SimulatorType::kQCSim;
    }

    /**
     * @brief Saves the state to internal storage.
     *
     * Saves the state to internal storage, if needed.
     * Calling this will not destroy the internal state, unlike the 'Destructive'
     * variant. To be used in order to recover the state after doing measurements,
     * for multiple shots executions. In the first phase, only qcsim will
     * implement this.
     *
     * For composite, at least for qcsim variant, it can be implemented but it's
     * cumbersome, everything needs to be copied calling SaveState on individual
     * simulators is not enough, as the structures will be changed after
     * measurements.
     */
    void SaveState() override
    {
        savedState = CloneStateForExecution(DeriveSeed(samplingSeed, 0x53415645ULL), false);
    }

    /**
     * @brief Restores the state from the internally saved state
     *
     * Restores the state from the internally saved state, if needed.
     * To be used in order to recover the state after doing measurements, for
     * multiple shots executions. In the first phase, only qcsim will implement
     * this.
     *
     * For composite, at least for qcsim variant, it can be implemented but it's
     * cumbersome, everything needs to be copied calling SaveState on individual
     * simulators is not enough, as the structures will be changed after
     * measurements.
     */
    void RestoreState() override
    {
        if (savedState)
        {
            const ImmediateCompositeSimulator *savedStatePtr = static_cast<ImmediateCompositeSimulator *>(savedState.get());

            type = savedStatePtr->type;           /**< The type of simulators to create and use */
            qubitsMap = savedStatePtr->qubitsMap; /**< A map between qubits (as identified from
                                                     outside) and the individual simulators ids */

            nrQubits = savedStatePtr->nrQubits;                         /**< The number of allocated qubits. */
            nextId = savedStatePtr->nextId;                             /**< The next id to be used for a new simulator. */
            enableMultithreading = savedStatePtr->enableMultithreading; /**< A flag to indicate if multithreading
                                                                           should be enabled. */

            simulators.clear();
            const uint64_t restoreSeed = DeriveSeed(samplingSeed, 0x52455354ULL + nextRestoreStream++);
            for (auto &[id, simulator] : savedStatePtr->simulators)
            {
                auto isim = simulator->CloneForExecution(DeriveSeed(restoreSeed, id));
                simulators[id] = std::unique_ptr<IndividualSimulator>(static_cast<IndividualSimulator *>(isim.release()));
            }
        }
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
        Types::qubit_t res = 0;

        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;

        for (auto &[id, simulator] : simulators)
        {
            const Types::qubit_t meas = simulator->MeasureNoCollapse();
            res |= meas;
        }

        return res;
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
        std::vector<bool> res(nrQubits, false);
        for (auto &[id, simulator] : simulators)
        {
            const std::vector<bool> meas = simulator->MeasureNoCollapseMany();
            for (size_t i = 0; i < meas.size(); ++i)
                if (meas[i])
                    res[i] = true;
        }
        return res;
    }

    /**
     * @brief Clones the simulator.
     *
     * Clones the simulator, including the state, the configuration and the
     * internally saved state, if any. Does not copy the observers. Should be used
     * mainly internally, to optimise multiple shots execution, copying the state
     * from the simulator used for timing.
     *
     * @return A shared pointer to the cloned simulator.
     */
    std::unique_ptr<ISimulator> Clone() override
    {
        if (nextCloneStream == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Composite clone stream exhausted");
        return CloneForExecution(DeriveSeed(samplingSeed, nextCloneStream++));
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        return CloneStateForExecution(seed, true);
    }

  private:
    std::unique_ptr<ISimulator> CloneStateForExecution(uint64_t seed, bool includeSnapshot)
    {
        auto clone = std::make_unique<ImmediateCompositeSimulator>(type);

        clone->type = type;           /**< The type of simulators to create and use */
        clone->qubitsMap = qubitsMap; /**< A map between qubits (as identified from outside) and
                                         the individual simulators ids */

        clone->nrQubits = nrQubits;                         /**< The number of allocated qubits. */
        clone->nextId = nextId;                             /**< The next id to be used for a new simulator. */
        clone->enableMultithreading = enableMultithreading; /**< A flag to indicate if multithreading should
                                                               be enabled. */

        clone->config = config; /**< The configuration of the simulator. */

        for (auto &[id, simulator] : simulators)
        {
            auto isim = simulator->CloneForExecution(DeriveSeed(seed, id));
            clone->simulators[id] = std::unique_ptr<IndividualSimulator>(static_cast<IndividualSimulator *>(isim.release()));
        }

        if (includeSnapshot && savedState)
            clone->savedState = static_cast<ImmediateCompositeSimulator *>(savedState.get())->CloneStateForExecution(
                DeriveSeed(seed, 0x53415645ULL), false);

        clone->SetSeed(seed);
        return clone;
    }

  public:
    const Configuration &GetConfiguration() const
    {
        return config;
    }

    const std::unordered_map<std::string, std::string> &GetConfigMap() const override
    {
        return config.GetConfigMap();
    }

  private:
    template <bool Many> Utils::Sampling::Counts<Many> SamplePrepared(const Types::qubits_vector &qubits, size_t shots)
    {
        using namespace Utils::Sampling;
        using Plan = QCSimStatevectorSimulator::SamplingPlan;
        Counts<Many> result;
        if (qubits.empty() || !shots)
            return result;
        ValidateQubits(qubits, nrQubits, Many);
        if (simulators.size() == 1)
        {
            auto &component = *simulators.begin()->second;
            if constexpr (Many) result = component.SampleCountsMany(qubits, shots);
            else result = component.SampleCounts(qubits, shots);
            NotifyObservers(qubits);
            return result;
        }

        // Duplicate selections share a joint bit; output positions are expanded
        // only when inserting counts. No global qubit ID is used as a shift.
        std::unordered_map<size_t, size_t> uniqueBits;
        std::vector<size_t> outputBits;
        for (const auto qubit : qubits)
            outputBits.push_back(uniqueBits.emplace(qubit, uniqueBits.size()).first->second);
        constexpr size_t wordBits = std::numeric_limits<size_t>::digits;
        const size_t words = (uniqueBits.size() + wordBits - 1) / wordBits;
        struct Component
        {
            size_t id;
            IndividualSimulator *simulator;
            std::vector<std::pair<size_t, size_t>> bits; // local bit -> joint bit
            std::unique_ptr<Plan> plan;
            std::vector<size_t> projection;
            size_t minimumBytes = 0;
        };
        std::vector<Component> components;
        components.reserve(simulators.size());
        for (auto &[id, simulator] : simulators)
        {
            Component component{id, simulator.get()};
            for (auto [qubit, local] : simulator->GetQubitsMap())
                if (const auto found = uniqueBits.find(qubit); found != uniqueBits.end())
                    component.bits.emplace_back(local, found->second);
            components.push_back(std::move(component));
        }
        // Stable budget assignment; neither hash iteration order nor worker
        // count is allowed to choose a component's sampling algorithm.
        std::sort(components.begin(), components.end(), [](const auto &a, const auto &b) { return a.id < b.id; });
        auto project = [&](const Component &component, size_t row, size_t *joint) {
            if (!component.projection.empty())
                joint[0] |= component.projection[row];
            else
                for (const auto [local, output] : component.bits)
                    joint[output / wordBits] |= ((row >> local) & size_t{1}) << (output % wordBits);
        };
        std::vector<bool> key;
        if constexpr (Many) key.resize(qubits.size());
        auto insert = [&](const size_t *joint, size_t count) {
            if constexpr (Many)
            {
                for (size_t bit = 0; bit < outputBits.size(); ++bit)
                    key[bit] = ((joint[outputBits[bit] / wordBits] >> (outputBits[bit] % wordBits)) & 1) != 0;
                result[key] += count;
            }
            else
            {
                Types::qubit_t packed = 0;
                for (size_t bit = 0; bit < outputBits.size(); ++bit)
                    packed |= Types::qubit_t((joint[outputBits[bit] / wordBits] >> (outputBits[bit] % wordBits)) & 1) << bit;
                result[packed] += count;
            }
        };
        if (shots == 1)
        {
            std::vector<size_t> joint(words, 0);
            for (auto &component : components)
                if (component.bits.empty())
                    component.simulator->DiscardLocalMeasurementDraw();
                else
                    project(component, component.simulator->SampleLocalOnce(), joint.data());
            insert(joint.data(), 1);
            NotifyObservers(qubits);
            return result;
        }

        const Options options = components.front().simulator->SamplingOptions();
        const size_t sharedReserve = options.memoryBytes / 4;
        size_t remaining = options.memoryBytes - sharedReserve, reserved = 0;
        const size_t selected = std::count_if(components.begin(), components.end(), [](const auto &c) { return !c.bits.empty(); });
        const size_t componentBudget = remaining / std::max<size_t>(1, selected);
        for (auto &component : components)
            if (!component.bits.empty())
            {
                component.minimumBytes = Plan::MinimumMemoryBytes(component.simulator->SamplingRows(), shots, componentBudget);
                if (component.minimumBytes > remaining - reserved)
                    throw std::length_error("Composite probability indexes exceed the sampling scratch budget");
                reserved += component.minimumBytes;
            }
        for (auto &component : components)
            if (!component.bits.empty())
            {
                reserved -= component.minimumBytes;
                auto localOptions = options;
                localOptions.memoryBytes = remaining - reserved;
                component.plan = std::make_unique<Plan>(component.simulator->PrepareSampling(shots, localOptions));
                remaining -= localOptions.memoryBytes - component.plan->ScratchAvailable();
            }
        // Small local projection tables avoid per-shot bit permutation work.
        // Their memory comes from the same shared budget as sampling tables.
        for (auto &component : components)
            if (words == 1 && component.bits.size() > 1 && component.plan && component.plan->Size() <= 65536 &&
                shots >= std::max<size_t>(256, component.plan->Size()) && component.plan->Size() <= remaining / sizeof(size_t))
            {
                std::vector<size_t> table(component.plan->Size(), 0);
                for (size_t row = 0; row < table.size(); ++row) project(component, row, &table[row]);
                remaining -= table.size() * sizeof(size_t);
                component.projection = std::move(table);
            }
        remaining += sharedReserve;
        const size_t categories = uniqueBits.size() <= 20 ? size_t{1} << uniqueBits.size() : 0;
        const bool histogram = categories && shots >= std::max<size_t>(1024, categories / 4) &&
                               categories <= remaining / (2 * sizeof(size_t));
        std::vector<size_t> bins(histogram ? categories : 0, 0);
        remaining -= bins.size() * sizeof(size_t);
        // Uniforms, sampled rows, sorted-CDF scratch, and joint output words.
        if (words > (std::numeric_limits<size_t>::max() - 32) / sizeof(size_t))
            throw std::length_error("Composite sampling output is too wide");
        const size_t bytesPerShot = 32 + words * sizeof(size_t);
        const size_t batch = std::min({shots, options.batchSize, remaining / bytesPerShot});
        if (!batch)
            throw std::length_error("Composite sampling batch exceeds the scratch budget");
        std::vector<size_t> joint(batch * words), sampled;
        std::vector<double> uniforms;
        // Unselected components still consume their original one word per shot.
        // Omitting their probability preparation cannot restart later streams.
        for (auto &component : components)
            if (component.bits.empty()) component.simulator->SamplingRng().discard(shots);
        for (size_t begin = 0; begin < shots;)
        {
            const size_t count = std::min(batch, shots - begin);
            std::fill(joint.begin(), joint.begin() + count * words, 0);
            for (auto &component : components)
            {
                if (!component.plan) continue;
                const auto &plan = *component.plan;
                auto &rng = component.simulator->SamplingRng();
                const int threads = plan.Categories() <= 64 && plan.Method() == Family::Alias ? 1 : Threads(options.multithreading, count);
                if (threads == 1 && (plan.Method() == Family::Alias || plan.Size() <= 65536 || count < 128))
                {
                    for (size_t shot = 0; shot < count; ++shot)
                        project(component, plan.Sample(Utils::RandomStream::Uniform(rng)), &joint[shot * words]);
                }
                else
                {
                    uniforms.resize(count);
                    for (double &value : uniforms) value = Utils::RandomStream::Uniform(rng);
                    plan.Draw(uniforms, sampled, threads);
#ifdef _OPENMP
#pragma omp parallel for if (threads > 1) num_threads(threads) schedule(static)
#endif
                    for (int64_t shot = 0; shot < static_cast<int64_t>(count); ++shot)
                        project(component, sampled[shot], &joint[size_t(shot) * words]);
                }
            }
            for (size_t shot = 0; shot < count; ++shot)
                if (histogram) ++bins[joint[shot]];
                else insert(&joint[shot * words], 1);
            begin += count;
        }
        for (size_t row = 0; row < bins.size(); ++row)
            if (bins[row]) insert(&row, bins[row]);
        NotifyObservers(qubits);
        return result;
    }

    void InitializeAlias()
    {
        for (auto &[id, simulator] : simulators)
            simulator->InitializeAlias();
    }

    void ClearAlias()
    {
        for (auto &[id, simulator] : simulators)
            simulator->ClearAlias();
    }

    /**
     * @brief Get the simulator corresponding to a qubit.
     *
     * Gets the simulator corresponding to a qubit.
     *
     * @param qubit The qubit to get the simulator for.
     * @return The simulator corresponding to the qubit.
     */
    inline std::unique_ptr<IndividualSimulator> &GetSimulator(size_t qubit)
    {
        assert(qubitsMap.size() > qubit);
        assert(simulators.find(qubitsMap[qubit]) != simulators.end());

        // assume it was called with a valid qubit
        return simulators[qubitsMap[qubit]];
    }

    // if executing a gate on two or three qubits, use this to see if it can be
    // executed directly if it returns true, join the two simulators together and
    // execute the gate (for three qubits, do that once again for the third qubit,
    // before execution)
    /**
     * @brief Check to see if a join is needed.
     *
     * Checks to see if a join of simulators is needed for the two passed qubit
     * ids.
     *
     * @param qubit1 The first qubit id.
     * @param qubit2 The second qubit id.
     * @return True if a join is needed, false otherwise.
     */
    inline bool JoinNeeded(size_t qubit1, size_t qubit2)
    {
        // TODO: Not really needed for all gates that act on multiple qubits, is
        // this worth pursuing? the obvious example is the identity gate, which does
        // nothing, so a join is not needed
        return qubitsMap[qubit1] != qubitsMap[qubit2];
    }

    /**
     * @brief Joins the simulators if needed.
     *
     * Joins the simulators if needed for the two passed qubit ids.
     *
     * @param qubit1 The first qubit id.
     * @param qubit2 The second qubit id.
     */
    inline void JoinIfNeeded(size_t qubit1, size_t qubit2)
    {
        if (JoinNeeded(qubit1, qubit2))
        {
            const size_t simId = qubitsMap[qubit1];
            const size_t eraseSimId = qubitsMap[qubit2];
            auto &sim1 = GetSimulator(qubit1);
            const auto &sim2 = GetSimulator(qubit2);

            sim1->Join(simId, sim2, qubitsMap, enableMultithreading);

            simulators.erase(eraseSimId);
        }
    }

    /**
     * @brief Splits the simulators if needed.
     *
     * Splits the simulator if needed for the passed qubit id.
     * The qubit outcome is the result of the measurement or the value to which
     * the qubit was reset to.
     *
     * @param qubit The qubit id to split out.
     * @param qubitOutcome The outcome of the qubit.
     */
    inline void Split(size_t qubit, bool qubitOutcome = false)
    {
        auto &sim = GetSimulator(qubit);   // get the simulator for the qubit, this one will be split
        if (sim->GetNumberOfQubits() == 1) // no need to split it, it's already for a single qubit
            return;

        if (nextSplitStream == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Composite split stream exhausted");
        // A detached component gets its own stream. Restores and register
        // rebuilds must not rewind this ordinal and reuse a child's seed.
        const uint64_t splitSeed = DeriveSeed(DeriveSeed(samplingSeed, 0x53504C4954ULL), nextSplitStream++);
        qubitsMap[qubit] = nextId; // the qubit will be in the new simulator
        simulators[nextId] = sim->Split(qubit, qubitOutcome, enableMultithreading, splitSeed);

        ++nextId;
    }

    SimulatorType type;                                                          /**< The type of simulators to create and use */
    std::vector<size_t> qubitsMap;                                               /**< A map between qubits (as identified from outside) and the
                                                                                    individual simulators ids */
    std::unordered_map<size_t, std::unique_ptr<IndividualSimulator>> simulators; /**< The individual simulators */
    size_t nrQubits = 0;                                                         /**< The number of allocated qubits. */
    size_t nextId = 0;                                                           /**< The next id to be used for a new simulator. */
    bool enableMultithreading = true;                                            /**< A flag to indicate if multithreading should be enabled. */

    std::unique_ptr<ISimulator> savedState; /**< The saved state, if any. */

    Configuration config; /**< The configuration of the simulator. */
    uint64_t samplingSeed = Utils::RandomStream::FreshSeed();
    uint64_t nextCloneStream = 0, nextRestoreStream = 0, nextInitialization = 0, nextSplitStream = 0;
};

} // namespace Private
} // namespace Simulators

#endif
#endif
