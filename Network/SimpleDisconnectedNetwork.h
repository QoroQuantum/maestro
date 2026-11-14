/**
 * @file SimpleDisconnectedNetwork.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * A simple network class, implementing a network containing a buch of hosts, no
 * communication (classical or quantum) among them.
 *
 * Can execute circuits locally on each host.
 */

#pragma once

#ifndef _SIMPLE_NETWORK_H_
#define _SIMPLE_NETWORK_H_

#include "../Circuit/ChainOrder.h"
#include "../Estimators/SimulatorsEstimatorInterface.h"
#include "../Utils/Sampling/Multinomial.h"
#include "NetworkJob.h"
#include "QubitRegister.h"
#include "SimpleController.h"
#include "SimpleHost.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <optional>

#include "../Simulators/TensorNetworks/MPSDummySimulator.h"

#include "Configuration.h"

namespace Network
{

/**
 * @class SimpleDisconnectedNetwork
 * @brief The simple network implementation.
 *
 * The simple network implementation.
 * A simple network class, implementing a network containing a buch of hosts, no
 * communication (classical or quantum) among them.
 *
 * @tparam Time The time type used for execution times.
 * @sa INetwork
 * @sa SimpleController
 * @sa SimpleHost
 */
template <typename Time = Types::time_type, class Controller = SimpleController<Time>> class SimpleDisconnectedNetwork : public INetwork<Time>
{
  public:
    using BaseClass = INetwork<Time>;                          /**< The base class type. */
    using ExecuteResults = typename BaseClass::ExecuteResults; /**< The execute results type. */

    /**
     * @brief The constructor.
     *
     * Constructs the simple network object, creating the network hosts and the
     * controller.
     *
     * @param qubits The number of qubits for each host.
     * @param cbits The number of classical bits for each host.
     */
    SimpleDisconnectedNetwork(const std::vector<Types::qubit_t> &qubits = {}, const std::vector<size_t> &cbits = {})
    {
        configuration.SetConfiguration("trajectory_policy", "shot_v1");
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector});
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer});
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState});
        // this needs more work until it becomes useful - it's typically way too
        // slow to be timed
        // simulatorsForOptimizations.Insert({
        // SimulatorType::kQCSim,
        // SimulationType::kTensorNetwork });
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kCompositeQCSim, Simulators::SimulationType::kStatevector});

#ifndef NO_QISKIT_AER
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector});
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStabilizer});
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState});
        simulatorsForOptimizations.Insert({Simulators::SimulatorType::kCompositeQiskitAer, Simulators::SimulationType::kStatevector});
#endif

#ifdef __linux__
        { // GPU candidates are resolved lazily when simulation is requested.
            simulatorsForOptimizations.Insert({Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector});
            simulatorsForOptimizations.Insert({Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState});
            simulatorsForOptimizations.Insert({Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kTensorNetwork});
        }
#endif

        if (!qubits.empty())
            CreateNetwork(qubits, cbits);
    }

    /**
     * @brief Creates the network hosts and controller.
     *
     * Creates the network hosts and controller with the specified number of
     * qubits and classical bits.
     *
     * @param qubits The number of qubits for each host.
     * @param cbits The number of classical bits for each host.
     */
    void CreateNetwork(const std::vector<Types::qubit_t> &qubits, const std::vector<size_t> &cbits)
    {
        size_t qubitsOffset = 0;
        size_t cbitsOffset = 0;

        for (size_t i = 0; i < qubits.size(); ++i)
        {
            const size_t numQubits = qubits[i];
            const size_t numBits = (i < cbits.size() ? cbits[i] : 0);
            hosts.emplace_back(std::make_shared<SimpleHost<Time>>(i, qubitsOffset, numQubits, cbitsOffset, numBits));
            qubitsOffset += numQubits;
            cbitsOffset += numBits;
        }

        for (size_t i = 0; i < hosts.size(); ++i)
        {
            std::static_pointer_cast<SimpleHost<Time>>(hosts[i])->SetEntangledQubitId(qubitsOffset);
            std::static_pointer_cast<SimpleHost<Time>>(hosts[i])->SetEntangledQubitMeasurementBit(cbitsOffset);
            ++qubitsOffset;
            ++cbitsOffset;
        }

        controller = std::make_shared<Controller>();
    }

    /**
     * @brief Execute the circuit on the network.
     *
     * Execute the circuit on the network, using the controller for distributing
     * the operations to the hosts. Ensure the quantum computing simulator has
     * been created before calling this.
     *
     * @param circuit The circuit to execute.
     * @sa Circuits::Circuit
     */
    void Execute(const std::shared_ptr<Circuits::Circuit<Time>> &circuit) override
    {
        const auto recreate = recreateIfNeeded;

        auto simType = Simulators::SimulatorType::kQCSim;
        auto method = Simulators::SimulationType::kMatrixProductState;
        size_t numQubits = 2;
        if (simulator)
        {
            simType = simulator->GetType();
            method = simulator->GetSimulationType();
            numQubits = simulator->GetNumberOfQubits();
        }
        const size_t templateQubits = simulatorQubits;
        // Without a simulator the network does not simulate (a derived one may
        // only translate the circuit): executing must not create one.
        const bool hadSimulator = simulator != nullptr;

        // Execute continues from the caller's prepared quantum state. Repeated
        // shots and expectation queries still start from zero, even for one shot.
        const auto reset = resetBeforeExecution;
        recreateIfNeeded = false;
        resetBeforeExecution = false;

        ExecuteResults res;
        try
        {
            res = RepeatedExecute(circuit, 1);
        }
        catch (...)
        {
            recreateIfNeeded = recreate;
            resetBeforeExecution = reset;
            throw;
        }

        recreateIfNeeded = recreate;
        resetBeforeExecution = reset;

        // put the results in the state
        if (!res.empty())
        {
            const auto &first = *res.begin();
            GetState().SetResultsInOrder(first.first);
        }

        if (recreate && hadSimulator &&
            (!simulator ||
             (simulator && (simType != simulator->GetType() || method != simulator->GetSimulationType() || simulator->GetNumberOfQubits() != numQubits))))
            RecreateSimulatorKeepingResults(simType, method, templateQubits);
    }

    /**
     * @brief Execute the circuit on the specified host.
     *
     * Execute the circuit on the specified host.
     * The circuit must fit on the host, otherwise an exception is thrown.
     * The circuit will be mapped on the specified host, if its qubits start with
     * indexing from 0 (if already mapped, the qubits won't be altered).
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @sa Circuits::Circuit
     */
    void ExecuteOnHost(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId) override
    {
        const auto recreate = recreateIfNeeded;

        auto simType = Simulators::SimulatorType::kQCSim;
        auto method = Simulators::SimulationType::kMatrixProductState;
        size_t numQubits = 2;
        if (simulator)
        {
            simType = simulator->GetType();
            method = simulator->GetSimulationType();
            numQubits = simulator->GetNumberOfQubits();
        }
        const size_t templateQubits = simulatorQubits;
        // Without a simulator the network does not simulate (a derived one may
        // only translate the circuit): executing must not create one.
        const bool hadSimulator = simulator != nullptr;

        // RAII: restore recreateIfNeeded on any exit path (including exceptions).
        struct ScopedRestoreFlag
        {
            bool &flag, saved;

            ScopedRestoreFlag(bool &f) : flag(f), saved(f)
            {
                flag = false;
            }

            ~ScopedRestoreFlag()
            {
                flag = saved;
            }
        };

        ExecuteResults res;
        {
            ScopedRestoreFlag restoreGuard(recreateIfNeeded);
            res = RepeatedExecuteOnHost(circuit, hostId, 1);
        }

        // put the results in the state
        if (!res.empty())
        {
            const auto &first = *res.begin();
            GetState().SetResultsInOrder(first.first);
        }

        if (recreate && hadSimulator &&
            (!simulator ||
             (simulator && (simType != simulator->GetType() || method != simulator->GetSimulationType() || simulator->GetNumberOfQubits() != numQubits))))
            RecreateSimulatorKeepingResults(simType, method, templateQubits);
    }

    /**
     * @brief Execute the circuit on the network and return the expectation values
     * for the specified Pauli strings.
     *
     * Execute the circuit on the network, using the controller for distributing
     * the operations to the hosts and return the expectation values for the
     * specified Pauli strings. The base class functionality is used for circuit
     * distribution, but then the distributed circuit is converted to netqasm.
     * Ensure the quantum computing simulator and the netqasm virtual machines
     * have been created before calling this.
     *
     * @param circuit The circuit to execute.
     * @param paulis The Pauli strings to measure the expectations for.
     * @sa Circuits::Circuit
     */
    std::vector<double> ExecuteExpectations(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, const std::vector<std::string> &paulis) override
    {
        const auto recreate = recreateIfNeeded;

        auto simType = Simulators::SimulatorType::kQCSim;
        auto method = Simulators::SimulationType::kMatrixProductState;
        size_t numQubits = 2;
        if (simulator)
        {
            simType = simulator->GetType();
            method = simulator->GetSimulationType();
            numQubits = simulator->GetNumberOfQubits();
        }
        const size_t templateQubits = simulatorQubits;
        // Without a simulator the network does not simulate (a derived one may
        // only translate the circuit): executing must not create one.
        const bool hadSimulator = simulator != nullptr;

        const auto cleanCircuit = circuit ? circuit->RemoveEndMeasurements() : circuit;

        // RAII: restore recreateIfNeeded and clear pauliStrings on any exit path.
        // A pointer left behind would outlive the caller's vector, and the next
        // execution would read it as observables.
        struct ScopedRestore
        {
            bool &flag, saved;
            const std::vector<std::string> **ps;

            ScopedRestore(bool &f, const std::vector<std::string> **p) : flag(f), saved(f), ps(p)
            {
                flag = false;
            }

            ~ScopedRestore()
            {
                flag = saved;
                *ps = nullptr;
            }
        };

        ExecuteResults res;
        {
            ScopedRestore restoreGuard(recreateIfNeeded, &pauliStrings);
            pauliStrings = &paulis;
            res = RepeatedExecute(cleanCircuit, 1);
        }

        // put the results in the state
        if (!res.empty())
        {
            const auto &first = *res.begin();
            GetState().SetResultsInOrder(first.first);
        }

        std::vector<double> expectations(paulis.size());
        if (simulator)
        {
            // translate the pauli strings to the mapped order of qubits
            const size_t numOps = simulator->GetNumberOfQubits();

            auto optimiser = controller->GetOptimiser();
            if (optimiser)
            {
                // convert the classical state results back to the expected order
                const auto &qubitsMap = optimiser->GetQubitsMap();
                std::vector<std::string> translatedPaulis;
                translatedPaulis.reserve(paulis.size());

                for (size_t i = 0; i < paulis.size(); ++i)
                {
                    std::string translated(numOps, 'I');

                    for (size_t j = 0; j < paulis[i].size(); ++j)
                    {
                        const auto pos = qubitsMap.find(j);
                        if (pos != qubitsMap.end())
                            translated[pos->second] = paulis[i][j];
                        else
                            translated[j] = paulis[i][j];
                    }

                    translatedPaulis.push_back(std::move(translated));
                }
                expectations = simulator->ExpectationValues(translatedPaulis);
            }
            else
            {
                expectations = simulator->ExpectationValues(paulis);
            }
        }

        if (recreate && hadSimulator &&
            (!simulator || simType != simulator->GetType() || method != simulator->GetSimulationType() || simulator->GetNumberOfQubits() != numQubits))
            RecreateSimulatorKeepingResults(simType, method, templateQubits);

        return expectations;
    }

    /**
     * @brief Execute the circuit on the specified host and return the expectation
     * values for the specified Pauli strings.
     *
     * Execute the circuit on the specified host and return the expectation values
     * for the specified Pauli strings. The circuit must fit on the host,
     * otherwise an exception is thrown. The circuit will be mapped on the
     * specified host, if its qubits start with indexing from 0 (if already
     * mapped, the qubits won't be altered).
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @param paulis The Pauli strings to measure the expectations for.
     * @sa Circuits::Circuit
     */
    std::vector<double> ExecuteOnHostExpectations(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId,
                                                  const std::vector<std::string> &paulis) override
    {
        const auto recreate = recreateIfNeeded;

        auto simType = Simulators::SimulatorType::kQCSim;
        auto method = Simulators::SimulationType::kMatrixProductState;
        size_t numQubits = 2;
        if (simulator)
        {
            simType = simulator->GetType();
            method = simulator->GetSimulationType();
            numQubits = simulator->GetNumberOfQubits();
        }
        const size_t templateQubits = simulatorQubits;
        // Without a simulator the network does not simulate (a derived one may
        // only translate the circuit): executing must not create one.
        const bool hadSimulator = simulator != nullptr;

        // RAII: restore recreateIfNeeded and clear pauliStrings on any exit path.
        struct ScopedRestore
        {
            bool &flag, saved;
            const std::vector<std::string> **ps;

            ScopedRestore(bool &f, const std::vector<std::string> **p) : flag(f), saved(f), ps(p)
            {
                flag = false;
            }

            ~ScopedRestore()
            {
                flag = saved;
                *ps = nullptr;
            }
        } restoreGuard(recreateIfNeeded, &pauliStrings);

        const auto cleanCircuit = circuit ? circuit->RemoveEndMeasurements() : circuit;

        pauliStrings = &paulis;
        const auto res = RepeatedExecuteOnHost(cleanCircuit, hostId, 1);

        // put the results in the state
        if (!res.empty())
        {
            const auto &first = *res.begin();
            GetState().SetResultsInOrder(first.first);
        }

        // for (const auto& m : qubitsMapOnHost)
        //	std::cout << "Mapping qubit " << m.first << " to " << m.second <<
        // std::endl;

        const size_t offsetBase = qubitsMapOnHost.size();

        std::vector<double> expectations(paulis.size(), 1.);
        if (simulator)
        {
            // translate the pauli strings to the mapped order of qubits
            const size_t numOps = simulator->GetNumberOfQubits();

            // convert the pauli strings to the actual qubits order
            std::vector<std::string> translatedPaulis;
            translatedPaulis.reserve(paulis.size());
            for (size_t i = 0; i < paulis.size(); ++i)
            {
                std::string translated(std::max(numOps, paulis[i].size()), 'I');

                size_t offset = offsetBase;

                for (size_t j = 0; j < paulis[i].size(); ++j)
                {
                    auto pos = qubitsMapOnHost.find(j);
                    if (pos != qubitsMapOnHost.end())
                        translated[pos->second] = paulis[i][j];
                    else
                    {
                        translated[offset] = paulis[i][j];
                        ++offset;
                    }
                }

                // std::cout << "Translated pauli string: " << translated << std::endl;

                translatedPaulis.push_back(std::move(translated));
            }
            expectations = simulator->ExpectationValues(translatedPaulis);
        }
        else
        {
            throw std::runtime_error("ExecuteOnHostExpectations: no simulator available after execution.");
        }

        if (recreate && hadSimulator &&
            (!simulator || simType != simulator->GetType() || method != simulator->GetSimulationType() || simulator->GetNumberOfQubits() != numQubits))
            RecreateSimulatorKeepingResults(simType, method, templateQubits);

        return expectations;
    }

    /**
     * @brief Execute circuit on host and return full statevector amplitudes.
     *
     * Execute the circuit on the specified host and return the complex amplitudes
     * of the resulting quantum state (one per basis state 0..2^n-1).
     * Retained qubits are ordered by ascending original index, with gaps compacted.
     * Supported for Statevector and MPS simulation types.
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @return A vector of complex<double> amplitudes of length 2^n.
     */
    std::vector<std::complex<double>> ExecuteOnHostAmplitudes(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId) override
    {
        return ExecuteOnHostBasisValues(circuit, hostId, "ExecuteOnHostAmplitudes", [](Simulators::ISimulator &state) { return state.GetStateVector(); });
    }

    /**
     * @brief Execute circuit on host and return the probability of each basis
     * state.
     *
     * Execute the circuit on the specified host and return the probabilities of
     * the basis states (one per basis state 0..2^n-1, ordered as the amplitudes
     * of ExecuteOnHostAmplitudes). They exist for every state, so every
     * simulation type answers: the mixed ones and the ones without amplitudes
     * too. End measurements are removed first: they would collapse the state.
     * A qubit that only they use keeps its place among the retained qubits,
     * in |0>, as it has in the amplitudes of the measured circuit.
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @return A vector of probabilities of length 2^n.
     */
    std::vector<double> ExecuteOnHostProbabilities(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId) override
    {
        const auto read = [](Simulators::ISimulator &state) { return Simulators::BasisProbabilities(state); };
        if (!circuit)
            return ExecuteOnHostBasisValues(circuit, hostId, "ExecuteOnHostProbabilities", read);
        const auto executed = circuit->RemoveEndMeasurements();
        auto values = ExecuteOnHostBasisValues(executed, hostId, "ExecuteOnHostProbabilities", read);

        // The register that ran holds the qubits the executed circuit uses. One
        // that only the removed measurements used was never touched: it is in
        // |0>, so it takes its place back with no probability where it is 1.
        const auto used = circuit->GetQubits(), kept = executed->GetQubits();
        const size_t ran = std::max<size_t>(1, kept.size());
        if (used.size() == kept.size() || used.size() >= std::numeric_limits<size_t>::digits || values.size() != (size_t{1} << ran))
            return values;
        std::vector<size_t> place; // of each qubit that ran, among the circuit's qubits
        size_t position = 0;
        for (const auto qubit : used)
        {
            if (kept.count(qubit))
                place.push_back(position);
            ++position;
        }
        std::vector<double> all(size_t{1} << used.size(), 0.0);
        for (size_t state = 0; state < (size_t{1} << place.size()); ++state)
        {
            size_t index = 0;
            for (size_t bit = 0; bit < place.size(); ++bit)
                if (state & (size_t{1} << bit))
                    index |= size_t{1} << place[bit];
            all[index] = values[state];
        }
        return all;
    }

    /**
     * @brief Execute circuit on host and read one value per basis state.
     *
     * The execution, register restoration and qubit ordering shared by
     * ExecuteOnHostAmplitudes and ExecuteOnHostProbabilities.
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @param caller The name of the calling function, for its error message.
     * @param read Returns the 2^n values of the executed simulator.
     * @return The values, with the retained qubits in ascending original order.
     */
    template <typename Read>
    auto ExecuteOnHostBasisValues(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId, const char *caller, Read &&read)
        -> decltype(read(std::declval<Simulators::ISimulator &>()))
    {
        const auto recreate = recreateIfNeeded;

        auto simType = Simulators::SimulatorType::kQCSim;
        auto method = Simulators::SimulationType::kMatrixProductState;
        size_t numQubits = 2;
        if (simulator)
        {
            simType = simulator->GetType();
            method = simulator->GetSimulationType();
            numQubits = simulator->GetNumberOfQubits();
        }
        const size_t templateQubits = simulatorQubits;
        // Without a simulator the network does not simulate (a derived one may
        // only translate the circuit): executing must not create one.
        const bool hadSimulator = simulator != nullptr;

        // RAII: restore recreateIfNeeded on any exit path (including exceptions).
        struct ScopedRestoreFlag
        {
            bool &flag, saved;

            ScopedRestoreFlag(bool &f) : flag(f), saved(f)
            {
                flag = false;
            }

            ~ScopedRestoreFlag()
            {
                flag = saved;
            }
        } restoreGuard(recreateIfNeeded);

        const auto res = RepeatedExecuteOnHost(circuit, hostId, 1);

        if (!res.empty())
        {
            const auto &first = *res.begin();
            GetState().SetResultsInOrder(first.first);
        }

        if (!simulator)
            throw std::runtime_error(std::string(caller) + ": no simulator available after execution.");

        const size_t n = simulator->GetNumberOfQubits();
        auto values = read(*simulator);
        const size_t dim = values.size();

        // Restore ascending original-qubit order within the compact register.
        // Original indices may be sparse, so use their ranks as output positions.
        if (!qubitsMapOnHost.empty())
        {
            // Build reverse mapping: simulator qubit position -> rank of its
            // original qubit among the mapped ones.
            std::vector<std::pair<Types::qubit_t, Types::qubit_t>> byOriginal(qubitsMapOnHost.begin(), qubitsMapOnHost.end());
            std::sort(byOriginal.begin(), byOriginal.end());

            std::vector<size_t> simToOrig(n);
            for (size_t pos = 0; pos < byOriginal.size(); ++pos)
                simToOrig[byOriginal[pos].second] = pos;

            decltype(values) remapped(dim);

            for (size_t sim_state = 0; sim_state < dim; ++sim_state)
            {
                size_t orig_state = 0;
                for (size_t qbit = 0; qbit < n; ++qbit)
                {
                    if (sim_state & (1ULL << qbit))
                        orig_state |= (1ULL << simToOrig[qbit]);
                }
                if (orig_state < dim)
                    remapped[orig_state] = values[sim_state];
            }
            values.swap(remapped);
        }

        if (recreate && hadSimulator &&
            (!simulator || simType != simulator->GetType() || method != simulator->GetSimulationType() || simulator->GetNumberOfQubits() != numQubits))
            RecreateSimulatorKeepingResults(simType, method, templateQubits);

        return values;
    }

    /**
     * @brief Execute circuit on host and return the projection onto the zero
     * state.
     *
     * Execute the circuit on the specified host and return <0|Psi>, the inner
     * product of the resulting quantum state with the all-zeros basis state.
     * This is equivalent to Amplitude(0) but can be faster for certain simulator
     * back-ends (e.g. MPS).
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @return The complex amplitude <0|Psi>.
     */
    std::complex<double> ExecuteOnHostProjectOnZero(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId) override
    {
        const auto recreate = recreateIfNeeded;

        auto simType = Simulators::SimulatorType::kQCSim;
        auto method = Simulators::SimulationType::kMatrixProductState;
        size_t numQubits = 2;
        if (simulator)
        {
            simType = simulator->GetType();
            method = simulator->GetSimulationType();
            numQubits = simulator->GetNumberOfQubits();
        }
        const size_t templateQubits = simulatorQubits;
        // Without a simulator the network does not simulate (a derived one may
        // only translate the circuit): executing must not create one.
        const bool hadSimulator = simulator != nullptr;

        // RAII: restore recreateIfNeeded on any exit path (including exceptions).
        struct ScopedRestoreFlag
        {
            bool &flag, saved;

            ScopedRestoreFlag(bool &f) : flag(f), saved(f)
            {
                flag = false;
            }

            ~ScopedRestoreFlag()
            {
                flag = saved;
            }
        } restoreGuard(recreateIfNeeded);

        const auto res = RepeatedExecuteOnHost(circuit, hostId, 1);

        if (!res.empty())
        {
            const auto &first = *res.begin();
            GetState().SetResultsInOrder(first.first);
        }

        if (!simulator)
            throw std::runtime_error("ExecuteOnHostProjectOnZero: no simulator available after "
                                     "execution.");

        const std::complex<double> result = simulator->ProjectOnZero();

        if (recreate && hadSimulator &&
            (!simulator || simType != simulator->GetType() || method != simulator->GetSimulationType() || simulator->GetNumberOfQubits() != numQubits))
            RecreateSimulatorKeepingResults(simType, method, templateQubits);

        return result;
    }

    /**
     * @brief Execute the circuit on the network, repeatedly.
     *
     * Execute the circuit on the network, distributing the operations to the
     * hosts, repeating the execution 'shots' times. The way the circuit is
     * distributed to the hosts depends on the specific interface implementations.
     *
     * @param circuit The circuit to execute.
     * @param shots The number of times to repeat the execution.
     * @return A map with the results of the execution, where the key is the qubit
     * id and the value is the number of times the qubit was measured to be 1.
     * @sa Circuits::Circuit
     */
    ExecuteResults RepeatedExecute(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t shots = 1000) override
    {
        if (!controller || !circuit)
            return {};

        distCirc = controller->DistributeCircuit(BaseClass::getptr(), circuit);
        if (!distCirc)
            return {};

#ifdef _DEBUG
        for (auto q : distCirc->AffectedQubits())
        {
            if (q >= GetNumQubits())
            {
                std::cout << "This is a distributed circuit, using entanglement or cutting" << std::endl;
                break;
            }
        }
#endif

        // Without shots nothing runs: the network's simulator keeps its
        // register and its state.
        if (!simulator || shots == 0)
            return {};
        InlineTerminalComposites();

        auto simType = simulator->GetType();
        if (distCirc->HasOpsAfterMeasurements() && (
#ifndef NO_QISKIT_AER
                                                       simType == Simulators::SimulatorType::kCompositeQiskitAer ||
#endif
                                                       simType == Simulators::SimulatorType::kCompositeQCSim))
            distCirc->MoveMeasurementsAndResets();

        auto method = simulator->GetSimulationType();

        const auto saveSimType = simType;
        const auto saveMethod = method;
        const size_t saveQubits = simulatorQubits;

        if (GetOptimizeSimulator() && distCirc->IsClifford() && method != Simulators::SimulationType::kStabilizer
        // this is for the gpu simulator, as it doesn't support stabilizer
#ifdef __linux__
            && !Simulators::IsGpuSimulator(simType)
#endif
        )
        {
            // Only to a stabilizer backend the candidate set lists.
            if (const auto stabilizer = StabilizerBackend(simType))
            {
                method = Simulators::SimulationType::kStabilizer;
                simType = *stabilizer;
            }
        }

        ExecuteResults res;
        const size_t nrQubits = GetNumQubits() + GetNumNetworkEntangledQubits();
        const size_t nrCbitsResults = GetNumClassicalBits();
        const size_t nrCbits = nrCbitsResults + GetNumNetworkEntangledQubits();

        // From here a failure (backend selection, the shots) leaves the
        // network's simulator with the backend and width it had.
        const SimulatorRestoreOnFailure restoreOnFailure(*this, saveSimType, saveMethod, saveQubits);

        CaptureSimulatorConfiguration();

        // do that only if the optimization for simulator is on and the estimator is
        // available, ortherwise an 'optimal' simulator won't be created
        if (optimizeSimulator && simulatorsEstimator && simulatorsEstimator->IsInitialized())
        {
            simulator->Clear();
            GetState().Clear();
        }

        curMaxBondDim = 0;

        std::vector<bool> executed;
        auto optSim = ChooseBestSimulator(distCirc, shots, nrQubits, nrCbits, nrCbitsResults, simType, method, executed);

        lastSimulatorType = simType;
        lastMethod = method;
        lastGpuDevice = -1;
        lastFusionWidth = 0;
        lastFusionEnabled = false;
        lastDistributedPlacement.clear();

        const bool reproducibleSampling = UsesReproducibleSampling(simType, method);
        size_t measuredWidth;
        const bool terminalSampling = !distCirc->HasOpsAfterMeasurementsWithWidth(Simulators::EvolvesExactQuantumChannels(simType, method), measuredWidth);
        const size_t nrThreads =
            ExecutionThreads(distCirc, nrQubits, nrCbits, nrCbitsResults, shots, simType, method, terminalSampling, measuredWidth, optSim, executed);

        std::mutex resultsMutex;

        auto dcirc = distCirc;

        const bool logicalTerminal = terminalSampling && shots > 1 && !UsesFullTerminalBatch(simType, method, shots, measuredWidth);
        if (reproducibleSampling &&
            (logicalTerminal || (!terminalSampling && (shots > 1 || (shots == 1 && configuration.GetConfiguration("trajectory_policy") == "shot_v1")))))
        {
            if (logicalTerminal && !optSim)
                optSim = PrepareTerminalState(dcirc, nrQubits, nrCbits, nrCbitsResults, simType, method, executed,
                                              nrThreads == 1 && simulator && method == saveMethod && simType == saveSimType ? simulator : nullptr);
            ExecuteLogicalShotBlocks(dcirc, res, shots, nrThreads, nrQubits, nrCbits, nrCbitsResults, simType, method, optSim, executed, resultsMutex,
                                     logicalTerminal);
        }
        else if (nrThreads > 1)
        {
            // Clones are owned by outer shot workers; avoid constructing inner pools.
            if (optSim && method == Simulators::SimulationType::kPauliPropagator && optSim->GetType() == Simulators::SimulatorType::kQCSim)
                optSim->SetMultithreading(false);
            // since it's going to execute on multiple threads, free the memory from
            // the network's simulator and state, it's going to use other ones,
            // created in the threads if optimization already exists, it will be
            // cloned in the threads, otherwise a new one will be created in the
            // threads
            if (!optimizeSimulator || !simulatorsEstimator || !simulatorsEstimator->IsInitialized()) // otherwise it was already cleared
            {
                simulator->Clear();
                GetState().Clear();
            }

            const size_t cntPerThread = std::max<size_t>(shots / nrThreads, 1ULL);

            threadsPool.Resize(nrThreads);
            threadsPool.SetFinishLimit(shots);

            uint64_t jobStream = 0;
            while (shots > 0)
            {
                const size_t curCnt = std::min(cntPerThread, shots);

                shots -= curCnt;

                auto job = std::make_shared<ExecuteJob<Time>>(dcirc, res, curCnt, nrQubits, nrCbits, nrCbitsResults, simType, method, resultsMutex);
                job->optimiseMultipleShotsExecution = curCnt > 1 || GetOptimizeSimulator();

                job->network = BaseClass::getptr();
                job->onSimulatorReady = [this](const Simulators::ISimulator &actual) {
                    lastFusionWidth.store(actual.GetGateFusionMaxQubits());
                    lastFusionEnabled.store(actual.IsGateFusionEnabled());
                };
                job->curMaxBondDim = &curMaxBondDim;

                job->config = ExecutionConfiguration(simType, nrQubits, method);
                job->randomStream = jobStream++;
                if (configuration.IsSet("seed"))
                {
                    const uint64_t childSeed = Simulators::IState::DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), job->randomStream);
                    job->config.SetConfiguration("seed", std::to_string(childSeed));
                }

                if (optSim)
                {
                    job->optSim = optSim->Clone();
                    job->executedGates = executed;
                    if (job->config.IsSet("seed"))
                        job->optSim->SetSeed(std::stoull(job->config.GetConfiguration("seed")));
                }

                threadsPool.AddRunJob(std::move(job));
            }

            // The workers stay for the next execution: creating them costs
            // more than a small batch of shots.
            threadsPool.WaitForFinish();
        }
        else
        {
            const size_t curCnt = shots;

            auto job = std::make_shared<ExecuteJob<Time>>(dcirc, res, curCnt, nrQubits, nrCbits, nrCbitsResults, simType, method, resultsMutex);
            job->optimiseMultipleShotsExecution = curCnt > 1 || GetOptimizeSimulator();

            job->network = BaseClass::getptr();
            job->onSimulatorReady = [this](const Simulators::ISimulator &actual) {
                lastFusionWidth.store(actual.GetGateFusionMaxQubits());
                lastFusionEnabled.store(actual.IsGateFusionEnabled());
            };
            job->curMaxBondDim = &curMaxBondDim;

            job->config = ExecutionConfiguration(simType, nrQubits, method);

            if (optSim)
            {
                optSim->SetMultithreading(true);
                job->optSim = optSim;
                job->executedGates = executed;
            }
            else
            {
                if (simulator && method == saveMethod && simType == saveSimType)
                {
                    // use the already created simulator
                    optSim = simulator;
                    job->optSim = optSim;
                    if (optSim->GetNumberOfQubits() == nrQubits)
                    {
                        if (resetBeforeExecution)
                        {
                            optSim->Reset();
                            optSim->SetGatesCounter(0);
                        }
                        job->config.ApplyConfigurationToSimulator(optSim);
                        // An initial mapping is only valid for a freshly reset register.
                        if (resetBeforeExecution)
                            OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);
                    }
                    job->executedGates.resize(dcirc->size(),
                                              false); // no gates executed yet
                    simulator = nullptr;
                }
            }

            // The job may hold the network's simulator (detached above); a
            // failure re-creates it (SimulatorRestoreOnFailure).
            job->DoWorkNoLock();
            lastGpuDevice = job->optSim ? job->optSim->GetGpuDevice() : -1;
            RecordDistributedPlacement(job->optSim);
            if (!recreateIfNeeded)
                simulator = job->optSim;
        }

        if (recreateIfNeeded)
            CreateSimulator(saveSimType, saveMethod, saveQubits);

        ConvertBackResults(res);

        return res;
    }

    /**
     * @brief Execute the circuit on the specified host, repeatedly.
     *
     * Execute the circuit on the specified host, repeating the execution 'shots'
     * times. The circuit must fit on the host, otherwise an exception is thrown.
     * The circuit will be mapped on the specified host, if its qubits start with
     * indexing from 0 (if already mapped, the qubits won't be altered).
     *
     * @param circuit The circuit to execute.
     * @param hostId The id of the host to execute the circuit on.
     * @param shots The number of times to repeat the execution.
     * @return A map with the results of the execution, where the key is the qubit
     * id and the value is the number of times the qubit was measured to be 1.
     * @sa Circuits::Circuit
     */
    ExecuteResults RepeatedExecuteOnHost(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId, size_t shots = 1000) override
    {
        if (!circuit || hostId >= GetNumHosts())
            return {};

        size_t nrQubits = 0;
        size_t nrCbits = 0;

        const bool distributed = simulator && Simulators::IsDistributedGpuSimulator(simulator->GetType());
        auto mappingCircuit = circuit;
        if (!distributed && GetController()->GetOptimizeCircuit())
        {
            mappingCircuit = std::static_pointer_cast<Circuits::Circuit<Time>>(circuit->Clone());
            mappingCircuit->Optimize();
        }
        const auto reverseQubitsMap = MapCircuitOnHost(mappingCircuit, hostId, nrQubits, nrCbits, true);
        // Resolve indexing before optimization can remove a wire that disambiguates
        // local from global numbering (for example a cancelling pair on qubit 0).
        if (distributed && distCirc && GetController()->GetOptimizeCircuit())
            distCirc->Optimize();
        if (nrCbits == 0)
            nrCbits = nrQubits;

        // Without shots nothing runs: the network's simulator keeps its
        // register and its state.
        if (!simulator || !distCirc || shots == 0)
            return {};
        InlineTerminalComposites();

        auto simType = simulator->GetType();

        CaptureSimulatorConfiguration();

        if (distCirc->HasOpsAfterMeasurements() && (
#ifndef NO_QISKIT_AER
                                                       simType == Simulators::SimulatorType::kCompositeQiskitAer ||
#endif
                                                       simType == Simulators::SimulatorType::kCompositeQCSim))
            distCirc->MoveMeasurementsAndResets();

        auto method = simulator->GetSimulationType();
        const auto saveSimType = simType;
        const auto saveMethod = method;
        const size_t saveQubits = simulatorQubits;

        if (GetOptimizeSimulator() && distCirc->IsClifford() && method != Simulators::SimulationType::kStabilizer
        // this is for the gpu simulator, as it doesn't support stabilizer
#ifdef __linux__
            && !Simulators::IsGpuSimulator(simType)
#endif
        )
        {
            // Only to a stabilizer backend the candidate set lists.
            if (const auto stabilizer = StabilizerBackend(simType))
            {
                method = Simulators::SimulationType::kStabilizer;
                simType = *stabilizer;
            }
        }

        ExecuteResults res;

        // From here a failure (backend selection, the shots) leaves the
        // network's simulator with the backend and width it had.
        const SimulatorRestoreOnFailure restoreOnFailure(*this, saveSimType, saveMethod, saveQubits);

        // Backend selection can allocate a replacement. Fixed single-worker jobs
        // retain the existing simulator; release it below if workers are needed.
        if (GetOptimizeSimulator() && !distributed)
            simulator->Clear();
        GetState().Clear();

        curMaxBondDim = 0;

        std::vector<bool> executed;
        auto optSim = ChooseBestSimulator(distCirc, shots, nrQubits, nrCbits, nrCbits, simType, method, executed);

        lastSimulatorType = simType;
        lastMethod = method;
        lastGpuDevice = -1;
        lastFusionWidth = 0;
        lastFusionEnabled = false;
        lastDistributedPlacement.clear();

        const bool reproducibleSampling = UsesReproducibleSampling(simType, method);
        size_t measuredWidth;
        const bool terminalSampling = !distCirc->HasOpsAfterMeasurementsWithWidth(Simulators::EvolvesExactQuantumChannels(simType, method), measuredWidth);
        // WARNING: be sure to not put this above ChooseBestSimulator, as that one
        // can change the shots value!
        const size_t nrThreads =
            ExecutionThreads(distCirc, nrQubits, nrCbits, nrCbits, shots, simType, method, terminalSampling, measuredWidth, optSim, executed);

        std::mutex resultsMutex;

        auto dcirc = distCirc;

        const bool logicalTerminal = terminalSampling && shots > 1 && !UsesFullTerminalBatch(simType, method, shots, measuredWidth);
        if (reproducibleSampling &&
            (logicalTerminal || (!terminalSampling && (shots > 1 || (shots == 1 && configuration.GetConfiguration("trajectory_policy") == "shot_v1")))))
        {
            if (logicalTerminal && !optSim)
                optSim = PrepareTerminalState(dcirc, nrQubits, nrCbits, nrCbits, simType, method, executed,
                                              nrThreads == 1 && simulator && method == saveMethod && simType == saveSimType ? simulator : nullptr);
            ExecuteLogicalShotBlocks(dcirc, res, shots, nrThreads, nrQubits, nrCbits, nrCbits, simType, method, optSim, executed, resultsMutex,
                                     logicalTerminal);
        }
        else if (nrThreads > 1)
        {
            // Clones are owned by outer shot workers; avoid constructing inner pools.
            if (optSim && method == Simulators::SimulationType::kPauliPropagator && optSim->GetType() == Simulators::SimulatorType::kQCSim)
                optSim->SetMultithreading(false);
            simulator->Clear();
            // this rounds up, rounding down is better
            // const size_t cntPerThread = static_cast<size_t>((shots - 1) / nrThreads
            // + 1);
            const size_t cntPerThread = std::max<size_t>(shots / nrThreads, 1ULL);

            threadsPool.Resize(nrThreads);
            threadsPool.SetFinishLimit(shots);

            uint64_t jobStream = 0;
            while (shots > 0)
            {
                const size_t curCnt = std::min(cntPerThread, shots);
                shots -= curCnt;

                auto job = std::make_shared<ExecuteJob<Time>>(dcirc, res, curCnt, nrQubits, nrCbits, nrCbits, simType, method, resultsMutex);
                job->optimiseMultipleShotsExecution = curCnt > 1 || GetOptimizeSimulator();

                job->network = BaseClass::getptr();
                job->onSimulatorReady = [this](const Simulators::ISimulator &actual) {
                    lastFusionWidth.store(actual.GetGateFusionMaxQubits());
                    lastFusionEnabled.store(actual.IsGateFusionEnabled());
                };
                job->curMaxBondDim = &curMaxBondDim;

                job->config = ExecutionConfiguration(simType, nrQubits, method);
                job->randomStream = jobStream++;
                if (configuration.IsSet("seed"))
                {
                    const uint64_t childSeed = Simulators::IState::DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), job->randomStream);
                    job->config.SetConfiguration("seed", std::to_string(childSeed));
                }

                if (optSim)
                {
                    job->optSim = optSim->Clone();
                    job->executedGates = executed;
                    if (job->config.IsSet("seed"))
                        job->optSim->SetSeed(std::stoull(job->config.GetConfiguration("seed")));
                }

                threadsPool.AddRunJob(std::move(job));
            }

            // The workers stay for the next execution: creating them costs
            // more than a small batch of shots.
            threadsPool.WaitForFinish();
        }
        else
        {
            const size_t curCnt = shots;

            auto job = std::make_shared<ExecuteJob<Time>>(dcirc, res, curCnt, nrQubits, nrCbits, nrCbits, simType, method, resultsMutex);
            job->optimiseMultipleShotsExecution = curCnt > 1 || GetOptimizeSimulator();

            job->network = BaseClass::getptr();
            job->onSimulatorReady = [this](const Simulators::ISimulator &actual) {
                lastFusionWidth.store(actual.GetGateFusionMaxQubits());
                lastFusionEnabled.store(actual.IsGateFusionEnabled());
            };
            job->curMaxBondDim = &curMaxBondDim;

            job->config = ExecutionConfiguration(simType, nrQubits, method);

            if (optSim)
            {
                optSim->SetMultithreading(true);
                job->optSim = optSim;
                job->executedGates = executed;
            }
            else if (simulator && method == saveMethod && simType == saveSimType)
            {
                optSim = simulator;
                job->optSim = optSim;
                if (optSim->GetNumberOfQubits() == nrQubits)
                {
                    // Host execution always starts from zero, even when the previous
                    // invocation retained its state for an expectation/state query.
                    optSim->Reset();
                    optSim->SetGatesCounter(0);
                    job->config.ApplyConfigurationToSimulator(optSim);
                    OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);
                }
                // DoWorkNoLock reallocates/reinitializes when the host width changed.
                job->executedGates.resize(dcirc->size(), false);
            }

            job->DoWorkNoLock();
            lastGpuDevice = job->optSim ? job->optSim->GetGpuDevice() : -1;
            RecordDistributedPlacement(job->optSim);
            if (!recreateIfNeeded)
                simulator = job->optSim;
        }

        if (recreateIfNeeded)
            CreateSimulator(saveSimType, saveMethod, saveQubits);

        if (!reverseQubitsMap.empty())
            ConvertBackResults(res, reverseQubitsMap);

        return res;
    }

    /**
     * @brief Get the number of gates that span more than one host.
     *
     * Get the number of gates that span more than one host for the given circuit.
     *
     * @param circuit The circuit to check.
     * @return The number of gates that need distribution or cutting.
     */
    size_t GetNumberOfGatesDistributedOrCut(const std::shared_ptr<Circuits::Circuit<Time>> &circuit) const override
    {
        if (!circuit)
            return 0;

        size_t distgates = 0;

        for (const auto &op : circuit->GetOperations())
            if (!IsLocalOperation(op))
                ++distgates;

        return distgates;
    }

    /**
     * @brief Schedule and execute circuits on the network.
     *
     * Execute the circuits on the network, scheduling their execution and
     * distributing the operations to the hosts. The way the circuits are
     * distributed to the hosts depends on the specific interface implementations.
     * The way they are scheduled depends on the network scheduler and
     * parametrization.
     *
     * @param circuits The circuits to execute, along with the number of shots.
     * @return A vector of maps with the results of each circuit execution, where
     * the key is the state as a vector of bools and the value is the number of
     * times it was measured.
     * @sa Circuits::Circuit
     * @sa ExecuteCircuit
     */
    std::vector<ExecuteResults> ExecuteScheduled(const std::vector<Schedulers::ExecuteCircuit<Time>> &circuits) override
    {
        // create a default one if not set
        if (!GetScheduler())
        {
            CreateScheduler();

            if (!GetScheduler())
                return {};
        }

        return GetScheduler()->ExecuteScheduled(circuits);
    }

    /**
     * @brief Create the simulator for the network.
     *
     * See INetwork::CreateSimulator: the simulator is mainly a backend
     * template, which execution may clear, replace or re-create. Recreating it
     * after an execution, Clone() and RemoveConfiguration() keep nrQubits.
     *
     * @param simType The type of the simulator to create.
     * @param simExecType The type of the simulation - statevector, composite,
     * matrix product state, stabilizer, tensor network...
     * @param nrQubits The number of qubits to allocate. 0 (default) allocates
     * the whole network, including each host's entanglement qubit.
     * @sa Simulators::SimulatorType
     */
    void CreateSimulator(Simulators::SimulatorType simType = Simulators::SimulatorType::kQCSim,
                         Simulators::SimulationType simExecType = Simulators::SimulationType::kMatrixProductState, size_t nrQubits = 0) override
    {
        simulatorQubits = nrQubits;
        classicalState.Clear();
        classicalState.AllocateBits(GetNumClassicalBits() + GetNumNetworkEntangledQubits());

        simulator = Simulators::SimulatorsFactory::CreateSimulator(simType, simExecType);

        if (simulator)
        {
            const size_t allocationQubits = nrQubits == 0 ? GetNumQubits() + GetNumNetworkEntangledQubits() : nrQubits;
            ExecutionConfiguration(simType, allocationQubits, simExecType).ApplyConfigurationToSimulator(simulator);

            simulator->AllocateQubits(allocationQubits);
            simulator->Initialize();

            // Pin the resolved default as well as explicit selections. Cloning or
            // recreating this network must not follow later process-default changes.
            if (simType == Simulators::SimulatorType::kGpuSim)
                configuration.SetConfiguration("gpu_device", std::to_string(simulator->GetGpuDevice()));

            if (Simulators::IsDistributedGpuSimulator(simType) && nrQubits == 0)
            {
                resolvedDistributedDevices[simType] = simulator->GetConfiguration("distributed_shard_devices");
            }
            simulator->setGrowthFactorGate(growthFactorGate);
            simulator->setGrowthFactorSwap(growthFactorSwap);
            simulator->SetLookaheadDepth(lookaheadDepth);
            simulator->SetLookaheadDepthWithHeuristic(lookaheadDepthWithHeuristic);
        }
    }

    /**
     * @brief Configures the network.
     *
     * This function is called to configure the network (for example the
     * simulator(s) used by the network.
     *
     * @param key The key of the configuration option.
     * @param value The value of the configuration.
     */
    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;

        if (std::string("trajectory_policy") == key)
        {
            const std::string policy(value);
            if (policy != "block_v1" && policy != "shot_v1")
                throw std::invalid_argument("trajectory_policy must be block_v1 or shot_v1");
        }
        if (std::string("counts_policy") == key)
        {
            const std::string policy(value);
            if (policy != "auto" && policy != "legacy" && policy != "frame_v1" && policy != "multinomial_v1")
                throw std::invalid_argument("counts_policy must be auto, legacy, frame_v1 or multinomial_v1");
        }
        if (std::string("trajectory_max_memory_mb") == key)
        {
            size_t mb = 0;
            const std::string_view text(value);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), mb);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !mb || mb > (std::numeric_limits<size_t>::max() >> 20))
                throw std::invalid_argument("trajectory_max_memory_mb must be a positive representable MiB count");
        }

        if (std::string("gate_fusion") == key)
        {
            const std::string setting(value);
            if (setting != "true" && setting != "false" && setting != "0" && setting != "1" && setting != "auto")
                throw std::invalid_argument("gate_fusion must be true, false, 1, 0 or auto");
            configuration.SetConfiguration(key, value);
            if (simulator && (simulator->GetType() == Simulators::SimulatorType::kQCSim || simulator->GetType() == Simulators::SimulatorType::kCompositeQCSim ||
                              Simulators::IsGpuSimulator(simulator->GetType())))
                simulator->Configure(key, value);
            return;
        }
        if (std::string("distributed_host_qubit_indexing") == key)
        {
            const std::string mode(value);
            if (mode != "auto" && mode != "local" && mode != "global")
                throw std::invalid_argument("distributed_host_qubit_indexing must be auto, local or global");
            distributedHostQubitIndexing = mode;
            return;
        }

        if (std::string("distributed_devices") == key || std::string("gpu_device") == key)
            resolvedDistributedDevices.clear();

        if (std::string("max_simulators") == key)
            maxSimulators = std::stoull(value);

        if (std::string("gpu_device") == key)
        {
            Simulators::Configuration::ParseGpuDevice(value);
            if (simulator)
                simulator->Configure(key, value);
            configuration.SetConfiguration(key, value);
            return;
        }
        if (!Simulators::Configuration::GpuSvdSettingGroup(key).empty())
        {
            if (simulator)
                simulator->Configure(key, value);
            configuration.SetConfiguration(key, value);
            return;
        }
        configuration.SetConfiguration(key, value);

        if (simulator)
            simulator->Configure(key, value);
    }

    void RemoveConfiguration(const char *key) override
    {
        if (!key)
            return;

        // Settings made directly on the simulator survive the re-creation: it
        // gets them from the network's configuration.
        configuration = EffectiveConfiguration();
        configuration.RemoveConfiguration(key);
        if (simulator && simulator->GetConfigMap().count(key))
            CreateSimulator(simulator->GetType(), simulator->GetSimulationType(), simulatorQubits);
    }

    void CaptureSimulatorSettings() override
    {
        CaptureSimulatorConfiguration();
    }

    /**
     * @brief Get the simulator for the network.
     *
     * Get the simulator for the network.
     *
     * @return The simulator for the network.
     * @sa Simulators::ISimulator
     */
    std::shared_ptr<Simulators::ISimulator> GetSimulator() const override
    {
        return simulator;
    }

    size_t GetSimulatorQubits() const override
    {
        return simulatorQubits;
    }

    /**
     * @brief Get the classical state of the network.
     *
     * Gets a reference to the classical state of the network.
     *
     * @return The classical state of the network.
     * @sa Circuits::ClassicalState
     */
    Circuits::OperationState &GetState() override
    {
        return classicalState;
    }

    /**
     * @brief Create the scheduler for the network.
     *
     * Creates the scheduler for the network.
     * Call this only after the network topology has been set up.
     * Should create the scheduler and set the network for it and any other
     * necessary parameters.
     *
     * @param simType The type of the scheduler to create.
     * @sa SchedulerType
     */
    void CreateScheduler(SchedulerType schType = SchedulerType::kNoEntanglementQubitsParallel) override
    {
        if (!controller)
            return;

        controller->CreateScheduler(BaseClass::getptr(), schType);
    }

    /**
     * @brief Get the scheduler for the network.
     *
     * Get the scheduler for the network.
     *
     * @return The scheduler for the network.
     * @sa Schedulers::IScheduler
     */
    std::shared_ptr<Schedulers::IScheduler<Time>> GetScheduler() const override
    {
        if (!controller)
            return nullptr;

        return controller->GetScheduler();
    }

    /**
     * @brief Get the host with the specified id.
     *
     * Get a smart pointer to the host that has the specified id.
     *
     * @param hostId The id of the host to get.
     * @return A smart pointer to the host that has the specified id.
     * @sa IHost
     */
    const std::shared_ptr<IHost<Time>> GetHost(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return nullptr;

        return hosts[hostId];
    }

    /**
     * @brief Get the controller for the network.
     *
     * Gets a smart pointer to the controller for the network.
     *
     * @return The controller for the network.
     * @sa IController
     */
    const std::shared_ptr<IController<Time>> GetController() const override
    {
        return controller;
    }

    /**
     * @brief Get the number of hosts in the network.
     *
     * Get the number of hosts in the network, excluding the controller.
     *
     * @return The number of hosts in the network.
     */
    size_t GetNumHosts() const override
    {
        return hosts.size();
    }

    /**
     * @brief Get the number of qubits in the network.
     *
     * Get the number of qubits in the network, excluding the qubits used for
     * entanglement between the hosts.
     *
     * @return The number of qubits in the network.
     */
    size_t GetNumQubits() const override
    {
        size_t res = 0;

        for (const auto &host : hosts)
            res += host->GetNumQubits();

        return res;
    }

    /**
     * @brief Get the number of qubits in the network for the specified host.
     *
     * Get the number of qubits in the network for the specified host, excluding
     * the qubits used for entanglement between the hosts.
     *
     * @param hostId The id of the host to get the number of qubits for.
     * @return The number of qubits in the network for the specified host.
     */
    size_t GetNumQubitsForHost(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return 0;

        return hosts[hostId]->GetNumQubits();
    }

    /**
     * @brief Get the number of qubits used for entanglement between hosts.
     *
     * Get the number of qubits used for entanglement between hosts in the
     * network. For the simple network it's one qubit per host.
     *
     * @return The number of qubits used for entanglement between hosts.
     */
    size_t GetNumNetworkEntangledQubits() const override
    {
        size_t res = 0;

        for (const auto &host : hosts)
            res += host->GetNumNetworkEntangledQubits();

        return res;
    }

    /**
     * @brief Get the number of qubits used for entanglement between hosts for the
     * specified host.
     *
     * Get the number of qubits used for entanglement between hosts in the network
     * for the specified host. For the simple network it's a single qubit.
     *
     * @param hostId The id of the host to get the number of qubits used for
     * entanglement between hosts for.
     * @return The number of qubits used for entanglement between hosts for the
     * specified host.
     */
    size_t GetNumNetworkEntangledQubitsForHost(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return 0;

        return hosts[hostId]->GetNumNetworkEntangledQubits();
    }

    /**
     * @brief Get the number of classical bits in the network.
     *
     * Get the number of classical bits in the network, excluding the classical
     * bits used for measurement of entanglement qubits between the hosts.
     *
     * @return The number of classical bits in the network.
     */
    size_t GetNumClassicalBits() const override
    {
        size_t res = 0;

        for (const auto &host : hosts)
            res += host->GetNumClassicalBits();

        return res;
    }

    /**
     * @brief Get the number of classical bits in the network for the specified
     * host.
     *
     * Get the number of classical bits in the network for the specified host,
     * excluding the classical bits used for measurement of entanglement qubits
     * between the hosts.
     *
     * @param hostId The id of the host to get the number of classical bits for.
     * @return The number of classical bits in the network for the specified host.
     */
    size_t GetNumClassicalBitsForHost(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return 0;

        return hosts[hostId]->GetNumClassicalBits();
    }

    /**
     * @brief Get the hosts in the network.
     *
     * Gets the hosts in the network.
     *
     * @return A vector with the hosts in the network.
     * @sa IHost
     * @sa SimpleHost
     */
    std::vector<std::shared_ptr<IHost<Time>>> &GetHosts()
    {
        return hosts;
    }

    /**
     * @brief Set the network controller host.
     *
     * Sets the network controller host.
     * @param[in] cntrl The controller host.
     * @sa IController
     */
    void SetController(const std::shared_ptr<IController<Time>> &cntrl)
    {
        controller = cntrl;
    }

    /**
     * @brief Sends a packet between two hosts.
     *
     * Sends a packet between the two specified hosts.
     * It's not used in the simple network, because sending/receiving the
     * classical information is done using a shared classical state for the
     * simulator.
     *
     * @param fromHostId The id of the host to send the packet from.
     * @param toHostId The id of the host to send the packet to.
     * @param packet The packet to send.
     * @return False.
     */
    bool SendPacket(size_t fromHostId, size_t toHostId, const std::vector<uint8_t> &packet) override
    {
        return false;
    }

    /**
     * @brief Get the type of the network.
     *
     * Get the type of the network.
     * @return The type of the network.
     * @sa NetworkType
     */
    NetworkType GetType() const override
    {
        return NetworkType::kSimpleDisconnectedNetwork;
    }

    /**
     * @brief Check if the circuit operation is local.
     *
     * Check if the specified circuit operation is local. A local operation is an
     * operation that is executed on a single host. This does not include the
     * operations that also operate on the entanglement qubits between hosts.
     *
     * @param op The operation to check.
     * @return True if the specified circuit operation is local, false otherwise.
     */
    bool IsLocalOperation(const std::shared_ptr<Circuits::IOperation<Time>> &op) const override
    {
        const auto qubits = op->AffectedQubits();

        if (qubits.empty())
            return true;

        size_t firstQubit = qubits[0];

        for (size_t q = 1; q < qubits.size(); ++q)
            if (!AreQubitsOnSameHost(firstQubit, qubits[q]))
                return false;

        return true;
    }

    /**
     * @brief Check if the circuit operation is distributed.
     *
     * Check if the specified circuit operation is distributed. A distributed
     * operation is an operation that is executed on multiple hosts. This does not
     * include the operations that also operate on the entanglement qubits between
     * hosts.
     *
     * @param op The operation to check.
     * @return True if the specified circuit operation is distributed, false
     * otherwise.
     */
    bool IsDistributedOperation(const std::shared_ptr<Circuits::IOperation<Time>> &op) const override
    {
        const auto qubits = op->AffectedQubits();

        if (qubits.empty())
            return false;

        // grab the first qubit that is on a host (skip over network entangled
        // qubits)
        size_t firstQubit = qubits[0];
        size_t q = 1;
        for (; IsNetworkEntangledQubit(firstQubit) && q < qubits.size(); ++q)
            firstQubit = qubits[q];

        // check to see one of the other qubits is on a different host, but ignore
        // the network entangled qubits
        for (; q < qubits.size(); ++q)
            if (!IsNetworkEntangledQubit(qubits[q]) && !AreQubitsOnSameHost(firstQubit, qubits[q]))
                return true;

        return false;
    }

    /**
     * @brief Check if the circuit operation operates on the entanglement qubits
     * between hosts.
     *
     * Check if the specified circuit operation operates on the entanglement
     * qubits between hosts.
     *
     * @param op The operation to check.
     * @return True if the specified circuit operation operates on the
     * entanglement qubits between hosts, false otherwise.
     */
    bool OperatesWithNetworkEntangledQubit(const std::shared_ptr<Circuits::IOperation<Time>> &op) const override
    {
        const auto qubits = op->AffectedQubits();

        if (qubits.empty())
            return false;

        for (size_t q = 0; q < qubits.size(); ++q)
            if (IsNetworkEntangledQubit(qubits[q]))
                return true;

        return false;
    }

    /**
     * @brief Checks if a gate is an entangling gate.
     *
     * An entangling gate is a gate that operates on two qubits that are used for
     * entanglement between hosts.
     * @param op The operation to check.
     * @return True if the specified circuit operation is an entangling gate,
     * false otherwise.
     */
    bool IsEntanglingGate(const std::shared_ptr<Circuits::IOperation<Time>> &op) const override
    {
        if (op->GetType() != Circuits::OperationType::kGate)
            return false;
        const auto qubits = op->AffectedQubits();
        if (qubits.size() != 2)
            return false;

        return IsNetworkEntangledQubit(qubits[0]) && IsNetworkEntangledQubit(qubits[1]);
    }

    /**
     * @brief Checks if a gate expects a classical bit from another host.
     *
     * It must be a conditional gate, conditioned on a classical bit from another
     * host. Use it on already distributed gates, not on the original circuit.
     *
     * @param op The operation to check.
     * @return True if the specified circuit operation needs a classical bit from
     * another host, false otherwise.
     */
    bool ExpectsClassicalBitFromOtherHost(const std::shared_ptr<Circuits::IOperation<Time>> &op) const override
    {
        if (!op->IsConditional())
            return false;

        const auto qubits = op->AffectedQubits();

        const std::shared_ptr<Circuits::IConditionalOperation<Time>> condOp = std::static_pointer_cast<Circuits::IConditionalOperation<Time>>(op);
        const auto &classicalBits = condOp->GetCondition()->GetBitsIndices();

        if (qubits.empty() && classicalBits.empty())
            throw std::runtime_error("No classical bits specified!"); // this would be odd!

        // consider it on the host where it has the first qubit (or bit, if there
        // are no qubits)

        const size_t hostId = GetHostIdForAnyQubit(qubits[0]);

        // now check the classical bits
        for (const auto bit : classicalBits)
            if (hostId != GetHostIdForClassicalBit(bit))
                return true;

        return false;
    }

    /**
     * @brief Get the host id where the classical control bit resides for a
     * conditioned gate.
     *
     * It must be a conditional gate, conditioned on a qubit from another host.
     * Use it on already distributed gates, not on the original circuit.
     *
     * @param op The operation to find the host of the control bit.
     * @return The host id.
     */
    size_t GetHostIdForClassicalControl(const std::shared_ptr<Circuits::IOperation<Time>> &op) const override
    {
        if (!op->IsConditional())
            throw std::runtime_error("Operation is not conditional!");

        std::shared_ptr<Circuits::IConditionalOperation<Time>> condOp = std::static_pointer_cast<Circuits::IConditionalOperation<Time>>(op);
        const auto classicalBits = condOp->AffectedBits();

        if (classicalBits.empty())
            throw std::runtime_error("No classical bits specified!");

        return GetHostIdForClassicalBit(classicalBits[0]);
    }

    /**
     * @brief Check if the specified qubits are on the same host.
     *
     * Check if the specified qubits are on the same host. This does not include
     * the qubits used for entanglement between the hosts.
     *
     * @return True if the specified qubits are on the same host, false otherwise.
     */
    bool AreQubitsOnSameHost(size_t qubitId1, size_t qubitId2) const override
    {
        for (const auto &host : hosts)
        {
            const bool present1 = host->IsQubitOnHost(qubitId1);
            const bool present2 = host->IsQubitOnHost(qubitId2);

            if (present1 && present2)
                return true;
            else if (present1 || present2)
                return false;
        }

        return false;
    }

    /**
     * @brief Check if the specified classical bits are on the same host.
     *
     * Check if the specified classical are on the same host. This does not
     * include the classical bits used for measuring the qubits used for
     * entanglement between the hosts.
     *
     * @return True if the specified qubits are on the same host, false otherwise.
     */
    bool AreClassicalBitsOnSameHost(size_t bitId1, size_t bitId2) const override
    {
        for (const auto &host : hosts)
        {
            const bool present1 = host->IsClassicalBitOnHost(bitId1);
            const bool present2 = host->IsClassicalBitOnHost(bitId2);

            if (present1 && present2)
                return true;
            else if (present1 || present2)
                return false;
        }

        return false;
    }

    /**
     * @brief Check if the specified qubit and classical bit are on the same host.
     *
     * Check if the specified qubit and classical bit are on the same host. This
     * does not include the qubits used for entanglement between the hosts and the
     * classical bits used for their measurement.
     *
     * @return True if the specified qubit and classical bit are on the same host,
     * false otherwise.
     */
    bool AreQubitAndClassicalBitOnSameHost(size_t qubitId, size_t bitId) const override
    {
        for (const auto &host : hosts)
        {
            const bool present1 = host->IsQubitOnHost(qubitId);
            const bool present2 = host->IsClassicalBitOnHost(bitId);

            if (present1 && present2)
                return true;
            else if (present1 || present2)
                return false;
        }

        return false;
    }

    /**
     * @brief Get the host id for the specified qubit.
     *
     * Get the host id for the specified qubit, excluding the qubits used for
     * entanglement between hosts.
     *
     * @param qubitId The id of the qubit to get the host id for.
     * @return The host id for the specified qubit.
     */
    size_t GetHostIdForQubit(size_t qubitId) const override
    {
        for (const auto &host : hosts)
            if (host->IsQubitOnHost(qubitId))
                return host->GetId();

        return std::numeric_limits<size_t>::max();
    }

    /**
     * @brief Get the host id for the specified qubit used for entanglement
     * between hosts.
     *
     * Get the host id for the specified qubit used for entanglement between
     * hosts.
     *
     * @param qubitId The id of the qubit used for entanglement between hosts to
     * get the host id for.
     * @return The host id for the specified qubit used for entanglement between
     * hosts.
     */
    size_t GetHostIdForEntangledQubit(size_t qubitId) const override
    {
        for (const auto &host : hosts)
            if (host->IsEntangledQubitOnHost(qubitId))
                return host->GetId();

        return std::numeric_limits<size_t>::max();
    }

    /**
     * @brief Get the host id for the specified qubit.
     *
     * Get the host id for the squbit, including the qubits used for entanglement
     * between hosts.
     *
     * @param qubitId The id of the qubit to get the host id for.
     * @return The host id for the specified qubit.
     */
    size_t GetHostIdForAnyQubit(size_t qubitId) const override
    {
        if (IsNetworkEntangledQubit(qubitId))
            return GetHostIdForEntangledQubit(qubitId);

        return GetHostIdForQubit(qubitId);
    }

    /**
     * @brief Get the host id for the specified classical bit.
     *
     * Get the host id for the specified classical bit, excluding the one(s) used
     * for measurement of entanglement qubits between the hosts.
     *
     * @param classicalBitId The id of the classical bit to get the host id for.
     * @return The host id for the specified classical bit.
     */
    size_t GetHostIdForClassicalBit(size_t classicalBitId) const override
    {
        for (const auto &host : hosts)
            if (host->IsClassicalBitOnHost(classicalBitId))
                return host->GetId();

        return std::numeric_limits<size_t>::max();
    }

    /**
     * @brief Get the qubit ids for the specified host.
     *
     * Get the qubit ids for the specified host.
     *
     * @param hostId The id of the host to get the qubit ids for.
     * @return A vector with the qubit ids.
     */
    std::vector<size_t> GetQubitsIds(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return std::vector<size_t>();

        return hosts[hostId]->GetQubitsIds();
    }

    /**
     * @brief Get the qubit ids used for entanglement between hosts for the
     * specified host.
     *
     * Get the qubit ids used for entanglement between hosts  for the specified
     * host.
     *
     * @param hostId The id of the host to get the qubit ids for.
     * @return A vector with the qubit ids.
     */
    std::vector<size_t> GetNetworkEntangledQubitsIds(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return std::vector<size_t>();

        return hosts[hostId]->GetNetworkEntangledQubitsIds();
    }

    /**
     * @brief Get the classical bit ids for the specified host.
     *
     * Get the classical bit ids for the specified host, excluding the one(s) used
     * for measurement of entanglemen qubits between the hosts.
     *
     * @param hostId The id of the host to get the classical bit ids for.
     * @return A vector with the classical bit ids.
     */
    std::vector<size_t> GetClassicalBitsIds(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return std::vector<size_t>();

        return hosts[hostId]->GetClassicalBitsIds();
    }

    /**
     * @brief Get the classical bit ids used for measurement of entanglement
     * qubits between the hosts for the specified host.
     *
     * Get the classical bit ids used for measurement of entanglement qubits
     * between the hosts for the specified host.
     *
     * @param hostId The id of the host to get the classical bit ids for.
     * @return A vector with the classical bit ids.
     */
    std::vector<size_t> GetEntangledQubitMeasurementBitIds(size_t hostId) const override
    {
        if (hostId >= hosts.size())
            return std::vector<size_t>();

        return hosts[hostId]->GetEntangledQubitMeasurementBitIds();
    }

    /**
     * @brief Check if the specified qubit id is for a qubit used for entanglement
     * between hosts.
     *
     * Check if the specified qubit id is for a qubit used for entanglement
     * between hosts.
     *
     * @param qubitId The id of the qubit to check.
     * @return True if the specified qubit id is for a qubit used for entanglement
     * between hosts, false otherwise.
     */
    bool IsNetworkEntangledQubit(size_t qubitId) const override
    {
        return qubitId >= GetNumQubits();
    }

    /**
     * @brief Check if the specified qubit used for entanglement between hosts is
     * busy.
     *
     * Check if the specified qubit used for entanglement between hosts is busy.
     * It's not used in the simple network, because it doesn't allow quantum
     * communication between hosts.
     *
     * @param qubitId The id of the qubit to check.
     * @return True if the specified qubit used for entanglement between hosts is
     * busy, false otherwise.
     */
    bool IsEntanglementQubitBusy(size_t qubitId) const override
    {
        return false;
    }

    /**
     * @brief Check if any of the two specified qubits used for entanglement
     * between hosts are busy.
     *
     * Check if any of the two specified qubits used for entanglement between
     * hosts are busy. This is used to check if the qubits are free in order to
     * use them for creating an entanglement between hosts. It's not used in the
     * simple network, because it doesn't allow quantum communication between
     * hosts.
     *
     * @param qubitId1 The id of the first qubit to check.
     * @param qubitId2 The id of the second qubit to check.
     * @return True if any of the two specified qubits used for entanglement
     * between hosts are busy, false otherwise.
     */
    bool AreEntanglementQubitsBusy(size_t qubitId1, size_t qubitId2) const override
    {
        return false;
    }

    /**
     * @brief Mark the pair of the specified qubits used for entanglement between
     * hosts as busy.
     *
     * Mark the pair of the specified qubits used for entanglement between hosts
     * as busy. This is used to mark the qubits as busy when they are used for
     * creating an entanglement between hosts. It's not used in the simple
     * network, because it doesn't allow quantum communication between hosts.
     *
     * @param qubitId1 The id of the first qubit to mark as busy.
     * @param qubitId2 The id of the second qubit to mark as busy.
     */
    void MarkEntangledQubitsBusy(size_t qubitId1, size_t qubitId2) override
    {
        throw std::runtime_error("Entanglement between hosts is not supported in the simple network");
    }

    /**
     * @brief Mark the specified qubit used for entanglement between hosts as
     * free.
     *
     * Mark the specified qubit used for entanglement between hosts as free.
     * This is used to mark the qubits as free when they are not used anymore for
     * an entanglement between hosts. It's not used in the simple network, because
     * it doesn't allow quantum communication between hosts.
     *
     * @param qubitId The id of the qubit to mark as free.
     */
    void MarkEntangledQubitFree(size_t qubitId) override
    {
        throw std::runtime_error("Entanglement between hosts is not supported in the simple network");
    }

    /**
     * @brief Clear all entanglements between hosts in the network.
     *
     * Clear all entanglements between hosts in the network.
     * This marks all qubits used for entanglement between hosts as free.
     * If the entanglements are explicitely coordinated in the network, all pairs
     * of entangled qubits are released. It's not used in the simple network,
     * because it doesn't allow quantum communication between hosts.
     */
    void ClearEntanglements() override
    {
        throw std::runtime_error("Entanglement between hosts is not supported in the simple network");
    }

    /**
     * @brief Get the distributed circuit.
     *
     * Get the distributed circuit.
     * Execute() must be called first, otherwise the return would be nullptr.
     *
     * @return The distributed circuit.
     * @sa Circuits::Circuit
     */
    std::shared_ptr<Circuits::Circuit<Time>> GetDistributedCircuit() const override
    {
        return distCirc;
    }

    /**
     * @brief Get the last used simulator type.
     *
     * Get the last used simulator type.
     *
     * @return The simulator type that was used last time.
     */
    int GetLastGpuDevice() const override
    {
        return lastGpuDevice;
    }

    unsigned GetLastGateFusionMaxQubits() const override
    {
        return lastFusionWidth.load();
    }

    bool WasGateFusionEnabled() const override
    {
        return lastFusionEnabled.load();
    }

    bool GetGateFusion() const override
    {
        if (simulator)
        {
            const auto &options = simulator->GetConfigMap();
            const auto found = options.find("gate_fusion");
            if (found != options.end())
                return found->second != "false" && found->second != "0";
        }
        if (!configuration.IsSet("gate_fusion"))
            return true;
        const auto value = configuration.GetConfiguration("gate_fusion");
        return value != "false" && value != "0";
    }

    Simulators::SimulatorType GetLastSimulatorType() const override
    {
        return lastSimulatorType;
    }

    /**
     * @brief Get the last used simulation type.
     *
     * Get the last used simulation type.
     *
     * @return The simulation type that was used last time.
     */
    Simulators::SimulationType GetLastSimulationType() const override
    {
        return lastMethod;
    }

    /**
     * @brief Get the maximum number of simulators that can be used in the
     * network.
     *
     * Get the maximum number of simulators that can be used in the network.
     * This is used to limit the number of simulators (and corresponding threads)
     * that can be used in the network.
     *
     * @return The maximum number of simulators that can be used in the network.
     */
    size_t GetMaxSimulators() const override
    {
        return maxSimulators;
    }

    /**
     * @brief Set the maximum number of simulators that can be used in the
     * network.
     *
     * Set the maximum number of simulators that can be used in the network.
     * This is used to limit the number of simulators (and corresponding threads)
     * that can be used in the network.
     *
     * @param maxSimulators The maximum number of simulators that can be used in
     * the network.
     */
    void SetMaxSimulators(size_t val) override
    {
        if (val < 1)
            val = 1;
        else if (val > (size_t)QC::QubitRegisterCalculator<>::GetNumberOfThreads())
            val = (size_t)QC::QubitRegisterCalculator<>::GetNumberOfThreads();

        maxSimulators = val;
    }

    /**
     * @brief Set the estimated work per trajectory worker.
     *
     * Per-shot (shot_v1) statevector and density-matrix trajectories use at
     * most one worker per this many microseconds of estimated serial work, so
     * tiny states don't pay for dispatching workers. 0 disables the limit.
     * Counts do not depend on it.
     *
     * @param micros The estimated serial work per worker, in microseconds.
     */
    void SetTrajectoryMinWorkerMicros(double micros)
    {
        trajectoryMinWorkerMicros = std::max(0., micros);
    }

    double GetTrajectoryMinWorkerMicros() const
    {
        return trajectoryMinWorkerMicros;
    }

    /**
     * @brief Allows using an optimized simulator.
     *
     * If set, allows changing the simulator with an optimized one.
     * States/amplitudes are not available in such a case, disable if you need
     * them.
     * Multiple shots still reuse the executable prefix with a fixed backend;
     * the existing single-shot path is retained for state queries.
     *
     * @param optimize If true, the simulator will be optimized if possible.
     */
    void SetOptimizeSimulator(bool optimize = true) override
    {
        optimizeSimulator = optimize;
    }

    /**
     * @brief Returns the 'optimize' flag.
     *
     * Returns the flag set by SetOptimizeSimulator().
     *
     * @return The 'optimize' flag.
     */
    bool GetOptimizeSimulator() const override
    {
        return optimizeSimulator;
    }

    /**
     * @brief Get the optimizations simulators set.
     *
     * Get the optimization simulators set.
     * To be used internally, will not be exposed from the library.
     *
     * @return The simulators set.
     */
    const typename BaseClass::SimulatorsSet &GetSimulatorsSet() const override
    {
        return simulatorsForOptimizations.AsSet();
    }

    std::vector<typename BaseClass::SimulatorPair> GetOptimizationCandidates() const override
    {
        if (!simulatorsForOptimizations.configured)
            return {};
        return simulatorsForOptimizations.Ordered();
    }

    bool HasCostModel() const override
    {
        return costModelEnabled && simulatorsEstimator && simulatorsEstimator->IsInitialized();
    }

    void SetCostModelEnabled(bool enabled) override
    {
        costModelEnabled = enabled;
    }

    std::optional<std::string> GetLastDistributedPlacement(const std::string &key) const override
    {
        const auto found = lastDistributedPlacement.find(key);
        if (found == lastDistributedPlacement.end())
            return std::nullopt;
        return found->second;
    }

    /**
     * @brief Adds a simulator to the simulators optimization set.
     *
     * Adds a simulator (if not already present) to the simulators optimization
     * set.
     *
     * @param type The type of the simulator to add.
     * @param kind The kind of the simulation to add.
     */
    void AddOptimizationSimulator(Simulators::SimulatorType type, Simulators::SimulationType kind) override
    {
        simulatorsForOptimizations.Insert({type, kind});
        simulatorsForOptimizations.configured = true;
    }

    /**
     * @brief Removes a simulator from the simulators optimization set.
     *
     * Removes a simulator from the simulators optimization set, if it exists.
     *
     * @param type The type of the simulator to remove.
     * @param kind The kind of the simulation to remove.
     */
    void RemoveOptimizationSimulator(Simulators::SimulatorType type, Simulators::SimulationType kind) override
    {
        simulatorsForOptimizations.Erase({type, kind});
        simulatorsForOptimizations.configured = true;
    }

    /**
     * @brief Removes all simulators from the simulators optimization set and adds
     * the one specified.
     *
     * Removes all simulators from the simulators optimization set and adds the
     * one specified.
     *
     * @param type The type of the simulator to add.
     * @param kind The kind of the simulation to add.
     */
    void RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType type, Simulators::SimulationType kind) override
    {
        simulatorsForOptimizations.Clear();
        simulatorsForOptimizations.Insert({type, kind});
        simulatorsForOptimizations.configured = true;
    }

    /**
     * @brief Checks if a simulator exists in the optimization set.
     *
     * Checks if a simulator exists in the optimization set.
     *
     * @param type The type of the simulator to check.
     * @param kind The kind of the simulation to check.
     * @return True if the simulator exists in the optimization set, false
     * otherwise.
     */
    bool OptimizationSimulatorExists(Simulators::SimulatorType type, Simulators::SimulationType kind) const override
    {
        if (simulatorsForOptimizations.empty())
            return true;

        return simulatorsForOptimizations.Contains({type, kind});
    }

    /**
     * @brief Clone the network.
     *
     * Clone the network in a pristine state: its settings (those set directly
     * on the simulator included) and a fresh simulator of the same backend and
     * creation width.
     * @return A shared pointer to the cloned network.
     */
    std::shared_ptr<INetwork<Time>> Clone() const override
    {
        return CloneWithSimulatorQubits(simulatorQubits);
    }

    /**
     * @brief Clone the network for execution only.
     *
     * See INetwork::CloneForExecution. A simple network allocates its own
     * register for each execution, so the clone's simulator only selects the
     * backend. Distribution keeps its configured register, and a derived
     * network that executes on its simulator itself keeps its own Clone().
     */
    std::shared_ptr<INetwork<Time>> CloneForExecution(size_t qubits) const override
    {
        if (GetType() != NetworkType::kSimpleDisconnectedNetwork || (simulator && Simulators::IsDistributedGpuSimulator(simulator->GetType())))
            return Clone();
        return CloneWithSimulatorQubits(qubits);
    }

  protected:
    // Clone()'s body, with the creation width of the clone's simulator.
    std::shared_ptr<INetwork<Time>> CloneWithSimulatorQubits(size_t simulatorWidth) const
    {
        const size_t numHosts = GetNumHosts();

        std::vector<Types::qubit_t> qubits(numHosts);
        std::vector<size_t> cbits(numHosts);

        for (size_t h = 0; h < numHosts; ++h)
        {
            qubits[h] = GetNumQubitsForHost(h);
            cbits[h] = GetNumClassicalBitsForHost(h);
        }

        const auto cloned = std::make_shared<SimpleDisconnectedNetwork<Time, Controller>>(qubits, cbits);

        cloned->configuration = EffectiveConfiguration();
        cloned->distributedHostQubitIndexing = distributedHostQubitIndexing;
        cloned->resolvedDistributedDevices = resolvedDistributedDevices;

        cloned->maxSimulators = maxSimulators;
        cloned->trajectoryMinWorkerMicros = trajectoryMinWorkerMicros;

        cloned->optimizeSimulator = optimizeSimulator;
        cloned->costModelEnabled = costModelEnabled;
        cloned->simulatorsForOptimizations = simulatorsForOptimizations;
        cloned->GetController()->SetOptimizeCircuit(GetController()->GetOptimizeCircuit());
        cloned->GetController()->SetOptimizeRotationGates(GetController()->GetOptimizeRotationGates());

        cloned->SetInitialQubitsMapOptimization(GetInitialQubitsMapOptimization());
        cloned->SetMPSOptimizeSwaps(GetMPSOptimizeSwaps());

        cloned->SetMPSOptimizationBondDimensionThreshold(GetMPSOptimizationBondDimensionThreshold());
        cloned->SetMPSOptimizationQubitsNumberThreshold(GetMPSOptimizationQubitsNumberThreshold());

        cloned->SetLookaheadDepth(GetLookaheadDepth());
        cloned->SetLookaheadDepthWithHeuristic(GetLookaheadDepthWithHeuristic());

        cloned->setGrowthFactorGate(getGrowthFactorGate());
        cloned->setGrowthFactorSwap(getGrowthFactorSwap());

        // Clone() keeps the width this one was created with: a backend
        // placeholder stays small, an exact register stays exact.
        if (GetSimulator())
            cloned->CreateSimulator(GetSimulator()->GetType(), GetSimulator()->GetSimulationType(), simulatorWidth);

        return cloned;
    }

    // Installs a simulator another step prepared, such as the register backend
    // selection allocated, as the network's own, instead of allocating one
    // more. The creation width (simulatorQubits) stays, for re-creation.
    void InstallSimulator(std::shared_ptr<Simulators::ISimulator> prepared)
    {
        classicalState.Clear();
        classicalState.AllocateBits(GetNumClassicalBits() + GetNumNetworkEntangledQubits());
        simulator = std::move(prepared);
        if (simulator)
        {
            simulator->setGrowthFactorGate(growthFactorGate);
            simulator->setGrowthFactorSwap(growthFactorSwap);
            simulator->SetLookaheadDepth(lookaheadDepth);
            simulator->SetLookaheadDepthWithHeuristic(lookaheadDepthWithHeuristic);
        }
    }

  public:
    std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t &counts, size_t nrQubits,
                                                                size_t nrCbits, size_t nrResultCbits, Simulators::SimulatorType &simType,
                                                                Simulators::SimulationType &method, std::vector<bool> &executed, bool multithreading = false,
                                                                bool dontRunCircuitStart = false) override
    {
        // Distribution is an explicit execution choice. Timing-based backend
        // selection must not replace it or diverge between MPI ranks.
        if (Simulators::IsDistributedGpuSimulator(simType))
            return nullptr;
        if (!optimizeSimulator)
            return nullptr;

        // Mixed-state candidates are never ranked by the estimator, so a set made
        // only of them is resolved below without it.
        const bool onlyMixedStateCandidates =
            !simulatorsForOptimizations.empty() && std::all_of(simulatorsForOptimizations.begin(), simulatorsForOptimizations.end(),
                                                               [](const auto &candidate) { return IsMixedStateMethod(candidate.second); });
        // Without a cost model a configured list decides, in its order (see
        // INetwork::GetOptimizationCandidates). The constructor's default list
        // is no request: the network simulator's backend runs then.
        const bool costModel = HasCostModel();
        const bool configuredList = simulatorsForOptimizations.configured && simulatorsForOptimizations.size() > 1;
        const bool listDecides = !costModel && configuredList;
        if (!costModel && simulatorsForOptimizations.size() != 1 && !onlyMixedStateCandidates && !listDecides)
            return nullptr;

        // when multithreading is set to true it means it needs a multithreaded
        // simulator

        std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>> simulatorTypes;

        const bool checkTensorNetwork = method == Simulators::SimulationType::kTensorNetwork;

        // the others are to be picked between statevector, composite, tensor
        // networks and mps, for now at least for tensor networks in the future it's
        // worth checking different contractors!!!!
        //
        // clifford was decided at higher level; a stabilizer simulator of the
        // network (not a Clifford switch) does not make a circuit Clifford
        if (method == Simulators::SimulationType::kStabilizer && dcirc->IsClifford())
        {
            // compare qcsim with qiskit aer if qiskit aer is available, let the best
            // one win
            if (OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer))
                simulatorTypes.emplace_back(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer);

#ifndef NO_QISKIT_AER
            // if the number of shots is too small, probably it's not worth it, it's
            // going to be better to just execute them multithreading
            if (OptimizationSimulatorExists(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStabilizer))
                simulatorTypes.emplace_back(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStabilizer);
#endif
        }

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kCompositeQCSim, Simulators::SimulationType::kStatevector))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kCompositeQCSim, Simulators::SimulationType::kStatevector);

        if (checkTensorNetwork && OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kTensorNetwork))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kTensorNetwork);

        const long long int maxBondDim = configuration.GetConfigurationAsInt("matrix_product_state_max_bond_dimension");

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState) &&
            (nrQubits <= 4 || maxBondDim > 0))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPauliPropagator))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPauliPropagator);

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPathIntegral))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPathIntegral);

#ifndef NO_QISKIT_AER
        // tensor networks are out of the picture for now for qiskit aer, since they
        // are available with cuda library, and work only on linux (obviously when
        // compiled properly and if there is the right hw an driver installed)

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kStatevector);

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kCompositeQiskitAer, Simulators::SimulationType::kStatevector))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kCompositeQiskitAer, Simulators::SimulationType::kStatevector);

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState) &&
            (nrQubits <= 4 || maxBondDim > 0))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQiskitAer, Simulators::SimulationType::kMatrixProductState);
#endif

#ifdef __linux__
        const int gpuDevice = configuration.IsSet("gpu_device") ? Simulators::Configuration::ParseGpuDevice(configuration.GetConfiguration("gpu_device"))
                                                                : Simulators::SimulatorsFactory::ResolveGpuDevice();
        Simulators::SimulatorsFactory::ScopedGpuDevice gpuDeviceScope(gpuDevice);
        const bool hasGpuCandidate = std::any_of(simulatorsForOptimizations.begin(), simulatorsForOptimizations.end(),
                                                 [](const auto &candidate) { return candidate.first == Simulators::SimulatorType::kGpuSim; });
        if (configuration.IsSet("gpu_device") && hasGpuCandidate && !Simulators::SimulatorsFactory::IsGpuLibraryAvailable(gpuDevice))
            throw std::runtime_error("Unable to initialize requested GPU device " + std::to_string(gpuDevice));
        if (hasGpuCandidate && Simulators::SimulatorsFactory::IsGpuLibraryAvailable(gpuDevice))
        {
            if (OptimizationSimulatorExists(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector))
                simulatorTypes.emplace_back(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector);
            if (OptimizationSimulatorExists(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState))
                simulatorTypes.emplace_back(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
            if (OptimizationSimulatorExists(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kTensorNetwork))
                simulatorTypes.emplace_back(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kTensorNetwork);
            if (OptimizationSimulatorExists(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kPauliPropagator))
                simulatorTypes.emplace_back(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kPauliPropagator);
        }
#endif

        if (OptimizationSimulatorExists(Simulators::SimulatorType::kQuestSim, Simulators::SimulationType::kStatevector))
            simulatorTypes.emplace_back(Simulators::SimulatorType::kQuestSim, Simulators::SimulationType::kStatevector);

        // Unranked candidates are tried in a fixed order: a singleton
        // optimization set (e.g. density_matrix), even one outside the hardcoded
        // list; several mixed-state candidates, which have no cost model; and,
        // without a cost model, any configured list. The pair selection starts
        // from comes first when it is listed: the network simulator's backend
        // (ConfigureNetwork makes it the first listed candidate), or the
        // stabilizer a Clifford circuit moved to. The others follow in the order
        // they were added, never a placeholder outside the set. Each is tried on
        // the register execution maps, so a density matrix that cannot hold a
        // 40-qubit circuit still runs one that uses a single qubit. The chosen
        // backend is prepared afresh, as for a singleton, so terminal shot
        // workers can clone it even after host execution cleared the network's
        // simulator.
        // A candidate whose register cannot be allocated (a density matrix of
        // 40 qubits) yields to the next. Skip backends that cannot actually be
        // constructed, such as GPU MPS when the GPU library is missing: the
        // caller then keeps the network simulator instead of recording 0 shots.
        // The candidates to try in order, filtered by those kept: the pair
        // selection starts from first when it is among them, the others in the
        // order they were added. Backends that cannot actually be constructed,
        // such as GPU MPS when the GPU library is missing, are skipped.
        const auto requested = std::make_pair(simType, method);
        const auto ordered = [&](bool all) {
            std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>> candidates;
            for (const auto &candidate : simulatorsForOptimizations)
                if (all || simulatorsForOptimizations.size() == 1 || IsMixedStateMethod(candidate.second))
                    candidates.push_back(candidate);
            if (const auto current = std::find(candidates.begin(), candidates.end(), requested); current != candidates.end())
                std::rotate(candidates.begin(), current, current + 1);
            candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                            [](const auto &candidate) {
                                                return candidate.first == Simulators::SimulatorType::kGpuSim &&
                                                       !Simulators::SimulatorsFactory::IsGpuLibraryAvailable();
                                            }),
                             candidates.end());
            // A stabilizer cannot run a non-Clifford gate: the other
            // candidates are tried first. Stabilizers stay last rather than
            // dropped, so that when no other candidate can be created (QuEST
            // not loaded) the stabilizer's failure is reported instead of the
            // run being left to the network simulator.
            if (!dcirc->IsClifford())
                std::stable_partition(candidates.begin(), candidates.end(),
                                      [](const auto &candidate) { return candidate.second != Simulators::SimulationType::kStabilizer; });
            return candidates;
        };
        // Prepares the first candidate whose register can be allocated and,
        // unless dontRunCircuitStart, that runs the start of the circuit. A
        // candidate that cannot (a density matrix of 40 qubits, an operation
        // the backend does not support) yields to the next. When none runs,
        // the first failure recorded is reported: the preferred candidate's,
        // not that of a stabilizer tried last. Only when none can be
        // constructed at all (typical for GPU on CI) do
        // simType/method stay as requested, and RepeatedExecuteOnHost keeps
        // the network simulator instead of recording empty counts.
        // report = false defers that to the caller: a configured list keeps
        // trying its other candidates after the cost model's choice fails.
        std::exception_ptr failure;
        // A backend that neither holds nor samples channels cannot run a
        // circuit's channel instructions. Like the stabilizer's Clifford test
        // below, that is decided before the candidate is accepted and not by
        // running the circuit's start: a channel whose branch is drawn is left
        // to the shots (ExecuteUpToMeasurements stops before it), so the start
        // would pass, the shots would fail, and a later candidate that runs the
        // circuit would never be tried.
        const bool hasChannels = dcirc->HasQuantumChannels();
        const auto requireChannels = [hasChannels](const std::shared_ptr<Simulators::ISimulator> &candidate) {
            if (hasChannels && !candidate->SupportsQuantumChannels() && !candidate->SamplesQuantumChannels())
                throw std::runtime_error("This backend neither holds nor samples quantum channels: use density_matrix or matrix_product_operator, "
                                         "which apply them, or statevector or matrix_product_state, which sample them");
        };
        const auto prepare = [&](const std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>> &candidates,
                                 bool report = true) -> std::shared_ptr<Simulators::ISimulator> {
            for (size_t index = 0; index < candidates.size(); ++index)
            {
                const auto [candidateType, candidateMethod] = candidates[index];
                // A stabilizer cannot run a non-Clifford circuit. That is
                // decided here, not by running the circuit's start: networks
                // that run their shots on their own simulator skip that
                // (dontRunCircuitStart), and would install the stabilizer and
                // fail only in the shots, hiding an earlier candidate's failure.
                if (candidateMethod == Simulators::SimulationType::kStabilizer && !dcirc->IsClifford())
                {
                    // A runtime error, as the stabilizer's own rejection of a
                    // non-Clifford gate is (Python RuntimeError, request API
                    // native_failure).
                    if (!failure)
                        failure = std::make_exception_ptr(std::runtime_error("A stabilizer simulator cannot run a non-Clifford circuit"));
                    if (report && index + 1 == candidates.size())
                        std::rethrow_exception(failure);
                    continue;
                }
                std::shared_ptr<Simulators::ISimulator> sim = Simulators::SimulatorsFactory::CreateSimulator(candidateType, candidateMethod);
                if (!sim)
                    continue;
                configuration.ApplyConfigurationToSimulator(sim);
                try
                {
                    requireChannels(sim);
                    sim->AllocateQubits(nrQubits);
                    sim->Initialize();
                }
                catch (const std::exception &)
                {
                    // The candidate does not run this circuit's channels, or
                    // this register does not fit it; another may.
                    if (!failure)
                        failure = std::current_exception();
                    if (report && index + 1 == candidates.size())
                        std::rethrow_exception(failure);
                    continue;
                }
                simType = candidateType;
                method = candidateMethod;

                if (method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator)
                {
                    sim->setGrowthFactorGate(growthFactorGate);
                    sim->setGrowthFactorSwap(growthFactorSwap);
                    sim->SetLookaheadDepth(lookaheadDepth);
                    sim->SetLookaheadDepthWithHeuristic(lookaheadDepthWithHeuristic);

                    OptimizeMPSInitialQubitsMap(sim, dcirc, nrQubits);
                }

                if (!dontRunCircuitStart)
                {
                    if (sim->GetType() == Simulators::SimulatorType::kQCSim)
                        sim->Configure("reproducible_trajectory", dcirc->HasOpsAfterMeasurements(sim->SupportsQuantumChannels()) ? "true" : "false");
                    sim->SetMultithreading(true);
                    try
                    {
                        Estimators::SimulatorsEstimatorInterface<Time>::ExecuteUpToMeasurements(dcirc, nrQubits, nrCbits, nrResultCbits, sim, executed,
                                                                                                &curMaxBondDim);
                    }
                    catch (const std::exception &)
                    {
                        // The backend cannot run this circuit; another may.
                        if (!failure)
                            failure = std::current_exception();
                        if (report && index + 1 == candidates.size())
                            std::rethrow_exception(failure);
                        executed.clear();
                        curMaxBondDim = 0;
                        continue;
                    }
                }
                sim->SetMultithreading(multithreading || GetMaxSimulators() == 1);

                return sim;
            }
            simType = requested.first;
            method = requested.second;
            // A candidate that could be created but not run must not leave the
            // run to the network simulator, which may be outside the list.
            if (report && failure)
                std::rethrow_exception(failure);
            return nullptr;
        };

        // The configured candidates other than one the cost model already
        // tried, in order. A failure recorded earlier is reported when none of
        // them runs.
        const auto others = [&](const std::pair<Simulators::SimulatorType, Simulators::SimulationType> &tried) -> std::shared_ptr<Simulators::ISimulator> {
            auto rest = ordered(true);
            rest.erase(std::remove(rest.begin(), rest.end(), tried), rest.end());
            return prepare(rest);
        };

        // A configured list the cost model cannot rank (none of its candidates
        // is a type it estimates, such as a tensor network or an extended
        // stabilizer) is tried in order too, never left to the network
        // simulator.
        if (listDecides || simulatorTypes.empty())
            return prepare(ordered(listDecides || configuredList));
        if (simulatorTypes.size() == 1)
        {
            // The one candidate the cost model would rank may not be
            // constructible (a QuEST plugin that is not loaded) or may not hold
            // the register: a configured list then still runs one of its
            // others.
            if (!configuredList)
                return prepare(simulatorTypes);
            if (auto sim = prepare(simulatorTypes, false))
                return sim;
            return others(simulatorTypes.front());
        }

        const double singularValueThreshold = configuration.GetConfigurationAsDouble("matrix_product_state_truncation_threshold");

        const std::string truncationMode = configuration.GetConfiguration("matrix_product_state_truncation_mode");

        const std::string mpsSample = configuration.GetConfiguration("mps_sample_measure_algorithm");

        std::shared_ptr<Simulators::ISimulator> sim =
            simulatorsEstimator->ChooseBestSimulator(simulatorTypes, dcirc, counts, nrQubits, nrCbits, nrResultCbits, simType, method, executed, maxBondDim,
                                                     singularValueThreshold, truncationMode, mpsSample, GetMaxSimulators(), pauliStrings, multithreading);

        if (sim)
        {
            const auto chosen = std::make_pair(simType, method);
            try
            {
                // The estimator sets only the MPS truncation. The network's
                // configuration (gate fusion, seed, precision, ...) must reach
                // the chosen simulator before it allocates and runs the prefix,
                // as for the unranked candidates above.
                configuration.ApplyConfigurationToSimulator(sim);
                requireChannels(sim);
                sim->AllocateQubits(nrQubits);
                sim->Initialize();

                sim->setGrowthFactorGate(growthFactorGate);
                sim->setGrowthFactorSwap(growthFactorSwap);
                sim->SetLookaheadDepth(lookaheadDepth);
                sim->SetLookaheadDepthWithHeuristic(lookaheadDepthWithHeuristic);

                OptimizeMPSInitialQubitsMap(sim, dcirc, nrQubits);

                if (!dontRunCircuitStart)
                {
                    if (sim->GetType() == Simulators::SimulatorType::kQCSim)
                        sim->Configure("reproducible_trajectory", dcirc->HasOpsAfterMeasurements(sim->SupportsQuantumChannels()) ? "true" : "false");
                    sim->SetMultithreading(true);
                    Estimators::SimulatorsEstimatorInterface<Time>::ExecuteUpToMeasurements(dcirc, nrQubits, nrCbits, nrResultCbits, sim, executed,
                                                                                            &curMaxBondDim);
                }
                sim->SetMultithreading(multithreading || GetMaxSimulators() == 1);
                return sim;
            }
            catch (const std::exception &)
            {
                // The ranked choice cannot hold or run this circuit.
                if (!configuredList)
                    throw;
                failure = std::current_exception();
                executed.clear();
                curMaxBondDim = 0;
            }
            simType = requested.first;
            method = requested.second;
            return others(chosen);
        }

        // The cost model chose nothing (it may not estimate this circuit): a
        // configured list still runs one of its own candidates.
        simType = requested.first;
        method = requested.second;
        if (configuredList)
            return prepare(ordered(true));
        return nullptr;
    }

    void SetInitialQubitsMapOptimization(bool optimize = true) override
    {
        optimizeInitialQubitsMap = optimize;
    }

    bool GetInitialQubitsMapOptimization() const override
    {
        return optimizeInitialQubitsMap;
    }

    void SetMPSOptimizeSwaps(bool optimize = true) override
    {
        mpsOptimizeSwaps = optimize;

        if (simulator)
        {
            simulator->SetLookaheadDepth(0);
            simulator->SetLookaheadDepthWithHeuristic(0);
        }
    }

    bool GetMPSOptimizeSwaps() const override
    {
        return mpsOptimizeSwaps;
    }

    void SetMPSOptimizationBondDimensionThreshold(size_t threshold) override
    {
        mpsOptimizationBondDimensionThreshold = threshold;

        if (simulator && std::stoull(simulator->GetConfiguration("matrix_product_state_max_bond_dimension")) < threshold)
        {
            simulator->SetLookaheadDepth(0);
            simulator->SetLookaheadDepthWithHeuristic(0);
        }
    }

    size_t GetMPSOptimizationBondDimensionThreshold() const override
    {
        return mpsOptimizationBondDimensionThreshold;
    }

    void SetMPSOptimizationQubitsNumberThreshold(size_t threshold) override
    {
        mpsOptimizationQubitsNumberThreshold = threshold;

        if (GetNumQubits() < threshold && simulator)
        {
            simulator->SetLookaheadDepth(0);
            simulator->SetLookaheadDepthWithHeuristic(0);
        }
    }

    size_t GetMPSOptimizationQubitsNumberThreshold() const override
    {
        return mpsOptimizationQubitsNumberThreshold;
    }

    void SetLookaheadDepth(int depth) override
    {
        if (depth < 0)
            depth = std::numeric_limits<int>::max();

        lookaheadDepth = depth;

        if (simulator && lookaheadDepth != std::numeric_limits<int>::max())
        {
            simulator->SetLookaheadDepth(depth);
            if (lookaheadDepthWithHeuristic != std::numeric_limits<int>::max())
                simulator->SetLookaheadDepthWithHeuristic(std::min(depth, lookaheadDepthWithHeuristic));
        }
    }

    int GetLookaheadDepth() const override
    {
        return lookaheadDepth;
    }

    void SetLookaheadDepthWithHeuristic(int depth) override
    {
        if (depth < 0)
            depth = std::numeric_limits<int>::max();

        if (depth > lookaheadDepth)
            depth = lookaheadDepth;

        lookaheadDepthWithHeuristic = depth;

        if (simulator && lookaheadDepthWithHeuristic != std::numeric_limits<int>::max())
            simulator->SetLookaheadDepthWithHeuristic(depth);
    }

    int GetLookaheadDepthWithHeuristic() const override
    {
        return lookaheadDepthWithHeuristic;
    }

    double getGrowthFactorSwap() const override
    {
        return growthFactorSwap;
    }

    double getGrowthFactorGate() const override
    {
        return growthFactorGate;
    }

    void setGrowthFactorSwap(double factor) override
    {
        growthFactorSwap = factor;

        if (simulator)
            simulator->setGrowthFactorSwap(factor);
    }

    void setGrowthFactorGate(double factor) override
    {
        growthFactorGate = factor;

        if (simulator)
            simulator->setGrowthFactorGate(factor);
    }

    /**
     * @brief Returns the maximum bond dimension reached.
     *
     * Returns the maximum bond dimension reached during execution, if applicable
     * (mps simulator, either qcsim or gpu).
     */
    size_t GetCurrentMaxBondDimension() const override
    {
        return curMaxBondDim;
    }

  protected:
    // Resolved placement is separate from user settings. Importing a smaller
    // simulator must not turn its automatic shard subset into an explicit choice.
    void CaptureSimulatorConfiguration()
    {
        configuration = EffectiveConfiguration();
    }

    // Where a distributed executor placed its register, read before the
    // network re-creates its simulator (GetLastDistributedPlacement).
    void RecordDistributedPlacement(const std::shared_ptr<Simulators::ISimulator> &executor)
    {
        lastDistributedPlacement.clear();
        if (!executor || !Simulators::IsDistributedGpuSimulator(executor->GetType()))
            return;
        for (const char *key : BaseClass::kDistributedPlacementKeys)
            lastDistributedPlacement[key] = executor->GetConfiguration(key);
    }

    // The stored configuration plus the settings made directly on the
    // network's simulator (GetSimulator()->Configure), which every execution
    // imports. Clones must carry them too, or a pristine copy would run
    // without them. The requested distributed placement stays the user's.
    Configuration<Time> EffectiveConfiguration() const
    {
        auto effective = configuration;
        if (simulator)
        {
            effective.ApplyConfigurationFromSimulator(simulator);
            if (Simulators::IsDistributedGpuSimulator(simulator->GetType()))
                effective.SetConfiguration("distributed_devices", configuration.GetConfiguration("distributed_devices"));
        }
        return effective;
    }

    // Re-creates the network's simulator with the backend and creation width it
    // had when the run that made the guard throws, if that run cleared it
    // (backend selection), replaced it (the register selection prepared, on
    // netqasm and discrete events) or detached it (a shot job). A simulator the
    // failed run did not touch stays. A run that succeeds restores the
    // simulator itself.
    //
    // Creating an MPI distributed state is collective across ranks: if only
    // some ranks failed, those would hang re-creating it. Such a simulator is
    // re-created only when the failed run left the network without one, as
    // before this guard existed.
    class SimulatorRestoreOnFailure
    {
      public:
        SimulatorRestoreOnFailure(SimpleDisconnectedNetwork &network, Simulators::SimulatorType type, Simulators::SimulationType method, size_t qubits)
            : network(network), type(type), method(method), qubits(qubits), exceptions(std::uncaught_exceptions())
        {
            // Held weakly: a run that installs another register must be able
            // to free this one meanwhile.
            if (const auto simulator = network.GetSimulator())
            {
                original = simulator;
                originalQubits = simulator->GetNumberOfQubits();
            }
        }

        SimulatorRestoreOnFailure(const SimulatorRestoreOnFailure &) = delete;
        SimulatorRestoreOnFailure &operator=(const SimulatorRestoreOnFailure &) = delete;

        ~SimulatorRestoreOnFailure()
        {
            if (std::uncaught_exceptions() <= exceptions)
                return;
            const auto current = network.GetSimulator();
            if (current && type == Simulators::SimulatorType::kDistMpiGpuSim)
                return;
            if (current && current == original.lock() && current->GetNumberOfQubits() == originalQubits)
                return;
            try
            {
                network.CreateSimulator(type, method, qubits);
            }
            catch (...)
            {
                // Report the run's failure, not this one.
            }
        }

      private:
        SimpleDisconnectedNetwork &network;
        Simulators::SimulatorType type;
        Simulators::SimulationType method;
        size_t qubits;
        int exceptions;
        std::weak_ptr<Simulators::ISimulator> original;
        size_t originalQubits = 0;
    };

    // Recreating the network simulator after a single execution must not drop
    // the classical results that execution stored in the network state. It
    // gets the width it had before the execution (a derived execution loop may
    // create simulators of other widths meanwhile).
    void RecreateSimulatorKeepingResults(Simulators::SimulatorType simType, Simulators::SimulationType method, size_t qubits)
    {
        const auto results = GetState().GetAllBits();
        CreateSimulator(simType, method, qubits);
        GetState().SetResultsInOrder(results);
    }

    // Nested circuits read and write the enclosing classical state, so inlining
    // them changes no result. When that leaves only terminal measurements, as
    // for a nested circuit that ends by measuring, the shots are sampled from
    // one prepared state instead of replayed. Other circuits keep their nesting.
    void InlineTerminalComposites()
    {
        if (!distCirc || !distCirc->HasCompositeOperations())
            return;
        auto inlined = distCirc->InlinedComposites();
        if (!inlined->HasOpsAfterMeasurements(true))
            distCirc = std::move(inlined);
    }

    // The stabilizer backend automatic selection may move a Clifford circuit
    // to: the current backend's own (a composite maps to its plain backend)
    // when the candidate set lists it, else another listed stabilizer backend;
    // none when no listed pair is a stabilizer. An empty set admits all.
    std::optional<Simulators::SimulatorType> StabilizerBackend(Simulators::SimulatorType simType) const
    {
        auto own = simType;
        if (own == Simulators::SimulatorType::kCompositeQCSim)
            own = Simulators::SimulatorType::kQCSim;
#ifndef NO_QISKIT_AER
        else if (own == Simulators::SimulatorType::kCompositeQiskitAer)
            own = Simulators::SimulatorType::kQiskitAer;
        const std::array<Simulators::SimulatorType, 2> stabilizers{Simulators::SimulatorType::kQCSim, Simulators::SimulatorType::kQiskitAer};
#else
        const std::array<Simulators::SimulatorType, 1> stabilizers{Simulators::SimulatorType::kQCSim};
#endif
        if (std::find(stabilizers.begin(), stabilizers.end(), own) != stabilizers.end() &&
            OptimizationSimulatorExists(own, Simulators::SimulationType::kStabilizer))
            return own;
        for (const auto type : stabilizers)
            if (OptimizationSimulatorExists(type, Simulators::SimulationType::kStabilizer))
                return type;
        return std::nullopt;
    }

    // Methods holding a mixed state. The cost estimator does not rank them.
    // Whether resets and Kraus channels are deterministic is a different
    // question, which the Pauli propagator also answers yes to: see
    // Simulators::EvolvesExactQuantumChannels.
    static bool IsMixedStateMethod(Simulators::SimulationType method)
    {
        return method == Simulators::SimulationType::kDensityMatrix || method == Simulators::SimulationType::kMatrixProductOperator;
    }

    // Keep adaptive terminal samplers on the full request: splitting it could
    // put every block below the crossover. This decision must not depend on
    // the requested worker count, since it also selects the logical RNG layout.
    bool UsesFullTerminalBatch(Simulators::SimulatorType type, Simulators::SimulationType method, size_t shots, size_t measuredWidth) const
    {
        if (method == Simulators::SimulationType::kStatevector || method == Simulators::SimulationType::kPathIntegral ||
            method == Simulators::SimulationType::kDensityMatrix)
            return true;
        if (type != Simulators::SimulatorType::kQCSim ||
            !Utils::Sampling::AllowsMultinomialCounts(configuration.GetConfiguration("counts_policy"), configuration.GetConfiguration("sampling_policy")))
            return false;
        if (method == Simulators::SimulationType::kExtendedStabilizer)
            return Utils::Sampling::PreferExtendedMarginalCounts(shots, measuredWidth);
        if (measuredWidth != 1 || !Utils::Sampling::PreferMultinomial(shots, 2))
            return false;
        if (method == Simulators::SimulationType::kMatrixProductState)
            return !configuration.IsSet("mps_sample_measure_algorithm") ||
                   configuration.GetConfiguration("mps_sample_measure_algorithm") == "mps_probabilities";
        // A truncating MPO retains its native sampler instead of using an
        // exact marginal contraction.
        const auto truncates = [&](const char *key) {
            const auto value = configuration.GetConfiguration(key);
            return !value.empty() && std::strtod(value.c_str(), nullptr) > 0.;
        };
        return method == Simulators::SimulationType::kMatrixProductOperator && !truncates("matrix_product_state_max_bond_dimension") &&
               !truncates("matrix_product_operator_max_bond_dimension") && !truncates("matrix_product_state_truncation_threshold") &&
               !truncates("matrix_product_operator_truncation_threshold");
    }

    // Shot workers for one execution. Dense states draw every terminal shot
    // from one prepared distribution, so extra workers would only repeat its
    // preparation. A CPU MPO draws each terminal shot by a serial contraction:
    // workers help, but only by sampling clones of one prepared state. Without
    // one from backend selection, prepare it here once; if that is impossible,
    // keep a single worker rather than evolving the MPO once per worker.
    size_t ExecutionThreads(const std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t nrQubits, size_t nrCbits, size_t nrResultCbits, size_t shots,
                            Simulators::SimulatorType simType, Simulators::SimulationType method, bool terminalSampling, size_t measuredWidth,
                            std::shared_ptr<Simulators::ISimulator> &optSim, std::vector<bool> &executed)
    {
        size_t nrThreads = GetMaxSimulators();
#ifdef __linux__
        if (Simulators::IsGpuSimulator(simType))
            return 1;
#endif
        if (simType == Simulators::SimulatorType::kQuestSim || (terminalSampling && UsesFullTerminalBatch(simType, method, shots, measuredWidth)))
            return 1;
        nrThreads = std::min(nrThreads, std::max<size_t>(shots, 1ULL));
        if (method == Simulators::SimulationType::kMatrixProductOperator && terminalSampling && nrThreads > 1)
        {
            if (!optSim)
                optSim = PrepareTerminalState(dcirc, nrQubits, nrCbits, nrResultCbits, simType, method, executed);
            if (!optSim)
                return 1;
        }
        return nrThreads;
    }

    // Prepare the selected backend on exactly this register, with every
    // non-measurement operation applied, for shot workers to clone. Serial
    // requests may reuse the existing simulator, including after resizing it;
    // every repeated execution still starts from zero.
    // Null when the backend cannot hold the register.
    std::shared_ptr<Simulators::ISimulator> PrepareTerminalState(const std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t nrQubits, size_t nrCbits,
                                                                 size_t nrResultCbits, Simulators::SimulatorType simType, Simulators::SimulationType method,
                                                                 std::vector<bool> &executed, std::shared_ptr<Simulators::ISimulator> reusable = nullptr)
    {
        auto prepared = reusable ? reusable : Simulators::SimulatorsFactory::CreateSimulator(simType, method);
        if (!prepared)
            return nullptr;
        try
        {
            if (reusable && prepared->GetNumberOfQubits() == nrQubits)
            {
                prepared->Reset();
                prepared->SetGatesCounter(0);
                ExecutionConfiguration(simType, nrQubits, method).ApplyConfigurationToSimulator(prepared);
            }
            else
            {
                prepared->Clear();
                ExecutionConfiguration(simType, nrQubits, method).ApplyConfigurationToSimulator(prepared);
                prepared->AllocateQubits(nrQubits);
                prepared->Initialize();
            }
        }
        catch (const std::exception &)
        {
            return nullptr; // the single-worker path reports what does not fit
        }
        prepared->setGrowthFactorGate(growthFactorGate);
        prepared->setGrowthFactorSwap(growthFactorSwap);
        prepared->SetLookaheadDepth(lookaheadDepth);
        prepared->SetLookaheadDepthWithHeuristic(lookaheadDepthWithHeuristic);
        auto circuit = dcirc; // the layout planner takes a mutable handle
        OptimizeMPSInitialQubitsMap(prepared, circuit, nrQubits);
        prepared->SetMultithreading(true);
        Estimators::SimulatorsEstimatorInterface<Time>::ExecuteUpToMeasurements(dcirc, nrQubits, nrCbits, nrResultCbits, prepared, executed, &curMaxBondDim);
        // Each worker samples its own clone; avoid nested thread pools there.
        prepared->SetMultithreading(false);
        return prepared;
    }

    bool UsesReproducibleSampling(Simulators::SimulatorType type, Simulators::SimulationType method) const
    {
        return configuration.GetConfiguration("sampling_policy") != "legacy" && ExecuteJob<Time>::ReproducibleMethod(type, method);
    }

    void ExecuteLogicalShotBlocks(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, ExecuteResults &results, size_t shots, size_t workers, size_t qubits,
                                  size_t cbits, size_t resultBits, Simulators::SimulatorType type, Simulators::SimulationType method,
                                  const std::shared_ptr<Simulators::ISimulator> &prepared, const std::vector<bool> &executed, std::mutex &resultsMutex,
                                  bool terminalSampling = false)
    {
        // Terminal native samplers consume one stream per batch. Fixed logical
        // batches preserve those streams across worker counts without replaying
        // the prepared circuit per shot. Dynamic shot_v1 uses per-shot streams.
        const bool independentShots = !terminalSampling && configuration.GetConfiguration("trajectory_policy") == "shot_v1";
        workers = std::max<size_t>(1, workers);
#ifdef _OPENMP
        // Outer jobs count against the same CPU budget as their internal teams.
        int threadBudget =
            omp_in_parallel() || Utils::RealizationDetail::IsRealizationWorker() ? 1 : std::max(1, Utils::RealizationDetail::CallerOpenMPBudget());
        workers = std::min(workers, static_cast<size_t>(threadBudget));
#endif
        if (method == Simulators::SimulationType::kStatevector || method == Simulators::SimulationType::kDensityMatrix)
        {
            const size_t bitsPerQubit = method == Simulators::SimulationType::kDensityMatrix ? 2 : 1;
            if (qubits <= (std::numeric_limits<size_t>::digits - 5) / bitsPerQubit)
            {
                const size_t elements = size_t{1} << (bitsPerQubit * qubits);
                const size_t bytes = elements * sizeof(std::complex<double>);
                const size_t mb =
                    configuration.IsSet("trajectory_max_memory_mb") ? std::stoull(configuration.GetConfiguration("trajectory_max_memory_mb")) : 1024;
                const size_t slots = (mb << 20) / bytes;
                // Prepared executions retain a source and one shared checkpoint.
                // Without preparation each worker owns its own checkpoint too.
                const size_t memoryWorkers = prepared ? (slots > 2 ? slots - 2 : 1) : slots / 2;
                workers = std::min(workers, std::max<size_t>(1, memoryWorkers));
#ifdef _OPENMP
                // Large independent states contend for cache and memory bandwidth.
                // Reserve a small internal team instead of filling every CPU with
                // a separate state, and use no more threads in all than QCSim's
                // team for one such register (half the processors by default):
                // memory-bound work gains nothing from the second hardware
                // thread of a core, and with the default OpenMP limit filling
                // every logical CPU made these runs 1.3-1.5x slower. Logical RNG
                // streams do not depend on this choice.
                if (elements >= (size_t{1} << 20))
                {
                    threadBudget = std::min(threadBudget, std::max(1, QC::QubitRegisterCalculator<>::GetParallelTeamSize(elements)));
                    workers = std::min(workers, static_cast<size_t>(std::max(1, threadBudget / 4)));
                }
#endif
            }
            else
                workers = 1;
        }
        else if (method == Simulators::SimulationType::kStabilizer || method == Simulators::SimulationType::kMatrixProductState ||
                 method == Simulators::SimulationType::kMatrixProductOperator)
        {
            // Budget these methods like dense states where their size is bounded: a tableau
            // always, MPS and MPO under a bond-dimension cap.
            const bool mpo = method == Simulators::SimulationType::kMatrixProductOperator;
            const auto state = method == Simulators::SimulationType::kStabilizer ? Utils::RealizationState::kStabilizer
                               : mpo                                             ? Utils::RealizationState::kMatrixProductOperator
                                                                                 : Utils::RealizationState::kMatrixProductState;
            size_t bond = 0;
            if (state != Utils::RealizationState::kStabilizer)
            {
                const char *key = mpo && configuration.IsSet("matrix_product_operator_max_bond_dimension") ? "matrix_product_operator_max_bond_dimension"
                                                                                                           : "matrix_product_state_max_bond_dimension";
                if (configuration.IsSet(key))
                    bond = std::stoull(configuration.GetConfiguration(key));
            }
            const long double bytes = Utils::RealizationStateBytes(state, qubits, bond);
            if (bytes > 0 && bytes < static_cast<long double>(std::numeric_limits<size_t>::max()))
            {
                const size_t mb =
                    configuration.IsSet("trajectory_max_memory_mb") ? std::stoull(configuration.GetConfiguration("trajectory_max_memory_mb")) : 1024;
                // Tensor/tableau clones copy their saved checkpoint rather
                // than sharing it. Keep two copies per worker also for terminal
                // batches: explicit MPS measurement and a wide stabilizer tail
                // can allocate a private sampling snapshot.
                workers = Utils::LimitRealizationWorkersByMemory(workers, static_cast<size_t>(bytes), mb, prepared ? 2 : 0, 2);
            }
        }
        // shot_v1 streams do not depend on the worker count, so tiny states can
        // skip workers whose dispatch would cost more than their shots.
        if (independentShots && trajectoryMinWorkerMicros > 0)
            workers = std::min(workers, TrajectoryWorkersForWork(circuit, shots, qubits, method, executed));
        // shot_v1 can change physical job boundaries without changing any RNG
        // stream. Keep batches bounded while exposing short requests to workers.
        // Terminal batches follow the shot count alone: 256 shots each, or fewer
        // and larger ones once there would be more than 256 of them.
        const size_t blockSize = independentShots   ? std::min<size_t>(256, std::max<size_t>(1, shots / workers + (shots % workers != 0)))
                                 : terminalSampling ? std::max<size_t>(256, shots / 256 + (shots % 256 != 0))
                                                    : 256;
        const uint64_t seed = configuration.IsSet("seed") ? std::stoull(configuration.GetConfiguration("seed"))
                                                          : Simulators::GenerateRandomSeed(type, configuration.GetConfigMap());
        const auto cloneMutex = std::make_shared<std::mutex>();
        // Execute() continues the live register when no optimizer replaced it.
        // A one-shot trajectory must not clear that state or invalidate handles
        // held by callers preparing and inspecting the register around Execute().
        const bool continueExecution = !resetBeforeExecution && shots == 1 && !prepared && simulator && simulator->GetType() == type &&
                                       simulator->GetSimulationType() == method && simulator->GetNumberOfQubits() == qubits;
        if (prepared)
            prepared->SetMultithreading(false);
        if (simulator && simulator != prepared && !continueExecution)
            simulator->Clear();
        // Terminal batches sample without collapsing, and a sampler prepares its
        // state once: an MPS moves the measured sites to the head of the chain, an
        // MPO traces its suffixes, a tableau solves its distribution. The first
        // batch therefore samples the prepared state itself, and only then do the
        // others copy it, with that work done. One worker keeps sampling the
        // prepared state, reseeded for each batch, and needs no copy at all.
        const bool sampleSource = terminalSampling && prepared;
        // Such a batch only reads what the evolution deferred, the measurements:
        // each job gets those instead of the whole circuit to scan and copy.
        auto jobCircuit = circuit;
        auto jobExecuted = executed;
        if (sampleSource && !executed.empty())
            jobCircuit = circuit->RemoveExecutedOperations(jobExecuted);
        // Every job runs the same configuration, apart from its seed.
        const auto jobConfiguration = ExecutionConfiguration(type, qubits, method);
        const size_t blocks = shots / blockSize + (shots % blockSize != 0);
        workers = std::max<size_t>(1, std::min(workers, sampleSource && blocks > 1 ? blocks - 1 : blocks));
        if (workers > 1)
        {
            threadsPool.Resize(workers);
            threadsPool.SetFinishLimit(sampleSource ? shots - std::min(blockSize, shots) : shots);
        }
        // The caller's OpenMP limit is a budget shared by concurrent shot jobs.
        // std::thread workers do not inherit the caller's OpenMP ICV, so pass
        // the budget explicitly and apply it only for the lifetime of each job.
        int internalThreads = 1;
#ifdef _OPENMP
        if (!omp_in_parallel() && method != Simulators::SimulationType::kPathIntegral)
            internalThreads = static_cast<int>(std::max<size_t>(1, static_cast<size_t>(threadBudget) / workers));
#endif
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
        // Share the process limit with noisy-realization dispatchers and other
        // networks. The final participant restores the original caller budget.
        Utils::RealizationDetail::ScopedProcessOpenMPBudget processBudget(internalThreads);
#endif
        std::shared_ptr<ExecuteJob<Time>> last;
        for (size_t begin = 0, block = 0; begin < shots; ++block)
        {
            const size_t count = std::min(blockSize, shots - begin);
            auto job = std::make_shared<ExecuteJob<Time>>(jobCircuit, results, count, qubits, cbits, resultBits, type, method, resultsMutex);
            // Even a one-shot tail must remove the same deterministic prefix:
            // classical instruction positions are part of shot_v1 stream IDs.
            job->optimiseMultipleShotsExecution = independentShots || terminalSampling || count > 1 || GetOptimizeSimulator();
            // Reproducible measurement retains serial scan/reduction order;
            // gates and disjoint collapse updates may use the remaining budget.
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
            job->internalThreads = 0; // The dispatcher owns the process-wide limit.
#else
            job->internalThreads = internalThreads;
#endif
            job->allowInternalMultithreading = internalThreads > 1;
            job->network = BaseClass::getptr();
            job->onSimulatorReady = [this](const Simulators::ISimulator &actual) {
                lastFusionWidth.store(actual.GetGateFusionMaxQubits());
                lastFusionEnabled.store(actual.IsGateFusionEnabled());
            };
            job->curMaxBondDim = &curMaxBondDim;
            job->config = jobConfiguration;
            const uint64_t blockSeed = Simulators::IState::DeriveSeed(seed, block);
            job->config.SetConfiguration("seed", std::to_string(blockSeed));
            // Execute() keeps the last job's simulator, and the next run captures
            // its configuration: an unseeded network must stay unseeded.
            job->transientSeed = !configuration.IsSet("seed");
            job->randomStream = block;
            job->independentShotStreams = independentShots;
            job->trajectorySeed = Simulators::IState::DeriveSeed(seed, 0x53484f545f5631ULL);
            job->firstShot = begin;
            const bool onSource = sampleSource && (block == 0 || workers == 1);
            if (onSource)
            {
                // The streams a copy made for this batch would get.
                if (job->transientSeed)
                    prepared->SeedForExecution(blockSeed);
                else
                    prepared->SetSeed(blockSeed);
                job->optSim = prepared;
            }
            else
            {
                job->cloneSource = prepared;
                job->cloneMutex = cloneMutex;
            }
            job->executedGates = jobExecuted;
            if (continueExecution)
            {
                job->optSim = simulator;
                job->ApplyConfiguration(job->optSim);
                job->ApplyTransientSeed(job->optSim);
                job->executedGates.assign(circuit->size(), false);
            }
            last = job;
            // The batch on the prepared state finishes before any copy is taken.
            if (workers > 1 && !onSource)
                threadsPool.AddRunJob(std::move(job));
            else
                job->DoWork();
            begin += count;
        }
        if (workers > 1)
            threadsPool.WaitForFinish();
        if (!recreateIfNeeded && last)
            simulator = last->optSim;
    }

    // Rough serial cost of the shots of a statevector or density-matrix
    // trajectory batch (fixed per-shot cost plus the operations replayed after
    // the checkpoint over every amplitude), measured on QCSim; one worker per
    // trajectoryMinWorkerMicros of it.
    size_t TrajectoryWorkersForWork(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t shots, size_t qubits, Simulators::SimulationType method,
                                    const std::vector<bool> &executed) const
    {
        if (method != Simulators::SimulationType::kStatevector && method != Simulators::SimulationType::kDensityMatrix)
            return std::numeric_limits<size_t>::max();
        const size_t bitsPerQubit = method == Simulators::SimulationType::kDensityMatrix ? 2 : 1;
        if (qubits > (std::numeric_limits<size_t>::digits - 5) / bitsPerQubit)
            return std::numeric_limits<size_t>::max();
        const double elements = static_cast<double>(size_t{1} << (bitsPerQubit * qubits));
        const size_t done = static_cast<size_t>(std::count(executed.begin(), executed.end(), true));
        const double ops = static_cast<double>(circuit->size() > done ? circuit->size() - done : 1);
        const double micros = static_cast<double>(shots) * (6. + 0.4e-3 * ops * elements);
        return std::max<size_t>(1, static_cast<size_t>(micros / trajectoryMinWorkerMicros));
    }

    Configuration<Time> ExecutionConfiguration(Simulators::SimulatorType type, size_t qubits, Simulators::SimulationType method) const
    {
        auto result = configuration;
        if (type == Simulators::SimulatorType::kQCSim)
            result.SetConfiguration("reproducible_trajectory",
                                    distCirc && distCirc->HasOpsAfterMeasurements(Simulators::EvolvesExactQuantumChannels(type, method)) ? "true" : "false");
        const auto found = resolvedDistributedDevices.find(type);
        if (!Simulators::IsDistributedGpuSimulator(type) || !configuration.GetConfiguration("distributed_devices").empty() ||
            found == resolvedDistributedDevices.end())
            return result;
        auto devices = found->second;
        // MPI shard count belongs to the communicator. Explicit global-qubit
        // settings also constrain the shard count and must be validated unchanged.
        if (type == Simulators::SimulatorType::kDistGpuSim && !configuration.IsSet("distributed_global_qubits"))
        {
            size_t count = 1;
            for (char c : devices)
                if (c == ',')
                    ++count;
            size_t bits = 0;
            for (size_t n = count; n > 1; n >>= 1)
                ++bits;
            while (count > 1 && bits >= qubits)
            {
                count /= 2;
                --bits;
            }
            size_t end = 0;
            for (size_t i = 0; i < count; ++i)
            {
                end = devices.find(',', end);
                if (end == std::string::npos)
                    break;
                if (i + 1 < count)
                    ++end;
            }
            devices = devices.substr(0, end);
        }
        result.SetConfiguration("distributed_devices", devices);
        return result;
    }

    void OptimizeMPSInitialQubitsMap(std::shared_ptr<Simulators::ISimulator> &sim, std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t nrQubits) const
    {
        if ((sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductState ||
             sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator) &&
            (optimizeInitialQubitsMap || mpsOptimizeSwaps) && sim->SupportsMPSSwapOptimization() && !dcirc->HasCompositeOperations())
        {
            // an MPO site carries a two-qubit (operator) physical index, so its
            // bonds grow like those of an MPS twice as long
            const size_t routedQubits = sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator ? 2 * nrQubits : nrQubits;
            bool planned = false;
            if (mpsOptimizationQubitsNumberThreshold <= routedQubits)
            {
                const auto maxBondDimValue =
                    configuration.GetConfigurationAsInt(sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator &&
                                                                configuration.IsSet("matrix_product_operator_max_bond_dimension")
                                                            ? "matrix_product_operator_max_bond_dimension"
                                                            : "matrix_product_state_max_bond_dimension");

                if (maxBondDimValue <= 0 || static_cast<int>(mpsOptimizationBondDimensionThreshold) <= maxBondDimValue)
                {
                    planned = true;
                    // need to be sure the circuit is correctly converted
                    dcirc->ConvertForCutting(); // convert the three qubit gates
                    auto layers = dcirc->ToMultipleQubitsLayersNoClone();
                    // Finalize source ordering before preparing fusion. Routing layers
                    // are only a cost model and must never replace executable operations.
                    auto ordered = Circuits::Circuit<Time>::LayersToCircuit(layers);
                    // A circuit whose two-qubit operations form lines that do not run in index order
                    // (a line through a device) gets its qubits placed along them: no swap is then
                    // needed, so no layout can cost less and none is searched for (Circuits::ChainLine).
                    const std::vector<long long int> line =
                        optimizeInitialQubitsMap ? Circuits::ChainLine(ordered, nrQubits, sim->SwapGatesRelabelQubits()) : std::vector<long long int>();
                    // Among the orders that are the same circuit, the one that keeps the chain's work
                    // local (Circuits::ChainOrdered): QCSim's chains pay for the distance between
                    // operations once a channel has taken them off their canonical form. The distance
                    // is the one on the chain, so along the line when the qubits are placed there.
                    if (sim->GetType() == Simulators::SimulatorType::kQCSim)
                        ordered = Circuits::ChainOrdered(ordered,
                                                         sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator
                                                             ? Circuits::ChainOrder::kMixedState
                                                         : sim->SamplesQuantumChannels() ? Circuits::ChainOrder::kSampledChannels
                                                                                         : Circuits::ChainOrder::kNone,
                                                         line.empty() ? nullptr : &line);
                    dcirc->SetOperations(ordered->GetOperations());
                    sim->SetUpcomingGates(dcirc->GetOperations());
                    if (sim->IsGateFusionEnabled())
                    {
                        Circuits::Circuit<Time> routingCircuit;
                        for (const auto &op : sim->GetUpcomingRoutingOperations())
                            if (op->GetType() == Circuits::OperationType::kGate)
                                routingCircuit.AddOperation(op);
                        layers = routingCircuit.ToMultipleQubitsLayersNoClone();
                    }

                    Simulators::MPSDummySimulator dummySim(nrQubits, sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator);
                    dummySim.SetSwapGatesRelabelQubits(sim->SwapGatesRelabelQubits());
                    dummySim.setGrowthFactorGate(growthFactorGate);
                    dummySim.setGrowthFactorSwap(growthFactorSwap);

                    if (maxBondDimValue > 0)
                    {
                        const bool virtualBonds = Simulators::RoutingConfig::VirtualBondsEnabled(configuration);
                        dummySim.SetMaxBondDimension(
                            virtualBonds ? Simulators::MPSDummySimulator::EffectiveMapMaxBondDimension(
                                               nrQubits, sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator, maxBondDimValue)
                                         : maxBondDimValue);
                        dummySim.SetRoutingCostModel(virtualBonds ? maxBondDimValue : 0, Simulators::RoutingConfig::TruncationWeight(configuration));
                    }

                    if (optimizeInitialQubitsMap)
                    {
                        const auto optimalMap = line.empty() ? dummySim.ComputeOptimalQubitsMap(layers) : line;
                        sim->SetInitialQubitsMap(optimalMap);
                    }

                    if (mpsOptimizeSwaps)
                    {
                        int lookaheadDepthLocal = lookaheadDepth;

                        if (lookaheadDepthLocal == std::numeric_limits<int>::max())
                        {
                            double avgTwoQubitGatesPerLayer = 0.0;
                            for (const auto &layer : layers)
                            {
                                int twoQubitGates = 0;
                                for (const auto &op : layer->GetOperations())
                                {
                                    if (op->AffectedQubits().size() >= 2)
                                    {
                                        ++twoQubitGates;
                                    }
                                }
                                avgTwoQubitGatesPerLayer += twoQubitGates;
                            }
                            if (!layers.empty())
                                avgTwoQubitGatesPerLayer /= layers.size();

                            int lookaheadVal = static_cast<int>(4. * avgTwoQubitGatesPerLayer);
                            if (lookaheadVal > 15)
                                lookaheadVal = 15;

                            lookaheadDepthLocal = layers.size() < 8 || routedQubits <= 10 ? 0
                                                  : layers.size() < 15                    ? static_cast<int>(lookaheadVal)
                                                  : layers.size() < 25                    ? static_cast<int>(1.5 * lookaheadVal)
                                                                                          : 2 * lookaheadVal;
                        }

                        int lookaheadHeuristicDepthLocal = lookaheadDepthWithHeuristic;

                        if (lookaheadHeuristicDepthLocal == std::numeric_limits<int>::max())
                            lookaheadHeuristicDepthLocal = layers.size() < 8 || routedQubits <= 10 ? 0 : lookaheadDepthLocal - 1;

                        if (lookaheadHeuristicDepthLocal < 0)
                            lookaheadHeuristicDepthLocal = 0;

                        sim->setGrowthFactorGate(growthFactorGate);
                        sim->setGrowthFactorSwap(growthFactorSwap);
                        sim->SetUseOptimalMeetingPosition(true);
                        if (lookaheadDepth == std::numeric_limits<int>::max())
                            sim->SetAutomaticLookaheadDepth(lookaheadDepthLocal);
                        else
                            sim->SetLookaheadDepth(lookaheadDepthLocal);
                        sim->SetLookaheadDepthWithHeuristic(lookaheadHeuristicDepthLocal);
                        // The CPU window installed before layout planning has not
                        // changed. Its routing setters already refresh the plan.
                        if (sim->GetType() != Simulators::SimulatorType::kQCSim)
                            sim->SetUpcomingGates(dcirc->GetOperations());
                    }
                }
            }
            // A register of that size whose bond limit is below the one a layout is planned for still gets
            // the qubits of lines out of index order placed along them: finding the lines is one pass
            // over the circuit, the chain then makes no swap, and QCSim's chains get the order along the
            // lines, as a chain in index order has it from the noise plan. A smaller register is left
            // alone: its swaps cost less than looking for the lines.
            if (!planned && optimizeInitialQubitsMap && mpsOptimizationQubitsNumberThreshold <= routedQubits)
            {
                const auto line = Circuits::ChainLine(dcirc, nrQubits, sim->SwapGatesRelabelQubits());
                if (!line.empty())
                {
                    sim->SetInitialQubitsMap(line);
                    if (sim->GetType() == Simulators::SimulatorType::kQCSim)
                    {
                        const auto ordered = Circuits::ChainOrdered(dcirc,
                                                                    sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator
                                                                        ? Circuits::ChainOrder::kMixedState
                                                                    : sim->SamplesQuantumChannels() ? Circuits::ChainOrder::kSampledChannels
                                                                                                    : Circuits::ChainOrder::kNone,
                                                                    &line);
                        if (ordered != dcirc)
                            dcirc->SetOperations(ordered->GetOperations());
                    }
                }
            }
        }
    }

    /**
     * @brief Converts back the state from the optimized network distribution
     * mapping
     *
     * Converts back the state from the optimized network distribution mapping.
     * It's used only if there is an optimiser set, that is, there was a
     * rempapping/optimizing of the circuit to be distributed.
     */
    void ConvertBackState()
    {
        auto optimiser = controller->GetOptimiser();
        if (optimiser)
        {
            // convert the classical state results back to the expected order
            const auto &qubitsMap = optimiser->GetReverseQubitsMap();

            ConvertBackState(qubitsMap);
        }
    }

    /**
     * @brief Converts back the state using the passed qubits map
     *
     * Converts back the state using the passed qubits map.
     * It's used only if there is an optimiser set, that is, there was a
     * rempapping/optimizing of the circuit to be distributed. Or if there is an
     * execution on a single host and the passed circuit had a different mapping
     * for qubits than the host qubits.
     *
     * @param qubitsMap The map with the qubits mapping.
     */
    void ConvertBackState(const std::unordered_map<Types::qubit_t, Types::qubit_t> &qubitsMap)
    {
        // might not be the one stored in the network, might exist in the DES
        Circuits::OperationState &theClassicalState = GetState();

        theClassicalState.Remap(qubitsMap);
    }

    /**
     * @brief Converts back the results from the optimized network distribution
     * mapping
     *
     * Converts back the results from the optimized network distribution mapping.
     * It's used only if there is an optimiser set, that is, there was a
     * rempapping/optimizing of the circuit to be distributed.
     *
     * @param res The results to convert back.
     */
    void ConvertBackResults(ExecuteResults &res)
    {
        auto optimiser = controller->GetOptimiser();
        if (optimiser)
        {
            // convert the classical state results back to the expected order
            const auto &qubitsMap = optimiser->GetReverseQubitsMap();

            ConvertBackResults(res, qubitsMap);
        }
    }

    /**
     * @brief Converts back the results using the passed qubits map
     *
     * Converts back the results using the passed qubits map
     * It's used only if there is an optimiser set, that is, there was a
     * rempapping/optimizing of the circuit to be distributed. Or if there is an
     * execution on a single host and the passed circuit had a different mapping
     * for qubits than the host qubits.
     *
     * @param res The results to convert back.
     * @param bitsMap The map with the bits mapping.
     */
    void ConvertBackResults(ExecuteResults &res, const std::unordered_map<Types::qubit_t, Types::qubit_t> &bitsMap) const
    {
        ExecuteResults translatedRes;

        size_t numClassicalBits = 0;
        for (const auto &[q, b] : bitsMap)
            if (b >= numClassicalBits)
                numClassicalBits = b + 1;

        numClassicalBits = std::max(numClassicalBits, GetNumClassicalBits());

        for (const auto &r : res)
        {
            Circuits::OperationState translatedState(r.first);

            translatedState.Remap(bitsMap, false, numClassicalBits);
            translatedRes[translatedState.GetAllBits()] = r.second;
        }

        res.swap(translatedRes);
    }

    /**
     * @brief Map the circuit on the host.
     *
     * Map the circuit on the host.
     * It's used only if there is an execution on a single host and the passed
     * circuit had a different mapping for qubits than the host qubits.
     *
     * @param circuit The circuit to map.
     * @param hostId The id of the host to map the circuit on.
     * @return The reverse qubits/cbits map.
     */
    std::unordered_map<Types::qubit_t, Types::qubit_t> MapCircuitOnHost(const std::shared_ptr<Circuits::Circuit<Time>> &circuit, size_t hostId,
                                                                        size_t &nrQubits, size_t &nrCbits, bool useSeparateSimForHosts = false)
    {
        qubitsMapOnHost.clear();
        nrQubits = 0;
        nrCbits = 0;
        if (!circuit)
            return {};

        const auto host = std::static_pointer_cast<SimpleHost<Time>>(GetHost(hostId));
        const size_t hostNrQubits = host->GetNumQubits();

        std::unordered_map<Types::qubit_t, Types::qubit_t> reverseQubitsMap;

        if (!useSeparateSimForHosts)
        {
            // A circuit already using this host's qubit IDs still replaces the
            // previous circuit (and must be available on the first invocation).
            distCirc = circuit;
            size_t mxq = 0;
            size_t mnq = std::numeric_limits<size_t>::max();
            size_t mxb = 0;
            size_t mnb = std::numeric_limits<size_t>::max();

            for (const auto &op : circuit->GetOperations())
            {
                const auto qbits = op->AffectedQubits();
                for (auto q : qbits)
                {
                    if (q > mxq)
                        mxq = q;
                    if (q < mnq)
                        mnq = q;
                }
            }
            for (auto b : circuit->GetBits())
            {
                if (b > mxb)
                    mxb = b;
                if (b < mnb)
                    mnb = b;
            }

            if (mnq > mxq)
                mnq = 0;
            if (mnb > mxb)
                mnb = 0;

            nrQubits = mxq - mnq + 1;
            nrCbits = mxb - mnb + 1;
            if (nrCbits < nrQubits)
                nrCbits = nrQubits;

            const size_t startQubit = host->GetStartQubitId();

            if (mnq < startQubit || mxq >= startQubit + hostNrQubits)
            {
                if (nrQubits > hostNrQubits + 1) // the host has an additional 'special' qubit for the
                                                 // entanglement or other operations (like those for cutting)
                    throw std::runtime_error("Circuit does not fit on the host!");

                for (size_t i = 0; i < nrCbits; ++i)
                {
                    const size_t mapFrom = mnq + i;
                    const size_t mapTo = startQubit + i;

                    qubitsMapOnHost[mapFrom] = mapTo;
                    reverseQubitsMap[mapTo] = mapFrom;
                }

                distCirc = std::static_pointer_cast<Circuits::Circuit<Time>>(circuit->Remap(qubitsMapOnHost, qubitsMapOnHost));
            }

            return reverseQubitsMap;
        }

        distCirc = circuit->RemapToContinuous(qubitsMapOnHost, reverseQubitsMap, nrQubits, nrCbits);

        if (simulator && Simulators::IsDistributedGpuSimulator(simulator->GetType()))
        {
            // Distribution settings refer to register qubits. Keep their numbering
            // and idle wires when the network creates a smaller per-host simulator.
            // Classical results retain RemapToContinuous's independent mapping.
            if (!hostNrQubits)
                throw std::runtime_error("Circuit does not fit on a host with no qubits!");
            const size_t start = host->GetStartQubitId();
            bool fitsLocal = true, fitsGlobal = true;
            for (const auto &entry : qubitsMapOnHost)
            {
                const auto q = entry.first;
                fitsLocal = fitsLocal && q < hostNrQubits;
                fitsGlobal = fitsGlobal && q >= start && q - start < hostNrQubits;
            }
            // Auto follows the host API's already-mapped precedence. Sparse local
            // circuits in the overlap must explicitly select local indexing.
            const bool global = distributedHostQubitIndexing == "global" || (distributedHostQubitIndexing == "auto" && fitsGlobal);
            if (!(global ? fitsGlobal : fitsLocal))
                throw std::runtime_error("Circuit does not fit on the host!");
            const size_t offset = global ? start : 0;
            if (pauliStrings)
                for (const auto &pauli : *pauliStrings)
                    if (pauli.size() > hostNrQubits)
                        throw std::invalid_argument("Host Pauli strings use local indices and must fit the host");
            std::unordered_map<Types::qubit_t, Types::qubit_t> restoreQubits;
            for (const auto &[original, compact] : qubitsMapOnHost)
            {
                if (original < offset || original - offset >= hostNrQubits)
                    throw std::runtime_error("Circuit does not fit on the host!");
                restoreQubits[compact] = original - offset;
            }
            distCirc = std::static_pointer_cast<Circuits::Circuit<Time>>(distCirc->Remap(restoreQubits));
            qubitsMapOnHost.clear();
            for (size_t q = 0; q < hostNrQubits; ++q)
                qubitsMapOnHost[q] = q;
            nrQubits = hostNrQubits;
        }
        else if (pauliStrings)
        {
            // Observables can mention idle qubits absent from the circuit. Allocate
            // them in |0> and include them in the expectation remapping.
            size_t width = 0;
            for (const auto &pauli : *pauliStrings)
                width = std::max(width, pauli.size());
            for (size_t q = 0; q < width; ++q)
                if (!qubitsMapOnHost.count(q))
                    qubitsMapOnHost[q] = nrQubits++;
        }

        assert(nrQubits == qubitsMapOnHost.size());

        if (nrQubits == 0)
            nrQubits = 1;

        if (nrQubits > hostNrQubits + 1) // the host has an additional 'special' qubit for the
                                         // entanglement or other operations (like those for cutting)
            throw std::runtime_error("Circuit does not fit on the host!");

        return reverseQubitsMap;
    }

    // The backends automatic selection may use, in the order they were added,
    // and whether a caller configured them (see GetOptimizationCandidates).
    // Copying it copies all three, so derived clones keep the order.
    class OptimizationCandidates
    {
      public:
        using Pair = typename BaseClass::SimulatorPair;

        void Insert(const Pair &candidate)
        {
            if (set.insert(candidate).second)
                order.push_back(candidate);
        }

        void Erase(const Pair &candidate)
        {
            if (set.erase(candidate))
                order.erase(std::remove(order.begin(), order.end(), candidate), order.end());
        }

        void Clear()
        {
            set.clear();
            order.clear();
        }

        bool Contains(const Pair &candidate) const
        {
            return set.count(candidate) != 0;
        }

        size_t size() const
        {
            return order.size();
        }

        bool empty() const
        {
            return order.empty();
        }

        auto begin() const
        {
            return order.begin();
        }

        auto end() const
        {
            return order.end();
        }

        const typename BaseClass::SimulatorsSet &AsSet() const
        {
            return set;
        }

        const std::vector<Pair> &Ordered() const
        {
            return order;
        }

        bool configured = false; /**< Set by the Add/Remove calls, not by the constructor's default list. */

      private:
        typename BaseClass::SimulatorsSet set;
        std::vector<Pair> order;
    };

    bool optimizeSimulator = true; /**< The flag to optimize the simulator. */
    bool costModelEnabled = true;  /**< SetCostModelEnabled. */
    OptimizationCandidates simulatorsForOptimizations;

    int lastGpuDevice = -1;
    std::unordered_map<std::string, std::string> lastDistributedPlacement; /**< GetLastDistributedPlacement. */
    std::atomic<unsigned> lastFusionWidth{0};
    std::atomic<bool> lastFusionEnabled{false};
    Simulators::SimulatorType lastSimulatorType = Simulators::SimulatorType::kQCSim;  /**< The last simulator type used. */
    Simulators::SimulationType lastMethod = Simulators::SimulationType::kStatevector; /**< The last simulation method
                                                                                         used. */

    Configuration<Time> configuration;
    std::string distributedHostQubitIndexing = "auto";
    std::unordered_map<Simulators::SimulatorType, std::string> resolvedDistributedDevices;

    size_t maxSimulators = QC::QubitRegisterCalculator<>::GetNumberOfThreads(); /**< The maximum number of simulators that can be
                                                                                   used in the network. */
    double trajectoryMinWorkerMicros = 100.; /**< Estimated serial work per trajectory worker, see SetTrajectoryMinWorkerMicros. */

    Circuits::OperationState classicalState;           /**< The classical state of the network. */
    std::shared_ptr<Simulators::ISimulator> simulator; /**< The quantum computing simulator for the network. */

    std::shared_ptr<Circuits::Circuit<Time>> distCirc; /**< The distributed circuit. */

    std::shared_ptr<IController<Time>> controller; /**< The controller for the network. */
    // TODO: depending on the network topology, we will have adiacency lists, etc.
    // or simply a vector of hosts for a totally connected network (or where the
    // communication details do not matter so much)
    std::vector<std::shared_ptr<IHost<Time>>> hosts; /**< The hosts in the network. */

    std::unique_ptr<Estimators::SimulatorsEstimatorInterface<Time>> simulatorsEstimator; /**< The simulators estimator. */

  private:
    Utils::ThreadsPool<ExecuteJob<Time>> threadsPool; /**< The threads pool for the execution of the circuits. */

  protected:
    // Only Execute() preserves a prepared input state. Derived shot loops must
    // honor this independently of retaining the final simulator for queries.
    bool resetBeforeExecution = true;

    // Derived execution loops must retain the final state for state queries.
    bool recreateIfNeeded = true; /**< The flag to recreate the simulator if needed. */

    // The nrQubits the network simulator was created with (0: the whole
    // network). Recreating it after an execution, cloning the network or
    // dropping a configuration option keeps this width, so a small backend
    // placeholder never grows into a full dense register nobody uses.
    size_t simulatorQubits = 0;

  private:
    std::unordered_map<Types::qubit_t, Types::qubit_t> qubitsMapOnHost; /**< The map with the qubits mapping when executing on a
                                                                           host. Relevant only when computing expectation values.
                                                                         */
    const std::vector<std::string> *pauliStrings = nullptr;             /**< Set to the vector of pauli strings if computing the
                                                                           expectation values. */

    bool optimizeInitialQubitsMap = true;              /**< The flag to optimize the initial
                                                           qubits mapping. */
    bool mpsOptimizeSwaps = true;                      /**< The flag to optimize the swaps in MPS */
    size_t mpsOptimizationBondDimensionThreshold = 32; /**< The bond dimension threshold for using MPS optimization. */
    size_t mpsOptimizationQubitsNumberThreshold = 12;  /**< The qubits number threshold for using MPS optimization. */

    int lookaheadDepth = std::numeric_limits<int>::max();              /**< The lookahead depth for MPS swap
                                                                          optimization. */
    int lookaheadDepthWithHeuristic = std::numeric_limits<int>::max(); /**< The
              lookahead depth with heuristic for MPS swap optimization. */

    double growthFactorSwap = 1.;
    double growthFactorGate = 0.7;
    size_t curMaxBondDim = 0;
};

} // namespace Network

#endif // !_SIMPLE_NETWORK_H_
