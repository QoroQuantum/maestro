/**
 * @file State.h
 * @ingroup simulators
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The State interface for a simulator.
 *
 * This interface is used to define the state of a simulator.
 * Exposes functions to initialize, configure, allocate qubits, measure, apply
 * reset on qubits, and obtain probabilities. Also allows to register observers
 * that will be notified when the state changes.
 *
 * The interface could be used to pass around the state of a simulator, if
 * operations on the state are not required.
 */

#pragma once

#ifndef _SIMULATOR_STATE_H_
#define _SIMULATOR_STATE_H_

#include <Eigen/Eigen>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef NO_QISKIT_AER
#include "framework/linalg/vector.hpp"
#endif

#include "../../Utils/RandomStream.h"
#include "../Core/PauliString.h"
#include "../Core/QuantumChannel.h"
#include "../TensorNetworks/TensorQueries.h"
#include "SimulatorObserver.h"

namespace Circuits
{
template <typename Time> class IOperation;
}

namespace Simulators
{

/**
 * @class avoid_init_allocator
 * @brief An allocator that avoids initializing the allocated memory.
 *
 * This allocator is used to avoid initializing the allocated memory.
 * It is used to avoid initializing the memory when the state is initialized
 * with a vector of amplitudes. std::vector will initialize the memory with 0,
 * which is not needed when the values in the vector are set right after the
 * allocation.
 *
 * @sa IState::InitializeState
 */
/*
template<class T> class avoid_init_allocator : public std::allocator<T>
{
public:
        using std::allocator<T>::allocator;

        template <class U, class... Args> void construct(U*, Args&&...) {}
};
*/

/**
 * @enum SimulatorType
 * @brief The type of simulator.
 */
enum class SimulatorType : int
{
#ifndef NO_QISKIT_AER
    kQiskitAer, /**< qiskit aer simulator type */
#endif
    kQCSim, /**< qcsim simulator type */
#ifndef NO_QISKIT_AER
    kCompositeQiskitAer, /**< composite qiskit aer simulator type */
#endif
    kCompositeQCSim, /**< composite qcsim simulator type */
    kGpuSim,         /**< gpu simulator type */
    kQuestSim,       /**< quest simulator type */
    kDistGpuSim,     /**< state distributed across local GPUs */
    kDistMpiGpuSim   /**< state distributed across MPI ranks/GPUs */
};

inline bool IsDistributedGpuSimulator(SimulatorType type)
{
    return type == SimulatorType::kDistGpuSim || type == SimulatorType::kDistMpiGpuSim;
}

inline bool IsGpuSimulator(SimulatorType type)
{
    return type == SimulatorType::kGpuSim || IsDistributedGpuSimulator(type);
}

/**
 * @enum SimulationType
 * @brief The type of simulation.
 */
enum class SimulationType : int
{
    kStatevector,           /**< statevector simulation type */
    kMatrixProductState,    /**< matrix product state simulation type */
    kStabilizer,            /**< Clifford gates simulation type */
    kTensorNetwork,         /**< Tensor network simulation type */
    kPauliPropagator,       /**< Pauli propagator simulation type */
    kExtendedStabilizer,    /**< Extended stabilizer simulation type */
    kPathIntegral,          /**< Path integral simulation type */
    kDensityMatrix,         /**< Density matrix simulation type */
    kMatrixProductOperator, /**< Matrix product operator simulation type */
    kOther                  /**< other simulation type, could occur for the aer simulator, which
                               also has unitary and superop methods */
};

// Whether a backend evolves the ensemble through exact channels: an unrecorded
// reset or a Kraus channel is then one deterministic operation, not a sampled
// branch. A density matrix and an MPO hold the mixed state itself. The QCSim
// Pauli propagator holds no state at all: it applies the adjoint channel to
// the observable it propagates, which yields the same ensemble averages,
// probabilities and samples. This is ISimulator::SupportsQuantumChannels() of
// the simulator the pair creates, for code that has only the pair.
inline bool EvolvesExactQuantumChannels(SimulatorType type, SimulationType method)
{
#ifndef NO_QISKIT_AER
    if (type == SimulatorType::kQiskitAer)
        return method == SimulationType::kDensityMatrix;
#endif
    if (type == SimulatorType::kQCSim && method == SimulationType::kPauliPropagator)
        return true;
    return (type == SimulatorType::kQCSim || type == SimulatorType::kGpuSim) &&
           (method == SimulationType::kDensityMatrix || method == SimulationType::kMatrixProductOperator);
}

// Whether a backend takes a quantum channel as one sampled Kraus branch: a
// pure-state backend that reads the branch probabilities from its state and
// renormalizes, as it does for a measurement. Each execution is then one
// trajectory, and averaging trajectories reproduces the channel. This is
// ISimulator::SamplesQuantumChannels() of the simulator the pair creates.
inline bool SamplesQuantumChannels(SimulatorType type, SimulationType method)
{
#ifndef NO_QISKIT_AER
    // Aer samples its Kraus instruction on the statevector and the matrix product state.
    if (type == SimulatorType::kQiskitAer)
        return method == SimulationType::kStatevector || method == SimulationType::kMatrixProductState;
#endif
    // A composite is made of statevectors of its backend, and samples where they do.
    if (type == SimulatorType::kCompositeQCSim)
        return method == SimulationType::kStatevector;
#ifndef NO_QISKIT_AER
    if (type == SimulatorType::kCompositeQiskitAer)
        return method == SimulationType::kStatevector;
#endif
    return type == SimulatorType::kQCSim &&
           (method == SimulationType::kStatevector || method == SimulationType::kMatrixProductState || method == SimulationType::kPathIntegral);
}

class IState;

// A shot planner can run before the selected simulator has allocated a register.
struct ShotPrefixPolicy
{
    bool exactChannels;
    bool contiguous;
    bool contiguousWithDrawnChannels = false;

    // QCSim's native default. At this tolerance measurement-only circuits can
    // share independent gates across shots. Coarser pruning preserves the
    // order of every collapse. Sampled channels retain their ordering guard
    // whenever pruning is enabled.
    static constexpr double kDefaultPathIntegralThreshold = 1e-15;

    static ShotPrefixPolicy ForPathIntegral(double threshold)
    {
        return {false, threshold > kDefaultPathIntegralThreshold, threshold > 0.};
    }

    static ShotPrefixPolicy FromConfiguration(SimulatorType type, SimulationType method, const std::unordered_map<std::string, std::string> &options,
                                              std::optional<double> pathIntegralThreshold = std::nullopt)
    {
        if (type != SimulatorType::kQCSim || method != SimulationType::kPathIntegral)
            return {EvolvesExactQuantumChannels(type, method), false};
        if (pathIntegralThreshold)
            return ForPathIntegral(*pathIntegralThreshold);
        const auto value = options.find("path_integral_threshold");
        return ForPathIntegral(value == options.end() ? kDefaultPathIntegralThreshold : std::stod(value->second));
    }
};

// The probability of every basis state, also where AllProbabilities is not defined (defined below).
inline std::vector<double> BasisProbabilities(IState &state);

/**
 * The number of basis states of a register, 2^qubits, where a vector of that
 * many values (probabilities, or amplitudes with Value = std::complex<double>)
 * can exist; otherwise std::length_error. Every reader of the whole
 * distribution or statevector checks it before listing: 1 << 64 is undefined
 * (it listed one basis state, which passed for the distribution), a count past
 * what a vector holds cannot be listed, only appended to without end, and an
 * allocator that sizes it in bytes wraps.
 */
template <class Value = double> inline size_t ListableBasisStateCount(size_t qubits)
{
    if (qubits >= static_cast<size_t>(std::numeric_limits<size_t>::digits) || (size_t{1} << qubits) > std::vector<Value>().max_size())
        throw std::length_error("The basis states of a register of " + std::to_string(qubits) + " qubits cannot all be listed");
    return size_t{1} << qubits;
}

/**
 * @class IState
 * @brief Interface class for a quantum computing simulator state.
 *
 * Use this interface if only the state of the simulator is required.
 * @sa ISimulator
 */
class IState
{
  public:
    /**
     * @brief Virtual destructor.
     *
     * Since this is a base class, the destructor should be virtual.
     */
    virtual ~IState() = default;

    /** Seed every random stream owned by this simulator. */
    virtual void SetSeed(uint64_t seed)
    {
        SeedAuxiliaryRng(seed);
        const std::string value = std::to_string(seed);
        Configure("seed", value.c_str());
    }

    /**
     * Seed every random stream for the current execution without configuring a
     * seed. An unseeded simulator stays unseeded: its configuration does not
     * report this seed and reinitializing it draws fresh entropy. Backends that
     * cannot separate the two configure the seed, as SetSeed does.
     */
    virtual void SeedForExecution(uint64_t seed)
    {
        SetSeed(seed);
    }

    /**
     * @brief Uniform random number in [0, 1) from the auxiliary stream.
     *
     * Used for classical post-processing that happens during execution, such as
     * readout-error flips applied when a measurement writes its bit. Each
     * simulator instance owns its own stream, so the per-job clones in the
     * multi-shot thread pool never share state. Deterministic after SetSeed or
     * Configure("seed") on the built-in backends.
     */
    double RandomUniform()
    {
        return auxUniform(auxRng);
    }

    /**
     * @brief Uniform random number in [0, 1) from the channel-branch stream.
     *
     * A backend that samples quantum channels draws their branches from this
     * stream, which is seeded with the simulator and separate from the
     * measurement and readout streams: adding a channel to a circuit does not
     * shift the measurement outcomes of the operations around it. A backend
     * that leaves a draw to the engine it wraps cannot promise that: the Aer
     * adapter draws Pauli and unitary-mixture branches here, but the branches
     * that need the state are drawn by Aer's apply_kraus from the stream that
     * also draws Aer's measurement outcomes (AerState::ApplyQuantumChannel).
     */
    double ChannelUniform()
    {
        return channelUniform(channelRng);
    }

    static uint64_t DeriveSeed(uint64_t seed, uint64_t stream)
    {
        return Utils::RandomStream::Derive(seed, stream);
    }

    /**
     * @brief Initializes the state.
     *
     * This function is called when the simulator is initialized.
     * Call it after the qubits allocation.
     * @sa IState::AllocateQubits
     */
    virtual void Initialize() = 0;

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
    virtual void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) = 0;

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
    virtual void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) = 0;
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
    virtual void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) = 0;

    /**
     * @brief Initializes the state to a computational basis state.
     *
     * Call it only on a non-initialized state, it allocates the qubits and
     * initializes the state itself. Every simulator supports this: backends
     * with a direct primitive (density matrix, matrix product operator, matrix
     * product state, statevector) use it; ISimulator provides a generic
     * fallback (reset to |0...0>, then apply X on every set bit) for the rest.
     *
     * Don't use it for more than 64 qubits, as the basis state is packed in a
     * single Types::qubit_t; use the std::vector<bool> overload instead, which
     * the matrix product operator and matrix product state backends (the only
     * ones that can scale that far) support natively.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param basisState The computational basis state, bit i selects qubit i.
     */
    virtual void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState)
    {
        (void)num_qubits;
        (void)basisState;
        throw std::runtime_error("This simulator does not support initialization to a computational "
                                 "basis state");
    }

    /**
     * @brief Initializes the state to a computational basis state.
     *
     * Same as the Types::qubit_t overload, but the basis state is given as one
     * bool per qubit so it is not limited to 64 qubits.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param basisState The computational basis state, entry i selects qubit i.
     */
    virtual void InitializeToBasisState(size_t num_qubits, const std::vector<bool> &basisState)
    {
        (void)num_qubits;
        (void)basisState;
        throw std::runtime_error("This simulator does not support initialization to a computational "
                                 "basis state");
    }

    /**
     * @brief Initializes the state to a classical mixture of computational
     * basis states.
     *
     * Sets the state to rho = sum_k weights[k] |states[k]><states[k]|. Call it
     * only on a non-initialized state, it allocates the qubits and initializes
     * the state itself. Weights are normalized so the trace is 1; only backends
     * that can represent a mixed state support this - currently the density
     * matrix and matrix product operator backends. Other backends throw, and
     * there is no generic fallback: unlike a basis state, a mixture cannot be
     * reached with unitary gates alone.
     *
     * Don't use it for more than 64 qubits, as each basis state is packed in a
     * single Types::qubit_t; use the std::vector<bool>-keyed overload instead
     * for the matrix product operator backend, which can scale that far.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param mixture The mixture, as pairs of (basis state, weight).
     */
    virtual void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<Types::qubit_t, double>> &mixture)
    {
        (void)num_qubits;
        (void)mixture;
        throw std::runtime_error("This simulator does not support initialization to a mixture of "
                                 "computational basis states");
    }

    /**
     * @brief Initializes the state to a classical mixture of computational
     * basis states.
     *
     * Same as the Types::qubit_t-keyed overload, but each basis state is given
     * as one bool per qubit so it is not limited to 64 qubits. Currently only
     * the matrix product operator backend supports this.
     *
     * @param num_qubits The number of qubits to initialize the state with.
     * @param mixture The mixture, as pairs of (basis state, weight).
     */
    virtual void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<std::vector<bool>, double>> &mixture)
    {
        (void)num_qubits;
        (void)mixture;
        throw std::runtime_error("This simulator does not support initialization to a mixture of "
                                 "computational basis states");
    }

    /**
     * @brief Just resets the state to 0.
     *
     * Does not destroy the internal state, just resets it to zero (as a 'reset'
     * op on each qubit would do).
     */
    virtual void Reset() = 0;

    /**
     * @brief Returns if the simulator supports MPS swap optimization.
     *
     * Used to check if the simulator supports MPS swap optimization.
     * @return True if the simulator supports MPS swap optimization, false
     * otherwise.
     */
    virtual bool SupportsMPSSwapOptimization() const
    {
        return false;
    }

    // Named SWAP gates exchange logical labels without moving tensor sites.
    // Internal routing swaps still require tensor contractions.
    virtual bool SwapGatesRelabelQubits() const
    {
        return false;
    }

    // True when external meeting-position lookahead can use an upcoming list.
    // Requires a live MPS/MPO, enabled lookahead and an authoritative qubit map.
    virtual bool IsRoutingLookaheadEnabled() const
    {
        return false;
    }

    /**
     * @brief Sets the initial qubits map, if possible.
     *
     * This will do nothing for most simulators, but for the MPS simulator it will
     * set the initial qubits if it supports it - that is, for qcsim and the gpu
     * simulator it can set the mapping of the qubits to the positions in the
     * chain, which can be used to optimize the swapping cost.
     */
    virtual void SetInitialQubitsMap(const std::vector<long long int> &initialMap)
    {
    }

    /**
     * @brief Enables or disables optimal meeting position for MPS swaps.
     *
     * When enabled, the MPS simulator uses actual bond dimensions to find
     * the cheapest meeting position for non-adjacent qubit swaps instead
     * of the default heuristic.
     * Does nothing for non-MPS simulators.
     */
    virtual void SetUseOptimalMeetingPosition(bool /*enable*/)
    {
    }

    /**
     * @brief Sets the lookahead depth for swap optimization.
     *
     * Controls how many upcoming 2-qubit gates are considered when
     * choosing the swap meeting position.  0 means no lookahead
     * (immediate cost only).  Only effective for MPS simulators.
     * Requires upcoming gates to be supplied via SetUpcomingGates.
     */
    virtual void SetLookaheadDepth(int /*depth*/)
    {
    }

    // Network execution calls this after computing the initial layout and
    // resolving its automatic depth. Explicit SetLookaheadDepth calls retain
    // their original meaning, including zero for immediate-cost routing.
    virtual void SetAutomaticLookaheadDepth(int depth)
    {
        SetLookaheadDepth(depth);
    }

    /**
     * @brief Sets the lookahead depth for swap optimization.
     *
     * Controls how many upcoming 2-qubit gates are considered when
     * choosing the swap meeting position.  0 means no lookahead
     * (immediate cost only).  Only effective for MPS simulators.
     * Requires upcoming gates to be supplied via SetUpcomingGates.
     * This value sets a number of gates to lookahead without much cost increase.
     */
    virtual void SetLookaheadDepthWithHeuristic(int /*depth*/)
    {
    }

    /**
     * @brief Supplies upcoming gates for lookahead swap optimization.
     *
     * The simulator uses these to evaluate swap costs for future gates
     * when choosing where to meet.  Only effective for MPS simulators
     * with lookahead depth > 0.
     */
    virtual void SetUpcomingGates(const std::vector<std::shared_ptr<Circuits::IOperation<double>>> & /*gates*/)
    {
    }

    // Prepared interactions used by initial-layout optimization. An active
    // routing plan may retain its complete indexed window during execution;
    // this is not a remaining-operations/suffix API. Inactive mid-execution
    // views are empty. Empty means the caller should retain its circuit view.
    virtual const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &GetUpcomingRoutingOperations() const
    {
        static const std::vector<std::shared_ptr<Circuits::IOperation<double>>> empty;
        return empty;
    }

    /**
     * @brief Returns the gates counter.
     *
     * Usually does nothing, except for MPS simulators that support swap
     * optimization.
     *
     * @return The number of gates executed in the circuit.
     */
    virtual long long int GetGatesCounter() const
    {
        return 0;
    }

    /**
     * @brief Sets the gates counter.
     *
     * Usually does nothing, except for MPS simulators that support swap
     * optimization.
     *
     * @param counter The position in the circuit from where the execution should
     * continue.
     */
    virtual void SetGatesCounter(long long int /*counter*/)
    {
    }

    /**
     * @brief Increments the gates counter.
     *
     * Usually does nothing, except for MPS simulators that support swap
     * optimization. Increments the position in the circuit from where the
     * execution should continue. Useful for classically controlled gates, for the
     * case when the controlled gate is not executed.
     */
    virtual void IncrementGatesCounter()
    {
    }

    // the following four functions are also for MPS swaps optimizations, might be
    // removed in the future

    virtual double getGrowthFactorSwap() const
    {
        return 0.;
    }

    virtual double getGrowthFactorGate() const
    {
        return 0.;
    }

    virtual void setGrowthFactorSwap(double factor)
    {
    }

    virtual void setGrowthFactorGate(double factor)
    {
    }

    /**
     * @brief Configures the state.
     *
     * This function is called to configure the simulator.
     * Currently only aer supports configuration, qcsim will gracefully ignore
     * this.
     * @param key The key of the configuration option.
     * @param value The value of the configuration.
     */
    virtual void Configure(const char *key, const char *value) = 0;

    /**
     * @brief Returns configuration value.
     *
     * This function is called get a configuration value.
     * @param key The key of the configuration value.
     * @return The configuration value as a string.
     */
    virtual std::string GetConfiguration(const char *key) const = 0;

    /**
     * @brief Allocates qubits.
     *
     * This function is called to allocate qubits.
     * @param num_qubits The number of qubits to allocate.
     * @return The index of the first qubit allocated.
     */
    virtual size_t AllocateQubits(size_t num_qubits) = 0;

    /**
     * @brief Returns the number of qubits.
     *
     * This function is called to obtain the number of the allocated qubits.
     * @return The number of qubits.
     */
    virtual size_t GetNumberOfQubits() const = 0;

    /**
     * @brief Clears the state.
     *
     * Sets the number of allocated qubits to 0 and clears the state.
     * After this qubits allocation is required then calling
     * IState::AllocateQubits in order to use the simulator.
     */
    virtual void Clear() = 0;

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
    virtual size_t Measure(const Types::qubits_vector &qubits) = 0;

    /**
     * @brief Performs a measurement on the specified qubits.
     *
     * @param qubits A vector with the qubits to be measured.
     * @return The outcome of the measurements
     */
    virtual std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) = 0;

    /**
     * @brief Performs a reset of the specified qubits.
     *
     * Measures the qubits and for those that are 1, applies X on them
     * @param qubits A vector with the qubits to be reset.
     */
    virtual void ApplyReset(const Types::qubits_vector &qubits) = 0;

    /**
     * @brief Returns whether this state retains channel ensembles directly.
     *
     * This is true for QCSim density-matrix/MPO states and Aer density-matrix
     * states. MPO results remain subject to configured bond truncation.
     * Statevectors, MPS and the path integral can sample Kraus trajectories.
     */
    virtual bool SupportsQuantumChannels() const
    {
        return false;
    }

    /**
     * @brief Returns whether ApplyQuantumChannel applies one sampled branch.
     *
     * True for a pure-state backend that draws a Kraus operator with the
     * probability its state gives it and renormalizes. The state after the
     * call is one trajectory of the channel, not the ensemble, so a state
     * that has passed through a channel must not serve several shots.
     */
    virtual bool SamplesQuantumChannels() const
    {
        return false;
    }

    // A gate on a disjoint wire may still depend on a preceding collapse or
    // conditional gate when the simulator prunes the whole state per operation.
    virtual ShotPrefixPolicy GetShotPrefixPolicy() const
    {
        return {SupportsQuantumChannels(), false};
    }

    /**
     * @brief Applies a local CPTP quantum channel to one or two qubits.
     *
     * The order of target qubits has the same matrix convention as
     * ApplyGenericTwoQubitGate: targets[0] is the least-significant local basis
     * bit. The backend applies the ensemble map or samples one trajectory
     * according to its channel capabilities.
     */
    virtual void ApplyQuantumChannel(const Types::qubits_vector &targets, const QuantumChannel &channel)
    {
        (void)targets;
        (void)channel;
        throw std::runtime_error("This simulator does not support exact quantum-channel evolution");
    }

    /** @brief Returns whether LocalDensityMatrix is available. */
    virtual bool SupportsLocalDensityMatrix() const
    {
        return false;
    }

    /**
     * @brief Reduced density matrix of one to three qubits of a pure state.
     *
     * qubits[0] is the least-significant bit of the row and column index. The
     * state is not changed. Every probability a local operation on those
     * qubits needs follows from it: for a Kraus operator E, tr(E rho E^dagger).
     */
    virtual Eigen::MatrixXcd LocalDensityMatrix(const Types::qubits_vector &qubits)
    {
        (void)qubits;
        throw std::runtime_error("This simulator does not provide local reduced density matrices");
    }

    /** @brief Returns whether ApplyLocalMapNorm is available. */
    virtual bool SupportsLocalMapNorm() const
    {
        return false;
    }

    /**
     * @brief Applies a matrix to one to three qubits of a pure state and returns the squared norm of the result.
     *
     * The matrix need not be unitary; qubits[0] is the least-significant bit
     * of its index. The state is left as the matrix leaves it, not
     * renormalized, and the norm comes from the pass that applies the matrix.
     * A caller that applies, on speculation, what a block of channels does
     * when nothing happens reads from it how probable that was.
     */
    virtual double ApplyLocalMapNorm(const Types::qubits_vector &qubits, const Eigen::MatrixXcd &matrix)
    {
        (void)qubits;
        (void)matrix;
        throw std::runtime_error("This simulator does not apply local maps with their norm");
    }

    /** @brief Returns whether ApplySuperoperator is available. */
    virtual bool SupportsSuperoperators() const
    {
        return false;
    }

    /**
     * @brief Applies a local linear map to a mixed state, in superoperator form.
     *
     * Every local density block, indexed row + d * column with targets[0] the
     * least-significant bit of both and d = 2 or 4, is multiplied by the 4x4 or
     * 16x16 matrix. For Kraus operators E_k that matrix is sum_k conj(E_k) (x)
     * E_k (QuantumChannel::Superoperator), and for a unitary U it is conj(U)
     * (x) U. Maps applied one after another multiply, later ones on the left,
     * which is how a gate, its noise and the gates around them become one pass.
     *
     * The matrix is trusted: only its shape is checked.
     */
    virtual void ApplySuperoperator(const Types::qubits_vector &targets, const Eigen::MatrixXcd &superoperator)
    {
        (void)targets;
        (void)superoperator;
        throw std::runtime_error("This simulator does not apply superoperators");
    }

    /** Probability of a full q0-first bitstring, including registers wider than a machine word. */
    virtual double ProbabilityBits(const std::vector<bool> &)
    {
        throw std::runtime_error("This simulator does not support ProbabilityBits");
    }

    /** MPO extensions: bit vectors are q0-first; matrices use logical basis
     * order. Element queries return the raw operator. Dense output and complex
     * Pauli expectations are trace-normalized unless normalized is false.
     * ApplyOperator computes A rho A^dagger; normalize also restores unit trace.
     */
    virtual std::complex<double> DensityMatrixElementBits(const std::vector<bool> &, const std::vector<bool> &) const
    {
        throw std::runtime_error("This simulator does not support DensityMatrixElementBits");
    }

    virtual Eigen::MatrixXcd GetDensityMatrix(bool = true) const
    {
        throw std::runtime_error("This simulator does not support GetDensityMatrix");
    }

    virtual std::complex<double> ExpectationValueComplex(const std::string &, bool = true) const
    {
        throw std::runtime_error("This simulator does not support ExpectationValueComplex");
    }

    // Same normalization and validation as the scalar complex query. Backends
    // may share contractions; other backends retain their scalar behavior.
    virtual std::vector<std::complex<double>> ExpectationValuesComplex(const std::vector<std::string> &paulis, bool normalized = true) const
    {
        std::vector<std::complex<double>> values;
        values.reserve(paulis.size());
        for (const auto &pauli : paulis)
            values.push_back(ExpectationValueComplex(pauli, normalized));
        return values;
    }

    virtual void ApplyOperator(const Types::qubits_vector &, const Eigen::MatrixXcd &, bool = false)
    {
        throw std::runtime_error("This simulator does not support ApplyOperator");
    }

    virtual void MoveAtBeginningOfChain(const Types::qubits_vector &)
    {
        throw std::runtime_error("This simulator does not support MoveAtBeginningOfChain");
    }

    /** Mixed-state diagnostics. Implemented by density-matrix and MPO backends.
     */
    virtual std::complex<double> DensityMatrixTrace() const
    {
        throw std::runtime_error("This simulator does not expose a density-matrix trace");
    }

    virtual double DensityMatrixPurity() const
    {
        throw std::runtime_error("This simulator does not expose density-matrix purity");
    }

    virtual std::complex<double> DensityMatrixTraceOfSquare() const
    {
        throw std::runtime_error("This simulator does not expose Tr(rho^2)");
    }

    virtual std::complex<double> DensityMatrixOverlap(const IState &) const
    {
        throw std::runtime_error("This simulator does not support density-matrix overlap");
    }

    virtual double DensityMatrixHermiticityResidual() const
    {
        throw std::runtime_error("This simulator does not expose a Hermiticity residual");
    }

    virtual bool IsDensityMatrixHermitian(double = 1e-10) const
    {
        throw std::runtime_error("This simulator does not expose a Hermiticity test");
    }

    virtual Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &) const
    {
        throw std::runtime_error("This simulator does not support partial trace");
    }

    virtual double FidelityWithStatevector(const Eigen::VectorXcd &) const
    {
        throw std::runtime_error("This simulator does not support mixed-state fidelity");
    }

    virtual void RestoreDensityMatrixTrace()
    {
        throw std::runtime_error("This simulator cannot restore density-matrix trace");
    }

    virtual void HermitizeDensityMatrix()
    {
        throw std::runtime_error("This simulator cannot hermitize its density matrix");
    }

    /** Compress a matrix-product state or operator to its configured limits. */
    virtual void Trim()
    {
        throw std::runtime_error("This simulator is not a matrix-product state or operator");
    }

    /** Restore the canonical form of a matrix-product state or operator. */
    virtual void ReCanonicalize()
    {
        throw std::runtime_error("This simulator is not a matrix-product state or operator");
    }

    /** Apply an arbitrary CPTP map supplied in Kraus form. */
    void ApplyKrausChannel(const Types::qubits_vector &targets, const QuantumChannel::KrausOperators &krausOperators)
    {
        ApplyQuantumChannel(targets, QuantumChannel(krausOperators));
    }

    /** Alias matching the channel terminology used by QCSim's dense backend. */
    void ApplyChannel(const Types::qubits_vector &targets, const QuantumChannel::KrausOperators &krausOperators)
    {
        ApplyKrausChannel(targets, krausOperators);
    }

    /** Apply an arbitrary one- or two-qubit Pauli channel. */
    void ApplyPauliChannel(const Types::qubits_vector &targets, const std::vector<double> &probabilities)
    {
        ApplyQuantumChannel(targets, QuantumChannel::Pauli(probabilities));
    }

    /** Apply a single-qubit Pauli channel specified by X/Y/Z probabilities. */
    void ApplyPauliChannel(Types::qubit_t qubit, double px, double py, double pz)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::Pauli(px, py, pz));
    }

    void ApplyBitFlipNoise(Types::qubit_t qubit, double probability)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::BitFlip(probability));
    }

    void ApplyBitPhaseFlipNoise(Types::qubit_t qubit, double probability)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::BitPhaseFlip(probability));
    }

    void ApplyPhaseFlipNoise(Types::qubit_t qubit, double probability)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::PhaseFlip(probability));
    }

    /** `Noise.h` dephasing is a stochastic phase flip, not phase damping. */
    void ApplyDephasingNoise(Types::qubit_t qubit, double probability)
    {
        ApplyPhaseFlipNoise(qubit, probability);
    }

    /** Depolarizing noise using total nonidentity-Pauli probability p. */
    void ApplyDepolarizingNoise(Types::qubit_t qubit, double errorProbability)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::Depolarizing(errorProbability));
    }

    /** Depolarizing replacement `(1-p)rho + p I/2`, fully mixed at p=1. */
    void ApplyDepolarizingMixingNoise(Types::qubit_t qubit, double mixingProbability)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::DepolarizingMixing(mixingProbability));
    }

    void ApplyAmplitudeDamping(Types::qubit_t qubit, double gamma)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::AmplitudeDamping(gamma));
    }

    /** Alias for the exact T1 amplitude-damping channel. */
    void ApplyT1Relaxation(Types::qubit_t qubit, double gamma)
    {
        ApplyAmplitudeDamping(qubit, gamma);
    }

    /** Exact T1 relaxation for a physical duration and time constant. */
    void ApplyT1RelaxationFromTime(Types::qubit_t qubit, double duration, double t1)
    {
        if (!std::isfinite(duration) || duration < 0.0)
            throw std::invalid_argument("T1-relaxation duration must be finite and nonnegative");
        if (std::isnan(t1) || t1 <= 0.0)
            throw std::invalid_argument("T1 must be positive (infinity is allowed)");
        const double gamma = std::isinf(t1) ? 0.0 : -std::expm1(-duration / t1);
        ApplyAmplitudeDamping(qubit, gamma);
    }

    /** Phase damping with coherence multiplier `sqrt(1-gamma)`. */
    void ApplyPhaseDamping(Types::qubit_t qubit, double gamma)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::PhaseDamping(gamma));
    }

    /**
     * Pure phase damping for a physical duration and T_phi.
     * gamma=1-exp(-2 duration/T_phi), so coherences decay as
     * exp(-duration/T_phi).
     */
    void ApplyPhaseDampingFromTime(Types::qubit_t qubit, double duration, double tPhi)
    {
        if (!std::isfinite(duration) || duration < 0.0)
            throw std::invalid_argument("Phase-damping duration must be finite and nonnegative");
        if (std::isnan(tPhi) || tPhi <= 0.0)
            throw std::invalid_argument("T_phi must be positive (infinity is allowed)");
        const double gamma = std::isinf(tPhi) ? 0.0 : -std::expm1(-2.0 * duration / tPhi);
        ApplyPhaseDamping(qubit, gamma);
    }

    void ApplyGeneralizedAmplitudeDamping(Types::qubit_t qubit, double gamma, double excitedStatePopulation)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::GeneralizedAmplitudeDamping(gamma, excitedStatePopulation));
    }

    void ApplyThermalRelaxation(Types::qubit_t qubit, double duration, double t1, double t2, double excitedStatePopulation = 0.0)
    {
        ApplyQuantumChannel({qubit}, QuantumChannel::ThermalRelaxation(duration, t1, t2, excitedStatePopulation));
    }

    void ApplyCorrelatedPhaseFlipNoise(Types::qubit_t qubit0, Types::qubit_t qubit1, double probability)
    {
        ApplyQuantumChannel({qubit0, qubit1}, QuantumChannel::CorrelatedPhaseFlip(probability));
    }

    void ApplyCorrelatedPhaseFlipNoise(Types::qubit_t qubit0, Types::qubit_t qubit1, double probability, double correlation)
    {
        ApplyQuantumChannel({qubit0, qubit1}, QuantumChannel::CorrelatedPhaseFlip(probability, correlation));
    }

    void ApplyTwoQubitDepolarizingNoise(Types::qubit_t qubit0, Types::qubit_t qubit1, double errorProbability)
    {
        ApplyQuantumChannel({qubit0, qubit1}, QuantumChannel::TwoQubitDepolarizing(errorProbability));
    }

    void ApplyTwoQubitDepolarizingMixingNoise(Types::qubit_t qubit0, Types::qubit_t qubit1, double mixingProbability)
    {
        ApplyQuantumChannel({qubit0, qubit1}, QuantumChannel::TwoQubitDepolarizingMixing(mixingProbability));
    }

    /**
     * @brief Returns the probability of the specified outcome.
     *
     * Use it to obtain the probability to obtain the specified outcome, if all
     * qubits are measured.
     * @sa IState::Amplitude
     * @sa IState::Probabilities
     *
     * @param outcome The outcome to obtain the probability for: a basis-state
     * index, qubit q as bit q. It must be below 2^GetNumberOfQubits();
     * otherwise std::out_of_range is thrown (see RequireBasisState). A
     * register without qubits throws std::runtime_error.
     * @return The probability of the specified outcome.
     */
    virtual double Probability(Types::qubit_t outcome) = 0;

    /**
     * @brief Returns the amplitude of the specified state.
     *
     * Use it to obtain the amplitude of the specified state.
     * @sa IState::Probability
     * @sa IState::Probabilities
     *
     * @param outcome The outcome to obtain the amplitude for, a basis-state
     * index below 2^GetNumberOfQubits(); otherwise std::out_of_range is thrown.
     * A method without amplitudes (density matrix, matrix product operator,
     * stabilizer, Pauli propagator, and QCSim's extended stabilizer and
     * tensor network) throws std::runtime_error first, whatever the index.
     * @return The amplitude of the specified outcome.
     */
    virtual std::complex<double> Amplitude(Types::qubit_t outcome) = 0;

    // Full pure-state readout, indexed by logical qubit (q0 is the low bit).
    // Backends with a bulk reader override the scalar fallback.
    virtual std::vector<std::complex<double>> GetStateVector()
    {
        if (GetSimulationType() == SimulationType::kDensityMatrix || GetSimulationType() == SimulationType::kMatrixProductOperator)
            throw std::runtime_error("Mixed states have no unique statevector");
        if (GetNumberOfQubits() == 0)
            return {};
        std::vector<std::complex<double>> result(TensorQueries::DenseSize(GetNumberOfQubits()));
        for (size_t i = 0; i < result.size(); ++i)
            result[i] = Amplitude(i);
        return result;
    }

    // <psi|O_(k-1)...O_0|psi>. Each matrix is 2x2; repeated targets are allowed.
    // This is a read-only query, including for non-Hermitian/non-unitary
    // operators.
    virtual std::complex<double> ExpectationValueOperators(const Types::qubits_vector &, const std::vector<Eigen::MatrixXcd> &)
    {
        throw std::runtime_error("This simulator does not support ExpectationValueOperators");
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
    virtual std::complex<double> ProjectOnZero() = 0;

    /**
     * @brief Returns the probabilities of all possible outcomes.
     *
     * Use it to obtain the probabilities of all possible outcomes.
     * @sa IState::Probability
     * @sa IState::Amplitude
     * @sa IState::Probabilities
     *
     * @return A vector with the probabilities of all possible outcomes.
     */
    virtual std::vector<double> AllProbabilities() = 0;

    /**
     * @brief Returns the probabilities of the specified outcomes.
     *
     * Use it to obtain the probabilities of the specified outcomes: one value
     * per outcome, in the order given, each what Probability returns for it.
     * The outcomes are basis-state indices (qubit q as bit q), not qubits; an
     * outcome of 2^GetNumberOfQubits() or more throws std::out_of_range. For
     * the distribution of some of the qubits, see MarginalProbabilities.
     * @sa IState::Probability
     * @sa IState::Amplitude
     * @sa IState::MarginalProbabilities
     *
     * @param outcomes The basis-state indices.
     * @return The probability of each outcome.
     */
    virtual std::vector<double> Probabilities(const Types::qubits_vector &outcomes)
    {
        RequireQubits(GetNumberOfQubits()); // also for no outcomes: a register without qubits has no state to query
        std::vector<double> result;
        result.reserve(outcomes.size());
        for (const Types::qubit_t outcome : outcomes)
            result.push_back(Probability(outcome));
        return result;
    }

    /**
     * @brief Returns the probability distribution of some of the qubits.
     *
     * The marginal distribution of the listed qubits, the others traced out:
     * 2^k values for k qubits, with the first listed qubit as the least
     * significant bit of the index (as the packed outcomes of SampleCounts).
     * The qubits must be distinct and below GetNumberOfQubits(); otherwise
     * std::invalid_argument, or std::out_of_range for a qubit past the
     * register, is thrown. No qubits is one value, the total probability of
     * the state (TotalProbability). A register without qubits throws
     * std::runtime_error.
     *
     * The default sums the whole distribution (BasisProbabilities: also on a
     * tensor network, whose AllProbabilities is not defined, from one
     * Probability per basis state), so it costs what that costs, and throws
     * where the register is too wide to list it (std::length_error, or the
     * backend's own error, as QCSim's stabilizer past 32 qubits), never
     * summing a short list; a backend that has the marginal directly (Qiskit
     * Aer) overrides it.
     *
     * @param qubits The qubits, the first one the least significant bit.
     * @return The 2^k probabilities.
     */
    virtual std::vector<double> MarginalProbabilities(const Types::qubits_vector &qubits)
    {
        RequireMarginalQubits(qubits);
        if (qubits.empty())
            return {TotalProbability()};
        // A reader that cannot list every basis state must not pass for a distribution with no weight.
        const size_t count = GetNumberOfQubits();
        const size_t states = ListableBasisStateCount(count);
        const std::vector<double> all = BasisProbabilities(*this);
        if (all.size() != states)
            throw std::length_error("The basis states of a register of " + std::to_string(count) + " qubits cannot all be listed");
        std::vector<double> result(size_t{1} << qubits.size(), 0.0);
        for (size_t index = 0; index < all.size(); ++index)
        {
            size_t outcome = 0;
            for (size_t k = 0; k < qubits.size(); ++k)
                if (qubits[k] < std::numeric_limits<size_t>::digits && ((index >> qubits[k]) & 1))
                    outcome |= size_t{1} << k;
            result[outcome] += all[index];
        }
        return result;
    }

    /**
     * @brief Returns the total probability of the state.
     *
     * The sum of the probabilities of every basis state, as the other
     * probability queries report them: 1 for a normalized state, and what an
     * operation that does not keep the norm leaves (a generic gate with a
     * non-unitary matrix, ApplyLocalMapNorm), since nothing renormalizes the
     * state after it. It is the one value of MarginalProbabilities({}). A
     * register without qubits throws std::runtime_error.
     *
     * The default sums the marginal of qubit 0, which on most backends needs
     * the whole distribution (so it throws std::length_error where that cannot
     * be listed). Overrides: Aer's dense methods read their own marginal of
     * qubit 0, and the tensor network contracts both weights of qubit 0, at any
     * width, over each of its independent groups of qubits; the path integral
     * sums the paths it keeps (its threshold drops some); the composite
     * multiplies its components'; the matrix product operators normalize their
     * probabilities by the trace, so they total 1 where they are defined; a
     * backend whose state cannot be left unnormalized is 1; and so is a
     * matrix product state, which restores its unit norm after a matrix that
     * moved it (NormDeviation), at any width.
     */
    virtual double TotalProbability()
    {
        RequireQubits(GetNumberOfQubits());
        const std::vector<double> first = MarginalProbabilities({0});
        return first.at(0) + first.at(1);
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
    virtual std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) = 0;

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
    virtual std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) = 0;

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
    virtual double ExpectationValue(const std::string &pauliString) = 0;

    // Backends may share contractions across this read-only batch. The default
    // retains the existing behavior for every other simulator.
    virtual std::vector<double> ExpectationValues(const std::vector<std::string> &paulis)
    {
        std::vector<double> values;
        values.reserve(paulis.size());
        for (const auto &pauli : paulis)
            values.push_back(ExpectationValue(pauli));
        return values;
    }

    /**
     * @brief Registers an observer.
     *
     * Registers an observer that will be notified when the state changes.
     * @sa ISimulatorObserver
     *
     * @param observer A smart pointer to an observer.
     */
    void RegisterObserver(const std::shared_ptr<ISimulatorObserver> &observer)
    {
        observers.insert(observer);
    }

    /**
     * @brief Unregisters an observer.
     *
     * Unegisters an observer.
     * @sa ISimulatorObserver
     *
     * @param observer A smart pointer to an observer.
     */
    void UnregisterObserver(const std::shared_ptr<ISimulatorObserver> &observer)
    {
        observers.erase(observer);
    }

    /**
     * @brief Clears all observers.
     *
     * Clears all observers.
     */
    void ClearObservers()
    {
        observers.clear();
    }

    /**
     * @brief Returns the type of simulator.
     *
     * Returns the type of simulator.
     *
     * @return The type of simulator.
     * @sa SimulatorType
     */
    virtual SimulatorType GetType() const = 0;

    /**
     * @brief Returns the type of simulation.
     *
     * Returns the type of simulation.
     *
     * @return The type of simulation.
     * @sa SimulationType
     */
    virtual SimulationType GetSimulationType() const = 0;

    /**
     * @brief Flushes the applied operations
     *
     * This function is called to flush the applied operations.
     * qcsim applies them right away, so this has no effect on it, but qiskit aer
     * does not.
     */
    virtual void Flush() = 0;

    /**
     * @brief Saves the state to internal storage.
     *
     * Saves the state to internal storage, if needed.
     * Calling this should consider as the simulator is gone to uninitialized.
     * Either do not use it except for getting amplitudes, or reinitialize the
     * simulator after calling it. This is needed only for the composite
     * simulator, for an optimization for qiskit aer.
     */
    virtual void SaveStateToInternalDestructive() = 0;

    /**
     * @brief Restores the state from the internally saved state
     *
     * Restores the state from the internally saved state, if needed.
     * This does something only for qiskit aer.
     */
    virtual void RestoreInternalDestructiveSavedState() = 0;

    /**
     * @brief Saves the state to internal storage.
     *
     * Saves the state to internal storage, if needed.
     * Calling this will not destroy the internal state, unlike the 'Destructive'
     * variant. To be used in order to recover the state after doing measurements,
     * for multiple shots executions. In the first phase, only qcsim will
     * implement this.
     */
    virtual void SaveState() = 0;

    /**
     * @brief Restores the state from the internally saved state
     *
     * Restores the state from the internally saved state, if needed.
     * To be used in order to recover the state after doing measurements, for
     * multiple shots executions. In the first phase, only qcsim will implement
     * this.
     */
    virtual void RestoreState() = 0;

    /**
     * @brief Gets the amplitude.
     *
     * Gets the amplitude, from the internal storage if needed.
     * This is needed only for the composite simulator, for an optimization for
     * qiskit aer.
     */
    virtual std::complex<double> AmplitudeRaw(Types::qubit_t outcome) = 0;

    /**
     * @brief Enable/disable multithreading.
     *
     * Enable/disable multithreading. Default is enabled.
     *
     * @param multithreading A flag to indicate if multithreading should be
     * enabled.
     */
    virtual void SetMultithreading(bool multithreading = true) = 0;

    /**
     * @brief Get the multithreading flag.
     *
     * Returns the multithreading flag.
     *
     * @return The multithreading flag.
     */
    virtual bool GetMultithreading() const = 0;

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
    virtual bool IsQcsim() const = 0;

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
    virtual Types::qubit_t MeasureNoCollapse() = 0;

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
    virtual std::vector<bool> MeasureNoCollapseMany() = 0;

    /**
     * @brief Returns the maximum bond dimension reached.
     *
     * Returns the maximum bond dimension reached during execution, if applicable
     * (mps simulator, either qcsim or gpu).
     */
    virtual size_t GetCurrentMaxBondDimension() const
    {
        return 0;
    }

    // Internal telemetry must not synchronize a pending gate after every call.
    virtual size_t GetExecutedMaxBondDimension() const
    {
        return GetCurrentMaxBondDimension();
    }

    virtual const std::unordered_map<std::string, std::string> &GetConfigMap() const = 0;

    // Configuration propagated between backends. Implementations with
    // implicit defaults can expose them in GetConfigMap while omitting them
    // here, so selecting another backend preserves its own defaults.
    virtual const std::unordered_map<std::string, std::string> &GetExplicitConfigMap() const
    {
        return GetConfigMap();
    }

    // Whether the current state admits the exact single-qubit marginal used
    // by batched terminal counts. This may change after truncation or restore.
    virtual bool CanSampleSingleQubitMarginal() const
    {
        return false;
    }

  protected:
    /**
     * Throws std::out_of_range unless `outcome` is a basis state of this
     * register: below 2^GetNumberOfQubits(). Every implementation of
     * Probability and Amplitude checks it first (AmplitudeRaw too, where the
     * state is still in place; Aer checks its saved amplitudes), so an outcome
     * past the register is the same error on every backend. (It used to be
     * 0 on some, the probability of its low bits on others, and a read past
     * the state on others again.)
     */
    void RequireBasisState(Types::qubit_t outcome) const
    {
        const size_t qubits = GetNumberOfQubits();
        RequireQubits(qubits);
        if (qubits < static_cast<size_t>(std::numeric_limits<Types::qubit_t>::digits) && (outcome >> qubits) != 0)
            throw std::out_of_range("Basis state " + std::to_string(outcome) + " is out of range for a register of " + std::to_string(qubits) + " qubits");
    }

    /**
     * The qubits of a marginal distribution (MarginalProbabilities): distinct,
     * in the register, and few enough that the 2^k values can be indexed:
     * k at most the bits of size_t less two (62 on 64-bit platforms, 30 on
     * 32-bit ones), so that size_t{1} << k is defined.
     */
    void RequireMarginalQubits(const Types::qubits_vector &qubits) const
    {
        const size_t count = GetNumberOfQubits();
        RequireQubits(count);
        constexpr size_t limit = static_cast<size_t>(std::numeric_limits<size_t>::digits) - 2;
        if (qubits.size() > limit)
            throw std::invalid_argument("A marginal distribution of more than " + std::to_string(limit) + " qubits cannot be returned");
        for (size_t k = 0; k < qubits.size(); ++k)
        {
            if (qubits[k] >= count)
                throw std::out_of_range("Qubit " + std::to_string(qubits[k]) + " is out of range for a register of " + std::to_string(count) + " qubits");
            for (size_t j = 0; j < k; ++j)
                if (qubits[j] == qubits[k])
                    throw std::invalid_argument("The qubits of a marginal distribution must be distinct");
        }
    }

    /**
     * A register without qubits holds no state to query: most backends then
     * have no state object at all, and reading it crashed.
     */
    static void RequireQubits(size_t qubits)
    {
        if (qubits == 0)
            throw std::runtime_error("The register has no qubits: there is no state to query");
    }

    /**
     * How far a generic gate's matrix can move the norm of a state: the
     * Frobenius norm of M^dagger M - I, which bounds |<psi|M^dagger M|psi> - 1|
     * for a unit psi. It rejects nothing.
     *
     * A matrix product state uses it to stay normalized. A matrix that does
     * not keep the norm leaves more than a weight on the chain: the canonical
     * form its reads of one qubit, its samples and its saves rely on no longer
     * holds. So the chain adds up the deviations of the matrices that reach
     * it (many matrices each close to unitary move the norm by their sum) and,
     * once the sum passes kNormDeviationTolerance, restores its form and its
     * unit norm, as its two-qubit updates always did. Its total probability is
     * therefore 1, and a matrix that leaves no state at all is an error.
     */
    template <class Matrix> static double NormDeviation(const Matrix &matrix)
    {
        // In the matrix's own type: the gates are fixed-size 2x2 and 4x4, which then stay off the heap.
        const auto product = (matrix.adjoint() * matrix).eval();
        return (product - std::decay_t<decltype(product)>::Identity(product.rows(), product.cols())).norm();
    }

    static constexpr double kNormDeviationTolerance = 1e-10;

    /** Shared by SetSeed and backend Configure("seed") implementations. */
    void SeedAuxiliaryRng(uint64_t seed)
    {
        auxRng.seed(DeriveSeed(seed, kAuxRngStream));
        auxUniform.reset();
        channelRng.seed(DeriveSeed(seed, kChannelRngStream));
        channelUniform.reset();
    }

    /**
     * @brief Stops notifying observers.
     *
     * Use it to stop notifying observers until Notify is called.
     */
    void DontNotify()
    {
        notifyObservers = false;
    }

    /**
     * @brief Starts notifying observers.
     *
     * Use it to allow notifying observers.
     */
    void Notify()
    {
        notifyObservers = true;
    }

    /**
     * @brief Notifies observers.
     *
     * Called when the state changes, to notify observers about it.
     * @param affectedQubits A vector with the qubits that were affected by the
     * change.
     */
    void NotifyObservers(const Types::qubits_vector &affectedQubits)
    {
        if (!notifyObservers)
            return;

        for (auto &observer : observers)
        {
            observer->Update(affectedQubits);
        }
    }

  protected:
    bool NotificationsEnabled() const noexcept
    {
        return notifyObservers;
    }

  private:
    /** Stream id for the auxiliary RNG. Large so it cannot collide with the
     *  small per-job / per-sub-simulator stream ids passed to DeriveSeed. */
    static constexpr uint64_t kAuxRngStream = 0x52454144ULL;     // 'READ'
    static constexpr uint64_t kChannelRngStream = 0x4B524155ULL; // 'KRAU'

    std::unordered_set<std::shared_ptr<ISimulatorObserver>> observers; /**< The registered observers. */
    bool notifyObservers = true;                                       /**< A flag to indicate if observers should be notified. */
    std::mt19937_64 auxRng{Utils::RandomStream::FreshSeed()};          /**< Auxiliary stream, see RandomUniform. */
    std::uniform_real_distribution<double> auxUniform{0.0, 1.0};
    std::mt19937_64 channelRng{Utils::RandomStream::FreshSeed()}; /**< Channel-branch stream, see ChannelUniform. */
    std::uniform_real_distribution<double> channelUniform{0.0, 1.0};
};

/**
 * @brief The probabilities of all the basis states of a prepared state.
 *
 * A tensor network has no bulk read: it contracts once for each basis state.
 * Every caller that returns the whole distribution reads it here (the native
 * probabilities request, and the network's ExecuteOnHostProbabilities behind
 * Python's get_probabilities), so that they answer on the same backends.
 *
 * @param state The prepared state.
 * @return The probability of each basis state, by basis index.
 */
inline std::vector<double> BasisProbabilities(IState &state)
{
    if (state.GetSimulationType() != SimulationType::kTensorNetwork)
        return state.AllProbabilities();
    std::vector<double> values(ListableBasisStateCount(state.GetNumberOfQubits()));
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = state.Probability(index);
    return values;
}

} // namespace Simulators

#endif // !_SIMULATOR_STATE_H_
