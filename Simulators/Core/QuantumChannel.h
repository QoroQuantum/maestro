/**
 * @file QuantumChannel.h
 * @ingroup simulators
 * @brief Backend-independent constructors for local CPTP quantum channels.
 */

#pragma once

#ifndef _SIMULATORS_QUANTUM_CHANNEL_H_
#define _SIMULATORS_QUANTUM_CHANNEL_H_

#include <Eigen/Eigen>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Simulators
{

/**
 * A local completely-positive, trace-preserving map.
 *
 * Matrices use Maestro's gate convention: for a target list {q0, q1}, q0 is
 * the least-significant local basis bit. Thus a product A on q0 and B on q1
 * is represented by B (x) A.
 *
 * A channel is a handle to shared immutable data: copying one, or cloning or
 * remapping an operation that holds one, copies a pointer. Forms derived from
 * the definition (the factor classification, the superoperator, the Kraus list
 * of a composed channel) are computed once on first use and shared by every
 * copy; that first use is thread-safe.
 *
 * Besides its Kraus operators a channel carries an ordered list of factors,
 * applied first to last. A factor records how one part of the channel can be
 * realized: as a Pauli mixture, a mixture of unitaries, a thermal relaxation,
 * or a general Kraus set. A channel built from Kraus operators has one factor;
 * Then() composes channels and keeps the factors of both, so that a backend can
 * draw the Pauli part of a gate error without reading its state while applying
 * the relaxation part as it must.
 */
class QuantumChannel
{
  public:
    using Matrix = Eigen::MatrixXcd;
    using KrausOperators = std::vector<Matrix>;

    /** How one factor of a channel can be realized. */
    enum class FactorKind
    {
        kPauli,          /**< Probabilistic Pauli: no state access is needed to draw it */
        kUnitaryMixture, /**< Probabilistic unitary: no state access is needed to draw it */
        kThermal,        /**< Amplitude damping or thermal relaxation of one qubit */
        kKraus           /**< General Kraus set: branch probabilities depend on the state */
    };

    /** A relaxation in the form backends lower it from. */
    struct ThermalParameters
    {
        double decay = 0.0;                  /**< 1 - exp(-duration/T1) */
        double excitedStatePopulation = 0.0; /**< Equilibrium population of |1> */
        double coherence = 1.0;              /**< Multiplier of the off-diagonal terms, exp(-duration/T2) */
    };

    struct Factor
    {
        FactorKind kind = FactorKind::kKraus;
        /** Local targets of the channel this factor acts on, in the factor's own target order. */
        std::vector<size_t> qubits;
        /** kPauli: 4^k probabilities in Pauli() order over `qubits`. kUnitaryMixture: one per operator. */
        std::vector<double> probabilities;
        /** kUnitaryMixture: the unitaries. kThermal and kKraus: Kraus operators over `qubits`. */
        KrausOperators operators;
        /** kThermal only. */
        ThermalParameters thermal;

        /** Whether the branch can be drawn without reading the state. */
        bool IsStateIndependent() const
        {
            return kind == FactorKind::kPauli || kind == FactorKind::kUnitaryMixture;
        }

        /** Whether the factor has one branch only: a mixture with one term, or one Kraus operator, which is then unitary. */
        bool IsDeterministic() const
        {
            if (kind == FactorKind::kPauli || kind == FactorKind::kUnitaryMixture)
            {
                // Stops at the second branch: circuit analyses ask this of every channel operation.
                size_t branches = 0;
                for (const double probability : probabilities)
                    if (probability > 0. && ++branches > 1)
                        return false;
                return branches == 1;
            }
            return operators.size() == 1;
        }
    };

    /**
     * A relaxation whose coherence decays at least as fast as its populations
     * (T2 <= T1) is a mixture: nothing happens, the phase flips, or the qubit
     * is reset to its thermal state. Only the reset reads the state, so a
     * backend that samples channels draws the branch first and seldom needs
     * the state at all.
     */
    struct ThermalMixture
    {
        double identity = 1.0;
        double phaseFlip = 0.0;
        double reset = 0.0;
    };

    /** The mixture a relaxation is; false for one that is none (T2 > T1, amplitude damping). */
    static bool MixtureOf(const ThermalParameters &thermal, ThermalMixture &mixture)
    {
        const double kept = 1.0 - thermal.decay;
        if (!(thermal.coherence <= kept + 1e-14))
            return false;
        mixture.reset = thermal.decay;
        mixture.phaseFlip = std::max(0.0, 0.5 * (kept - thermal.coherence));
        mixture.identity = std::max(0.0, 1.0 - mixture.reset - mixture.phaseFlip);
        return true;
    }

    /** Kraus operators of the reset of one qubit to its thermal state. */
    static KrausOperators ResetOperators(double excitedStatePopulation)
    {
        KrausOperators kraus;
        const double populations[2] = {1.0 - excitedStatePopulation, excitedStatePopulation};
        for (int to = 0; to < 2; ++to)
            if (populations[to] > 0.0)
                for (int from = 0; from < 2; ++from)
                {
                    Matrix op = Matrix::Zero(2, 2);
                    op(to, from) = std::sqrt(populations[to]);
                    kraus.emplace_back(std::move(op));
                }
        return kraus;
    }

    /** Construct and validate a CPTP channel from Kraus operators. */
    explicit QuantumChannel(KrausOperators krausOperators) : data_(MakeKrausData(std::move(krausOperators)))
    {
    }

    /**
     * One factor of the channel as a channel of its own, on the factor's
     * qubits in the factor's order. A backend that routes the parts of a
     * channel to different places (a composite simulator, whose components
     * hold different qubits) applies them one by one.
     */
    const QuantumChannel &FactorChannel(size_t index) const
    {
        std::call_once(data_->partsOnce, [this]() {
            for (const Factor &factor : GetFactors())
            {
                Factor own = factor;
                own.qubits = AllQubits(factor.qubits.size());
                data_->parts.push_back(FromFactors({std::move(own)}, factor.qubits.size()));
            }
        });
        return data_->parts.at(index);
    }

    /**
     * What is known, whatever the state, about the first operator of a factor
     * whose branch is drawn from the state. With operators that preserve the
     * trace its probability is <psi| K0^dagger K0 |psi>, which lies between
     * the smallest and the largest eigenvalue of K0^dagger K0. A backend that
     * samples channels applies K0 without reading the state first where that
     * probability is high (see FusionState::ResolveWithoutReading).
     */
    struct FirstOperator
    {
        bool tracePreserving = false;
        double low = 0.0;  // the smallest probability of operators[0]
        double high = 0.0; // the largest
    };

    const FirstOperator &FirstOperatorOf(size_t factor) const
    {
        std::call_once(data_->firstOnce, [this]() {
            for (const Factor &part : GetFactors())
            {
                FirstOperator first;
                if (!part.IsStateIndependent() && !part.operators.empty())
                {
                    const Eigen::Index dimension = part.operators.front().rows();
                    Matrix sum = Matrix::Zero(dimension, dimension);
                    for (const Matrix &op : part.operators)
                        sum.noalias() += op.adjoint() * op;
                    first.tracePreserving = (sum - Matrix::Identity(dimension, dimension)).cwiseAbs().maxCoeff() < 1e-10;
                    const Matrix gram = part.operators.front().adjoint() * part.operators.front();
                    const Eigen::SelfAdjointEigenSolver<Matrix> solver(gram, Eigen::EigenvaluesOnly);
                    first.low = std::max(solver.eigenvalues().minCoeff(), 0.0);
                    first.high = solver.eigenvalues().maxCoeff();
                }
                data_->first.push_back(first);
            }
        });
        return data_->first.at(factor);
    }

    /**
     * Kraus operators of the channel. A channel built from Kraus operators
     * returns them as given. A composed channel returns its canonical set: at
     * most 4^n operators, largest weight first.
     */
    const KrausOperators &GetKrausOperators() const
    {
        if (data_->composed)
            std::call_once(data_->krausOnce, [this]() { data_->kraus = CanonicalKraus(Superoperator(), Dimension()); });
        return data_->kraus;
    }

    size_t GetNumberOfQubits() const
    {
        return data_->numQubits;
    }

    /**
     * Probabilities of a channel built by Pauli(), in Pauli()'s index order;
     * empty for any other channel. Sampled (pure-state) noise injection uses
     * them to draw one Pauli per realization instead of applying the channel.
     */
    const std::vector<double> &GetPauliProbabilities() const
    {
        return data_->pauliProbabilities;
    }

    /**
     * The factors of the channel, applied first to last. A channel built from
     * Kraus operators is classified on first use: a Pauli mixture and a mixture
     * of unitaries are recognized whatever Kraus form they were given in.
     */
    const std::vector<Factor> &GetFactors() const
    {
        std::call_once(data_->factorsOnce, [this]() {
            if (data_->factors.empty())
                data_->factors.push_back(Classify(data_->kraus, data_->numQubits));
        });
        return data_->factors;
    }

    /** Whether every factor is a Pauli or unitary mixture, so a trajectory can draw the whole channel without reading its state. */
    bool IsStateIndependent() const
    {
        const auto &factors = GetFactors();
        return std::all_of(factors.begin(), factors.end(), [](const Factor &factor) { return factor.IsStateIndependent(); });
    }

    /**
     * Whether the channel has one branch only, a unitary. It is then a gate: every trajectory applies the same matrix,
     * nothing is drawn, and a state that has passed through it can serve any number of shots.
     */
    bool IsDeterministic() const
    {
        // Found once per channel and kept with its shared data: circuit analyses ask it of every channel operation.
        signed char known = data_->deterministic.load(std::memory_order_relaxed);
        if (known < 0)
        {
            const auto &factors = GetFactors();
            known = std::all_of(factors.begin(), factors.end(), [](const Factor &factor) { return factor.IsDeterministic(); }) ? 1 : 0;
            data_->deterministic.store(known, std::memory_order_relaxed);
        }
        return known != 0;
    }

    /**
     * The channel as a matrix on vectorized local density blocks.
     *
     * With d = 2^n and a density block indexed row + d * column, the entry at
     * (output, input) is sum_k K(output % d, input % d) * conj(K(output / d,
     * input / d)), that is sum_k conj(K) (x) K. Applying channels one after
     * another multiplies their superoperators, later ones on the left.
     */
    const Matrix &Superoperator() const
    {
        std::call_once(data_->superoperatorOnce, [this]() {
            if (!data_->composed)
            {
                data_->superoperator = KrausSuperoperator(data_->kraus);
                return;
            }
            const Eigen::Index size = static_cast<Eigen::Index>(Dimension() * Dimension());
            Matrix product = Matrix::Identity(size, size);
            for (const Factor &factor : data_->factors)
                product = FactorSuperoperator(factor, data_->numQubits) * product;
            data_->superoperator = std::move(product);
        });
        return data_->superoperator;
    }

    /** This channel followed by another on the same targets. */
    QuantumChannel Then(const QuantumChannel &next) const
    {
        std::vector<size_t> all(next.GetNumberOfQubits());
        for (size_t q = 0; q < all.size(); ++q)
            all[q] = q;
        return Then(next, all);
    }

    /**
     * This channel followed by another acting on some of its targets.
     * localQubits[k] is the target of this channel that target k of `next`
     * acts on, so a one-qubit relaxation follows a two-qubit gate error on its
     * second qubit with Then(relaxation, {1}).
     */
    QuantumChannel Then(const QuantumChannel &next, const std::vector<size_t> &localQubits) const
    {
        if (localQubits.size() != next.GetNumberOfQubits())
            throw std::invalid_argument("A composed channel needs one local qubit per target of the next channel");
        for (size_t k = 0; k < localQubits.size(); ++k)
        {
            if (localQubits[k] >= GetNumberOfQubits())
                throw std::invalid_argument("A composed channel acts outside the qubits of the channel it follows");
            if (std::find(localQubits.begin(), localQubits.begin() + static_cast<std::ptrdiff_t>(k), localQubits[k]) !=
                localQubits.begin() + static_cast<std::ptrdiff_t>(k))
                throw std::invalid_argument("The local qubits of a composed channel must be distinct");
        }
        std::vector<Factor> factors = GetFactors();
        for (Factor factor : next.GetFactors())
        {
            for (size_t &q : factor.qubits)
                q = localQubits[q];
            factors.push_back(std::move(factor));
        }
        return FromFactors(std::move(factors), GetNumberOfQubits());
    }

    /** A one-qubit channel on one target of a two-qubit channel that leaves the other target alone. */
    static QuantumChannel OnQubit(const QuantumChannel &channel, size_t localQubit, size_t numQubits = 2)
    {
        if (channel.GetNumberOfQubits() != 1 || numQubits < 1 || numQubits > 2 || localQubit >= numQubits)
            throw std::invalid_argument("OnQubit places a one-qubit channel on a target of a one- or two-qubit channel");
        std::vector<Factor> factors = channel.GetFactors();
        for (Factor &factor : factors)
            for (size_t &q : factor.qubits)
                q = localQubit;
        return FromFactors(std::move(factors), numQubits);
    }

    /**
     * A channel placed on targets of a channel of numQubits targets, which it
     * leaves alone elsewhere: localQubits[k] is the target that target k of
     * `channel` acts on. OnQubit is the one-qubit case; {1, 0} exchanges the
     * targets of a two-qubit channel. Followed by Then(), it builds a channel
     * from a list of parts, each on its own targets.
     */
    static QuantumChannel Placed(const QuantumChannel &channel, const std::vector<size_t> &localQubits, size_t numQubits = 2)
    {
        if (numQubits < 1 || numQubits > 2 || localQubits.size() != channel.GetNumberOfQubits())
            throw std::invalid_argument("A placed channel needs one target of a one- or two-qubit channel per target of its own");
        bool inPlace = localQubits.size() == numQubits;
        for (size_t k = 0; k < localQubits.size(); ++k)
        {
            if (localQubits[k] >= numQubits)
                throw std::invalid_argument("A placed channel acts outside the qubits of the channel it is placed in");
            if (std::find(localQubits.begin(), localQubits.begin() + static_cast<std::ptrdiff_t>(k), localQubits[k]) !=
                localQubits.begin() + static_cast<std::ptrdiff_t>(k))
                throw std::invalid_argument("The targets of a placed channel must be distinct");
            inPlace = inPlace && localQubits[k] == k;
        }
        if (inPlace)
            return channel;
        std::vector<Factor> factors = channel.GetFactors();
        for (Factor &factor : factors)
            for (size_t &q : factor.qubits)
                q = localQubits[q];
        return FromFactors(std::move(factors), numQubits);
    }

    /**
     * Compare two channels as MAPS, not as Kraus lists.
     *
     * The Kraus representation is not unique: {E_k} and {sum_j u_kj E_j} for any
     * unitary u describe the same channel, and a channel may legitimately be
     * built with a different number of operators (Pauli(), for instance, drops
     * zero-probability terms, so the operator count depends on the parameter
     * values). Comparing coefficient by coefficient therefore reports
     * physically identical channels as different.
     *
     * The Choi matrix sum_k vec(E_k) vec(E_k)^dagger is unique to the map, so
     * comparing Choi matrices answers the question actually being asked.
     */
    bool IsApprox(const QuantumChannel &other, double tolerance = 1e-12) const
    {
        if (tolerance < 0.0 || !std::isfinite(tolerance))
            throw std::invalid_argument("Quantum-channel comparison tolerance must be finite and nonnegative");
        if (data_ == other.data_)
            return true;
        if (GetNumberOfQubits() != other.GetNumberOfQubits())
            return false;

        const Matrix left = ChoiMatrix();
        const Matrix right = other.ChoiMatrix();
        if (left.rows() != right.rows() || left.cols() != right.cols())
            return false;
        return (left - right).cwiseAbs().maxCoeff() <= tolerance;
    }

    /**
     * Choi matrix of the channel: sum_k vec(E_k) vec(E_k)^dagger, with vec()
     * stacking the operator column by column. Two channels are the same map if
     * and only if their Choi matrices are equal.
     */
    Matrix ChoiMatrix() const
    {
        const KrausOperators &krausOperators = GetKrausOperators();
        const Eigen::Index dimension = krausOperators.front().rows();
        const Eigen::Index choiDimension = dimension * dimension;
        Matrix choi = Matrix::Zero(choiDimension, choiDimension);
        for (const Matrix &krausOperator : krausOperators)
        {
            const Eigen::Map<const Eigen::VectorXcd> vectorized(krausOperator.data(), choiDimension);
            choi.noalias() += vectorized * vectorized.adjoint();
        }
        return choi;
    }

    /** (1-p) rho + p X rho X. */
    static QuantumChannel BitFlip(double probability)
    {
        return Pauli({1.0 - CheckedProbability(probability), probability, 0.0, 0.0});
    }

    /** (1-p) rho + p Y rho Y. */
    static QuantumChannel BitPhaseFlip(double probability)
    {
        return Pauli({1.0 - CheckedProbability(probability), 0.0, probability, 0.0});
    }

    /** (1-p) rho + p Z rho Z. */
    static QuantumChannel PhaseFlip(double probability)
    {
        return Pauli({1.0 - CheckedProbability(probability), 0.0, 0.0, probability});
    }

    /**
     * Single-qubit Pauli channel with X, Y and Z error probabilities.
     * The identity probability is 1-px-py-pz.
     */
    static QuantumChannel Pauli(double px, double py, double pz)
    {
        CheckedProbability(px);
        CheckedProbability(py);
        CheckedProbability(pz);
        const double total = px + py + pz;
        if (!std::isfinite(total) || total > 1.0 + kProbabilityTolerance)
            throw std::invalid_argument("Pauli error probabilities must sum to at most one");
        return Pauli({std::max(0.0, 1.0 - total), px, py, pz});
    }

    /**
     * Arbitrary local Pauli channel.
     *
     * The array must contain 4^n probabilities. Base-4 digit i of an array
     * index selects I, X, Y, or Z on targets[i], respectively. For two targets
     * the order is therefore II, XI, YI, ZI, IX, XX, ... , ZZ.
     */
    static QuantumChannel Pauli(const std::vector<double> &probabilities)
    {
        if (probabilities.empty())
            throw std::invalid_argument("A Pauli channel must contain probabilities");

        size_t count = probabilities.size();
        size_t numQubits = 0;
        while (count > 1 && count % 4 == 0)
        {
            count /= 4;
            ++numQubits;
        }
        if (count != 1 || numQubits == 0)
            throw std::invalid_argument("A Pauli channel needs exactly 4^n probabilities for n >= 1");

        double total = 0.0;
        for (const double probability : probabilities)
        {
            CheckedProbability(probability);
            total += probability;
        }
        if (!std::isfinite(total) || std::abs(total - 1.0) > kProbabilityTolerance)
            throw std::invalid_argument("Pauli channel probabilities must sum to one");

        auto data = MakeKrausData(PauliKraus(probabilities, numQubits));
        data->pauliProbabilities = probabilities;
        data->factors.push_back(PauliFactor(probabilities, numQubits));
        return QuantumChannel(std::move(data));
    }

    /**
     * Mixture of unitaries: unitaries[i] is applied with probabilities[i].
     * The probabilities must sum to one and every matrix must be unitary.
     */
    static QuantumChannel UnitaryMixture(const std::vector<double> &probabilities, const KrausOperators &unitaries)
    {
        if (probabilities.empty() || probabilities.size() != unitaries.size())
            throw std::invalid_argument("A unitary mixture needs one probability per unitary");
        double total = 0.0;
        for (const double probability : probabilities)
            total += CheckedProbability(probability);
        if (!std::isfinite(total) || std::abs(total - 1.0) > kProbabilityTolerance)
            throw std::invalid_argument("Unitary-mixture probabilities must sum to one");

        const Eigen::Index dimension = unitaries.front().rows();
        KrausOperators kraus;
        Factor factor;
        factor.kind = FactorKind::kUnitaryMixture;
        for (size_t i = 0; i < unitaries.size(); ++i)
        {
            const Matrix &unitary = unitaries[i];
            if (unitary.rows() != dimension || unitary.cols() != dimension || dimension < 2 || !unitary.allFinite() ||
                (unitary.adjoint() * unitary - Matrix::Identity(dimension, dimension)).norm() > 1e-10 * static_cast<double>(dimension))
                throw std::invalid_argument("Every operator of a unitary mixture must be a unitary matrix of the same qubit dimension");
            if (probabilities[i] == 0.0)
                continue;
            kraus.emplace_back(std::sqrt(probabilities[i]) * unitary);
            factor.probabilities.push_back(probabilities[i]);
            factor.operators.push_back(unitary);
        }
        auto data = MakeKrausData(std::move(kraus));
        factor.qubits = AllQubits(data->numQubits);
        data->factors.push_back(std::move(factor));
        return QuantumChannel(std::move(data));
    }

    /**
     * Depolarizing channel in the NoiseModel/QCSim convention:
     * (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ).
     */
    static QuantumChannel Depolarizing(double errorProbability)
    {
        CheckedProbability(errorProbability);
        return Pauli({1.0 - errorProbability, errorProbability / 3.0, errorProbability / 3.0, errorProbability / 3.0});
    }

    /**
     * Depolarizing channel in the replacement convention:
     * (1-p)rho + p I/2. Full mixing occurs at p=1.
     */
    static QuantumChannel DepolarizingMixing(double mixingProbability)
    {
        CheckedProbability(mixingProbability);
        return Depolarizing(0.75 * mixingProbability);
    }

    /** |1> -> |0> relaxation with probability gamma. */
    static QuantumChannel AmplitudeDamping(double gamma)
    {
        CheckedProbability(gamma);
        Matrix k0 = Matrix::Zero(2, 2);
        k0(0, 0) = 1.0;
        k0(1, 1) = std::sqrt(1.0 - gamma);
        Matrix k1 = Matrix::Zero(2, 2);
        k1(0, 1) = std::sqrt(gamma);
        return Thermal({std::move(k0), std::move(k1)}, {gamma, 0.0, std::sqrt(1.0 - gamma)});
    }

    /**
     * Phase damping with coherence multiplier sqrt(1-gamma).
     * This differs from phase flip, whose multiplier is 1-2p.
     */
    static QuantumChannel PhaseDamping(double gamma)
    {
        CheckedProbability(gamma);
        Matrix k0 = Matrix::Zero(2, 2);
        k0(0, 0) = 1.0;
        k0(1, 1) = std::sqrt(1.0 - gamma);
        Matrix k1 = Matrix::Zero(2, 2);
        k1(1, 1) = std::sqrt(gamma);
        // The same map as a phase flip with 1 - 2p = sqrt(1 - gamma).
        const double flip = 0.5 * (1.0 - std::sqrt(1.0 - gamma));
        auto data = MakeKrausData({std::move(k0), std::move(k1)});
        data->factors.push_back(PauliFactor({1.0 - flip, 0.0, 0.0, flip}, 1));
        return QuantumChannel(std::move(data));
    }

    /**
     * Finite-temperature amplitude damping.
     * excitedStatePopulation is the equilibrium population of |1>.
     */
    static QuantumChannel GeneralizedAmplitudeDamping(double gamma, double excitedStatePopulation)
    {
        CheckedProbability(gamma);
        CheckedProbability(excitedStatePopulation);
        const double groundStatePopulation = 1.0 - excitedStatePopulation;

        Matrix k0 = Matrix::Zero(2, 2);
        k0(0, 0) = std::sqrt(groundStatePopulation);
        k0(1, 1) = std::sqrt(groundStatePopulation * (1.0 - gamma));
        Matrix k1 = Matrix::Zero(2, 2);
        k1(0, 1) = std::sqrt(groundStatePopulation * gamma);
        Matrix k2 = Matrix::Zero(2, 2);
        k2(0, 0) = std::sqrt(excitedStatePopulation * (1.0 - gamma));
        k2(1, 1) = std::sqrt(excitedStatePopulation);
        Matrix k3 = Matrix::Zero(2, 2);
        k3(1, 0) = std::sqrt(excitedStatePopulation * gamma);
        return Thermal({std::move(k0), std::move(k1), std::move(k2), std::move(k3)}, {gamma, excitedStatePopulation, std::sqrt(1.0 - gamma)});
    }

    /**
     * Hardware-style thermal relaxation for a duration, T1 and T2.
     *
     * Populations relax toward excitedStatePopulation with exp(-duration/T1),
     * while coherences are multiplied by exp(-duration/T2). The physical
     * Markovian constraint T2 <= 2*T1 is enforced. Positive infinity is
     * accepted for either time constant.
     */
    static QuantumChannel ThermalRelaxation(double duration, double t1, double t2, double excitedStatePopulation = 0.0)
    {
        if (!std::isfinite(duration) || duration < 0.0)
            throw std::invalid_argument("Thermal-relaxation duration must be finite and nonnegative");
        ValidatePositiveTime(t1, "T1");
        ValidatePositiveTime(t2, "T2");
        CheckedProbability(excitedStatePopulation);

        if (std::isinf(t2) && !std::isinf(t1))
            throw std::invalid_argument("Thermal relaxation requires T2 <= 2*T1");
        if (std::isfinite(t1) && std::isfinite(t2) && t2 > 2.0 * t1 * (1.0 + kProbabilityTolerance))
            throw std::invalid_argument("Thermal relaxation requires T2 <= 2*T1");

        const double coherenceMultiplier = std::isinf(t2) ? 1.0 : std::exp(-duration / t2);
        const double gamma = std::isinf(t1) ? 0.0 : -std::expm1(-duration / t1);
        const double up = gamma * excitedStatePopulation;
        const double down = gamma * (1.0 - excitedStatePopulation);

        // The Choi matrix has a 2x2 block
        // [[1-up, coherence], [coherence, 1-down]]. Decomposing that block gives
        // at most two diagonal Kraus operators; excitation and decay add one each.
        const double a = 1.0 - up;
        const double c = 1.0 - down;
        if (coherenceMultiplier * coherenceMultiplier > a * c + kProbabilityTolerance)
            throw std::invalid_argument("The supplied T1 and T2 do not define a completely-positive map");

        const double discriminant = std::hypot(a - c, 2.0 * coherenceMultiplier);
        const double lambdaPlus = 0.5 * (a + c + discriminant);
        const double lambdaMinus = std::max(0.0, 0.5 * (a + c - discriminant));
        const double angle = 0.5 * std::atan2(2.0 * coherenceMultiplier, a - c);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        KrausOperators kraus;
        if (lambdaPlus > kZeroTolerance)
        {
            Matrix diagonal = Matrix::Zero(2, 2);
            diagonal(0, 0) = std::sqrt(lambdaPlus) * cosine;
            diagonal(1, 1) = std::sqrt(lambdaPlus) * sine;
            kraus.emplace_back(std::move(diagonal));
        }
        if (lambdaMinus > kZeroTolerance)
        {
            Matrix diagonal = Matrix::Zero(2, 2);
            diagonal(0, 0) = -std::sqrt(lambdaMinus) * sine;
            diagonal(1, 1) = std::sqrt(lambdaMinus) * cosine;
            kraus.emplace_back(std::move(diagonal));
        }
        if (up > kZeroTolerance)
        {
            Matrix excitation = Matrix::Zero(2, 2);
            excitation(1, 0) = std::sqrt(up);
            kraus.emplace_back(std::move(excitation));
        }
        if (down > kZeroTolerance)
        {
            Matrix decay = Matrix::Zero(2, 2);
            decay(0, 1) = std::sqrt(down);
            kraus.emplace_back(std::move(decay));
        }
        return Thermal(std::move(kraus), {gamma, excitedStatePopulation, coherenceMultiplier});
    }

    /** (1-p)rho + p (Z (x) Z) rho (Z (x) Z). */
    static QuantumChannel CorrelatedPhaseFlip(double probability)
    {
        return CorrelatedPhaseFlip(probability, 1.0);
    }

    /**
     * Interpolate between independent phase flips and a fully correlated ZZ
     * phase flip. correlation=0 is E_Z (x) E_Z; correlation=1 applies either II
     * or ZZ.
     */
    static QuantumChannel CorrelatedPhaseFlip(double probability, double correlation)
    {
        CheckedProbability(probability);
        CheckedProbability(correlation);
        std::vector<double> probabilities(16, 0.0);
        const double independent = 1.0 - correlation;
        probabilities[0] = independent * (1.0 - probability) * (1.0 - probability) + correlation * (1.0 - probability);
        probabilities[3] = independent * probability * (1.0 - probability);
        probabilities[12] = probabilities[3];
        probabilities[15] = independent * probability * probability + correlation * probability;
        return Pauli(probabilities);
    }

    /**
     * Two-qubit depolarizing in the total-Pauli-error convention:
     * identity has probability 1-p and each other Pauli has p/15.
     */
    static QuantumChannel TwoQubitDepolarizing(double errorProbability)
    {
        CheckedProbability(errorProbability);
        std::vector<double> probabilities(16, errorProbability / 15.0);
        probabilities[0] = 1.0 - errorProbability;
        return Pauli(probabilities);
    }

    /** Two-qubit replacement depolarizing, fully mixed at probability one. */
    static QuantumChannel TwoQubitDepolarizingMixing(double mixingProbability)
    {
        CheckedProbability(mixingProbability);
        return TwoQubitDepolarizing((15.0 / 16.0) * mixingProbability);
    }

    /**
     * Probabilities of the channel as a Pauli mixture, in Pauli() order, when
     * its Kraus operators describe one: with K_i = sum_a c_ia P_a the process
     * matrix chi_ab = sum_i c_ia conj(c_ib) must be diagonal. Empty otherwise.
     */
    static std::vector<double> PauliMixtureOf(const KrausOperators &kraus, size_t numQubits)
    {
        if (numQubits < 1 || numQubits > 2 || kraus.empty())
            return {};
        const size_t terms = numQubits == 1 ? 4 : 16;
        const double dimension = numQubits == 1 ? 2.0 : 4.0;
        if (kraus.front().rows() != static_cast<Eigen::Index>(dimension))
            return {};
        std::vector<Matrix> paulis;
        paulis.reserve(terms);
        for (size_t a = 0; a < terms; ++a)
            paulis.push_back(PauliProduct(a, numQubits));
        std::vector<std::vector<std::complex<double>>> coefficients(kraus.size(), std::vector<std::complex<double>>(terms));
        for (size_t i = 0; i < kraus.size(); ++i)
            for (size_t a = 0; a < terms; ++a)
                coefficients[i][a] = (paulis[a].adjoint() * kraus[i]).trace() / dimension;
        constexpr double tolerance = 1e-12;
        std::vector<double> probabilities(terms, 0.0);
        double total = 0;
        for (size_t a = 0; a < terms; ++a)
            for (size_t b = 0; b < terms; ++b)
            {
                std::complex<double> chi = 0;
                for (size_t i = 0; i < kraus.size(); ++i)
                    chi += coefficients[i][a] * std::conj(coefficients[i][b]);
                if (a == b)
                {
                    if (chi.real() < -tolerance || std::abs(chi.imag()) > tolerance)
                        return {};
                    probabilities[a] = std::max(0.0, chi.real());
                    total += probabilities[a];
                }
                else if (std::abs(chi) > tolerance)
                    return {};
            }
        if (std::abs(total - 1.0) > 1e-10)
            return {};
        return probabilities;
    }

  private:
    static constexpr double kProbabilityTolerance = 1e-12;
    static constexpr double kZeroTolerance = 1e-15;

    struct Data
    {
        // Kraus operators as given; for a composed channel, derived on first use.
        mutable KrausOperators kraus;
        mutable std::once_flag krausOnce;
        size_t numQubits = 0;
        std::vector<double> pauliProbabilities;
        // Set when the channel is built with its structure, classified on first use otherwise.
        mutable std::vector<Factor> factors;
        mutable std::once_flag factorsOnce;
        mutable Matrix superoperator;
        mutable std::once_flag superoperatorOnce;
        // The factors define the channel; the Kraus operators follow from them.
        bool composed = false;
        // Each factor as a channel, built on first use (FactorChannel).
        mutable std::vector<QuantumChannel> parts;
        mutable std::once_flag partsOnce;
        // What each factor's first operator may weigh, found on first use (FirstOperatorOf).
        mutable std::vector<FirstOperator> first;
        mutable std::once_flag firstOnce;
        // Whether the channel has one branch only (IsDeterministic): -1 until asked.
        mutable std::atomic<signed char> deterministic{-1};
    };

    explicit QuantumChannel(std::shared_ptr<Data> data) : data_(std::move(data))
    {
    }

    size_t Dimension() const
    {
        return size_t{1} << data_->numQubits;
    }

    static std::vector<size_t> AllQubits(size_t numQubits)
    {
        std::vector<size_t> qubits(numQubits);
        for (size_t q = 0; q < numQubits; ++q)
            qubits[q] = q;
        return qubits;
    }

    static double CheckedProbability(double probability)
    {
        if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
            throw std::invalid_argument("Quantum-channel probabilities must be finite and in [0, 1]");
        return probability;
    }

    static void ValidatePositiveTime(double value, const char *name)
    {
        if (std::isnan(value) || value <= 0.0)
            throw std::invalid_argument(std::string(name) + " must be positive (infinity is allowed)");
    }

    static Matrix PauliMatrix(size_t pauliId)
    {
        Matrix matrix = Matrix::Zero(2, 2);
        switch (pauliId)
        {
        case 0:
            matrix.setIdentity();
            break;
        case 1:
            matrix(0, 1) = 1.0;
            matrix(1, 0) = 1.0;
            break;
        case 2:
            matrix(0, 1) = std::complex<double>(0.0, -1.0);
            matrix(1, 0) = std::complex<double>(0.0, 1.0);
            break;
        case 3:
            matrix(0, 0) = 1.0;
            matrix(1, 1) = -1.0;
            break;
        default:
            throw std::invalid_argument("Invalid Pauli index");
        }
        return matrix;
    }

    static Matrix Kronecker(const Matrix &left, const Matrix &right)
    {
        Matrix result(left.rows() * right.rows(), left.cols() * right.cols());
        for (Eigen::Index row = 0; row < left.rows(); ++row)
            for (Eigen::Index column = 0; column < left.cols(); ++column)
                result.block(row * right.rows(), column * right.cols(), right.rows(), right.cols()) = left(row, column) * right;
        return result;
    }

    // The Pauli product selected by a Pauli() index: base-4 digit k acts on target k.
    static Matrix PauliProduct(size_t index, size_t numQubits)
    {
        Matrix pauli(1, 1);
        pauli(0, 0) = 1.0;
        size_t divisor = 1;
        for (size_t q = 0; q < numQubits; ++q)
            divisor *= 4;
        for (size_t q = numQubits; q-- > 0;)
        {
            divisor /= 4;
            pauli = Kronecker(pauli, PauliMatrix((index / divisor) % 4));
        }
        return pauli;
    }

    static KrausOperators PauliKraus(const std::vector<double> &probabilities, size_t numQubits)
    {
        KrausOperators kraus;
        kraus.reserve(static_cast<size_t>(std::count_if(probabilities.begin(), probabilities.end(), [](double probability) { return probability > 0.0; })));
        for (size_t index = 0; index < probabilities.size(); ++index)
            if (probabilities[index] != 0.0)
                kraus.emplace_back(std::sqrt(probabilities[index]) * PauliProduct(index, numQubits));
        return kraus;
    }

    static Factor PauliFactor(std::vector<double> probabilities, size_t numQubits)
    {
        Factor factor;
        factor.kind = FactorKind::kPauli;
        factor.qubits = AllQubits(numQubits);
        factor.probabilities = std::move(probabilities);
        return factor;
    }

    // A one-qubit channel with its relaxation parameters attached.
    static QuantumChannel Thermal(KrausOperators kraus, const ThermalParameters &parameters)
    {
        // A relaxation that changes nothing is the identity, one branch, however many operators it was written with
        // (a generalized damping of zero has two multiples of it). Otherwise an operator that is zero is no branch:
        // it is never drawn, and the ones left decide whether the channel is a gate (IsDeterministic).
        if (parameters.decay == 0.0 && parameters.coherence == 1.0)
            kraus = {Matrix::Identity(2, 2)};
        else
            kraus.erase(std::remove_if(kraus.begin(), kraus.end(), [](const Matrix &op) { return op.isZero(0.0); }), kraus.end());
        auto data = MakeKrausData(std::move(kraus));
        Factor factor;
        factor.kind = FactorKind::kThermal;
        factor.qubits = {0};
        factor.operators = data->kraus;
        factor.thermal = parameters;
        data->factors.push_back(std::move(factor));
        return QuantumChannel(std::move(data));
    }

    static std::shared_ptr<Data> MakeKrausData(KrausOperators krausOperators)
    {
        auto data = std::make_shared<Data>();
        data->kraus = std::move(krausOperators);
        data->numQubits = Validate(data->kraus);
        return data;
    }

    // Returns the number of qubits the validated operators act on.
    static size_t Validate(const KrausOperators &krausOperators)
    {
        if (krausOperators.empty())
            throw std::invalid_argument("A quantum channel must contain at least one Kraus operator");

        const Eigen::Index dimension = krausOperators.front().rows();
        if (dimension < 2 || krausOperators.front().cols() != dimension)
            throw std::invalid_argument("Kraus operators must be nonempty square qubit matrices");

        size_t numQubits = 0;
        size_t remainingDimension = static_cast<size_t>(dimension);
        while (remainingDimension > 1 && remainingDimension % 2 == 0)
        {
            remainingDimension /= 2;
            ++numQubits;
        }
        if (remainingDimension != 1)
            throw std::invalid_argument("Kraus-operator dimensions must be powers of two");

        Matrix completeness = Matrix::Zero(dimension, dimension);
        for (const Matrix &krausOperator : krausOperators)
        {
            if (krausOperator.rows() != dimension || krausOperator.cols() != dimension)
                throw std::invalid_argument("All Kraus operators must have the same dimensions");
            if (!krausOperator.allFinite())
                throw std::invalid_argument("Kraus operators must contain only finite values");
            completeness.noalias() += krausOperator.adjoint() * krausOperator;
        }

        const Matrix identity = Matrix::Identity(dimension, dimension);
        const double tolerance = 1e-10 * std::max<Eigen::Index>(1, dimension);
        if ((completeness - identity).norm() > tolerance)
            throw std::invalid_argument("Kraus operators do not define a trace-preserving channel");
        return numQubits;
    }

    // What a Kraus set is: a Pauli mixture, a mixture of unitaries, or neither.
    static Factor Classify(const KrausOperators &kraus, size_t numQubits)
    {
        Factor factor;
        factor.qubits = AllQubits(numQubits);
        std::vector<double> pauli = PauliMixtureOf(kraus, numQubits);
        if (!pauli.empty())
        {
            factor.kind = FactorKind::kPauli;
            factor.probabilities = std::move(pauli);
            return factor;
        }

        const Eigen::Index dimension = kraus.front().rows();
        const Matrix identity = Matrix::Identity(dimension, dimension);
        bool unitaryMixture = true;
        for (const Matrix &op : kraus)
        {
            const Matrix gram = op.adjoint() * op;
            const double probability = gram.trace().real() / static_cast<double>(dimension);
            if ((gram - probability * identity).norm() > 1e-12 * static_cast<double>(dimension))
            {
                unitaryMixture = false;
                break;
            }
            if (probability <= kZeroTolerance)
                continue;
            factor.probabilities.push_back(probability);
            factor.operators.emplace_back(op / std::sqrt(probability));
        }
        if (unitaryMixture && !factor.operators.empty())
        {
            factor.kind = FactorKind::kUnitaryMixture;
            return factor;
        }

        factor.kind = FactorKind::kKraus;
        factor.probabilities.clear();
        factor.operators = kraus;
        return factor;
    }

    static KrausOperators FactorKraus(const Factor &factor)
    {
        switch (factor.kind)
        {
        case FactorKind::kPauli:
            return PauliKraus(factor.probabilities, factor.qubits.size());
        case FactorKind::kUnitaryMixture: {
            KrausOperators kraus;
            kraus.reserve(factor.operators.size());
            for (size_t i = 0; i < factor.operators.size(); ++i)
                kraus.emplace_back(std::sqrt(factor.probabilities[i]) * factor.operators[i]);
            return kraus;
        }
        default:
            return factor.operators;
        }
    }

    // An operator over a factor's targets as an operator over all targets of the channel.
    static Matrix Embed(const Matrix &op, const std::vector<size_t> &qubits, size_t numQubits)
    {
        if (qubits.size() == numQubits)
        {
            if (numQubits == 1 || qubits[0] == 0)
                return op;
            // Targets given as {1, 0}: exchange the two local bits of both indices.
            Matrix swapped(4, 4);
            const std::array<Eigen::Index, 4> exchange{0, 2, 1, 3};
            for (Eigen::Index row = 0; row < 4; ++row)
                for (Eigen::Index column = 0; column < 4; ++column)
                    swapped(exchange[static_cast<size_t>(row)], exchange[static_cast<size_t>(column)]) = op(row, column);
            return swapped;
        }
        const Matrix identity = Matrix::Identity(2, 2);
        return qubits[0] == 0 ? Kronecker(identity, op) : Kronecker(op, identity);
    }

    static Matrix KrausSuperoperator(const KrausOperators &kraus)
    {
        const Eigen::Index dimension = kraus.front().rows();
        const Eigen::Index size = dimension * dimension;
        Matrix superoperator = Matrix::Zero(size, size);
        for (const Matrix &op : kraus)
            for (Eigen::Index input = 0; input < size; ++input)
                for (Eigen::Index output = 0; output < size; ++output)
                    superoperator(output, input) += op(output % dimension, input % dimension) * std::conj(op(output / dimension, input / dimension));
        return superoperator;
    }

    static Matrix FactorSuperoperator(const Factor &factor, size_t numQubits)
    {
        KrausOperators kraus = FactorKraus(factor);
        for (Matrix &op : kraus)
            op = Embed(op, factor.qubits, numQubits);
        return KrausSuperoperator(kraus);
    }

    // The minimal Kraus set of a map, from the eigenvectors of its Choi matrix.
    static KrausOperators CanonicalKraus(const Matrix &superoperator, size_t dimension)
    {
        const Eigen::Index d = static_cast<Eigen::Index>(dimension);
        const Eigen::Index size = d * d;
        Matrix choi(size, size);
        for (Eigen::Index r = 0; r < d; ++r)
            for (Eigen::Index c = 0; c < d; ++c)
                for (Eigen::Index r2 = 0; r2 < d; ++r2)
                    for (Eigen::Index c2 = 0; c2 < d; ++c2)
                        choi(r + d * c, r2 + d * c2) = superoperator(r + d * r2, c + d * c2);

        const Eigen::SelfAdjointEigenSolver<Matrix> solver(choi);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Could not decompose a composed quantum channel");
        const auto &values = solver.eigenvalues();
        const double cutoff = 1e-14 * std::max(1.0, values.cwiseAbs().maxCoeff());
        KrausOperators kraus;
        for (Eigen::Index k = size; k-- > 0;)
        {
            if (values[k] <= cutoff)
                continue;
            Matrix op(d, d);
            for (Eigen::Index c = 0; c < d; ++c)
                for (Eigen::Index r = 0; r < d; ++r)
                    op(r, c) = std::sqrt(values[k]) * solver.eigenvectors()(r + d * c, k);
            kraus.emplace_back(std::move(op));
        }
        return kraus;
    }

    // Probabilities over a factor's targets as probabilities over all targets, identity elsewhere.
    static std::vector<double> ExpandPauli(const Factor &factor, size_t numQubits)
    {
        const size_t terms = size_t{1} << (2 * numQubits);
        std::vector<double> expanded(terms, 0.0);
        for (size_t index = 0; index < factor.probabilities.size(); ++index)
        {
            size_t target = 0;
            for (size_t k = 0; k < factor.qubits.size(); ++k)
                target |= ((index >> (2 * k)) & 3) << (2 * factor.qubits[k]);
            expanded[target] += factor.probabilities[index];
        }
        return expanded;
    }

    // Two Pauli mixtures one after the other: phases do not matter, so the result is the convolution over the Pauli group.
    static std::vector<double> ConvolvePauli(const std::vector<double> &first, const std::vector<double> &second, size_t numQubits)
    {
        static constexpr std::array<std::array<size_t, 4>, 4> product{{{0, 1, 2, 3}, {1, 0, 3, 2}, {2, 3, 0, 1}, {3, 2, 1, 0}}};
        std::vector<double> result(first.size(), 0.0);
        for (size_t a = 0; a < first.size(); ++a)
        {
            if (first[a] == 0.0)
                continue;
            for (size_t b = 0; b < second.size(); ++b)
            {
                if (second[b] == 0.0)
                    continue;
                size_t c = 0;
                for (size_t q = 0; q < numQubits; ++q)
                    c |= product[(a >> (2 * q)) & 3][(b >> (2 * q)) & 3] << (2 * q);
                result[c] += first[a] * second[b];
            }
        }
        return result;
    }

    static QuantumChannel FromFactors(std::vector<Factor> factors, size_t numQubits)
    {
        // Adjacent Pauli factors are one Pauli factor.
        std::vector<Factor> merged;
        for (Factor &factor : factors)
        {
            if (factor.kind == FactorKind::kPauli && !merged.empty() && merged.back().kind == FactorKind::kPauli)
            {
                const std::vector<double> combined = ConvolvePauli(ExpandPauli(merged.back(), numQubits), ExpandPauli(factor, numQubits), numQubits);
                merged.back() = PauliFactor(combined, numQubits);
            }
            else
                merged.push_back(std::move(factor));
        }

        if (merged.size() == 1 && merged.front().kind == FactorKind::kPauli)
        {
            std::vector<double> probabilities = ExpandPauli(merged.front(), numQubits);
            // Rounding in a convolution can leave the sum a few ulps off one.
            double total = 0.0;
            for (const double probability : probabilities)
                total += probability;
            for (double &probability : probabilities)
                probability /= total;
            return Pauli(probabilities);
        }

        auto data = std::make_shared<Data>();
        data->numQubits = numQubits;
        data->factors = std::move(merged);
        data->composed = true;
        return QuantumChannel(std::move(data));
    }

    std::shared_ptr<const Data> data_;
};

} // namespace Simulators

#endif // _SIMULATORS_QUANTUM_CHANNEL_H_
