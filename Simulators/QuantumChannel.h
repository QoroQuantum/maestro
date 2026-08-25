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
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Simulators {

/**
 * A local completely-positive, trace-preserving map in Kraus form.
 *
 * Matrices use Maestro's gate convention: for a target list {q0, q1}, q0 is
 * the least-significant local basis bit. Thus a product A on q0 and B on q1
 * is represented by B (x) A.
 */
class QuantumChannel {
 public:
  using Matrix = Eigen::MatrixXcd;
  using KrausOperators = std::vector<Matrix>;

  /** Construct and validate a CPTP channel from Kraus operators. */
  explicit QuantumChannel(KrausOperators krausOperators)
      : krausOperators_(std::move(krausOperators)), numQubits_(0) {
    Validate();
  }

  const KrausOperators& GetKrausOperators() const { return krausOperators_; }
  size_t GetNumberOfQubits() const { return numQubits_; }

  /** Compare two Kraus representations coefficient by coefficient. */
  bool IsApprox(const QuantumChannel& other, double tolerance = 1e-12) const {
    if (tolerance < 0.0 || !std::isfinite(tolerance))
      throw std::invalid_argument(
          "Quantum-channel comparison tolerance must be finite and nonnegative");
    if (numQubits_ != other.numQubits_ ||
        krausOperators_.size() != other.krausOperators_.size())
      return false;
    for (size_t i = 0; i < krausOperators_.size(); ++i) {
      const Matrix& left = krausOperators_[i];
      const Matrix& right = other.krausOperators_[i];
      if (left.rows() != right.rows() || left.cols() != right.cols() ||
          (left - right).cwiseAbs().maxCoeff() > tolerance)
        return false;
    }
    return true;
  }

  /** (1-p) rho + p X rho X. */
  static QuantumChannel BitFlip(double probability) {
    return Pauli(
        {1.0 - CheckedProbability(probability), probability, 0.0, 0.0});
  }

  /** (1-p) rho + p Y rho Y. */
  static QuantumChannel BitPhaseFlip(double probability) {
    return Pauli(
        {1.0 - CheckedProbability(probability), 0.0, probability, 0.0});
  }

  /** (1-p) rho + p Z rho Z. */
  static QuantumChannel PhaseFlip(double probability) {
    return Pauli(
        {1.0 - CheckedProbability(probability), 0.0, 0.0, probability});
  }

  /**
   * Single-qubit Pauli channel with X, Y and Z error probabilities.
   * The identity probability is 1-px-py-pz.
   */
  static QuantumChannel Pauli(double px, double py, double pz) {
    CheckedProbability(px);
    CheckedProbability(py);
    CheckedProbability(pz);
    const double total = px + py + pz;
    if (!std::isfinite(total) || total > 1.0 + kProbabilityTolerance)
      throw std::invalid_argument(
          "Pauli error probabilities must sum to at most one");
    return Pauli({std::max(0.0, 1.0 - total), px, py, pz});
  }

  /**
   * Arbitrary local Pauli channel.
   *
   * The array must contain 4^n probabilities. Base-4 digit i of an array
   * index selects I, X, Y, or Z on targets[i], respectively. For two targets
   * the order is therefore II, XI, YI, ZI, IX, XX, ... , ZZ.
   */
  static QuantumChannel Pauli(const std::vector<double>& probabilities) {
    if (probabilities.empty())
      throw std::invalid_argument("A Pauli channel must contain probabilities");

    size_t count = probabilities.size();
    size_t numQubits = 0;
    while (count > 1 && count % 4 == 0) {
      count /= 4;
      ++numQubits;
    }
    if (count != 1 || numQubits == 0)
      throw std::invalid_argument(
          "A Pauli channel needs exactly 4^n probabilities for n >= 1");

    double total = 0.0;
    for (const double probability : probabilities) {
      CheckedProbability(probability);
      total += probability;
    }
    if (!std::isfinite(total) || std::abs(total - 1.0) > kProbabilityTolerance)
      throw std::invalid_argument(
          "Pauli channel probabilities must sum to one");

    KrausOperators kraus;
    kraus.reserve(static_cast<size_t>(
        std::count_if(probabilities.begin(), probabilities.end(),
                      [](double probability) { return probability > 0.0; })));
    for (size_t index = 0; index < probabilities.size(); ++index) {
      if (probabilities[index] == 0.0) continue;

      Matrix pauli(1, 1);
      pauli(0, 0) = 1.0;
      size_t divisor = 1;
      for (size_t q = 0; q < numQubits; ++q) divisor *= 4;
      for (size_t q = numQubits; q-- > 0;) {
        divisor /= 4;
        const size_t pauliId = (index / divisor) % 4;
        pauli = Kronecker(pauli, PauliMatrix(pauliId));
      }
      kraus.emplace_back(std::sqrt(probabilities[index]) * pauli);
    }
    return QuantumChannel(std::move(kraus));
  }

  /**
   * Depolarizing channel in the NoiseModel/QCSim convention:
   * (1-p)rho + p/3 (XrhoX + YrhoY + ZrhoZ).
   */
  static QuantumChannel Depolarizing(double errorProbability) {
    CheckedProbability(errorProbability);
    return Pauli({1.0 - errorProbability, errorProbability / 3.0,
                  errorProbability / 3.0, errorProbability / 3.0});
  }

  /**
   * Depolarizing channel in the replacement convention:
   * (1-p)rho + p I/2. Full mixing occurs at p=1.
   */
  static QuantumChannel DepolarizingMixing(double mixingProbability) {
    CheckedProbability(mixingProbability);
    return Depolarizing(0.75 * mixingProbability);
  }

  /** |1> -> |0> relaxation with probability gamma. */
  static QuantumChannel AmplitudeDamping(double gamma) {
    CheckedProbability(gamma);
    Matrix k0 = Matrix::Zero(2, 2);
    k0(0, 0) = 1.0;
    k0(1, 1) = std::sqrt(1.0 - gamma);
    Matrix k1 = Matrix::Zero(2, 2);
    k1(0, 1) = std::sqrt(gamma);
    return QuantumChannel({std::move(k0), std::move(k1)});
  }

  /**
   * Phase damping with coherence multiplier sqrt(1-gamma).
   * This differs from phase flip, whose multiplier is 1-2p.
   */
  static QuantumChannel PhaseDamping(double gamma) {
    CheckedProbability(gamma);
    Matrix k0 = Matrix::Zero(2, 2);
    k0(0, 0) = 1.0;
    k0(1, 1) = std::sqrt(1.0 - gamma);
    Matrix k1 = Matrix::Zero(2, 2);
    k1(1, 1) = std::sqrt(gamma);
    return QuantumChannel({std::move(k0), std::move(k1)});
  }

  /**
   * Finite-temperature amplitude damping.
   * excitedStatePopulation is the equilibrium population of |1>.
   */
  static QuantumChannel GeneralizedAmplitudeDamping(
      double gamma, double excitedStatePopulation) {
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
    return QuantumChannel(
        {std::move(k0), std::move(k1), std::move(k2), std::move(k3)});
  }

  /**
   * Hardware-style thermal relaxation for a duration, T1 and T2.
   *
   * Populations relax toward excitedStatePopulation with exp(-duration/T1),
   * while coherences are multiplied by exp(-duration/T2). The physical
   * Markovian constraint T2 <= 2*T1 is enforced. Positive infinity is
   * accepted for either time constant.
   */
  static QuantumChannel ThermalRelaxation(double duration, double t1, double t2,
                                          double excitedStatePopulation = 0.0) {
    if (!std::isfinite(duration) || duration < 0.0)
      throw std::invalid_argument(
          "Thermal-relaxation duration must be finite and nonnegative");
    ValidatePositiveTime(t1, "T1");
    ValidatePositiveTime(t2, "T2");
    CheckedProbability(excitedStatePopulation);

    if (std::isinf(t2) && !std::isinf(t1))
      throw std::invalid_argument("Thermal relaxation requires T2 <= 2*T1");
    if (std::isfinite(t1) && std::isfinite(t2) &&
        t2 > 2.0 * t1 * (1.0 + kProbabilityTolerance))
      throw std::invalid_argument("Thermal relaxation requires T2 <= 2*T1");

    const double coherenceMultiplier =
        std::isinf(t2) ? 1.0 : std::exp(-duration / t2);
    const double gamma = std::isinf(t1) ? 0.0 : -std::expm1(-duration / t1);
    const double up = gamma * excitedStatePopulation;
    const double down = gamma * (1.0 - excitedStatePopulation);

    // The Choi matrix has a 2x2 block
    // [[1-up, coherence], [coherence, 1-down]]. Decomposing that block gives
    // at most two diagonal Kraus operators; excitation and decay add one each.
    const double a = 1.0 - up;
    const double c = 1.0 - down;
    if (coherenceMultiplier * coherenceMultiplier >
        a * c + kProbabilityTolerance)
      throw std::invalid_argument(
          "The supplied T1 and T2 do not define a completely-positive map");

    const double discriminant = std::hypot(a - c, 2.0 * coherenceMultiplier);
    const double lambdaPlus = 0.5 * (a + c + discriminant);
    const double lambdaMinus = std::max(0.0, 0.5 * (a + c - discriminant));
    const double angle = 0.5 * std::atan2(2.0 * coherenceMultiplier, a - c);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    KrausOperators kraus;
    if (lambdaPlus > kZeroTolerance) {
      Matrix diagonal = Matrix::Zero(2, 2);
      diagonal(0, 0) = std::sqrt(lambdaPlus) * cosine;
      diagonal(1, 1) = std::sqrt(lambdaPlus) * sine;
      kraus.emplace_back(std::move(diagonal));
    }
    if (lambdaMinus > kZeroTolerance) {
      Matrix diagonal = Matrix::Zero(2, 2);
      diagonal(0, 0) = -std::sqrt(lambdaMinus) * sine;
      diagonal(1, 1) = std::sqrt(lambdaMinus) * cosine;
      kraus.emplace_back(std::move(diagonal));
    }
    if (up > kZeroTolerance) {
      Matrix excitation = Matrix::Zero(2, 2);
      excitation(1, 0) = std::sqrt(up);
      kraus.emplace_back(std::move(excitation));
    }
    if (down > kZeroTolerance) {
      Matrix decay = Matrix::Zero(2, 2);
      decay(0, 1) = std::sqrt(down);
      kraus.emplace_back(std::move(decay));
    }
    return QuantumChannel(std::move(kraus));
  }

  /** (1-p)rho + p (Z (x) Z) rho (Z (x) Z). */
  static QuantumChannel CorrelatedPhaseFlip(double probability) {
    return CorrelatedPhaseFlip(probability, 1.0);
  }

  /**
   * Interpolate between independent phase flips and a fully correlated ZZ
   * phase flip. correlation=0 is E_Z (x) E_Z; correlation=1 applies either II
   * or ZZ.
   */
  static QuantumChannel CorrelatedPhaseFlip(double probability,
                                             double correlation) {
    CheckedProbability(probability);
    CheckedProbability(correlation);
    std::vector<double> probabilities(16, 0.0);
    const double independent = 1.0 - correlation;
    probabilities[0] =
        independent * (1.0 - probability) * (1.0 - probability) +
        correlation * (1.0 - probability);
    probabilities[3] = independent * probability * (1.0 - probability);
    probabilities[12] = probabilities[3];
    probabilities[15] = independent * probability * probability +
                        correlation * probability;
    return Pauli(probabilities);
  }

  /**
   * Two-qubit depolarizing in the total-Pauli-error convention:
   * identity has probability 1-p and each other Pauli has p/15.
   */
  static QuantumChannel TwoQubitDepolarizing(double errorProbability) {
    CheckedProbability(errorProbability);
    std::vector<double> probabilities(16, errorProbability / 15.0);
    probabilities[0] = 1.0 - errorProbability;
    return Pauli(probabilities);
  }

  /** Two-qubit replacement depolarizing, fully mixed at probability one. */
  static QuantumChannel TwoQubitDepolarizingMixing(
      double mixingProbability) {
    CheckedProbability(mixingProbability);
    return TwoQubitDepolarizing((15.0 / 16.0) * mixingProbability);
  }

 private:
  static constexpr double kProbabilityTolerance = 1e-12;
  static constexpr double kZeroTolerance = 1e-15;

  static double CheckedProbability(double probability) {
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0)
      throw std::invalid_argument(
          "Quantum-channel probabilities must be finite and in [0, 1]");
    return probability;
  }

  static void ValidatePositiveTime(double value, const char* name) {
    if (std::isnan(value) || value <= 0.0)
      throw std::invalid_argument(std::string(name) +
                                  " must be positive (infinity is allowed)");
  }

  static Matrix PauliMatrix(size_t pauliId) {
    Matrix matrix = Matrix::Zero(2, 2);
    switch (pauliId) {
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

  static Matrix Kronecker(const Matrix& left, const Matrix& right) {
    Matrix result(left.rows() * right.rows(), left.cols() * right.cols());
    for (Eigen::Index row = 0; row < left.rows(); ++row)
      for (Eigen::Index column = 0; column < left.cols(); ++column)
        result.block(row * right.rows(), column * right.cols(), right.rows(),
                     right.cols()) = left(row, column) * right;
    return result;
  }

  void Validate() {
    if (krausOperators_.empty())
      throw std::invalid_argument(
          "A quantum channel must contain at least one Kraus operator");

    const Eigen::Index dimension = krausOperators_.front().rows();
    if (dimension < 2 || krausOperators_.front().cols() != dimension)
      throw std::invalid_argument(
          "Kraus operators must be nonempty square qubit matrices");

    size_t remainingDimension = static_cast<size_t>(dimension);
    while (remainingDimension > 1 && remainingDimension % 2 == 0) {
      remainingDimension /= 2;
      ++numQubits_;
    }
    if (remainingDimension != 1)
      throw std::invalid_argument(
          "Kraus-operator dimensions must be powers of two");

    Matrix completeness = Matrix::Zero(dimension, dimension);
    for (const Matrix& krausOperator : krausOperators_) {
      if (krausOperator.rows() != dimension ||
          krausOperator.cols() != dimension)
        throw std::invalid_argument(
            "All Kraus operators must have the same dimensions");
      if (!krausOperator.allFinite())
        throw std::invalid_argument(
            "Kraus operators must contain only finite values");
      completeness.noalias() += krausOperator.adjoint() * krausOperator;
    }

    const Matrix identity = Matrix::Identity(dimension, dimension);
    const double tolerance = 1e-10 * std::max<Eigen::Index>(1, dimension);
    if ((completeness - identity).norm() > tolerance)
      throw std::invalid_argument(
          "Kraus operators do not define a trace-preserving channel");
  }

  KrausOperators krausOperators_;
  size_t numQubits_;
};

}  // namespace Simulators

#endif  // _SIMULATORS_QUANTUM_CHANNEL_H_
