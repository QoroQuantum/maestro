/**
 * @file noise.h
 * @brief Pauli and coherent noise models for quantum circuit simulation.
 *
 * Defines a NoiseModel that maps per-qubit noise parameters and provides:
 *   - Analytical noise damping for expectation values (zero overhead).
 *   - Monte Carlo Pauli noise injection for shot-based execution.
 *   - Coherent noise injection via systematic rotation gates.
 *
 * ## Pauli (incoherent) noise
 *
 * Pauli channel: Λ(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ
 *
 * For expectation values of a Pauli string P = P_0 ⊗ ... ⊗ P_{n-1}:
 *   ⟨P⟩_noisy = (∏_q damping(P_q, q)) · ⟨P⟩_ideal
 * where damping(X,q) = 1 - 2(py_q + pz_q), etc.
 *
 * ## Coherent noise
 *
 * Instead of stochastic Pauli gates, coherent noise injects deterministic
 * rotation gates after every gate. For a depolarising probability p, the
 * rotation angle is ε = 2·arcsin(√p), which produces the same gate
 * infidelity as the corresponding Pauli channel while preserving phase
 * coherence.
 *
 * The coherent model supports per-qubit, per-axis rotation angles.
 * Convenience setters (set_coherent_depolarizing, etc.) convert from
 * error probability to angle automatically.
 */

#pragma once

#include <cmath>
#include <random>
#include <string>
#include <unordered_map>

#include "Circuit/Circuit.h"

namespace noise {

/// Per-qubit Pauli noise parameters.
struct QubitNoise {
  double px = 0.0;  ///< X (bit-flip) error probability
  double py = 0.0;  ///< Y (bit-phase-flip) error probability
  double pz = 0.0;  ///< Z (phase-flip) error probability

  /// Damping factor a single-qubit Pauli operator acquires from this channel.
  double damping_for(char pauli) const {
    switch (toupper(pauli)) {
      case 'X': return 1.0 - 2.0 * (py + pz);
      case 'Y': return 1.0 - 2.0 * (px + pz);
      case 'Z': return 1.0 - 2.0 * (px + py);
      default:  return 1.0;  // Identity
    }
  }

  double total() const { return px + py + pz; }
};

/// Per-qubit coherent noise parameters (rotation angles in radians).
struct CoherentNoise {
  double rx = 0.0;  ///< X-axis rotation angle
  double ry = 0.0;  ///< Y-axis rotation angle
  double rz = 0.0;  ///< Z-axis rotation angle
};

/**
 * @class NoiseModel
 * @brief Maps qubit indices to Pauli channel and/or coherent noise parameters.
 *
 * Supports two noise modes:
 *   1. **Pauli (incoherent)**: stochastic Pauli gate injection or analytical
 *      damping. Configured via set_depolarizing(), set_dephasing(), etc.
 *   2. **Coherent**: systematic rotation gate injection. Configured via
 *      set_coherent_depolarizing(), set_coherent_rotation(), etc.
 *
 * Both modes can be configured on the same model — the caller chooses
 * which injection function to use (inject_noise vs inject_coherent_noise).
 *
 * Usage:
 *   NoiseModel nm;
 *   nm.set_depolarizing(0, 0.01);              // Pauli: 1% depolarizing
 *   nm.set_coherent_depolarizing(1, 0.01);     // Coherent: same infidelity
 *   nm.set_coherent_rotation(2, 0.0, 0.0, 0.05); // Coherent: custom angles
 *   double d = nm.compute_damping("ZZ");       // damping for ⟨ZZ⟩
 */
class NoiseModel {
 public:
  // ── Pauli (incoherent) noise setters ──

  /// Set arbitrary Pauli channel on a qubit.
  void set_qubit_noise(int q, double px, double py, double pz) {
    noise_[q] = {px, py, pz};
  }

  /// Symmetric depolarizing: px = py = pz = p/3.
  void set_depolarizing(int q, double p) {
    noise_[q] = {p / 3.0, p / 3.0, p / 3.0};
  }

  /// Pure dephasing (Z noise only).
  void set_dephasing(int q, double p) { noise_[q] = {0, 0, p}; }

  /// Pure bit-flip (X noise only).
  void set_bit_flip(int q, double p) { noise_[q] = {p, 0, 0}; }

  /// Apply uniform depolarizing noise to qubits 0..n-1.
  void set_all_depolarizing(int n, double p) {
    for (int q = 0; q < n; ++q) set_depolarizing(q, p);
  }

  /// Apply uniform dephasing noise to qubits 0..n-1.
  void set_all_dephasing(int n, double p) {
    for (int q = 0; q < n; ++q) set_dephasing(q, p);
  }

  // ── Coherent noise setters ──

  /// Set arbitrary coherent rotation angles on a qubit (radians).
  void set_coherent_rotation(int q, double rx, double ry, double rz) {
    coherent_[q] = {rx, ry, rz};
  }

  /**
   * Coherent version of depolarizing noise.
   * Converts probability p to a Z-axis rotation angle:
   *   ε = 2·arcsin(√p)
   * This produces the same per-gate infidelity as DEPOLARIZE1(p).
   */
  void set_coherent_depolarizing(int q, double p) {
    double eps = (p > 0) ? 2.0 * std::asin(std::sqrt(p)) : 0.0;
    coherent_[q] = {0.0, 0.0, eps};
  }

  /// Coherent dephasing: Rz rotation with angle from probability.
  void set_coherent_dephasing(int q, double p) {
    double eps = (p > 0) ? 2.0 * std::asin(std::sqrt(p)) : 0.0;
    coherent_[q] = {0.0, 0.0, eps};
  }

  /// Coherent bit-flip: Rx rotation with angle from probability.
  void set_coherent_bit_flip(int q, double p) {
    double eps = (p > 0) ? 2.0 * std::asin(std::sqrt(p)) : 0.0;
    coherent_[q] = {eps, 0.0, 0.0};
  }

  /// Apply uniform coherent depolarizing noise to qubits 0..n-1.
  void set_all_coherent_depolarizing(int n, double p) {
    for (int q = 0; q < n; ++q) set_coherent_depolarizing(q, p);
  }

  /// Apply uniform coherent dephasing to qubits 0..n-1.
  void set_all_coherent_dephasing(int n, double p) {
    for (int q = 0; q < n; ++q) set_coherent_dephasing(q, p);
  }

  /**
   * Set a global coherent noise strength that scales all axes uniformly.
   * Convenience method: sets Rz angle = 2·arcsin(√p) on every qubit.
   * Equivalent to set_all_coherent_depolarizing.
   */
  void set_coherent_strength(int n, double p) {
    set_all_coherent_depolarizing(n, p);
  }

  // ── Analytical damping ──

  /**
   * Compute the multiplicative damping factor for a Pauli string observable.
   * ⟨P⟩_noisy = compute_damping(P) × ⟨P⟩_ideal
   */
  double compute_damping(const std::string &pauli) const {
    double d = 1.0;
    for (size_t q = 0; q < pauli.size(); ++q) {
      auto it = noise_.find(static_cast<int>(q));
      if (it != noise_.end()) d *= it->second.damping_for(pauli[q]);
    }
    return d;
  }

  // ── Accessors ──

  /// Get Pauli noise for a specific qubit (nullptr if none set).
  const QubitNoise *get(int q) const {
    auto it = noise_.find(q);
    return (it != noise_.end()) ? &it->second : nullptr;
  }

  /// Get coherent noise for a specific qubit (nullptr if none set).
  const CoherentNoise *get_coherent(int q) const {
    auto it = coherent_.find(q);
    return (it != coherent_.end()) ? &it->second : nullptr;
  }

  bool empty() const { return noise_.empty(); }
  bool coherent_empty() const { return coherent_.empty(); }
  bool has_coherent() const { return !coherent_.empty(); }

  // ── T1 amplitude damping setters ──

  /// Set per-gate T1 decay probability on a qubit.
  /// After each gate, with probability gamma, the qubit decays to |0⟩.
  void set_t1(int q, double gamma) { t1_[q] = gamma; }

  /// Set uniform T1 decay probability on qubits 0..n-1.
  void set_all_t1(int n, double gamma) {
    for (int q = 0; q < n; ++q) t1_[q] = gamma;
  }

  /**
   * Set T1 from physical time constants.
   * gamma = 1 - exp(-gate_time / t1_time)
   */
  void set_t1_from_time(int q, double gate_time_s, double t1_time_s) {
    t1_[q] = 1.0 - std::exp(-gate_time_s / t1_time_s);
  }

  /// Get T1 decay probability for a qubit (0 if not set).
  double get_t1(int q) const {
    auto it = t1_.find(q);
    return (it != t1_.end()) ? it->second : 0.0;
  }

  bool has_t1() const { return !t1_.empty(); }

  // ── Crosstalk setters ──

  /**
   * Set ZZ crosstalk coupling between two qubits.
   * After a gate on q1, an Rz(strength) rotation is applied on q2
   * (and vice versa). Symmetric by default.
   */
  void set_crosstalk(int q1, int q2, double strength) {
    crosstalk_[q1][q2] = strength;
    crosstalk_[q2][q1] = strength;
  }

  /// Get crosstalk neighbor map for a qubit (nullptr if none).
  const std::unordered_map<int, double> *get_crosstalk_neighbors(int q) const {
    auto it = crosstalk_.find(q);
    return (it != crosstalk_.end()) ? &it->second : nullptr;
  }

  bool has_crosstalk() const { return !crosstalk_.empty(); }

  /// True if any noise of any type has been configured.
  bool has_any() const {
    return !noise_.empty() || !coherent_.empty() ||
           !t1_.empty() || !crosstalk_.empty();
  }

 private:
  std::unordered_map<int, QubitNoise> noise_;
  std::unordered_map<int, CoherentNoise> coherent_;
  std::unordered_map<int, double> t1_;
  std::unordered_map<int, std::unordered_map<int, double>> crosstalk_;
};

/**
 * Inject random Pauli error gates into a circuit copy (Monte Carlo sample).
 * After each gate, for every affected qubit with noise, a random Pauli
 * (X, Y, Z, or I) is applied according to the channel probabilities.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    for (auto q : op->AffectedQubits()) {
      const auto *qn = nm.get(q);
      if (!qn || qn->total() <= 0) continue;

      double r = dist(rng);
      if (r < qn->px)
        out->AddOperation(std::make_shared<Circuits::XGate<>>(q));
      else if (r < qn->px + qn->py)
        out->AddOperation(std::make_shared<Circuits::YGate<>>(q));
      else if (r < qn->total())
        out->AddOperation(std::make_shared<Circuits::ZGate<>>(q));
      // else: identity (no error)
    }
  }
  return out;
}

/**
 * Inject coherent rotation noise into a circuit copy.
 *
 * After each gate, for every affected qubit with coherent noise parameters,
 * rotation gates Rx(±θx), Ry(±θy), Rz(±θz) are applied. The ± sign is
 * sampled randomly per-gate to model coherent over/under-rotation.
 *
 * This produces a SINGLE deterministic noisy circuit that should be run
 * for all shots — unlike Pauli noise where each shot samples independently.
 * For richer statistics, call this multiple times with different RNG states
 * and average the results (like noisy_execute does with noise_realizations).
 *
 * @param circ  Input circuit (not modified).
 * @param nm    NoiseModel with coherent noise parameters set.
 * @param rng   Random number generator for sign sampling.
 * @return New circuit with coherent rotation gates inserted.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_coherent_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_int_distribution<int> sign_dist(0, 1);

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    for (auto q : op->AffectedQubits()) {
      const auto *cn = nm.get_coherent(q);
      if (!cn) continue;

      // Apply rotation on each axis with non-zero angle
      if (std::abs(cn->rx) > 1e-15) {
        double sign = sign_dist(rng) ? 1.0 : -1.0;
        out->AddOperation(
            std::make_shared<Circuits::RxGate<>>(q, sign * cn->rx));
      }
      if (std::abs(cn->ry) > 1e-15) {
        double sign = sign_dist(rng) ? 1.0 : -1.0;
        out->AddOperation(
            std::make_shared<Circuits::RyGate<>>(q, sign * cn->ry));
      }
      if (std::abs(cn->rz) > 1e-15) {
        double sign = sign_dist(rng) ? 1.0 : -1.0;
        out->AddOperation(
            std::make_shared<Circuits::RzGate<>>(q, sign * cn->rz));
      }
    }
  }
  return out;
}

/**
 * Inject ALL configured noise types into a circuit in physical order:
 *   1. Coherent over-rotations (systematic gate errors)
 *   2. Crosstalk (ZZ coupling to spectator neighbors)
 *   3. T1 amplitude damping (probabilistic decay to |0⟩)
 *   4. Pauli noise (stochastic bit/phase flips)
 *
 * This is the "realistic" noise injection that combines every layer.
 * Only noise types that have been configured on the NoiseModel are applied.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_combined_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::uniform_int_distribution<int> sign_dist(0, 1);

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    auto affected = op->AffectedQubits();

    // Helper: check if qubit is in the affected set
    auto is_affected = [&affected](int q) {
      for (auto aq : affected)
        if (static_cast<int>(aq) == q) return true;
      return false;
    };

    // ── 1. Coherent over-rotations on affected qubits ──
    for (auto q : affected) {
      const auto *cn = nm.get_coherent(q);
      if (!cn) continue;
      if (std::abs(cn->rx) > 1e-15) {
        double s = sign_dist(rng) ? 1.0 : -1.0;
        out->AddOperation(
            std::make_shared<Circuits::RxGate<>>(q, s * cn->rx));
      }
      if (std::abs(cn->ry) > 1e-15) {
        double s = sign_dist(rng) ? 1.0 : -1.0;
        out->AddOperation(
            std::make_shared<Circuits::RyGate<>>(q, s * cn->ry));
      }
      if (std::abs(cn->rz) > 1e-15) {
        double s = sign_dist(rng) ? 1.0 : -1.0;
        out->AddOperation(
            std::make_shared<Circuits::RzGate<>>(q, s * cn->rz));
      }
    }

    // ── 2. Crosstalk: Rz on spectator neighbors ──
    // Accumulate crosstalk from all affected qubits, then apply once
    // per spectator (avoids double-counting).
    std::unordered_map<int, double> spectator_rotations;
    for (auto q : affected) {
      const auto *xt = nm.get_crosstalk_neighbors(q);
      if (!xt) continue;
      for (const auto &[neighbor, strength] : *xt) {
        if (!is_affected(neighbor))
          spectator_rotations[neighbor] += strength;
      }
    }
    for (const auto &[spectator, total] : spectator_rotations) {
      out->AddOperation(std::make_shared<Circuits::RzGate<>>(
          static_cast<Types::qubit_t>(spectator), total));
    }

    // ── 3. T1 amplitude damping (quantum trajectory) ──
    for (auto q : affected) {
      double gamma = nm.get_t1(q);
      if (gamma > 0 && dist(rng) < gamma) {
        out->AddOperation(std::make_shared<Circuits::Reset<>>(
            Types::qubits_vector{q}));
      }
    }

    // ── 4. Pauli (incoherent) noise ──
    for (auto q : affected) {
      const auto *qn = nm.get(q);
      if (!qn || qn->total() <= 0) continue;
      double r = dist(rng);
      if (r < qn->px)
        out->AddOperation(std::make_shared<Circuits::XGate<>>(q));
      else if (r < qn->px + qn->py)
        out->AddOperation(std::make_shared<Circuits::YGate<>>(q));
      else if (r < qn->total())
        out->AddOperation(std::make_shared<Circuits::ZGate<>>(q));
    }
  }
  return out;
}

}  // namespace noise
