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

/// Per-qubit time-correlated (OU → AR(1)) dephasing noise parameters.
struct CorrelatedNoise {
  double phi = 0.0;        ///< AR(1) coefficient Ω = exp(-θ·dt)
  double sigma_eta = 0.0;  ///< Driving noise std dev
  bool inject_after_1q = true;  ///< Inject after 1Q gates (default: yes)
  bool inject_after_2q = true;  ///< Inject after 2Q gates (default: yes)
};

/// Per-qubit readout error parameters (classical post-measurement channel).
struct ReadoutError {
  double p_meas1_prep0 = 0.0;  ///< P(measure 1 | prepared 0) — false positive
  double p_meas0_prep1 = 0.0;  ///< P(measure 0 | prepared 1) — false negative
};

/**
 * @class NoiseModel
 * @brief Maps qubit indices to noise parameters for realistic device simulation.
 *
 * Supports six noise categories, all combinable on the same model:
 *   1. **Pauli (incoherent)**: stochastic X/Y/Z gate injection or analytical
 *      damping. Configured via set_depolarizing(), set_dephasing(), etc.
 *   2. **Coherent**: systematic rotation gate injection. Configured via
 *      set_coherent_depolarizing(), set_coherent_rotation(), etc.
 *   3. **Correlated (time-correlated)**: non-Markovian dephasing via
 *      per-qubit AR(1) processes. Configured via set_correlated_ou(),
 *      set_correlated_ar1(), set_all_correlated_from_power(), etc.
 *   4. **T1 amplitude damping**: probabilistic |1⟩→|0⟩ decay.
 *   5. **ZZ crosstalk**: parasitic Rz rotations on spectator qubits.
 *   6. **Readout error**: classical post-measurement bit-flip channel.
 *
 * Use inject_combined_noise() to apply all configured layers in a single pass,
 * or use the individual inject_noise/inject_coherent_noise/inject_correlated_noise
 * functions for specific noise types.
 *
 * Usage:
 *   NoiseModel nm;
 *   nm.set_depolarizing(0, 0.01);              // Pauli: 1% depolarizing
 *   nm.set_coherent_depolarizing(1, 0.01);     // Coherent: same infidelity
 *   nm.set_all_correlated_ou(4, 15.0, 0.5, 100e-9); // Correlated: OU noise
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

  // ── Correlated (time-correlated) noise setters ──

  /**
   * Set AR(1) correlated dephasing noise on a qubit.
   * After every gate, an Rz(y[k]) rotation is injected where:
   *   y[k] = phi * y[k-1] + eta[k],  eta ~ N(0, sigma_eta²)
   *
   * @param q Qubit index.
   * @param phi AR(1) autoregressive coefficient.
   * @param sigma_eta Driving noise standard deviation.
   * @param after_1q If true (default), inject after 1Q gates too.
   */
  void set_correlated_ar1(int q, double phi, double sigma_eta,
                          bool after_1q = true, bool after_2q = true) {
    correlated_[q] = {phi, sigma_eta, after_1q, after_2q};
  }

  /**
   * Set correlated noise from Ornstein-Uhlenbeck parameters.
   * OU: dX = -θ·X·dt + σ·dW, discretized as AR(1).
   *   θ = 1/(α · gate_time)
   *   Ω = exp(-θ · gate_time)
   *   σ_η² = (σ²/2θ)(1 - Ω²)
   *
   * @param q Qubit index.
   * @param sigma OU diffusion coefficient (noise strength).
   * @param alpha Correlation time in gate-time units (τ/t_g).
   * @param gate_time Gate duration in seconds.
   * @param after_1q If true (default), inject after 1Q gates too.
   */
  void set_correlated_ou(int q, double sigma, double alpha,
                         double gate_time, bool after_1q = true,
                         bool after_2q = true) {
    double theta = 1.0 / (alpha * gate_time);
    double phi = std::exp(-theta * gate_time);
    double sigma_eta_sq = (sigma * sigma / (2.0 * theta)) * (1.0 - phi * phi);
    double sigma_eta = std::sqrt(std::max(sigma_eta_sq, 0.0));
    correlated_[q] = {phi, sigma_eta, after_1q, after_2q};
  }

  /// Set identical OU correlated noise on qubits [0, n).
  void set_all_correlated_ou(int n, double sigma, double alpha,
                             double gate_time, bool after_1q = true,
                             bool after_2q = true) {
    for (int q = 0; q < n; ++q)
      set_correlated_ou(q, sigma, alpha, gate_time, after_1q, after_2q);
  }

  /**
   * Set correlated noise from total noise power.
   * P_tot = N·σ²·π·α·t_g  →  σ = sqrt(P_tot / (N·π·α·t_g))
   *
   * @param n Number of qubits.
   * @param power Total noise power P_tot.
   * @param alpha Correlation time in gate-time units.
   * @param gate_time Gate duration.
   * @param after_1q If true (default), inject after 1Q gates too.
   */
  void set_all_correlated_from_power(int n, double power, double alpha,
                                     double gate_time,
                                     bool after_1q = true,
                                     bool after_2q = true) {
    double sigma = std::sqrt(power / (n * M_PI * alpha * gate_time));
    set_all_correlated_ou(n, sigma, alpha, gate_time, after_1q, after_2q);
  }

  /// Get correlated noise for a qubit (nullptr if not set).
  const CorrelatedNoise *get_correlated(int q) const {
    auto it = correlated_.find(q);
    return (it != correlated_.end()) ? &it->second : nullptr;
  }

  bool has_correlated() const { return !correlated_.empty();
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

  /// Set per-gate T1 decay probability on a qubit (applied after ALL gates).
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

  bool has_t1() const { return !t1_.empty() || !t1_2q_.empty(); }

  // ── T1 gate-type-specific overrides ──

  /// Set T1 decay probability applied only after two-qubit gates.
  /// When set, 2Q gates use this gamma instead of the "all gates" value.
  void set_t1_2q(int q, double gamma) { t1_2q_[q] = gamma; }

  /// Set T1 for 2Q gates from physical time constants.
  void set_t1_2q_from_time(int q, double gate_time_s, double t1_time_s) {
    t1_2q_[q] = 1.0 - std::exp(-gate_time_s / t1_time_s);
  }

  /// Get T1 decay probability for 2Q gates (falls back to get_t1 if not set).
  double get_t1_2q(int q) const {
    auto it = t1_2q_.find(q);
    if (it != t1_2q_.end()) return it->second;
    return get_t1(q);  // fallback to "all gates" T1
  }

  /// Get gate-type-aware T1: returns t1_2q if is_2q and set, else t1.
  double get_t1_for_gate(int q, bool is_2q) const {
    if (is_2q) return get_t1_2q(q);
    return get_t1(q);
  }

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

  // ── Readout error setters ──

  /**
   * Set asymmetric readout error on a qubit.
   * @param q Qubit index.
   * @param p_meas1_prep0 P(measure 1 | state was 0) — false positive.
   * @param p_meas0_prep1 P(measure 0 | state was 1) — false negative.
   */
  void set_readout_error(int q, double p_meas1_prep0, double p_meas0_prep1) {
    readout_[q] = {p_meas1_prep0, p_meas0_prep1};
  }

  /// Symmetric readout error: both directions use the same rate.
  void set_readout_error_symmetric(int q, double p_error) {
    readout_[q] = {p_error, p_error};
  }

  /// Apply uniform symmetric readout error to qubits 0..n-1.
  void set_all_readout_error(int n, double p_error) {
    for (int q = 0; q < n; ++q) readout_[q] = {p_error, p_error};
  }

  /// Get readout error for a qubit (nullptr if not set).
  const ReadoutError *get_readout_error(int q) const {
    auto it = readout_.find(q);
    return (it != readout_.end()) ? &it->second : nullptr;
  }

  bool has_readout_error() const { return !readout_.empty(); }

  // ── Two-qubit depolarizing setters ──

  /**
   * Set two-qubit depolarizing channel applied after CX/CZ gates on (q1, q2).
   * Channel: Λ(ρ) = (1-p)ρ + p/15 · Σ_{P∈{I,X,Y,Z}⊗2 \ {II}} PρP†
   * Stored symmetrically: set_2q_depolarizing(a,b,p) == set_2q_depolarizing(b,a,p).
   */
  void set_2q_depolarizing(int q1, int q2, double p) {
    depol_2q_[q1][q2] = p;
    depol_2q_[q2][q1] = p;
  }

  /// Get two-qubit depolarizing probability for a qubit pair (0 if not set).
  double get_2q_depolarizing(int q1, int q2) const {
    auto it = depol_2q_.find(q1);
    if (it == depol_2q_.end()) return 0.0;
    auto jt = it->second.find(q2);
    return (jt != it->second.end()) ? jt->second : 0.0;
  }

  bool has_2q_depolarizing(int q1, int q2) const {
    return get_2q_depolarizing(q1, q2) > 0.0;
  }

  bool has_any_2q_depolarizing() const { return !depol_2q_.empty(); }

  // ── Gate-type-specific noise setters ──

  /// Per-qubit depolarizing applied only after single-qubit gates.
  void set_1q_gate_depolarizing(int q, double p) {
    noise_1q_[q] = {p / 3.0, p / 3.0, p / 3.0};
  }

  /// Per-qubit depolarizing applied only after two-qubit gates.
  void set_2q_gate_depolarizing(int q, double p) {
    noise_2q_[q] = {p / 3.0, p / 3.0, p / 3.0};
  }

  /// Bulk: 1Q gate depolarizing on qubits 0..n-1.
  void set_all_1q_gate_depolarizing(int n, double p) {
    for (int q = 0; q < n; ++q) set_1q_gate_depolarizing(q, p);
  }

  /// Bulk: 2Q gate depolarizing on qubits 0..n-1.
  void set_all_2q_gate_depolarizing(int n, double p) {
    for (int q = 0; q < n; ++q) set_2q_gate_depolarizing(q, p);
  }

  /// Get 1Q-gate-specific noise for a qubit (nullptr if none set).
  const QubitNoise *get_1q_gate_noise(int q) const {
    auto it = noise_1q_.find(q);
    return (it != noise_1q_.end()) ? &it->second : nullptr;
  }

  /// Get 2Q-gate-specific noise for a qubit (nullptr if none set).
  const QubitNoise *get_2q_gate_noise(int q) const {
    auto it = noise_2q_.find(q);
    return (it != noise_2q_.end()) ? &it->second : nullptr;
  }

  bool has_1q_gate_noise() const { return !noise_1q_.empty(); }
  bool has_2q_gate_noise() const { return !noise_2q_.empty(); }

  /// True if any noise of any type has been configured.
  bool has_any() const {
    return !noise_.empty() || !coherent_.empty() ||
           !correlated_.empty() ||
           !t1_.empty() || !t1_2q_.empty() || !crosstalk_.empty() ||
           !readout_.empty() || !depol_2q_.empty() ||
           !noise_1q_.empty() || !noise_2q_.empty();
  }

 private:
  std::unordered_map<int, QubitNoise> noise_;
  std::unordered_map<int, CoherentNoise> coherent_;
  std::unordered_map<int, double> t1_;
  std::unordered_map<int, double> t1_2q_;  ///< T1 decay after 2Q gates only
  std::unordered_map<int, std::unordered_map<int, double>> crosstalk_;
  std::unordered_map<int, ReadoutError> readout_;
  std::unordered_map<int, std::unordered_map<int, double>> depol_2q_;
  std::unordered_map<int, QubitNoise> noise_1q_;  ///< after 1Q gates only
  std::unordered_map<int, QubitNoise> noise_2q_;  ///< after 2Q gates only
  std::unordered_map<int, CorrelatedNoise> correlated_;  ///< time-correlated
};

/// Helper: inject a single-qubit Pauli error on qubit q with probabilities qn.
inline void inject_1q_pauli_(std::shared_ptr<Circuits::Circuit<double>> &out,
                             Types::qubit_t q, const QubitNoise &qn,
                             std::uniform_real_distribution<double> &dist,
                             std::mt19937 &rng) {
  if (qn.total() <= 0) return;
  double r = dist(rng);
  if (r < qn.px)
    out->AddOperation(std::make_shared<Circuits::XGate<>>(q));
  else if (r < qn.px + qn.py)
    out->AddOperation(std::make_shared<Circuits::YGate<>>(q));
  else if (r < qn.total())
    out->AddOperation(std::make_shared<Circuits::ZGate<>>(q));
}

/**
 * Helper: inject a two-qubit depolarizing channel on (q1, q2).
 * Samples from 15 non-identity two-qubit Paulis each with probability p/15.
 * Total error probability = p.  With probability (1-p), identity is applied.
 */
inline void inject_2q_depol_(
    std::shared_ptr<Circuits::Circuit<double>> &out,
    Types::qubit_t q1, Types::qubit_t q2, double p,
    std::uniform_real_distribution<double> &dist, std::mt19937 &rng) {
  if (p <= 0) return;
  double r = dist(rng);
  if (r >= p) return;  // identity (no error)

  // 15 equally-weighted non-identity Paulis on 2 qubits.
  // Map r ∈ [0, p) to one of 15 bins.
  int idx = static_cast<int>(r / p * 15.0);
  if (idx >= 15) idx = 14;

  // Paulis: I=0, X=1, Y=2, Z=3.  Pair index = 4*a + b, skip II (0).
  // idx 0..14 → pair indices 1..15.
  int pair = idx + 1;
  int pa = pair / 4;  // Pauli on q1
  int pb = pair % 4;  // Pauli on q2

  auto apply_pauli = [&](Types::qubit_t q, int pauli_id) {
    switch (pauli_id) {
      case 1: out->AddOperation(std::make_shared<Circuits::XGate<>>(q)); break;
      case 2: out->AddOperation(std::make_shared<Circuits::YGate<>>(q)); break;
      case 3: out->AddOperation(std::make_shared<Circuits::ZGate<>>(q)); break;
      default: break;  // Identity
    }
  };
  apply_pauli(q1, pa);
  apply_pauli(q2, pb);
}

/**
 * Inject random Pauli error gates into a circuit copy (Monte Carlo sample).
 * After each gate, for every affected qubit with noise, a random Pauli
 * (X, Y, Z, or I) is applied according to the channel probabilities.
 *
 * Supports gate-type-specific noise: if 1Q/2Q gate noise maps are set,
 * they are applied in addition to the "all gates" channel.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    auto affected = op->AffectedQubits();
    bool is_2q = affected.size() >= 2;

    for (auto q : affected) {
      // T1 amplitude damping (gate-type-aware)
      double gamma = nm.get_t1_for_gate(static_cast<int>(q), is_2q);
      if (gamma > 0 && dist(rng) < gamma) {
        out->AddOperation(std::make_shared<Circuits::Reset<>>(
            Types::qubits_vector{q}));
      }

      // "All gates" channel (existing behavior)
      const auto *qn = nm.get(q);
      if (qn) inject_1q_pauli_(out, q, *qn, dist, rng);

      // Gate-type-specific channels
      if (is_2q) {
        const auto *qn2 = nm.get_2q_gate_noise(q);
        if (qn2) inject_1q_pauli_(out, q, *qn2, dist, rng);
      } else {
        const auto *qn1 = nm.get_1q_gate_noise(q);
        if (qn1) inject_1q_pauli_(out, q, *qn1, dist, rng);
      }
    }

    // Two-qubit depolarizing (correlated 2Q Pauli channel)
    if (is_2q && affected.size() >= 2) {
      auto q1 = affected[0];
      auto q2 = affected[1];
      double p2q = nm.get_2q_depolarizing(
          static_cast<int>(q1), static_cast<int>(q2));
      if (p2q > 0) inject_2q_depol_(out, q1, q2, p2q, dist, rng);
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
  std::bernoulli_distribution sign_dist(0.5);

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
 * Inject time-correlated (OU → AR(1)) dephasing noise into a circuit copy.
 *
 * Each qubit with correlated noise parameters gets an independent AR(1)
 * trajectory: y[k] = φ·y[k-1] + η[k], η ~ N(0, σ_η²).
 * After each gate affecting qubit q, an Rz(y[k]) rotation is injected.
 *
 * Unlike coherent noise (fixed amplitude, random sign), correlated noise
 * produces time-varying amplitudes with temporal correlations governed by φ.
 * This models realistic non-Markovian dephasing from 1/f or OU noise sources.
 *
 * The inject_after_1q and inject_after_2q flags on each qubit's
 * CorrelatedNoise control whether noise is injected after single-qubit
 * and/or multi-qubit gates respectively.
 *
 * @param circ  Input circuit (not modified).
 * @param nm    NoiseModel with correlated noise parameters set.
 * @param rng   Random number generator for AR(1) driving noise.
 * @return New circuit with correlated Rz gates inserted.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_correlated_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::normal_distribution<double> normal(0.0, 1.0);

  // Per-qubit AR(1) state: y[k] = phi * y[k-1] + sigma_eta * eta[k]
  std::unordered_map<int, double> state;  // current y value per qubit

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    auto affected = op->AffectedQubits();
    bool is_multiq = affected.size() >= 2;

    for (auto q : affected) {
      const auto *cn = nm.get_correlated(static_cast<int>(q));
      if (!cn) continue;

      // Check gate-type flags
      if (!is_multiq && !cn->inject_after_1q) continue;
      if (is_multiq && !cn->inject_after_2q) continue;

      // Advance AR(1): y[k] = phi * y[k-1] + sigma_eta * eta
      double prev = state[static_cast<int>(q)];  // 0 if not yet set
      double eta = normal(rng);
      double y = cn->phi * prev + cn->sigma_eta * eta;
      state[static_cast<int>(q)] = y;

      // Inject Rz(y)
      if (std::abs(y) > 1e-18) {
        out->AddOperation(
            std::make_shared<Circuits::RzGate<>>(q, y));
      }
    }
  }
  return out;
}

/**
 * Inject ALL configured noise types into a circuit in physical order:
 *   1. Correlated dephasing (time-correlated OU/AR(1) noise)
 *   2. Coherent over-rotations (systematic gate errors)
 *   3. Crosstalk (ZZ coupling to spectator neighbors)
 *   4. T1 amplitude damping (probabilistic decay to |0⟩)
 *   5. Pauli noise (stochastic bit/phase flips)
 *   6. Gate-type-specific Pauli noise
 *   7. Two-qubit depolarizing
 *
 * This is the "realistic" noise injection that combines every layer.
 * Only noise types that have been configured on the NoiseModel are applied.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_combined_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::bernoulli_distribution sign_dist(0.5);
  std::normal_distribution<double> normal_dist(0.0, 1.0);
  std::unordered_map<int, double> corr_state;  // AR(1) state per qubit

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    auto affected = op->AffectedQubits();
    bool is_2q = affected.size() >= 2;

    // Helper: check if qubit is in the affected set
    auto is_affected = [&affected](int q) {
      for (auto aq : affected)
        if (static_cast<int>(aq) == q) return true;
      return false;
    };

    // ── 1. Correlated (time-correlated) dephasing on affected qubits ──
    for (auto q : affected) {
      const auto *crn = nm.get_correlated(static_cast<int>(q));
      if (!crn) continue;
      if (!is_2q && !crn->inject_after_1q) continue;
      if (is_2q && !crn->inject_after_2q) continue;

      double prev = corr_state[static_cast<int>(q)];
      double eta = normal_dist(rng);
      double y = crn->phi * prev + crn->sigma_eta * eta;
      corr_state[static_cast<int>(q)] = y;

      if (std::abs(y) > 1e-18) {
        out->AddOperation(
            std::make_shared<Circuits::RzGate<>>(q, y));
      }
    }

    // ── 2. Coherent over-rotations on affected qubits ──
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

    // ── 3. Crosstalk: Rz on spectator neighbors ──
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

    // ── 4. T1 amplitude damping (quantum trajectory, gate-type-aware) ──
    for (auto q : affected) {
      double gamma = nm.get_t1_for_gate(static_cast<int>(q), is_2q);
      if (gamma > 0 && dist(rng) < gamma) {
        out->AddOperation(std::make_shared<Circuits::Reset<>>(
            Types::qubits_vector{q}));
      }
    }

    // ── 5. Pauli (incoherent) noise — "all gates" channel ──
    for (auto q : affected) {
      const auto *qn = nm.get(q);
      if (qn) inject_1q_pauli_(out, q, *qn, dist, rng);
    }

    // ── 6. Gate-type-specific Pauli noise ──
    for (auto q : affected) {
      if (is_2q) {
        const auto *qn2 = nm.get_2q_gate_noise(q);
        if (qn2) inject_1q_pauli_(out, q, *qn2, dist, rng);
      } else {
        const auto *qn1 = nm.get_1q_gate_noise(q);
        if (qn1) inject_1q_pauli_(out, q, *qn1, dist, rng);
      }
    }

    // ── 7. Two-qubit depolarizing (correlated 2Q Pauli channel) ──
    if (is_2q && affected.size() >= 2) {
      auto q1 = affected[0];
      auto q2 = affected[1];
      double p2q = nm.get_2q_depolarizing(
          static_cast<int>(q1), static_cast<int>(q2));
      if (p2q > 0) inject_2q_depol_(out, q1, q2, p2q, dist, rng);
    }
  }
  return out;
}

}  // namespace noise
