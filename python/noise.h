/**
 * @file noise.h
 * @brief Pauli and coherent noise models for quantum circuit simulation.
 *
 * Defines a NoiseModel that maps per-qubit noise parameters and provides:
 *   - A single-layer analytical damping estimate for expectation values.
 *   - Monte Carlo Pauli noise injection for shot-based execution.
 *   - Coherent-unitary noise injection via systematic over/under-rotations.
 *
 * ## Pauli (incoherent) noise
 *
 * Pauli channel: Λ(ρ) = (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ
 *
 * ## Analytical damping — scope and limitations
 *
 * compute_damping() returns ∏_q damping(P_q, q) for a Pauli string
 * P = P_0 ⊗ ... ⊗ P_{n-1}, where damping(X,q) = 1 - 2(py_q + pz_q), etc.
 *
 * This factor is EXACT only for a single Pauli layer applied immediately
 * before measurement:
 *   ⟨P⟩ = damping(P) · ⟨P⟩_ideal   iff the channel acts once, at the end.
 *
 * It is an APPROXIMATION for a circuit carrying noise after every gate:
 *   - Pauli channels do not commute through general (non-Clifford) gates, so
 *     they cannot be collected into one terminal layer.
 *   - Even in the commuting case the damping compounds with the number of
 *     noisy layers acting on each qubit (≈ damping^(gates on q)), which this
 *     single factor does not model.
 * The result is therefore an upper bound on ⟨P⟩ (noise is underestimated),
 * and the gap grows with circuit depth. Use noisy_execute / a density-matrix
 * or MPO backend when the magnitude of the noise matters.
 *
 * ## Coherent noise
 *
 * Instead of stochastic Pauli gates, coherent noise injects deterministic
 * rotation gates after every gate. For a depolarising probability p, the
 * rotation angle is ε = 2·arcsin(√p), which produces the same average gate
 * infidelity as the corresponding Pauli channel while preserving coherence.
 * The existing "coherent depolarizing" convenience model is specifically a
 * Z-axis error; it is not an isotropic depolarizing channel.
 *
 * The over/under-rotation sign is drawn ONCE PER QUBIT per realization and
 * then held fixed for the whole circuit, so the error is systematic and
 * accumulates coherently (the physical situation this model exists for).
 * Re-drawing the sign at every gate would instead produce a random walk whose
 * ensemble average is just the phase-flip channel with probability p, i.e.
 * nothing that set_dephasing(p) does not already provide.
 *
 * The coherent model supports per-qubit, per-axis rotation angles.
 * Convenience setters (set_coherent_depolarizing, etc.) convert from
 * error probability to angle automatically.
 */

#pragma once

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../Circuit/Circuit.h"

namespace noise {

/// Validate a probability-like parameter, mirroring QuantumChannel's checks.
/// The sampled (circuit-rewrite) injectors cannot detect bad input on their
/// own -- a negative or >1 probability silently skews the sampling -- while
/// the exact-channel path throws deep inside QuantumChannel. Validating in the
/// setters makes both backends reject the same inputs, at the point of error.
inline double checked_probability_(double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0 || value > 1.0)
    throw std::invalid_argument(std::string(name) +
                                " must be finite and in [0, 1]");
  return value;
}

/// Validate that X/Y/Z error probabilities form a valid Pauli channel.
inline void checked_pauli_probabilities_(double px, double py, double pz) {
  checked_probability_(px, "Pauli X probability");
  checked_probability_(py, "Pauli Y probability");
  checked_probability_(pz, "Pauli Z probability");
  const double total = px + py + pz;
  if (!std::isfinite(total) || total > 1.0 + 1e-12)
    throw std::invalid_argument(
        "Pauli error probabilities must sum to at most one");
}

/// Validate a duration in seconds (finite, nonnegative).
inline double checked_duration_(double value, const char *name) {
  if (!std::isfinite(value) || value < 0.0)
    throw std::invalid_argument(std::string(name) +
                                " must be finite and nonnegative");
  return value;
}

/// Validate a relaxation/coherence time (positive; infinity means "no decay").
inline double checked_time_constant_(double value, const char *name) {
  if (std::isnan(value) || value <= 0.0)
    throw std::invalid_argument(std::string(name) +
                                " must be positive (infinity is allowed)");
  return value;
}

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
  double sigma_stat = 0.0; ///< Stationary std dev σ_stat = σ_η/√(1-Ω²); AR(1) state seeded from N(0, σ_stat²) to avoid cold start
  bool inject_after_1q = true;  ///< Inject after 1Q gates (default: yes)
  bool inject_after_2q = true;  ///< Inject after 2Q gates (default: yes)
};

/// Per-qubit readout error parameters (classical post-measurement channel).
struct ReadoutError {
  double p_meas1_prep0 = 0.0;  ///< P(measure 1 | prepared 0) — false positive
  double p_meas0_prep1 = 0.0;  ///< P(measure 0 | prepared 1) — false negative
};

/** A caller-supplied local Kraus channel and the targets it is attached to. */
struct ConfiguredKrausChannel {
  Types::qubits_vector targets;
  Simulators::QuantumChannel channel;
};

/**
 * Physical T1/T2 thermal relaxation for one gate duration.
 *
 * Both the parameters and the exact channel they generate are kept, so the
 * same configuration can be realized on either backend family:
 *   - exact (density matrix / MPO): the CPTP channel, coherences decay by
 *     exactly exp(-duration/T2);
 *   - sampled (pure state / MPS): the stochastic reset+Z mixture below, which
 *     reproduces the same populations AND the same exp(-duration/T2)
 *     coherence decay.
 *
 * Deriving both from one set of (duration, T1, T2) is what keeps the two
 * paths physically consistent. Configuring T1 and T2 separately via
 * set_t1()+set_dephasing() cannot do this: a phase-flip probability tuned so
 * that (1-gamma)(1-2p_z) = exp(-t/T2) for the sampled reset model becomes
 * sqrt(1-gamma)(1-2p_z) on the exact amplitude-damping path, which
 * under-dephases by exp(t/2T1) per gate.
 */
struct ThermalRelaxation {
  double duration = 0.0;             ///< Gate duration in seconds
  double t1 = 0.0;                   ///< T1 in seconds (may be infinite)
  double t2 = 0.0;                   ///< T2 in seconds (may be infinite)
  double excited_population = 0.0;   ///< Equilibrium population of |1>

  /// Probability the qubit is reset to |0> during this gate.
  double reset_to_zero_probability() const {
    return decay_probability() * (1.0 - excited_population);
  }

  /// Probability the qubit is excited to |1> during this gate.
  double reset_to_one_probability() const {
    return decay_probability() * excited_population;
  }

  /**
   * Phase-flip probability applied to the trajectories that did not reset,
   * chosen so the total coherence multiplier is exp(-duration/T2):
   *   (1 - gamma)(1 - 2 p_z) = exp(-duration/T2).
   *
   * This has a solution only when T2 <= T1. For T1 < T2 <= 2*T1 the reset
   * mixture already destroys more coherence than physics allows and no
   * nonnegative p_z can compensate, so it is clamped to zero and the sampled
   * path over-dephases (exp(-t/T1) instead of exp(-t/T2)). Aer's
   * thermal_relaxation_error has the same limitation and switches to a Kraus
   * representation there; use a density-matrix or MPO backend, where the
   * exact channel is used instead, if that regime matters.
   */
  double phase_flip_probability() const {
    const double survival = 1.0 - decay_probability();
    if (survival <= 0.0) return 0.0;
    const double coherence = std::isinf(t2) ? 1.0 : std::exp(-duration / t2);
    const double ratio = coherence / survival;
    if (!std::isfinite(ratio) || ratio >= 1.0) return 0.0;
    return 0.5 * (1.0 - ratio);
  }

  /// gamma = 1 - exp(-duration/T1), the total probability of a T1 event.
  double decay_probability() const {
    return std::isinf(t1) ? 0.0 : -std::expm1(-duration / t1);
  }

  /**
   * True when the sampled reset+Z mixture reproduces the exact channel's
   * coherence decay exp(-duration/T2). False in the T1 < T2 <= 2*T1 window,
   * where the mixture over-dephases and a density-matrix or MPO backend is
   * required.
   */
  bool sampled_realization_is_exact() const {
    if (!std::isfinite(t1) || !std::isfinite(t2)) return true;
    return t2 <= t1 * (1.0 + 1e-12);
  }
};

/**
 * @class NoiseModel
 * @brief Maps qubit indices to noise parameters for realistic device simulation.
 *
 * Supports seven noise categories, all combinable on the same model:
 *   1. **Pauli (incoherent)**: stochastic X/Y/Z gate injection or analytical
 *      damping. Configured via set_depolarizing(), set_dephasing(), etc.
 *   2. **Coherent**: sampled coherent over/under-rotation injection. Configured via
 *      set_coherent_depolarizing(), set_coherent_rotation(), etc.
 *   3. **Correlated (time-correlated)**: non-Markovian dephasing via
 *      per-qubit AR(1) processes. Configured via set_correlated_ou(),
 *      set_correlated_ar1(), set_all_correlated_from_power(), etc.
 *   4. **T1 amplitude damping**: exact on DM/MPO, with a legacy reset
 *      approximation on pure-state circuit-rewrite paths.
 *   5. **Spectator-Z crosstalk**: parasitic Rz rotations on neighboring
 *      spectator qubits (not a genuine two-qubit ZZ interaction).
 *   6. **Readout error**: classical post-measurement bit-flip channel.
 *   7. **Additional CPTP channels**: phase damping and T1/T2 thermal
 *      relaxation (exact or sampled), plus generalized amplitude damping,
 *      correlated phase flips and caller-supplied Kraus maps. The last three
 *      require a density-matrix/MPO path; thermal with T2 > T1 does too.
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
    checked_pauli_probabilities_(px, py, pz);
    noise_[q] = {px, py, pz};
  }

  /// Symmetric depolarizing: px = py = pz = p/3.
  void set_depolarizing(int q, double p) {
    checked_probability_(p, "Depolarizing probability");
    noise_[q] = {p / 3.0, p / 3.0, p / 3.0};
  }

  /// Pure dephasing (Z noise only).
  void set_dephasing(int q, double p) {
    checked_probability_(p, "Dephasing probability");
    noise_[q] = {0, 0, p};
  }

  /// Pure bit-flip (X noise only).
  void set_bit_flip(int q, double p) {
    checked_probability_(p, "Bit-flip probability");
    noise_[q] = {p, 0, 0};
  }

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
   * Single-axis coherent error calibrated to depolarizing infidelity.
   * Converts probability p to a Z-axis rotation angle:
   *   ε = 2·arcsin(√p)
   * This produces the same per-gate infidelity as DEPOLARIZE1(p).
   */
  void set_coherent_depolarizing(int q, double p) {
    checked_probability_(p, "Coherent depolarizing probability");
    double eps = (p > 0) ? 2.0 * std::asin(std::sqrt(p)) : 0.0;
    coherent_[q] = {0.0, 0.0, eps};
  }

  /// Coherent dephasing: Rz rotation with angle from probability.
  void set_coherent_dephasing(int q, double p) {
    checked_probability_(p, "Coherent dephasing probability");
    double eps = (p > 0) ? 2.0 * std::asin(std::sqrt(p)) : 0.0;
    coherent_[q] = {0.0, 0.0, eps};
  }

  /// Coherent bit-flip: Rx rotation with angle from probability.
  void set_coherent_bit_flip(int q, double p) {
    checked_probability_(p, "Coherent bit-flip probability");
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
    // Stationary std dev σ_η/√(1-φ²); guard the φ²→1 (quasi-static) limit where
    // 1-φ² underflows — fall back to σ_η so the seed is finite (a truly static
    // process is degenerate and the caller should use a finite τ_c instead).
    double one_minus_phi2 = 1.0 - phi * phi;
    double sigma_stat = (one_minus_phi2 > 1e-15)
                            ? sigma_eta / std::sqrt(one_minus_phi2)
                            : sigma_eta;
    correlated_[q] = {phi, sigma_eta, sigma_stat, after_1q, after_2q};
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
    if (!std::isfinite(sigma) || sigma < 0.0)
      throw std::invalid_argument(
          "OU diffusion coefficient must be finite and nonnegative");
    if (!std::isfinite(alpha) || alpha <= 0.0)
      throw std::invalid_argument(
          "OU correlation time (alpha) must be finite and positive");
    if (!std::isfinite(gate_time) || gate_time <= 0.0)
      throw std::invalid_argument(
          "OU gate time must be finite and positive");
    double theta = 1.0 / (alpha * gate_time);
    double phi = std::exp(-theta * gate_time);
    double sigma_stat_sq = sigma * sigma / (2.0 * theta);  // stationary variance
    double sigma_eta_sq = sigma_stat_sq * (1.0 - phi * phi);
    double sigma_eta = std::sqrt(std::max(sigma_eta_sq, 0.0));
    double sigma_stat = std::sqrt(std::max(sigma_stat_sq, 0.0));
    correlated_[q] = {phi, sigma_eta, sigma_stat, after_1q, after_2q};
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
   * NOTE ON THE CONVENTION: P_tot here is N·∫S(ω)dω with ω in rad/s, i.e.
   * 2π times the summed stationary variance of the N AR(1) processes (each
   * process has variance σ²·α·t_g/2). If you want to specify the total
   * variance directly, use set_all_correlated_ou() with
   * σ = sqrt(2·Var_tot / (N·α·t_g)) instead.
   *
   * @param n Number of qubits (must be positive).
   * @param power Total noise power P_tot.
   * @param alpha Correlation time in gate-time units.
   * @param gate_time Gate duration.
   * @param after_1q If true (default), inject after 1Q gates too.
   */
  void set_all_correlated_from_power(int n, double power, double alpha,
                                     double gate_time,
                                     bool after_1q = true,
                                     bool after_2q = true) {
    if (n <= 0)
      throw std::invalid_argument(
          "Correlated-noise qubit count must be positive");
    if (!std::isfinite(power) || power < 0.0)
      throw std::invalid_argument(
          "Total noise power must be finite and nonnegative");
    if (!std::isfinite(alpha) || alpha <= 0.0)
      throw std::invalid_argument(
          "OU correlation time (alpha) must be finite and positive");
    if (!std::isfinite(gate_time) || gate_time <= 0.0)
      throw std::invalid_argument(
          "OU gate time must be finite and positive");
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
   * Multiplicative damping factor for a Pauli string observable from ONE
   * application of the configured Pauli channels:
   *   ⟨P⟩ = compute_damping(P) × ⟨P⟩_ideal
   *
   * Exact only when the channel acts once, immediately before measurement.
   * For a circuit with noise after every gate this underestimates the noise
   * (see the "Analytical damping" section in the file header): the factor
   * neither compounds with depth nor accounts for the fact that Pauli
   * channels do not commute through non-Clifford gates. Only the "all gates"
   * Pauli layer is included -- T1, thermal, coherent, correlated, gate-type
   * specific and two-qubit layers do not contribute to this factor at all.
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

  /**
   * Set per-gate T1 decay probability on a qubit (applied after ALL gates).
   *
   * On density-matrix/MPO backends this becomes the exact amplitude-damping
   * channel. On pure-state backends it becomes a Reset applied with
   * probability gamma, which reproduces the populations but damps coherences
   * by (1-gamma) instead of the correct sqrt(1-gamma) -- an error of order
   * gamma/2, i.e. FIRST order, not a small correction.
   *
   * Because of that, T1 alone is not enough to make the two backend families
   * agree on T2. Prefer set_thermal_relaxation(), which specifies T1 and T2
   * together and is consistent on both paths.
   */
  void set_t1(int q, double gamma) {
    checked_probability_(gamma, "T1 decay probability");
    t1_[q] = gamma;
    EnsureNoT1ThermalStack();
  }

  /// Set uniform T1 decay probability on qubits 0..n-1.
  void set_all_t1(int n, double gamma) {
    checked_probability_(gamma, "T1 decay probability");
    for (int q = 0; q < n; ++q) t1_[q] = gamma;
    EnsureNoT1ThermalStack();
  }

  /**
   * Set T1 from physical time constants.
   * gamma = 1 - exp(-gate_time / t1_time)
   */
  void set_t1_from_time(int q, double gate_time_s, double t1_time_s) {
    checked_duration_(gate_time_s, "T1 gate time");
    checked_time_constant_(t1_time_s, "T1");
    t1_[q] = std::isinf(t1_time_s) ? 0.0
                                   : -std::expm1(-gate_time_s / t1_time_s);
    EnsureNoT1ThermalStack();
  }

  /// Get T1 decay probability for a qubit (0 if not set).
  double get_t1(int q) const {
    auto it = t1_.find(q);
    return (it != t1_.end()) ? it->second : 0.0;
  }

  bool has_t1() const { return !t1_.empty() || !t1_2q_.empty(); }

  // ── Dephasing / thermal relaxation (consistent on both backends) ──

  /**
   * Phase damping with coherence multiplier sqrt(1-gamma).
   *
   * Phase damping and the stochastic phase flip are THE SAME channel: both
   * leave populations alone and multiply coherences by a constant, so
   * sqrt(1-gamma) = 1 - 2p. The equivalent flip probability is stored so the
   * sampled backends can realize this channel exactly rather than reject it.
   */
  void set_phase_damping(int q, double gamma) {
    checked_probability_(gamma, "Phase-damping probability");
    phase_damping_.insert_or_assign(
        q, Simulators::QuantumChannel::PhaseDamping(gamma));
    // sqrt(1 - gamma) = 1 - 2p  =>  p = (1 - sqrt(1 - gamma)) / 2
    phase_damping_flip_probability_.insert_or_assign(
        q, 0.5 * (1.0 - std::sqrt(std::max(0.0, 1.0 - gamma))));
  }

  /** Pure phase damping with coherence exp(-gate_time/T_phi). */
  void set_phase_damping_from_time(int q, double gate_time_s,
                                   double t_phi_s) {
    checked_duration_(gate_time_s, "Phase-damping gate time");
    checked_time_constant_(t_phi_s, "T_phi");
    const double gamma = std::isinf(t_phi_s)
                             ? 0.0
                             : -std::expm1(-2.0 * gate_time_s / t_phi_s);
    set_phase_damping(q, gamma);
  }

  /** Finite-temperature amplitude damping after every gate on a qubit. */
  void set_generalized_amplitude_damping(int q, double gamma,
                                         double excited_population) {
    generalized_amplitude_damping_.insert_or_assign(
        q, Simulators::QuantumChannel::GeneralizedAmplitudeDamping(
               gamma, excited_population));
  }

  /**
   * Hardware-style T1/T2 thermal relaxation after every gate on a qubit.
   *
   * This is the preferred way to specify decoherence: because T1 and T2 are
   * given together, both backend families can be driven from one physical
   * description and agree on the resulting coherence decay exp(-t/T2).
   * Specifying set_t1() and set_dephasing() separately cannot achieve that
   * (see the ThermalRelaxation struct docs).
   *
   * The physical constraint T2 <= 2*T1 is enforced by QuantumChannel.
   */
  void set_thermal_relaxation(int q, double gate_time_s, double t1_s,
                              double t2_s,
                              double excited_population = 0.0) {
    // Constructing the channel validates duration, T1, T2, T2 <= 2*T1 and
    // complete positivity, so an invalid configuration is rejected here
    // rather than at injection time, on either backend.
    thermal_relaxation_.insert_or_assign(
        q, Simulators::QuantumChannel::ThermalRelaxation(
               gate_time_s, t1_s, t2_s, excited_population));
    thermal_relaxation_params_.insert_or_assign(
        q, ThermalRelaxation{gate_time_s, t1_s, t2_s, excited_population});
    EnsureNoT1ThermalStack();
  }

  /**
   * Thermal relaxation applied only after two-qubit gates.
   * When set, 2Q gates use this instead of the "all gates" relaxation --
   * the usual case, since 2Q gates are longer than 1Q gates.
   */
  void set_thermal_relaxation_2q(int q, double gate_time_s, double t1_s,
                                 double t2_s,
                                 double excited_population = 0.0) {
    thermal_relaxation_2q_.insert_or_assign(
        q, Simulators::QuantumChannel::ThermalRelaxation(
               gate_time_s, t1_s, t2_s, excited_population));
    thermal_relaxation_params_2q_.insert_or_assign(
        q, ThermalRelaxation{gate_time_s, t1_s, t2_s, excited_population});
    EnsureNoT1ThermalStack();
  }

  const Simulators::QuantumChannel *get_phase_damping(int q) const {
    const auto it = phase_damping_.find(q);
    return it == phase_damping_.end() ? nullptr : &it->second;
  }

  /// Phase-flip probability equivalent to the configured phase damping.
  double get_phase_damping_flip_probability(int q) const {
    const auto it = phase_damping_flip_probability_.find(q);
    return it == phase_damping_flip_probability_.end() ? 0.0 : it->second;
  }

  const Simulators::QuantumChannel *get_generalized_amplitude_damping(
      int q) const {
    const auto it = generalized_amplitude_damping_.find(q);
    return it == generalized_amplitude_damping_.end() ? nullptr : &it->second;
  }

  const Simulators::QuantumChannel *get_thermal_relaxation(int q) const {
    const auto it = thermal_relaxation_.find(q);
    return it == thermal_relaxation_.end() ? nullptr : &it->second;
  }

  /// Thermal-relaxation channel for a gate type (2Q override, else all-gates).
  const Simulators::QuantumChannel *get_thermal_relaxation_for_gate(
      int q, bool is_2q) const {
    if (is_2q) {
      const auto it = thermal_relaxation_2q_.find(q);
      if (it != thermal_relaxation_2q_.end()) return &it->second;
    }
    return get_thermal_relaxation(q);
  }

  /// Thermal-relaxation parameters for a gate type (2Q override, else all).
  const ThermalRelaxation *get_thermal_relaxation_params(int q,
                                                         bool is_2q) const {
    if (is_2q) {
      const auto it = thermal_relaxation_params_2q_.find(q);
      if (it != thermal_relaxation_params_2q_.end()) return &it->second;
    }
    const auto it = thermal_relaxation_params_.find(q);
    return it == thermal_relaxation_params_.end() ? nullptr : &it->second;
  }

  bool has_thermal_relaxation() const {
    return !thermal_relaxation_.empty() || !thermal_relaxation_2q_.empty();
  }

  // ── T1 gate-type-specific overrides ──

  /// Set T1 decay probability applied only after two-qubit gates.
  /// When set, 2Q gates use this gamma instead of the "all gates" value.
  void set_t1_2q(int q, double gamma) {
    checked_probability_(gamma, "2Q T1 decay probability");
    t1_2q_[q] = gamma;
    EnsureNoT1ThermalStack();
  }

  /// Set T1 for 2Q gates from physical time constants.
  void set_t1_2q_from_time(int q, double gate_time_s, double t1_time_s) {
    checked_duration_(gate_time_s, "2Q T1 gate time");
    checked_time_constant_(t1_time_s, "T1");
    t1_2q_[q] = std::isinf(t1_time_s)
                    ? 0.0
                    : -std::expm1(-gate_time_s / t1_time_s);
    EnsureNoT1ThermalStack();
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
   * Set spectator-Z crosstalk coupling between two qubits.
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
    checked_probability_(p_meas1_prep0, "P(measure 1 | prepared 0)");
    checked_probability_(p_meas0_prep1, "P(measure 0 | prepared 1)");
    readout_[q] = {p_meas1_prep0, p_meas0_prep1};
  }

  /// Symmetric readout error: both directions use the same rate.
  void set_readout_error_symmetric(int q, double p_error) {
    checked_probability_(p_error, "Readout error probability");
    readout_[q] = {p_error, p_error};
  }

  /// Apply uniform symmetric readout error to qubits 0..n-1.
  void set_all_readout_error(int n, double p_error) {
    checked_probability_(p_error, "Readout error probability");
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
    if (q1 == q2)
      throw std::invalid_argument(
          "Two-qubit depolarizing qubits must be distinct");
    checked_probability_(p, "Two-qubit depolarizing probability");
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

  /**
   * Correlated phase flip after a two-qubit gate on the configured pair.
   * correlation=0 gives independent phase flips; correlation=1 gives II/ZZ.
   */
  void set_correlated_phase_flip(int q1, int q2, double probability,
                                 double correlation = 1.0) {
    if (q1 == q2)
      throw std::invalid_argument(
          "Correlated phase-flip qubits must be distinct");
    const auto channel = Simulators::QuantumChannel::CorrelatedPhaseFlip(
        probability, correlation);
    correlated_phase_flip_[q1].insert_or_assign(q2, channel);
    correlated_phase_flip_[q2].insert_or_assign(q1, channel);
  }

  const Simulators::QuantumChannel *get_correlated_phase_flip(int q1,
                                                               int q2) const {
    const auto first = correlated_phase_flip_.find(q1);
    if (first == correlated_phase_flip_.end()) return nullptr;
    const auto second = first->second.find(q2);
    return second == first->second.end() ? nullptr : &second->second;
  }

  /**
   * Attach an arbitrary one- or two-qubit CPTP channel after gates acting on
   * the same target set. Reconfiguring an identical ordered target list
   * replaces the previous custom channel.
   */
  void set_kraus_channel(
      const Types::qubits_vector &targets,
      const Simulators::QuantumChannel::KrausOperators &kraus_operators) {
    if (targets.empty() || targets.size() > 2)
      throw std::invalid_argument(
          "NoiseModel Kraus channels support one or two targets");
    if (targets.size() == 2 && targets[0] == targets[1])
      throw std::invalid_argument(
          "NoiseModel Kraus-channel targets must be distinct");
    Simulators::QuantumChannel channel(kraus_operators);
    if (channel.GetNumberOfQubits() != targets.size())
      throw std::invalid_argument(
          "Kraus dimensions do not match the configured targets");
    for (ConfiguredKrausChannel &configured : kraus_channels_) {
      if (configured.targets == targets) {
        configured.channel = std::move(channel);
        return;
      }
    }
    kraus_channels_.push_back({targets, std::move(channel)});
  }

  const std::vector<ConfiguredKrausChannel> &get_kraus_channels() const {
    return kraus_channels_;
  }

  /**
   * True if a configured channel can only be realized on an exact
   * (density-matrix / MPO) backend.
   *
   * Phase damping and thermal relaxation are deliberately NOT in this set:
   * both have an exact or well-defined stochastic realization that the
   * sampled backends use, so they no longer force an exact backend.
   *   - phase damping IS the phase-flip channel (sqrt(1-gamma) = 1-2p);
   *   - thermal relaxation maps to the reset+Z mixture, exact for T2 <= T1
   *     (T2 > T1 is reported by requires_exact_quantum_channels() instead).
   * Generalized amplitude damping, correlated phase flips and caller-supplied
   * Kraus maps have no such realization and still require an exact backend.
   */
  bool has_additional_quantum_channels() const {
    return !generalized_amplitude_damping_.empty() ||
           !correlated_phase_flip_.empty() || !kraus_channels_.empty();
  }

  /**
   * True if any configured thermal-relaxation layer sits in T1 < T2 <= 2*T1,
   * where the sampled reset+Z mixture over-dephases. Density-matrix or MPO
   * execution is required in that regime.
   */
  bool has_thermal_in_sampled_overdephasing_regime() const {
    for (const auto &entry : thermal_relaxation_params_)
      if (!entry.second.sampled_realization_is_exact()) return true;
    for (const auto &entry : thermal_relaxation_params_2q_)
      if (!entry.second.sampled_realization_is_exact()) return true;
    return false;
  }

  /**
   * True if sampled (circuit-rewrite) injection cannot realize the model:
   * exact-only Kraus maps, or thermal relaxation with T2 > T1.
   */
  bool requires_exact_quantum_channels() const {
    return has_additional_quantum_channels() ||
           has_thermal_in_sampled_overdephasing_regime();
  }

  /**
   * True iff compute_damping() captures every layer that affects Pauli
   * expectations. Readout is ignored (it is a classical post-measurement
   * channel). Thermal, T1, gate-type Pauli, 2Q depolarizing, phase damping,
   * coherent, correlated and crosstalk layers all make this false — use
   * Monte Carlo or an exact density-matrix/MPO execution instead.
   */
  bool compute_damping_covers_model() const {
    return noise_1q_.empty() && noise_2q_.empty() && depol_2q_.empty() &&
           t1_.empty() && t1_2q_.empty() && !has_thermal_relaxation() &&
           phase_damping_.empty() && coherent_.empty() && correlated_.empty() &&
           crosstalk_.empty() && !has_additional_quantum_channels();
  }

  /**
   * T1 amplitude damping and T1/T2 thermal relaxation both implement T1
   * decay. Applying them on the same qubit and gate type double-counts it.
   * Called from the setters and from the injectors.
   */
  void EnsureNoT1ThermalStack() const {
    std::unordered_set<int> qubits;
    for (const auto &entry : t1_) qubits.insert(entry.first);
    for (const auto &entry : t1_2q_) qubits.insert(entry.first);
    for (const auto &entry : thermal_relaxation_) qubits.insert(entry.first);
    for (const auto &entry : thermal_relaxation_2q_)
      qubits.insert(entry.first);
    for (const int q : qubits) {
      if (get_t1(q) != 0.0 && get_thermal_relaxation(q) != nullptr)
        throw std::invalid_argument(
            "set_t1 and set_thermal_relaxation both apply after 1-qubit "
            "gates on qubit " +
            std::to_string(q) +
            "; they would double-count T1. Use set_thermal_relaxation only.");
      if (get_t1_for_gate(q, true) != 0.0 &&
          get_thermal_relaxation_for_gate(q, true) != nullptr)
        throw std::invalid_argument(
            "T1 and thermal relaxation both apply after 2-qubit gates on "
            "qubit " +
            std::to_string(q) +
            "; they would double-count T1. Use set_thermal_relaxation / "
            "set_thermal_relaxation_2q only.");
    }
  }

  // ── Gate-type-specific noise setters ──

  /// Per-qubit depolarizing applied only after single-qubit gates.
  void set_1q_gate_depolarizing(int q, double p) {
    checked_probability_(p, "1Q-gate depolarizing probability");
    noise_1q_[q] = {p / 3.0, p / 3.0, p / 3.0};
  }

  /// Per-qubit depolarizing applied only after two-qubit gates.
  void set_2q_gate_depolarizing(int q, double p) {
    checked_probability_(p, "2Q-gate depolarizing probability");
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
           !noise_1q_.empty() || !noise_2q_.empty() ||
           !phase_damping_.empty() || has_thermal_relaxation() ||
           has_additional_quantum_channels();
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
  std::unordered_map<int, Simulators::QuantumChannel> phase_damping_;
  /// Phase-flip probability equivalent to phase_damping_ (same channel).
  std::unordered_map<int, double> phase_damping_flip_probability_;
  std::unordered_map<int, Simulators::QuantumChannel>
      generalized_amplitude_damping_;
  std::unordered_map<int, Simulators::QuantumChannel> thermal_relaxation_;
  std::unordered_map<int, Simulators::QuantumChannel> thermal_relaxation_2q_;
  /// Parameters behind thermal_relaxation_, kept so the sampled backends can
  /// build an equivalent reset+Z mixture instead of rejecting the model.
  std::unordered_map<int, ThermalRelaxation> thermal_relaxation_params_;
  std::unordered_map<int, ThermalRelaxation> thermal_relaxation_params_2q_;
  std::unordered_map<
      int, std::unordered_map<int, Simulators::QuantumChannel>>
      correlated_phase_flip_;
  std::vector<ConfiguredKrausChannel> kraus_channels_;
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

/// Append the exact CPTP form of a configured single-qubit Pauli channel.
inline void inject_1q_pauli_exact_(
    std::shared_ptr<Circuits::Circuit<double>> &out, Types::qubit_t q,
    const QubitNoise &qn) {
  if (qn.px == 0.0 && qn.py == 0.0 && qn.pz == 0.0) return;
  out->AddOperation(
      std::make_shared<Circuits::QuantumChannelOperation<>>(
          Types::qubits_vector{q},
          Simulators::QuantumChannel::Pauli(qn.px, qn.py, qn.pz)));
}

/**
 * Append the T1/T2 thermal relaxation configured for a qubit and gate type.
 *
 * Exact backends get the CPTP channel; sampled backends get the equivalent
 * stochastic mixture (reset to |0>/|1> with the T1 probabilities, else a
 * phase flip sized so the surviving trajectories carry the whole
 * exp(-duration/T2) coherence decay). Both are driven from the same physical
 * (duration, T1, T2, excited population), so the two paths agree.
 */
inline void inject_thermal_relaxation_(
    std::shared_ptr<Circuits::Circuit<double>> &out, const NoiseModel &nm,
    Types::qubit_t q, bool is_2q, bool exact_channels,
    std::uniform_real_distribution<double> &dist, std::mt19937 &rng) {
  const int qi = static_cast<int>(q);
  if (exact_channels) {
    if (const auto *channel = nm.get_thermal_relaxation_for_gate(qi, is_2q))
      out->AddOperation(
          std::make_shared<Circuits::QuantumChannelOperation<>>(
              Types::qubits_vector{q}, *channel));
    return;
  }

  const ThermalRelaxation *params =
      nm.get_thermal_relaxation_params(qi, is_2q);
  if (!params) return;

  const double reset0 = params->reset_to_zero_probability();
  const double reset1 = params->reset_to_one_probability();
  const double r = dist(rng);
  if (r < reset0) {
    out->AddOperation(
        std::make_shared<Circuits::Reset<>>(Types::qubits_vector{q}));
    return;
  }
  if (r < reset0 + reset1) {
    // Reset targets |0>, so an X afterwards realizes the thermal excitation.
    out->AddOperation(
        std::make_shared<Circuits::Reset<>>(Types::qubits_vector{q}));
    out->AddOperation(std::make_shared<Circuits::XGate<>>(q));
    return;
  }

  const double pz = params->phase_flip_probability();
  if (pz > 0.0 && dist(rng) < pz)
    out->AddOperation(std::make_shared<Circuits::ZGate<>>(q));
}

/**
 * Append the configured phase damping for a qubit.
 *
 * Phase damping and the stochastic phase flip are the same channel, so the
 * sampled path realizes it exactly rather than rejecting the model.
 */
inline void inject_phase_damping_(
    std::shared_ptr<Circuits::Circuit<double>> &out, const NoiseModel &nm,
    Types::qubit_t q, bool exact_channels,
    std::uniform_real_distribution<double> &dist, std::mt19937 &rng) {
  const int qi = static_cast<int>(q);
  if (exact_channels) {
    if (const auto *channel = nm.get_phase_damping(qi))
      out->AddOperation(
          std::make_shared<Circuits::QuantumChannelOperation<>>(
              Types::qubits_vector{q}, *channel));
    return;
  }

  const double p = nm.get_phase_damping_flip_probability(qi);
  if (p > 0.0 && dist(rng) < p)
    out->AddOperation(std::make_shared<Circuits::ZGate<>>(q));
}

/** Append the extra channels that are intentionally exact-backend-only. */
inline void inject_additional_exact_channels_(
    std::shared_ptr<Circuits::Circuit<double>> &out, const NoiseModel &nm,
    const Types::qubits_vector &affected) {
  for (const Types::qubit_t q : affected) {
    const int qi = static_cast<int>(q);
    if (const auto *channel = nm.get_generalized_amplitude_damping(qi))
      out->AddOperation(
          std::make_shared<Circuits::QuantumChannelOperation<>>(
              Types::qubits_vector{q}, *channel));

    for (const ConfiguredKrausChannel &configured :
         nm.get_kraus_channels())
      if (configured.targets.size() == 1 && configured.targets[0] == q)
        out->AddOperation(
            std::make_shared<Circuits::QuantumChannelOperation<>>(
                configured.targets, configured.channel));
  }

  if (affected.size() != 2) return;
  const Types::qubit_t q0 = affected[0];
  const Types::qubit_t q1 = affected[1];
  if (const auto *channel = nm.get_correlated_phase_flip(
          static_cast<int>(q0), static_cast<int>(q1)))
    out->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<>>(
            Types::qubits_vector{q0, q1}, *channel));

  for (const ConfiguredKrausChannel &configured : nm.get_kraus_channels()) {
    if (configured.targets.size() != 2) continue;
    const bool sameTargetSet =
        (configured.targets[0] == q0 && configured.targets[1] == q1) ||
        (configured.targets[0] == q1 && configured.targets[1] == q0);
    if (sameTargetSet)
      out->AddOperation(
          std::make_shared<Circuits::QuantumChannelOperation<>>(
              configured.targets, configured.channel));
  }
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
inline std::shared_ptr<Circuits::Circuit<double>> inject_noise_impl_(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng, bool exact_channels) {
  nm.EnsureNoT1ThermalStack();
  if (!exact_channels && nm.has_additional_quantum_channels())
    throw std::invalid_argument(
        "Generalized amplitude damping, correlated phase flips and arbitrary "
        "Kraus channels require a density-matrix or MPO exact-noise backend");
  if (!exact_channels && nm.has_thermal_in_sampled_overdephasing_regime())
    throw std::invalid_argument(
        "Thermal relaxation with T2 > T1 cannot be realized by the sampled "
        "reset+Z mixture (it over-dephases as exp(-t/T1)). Use a "
        "density-matrix or MPO exact-noise backend");

  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    auto affected = op->AffectedQubits();
    const bool is_2q = affected.size() == 2;

    for (auto q : affected) {
      // T1 amplitude damping (gate-type-aware). The sampled realization is a
      // Reset, which gets the populations right but damps coherences by
      // (1-gamma) instead of sqrt(1-gamma) -- a first-order error. Use
      // set_thermal_relaxation() to specify T1 and T2 together and get
      // matching coherence decay on both backend families.
      double gamma = nm.get_t1_for_gate(static_cast<int>(q), is_2q);
      if (exact_channels) {
        if (gamma != 0.0)
          out->AddOperation(
              std::make_shared<Circuits::QuantumChannelOperation<>>(
                  Types::qubits_vector{q},
                  Simulators::QuantumChannel::AmplitudeDamping(gamma)));
      } else if (gamma > 0 && dist(rng) < gamma) {
          out->AddOperation(std::make_shared<Circuits::Reset<>>(
              Types::qubits_vector{q}));
      }

      // T1/T2 thermal relaxation (exact channel or equivalent reset+Z sample)
      inject_thermal_relaxation_(out, nm, q, is_2q, exact_channels, dist, rng);

      // Phase damping (identical to a phase flip, so exact on both paths)
      inject_phase_damping_(out, nm, q, exact_channels, dist, rng);

      // "All gates" channel (existing behavior)
      const auto *qn = nm.get(static_cast<int>(q));
      if (qn) {
        if (exact_channels)
          inject_1q_pauli_exact_(out, q, *qn);
        else
          inject_1q_pauli_(out, q, *qn, dist, rng);
      }

      // Gate-type-specific channels
      if (is_2q) {
        const auto *qn2 = nm.get_2q_gate_noise(static_cast<int>(q));
        if (qn2) {
          if (exact_channels)
            inject_1q_pauli_exact_(out, q, *qn2);
          else
            inject_1q_pauli_(out, q, *qn2, dist, rng);
        }
      } else if (affected.size() == 1) {
        const auto *qn1 = nm.get_1q_gate_noise(static_cast<int>(q));
        if (qn1) {
          if (exact_channels)
            inject_1q_pauli_exact_(out, q, *qn1);
          else
            inject_1q_pauli_(out, q, *qn1, dist, rng);
        }
      }
    }

    // Additional exact CPTP channels.
    if (exact_channels)
      inject_additional_exact_channels_(out, nm, affected);

    // Two-qubit depolarizing (correlated 2Q Pauli channel)
    if (is_2q) {
      auto q1 = affected[0];
      auto q2 = affected[1];
      double p2q = nm.get_2q_depolarizing(
          static_cast<int>(q1), static_cast<int>(q2));
      if (exact_channels) {
        if (p2q != 0.0)
          out->AddOperation(
              std::make_shared<Circuits::QuantumChannelOperation<>>(
                  Types::qubits_vector{q1, q2},
                  Simulators::QuantumChannel::TwoQubitDepolarizing(p2q)));
      } else if (p2q > 0) {
        inject_2q_depol_(out, q1, q2, p2q, dist, rng);
      }
    }
  }
  return out;
}

/** Preserve the legacy sampled-trajectory behavior for pure-state backends. */
inline std::shared_ptr<Circuits::Circuit<double>> inject_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  return inject_noise_impl_(circ, nm, rng, false);
}

/** Insert deterministic Kraus-channel operations for DM/MPO execution. */
inline std::shared_ptr<Circuits::Circuit<double>> inject_exact_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm) {
  // The implementation does not consume randomness in exact mode, but sharing
  // one implementation guarantees identical gate scheduling in both paths.
  std::mt19937 unused;
  return inject_noise_impl_(circ, nm, unused, true);
}

/**
 * Per-qubit, per-axis over/under-rotation signs for one noise realization.
 *
 * The sign of a miscalibration is a property of the DEVICE, not of each
 * individual gate: a qubit that over-rotates does so every time. Drawing the
 * sign once per realization and holding it fixed is what makes the error
 * systematic, so it accumulates coherently (amplitude ~ n·eps, and therefore
 * infidelity ~ n²·eps²) over the n gates on that qubit.
 *
 * Re-drawing per gate would instead produce a random walk whose ensemble
 * average is exactly the phase-flip channel with probability p = sin²(eps/2)
 * -- i.e. nothing beyond set_dephasing(p), and with none of the coherent
 * accumulation this model exists to capture.
 */
class CoherentSigns {
 public:
  explicit CoherentSigns(std::mt19937 &rng) : rng_(&rng) {}

  /// Signs for qubit q, drawn on first use and stable thereafter.
  const std::array<double, 3> &for_qubit(int q) {
    auto it = signs_.find(q);
    if (it != signs_.end()) return it->second;

    std::bernoulli_distribution sign_dist(0.5);
    std::array<double, 3> drawn = {sign_dist(*rng_) ? 1.0 : -1.0,
                                   sign_dist(*rng_) ? 1.0 : -1.0,
                                   sign_dist(*rng_) ? 1.0 : -1.0};
    return signs_.emplace(q, drawn).first->second;
  }

 private:
  std::mt19937 *rng_;
  std::unordered_map<int, std::array<double, 3>> signs_;
};

/// Append the coherent over/under-rotations configured for a qubit.
inline void inject_coherent_rotations_(
    std::shared_ptr<Circuits::Circuit<double>> &out, const NoiseModel &nm,
    Types::qubit_t q, CoherentSigns &signs) {
  const auto *cn = nm.get_coherent(static_cast<int>(q));
  if (!cn) return;

  const auto &sign = signs.for_qubit(static_cast<int>(q));
  if (std::abs(cn->rx) > 1e-15)
    out->AddOperation(
        std::make_shared<Circuits::RxGate<>>(q, sign[0] * cn->rx));
  if (std::abs(cn->ry) > 1e-15)
    out->AddOperation(
        std::make_shared<Circuits::RyGate<>>(q, sign[1] * cn->ry));
  if (std::abs(cn->rz) > 1e-15)
    out->AddOperation(
        std::make_shared<Circuits::RzGate<>>(q, sign[2] * cn->rz));
}

/**
 * Inject coherent rotation noise into a circuit copy.
 *
 * After each gate, for every affected qubit with coherent noise parameters,
 * rotation gates Rx(±θx), Ry(±θy), Rz(±θz) are applied. The ± sign is drawn
 * once per qubit and axis for the whole circuit, modelling a systematic
 * over/under-rotation (see CoherentSigns).
 *
 * This produces a SINGLE deterministic noisy circuit that should be run
 * for all shots — unlike Pauli noise where each shot samples independently.
 * For richer statistics, call this multiple times with different RNG states
 * and average the results (like noisy_execute does with noise_realizations);
 * that ensemble is what averages over the sign of the miscalibration.
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
  CoherentSigns signs(rng);

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    for (auto q : op->AffectedQubits())
      inject_coherent_rotations_(out, nm, q, signs);
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
 * A single OU process has a Lorentzian spectrum. More general AR(1)
 * parameters can be used phenomenologically, but do not by themselves model
 * a full 1/f spectrum.
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

      // Advance AR(1): y[k] = phi * y[k-1] + sigma_eta * eta. Seed from the
      // stationary distribution N(0, σ_stat²) on first touch (see the note in
      // inject_combined_noise) rather than climbing off 0.
      int qi = static_cast<int>(q);
      auto it = state.find(qi);
      double prev = (it == state.end())
                        ? cn->sigma_stat * normal(rng)
                        : it->second;
      double eta = normal(rng);
      double y = cn->phi * prev + cn->sigma_eta * eta;
      state[qi] = y;

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
 *   3. Spectator-Z crosstalk
 *   4. T1 amplitude damping, T1/T2 thermal relaxation, phase damping
 *   5. Pauli noise
 *   6. Gate-type-specific Pauli noise
 *   7. Additional exact CPTP channels (on density-matrix/MPO backends)
 *   8. Two-qubit depolarizing
 *
 * This is the "realistic" noise injection that combines every layer.
 * Only noise types that have been configured on the NoiseModel are applied.
 */
inline std::shared_ptr<Circuits::Circuit<double>> inject_combined_noise_impl_(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng, bool exact_channels) {
  nm.EnsureNoT1ThermalStack();
  if (!exact_channels && nm.has_additional_quantum_channels())
    throw std::invalid_argument(
        "Generalized amplitude damping, correlated phase flips and arbitrary "
        "Kraus channels require a density-matrix or MPO exact-noise backend");
  if (!exact_channels && nm.has_thermal_in_sampled_overdephasing_regime())
    throw std::invalid_argument(
        "Thermal relaxation with T2 > T1 cannot be realized by the sampled "
        "reset+Z mixture (it over-dephases as exp(-t/T1)). Use a "
        "density-matrix or MPO exact-noise backend");

  auto out = std::make_shared<Circuits::Circuit<double>>();
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::normal_distribution<double> normal_dist(0.0, 1.0);
  CoherentSigns coherent_signs(rng);            // systematic, per realization
  std::unordered_map<int, double> corr_state;   // AR(1) state per qubit

  for (const auto &op : circ->GetOperations()) {
    out->AddOperation(op->Clone());

    if (op->GetType() != Circuits::OperationType::kGate) continue;

    auto affected = op->AffectedQubits();
    const bool is_multiq = affected.size() >= 2;
    const bool is_2q = affected.size() == 2;

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
      if (!is_multiq && !crn->inject_after_1q) continue;
      if (is_multiq && !crn->inject_after_2q) continue;

      int qi = static_cast<int>(q);
      auto it = corr_state.find(qi);
      // Seed the AR(1) state from its stationary distribution N(0, σ_stat²)
      // so a long-τ_c (Ω→1) process delivers its full variance from
      // the start instead of climbing off 0 over the (finite) circuit.
      double prev = (it == corr_state.end())
                        ? crn->sigma_stat * normal_dist(rng)
                        : it->second;
      double eta = normal_dist(rng);
      double y = crn->phi * prev + crn->sigma_eta * eta;
      corr_state[qi] = y;

      if (std::abs(y) > 1e-18) {
        out->AddOperation(
            std::make_shared<Circuits::RzGate<>>(q, y));
      }
    }

    // ── 2. Coherent over-rotations on affected qubits ──
    for (auto q : affected)
      inject_coherent_rotations_(out, nm, q, coherent_signs);

    // ── 3. Crosstalk: Rz on spectator neighbors ──
    // Accumulate crosstalk from all affected qubits, then apply once
    // per spectator (avoids double-counting).
    std::unordered_map<int, double> spectator_rotations;
    for (auto q : affected) {
      const auto *xt = nm.get_crosstalk_neighbors(static_cast<int>(q));
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

    // ── 4. T1 amplitude damping (exact channel or legacy reset sample) ──
    for (auto q : affected) {
      double gamma = nm.get_t1_for_gate(static_cast<int>(q), is_2q);
      if (exact_channels) {
        if (gamma != 0.0)
          out->AddOperation(
              std::make_shared<Circuits::QuantumChannelOperation<>>(
                  Types::qubits_vector{q},
                  Simulators::QuantumChannel::AmplitudeDamping(gamma)));
      } else if (gamma > 0 && dist(rng) < gamma) {
          out->AddOperation(std::make_shared<Circuits::Reset<>>(
              Types::qubits_vector{q}));
      }

      // T1/T2 thermal relaxation and phase damping. Both are driven from the
      // same physical parameters on either backend, so the coherence decay
      // matches; see inject_thermal_relaxation_.
      inject_thermal_relaxation_(out, nm, q, is_2q, exact_channels, dist, rng);
      inject_phase_damping_(out, nm, q, exact_channels, dist, rng);
    }

    // ── 5. Pauli (incoherent) noise — "all gates" channel ──
    for (auto q : affected) {
      const auto *qn = nm.get(static_cast<int>(q));
      if (qn) {
        if (exact_channels)
          inject_1q_pauli_exact_(out, q, *qn);
        else
          inject_1q_pauli_(out, q, *qn, dist, rng);
      }
    }

    // ── 6. Gate-type-specific Pauli noise ──
    for (auto q : affected) {
      if (is_2q) {
        const auto *qn2 = nm.get_2q_gate_noise(static_cast<int>(q));
        if (qn2) {
          if (exact_channels)
            inject_1q_pauli_exact_(out, q, *qn2);
          else
            inject_1q_pauli_(out, q, *qn2, dist, rng);
        }
      } else if (affected.size() == 1) {
        const auto *qn1 = nm.get_1q_gate_noise(static_cast<int>(q));
        if (qn1) {
          if (exact_channels)
            inject_1q_pauli_exact_(out, q, *qn1);
          else
            inject_1q_pauli_(out, q, *qn1, dist, rng);
        }
      }
    }

    // ── 7. Additional exact CPTP channels ──
    if (exact_channels)
      inject_additional_exact_channels_(out, nm, affected);

    // ── 8. Two-qubit depolarizing (correlated 2Q Pauli channel) ──
    if (is_2q) {
      auto q1 = affected[0];
      auto q2 = affected[1];
      double p2q = nm.get_2q_depolarizing(
          static_cast<int>(q1), static_cast<int>(q2));
      if (exact_channels) {
        if (p2q != 0.0)
          out->AddOperation(
              std::make_shared<Circuits::QuantumChannelOperation<>>(
                  Types::qubits_vector{q1, q2},
                  Simulators::QuantumChannel::TwoQubitDepolarizing(p2q)));
      } else if (p2q > 0) {
        inject_2q_depol_(out, q1, q2, p2q, dist, rng);
      }
    }
  }
  return out;
}

inline std::shared_ptr<Circuits::Circuit<double>> inject_combined_noise(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  return inject_combined_noise_impl_(circ, nm, rng, false);
}

/**
 * Insert exact CPTP operations for Pauli, T1, phase/thermal damping, custom
 * Kraus and two-qubit correlated layers, while retaining sampled unitary
 * trajectories for coherent and temporally correlated layers.
 */
inline std::shared_ptr<Circuits::Circuit<double>>
inject_combined_noise_exact(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm, std::mt19937 &rng) {
  return inject_combined_noise_impl_(circ, nm, rng, true);
}

}  // namespace noise
