#pragma once
#include "Options.h"
#include "Noise/NoiseModel.h"

namespace MaestroExecution {
inline const std::map<std::string, std::string>& NoiseKinds() {
  static const std::map<std::string, std::string> kinds{
      {"pauli", "px py pz"},
      {"depolarizing", "probability"},
      {"dephasing", "probability"},
      {"bit_flip", "probability"},
      {"gate_depolarizing_1q", "probability"},
      {"gate_depolarizing_2q", "probability"},
      {"pair_depolarizing", "probability"},
      {"coherent_rotation", "rx ry rz"},
      {"coherent_depolarizing", "probability"},
      {"coherent_dephasing", "probability"},
      {"coherent_bit_flip", "probability"},
      {"t1", "gamma duration t1"},
      {"t1_2q", "gamma duration t1"},
      {"phase_damping", "gamma duration t_phi"},
      {"generalized_amplitude_damping", "gamma excited_population"},
      {"thermal_relaxation", "duration t1 t2 excited_population"},
      {"thermal_relaxation_2q", "duration t1 t2 excited_population"},
      {"idle", "t1 t2 excited_population detuning_hz"},
      {"crosstalk", "strength"},
      {"readout", "p_meas1_prep0 p_meas0_prep1 probability"},
      {"correlated_phase_flip", "probability correlation"},
      {"kraus", "operators"},
      {"correlated_ar1", "phi sigma_eta after_1q after_2q stationary_init"},
      {"correlated_ou",
       "sigma alpha gate_time after_1q after_2q stationary_init"},
      {"correlated_ou_band",
       "sigma alpha gate_time after_1q after_2q stationary_init"},
      {"multi_correlated_ou",
       "bands gate_time after_1q after_2q stationary_init"},
      {"one_over_f",
       "total_power f_min f_max num_bands gate_time after_1q after_2q "
       "stationary_init"},
  };
  return kinds;
}
inline double TimeConstant(const json::object& object, const char* name) {
  const auto& value = Field(object, name);
  if (value.is_string() && String(value) == "infinity")
    return std::numeric_limits<double>::infinity();
  const auto result = Number(value);
  Require(result > 0, std::string(name) + " must be positive");
  return result;
}
struct NoiseConfig {
  noise::NoiseModel model;
  bool enabled = false;
  bool exact = false;
  std::string mode = "combined";
  uint32_t seed = 0;
  size_t realizations = 1;
  json::array approximations;
  std::string thermal_approximation_warning;
};
inline NoiseConfig ParseNoise(const json::object& object,
                              const SimulatorConfig& config, size_t qubits) {
  NoiseConfig result;
  if (object.empty()) return result;
  Keys(object, {"mode", "evaluation", "realizations", "seed", "channels"});
  result.enabled = true;
  result.mode = String(object, "mode", "combined");
  Require(result.mode == "combined" || result.mode == "pauli" ||
              result.mode == "coherent" || result.mode == "analytical",
          "Unknown noise mode");
  const auto evaluation = String(object, "evaluation", "auto");
  Require(evaluation == "auto" || evaluation == "exact" ||
              evaluation == "trajectories",
          "Unknown noise evaluation");
  result.exact =
      evaluation == "exact" || (evaluation == "auto" && Mixed(config));
  Supported(!result.exact || Mixed(config),
            "Exact noise requires density_matrix or matrix_product_operator");
  result.realizations = UInt(object, "realizations", result.exact ? 1 : 64);
  Require(result.realizations > 0 && result.realizations <= 1000000,
          "Invalid noise realization count");
  const auto seed = UInt(object, "seed", config.seed.value_or(0) & UINT32_MAX);
  Require(seed <= UINT32_MAX, "noise.seed must fit in 32 bits");
  result.seed = static_cast<uint32_t>(seed);
  std::set<std::pair<std::string, uint64_t>> configured;
  std::set<std::pair<std::string, Types::qubits_vector>> configuredPairs;
  const auto& channels = Array(Field(object, "channels"));
  Require(!channels.empty() && channels.size() <= 100000,
          "Invalid channel count");
  for (const auto& item : channels) {
    const auto& channel = Object(item);
    const std::string kind = String(Field(channel, "kind"));
    const auto spec = NoiseKinds().find(kind);
    Require(spec != NoiseKinds().end(), "Unknown noise channel: " + kind);
    std::set<std::string> allowed{"kind", "targets", "placement"};
    std::istringstream names(spec->second);
    for (std::string name; names >> name;) allowed.insert(name);
    for (const auto& entry : channel)
      Require(allowed.count(std::string(entry.key())),
              "Unknown field for " + kind + ": " + std::string(entry.key()));
    Require(
        String(channel, "placement", "after_each_gate") == "after_each_gate",
        "Noise placement is defined by channel kind; only after_each_gate is "
        "accepted");
    Types::qubits_vector targets;
    std::set<uint64_t> unique;
    for (const auto& target : Array(Field(channel, "targets"))) {
      auto q = UInt(target);
      Require(q < qubits && unique.insert(q).second,
              "Noise target is invalid or duplicated");
      targets.push_back(q);
      if (kind != "correlated_ou_band" && kind != "kraus" &&
          kind != "pair_depolarizing" && kind != "correlated_phase_flip" &&
          kind != "crosstalk")
        Require(configured.emplace(kind, q).second,
                "Repeated channel would replace an existing model setting");
    }
    Require(!targets.empty(), "Noise targets must not be empty");
    const auto number = [&](const char* key) {
      return Number(Field(channel, key));
    };
    const auto p = [&]() { return number("probability"); };
    const bool pair = kind == "pair_depolarizing" ||
                      kind == "correlated_phase_flip" || kind == "crosstalk";
    Require(!pair || targets.size() == 2,
            "Pair channel requires exactly two targets");
    if (pair || kind == "kraus")
      Require(configuredPairs.emplace(kind, targets).second,
              "Repeated channel target tuple");
    auto& model = result.model;
    if (kind == "kraus") {
      Require(targets.size() <= 2, "Kraus channels support one or two targets");
      const auto dim = size_t{1} << targets.size();
      Simulators::QuantumChannel::KrausOperators operators;
      for (const auto& matrix : Array(Field(channel, "operators"))) {
        const auto& entries = Array(matrix);
        Require(entries.size() == dim * dim,
                "Kraus matrix must be a flat row-major array");
        Eigen::MatrixXcd op(dim, dim);
        for (size_t row = 0; row < dim; ++row)
          for (size_t col = 0; col < dim; ++col)
            op(row, col) = Complex(entries[row * dim + col]);
        operators.push_back(std::move(op));
      }
      model.set_kraus_channel(targets, operators);
    } else if (kind == "pair_depolarizing")
      model.set_2q_depolarizing(targets[0], targets[1], p());
    else if (kind == "correlated_phase_flip")
      model.set_correlated_phase_flip(targets[0], targets[1], p(),
                                      Number(channel, "correlation", 1));
    else if (kind == "crosstalk")
      model.set_crosstalk(targets[0], targets[1], number("strength"));
    else
      for (const auto q : targets) {
        if (kind == "pauli")
          model.set_qubit_noise(q, number("px"), number("py"), number("pz"));
        else if (kind == "depolarizing")
          model.set_depolarizing(q, p());
        else if (kind == "dephasing")
          model.set_dephasing(q, p());
        else if (kind == "bit_flip")
          model.set_bit_flip(q, p());
        else if (kind == "gate_depolarizing_1q")
          model.set_1q_gate_depolarizing(q, p());
        else if (kind == "gate_depolarizing_2q")
          model.set_2q_gate_depolarizing(q, p());
        else if (kind == "coherent_rotation")
          model.set_coherent_rotation(q, number("rx"), number("ry"),
                                      number("rz"));
        else if (kind == "coherent_depolarizing")
          model.set_coherent_depolarizing(q, p());
        else if (kind == "coherent_dephasing")
          model.set_coherent_dephasing(q, p());
        else if (kind == "coherent_bit_flip")
          model.set_coherent_bit_flip(q, p());
        else if (kind == "t1" || kind == "t1_2q" || kind == "phase_damping") {
          const bool gamma = channel.contains("gamma");
          Require(gamma != channel.contains("duration"),
                  "Specify gamma or duration/time constant");
          if (gamma) {
            Require(!channel.contains("t1") && !channel.contains("t_phi"),
                    "Time constant is unused with gamma");
            if (kind == "t1")
              model.set_t1(q, number("gamma"));
            else if (kind == "t1_2q")
              model.set_t1_2q(q, number("gamma"));
            else
              model.set_phase_damping(q, number("gamma"));
          } else {
            if (kind == "t1")
              model.set_t1_from_time(q, number("duration"),
                                     TimeConstant(channel, "t1"));
            else if (kind == "t1_2q")
              model.set_t1_2q_from_time(q, number("duration"),
                                        TimeConstant(channel, "t1"));
            else
              model.set_phase_damping_from_time(q, number("duration"),
                                                TimeConstant(channel, "t_phi"));
          }
        } else if (kind == "generalized_amplitude_damping")
          model.set_generalized_amplitude_damping(q, number("gamma"),
                                                  number("excited_population"));
        else if (kind == "thermal_relaxation")
          model.set_thermal_relaxation(
              q, number("duration"), TimeConstant(channel, "t1"),
              TimeConstant(channel, "t2"),
              Number(channel, "excited_population", 0));
        else if (kind == "thermal_relaxation_2q")
          model.set_thermal_relaxation_2q(
              q, number("duration"), TimeConstant(channel, "t1"),
              TimeConstant(channel, "t2"),
              Number(channel, "excited_population", 0));
        else if (kind == "idle")
          model.set_idle_noise(q, TimeConstant(channel, "t1"),
                               TimeConstant(channel, "t2"),
                               Number(channel, "excited_population", 0),
                               Number(channel, "detuning_hz", 0));
        else if (kind == "readout") {
          if (channel.contains("probability")) {
            Require(!channel.contains("p_meas1_prep0") &&
                        !channel.contains("p_meas0_prep1"),
                    "Conflicting readout parameters");
            model.set_readout_error_symmetric(q, p());
          } else
            model.set_readout_error(q, number("p_meas1_prep0"),
                                    number("p_meas0_prep1"));
        } else {
          const bool after1 = Boolean(channel, "after_1q", true);
          const bool after2 = Boolean(channel, "after_2q", true);
          const bool stationary = Boolean(channel, "stationary_init", true);
          if (kind == "correlated_ar1") {
            const auto phi = number("phi"), sigma = number("sigma_eta");
            Require(std::abs(phi) < 1 && sigma >= 0,
                    "AR(1) requires abs(phi)<1 and sigma_eta>=0");
            model.set_correlated_ar1(q, phi, sigma, after1, after2, stationary);
          } else if (kind == "correlated_ou")
            model.set_correlated_ou(q, number("sigma"), number("alpha"),
                                    number("gate_time"), after1, after2,
                                    stationary);
          else if (kind == "correlated_ou_band")
            model.add_correlated_ou_band(q, number("sigma"), number("alpha"),
                                         number("gate_time"), after1, after2,
                                         stationary);
          else if (kind == "multi_correlated_ou") {
            std::vector<std::pair<double, double>> bands;
            for (const auto& band : Array(Field(channel, "bands"))) {
              const auto& values = Array(band);
              Require(values.size() == 2, "An OU band is [sigma, alpha]");
              bands.emplace_back(Number(values[0]), Number(values[1]));
            }
            Require(!bands.empty(), "OU bands must not be empty");
            model.set_multi_correlated_ou(q, bands, number("gate_time"), after1,
                                          after2, stationary);
          } else if (kind == "one_over_f") {
            const auto bands = UInt(Field(channel, "num_bands"));
            Require(bands > 0 && bands <= 10000, "Invalid OU band count");
            model.set_1_over_f_noise(
                q, number("total_power"), number("f_min"), number("f_max"),
                bands, number("gate_time"), after1, after2, stationary);
          }
        }
      }
    if (result.mode == "pauli" || result.mode == "analytical")
      Supported(
          kind == "pauli" || kind == "depolarizing" || kind == "dephasing" ||
              kind == "bit_flip" ||
              (kind == "readout" && result.mode == "pauli"),
          "This noise mode does not model the selected channel; use combined");
    if (result.mode == "coherent")
      Supported(kind.find("coherent_") == 0,
                "coherent mode accepts only coherent channels");
  }
  result.model.EnsureNoT1ThermalStack();
  Supported(result.exact || !result.model.has_additional_quantum_channels(),
            "This noise model requires exact density-matrix/MPO evolution");
  // Match Python's sampled thermal policy without altering calibrated channels
  // used by exact density-matrix/MPO evolution.
  if (!result.exact) {
    const auto qubits = result.model.thermal_approximation_qubits();
    if (!qubits.empty()) {
      result.approximations.emplace_back("thermal_T2_clamped_to_T1");
      std::ostringstream message;
      message << "Sampled thermal approximation for circuit qubits [";
      for (size_t i = 0; i < qubits.size(); ++i) {
        if (i) message << ", ";
        message << qubits[i];
      }
      message
          << "]: effective T2 clamped to T1. Use density matrix or a supported "
             "MPO backend to preserve calibrated T2. Kraus trajectories are "
             "not supported by Maestro's current SV/MPS noise path.";
      result.thermal_approximation_warning = message.str();
    }
  }
  if (!result.exact && result.model.has_t1())
    result.approximations.emplace_back("sampled_T1_reset");
  if (!result.exact && result.mode != "analytical")
    result.approximations.emplace_back("finite_noise_realizations");
  if (result.mode == "analytical")
    result.approximations.emplace_back("terminal_Pauli_damping");
  if (config.simulation_type == Method::kMatrixProductOperator)
    result.approximations.emplace_back("MPO_truncation");
  return result;
}
}  // namespace MaestroExecution
