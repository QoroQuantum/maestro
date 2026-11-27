/**
 * @file PrefixCheckpoint.h
 * @ingroup execution
 * @brief Simulator checkpointed after a prefix circuit for repeated suffix sampling.
 */
#pragma once

#include "SampleRecords.h"

#include <memory>
#include <optional>

namespace MaestroExecution
{
/// @ingroup execution
class PrefixCheckpoint
{
  public:
    // Executes prefix (which may be null) once on a register of num_qubits
    // qubits, or of the prefix's width when num_qubits <= 0, and saves the
    // state. The logical register is fixed. MPS/MPO can place the fresh
    // prefix register and route each suffix from its restored physical map
    // according to the configured swap optimization.
    // The prefix runs without its end measurements (ExecutedCheckpointPrefix):
    // only suffixes are sampled, and those measurements would collapse the
    // saved state once for every suffix shot. A measurement that later
    // operations of the prefix follow or read runs and collapses once. Every
    // suffix shot starts with that quantum state and its classical results.
    // prefix_noise, when it has noise, is injected into the prefix as exact
    // channels: it evolves only once, so it must be deterministic there (a
    // density-matrix/MPO backend, or the QCSim Pauli propagator for noise it
    // applies exactly by default, without coherent or correlated layers);
    // otherwise std::invalid_argument is thrown.
    MAESTRO_TRANSFORM_API PrefixCheckpoint(const RecordCircuitPtr &prefix, int num_qubits, const SimulatorConfig &config = SimulatorConfig{},
                                           const noise::NoiseModel *prefix_noise = nullptr);
    MAESTRO_TRANSFORM_API ~PrefixCheckpoint();
    PrefixCheckpoint(const PrefixCheckpoint &) = delete;
    PrefixCheckpoint &operator=(const PrefixCheckpoint &) = delete;

    // Sample suffix from the saved state. Terminal measurements on mixed-state
    // backends, the QCSim Pauli propagator and QCSim statevector/stabilizer/MPS
    // backends reuse one prepared suffix state per realization. QCSim path
    // integral and extended stabilizer also preserve the saved state for
    // counts and rows. Pure-state suffixes with resets replay
    // per shot, as the Pauli propagator does under the legacy counts policy
    // (PreservesSavedStateWhileSampling).
    // Records span
    // max(qubits, num_measurements, the suffix's highest classical bit + 1)
    // bits; replayed rows keep shot order, prepared-state rows are shuffled
    // within their realization. When the model or the suffix's own channels
    // add noise, shots are split across min(shots, noise_realizations)
    // combined-noise realizations, run on realization workers. An absent
    // realization count selects one for deterministic exact channels, or the
    // usual trajectory default otherwise. A call's
    // noise_seed (32 bits) seeds injection, measurement and readout. A seeded
    // checkpoint called without one derives a new seed per call, so a sequence
    // of calls replays without repeating a batch. On the QCSim extended
    // stabilizer, after a gate-only prefix, noisy suffixes that frames support
    // run on batched Pauli frames (extended_frame_v1) when PreferExtendedFrames
    // admits the call: counts_policy auto or frame_v1, with or without a seed.
    // Not safe for concurrent calls.
    MAESTRO_TRANSFORM_API ShotRecords ExecuteSuffix(const RecordCircuitPtr &suffix, size_t shots, const noise::NoiseModel *noise_model,
                                                    std::optional<size_t> noise_realizations, std::optional<uint64_t> noise_seed, size_t num_measurements,
                                                    bool rows);

    // Whether ExecuteSuffix injects noise for this suffix and model.
    MAESTRO_TRANSFORM_API static bool HasNoise(const RecordCircuitPtr &suffix, const noise::NoiseModel *noise_model);

    // The largest bond dimension reached so far, by the prefix or any suffix.
    size_t MaxBondDimension() const noexcept
    {
        return max_bond_dim_;
    }

    size_t Qubits() const noexcept
    {
        return static_cast<size_t>(num_qubits_);
    }

    // The configuration every suffix runs with: the requested one, with a
    // candidate list resolved to the backend that holds the register.
    const SimulatorConfig &Config() const noexcept
    {
        return config_;
    }

  private:
    size_t RealizationWorkers(size_t batches, size_t copies_per_worker) const;

    unsigned long handle_ = 0;
    int num_qubits_;
    SimulatorConfig config_;
    std::shared_ptr<Network::INetwork<double>> network_;
    std::shared_ptr<Simulators::ISimulator> simulator_;
    Circuits::OperationState prefix_state_;
    size_t max_bond_dim_ = 0;
    uint64_t calls_ = 0; // ExecuteSuffix calls, for per-call seeds
    // The ideal gate-only prefix of an extended_stabilizer checkpoint, which
    // batched frames evolve once per call for noisy suffixes.
    RecordCircuitPtr frame_prefix_;
    bool frame_prefix_usable_ = false;
};
} // namespace MaestroExecution
