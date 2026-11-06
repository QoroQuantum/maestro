// A simulator checkpointed after a prefix circuit, from which suffixes are
// sampled repeatedly. Shared by C++ callers and the Python bindings
// (PrefixCheckpointedSimulator). This header has no Python dependency.
#pragma once

#include "SampleRecords.h"

#include <memory>
#include <optional>

namespace MaestroExecution
{
class PrefixCheckpoint
{
  public:
    // Executes prefix (which may be null) once on a register of num_qubits
    // qubits, or of the prefix's width when num_qubits <= 0, and saves the
    // state. The register is fixed: circuit optimization, initial MPS mapping
    // and swap optimization are off, so suffix qubits keep their indices.
    MAESTRO_TRANSFORM_API PrefixCheckpoint(const RecordCircuitPtr &prefix, int num_qubits, const SimulatorConfig &config = SimulatorConfig{});
    MAESTRO_TRANSFORM_API ~PrefixCheckpoint();
    PrefixCheckpoint(const PrefixCheckpoint &) = delete;
    PrefixCheckpoint &operator=(const PrefixCheckpoint &) = delete;

    // shots executions of suffix from the saved state. Records span
    // max(qubits, num_measurements, the suffix's highest classical bit + 1)
    // bits; rows keep shot order. When the model or the suffix's own channels
    // add noise, shots are split across min(shots, noise_realizations)
    // combined-noise realizations, run on realization workers. A call's
    // noise_seed (32 bits) seeds injection, measurement and readout. A seeded
    // checkpoint called without one derives a new seed per call, so a sequence
    // of calls replays without repeating a batch. Not safe for concurrent calls.
    MAESTRO_TRANSFORM_API ShotRecords ExecuteSuffix(const RecordCircuitPtr &suffix, size_t shots, const noise::NoiseModel *noise_model,
                                                    size_t noise_realizations, std::optional<uint64_t> noise_seed, size_t num_measurements, bool rows);

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

  private:
    size_t RealizationWorkers(size_t batches) const;

    unsigned long handle_ = 0;
    int num_qubits_;
    SimulatorConfig config_;
    std::shared_ptr<Network::INetwork<double>> network_;
    std::shared_ptr<Simulators::ISimulator> simulator_;
    size_t max_bond_dim_ = 0;
    uint64_t calls_ = 0; // ExecuteSuffix calls, for per-call seeds
};
} // namespace MaestroExecution
