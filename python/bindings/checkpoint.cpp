#include "Execution/PrefixCheckpoint.h"
#include "core.h"
#include "module.h"
#include "samples.h"

namespace maestro_bindings
{

// Checkpointed simulator for fast prefix state reuse across shots; the
// algorithm is MaestroExecution::PrefixCheckpoint, shared with C++ callers.
class PrefixCheckpointedSimulator
{
  public:
    PrefixCheckpointedSimulator(std::shared_ptr<Circuits::Circuit<double>> prefix_circuit, int num_qubits, const SimulatorConfig &config = SimulatorConfig{},
                                const noise::NoiseModel *prefix_noise_model = nullptr)
        : checkpoint_(std::make_unique<MaestroExecution::PrefixCheckpoint>(prefix_circuit, num_qubits, config, prefix_noise_model))
    {
    }

    nb::dict execute_suffix(std::shared_ptr<Circuits::Circuit<double>> suffix_circuit, int shots = 1024, const noise::NoiseModel *noise_model = nullptr,
                            std::optional<int> noise_realizations = std::nullopt, std::optional<uint64_t> noise_seed = std::nullopt, int num_measurements = 0,
                            bool return_samples = false)
    {
        if (!suffix_circuit)
            throw nb::value_error("suffix_circuit is null.");
        if (shots < 1)
            throw nb::value_error("shots must be >= 1.");
        if (num_measurements < 0)
            throw nb::value_error("num_measurements must be nonnegative.");
        require_realizations(noise_realizations);
        const auto requested = noise_realizations ? std::optional<size_t>(static_cast<size_t>(*noise_realizations)) : std::nullopt;
        // Before the run: a warning raised as an error must leave the checkpoint
        // as it was, and a suffix call advances its per-call seed. The checkpoint
        // resolved a candidate list to one backend when it was built, so its
        // configuration decides as the suffix will.
        static const noise::NoiseModel no_noise;
        if (MaestroExecution::PrefixCheckpoint::HasNoise(suffix_circuit, noise_model))
            warn_thermal_approximation(noise_model ? *noise_model : no_noise, checkpoint_->Config(), suffix_circuit);

        MaestroExecution::ShotRecords records;
        {
            nb::gil_scoped_release release;
            records = checkpoint_->ExecuteSuffix(suffix_circuit, static_cast<size_t>(shots), noise_model, requested, noise_seed,
                                                 static_cast<size_t>(num_measurements), return_samples);
        }
        nb::dict out;
        if (return_samples)
            out["samples"] = MakeSamples(std::move(records.rows), records.shots, records.width);
        else
            out["counts"] = CountsDict(records.counts);
        out["time_taken"] = records.time_taken;
        out["simulator"] = records.simulator;
        out["method"] = records.method;
        out["noise_realizations"] = records.realizations;
        AddNoisePolicyMetadata(out, records.policy);
        if (!records.sampling_strategy.empty())
            out["sampling_strategy"] = records.sampling_strategy;
        if (records.max_bond_dimension > 0)
            out["max_bond_dim_reached"] = records.max_bond_dimension;
        return out;
    }

    size_t max_bond_dim() const
    {
        return checkpoint_->MaxBondDimension();
    }

  private:
    std::unique_ptr<MaestroExecution::PrefixCheckpoint> checkpoint_;
};

void bind_checkpoint(nb::module_ &m)
{
    nb::class_<PrefixCheckpointedSimulator>(m, "PrefixCheckpointedSimulator",
                                            "Executes a prefix circuit once and saves its state, so that execute_suffix can sample many "
                                            "suffix circuits from it without re-running the prefix. The config is fixed at construction.")
        .def(nb::init<std::shared_ptr<Circuits::Circuit<double>>, int, const SimulatorConfig &, const noise::NoiseModel *>(), nb::arg("prefix_circuit"),
             nb::arg("num_qubits"), nb::arg("config") = SimulatorConfig{}, nb::arg("prefix_noise_model").none() = nb::none(),
             "Execute prefix_circuit on num_qubits qubits (zero or less: the prefix's width) and save the state. "
             "config.seed seeds the prefix and, absent a per-call seed, the suffix calls. prefix_noise_model, if given, "
             "is applied to the prefix once as exact channels; it must be deterministic there (a density-matrix or MPO "
             "backend, or the QCSim Pauli propagator for noise it applies exactly by default, without coherent or "
             "correlated layers), otherwise ValueError is raised. The Pauli propagator samples relaxation that acts on "
             "the circuit (T1, thermal or idle noise towards unequal populations), so it refuses such a prefix.")
        .def("execute_suffix", &PrefixCheckpointedSimulator::execute_suffix, nb::arg("suffix_circuit"), nb::arg("shots") = 1024,
             nb::arg("noise_model").none() = nb::none(), nb::arg("noise_realizations") = nb::none(), nb::arg("noise_seed") = nb::none(),
             nb::arg("num_measurements") = 0, nb::arg("return_samples") = false,
             "Sample suffix_circuit from the saved prefix state. Mixed-state backends, the QCSim Pauli propagator and the QCSim statevector, stabilizer "
             "and MPS backends sample terminal measurements from one prepared suffix state per realization; other executions restore the prefix and "
             "replay the suffix per shot, as the Pauli propagator does under counts_policy='legacy'.\n\n"
             "noise_model, and channel instructions in the suffix, are injected afresh for each of "
             "noise_realizations realizations, which share the shots between them. None selects one realization for "
             "deterministic exact-channel noise (density-matrix and MPO backends, and the QCSim Pauli propagator for "
             "noise it applies exactly by default), or 64 otherwise. "
             "An explicit positive count is honored. noise_seed (alias seed) is a "
             "32-bit seed for this call's noise and measurements. Without one, a checkpoint whose config has a "
             "seed derives a new seed per call, so a sequence of calls replays but no two calls repeat. "
             "num_measurements widens the classical register to at least that many bits.\n\n"
             "Returns a dict with 'counts' or, with return_samples=True, 'samples' (a uint8 array of shape "
             "(shots, classical width): replayed shots keep shot order, terminal_v1 rows are shuffled within each realization), plus 'time_taken', "
             "'simulator', 'method', 'noise_realizations', "
             "'sampling_strategy' and, on MPS "
             "and MPO, 'max_bond_dim_reached'.")
        .def_prop_ro("max_bond_dim", &PrefixCheckpointedSimulator::max_bond_dim,
                     "Largest MPS/MPO bond dimension reached so far, by the prefix and every suffix; 0 on other methods.");
}

} // namespace maestro_bindings
