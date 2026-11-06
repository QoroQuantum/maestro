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
    PrefixCheckpointedSimulator(std::shared_ptr<Circuits::Circuit<double>> prefix_circuit, int num_qubits, const SimulatorConfig &config = SimulatorConfig{})
        : config_(config), checkpoint_(std::make_unique<MaestroExecution::PrefixCheckpoint>(prefix_circuit, num_qubits, config))
    {
    }

    nb::dict execute_suffix(std::shared_ptr<Circuits::Circuit<double>> suffix_circuit, int shots = 1024, const noise::NoiseModel *noise_model = nullptr,
                            int noise_realizations = 64, std::optional<uint64_t> noise_seed = std::nullopt, int num_measurements = 0,
                            bool return_samples = false)
    {
        if (!suffix_circuit)
            throw nb::value_error("suffix_circuit is null.");
        if (shots < 1)
            throw nb::value_error("shots must be >= 1.");
        if (num_measurements < 0)
            throw nb::value_error("num_measurements must be nonnegative.");
        static const noise::NoiseModel no_noise;
        if (MaestroExecution::PrefixCheckpoint::HasNoise(suffix_circuit, noise_model))
            warn_thermal_approximation(noise_model ? *noise_model : no_noise, config_);

        MaestroExecution::ShotRecords records;
        {
            nb::gil_scoped_release release;
            records = checkpoint_->ExecuteSuffix(suffix_circuit, static_cast<size_t>(shots), noise_model, static_cast<size_t>(std::max(1, noise_realizations)),
                                                 noise_seed, static_cast<size_t>(num_measurements), return_samples);
        }
        nb::dict out;
        if (return_samples)
            out["samples"] = MakeSamples(std::move(records.rows), records.shots, records.width);
        else
            out["counts"] = CountsDict(records.counts);
        out["time_taken"] = records.time_taken;
        out["simulator"] = records.simulator;
        out["method"] = records.method;
        if (records.max_bond_dimension > 0)
            out["max_bond_dim_reached"] = records.max_bond_dimension;
        return out;
    }

    size_t max_bond_dim() const
    {
        return checkpoint_->MaxBondDimension();
    }

  private:
    SimulatorConfig config_;
    std::unique_ptr<MaestroExecution::PrefixCheckpoint> checkpoint_;
};

void bind_checkpoint(nb::module_ &m)
{
    nb::class_<PrefixCheckpointedSimulator>(m, "PrefixCheckpointedSimulator",
                                            "Executes a prefix circuit once and saves its state, so that execute_suffix can sample many "
                                            "suffix circuits from it without re-running the prefix. The config is fixed at construction.")
        .def(nb::init<std::shared_ptr<Circuits::Circuit<double>>, int, const SimulatorConfig &>(), nb::arg("prefix_circuit"), nb::arg("num_qubits"),
             nb::arg("config") = SimulatorConfig{},
             "Execute prefix_circuit on num_qubits qubits (zero or less: the prefix's width) and save the state. "
             "config.seed seeds the prefix and, absent a per-call seed, the suffix calls.")
        .def("execute_suffix", &PrefixCheckpointedSimulator::execute_suffix, nb::arg("suffix_circuit"), nb::arg("shots") = 1024,
             nb::arg("noise_model").none() = nb::none(), nb::arg("noise_realizations") = 64, nb::arg("noise_seed") = nb::none(),
             nb::arg("num_measurements") = 0, nb::arg("return_samples") = false,
             "Sample suffix_circuit from the saved prefix state; every shot restores that state.\n\n"
             "noise_model, and channel instructions in the suffix, are injected afresh for each of "
             "noise_realizations realizations, which share the shots between them. noise_seed (alias seed) is a "
             "32-bit seed for this call's noise and measurements. Without one, a checkpoint whose config has a "
             "seed derives a new seed per call, so a sequence of calls replays but no two calls repeat. "
             "num_measurements widens the classical register to at least that many bits.\n\n"
             "Returns a dict with 'counts' or, with return_samples=True, 'samples' (a uint8 array of shape "
             "(shots, classical width), in shot order), plus 'time_taken', 'simulator', 'method' and, on MPS "
             "and MPO, 'max_bond_dim_reached'.")
        .def_prop_ro("max_bond_dim", &PrefixCheckpointedSimulator::max_bond_dim,
                     "Largest MPS/MPO bond dimension reached so far, by the prefix and every suffix; 0 on other methods.");
}

} // namespace maestro_bindings
