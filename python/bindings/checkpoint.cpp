#include "core.h"
#include "module.h"
#include "runtime.h"

namespace maestro_bindings
{

// Checkpointed simulator for fast prefix state reuse across shots
class PrefixCheckpointedSimulator
{
  public:
    PrefixCheckpointedSimulator(std::shared_ptr<Circuits::Circuit<double>> prefix_circuit, int num_qubits, const SimulatorConfig &config = SimulatorConfig{})
        : num_qubits_(std::max(1, num_qubits)), config_(config)
    {
        if (prefix_circuit && num_qubits <= 0)
        {
            num_qubits_ = std::max(1, static_cast<int>(prefix_circuit->GetMaxQubitIndex()) + 1);
        }
        sim_ = std::make_unique<ScopedSimulator>(num_qubits_);
        if (sim_->handle == 0)
        {
            throw std::runtime_error("PrefixCheckpointedSimulator: failed to create simulator handle.");
        }
        network_ = ConfigureNetwork(sim_->handle, config_);
        if (!network_)
        {
            throw std::runtime_error("PrefixCheckpointedSimulator: failed to configure network.");
        }
        network_->CreateSimulator(config_.simulator_type, config_.simulation_type);
        simulator_ = network_->GetSimulator();
        if (!simulator_)
        {
            throw std::runtime_error("PrefixCheckpointedSimulator: requested simulator/simulation type is "
                                     "not available.");
        }
        network_->GetController()->SetOptimizeCircuit(false);
        network_->SetInitialQubitsMapOptimization(false);
        network_->SetMPSOptimizeSwaps(false);

        if (config_.seed)
        {
            simulator_->SetSeed(*config_.seed);
        }

        Circuits::OperationState opState(num_qubits_);
        if (prefix_circuit && !prefix_circuit->GetOperations().empty())
        {
            prefix_circuit->ExecuteBD(simulator_, opState, &max_bond_dim_);
        }
        simulator_->SaveState();
    }

    nb::dict execute_suffix(std::shared_ptr<Circuits::Circuit<double>> suffix_circuit, int shots = 1024, const noise::NoiseModel *noise_model = nullptr,
                            int noise_realizations = 64, std::optional<uint64_t> noise_seed = std::nullopt, int num_measurements = 0)
    {
        if (!suffix_circuit)
            throw nb::value_error("suffix_circuit is null.");
        if (shots <= 0)
            shots = 1;

        size_t total_cbits = std::max((size_t)num_qubits_, (size_t)num_measurements);
        const auto cbits_set = suffix_circuit->GetBits();
        if (!cbits_set.empty())
        {
            total_cbits = std::max(total_cbits, *cbits_set.rbegin() + 1);
        }

        const bool has_noise = (noise_model != nullptr) && noise_model->has_any();
        if (has_noise)
            warn_thermal_approximation(*noise_model, config_);

        RequireNoiseSeed(noise_seed);
        std::mt19937 rng(static_cast<uint32_t>(noise_seed.value_or(config_.seed.value_or(std::random_device{}()))));

        if (noise_seed)
        {
            simulator_->SetSeed(*noise_seed);
        }

        std::unordered_map<std::string, size_t> combined;
        auto start = std::chrono::high_resolution_clock::now();

        {
            nb::gil_scoped_release release;

            if (!has_noise)
            {
                Circuits::OperationState opState(total_cbits);
                for (int s = 0; s < shots; ++s)
                {
                    opState.Reset();
                    simulator_->RestoreState();
                    simulator_->SetGatesCounter(0);

                    suffix_circuit->ExecuteBD(simulator_, opState, &max_bond_dim_);

                    const auto &bits = opState.GetAllBits();
                    std::string bitstring(total_cbits, '0');
                    for (size_t i = 0; i < std::min(bits.size(), total_cbits); ++i)
                    {
                        if (bits[i])
                            bitstring[i] = '1';
                    }
                    ++combined[bitstring];
                }
            }
            else
            {
                const int batches = std::min(shots, std::max(1, noise_realizations));
                const int base_batch = shots / batches;
                int leftover = shots % batches;

                Circuits::OperationState opState(total_cbits);
                for (int b = 0; b < batches; ++b)
                {
                    int batch_shots = base_batch + (b < leftover ? 1 : 0);
                    if (batch_shots <= 0)
                        continue;

                    auto noisy_suffix = inject_combined_noise_for_config(suffix_circuit, *noise_model, rng, config_);

                    for (int s = 0; s < batch_shots; ++s)
                    {
                        opState.Reset();
                        simulator_->RestoreState();
                        simulator_->SetGatesCounter(0);

                        noisy_suffix->ExecuteBD(simulator_, opState, &max_bond_dim_);

                        const auto &bits = opState.GetAllBits();
                        std::string bitstring(total_cbits, '0');
                        for (size_t i = 0; i < std::min(bits.size(), total_cbits); ++i)
                        {
                            if (bits[i])
                                bitstring[i] = '1';
                        }
                        ++combined[bitstring];
                    }
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        nb::dict py_counts;
        for (const auto &[k, v] : combined)
        {
            py_counts[k.c_str()] = v;
        }

        nb::dict out;
        out["counts"] = py_counts;
        out["time_taken"] = std::chrono::duration<double>(end - start).count();
        out["simulator"] = (int)config_.simulator_type;
        out["method"] = (int)config_.simulation_type;
        if (max_bond_dim_ > 0)
        {
            out["max_bond_dim_reached"] = max_bond_dim_;
        }
        return out;
    }

    size_t max_bond_dim() const
    {
        return max_bond_dim_;
    }

  private:
    std::unique_ptr<ScopedSimulator> sim_;
    int num_qubits_;
    SimulatorConfig config_;
    std::shared_ptr<Network::INetwork<double>> network_;
    std::shared_ptr<Simulators::ISimulator> simulator_;
    size_t max_bond_dim_ = 0;
};

void bind_checkpoint(nb::module_ &m)
{
    nb::class_<PrefixCheckpointedSimulator>(m, "PrefixCheckpointedSimulator")
        .def(nb::init<std::shared_ptr<Circuits::Circuit<double>>, int, const SimulatorConfig &>(), nb::arg("prefix_circuit"), nb::arg("num_qubits"),
             nb::arg("config") = SimulatorConfig{}, "Create a simulator checkpointed after executing prefix_circuit.")
        .def("execute_suffix", &PrefixCheckpointedSimulator::execute_suffix, nb::arg("suffix_circuit"), nb::arg("shots") = 1024,
             nb::arg("noise_model").none() = nb::none(), nb::arg("noise_realizations") = 64, nb::arg("noise_seed") = nb::none(),
             nb::arg("num_measurements") = 0, "Execute suffix circuit from checkpointed prefix state.")
        .def_prop_ro("max_bond_dim", &PrefixCheckpointedSimulator::max_bond_dim);
}

} // namespace maestro_bindings
