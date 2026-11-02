#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Core/Factory.h"
#include "ImmediateGpuSimulator.h"

namespace Simulators::Private
{
// pauli_propagator backend. Owns exactly one native Gpu implementation.
class GpuPauliPropagatorSimulator : public ImmediateGpuSimulator
{
  public:
    GpuPauliPropagatorSimulator()
    {
        configuration.SetConfiguration("method", "pauli_propagator");
    }

    const char *MethodName() const override
    {
        return "pauli_propagator";
    }

    void Initialize() override
    {
        if (nrQubits)
        {
            auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
            const int gpuDevice = configuration.IsSet("gpu_device") ? Configuration::ParseGpuDevice(configuration.GetConfiguration("gpu_device"))
                                                                    : SimulatorsFactory::ResolveGpuDevice();
            configuration.SetConfiguration("gpu_device", std::to_string(gpuDevice));
            if (!SimulatorsFactory::GetGpuLibrary(gpuDevice))
                throw std::runtime_error("GpuState::Initialize: Unable to initialize GPU device " + std::to_string(gpuDevice));
            {
                pp = SimulatorsFactory::CreateGpuPauliPropagatorSimulatorUnique(gpuDevice);
                if (pp)
                {
                    const bool res = pp->CreateSimulator(static_cast<int>(nrQubits));
                    if (!res)
                        throw std::runtime_error("GpuState::Initialize: Failed to create "
                                                 "and initialize the Pauli propagator state.");
                    // Older plugins size their sampling buffers during allocation.
                    if (!pp->SetWillUseSampling(true))
                        throw std::runtime_error("GpuState::Initialize: Failed to enable Pauli sampling");
                    for (const auto &[key, value] : configuration.GetConfigMap())
                        if (key != "method")
                            Configure(key.c_str(), value.c_str());
                    const bool allocated = configuration.IsSet("pauli_propagator_memory_limit_bytes")
                                               ? pp->AllocateMemoryBytes(configuration.GetConfigurationAsUnsigned("pauli_propagator_memory_limit_bytes"))
                                               : pp->AllocateMemory(0.9);
                    if (!allocated)
                        throw std::runtime_error("GpuState::Initialize: Failed to allocate memory for the "
                                                 "Pauli propagator state.");
                }
                else
                    throw std::runtime_error("GpuState::Initialize: Failed to create the Pauli propagator "
                                             "state.");
            }
            if (GetGpuDevice() != gpuDevice)
                throw std::runtime_error("GpuState::Initialize: GPU plugin did not confirm the requested "
                                         "device; update the GPU library");
        }
    }

    void Reset() override
    {
        if (pp)
            RequirePauli(pp->ClearOperators());
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;
        if (std::string("gpu_device") == key)
        {
            const int device = Configuration::ParseGpuDevice(value);
            if ((pp) && device != Configuration::ParseGpuDevice(configuration.GetConfiguration(key)))
                throw std::invalid_argument("gpu_device cannot change after initialization; clear the "
                                            "simulator first");
            configuration.SetConfiguration(key, std::to_string(device));
            return;
        }
        const auto svdGroup = Configuration::GpuSvdSettingGroup(key);
        if (!svdGroup.empty())
        {
            Configuration::ParseGpuSvdFlag(value);
            configuration.SetConfiguration(key, value);
            return;
        }
        if (std::string("method") == key)
        {
            ValidateMethod(value);
            configuration.SetConfiguration(key, value);
            return;
        }
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = Configuration::ParsePrecision(key, value);
            if (pp && !pp->SetUseDoublePrecision(useDouble))
                throw std::runtime_error("Cannot configure Pauli precision; update the GPU library");
            configuration.SetConfiguration(key, value);
            return;
        }
        const std::string setting(key);
        if (setting == "pauli_propagator_memory_limit_bytes" || setting == "pauli_propagator_workers" || setting == "pauli_propagator_max_fused_gates" ||
            setting == "pauli_propagator_sampling_cache_nodes")
        {
            unsigned long long parsed = 0;
            const std::string text(value);
            const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
            if (result.ec != std::errc() || result.ptr != text.data() + text.size() || (setting != "pauli_propagator_sampling_cache_nodes" && !parsed) ||
                (setting == "pauli_propagator_workers" && parsed > 64) || (setting == "pauli_propagator_max_fused_gates" && parsed > 256) ||
                parsed > static_cast<unsigned long long>(INT64_MAX))
                throw std::invalid_argument("Invalid Pauli propagation configuration");
            if (pp)
            {
                bool applied = true;
                if (setting == "pauli_propagator_memory_limit_bytes")
                    applied = pp->AllocateMemoryBytes(parsed);
                else if (setting == "pauli_propagator_max_fused_gates")
                    applied = pp->SetMaxFusedGates(parsed);
                else if (setting == "pauli_propagator_sampling_cache_nodes")
                    applied = pp->SetSamplingCacheMaxNodes(parsed);
                if (!applied)
                    throw std::runtime_error("Cannot configure Pauli propagation; check the memory budget and update the GPU library");
            }
            configuration.SetConfiguration(key, value);
            return;
        }
        if (setting == "pauli_propagator_steps_between_trims" || setting == "pauli_propagator_num_gates_between_deduplications")
        {
            int interval = 0;
            const std::string text(value);
            const auto result = std::from_chars(text.data(), text.data() + text.size(), interval);
            if (result.ec != std::errc() || result.ptr != text.data() + text.size() || interval < 1)
                throw std::invalid_argument("Pauli intervals must be positive integers");
        }
        if (setting == "pauli_propagator_coefficient_threshold" || setting == "pauli_propagator_pauli_weight_threshold")
        {
            size_t end = 0;
            const double threshold = std::stod(value, &end);
            if (end != std::string(value).size() || !std::isfinite(threshold) || threshold < 0 ||
                (setting == "pauli_propagator_pauli_weight_threshold" && (threshold > INT32_MAX || threshold != std::floor(threshold))))
                throw std::invalid_argument("Invalid Pauli truncation threshold");
        }
        if (!configuration.WasApplied(key, value))
            configuration.SetConfiguration(key, value);
        if (std::string("seed") == key)
        {
            const uint64_t seed = std::stoull(value);
            SeedAuxiliaryRng(seed);
            nextSeedStream = 0;
            if (pp)
                pp->SetSeed(seed);
            return;
        }
        // Configuration can be replayed from another method. Preserve the
        // numeric validation, but tensor truncation has no native effect here.
        if (std::string("matrix_product_state_truncation_threshold") == key || std::string("matrix_product_operator_truncation_threshold") == key)
            std::stod(value);
        else if (std::string("matrix_product_state_max_bond_dimension") == key || std::string("matrix_product_operator_max_bond_dimension") == key)
            std::stoi(value);

        if (pp)
        {
            if (std::string("pauli_propagator_coefficient_threshold") == key)
            {
                const double coefficientThreshold = std::stod(value);
                pp->SetCoefficientTruncationCutoff(coefficientThreshold);
            }
            else if (std::string("pauli_propagator_pauli_weight_threshold") == key)
            {
                const double pauliWeightThreshold = std::stod(value);
                pp->SetWeightTruncationCutoff(pauliWeightThreshold);
            }
            else if (std::string("pauli_propagator_steps_between_trims") == key)
            {
                const int stepsBetweenTrims = std::stoi(value);
                pp->SetNumGatesBetweenTruncations(stepsBetweenTrims);
            }
            else if (std::string("pauli_propagator_num_gates_between_deduplications") == key)
            {
                const int numGatesBetweenDeduplications = std::stoi(value);
                pp->SetNumGatesBetweenDeduplications(numGatesBetweenDeduplications);
            }
        }
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = pp ? pp->GetUseDoublePrecision() : configuration.GetConfiguration("precision") == "double";
            return std::string(key) == "precision" ? (useDouble ? "double" : "single") : (useDouble ? "1" : "0");
        }
        if (std::string("method") == key)
            return MethodName();
        return configuration.GetConfiguration(key);
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (pp)
            return 0;
        if (num_qubits > static_cast<size_t>(INT_MAX) - nrQubits)
            throw std::length_error("GPU Pauli qubit count exceeds the native API range");
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        pp = nullptr;
        nrQubits = 0;
        upcomingGateIndex = 0;
        upcomingGates.clear();
    }

    size_t Measure(const Types::qubits_vector &qubits) override
    {
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        const auto qb = CheckedQubits(qubits);
        size_t res = 0;
        size_t mask = 1ULL;
        {
            NotificationGuard guard(*this);
            for (int qubit : qb)
            {
                if (pp->MeasureQubit(qubit))
                    res |= mask;
                mask <<= 1;
            }
        }
        NotifyObservers(qubits);
        return res;
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        const auto qb = CheckedQubits(qubits);
        std::vector<bool> res(qubits.size(), false);
        {
            NotificationGuard guard(*this);
            for (size_t i = 0; i < qubits.size(); ++i)
                res[i] = pp->MeasureQubit(qb[i]);
        }
        NotifyObservers(qubits);
        return res;
    }

    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        const auto qb = CheckedQubits(qubits);
        {
            NotificationGuard guard(*this);
            for (int qubit : qb)
                if (pp->MeasureQubit(qubit))
                    RequirePauli(pp->ApplyX(qubit));
        }
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        return pp->Probability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        {
            throw std::runtime_error("GpuState::Amplitude: Invalid simulation type for amplitude "
                                     "calculation.");
        }
    }

    std::vector<double> AllProbabilities() override
    {
        if (nrQubits == 0)
            return {};
        if (nrQubits >= std::numeric_limits<size_t>::digits)
            throw std::length_error("Full probability enumeration exceeds the basis-index API; query "
                                    "selected bit vectors");
        const size_t numStates = size_t{1} << nrQubits;
        std::vector<double> result(numStates);
        {
            for (Types::qubit_t i = 0; i < (Types::qubit_t)numStates; ++i)
            {
                result[i] = pp->Probability(i);
            }
        }
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (size_t i = 0; i < qubits.size(); ++i)
                result[i] = pp->Probability(qubits[i]);
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > std::numeric_limits<Types::qubit_t>::digits)
            throw std::length_error("Packed Pauli samples exceed the outcome width; use SampleCountsMany");
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        {
            NotificationGuard guard(*this);
            const auto qb = CheckedQubits(qubits);
            for (size_t shot = 0; shot < shots; ++shot)
            {
                size_t meas = 0;
                auto res = pp->SampleQubits(qb);
                for (size_t i = 0; i < qubits.size(); ++i)
                {
                    if (res[i])
                        meas |= (1ULL << i);
                }
                ++result[meas];
            }
        }
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        {
            NotificationGuard guard(*this);
            const auto qb = CheckedQubits(qubits);
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto res = pp->SampleQubits(qb);
                ++result[res];
            }
        }
        NotifyObservers(qubits);
        return result;
    }

    double ExpectationValue(const std::string &pauliString) override
    {
        double result = 0.0;
        result = pp->ExpectationValue(pauliString);
        return result;
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        const int workers =
            configuration.IsSet("pauli_propagator_workers") ? static_cast<int>(configuration.GetConfigurationAsUnsigned("pauli_propagator_workers")) : 1;
        return pp->ExpectationValues(paulis, workers);
    }

    int GetGpuDevice() const override
    {
        if (pp)
            return pp->GetGpuDevice();
        return -1;
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kPauliPropagator;
    }

    void SaveStateToInternalDestructive() override
    {
        return;
    }

    void RestoreInternalDestructiveSavedState() override
    {
        return;
    }

    void SaveState() override
    {
        pp->SaveState();
    }

    void RestoreState() override
    {
        pp->RestoreState();
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        RequirePauli(pp->ApplyP(CheckedQubit(qubit), lambda));
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyX(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyY(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyZ(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyH(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyS(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplySDG(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyT(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyTDG(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplySQRTX(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplySxDAG(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        RequirePauli(pp->ApplyK(CheckedQubit(qubit)));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        RequirePauli(pp->ApplyRX(CheckedQubit(qubit), theta));
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        RequirePauli(pp->ApplyRY(CheckedQubit(qubit), theta));
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        RequirePauli(pp->ApplyRZ(CheckedQubit(qubit), theta));
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        RequirePauli(pp->ApplyU(CheckedQubit(qubit), theta, phi, lambda, gamma));
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        RequirePauli(pp->ApplyCX(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit)));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        RequirePauli(pp->ApplyCY(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit)));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        RequirePauli(pp->ApplyCZ(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit)));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        RequirePauli(pp->ApplyCP(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit), lambda));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        RequirePauli(pp->ApplyCRX(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit), theta));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        RequirePauli(pp->ApplyCRY(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit), theta));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        RequirePauli(pp->ApplyCRZ(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit), theta));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        RequirePauli(pp->ApplyCH(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit)));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        RequirePauli(pp->ApplyCSX(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit)));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        RequirePauli(pp->ApplyCSXDAG(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit)));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        RequirePauli(pp->ApplySWAP(CheckedQubit(qubit0), CheckedQubit(qubit1)));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        RequirePauli(pp->ApplyCCX(CheckedQubit(qubit0), CheckedQubit(qubit1), CheckedQubit(qubit2)));
        NotifyObservers({qubit0, qubit1, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        RequirePauli(pp->ApplyCSwap(CheckedQubit(ctrl_qubit), CheckedQubit(qubit0), CheckedQubit(qubit1)));
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        RequirePauli(pp->ApplyCU(CheckedQubit(ctrl_qubit), CheckedQubit(tgt_qubit), theta, phi, lambda, gamma));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        throw std::runtime_error("GpuSimulator::Clone: Cloning Tensor Network or Pauli Propagator "
                                 "simulation is not "
                                 "supported.");
    }

  protected:
    class NotificationGuard
    {
      public:
        explicit NotificationGuard(GpuPauliPropagatorSimulator &state) : state(state), enabled(state.NotificationsEnabled())
        {
            state.DontNotify();
        }

        ~NotificationGuard()
        {
            if (enabled)
                state.Notify();
            else
                state.DontNotify();
        }

      private:
        GpuPauliPropagatorSimulator &state;
        bool enabled;
    };

    int CheckedQubit(Types::qubit_t qubit) const
    {
        if (qubit >= nrQubits || qubit > static_cast<Types::qubit_t>(INT_MAX))
            throw std::out_of_range("Invalid Pauli qubit");
        return static_cast<int>(qubit);
    }

    std::vector<int> CheckedQubits(const Types::qubits_vector &qubits) const
    {
        std::vector<int> result;
        result.reserve(qubits.size());
        for (auto qubit : qubits)
            result.push_back(CheckedQubit(qubit));
        return result;
    }

    static void RequirePauli(bool ok)
    {
        if (!ok)
            throw std::runtime_error("GPU Pauli operation failed");
    }

    std::unique_ptr<GpuPauliPropagator> pp;
};
} // namespace Simulators::Private
#endif
