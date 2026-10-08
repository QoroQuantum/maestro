#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Core/Factory.h"
#include "ImmediateGpuSimulator.h"

namespace Simulators::Private
{
// tensor_network backend. Owns exactly one native Gpu implementation.
class GpuTensorNetworkSimulator : public ImmediateGpuSimulator
{
  public:
    GpuTensorNetworkSimulator()
    {
        configuration.SetConfiguration("method", "tensor_network");
    }

    const char *MethodName() const override
    {
        return "tensor_network";
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
                tn = SimulatorsFactory::CreateGpuLibTensorNetSim(gpuDevice);
                if (tn)
                {
                    for (const auto &[key, value] : configuration.GetConfigMap())
                        if (key != "method")
                            Configure(key.c_str(), value.c_str());
                    const bool res = tn->Create(nrQubits);
                    if (!res)
                        throw std::runtime_error("GpuState::Initialize: Failed to create "
                                                 "and initialize the tensor network state.");
                }
                else
                    throw std::runtime_error("GpuState::Initialize: Failed to create the tensor network "
                                             "state.");
            }
            if (GetGpuDevice() != gpuDevice)
                throw std::runtime_error("GpuState::Initialize: GPU plugin did not confirm the requested "
                                         "device; update the GPU library");
        }
    }

    void Reset() override
    {
        if (tn)
            tn->Reset();
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;
        if (std::string("gpu_device") == key)
        {
            const int device = Configuration::ParseGpuDevice(value);
            if ((tn) && device != Configuration::ParseGpuDevice(configuration.GetConfiguration(key)))
                throw std::invalid_argument("gpu_device cannot change after initialization; clear the "
                                            "simulator first");
            configuration.SetConfiguration(key, std::to_string(device));
            return;
        }
        const auto svdGroup = Configuration::GpuSvdSettingGroup(key);
        if (!svdGroup.empty())
        {
            const bool enabled = Configuration::ParseGpuSvdFlag(value);
            const bool gesvd = svdGroup == key;
            const char algorithm = std::string(key).back();
            const auto apply = [algorithm, enabled](auto &backend) {
                if (!backend)
                    return true;
                if (algorithm == 'j')
                    return backend->SetGesvdJ(enabled);
                if (algorithm == 'p')
                    return backend->SetGesvdP(enabled);
                return backend->SetGesvdR(enabled);
            };
            const auto applyGesvd = [enabled](auto &backend) {
                if (!enabled || !backend)
                    return true;
                return backend->SetGesvdJ(false) && backend->SetGesvdP(false) && backend->SetGesvdR(false);
            };
            bool applied = true;
            if (svdGroup == "tensor_network_use_gesvd")
                applied = gesvd ? applyGesvd(tn) : apply(tn);
            if (!applied)
                throw std::runtime_error(std::string("GPU library cannot apply ") + key + "; an updated GPU plugin may be required");
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
            // Precision is immutable after native storage is allocated.
            // Replayed settings take effect after Clear and Initialize.
            const bool useDouble = Configuration::ParsePrecision(key, value);
            if ((tn && tn->IsCreated()))
                return;
            if ((tn && !tn->SetDataType(useDouble)))
                throw std::runtime_error("GPU precision configuration failed");
            configuration.SetConfiguration(key, value);
            return;
        }
        if (!configuration.WasApplied(key, value))
            configuration.SetConfiguration(key, value);
        if (std::string("seed") == key)
        {
            const uint64_t seed = std::stoull(value);
            SeedAuxiliaryRng(seed);
            nextSeedStream = 0;
            if (tn)
                tn->SetSeed(seed);
            return;
        }
        if (std::string("matrix_product_state_truncation_threshold") == key || std::string("matrix_product_operator_truncation_threshold") == key)
        {
            const double singularValueThreshold = std::stod(value);
            if (singularValueThreshold >= 0.)
            {
                if (tn)
                    tn->SetCutoff(singularValueThreshold);
            }
        }
        else if (std::string("matrix_product_state_truncation_mode") == key || std::string("matrix_product_operator_truncation_mode") == key)
        {
            int truncationMode = -1;
            if (std::string("relative_max") == value)
                truncationMode = 0;
            else if (std::string("discarded_weight") == value)
                truncationMode = 1;
            if (truncationMode >= 0)
            {
                if (tn)
                    tn->SetTruncationMode(truncationMode);
            }
        }
        else if (std::string("matrix_product_state_max_bond_dimension") == key || std::string("matrix_product_operator_max_bond_dimension") == key)
        {
            const long long int chi = std::stoi(value);
            if (chi > 0)
            {
                if (tn)
                    tn->SetMaxExtent(chi);
            }
        }
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = tn ? tn->IsDoublePrecision() : configuration.GetConfiguration("precision") == "double";
            return std::string(key) == "precision" ? (useDouble ? "double" : "single") : (useDouble ? "1" : "0");
        }
        if (!key)
            return {};
        const auto svdGroup = Configuration::GpuSvdSettingGroup(key);
        if (!svdGroup.empty())
        {
            if (svdGroup == key)
            {
                const auto readGesvd = [](const auto &backend) { return !backend->GetGesvdJ() && !backend->GetGesvdP() && !backend->GetGesvdR(); };
                if (svdGroup == "tensor_network_use_gesvd" && tn)
                    return readGesvd(tn) ? "true" : "false";
            }
            const char algorithm = std::string(key).back();
            const auto read = [algorithm](const auto &backend) {
                if (algorithm == 'j')
                    return backend->GetGesvdJ();
                if (algorithm == 'p')
                    return backend->GetGesvdP();
                return backend->GetGesvdR();
            };
            if (svdGroup == "tensor_network_use_gesvd" && tn)
                return read(tn) ? "true" : "false";
        }
        if (std::string("method") == key)
            return MethodName();
        return configuration.GetConfiguration(key);
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        tn = nullptr;
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
        size_t res = 0;
        size_t mask = 1ULL;
        DontNotify();
        {
            for (size_t qubit : qubits)
            {
                if (tn->Measure(static_cast<unsigned int>(qubit)))
                    res |= mask;
                mask <<= 1;
            }
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        std::vector<bool> res(qubits.size(), false);
        DontNotify();
        {
            for (size_t i = 0; i < qubits.size(); ++i)
                res[i] = tn->Measure(static_cast<unsigned int>(qubits[i]));
        }
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        DontNotify();
        {
            for (size_t qubit : qubits)
                if (tn->Measure(static_cast<unsigned int>(qubit)))
                    tn->ApplyX(static_cast<unsigned int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        double real = 0.0;
        double imag = 0.0;
        {
            std::vector<long int> fixedValues(nrQubits);
            for (size_t i = 0; i < nrQubits; ++i)
                fixedValues[i] = i < std::numeric_limits<Types::qubit_t>::digits ? (outcome >> i) & 1 : 0;
            tn->Amplitude(nrQubits, fixedValues.data(), &real, &imag);
        }
        return std::complex<double>(real, imag);
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
                const auto val = Amplitude(i);
                result[i] = std::norm(std::complex<double>(val.real(), val.imag()));
            }
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();
        {
            std::unordered_map<std::vector<bool>, int64_t> *map = tn->GetMapForSample();
            std::vector<unsigned int> qubitsIndices(qubits.begin(), qubits.end());
            tn->Sample(shots, qubitsIndices.size(), qubitsIndices.data(), map);
            const auto positions = SampleBitPositions(qubits);
            for (const auto &[meas, cnt] : *map)
            {
                Types::qubit_t outcome = 0;
                Types::qubit_t mask = 1ULL;
                for (Types::qubit_t q = 0; q < qubits.size(); ++q)
                {
                    if (meas[positions[q]])
                        outcome |= mask;
                    mask <<= 1;
                }
                result[outcome] += cnt;
            }
            tn->FreeMapForSample(map);
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();
        {
            std::unordered_map<std::vector<bool>, int64_t> *map = tn->GetMapForSample();
            std::vector<unsigned int> qubitsIndices(qubits.begin(), qubits.end());
            tn->Sample(shots, qubitsIndices.size(), qubitsIndices.data(), map);
            const auto positions = SampleBitPositions(qubits);
            for (const auto &[meas, cnt] : *map)
            {
                std::vector<bool> ordered(qubits.size());
                for (size_t q = 0; q < qubits.size(); ++q)
                    ordered[q] = meas[positions[q]];
                result[ordered] += cnt;
            }
            tn->FreeMapForSample(map);
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    double ExpectationValue(const std::string &pauliString) override
    {
        double result = 0.0;
        result = tn->ExpectationValue(pauliString);
        return result;
    }

    int GetGpuDevice() const override
    {
        if (tn)
            return tn->GetGpuDevice();
        return -1;
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kTensorNetwork;
    }

    void SaveState() override
    {
        tn->SaveState();
    }

    void RestoreState() override
    {
        tn->RestoreState();
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        tn->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        tn->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        tn->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        tn->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        tn->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        tn->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        tn->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        tn->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        tn->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        tn->ApplySX(qubit);
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        tn->ApplySXDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        tn->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        tn->ApplyRx(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        tn->ApplyRy(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        tn->ApplyRz(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        tn->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tn->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tn->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tn->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        tn->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        tn->ApplyCRx(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        tn->ApplyCRy(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        tn->ApplyCRz(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tn->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tn->ApplyCSX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        tn->ApplyCSXDG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        tn->ApplySwap(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        tn->ApplyCCX(qubit0, qubit1, qubit2);
        NotifyObservers({qubit0, qubit1, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        tn->ApplyCSwap(ctrl_qubit, qubit0, qubit1);
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        tn->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        throw std::runtime_error("GpuSimulator::Clone: Cloning Tensor Network or Pauli Propagator "
                                 "simulation is not "
                                 "supported.");
    }

  protected:
    std::unique_ptr<GpuLibTNSim> tn;
};
} // namespace Simulators::Private
#endif
