#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Core/Factory.h"
#include "ImmediateGpuSimulator.h"

namespace Simulators::Private
{
// statevector backend. Owns exactly one native Gpu implementation.
class GpuStatevectorSimulator : public ImmediateGpuSimulator
{
  public:
    GpuStatevectorSimulator()
    {
        configuration.SetConfiguration("method", "statevector");
    }

    const char *MethodName() const override
    {
        return "statevector";
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
                state = SimulatorsFactory::CreateGpuLibStateVectorSim(gpuDevice);
                if (state)
                {
                    for (const auto &[key, value] : configuration.GetConfigMap())
                        if (key != "method")
                            Configure(key.c_str(), value.c_str());
                    const bool res = state->Create(nrQubits);
                    if (!res)
                        throw std::runtime_error("GpuState::Initialize: Failed to create "
                                                 "and initialize the statevector state.");
                }
                else
                    throw std::runtime_error("GpuState::Initialize: Failed to create the statevector state.");
            }
            if (GetGpuDevice() != gpuDevice)
                throw std::runtime_error("GpuState::Initialize: GPU plugin did not confirm the requested "
                                         "device; update the GPU library");
        }
    }

    void InitializeState(size_t num_qubits, std::vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        const bool created = state->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
        if (!created)
            throw std::runtime_error("GpuState::InitializeState: Failed to initialize the state.");
    }

#ifndef NO_QISKIT_AER
    void InitializeState(size_t num_qubits, AER::Vector<std::complex<double>> &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        const bool created = state->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
        if (!created)
            throw std::runtime_error("GpuState::InitializeState: Failed to initialize the state.");
    }

#endif

    void InitializeState(size_t num_qubits, Eigen::VectorXcd &amplitudes) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        const bool created = state->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
        if (!created)
            throw std::runtime_error("GpuState::InitializeState: Failed to initialize the state.");
    }

    void Reset() override
    {
        if (state)
            state->Reset();
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;
        if (std::string("gpu_device") == key)
        {
            const int device = Configuration::ParseGpuDevice(value);
            if ((state) && device != Configuration::ParseGpuDevice(configuration.GetConfiguration(key)))
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
            // Precision is immutable after native storage is allocated.
            // Replayed settings take effect after Clear and Initialize.
            const bool useDouble = Configuration::ParsePrecision(key, value);
            if ((state && state->GetNrQubits() > 0))
                return;
            if ((state && !state->SetDataType(useDouble)))
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
            if (state)
                state->SetSeed(seed);
            return;
        }
        // Configuration can be replayed from another method. Preserve the
        // numeric validation, but tensor truncation has no native effect here.
        if (std::string("matrix_product_state_truncation_threshold") == key || std::string("matrix_product_operator_truncation_threshold") == key)
            std::stod(value);
        else if (std::string("matrix_product_state_max_bond_dimension") == key || std::string("matrix_product_operator_max_bond_dimension") == key)
            std::stoi(value);
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = state ? state->IsDoublePrecision() : configuration.GetConfiguration("precision") == "double";
            return std::string(key) == "precision" ? (useDouble ? "double" : "single") : (useDouble ? "1" : "0");
        }
        if (!key)
            return {};
        if (std::string("method") == key)
            return MethodName();
        return configuration.GetConfiguration(key);
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (state)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        state = nullptr;
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
                if (state->MeasureQubitCollapse(static_cast<int>(qubit)))
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
                res[i] = state->MeasureQubitCollapse(static_cast<int>(qubits[i]));
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
                if (state->MeasureQubitCollapse(static_cast<int>(qubit)))
                    state->ApplyX(static_cast<int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    double Probability(Types::qubit_t outcome) override
    {
        return state->BasisStateProbability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        double real = 0.0;
        double imag = 0.0;
        state->Amplitude(outcome, &real, &imag);
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
        state->AllProbabilities(result.data());
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (size_t i = 0; i < qubits.size(); ++i)
                result[i] = state->BasisStateProbability(qubits[i]);
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
            std::vector<long int> samples(shots);
            state->SampleAll(shots, samples.data());
            for (auto outcome : samples)
            {
                Types::qubit_t translatedOutcome = 0;
                Types::qubit_t mask = 1ULL;
                for (size_t i = 0; i < qubits.size(); ++i)
                {
                    if (outcome & (1ULL << qubits[i]))
                        translatedOutcome |= mask;
                    mask <<= 1;
                }
                ++result[translatedOutcome];
            }
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
            std::vector<long int> samples(shots);
            state->SampleAll(shots, samples.data());
            std::vector<bool> outcomeVec(qubits.size());
            for (auto outcome : samples)
            {
                for (size_t i = 0; i < qubits.size(); ++i)
                    outcomeVec[i] = ((outcome >> qubits[i]) & 1) == 1;
                ++result[outcomeVec];
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    double ExpectationValue(const std::string &pauliString) override
    {
        double result = 0.0;
        result = state->ExpectationValue(pauliString);
        return result;
    }

    int GetGpuDevice() const override
    {
        if (state)
            return state->GetGpuDevice();
        return -1;
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kStatevector;
    }

    void Flush() override
    {
        const bool done = state ? state->Synchronize() : true;
        if (!done)
            throw std::runtime_error("GpuState::Flush: device synchronization failed");
    }

    void SaveStateToInternalDestructive() override
    {
        state->SaveStateDestructive();
    }

    void RestoreInternalDestructiveSavedState() override
    {
        state->RestoreStateFreeSaved();
    }

    void SaveState() override
    {
        state->SaveState();
    }

    void RestoreState() override
    {
        state->RestoreStateNoFreeSaved();
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        return state->MeasureAllQubitsNoCollapse();
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        const auto meas = state->MeasureAllQubitsNoCollapse();
        std::vector<bool> result(nrQubits, false);
        for (size_t i = 0; i < nrQubits; ++i)
            result[i] = ((meas >> i) & 1) == 1;
        return result;
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        ValidateStatevectorTarget(qubit);
        if (!state->ApplyOneQubitMatrixWithLayout(static_cast<int>(qubit), reinterpret_cast<const double *>(gate.data()), GpuLibrary::MATRIX_COLUMN_MAJOR))
            throw std::runtime_error("GPU statevector failed to apply one-qubit matrix");
        NotifyObservers({qubit});
        return;
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        ValidateStatevectorTarget(qubit0);
        ValidateStatevectorTarget(qubit1);
        if (!state->ApplyTwoQubitMatrixWithLayout(static_cast<int>(qubit0), static_cast<int>(qubit1), reinterpret_cast<const double *>(gate.data()),
                                                  GpuLibrary::MATRIX_COLUMN_MAJOR))
            throw std::runtime_error("GPU statevector failed to apply two-qubit matrix");
        NotifyObservers({qubit0, qubit1});
        return;
    }

    void ApplyGenericThreeQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2,
                                    const Eigen::Matrix<std::complex<double>, 8, 8> &gate) override
    {
        ValidateStatevectorTarget(qubit0);
        ValidateStatevectorTarget(qubit1);
        ValidateStatevectorTarget(qubit2);
        if (!state->ApplyThreeQubitMatrixWithLayout(static_cast<int>(qubit0), static_cast<int>(qubit1), static_cast<int>(qubit2),
                                                    reinterpret_cast<const double *>(gate.data()), GpuLibrary::MATRIX_COLUMN_MAJOR))
            throw std::runtime_error("GPU statevector failed to apply three-qubit matrix");
        NotifyObservers({qubit0, qubit1, qubit2});
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        state->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        state->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        state->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        state->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        state->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        state->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        state->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        state->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        state->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        state->ApplySX(qubit);
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        state->ApplySXDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        state->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        state->ApplyRx(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        state->ApplyRy(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        state->ApplyRz(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        state->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        state->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        state->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        state->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        state->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        state->ApplyCRx(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        state->ApplyCRy(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        state->ApplyCRz(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        state->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        state->ApplyCSX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        state->ApplyCSXDG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        state->ApplySwap(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        state->ApplyCCX(qubit0, qubit1, qubit2);
        NotifyObservers({qubit0, qubit1, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        state->ApplyCSwap(ctrl_qubit, qubit0, qubit1);
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        state->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<GpuStatevectorSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->configuration = configuration;
        if (state)
            cloned->state = state->Clone();
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::unique_ptr<GpuLibStateVectorSim> state;

  private:
    void ValidateStatevectorTarget(Types::qubit_t qubit) const
    {
        if (!state)
            throw std::logic_error("GPU statevector is not initialized");
        if (qubit >= GetNumberOfQubits())
            throw std::out_of_range("GPU statevector qubit is out of range");
    }
};
} // namespace Simulators::Private
#endif
