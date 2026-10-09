#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Core/Factory.h"
#include "ImmediateGpuSimulator.h"

namespace Simulators::Private
{
// density_matrix backend. Owns exactly one native Gpu implementation.
class GpuDensityMatrixSimulator : public ImmediateGpuSimulator
{
  public:
    GpuDensityMatrixSimulator()
    {
        configuration.SetConfiguration("method", "density_matrix");
    }

    const char *MethodName() const override
    {
        return "density_matrix";
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
                densityMatrix = SimulatorsFactory::CreateGpuDensityMatrix(gpuDevice);
                if (!densityMatrix)
                    throw std::runtime_error("GpuState::Initialize: Failed to create the density matrix "
                                             "state.");
                for (const auto &[key, value] : configuration.GetConfigMap())
                    if (key != "method")
                        Configure(key.c_str(), value.c_str());
                if (!densityMatrix->Create(nrQubits))
                    throw std::runtime_error("GpuState::Initialize: Failed to initialize the density matrix "
                                             "state.");
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
        const bool created = densityMatrix->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
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
        const bool created = densityMatrix->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
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
        const bool created = densityMatrix->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
        if (!created)
            throw std::runtime_error("GpuState::InitializeState: Failed to initialize the state.");
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        const bool created = densityMatrix->CreateWithBasisState(nrQubits, static_cast<unsigned long long>(basisState));
        if (!created)
            throw std::runtime_error("GpuState::InitializeToBasisState: Failed to initialize the "
                                     "state.");
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<Types::qubit_t, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        std::vector<std::pair<unsigned long long, double>> converted;
        converted.reserve(mixture.size());
        for (const auto &[basisState, weight] : mixture)
            converted.emplace_back(static_cast<unsigned long long>(basisState), weight);
        const bool created = densityMatrix->CreateWithMixtureOfBasisStates(nrQubits, converted);
        if (!created)
            throw std::runtime_error("GpuState::InitializeToMixtureOfBasisStates: Failed to initialize "
                                     "the state.");
    }

    void Reset() override
    {
        if (densityMatrix)
            densityMatrix->Reset();
        upcomingGateIndex = 0;
    }

    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;
        if (std::string("gpu_device") == key)
        {
            const int device = Configuration::ParseGpuDevice(value);
            if ((densityMatrix) && device != Configuration::ParseGpuDevice(configuration.GetConfiguration(key)))
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
            if ((densityMatrix && densityMatrix->IsCreated()))
                return;
            if (densityMatrix)
                densityMatrix->SetDataType(useDouble);
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
            if (densityMatrix)
                densityMatrix->SetSeed(seed);
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
            const bool useDouble = densityMatrix ? densityMatrix->IsDoublePrecision() : configuration.GetConfiguration("precision") == "double";
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
        if (densityMatrix)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        densityMatrix = nullptr;
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
                if (densityMatrix->Measure(static_cast<unsigned int>(qubit)))
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
                res[i] = densityMatrix->Measure(qubits[i]);
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
                densityMatrix->ApplyReset(qubit);
        }
        Notify();
        NotifyObservers(qubits);
    }

    bool SupportsQuantumChannels() const override
    {
        return true;
    }

    void ApplyQuantumChannel(const Types::qubits_vector &targets, const QuantumChannel &channel) override
    {
        if (!densityMatrix)
            throw std::runtime_error("GPU quantum channels require an initialized density matrix or "
                                     "matrix product operator");
        if (targets.size() != channel.GetNumberOfQubits() || targets.empty() || targets.size() > 2)
            throw std::invalid_argument("GPU density matrices and matrix product operators support one- "
                                        "and two-qubit local channels");
        std::vector<int> gpuTargets;
        gpuTargets.reserve(targets.size());
        for (auto target : targets)
        {
            if (target >= nrQubits || std::find(gpuTargets.begin(), gpuTargets.end(), target) != gpuTargets.end())
                throw std::invalid_argument("Invalid GPU quantum-channel target");
            gpuTargets.push_back(static_cast<int>(target));
        }
        const auto &kraus = channel.GetKrausOperators();
        std::vector<double> interleaved;
        interleaved.reserve(kraus.size() * kraus.front().size() * 2);
        for (const auto &op : kraus)
            for (Eigen::Index i = 0; i < op.size(); ++i)
            {
                interleaved.push_back(op.data()[i].real());
                interleaved.push_back(op.data()[i].imag());
            }
        const bool applied = densityMatrix->ApplyKraus(gpuTargets, kraus.size(), interleaved.data());
        if (!applied)
            throw std::runtime_error("GPU density-matrix/matrix-product-operator channel application "
                                     "failed");
        NotifyObservers(targets);
    }

    std::complex<double> DensityMatrixTrace() const override
    {
        if (densityMatrix)
            return densityMatrix->Trace();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    double DensityMatrixPurity() const override
    {
        if (densityMatrix)
            return densityMatrix->Purity();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        if (densityMatrix)
        {
            const double tr = densityMatrix->Trace();
            return densityMatrix->Purity() * tr * tr;
        }
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        const auto *rhs = dynamic_cast<const GpuDensityMatrixSimulator *>(&other);
        if (!rhs)
            throw std::invalid_argument("Density-matrix overlap requires matching GPU backends");
        if (densityMatrix && rhs->densityMatrix)
            return densityMatrix->HilbertSchmidtOverlap(*rhs->densityMatrix);
        throw std::invalid_argument("Density-matrix overlap requires two density matrices or two MPOs");
    }

    double DensityMatrixHermiticityResidual() const override
    {
        if (densityMatrix)
            return densityMatrix->IsHermitian() ? 0. : std::numeric_limits<double>::infinity();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        if (densityMatrix)
            return densityMatrix->IsHermitian(eps);
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qubits) const override
    {
        std::vector<int> keep(qubits.begin(), qubits.end());
        const auto values = densityMatrix ? densityMatrix->PartialTrace(keep) : throw std::runtime_error("GPU partial trace requires a mixed-state backend");
        const Eigen::Index dim = static_cast<Eigen::Index>(size_t{1} << keep.size());
        return Eigen::Map<const Eigen::MatrixXcd>(values.data(), dim, dim);
    }

    double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override
    {
        std::vector<double> raw(2 * static_cast<size_t>(psi.size()));
        for (Eigen::Index i = 0; i < psi.size(); ++i)
        {
            raw[2 * i] = psi[i].real();
            raw[2 * i + 1] = psi[i].imag();
        }
        if (densityMatrix)
            return densityMatrix->FidelityWithStatevector(raw.data());
        throw std::runtime_error("GPU mixed-state fidelity requires density_matrix or "
                                 "matrix_product_operator");
    }

    double Probability(Types::qubit_t outcome) override
    {
        return densityMatrix->Probability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("GpuState::Amplitude: Amplitudes are not defined for density "
                                 "matrices.");
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
        densityMatrix->AllProbabilities(result.data());
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (size_t i = 0; i < qubits.size(); ++i)
                result[i] = densityMatrix->Probability(qubits[i]);
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
            if (!densityMatrix->SampleAll(shots, samples.data()))
            {
                Notify();
                throw std::runtime_error("GpuState::SampleCounts: Density-matrix sampling failed.");
            }
            for (auto outcome : samples)
            {
                Types::qubit_t translatedOutcome = 0;
                for (size_t i = 0; i < qubits.size(); ++i)
                    if (outcome & (1ULL << qubits[i]))
                        translatedOutcome |= 1ULL << i;
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
            if (!densityMatrix->SampleAll(shots, samples.data()))
            {
                Notify();
                throw std::runtime_error("GpuState::SampleCountsMany: Density-matrix sampling failed.");
            }
            std::vector<bool> outcomeVec(qubits.size());
            for (auto outcome : samples)
            {
                for (size_t i = 0; i < qubits.size(); ++i)
                    outcomeVec[i] = ((outcome >> qubits[i]) & 1) != 0;
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
        result = densityMatrix->ExpectationValue(pauliString);
        return result;
    }

    int GetGpuDevice() const override
    {
        if (densityMatrix)
            return densityMatrix->GetGpuDevice();
        return -1;
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kDensityMatrix;
    }

    void Flush() override
    {
        const bool done = densityMatrix ? densityMatrix->Synchronize() : true;
        if (!done)
            throw std::runtime_error("GpuState::Flush: device synchronization failed");
    }

    void SaveState() override
    {
        densityMatrix->SaveState();
    }

    void RestoreState() override
    {
        densityMatrix->RestoreState();
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        std::vector<long int> samples(1);
        if (!densityMatrix->SampleAll(1, samples.data()))
            throw std::runtime_error("GpuState::MeasureNoCollapse: Density-matrix sampling failed.");
        return static_cast<Types::qubit_t>(samples.front());
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        std::vector<long int> samples(1);
        if (!densityMatrix->SampleAll(1, samples.data()))
            throw std::runtime_error("GpuState::MeasureNoCollapseMany: Density-matrix sampling failed.");
        std::vector<bool> result(nrQubits, false);
        for (size_t i = 0; i < nrQubits; ++i)
            result[i] = ((samples.front() >> i) & 1) != 0;
        return result;
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        ApplyQuantumChannel({qubit}, QuantumChannel({gate}));
        return;
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        ApplyQuantumChannel({qubit0, qubit1}, QuantumChannel({gate}));
        return;
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        densityMatrix->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        densityMatrix->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        densityMatrix->ApplySX(qubit);
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        densityMatrix->ApplySXDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        densityMatrix->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        densityMatrix->ApplyRx(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        densityMatrix->ApplyRy(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        densityMatrix->ApplyRz(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        densityMatrix->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        densityMatrix->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        densityMatrix->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        densityMatrix->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        densityMatrix->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        densityMatrix->ApplyCRx(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        densityMatrix->ApplyCRy(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        densityMatrix->ApplyCRz(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        densityMatrix->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        densityMatrix->ApplyCSX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        densityMatrix->ApplyCSXDG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        densityMatrix->ApplySwap(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        densityMatrix->ApplyCCX(qubit0, qubit1, qubit2);
        NotifyObservers({qubit0, qubit1, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        densityMatrix->ApplyCSwap(ctrl_qubit, qubit0, qubit1);
        NotifyObservers({qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        densityMatrix->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<GpuDensityMatrixSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->configuration = configuration;
        if (densityMatrix)
        {
            cloned->densityMatrix = densityMatrix->Clone();
            if (!cloned->densityMatrix)
                throw std::runtime_error("GpuSimulator::Clone: Failed to clone density matrix state.");
        }
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::unique_ptr<GpuDensityMatrix> densityMatrix;
};
} // namespace Simulators::Private
#endif
