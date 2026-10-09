#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Core/Factory.h"
#include "GpuTensorChainSimulator.h"

namespace Simulators::Private
{
// matrix_product_operator backend. Owns exactly one native Gpu implementation.
class GpuMPOSimulator : public GpuTensorChainSimulator
{
  public:
    GpuMPOSimulator() : GpuTensorChainSimulator(true)
    {
        configuration.SetConfiguration("method", "matrix_product_operator");
        configuration.SetConfiguration("matrix_product_state_max_bond_dimension", "128");
        configuration.SetConfiguration("matrix_product_operator_max_bond_dimension", "128");
    }

    const char *MethodName() const override
    {
        return "matrix_product_operator";
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
                mpo = SimulatorsFactory::CreateGpuMPO(gpuDevice);
                if (!mpo)
                    throw std::runtime_error("GpuState::Initialize: Failed to create the matrix product "
                                             "operator state.");
                mpo->SetCallbackContext(static_cast<GpuTensorChainSimulator *>(this));
                curMaxBondDim = 1;
                mpo->InstallBondSummary(&GpuTensorChainSimulator::BondSummaryCallback, &GpuTensorChainSimulator::BondDimCallback);
                for (const auto &[key, value] : configuration.GetConfigMap())
                    if (key != "method")
                        Configure(key.c_str(), value.c_str());
                if (!mpo->Create(nrQubits))
                    throw std::runtime_error("GpuState::Initialize: Failed to initialize the matrix product "
                                             "operator state.");
                if (!useOptimalMeetingPosition)
                    mpo->SetUseOptimalMeetingPosition(false);
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
        const bool created = mpo->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
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
        const bool created = mpo->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
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
        const bool created = mpo->CreateWithState(nrQubits, reinterpret_cast<const double *>(amplitudes.data()));
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
        const bool created = mpo->CreateWithBasisState(nrQubits, static_cast<unsigned long long>(basisState));
        if (!created)
            throw std::runtime_error("GpuState::InitializeToBasisState: Failed to initialize the "
                                     "state.");
    }

    void InitializeToBasisState(size_t num_qubits, const std::vector<bool> &basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        bool created = true;
        {
            std::vector<unsigned char> stateBits(num_qubits, 0);
            for (size_t q = 0; q < num_qubits && q < basisState.size(); ++q)
                stateBits[q] = basisState[q] ? 1 : 0;
            created = mpo->CreateWithBasisStateBits(nrQubits, stateBits);
        }
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
        const bool created = mpo->CreateWithMixtureOfBasisStates(nrQubits, converted);
        if (!created)
            throw std::runtime_error("GpuState::InitializeToMixtureOfBasisStates: Failed to initialize "
                                     "the state.");
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<std::vector<bool>, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        std::vector<double> weights;
        weights.reserve(mixture.size());
        std::vector<unsigned char> stateBitsFlat;
        stateBitsFlat.reserve(mixture.size() * num_qubits);
        for (const auto &[basisState, weight] : mixture)
        {
            weights.push_back(weight);
            for (size_t q = 0; q < num_qubits; ++q)
                stateBitsFlat.push_back((q < basisState.size() && basisState[q]) ? 1 : 0);
        }
        if (!mpo->CreateWithMixtureOfBasisStatesBits(nrQubits, stateBitsFlat, weights))
            throw std::runtime_error("GpuState::InitializeToMixtureOfBasisStates: Failed to initialize "
                                     "the state.");
    }

    void Reset() override
    {
        if (mpo)
        {
            mpo->Reset();
            curMaxBondDim = 1;
        }
        upcomingGateIndex = 0;
    }

    void SetInitialQubitsMap(const std::vector<long long int> &initialMap) override
    {
        if (mpo)
        {
            mpo->SetInitialQubitsMap(initialMap);
            if (!dummySim || dummySim->getNrQubits() != initialMap.size() || dummySim->IsOperatorChain() != true)
            {
                dummySim = std::make_unique<Simulators::MPSDummySimulator>(initialMap.size(), true);
                dummySim->SetMaxBondDimension(configuration.GetConfigurationAsInt(MaxBondDimensionConfigKey()));
            }
            dummySim->setGrowthFactorGate(growthFactorGate);
            dummySim->setGrowthFactorSwap(growthFactorSwap);
            dummySim->SetInitialQubitsMap(initialMap);
        }
    }

    void SetUseOptimalMeetingPosition(bool enable) override
    {
        useOptimalMeetingPosition = enable;
        if (mpo)
            mpo->SetUseOptimalMeetingPosition(enable);
        RefreshRoutingCallback();
    }

    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;
        if (std::string("gpu_device") == key)
        {
            const int device = Configuration::ParseGpuDevice(value);
            if ((mpo) && device != Configuration::ParseGpuDevice(configuration.GetConfiguration(key)))
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
            if (svdGroup == "matrix_product_operator_use_gesvd")
                applied = gesvd ? applyGesvd(mpo) : apply(mpo);
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
            if ((mpo && mpo->IsCreated()))
                return;
            if (mpo)
                mpo->SetDataType(useDouble);
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
            if (mpo)
                mpo->SetSeed(seed);
            return;
        }
        if (std::string("matrix_product_state_truncation_threshold") == key || std::string("matrix_product_operator_truncation_threshold") == key)
        {
            const double singularValueThreshold = std::stod(value);
            if (singularValueThreshold >= 0.)
            {
                if (mpo)
                    mpo->SetCutoff(singularValueThreshold);
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
                if (mpo)
                    mpo->SetTruncationMode(truncationMode);
            }
        }
        else if (std::string("matrix_product_state_max_bond_dimension") == key || std::string("matrix_product_operator_max_bond_dimension") == key)
        {
            const long long int chi = std::stoi(value);
            {
                const std::string bondDimension(value);
                for (const char *alias : {"matrix_product_state_max_bond_dimension", "matrix_product_operator_max_bond_dimension"})
                    if (!configuration.WasApplied(alias, bondDimension))
                        configuration.SetConfiguration(alias, bondDimension);
            }
            if (mpo)
                mpo->SetMaxExtent(chi);
            if (chi > 0)
            {
                if (dummySim)
                    dummySim->SetMaxBondDimension(chi);
            }
        }
        else if (std::string("matrix_product_operator_kraus_completeness_check") == key)
        {
            int mode = -1;
            if (std::string(value) == "ignore")
                mode = 0;
            else if (std::string(value) == "warn")
                mode = 1;
            else if (std::string(value) == "strict")
                mode = 2;
            if (mpo && mode >= 0 && !mpo->SetKrausCompletenessCheck(mode))
                throw std::runtime_error("Invalid GPU MPO Kraus completeness mode");
        }
        else if (std::string("matrix_product_operator_restore_trace_after_truncation") == key)
        {
            if (mpo && !mpo->SetRestoreTraceAfterTruncation(std::string(value) == "true" || std::string(value) == "1"))
                throw std::runtime_error("GPU MPO truncation configuration failed");
        }
        else if (std::string("matrix_product_operator_hermitize_after_truncation") == key)
        {
            if (mpo && !mpo->SetHermitizeAfterTruncation(std::string(value) == "true" || std::string(value) == "1"))
                throw std::runtime_error("GPU MPO truncation configuration failed");
        }
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = mpo ? mpo->IsDoublePrecision() : configuration.GetConfiguration("precision") == "double";
            return std::string(key) == "precision" ? (useDouble ? "double" : "single") : (useDouble ? "1" : "0");
        }
        if (mpo)
        {
            if (std::string(key) == "matrix_product_operator_restore_trace_after_truncation")
                return mpo->GetRestoreTraceAfterTruncation() ? "true" : "false";
            if (std::string(key) == "matrix_product_operator_hermitize_after_truncation")
                return mpo->GetHermitizeAfterTruncation() ? "true" : "false";
        }
        if (!key)
            return {};
        const auto svdGroup = Configuration::GpuSvdSettingGroup(key);
        if (!svdGroup.empty())
        {
            if (svdGroup == key)
            {
                const auto readGesvd = [](const auto &backend) { return !backend->GetGesvdJ() && !backend->GetGesvdP() && !backend->GetGesvdR(); };
                if (svdGroup == "matrix_product_operator_use_gesvd" && mpo)
                    return readGesvd(mpo) ? "true" : "false";
            }
            const char algorithm = std::string(key).back();
            const auto read = [algorithm](const auto &backend) {
                if (algorithm == 'j')
                    return backend->GetGesvdJ();
                if (algorithm == 'p')
                    return backend->GetGesvdP();
                return backend->GetGesvdR();
            };
            if (svdGroup == "matrix_product_operator_use_gesvd" && mpo)
                return read(mpo) ? "true" : "false";
        }
        if (std::string("method") == key)
            return MethodName();
        return configuration.GetConfiguration(key);
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (mpo)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        mpo = nullptr;
        nrQubits = 0;
        dummySim = nullptr;
        upcomingGateIndex = 0;
        upcomingGates.clear();
    }

    size_t Measure(const Types::qubits_vector &qubits) override
    {
        if (mpo && qubits.size() > sizeof(size_t) * 8)
            throw std::invalid_argument("Use MeasureMany for more than 64 measured qubits");
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
                if (mpo->Measure(static_cast<unsigned int>(qubit)))
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
                res[i] = mpo->Measure(static_cast<unsigned int>(qubits[i]));
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
                mpo->ApplyReset(static_cast<int>(qubit));
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
        if (!mpo)
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
        const bool applied = mpo->ApplyKraus(gpuTargets, kraus.size(), interleaved.data());
        if (!applied)
            throw std::runtime_error("GPU density-matrix/matrix-product-operator channel application "
                                     "failed");
        NotifyObservers(targets);
    }

    double ProbabilityBits(const std::vector<bool> &bits) override
    {
        if (!mpo)
            return IState::ProbabilityBits(bits);
        return mpo->ProbabilityBits(std::vector<unsigned char>(bits.begin(), bits.end()));
    }

    std::complex<double> DensityMatrixElementBits(const std::vector<bool> &row, const std::vector<bool> &col) const override
    {
        if (!mpo)
            return IState::DensityMatrixElementBits(row, col);
        return mpo->GetElementBits(std::vector<unsigned char>(row.begin(), row.end()), std::vector<unsigned char>(col.begin(), col.end()));
    }

    Eigen::MatrixXcd GetDensityMatrix(bool normalized = true) const override
    {
        if (!mpo)
            return IState::GetDensityMatrix(normalized);
        const auto values = mpo->GetDensityMatrix(normalized);
        const Eigen::Index dim = Eigen::Index{1} << nrQubits;
        return Eigen::Map<const Eigen::MatrixXcd>(values.data(), dim, dim);
    }

    std::complex<double> ExpectationValueComplex(const std::string &pauli, bool normalized = true) const override
    {
        if (!mpo)
            return IState::ExpectationValueComplex(pauli, normalized);
        MPOValidation::Pauli(pauli, nrQubits);
        return mpo->ExpectationValueComplex(pauli, normalized);
    }

    std::vector<std::complex<double>> ExpectationValuesComplex(const std::vector<std::string> &paulis, bool normalized = true) const override
    {
        if (!mpo)
            return IState::ExpectationValuesComplex(paulis, normalized);
        for (const auto &pauli : paulis)
            MPOValidation::Pauli(pauli, nrQubits);
        return mpo->ExpectationValuesComplex(paulis, normalized);
    }

    void ApplyOperator(const Types::qubits_vector &qubits, const Eigen::MatrixXcd &matrix, bool normalize = false) override
    {
        if (!mpo)
            return IState::ApplyOperator(qubits, matrix, normalize);
        const auto selected = MPOValidation::Operator(qubits, matrix, nrQubits);
        std::vector<double> raw(2 * matrix.size());
        for (Eigen::Index i = 0; i < matrix.size(); ++i)
        {
            raw[2 * i] = matrix.data()[i].real();
            raw[2 * i + 1] = matrix.data()[i].imag();
        }
        if (!mpo->ApplyOperator(selected, raw.data(), normalize))
            throw std::runtime_error("GPU MPO operator application failed");
        NotifyObservers(qubits);
    }

    void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override
    {
        if (!mpo)
            return IState::MoveAtBeginningOfChain(qubits);
        const auto selected = MPOValidation::Qubits(qubits, nrQubits);
        if (selected.empty())
            return;
        mpo->MoveAtBeginningOfChain(selected);
    }

    std::complex<double> DensityMatrixTrace() const override
    {
        if (mpo)
            return mpo->TraceComplex();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    double DensityMatrixPurity() const override
    {
        if (mpo)
            return mpo->Purity();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        if (mpo)
            return mpo->TraceOfSquare();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        const auto *rhs = dynamic_cast<const GpuMPOSimulator *>(&other);
        if (!rhs)
            throw std::invalid_argument("Density-matrix overlap requires matching GPU backends");
        if (mpo && rhs->mpo)
            return mpo->HilbertSchmidtOverlap(*rhs->mpo);
        throw std::invalid_argument("Density-matrix overlap requires two density matrices or two MPOs");
    }

    double DensityMatrixHermiticityResidual() const override
    {
        if (mpo)
            return mpo->HermiticityResidual();
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        if (mpo)
            return mpo->IsHermitian(eps);
        throw std::runtime_error("GPU mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qubits) const override
    {
        std::vector<int> keep(qubits.begin(), qubits.end());
        const auto values = mpo ? mpo->PartialTrace(keep) : throw std::runtime_error("GPU partial trace requires a mixed-state backend");
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
        if (mpo)
            return mpo->FidelityWithStatevector(raw.data());
        throw std::runtime_error("GPU mixed-state fidelity requires density_matrix or "
                                 "matrix_product_operator");
    }

    void RestoreDensityMatrixTrace() override
    {
        if (!mpo)
            throw std::runtime_error("Trace restoration is only available for GPU MPO");
        mpo->RestoreTrace();
    }

    void HermitizeDensityMatrix() override
    {
        if (!mpo)
            throw std::runtime_error("Hermitization is only available for GPU MPO");
        mpo->Hermitize();
    }

    void Trim() override
    {
        if (mpo)
            mpo->Trim();
        else
            throw std::runtime_error("Trim is only available for GPU MPS and MPO");
    }

    void ReCanonicalize() override
    {
        if (mpo)
            mpo->ReCanonicalize();
        else
            throw std::runtime_error("Canonicalization is only available for GPU MPS and MPO");
    }

    double Probability(Types::qubit_t outcome) override
    {
        if (nrQubits < sizeof(Types::qubit_t) * 8 && outcome >= (Types::qubit_t{1} << nrQubits))
            throw std::out_of_range("GPU MPO basis state is out of range");
        std::vector<bool> bits(nrQubits);
        for (size_t q = 0; q < std::min(nrQubits, sizeof(Types::qubit_t) * 8); ++q)
            bits[q] = ((outcome >> q) & 1) != 0;
        return ProbabilityBits(bits);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("GpuState::Amplitude: Amplitudes are not defined for matrix "
                                 "product operators.");
    }

    std::vector<double> AllProbabilities() override
    {
        if (nrQubits == 0)
            return {};
        if (nrQubits >= std::numeric_limits<size_t>::digits || (mpo && nrQubits >= 63))
            throw std::length_error("Full probability enumeration exceeds the basis-index API; query "
                                    "selected bit vectors");
        const size_t numStates = size_t{1} << nrQubits;
        std::vector<double> result(numStates);
        mpo->AllProbabilities(result.data());
        return result;
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qubits) override
    {
        std::vector<double> result(qubits.size());
        {
            for (size_t i = 0; i < qubits.size(); ++i)
                result[i] = Probability(qubits[i]);
        }
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (mpo)
        {
            if (qubits.size() > sizeof(Types::qubit_t) * 8)
                throw std::invalid_argument("Use SampleCountsMany for more than 64 measured qubits");
            std::unordered_map<Types::qubit_t, Types::qubit_t> result;
            for (const auto &entry : SampleCountsMany(qubits, shots))
            {
                Types::qubit_t value = 0;
                for (size_t i = 0; i < entry.first.size(); ++i)
                    if (entry.first[i])
                        value |= Types::qubit_t{1} << i;
                result[value] += entry.second;
            }
            return result;
        }
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        DontNotify();
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (mpo)
        {
            std::vector<unsigned int> selected;
            for (auto q : qubits)
            {
                if (q >= nrQubits || q > static_cast<size_t>(std::numeric_limits<int>::max()))
                    throw std::out_of_range("GPU tensor sampled qubit is out of range");
                selected.push_back(static_cast<unsigned int>(q));
            }
            if (qubits.empty() || shots == 0)
                return {};
            std::sort(selected.begin(), selected.end());
            selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
            const auto counts = mpo->SampleHistogram(shots, std::move(selected));
            const auto positions = SampleBitPositions(qubits);
            std::unordered_map<std::vector<bool>, Types::qubit_t> result;
            for (const auto &entry : counts)
            {
                std::vector<bool> bits(qubits.size());
                for (size_t i = 0; i < bits.size(); ++i)
                    bits[i] = entry.first[positions[i]];
                result[std::move(bits)] += entry.second;
            }
            NotifyObservers(qubits);
            return result;
        }
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        DontNotify();
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    double ExpectationValue(const std::string &pauliString) override
    {
        double result = 0.0;
        result = mpo->ExpectationValue(pauliString);
        return result;
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        if (!mpo)
            return IState::ExpectationValues(paulis);
        std::vector<double> result(paulis.size(), 0.);
        std::vector<std::string> selected;
        std::vector<size_t> indices;
        for (size_t i = 0; i < paulis.size(); ++i)
        {
            if (paulis[i].empty())
            {
                result[i] = 1.;
                continue;
            }
            if (paulis[i].size() > nrQubits)
                continue;
            auto pauli = paulis[i];
            for (char &p : pauli)
                p = static_cast<char>(std::toupper(static_cast<unsigned char>(p)));
            if (pauli.find_first_not_of("IXYZ") != std::string::npos)
                continue;
            indices.push_back(i);
            selected.push_back(std::move(pauli));
        }
        if (selected.empty())
            return result;
        const auto values = mpo->ExpectationValues(selected);
        for (size_t i = 0; i < indices.size(); ++i)
            result[indices[i]] = values[i];
        return result;
    }

    int GetGpuDevice() const override
    {
        if (mpo)
            return mpo->GetGpuDevice();
        return -1;
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kMatrixProductOperator;
    }

    void SaveState() override
    {
        mpo->SaveState();
    }

    void RestoreState() override
    {
        mpo->RestoreState();
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (mpo && nrQubits > sizeof(Types::qubit_t) * 8)
            throw std::invalid_argument("Use MeasureNoCollapseMany for more than 64 qubits");
        {
            std::vector<long int> samples(1);
            if (!mpo->SampleAll(1, samples.data()))
                throw std::runtime_error("GpuState::MeasureNoCollapse: Matrix-product-operator sampling "
                                         "failed.");
            return static_cast<Types::qubit_t>(samples.front());
        }
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        std::vector<int> qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        const auto bits = mpo->SampleBits(1, qubits);
        return std::vector<bool>(bits.begin(), bits.end());
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        if (!mpo->ApplyOneQubitMatrix(static_cast<int>(qubit), reinterpret_cast<const double *>(gate.data())))
            throw std::runtime_error("GpuSimulator::ApplyGenericOneQubitGate: Failed to apply the "
                                     "generic one-qubit gate on the matrix product operator "
                                     "simulator.");
        NotifyObservers({qubit});
        return;
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        if (!mpo->ApplyTwoQubitMatrix(static_cast<int>(qubit0), static_cast<int>(qubit1), reinterpret_cast<const double *>(gate.data())))
            throw std::runtime_error("GpuSimulator::ApplyGenericTwoQubitGate: Failed to apply the "
                                     "generic two-qubit gate on the matrix product operator "
                                     "simulator.");
        NotifyObservers({qubit0, qubit1});
        return;
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        mpo->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        mpo->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        mpo->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        mpo->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        mpo->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        mpo->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        mpo->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        mpo->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        mpo->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        mpo->ApplySX(qubit);
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        mpo->ApplySXDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        mpo->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        mpo->ApplyRx(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        mpo->ApplyRy(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        mpo->ApplyRz(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        mpo->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpo->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpo->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpo->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        mpo->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        mpo->ApplyCRx(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        mpo->ApplyCRy(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        mpo->ApplyCRz(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpo->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpo->ApplyCSX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mpo->ApplyCSXDG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        mpo->ApplySwap(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        const size_t q1 = qubit0;
        const size_t q2 = qubit1;
        const size_t q3 = qubit2;
        mpo->ApplyCSX(static_cast<int>(q2), static_cast<int>(q3));
        NotifyObservers({qubit1, qubit2});
        mpo->ApplyCX(static_cast<int>(q1), static_cast<int>(q2));
        NotifyObservers({qubit0, qubit1});
        mpo->ApplyCSXDG(static_cast<int>(q2), static_cast<int>(q3));
        NotifyObservers({qubit1, qubit2});
        mpo->ApplyCX(static_cast<int>(q1), static_cast<int>(q2));
        NotifyObservers({qubit0, qubit1});
        mpo->ApplyCSX(static_cast<int>(q1), static_cast<int>(q3));
        NotifyObservers({qubit0, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        const size_t q1 = ctrl_qubit;
        const size_t q2 = qubit0;
        const size_t q3 = qubit1;
        mpo->ApplyCX(static_cast<int>(q3), static_cast<int>(q2));
        NotifyObservers({qubit1, qubit0});
        mpo->ApplyCSX(static_cast<int>(q2), static_cast<int>(q3));
        NotifyObservers({qubit0, qubit1});
        mpo->ApplyCX(static_cast<int>(q1), static_cast<int>(q2));
        NotifyObservers({ctrl_qubit, qubit0});
        mpo->ApplyP(static_cast<int>(q3), M_PI);
        NotifyObservers({qubit1});
        mpo->ApplyP(static_cast<int>(q2), -M_PI_2);
        NotifyObservers({qubit0});
        mpo->ApplyCSX(static_cast<int>(q2), static_cast<int>(q3));
        NotifyObservers({qubit0, qubit1});
        mpo->ApplyCX(static_cast<int>(q1), static_cast<int>(q2));
        NotifyObservers({ctrl_qubit, qubit0});
        mpo->ApplyP(static_cast<int>(q3), M_PI);
        NotifyObservers({qubit1});
        mpo->ApplyCSX(static_cast<int>(q1), static_cast<int>(q3));
        NotifyObservers({ctrl_qubit, qubit1});
        mpo->ApplyCX(static_cast<int>(q3), static_cast<int>(q2));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        mpo->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<GpuMPOSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->lookaheadDepth = lookaheadDepth;
        cloned->lookaheadDepthWithHeuristic = lookaheadDepthWithHeuristic;
        cloned->useOptimalMeetingPosition = useOptimalMeetingPosition;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->growthFactorGate = growthFactorGate;
        cloned->growthFactorSwap = growthFactorSwap;
        cloned->configuration = configuration;
        if (mpo)
        {
            cloned->mpo = mpo->Clone();
            if (!cloned->mpo)
                throw std::runtime_error("GpuSimulator::Clone: Failed to clone matrix product operator "
                                         "state.");
            cloned->gateCounterObserver = std::make_shared<GateCounterObserver>(cloned->upcomingGateIndex);
            cloned->RegisterObserver(cloned->gateCounterObserver);
            cloned->dummySim = dummySim ? dummySim->Clone() : nullptr;
            cloned->curMaxBondDim = curMaxBondDim;
            cloned->mpo->SetCallbackContext(static_cast<GpuTensorChainSimulator *>(cloned.get()));
        }
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::vector<long long> CurrentRoutingMap() const override
    {
        if (mpo)
            return mpo->GetQubitsMap();
        return {};
    }

    void RefreshRoutingCallback() override
    {
        const bool active = IsRoutingLookaheadEnabled() && !upcomingGates.empty();
        if (mpo)
            mpo->SetMeetingPositionCallback(active ? &GpuTensorChainSimulator::FindBestMeetingPosition : nullptr);
    }

    std::unique_ptr<GpuMPO> mpo;
};
} // namespace Simulators::Private
#endif
