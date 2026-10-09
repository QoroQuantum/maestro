#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Core/Factory.h"
#include "GpuTensorChainSimulator.h"

namespace Simulators::Private
{
// matrix_product_state backend. Owns exactly one native Gpu implementation.
class GpuMPSSimulator : public GpuTensorChainSimulator
{
  public:
    GpuMPSSimulator() : GpuTensorChainSimulator(false)
    {
        configuration.SetConfiguration("method", "matrix_product_state");
        configuration.SetConfiguration("matrix_product_state_max_bond_dimension", "128");
    }

    const char *MethodName() const override
    {
        return "matrix_product_state";
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
                mps = SimulatorsFactory::CreateGpuLibMPSSim(gpuDevice);
                if (mps)
                {
                    mps->SetCallbackContext(static_cast<GpuTensorChainSimulator *>(this));
                    curMaxBondDim = 1;
                    mps->InstallBondSummary(&GpuTensorChainSimulator::BondSummaryCallback, &GpuTensorChainSimulator::BondDimCallback);
                    for (const auto &[key, value] : configuration.GetConfigMap())
                        if (key != "method")
                            Configure(key.c_str(), value.c_str());
                    const bool res = mps->Create(nrQubits);
                    if (!res)
                        throw std::runtime_error("GpuState::Initialize: Failed to create "
                                                 "and initialize the MPS state.");
                }
                else
                    throw std::runtime_error("GpuState::Initialize: Failed to create the MPS state.");
                if (!useOptimalMeetingPosition)
                    mps->SetUseOptimalMeetingPosition(false);
            }
            if (GetGpuDevice() != gpuDevice)
                throw std::runtime_error("GpuState::Initialize: GPU plugin did not confirm the requested "
                                         "device; update the GPU library");
        }
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        const bool created = mps->CreateWithBasisState(nrQubits, static_cast<unsigned long long>(basisState));
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
            created = mps->CreateWithBasisStateBits(nrQubits, stateBits);
        }
        if (!created)
            throw std::runtime_error("GpuState::InitializeToBasisState: Failed to initialize the "
                                     "state.");
    }

    void Reset() override
    {
        if (mps)
        {
            mps->Reset();
            curMaxBondDim = 1;
        }
        upcomingGateIndex = 0;
    }

    void SetInitialQubitsMap(const std::vector<long long int> &initialMap) override
    {
        if (mps)
        {
            mps->SetInitialQubitsMap(initialMap);
            if (!dummySim || dummySim->getNrQubits() != initialMap.size() || dummySim->IsOperatorChain() != false)
            {
                dummySim = std::make_unique<Simulators::MPSDummySimulator>(initialMap.size(), false);
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
        if (mps)
            mps->SetUseOptimalMeetingPosition(enable);
        RefreshRoutingCallback();
    }

    void Configure(const char *key, const char *value) override
    {
        if (!key || !value)
            return;
        if (std::string("gpu_device") == key)
        {
            const int device = Configuration::ParseGpuDevice(value);
            if ((mps) && device != Configuration::ParseGpuDevice(configuration.GetConfiguration(key)))
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
            if (svdGroup == "matrix_product_state_use_gesvd")
                applied = gesvd ? applyGesvd(mps) : apply(mps);
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
            if ((mps && mps->IsCreated()))
                return;
            if ((mps && !mps->SetDataType(useDouble)))
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
            if (mps)
                mps->SetSeed(seed);
            return;
        }
        if (std::string("matrix_product_state_truncation_threshold") == key || std::string("matrix_product_operator_truncation_threshold") == key)
        {
            const double singularValueThreshold = std::stod(value);
            if (singularValueThreshold >= 0.)
            {
                if (mps)
                    mps->SetCutoff(singularValueThreshold);
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
                if (mps)
                    mps->SetTruncationMode(truncationMode);
            }
        }
        else if (std::string("matrix_product_state_max_bond_dimension") == key || std::string("matrix_product_operator_max_bond_dimension") == key)
        {
            const long long int chi = std::stoi(value);
            if (chi > 0)
            {
                if (mps)
                    mps->SetMaxExtent(chi);
                if (dummySim)
                    dummySim->SetMaxBondDimension(chi);
            }
        }
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision" || std::string(key) == "use_double_precision")
        {
            const bool useDouble = mps ? mps->IsDoublePrecision() : configuration.GetConfiguration("precision") == "double";
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
                if (svdGroup == "matrix_product_state_use_gesvd" && mps)
                    return readGesvd(mps) ? "true" : "false";
            }
            const char algorithm = std::string(key).back();
            const auto read = [algorithm](const auto &backend) {
                if (algorithm == 'j')
                    return backend->GetGesvdJ();
                if (algorithm == 'p')
                    return backend->GetGesvdP();
                return backend->GetGesvdR();
            };
            if (svdGroup == "matrix_product_state_use_gesvd" && mps)
                return read(mps) ? "true" : "false";
        }
        if (std::string("method") == key)
            return MethodName();
        return configuration.GetConfiguration(key);
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (mps)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        mps = nullptr;
        nrQubits = 0;
        dummySim = nullptr;
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
                if (mps->Measure(static_cast<unsigned int>(qubit)))
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
                res[i] = mps->Measure(static_cast<unsigned int>(qubits[i]));
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
                if (mps->Measure(static_cast<unsigned int>(qubit)))
                    mps->ApplyX(static_cast<unsigned int>(qubit));
        }
        Notify();
        NotifyObservers(qubits);
    }

    std::complex<double> ExpectationValueOperators(const Types::qubits_vector &qubits, const std::vector<Eigen::MatrixXcd> &matrices) override
    {
        if (!mps)
            return IState::ExpectationValueOperators(qubits, matrices);
        TensorQueries::Operators(qubits, matrices, nrQubits);
        std::vector<int> targets;
        std::vector<double> raw;
        targets.reserve(qubits.size());
        raw.reserve(8 * matrices.size());
        for (size_t i = 0; i < qubits.size(); ++i)
        {
            targets.push_back(TensorQueries::Count(qubits[i]));
            for (Eigen::Index e = 0; e < 4; ++e)
            {
                raw.push_back(matrices[i].data()[e].real());
                raw.push_back(matrices[i].data()[e].imag());
            }
        }
        return mps->ExpectationValueOperators(targets, raw);
    }

    void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override
    {
        if (!mps)
            return IState::MoveAtBeginningOfChain(qubits);
        const auto selected = MPOValidation::Qubits(qubits, nrQubits);
        if (selected.empty())
            return;
        mps->MoveAtBeginningOfChain(selected);
    }

    std::vector<std::complex<double>> GetStateVector() override
    {
        if (!mps)
            return IState::GetStateVector();
        return mps->GetStateVector();
    }

    void Trim() override
    {
        if (mps)
        {
            if (!mps->Trim())
                throw std::runtime_error("GPU MPS trim failed");
        }
        else
            throw std::runtime_error("Trim is only available for GPU MPS and MPO");
    }

    void ReCanonicalize() override
    {
        if (mps)
        {
            if (!mps->ReCanonicalize())
                throw std::runtime_error("GPU MPS canonicalization failed");
        }
        else
            throw std::runtime_error("Canonicalization is only available for GPU MPS and MPO");
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        double real = 0.0;
        double imag = 0.0;
        {
            std::vector<long int> fixedValues(nrQubits);
            for (size_t i = 0; i < nrQubits; ++i)
                fixedValues[i] = i < std::numeric_limits<Types::qubit_t>::digits ? (outcome >> i) & 1 : 0;
            {
                if (!mps->Amplitude(nrQubits, fixedValues.data(), &real, &imag) || !std::isfinite(real) || !std::isfinite(imag))
                    throw std::runtime_error("GPU MPS amplitude query failed");
            }
        }
        return std::complex<double>(real, imag);
    }

    std::complex<double> ProjectOnZero() override
    {
        return mps->ProjectOnZero();
    }

    std::vector<double> AllProbabilities() override
    {
        if (nrQubits == 0)
            return {};
        if (mps)
            return mps->AllProbabilities();
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
        if (mps)
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
        if (mps)
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
            const auto counts = mps->SampleHistogram(shots, std::move(selected));
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
        result = mps->ExpectationValue(pauliString);
        return result;
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        if (!mps)
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
        const auto values = mps->ExpectationValues(selected);
        for (size_t i = 0; i < indices.size(); ++i)
            result[indices[i]] = values[i];
        return result;
    }

    int GetGpuDevice() const override
    {
        if (mps)
            return mps->GetGpuDevice();
        return -1;
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kMatrixProductState;
    }

    void SaveState() override
    {
        if (!mps->SaveState())
            throw std::runtime_error("GPU MPS state save failed");
    }

    void RestoreState() override
    {
        if (!mps->RestoreState())
            throw std::runtime_error("GPU MPS state restore failed");
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        const Eigen::Matrix2cd inputFirst = gate.transpose();
        if (!mps->ApplyOneQubitMatrix(static_cast<unsigned int>(qubit), reinterpret_cast<const double *>(inputFirst.data())))
            throw std::runtime_error("GpuSimulator::ApplyGenericOneQubitGate: Failed to apply the "
                                     "generic one-qubit gate on the matrix product state simulator.");
        NotifyObservers({qubit});
        return;
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        const Eigen::Matrix4cd inputFirst = gate.transpose();
        if (!mps->ApplyTwoQubitMatrix(static_cast<unsigned int>(qubit0), static_cast<unsigned int>(qubit1),
                                      reinterpret_cast<const double *>(inputFirst.data())))
            throw std::runtime_error("GpuSimulator::ApplyGenericTwoQubitGate: Failed to apply the "
                                     "generic two-qubit gate on the matrix product state simulator.");
        NotifyObservers({qubit0, qubit1});
        return;
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        mps->ApplyP(qubit, lambda);
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        mps->ApplyX(qubit);
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        mps->ApplyY(qubit);
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        mps->ApplyZ(qubit);
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        mps->ApplyH(qubit);
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        mps->ApplyS(qubit);
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        mps->ApplySDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        mps->ApplyT(qubit);
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        mps->ApplyTDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        mps->ApplySX(qubit);
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        mps->ApplySXDG(qubit);
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        mps->ApplyK(qubit);
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        mps->ApplyRx(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        mps->ApplyRy(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        mps->ApplyRz(qubit, theta);
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        mps->ApplyU(qubit, theta, phi, lambda, gamma);
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mps->ApplyCX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mps->ApplyCY(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mps->ApplyCZ(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        mps->ApplyCP(ctrl_qubit, tgt_qubit, lambda);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        mps->ApplyCRx(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        mps->ApplyCRy(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        mps->ApplyCRz(ctrl_qubit, tgt_qubit, theta);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mps->ApplyCH(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mps->ApplyCSX(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        mps->ApplyCSXDG(ctrl_qubit, tgt_qubit);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        mps->ApplySwap(qubit0, qubit1);
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        const size_t q1 = qubit0;
        const size_t q2 = qubit1;
        const size_t q3 = qubit2;
        mps->ApplyCSX(static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit1, qubit2});
        mps->ApplyCX(static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({qubit0, qubit1});
        mps->ApplyCSXDG(static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit1, qubit2});
        mps->ApplyCX(static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({qubit0, qubit1});
        mps->ApplyCSX(static_cast<unsigned int>(q1), static_cast<unsigned int>(q3));
        NotifyObservers({qubit0, qubit2});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        const size_t q1 = ctrl_qubit;
        const size_t q2 = qubit0;
        const size_t q3 = qubit1;
        mps->ApplyCX(static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit1, qubit0});
        mps->ApplyCSX(static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit0, qubit1});
        mps->ApplyCX(static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({ctrl_qubit, qubit0});
        mps->ApplyP(static_cast<unsigned int>(q3), M_PI);
        NotifyObservers({qubit1});
        mps->ApplyP(static_cast<unsigned int>(q2), -M_PI_2);
        NotifyObservers({qubit0});
        mps->ApplyCSX(static_cast<unsigned int>(q2), static_cast<unsigned int>(q3));
        NotifyObservers({qubit0, qubit1});
        mps->ApplyCX(static_cast<unsigned int>(q1), static_cast<unsigned int>(q2));
        NotifyObservers({ctrl_qubit, qubit0});
        mps->ApplyP(static_cast<unsigned int>(q3), M_PI);
        NotifyObservers({qubit1});
        mps->ApplyCSX(static_cast<unsigned int>(q1), static_cast<unsigned int>(q3));
        NotifyObservers({ctrl_qubit, qubit1});
        mps->ApplyCX(static_cast<unsigned int>(q3), static_cast<unsigned int>(q2));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        mps->ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        auto cloned = std::make_unique<GpuMPSSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->lookaheadDepth = lookaheadDepth;
        cloned->lookaheadDepthWithHeuristic = lookaheadDepthWithHeuristic;
        cloned->useOptimalMeetingPosition = useOptimalMeetingPosition;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->growthFactorGate = growthFactorGate;
        cloned->growthFactorSwap = growthFactorSwap;
        cloned->configuration = configuration;
        if (mps)
        {
            cloned->mps = mps->Clone();
            cloned->gateCounterObserver = std::make_shared<GateCounterObserver>(cloned->upcomingGateIndex);
            cloned->RegisterObserver(cloned->gateCounterObserver);
            cloned->dummySim = dummySim ? dummySim->Clone() : nullptr;
            cloned->curMaxBondDim = curMaxBondDim;
            cloned->mps->SetCallbackContext(static_cast<GpuTensorChainSimulator *>(cloned.get()));
        }
        if (configuration.IsSet("seed"))
            cloned->SetSeed(DeriveSeed(std::stoull(configuration.GetConfiguration("seed")), nextSeedStream++));
        return cloned;
    }

  protected:
    std::vector<long long> CurrentRoutingMap() const override
    {
        if (mps)
            return mps->GetQubitsMap();
        return {};
    }

    void RefreshRoutingCallback() override
    {
        const bool active = IsRoutingLookaheadEnabled() && !upcomingGates.empty();
        if (mps)
            mps->SetMeetingPositionCallback(active ? &GpuTensorChainSimulator::FindBestMeetingPosition : nullptr);
    }

    std::unique_ptr<GpuLibMPSSim> mps;
};
} // namespace Simulators::Private
#endif
