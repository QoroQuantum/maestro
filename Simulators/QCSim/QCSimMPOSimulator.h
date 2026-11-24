#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "MPOSimulator.h"
#include "QCSimTensorChainSimulator.h"
#define _USE_MATH_DEFINES
#include <math.h>

namespace Simulators::Private
{
// matrix_product_operator backend. Owns exactly one native QCSim implementation.
class QCSimMPOSimulator : public QCSimTensorChainSimulator
{
  public:
    QCSimMPOSimulator() : QCSimTensorChainSimulator(true)
    {
        configuration.SetConfiguration("method", "matrix_product_operator");
    }

    const char *MethodName() const override
    {
        return "matrix_product_operator";
    }

    void Initialize() override
    {
        InvalidateBoundedRouting(true);
        RefreshRoutingSettings();
        InvalidateDummyHistory();
        ClearSavedRoutingState();
        ResetLiveSiteRanks();
        pendingDeflate = {};
        savedPendingDeflate.reset();
        if (nrQubits != 0)
        {
            InitializeSamplingSeed();
            {
                mpoSimulator = std::make_unique<QC::TensorNetworks::MPOSimulator>(nrQubits, Utils::RandomStream::FreshNativeSeed());
                const double threshold = std::stod(GetConfiguration("matrix_product_state_truncation_threshold"));
                mpoSimulator->setLimitEntanglement(threshold);
                cutoffEnabled = threshold > 0.;
                bondLimitEnabled = false;
                ForgetTruncationHistory();
                if (!useOptimalMeetingPosition)
                    mpoSimulator->SetUseOptimalMeetingPosition(false);
                InstallBondSummary(*mpoSimulator, bondDimensionCallback);
                curMaxBondDim = 1;
            }
            SeedBackend(mpoSimulator.get(), samplingSeed);
            SetMultithreading(enableMultithreading);
            for (const auto &[key, value] : configuration.GetConfigMap())
                if (key != "method")
                    Configure(key.c_str(), value.c_str());
        }
        RefreshRoutingCallback();
    }

    void InitializeToBasisState(size_t num_qubits, Types::qubit_t basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        mpoSimulator->setToBasisState(static_cast<size_t>(basisState));
    }

    void InitializeToBasisState(size_t num_qubits, const std::vector<bool> &basisState) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        mpoSimulator->setToBasisState(basisState);
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<Types::qubit_t, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        std::vector<std::pair<size_t, double>> converted;
        converted.reserve(mixture.size());
        for (const auto &[basisState, weight] : mixture)
            converted.emplace_back(static_cast<size_t>(basisState), weight);
        NotePossibleTruncation();
        mpoSimulator->setToMixtureOfBasisStates(converted);
        liveSiteRank.assign(num_qubits, mixture.size() > 1 ? 2. : 1.);
        if (mixture.size() > 1)
            MarkAllTouched(); // a mixture of basis states correlates its qubits
    }

    void InitializeToMixtureOfBasisStates(size_t num_qubits, const std::vector<std::pair<std::vector<bool>, double>> &mixture) override
    {
        if (num_qubits == 0)
            return;
        Clear();
        nrQubits = num_qubits;
        Initialize();
        NotePossibleTruncation();
        mpoSimulator->setToMixtureOfBasisStates(mixture);
        liveSiteRank.assign(num_qubits, mixture.size() > 1 ? 2. : 1.);
        if (mixture.size() > 1)
            MarkAllTouched(); // a mixture of basis states correlates its qubits
    }

    void Reset() override
    {
        if (mpoSimulator)
        {
            mpoSimulator->Clear();
            curMaxBondDim = 1;
            stateMayHaveTruncated = false;
        }
        upcomingGateIndex = 0;
        pendingDeflate = {};
        ResetDummySimulator();
    }

    void SetInitialQubitsMap(const std::vector<long long int> &initialMap) override
    {
        if (mpoSimulator && !RelabelQubitsMap(*mpoSimulator, initialMap))
            pendingDeflate = {};
    }

    void SetUseOptimalMeetingPosition(bool enable) override
    {
        useOptimalMeetingPosition = enable;
        RefreshRoutingSettings();
        if (mpoSimulator)
            mpoSimulator->SetUseOptimalMeetingPosition(enable);
        RefreshRoutingCallback();
    }

    void Configure(const char *key, const char *value) override
    {
        if (std::string("method") == key)
        {
            ValidateMethod(value);
            configuration.SetConfiguration(key, value);
            return;
        }
        if ((std::string(key) == "matrix_product_state_max_bond_dimension" || std::string(key) == "matrix_product_operator_max_bond_dimension"))
        {
            const auto limit = std::stoll(value);
            if (limit < 0)
                throw std::invalid_argument("Negative MPO bond dimension");
            bondLimitEnabled = limit > 0;
            if (mpoSimulator)
            {
                if (limit == 0)
                    mpoSimulator->dontLimitBondDimension();
                else
                {
                    mpoSimulator->setLimitBondDimension(limit);
                }
            }
            const std::string limitValue(value);
            configuration.SetConfiguration("matrix_product_state_max_bond_dimension", limitValue);
            configuration.SetConfiguration("matrix_product_operator_max_bond_dimension", limitValue);
            RefreshRoutingSettings();
            RefreshRoutingCallback();
            return;
        }
        if (!configuration.WasApplied(key, value))
            configuration.SetConfiguration(key, value);
        RefreshRoutingSettings();
        if (std::string("seed") == key)
        {
            SeedForExecution(std::stoull(value));
            return;
        }
        if (mpoSimulator)
        {
            if (Configuration::CanonicalKey(key) == "matrix_product_state_truncation_threshold")
            {
                const double threshold = configuration.GetConfigurationAsDouble(key);
                if (threshold >= 0.)
                {
                    mpoSimulator->setLimitEntanglement(threshold);
                    cutoffEnabled = threshold > 0.;
                }
            }
            else if (std::string(key) == "matrix_product_state_truncation_mode" || std::string(key) == "matrix_product_operator_truncation_mode")
            {
                if (std::string(value) == "relative_max")
                    mpoSimulator->setTruncationMode(QC::TensorNetworks::MPOSimulator::TruncationMode::RelativeToMax);
                else if (std::string(value) == "discarded_weight")
                    mpoSimulator->setTruncationMode(QC::TensorNetworks::MPOSimulator::TruncationMode::DiscardedWeight);
            }
            else if (std::string(key) == "matrix_product_operator_kraus_completeness_check")
            {
                using Check = QC::TensorNetworks::MPOSimulator::KrausCompletenessCheck;
                if (std::string(value) == "ignore")
                    mpoSimulator->setKrausCompletenessCheck(Check::Ignore);
                else if (std::string(value) == "warn")
                    mpoSimulator->setKrausCompletenessCheck(Check::Warn);
                else if (std::string(value) == "strict")
                    mpoSimulator->setKrausCompletenessCheck(Check::Strict);
            }
            else if (std::string(key) == "matrix_product_operator_restore_trace_after_truncation")
            {
                mpoSimulator->setRestoreTraceAfterTruncation(std::string(value) == "1" || std::string(value) == "true");
            }
            else if (std::string(key) == "matrix_product_operator_hermitize_after_truncation")
            {
                mpoSimulator->setHermitizeAfterTruncation(std::string(value) == "1" || std::string(value) == "true");
            }
        }
        if (std::string(key) == "pauli_propagator_workers")
        {
            const auto workers = configuration.GetConfigurationAsUnsigned(key);
            if (workers > 1024)
                throw std::invalid_argument("pauli_propagator_workers exceeds 1024");
        }
        RefreshRoutingCallback();
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (!key)
            return {};
        if (std::string(key) == "precision")
            return "double";
        if (std::string(key) == "use_double_precision")
            return "1";
        if (mpoSimulator)
        {
            if (std::string(key) == "matrix_product_operator_restore_trace_after_truncation")
                return mpoSimulator->getRestoreTraceAfterTruncation() ? "true" : "false";
            if (std::string(key) == "matrix_product_operator_hermitize_after_truncation")
                return mpoSimulator->getHermitizeAfterTruncation() ? "true" : "false";
        }
        if (std::string("method") == key)
            return MethodName();
        return QCSimTensorChainSimulator::GetConfiguration(key);
    }

    void SeedForExecution(uint64_t seed) override
    {
        SeedAuxiliaryRng(seed);
        SeedSampling(seed);
        if (mpoSimulator)
            SeedBackend(mpoSimulator.get(), seed);
    }

    size_t AllocateQubits(size_t num_qubits) override
    {
        if (mpoSimulator)
            return 0;
        const size_t oldNrQubits = nrQubits;
        nrQubits += num_qubits;
        return oldNrQubits;
    }

    void Clear() override
    {
        automaticRoutingDepth = false;
        InvalidateBoundedRouting(true);
        mpoSimulator = nullptr;
        ForgetTruncationHistory();
        ClearDummySimulator();
        pendingDeflate = {};
        savedPendingDeflate.reset();
        nrQubits = 0;
        upcomingGateIndex = 0;
        upcomingGates.clear();
    }

    size_t Measure(const Types::qubits_vector &qubits) override
    {
        if (mpoSimulator && qubits.size() > sizeof(size_t) * 8)
            throw std::invalid_argument("Use MeasureMany for more than 64 measured qubits");
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        size_t res = 0;
        size_t mask = 1ULL;
        DontNotify();
        {
            const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
            const auto measured = mpoSimulator->MeasureQubits(qubitsSet);
            for (Types::qubit_t qubit : qubits)
            {
                if (measured.at(static_cast<Eigen::Index>(qubit)))
                    res |= mask;
                mask <<= 1;
            }
        }
        NotePendingDisentangle(qubits);
        RecordDisentangledQubits(qubits);
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override
    {
        std::vector<bool> res(qubits.size(), false);
        DontNotify();
        {
            const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
            const auto measured = mpoSimulator->MeasureQubits(qubitsSet);
            for (size_t q = 0; q < qubits.size(); ++q)
                res[q] = measured.at(static_cast<Eigen::Index>(qubits[q]));
        }
        NotePendingDisentangle(qubits);
        RecordDisentangledQubits(qubits);
        Notify();
        NotifyObservers(qubits);
        return res;
    }

    void ApplyReset(const Types::qubits_vector &qubits) override
    {
        DontNotify();
        {
            for (size_t qubit : qubits)
                mpoSimulator->ApplyReset(static_cast<Eigen::Index>(qubit));
        }
        NotePendingDisentangle(qubits);
        RecordDisentangledQubits(qubits);
        Notify();
        NotifyObservers(qubits);
    }

    void ApplyResetToTargets(const Types::qubits_vector &qubits, const std::vector<bool> &resetTargets) override
    {
        DontNotify();
        {
            for (const auto qubit : qubits)
                mpoSimulator->ApplyReset(static_cast<Eigen::Index>(qubit));
            for (size_t qi = 0; qi < std::min(qubits.size(), resetTargets.size()); ++qi)
                if (resetTargets[qi])
                    mpoSimulator->ApplyGate(xgate, static_cast<Eigen::Index>(qubits[qi]));
        }
        NotePendingDisentangle(qubits);
        RecordDisentangledQubits(qubits);
        Notify();
        NotifyObservers(qubits);
    }

    bool SupportsQuantumChannels() const override
    {
        return true;
    }

    bool SupportsSuperoperators() const override
    {
        return true;
    }

    /**
     * @brief Applies a local map on one or two qubits, in superoperator form.
     *
     * The fusion layer hands a gate, its noise and the gates around them on
     * the same one or two qubits over as one map: one update of the chain,
     * and one decomposition, where a gate followed by a two-qubit channel
     * costs two. The matrix is trusted, as on the density matrix.
     */
    void ApplySuperoperator(const Types::qubits_vector &targets, const Eigen::MatrixXcd &superoperator) override
    {
        if (targets.empty() || targets.size() > 2 || (targets.size() == 2 && targets[0] == targets[1]))
            throw std::invalid_argument("A superoperator acts on one or two distinct qubits");
        const Eigen::Index dimension = Eigen::Index{1} << (2 * targets.size());
        if (superoperator.rows() != dimension || superoperator.cols() != dimension)
            throw std::invalid_argument("The superoperator does not match its targets");
        for (const Types::qubit_t target : targets)
            if (target >= nrQubits)
                throw std::invalid_argument("Superoperator qubit is out of range");
        if (!mpoSimulator)
            throw std::runtime_error("QCSim MPO state is not initialized");
        // As for a channel: the native update can route before it finishes.
        InvalidateDummyHistory();
        const auto mapBefore = targets.size() == 2 ? PendingDeflateMap() : std::vector<long long>{};
        if (targets.size() == 2)
            NotePossibleTruncation();
        mpoSimulator->ApplySuperoperator(superoperator, static_cast<Eigen::Index>(targets[0]),
                                         targets.size() > 1 ? static_cast<Eigen::Index>(targets[1]) : Eigen::Index{0});
        if (targets.size() == 2)
        {
            ClearSpanPendingDeflate(mapBefore, targets[0], targets[1]);
            NoteTwoQubitNonUnitaryTouch(targets[0], targets[1]);
        }
        InvalidateDummyHistory();
        NotifyObservers(targets);
    }

    void ApplyQuantumChannel(const Types::qubits_vector &targets, const QuantumChannel &channel) override
    {
        if (!SupportsQuantumChannels())
            throw std::runtime_error("QCSim quantum channels require density_matrix or "
                                     "matrix_product_operator simulation");
        if (targets.size() != channel.GetNumberOfQubits())
            throw std::invalid_argument("The number of channel targets does not match its Kraus operators");
        if (targets.empty() || targets.size() > 2)
            throw std::invalid_argument("QCSim supports only one- and two-qubit local channels");
        std::unordered_set<Types::qubit_t> uniqueTargets;
        for (const Types::qubit_t target : targets)
        {
            if (target >= nrQubits)
                throw std::invalid_argument("Quantum-channel qubit is out of range");
            if (!uniqueTargets.insert(target).second)
                throw std::invalid_argument("Quantum-channel target qubits must be distinct");
        }
        const auto &krausOperators = channel.GetKrausOperators();
        // Native channel/operator application can invoke routing before it
        // finishes; discard old estimates both before and after that call.
        InvalidateDummyHistory();
        {
            if (!mpoSimulator)
                throw std::runtime_error("QCSim MPO state is not initialized");
            if (targets.size() == 1)
                mpoSimulator->ApplyKrausOperators(krausOperators, static_cast<Eigen::Index>(targets[0]));
            else
            {
                const auto mapBefore = PendingDeflateMap();
                NotePossibleTruncation();
                mpoSimulator->ApplyKrausOperators(krausOperators, static_cast<Eigen::Index>(targets[0]), static_cast<Eigen::Index>(targets[1]));
                ClearSpanPendingDeflate(mapBefore, targets[0], targets[1]);
                NoteTwoQubitNonUnitaryTouch(targets[0], targets[1]);
            }
        }
        InvalidateDummyHistory();
        NotifyObservers(targets);
    }

    double ProbabilityBits(const std::vector<bool> &bits) override
    {
        if (!mpoSimulator)
            return IState::ProbabilityBits(bits);
        MPOValidation::Bits(bits, nrQubits);
        return mpoSimulator->getBasisStateProbability(bits);
    }

    std::complex<double> DensityMatrixElementBits(const std::vector<bool> &row, const std::vector<bool> &col) const override
    {
        if (!mpoSimulator)
            return IState::DensityMatrixElementBits(row, col);
        MPOValidation::Bits(row, nrQubits);
        MPOValidation::Bits(col, nrQubits);
        return mpoSimulator->getBasisStateMatrixElement(row, col);
    }

    Eigen::MatrixXcd GetDensityMatrix(bool normalized = true) const override
    {
        if (!mpoSimulator)
            return IState::GetDensityMatrix(normalized);
        if (nrQubits > 13)
            throw std::invalid_argument("MPO dense output supports at most 13 qubits");
        return normalized ? mpoSimulator->getDensityMatrix() : mpoSimulator->getUnnormalizedDensityMatrix();
    }

    std::complex<double> ExpectationValueComplex(const std::string &pauli, bool normalized = true) const override
    {
        if (!mpoSimulator)
            return IState::ExpectationValueComplex(pauli, normalized);
        MPOValidation::Pauli(pauli, nrQubits);
        return normalized ? mpoSimulator->ExpectationValue(pauli) : mpoSimulator->UnnormalizedExpectationValue(pauli);
    }

    std::vector<std::complex<double>> ExpectationValuesComplex(const std::vector<std::string> &paulis, bool normalized = true) const override
    {
        if (!mpoSimulator)
            return IState::ExpectationValuesComplex(paulis, normalized);
        for (const auto &pauli : paulis)
            MPOValidation::Pauli(pauli, nrQubits);
        return ComplexTensorExpectationBatch(*mpoSimulator, paulis, normalized);
    }

    void ApplyOperator(const Types::qubits_vector &qubits, const Eigen::MatrixXcd &matrix, bool normalize = false) override
    {
        if (!mpoSimulator)
            return IState::ApplyOperator(qubits, matrix, normalize);
        const auto selected = MPOValidation::Operator(qubits, matrix, nrQubits);
        if (!normalize && qubits.size() == 2 && IsExactSwapMatrix(matrix))
            return ApplySwap(qubits[0], qubits[1]);
        const QC::Gates::AppliedGate<> op(matrix, selected[0], selected.size() == 2 ? selected[1] : 0);
        // Native channel/operator application can invoke routing before it
        // finishes; discard old estimates both before and after that call.
        InvalidateDummyHistory();
        const auto mapBefore = qubits.size() == 2 ? PendingDeflateMap() : std::vector<long long>{};
        if (qubits.size() == 2)
            NotePossibleTruncation();
        if (normalize)
            mpoSimulator->ApplyOperatorAndNormalize(op);
        else
            mpoSimulator->ApplyOperator(op);
        if (qubits.size() == 2)
        {
            ClearSpanPendingDeflate(mapBefore, qubits[0], qubits[1]);
            NoteTwoQubitNonUnitaryTouch(qubits[0], qubits[1]);
        }
        InvalidateDummyHistory();
        NotifyObservers(qubits);
    }

    void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override
    {
        if (!mpoSimulator)
            return IState::MoveAtBeginningOfChain(qubits);
        const auto selected = MPOValidation::Qubits(qubits, nrQubits);
        const std::set<Eigen::Index> targets(selected.begin(), selected.end());
        const auto before = PendingDeflateMap();
        mpoSimulator->MoveAtBeginningOfChain(targets);
        if (!before.empty())
        {
            const auto after = CurrentRoutingMap();
            if (before == after)
                return;
            // QCSim stably partitions the chain. Every moved site crosses at
            // least one site of the other group; sites that stayed put cross none.
            RetirePendingDeflation([&](size_t q) { return before[q] != after[q]; });
        }
        NotePossibleTruncation();
        InvalidateDummyHistory();
    }

    std::complex<double> DensityMatrixTrace() const override
    {
        if (mpoSimulator)
            return mpoSimulator->Trace();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    double DensityMatrixPurity() const override
    {
        if (mpoSimulator)
            return mpoSimulator->Purity();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        if (mpoSimulator)
            return mpoSimulator->TraceOfSquare();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        const auto *rhs = dynamic_cast<const QCSimMPOSimulator *>(&other);
        if (!rhs)
            throw std::invalid_argument("Density-matrix overlap requires matching QCSim backends");
        if (mpoSimulator && rhs->mpoSimulator)
            return mpoSimulator->HilbertSchmidtOverlap(*rhs->mpoSimulator);
        throw std::invalid_argument("Density-matrix overlap requires two density matrices or two MPOs");
    }

    double DensityMatrixHermiticityResidual() const override
    {
        if (mpoSimulator)
            return mpoSimulator->HermiticityResidual();
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        if (mpoSimulator)
            return mpoSimulator->IsHermitian(eps);
        throw std::runtime_error("Mixed-state diagnostics require density_matrix or "
                                 "matrix_product_operator");
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qubits) const override
    {
        if (mpoSimulator)
            return mpoSimulator->PartialTrace(std::vector<Eigen::Index>(qubits.begin(), qubits.end()));
        throw std::runtime_error("Partial trace requires density_matrix or matrix_product_operator");
    }

    double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override
    {
        if (mpoSimulator)
            return mpoSimulator->FidelityWithStatevector(psi);
        throw std::runtime_error("Mixed-state fidelity requires density_matrix or "
                                 "matrix_product_operator");
    }

    void RestoreDensityMatrixTrace() override
    {
        if (!mpoSimulator)
            throw std::runtime_error("Trace restoration is only available for QCSim MPO");
        mpoSimulator->RestoreTrace();
    }

    void HermitizeDensityMatrix() override
    {
        if (!mpoSimulator)
            throw std::runtime_error("Hermitization is only available for QCSim MPO");
        NotePossibleTruncation();
        mpoSimulator->Hermitize();
        pendingDeflate = {};
        InvalidateDummyHistory();
    }

    void Trim() override
    {
        InvalidateDummyHistory();
        if (mpoSimulator)
        {
            NotePossibleTruncation();
            mpoSimulator->Trim();
            // Initialize installs QCSim's singular-value threshold even at zero,
            // so its compression pass visits every bond.
            pendingDeflate = {};
        }
        else
            throw std::runtime_error("Trim is only available for QCSim MPS and MPO");
    }

    void ReCanonicalize() override
    {
        InvalidateDummyHistory();
        if (mpoSimulator)
        {
            mpoSimulator->ReCanonicalize();
            pendingDeflate = {};
        }
        else
            throw std::runtime_error("Canonicalization is only available for QCSim MPS and MPO");
    }

    double Probability(Types::qubit_t outcome) override
    {
        RequireBasisState(outcome);
        std::vector<bool> bits(nrQubits);
        for (size_t q = 0; q < std::min(nrQubits, sizeof(Types::qubit_t) * 8); ++q)
            bits[q] = ((outcome >> q) & 1) != 0;
        return ProbabilityBits(bits);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        throw std::runtime_error("QCSimState::Amplitude: Amplitudes are not defined for the matrix "
                                 "product operator simulator.");
    }

    // The operator reports its probabilities normalized by its trace (ApplyOperator without normalize leaves the
    // trace where it falls), so they total 1 wherever they are defined. Where the trace cannot normalize them they
    // all throw, and so does this: it asks the probability of |0...0>, one contraction at any width.
    double TotalProbability() override
    {
        RequireQubits(GetNumberOfQubits());
        (void)Probability(0);
        return 1.;
    }

    std::vector<double> AllProbabilities() override
    {
        RequireQubits(GetNumberOfQubits()); // a register without qubits has no state to list
        const size_t nrBasisStates = CheckedBasisStateCountForQueries();
        std::vector<double> result(nrBasisStates);
        for (size_t i = 0; i < nrBasisStates; ++i)
            result[i] = mpoSimulator->getBasisStateProbability(i);
        return result;
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        if (mpoSimulator && qubits.size() > sizeof(Types::qubit_t) * 8)
            throw std::invalid_argument("Use SampleCountsMany for more than 64 measured qubits");
        if (mpoSimulator)
            for (auto q : qubits)
                if (q >= nrQubits)
                    throw std::out_of_range("MPO sampled qubit is out of range");
        if (qubits.empty() || shots == 0)
            return {};
        if (qubits.size() > sizeof(size_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the size_t type, the outcome will be undefined"
                      << std::endl;
        std::unordered_map<Types::qubit_t, Types::qubit_t> result;
        if (TrySingleQubitCounts<false>(qubits, shots, result))
            return result;
        DontNotify();
        {
            const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto measured = mpoSimulator->MeasureNoCollapse(qubitsSet);
                Types::qubit_t packed = 0;
                for (size_t i = 0; i < qubits.size(); ++i)
                    if (measured.at(static_cast<Eigen::Index>(qubits[i])))
                        packed |= Types::BitAt(i);
                ++result[packed];
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qubits, size_t shots = 1000) override
    {
        // Accumulate narrow outcomes as integers and materialize each distinct
        // bit vector once. Native MPO sampling keeps the same per-shot RNG
        // sequence, without allocating/hashing a bit vector for every draw.
        if (qubits.size() <= sizeof(Types::qubit_t) * 8)
        {
            const auto packedCounts = SampleCounts(qubits, shots);
            std::unordered_map<std::vector<bool>, Types::qubit_t> result;
            result.reserve(packedCounts.size());
            for (const auto &[outcome, count] : packedCounts)
            {
                std::vector<bool> bits(qubits.size());
                for (size_t i = 0; i < qubits.size(); ++i)
                    bits[i] = (outcome & Types::BitAt(i)) != 0;
                result.emplace(std::move(bits), count);
            }
            return result;
        }
        if (mpoSimulator)
            for (auto q : qubits)
                if (q >= nrQubits)
                    throw std::out_of_range("MPO sampled qubit is out of range");
        if (qubits.empty() || shots == 0)
            return {};
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        if (TrySingleQubitCounts<true>(qubits, shots, result))
            return result;
        DontNotify();
        {
            const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
            std::vector<bool> packed(qubits.size(), false);
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto measured = mpoSimulator->MeasureNoCollapse(qubitsSet);
                for (size_t i = 0; i < qubits.size(); ++i)
                    packed[i] = measured.at(static_cast<Eigen::Index>(qubits[i]));
                ++result[packed];
            }
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        std::vector<double> result(paulis.size(), 1.);
        std::vector<std::string> selected;
        std::vector<size_t> indices;
        const size_t n = GetNumberOfQubits();
        for (size_t i = 0; i < paulis.size(); ++i)
        {
            if (paulis[i].empty())
                continue;
            bool zero = false;
            for (size_t q = n; q < paulis[i].size(); ++q)
            {
                const auto p = toupper(static_cast<unsigned char>(paulis[i][q]));
                if (p != 'I' && p != 'Z')
                {
                    zero = true;
                    break;
                }
            }
            if (zero)
            {
                result[i] = 0.;
                continue;
            }
            std::string pauli = paulis[i];
            pauli.resize(n, 'I');
            for (char &p : pauli)
            {
                p = static_cast<char>(toupper(static_cast<unsigned char>(p)));
            }
            indices.push_back(i);
            selected.push_back(std::move(pauli));
        }
        if (selected.empty())
            return result;
        const auto values = TensorExpectationBatch(*mpoSimulator, selected);
        for (size_t i = 0; i < indices.size(); ++i)
            result[indices[i]] = values[i].real();
        return result;
    }

    double ExpectationValue(const std::string &pauliStringOrig) override
    {
        if (pauliStringOrig.empty())
            return 1.0;
        std::string pauliString = pauliStringOrig;
        if (pauliString.size() > GetNumberOfQubits())
        {
            for (size_t i = GetNumberOfQubits(); i < pauliString.size(); ++i)
            {
                const auto pauliOp = toupper(pauliString[i]);
                if (pauliOp != 'I' && pauliOp != 'Z')
                    return 0.0;
            }
            pauliString.resize(GetNumberOfQubits());
        }
        {
            pauliString.resize(GetNumberOfQubits(), 'I');
            return mpoSimulator->ExpectationValue(pauliString).real();
        }
    }

    SimulationType GetSimulationType() const override
    {
        return SimulationType::kMatrixProductOperator;
    }

    void SaveState() override
    {
        mpoSimulator->SaveState();
        savedStateMayHaveTruncated = stateMayHaveTruncated;
        savedPendingDeflate = pendingDeflate;
        SaveDummyState();
    }

    void RestoreState() override
    {
        mpoSimulator->RestoreState();
        if (savedStateMayHaveTruncated)
            stateMayHaveTruncated = *savedStateMayHaveTruncated;
        if (savedPendingDeflate)
            pendingDeflate = *savedPendingDeflate;
        RestoreDummyState();
    }

    void SetMultithreading(bool multithreading = true) override
    {
        enableMultithreading = multithreading;
        if (mpoSimulator)
            mpoSimulator->SetMultithreading(multithreading);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        if (mpoSimulator && nrQubits > sizeof(Types::qubit_t) * 8)
            throw std::invalid_argument("Use MeasureNoCollapseMany for more than 64 qubits");
        if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
            std::cerr << "Warning: The number of qubits to measure is larger than the "
                         "number of bits in the Types::qubit_t type, the outcome will be "
                         "undefined"
                      << std::endl;
        {
            const auto measured = mpoSimulator->MeasureNoCollapse();
            Types::qubit_t result = 0;
            for (size_t qubit = 0; qubit < nrQubits; ++qubit)
                if (measured.at(static_cast<Eigen::Index>(qubit)))
                    result |= Types::BitAt(qubit);
            return result;
        }
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        const auto measured = mpoSimulator->MeasureNoCollapse();
        std::vector<bool> res(nrQubits);
        for (size_t i = 0; i < nrQubits; ++i)
            res[i] = measured.at(static_cast<Eigen::Index>(i));
        return res;
    }

    void ApplyGenericOneQubitGate(Types::qubit_t qubit, const Eigen::Matrix2cd &gate) override
    {
        const QC::Gates::AppliedGate<> agate(gate, qubit);
        ApplyNativeGate(agate);
        NotifyObservers({qubit});
    }

    void ApplyGenericTwoQubitGate(Types::qubit_t qubit0, Types::qubit_t qubit1, const Eigen::Matrix4cd &gate) override
    {
        if (IsExactSwapMatrix(gate))
            return ApplySwap(qubit0, qubit1);
        const QC::Gates::AppliedGate<> agate(gate, qubit0, qubit1);
        ApplyNativeGate(agate);
        NotifyObservers({qubit0, qubit1});
    }

    void ApplyP(Types::qubit_t qubit, double lambda) override
    {
        pgate.SetPhaseShift(lambda);
        ApplyNativeGate(pgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyX(Types::qubit_t qubit) override
    {
        ApplyNativeGate(xgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyY(Types::qubit_t qubit) override
    {
        ApplyNativeGate(ygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyZ(Types::qubit_t qubit) override
    {
        ApplyNativeGate(zgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyH(Types::qubit_t qubit) override
    {
        ApplyNativeGate(h, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyS(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySDG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyT(Types::qubit_t qubit) override
    {
        ApplyNativeGate(tgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyTDG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(tdggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySx(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplySxDAG(Types::qubit_t qubit) override
    {
        ApplyNativeGate(sxdaggate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyK(Types::qubit_t qubit) override
    {
        ApplyNativeGate(k, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRx(Types::qubit_t qubit, double theta) override
    {
        rxgate.SetTheta(theta);
        ApplyNativeGate(rxgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRy(Types::qubit_t qubit, double theta) override
    {
        rygate.SetTheta(theta);
        ApplyNativeGate(rygate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyRz(Types::qubit_t qubit, double theta) override
    {
        rzgate.SetTheta(theta);
        ApplyNativeGate(rzgate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda, double gamma) override
    {
        ugate.SetParams(theta, phi, lambda, gamma);
        ApplyNativeGate(ugate, static_cast<unsigned int>(qubit));
        NotifyObservers({qubit});
    }

    void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(cxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(cygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(czgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double lambda) override
    {
        cpgate.SetPhaseShift(lambda);
        ApplyNativeGate(cpgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crxgate.SetTheta(theta);
        ApplyNativeGate(crxgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crygate.SetTheta(theta);
        ApplyNativeGate(crygate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta) override
    {
        crzgate.SetTheta(theta);
        ApplyNativeGate(crzgate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(ch, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(csx, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplyCSxDAG(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override
    {
        ApplyNativeGate(csxdag, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        ApplyNativeGate(swapgate, static_cast<unsigned int>(qubit1), static_cast<unsigned int>(qubit0));
        NotifyObservers({qubit1, qubit0});
    }

    void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1, Types::qubit_t qubit2) override
    {
        ApplyDecomposedGate(
            [&]() {
                ApplyCSx(qubit1, qubit2);
                ApplyCX(qubit0, qubit1);
                ApplyCSxDAG(qubit1, qubit2);
                ApplyCX(qubit0, qubit1);
                ApplyCSx(qubit0, qubit2);
            },
            {qubit2, qubit1, qubit0});
    }

    void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0, Types::qubit_t qubit1) override
    {
        ApplyDecomposedGate(
            [&]() {
                ApplyCX(qubit1, qubit0);
                ApplyCSx(qubit0, qubit1);
                ApplyCX(ctrl_qubit, qubit0);
                ApplyP(qubit1, M_PI);
                ApplyP(qubit0, -M_PI_2);
                ApplyCSx(qubit0, qubit1);
                ApplyCX(ctrl_qubit, qubit0);
                ApplyP(qubit1, M_PI);
                ApplyCSx(ctrl_qubit, qubit1);
                ApplyCX(qubit1, qubit0);
            },
            {qubit1, qubit0, ctrl_qubit});
    }

    void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit, double theta, double phi, double lambda, double gamma) override
    {
        cugate.SetParams(theta, phi, lambda, gamma);
        ApplyNativeGate(cugate, static_cast<unsigned int>(tgt_qubit), static_cast<unsigned int>(ctrl_qubit));
        NotifyObservers({tgt_qubit, ctrl_qubit});
    }

    std::unique_ptr<ISimulator> Clone() override
    {
        return CloneForExecution(NextCloneSeed());
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        auto cloned = std::make_unique<QCSimMPOSimulator>();
        cloned->nrQubits = nrQubits;
        cloned->enableMultithreading = enableMultithreading;
        cloned->lookaheadDepth = lookaheadDepth;
        cloned->automaticRoutingDepth = automaticRoutingDepth;
        cloned->lookaheadDepthWithHeuristic = lookaheadDepthWithHeuristic;
        cloned->useOptimalMeetingPosition = useOptimalMeetingPosition;
        cloned->upcomingGates = upcomingGates;
        cloned->upcomingGateIndex = upcomingGateIndex;
        cloned->boundedRoutingOperations = boundedRoutingOperations;
        cloned->boundedRoutingOperationsQubits = boundedRoutingOperationsQubits;
        cloned->growthFactorGate = growthFactorGate;
        cloned->growthFactorSwap = growthFactorSwap;
        if (mpoSimulator)
        {
            cloned->mpoSimulator = mpoSimulator->Clone();
            cloned->curMaxBondDim = curMaxBondDim;
            cloned->gateCounterObserver = std::make_shared<GateCounterObserver>(cloned->upcomingGateIndex);
            cloned->RegisterObserver(cloned->gateCounterObserver);
            cloned->mpoSimulator->SetMeetingPositionCallback(cloned->meetingPositionCallback);
            InstallBondSummary(*cloned->mpoSimulator, cloned->bondDimensionCallback);
        }
        for (const auto &[key, value] : configuration.GetConfigMap())
            cloned->Configure(key.c_str(), value.c_str());
        cloned->cutoffEnabled = cutoffEnabled;
        cloned->bondLimitEnabled = bondLimitEnabled;
        cloned->stateMayHaveTruncated = stateMayHaveTruncated;
        cloned->savedStateMayHaveTruncated = savedStateMayHaveTruncated;
        cloned->upcomingHasDisentangle = upcomingHasDisentangle;
        cloned->RefreshRoutingSettings();
        CopyRoutingHistoryTo(*cloned);
        cloned->pendingDeflate = pendingDeflate;
        cloned->savedPendingDeflate = savedPendingDeflate;
        if (configuration.IsSet("seed"))
            cloned->SetSeed(seed);
        else
            cloned->SeedForExecution(seed);
        cloned->RefreshRoutingCallback();
        return cloned;
    }

    bool CanSampleSingleQubitMarginal() const override
    {
        return mpoSimulator && !stateMayHaveTruncated;
    }

  protected:
    // P(q = 0) = Tr(rho P0_q) / Tr(rho): two traced chain contractions, O(n chi^2)
    // with one chi x chi site matrix of scratch at a time. The native sampler
    // conditions every shot along the chain up to the measured site instead.
    // Both follow the Born law of an exact operator. Truncation can leave
    // negative populations, where the native sampler clamps each conditional
    // and the single clamped marginal would sample another law, so a possibly
    // truncated operator keeps the native sampler. So does a trace too small
    // to normalize, which the native sampler rescales along the chain.
    template <bool Many> bool TrySingleQubitCounts(const Types::qubits_vector &qubits, size_t shots, Utils::Sampling::Counts<Many> &result)
    {
        return mpoSimulator && SingleQubitCounts<Many>(qubits, shots, result, [&](Types::qubit_t q) -> std::optional<double> {
                   if (!CanSampleSingleQubitMarginal())
                       return std::nullopt;
                   try
                   {
                       return mpoSimulator->GetProbability(static_cast<Eigen::Index>(q));
                   }
                   catch (const std::runtime_error &)
                   {
                       return std::nullopt;
                   }
               });
    }

    // Configuring a limit cannot truncate a state. Record history only when
    // an operation can use it; resetting the state clears that history while
    // a saved snapshot retains its own history for a later restore.
    void NotePossibleTruncation()
    {
        stateMayHaveTruncated = stateMayHaveTruncated || (nrQubits > 1 && (cutoffEnabled || bondLimitEnabled));
    }

    void ForgetTruncationHistory()
    {
        stateMayHaveTruncated = false;
        savedStateMayHaveTruncated.reset();
    }

    std::vector<long long> PendingDeflateMap() const
    {
        return pendingDeflate.count ? CurrentRoutingMap() : std::vector<long long>{};
    }

    void NotePendingDisentangle(const Types::qubits_vector &qubits) noexcept
    {
        try
        {
            EnsureLiveSiteRanks();
            if (pendingDeflate.ranks.size() != nrQubits)
                pendingDeflate.ranks.assign(nrQubits, 0.);
            for (const Types::qubit_t q : qubits)
            {
                if (q >= nrQubits)
                    continue;
                auto &rank = pendingDeflate.ranks[q];
                if (rank == 0.)
                    ++pendingDeflate.count;
                rank = std::max(rank, liveSiteRank[q]);
            }
        }
        catch (...)
        {
        }
    }

    void ExchangeTrackedStates(Types::qubit_t q0, Types::qubit_t q1) noexcept override
    {
        if (pendingDeflate.count && q0 < nrQubits && q1 < nrQubits)
            std::swap(pendingDeflate.ranks[q0], pendingDeflate.ranks[q1]);
    }

    void ClearSpanPendingDeflate(const std::vector<long long> &mapBefore, Types::qubit_t q0, Types::qubit_t q1) noexcept
    {
        if (!pendingDeflate.count || mapBefore.size() != nrQubits || q0 >= nrQubits || q1 >= nrQubits)
            return;
        const long long lo = std::min(mapBefore[q0], mapBefore[q1]);
        const long long hi = std::max(mapBefore[q0], mapBefore[q1]);
        RetirePendingDeflation([&](size_t q) { return mapBefore[q] >= lo && mapBefore[q] <= hi; });
    }

    template <class SweptSite> void RetirePendingDeflation(SweptSite swept) noexcept
    {
        // A truncation may trigger a full-chain Hermitize. QCSim does not
        // report whether that sweep occurred during a gate or physical swap.
        if ((cutoffEnabled || bondLimitEnabled) && mpoSimulator->getHermitizeAfterTruncation())
        {
            pendingDeflate = {};
            return;
        }
        for (size_t iq = 0; iq < nrQubits && pendingDeflate.count; ++iq)
        {
            if (pendingDeflate.ranks[iq] && swept(iq))
            {
                pendingDeflate.ranks[iq] = 0.;
                --pendingDeflate.count;
            }
        }
    }

    // QCSim's MPOSimulatorImpl applies single-qubit Reset and Measure in-place
    // without running an SVD sweep, so mpoSimulator->getBondDimensions() retains
    // pre-reset bond dimensions until a two-site gate sweeps across those bonds.
    void DeflateLiveBondDimensions(const std::vector<long long> &map, double *dims, size_t count) const override
    {
        if (!pendingDeflate.count || !dims || !count || map.size() != nrQubits || count + 1 != nrQubits)
            return;
        // Every cut across a run of product sites has the same rank. Compute the
        // estimates from the incoming bonds before assigning one rank to the run.
        std::vector<double> physicalRanks(nrQubits);
        for (size_t q = 0; q < nrQubits; ++q)
            physicalRanks[map[q]] = pendingDeflate.ranks[q];
        for (size_t p = 0; p < nrQubits;)
        {
            if (!physicalRanks[p])
            {
                ++p;
                continue;
            }
            const size_t first = p;
            double rank = std::numeric_limits<double>::infinity();
            for (; p < nrQubits && physicalRanks[p]; ++p)
            {
                if (p == 0 || p == count)
                    rank = 1.;
                else if (rank > 1.)
                {
                    const auto [low, high] = std::minmax(dims[p - 1], dims[p]);
                    const double symmetricRank = std::max(1., physicalRanks[p] * low / high);
                    rank = std::min(rank, std::max(1., low / std::pow(symmetricRank, MPSDummySimulator::kDisentangleDropExponent)));
                }
            }
            std::fill(dims + (first ? first - 1 : 0), dims + std::min(p, count), rank);
        }
    }

    std::vector<double> CurrentBondDimensions() const override
    {
        if (!mpoSimulator)
            return {};
        const auto dims = mpoSimulator->getBondDimensions();
        return {dims.begin(), dims.end()};
    }

    std::vector<long long> CurrentRoutingMap() const override
    {
        if (mpoSimulator)
            return ReadRoutingMap(mpoSimulator.get());
        return {};
    }

    void RefreshRoutingCallback() override
    {
        RefreshBoundedRouting();
        const bool active = IsRoutingLookaheadEnabled() && !upcomingGates.empty();
        if (mpoSimulator)
            mpoSimulator->SetMeetingPositionCallback(active ? meetingPositionCallback : nullptr);
    }

    std::unique_ptr<QC::TensorNetworks::MPOSimulator> mpoSimulator;
    bool cutoffEnabled = true;
    bool bondLimitEnabled = false;
    bool stateMayHaveTruncated = false;
    std::optional<bool> savedStateMayHaveTruncated;

    struct PendingDeflation
    {
        std::vector<double> ranks; // pre-collapse site rank by logical qubit, zero for unmarked sites
        size_t count = 0;
    };

    PendingDeflation pendingDeflate;
    std::optional<PendingDeflation> savedPendingDeflate;

  private:
    template <class Function> void ApplyDecomposedGate(Function &&apply, const Types::qubits_vector &affectedQubits)
    {
        {
            DecompositionScope scope(*this);
            std::forward<Function>(apply)();
        }
        NotifyObservers(affectedQubits);
    }

    template <class Gate, class... Qubits> void ApplyNativeGate(const Gate &gate, Qubits... qubits)
    {
        if constexpr (sizeof...(Qubits) <= 2)
        {
            const bool isTwoSite = gate.getQubitsNumber() > 1;
            const auto mapBefore = isTwoSite && !gate.isSwapGate() ? PendingDeflateMap() : std::vector<long long>{};
            if (isTwoSite)
                NotePossibleTruncation();
            mpoSimulator->ApplyGate(gate, static_cast<Eigen::Index>(qubits)...);
            if constexpr (sizeof...(Qubits) == 2)
            {
                const std::array<Types::qubit_t, 2> qs{static_cast<Types::qubit_t>(qubits)...};
                if (!gate.isSwapGate())
                    ClearSpanPendingDeflate(mapBefore, qs[0], qs[1]);
                RecordExecutedTwoQubitGate(qs[0], qs[1], gate.isSwapGate(), NativeGateSchmidtRank(gate));
            }
            else if constexpr (sizeof...(Qubits) == 0)
            {
                if (gate.getQubitsNumber() == 2)
                {
                    const Types::qubit_t q0 = static_cast<Types::qubit_t>(gate.getQubit1());
                    const Types::qubit_t q1 = static_cast<Types::qubit_t>(gate.getQubit2());
                    if (!gate.isSwapGate())
                        ClearSpanPendingDeflate(mapBefore, q0, q1);
                    RecordExecutedTwoQubitGate(q0, q1, gate.isSwapGate(), NativeGateSchmidtRank(gate));
                }
            }
        }
        else
        {
            throw std::runtime_error("ImmediateQCSimSimulator: The matrix product operator simulator "
                                     "supports "
                                     "only one- and two-qubit gate operators.");
        }
    }
};
} // namespace Simulators::Private
#endif
