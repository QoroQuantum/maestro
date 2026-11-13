#pragma once
#include "FusionGate.h"
#include <memory>
#include <optional>
#include <unordered_map>

namespace Simulators::Private
{
// State boundary for the fusion-aware adapters. The owned implementation is
// immediate and never calls back into this cache. Composite children use that
// implementation directly and do not instantiate this class.
class FusionState : public ISimulator
{
  public:
    explicit FusionState(std::shared_ptr<ISimulator> immediate) : immediate_(std::move(immediate)), cache_(3)
    {
        const auto seed = immediate_->GetConfiguration("sampling_seed");
        if (!seed.empty())
            SeedAuxiliaryRng(std::stoull(seed));
    }

    GateFusionStatistics GetGateFusionStatistics() const override
    {
        return stats_;
    }

    // An explicit gate_fusion setting always wins. Without one, fusion is on
    // from the backend's minimum register size; below it the fusion cache and
    // dense kernels cost more than the gates they replace.
    bool IsGateFusionEnabled() const override
    {
        if (GetGateFusionMaxQubits() == 0)
            return false;
        if (requested_)
            return *requested_;
        const auto qubits = GetNumberOfQubits();
        return qubits == 0 || qubits >= DefaultGateFusionMinQubits();
    }

    void Configure(const char *key, const char *value) override
    {
        const std::string name(key), setting(value);
        const auto oldWidth = GetGateFusionMaxQubits();
        const bool oldEnabled = IsGateFusionEnabled();
        const bool oldLookahead = IsRoutingLookaheadEnabled();
        const auto seed = name == "seed" ? std::optional<uint64_t>(std::stoull(setting)) : std::nullopt;
        if (name == "gate_fusion")
        {
            if (setting != "true" && setting != "false" && setting != "1" && setting != "0" && setting != "auto")
                throw std::invalid_argument("gate_fusion must be true, false, 1, 0 or auto");
            Flush();
            if (setting == "auto")
                requested_.reset();
            else
                requested_ = setting == "true" || setting == "1";
        }
        else if (name == kReadBlocksFirst)
        {
            if (setting != "true" && setting != "false" && setting != "1" && setting != "0")
                throw std::invalid_argument("trajectory_blocks_read_first must be true, false, 1 or 0");
            Flush();
            readBlocksFirst_ = setting == "true" || setting == "1";
        }
        else
        {
            Flush();
            immediate_->Configure(key, value);
            if (seed)
                SeedAuxiliaryRng(*seed);
        }
        cache_.SetWidth(GetGateFusionMaxQubits());
        if (oldWidth != GetGateFusionMaxQubits() || oldEnabled != IsGateFusionEnabled() || oldLookahead != IsRoutingLookaheadEnabled())
            RebuildPlan();
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (std::string(key) == "gate_fusion")
            return RequestedSetting();
        if (std::string(key) == kReadBlocksFirst)
            return readBlocksFirst_ ? "true" : "false";
        return immediate_->GetConfiguration(key);
    }

    const std::unordered_map<std::string, std::string> &GetConfigMap() const override
    {
        config_ = immediate_->GetConfigMap();
        config_["gate_fusion"] = RequestedSetting();
        return config_;
    }

    void SetSeed(uint64_t seed) override
    {
        Flush();
        SeedAuxiliaryRng(seed);
        immediate_->SetSeed(seed);
    }

    void SeedForExecution(uint64_t seed) override
    {
        Flush();
        SeedAuxiliaryRng(seed);
        immediate_->SeedForExecution(seed);
    }

    void FlushPendingGates() override
    {
        cache_.Flush([this](const auto &block) { Emit(block); });
        // Pending blocks never share a qubit, so their order is free.
        for (const auto &block : channelBlocks_)
            EmitChannelBlock(block);
        channelBlocks_.clear();
        if (!trajectoryBlocks_.empty())
        {
            // Each is resolved from the state as the ones before it left it.
            const auto blocks = std::move(trajectoryBlocks_);
            trajectoryBlocks_.clear();
            for (const auto &block : blocks)
                ResolveTrajectoryBlock(block);
        }
        // Whatever follows may read the state.
        SettleNorm();
    }

    // A mixed-state backend that applies a local map in one pass fuses a
    // channel with the gates around it: the pending gates on its qubits, the
    // channel and what follows become one superoperator on at most two qubits.
    bool FusesQuantumChannels() const override
    {
        return ready_ && IsGateFusionEnabled() && (immediate_->SupportsSuperoperators() || ResolvesChannelBlocks());
    }

    // A pure-state backend that samples channels and reads a local reduced
    // density matrix resolves a fused block from its state: one read gives the
    // probability of every branch of every channel in the block, and the
    // operators selected are then applied as one matrix.
    bool ResolvesChannelBlocks() const
    {
        return immediate_->SamplesQuantumChannels() && immediate_->SupportsLocalDensityMatrix();
    }

    bool SupportsLocalDensityMatrix() const override
    {
        return immediate_->SupportsLocalDensityMatrix();
    }

    Eigen::MatrixXcd LocalDensityMatrix(const Types::qubits_vector &qs) override
    {
        Flush();
        return immediate_->LocalDensityMatrix(qs);
    }

    bool SupportsLocalMapNorm() const override
    {
        return immediate_->SupportsLocalMapNorm();
    }

    double ApplyLocalMapNorm(const Types::qubits_vector &qs, const Eigen::MatrixXcd &matrix) override
    {
        Flush();
        const double norm = immediate_->ApplyLocalMapNorm(qs, matrix);
        // The caller's matrix need not keep the norm, and the backend has just measured what it left.
        normKnown_ = std::abs(norm - 1.) <= kNormTolerance;
        NotifyObservers(qs);
        return norm;
    }

    bool SkipIdleOperation() override
    {
        if (!FusesQuantumChannels())
            return false;
        ConsumeBoundary(Circuits::OperationType::kDelay);
        return true;
    }

    bool SupportsSuperoperators() const override
    {
        return immediate_->SupportsSuperoperators();
    }

    bool SamplesQuantumChannels() const override
    {
        return immediate_->SamplesQuantumChannels();
    }

    void ApplySuperoperator(const Types::qubits_vector &qs, const Eigen::MatrixXcd &superoperator) override
    {
        Flush();
        immediate_->ApplySuperoperator(qs, superoperator);
        NotifyObservers(qs);
    }

    void Flush() override
    {
        FlushPendingGates();
        // Distributed backends cannot synchronize until a native state exists.
        if (ready_)
            immediate_->Flush();
    }

    void SetUpcomingGates(const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &gates) override
    {
        Flush();
        sources_ = gates;
        sourceIndex_ = 0;
        RebuildPlan();
    }

    const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &GetUpcomingRoutingOperations() const override
    {
        static const std::vector<Operation> empty;
        // An active plan retains its complete indexed window, not a suffix.
        // Inactive metadata is only an initial-layout view. Reading it after
        // execution starts must neither expose that stale view nor flush pending
        // fusion blocks or install a new backend operation window.
        if (!planValid_ && sourceIndex_ != 0)
            return empty;
        // Initial placement may request a fused view even without lookahead.
        // Build it lazily; statevectors never need a routing plan.
        if ((!routingPlan_ || routingPlan_->routing.empty()) && !sources_.empty() && IsGateFusionEnabled() &&
            (GetSimulationType() == SimulationType::kMatrixProductState || GetSimulationType() == SimulationType::kMatrixProductOperator))
            const_cast<FusionState *>(this)->PreparePlan();
        return routingPlan_ ? routingPlan_->routing : empty;
    }

    long long GetGatesCounter() const override
    {
        return sourceIndex_;
    }

    void SetGatesCounter(long long counter) override
    {
        if (counter < 0)
            throw std::invalid_argument("Negative circuit position");
        if (counter == sourceIndex_)
            return;
        // Moving the counter only needs the pending blocks emitted at the old
        // position; it does not need the backend to finish its queued work.
        FlushPendingGates();
        // Advancing over a classical operation does not change the prepared
        // quantum blocks. Arbitrary jumps use local routing until explicitly
        // prepared again, rather than rescanning the suffix on every skipped gate.
        if (counter != sourceIndex_ + 1 || !AtBoundary())
            InvalidatePlan();
        sourceIndex_ = counter;
        if (!planValid_)
            immediate_->SetGatesCounter(sourceIndex_);
    }

    void IncrementGatesCounter() override
    {
        SetGatesCounter(sourceIndex_ + 1);
    }

    void Initialize() override
    {
        Discard();
        immediate_->Initialize();
        FinishInitialization(GetNumberOfQubits(), !InitializationPreservesSnapshots());
    }

    void Reset() override
    {
        Discard();
        immediate_->Reset();
        sourceIndex_ = 0;
        RebuildPlan();
    }

    void Clear() override
    {
        Discard();
        immediate_->Clear();
        ready_ = false;
        sourceIndex_ = 0;
        sources_.clear();
        routingPlan_.reset();
        backendRoutingInstalled_ = false;
        saved_.reset();
        destructiveSaved_.reset();
    }

    size_t AllocateQubits(size_t n) override
    {
        Flush();
        return immediate_->AllocateQubits(n);
    }

    void SaveState() override
    {
        Flush();
        immediate_->SaveState();
        saved_ = Context{sourceIndex_, sources_, normKnown_};
    }

    void RestoreState() override
    {
        // Several immediate implementations treat restore-without-save as a no-op.
        // Such a call must not silently throw away newly submitted gates.
        if (saved_)
            Discard();
        else
            Flush();
        immediate_->RestoreState();
        if (saved_)
        {
            sourceIndex_ = saved_->source;
            sources_ = saved_->operations;
            normKnown_ = saved_->normKnown;
        }
        RebuildPlan();
    }

    void SaveStateToInternalDestructive() override
    {
        Flush();
        immediate_->SaveStateToInternalDestructive();
        if (UsesDestructiveStateStorage())
            destructiveSaved_ = Context{sourceIndex_, sources_, normKnown_};
    }

    void RestoreInternalDestructiveSavedState() override
    {
        if (destructiveSaved_)
            Discard();
        else
            Flush();
        immediate_->RestoreInternalDestructiveSavedState();
        if (destructiveSaved_)
        {
            sourceIndex_ = destructiveSaved_->source;
            sources_ = destructiveSaved_->operations;
            normKnown_ = destructiveSaved_->normKnown;
            destructiveSaved_.reset();
        }
        RebuildPlan();
    }

    size_t Measure(const Types::qubits_vector &qs) override
    {
        Flush();
        auto result = immediate_->Measure(qs);
        ConsumeBoundary(Circuits::OperationType::kMeasurement);
        NotifyObservers(qs);
        return result;
    }

    std::vector<bool> MeasureMany(const Types::qubits_vector &qs) override
    {
        Flush();
        auto result = immediate_->MeasureMany(qs);
        ConsumeBoundary(Circuits::OperationType::kMeasurement);
        NotifyObservers(qs);
        return result;
    }

    void ApplyReset(const Types::qubits_vector &qs) override
    {
        Flush();
        immediate_->ApplyReset(qs);
        ConsumeBoundary(Circuits::OperationType::kReset);
        NotifyObservers(qs);
    }

    void ApplyQuantumChannel(const Types::qubits_vector &qs, const QuantumChannel &channel) override
    {
        if (FusesQuantumChannels())
        {
            // The checks the backend would make, since it now sees only the fused map.
            if (qs.size() != channel.GetNumberOfQubits())
                throw std::invalid_argument("The number of channel targets does not match its Kraus operators");
            if (qs.empty() || qs.size() > 2 || (qs.size() == 2 && qs[0] == qs[1]))
                throw std::invalid_argument("Quantum-channel target qubits must be one or two distinct qubits");
            for (const auto q : qs)
                if (q >= GetNumberOfQubits())
                    throw std::invalid_argument("Quantum-channel qubit is out of range");
            if (immediate_->SupportsSuperoperators())
                SubmitSuperoperator(qs, channel.Superoperator());
            else
                SubmitSampledChannel(qs, channel);
            ConsumeBoundary(Circuits::OperationType::kQuantumChannel);
            NotifyObservers(qs);
            return;
        }
        Flush();
        immediate_->ApplyQuantumChannel(qs, channel);
        ConsumeBoundary(Circuits::OperationType::kQuantumChannel);
        NotifyObservers(qs);
    }

    std::complex<double> DensityMatrixOverlap(const IState &other) const override
    {
        Synchronize();
        const auto *adapter = dynamic_cast<const FusionState *>(&other);
        if (adapter)
        {
            adapter->Synchronize();
            return immediate_->DensityMatrixOverlap(*adapter->immediate_);
        }
        return immediate_->DensityMatrixOverlap(other);
    }

    size_t GetCurrentMaxBondDimension() const override
    {
        Synchronize();
        return immediate_->GetCurrentMaxBondDimension();
    }

    size_t GetExecutedMaxBondDimension() const override
    {
        return immediate_->GetCurrentMaxBondDimension();
    }

    size_t GetNumberOfQubits() const override
    {
        return immediate_->GetNumberOfQubits();
    }

    SimulatorType GetType() const override
    {
        return immediate_->GetType();
    }

    SimulationType GetSimulationType() const override
    {
        return immediate_->GetSimulationType();
    }

    int GetGpuDevice() const override
    {
        return immediate_->GetGpuDevice();
    }

    bool IsQcsim() const override
    {
        return immediate_->IsQcsim();
    }

    bool GetMultithreading() const override
    {
        return immediate_->GetMultithreading();
    }

    bool SupportsMPSSwapOptimization() const override
    {
        return immediate_->SupportsMPSSwapOptimization();
    }

    bool SwapGatesRelabelQubits() const override
    {
        return immediate_->SwapGatesRelabelQubits();
    }

    bool IsRoutingLookaheadEnabled() const override
    {
        return immediate_->IsRoutingLookaheadEnabled();
    }

    bool SupportsQuantumChannels() const override
    {
        return immediate_->SupportsQuantumChannels();
    }

    double getGrowthFactorSwap() const override
    {
        return immediate_->getGrowthFactorSwap();
    }

    double getGrowthFactorGate() const override
    {
        return immediate_->getGrowthFactorGate();
    }

    double Probability(Types::qubit_t outcome) override
    {
        Synchronize();
        return immediate_->Probability(outcome);
    }

    std::complex<double> Amplitude(Types::qubit_t outcome) override
    {
        Synchronize();
        return immediate_->Amplitude(outcome);
    }

    std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override
    {
        Synchronize();
        return immediate_->AmplitudeRaw(outcome);
    }

    std::complex<double> ProjectOnZero() override
    {
        Synchronize();
        return immediate_->ProjectOnZero();
    }

    std::vector<double> AllProbabilities() override
    {
        Synchronize();
        return immediate_->AllProbabilities();
    }

    std::vector<std::complex<double>> GetStateVector() override
    {
        Synchronize();
        return immediate_->GetStateVector();
    }

    std::complex<double> ExpectationValueOperators(const Types::qubits_vector &qubits, const std::vector<Eigen::MatrixXcd> &matrices) override
    {
        Synchronize();
        return immediate_->ExpectationValueOperators(qubits, matrices);
    }

    std::vector<double> Probabilities(const Types::qubits_vector &qs) override
    {
        Synchronize();
        return immediate_->Probabilities(qs);
    }

    std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(const Types::qubits_vector &qs, size_t shots = 1000) override
    {
        Synchronize();
        return immediate_->SampleCounts(qs, shots);
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(const Types::qubits_vector &qs, size_t shots = 1000) override
    {
        Synchronize();
        return immediate_->SampleCountsMany(qs, shots);
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &paulis) override
    {
        if (paulis.empty())
            return {};
        Synchronize();
        return immediate_->ExpectationValues(paulis);
    }

    double ExpectationValue(const std::string &pauli) override
    {
        Synchronize();
        return immediate_->ExpectationValue(pauli);
    }

    Types::qubit_t MeasureNoCollapse() override
    {
        Synchronize();
        return immediate_->MeasureNoCollapse();
    }

    std::vector<bool> MeasureNoCollapseMany() override
    {
        Synchronize();
        return immediate_->MeasureNoCollapseMany();
    }

    double ProbabilityBits(const std::vector<bool> &bits) override
    {
        Synchronize();
        return immediate_->ProbabilityBits(bits);
    }

    std::complex<double> DensityMatrixElementBits(const std::vector<bool> &row, const std::vector<bool> &col) const override
    {
        Synchronize();
        return immediate_->DensityMatrixElementBits(row, col);
    }

    Eigen::MatrixXcd GetDensityMatrix(bool normalized = true) const override
    {
        Synchronize();
        return immediate_->GetDensityMatrix(normalized);
    }

    std::complex<double> ExpectationValueComplex(const std::string &pauli, bool normalized = true) const override
    {
        Synchronize();
        return immediate_->ExpectationValueComplex(pauli, normalized);
    }

    std::vector<std::complex<double>> ExpectationValuesComplex(const std::vector<std::string> &paulis, bool normalized = true) const override
    {
        if (paulis.empty())
            return {};
        Synchronize();
        return immediate_->ExpectationValuesComplex(paulis, normalized);
    }

    void ApplyOperator(const Types::qubits_vector &qubits, const Eigen::MatrixXcd &matrix, bool normalize = false) override
    {
        Flush();
        immediate_->ApplyOperator(qubits, matrix, normalize);
        if (!normalize && normKnown_ && !KeepsNorm(matrix))
            normKnown_ = false;
        InvalidatePlan();
        NotifyObservers(qubits);
    }

    void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override
    {
        Flush();
        // A failed routing SWAP may have already changed the physical qubit map.
        InvalidatePlan();
        immediate_->MoveAtBeginningOfChain(qubits);
        RebuildPlan();
    }

    std::complex<double> DensityMatrixTrace() const override
    {
        Synchronize();
        return immediate_->DensityMatrixTrace();
    }

    double DensityMatrixPurity() const override
    {
        Synchronize();
        return immediate_->DensityMatrixPurity();
    }

    std::complex<double> DensityMatrixTraceOfSquare() const override
    {
        Synchronize();
        return immediate_->DensityMatrixTraceOfSquare();
    }

    double DensityMatrixHermiticityResidual() const override
    {
        Synchronize();
        return immediate_->DensityMatrixHermiticityResidual();
    }

    bool IsDensityMatrixHermitian(double eps = 1e-10) const override
    {
        Synchronize();
        return immediate_->IsDensityMatrixHermitian(eps);
    }

    Eigen::MatrixXcd PartialTrace(const Types::qubits_vector &qs) const override
    {
        Synchronize();
        return immediate_->PartialTrace(qs);
    }

    double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override
    {
        Synchronize();
        return immediate_->FidelityWithStatevector(psi);
    }

    void RestoreDensityMatrixTrace() override
    {
        Flush();
        immediate_->RestoreDensityMatrixTrace();
    }

    void HermitizeDensityMatrix() override
    {
        Flush();
        immediate_->HermitizeDensityMatrix();
    }

    void Trim() override
    {
        Flush();
        immediate_->Trim();
    }

    void ReCanonicalize() override
    {
        Flush();
        immediate_->ReCanonicalize();
    }

    void SetMultithreading(bool value = true) override
    {
        Flush();
        immediate_->SetMultithreading(value);
    }

    void SetInitialQubitsMap(const std::vector<long long> &value) override
    {
        Flush();
        immediate_->SetInitialQubitsMap(value);
    }

    void SetUseOptimalMeetingPosition(bool value) override
    {
        UpdateRoutingSetting([&] { immediate_->SetUseOptimalMeetingPosition(value); });
    }

    void SetLookaheadDepth(int value) override
    {
        UpdateRoutingSetting([&] { immediate_->SetLookaheadDepth(value); });
    }

    void SetAutomaticLookaheadDepth(int value) override
    {
        UpdateRoutingSetting([&] { immediate_->SetAutomaticLookaheadDepth(value); });
    }

    void SetLookaheadDepthWithHeuristic(int value) override
    {
        UpdateRoutingSetting([&] { immediate_->SetLookaheadDepthWithHeuristic(value); });
    }

    void setGrowthFactorSwap(double value) override
    {
        Flush();
        const bool oldLookahead = IsRoutingLookaheadEnabled();
        immediate_->setGrowthFactorSwap(value);
        if (oldLookahead != IsRoutingLookaheadEnabled())
            RebuildPlan();
    }

    void setGrowthFactorGate(double value) override
    {
        Flush();
        const bool oldLookahead = IsRoutingLookaheadEnabled();
        immediate_->setGrowthFactorGate(value);
        if (oldLookahead != IsRoutingLookaheadEnabled())
            RebuildPlan();
    }

    void InitializeState(size_t n, std::vector<std::complex<double>> &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeState(n, value);
            return;
        }
        // Read before the backend takes the amplitudes: they need not have unit norm.
        const bool unitNorm = !immediate_->SupportsLocalMapNorm() || HasUnitNorm(value);
        Discard();
        immediate_->InitializeState(n, value);
        normKnown_ = unitNorm;
        FinishInitialization(n);
    }
#ifndef NO_QISKIT_AER
    void InitializeState(size_t n, AER::Vector<std::complex<double>> &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeState(n, value);
            return;
        }
        // Read before the backend takes the amplitudes: they need not have unit norm.
        const bool unitNorm = !immediate_->SupportsLocalMapNorm() || HasUnitNorm(value);
        Discard();
        immediate_->InitializeState(n, value);
        normKnown_ = unitNorm;
        FinishInitialization(n);
    }
#endif
    void InitializeState(size_t n, Eigen::VectorXcd &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeState(n, value);
            return;
        }
        // Read before the backend takes the amplitudes: they need not have unit norm.
        const bool unitNorm = !immediate_->SupportsLocalMapNorm() || HasUnitNorm(value);
        Discard();
        immediate_->InitializeState(n, value);
        normKnown_ = unitNorm;
        FinishInitialization(n);
    }

    void InitializeToBasisState(size_t n, Types::qubit_t value) override
    {
        if (n == 0)
        {
            immediate_->InitializeToBasisState(n, value);
            return;
        }
        Discard();
        immediate_->InitializeToBasisState(n, value);
        FinishInitialization(n);
    }

    void InitializeToBasisState(size_t n, const std::vector<bool> &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeToBasisState(n, value);
            return;
        }
        Discard();
        immediate_->InitializeToBasisState(n, value);
        FinishInitialization(n);
    }

    void InitializeToMixtureOfBasisStates(size_t n, const std::vector<std::pair<Types::qubit_t, double>> &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeToMixtureOfBasisStates(n, value);
            return;
        }
        Discard();
        immediate_->InitializeToMixtureOfBasisStates(n, value);
        FinishInitialization(n);
    }

    void InitializeToMixtureOfBasisStates(size_t n, const std::vector<std::pair<std::vector<bool>, double>> &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeToMixtureOfBasisStates(n, value);
            return;
        }
        Discard();
        immediate_->InitializeToMixtureOfBasisStates(n, value);
        FinishInitialization(n);
    }

  protected:
    // QCSim's destructive save/restore hooks are no-ops. Backends that move
    // their storage opt in so the routing context follows that storage.
    // Smallest register for which fusion is on when gate_fusion is not set.
    virtual size_t DefaultGateFusionMinQubits() const
    {
        return 0;
    }

    virtual bool UsesDestructiveStateStorage() const
    {
        return false;
    }

    virtual bool InitializationPreservesSnapshots() const
    {
        return false;
    }

    // Backends where a dense matrix costs more than native structured gates
    // (e.g. communication on distributed global qubits) can decline merges of
    // exclusively diagonal/permutation gates without changing the cache
    // algorithm. Elsewhere merging them measured faster or equal.
    virtual bool PreserveStructuredGates() const
    {
        return false;
    }

    using Cache = GateFusion<FusionGate>;
    using Operation = std::shared_ptr<Circuits::IOperation<double>>;
    std::shared_ptr<ISimulator> immediate_;

    void CloneInto(FusionState &copy, const uint64_t *executionSeed = nullptr)
    {
        Flush();
        copy.immediate_ = executionSeed ? immediate_->CloneForExecution(*executionSeed) : immediate_->Clone();
        copy.requested_ = requested_;
        copy.stats_ = stats_;
        copy.ready_ = ready_;
        copy.sources_ = sources_;
        copy.sourceIndex_ = sourceIndex_;
        copy.saved_ = saved_;
        copy.destructiveSaved_ = destructiveSaved_;
        copy.readBlocksFirst_ = readBlocksFirst_;
        copy.normKnown_ = normKnown_;
        copy.cache_.SetWidth(copy.GetGateFusionMaxQubits());
        // An unseeded backend clone keeps its derived root as its sampling seed.
        auto seed = copy.immediate_->GetConfiguration("seed");
        if (seed.empty())
            seed = copy.immediate_->GetConfiguration("sampling_seed");
        if (!seed.empty())
            copy.SeedAuxiliaryRng(std::stoull(seed));
        // Flush above either completed the pending planned block or invalidated
        // it. A still-valid plan uses the same complete backend operation window
        // and cursor copied by the native wrapper; do not reparse its suffix.
        const auto method = GetSimulationType();
        const bool cpuChain =
            GetType() == SimulatorType::kQCSim && (method == SimulationType::kMatrixProductState || method == SimulationType::kMatrixProductOperator);
        if (cpuChain && (!IsGateFusionEnabled() || (planValid_ && routingPlan_)))
        {
            copy.routingPlan_ = routingPlan_;
            copy.planIndex_ = planIndex_;
            copy.planValid_ = planValid_;
            copy.backendRoutingInstalled_ = backendRoutingInstalled_;
        }
        else
            copy.RebuildPlan();
    }

    // A generic gate need not be unitary (GenericGateValidation.h): one that is not scales the state by an amount
    // this layer cannot know without reading it, whether or not the gate is fused. Called where a generic matrix
    // comes in from the caller, before SubmitGate: every gate of every shot passes SubmitGate, and none of a
    // circuit is generic.
    template <class Matrix> void NoteGenericGate(const Matrix &matrix)
    {
        if (normKnown_ && immediate_->SupportsLocalMapNorm() && !KeepsNorm(matrix))
            normKnown_ = false;
    }

    void SubmitGate(const FusionGate &gate)
    {
        if (!IsGateFusionEnabled())
        {
            immediate_->SetGatesCounter(sourceIndex_);
            gate.Apply(*immediate_);
            ++stats_.submittedGates;
            ++stats_.backendGates;
            ++sourceIndex_;
            NotifyObservers(gate.qubits);
            return;
        }
        if (!ready_)
            throw std::logic_error("Initialize the simulator before applying gates");
        auto qs = gate.qubits;
        std::sort(qs.begin(), qs.end());
        if (qs.empty() || qs.back() >= GetNumberOfQubits() || std::adjacent_find(qs.begin(), qs.end()) != qs.end())
            throw std::invalid_argument("Invalid or duplicate gate target");
        for (auto p : gate.params)
            if (!std::isfinite(p))
                throw std::invalid_argument("Non-finite gate parameter");
        if (gate.kind == FusionGate::Kind::kNone && !gate.generic.allFinite())
            throw std::invalid_argument("Non-finite gate matrix");
        ++stats_.submittedGates;
        const auto width = GetGateFusionMaxQubits();
        if (gate.kind == FusionGate::Kind::kNone && gate.qubits.size() > width)
        {
            FlushPendingGates();
            InvalidatePlan();
            gate.Apply(*immediate_);
            ++stats_.backendGates;
            ++sourceIndex_;
            NotifyObservers(gate.qubits);
            return;
        }
        // Nonunitary generic operators retain their backend's existing semantics.
        if (gate.kind == FusionGate::Kind::kNone &&
            !(gate.generic.adjoint() * gate.generic).isApprox(Eigen::MatrixXcd::Identity(gate.generic.rows(), gate.generic.rows()), 1e-10))
        {
            FlushPendingGates();
            InvalidatePlan();
            gate.Apply(*immediate_);
            ++stats_.backendGates;
            ++sourceIndex_;
            NotifyObservers(gate.qubits);
            return;
        }
        if (planValid_)
        {
            const auto index = static_cast<size_t>(sourceIndex_);
            const auto boundary = routingPlan_->boundaries.find(index);
            if (boundary != routingPlan_->boundaries.end() && sources_[index]->GetType() == Circuits::OperationType::kConditionalGate)
            {
                // A conditional gate that fires is a boundary of the plan: apply it on
                // its own, at its routing position, and keep the plan.
                FlushPendingGates();
                immediate_->SetGatesCounter(static_cast<long long>(boundary->second));
                gate.Apply(*immediate_);
                ++stats_.backendGates;
                ++sourceIndex_;
                NotifyObservers(gate.qubits);
                return;
            }
            const auto original = index < sources_.size() ? std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(sources_[index]) : nullptr;
            if (!original || !FusionGate::FromCircuit(*original).Matches(gate))
                InvalidatePlan();
        }
        if ((!channelBlocks_.empty() && FuseGateWithChannelBlocks(gate)) || (!trajectoryBlocks_.empty() && FuseGateWithTrajectoryBlocks(gate)))
        {
            ++sourceIndex_;
            NotifyObservers(gate.qubits);
            return;
        }
        const auto source = static_cast<uint64_t>(sourceIndex_++);
        if (cache_.Empty())
            cache_.SetWidth(width);
        const auto emit = [this](const auto &block) { Emit(block); };
        if (gate.qubits.size() <= width) // Expand would return {gate}
            SubmitToCache(cache_, gate, source << 4, emit);
        else
        {
            const auto primitives = gate.Expand(width);
            for (size_t i = 0; i < primitives.size(); ++i)
                SubmitToCache(cache_, primitives[i], (source << 4) | i, emit);
        }
        NotifyObservers(gate.qubits);
    }

    void ConsumeBoundary(Circuits::OperationType type)
    {
        if (!AtBoundary() || sources_[sourceIndex_]->GetType() != type)
            InvalidatePlan();
        ++sourceIndex_;
        if (!planValid_)
            immediate_->SetGatesCounter(sourceIndex_);
    }

  private:
    template <class Setter> void UpdateRoutingSetting(Setter &&setter)
    {
        Flush();
        const bool oldLookahead = IsRoutingLookaheadEnabled();
        std::forward<Setter>(setter)();
        const bool newLookahead = IsRoutingLookaheadEnabled();
        // Reuse is limited to the CPU tensor chains verified here. Initial
        // routing metadata does not depend on their depth/heuristic.
        // Mid-circuit calls still rebuild the suffix: Flush may split a planned
        // block, and a counter rewind can leave an invalid plan even at zero.
        const auto method = GetSimulationType();
        const bool cpuChain =
            GetType() == SimulatorType::kQCSim && (method == SimulationType::kMatrixProductState || method == SimulationType::kMatrixProductOperator);
        if (!cpuChain || sourceIndex_ != 0 || oldLookahead != newLookahead || (newLookahead && !planValid_))
            RebuildPlan();
    }

    template <class EmitBlock> void SubmitToCache(Cache &cache, const FusionGate &gate, uint64_t source, EmitBlock &&emit)
    {
        cache.Submit(gate, source, std::forward<EmitBlock>(emit), PreserveStructuredGates() && gate.IsStructured(),
                     SwapGatesRelabelQubits() && gate.kind == FusionGate::Kind::kSwapGateType);
    }

    // Operations fused into one local map of a mixed state: a superoperator
    // over the block's sorted targets, on density blocks indexed row + d *
    // column with targets[0] the low bit of both.
    struct ChannelBlock
    {
        Types::qubits_vector targets;
        Eigen::MatrixXcd superoperator;
        uint64_t operations = 1;
    };

    static bool Overlaps(const Types::qubits_vector &a, const Types::qubits_vector &b)
    {
        return std::any_of(a.begin(), a.end(), [&](auto q) { return std::find(b.begin(), b.end(), q) != b.end(); });
    }

    static Types::qubits_vector Union(Types::qubits_vector a, const Types::qubits_vector &b)
    {
        a.insert(a.end(), b.begin(), b.end());
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
        return a;
    }

    // conj(U) (x) U: U on the row (ket) index and its conjugate on the column (bra) index.
    static Eigen::MatrixXcd UnitarySuperoperator(const Eigen::MatrixXcd &unitary)
    {
        const Eigen::Index d = unitary.rows();
        Eigen::MatrixXcd superoperator(d * d, d * d);
        for (Eigen::Index c2 = 0; c2 < d; ++c2)
            for (Eigen::Index c = 0; c < d; ++c)
                for (Eigen::Index r2 = 0; r2 < d; ++r2)
                    for (Eigen::Index r = 0; r < d; ++r)
                        superoperator(r + d * r2, c + d * c2) = unitary(r, c) * std::conj(unitary(r2, c2));
        return superoperator;
    }

    // A superoperator over k targets is a matrix over 2k index bits: the ket
    // bits of the targets in order, then their bra bits. Naming those bits
    // lets the gate cache's embedding place or permute a superoperator.
    static Types::qubits_vector IndexBits(const Types::qubits_vector &targets)
    {
        Types::qubits_vector bits;
        bits.reserve(2 * targets.size());
        for (const auto q : targets)
            bits.push_back(2 * q);
        for (const auto q : targets)
            bits.push_back(2 * q + 1);
        return bits;
    }

    void EmitChannelBlock(const ChannelBlock &block)
    {
        // A routing plan lists the gate blocks and the channels one by one;
        // a block that holds both is none of its entries. The backend routes
        // the rest without the plan's lookahead.
        if (planValid_)
            InvalidatePlan();
        immediate_->ApplySuperoperator(block.targets, block.superoperator);
        ++stats_.backendGates;
        if (block.operations > 1)
            ++stats_.fusedBlocks;
    }

    // Adds a map on `order` (its matrix's target order) to the pending
    // operations of a mixed state. Pending gates and channel blocks on those
    // qubits come before it: the ones that fit with it on two qubits become
    // part of its block and the others are emitted.
    void SubmitSuperoperator(const Types::qubits_vector &order, const Eigen::MatrixXcd &superoperator)
    {
        Types::qubits_vector support = order;
        std::sort(support.begin(), support.end());
        std::vector<ChannelBlock> earlier;
        cache_.Extract(order, [&](Cache::Block &&block) {
            auto joined = Union(support, block.targets);
            if (joined.size() > 2)
            {
                Emit(block);
                return;
            }
            support = std::move(joined);
            earlier.push_back({block.targets, UnitarySuperoperator(Cache::BlockMatrix(block)), block.sources.count});
        });
        for (auto it = channelBlocks_.begin(); it != channelBlocks_.end();)
        {
            if (!Overlaps(it->targets, order))
            {
                ++it;
                continue;
            }
            auto joined = Union(support, it->targets);
            if (joined.size() > 2)
                EmitChannelBlock(*it);
            else
            {
                support = std::move(joined);
                earlier.push_back(std::move(*it));
            }
            it = channelBlocks_.erase(it);
        }

        // The earlier parts act on different qubits, so they commute.
        ChannelBlock block{support, {}, 1};
        const auto bits = IndexBits(support);
        for (const auto &part : earlier)
        {
            Eigen::MatrixXcd embedded = Cache::Embed(part.superoperator, IndexBits(part.targets), bits);
            block.superoperator = block.superoperator.size() ? (embedded * block.superoperator).eval() : std::move(embedded);
            block.operations += part.operations;
        }
        Eigen::MatrixXcd embedded = Cache::Embed(superoperator, IndexBits(order), bits);
        block.superoperator = block.superoperator.size() ? (embedded * block.superoperator).eval() : std::move(embedded);
        channelBlocks_.push_back(std::move(block));
    }

    // A gate on qubits that hold a pending channel block joins that block
    // when the two fit on two qubits. Otherwise the block is emitted and the
    // gate takes the ordinary path.
    bool FuseGateWithChannelBlocks(const FusionGate &gate)
    {
        Types::qubits_vector support = gate.qubits;
        bool overlaps = false;
        for (const auto &block : channelBlocks_)
            if (Overlaps(block.targets, gate.qubits))
            {
                overlaps = true;
                support = Union(std::move(support), block.targets);
            }
        if (!overlaps)
            return false;
        if (gate.qubits.size() <= 2 && Union(std::move(support), {}).size() <= 2)
        {
            SubmitSuperoperator(gate.Targets(), UnitarySuperoperator(gate.Matrix()));
            return true;
        }
        for (auto it = channelBlocks_.begin(); it != channelBlocks_.end();)
            if (Overlaps(it->targets, gate.qubits))
            {
                EmitChannelBlock(*it);
                it = channelBlocks_.erase(it);
            }
            else
                ++it;
        return false;
    }

    // One step of a block that a pure-state backend resolves from its state:
    // a unitary, or a channel factor whose branch is drawn at that time.
    struct TrajectoryStep
    {
        Types::qubits_vector order;            // targets in the order of the matrices
        Eigen::MatrixXcd unitary;              // empty for a channel factor
        std::optional<QuantumChannel> channel; // holds the factor
        size_t factor = 0;
        // Drawn when the channel arrives: the channel stream is consumed in
        // circuit order, whatever the order in which blocks are resolved.
        double random = 0.;
        QuantumChannel::KrausOperators own; // the operators when no channel holds them (a reset)
    };

    struct TrajectoryBlock
    {
        Types::qubits_vector targets; // sorted
        std::vector<TrajectoryStep> steps;
        uint64_t operations = 0;
        size_t channels = 0; // steps that draw a branch
    };

    // As wide as the gate blocks, and as the reduced density matrix a backend reads in one pass.
    size_t TrajectoryWidth() const
    {
        return std::min<size_t>(3, GetGateFusionMaxQubits());
    }

    static size_t DrawBranch(const std::vector<double> &probabilities, double random)
    {
        size_t chosen = 0;
        double accumulated = 0.;
        for (size_t index = 0; index < probabilities.size(); ++index)
        {
            if (probabilities[index] <= 0.)
                continue;
            chosen = index; // also the fallback if rounding leaves the draw above the last sum
            accumulated += probabilities[index];
            if (random < accumulated)
                break;
        }
        return chosen;
    }

    // A channel on a backend that samples: its Pauli and unitary-mixture
    // factors are drawn now, without the state, and join the pending gates;
    // every other factor waits in a block for the state it is drawn from.
    void SubmitSampledChannel(const Types::qubits_vector &qs, const QuantumChannel &channel)
    {
        static constexpr FusionGate::Kind paulis[4] = {FusionGate::Kind::kNone, FusionGate::Kind::kXGateType, FusionGate::Kind::kYGateType,
                                                       FusionGate::Kind::kZGateType};
        unsigned drawn = 0;
        const auto &factors = channel.GetFactors();
        for (size_t f = 0; f < factors.size(); ++f)
        {
            const auto &factor = factors[f];
            if (factor.kind == QuantumChannel::FactorKind::kPauli)
            {
                const size_t index = DrawBranch(factor.probabilities, immediate_->ChannelUniform());
                for (size_t k = 0; k < factor.qubits.size(); ++k)
                    if (const size_t pauli = (index >> (2 * k)) & 3)
                        SubmitDrawnGate({paulis[pauli], {qs[factor.qubits[k]]}, {}, {}}, drawn);
                continue;
            }
            Types::qubits_vector order;
            for (const auto q : factor.qubits)
                order.push_back(qs[q]);
            if (factor.kind == QuantumChannel::FactorKind::kUnitaryMixture)
            {
                const auto &unitary = factor.operators[DrawBranch(factor.probabilities, immediate_->ChannelUniform())];
                if (!unitary.isIdentity(0.))
                    SubmitDrawnGate({FusionGate::Kind::kNone, std::move(order), {}, unitary}, drawn);
            }
            else
            {
                double random = immediate_->ChannelUniform();
                QuantumChannel::ThermalMixture mixture;
                if (factor.kind != QuantumChannel::FactorKind::kThermal || !QuantumChannel::MixtureOf(factor.thermal, mixture))
                    SubmitTrajectoryStep({std::move(order), {}, channel, f, random, {}});
                else if (random >= mixture.identity + mixture.phaseFlip)
                {
                    // Only the reset reads the state: its outcome takes what is left of the draw.
                    random = std::min((random - mixture.identity - mixture.phaseFlip) / mixture.reset, std::nextafter(1., 0.));
                    SubmitTrajectoryStep(
                        {std::move(order), {}, std::nullopt, 0, random, QuantumChannel::ResetOperators(factor.thermal.excitedStatePopulation)});
                }
                else if (random >= mixture.identity)
                    SubmitDrawnGate({FusionGate::Kind::kZGateType, std::move(order), {}, {}}, drawn);
            }
        }
    }

    // A unitary drawn from a channel takes the path of a gate without being a source operation.
    void SubmitDrawnGate(const FusionGate &gate, unsigned &drawn)
    {
        if (!trajectoryBlocks_.empty() && FuseGateWithTrajectoryBlocks(gate))
            return;
        if (drawn == 16)
        { // the cache identifiers of this source position are used up
            FlushPendingGates();
            drawn = 0;
        }
        if (cache_.Empty())
            cache_.SetWidth(GetGateFusionMaxQubits());
        SubmitToCache(cache_, gate, (static_cast<uint64_t>(sourceIndex_) << 4) | drawn++, [this](const auto &block) { Emit(block); });
    }

    // Adds a step to the pending blocks of a sampling backend. Pending blocks
    // and gates on its qubits come before it: those that fit with it on the
    // block width become part of its block, the others are resolved or emitted.
    // Pending blocks never share a qubit, so the ones taken in commute.
    void SubmitTrajectoryStep(TrajectoryStep step)
    {
        const size_t width = std::max(step.order.size(), TrajectoryWidth());
        Types::qubits_vector support = step.order;
        std::sort(support.begin(), support.end());
        TrajectoryBlock block;
        for (auto it = trajectoryBlocks_.begin(); it != trajectoryBlocks_.end();)
        {
            if (!Overlaps(it->targets, step.order))
            {
                ++it;
                continue;
            }
            auto joined = Union(support, it->targets);
            if (joined.size() > width)
                ResolveTrajectoryBlock(*it);
            else
            {
                support = std::move(joined);
                block.operations += it->operations;
                block.channels += it->channels;
                if (block.steps.empty())
                    block.steps = std::move(it->steps);
                else
                    for (auto &earlier : it->steps)
                        block.steps.push_back(std::move(earlier));
            }
            it = trajectoryBlocks_.erase(it);
        }
        cache_.Extract(step.order, [&](Cache::Block &&pending) {
            auto joined = Union(support, pending.targets);
            if (joined.size() > width)
            {
                Emit(pending);
                return;
            }
            support = std::move(joined);
            block.operations += pending.sources.count;
            block.steps.push_back({pending.targets, Cache::BlockMatrix(pending), std::nullopt, 0, 0.});
        });
        // A unitary that follows a unitary on the same qubits is multiplied into it.
        const auto within = [](const Types::qubits_vector &part, const Types::qubits_vector &whole) {
            return std::all_of(part.begin(), part.end(), [&](auto q) { return std::find(whole.begin(), whole.end(), q) != whole.end(); });
        };
        if (step.unitary.size() && !block.steps.empty() && block.steps.back().unitary.size() && within(step.order, block.steps.back().order))
            Cache::ApplyLeft(step.unitary, step.order, block.steps.back().order, block.steps.back().unitary);
        else
        {
            block.channels += step.unitary.size() == 0;
            block.steps.push_back(std::move(step));
        }
        ++block.operations;
        block.targets = std::move(support);
        trajectoryBlocks_.push_back(std::move(block));
    }

    // A gate on qubits that hold a pending trajectory block joins it when the
    // two fit; otherwise the block is resolved and the gate takes the ordinary path.
    bool FuseGateWithTrajectoryBlocks(const FusionGate &gate)
    {
        Types::qubits_vector support = gate.qubits;
        bool overlaps = false;
        for (const auto &block : trajectoryBlocks_)
            if (Overlaps(block.targets, gate.qubits))
            {
                overlaps = true;
                support = Union(std::move(support), block.targets);
            }
        if (!overlaps)
            return false;
        if (support.size() <= TrajectoryWidth())
        {
            SubmitTrajectoryStep({gate.Targets(), gate.Matrix(), std::nullopt, 0, 0.});
            return true;
        }
        for (auto it = trajectoryBlocks_.begin(); it != trajectoryBlocks_.end();)
            if (Overlaps(it->targets, gate.qubits))
            {
                ResolveTrajectoryBlock(*it);
                it = trajectoryBlocks_.erase(it);
            }
            else
                ++it;
        return false;
    }

    // rho <- op rho op^dagger for a Hermitian rho, op acting on `order` within `targets`.
    template <class Operator, class Matrix>
    static void Conjugate(const Operator &op, const Types::qubits_vector &order, const Types::qubits_vector &targets, Matrix &rho)
    {
        Cache::ApplyLeft(op, order, targets, rho);
        rho.adjointInPlace();
        Cache::ApplyLeft(op, order, targets, rho);
    }

    // Fixed capacity: the algebra of a block allocates nothing.
    using Local = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 8, 8>;

    // Every operator of the block acts on its targets only, so the reduced
    // density matrix of the targets before the block gives the probability of
    // every branch. The steps are walked on that small matrix, each channel
    // factor draws its operator there, and the product of the operators
    // selected reaches the state as one matrix: one read and one write of the
    // state for the whole block, however many gates and channels it holds.
    // Where the channels most probably do nothing, the read is saved too
    // (ResolveWithoutReading).
    void ResolveTrajectoryBlock(const TrajectoryBlock &block)
    {
        if (planValid_)
            InvalidatePlan();
        if (ResolveWithoutReading(block))
            return;
        Local rho = immediate_->LocalDensityMatrix(block.targets);
        if (stateNormSq_ != 1.)
            rho /= stateNormSq_;
        Local product = WalkTrajectoryBlock(block, rho);
        if (stateNormSq_ != 1.)
        {
            // The norm an earlier block left is restored by this one.
            product /= std::sqrt(stateNormSq_);
            stateNormSq_ = 1.;
        }
        ApplyBlockMatrix(block.targets, product);
        ++stats_.backendGates;
        if (block.operations > 1)
            ++stats_.fusedBlocks;
    }

    // A block whose channels most probably do nothing is applied on that
    // assumption, without reading the state first.
    //
    // For operators that preserve the trace, a channel takes its first
    // operator when its draw lies below the probability of that operator. The
    // product of those probabilities over the block is the squared norm the
    // state has once the gates and the first operators of the block have been
    // applied to it, and the backend returns that norm from the pass that
    // applies them (ApplyLocalMapNorm). Each probability is at least the
    // product. So if every draw of the block lies below the norm, every
    // channel took its first operator, as resolving the block from the
    // reduced density matrix would have found, and the block has cost one
    // pass over the state instead of two. The state keeps that norm, which
    // this layer remembers (stateNormSq_) for the next block to divide out.
    //
    // Otherwise some channel may have taken another operator. The reduced
    // density matrix is read then, of the state as it is now. Undoing the
    // applied matrix on that small matrix gives the one before the block; the
    // block is resolved from it as usual, and the state receives the operators
    // selected together with the inverse of what was applied. That costs three
    // passes where reading first costs two, the read being the cheaper one, so
    // the attempt has to hold about five times in eight to pay. It is made
    // when the draws leave that likely: below the smallest probability the
    // block can have it is certain to hold, and between the smallest and the
    // largest it is taken to be as likely as the largest draw is close to the
    // smallest. A block whose first operators can be improbable is read first:
    // the matrix applied is then always well conditioned.
    bool ResolveWithoutReading(const TrajectoryBlock &block)
    {
        // The attempt compares the draws with the norm the pass returns, which says how probable the first
        // operators were only on a state whose norm this layer knows.
        if (!block.channels || readBlocksFirst_ || !normKnown_ || !immediate_->SupportsLocalMapNorm())
            return false;
        double low = 1., high = 1., largest = 0.;
        for (const auto &step : block.steps)
        {
            if (step.unitary.size())
                continue;
            if (!step.channel)
                return false; // a reset: its operators are projections
            const auto &first = step.channel->FirstOperatorOf(step.factor);
            if (!first.tracePreserving)
                return false;
            low *= first.low;
            high *= first.high;
            largest = std::max(largest, step.random);
        }
        if (!(low >= .25 * high) || !(largest < low + .375 * (high - low)))
            return false;

        const size_t dimension = size_t{1} << block.targets.size();
        Local product = Local::Identity(dimension, dimension);
        for (const auto &step : block.steps)
            if (step.unitary.size())
                Cache::ApplyLeft(step.unitary, step.order, block.targets, product);
            else
                Cache::ApplyLeft(step.channel->GetFactors()[step.factor].operators.front(), step.order, block.targets, product);
        // On the state at unit norm: what an earlier block left is divided out here.
        const Eigen::MatrixXcd applied = product / std::sqrt(stateNormSq_);
        const double kept = immediate_->ApplyLocalMapNorm(block.targets, applied);
        ++stats_.backendGates;
        ++stats_.unreadBlocks;
        if (block.operations > 1)
            ++stats_.fusedBlocks;
        if (largest < kept)
        {
            stateNormSq_ = kept;
            return true;
        }
        Local after = immediate_->LocalDensityMatrix(block.targets);
        const Local inverse = product.inverse();
        Local before = inverse * after * inverse.adjoint();
        after = before.adjoint();
        before = .5 * (before + after);
        const Local resolved = WalkTrajectoryBlock(block, before) * inverse;
        stateNormSq_ = 1.;
        ApplyBlockMatrix(block.targets, resolved);
        ++stats_.backendGates;
        ++stats_.rereadBlocks;
        return true;
    }

    // After a block applied without reading the state, the backend's state is
    // short of unit norm by a factor this layer keeps. A block that follows
    // divides it out; anything that reads the state needs it applied.
    void SettleNorm()
    {
        if (stateNormSq_ == 1.)
            return;
        const Eigen::MatrixXcd scale = Eigen::MatrixXcd::Identity(2, 2) / std::sqrt(stateNormSq_);
        stateNormSq_ = 1.;
        immediate_->ApplyLocalMapNorm({0}, scale);
    }

    void ApplyBlockMatrix(const Types::qubits_vector &targets, const Local &product)
    {
        if (targets.size() == 1)
            immediate_->ApplyGenericOneQubitGate(targets[0], Eigen::Matrix2cd(product));
        else if (targets.size() == 2)
            immediate_->ApplyGenericTwoQubitGate(targets[0], targets[1], Eigen::Matrix4cd(product));
        else
            immediate_->ApplyGenericThreeQubitGate(targets[0], targets[1], targets[2], Matrix8cd(product));
    }

    // Walks the steps of a block on the reduced density matrix of its targets,
    // normalized to unit trace: each channel factor draws its operator there.
    // Returns the product of the gates and of the operators selected, each
    // scaled by its weight, so that it leaves a state of unit norm with one.
    Local WalkTrajectoryBlock(const TrajectoryBlock &block, Local rho)
    {
        using Selected = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor, 4, 4>;
        const size_t dimension = size_t{1} << block.targets.size();
        Local product = Local::Identity(dimension, dimension);
        size_t undrawn = block.channels;
        for (const auto &step : block.steps)
        {
            if (step.unitary.size())
            {
                Cache::ApplyLeft(step.unitary, step.order, block.targets, product);
                if (undrawn)
                    Conjugate(step.unitary, step.order, block.targets, rho);
                continue;
            }
            // The weights of a factor need only the density matrix of its own targets.
            const auto &operators = step.channel ? step.channel->GetFactors()[step.factor].operators : step.own;
            const size_t sub = size_t{1} << step.order.size();
            size_t offsets[4] = {}, mask = 0;
            for (size_t i = 0; i < step.order.size(); ++i)
            {
                const size_t bit = size_t{1} << (std::find(block.targets.begin(), block.targets.end(), step.order[i]) - block.targets.begin());
                mask |= bit;
                for (size_t l = 0; l < sub; ++l)
                    if (l & (size_t{1} << i))
                        offsets[l] |= bit;
            }
            std::complex<double> local[4][4] = {};
            for (size_t base = 0; base < dimension; ++base)
            {
                if (base & mask)
                    continue;
                for (size_t a = 0; a < sub; ++a)
                    for (size_t b = 0; b < sub; ++b)
                        local[a][b] += rho(base | offsets[a], base | offsets[b]);
            }
            weights_.assign(operators.size(), 0.);
            double total = 0.;
            for (size_t k = 0; k < operators.size(); ++k)
            {
                const auto &op = operators[k];
                double weight = 0.;
                for (size_t row = 0; row < sub; ++row)
                    for (size_t a = 0; a < sub; ++a)
                        for (size_t b = 0; b < sub; ++b)
                            weight += std::real(op(row, a) * local[a][b] * std::conj(op(row, b)));
                weights_[k] = weight > 0. ? weight : 0.;
                total += weights_[k];
            }
            if (!std::isfinite(total) || total <= 0.)
                throw std::domain_error("No Kraus operator of a channel has any probability on this state");
            const double threshold = step.random * total;
            size_t chosen = 0;
            double accumulated = 0.;
            for (size_t k = 0; k < operators.size(); ++k)
            {
                if (weights_[k] <= 0.)
                    continue;
                chosen = k; // also the fallback if rounding leaves the threshold above the last sum
                accumulated += weights_[k];
                if (threshold < accumulated)
                    break;
            }
            // Scaled by its weight, the operator leaves the state with unit norm.
            const Selected selected = operators[chosen] / std::sqrt(weights_[chosen]);
            Cache::ApplyLeft(selected, step.order, block.targets, product);
            if (--undrawn)
                Conjugate(selected, step.order, block.targets, rho);
        }
        return product;
    }

    struct Context
    {
        long long source;
        std::vector<Operation> operations;
        bool normKnown = true;
    };

    struct Planned
    {
        Types::qubits_vector targets;
        Cache::Sources sources;
        size_t routingIndex;
    };

    struct RoutingPlan
    {
        std::vector<Operation> routing;
        std::vector<Planned> planned;
        // Source position -> routing position for non-gate boundaries.
        std::unordered_map<size_t, size_t> boundaries;
    };

    void Synchronize() const
    {
        const_cast<FusionState *>(this)->Flush();
    }

    bool AtBoundary() const
    {
        return sourceIndex_ >= 0 && static_cast<size_t>(sourceIndex_) < sources_.size() && sources_[sourceIndex_]->GetType() != Circuits::OperationType::kGate;
    }

    void FinishInitialization(size_t n, bool invalidateSnapshots = true)
    {
        // An unseeded native initialization may have generated a fresh root.
        // Keep the wrapper's readout stream attached to that same root.
        const auto seed = immediate_->GetConfiguration("sampling_seed");
        if (!seed.empty())
            SeedAuxiliaryRng(std::stoull(seed));
        ready_ = n != 0;
        sourceIndex_ = 0;
        if (invalidateSnapshots)
        {
            saved_.reset();
            destructiveSaved_.reset();
        }
        // Native initialization may have cleared its operation list.
        backendRoutingInstalled_ = false;
        RebuildPlan();
    }

    void Discard()
    {
        cache_.Clear();
        channelBlocks_.clear();
        trajectoryBlocks_.clear();
        stateNormSq_ = 1.; // the state is replaced
        normKnown_ = true;
        routingPlan_.reset();
        planValid_ = false;
    }

    void InvalidatePlan()
    {
        planValid_ = false;
        routingPlan_.reset();
        if (IsGateFusionEnabled() && backendRoutingInstalled_)
        {
            immediate_->SetUpcomingGates({});
            backendRoutingInstalled_ = false;
        }
    }

    void RebuildPlan()
    {
        routingPlan_.reset();
        planIndex_ = 0;
        planValid_ = false;
        if (!IsGateFusionEnabled())
        {
            if (!sources_.empty() || backendRoutingInstalled_)
                immediate_->SetUpcomingGates(sources_);
            backendRoutingInstalled_ = !sources_.empty();
            immediate_->SetGatesCounter(sourceIndex_);
            return;
        }
        if (!sources_.empty() && IsRoutingLookaheadEnabled())
            PreparePlan();
        else if (backendRoutingInstalled_)
        {
            immediate_->SetUpcomingGates({});
            backendRoutingInstalled_ = false;
        }
    }

    void PreparePlan()
    {
        planValid_ = false;
        routingPlan_.reset();
        auto prepared = std::make_shared<RoutingPlan>();
        planIndex_ = 0;
        const bool installRouting = IsRoutingLookaheadEnabled();
        Cache planner(GetGateFusionMaxQubits(), false);
        auto emit = [&prepared](const auto &block) {
            prepared->planned.push_back({block.targets, block.sources, prepared->routing.size()});
            const bool swap = block.IsSingle() && block.original.kind == Circuits::QuantumGateType::kSwapGateType;
            prepared->routing.push_back(std::make_shared<FusionRoutingOperation>(block.targets, block.schmidtRank, swap));
        };
        for (size_t i = static_cast<size_t>(sourceIndex_); i < sources_.size(); ++i)
        {
            const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(sources_[i]);
            if (gate && sources_[i]->GetType() == Circuits::OperationType::kGate)
            {
                const auto primitives = FusionGate::FromCircuit(*gate).Expand(GetGateFusionMaxQubits());
                for (size_t j = 0; j < primitives.size(); ++j)
                    SubmitToCache(planner, primitives[j], (uint64_t(i) << 4) | j, emit);
            }
            else
            {
                planner.Flush(emit);
                prepared->boundaries[i] = prepared->routing.size();
                prepared->routing.push_back(sources_[i]);
            }
        }
        planner.Flush(emit);
        routingPlan_ = prepared;
        planValid_ = installRouting;
        if (planValid_)
        {
            immediate_->SetUpcomingGates(prepared->routing);
            backendRoutingInstalled_ = !prepared->routing.empty();
        }
    }

    void Emit(const Cache::Block &block)
    {
        if (planValid_)
        {
            if (planIndex_ >= routingPlan_->planned.size() || routingPlan_->planned[planIndex_].sources != block.sources ||
                routingPlan_->planned[planIndex_].targets != block.targets)
                InvalidatePlan();
            else
                immediate_->SetGatesCounter(routingPlan_->planned[planIndex_].routingIndex);
        }
        if (block.IsSingle())
            block.original.Apply(*immediate_);
        else if (block.targets.size() == 1)
            immediate_->ApplyGenericOneQubitGate(block.targets[0], block.matrix);
        else if (block.targets.size() == 2)
            immediate_->ApplyGenericTwoQubitGate(block.targets[0], block.targets[1], block.matrix);
        else
            immediate_->ApplyGenericThreeQubitGate(block.targets[0], block.targets[1], block.targets[2], block.matrix);
        ++stats_.backendGates;
        if (!block.IsSingle())
            ++stats_.fusedBlocks;
        if (planValid_)
            ++planIndex_;
    }

    GateFusionStatistics stats_;

    const char *RequestedSetting() const
    {
        return !requested_ ? "auto" : *requested_ ? "true" : "false";
    }

    std::optional<bool> requested_; // unset: the backend default
    bool ready_ = false;
    Cache cache_;
    // Pending fused maps of a mixed state; no qubit is in two pending blocks of either kind.
    std::vector<ChannelBlock> channelBlocks_;
    // Pending blocks of a backend that samples channels from its state.
    std::vector<TrajectoryBlock> trajectoryBlocks_;
    std::vector<double> weights_;
    // The squared norm of the backend's state: below one after a block applied
    // without reading the state (ResolveWithoutReading), until the next block
    // or a flush restores it.
    double stateNormSq_ = 1.;
    // Whether the backend's state has the norm this layer accounts for (stateNormSq_). A generic gate or operator
    // need not be unitary and amplitudes given by the caller need not have unit norm: after any of them only a read
    // can tell the norm, so no block is applied without reading the state first (ResolveWithoutReading) until the
    // state is replaced. Resolving a block from the reduced density matrix it reads does not need the norm.
    bool normKnown_ = true;
    static constexpr double kNormTolerance = 1e-10;

    // Whether a matrix keeps the norm of every state: its columns are orthonormal.
    template <class Matrix> static bool KeepsNorm(const Matrix &matrix)
    {
        if (matrix.rows() != matrix.cols())
            return false;
        for (Eigen::Index j = 0; j < matrix.cols(); ++j)
            for (Eigen::Index i = 0; i <= j; ++i)
                if (!(std::abs(matrix.col(i).dot(matrix.col(j)) - (i == j ? 1. : 0.)) <= kNormTolerance))
                    return false;
        return true;
    }

    template <class Amplitudes> static bool HasUnitNorm(const Amplitudes &amplitudes)
    {
        double norm = 0.;
        const size_t size = static_cast<size_t>(amplitudes.size());
        for (size_t i = 0; i < size; ++i)
            norm += std::norm(amplitudes[i]);
        return std::abs(norm - 1.) <= kNormTolerance;
    }

    // Configuration "trajectory_blocks_read_first": every block of sampled
    // channels is resolved from the reduced density matrix, never applied
    // first. The branches drawn are the same either way; this is the reference
    // the other path is compared with.
    static constexpr const char *kReadBlocksFirst = "trajectory_blocks_read_first";
    bool readBlocksFirst_ = false;
    long long sourceIndex_ = 0;
    std::vector<Operation> sources_;
    // Gate/index metadata is immutable after preparation and can be shared by
    // compatible clones. Execution cursors and the pending fusion cache stay local.
    std::shared_ptr<const RoutingPlan> routingPlan_;
    size_t planIndex_ = 0;
    bool planValid_ = false;
    bool backendRoutingInstalled_ = false;
    std::optional<Context> saved_, destructiveSaved_;
    mutable std::unordered_map<std::string, std::string> config_;
};
} // namespace Simulators::Private
