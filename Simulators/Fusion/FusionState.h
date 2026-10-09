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

    void FlushPendingGates() override
    {
        cache_.Flush([this](const auto &block) { Emit(block); });
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
        // Initial placement may request a fused view even without lookahead.
        // Build it lazily; statevectors never need a routing plan.
        if (routing_.empty() && !sources_.empty() && IsGateFusionEnabled() &&
            (GetSimulationType() == SimulationType::kMatrixProductState || GetSimulationType() == SimulationType::kMatrixProductOperator))
            const_cast<FusionState *>(this)->PreparePlan();
        return routing_;
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
        routing_.clear();
        boundaryRouting_.clear();
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
        saved_ = Context{sourceIndex_, sources_};
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
        }
        RebuildPlan();
    }

    void SaveStateToInternalDestructive() override
    {
        Flush();
        immediate_->SaveStateToInternalDestructive();
        if (UsesDestructiveStateStorage())
            destructiveSaved_ = Context{sourceIndex_, sources_};
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
        Flush();
        immediate_->SetUseOptimalMeetingPosition(value);
        RebuildPlan();
    }

    void SetLookaheadDepth(int value) override
    {
        Flush();
        immediate_->SetLookaheadDepth(value);
        RebuildPlan();
    }

    void SetLookaheadDepthWithHeuristic(int value) override
    {
        Flush();
        immediate_->SetLookaheadDepthWithHeuristic(value);
        RebuildPlan();
    }

    void setGrowthFactorSwap(double value) override
    {
        Flush();
        immediate_->setGrowthFactorSwap(value);
    }

    void setGrowthFactorGate(double value) override
    {
        Flush();
        immediate_->setGrowthFactorGate(value);
    }

    void InitializeState(size_t n, std::vector<std::complex<double>> &value) override
    {
        if (n == 0)
        {
            immediate_->InitializeState(n, value);
            return;
        }
        Discard();
        immediate_->InitializeState(n, value);
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
        Discard();
        immediate_->InitializeState(n, value);
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
        Discard();
        immediate_->InitializeState(n, value);
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
        copy.cache_.SetWidth(copy.GetGateFusionMaxQubits());
        const auto seed = copy.immediate_->GetConfiguration("seed");
        if (!seed.empty())
            copy.SeedAuxiliaryRng(std::stoull(seed));
        copy.RebuildPlan();
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
            const auto boundary = boundaryRouting_.find(index);
            if (boundary != boundaryRouting_.end() && sources_[index]->GetType() == Circuits::OperationType::kConditionalGate)
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
        const auto source = static_cast<uint64_t>(sourceIndex_++);
        if (cache_.Empty())
            cache_.SetWidth(width);
        const auto emit = [this](const auto &block) { Emit(block); };
        if (gate.qubits.size() <= width) // Expand would return {gate}
            cache_.Submit(gate, source << 4, emit, PreserveStructuredGates() && gate.IsStructured());
        else
        {
            const auto primitives = gate.Expand(width);
            for (size_t i = 0; i < primitives.size(); ++i)
                cache_.Submit(primitives[i], (source << 4) | i, emit, PreserveStructuredGates() && primitives[i].IsStructured());
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
    struct Context
    {
        long long source;
        std::vector<Operation> operations;
    };

    struct Planned
    {
        Types::qubits_vector targets;
        Cache::Sources sources;
        size_t routingIndex;
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
        planned_.clear();
        routing_.clear();
        boundaryRouting_.clear();
        planValid_ = false;
    }

    void InvalidatePlan()
    {
        planValid_ = false;
        planned_.clear();
        routing_.clear();
        boundaryRouting_.clear();
        if (IsGateFusionEnabled() && backendRoutingInstalled_)
        {
            immediate_->SetUpcomingGates({});
            backendRoutingInstalled_ = false;
        }
    }

    void RebuildPlan()
    {
        planned_.clear();
        routing_.clear();
        boundaryRouting_.clear();
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
        planned_.clear();
        routing_.clear();
        boundaryRouting_.clear();
        planIndex_ = 0;
        planValid_ = IsRoutingLookaheadEnabled();
        Cache planner(GetGateFusionMaxQubits(), false);
        auto emit = [this](const auto &block) {
            planned_.push_back({block.targets, block.sources, routing_.size()});
            routing_.push_back(std::make_shared<FusionRoutingOperation>(block.targets));
        };
        for (size_t i = static_cast<size_t>(sourceIndex_); i < sources_.size(); ++i)
        {
            const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(sources_[i]);
            if (gate && sources_[i]->GetType() == Circuits::OperationType::kGate)
            {
                const auto primitives = FusionGate::FromCircuit(*gate).Expand(GetGateFusionMaxQubits());
                for (size_t j = 0; j < primitives.size(); ++j)
                    planner.Submit(primitives[j], (uint64_t(i) << 4) | j, emit);
            }
            else
            {
                planner.Flush(emit);
                boundaryRouting_[i] = routing_.size();
                routing_.push_back(sources_[i]);
            }
        }
        planner.Flush(emit);
        if (planValid_)
        {
            immediate_->SetUpcomingGates(routing_);
            backendRoutingInstalled_ = !routing_.empty();
        }
    }

    void Emit(const Cache::Block &block)
    {
        if (planValid_)
        {
            if (planIndex_ >= planned_.size() || planned_[planIndex_].sources != block.sources || planned_[planIndex_].targets != block.targets)
                InvalidatePlan();
            else
                immediate_->SetGatesCounter(planned_[planIndex_].routingIndex);
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
    long long sourceIndex_ = 0;
    std::vector<Operation> sources_;
    std::vector<Operation> routing_;
    std::vector<Planned> planned_;
    // routing position of every non-gate source operation of the current plan
    std::unordered_map<size_t, size_t> boundaryRouting_;
    size_t planIndex_ = 0;
    bool planValid_ = false;
    bool backendRoutingInstalled_ = false;
    std::optional<Context> saved_, destructiveSaved_;
    mutable std::unordered_map<std::string, std::string> config_;
};
} // namespace Simulators::Private
