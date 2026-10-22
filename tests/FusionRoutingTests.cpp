#include "Simulators/Core/Factory.h"
#define INCLUDED_BY_FACTORY
#include "Network/SimpleDisconnectedNetwork.h"
#include "Simulators/Fusion/FusionSimulator.h"
#include "Simulators/Gpu/GpuMPOSimulator.h"
#include "Simulators/Gpu/GpuMPSSimulator.h"
#include "Simulators/QCSim/QCSimMPOSimulator.h"
#include "Simulators/QCSim/QCSimMPSSimulator.h"
#include "Simulators/QCSim/QCSimStatevectorSimulator.h"
#include <iostream>

using namespace Simulators;
using Op = std::shared_ptr<Circuits::IOperation<>>;
using CF = Circuits::CircuitFactory<>;
using Kind = Circuits::QuantumGateType;

static void Check(bool ok, const char *message)
{
    if (!ok)
        throw std::runtime_error(message);
}

struct RecordingStats
{
    virtual ~RecordingStats() = default;
    virtual void SetNotificationsForTest(bool enabled) = 0;
    size_t installs = 0, copied = 0, callbacks = 0, flushes = 0;
    std::vector<Op> installed;
};

template <class Backend> struct RecordingImmediate : Backend, RecordingStats
{
    void SetNotificationsForTest(bool enabled) override
    {
        if (enabled)
            this->Notify();
        else
            this->DontNotify();
    }

    RecordingImmediate()
    {
        if constexpr (std::is_base_of_v<Private::QCSimTensorChainSimulator, Backend>)
        {
            auto original = this->meetingPositionCallback;
            this->meetingPositionCallback = [this, original](const auto &bonds) {
                ++callbacks;
                return original(bonds);
            };
        }
    }

    void SetUpcomingGates(const std::vector<Op> &operations) override
    {
        ++installs;
        copied += operations.size();
        installed = operations;
        Backend::SetUpcomingGates(operations);
    }

    void Flush() override
    {
        ++flushes;
    }

  protected:
#ifdef __linux__
    using GpuMeetingPositionCallback = int64_t (*)(void *, const int64_t *);

    GpuMeetingPositionCallback GetMeetingPositionCallback() const
    {
        return &RecordGpuMeetingPosition;
    }

    static int64_t RecordGpuMeetingPosition(void *context, const int64_t *bonds) noexcept
    {
        auto *self = static_cast<RecordingImmediate *>(static_cast<Private::GpuTensorChainSimulator *>(context));
        ++self->callbacks;
        return Backend::FindBestMeetingPosition(context, bonds);
    }
#endif
};

static std::shared_ptr<ISimulator> MakeRecording(const char *method, bool gpu = false)
{
#ifdef __linux__
    if (gpu)
    {
        if (std::string(method) == "matrix_product_state")
            return std::make_shared<RecordingImmediate<Private::GpuMPSSimulator>>();
        return std::make_shared<RecordingImmediate<Private::GpuMPOSimulator>>();
    }
#endif
    if (std::string(method) == "statevector")
        return std::make_shared<RecordingImmediate<Private::QCSimStatevectorSimulator>>();
    if (std::string(method) == "matrix_product_state")
        return std::make_shared<RecordingImmediate<Private::QCSimMPSSimulator>>();
    return std::make_shared<RecordingImmediate<Private::QCSimMPOSimulator>>();
}

static RecordingStats &Stats(const std::shared_ptr<ISimulator> &sim)
{
    return dynamic_cast<RecordingStats &>(*sim);
}

struct RecordingState : Private::FusionState
{
    explicit RecordingState(std::shared_ptr<ISimulator> raw) : FusionState(std::move(raw))
    {
    }

    unsigned GetGateFusionMaxQubits() const override
    {
        return 2;
    }
};

struct RecordingSimulator : Private::FusionSimulator<RecordingState>
{
    using FusionSimulator::FusionSimulator;

    std::unique_ptr<ISimulator> Clone() override
    {
        throw std::logic_error("Recording simulator is not cloned");
    }
};

struct Observer : ISimulatorObserver
{
    size_t calls = 0;

    void Update(const Types::qubits_vector &) override
    {
        ++calls;
    }
};

static void BoundaryPlanning()
{
    for (bool fusion : {false, true})
        for (const char *method : {"statevector", "matrix_product_state", "matrix_product_operator"})
            for (int n : {50, 100})
            {
                auto raw = MakeRecording(method);
                RecordingSimulator sim(raw);
                sim.Configure("method", method);
                sim.Configure("gate_fusion", fusion ? "true" : "false");
                sim.SetMultithreading(false);
                sim.AllocateQubits(4);
                sim.Initialize();
                sim.SetLookaheadDepth(2);
                std::vector<Op> operations;
                for (int i = 0; i < n; ++i)
                {
                    operations.push_back(CF::CreateGate(Kind::kCXGateType, 0, 3));
                    operations.push_back(CF::CreateMeasurement({{0, 0}}));
                    operations.push_back(CF::CreateDelay(1));
                }
                auto observer = std::make_shared<Observer>();
                raw->RegisterObserver(observer);
                sim.SetUpcomingGates(operations);
                const auto installs = Stats(raw).installs, copies = Stats(raw).copied;
                for (int i = 0; i < n; ++i)
                {
                    sim.ApplyCX(0, 3);
                    sim.Measure({0});
                    const auto flushes = Stats(raw).flushes;
                    sim.IncrementGatesCounter(); // delay/classical boundary
                    Check(Stats(raw).flushes == flushes, "moving the gate counter synchronized the backend");
                }
                Check(Stats(raw).installs == installs && Stats(raw).copied == copies, "ordinary boundaries reinstalled the prepared operation list");
                Check(observer->calls >= size_t(n), "preparation removed backend observer");
                Check(sim.GetGatesCounter() == 3 * n, "source counter lost boundaries");
                if (sim.IsRoutingLookaheadEnabled())
                    Check(Stats(raw).callbacks > 0, "lookahead was enabled but never called");
                if (std::string(method) == "statevector" && fusion)
                    Check(Stats(raw).installs == 0, "statevector unnecessarily prepared routing");
            }
}

static void MultishotNetwork()
{
    for (bool fusion : {false, true})
        for (auto method : {SimulationType::kMatrixProductState, SimulationType::kMatrixProductOperator})
            for (int preparation : {0, 1, 2, 3})
            {
                auto raw = MakeRecording(method == SimulationType::kMatrixProductState ? "matrix_product_state" : "matrix_product_operator");
                auto sim = std::make_shared<RecordingSimulator>(raw);
                sim->Configure("method", method == SimulationType::kMatrixProductState ? "matrix_product_state" : "matrix_product_operator");
                sim->Configure("gate_fusion", fusion ? "true" : "false");
                sim->AllocateQubits(preparation == 1 ? 2 : 4);
                sim->Initialize();
                sim->SetLookaheadDepth(2);
                auto circuit = CF::CreateCircuit({CF::CreateGate(Kind::kXGateType, 0), CF::CreateMeasurement({{0, 0}}), CF::CreateGate(Kind::kCXGateType, 0, 3),
                                                  CF::CreateMeasurement({{3, 1}})});
                sim->SetUpcomingGates(circuit->GetOperations());
                auto net = std::make_shared<Network::SimpleDisconnectedNetwork<>>(Types::qubits_vector{4}, std::vector<size_t>{2});
                net->SetMPSOptimizeSwaps(true);
                net->SetInitialQubitsMapOptimization(false);
                net->SetMPSOptimizationQubitsNumberThreshold(0);
                net->SetMPSOptimizationBondDimensionThreshold(0);
                net->SetLookaheadDepth(2);
                Circuits::Circuit<>::ExecuteResults results;
                std::mutex mutex;
                Network::ExecuteJob<> job(circuit, results, 3, 4, 2, 2, SimulatorType::kQCSim, method, mutex);
                job.network = net;
                job.optSim = sim;
                job.executedGates.assign(circuit->size(), false);
                job.config.SetConfiguration("method", method == SimulationType::kMatrixProductState ? "matrix_product_state" : "matrix_product_operator");
                if (preparation >= 2)
                    Estimators::SimulatorsEstimatorInterface<>::ExecuteUpToMeasurements(circuit, 4, 2, 2, sim, job.executedGates);
                if (preparation == 3)
                    job.DoWork();
                else
                    job.DoWorkNoLock();
                sim->RestoreState();
                Check(sim->GetGatesCounter() == 0, "shot snapshot points inside the full circuit");
                if (!fusion || sim->IsRoutingLookaheadEnabled())
                {
                    Check(!Stats(raw).installed.empty(), "shot snapshot lost its routing list");
                    Check(Stats(raw).installed.front()->GetType() == Circuits::OperationType::kMeasurement,
                          "shot snapshot restored the prefix instead of the suffix");
                }
                sim->Measure({0});
                sim->ApplyCX(0, 3);
                sim->Flush();
                Check(std::abs(sim->Probability(9) - 1.) < 1e-10, "restored shot changed state");
            }
}

static void FlushScopeAndRoutingSettings()
{
    auto raw = MakeRecording("statevector");
    raw->AllocateQubits(2);
    raw->Initialize();
    auto circuit = CF::CreateCircuit({CF::CreateGate(Kind::kXGateType, 0)});
    Circuits::OperationState state;
    state.AllocateBits(2);
    circuit->Execute(raw, state);
    circuit->ExecuteNonMeasurements(raw, state);
    circuit->ExecuteMeasurements(raw, state, {});
    Check(Stats(raw).flushes == 0, "circuit completion flushed an unfused backend");
    raw = MakeRecording("matrix_product_state");
    raw->AllocateQubits(4);
    raw->Initialize();
    raw->SetUseOptimalMeetingPosition(false);
    raw->SetLookaheadDepth(3);
    Check(!raw->IsRoutingLookaheadEnabled(), "setting depth overrode explicit routing disable");
}

// Circuit boundaries only emit pending fused blocks. Waiting for the backend
// (a device sync on GPU) is left to reads of the state.
static void FusedBoundariesDoNotWait()
{
    for (const char *method : {"statevector", "matrix_product_state"})
    {
        auto raw = MakeRecording(method);
        auto sim = std::make_shared<RecordingSimulator>(raw);
        sim->Configure("method", method);
        sim->Configure("gate_fusion", "true");
        sim->SetMultithreading(false);
        sim->AllocateQubits(4);
        sim->Initialize();
        const auto x = [](Types::qubit_t q) { return std::static_pointer_cast<Circuits::IGateOperation<>>(CF::CreateGate(Kind::kXGateType, q)); };
        auto circuit = CF::CreateCircuit({CF::CreateGate(Kind::kHadamardGateType, 0), CF::CreateGate(Kind::kCXGateType, 0, 1), CF::CreateDelay(1),
                                          CF::CreateNoOp(), CF::CreateConditionalGate(x(2), CF::CreateEqualCondition({1}, {true})),
                                          CF::CreateConditionalGate(x(2), CF::CreateEqualCondition({0}, {false})),
                                          CF::CreateCircuit({CF::CreateGate(Kind::kXGateType, 3)})});
        Circuits::OperationState state;
        state.AllocateBits(2);
        circuit->Execute(sim, state);
        Check(Stats(raw).flushes == 0, "a fused circuit boundary waited for the backend");
        Check(std::abs(sim->Probability(12) - .5) < 1e-10 && std::abs(sim->Probability(15) - .5) < 1e-10, "boundaries changed the fused circuit's result");
    }
}

// Exercise the exception barrier using an operation whose routing inspection
// fails; the quantum gate itself remains valid and must use native fallback.
struct ThrowingRoutingGate : Circuits::CXGate<>
{
    ThrowingRoutingGate() : CXGate(0, 3)
    {
    }

    Types::qubits_vector AffectedQubits() const override
    {
        throw std::runtime_error("unavailable routing metadata");
    }
};

static void CallbackFallback()
{
    RecordingImmediate<Private::QCSimMPSSimulator> raw;
    raw.Configure("method", "matrix_product_state");
    raw.AllocateQubits(4);
    raw.Initialize();
    raw.SetLookaheadDepth(2);
    raw.SetUpcomingGates({std::make_shared<ThrowingRoutingGate>()});
    raw.ApplyX(0);
    raw.SetGatesCounter(0);
    raw.ApplyCX(0, 3);
    Check(std::abs(raw.Probability(9) - 1.) < 1e-10, "callback fallback lost gate");
    if (raw.IsRoutingLookaheadEnabled())
        Check(raw.callbacks == 1, "exception fallback callback never ran");
}

static void PrepareChain(RecordingSimulator &sim, const char *method, bool fusion, size_t qubits)
{
    sim.Configure("method", method);
    sim.Configure("gate_fusion", fusion ? "true" : "false");
    sim.SetMultithreading(false);
    sim.AllocateQubits(qubits);
    sim.Initialize();
    sim.SetLookaheadDepth(4);
}

// Gates that depend on a mid-circuit measurement are deferred to the shots
// while the rest of the prefix still runs. With fusion, stepping over them
// used to drop the look-ahead routing for the remainder of the prefix.
static void DeferredGatesKeepRouting()
{
    for (const char *method : {"matrix_product_state", "matrix_product_operator"})
    {
        std::vector<double> reference;
        for (bool fusion : {false, true})
        {
            auto raw = MakeRecording(method);
            auto sim = std::make_shared<RecordingSimulator>(raw);
            PrepareChain(*sim, method, fusion, 6);
            std::vector<Op> operations;
            for (Types::qubit_t q = 0; q < 6; ++q)
                operations.push_back(CF::CreateGate(Kind::kHadamardGateType, q));
            operations.push_back(CF::CreateGate(Kind::kCXGateType, 0, 5));
            operations.push_back(CF::CreateMeasurement({{0, 0}}));
            operations.push_back(CF::CreateGate(Kind::kCXGateType, 0, 4)); // deferred
            operations.push_back(CF::CreateGate(Kind::kCXGateType, 1, 5));
            operations.push_back(CF::CreateGate(Kind::kCZGateType, 2, 5));
            operations.push_back(CF::CreateGate(Kind::kCXGateType, 3, 1));
            auto circuit = CF::CreateCircuit(operations);
            sim->SetUpcomingGates(circuit->GetOperations());
            Circuits::OperationState state;
            state.AllocateBits(1);
            const auto executed = circuit->ExecuteNonMeasurements(sim, state);
            // the mask starts at the measurement, the first operation not executed
            Check(executed == std::vector<bool>{false, false, true, true, true}, "unexpected executed mask");
            if (fusion && sim->IsRoutingLookaheadEnabled())
                Check(!Stats(raw).installed.empty(), "a deferred gate dropped the look-ahead routing");
            const auto probabilities = sim->AllProbabilities();
            if (!fusion)
                reference = probabilities;
            else
                for (size_t i = 0; i < reference.size(); ++i)
                    Check(std::abs(reference[i] - probabilities[i]) < 1e-10, "planned execution order changed the prefix state");
        }
    }
}

// A conditional gate that fires is applied as a plan boundary and keeps the
// look-ahead routing of the rest of the shot.
static void FiringConditionalKeepsRouting()
{
    for (const char *method : {"matrix_product_state", "matrix_product_operator"})
    {
        auto raw = MakeRecording(method);
        auto sim = std::make_shared<RecordingSimulator>(raw);
        PrepareChain(*sim, method, true, 6);
        sim->ApplyX(0);
        const auto flip = std::static_pointer_cast<Circuits::IGateOperation<>>(CF::CreateGate(Kind::kXGateType, 3));
        auto suffix = CF::CreateCircuit({CF::CreateMeasurement({{0, 0}}), CF::CreateConditionalGate(flip, CF::CreateEqualCondition({0}, {true})),
                                         CF::CreateGate(Kind::kCXGateType, 1, 5), CF::CreateGate(Kind::kCXGateType, 2, 4)});
        sim->SetUpcomingGates(suffix->GetOperations());
        Circuits::OperationState state;
        state.AllocateBits(1);
        suffix->ExecuteMeasurements(sim, state, std::vector<bool>(suffix->size(), false));
        if (sim->IsRoutingLookaheadEnabled())
            Check(!Stats(raw).installed.empty(), "a firing conditional gate dropped the look-ahead routing");
        Check(std::abs(sim->Probability(9) - 1.) < 1e-10, "the conditional gate was not applied");
    }
}

template <class Backend> struct HistoryProbe : RecordingImmediate<Backend>
{
    mutable size_t mapReads = 0, bondReads = 0;
    bool failMapRead = false;

    const void *Planner() const
    {
        return this->dummySim.get();
    }

    long long RoutingCap() const
    {
        return this->physicalBondCap;
    }

    void InvalidateHistory()
    {
        this->InvalidateDummyHistory();
    }

    void CorruptMap()
    {
        auto map = this->dummySim->getQubitsMap();
        std::swap(map[0], map[8]);
        this->dummySim->SetRoutingMap(map);
    }

    std::vector<long long> PredictedMap() const
    {
        return this->dummySim->getQubitsMap();
    }

    std::vector<double> VirtualBonds() const
    {
        return this->CaptureDummyHistory();
    }

    std::vector<double> NativeBonds() const
    {
        return Backend::CurrentBondDimensions();
    }

    std::vector<long long> NativeMap() const
    {
        return Backend::CurrentRoutingMap();
    }

  protected:
    std::vector<long long> CurrentRoutingMap() const override
    {
        ++mapReads;
        if (failMapRead)
            throw std::runtime_error("test routing read failure");
        return Backend::CurrentRoutingMap();
    }

    std::vector<double> CurrentBondDimensions() const override
    {
        ++bondReads;
        return Backend::CurrentBondDimensions();
    }
};

// Access inherited protected state through member pointers, so a real backend
// clone can be inspected without pretending it is a HistoryProbe instance.
template <class Backend> struct CloneRoutingInspection : Backend
{
    static auto NativeMap(const Backend &sim)
    {
        return (sim.*(&CloneRoutingInspection::CurrentRoutingMap))();
    }

    static auto VirtualBonds(const Backend &sim)
    {
        return (sim.*(&CloneRoutingInspection::CaptureDummyHistory))();
    }

    static auto PredictedMap(const Backend &sim)
    {
        return (sim.*(&CloneRoutingInspection::dummySim))->getQubitsMap();
    }
};

// The planner counts each CP(pi) as a rank-2 gate, but every pair on the same
// qubits cancels, so the estimated (virtual) bonds outgrow the real ones. A
// small-angle CP no longer works here: its effective rank is close to 1.
static void GrowVirtualHistory(ISimulator &sim)
{
    sim.ApplyH(0);
    sim.ApplyH(1);
    sim.ApplyCP(0, 1, M_PI);
    sim.ApplyH(2);
    sim.ApplyCP(1, 2, M_PI);
    sim.ApplyCX(2, 3);
    sim.ApplyCP(1, 2, M_PI);
    sim.ApplyH(4);
    sim.ApplyCP(3, 4, M_PI);
    sim.ApplyCX(4, 5);
    sim.ApplyCP(3, 4, M_PI);
}

enum class SwapRepresentation
{
    Named,
    Matrix,
    Operator
};

template <class Backend>
static void LogicalSwapHistory(const char *cap, SwapRepresentation representation = SwapRepresentation::Named, bool doublePrecision = true)
{
    HistoryProbe<Backend> sim;
    if (sim.GetType() == SimulatorType::kGpuSim)
        sim.Configure("use_double_precision", doublePrecision ? "true" : "false");
    sim.AllocateQubits(9);
    sim.Configure("mps_routing_virtual_bonds", "true");
    sim.Configure("matrix_product_state_max_bond_dimension", cap);
    sim.Configure("matrix_product_operator_max_bond_dimension", cap);
    sim.Initialize();
    sim.SetLookaheadDepth(3);
    sim.setGrowthFactorGate(1.);
    GrowVirtualHistory(sim);
    const auto history = sim.VirtualBonds(), bonds = sim.NativeBonds();
    Check(!history.empty(), "logical SWAP test needs virtual history");
    Check(sim.SwapGatesRelabelQubits(), "tensor chain did not advertise SWAP relabeling");
    auto map = sim.NativeMap();
    const auto before = sim.AllProbabilities();
    const auto planner = sim.Planner();
    sim.SetUpcomingGates({CF::CreateGate(Kind::kSwapGateType, 0, 8), CF::CreateGate(Kind::kSwapGateType, 2, 3), CF::CreateGate(Kind::kCXGateType, 0, 4)});
    const auto callbacks = sim.callbacks;
    for (const auto &[a, b] : {std::pair{0, 8}, std::pair{2, 3}})
    {
        if (representation == SwapRepresentation::Operator)
            sim.ApplyOperator({static_cast<Types::qubit_t>(a), static_cast<Types::qubit_t>(b)}, Circuits::SwapGate<>(a, b).GetMatrix(), false);
        else if (representation == SwapRepresentation::Matrix)
            sim.ApplyGenericTwoQubitGate(a, b, Circuits::SwapGate<>(a, b).GetMatrix());
        else
            sim.ApplySwap(a, b);
        std::swap(map[a], map[b]);
        Check(sim.NativeMap() == map && sim.PredictedMap() == map, "logical SWAP history map differs from backend");
        Check(sim.NativeBonds() == bonds && sim.VirtualBonds() == history, "logical SWAP changed bonds or discarded virtual history");
        Check(sim.callbacks == callbacks && sim.Planner() == planner, "logical SWAP routed qubits or replaced the planner");
    }
    Check(sim.GetGatesCounter() == 2, "logical SWAP did not advance the gate counter once");
    const auto after = sim.AllProbabilities();
    for (size_t i = 0; i < before.size(); ++i)
    {
        size_t permuted = i;
        for (const auto &[a, b] : {std::pair{0, 8}, std::pair{2, 3}})
            if (((permuted >> a) & 1) != ((permuted >> b) & 1))
                permuted ^= (size_t{1} << a) | (size_t{1} << b);
        Check(std::abs(after[permuted] - before[i]) < (doublePrecision ? 1e-10 : 1e-5), "logical SWAP changed the state beyond relabeling");
    }
#ifdef __linux__
    if constexpr (std::is_base_of_v<Private::GpuTensorChainSimulator, Backend>)
        for (const auto &[a, b] : {std::pair{0, 0}, std::pair{0, 9}})
        {
            bool rejected = false;
            try
            {
                sim.ApplySwap(a, b);
            }
            catch (const std::runtime_error &)
            {
                rejected = true;
            }
            Check(rejected && sim.GetGatesCounter() == 2 && sim.NativeMap() == map && sim.VirtualBonds() == history,
                  "invalid GPU SWAP was accepted or changed routing history");
        }
#endif
    sim.SaveState();
    auto clone = sim.Clone();
    auto &cloned = dynamic_cast<Backend &>(*clone);
    using Inspect = CloneRoutingInspection<Backend>;
    Check(clone->SwapGatesRelabelQubits(), "clone lost SWAP capability");
    Check(Inspect::NativeMap(cloned) == map && Inspect::PredictedMap(cloned) == map && Inspect::VirtualBonds(cloned) == history,
          "clone lost relabeled routing history");
    sim.ApplySwap(1, 7);
    sim.RestoreState();
    Check(sim.NativeMap() == map && sim.PredictedMap() == map && sim.VirtualBonds() == history, "restore lost relabeled routing history");
    sim.SetGatesCounter(2);
    clone->ApplyCX(0, 4);
    Check(Inspect::PredictedMap(cloned) == Inspect::NativeMap(cloned), "clone routing ignored relabeled map");
    Check(sim.NativeMap() == map && sim.VirtualBonds() == history, "clone routing changed original history");
    sim.ApplyCX(0, 4);
    Check(sim.callbacks > callbacks && sim.GetGatesCounter() == 3, "routing did not resume after logical SWAPs");
    Check(sim.PredictedMap() == sim.NativeMap(), "later routing ignored the relabeled map");
    sim.CorruptMap();
    sim.ApplySwap(0, 8);
    Check(sim.PredictedMap() == sim.NativeMap() && sim.VirtualBonds() == sim.NativeBonds(), "SWAP accepted unrelated map drift");
}

template <class Backend> static void PhasedSwapHistory(const char *cap, bool doublePrecision = true)
{
    HistoryProbe<Backend> sim;
    if (sim.GetType() == SimulatorType::kGpuSim)
        sim.Configure("use_double_precision", doublePrecision ? "true" : "false");
    sim.AllocateQubits(9);
    sim.Configure("mps_routing_virtual_bonds", "true");
    sim.Configure("matrix_product_state_max_bond_dimension", cap);
    sim.Configure("matrix_product_operator_max_bond_dimension", cap);
    sim.Initialize();
    sim.SetLookaheadDepth(3);
    sim.setGrowthFactorGate(1.);
    GrowVirtualHistory(sim);
    const auto before = sim.VirtualBonds();
    const auto map = sim.NativeMap();
    const Eigen::Matrix4cd phased = std::complex<double>(0., 1.) * Circuits::SwapGate<>(6, 7).GetMatrix();
    sim.ApplyGenericTwoQubitGate(6, 7, phased);
    Check(sim.NativeMap() == map && sim.PredictedMap() == map, "phased SWAP was incorrectly relabeled");
    Check(!sim.VirtualBonds().empty() && sim.VirtualBonds() != before, "phased SWAP was not recorded as a tensor gate");
}

template <class Backend> static void FusedMatrixSwapHistory(const char *cap, bool doublePrecision = true)
{
    auto raw = std::make_shared<HistoryProbe<Backend>>();
    RecordingSimulator sim(raw);
    if (sim.GetType() == SimulatorType::kGpuSim)
        sim.Configure("use_double_precision", doublePrecision ? "true" : "false");
    sim.Configure("gate_fusion", "true");
    sim.Configure("mps_routing_virtual_bonds", "true");
    sim.Configure("matrix_product_state_max_bond_dimension", cap);
    sim.Configure("matrix_product_operator_max_bond_dimension", cap);
    sim.AllocateQubits(9);
    sim.Initialize();
    sim.SetLookaheadDepth(3);
    sim.setGrowthFactorGate(1.);
    GrowVirtualHistory(*raw);
    const auto history = raw->VirtualBonds(), bonds = raw->NativeBonds();
    auto map = raw->NativeMap();
    sim.SetUpcomingGates({CF::CreateGate(Kind::kCXGateType, 0, 8), CF::CreateGate(Kind::kCXGateType, 8, 0), CF::CreateGate(Kind::kCXGateType, 0, 8),
                          CF::CreateGate(Kind::kCXGateType, 0, 4)});
    const auto callbacks = raw->callbacks;
    sim.ApplyCX(0, 8);
    sim.ApplyCX(8, 0);
    sim.ApplyCX(0, 8);
    sim.Flush();
    std::swap(map[0], map[8]);
    Check(sim.GetGateFusionStatistics().backendGates == 1, "CX SWAP decomposition did not fuse into one matrix");
    Check(raw->callbacks == callbacks && raw->NativeMap() == map && raw->PredictedMap() == map, "fused matrix SWAP did not relabel exactly");
    Check(raw->NativeBonds() == bonds && raw->VirtualBonds() == history, "fused matrix SWAP changed bond history");
    sim.ApplyCX(0, 4);
    sim.Flush();
    Check(raw->PredictedMap() == raw->NativeMap(), "routing after a fused matrix SWAP lost the relabeled map");
}

static void LogicalSwapFusion(bool gpu = false, bool doublePrecision = true)
{
    for (auto method : {SimulationType::kStatevector, SimulationType::kDensityMatrix})
    {
        auto sim = SimulatorsFactory::CreateSimulator(gpu ? SimulatorType::kGpuSim : SimulatorType::kQCSim, method);
        Check(!sim->SwapGatesRelabelQubits(), "non-chain backend advertised SWAP relabeling");
        sim->Configure("gate_fusion", "true");
        sim->AllocateQubits(4);
        sim->Initialize();
        sim->ApplyH(0);
        sim->ApplySwap(0, 1);
        sim->ApplyH(1);
        sim->Flush();
        Check(sim->GetGateFusionStatistics().backendGates == 1, "SWAP stopped fusing on a non-chain backend");
    }
    for (bool fusion : {false, true})
        for (const char *method : {"matrix_product_state", "matrix_product_operator"})
        {
            auto raw = MakeRecording(method, gpu);
            if (gpu)
                raw->Configure("use_double_precision", doublePrecision ? "true" : "false");
            auto sim = std::make_shared<RecordingSimulator>(raw);
            auto reference = MakeRecording("statevector");
            sim->Configure("gate_fusion", fusion ? "true" : "false");
            sim->Configure("matrix_product_state_max_bond_dimension", "64");
            sim->Configure("matrix_product_operator_max_bond_dimension", "64");
            for (const auto &s : std::vector<std::shared_ptr<ISimulator>>{sim, reference})
            {
                s->AllocateQubits(6);
                s->Initialize();
            }
            sim->SetLookaheadDepth(3);
            Check(sim->SwapGatesRelabelQubits(), "fusion wrapper lost SWAP capability");
            const std::vector<Op> ops{
                CF::CreateGate(Kind::kHadamardGateType, 0),      CF::CreateGate(Kind::kRyGateType, 2, 0, 0, 0.3), CF::CreateGate(Kind::kSwapGateType, 0, 5),
                CF::CreateGate(Kind::kRyGateType, 0, 0, 0, 0.7), CF::CreateGate(Kind::kCXGateType, 5, 4),         CF::CreateGate(Kind::kSwapGateType, 4, 1),
                CF::CreateGate(Kind::kCXGateType, 1, 3),         CF::CreateGate(Kind::kSwapGateType, 0, 2),       CF::CreateGate(Kind::kHadamardGateType, 2)};
            sim->SetUpcomingGates(ops);
            if (fusion)
            {
                size_t swaps = 0;
                for (const auto &op : Stats(raw).installed)
                    if (const auto info = std::dynamic_pointer_cast<RoutingGateInfo>(op); info && info->isSwap)
                        ++swaps;
                Check(swaps == 3, "fusion planner merged a relabeling SWAP into a matrix");
            }
            Circuits::OperationState state;
            for (size_t i = 0; i < ops.size(); ++i)
            {
                ops[i]->Execute(sim, state);
                ops[i]->Execute(reference, state);
                if (i == 2)
                {
                    Check(Stats(raw).callbacks == 0, "fusion turned a relabeling SWAP into a routed matrix");
                    if (fusion)
                        Check(sim->GetGateFusionStatistics().backendGates == 2, "standalone SWAP flushed a disjoint pending gate");
                }
            }
            const auto actual = sim->AllProbabilities(), expected = reference->AllProbabilities();
            for (size_t i = 0; i < actual.size(); ++i)
                Check(std::abs(actual[i] - expected[i]) < (doublePrecision ? 1e-10 : 1e-5), "fusion or routing changed a circuit with logical SWAPs");
        }
}

#ifdef __linux__
template <class Backend> static void CachedGpuRoutingSettings()
{
    // Configuration replay must not depend on unordered alias iteration.
    // This check requires no initialized device or GPU plugin.
    HistoryProbe<Backend> sim;
    Check(!sim.SwapGatesRelabelQubits(), "uninitialized GPU tensor chain assumed SWAP relabeling");
    sim.Configure("matrix_product_state_max_bond_dimension", "2");
    Check(sim.RoutingCap() == 2 && sim.GetConfiguration("matrix_product_operator_max_bond_dimension") == "2", "MPS cap alias left stale GPU settings");
    sim.Configure("matrix_product_operator_max_bond_dimension", "4");
    Check(sim.RoutingCap() == 4 && sim.GetConfiguration("matrix_product_state_max_bond_dimension") == "4", "MPO cap alias left stale GPU settings");
    HistoryProbe<Backend> replay;
    for (const auto &[key, value] : sim.GetConfigMap())
        replay.Configure(key.c_str(), value.c_str());
    Check(replay.RoutingCap() == 4, "GPU cap changed during configuration replay");
}
#endif

template <class Backend> static void VirtualHistoryBoundaries(const char *cap)
{
    HistoryProbe<Backend> sim;
    sim.AllocateQubits(9);
    sim.Configure("mps_routing_virtual_bonds", "true");
    sim.Configure("matrix_product_state_max_bond_dimension", cap);
    sim.Configure("matrix_product_operator_max_bond_dimension", cap);
    sim.Configure("matrix_product_state_truncation_threshold", "1e-16");
    sim.Configure("matrix_product_operator_truncation_threshold", "1e-16");
    sim.SetMultithreading(false);
    sim.Initialize();
    sim.setGrowthFactorGate(1.);
    sim.setGrowthFactorSwap(1.);
    sim.SetLookaheadDepth(2);
    const Types::qubits_vector all{0, 1, 2, 3, 4, 5, 6, 7, 8};

    for (int boundary = 0; boundary < 10; ++boundary)
    {
        sim.Reset();
        GrowVirtualHistory(sim);
        const auto history = sim.VirtualBonds();
        Check(history.size() == 8 && history[1] > sim.NativeBonds()[1], "test prefix did not build virtual pressure");
        sim.SaveState();
        switch (boundary)
        {
        case 0:
            sim.ApplyReset(all);
            break;
        case 1:
            sim.Measure(all);
            break;
        case 2:
            sim.MeasureMany(all);
            break;
        case 3:
            sim.MoveAtBeginningOfChain({8});
            Check(sim.NativeMap()[8] == 0, "explicit reorder did not move the requested qubit");
            break;
        case 4:
            sim.Trim();
            break;
        case 5:
            sim.ReCanonicalize();
            break;
        case 6:
            if (!sim.SupportsQuantumChannels())
                continue;
            sim.ApplyQuantumChannel({1, 3}, QuantumChannel::TwoQubitDepolarizing(1.));
            break;
        case 7:
            if (!sim.SupportsQuantumChannels())
                continue;
            sim.ApplyOperator({1}, Eigen::Matrix2cd::Identity());
            break;
        case 8:
        case 9:
            // These CPU MPS sampling paths reorder the selected sites.
            if constexpr (std::is_base_of_v<Private::QCSimMPSSimulator, Backend>)
            {
                if (boundary == 8)
                    sim.SampleCounts({7, 8}, 4);
                else
                    sim.SampleCountsMany({7, 8}, 4);
                Check(sim.NativeMap()[7] == 0 && sim.NativeMap()[8] == 1, "sampling did not exercise chain reordering");
            }
            else
                continue;
            break;
        }
        Check(sim.VirtualBonds().empty(), "state boundary retained stale virtual bonds");
        const auto map = sim.NativeMap();
        const auto q0 = static_cast<Types::qubit_t>(std::find(map.begin(), map.end(), 2) - map.begin());
        const auto q1 = static_cast<Types::qubit_t>(std::find(map.begin(), map.end(), 3) - map.begin());
        sim.ApplyH(q0);
        sim.ApplyCX(q0, q1);
        Check(sim.VirtualBonds() == sim.NativeBonds(), "first gate after a boundary did not seed from the native state");
        sim.RestoreState();
        Check(sim.VirtualBonds() == history, "history invalidation destroyed the saved snapshot");
    }

    // Saving while history is invalid must not resurrect a later gate's history.
    sim.ApplyReset(all);
    sim.SaveState();
    GrowVirtualHistory(sim);
    sim.RestoreState();
    Check(sim.VirtualBonds().empty(), "restoring a snapshot without history retained later virtual bonds");
    sim.ApplyH(2);
    sim.ApplyCX(2, 3);
    Check(sim.VirtualBonds() == sim.NativeBonds(), "restored state was not used to seed history");

    // Disabled lookahead, disabled optimal routing, legacy mode, unlimited bonds,
    // and caps without extra virtual capacity must avoid native map/bond queries.
    for (int mode = 0; mode < 6; ++mode)
    {
        // QCSim's MPS setter rejects zero; MPO exposes unlimited bonds as zero.
        if (mode == 3 && !sim.SupportsQuantumChannels())
            continue;
        sim.Reset();
        sim.Configure("mps_routing_legacy_cap", "false");
        sim.Configure("mps_routing_virtual_bonds", "true");
        sim.Configure("matrix_product_state_max_bond_dimension", cap);
        sim.Configure("matrix_product_operator_max_bond_dimension", cap);
        sim.SetUseOptimalMeetingPosition(true);
        sim.SetLookaheadDepth(2);
        GrowVirtualHistory(sim);
        if (mode == 0)
            sim.SetLookaheadDepth(0);
        else if (mode == 1)
            sim.SetUseOptimalMeetingPosition(false);
        else if (mode == 2)
            sim.Configure("mps_routing_legacy_cap", "true");
        else if (mode == 5)
            sim.Configure("mps_routing_virtual_bonds", "false");
        else
        {
            sim.Configure("matrix_product_state_max_bond_dimension", mode == 3 ? "0" : "256");
            sim.Configure("matrix_product_operator_max_bond_dimension", mode == 3 ? "0" : "256");
        }
        sim.mapReads = sim.bondReads = 0;
        for (int i = 0; i < 100; ++i)
            sim.ApplyCX(2, 3);
        Check(sim.mapReads == 0 && sim.bondReads == 0, "inactive virtual tracking queried the native chain");
        Check(sim.VirtualBonds().empty(), "inactive virtual tracking retained stale history");
        sim.Configure("mps_routing_legacy_cap", "false");
        sim.Configure("mps_routing_virtual_bonds", "true");
        sim.Configure("matrix_product_state_max_bond_dimension", cap);
        sim.Configure("matrix_product_operator_max_bond_dimension", cap);
        sim.SetUseOptimalMeetingPosition(true);
        sim.SetLookaheadDepth(2);
        sim.ApplyCX(2, 3);
        Check(sim.VirtualBonds() == sim.NativeBonds(), "reenabling tracking reused stale history");
    }
}

template <class Backend> static void RoutingHistoryRecovery(const char *cap)
{
    HistoryProbe<Backend> sim;
    sim.AllocateQubits(9);
    sim.Configure("matrix_product_state_max_bond_dimension", cap);
    sim.Configure("matrix_product_operator_max_bond_dimension", cap);
    sim.Initialize();
    sim.SetLookaheadDepth(2);
    sim.setGrowthFactorGate(1.);
    sim.setGrowthFactorSwap(1.);
    GrowVirtualHistory(sim);
    Check(sim.VirtualBonds().empty(), "virtual history must be opt-in");
    sim.Configure("mps_routing_virtual_bonds", "true");
    sim.Reset();
    GrowVirtualHistory(sim);
    const auto history = sim.VirtualBonds();
    const auto *planner = sim.Planner();
    Check(!history.empty(), "opt-in did not enable history");
    sim.SaveState();
    sim.InvalidateHistory();
    Check(sim.Planner() == planner, "history invalidation destroyed the planner/cache");
    sim.RestoreState();
    Check(sim.Planner() == planner && sim.VirtualBonds() == history, "restore replaced the planner or lost history");

    // Both gate operands remain adjacent; an unrelated part of the map differs.
    // A check of the active pair alone cannot catch this drift.
    sim.CorruptMap();
    sim.ApplyCX(2, 3);
    Check(sim.PredictedMap() == sim.NativeMap() && sim.VirtualBonds() == sim.NativeBonds(), "full-map mismatch did not reseed native bonds");

    const std::vector<Op> oneGate{CF::CreateGate(Kind::kCXGateType, 2, 3)};
    sim.SetUpcomingGates(oneGate);
    sim.failMapRead = true;
    sim.ApplyCX(2, 3);
    sim.failMapRead = false;
    Check(sim.GetGatesCounter() == 1 && sim.VirtualBonds().empty(), "advisory recording failure suppressed gate notification");

    // Reset must follow a nonidentity native map, not assume logical order.
    sim.MoveAtBeginningOfChain({8});
    sim.Reset();
    Check(sim.PredictedMap() == sim.NativeMap(), "reset installed a different map from the backend");
    Check(sim.Planner() == planner, "reset discarded the planner cache");

    if constexpr (std::is_base_of_v<Private::QCSimMPSSimulator, Backend>)
    {
        sim.Initialize();
        GrowVirtualHistory(sim);
        sim.Configure("mps_sample_measure_algorithm", "mps_apply_measure");
        const auto beforeSampling = sim.VirtualBonds();
        const auto probabilities = sim.AllProbabilities();
        sim.SaveState();
        sim.SampleCounts({2, 3}, 8);
        Check(sim.VirtualBonds() == beforeSampling, "measure/restore sampling discarded history");
        sim.SampleCountsMany({2, 3}, 8);
        Check(sim.VirtualBonds() == beforeSampling, "wide measure/restore sampling discarded history");
        const auto after = sim.AllProbabilities();
        for (size_t i = 0; i < after.size(); ++i)
            Check(std::abs(after[i] - probabilities[i]) < 1e-10, "sampling changed the restored state");
        sim.Reset();
        sim.RestoreState();
        Check(sim.VirtualBonds() == beforeSampling, "sampling overwrote the caller's saved snapshot");
    }
}

static void DecompositionAndFusionRanks(bool gpu = false)
{
    for (const char *method : {"matrix_product_state", "matrix_product_operator"})
    {
        auto raw = MakeRecording(method, gpu);
        auto reference = MakeRecording("statevector");
        raw->Configure("mps_routing_virtual_bonds", "true");
        if (gpu)
            raw->Configure("use_double_precision", "true");
        for (const auto &sim : {raw, reference})
        {
            sim->AllocateQubits(6);
            sim->Configure("matrix_product_state_max_bond_dimension", "64");
            sim->Initialize();
            sim->SetLookaheadDepth(2);
        }
        const std::vector<Op> ops{CF::CreateGate(Kind::kHadamardGateType, 0), CF::CreateGate(Kind::kXGateType, 4), CF::CreateGate(Kind::kCCXGateType, 0, 4, 2),
                                  CF::CreateGate(Kind::kCSwapGateType, 0, 1, 5), CF::CreateGate(Kind::kCXGateType, 1, 3)};
        raw->SetUpcomingGates(ops);
        Circuits::OperationState state;
        for (size_t i = 0; i < ops.size(); ++i)
        {
            ops[i]->Execute(raw, state);
            ops[i]->Execute(reference, state);
            Check(raw->GetGatesCounter() == static_cast<long long>(i + 1), "decomposition advanced past its logical operation");
        }
        const auto actual = raw->AllProbabilities(), expected = reference->AllProbabilities();
        for (size_t i = 0; i < actual.size(); ++i)
            Check(std::abs(actual[i] - expected[i]) < 1e-10, "decomposition/routing changed the circuit result");
        raw->SetGatesCounter(0);
        Stats(raw).SetNotificationsForTest(false);
        raw->ApplyCCX(0, 4, 2);
        raw->ApplyCX(1, 3);
        Check(raw->GetGatesCounter() == 0, "decomposition enabled suppressed observers");
        Stats(raw).SetNotificationsForTest(true);
        raw->ApplyCSwap(0, 1, 5);
        Check(raw->GetGatesCounter() == 1, "decomposition failed to restore observer delivery");
    }

    auto raw = MakeRecording("matrix_product_state", gpu);
    auto fused = std::make_shared<RecordingSimulator>(raw);
    fused->AllocateQubits(6);
    fused->Initialize();
    fused->Configure("gate_fusion", "true");
    fused->SetLookaheadDepth(2);
    fused->SetUpcomingGates({CF::CreateGate(Kind::kCXGateType, 0, 2), CF::CreateGate(Kind::kSwapGateType, 1, 3)});
    const auto &routing = Stats(raw).installed;
    Check(routing.size() == 2, "isolated gates unexpectedly merged");
    const auto first = std::dynamic_pointer_cast<RoutingGateInfo>(routing[0]);
    const auto second = std::dynamic_pointer_cast<RoutingGateInfo>(routing[1]);
    const auto cx = first && !first->isSwap ? first : second;
    const auto swap = first && first->isSwap ? first : second;
    Check(cx && cx->schmidtRank == 2 && !cx->isSwap && swap && swap->isSwap, "fusion discarded single-gate rank/swap metadata");
    const auto &cxOperation = routing[first == cx ? 0 : 1];
    const auto copied = std::dynamic_pointer_cast<RoutingGateInfo>(cxOperation->Clone());
    Check(copied && copied->schmidtRank == 2, "routing metadata lost on clone");
    for (bool mpo : {false, true})
    {
        MPSDummySimulator predicted(6, mpo), actual(6, mpo);
        predicted.setGrowthFactorGate(1.);
        actual.setGrowthFactorGate(1.);
        predicted.SetCurrentBondDimensions({2, 2, 2, 2, 2});
        actual.SetCurrentBondDimensions({2, 2, 2, 2, 2});
        predicted.ApplyGate(cxOperation);
        actual.ApplyGate(CF::CreateGate(Kind::kCXGateType, 0, 2));
        Check(predicted.getCurrentBondDimensions() == actual.getCurrentBondDimensions(), "planner rank differs from the emitted native gate");
    }
}

static void NonGateRoutingFallbacks(bool gpu = false)
{
    for (bool fusion : {false, true})
        for (const char *method : {"matrix_product_state", "matrix_product_operator"})
        {
            const bool mpo = std::string(method) == "matrix_product_operator";
            for (bool channel : {false, true})
            {
                if (channel && !mpo)
                    continue;
                auto raw = MakeRecording(method, gpu);
                // The GPU defaults to fp32; the state is compared at 1e-10.
                if (gpu)
                    raw->Configure("use_double_precision", "true");
                auto sim = std::make_shared<RecordingSimulator>(raw);
                PrepareChain(*sim, method, fusion, 4);
                std::vector<Op> operations;
                if (channel)
                    operations.push_back(
                        std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{1, 3}, QuantumChannel::TwoQubitDepolarizing(0.2)));
                else
                {
                    operations.push_back(CF::CreateGate(Kind::kHadamardGateType, 1));
                    const auto cx = std::static_pointer_cast<Circuits::IGateOperation<>>(CF::CreateGate(Kind::kCXGateType, 1, 3));
                    operations.push_back(CF::CreateConditionalGate(cx, CF::CreateEqualCondition({0}, {false})));
                }
                operations.push_back(CF::CreateGate(Kind::kCXGateType, 0, 2));
                auto circuit = CF::CreateCircuit(operations);
                sim->SetUpcomingGates(operations);
                Circuits::OperationState state;
                state.AllocateBits(1);
                circuit->Execute(sim, state);
                sim->Flush();
                if (!gpu)
                    Check(Stats(raw).callbacks > 0, "non-gate routing test did not exercise the callback");

                auto reference = SimulatorsFactory::CreateSimulator(SimulatorType::kQCSim, mpo ? SimulationType::kDensityMatrix : SimulationType::kStatevector);
                reference->AllocateQubits(4);
                reference->Initialize();
                Circuits::OperationState referenceState;
                referenceState.AllocateBits(1);
                circuit->Execute(reference, referenceState);
                const auto expected = reference->AllProbabilities(), actual = sim->AllProbabilities();
                Check(expected.size() == actual.size(), "fallback probability size mismatch");
                for (size_t i = 0; i < expected.size(); ++i)
                    Check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 1e-10, "non-gate fallback changed the quantum state");
            }
        }

    MPSDummySimulator dummy(4);
    double cost = std::numeric_limits<double>::infinity();
    Check(dummy.FindBestMeetingPosition({}, 0, 2, 0, 0., cost) == -1, "empty lookahead returned a meeting position");
    const std::vector<Op> oneQubit{CF::CreateGate(Kind::kHadamardGateType, 1)};
    Check(dummy.FindBestMeetingPosition(oneQubit, 0, 2, 0, 0., cost) == -1, "empty two-qubit window returned a meeting position");
    const std::vector<Op> channel{std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{1, 3}, QuantumChannel::TwoQubitDepolarizing(0.2))};
    Check(dummy.FindBestMeetingPosition(channel, 0, 2, 0, 0., cost) == -1, "channel boundary returned a meeting position");
}

int main(int argc, char **argv)
{
    try
    {
#ifdef __linux__
        CachedGpuRoutingSettings<Private::GpuMPSSimulator>();
        CachedGpuRoutingSettings<Private::GpuMPOSimulator>();
        if (argc > 1 && std::string(argv[1]) == "--gpu")
        {
            if (!SimulatorsFactory::InitGpuLibraryWithMute())
                return 77;
            NonGateRoutingFallbacks(true);
            VirtualHistoryBoundaries<Private::GpuMPSSimulator>("2");
            VirtualHistoryBoundaries<Private::GpuMPOSimulator>("4");
            RoutingHistoryRecovery<Private::GpuMPSSimulator>("2");
            RoutingHistoryRecovery<Private::GpuMPOSimulator>("4");
            DecompositionAndFusionRanks(true);
            for (bool doublePrecision : {false, true})
            {
                for (auto representation : {SwapRepresentation::Named, SwapRepresentation::Matrix})
                {
                    LogicalSwapHistory<Private::GpuMPSSimulator>("2", representation, doublePrecision);
                    LogicalSwapHistory<Private::GpuMPOSimulator>("4", representation, doublePrecision);
                }
                LogicalSwapHistory<Private::GpuMPOSimulator>("4", SwapRepresentation::Operator, doublePrecision);
                PhasedSwapHistory<Private::GpuMPSSimulator>("2", doublePrecision);
                PhasedSwapHistory<Private::GpuMPOSimulator>("4", doublePrecision);
                FusedMatrixSwapHistory<Private::GpuMPSSimulator>("2", doublePrecision);
                FusedMatrixSwapHistory<Private::GpuMPOSimulator>("4", doublePrecision);
                LogicalSwapFusion(true, doublePrecision);
            }
            std::cout << "GPU routing fallbacks, virtual history boundaries and inactive tracking passed\n";
            return 0;
        }
#endif
        BoundaryPlanning();
        MultishotNetwork();
        FlushScopeAndRoutingSettings();
        FusedBoundariesDoNotWait();
        CallbackFallback();
        DeferredGatesKeepRouting();
        FiringConditionalKeepsRouting();
        NonGateRoutingFallbacks();
        VirtualHistoryBoundaries<Private::QCSimMPSSimulator>("2");
        VirtualHistoryBoundaries<Private::QCSimMPOSimulator>("4");
        RoutingHistoryRecovery<Private::QCSimMPSSimulator>("2");
        RoutingHistoryRecovery<Private::QCSimMPOSimulator>("4");
        DecompositionAndFusionRanks();
        for (auto representation : {SwapRepresentation::Named, SwapRepresentation::Matrix})
        {
            LogicalSwapHistory<Private::QCSimMPSSimulator>("2", representation);
            LogicalSwapHistory<Private::QCSimMPOSimulator>("4", representation);
        }
        LogicalSwapHistory<Private::QCSimMPOSimulator>("4", SwapRepresentation::Operator);
        PhasedSwapHistory<Private::QCSimMPSSimulator>("2");
        PhasedSwapHistory<Private::QCSimMPOSimulator>("4");
        FusedMatrixSwapHistory<Private::QCSimMPSSimulator>("2");
        FusedMatrixSwapHistory<Private::QCSimMPOSimulator>("4");
        LogicalSwapFusion();
        std::cout << "Routing reuse, multishot snapshots, observers, flush scope, "
                     "fused boundaries, callback fallback, deferred gates and "
                     "conditional gates, virtual history boundaries and inactive tracking passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
