#include "Simulators/Core/Factory.h"
#define INCLUDED_BY_FACTORY
#include "Network/SimpleDisconnectedNetwork.h"
#include "Simulators/Fusion/FusionSimulator.h"
#include "Simulators/QCSim/ImmediateQCSimSimulator.h"
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

struct RecordingImmediate : Private::ImmediateQCSimSimulator
{
    size_t installs = 0, copied = 0, callbacks = 0, flushes = 0;
    std::vector<Op> installed;

    RecordingImmediate()
    {
        auto original = meetingPositionCallback;
        meetingPositionCallback = [this, original](const auto &bonds) {
            ++callbacks;
            return original(bonds);
        };
    }

    void SetUpcomingGates(const std::vector<Op> &operations) override
    {
        ++installs;
        copied += operations.size();
        installed = operations;
        ImmediateQCSimSimulator::SetUpcomingGates(operations);
    }

    void Flush() override
    {
        ++flushes;
    }
};

struct RecordingState : Private::FusionState
{
    explicit RecordingState(std::shared_ptr<RecordingImmediate> raw) : FusionState(std::move(raw))
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
                auto raw = std::make_shared<RecordingImmediate>();
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
                const auto installs = raw->installs, copies = raw->copied;
                for (int i = 0; i < n; ++i)
                {
                    sim.ApplyCX(0, 3);
                    sim.Measure({0});
                    const auto flushes = raw->flushes;
                    sim.IncrementGatesCounter(); // delay/classical boundary
                    Check(raw->flushes == flushes, "moving the gate counter synchronized the backend");
                }
                Check(raw->installs == installs && raw->copied == copies, "ordinary boundaries reinstalled the prepared operation list");
                Check(observer->calls >= size_t(n), "preparation removed backend observer");
                Check(sim.GetGatesCounter() == 3 * n, "source counter lost boundaries");
                if (sim.IsRoutingLookaheadEnabled())
                    Check(raw->callbacks > 0, "lookahead was enabled but never called");
                if (std::string(method) == "statevector" && fusion)
                    Check(raw->installs == 0, "statevector unnecessarily prepared routing");
            }
}

static void MultishotNetwork()
{
    for (bool fusion : {false, true})
        for (auto method : {SimulationType::kMatrixProductState, SimulationType::kMatrixProductOperator})
            for (int preparation : {0, 1, 2, 3})
            {
                auto raw = std::make_shared<RecordingImmediate>();
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
                    Check(!raw->installed.empty(), "shot snapshot lost its routing list");
                    Check(raw->installed.front()->GetType() == Circuits::OperationType::kMeasurement,
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
    auto raw = std::make_shared<RecordingImmediate>();
    raw->AllocateQubits(2);
    raw->Initialize();
    auto circuit = CF::CreateCircuit({CF::CreateGate(Kind::kXGateType, 0)});
    Circuits::OperationState state;
    state.AllocateBits(2);
    circuit->Execute(raw, state);
    circuit->ExecuteNonMeasurements(raw, state);
    circuit->ExecuteMeasurements(raw, state, {});
    Check(raw->flushes == 0, "circuit completion flushed an unfused backend");
    raw->Clear();
    raw->Configure("method", "matrix_product_state");
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
        auto raw = std::make_shared<RecordingImmediate>();
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
        Check(raw->flushes == 0, "a fused circuit boundary waited for the backend");
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
    RecordingImmediate raw;
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
            auto raw = std::make_shared<RecordingImmediate>();
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
                Check(!raw->installed.empty(), "a deferred gate dropped the look-ahead routing");
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
        auto raw = std::make_shared<RecordingImmediate>();
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
            Check(!raw->installed.empty(), "a firing conditional gate dropped the look-ahead routing");
        Check(std::abs(sim->Probability(9) - 1.) < 1e-10, "the conditional gate was not applied");
    }
}

int main()
{
    try
    {
        BoundaryPlanning();
        MultishotNetwork();
        FlushScopeAndRoutingSettings();
        FusedBoundariesDoNotWait();
        CallbackFallback();
        DeferredGatesKeepRouting();
        FiringConditionalKeepsRouting();
        std::cout << "Routing reuse, multishot snapshots, observers, flush scope, "
                     "fused boundaries, callback fallback, deferred gates and "
                     "conditional gates passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
