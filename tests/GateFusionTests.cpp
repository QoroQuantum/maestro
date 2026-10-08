#include "Simulators/Core/Factory.h"
#include "Simulators/Fusion/FusionGate.h"
#include <Eigen/QR>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#ifdef __linux__
#include "Simulators/DistributedGpu/DistributedGpuLibStateVectorSim.h"
#endif

using namespace Simulators;
using Method = SimulationType;
using Backend = SimulatorType;
using Gate = FusionGate;
using Kind = Gate::Kind;
using Vec = Eigen::VectorXcd;

void Check(bool value, const std::string &message)
{
    if (!value)
        throw std::runtime_error(message);
}

void Close(const Vec &a, const Vec &b, const std::string &message, double eps = 2e-9)
{
    Check(a.size() == b.size() && (a - b).norm() < eps, message + " error=" + std::to_string((a - b).norm()));
}

std::shared_ptr<ISimulator> Make(Method method, bool fusion = true, Backend backend = Backend::kQCSim, size_t n = 4)
{
    auto sim = SimulatorsFactory::CreateSimulator(backend, method);
    Check(bool(sim), "Simulator unavailable");
    sim->Configure("gate_fusion", fusion ? "true" : "false");
    sim->Configure("matrix_product_state_max_bond_dimension", "256");
    sim->Configure("matrix_product_state_truncation_threshold", "0");
    sim->Configure("use_double_precision", "true");
    sim->SetMultithreading(false);
    sim->AllocateQubits(n);
    sim->Initialize();
    sim->SetSeed(42);
    return sim;
}

Vec State(ISimulator &sim)
{
    const size_t n = size_t{1} << sim.GetNumberOfQubits();
    Vec result(n);
    for (size_t i = 0; i < n; ++i)
        result[i] = sim.Amplitude(i);
    return result;
}

// Independent bit-index application, without the cache's embedding utility.
Vec ApplyDense(const Vec &input, const Gate &gate)
{
    const auto qs = gate.Targets();
    const auto m = gate.Matrix();
    Vec output = Vec::Zero(input.size());
    for (size_t col = 0; col < static_cast<size_t>(input.size()); ++col)
    {
        size_t local = 0, cleared = col;
        for (size_t k = 0; k < qs.size(); ++k)
        {
            local |= ((col >> qs[k]) & 1) << k;
            cleared &= ~(size_t{1} << qs[k]);
        }
        for (size_t row = 0; row < (size_t{1} << qs.size()); ++row)
        {
            size_t global = cleared;
            for (size_t k = 0; k < qs.size(); ++k)
                global |= ((row >> k) & 1) << qs[k];
            output[global] += m(row, local) * input[col];
        }
    }
    return output;
}

Eigen::MatrixXcd Unitary(size_t dim, std::mt19937 &rng)
{
    std::normal_distribution<double> normal;
    Eigen::MatrixXcd m(dim, dim);
    for (Eigen::Index i = 0; i < m.size(); ++i)
        m.data()[i] = {normal(rng), normal(rng)};
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(m);
    return qr.householderQ() * Eigen::MatrixXcd::Identity(dim, dim);
}

void Verify(ISimulator &sim, const Vec &expected, const std::string &name)
{
    const auto method = sim.GetSimulationType();
    if (method == Method::kDensityMatrix || method == Method::kMatrixProductOperator)
    {
        Types::qubits_vector qs(sim.GetNumberOfQubits());
        std::iota(qs.begin(), qs.end(), 0);
        const auto rho = sim.PartialTrace(qs);
        Check((rho - expected * expected.adjoint()).norm() < 3e-8, name + " density matrix");
    }
    else if (method == Method::kTensorNetwork)
    {
        for (Eigen::Index i = 0; i < expected.size(); ++i)
            Check(std::abs(sim.Probability(i) - std::norm(expected[i])) < 2e-8, name + " probabilities");
    }
    else
        Close(State(sim), expected, name);
}

void Algebra()
{
    {
        GateFusion<Gate> cache(2);
        size_t emitted = 0;
        auto emit = [&](const auto &) { ++emitted; };
        cache.Submit({Kind::kCXGateType, {0, 1}, {}, {}}, 0, emit);
        cache.Submit({Kind::kHadamardGateType, {2}, {}, {}}, 1, emit);
        cache.Submit({Kind::kCXGateType, {1, 2}, {}, {}}, 2, emit);
        cache.Flush(emit);
        Check(emitted == 2, "width conflict prematurely flushed a compatible single-qubit gate");
    }
    {
        // Merges reuse the oldest block's entry; a source older than every block
        // it merges with takes the fallback path. Both must keep the blocks, the
        // qubit owners and the matrices consistent.
        GateFusion<Gate> cache(3);
        std::vector<GateFusion<Gate>::Sources> emitted;
        Vec actual = Vec::Zero(16), expected = actual;
        actual[0] = expected[0] = 1;
        auto emit = [&](const auto &block) {
            auto gate = block.IsSingle() ? block.original : Gate{Kind::kNone, block.targets, {}, block.matrix};
            actual = ApplyDense(actual, gate);
            emitted.push_back(block.sources);
        };
        const std::vector<std::pair<Gate, uint64_t>> submissions{
            {{Kind::kHadamardGateType, {0}, {}, {}}, 20}, {{Kind::kHadamardGateType, {1}, {}, {}}, 30},
            {{Kind::kCXGateType, {0, 1}, {}, {}}, 5},                                                 // older than both blocks
            {{Kind::kRyGateType, {2}, {.4}, {}}, 40},     {{Kind::kCZGateType, {1, 2}, {}, {}}, 50},  // merges into key 5
            {{Kind::kXGateType, {3}, {}, {}}, 60},        {{Kind::kCXGateType, {3, 0}, {}, {}}, 70}}; // too wide: emits key 5
        for (const auto &[gate, source] : submissions)
        {
            expected = ApplyDense(expected, gate);
            cache.Submit(gate, source, emit);
        }
        Check(emitted.size() == 1 && emitted[0].first == 5 && emitted[0].count == 5, "merged block lost its sources");
        cache.Flush(emit);
        Check(cache.Empty() && emitted.size() == 2 && emitted[1].first == 60 && emitted[1].count == 2, "in-place merge left stale blocks or owners");
        Close(actual, expected, "in-place merge algebra");
    }
    std::mt19937 rng(19);
    // The in-place left application must equal the dense embedded product for
    // every gate size, target order and block size.
    for (size_t blockQubits = 1; blockQubits <= 3; ++blockQubits)
        for (size_t k = 1; k <= blockQubits; ++k)
            for (int trial = 0; trial < 20; ++trial)
            {
                Types::qubits_vector to{4, 1, 7};
                to.resize(blockQubits);
                auto order = to;
                std::shuffle(order.begin(), order.end(), rng);
                order.resize(k);
                const auto gate = Unitary(size_t{1} << k, rng);
                const Eigen::MatrixXcd start = Unitary(size_t{1} << blockQubits, rng);
                Eigen::MatrixXcd actual = start;
                GateFusion<Gate>::ApplyLeft(gate, order, to, actual);
                const Eigen::MatrixXcd expected = GateFusion<Gate>::Embed(gate, order, to) * start;
                Check((actual - expected).norm() < 1e-12, "fusion left application");
                Vec basis = Vec::Zero(size_t{1} << 8);
                for (size_t col = 0; col < (size_t{1} << blockQubits); ++col)
                {
                    // Embed agrees with independent bit-index application.
                    size_t index = 0;
                    for (size_t b = 0; b < blockQubits; ++b)
                        if (col & (size_t{1} << b))
                            index |= size_t{1} << to[b];
                    basis.setZero();
                    basis[index] = 1;
                    const Vec applied = ApplyDense(basis, Gate{Kind::kNone, order, {}, gate});
                    const Eigen::MatrixXcd embedded = GateFusion<Gate>::Embed(gate, order, to);
                    for (size_t row = 0; row < (size_t{1} << blockQubits); ++row)
                    {
                        size_t out = 0;
                        for (size_t b = 0; b < blockQubits; ++b)
                            if (row & (size_t{1} << b))
                                out |= size_t{1} << to[b];
                        Check(std::abs(applied[out] - embedded(row, col)) < 1e-12, "fusion embedding");
                    }
                }
            }
    for (unsigned width : {2u, 3u})
    {
        GateFusion<Gate> cache(width), planner(width, false);
        Vec actual = Vec::Zero(32), expected = actual;
        actual[0] = expected[0] = 1;
        std::vector<GateFusion<Gate>::Sources> emitted, planned;
        auto emit = [&](const auto &block) {
            auto gate = block.IsSingle() ? block.original : Gate{Kind::kNone, block.targets, {}, block.matrix};
            actual = ApplyDense(actual, gate);
            emitted.push_back(block.sources);
        };
        auto forecast = [&](const auto &block) { planned.push_back(block.sources); };
        for (uint64_t i = 0; i < 180; ++i)
        {
            const size_t arity = 1 + rng() % width;
            Types::qubits_vector qs{0, 1, 2, 3, 4};
            std::shuffle(qs.begin(), qs.end(), rng);
            qs.resize(arity);
            Gate gate{Kind::kNone, qs, {}, Unitary(size_t{1} << arity, rng)};
            expected = ApplyDense(expected, gate);
            cache.Submit(gate, i, emit);
            planner.Submit(gate, i, forecast);
        }
        cache.Flush(emit);
        planner.Flush(forecast);
        Close(actual, expected, "random cache algebra");
        Check(emitted == planned, "planner differs from execution");
    }
}

std::vector<Gate> Sequence()
{
    return {{Kind::kHadamardGateType, {0}, {}, {}},
            {Kind::kRyGateType, {3}, {.43}, {}},
            {Kind::kCXGateType, {0, 3}, {}, {}},
            {Kind::kRzGateType, {0}, {.29}, {}},
            {Kind::kCYGateType, {3, 0}, {}, {}},
            {Kind::kUGateType, {2}, {.41, .21, -.31, .17}, {}},
            {Kind::kCXGateType, {0, 2}, {}, {}},
            {Kind::kCSwapGateType, {0, 3, 2}, {}, {}},
            {Kind::kCCXGateType, {2, 3, 1}, {}, {}},
            {Kind::kSwapGateType, {1, 0}, {}, {}},
            {Kind::kCUGateType, {3, 1}, {.19, .39, .77, -.21}, {}},
            {Kind::kSxDagGateType, {1}, {}, {}},
            {Kind::kKGateType, {2}, {}, {}}};
}

void Backends()
{
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductState, Method::kMatrixProductOperator, Method::kTensorNetwork})
    {
        auto fused = Make(method);
        auto plain = Make(method, false);
        Check(fused->IsGateFusionEnabled(), "default eligibility");
        const unsigned width = method == Method::kStatevector || method == Method::kDensityMatrix ? 3 : 2;
        Check(fused->GetGateFusionMaxQubits() == width, "capability width");
        Vec expected = Vec::Zero(16);
        expected[0] = 1;
        for (const auto &gate : Sequence())
        {
            gate.Apply(*fused);
            gate.Apply(*plain);
            expected = ApplyDense(expected, gate);
        }
        Verify(*fused, expected, "fused backend " + std::to_string(int(method)));
        Verify(*plain, expected, "plain backend " + std::to_string(int(method)));
        Check(fused->GetGateFusionStatistics().fusedBlocks > 0, "no fusion performed");
    }
    auto composite = Make(Method::kStatevector, true, Backend::kCompositeQCSim);
    Vec expected = Vec::Zero(16);
    expected[0] = 1;
    for (const auto &gate : Sequence())
    {
        gate.Apply(*composite);
        expected = ApplyDense(expected, gate);
    }
    Verify(*composite, expected, "composite");
    composite->SaveState();
    auto copy = composite->Clone();
    composite->Measure({0});
    composite->ApplyReset({3});
    composite->ApplyH(2);
    composite->RestoreState();
    Verify(*composite, expected, "composite restore after split");
    Verify(*copy, expected, "composite clone");
    auto child = SimulatorsFactory::CreateImmediateSimulatorUnique(Backend::kQCSim);
    Check(!child->IsGateFusionEnabled() && child->GetGateFusionMaxQubits() == 0, "child must be immediate");
}

// Replacing native storage invalidates its snapshots; resetting it does not.
void SnapshotLifetime(Backend backend = Backend::kQCSim)
{
    for (bool fusion : {false, true})
    {
        for (int replacement = 0; replacement < 7; ++replacement)
        {
            auto sim = Make(Method::kStatevector, fusion, backend);
            sim->SaveState();
            if (replacement == 0)
                sim->Initialize();
            if (replacement == 1)
            {
                std::vector<std::complex<double>> zero(16);
                zero[0] = 1;
                sim->InitializeState(4, zero);
            }
            if (replacement == 2)
            {
                Vec zero = Vec::Zero(16);
                zero[0] = 1;
                sim->InitializeState(4, zero);
            }
            if (replacement == 3)
                sim->InitializeToBasisState(4, Types::qubit_t{0});
            if (replacement == 4)
                sim->InitializeToBasisState(4, std::vector<bool>(4));
            if (replacement == 5)
            {
                sim->Clear();
                sim->AllocateQubits(4);
                sim->Initialize();
            }
            if (replacement == 6)
            {
                sim->Reset();
                sim->ApplyX(0);
                sim->RestoreState();
                Check(std::abs(sim->Probability(0) - 1) < 1e-12, "reset must retain the native snapshot");
                continue;
            }
            sim->ApplyX(0);
            sim->RestoreState();
            Check(std::abs(sim->Probability(1) - 1) < 1e-12, "state replacement left stale snapshot metadata");
        }
    }
}

void NativeCloneCounters(Backend backend)
{
    for (const auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        auto original = SimulatorsFactory::CreateImmediateSimulatorUnique(backend, method);
        original->Configure("use_double_precision", "true");
        original->AllocateQubits(4);
        original->Initialize();
        original->SetUpcomingGates({Circuits::CircuitFactory<>::CreateGate(Kind::kXGateType, 0)});
        auto copy = original->Clone();
        copy->ApplyX(0);
        Check(original->GetGatesCounter() == 0 && copy->GetGatesCounter() == 1, "clone observer increments the original counter");
        original.reset();
        copy->ApplyCX(0, 3);
        Check(copy->GetGatesCounter() == 2 && std::abs(copy->Probability(9) - 1.) < 1e-10, "clone observer depends on original lifetime");
    }
}

void AdditionalSnapshotLifetimes(Backend backend)
{
    for (bool fusion : {false, true})
    {
        // MPO mixture imports replace the native saved state too.
        for (bool bits : {false, true})
        {
            auto sim = Make(Method::kMatrixProductOperator, fusion, backend);
            sim->SaveState();
            if (bits)
                sim->InitializeToMixtureOfBasisStates(4, std::vector<std::pair<std::vector<bool>, double>>{{std::vector<bool>(4), 1.}});
            else
                sim->InitializeToMixtureOfBasisStates(4, std::vector<std::pair<Types::qubit_t, double>>{{0, 1.}});
            sim->ApplyX(0);
            try
            {
                sim->RestoreState();
            }
            catch (const std::runtime_error &e)
            {
                // GPU MPO reports a missing snapshot as an error. Fusion must retain
                // the newly submitted gate even when native restore rejects the call.
                Check(backend == Backend::kGpuSim && std::string(e.what()) == "GPU matrix-product-operator state restore failed",
                      "unexpected failure restoring after mixture import");
            }
            Check(std::abs(sim->Probability(1) - 1.) < 1e-10, "mixture import retained stale snapshot metadata");
        }
        if (backend == Backend::kGpuSim)
        {
            auto sim = Make(Method::kStatevector, fusion, backend);
            sim->SaveStateToInternalDestructive();
            sim->Initialize();
            sim->ApplyX(0);
            sim->RestoreInternalDestructiveSavedState();
            Check(std::abs(sim->Probability(1) - 1.) < 1e-10, "reinitialization retained stale destructive snapshot metadata");
        }
        else
        {
            auto sim = Make(Method::kStatevector, fusion, Backend::kCompositeQCSim);
            sim->ApplyH(0);
            sim->SaveState();
            sim->Initialize();
            sim->ApplyX(0);
            sim->RestoreState();
            Check(std::abs(sim->Probability(0) - .5) < 1e-10, "composite initialization lost its surviving native snapshot");
        }
    }
}

void GenericTargetOrder(Backend backend, Method method)
{
    for (bool fusion : {false, true})
    {
        auto sim = Make(method, fusion, backend);
        Vec expected = Vec::Zero(16);
        expected[0] = 1;
        std::mt19937 rng(48);
        for (int i = 0; i < 8; ++i)
        {
            Gate gate{Kind::kNone, i % 2 ? Types::qubits_vector{3, 0} : Types::qubits_vector{0, 3}, {}, Unitary(4, rng)};
            gate.Apply(*sim);
            expected = ApplyDense(expected, gate);
        }
        Verify(*sim, expected, "asymmetric generic matrices with both target orders");
    }
}

void GenericAndLifecycle(Backend backend = Backend::kQCSim)
{
    std::mt19937 rng(33);
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        auto sim = Make(method, true, backend);
        Vec expected = Vec::Zero(16);
        expected[0] = 1;
        for (int i = 0; i < 30; ++i)
        {
            Types::qubits_vector qs = i % 3 ? Types::qubits_vector{3, 0} : Types::qubits_vector{0};
            Gate gate{Kind::kNone, qs, {}, Unitary(size_t{1} << qs.size(), rng)};
            gate.Apply(*sim);
            expected = ApplyDense(expected, gate);
        }
        sim->SaveState();
        auto copy = sim->Clone();
        sim->ApplyX(2);
        sim->ApplyCX(0, 2);
        sim->RestoreState();
        Verify(*sim, expected, "restore pending");
        Verify(*copy, expected, "clone pending");
        sim->ApplyH(1);
        expected = ApplyDense(expected, {Kind::kHadamardGateType, {1}, {}, {}});
        sim->Configure("gate_fusion", "false");
        Verify(*sim, expected, "disable flush");
        sim->Configure("gate_fusion", "true");
        sim->ApplyY(0);
        sim->Reset();
        expected = Vec::Zero(16);
        expected[0] = 1;
        Verify(*sim, expected, "reset pending");
        bool rejected = false;
        try
        {
            sim->ApplyCX(1, 1);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        Check(rejected, "duplicate targets accepted");
    }
    for (auto method : {Method::kStatevector, Method::kDensityMatrix})
    {
        if (backend == Backend::kGpuSim && method == Method::kDensityMatrix)
            continue;
        auto sim = Make(method, true, backend);
        Vec expected = Vec::Zero(16);
        expected[0] = 1;
        for (const auto &qs : {Types::qubits_vector{3, 0, 2}, Types::qubits_vector{2, 3, 0}})
        {
            Gate gate{Kind::kNone, qs, {}, Unitary(8, rng)};
            gate.Apply(*sim);
            expected = ApplyDense(expected, gate);
        }
        Verify(*sim, expected, "three-qubit generic permutation");
    }
}

void Routing(Backend backend = Backend::kQCSim)
{
    for (auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        auto sim = Make(method, true, backend, 6);
        sim->SetInitialQubitsMap({5, 3, 0, 4, 1, 2});
        sim->SetLookaheadDepth(2);
        sim->SetLookaheadDepthWithHeuristic(1);
        std::vector<std::shared_ptr<Circuits::IOperation<>>> circuit;
        std::vector<Gate> gates;
        for (int i = 0; i < 18; ++i)
        {
            const auto a = Types::qubit_t(i % 6), b = Types::qubit_t((i + 3) % 6);
            gates.push_back({Kind::kHadamardGateType, {a}, {}, {}});
            gates.push_back({Kind::kCXGateType, {a, b}, {}, {}});
            gates.push_back({Kind::kRzGateType, {b}, {.13}, {}});
            gates.push_back({Kind::kCYGateType, {b, a}, {}, {}});
        }
        for (const auto &g : gates)
            circuit.push_back(Circuits::CircuitFactory<>::CreateGate(g.kind, g.qubits[0], g.qubits.size() > 1 ? g.qubits[1] : 0, 0, g.params[0]));
        sim->SetUpcomingGates(circuit);
        Check(sim->GetUpcomingRoutingOperations().size() < gates.size(), "routing sequence not fused");
        Vec expected = Vec::Zero(64);
        expected[0] = 1;
        for (size_t i = 0; i < gates.size(); ++i)
        {
            gates[i].Apply(*sim);
            expected = ApplyDense(expected, gates[i]);
            if (i == 12)
                Verify(*sim, expected, "early observation");
            if (i == 25)
            {
                sim->SaveState();
                sim->ApplyX(0);
                sim->RestoreState();
            }
        }
        Verify(*sim, expected, "routing with fusion");
        Check(sim->GetGatesCounter() == static_cast<long long>(gates.size()), "source cursor misaligned");
        Check(sim->GetGateFusionStatistics().backendGates < gates.size(), "routing prevents fusion");
    }
}

void Channels()
{
    auto fused = Make(Method::kDensityMatrix), plain = Make(Method::kDensityMatrix, false);
    for (auto sim : {fused, plain})
    {
        sim->ApplyH(0);
        sim->ApplyCX(0, 3);
        sim->ApplyRz(3, .13);
        sim->ApplyAmplitudeDamping(0, .2);
        sim->ApplyRy(3, .42);
        sim->ApplyCX(3, 0);
    }
    Check((fused->PartialTrace({0, 1, 2, 3}) - plain->PartialTrace({0, 1, 2, 3})).norm() < 1e-10, "noise boundary");
    const auto overlap = fused->DensityMatrixOverlap(*plain);
    Check(std::abs(overlap.real() - fused->DensityMatrixPurity()) < 1e-10, "overlap synchronization");
}

void Capabilities()
{
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductState, Method::kMatrixProductOperator, Method::kTensorNetwork,
                        Method::kStabilizer, Method::kExtendedStabilizer, Method::kPauliPropagator, Method::kPathIntegral})
    {
        auto sim = SimulatorsFactory::CreateSimulator(Backend::kQCSim, method);
        const bool supported = method == Method::kStatevector || method == Method::kDensityMatrix || method == Method::kMatrixProductState ||
                               method == Method::kMatrixProductOperator || method == Method::kTensorNetwork;
        Check(sim->IsGateFusionEnabled() == supported, "incorrect default capability");
        Check(sim->GetConfiguration("gate_fusion") == "auto", "fusion default is not automatic");
        bool rejected = false;
        try
        {
            sim->Configure("gate_fusion", "invalid");
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        Check(rejected, "invalid fusion flag accepted");
    }
#ifndef NO_QISKIT_AER
    for (auto backend : {Backend::kQiskitAer, Backend::kCompositeQiskitAer})
    {
        auto sim = SimulatorsFactory::CreateSimulator(backend, Method::kStatevector);
        Check(!sim->IsGateFusionEnabled() && sim->GetGateFusionMaxQubits() == 0, "Aer fusion changed");
    }
#endif
}

// Without an explicit setting, small CPU statevectors and density matrices
// run unfused; an explicit setting always wins, and "auto" restores the default.
void DefaultThreshold()
{
    struct Case
    {
        Method method;
        size_t below, from;
    };

    for (const auto &c : {Case{Method::kStatevector, 10, 11}, Case{Method::kDensityMatrix, 4, 5}})
    {
        for (size_t n : {c.below, c.from})
        {
            auto sim = SimulatorsFactory::CreateSimulator(Backend::kQCSim, c.method);
            Check(sim->IsGateFusionEnabled(), "unallocated default capability");
            sim->AllocateQubits(n);
            sim->Initialize();
            const bool expected = n >= c.from;
            Check(sim->IsGateFusionEnabled() == expected, "default fusion threshold");
            Check(sim->GetConfiguration("gate_fusion") == "auto" && sim->GetConfigMap().at("gate_fusion") == "auto",
                  "default setting is not reported as automatic");
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            sim->ApplyRz(1, .3);
            Check(std::abs(sim->Probability(0) - .5) < 1e-12 && std::abs(sim->Probability(3) - .5) < 1e-12, "default fusion changed the result");
            const auto stats = sim->GetGateFusionStatistics();
            Check(expected ? stats.fusedBlocks == 1 && stats.backendGates == 1 : stats.fusedBlocks == 0 && stats.backendGates == 3,
                  "default fusion did not follow the threshold");
            auto clone = sim->Clone();
            Check(clone->IsGateFusionEnabled() == expected && clone->GetConfiguration("gate_fusion") == "auto", "clone lost the automatic setting");
            // Explicit settings win on either side of the threshold.
            for (const char *setting : {"true", "false", "1", "0"})
            {
                sim->Configure("gate_fusion", setting);
                const bool on = std::string(setting) == "true" || std::string(setting) == "1";
                Check(sim->IsGateFusionEnabled() == on && sim->GetConfiguration("gate_fusion") == (on ? "true" : "false"), "explicit fusion setting ignored");
            }
            sim->Configure("gate_fusion", "auto");
            Check(sim->IsGateFusionEnabled() == expected, "auto did not restore the default");
            // The default follows the register size.
            sim->Clear();
            sim->AllocateQubits(expected ? c.below : c.from);
            sim->Initialize();
            Check(sim->IsGateFusionEnabled() != expected, "default did not follow the new register size");
        }
    }
    // Tensor-network methods keep fusing at any size: a merge saves an SVD.
    for (auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator, Method::kTensorNetwork})
    {
        auto sim = SimulatorsFactory::CreateSimulator(Backend::kQCSim, method);
        sim->AllocateQubits(2);
        sim->Initialize();
        Check(sim->IsGateFusionEnabled(), "small tensor network default");
    }
}

void CircuitBoundaries()
{
    using CF = Circuits::CircuitFactory<>;
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Kind::kXGateType, 0));
    circuit->AddOperation(std::make_shared<Circuits::Delay<>>(1, .1));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Kind::kXGateType, 1), 0));
    circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Kind::kXGateType, 2), 1));
    auto nested = CF::CreateCircuit();
    nested->AddOperation(CF::CreateGate(Kind::kHadamardGateType, 2));
    nested->AddOperation(CF::CreateGate(Kind::kCXGateType, 2, 3));
    circuit->AddOperation(nested);
    circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}, 0, std::vector<bool>{true}));
    circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, 2, 0, 0, .31));
    for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        auto sim = Make(method);
        auto plain = Make(method, false);
        sim->SetUpcomingGates(circuit->GetOperations());
        Circuits::OperationState state;
        state.AllocateBits(4);
        circuit->Execute(sim, state);
        circuit->Execute(plain, state);
        Check(sim->GetGatesCounter() == static_cast<long long>(circuit->size()), "boundary source cursor");
        if (method == Method::kMatrixProductOperator)
            Check((sim->PartialTrace({0, 1, 2, 3}) - plain->PartialTrace({0, 1, 2, 3})).norm() < 1e-9, "boundary MPO");
        else
            Close(State(*sim), State(*plain), "dynamic circuit boundaries");
    }
}

void SamplingAndObservers()
{
    auto sim = Make(Method::kStatevector);
    sim->ApplyX(1);
    sim->RestoreState();
    Check(std::abs(sim->Probability(2) - 1) < 1e-12, "restore without save lost pending work");
    sim->Reset();

    struct Observer : ISimulatorObserver
    {
        size_t count = 0;

        void Update(const Types::qubits_vector &) override
        {
            ++count;
        }
    };

    auto observer = std::make_shared<Observer>();
    sim->RegisterObserver(observer);
    sim->ApplyH(0);
    sim->ApplyCX(0, 3);
    sim->ApplyZ(0);
    sim->ApplyZ(0);
    Check(observer->count == 4, "notifications must count source gates");
    const auto counts = sim->SampleCounts({3, 0}, 100);
    size_t shots = 0;
    for (const auto &[outcome, count] : counts)
    {
        Check(outcome == 0 || outcome == 3, "sampling pending cache");
        shots += count;
    }
    Check(shots == 100 && observer->count == 4, "sampling changed notifications");
    auto expected = State(*sim);
    sim->SaveStateToInternalDestructive();
    sim->ApplyX(1);
    expected = ApplyDense(expected, {Kind::kXGateType, {1}, {}, {}});
    sim->RestoreInternalDestructiveSavedState();
    Verify(*sim, expected, "QCSim destructive hooks retain their no-op semantics");
    sim->ApplyY(2);
    sim->Clear();
    sim->AllocateQubits(4);
    sim->Initialize();
    Vec zero = Vec::Zero(16);
    zero[0] = 1;
    Verify(*sim, zero, "clear pending");
}
#ifdef __linux__
// Exercise every low-level entry point and its storage contract before the
// generic interface (which uses explicit Eigen column-major storage).
template <class Native> void NativeMatrices(Native &sim, bool fp64)
{
    Check(sim.SetDataType(fp64), "native precision");
    Check(sim.Create(4), "native allocation");
    Vec expected = Vec::Zero(16);
    expected[0] = 1;
    std::mt19937 rng(73);
    for (const auto &qs : {Types::qubits_vector{3}, Types::qubits_vector{3, 0}, Types::qubits_vector{2, 3, 0}})
    {
        Gate gate{Kind::kNone, qs, {}, Unitary(size_t{1} << qs.size(), rng)};
        for (int mode = 0; mode < 3; ++mode)
        {
            const int dim = gate.generic.rows();
            std::vector<double> buffer(2 * dim * dim);
            for (int row = 0; row < dim; ++row)
                for (int col = 0; col < dim; ++col)
                {
                    const int i = mode == 2 ? col * dim + row : row * dim + col;
                    buffer[2 * i] = gate.generic(row, col).real();
                    buffer[2 * i + 1] = gate.generic(row, col).imag();
                }
            bool result;
            if (qs.size() == 1)
                result = mode == 0 ? sim.ApplyOneQubitMatrix(qs[0], buffer.data()) : sim.ApplyOneQubitMatrixWithLayout(qs[0], buffer.data(), mode == 2);
            else if (qs.size() == 2)
                result = mode == 0 ? sim.ApplyTwoQubitMatrix(qs[0], qs[1], buffer.data())
                                   : sim.ApplyTwoQubitMatrixWithLayout(qs[0], qs[1], buffer.data(), mode == 2);
            else
                result = mode == 0 ? sim.ApplyThreeQubitMatrix(qs[0], qs[1], qs[2], buffer.data())
                                   : sim.ApplyThreeQubitMatrixWithLayout(qs[0], qs[1], qs[2], buffer.data(), mode == 2);
            Check(result, "native matrix application");
            std::fill(buffer.begin(), buffer.end(), 0.);
            expected = ApplyDense(expected, gate);
        }
        Vec actual(16);
        for (int i = 0; i < 16; ++i)
        {
            double real = 0, imag = 0;
            Check(sim.Amplitude(i, &real, &imag), "native amplitude");
            actual[i] = {real, imag};
        }
        Close(actual, expected, "native matrix layouts", fp64 ? 2e-10 : 2e-5);
    }
}

void StatevectorFusion(Backend backend, const char *nativeBackend = "")
{
    for (bool fp64 : {false, true})
        for (bool fusion : {false, true})
        {
            auto sim = SimulatorsFactory::CreateSimulator(backend, Method::kStatevector);
            Check(bool(sim), "GPU statevector unavailable");
            if (*nativeBackend)
                sim->Configure("distributed_backend", nativeBackend);
            const bool eligible = backend == Backend::kGpuSim || std::string(nativeBackend) == "conventional";
            Check(sim->IsGateFusionEnabled() == eligible, "default statevector fusion");
            Check(sim->GetGateFusionMaxQubits() == (eligible ? 3u : 0u), "statevector width");
            sim->Configure("gate_fusion", fusion ? "true" : "false");
            sim->Configure("use_double_precision", fp64 ? "true" : "false");
            sim->AllocateQubits(4);
            sim->Initialize();
            Vec expected = Vec::Zero(16);
            expected[0] = 1;
            std::mt19937 rng(57);
            std::vector<Gate> gates{{Kind::kHadamardGateType, {0}, {}, {}},
                                    {Kind::kCXGateType, {0, 3}, {}, {}},
                                    {Kind::kCYGateType, {3, 2}, {}, {}},
                                    {Kind::kNone, {3, 0, 2}, {}, Unitary(8, rng)},
                                    {Kind::kNone, {2, 3, 0}, {}, Unitary(8, rng)}};
            for (const auto &gate : gates)
            {
                gate.Apply(*sim);
                expected = ApplyDense(expected, gate);
            }
            const auto before = sim->GetGateFusionStatistics();
            Check(before.backendGates == (eligible && fusion ? 0u : gates.size()), "wrong gate caching behavior");
            const double eps = fp64 ? 2e-10 : 2e-5;
            // Clone flushes pending gates; restore discards later cached work.
            auto copy = sim->Clone();
            Close(State(*sim), expected, "statevector generic gates", eps);
            Close(State(*copy), expected, "statevector clone", eps);
            auto after = sim->GetGateFusionStatistics();
            Check(after.fusedBlocks == (eligible && fusion ? 1u : 0u), "three-qubit block was not fused as expected");
            sim->SaveState();
            sim->ApplyX(1);
            sim->ApplyH(1);
            sim->RestoreState();
            Close(State(*sim), expected, "statevector restore", eps);
            sim->ApplyX(1);
            expected = ApplyDense(expected, {Kind::kXGateType, {1}, {}, {}});
            sim->Configure("gate_fusion", "false");
            Close(State(*sim), expected, "statevector disabling fusion", eps);
            for (const auto &targets : {Types::qubits_vector{0, 0, 2}, Types::qubits_vector{0, 4, 2}, Types::qubits_vector{0, Types::qubit_t{1} << 32, 2}})
            {
                bool rejected = false;
                try
                {
                    sim->ApplyGenericThreeQubitGate(targets[0], targets[1], targets[2], Eigen::Matrix<std::complex<double>, 8, 8>::Identity());
                }
                catch (const std::exception &)
                {
                    rejected = true;
                }
                Check(rejected, "invalid native generic target accepted");
            }
            Close(State(*sim), expected, "rejected generic gate changed state", eps);
            sim->Configure("gate_fusion", "true");
            sim->SaveStateToInternalDestructive();
            sim->RestoreInternalDestructiveSavedState();
            Close(State(*sim), expected, "statevector destructive snapshot", eps);
            sim->ApplyH(1);
            sim->Reset();
            sim->ApplyX(0);
            sim->ApplyX(3);
            const auto counts = sim->SampleCounts({3, 0}, 16);
            Check(counts.size() == 1 && counts.at(3) == 16, "statevector sampling flush");
            Check(sim->Measure({0, 3}) == 3, "statevector measurement flush");
        }
}

void StructuredDistributedGates()
{
    for (bool fusion : {false, true})
    {
        auto sim = SimulatorsFactory::CreateSimulator(Backend::kDistGpuSim, Method::kStatevector);
        sim->Configure("distributed_backend", "conventional");
        sim->Configure("distributed_devices", "0,0");
        sim->Configure("distributed_flags", "1");
        sim->Configure("use_double_precision", "true");
        sim->Configure("gate_fusion", fusion ? "true" : "false");
        sim->AllocateQubits(4);
        sim->Initialize();
        for (size_t q = 0; q < 4; ++q)
            sim->ApplyH(q);
        sim->Flush();
        const auto layout = sim->GetConfiguration("distributed_qubit_layout");
        Vec expected = State(*sim);
        for (const auto &gate :
             std::vector<Gate>{{Kind::kCZGateType, {0, 3}, {}, {}}, {Kind::kCPGateType, {0, 3}, {.31}, {}}, {Kind::kRzGateType, {0}, {.17}, {}}})
        {
            gate.Apply(*sim);
            expected = ApplyDense(expected, gate);
        }
        Check(sim->GetConfiguration("distributed_qubit_layout") == layout, "diagonal fusion introduced a global/local exchange");
        Close(State(*sim), expected, "structured distributed gates");
    }
}

Vec Collapse(Vec v, size_t q, bool bit)
{
    for (Eigen::Index i = 0; i < v.size(); ++i)
        if (bool((size_t(i) >> q) & 1) != bit)
            v[i] = 0;
    return v / v.norm();
}

// Conventional is the only distributed backend that Maestro fuses for. Several
// shards share one GPU (flag 1), so global qubits and every layout policy run
// without multiple devices. Random circuits mix structured, parametric and
// generic gates up to three qubits, with measurements on local and global
// qubits in between.
void ConventionalShardedFusion()
{
    const size_t n = 7;
    const std::vector<Kind> one{Kind::kHadamardGateType, Kind::kXGateType,   Kind::kYGateType,  Kind::kZGateType,     Kind::kSGateType, Kind::kSdgGateType,
                                Kind::kTGateType,        Kind::kTdgGateType, Kind::kSxGateType, Kind::kSxDagGateType, Kind::kKGateType, Kind::kPhaseGateType,
                                Kind::kRxGateType,       Kind::kRyGateType,  Kind::kRzGateType, Kind::kUGateType,     Kind::kNone};
    const std::vector<Kind> two{Kind::kCXGateType,  Kind::kCYGateType,   Kind::kCZGateType, Kind::kCPGateType,  Kind::kCRxGateType,
                                Kind::kCRyGateType, Kind::kCRzGateType,  Kind::kCHGateType, Kind::kCSxGateType, Kind::kCSxDagGateType,
                                Kind::kCUGateType,  Kind::kSwapGateType, Kind::kNone};
    const std::vector<Kind> three{Kind::kCCXGateType, Kind::kCSwapGateType, Kind::kNone};
    // Two-qubit gates are drawn twice as often as one- or three-qubit gates.
    const std::vector<const std::vector<Kind> *> pools{&one, &two, &two, &three};
    for (size_t shards : {2, 4, 8})
    {
        size_t globalBits = 0;
        while ((size_t{1} << globalBits) < shards)
            ++globalBits;
        std::string devices = "0", lowGlobals, highGlobals;
        for (size_t s = 1; s < shards; ++s)
            devices += ",0";
        for (size_t g = 0; g < globalBits; ++g)
        {
            lowGlobals += (g ? "," : "") + std::to_string(g);
            highGlobals += (g ? "," : "") + std::to_string(n - 1 - g);
        }
        for (int policy : {0, 2, 8})
            for (bool fp64 : {false, true})
                for (const auto &globals : {lowGlobals, highGlobals})
                    for (bool fusion : {false, true})
                    {
                        const std::string name = "conventional shards=" + std::to_string(shards) + " flags=" + std::to_string(policy) + " globals=" + globals +
                                                 (fp64 ? " fp64" : " fp32") + (fusion ? " fused" : " plain");
                        auto sim = SimulatorsFactory::CreateSimulator(Backend::kDistGpuSim, Method::kStatevector);
                        sim->Configure("distributed_backend", "conventional");
                        sim->Configure("distributed_devices", devices.c_str());
                        sim->Configure("distributed_flags", std::to_string(policy | 1).c_str());
                        sim->Configure("distributed_global_qubits", globals.c_str());
                        sim->Configure("use_double_precision", fp64 ? "true" : "false");
                        sim->Configure("gate_fusion", fusion ? "true" : "false");
                        sim->AllocateQubits(n);
                        sim->Initialize();
                        sim->SetSeed(7);
                        Check(sim->GetGateFusionMaxQubits() == 3 && sim->IsGateFusionEnabled() == fusion, name + ": fusion capability");
                        const auto layout = sim->GetConfiguration("distributed_qubit_layout");
                        Vec expected = Vec::Zero(size_t{1} << n);
                        expected[0] = 1;
                        std::mt19937 rng(1234 + shards);
                        std::uniform_real_distribution<double> angle(-3.1, 3.1);
                        const double eps = fp64 ? 1e-9 : 2e-4;
                        for (int i = 0; i < 150; ++i)
                        {
                            if (i == 50 || i == 100)
                            {
                                // One local and one global measurement per checkpoint.
                                for (size_t q : {size_t(n / 2), size_t(globals == lowGlobals ? 0 : n - 1)})
                                {
                                    const bool bit = sim->Measure({q}) & 1;
                                    expected = Collapse(expected, q, bit);
                                }
                                Close(State(*sim), expected, name + " after measurement", eps);
                            }
                            const auto *kinds = pools[rng() % pools.size()];
                            Gate gate;
                            gate.kind = (*kinds)[rng() % kinds->size()];
                            std::vector<Types::qubit_t> qs(n);
                            std::iota(qs.begin(), qs.end(), 0);
                            std::shuffle(qs.begin(), qs.end(), rng);
                            const size_t arity = kinds == &one ? 1 : kinds == &two ? 2 : 3;
                            gate.qubits.assign(qs.begin(), qs.begin() + arity);
                            for (auto &p : gate.params)
                                p = angle(rng);
                            if (gate.kind == Kind::kNone)
                                gate.generic = Unitary(size_t{1} << arity, rng);
                            gate.Apply(*sim);
                            expected = ApplyDense(expected, gate);
                        }
                        Close(State(*sim), expected, name, eps);
                        const auto stats = sim->GetGateFusionStatistics();
                        Check(fusion ? stats.fusedBlocks > 0 && stats.backendGates < stats.submittedGates : stats.fusedBlocks == 0,
                              name + ": fusion statistics");
                        if (policy == 8)
                            Check(sim->GetConfiguration("distributed_qubit_layout") == layout, name + ": pinned layout changed");
                    }
    }
}

void DistributedMatrices()
{
    auto lib = SimulatorsFactory::GetDistributedGpuLibrary();
    lib->RequireLoaded();
    Check(lib->HasThreeQubitMatrixAPI(), "distributed three-qubit API unavailable");
    for (int backend : {0, 1})
    {
        for (bool fp64 : {false, true})
        {
            DistributedGpuLibStateVectorSim sim(lib, lib->CreateNative(0, backend));
            NativeMatrices(sim, fp64);
        }
        StatevectorFusion(Backend::kDistGpuSim, backend == 0 ? "conventional" : "ex");
    }
}

void GpuBackends()
{
    Check(SimulatorsFactory::InitGpuLibrary(), "GPU plugin unavailable");
    for (bool fp64 : {false, true})
    {
        auto sim = SimulatorsFactory::CreateGpuLibStateVectorSim();
        Check(bool(sim), "native GPU statevector unavailable");
        NativeMatrices(*sim, fp64);
    }
    StatevectorFusion(Backend::kGpuSim);
    // Prefix checks localize native/generic ordering errors while retaining
    // fusion within each prefix.
    for (size_t prefix = 1; prefix <= Sequence().size(); ++prefix)
    {
        auto sim = Make(Method::kMatrixProductState, true, Backend::kGpuSim);
        Vec expected = Vec::Zero(16);
        expected[0] = 1;
        const auto gates = Sequence();
        for (size_t i = 0; i < prefix; ++i)
        {
            gates[i].Apply(*sim);
            expected = ApplyDense(expected, gates[i]);
        }
        Verify(*sim, expected, "GPU MPS prefix=" + std::to_string(prefix));
    }
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        auto fused = Make(method, true, Backend::kGpuSim), plain = Make(method, false, Backend::kGpuSim);
        Check(fused->GetGateFusionMaxQubits() == (method == Method::kStatevector ? 3u : 2u) && fused->IsGateFusionEnabled(), "GPU capability");
        Vec expected = Vec::Zero(16);
        expected[0] = 1;
        for (const auto &gate : Sequence())
        {
            gate.Apply(*fused);
            gate.Apply(*plain);
            expected = ApplyDense(expected, gate);
        }
        Verify(*fused, expected, "GPU fused method=" + std::to_string(int(method)));
        Verify(*plain, expected, "GPU plain");
    }
    for (auto method : {Method::kTensorNetwork, Method::kStabilizer, Method::kPauliPropagator})
    {
        auto sim = SimulatorsFactory::CreateSimulator(Backend::kGpuSim, method);
        Check(!sim || !sim->IsGateFusionEnabled(), "GPU unsupported method enabled");
    }
    SnapshotLifetime(Backend::kGpuSim);
    AdditionalSnapshotLifetimes(Backend::kGpuSim);
    NativeCloneCounters(Backend::kGpuSim);
    GenericTargetOrder(Backend::kGpuSim, Method::kMatrixProductState);
    GenericAndLifecycle(Backend::kGpuSim);
    Routing(Backend::kGpuSim);
}
#endif
int main(int argc, char **argv)
{
    try
    {
#ifdef __linux__
        if (argc > 1 && std::string(argv[1]) == "--distributed")
        {
            if (!SimulatorsFactory::IsDistributedGpuAvailable())
                return 77;
            DistributedMatrices();
            StructuredDistributedGates();
            ConventionalShardedFusion();
            std::cout << "Distributed matrix wrappers and fusion selection passed\n";
            return 0;
        }
#endif
        if (argc > 1 && std::string(argv[1]) == "--gpu")
        {
#ifdef __linux__
            if (!SimulatorsFactory::InitGpuLibraryWithMute())
                return 77;
            GpuBackends();
#else
            throw std::runtime_error("GPU backend requires Linux");
#endif
            std::cout << "GPU equivalence, lifecycle, and routing passed\n";
            return 0;
        }
        Capabilities();
        DefaultThreshold();
        CircuitBoundaries();
        SamplingAndObservers();
        std::cout << "configuration, circuit boundaries, sampling, observers passed\n";
        Algebra();
        std::cout << "cache algebra and preparation passed\n";
        Backends();
        std::cout << "CPU backend equivalence passed\n";
        SnapshotLifetime();
        AdditionalSnapshotLifetimes(Backend::kQCSim);
        NativeCloneCounters(Backend::kQCSim);
        GenericTargetOrder(Backend::kQCSim, Method::kTensorNetwork);
        GenericAndLifecycle();
        std::cout << "generic matrices and lifecycle passed\n";
        Routing();
        std::cout << "MPS/MPO routing passed\n";
        Channels();
        std::cout << "channel boundaries passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
