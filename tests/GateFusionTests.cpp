#include "Simulators/Core/Factory.h"
#include "Simulators/Fusion/FusionGate.h"
#include <Eigen/QR>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
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

// The CPU density matrix fuses a channel with the gates around it into one
// superoperator pass. Fused and unfused runs must agree to rounding on noisy
// circuits with every target order, delays, three-qubit gates crossing blocks,
// mid-circuit measurement, classical control and reset.
// On a mixed state a gate, its noise and the gates around them on the same one
// or two qubits become one map. `tolerance`: the density matrix multiplies the
// same numbers in another order; the matrix product operator decomposes them.
void ChannelFusionOn(Method method, double tolerance)
{
    using CF = Circuits::CircuitFactory<>;
    using Channel = QuantumChannel;
    constexpr size_t n = 5;
    const Types::qubits_vector all{0, 1, 2, 3, 4};
    const Channel relax0 = Channel::ThermalRelaxation(1., 30., 45.), relax1 = Channel::ThermalRelaxation(2., 20., 12., .2);
    const Channel gateError = Channel::TwoQubitDepolarizing(.04).Then(relax0, {0}).Then(relax1, {1});
    const Channel damping = Channel::AmplitudeDamping(.1), flip = Channel::Pauli(.02, .03, .05);
    const auto channel = [](const Types::qubits_vector &qubits, const Channel &c) { return std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, c); };
    const auto run = [&](const std::shared_ptr<Circuits::Circuit<>> &circuit, bool fusion) {
        auto sim = Make(method, fusion, Backend::kQCSim, n);
        Circuits::OperationState state;
        state.AllocateBits(n);
        circuit->Execute(sim, state);
        sim->Flush();
        return sim;
    };

    // One qubit pair, gates and noise interleaved: a single pass.
    {
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Kind::kCXGateType, 0, 1));
        circuit->AddOperation(channel({0, 1}, gateError));
        circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, 0, 0, 0, .7));
        circuit->AddOperation(channel({0}, damping));
        circuit->AddOperation(CF::CreateGate(Kind::kRyGateType, 1, 0, 0, -.4));
        circuit->AddOperation(channel({1}, flip));
        circuit->AddOperation(std::make_shared<Circuits::Delay<>>(0, 1e-7));
        circuit->AddOperation(channel({0}, relax0));
        circuit->AddOperation(CF::CreateGate(Kind::kCXGateType, 1, 0));
        circuit->AddOperation(channel({1, 0}, gateError));
        const auto fused = run(circuit, true), plain = run(circuit, false);
        Check((fused->PartialTrace(all) - plain->PartialTrace(all)).norm() < tolerance, "one-pair channel fusion changed the state");
        const auto stats = fused->GetGateFusionStatistics();
        Check(stats.backendGates == 1 && stats.fusedBlocks == 1,
              "a gate, its noise and its neighbours were not one pass: " + std::to_string(stats.backendGates) + " backend operations");
        Check(fused->GetGatesCounter() == static_cast<long long>(circuit->size()), "channel fusion source cursor");
    }

    std::mt19937_64 rng(2026);
    std::uniform_real_distribution<double> angle(-3., 3.);
    for (int trial = 0; trial < 24; ++trial)
    {
        auto circuit = CF::CreateCircuit();
        for (int step = 0; step < 70; ++step)
        {
            const size_t a = rng() % n;
            size_t b = rng() % (n - 1), c = rng() % (n - 2);
            b += b >= a;
            c += c >= std::min(a, b);
            c += c >= std::max(a, b);
            switch (rng() % (trial % 4 == 3 ? 10 : 8))
            {
            case 0:
                circuit->AddOperation(CF::CreateGate(Kind::kCXGateType, a, b));
                circuit->AddOperation(channel(rng() % 2 ? Types::qubits_vector{a, b} : Types::qubits_vector{b, a}, gateError));
                break;
            case 1:
                circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, a, 0, 0, angle(rng)));
                break;
            case 2:
                circuit->AddOperation(CF::CreateGate(Kind::kRyGateType, a, 0, 0, angle(rng)));
                circuit->AddOperation(channel({a}, damping));
                break;
            case 3:
                circuit->AddOperation(CF::CreateGate(Kind::kCZGateType, a, b));
                circuit->AddOperation(channel({a}, flip));
                circuit->AddOperation(channel({b}, relax1));
                break;
            case 4:
                circuit->AddOperation(std::make_shared<Circuits::Delay<>>(a, 1e-7));
                circuit->AddOperation(channel({a}, relax0));
                break;
            case 5:
                circuit->AddOperation(CF::CreateGate(Kind::kCCXGateType, a, b, c));
                break;
            case 6:
                circuit->AddOperation(channel({b, a}, gateError));
                break;
            case 7:
                circuit->AddOperation(CF::CreateGate(Kind::kHadamardGateType, a));
                break;
            case 8:
                circuit->AddOperation(CF::CreateMeasurement({{a, a}}));
                circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Kind::kXGateType, b), a));
                break;
            default:
                circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{a}));
                break;
            }
        }
        const auto fused = run(circuit, true), plain = run(circuit, false);
        const double error = (fused->PartialTrace(all) - plain->PartialTrace(all)).norm();
        Check(error < tolerance, "channel fusion changed the state in trial " + std::to_string(trial) + ": error " + std::to_string(error));
        Check(std::abs(fused->PartialTrace(all).trace().real() - 1.) < tolerance, "channel fusion changed the trace");
        // The unfused run applies every gate and channel on its own; a chain
        // takes a three-qubit gate as five two-qubit ones.
        size_t applied = 0;
        for (const auto &op : circuit->GetOperations())
            applied += op->GetType() == Circuits::OperationType::kQuantumChannel              ? 1
                       : op->GetType() != Circuits::OperationType::kGate                      ? 0
                       : method != Method::kDensityMatrix && op->AffectedQubits().size() == 3 ? 5
                                                                                              : 1;
        Check(fused->GetGateFusionStatistics().backendGates < applied, "channel fusion saved no passes");
    }

    Check(Make(method, true, Backend::kQCSim, n)->FusesQuantumChannels(), "channel fusion capability");
    Check(!Make(method, false, Backend::kQCSim, n)->FusesQuantumChannels(), "channel fusion ignores gate_fusion=false");
}

void ChannelFusion()
{
    ChannelFusionOn(Method::kDensityMatrix, 1e-12);
    ChannelFusionOn(Method::kMatrixProductOperator, 1e-9);
    // A pure state holds no map of a mixed one.
    Check(!Make(Method::kMatrixProductState, true)->FusesQuantumChannels(), "channel fusion on a pure state");
}

// A statevector samples channels from its state. With fusion on, the gates and
// channels on up to three qubits form one block: its branch probabilities come
// from one reduced density matrix and the operators selected are applied once.
void TrajectoryBlocks()
{
    using CF = Circuits::CircuitFactory<>;
    using Channel = QuantumChannel;
    const Channel relax0 = Channel::ThermalRelaxation(1., 30., 45.), relax1 = Channel::ThermalRelaxation(2., 20., 12., .2);
    const Channel damping = Channel::AmplitudeDamping(.25), generalized = Channel::GeneralizedAmplitudeDamping(.2, .3);
    const Channel flip = Channel::Pauli(.06, .04, .1);
    Eigen::MatrixXcd hadamard(2, 2);
    hadamard << 1., 1., 1., -1.;
    hadamard /= std::sqrt(2.);
    const Channel mixture = Channel::UnitaryMixture({.7, .3}, {Eigen::MatrixXcd::Identity(2, 2), hadamard});
    // Both qubits decay together, or neither does: a two-qubit map that is no product of one-qubit maps.
    Eigen::MatrixXcd keep = Eigen::MatrixXcd::Identity(4, 4), decay = Eigen::MatrixXcd::Zero(4, 4);
    keep(3, 3) = std::sqrt(.7);
    decay(0, 3) = std::sqrt(.3);
    const Channel pairDecay(Channel::KrausOperators{keep, decay});
    // Factors drawn from the state only, and a gate error that also holds a Pauli factor.
    const Channel relaxBoth = Channel::OnQubit(relax0, 0).Then(relax1, {1}).Then(pairDecay);
    const Channel gateError = Channel::TwoQubitDepolarizing(.08).Then(relax0, {0}).Then(relax1, {1});
    const auto channel = [](const Types::qubits_vector &qubits, const Channel &c) { return std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, c); };
    const auto run = [](const std::shared_ptr<Circuits::Circuit<>> &circuit, Method method, bool fusion, size_t n, uint64_t seed) {
        auto sim = Make(method, fusion, Backend::kQCSim, n);
        sim->SetSeed(seed);
        Circuits::OperationState state;
        state.AllocateBits(n);
        circuit->Execute(sim, state);
        sim->Flush();
        return sim;
    };

    Check(Make(Method::kStatevector, true)->FusesQuantumChannels() && !Make(Method::kStatevector, false)->FusesQuantumChannels() &&
              !Make(Method::kMatrixProductState, true)->FusesQuantumChannels(),
          "trajectory block capability");

    // A register no wider than a block holds one block, whose channels draw in
    // circuit order: with factors drawn from the state only, the fused run
    // takes the branches of the unfused one and ends in the same state.
    std::mt19937_64 rng(511);
    std::uniform_real_distribution<double> angle(-3., 3.);
    for (size_t n = 1; n <= 3; ++n)
        for (int trial = 0; trial < 40; ++trial)
        {
            auto circuit = CF::CreateCircuit();
            size_t channels = 0;
            for (int step = 0; step < 40; ++step)
            {
                const size_t a = rng() % n, b = n > 1 ? (a + 1 + rng() % (n - 1)) % n : 0;
                switch (rng() % (n == 1 ? 4 : n == 2 ? 7 : 8))
                {
                case 0:
                    circuit->AddOperation(CF::CreateGate(Kind::kRyGateType, a, 0, 0, angle(rng)));
                    break;
                case 1:
                    circuit->AddOperation(CF::CreateGate(Kind::kHadamardGateType, a));
                    circuit->AddOperation(channel({a}, rng() % 2 ? damping : generalized));
                    ++channels;
                    break;
                case 2:
                    circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, a, 0, 0, angle(rng)));
                    circuit->AddOperation(std::make_shared<Circuits::Delay<>>(a, 1e-7));
                    circuit->AddOperation(channel({a}, rng() % 2 ? relax0 : relax1));
                    ++channels;
                    break;
                case 3:
                    circuit->AddOperation(CF::CreateGate(Kind::kSxGateType, a));
                    break;
                case 4:
                    circuit->AddOperation(CF::CreateGate(Kind::kCXGateType, a, b));
                    circuit->AddOperation(channel({b, a}, relaxBoth));
                    ++channels;
                    break;
                case 5:
                    circuit->AddOperation(CF::CreateGate(Kind::kCZGateType, a, b));
                    circuit->AddOperation(channel({a}, relax1));
                    circuit->AddOperation(channel({b}, damping));
                    channels += 2;
                    break;
                case 6:
                    circuit->AddOperation(channel({a, b}, pairDecay));
                    ++channels;
                    break;
                default:
                    circuit->AddOperation(CF::CreateGate(Kind::kCCXGateType, a, b, 3 - a - b));
                    break;
                }
            }
            const uint64_t seed = 1000 + trial;
            const auto fused = run(circuit, Method::kStatevector, true, n, seed), plain = run(circuit, Method::kStatevector, false, n, seed);
            const std::string name = std::to_string(n) + "-qubit trajectory block, trial " + std::to_string(trial);
            Close(State(*fused), State(*plain), name + " left the unfused trajectory", 1e-10);
            Check(std::abs(State(*fused).norm() - 1.) < 1e-12, name + " is not normalized");
            // One pass, or two where the block was applied before it was read and had to be corrected.
            const auto stats = fused->GetGateFusionStatistics();
            Check(channels > 0 && stats.backendGates == 1 + stats.rereadBlocks && stats.rereadBlocks <= stats.unreadBlocks && stats.unreadBlocks <= 1,
                  name + " took " + std::to_string(stats.backendGates) + " passes");
            Check(fused->GetGatesCounter() == static_cast<long long>(circuit->size()), name + " source cursor");
            Close(State(*run(circuit, Method::kStatevector, true, n, seed)), State(*fused), name + " does not replay", 1e-14);
        }

    // Wider registers resolve their blocks in another order than the circuit's,
    // so single trajectories differ; their average is the density matrix.
    // Every kind of factor takes part here, with resets between them.
    constexpr size_t n = 5, trajectories = 6000;
    const Types::qubits_vector all{0, 1, 2, 3, 4};
    for (int trial = 0; trial < 3; ++trial)
    {
        auto circuit = CF::CreateCircuit();
        size_t applied = 0;
        for (int step = 0; step < 45; ++step)
        {
            const size_t a = rng() % n;
            size_t b = rng() % (n - 1), c = rng() % (n - 2);
            b += b >= a;
            c += c >= std::min(a, b);
            c += c >= std::max(a, b);
            switch (rng() % 10)
            {
            case 0:
                circuit->AddOperation(CF::CreateGate(Kind::kCXGateType, a, b));
                circuit->AddOperation(channel(rng() % 2 ? Types::qubits_vector{a, b} : Types::qubits_vector{b, a}, gateError));
                applied += 2;
                break;
            case 1:
                circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, a, 0, 0, angle(rng)));
                circuit->AddOperation(channel({a}, flip));
                applied += 2;
                break;
            case 2:
                circuit->AddOperation(CF::CreateGate(Kind::kRyGateType, a, 0, 0, angle(rng)));
                circuit->AddOperation(channel({a}, damping));
                applied += 2;
                break;
            case 3:
                circuit->AddOperation(CF::CreateGate(Kind::kCZGateType, a, b));
                circuit->AddOperation(channel({a}, mixture));
                circuit->AddOperation(channel({b}, relax1));
                applied += 3;
                break;
            case 4:
                circuit->AddOperation(std::make_shared<Circuits::Delay<>>(a, 1e-7));
                circuit->AddOperation(channel({a}, relax0));
                ++applied;
                break;
            case 5:
                circuit->AddOperation(CF::CreateGate(Kind::kCCXGateType, a, b, c));
                ++applied;
                break;
            case 6:
                circuit->AddOperation(channel({b, a}, relaxBoth));
                ++applied;
                break;
            case 7:
                circuit->AddOperation(CF::CreateGate(Kind::kHadamardGateType, a));
                circuit->AddOperation(channel({a}, generalized));
                applied += 2;
                break;
            case 8:
                circuit->AddOperation(CF::CreateGate(Kind::kSxGateType, a));
                ++applied;
                break;
            default:
                circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{a}));
                ++applied;
                break;
            }
        }
        const Eigen::MatrixXcd exact = run(circuit, Method::kDensityMatrix, false, n, 1)->PartialTrace(all);
        const double purity = (exact * exact).trace().real();
        Eigen::MatrixXcd mean = Eigen::MatrixXcd::Zero(exact.rows(), exact.cols());
        uint64_t passes = 0;
        for (size_t trajectory = 0; trajectory < trajectories; ++trajectory)
        {
            const auto sim = run(circuit, Method::kStatevector, true, n, 77000 + 1000 * trial + trajectory);
            const Vec state = State(*sim);
            mean += state * state.adjoint();
            passes += sim->GetGateFusionStatistics().backendGates;
        }
        mean /= static_cast<double>(trajectories);
        // The mean of the projectors has squared distance (1 - purity) / trajectories from the density matrix, on average.
        const double error = (mean - exact).norm(), expected = std::sqrt((1. - purity) / trajectories);
        Check(purity < .9, "the trajectory test circuit is too clean to test anything");
        Check(error < 2. * expected, "fused trajectories do not average to the density matrix in trial " + std::to_string(trial) + ": error " +
                                         std::to_string(error) + ", expected about " + std::to_string(expected));
        Check(passes < trajectories * applied * 6 / 10,
              "trajectory blocks saved too few passes: " + std::to_string(passes / trajectories) + " of " + std::to_string(applied));
    }
}

// A block whose channels most probably do nothing is applied without reading
// the state first: the backend returns the norm of the result from the pass
// that applies it, and the draws of the block tell from that norm whether any
// channel did something. The branches are those the reduced density matrix
// gives, so the trajectory is the same with the read first.
void UnreadTrajectoryBlocks()
{
    using CF = Circuits::CircuitFactory<>;
    using Channel = QuantumChannel;
    std::mt19937_64 rng(733);
    std::uniform_real_distribution<double> angle(-3., 3.);
    std::normal_distribution<double> normal;

    // The backend entry: any matrix on one to three qubits, in any order, with
    // or without qubit 0 among them, and the squared norm it leaves.
    for (size_t n = 1; n <= 7; ++n)
        for (int trial = 0; trial < 12; ++trial)
        {
            auto sim = Make(Method::kStatevector, false, Backend::kQCSim, n), reference = Make(Method::kStatevector, false, Backend::kQCSim, n);
            Check(sim->SupportsLocalMapNorm(), "the statevector does not apply local maps with their norm");
            for (auto *s : {sim.get(), reference.get()})
            {
                for (size_t q = 0; q < n; ++q)
                    s->ApplyRy(q, .3 + .4 * static_cast<double>(q + trial));
                for (size_t q = 0; q + 1 < n; ++q)
                    s->ApplyCX(q, q + 1);
                for (size_t q = 0; q < n; ++q)
                    s->ApplyRx(q, .7 - .2 * static_cast<double>(q));
            }
            Types::qubits_vector targets(n);
            std::iota(targets.begin(), targets.end(), Types::qubit_t{0});
            std::shuffle(targets.begin(), targets.end(), rng);
            targets.resize(1 + rng() % std::min<size_t>(3, n));
            const Eigen::Index dimension = Eigen::Index{1} << targets.size();
            Eigen::MatrixXcd matrix(dimension, dimension);
            for (Eigen::Index k = 0; k < matrix.size(); ++k)
                matrix.data()[k] = {normal(rng), normal(rng)};
            const double norm = sim->ApplyLocalMapNorm(targets, matrix);
            if (targets.size() == 1)
                reference->ApplyGenericOneQubitGate(targets[0], Eigen::Matrix2cd(matrix));
            else if (targets.size() == 2)
                reference->ApplyGenericTwoQubitGate(targets[0], targets[1], Eigen::Matrix4cd(matrix));
            else
                reference->ApplyGenericThreeQubitGate(targets[0], targets[1], targets[2], ISimulator::Matrix8cd(matrix));
            const std::string name = "local map with norm on " + std::to_string(targets.size()) + " of " + std::to_string(n) + " qubits";
            Close(State(*sim), State(*reference), name, 1e-10 * State(*reference).norm());
            Check(std::abs(norm - State(*reference).squaredNorm()) < 1e-10 * norm, name + " returned the wrong norm");
        }

    // What a channel says about its first operator.
    {
        // The channel owns what the reference points to.
        const Channel quarter = Channel::AmplitudeDamping(.25);
        const auto &first = quarter.FirstOperatorOf(0);
        Check(first.tracePreserving && std::abs(first.low - .75) < 1e-12 && std::abs(first.high - 1.) < 1e-12, "amplitude damping bounds");
        const Channel relax = Channel::ThermalRelaxation(1., 30., 45.);
        const auto &slow = relax.FirstOperatorOf(0);
        Check(relax.GetFactors().size() == 1 && slow.tracePreserving && slow.low > .9 && slow.high <= 1. + 1e-12,
              "relaxation with T2 > T1 does not lead with its largest operator");
        Check(!Channel::Depolarizing(.1).FirstOperatorOf(0).tracePreserving, "a factor drawn without the state has no first operator to try");
    }

    const Channel weakRelax = Channel::ThermalRelaxation(.5, 30., 45.), weakDamping = Channel::AmplitudeDamping(.02);
    const Channel weakGeneralized = Channel::GeneralizedAmplitudeDamping(.03, .02), damping = Channel::AmplitudeDamping(.25);
    Eigen::MatrixXcd keep = Eigen::MatrixXcd::Identity(4, 4), decay = Eigen::MatrixXcd::Zero(4, 4);
    keep(3, 3) = std::sqrt(.95);
    decay(0, 3) = std::sqrt(.05);
    const Channel pairDecay(Channel::KrausOperators{keep, decay});
    const Channel gateError = Channel::TwoQubitDepolarizing(.02).Then(weakRelax, {0}).Then(weakRelax, {1});
    const auto channel = [](const Types::qubits_vector &qubits, const Channel &c) { return std::make_shared<Circuits::QuantumChannelOperation<>>(qubits, c); };
    const auto run = [](const std::shared_ptr<Circuits::Circuit<>> &circuit, bool fusion, bool readFirst, size_t n, uint64_t seed) {
        auto sim = Make(Method::kStatevector, fusion, Backend::kQCSim, n);
        if (readFirst)
            sim->Configure("trajectory_blocks_read_first", "true");
        sim->SetSeed(seed);
        Circuits::OperationState state;
        state.AllocateBits(n);
        circuit->Execute(sim, state);
        sim->Flush();
        return sim;
    };
    // strong: amplitude damping of a quarter, which fails the attempt often.
    const auto build = [&](size_t n, int steps, bool strong, bool resets) {
        auto circuit = CF::CreateCircuit();
        for (int step = 0; step < steps; ++step)
        {
            const size_t a = rng() % n, b = n > 1 ? (a + 1 + rng() % (n - 1)) % n : 0;
            switch (rng() % (n == 1 ? 4 : 8))
            {
            case 0:
                circuit->AddOperation(CF::CreateGate(Kind::kRyGateType, a, 0, 0, angle(rng)));
                circuit->AddOperation(channel({a}, strong ? damping : weakDamping));
                break;
            case 1:
                circuit->AddOperation(CF::CreateGate(Kind::kHadamardGateType, a));
                circuit->AddOperation(channel({a}, weakGeneralized));
                break;
            case 2:
                circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, a, 0, 0, angle(rng)));
                circuit->AddOperation(channel({a}, weakRelax));
                break;
            case 3:
                circuit->AddOperation(CF::CreateGate(Kind::kSxGateType, a));
                if (resets && rng() % 4 == 0)
                    circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{a}));
                break;
            case 4:
                circuit->AddOperation(CF::CreateGate(Kind::kCXGateType, a, b));
                circuit->AddOperation(channel({b, a}, gateError));
                break;
            case 5:
                circuit->AddOperation(CF::CreateGate(Kind::kCZGateType, a, b));
                circuit->AddOperation(channel({a}, weakRelax));
                circuit->AddOperation(channel({b}, strong ? damping : weakDamping));
                break;
            case 6:
                circuit->AddOperation(channel({a, b}, pairDecay));
                break;
            default:
                circuit->AddOperation(CF::CreateGate(Kind::kRxGateType, b, 0, 0, angle(rng)));
                break;
            }
        }
        return circuit;
    };

    Check(Make(Method::kStatevector, true)->GetConfiguration("trajectory_blocks_read_first") == "false", "blocks are read first by default");

    // One block on a register no wider than it: the unfused run draws the same
    // branches, whether the block was applied first or read first.
    uint64_t unread = 0, reread = 0;
    for (size_t n = 1; n <= 3; ++n)
        for (int trial = 0; trial < 150; ++trial)
        {
            const bool strong = trial % 3 == 0;
            const auto circuit = build(n, strong ? 6 : 14, strong, false);
            if (!circuit->HasQuantumChannels())
                continue;
            const uint64_t seed = 5000 + trial;
            const auto fused = run(circuit, true, false, n, seed), plain = run(circuit, false, false, n, seed), first = run(circuit, true, true, n, seed);
            const std::string name = std::to_string(n) + "-qubit unread block, trial " + std::to_string(trial);
            Close(State(*fused), State(*plain), name + " left the unfused trajectory", 1e-10);
            Close(State(*fused), State(*first), name + " differs from the block read first", 1e-10);
            Check(std::abs(State(*fused).norm() - 1.) < 1e-12, name + " is not normalized");
            const auto stats = fused->GetGateFusionStatistics(), reference = first->GetGateFusionStatistics();
            // A block that had to be read after all costs one more write, to correct it.
            Check(reference.unreadBlocks == 0 && reference.rereadBlocks == 0, name + " was not read first when asked");
            Check(stats.backendGates == reference.backendGates + stats.rereadBlocks && stats.unreadBlocks <= n && stats.rereadBlocks <= stats.unreadBlocks,
                  name + " passes: " + std::to_string(stats.backendGates) + " writes, " + std::to_string(stats.unreadBlocks) + " unread, " +
                      std::to_string(stats.rereadBlocks) + " reread");
            unread += stats.unreadBlocks;
            reread += stats.rereadBlocks;
        }
    std::cout << "  one block: " << unread << " applied before they were read, " << reread << " read after all" << std::endl;
    Check(unread > 40, "few blocks were applied without reading the state: " + std::to_string(unread));
    Check(reread < unread / 2, std::to_string(reread) + " of " + std::to_string(unread) + " unread blocks had to be read");

    // The blocks that have to be read after all, on purpose. Close to |1>, amplitude damping of a quarter
    // leaves a norm near 0.77; the block is applied first for draws up to 0.84, so the draws in between
    // find that the channel did something, and the block is corrected from the state it left.
    uint64_t tried = 0, corrected = 0;
    for (int trial = 0; trial < 400; ++trial)
    {
        const size_t n = 1 + trial % 3;
        auto circuit = CF::CreateCircuit();
        for (size_t q = 0; q < n; ++q)
        {
            circuit->AddOperation(CF::CreateGate(Kind::kRyGateType, q, 0, 0, 2.6));
            circuit->AddOperation(channel({q}, damping));
            circuit->AddOperation(CF::CreateGate(Kind::kRzGateType, q, 0, 0, angle(rng)));
        }
        const uint64_t seed = 7000 + trial;
        const auto fused = run(circuit, true, false, n, seed), plain = run(circuit, false, false, n, seed), first = run(circuit, true, true, n, seed);
        const std::string name = "corrected block, trial " + std::to_string(trial);
        Close(State(*fused), State(*plain), name + " left the unfused trajectory", 1e-10);
        Close(State(*fused), State(*first), name + " differs from the block read first", 1e-10);
        Check(std::abs(State(*fused).norm() - 1.) < 1e-12, name + " is not normalized");
        const auto stats = fused->GetGateFusionStatistics();
        Check(stats.backendGates == first->GetGateFusionStatistics().backendGates + stats.rereadBlocks, name + " passes");
        tried += stats.unreadBlocks;
        corrected += stats.rereadBlocks;
    }
    std::cout << "  on purpose: " << tried << " applied before they were read, " << corrected << " read after all" << std::endl;
    Check(corrected >= 15 && corrected < tried / 3, std::to_string(corrected) + " of " + std::to_string(tried) + " blocks were corrected");

    // Wider registers hold many blocks: the norm one block leaves is divided
    // out by the next, or by the flush before anything reads the state. The
    // blocks are resolved in the same order either way, so the trajectory is
    // the one of the blocks read first, with resets and measurements between.
    unread = reread = 0;
    uint64_t passes = 0, passesReadFirst = 0;
    for (const size_t n : {size_t{5}, size_t{8}, size_t{11}})
        for (int trial = 0; trial < 40; ++trial)
        {
            const bool strong = trial % 4 == 0;
            auto circuit = build(n, 90, strong, true);
            // A read in the middle: the state must have its norm there.
            circuit->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{n - 1, 0}}));
            // Named: a range-for does not keep the temporary circuit of its initializer alive.
            const auto tail = build(n, 30, strong, true);
            for (const auto &op : tail->GetOperations())
                circuit->AddOperation(op);
            const uint64_t seed = 9000 + trial;
            const auto fused = run(circuit, true, false, n, seed), first = run(circuit, true, true, n, seed);
            const std::string name = std::to_string(n) + "-qubit unread blocks, trial " + std::to_string(trial);
            Close(State(*fused), State(*first), name + " differ from the blocks read first", 1e-9);
            Check(std::abs(State(*fused).norm() - 1.) < 1e-12, name + " are not normalized");
            const auto stats = fused->GetGateFusionStatistics(), reference = first->GetGateFusionStatistics();
            Check(reference.unreadBlocks == 0 && stats.backendGates == reference.backendGates + stats.rereadBlocks, name + " passes");
            unread += stats.unreadBlocks;
            reread += stats.rereadBlocks;
            passes += stats.backendGates + stats.rereadBlocks; // writes, and the reads of the blocks corrected
            passesReadFirst += 2 * reference.backendGates;     // an upper bound that counts a read for every block
        }
    std::cout << "  many blocks: " << unread << " applied before they were read, " << reread << " read after all" << std::endl;
    Check(unread > 1000 && reread < unread / 4, std::to_string(reread) + " of " + std::to_string(unread) + " unread blocks had to be read");
    Check(passes < passesReadFirst, "applying blocks first saved no pass");

    // A copy taken with blocks still pending is a state of its own, at unit norm.
    {
        const size_t n = 6;
        const auto circuit = build(n, 60, false, false);
        auto sim = Make(Method::kStatevector, true, Backend::kQCSim, n);
        sim->SetSeed(31);
        Circuits::OperationState state;
        state.AllocateBits(n);
        circuit->Execute(sim, state);
        const auto copy = sim->Clone();
        Check(std::abs(State(*copy).norm() - 1.) < 1e-12, "copy of a state with unread blocks is not normalized");
        Close(State(*copy), State(*sim), "copy of a state with unread blocks", 1e-12);
    }

    // A generic gate need not be unitary, and amplitudes given by the caller need not have unit norm. Either leaves
    // the state with a norm this layer cannot know without reading it, and an unread block would take that norm for
    // the probability of its first operators. The blocks that follow are therefore read first, which needs no norm:
    // the branch drawn is the one of the blocks read first, seed by seed. On |1> under a damping of a quarter, with
    // 2 I and I / 2, on the damped qubit (inside the block of the channel) and on another one, through a saved and
    // restored state, on a copy, and from amplitudes; a unitary generic gate keeps the blocks unread.
    {
        const size_t n = 3;
        const uint64_t seeds = 300;
        enum class Via
        {
            kGate,
            kSaved,
            kCopy,
            kAmplitudes
        };
        const auto decays = [&](double scale, Types::qubit_t where, Via via, bool readFirst, uint64_t &unread, std::vector<bool> &outcomes) {
            size_t count = 0;
            for (uint64_t seed = 1; seed <= seeds; ++seed)
            {
                auto sim = Make(Method::kStatevector, true, Backend::kQCSim, n);
                if (readFirst)
                    sim->Configure("trajectory_blocks_read_first", "true");
                sim->SetSeed(seed);
                if (via == Via::kAmplitudes)
                {
                    // |001> times the scale: the state the gates below prepare
                    Eigen::VectorXcd amplitudes = Eigen::VectorXcd::Zero(Eigen::Index{1} << n);
                    amplitudes[1] = scale;
                    sim->InitializeState(n, amplitudes);
                    sim->SetSeed(seed);
                }
                else
                {
                    sim->ApplyX(0);
                    sim->ApplyGenericOneQubitGate(where, Eigen::Matrix2cd(scale * Eigen::Matrix2cd::Identity()));
                }
                if (via == Via::kSaved)
                {
                    sim->SaveState();
                    sim->ApplyH(1);
                    sim->RestoreState();
                }
                if (via == Via::kCopy)
                    sim = sim->Clone();
                sim->ApplyQuantumChannel({0}, damping);
                sim->Flush();
                unread += sim->GetGateFusionStatistics().unreadBlocks;
                // qubit 0 kept in |1> or decayed, whatever norm the state was given
                const Vec state = State(*sim);
                const bool decayed = std::norm(state[0]) > std::norm(state[1]);
                // A sampled channel scales the operator it draws by its weight on the state as it is, so it
                // leaves the state normalized, as with fusion off.
                Check(std::abs(state.squaredNorm() - 1.) < 1e-9, "a channel on a scaled state did not leave it normalized");
                outcomes.push_back(decayed);
                count += decayed;
            }
            return count;
        };
        uint64_t unread = 0, none = 0;
        std::vector<bool> reference, outcomes;
        const size_t expected = decays(1., 0, Via::kGate, true, none, reference);
        // five sigma around a quarter
        Check(none == 0 && std::abs(static_cast<double>(expected) - .25 * seeds) < 5. * std::sqrt(.1875 * seeds), "decays of |1> read first");
        outcomes.clear();
        Check(decays(1., 0, Via::kGate, false, unread, outcomes) == expected && outcomes == reference && unread > seeds / 2,
              "a unitary generic gate changed the blocks after it");
        for (const double scale : {2., .5})
            for (const Via via : {Via::kGate, Via::kSaved, Via::kCopy, Via::kAmplitudes})
                for (const Types::qubit_t where : {Types::qubit_t{0}, Types::qubit_t{2}})
                {
                    if (via == Via::kAmplitudes && where != 0)
                        continue;
                    unread = 0;
                    outcomes.clear();
                    const size_t count = decays(scale, where, via, false, unread, outcomes);
                    const std::string name = "a state scaled by " + std::to_string(scale) +
                                             (via == Via::kAmplitudes ? " from amplitudes" : " by a generic gate") +
                                             (via == Via::kSaved  ? ", saved and restored"
                                              : via == Via::kCopy ? ", copied"
                                                                  : "") +
                                             (where ? " (on another qubit)" : "");
                    Check(unread == 0, name + ": " + std::to_string(unread) + " blocks were applied without reading it");
                    // A copy draws from a stream of its own: its decays keep the rate, not the seeds.
                    Check(via == Via::kCopy ? std::abs(static_cast<double>(count) - .25 * seeds) < 5. * std::sqrt(.1875 * seeds)
                                            : count == expected && outcomes == reference,
                          name + ": " + std::to_string(count) + " decays where the blocks read first give " + std::to_string(expected));
                }
        // A state replaced by one of unit norm is known again.
        {
            auto sim = Make(Method::kStatevector, true, Backend::kQCSim, n);
            sim->ApplyGenericOneQubitGate(0, Eigen::Matrix2cd(2. * Eigen::Matrix2cd::Identity()));
            sim->Reset();
            sim->SetSeed(1);
            sim->ApplyX(0);
            sim->ApplyQuantumChannel({0}, weakDamping);
            sim->Flush();
            Check(sim->GetGateFusionStatistics().unreadBlocks == 1, "a reset state is still read before its blocks");
        }
    }
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
        if (!supported)
        {
            sim->Configure("gate_fusion", "true");
            Check(!sim->IsGateFusionEnabled() && sim->GetGateFusionMaxQubits() == 0, "explicit fusion enabled an unsupported method");
        }
        if (method == Method::kStabilizer)
        {
            sim->AllocateQubits(2);
            sim->Initialize();
            Check(sim->GetSimulationType() == Method::kStabilizer && !sim->IsGateFusionEnabled() && sim->GetGateFusionMaxQubits() == 0,
                  "stabilizer initialization enabled fusion or changed method");
            sim->ApplyH(0);
            sim->ApplyCX(0, 1);
            sim->ApplyS(1);
            sim->Flush();
            Check(std::abs(sim->Probability(0) - .5) < 1e-12 && std::abs(sim->Probability(3) - .5) < 1e-12, "stabilizer Clifford circuit changed");
            Check(sim->GetGateFusionStatistics().fusedBlocks == 0, "stabilizer Clifford gates were fused");
        }
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
        ChannelFusion();
        std::cout << "density-matrix channel fusion passed\n";
        UnreadTrajectoryBlocks();
        std::cout << "statevector trajectory blocks applied before they are read passed\n";
        TrajectoryBlocks();
        std::cout << "statevector trajectory blocks passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
