// Regression tests for worker ownership, terminal measurements and noise
// clones.
#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Noise/NoiseAdd.h"

#include <array>
#include <cmath>
#include <atomic>
#include <chrono>
#include <iostream>
#include <optional>
#include <set>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef MAESTRO_POOL_FORK_HANDLERS
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace
{
using Backend = Simulators::SimulatorType;
using Method = Simulators::SimulationType;
using Circuit = Circuits::Circuit<>;
using CF = Circuits::CircuitFactory<>;
using Gate = Circuits::QuantumGateType;
using Net = Network::SimpleDisconnectedNetwork<>;
using Counts = Circuit::ExecuteResults;
size_t checks = 0;

void Check(bool condition, const char *message)
{
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

void Check(bool condition, const std::string &message)
{
    Check(condition, message.c_str());
}

std::shared_ptr<Net> MakeNetwork(Backend backend, Method method, size_t bits = 2, size_t qubits = 2)
{
    auto network = std::make_shared<Net>(Types::qubits_vector{qubits}, std::vector<size_t>{bits});
    network->SetOptimizeSimulator(false);
    network->GetController()->SetOptimizeCircuit(false);
    network->SetInitialQubitsMapOptimization(false);
    network->SetMPSOptimizeSwaps(false);
    network->SetMaxSimulators(2);
    network->RemoveAllOptimizationSimulatorsAndAdd(backend, method);
    network->CreateSimulator(backend, method);
    return network;
}

class CountedX : public Circuits::XGate<>
{
  public:
    explicit CountedX(std::shared_ptr<std::atomic<size_t>> calls) : XGate(0), calls(std::move(calls))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
    {
        ++*calls;
        XGate::Execute(sim, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<CountedX>(*this);
    }

  private:
    std::shared_ptr<std::atomic<size_t>> calls;
};

class SingleHostRemapper : public Distribution::IRemapper<>
{
  public:
    std::shared_ptr<Circuit> Remap(const std::shared_ptr<Network::INetwork<>> &, const std::shared_ptr<Circuit> &circuit) override
    {
        return circuit;
    }

    unsigned int GetNumberOfOperationsForDistribution() const override
    {
        return 0;
    }

    unsigned int GetNumberOfDistributions() const override
    {
        return 0;
    }

    Distribution::RemapperType GetType() const override
    {
        return Distribution::RemapperType::kLayersRemapper;
    }
};

std::shared_ptr<Circuit> RandomCircuit(bool conditional, std::optional<size_t> seed = 123)
{
    auto circuit = CF::CreateCircuit();
    std::vector<size_t> bits(64);
    std::iota(bits.begin(), bits.end(), 1);
    auto random = std::static_pointer_cast<Circuits::Random<>>(seed ? CF::CreateRandom(bits, *seed) : CF::CreateRandom(bits));
    if (conditional)
        circuit->AddOperation(
            CF::CreateConditionalRandomGen(random, std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{0}, std::vector<bool>{false})));
    else
        circuit->AddOperation(random);
    circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 0), 1));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    return circuit;
}

std::array<Counts, 2> RunRandomJobs(const std::shared_ptr<Circuit> &circuit, bool locked, bool reuse, bool seeded, uint64_t executionSeed = 42)
{
    std::array<Counts, 2> results;
    std::array<std::exception_ptr, 2> errors;
    std::array<std::thread, 2> threads;
    std::mutex mutex;
    auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 65);
    std::atomic<size_t> ready{0};
    for (size_t worker = 0; worker < threads.size(); ++worker)
        threads[worker] = std::thread([&, worker] {
            try
            {
                Network::ExecuteJob<> job(circuit, results[worker], 128, 2, 65, 65, Backend::kQCSim, Method::kMatrixProductState, mutex);
                job.network = network;
                job.optimiseMultipleShotsExecution = reuse;
                job.randomStream = worker;
                if (seeded)
                    job.config.SetConfiguration("seed", std::to_string(Simulators::IState::DeriveSeed(executionSeed, worker)));
                ++ready;
                while (ready.load() != threads.size())
                    std::this_thread::yield();
                if (locked)
                    job.DoWork();
                else
                    job.DoWorkNoLock();
            }
            catch (...)
            {
                errors[worker] = std::current_exception();
            }
        });
    for (auto &thread : threads)
        thread.join();
    for (const auto &error : errors)
        if (error)
            std::rethrow_exception(error);
    return results;
}

void RandomJobs()
{
    for (bool conditional : {false, true})
        for (bool locked : {false, true})
            for (bool reuse : {false, true})
                for (bool seeded : {false, true})
                {
                    const auto circuit = RandomCircuit(conditional);
                    const auto counts = RunRandomJobs(circuit, locked, reuse, seeded);
                    Check(counts == RunRandomJobs(circuit, locked, reuse, seeded), "Classical random results depend on worker scheduling/history");
                    for (const auto &worker : counts)
                    {
                        Check(worker.size() == 128, "Classical RNG was rewound between shots");
                        for (const auto &[bits, count] : worker)
                            Check(bits[0] == bits[1] && count == 1, "Random-controlled gate or per-shot reset failed");
                    }
                    // The chance that independent 64-bit streams overlap here is tiny;
                    // identical seeds/clones instead produce the same 128 outcomes.
                    for (const auto &[bits, count] : counts[0])
                        Check(counts[1].count(bits) == 0, "Workers share a classical RNG stream");
                }

    // Exercise both network dispatchers, including the uneven final job and
    // cloned planner simulators. No-seed jobs also need different streams.
    for (bool onHost : {false, true})
        for (bool optimize : {false, true})
            for (bool seeded : {false, true})
            {
                auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 65);
                network->SetOptimizeSimulator(optimize);
                if (seeded)
                    network->Configure("seed", "42");
                if (!onHost)
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                const auto circuit = RandomCircuit(true);
                const auto run = [&] { return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 257) : network->RepeatedExecute(circuit, 257); };
                const auto counts = run();
                if (counts.size() != 257)
                    std::cerr << "Random dispatch: onHost=" << onHost << " optimize=" << optimize << " seeded=" << seeded << " outcomes=" << counts.size()
                              << " width=" << (counts.empty() ? 0 : counts.begin()->first.size()) << '\n';
                Check(counts.size() == 257, "Network dispatcher reused a classical stream for multiple jobs");
                Check(counts == run(), "Network classical randomness is not reproducible");
            }

    // Check nested generators too, and that job seeding never mutates the input.
    auto nested = CF::CreateCircuit();
    nested->AddOperation(RandomCircuit(true));
    auto first = nested->CloneForExecution(42);
    auto second = nested->CloneForExecution(43);
    auto repeat = nested->CloneForExecution(42);
    auto generator = [](const std::shared_ptr<Circuit> &outer) {
        auto inner = std::static_pointer_cast<Circuit>(outer->GetOperation(0));
        auto conditional = std::static_pointer_cast<Circuits::ConditionalRandomGen<>>(inner->GetOperation(0));
        return conditional->GetOperation();
    };
    Check(generator(first) != generator(second) && generator(first) != generator(nested), "Nested conditional generator is still shared");
    Circuits::OperationState a(65), b(65), c(65), original(65), reference(65);
    generator(first)->Execute(nullptr, a);
    generator(second)->Execute(nullptr, b);
    generator(repeat)->Execute(nullptr, c);
    generator(nested)->Execute(nullptr, original);
    generator(nested)->Clone()->Execute(nullptr, reference);
    Check(a.GetAllBits() == c.GetAllBits() && a.GetAllBits() != b.GetAllBits(), "Nested generator streams are not reproducible and distinct");
    Check(original.GetAllBits() == reference.GetAllBits(), "Executing job copies advanced the caller's generator");
}

void RandomSeedPolicy()
{
    for (bool conditional : {false, true})
        for (bool locked : {false, true})
            for (bool reuse : {false, true})
            {
                const auto unseeded = RandomCircuit(conditional, std::nullopt);
                const auto fresh = RunRandomJobs(unseeded, locked, reuse, false);
                Check(fresh != RunRandomJobs(unseeded, locked, reuse, false), "Unseeded jobs repeated the same classical random sequence");
                Check(fresh[0] != fresh[1], "Unseeded workers reused a random stream");
                const auto seeded = RunRandomJobs(unseeded, locked, reuse, true, 0);
                Check(seeded == RunRandomJobs(unseeded, locked, reuse, true, 0) &&
                          seeded == RunRandomJobs(RandomCircuit(conditional, std::nullopt), locked, reuse, true, 0),
                      "Execution seed zero did not control default-seeded generators");
                Check(seeded != RunRandomJobs(unseeded, locked, reuse, true, 1), "Classical random operations ignored the execution seed");
                const auto explicitZero = RandomCircuit(conditional, 0);
                Check(RunRandomJobs(explicitZero, locked, reuse, false) == RunRandomJobs(RandomCircuit(conditional, 0), locked, reuse, false),
                      "Explicit instruction seed zero was treated as omitted");
            }

    // Both dispatchers remap/clone circuits before creating jobs. Those copies
    // must preserve whether each instruction's seed was explicitly supplied.
    for (bool onHost : {false, true})
        for (bool optimize : {false, true})
            for (size_t workers : {1, 2})
            {
                auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 65);
                network->SetOptimizeSimulator(optimize);
                network->SetMaxSimulators(workers);
                if (!onHost)
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                const auto run = [&](const std::shared_ptr<Circuit> &circuit) {
                    return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 257) : network->RepeatedExecute(circuit, 257);
                };
                const auto circuit = RandomCircuit(true, std::nullopt);
                const auto unseeded = run(circuit);
                Check(unseeded.size() == 257 && unseeded != run(circuit), "Network execution reused an unseeded classical stream");
                Check(run(RandomCircuit(true, 0)) == run(RandomCircuit(true, 0)), "Network remapping lost an explicit instruction seed of zero");
                network->Configure("seed", "0");
                const auto seeded = run(circuit);
                Check(seeded == run(circuit) && seeded == run(RandomCircuit(true, std::nullopt)), "Network seed zero did not control unseeded instructions");
            }

    std::vector<size_t> bits(64);
    std::iota(bits.begin(), bits.end(), 0);
    const auto sample = [](const std::shared_ptr<Circuits::IOperation<>> &op) {
        Circuits::OperationState state(64);
        op->Execute(nullptr, state);
        return state.GetAllBits();
    };
    Check(sample(CF::CreateRandom(bits)) != sample(CF::CreateRandom(bits)), "Direct unseeded generators still default to the same seed");
    Check(sample(CF::CreateRandom(bits, 0)) == sample(CF::CreateRandom(bits, 0)), "Direct generators ignored an explicit seed of zero");
    auto manual = std::make_shared<Circuits::Random<>>(bits);
    manual->Seed(0);
    Check(manual->HasExplicitSeed() && sample(manual) == sample(CF::CreateRandom(bits, 0)), "Calling Seed(0) did not mark an explicitly seeded generator");

    // Entropy for unseeded instructions must not affect explicitly seeded ones,
    // even when they coexist in a nested circuit.
    auto mixed = CF::CreateCircuit();
    mixed->AddOperation(CF::CreateRandom(bits, 0));
    mixed->AddOperation(CF::CreateRandom(bits));
    auto nested = CF::CreateCircuit({mixed});
    Check(nested->HasUnseededRandomOperations(), "Nested unseeded generators were not detected");
    const auto first = std::static_pointer_cast<Circuit>(nested->CloneForExecution(0, 123)->GetOperation(0));
    const auto second = std::static_pointer_cast<Circuit>(nested->CloneForExecution(0, 456)->GetOperation(0));
    Check(sample(first->GetOperation(0)) == sample(second->GetOperation(0)), "Fresh entropy affected an explicitly seeded instruction");
    Check(sample(first->GetOperation(1)) != sample(second->GetOperation(1)), "Fresh entropy did not reach a nested unseeded instruction");
}

void MeasurementJobs()
{
    std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Backend::kQiskitAer);
#endif
    for (auto backend : backends)
        for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
        {
            if (backend != Backend::kQCSim && method == Method::kMatrixProductOperator)
                continue;
            for (bool locked : {false, true})
                for (bool reuse : {false, true})
                    for (bool prepared : {false, true})
                        for (int noise : {0, 1, 2, 3})
                        {
                            if (prepared && !reuse)
                                continue;
                            const auto circuit = CF::CreateCircuit();
                            circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
                            auto earlier = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{1, 0}});
                            auto last = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}});
                            if (noise == 1 || noise == 2)
                                earlier->SetReadout({{1, 0}});
                            if (noise == 1)
                                last->SetReadout({{0, 1}});
                            if (noise == 3)
                                last->SetReadout({{0, 0.5}});
                            circuit->AddOperation(earlier);
                            circuit->AddOperation(last);
                            auto network = MakeNetwork(backend, method);
                            Counts counts;
                            std::mutex mutex;
                            Network::ExecuteJob<> job(circuit, counts, 128, 2, 1, 1, backend, method, mutex);
                            job.network = network;
                            job.config.SetConfiguration("seed", "123");
                            job.optimiseMultipleShotsExecution = reuse;
                            if (prepared)
                            {
                                job.optSim = Simulators::SimulatorsFactory::CreateSimulator(backend, method);
                                job.optSim->SetSeed(123);
                                job.optSim->AllocateQubits(2);
                                job.optSim->Initialize();
                                Circuits::OperationState state(1);
                                job.executedGates = circuit->ExecuteNonMeasurements(job.optSim, state);
                            }
                            if (locked)
                                job.DoWork();
                            else
                                job.DoWorkNoLock();
                            if (noise == 3)
                            {
                                Check(counts.size() == 2, "Readout flips must vary per shot");
                                Check(counts.at({false}) + counts.at({true}) == 128, "Readout sampling lost shots");
                            }
                            else
                            {
                                const bool expected = noise != 1;
                                Check(counts.size() == 1 && counts.at({expected}) == 128, "Terminal sampling changed the last classical "
                                                                                          "write/readout");
                            }
                        }
        }
}

void NoiseClonePolicy()
{
    for (bool optimizeCircuit : {false, true})
        for (bool optimizeSimulator : {false, true})
        {
            auto network = MakeNetwork(Backend::kQCSim, Method::kStatevector);
            network->GetController()->SetOptimizeCircuit(optimizeCircuit);
            network->GetController()->SetOptimizeRotationGates(!optimizeCircuit);
            network->SetOptimizeSimulator(optimizeSimulator);
            network->SetInitialQubitsMapOptimization(!optimizeSimulator);
            auto clone = network->Clone();
            Check(clone->GetController()->GetOptimizeCircuit() == optimizeCircuit, "Network clone lost circuit-optimization policy");
            Check(clone->GetOptimizeSimulator() == optimizeSimulator, "Network clone lost backend-selection policy");
            Check(clone->GetController()->GetOptimizeRotationGates() == !optimizeCircuit, "Network clone lost rotation-optimization policy");
            Check(clone->GetInitialQubitsMapOptimization() == !optimizeSimulator, "Network clone lost initial-qubit-map optimization policy");
        }

    // The cancelling X pair must still execute when circuit optimization is off.
    // Also verify that it executes once per realization, not once per shot.
    for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
        for (size_t workers : {size_t{1}, size_t{2}})
        {
            auto network = MakeNetwork(Backend::kQCSim, method);
            network->SetMaxSimulators(workers);
            auto calls = std::make_shared<std::atomic<size_t>>(0);
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(std::make_shared<CountedX>(calls));
            circuit->AddOperation(std::make_shared<CountedX>(calls));
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
            noise::NoiseModel model;
            model.set_bit_flip(1, 0.5);
            noise::NoiseAdd add;
            add.seed(42);
            const auto counts = add.noisy_execute(circuit, network, 0, model, 56, 7);
            Check(counts.size() == 1 && counts.begin()->second == 56 && !counts.begin()->first[0], "Noise realization batching changed the output");
            // Both mixed-state methods prepare once per explicitly requested
            // realization, independent of the sampling worker budget.
            const size_t jobs = method == Method::kStatevector || method == Method::kDensityMatrix || method == Method::kMatrixProductOperator ? 1 : workers;
            Check(calls->load() == 2 * 7 * jobs, "Noise clone optimized away gates or replayed the prefix per shot");
        }
}

void ConditionalOutputMapping()
{
    for (size_t workers : {size_t{1}, size_t{2}})
        for (bool readout : {false, true})
        {
            auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 20);
            network->SetMaxSimulators(workers);
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
            auto measurement = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 17}});
            if (readout)
                measurement->SetReadout({{0, 1}});
            circuit->AddOperation(std::make_shared<Circuits::ConditionalMeasurement<>>(
                measurement, std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{3}, std::vector<bool>{false})));
            const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 128);
            std::vector<bool> expected(20, false);
            expected[17] = !readout;
            Check(counts.size() == 1 && counts.at(expected) == 128, "Host mapping lost a conditional measurement destination/readout");
        }
}

void NestedCircuitJobs()
{
    // Keep the flat circuit as an independent execution reference. Include gates
    // after nested measurements, and classical data crossing a circuit boundary.
    for (int scenario = 0; scenario < 4; ++scenario)
    {
        auto inner = CF::CreateCircuit();
        auto outer = CF::CreateCircuit();
        auto flat = CF::CreateCircuit();
        if (scenario == 2)
        {
            auto prepare = CF::CreateGate(Gate::kXGateType, 1);
            auto measure = CF::CreateMeasurement({{1, 1}});
            outer->AddOperations({prepare, measure});
            flat->AddOperations({prepare, measure});
            inner->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 0), 1));
        }
        else if (scenario == 3)
        {
            inner->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            inner->AddOperation(CF::CreateMeasurement({{0, 0}}));
            inner->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 1), 0));
            inner->AddOperation(CF::CreateReset({0}));
        }
        else
        {
            inner->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        }
        if (scenario == 0 || scenario == 2)
            inner->AddOperation(CF::CreateMeasurement({{0, 0}}));
        outer->AddOperation(CF::CreateCircuit({inner}));
        flat->AddCircuit(inner);
        if (scenario == 1 || scenario == 3)
        {
            const auto measurement = CF::CreateMeasurement({{0, 0}, {1, 1}});
            outer->AddOperation(measurement);
            flat->AddOperation(measurement);
        }

        for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
            for (bool locked : {false, true})
            {
                auto network = MakeNetwork(Backend::kQCSim, method);
                network->SetInitialQubitsMapOptimization(true);
                network->SetMPSOptimizeSwaps(true);
                network->Configure("matrix_product_state_max_bond_dimension", "16");
                network->SetMPSOptimizationQubitsNumberThreshold(0);
                network->SetMPSOptimizationBondDimensionThreshold(0);
                network->SetLookaheadDepth(0);
                const auto run = [&](const std::shared_ptr<Circuit> &circuit, bool reuse) {
                    Counts counts;
                    std::mutex mutex;
                    Network::ExecuteJob<> job(circuit, counts, 128, 2, 2, 2, Backend::kQCSim, method, mutex);
                    job.network = network;
                    job.config.SetConfiguration("seed", "123");
                    job.optimiseMultipleShotsExecution = reuse;
                    if (locked)
                        job.DoWork();
                    else
                        job.DoWorkNoLock();
                    return counts;
                };
                const auto expected = run(flat, false);
                Check(run(outer, true) == expected, "Nested multishot execution differs from the flat circuit");
                Check(run(outer, false) == expected, "Nested execution cleared the enclosing classical state");
            }

        for (bool onHost : {false, true})
            for (size_t workers : {size_t{1}, size_t{2}})
                for (bool optimize : {false, true})
                {
                    auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState);
                    network->SetMaxSimulators(workers);
                    network->SetOptimizeSimulator(optimize);
                    network->Configure("seed", "123");
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                    const auto run = [&](const std::shared_ptr<Circuit> &circuit) {
                        return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 128) : network->RepeatedExecute(circuit, 128);
                    };
                    Check(run(outer) == run(flat), "Network dispatcher skipped or misexecuted a nested circuit");
                }
    }

    // A nested random stream must advance per shot, with both boundaries
    // retaining the classical state. CloneForExecution must still isolate its
    // generators.
    auto nested = CF::CreateCircuit({CF::CreateCircuit({RandomCircuit(true)})});
    const auto counts = RunRandomJobs(nested, true, true, true);
    Check(counts == RunRandomJobs(nested, true, true, true), "Nested random execution is not reproducible");
    for (const auto &worker : counts)
    {
        Check(worker.size() == 128, "Nested random generator was not run per shot");
        for (const auto &[bits, count] : worker)
            Check(bits[0] == bits[1] && count == 1, "Nested random-controlled gate lost its classical input");
    }

    // A composite containing only classical operations has no qubit queue for
    // MoveMeasurementsAndResets. The composite backend must still complete it.
    std::vector<size_t> bits(64);
    std::iota(bits.begin(), bits.end(), 0);
    const auto classical = CF::CreateCircuit({CF::CreateCircuit({CF::CreateRandom(bits, 123)})});
    for (bool onHost : {false, true})
    {
        auto network = MakeNetwork(Backend::kCompositeQCSim, Method::kStatevector, 64);
        network->SetMaxSimulators(1);
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        const auto counts = onHost ? network->RepeatedExecuteOnHost(classical, 0, 128) : network->RepeatedExecute(classical, 128);
        Check(counts.size() == 128, "Classical-only composite was not executed on the composite backend");
    }
}

void NestedClassicalDataflow()
{
    // A Boolean oracle keeps these checks independent of the flat executor.
    // Exercise both outcomes of each condition, readout flips, reset-to-one,
    // overwritten predicate bits, sparse destinations and empty composites.
    std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Backend::kQiskitAer);
#endif
    for (unsigned scenario = 0; scenario < 16; ++scenario)
    {
        const bool initial0 = scenario & 1, initial1 = scenario & 2;
        const bool flip = scenario & 4, resetToOne = scenario & 8;
        auto flat = CF::CreateCircuit();
        if (initial0)
            flat->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        if (initial1)
            flat->AddOperation(CF::CreateGate(Gate::kXGateType, 1));
        auto first = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 17}}));
        if (flip)
            first->SetReadout({{1, 1}});
        flat->AddOperation(first);
        flat->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 1), 17));
        flat->AddOperation(CF::CreateMeasurement({{1, 3}}));
        flat->AddOperation(CF::CreateConditionalMeasurement(std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}})),
                                                            std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{3}, std::vector<bool>{true})));
        flat->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}, 0, std::vector<bool>{resetToOne}));
        flat->AddOperation(CF::CreateMeasurement({{0, 3}}));
        flat->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 1), 0));
        flat->AddOperation(CF::CreateMeasurement({{1, 1}}));

        std::vector<bool> expected(18, false);
        expected[17] = initial0 != flip;
        const bool measured1 = initial1 != expected[17];
        expected[0] = measured1 && initial0;
        expected[3] = resetToOne;
        expected[1] = measured1 != expected[0];
        const auto &ops = flat->GetOperations();
        for (size_t cut : {size_t{0}, ops.size() / 2, ops.size()})
        {
            auto nested = CF::CreateCircuit({CF::CreateCircuit(Circuit::OperationsVector(ops.begin(), ops.begin() + cut)),
                                             CF::CreateCircuit({CF::CreateCircuit(Circuit::OperationsVector(ops.begin() + cut, ops.end()))})});
            for (auto backend : backends)
                for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
                {
                    if (backend != Backend::kQCSim && method == Method::kMatrixProductOperator)
                        continue;
                    auto network = MakeNetwork(backend, method, 18);
                    for (bool locked : {false, true})
                        for (bool reuse : {false, true})
                        {
                            Counts counts;
                            std::mutex mutex;
                            Network::ExecuteJob<> job(nested, counts, 17, 2, 18, 18, backend, method, mutex);
                            job.network = network;
                            job.config.SetConfiguration("seed", "0");
                            job.optimiseMultipleShotsExecution = reuse;
                            if (locked)
                                job.DoWork();
                            else
                                job.DoWorkNoLock();
                            Check(counts == Counts{{expected, 17}}, "Nested classical dataflow differs from the Boolean oracle");
                        }
                }
        }
    }
}

void NestedShotBoundaries()
{
    auto circuit = CF::CreateCircuit({CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 0), CF::CreateMeasurement({{0, 0}})})});
    circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 1), 0));
    circuit->AddOperation(CF::CreateMeasurement({{1, 1}}));
    for (bool onHost : {false, true})
        for (bool optimize : {false, true})
            for (size_t workers : {size_t{1}, size_t{3}, size_t{8}})
            {
                auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState);
                network->SetMaxSimulators(workers);
                network->SetOptimizeSimulator(optimize);
                network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                for (size_t shots : {size_t{0}, size_t{1}, size_t{2}, size_t{5}, size_t{7}, size_t{257}})
                {
                    const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots) : network->RepeatedExecute(circuit, shots);
                    const Counts expected = shots ? Counts{{{true, true}, shots}} : Counts{};
                    Check(counts == expected, "Nested worker dispatch lost or duplicated shots");
                }
            }
}

class MappingNetwork : public Net
{
  public:
    using Net::MapCircuitOnHost;
    using Net::Net;
};

void ConditionalBitHelpers()
{
    for (bool random : {false, true})
        for (bool nested : {false, true})
        {
            auto circuit = CF::CreateCircuit();
            auto condition = std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{3}, std::vector<bool>{false});
            std::shared_ptr<Circuits::IOperation<>> conditional;
            if (random)
                conditional = CF::CreateConditionalRandomGen(std::static_pointer_cast<Circuits::Random<>>(CF::CreateRandom({0, 17}, 123)), condition);
            else
                conditional = CF::CreateConditionalMeasurement(
                    std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}, {1, 17}})), condition);
            circuit->AddOperation(conditional);
            if (nested)
                circuit = CF::CreateCircuit({circuit});
            Check(conditional->AffectedBits() == std::vector<size_t>{3}, "Conditional predicate bits changed meaning");
            Check(circuit->GetBits() == std::set<size_t>{0, 3, 17} && circuit->AffectedBits() == std::vector<size_t>({0, 3, 17}) &&
                      circuit->GetMinCbitIndex() == 0 && circuit->GetMaxCbitIndex() == 17,
                  "Circuit bit helpers omitted conditional destinations");
            Circuit::BitMapping qubits, reverseBits;
            size_t nq = 0, nc = 0;
            const auto mapped = circuit->RemapToContinuous(qubits, reverseBits, nq, nc);
            Check(nc == 3 && mapped->GetBits() == std::set<size_t>{0, 1, 2}, "Mapping omitted a nested conditional destination");
            auto network = std::make_shared<MappingNetwork>(Types::qubits_vector{2}, std::vector<size_t>{20});
            network->MapCircuitOnHost(circuit, 0, nq, nc, false);
            Check(nc == 18, "Shared host mapping omitted conditional destinations");
        }
}

class ObservedX : public Circuits::XGate<>
{
  public:
    explicit ObservedX(std::shared_ptr<Simulators::ISimulator> &observed) : XGate(0), observed(observed)
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
    {
        observed = sim;
        XGate::Execute(sim, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<ObservedX>(*this);
    }

  private:
    std::shared_ptr<Simulators::ISimulator> &observed;
};

void HostSimulatorReuse()
{
    std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Backend::kQiskitAer);
#endif
    for (auto backend : backends)
        for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
        {
            if (backend != Backend::kQCSim && method == Method::kMatrixProductOperator)
                continue;
            auto network = MakeNetwork(backend, method);
            network->SetMaxSimulators(1);
            network->Configure("seed", "123");
            network->CreateSimulator(backend, method, 2);
            std::shared_ptr<Simulators::ISimulator> observed;
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(std::make_shared<ObservedX>(observed));
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
            // Both an already-sized simulator and one resized from the whole network
            // must be reused, and every invocation must begin in the zero state.
            for (size_t shots : {size_t{8}, size_t{8}, size_t{1}})
            {
                const auto existing = network->GetSimulator();
                existing->ApplyX(0);
                const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
                Check(observed == existing, "Single-worker host execution replaced its simulator");
                Check(counts == Counts{{{true, false}, shots}}, "Reused host simulator retained a previous quantum state");
            }
            // ExecuteOnHost retains a simulator with the caller's register width.
            network->CreateSimulator(backend, method, 2);
            // Exercise consecutive same-sized calls, then a change of register width.
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                const auto existing = network->GetSimulator();
                network->ExecuteOnHost(circuit, 0);
                Check(observed == existing && network->GetState().GetAllBits() == std::vector<bool>({true, false}),
                      "ExecuteOnHost failed to reset/reuse its simulator");
            }
            auto smaller = CF::CreateCircuit();
            smaller->AddOperation(std::make_shared<ObservedX>(observed));
            smaller->AddOperation(CF::CreateMeasurement({{0, 0}}));
            const auto existing = network->GetSimulator();
            const auto counts = network->RepeatedExecuteOnHost(smaller, 0, 8);
            Check(observed == existing && counts == Counts{{{true, false}, 8}}, "Host simulator reuse failed after changing the register width");

            auto random = CF::CreateCircuit();
            random->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            random->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
            network->CreateSimulator(backend, method, 2);
            const auto seeded = network->RepeatedExecuteOnHost(random, 0, 128);
            Check(seeded == network->RepeatedExecuteOnHost(random, 0, 128), "Simulator reuse changed the configured sampling seed");

            // Readout has its own RNG. Retaining a measured state must preserve
            // neither that RNG's position nor quantum/classical state on a new call.
            const auto makeSeeded = [&] {
                auto result = MakeNetwork(backend, method);
                result->SetMaxSimulators(1);
                result->Configure("seed", "0");
                result->CreateSimulator(backend, method, 2);
                return result;
            };
            auto measured = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}, {1, 1}}));
            measured->SetReadout({{0.2, 0.35}, {0.4, 0.1}});
            auto noisy = CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 0), measured});
            const auto fresh = makeSeeded()->RepeatedExecuteOnHost(noisy, 0, 257);
            auto reused = makeSeeded();
            reused->ExecuteOnHost(CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 1), measured}), 0);
            Check(fresh.size() == 4 && reused->RepeatedExecuteOnHost(noisy, 0, 257) == fresh, "Host reuse leaked retained state or readout RNG position");
        }
}

void QCSimWorkerRandomStreams()
{
    // Each shot writes 64 independent quantum measurements into classical bits.
    // Reusing a planner's RNG in multiple workers duplicates entire bitstrings.
    auto circuit = CF::CreateCircuit();
    for (size_t bit = 0; bit < 64; ++bit)
    {
        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
        circuit->AddOperation(CF::CreateMeasurement({{0, bit}}));
    }
    for (auto method : {Method::kDensityMatrix, Method::kStabilizer, Method::kExtendedStabilizer})
        for (bool onHost : {false, true})
            for (bool seeded : {false, true})
            {
                auto network = MakeNetwork(Backend::kQCSim, method, 64);
                network->SetOptimizeSimulator(true);
                if (seeded)
                    network->Configure("seed", "0");
                if (!onHost)
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                const auto run = [&] { return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 257) : network->RepeatedExecute(circuit, 257); };
                const auto first = run();
                Check(first.size() == 257, "QCSim workers duplicated quantum RNG streams");
                for (const auto &[bits, count] : first)
                    Check(count == 1, "QCSim workers repeated a 64-bit measurement");
                Check((first == run()) == seeded, "QCSim worker streams mishandled an omitted or zero seed");
            }
}

void QCSimSamplingRandomStreams(Method method)
{
    // The Clifford backend has no MeasureNoCollapse; it samples through counts.
    const int apis = method == Method::kStabilizer ? 2 : 4;
    const std::string name = method == Method::kStabilizer ? "Clifford" : "Extended-stabilizer";
    const auto create = [method] {
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQCSim, method);
        sim->AllocateQubits(3);
        sim->Initialize();
        sim->ApplyX(0);
        sim->SaveState(); // A checkpoint distinct from the sampled superposition.
        sim->ApplyX(0);
        for (size_t q = 0; q < 3; ++q)
            sim->ApplyH(q);
        return sim;
    };
    const auto sequence = [](const auto &sim, int api) {
        std::vector<std::vector<bool>> results;
        for (size_t shot = 0; shot < 128; ++shot)
        {
            if (api == 1)
            {
                results.push_back(sim->SampleCountsMany({2, 0, 1}, 1).begin()->first);
            }
            else if (api == 3)
            {
                results.push_back(sim->MeasureNoCollapseMany());
            }
            else
            {
                const auto packed = api == 0 ? sim->SampleCounts({2, 0, 1}, 1).begin()->first : sim->MeasureNoCollapse();
                results.push_back({bool(packed & 1), bool(packed & 2), bool(packed & 4)});
            }
        }
        return results;
    };
    for (int api = 0; api < apis; ++api)
    {
        auto sim = create();
        const auto unseeded = sequence(sim, api);
        Check(std::set<std::vector<bool>>(unseeded.begin(), unseeded.end()).size() > 1, "Unseeded " + name + " sampling replayed one outcome");
        // Seeds differing only above bit 31 must give different streams.
        for (uint64_t seed : {uint64_t{0}, uint64_t{1} << 32, UINT64_MAX})
        {
            sim->SetSeed(seed);
            const auto first = sequence(sim, api);
            Check(first != sequence(sim, api), name + " sampling did not advance its RNG");
            sim->SetSeed(seed);
            Check(first == sequence(sim, api), name + " sampling did not replay an explicit seed");
            if (seed == 0)
            {
                sim->SetSeed(uint64_t{1} << 32);
                Check(first != sequence(sim, api), name + " sampling ignored the high seed bits");
            }
        }
        Check(std::abs(sim->ExpectationValue("XXX") - 1.) < 1e-12, name + " sampling changed the quantum state");
        sim->RestoreState();
        Check(std::abs(sim->Probability(1) - 1.) < 1e-12, name + " sampling changed the saved checkpoint");
    }
    // Collapsing measurements draw from the backend's own RNG, which must see
    // every seed bit; sampling above draws worker seeds from the state's RNG.
    const auto measured = [&](uint64_t seed) {
        auto sim = create();
        sim->SetSeed(seed);
        std::vector<size_t> outcomes;
        for (size_t shot = 0; shot < 64; ++shot)
        {
            outcomes.push_back(sim->Measure({0, 1, 2}));
            for (size_t q = 0; q < 3; ++q)
                sim->ApplyH(q);
        }
        return outcomes;
    };
    Check(measured(0) == measured(0), name + " measurements did not replay an explicit seed");
    Check(measured(0) != measured(uint64_t{1} << 32), name + " measurements ignored the high seed bits");

    auto sim = create();
    sim->SetSeed(0);
    const auto first = sim->SampleCounts({2, 0, 1}, 1024);
    Check(first != sim->SampleCounts({2, 0, 1}, 1024), name + " sampling replayed an entire batch");
    sim->SetSeed(0);
    Check(first == sim->SampleCounts({2, 0, 1}, 1024), name + " batch sampling ignored reseeding");
}

void AdaptiveWorkerCounts()
{
    auto terminal = CF::CreateCircuit();
    terminal->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    terminal->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    auto readout = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    readout->SetReadout({{0.2, 0.35}, {0.4, 0.1}});
    terminal->AddOperation(readout);
    auto trajectory = CF::CreateCircuit();
    for (size_t bit = 0; bit < 16; ++bit)
    {
        trajectory->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
        trajectory->AddOperation(CF::CreateMeasurement({{0, bit}}));
    }
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kPathIntegral})
        for (bool onHost : {false, true})
            for (bool optimize : {false, true})
                for (const auto &circuit : {terminal, trajectory})
                {
                    Counts reference;
                    for (size_t workers : {1, 2, 3, 8})
                    {
                        auto network = MakeNetwork(Backend::kQCSim, method, 16);
                        network->SetOptimizeSimulator(optimize);
                        network->SetMaxSimulators(workers);
                        network->SetTrajectoryMinWorkerMicros(0);
                        network->Configure("seed", "1099511627776");
                        if (!onHost)
                            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                        const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, 1027) : network->RepeatedExecute(circuit, 1027);
                        if (workers == 1)
                            reference = counts;
                        Check(counts == reference, "QCSim counts/readout changed with outer worker count");
                    }
                }
}

class ThreadObservedX : public Circuits::XGate<>
{
  public:
    ThreadObservedX(bool expected, std::shared_ptr<std::atomic<size_t>> calls, std::shared_ptr<std::atomic<bool>> wrong,
                    std::shared_ptr<std::atomic<int>> budget = {})
        : XGate(0), expected(expected), calls(std::move(calls)), wrong(std::move(wrong)), budget(std::move(budget))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
    {
        if (sim->GetMultithreading() != (budget ? budget->load() > 1 : expected))
            wrong->store(true);
        if (budget && sim->GetConfiguration("reproducible_trajectory") != "true")
            wrong->store(true);
#ifdef _OPENMP
        if (budget && omp_get_max_threads() != budget->load())
            wrong->store(true);
#endif
        ++*calls;
        XGate::Execute(sim, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<ThreadObservedX>(*this);
    }

  private:
    bool expected;
    std::shared_ptr<std::atomic<size_t>> calls;
    std::shared_ptr<std::atomic<bool>> wrong;
    std::shared_ptr<std::atomic<int>> budget;
};

#if defined(_OPENMP) && !defined(NO_QISKIT_AER)
class WorkerThreadsX : public Circuits::XGate<>
{
  public:
    WorkerThreadsX(std::shared_ptr<std::atomic<size_t>> calls, std::shared_ptr<std::atomic<bool>> wrong)
        : XGate(0), calls(std::move(calls)), wrong(std::move(wrong))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
    {
        if (sim->GetMultithreading())
            wrong->store(true);
#if !defined(_MSC_VER) || defined(__clang__) || defined(_OPENMP_LLVM_RUNTIME)
        // Shot workers set this limit except on legacy MSVC OpenMP, where it is process-wide.
        if (omp_get_max_threads() != 1)
            wrong->store(true);
#endif
        ++*calls;
        XGate::Execute(sim, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<WorkerThreadsX>(*this);
    }

  private:
    std::shared_ptr<std::atomic<size_t>> calls;
    std::shared_ptr<std::atomic<bool>> wrong;
};

// Anything the simulator calls on a shot worker, a BLAS built on OpenMP for example, has to stay on
// that thread when multithreading is off, for prepared simulators and for those the job creates.
void WorkerThreadsFollowFlag()
{
    const int previous = omp_get_max_threads();
    omp_set_num_threads(4);
    for (bool prepared : {false, true})
    {
        const auto calls = std::make_shared<std::atomic<size_t>>(0);
        const auto wrong = std::make_shared<std::atomic<bool>>(false);
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
        circuit->AddOperation(std::make_shared<WorkerThreadsX>(calls, wrong));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
        Counts counts;
        std::mutex mutex;
        Network::ExecuteJob<> job(circuit, counts, 4, 2, 1, 1, Backend::kQiskitAer, Method::kMatrixProductState, mutex);
        job.network = MakeNetwork(Backend::kQiskitAer, Method::kMatrixProductState);
        job.optimiseMultipleShotsExecution = true;
        if (prepared)
        {
            job.optSim = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQiskitAer, Method::kMatrixProductState);
            job.optSim->SetMultithreading(false);
            job.optSim->AllocateQubits(2);
            job.optSim->Initialize();
            job.executedGates.assign(circuit->size(), false);
        }
        job.DoWork();
        Check(calls->load() > 0 && !wrong->load(), "Shot worker kept its OpenMP team with multithreading off");
        Check(omp_get_max_threads() == 4, "Shot worker leaked its thread limit into the caller");
    }
    omp_set_num_threads(previous);
}
#endif

void LargeTrajectoryWorkers()
{
    const auto calls = std::make_shared<std::atomic<size_t>>(0);
    const auto wrong = std::make_shared<std::atomic<bool>>(false);
    const auto budget = std::make_shared<std::atomic<int>>(1);
    auto circuit = CF::CreateCircuit();
    for (size_t q = 0; q < 15; ++q)
    {
        circuit->AddOperation(CF::CreateGate(Gate::kRyGateType, q, 0, 0, 0.137 + 0.039 * q));
        if (q)
            circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(q - 1, q));
    }
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    // After the first collapse, so simulator-estimation/prefix preparation is
    // not mistaken for execution inside a logical trajectory block.
    circuit->AddOperation(std::make_shared<ThreadObservedX>(false, calls, wrong, budget));
    circuit->AddOperation(CF::CreateReset({7}));
    circuit->AddOperation(CF::CreateGate(Gate::kRyGateType, 14, 0, 0, 0.732));
    circuit->AddOperation(CF::CreateMeasurement({{14, 1}, {0, 2}}));
    Counts expected;
    for (bool onHost : {false, true})
        for (int team : {1, 3, 8})
            for (size_t workers : {1, 2, 3, 8})
            {
                auto network = MakeNetwork(Backend::kQCSim, Method::kStatevector, 3, 15);
                // This checks the fixed 256-shot compatibility policy's teams.
                network->Configure("trajectory_policy", "block_v1");
                network->Configure("gate_fusion", "false");
                network->Configure("seed", "1099511627776");
                network->SetMaxSimulators(workers);
                network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
#ifdef _OPENMP
                omp_set_num_threads(team);
                // The network caps workers at the hardware thread count (2 on a
                // small CI runner), and 1027 shots make five 256-shot blocks.
                const size_t outer = std::min<size_t>(network->GetMaxSimulators(), 5);
                budget->store(std::max(1, team / static_cast<int>(outer)));
#endif
                calls->store(0);
                wrong->store(false);
                const auto actual = onHost ? network->RepeatedExecuteOnHost(circuit, 0, 1027) : network->RepeatedExecute(circuit, 1027);
                Check(calls->load() == 1027 && !wrong->load(), "Trajectory blocks did not respect their internal thread budget");
#ifdef _OPENMP
                Check(omp_get_max_threads() == team, "Trajectory job leaked its OpenMP budget into the caller");
#endif
                if (expected.empty())
                    expected = actual;
                Check(actual == expected, "Large trajectory counts changed with outer workers/OpenMP team size");
            }
}

class ObserveThreads : public Circuits::XGate<>
{
  public:
    struct Seen
    {
        std::mutex mutex;
        std::set<std::thread::id> workers;
        int budget = 0;
    };

    explicit ObserveThreads(std::shared_ptr<Seen> seen) : XGate(0), seen(std::move(seen))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
    {
        {
            const std::lock_guard lock(seen->mutex);
            seen->workers.insert(std::this_thread::get_id());
#ifdef _OPENMP
            seen->budget = std::max(seen->budget, omp_get_max_threads());
#endif
        }
        XGate::Execute(sim, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<ObserveThreads>(*this);
    }

  private:
    std::shared_ptr<Seen> seen;
};

void LargeTrajectoryThreadBudget()
{
#ifdef _OPENMP
    // Memory-bound registers: all workers together use no more threads than
    // QCSim's team for one such register, even when OpenMP allows every CPU.
    const int previous = omp_get_max_threads();
    omp_set_num_threads(std::max(8, omp_get_num_procs()));
    const size_t qubits = 20;
    const int cap = QC::QubitRegisterCalculator<>::GetParallelTeamSize(size_t{1} << qubits);
    for (const char *policy : {"shot_v1", "block_v1"})
    {
        const auto seen = std::make_shared<ObserveThreads::Seen>();
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
        circuit->AddOperation(std::make_shared<ObserveThreads>(seen));
        circuit->AddOperation(CF::CreateMeasurement({{0, 1}}));
        auto network = MakeNetwork(Backend::kQCSim, Method::kStatevector, 2, qubits);
        network->SetMaxSimulators(64);
        network->SetTrajectoryMinWorkerMicros(0);
        network->Configure("trajectory_policy", policy);
        network->Configure("seed", "80");
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        const auto counts = network->RepeatedExecute(circuit, 600);
        size_t total = 0;
        for (const auto &[bits, count] : counts)
            total += count;
        Check(total == 600 && seen->budget > 0, "Large trajectory budget test did not execute");
        Check(static_cast<int>(seen->workers.size()) * seen->budget <= cap,
              std::string("Large trajectories used more threads than QCSim's team (") + policy + ")");
    }
    omp_set_num_threads(previous);
#endif
}

void SamplingJobThreading()
{
    for (auto method : {Method::kStatevector, Method::kDensityMatrix})
        for (const char *policy : {"legacy", "reproducible_v1"})
            for (bool noLock : {false, true})
            {
                auto circuit = CF::CreateCircuit();
                circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
                auto network = MakeNetwork(Backend::kQCSim, method);
                Counts counts;
                std::mutex mutex;
                Network::ExecuteJob<> job(circuit, counts, 32, 2, 1, 1, Backend::kQCSim, method, mutex);
                job.network = network;
                job.config.SetConfiguration("sampling_policy", policy);
                job.config.SetConfiguration("seed", "81");
                job.optimiseMultipleShotsExecution = true;
                bool checked = false;
                job.onSimulatorReady = [&](const Simulators::ISimulator &sim) {
                    checked = true;
                    Check(sim.GetMultithreading() == (noLock || std::string(policy) == "legacy"),
                          "Fresh simulator used sampling policy before applying its configuration");
                };
                if (noLock)
                    job.DoWorkNoLock();
                else
                    job.DoWork();
                Check(checked && counts.size() == 2, "Sampling job threading test did not execute");
            }
    // Protect the real single-job network path: terminal batches still enable
    // internal threading even after trajectories were made serial internally.
    for (bool onHost : {false, true})
        for (auto method : {Method::kStatevector, Method::kDensityMatrix})
        {
            const auto calls = std::make_shared<std::atomic<size_t>>(0);
            const auto wrong = std::make_shared<std::atomic<bool>>(false);
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(std::make_shared<ThreadObservedX>(true, calls, wrong));
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
            auto network = MakeNetwork(Backend::kQCSim, method);
            network->Configure("gate_fusion", "false");
            network->SetMaxSimulators(1);
            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
            if (onHost)
                network->RepeatedExecuteOnHost(circuit, 0, 32);
            else
                network->RepeatedExecute(circuit, 32);
            Check(calls->load() > 0 && !wrong->load(), "Terminal single-job sampling disabled internal threading");
        }
}

void TerminalDensityDispatch()
{
    std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Backend::kQiskitAer);
#endif
    for (auto backend : backends)
        for (const char *policy : {"legacy", "reproducible_v1"})
            for (bool prepared : {false, true})
                for (bool onHost : {false, true})
                {
                    Counts reference;
                    for (size_t workers : {1, 4})
                    {
                        auto calls = std::make_shared<std::atomic<size_t>>(0);
                        auto circuit = CF::CreateCircuit();
                        circuit->AddOperation(std::make_shared<CountedX>(calls));
                        circuit->AddOperation(std::make_shared<CountedX>(calls));
                        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                        circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
                        auto measurement = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}, {1, 1}}));
                        measurement->SetReadout({{1., 1.}, {0., 0.}});
                        circuit->AddOperation(measurement);
                        auto network = MakeNetwork(backend, Method::kDensityMatrix);
                        network->SetOptimizeSimulator(prepared);
                        network->SetMaxSimulators(workers);
                        network->Configure("sampling_policy", policy);
                        network->Configure("seed", "81");
                        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                        const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, 1000) : network->RepeatedExecute(circuit, 1000);
                        Check(calls->load() == 2, "Terminal density sampling repeated preparation across workers");
                        Check(counts.size() == 2 && counts.at({true, false}) + counts.at({false, true}) == 1000,
                              "Terminal density sampling lost Bell correlations or readout flips");
                        if (workers == 1)
                            reference = counts;
                        Check(counts == reference, "Terminal density counts changed with maximum simulator count");
                    }
                }

    // A terminal batch must not replace a caller's existing checkpoint with a
    // snapshot that will never be restored by the batch.
    for (auto backend : backends)
        for (bool estimator : {false, true})
        {
            auto network = MakeNetwork(backend, Method::kDensityMatrix);
            auto sim = network->GetSimulator();
            sim->Clear();
            sim->AllocateQubits(2);
            sim->Initialize();
            sim->ApplyX(1);
            sim->SaveState();
            sim->Reset();
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
            if (estimator)
            {
                std::vector<bool> executed;
                Estimators::SimulatorsEstimatorInterface<>::ExecuteUpToMeasurements(circuit, 2, 2, 2, sim, executed);
            }
            else
            {
                Counts counts;
                std::mutex mutex;
                Network::ExecuteJob<> job(circuit, counts, 10, 2, 2, 2, backend, Method::kDensityMatrix, mutex);
                job.network = network;
                job.optSim = sim;
                if (backend == Backend::kQCSim)
                {
                    sim->Configure("reproducible_trajectory", "true");
                    job.config.SetConfiguration("reproducible_trajectory", "false");
                }
                job.executedGates.assign(circuit->size(), false);
                job.DoWorkNoLock();
                if (backend == Backend::kQCSim)
                    Check(sim->GetConfiguration("reproducible_trajectory") == "false", "Terminal job retained trajectory arithmetic mode");
                Check(counts.at({true, false}) == 10, "Terminal density job did not execute its prefix");
            }
            sim->RestoreState();
            Check(std::abs(sim->Probability(2) - 1.) < 1e-12, "Terminal density execution saved an unused checkpoint");
        }
}

void TerminalMpoDispatch()
{
    // A CPU MPO draws each terminal shot by a serial contraction. Shot workers
    // may share that work only by cloning one prepared MPO, never by evolving
    // the circuit again: with backend selection (which prepares the MPO) and
    // with a fixed backend (prepared by the network itself).
    for (bool prepared : {false, true})
        for (bool onHost : {false, true})
            for (size_t workers : {1, 4})
                for (bool seeded : {false, true})
                {
                    auto calls = std::make_shared<std::atomic<size_t>>(0);
                    auto circuit = CF::CreateCircuit();
                    circuit->AddOperation(std::make_shared<CountedX>(calls));
                    circuit->AddOperation(std::make_shared<CountedX>(calls));
                    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
                    auto measurement = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}, {1, 1}}));
                    measurement->SetReadout({{1., 1.}, {0., 0.}});
                    circuit->AddOperation(measurement);
                    auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductOperator);
                    network->SetOptimizeSimulator(prepared);
                    network->SetMaxSimulators(workers);
                    if (seeded)
                        network->Configure("seed", "81");
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                    const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, 2000) : network->RepeatedExecute(circuit, 2000);
                    Check(calls->load() == 2, "Terminal MPO sampling repeated preparation across workers");
                    Check(counts.size() == 2 && counts.at({true, false}) + counts.at({false, true}) == 2000,
                          "Terminal MPO sampling lost Bell correlations or readout flips");
                    // Identical clone streams would still balance on average;
                    // they cannot leave both outcomes this close to 1000.
                    Check(counts.at({true, false}) > 850 && counts.at({false, true}) > 850, "Terminal MPO workers sampled a biased distribution");
                    Check(network->GetLastSimulationType() == Method::kMatrixProductOperator, "Terminal MPO sampling changed the backend");
                }
}

void NestedTerminalMeasurementsShareThePrefix()
{
    // A nested circuit that ends by measuring is inlined: its gates evolve
    // once and the shots are sampled, not replayed.
    for (const auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductOperator})
        for (bool onHost : {false, true})
        {
            auto calls = std::make_shared<std::atomic<size_t>>(0);
            auto nested = CF::CreateCircuit();
            nested->AddOperation(std::make_shared<CountedX>(calls));
            nested->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
            nested->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(nested);
            auto network = MakeNetwork(Backend::kQCSim, method);
            network->SetMaxSimulators(4);
            network->Configure("seed", "5");
            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
            const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, 2000) : network->RepeatedExecute(circuit, 2000);
            Check(calls->load() == 1, "A nested terminal measurement replayed its prefix per shot");
            Check(counts.size() == 2 && counts.at({true, false}) > 850 && counts.at({true, true}) > 850, "Inlined nested measurements changed the outcome");
            Check(nested->GetOperations().size() == 3, "Inlining modified the caller's nested circuit");
        }
}

void KeptMixedCandidateIsPrepared()
{
    // Several mixed-state candidates: the network's current one is chosen
    // first and prepared afresh, even after host execution cleared it, so
    // terminal MPO workers can clone it.
    auto network = std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{2});
    network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kMatrixProductOperator);
    network->AddOptimizationSimulator(Backend::kQCSim, Method::kDensityMatrix);
    network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
    network->SetOptimizeSimulator(true);
    network->GetSimulator()->Clear();
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    size_t shots = 100;
    auto type = Backend::kQCSim;
    auto method = Method::kMatrixProductOperator;
    std::vector<bool> executed;
    const auto prepared = network->ChooseBestSimulator(circuit, shots, 2, 2, 2, type, method, executed);
    Check(prepared && method == Method::kMatrixProductOperator && prepared->GetNumberOfQubits() == 2,
          "The kept mixed-state candidate was not prepared for shot workers");

    for (bool onHost : {false, true})
    {
        auto calls = std::make_shared<std::atomic<size_t>>(0);
        auto terminal = CF::CreateCircuit();
        terminal->AddOperation(std::make_shared<CountedX>(calls));
        terminal->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
        terminal->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
        network->SetMaxSimulators(4);
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        const auto counts = onHost ? network->RepeatedExecuteOnHost(terminal, 0, 2000) : network->RepeatedExecute(terminal, 2000);
        Check(calls->load() == 1 && network->GetLastSimulationType() == Method::kMatrixProductOperator, "The kept MPO candidate was evolved more than once");
        Check(counts.size() == 2 && counts.at({true, false}) > 850 && counts.at({true, true}) > 850, "The kept MPO candidate sampled wrongly");
    }
}

class ThreadingNetwork : public Net
{
  public:
    using Net::ExecutionThreads;
    using Net::Net;
};

void MultinomialTerminalDispatch()
{
    auto network = std::make_shared<ThreadingNetwork>(Types::qubits_vector{10}, std::vector<size_t>{10});
    network->SetMaxSimulators(4);
    network->Configure("counts_policy", "multinomial_v1");
    auto circuit = CF::CreateCircuit();
    std::shared_ptr<Simulators::ISimulator> prepared;
    std::vector<bool> executed;
    const auto workers = [&](Method method, size_t bits, size_t shots, bool terminal = true) {
        return network->ExecutionThreads(circuit, 10, 10, 10, shots, Backend::kQCSim, method, terminal, bits, prepared, executed);
    };
    Check(workers(Method::kExtendedStabilizer, 8, 2048) == 1 && workers(Method::kExtendedStabilizer, 8, 2047) == 4, "Extended marginal crossover ignored");
    Check(workers(Method::kExtendedStabilizer, 9, 8192) == 4 && workers(Method::kExtendedStabilizer, 1, 2048, false) == 4,
          "Unsupported extended sampling lost its parallel workers");
    Check(workers(Method::kMatrixProductState, 1, 256) == 1 && workers(Method::kMatrixProductState, 2, 8192) == 4,
          "MPS marginal dispatch used the wrong measurement width");
    Check(workers(Method::kMatrixProductOperator, 1, 256) == 1, "MPO marginal dispatch split its batch");
    // A truncating MPO keeps the native sampler, and with it the shot workers.
    network->Configure("matrix_product_state_max_bond_dimension", "8");
    Check(workers(Method::kMatrixProductOperator, 1, 256) != 1, "A truncating MPO lost its shot workers");
    network->RemoveConfiguration("matrix_product_state_max_bond_dimension");
    prepared.reset();
    network->Configure("mps_sample_measure_algorithm", "mps_apply_measure");
    Check(workers(Method::kMatrixProductState, 1, 8192) == 4, "Explicit MPS measurement algorithm lost its parallel workers");
    network->Configure("sampling_policy", "reproducible_v1");
    Check(workers(Method::kExtendedStabilizer, 1, 8192) == 4, "Explicit sampling policy changed terminal shot partitioning");

    // Both public entry points should pass the distinct measured width and
    // draw one batch with the same seed regardless of the worker limit.
    for (auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator, Method::kExtendedStabilizer})
    {
        auto net = MakeNetwork(Backend::kQCSim, method);
        net->Configure("counts_policy", "multinomial_v1");
        auto measured = CF::CreateCircuit();
        measured->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
        measured->AddOperation(CF::CreateMeasurement({{0, 0}, {0, 1}}));
        for (bool onHost : {true, false})
        {
            net->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
            const auto run = [&](size_t limit) {
                net->SetMaxSimulators(limit);
                net->Configure("seed", "0");
                return onHost ? net->RepeatedExecuteOnHost(measured, 0, 256) : net->RepeatedExecute(measured, 256);
            };
            const auto one = run(1);
            Check(one == run(4) && one.size() == 2 && one.at({false, false}) + one.at({true, true}) == 256,
                  "Terminal count policy split a seeded marginal batch");
        }
    }
}

void ClearedMpoStillSamplesInParallel()
{
    // [MPO, statevector] without an estimator keeps the network's MPO, which
    // host execution has cleared. Terminal workers must still get a prepared
    // MPO of the host register to clone, not fall back to one worker.
    auto network = std::make_shared<ThreadingNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
    network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kMatrixProductOperator);
    network->AddOptimizationSimulator(Backend::kQCSim, Method::kStatevector);
    network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
    network->SetOptimizeSimulator(true);
    network->SetMaxSimulators(4);
    network->GetSimulator()->Clear();
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(std::make_shared<CountedX>(calls));
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    std::shared_ptr<Simulators::ISimulator> prepared;
    std::vector<bool> executed;
    const auto threads = network->ExecutionThreads(circuit, 2, 2, 2, 1000, Backend::kQCSim, Method::kMatrixProductOperator, true, 2, prepared, executed);
    Check(threads == 4 && prepared && prepared->GetNumberOfQubits() == 2 && calls->load() == 1, "A cleared MPO forced terminal sampling onto one worker");

    for (bool onHost : {false, true})
    {
        calls->store(0);
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, 2000) : network->RepeatedExecute(circuit, 2000);
        Check(calls->load() == 1 && network->GetLastSimulationType() == Method::kMatrixProductOperator,
              "The mixed/pure fallback evolved its MPO more than once");
        Check(counts.size() == 2 && counts.at({true, false}) > 850 && counts.at({true, true}) > 850, "The mixed/pure fallback sampled wrongly");
    }
}

void ClassicalBitIdsDoNotDeferWires()
{
    // A deferred conditional reads classical bit c3. That says nothing about
    // wire q3: the deterministic gate on q3 belongs to the shared prefix.
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 3}}));
    circuit->AddOperation(CF::CreateConditionalGate(std::static_pointer_cast<Circuits::IGateOperation<>>(CF::CreateGate(Gate::kXGateType, 0)),
                                                    std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{3}, std::vector<bool>{true})));
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 3));
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    for (const bool exact : {false, true})
    {
        Circuits::OperationState state(4);
        const auto mask = circuit->ExecuteNonMeasurements(nullptr, state, nullptr, exact);
        Check(mask == std::vector<bool>({false, false, true, false}), "A classical bit ID deferred the gate on the same-numbered wire");
    }
}

void MixedStateCandidates()
{
    // Several mixed-state candidates have no cost model. Selection must still
    // choose one of them, not keep a placeholder simulator outside the set
    // (which cannot apply the channel), even without an estimator.
    for (const bool clifford : {false, true})
    {
        auto network = std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{2});
        network->GetController()->SetOptimizeCircuit(false);
        network->SetInitialQubitsMapOptimization(false);
        network->SetMPSOptimizeSwaps(false);
        network->SetMaxSimulators(1);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kDensityMatrix);
        network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
        network->CreateSimulator(); // the MPS placeholder ConfigureNetwork leaves
        network->SetOptimizeSimulator(true);
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        if (!clifford)
            circuit->AddOperation(
                std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(1.)));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
        const auto counts = network->RepeatedExecute(circuit, 64);
        const auto method = network->GetLastSimulationType();
        Check(method == Method::kDensityMatrix || method == Method::kMatrixProductOperator,
              "Mixed-state candidates fell back to a non-candidate backend (method " + std::to_string(static_cast<int>(method)) +
                  (clifford ? ", Clifford)" : ", channel)"));
        Check(counts.size() == 1 && counts.begin()->second == 64 && counts.begin()->first[0] == clifford,
              "Mixed-state candidate execution changed the outcome");
    }
}

void SmallTrajectoryBudgets()
{
    for (auto method : {Method::kStatevector, Method::kDensityMatrix})
        for (size_t shots : {size_t{100}, size_t{257}})
            for (bool prepared : {false, true})
            {
                const size_t qubits = method == Method::kStatevector ? 15 : 8;
                Counts reference;
                for (bool onHost : {false, true})
                    for (int team : {1, 3, 8})
                    {
#ifdef _OPENMP
                        omp_set_num_threads(team);
#endif
                        auto circuit = CF::CreateCircuit();
                        for (size_t q = 0; q < qubits; ++q)
                        {
                            circuit->AddOperation(CF::CreateGate(Gate::kRyGateType, q, 0, 0, .31 + .03 * q));
                            circuit->AddOperation(CF::CreateGate(Gate::kRzGateType, q, 0, 0, .17 + .02 * q));
                        }
                        auto readout = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}}));
                        readout->SetReadout({{.2, .35}});
                        circuit->AddOperation(readout);
                        circuit->AddOperation(CF::CreateReset({qubits / 2}));
                        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, qubits / 2));
                        circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kRyGateType, qubits - 1, 0, 0, .51), 0));
                        circuit->AddOperation(CF::CreateMeasurement({{qubits / 2, 1}, {qubits - 1, 2}}));
                        auto network = MakeNetwork(Backend::kQCSim, method, 3, qubits);
                        network->SetOptimizeSimulator(prepared);
                        network->SetMaxSimulators(8);
                        network->Configure("gate_fusion", "true");
                        network->Configure("seed", "0");
                        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                        const auto actual = onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots) : network->RepeatedExecute(circuit, shots);
                        if (reference.empty())
                            reference = actual;
                        Check(actual == reference, "Small trajectory budgets changed seeded resets/readout");
#ifdef _OPENMP
                        Check(omp_get_max_threads() == team, "Small trajectory changed the caller's OpenMP settings");
#endif
                    }
            }
}

void IndependentShotStreams()
{
    struct Seeds
    {
        std::mutex mutex;
        std::set<std::string> values;
        size_t calls = 0;
    };

    class ObserveSeed : public Circuits::XGate<>
    {
      public:
        explicit ObserveSeed(std::shared_ptr<Seeds> seeds) : XGate(0), seeds(std::move(seeds))
        {
        }

        void Execute(const std::shared_ptr<Simulators::ISimulator> &sim, Circuits::OperationState &state) const override
        {
            {
                std::lock_guard lock(seeds->mutex);
                seeds->values.insert(sim->GetConfiguration("seed"));
                ++seeds->calls;
            }
            XGate::Execute(sim, state);
        }

        std::shared_ptr<Circuits::IOperation<>> Clone() const override
        {
            return std::make_shared<ObserveSeed>(*this);
        }

      private:
        std::shared_ptr<Seeds> seeds;
    };
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
#endif
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kPathIntegral})
        for (bool prepared : {false, true})
            for (size_t shots : {size_t{1}, size_t{17}, size_t{257}})
            {
                Counts reference;
                std::set<std::string> referenceSeeds;
                for (bool onHost : {false, true})
                    for (int team : {1, 4})
                        for (size_t workers : {size_t{1}, size_t{3}, size_t{8}})
                        {
#ifdef _OPENMP
                            omp_set_num_threads(team);
#endif
                            const auto seeds = std::make_shared<Seeds>();
                            auto circuit = CF::CreateCircuit();
                            circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                            auto measured = std::static_pointer_cast<Circuits::MeasurementOperation<>>(CF::CreateMeasurement({{0, 0}}));
                            measured->SetReadout({{.2, .35}});
                            circuit->AddOperation(measured);
                            circuit->AddOperation(std::make_shared<ObserveSeed>(seeds));
                            circuit->AddOperation(CF::CreateCircuit({CF::CreateRandom({3, 4}, 0), CF::CreateRandom({5, 6})}));
                            circuit->AddOperation(CF::CreateReset({1}));
                            circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kHadamardGateType, 1), 3));
                            circuit->AddOperation(CF::CreateMeasurement({{0, 1}, {1, 2}}));
                            auto network = MakeNetwork(Backend::kQCSim, method, 7, 2);
                            network->SetOptimizeSimulator(prepared);
                            network->SetMaxSimulators(workers);
                            network->SetTrajectoryMinWorkerMicros(0);
                            Check(network->GetSimulator()->GetConfiguration("trajectory_policy") == "shot_v1",
                                  "Network default did not select per-shot streams");
                            if (workers != 3)
                                network->Configure("trajectory_policy", "shot_v1");
                            network->Configure("gate_fusion", "true");
                            network->Configure("seed", "1099511627776");
                            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                            const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots) : network->RepeatedExecute(circuit, shots);
                            Check(seeds->calls == shots && seeds->values.size() == shots, "Shots did not receive unique RNG streams");
                            if (reference.empty())
                            {
                                reference = counts;
                                referenceSeeds = seeds->values;
                            }
                            if (reference != counts || referenceSeeds != seeds->values)
                                std::cerr << "Shot stream mismatch: method=" << static_cast<int>(method) << " prepared=" << prepared << " shots=" << shots
                                          << " host=" << onHost << " team=" << team << " workers=" << workers
                                          << " equal seeds=" << (referenceSeeds == seeds->values) << '\n';
                            Check(reference == counts && referenceSeeds == seeds->values, "Per-shot quantum/classical/readout streams depend on scheduling");
                            bool rejected = false;
                            try
                            {
                                network->Configure("trajectory_policy", "typo");
                            }
                            catch (const std::invalid_argument &)
                            {
                                rejected = true;
                            }
                            Check(rejected, "Invalid trajectory policy was accepted");
#ifdef _OPENMP
                            Check(omp_get_max_threads() == team, "Per-shot execution leaked its thread budget");
#endif
                        }
            }
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
}

void SharedReset()
{
    // Reset descriptions are shared by jobs. Resetting several targets to |1>
    // must not mutate an instruction-owned X gate's target between threads.
    Circuits::Reset<> reset({0, 1}, 0, {true, true});
    std::array<std::thread, 2> threads;
    std::array<std::exception_ptr, 2> errors;
    std::atomic<size_t> ready{0}, wrong{0};
    for (size_t worker = 0; worker < threads.size(); ++worker)
        threads[worker] = std::thread([&, worker] {
            try
            {
                auto simulator = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQCSim, Method::kStatevector);
                simulator->AllocateQubits(2);
                simulator->Initialize();
                Circuits::OperationState state;
                ++ready;
                while (ready.load() != threads.size())
                    std::this_thread::yield();
                for (size_t shot = 0; shot < 1024; ++shot)
                {
                    reset.Execute(simulator, state);
                    if (simulator->Probability(3) < 1.0 - 1e-12)
                        ++wrong;
                }
            }
            catch (...)
            {
                errors[worker] = std::current_exception();
            }
        });
    for (auto &thread : threads)
        thread.join();
    for (const auto &error : errors)
        if (error)
            std::rethrow_exception(error);
    Check(wrong.load() == 0, "Shared reset changed another worker's X target");
}

void TrailingReset()
{
    // Deferred classical operations must not block an unrelated reset merely
    // because a compacted classical bit has the same numeric ID as its qubit.
    for (const size_t firstBit : {size_t{1}, size_t{3}})
        for (const bool exact : {false, true})
            for (const bool used : {false, true})
            {
                auto prefix = CF::CreateCircuit();
                if (used)
                    prefix->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
                prefix->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                prefix->AddOperation(CF::CreateMeasurement({{0, 0}}));
                prefix->AddOperation(CF::CreateCircuit({CF::CreateRandom({firstBit, firstBit + 1}, 0)}));
                prefix->AddOperation(CF::CreateReset({1}));
                Circuits::OperationState state(7);
                const auto mask = prefix->ExecuteNonMeasurements(nullptr, state, nullptr, exact);
                Check(mask.size() == 3 && mask[2] == (exact || !used), "Classical bit numbering changed reset prefix eligibility");
            }
    // Genuine deferred quantum operations, including those inside an opaque
    // composite, must keep a mixed-state reset in the per-shot suffix.
    for (const int nested : {0, 1, 2})
    {
        auto prefix = CF::CreateCircuit();
        auto measured = CF::CreateMeasurement({{1, 0}});
        prefix->AddOperation(nested == 1 ? CF::CreateCircuit({measured}) : measured);
        if (nested == 2)
            prefix->AddOperation(CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 1)}));
        prefix->AddOperation(CF::CreateReset({1}));
        Circuits::OperationState state(2);
        const auto mask = prefix->ExecuteNonMeasurements(nullptr, state, nullptr, true);
        Check(mask == std::vector<bool>(nested == 2 ? 3 : 2, false), "Mixed-state reset crossed a deferred quantum dependency");
    }

    // Sampled T1 noise injects resets after the last gate. Without a later
    // measurement they must still run instead of being deferred and dropped.
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    circuit->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
    Check(circuit->HasOpsAfterMeasurements(), "Trailing reset allowed terminal-measurement execution");

    auto leading = CF::CreateCircuit();
    leading->AddOperation(std::make_shared<Circuits::Reset<>>(Types::qubits_vector{0}));
    leading->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    leading->AddOperation(CF::CreateMeasurement({{0, 0}}));
    Check(!leading->HasOpsAfterMeasurements(), "Initial reset disabled terminal-measurement execution");

    for (const auto method : {Method::kStatevector, Method::kMatrixProductState})
    {
        auto network = MakeNetwork(Backend::kQCSim, method);
        network->SetOptimizeSimulator(true);
        const auto values = network->ExecuteOnHostExpectations(circuit, 0, {"Z", "IZ"});
        Check(values.size() == 2 && std::abs(values[0] - 1.0) < 1e-12 && std::abs(values[1] - 1.0) < 1e-12,
              "Trailing reset was skipped before an expectation value");
    }
}

void ExecuteTrajectoryPreservesState()
{
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kPathIntegral})
        for (const char *policy : {"shot_v1", "block_v1"})
            for (const char *fusion : {"false", "true"})
            {
                const std::string context = " (method=" + std::to_string(static_cast<int>(method)) + ", policy=" + policy + ", fusion=" + fusion + ")";
                auto network = MakeNetwork(Backend::kQCSim, method, 4);
                network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                network->Configure("trajectory_policy", policy);
                network->Configure("gate_fusion", fusion);
                network->Configure("seed", "99");
                const auto existing = network->GetSimulator();
                const auto resultBits = [&] {
                    auto bits = network->GetState().GetAllBits();
                    bits.resize(network->GetNumClassicalBits());
                    return bits;
                };

                // Mid-circuit measurement selects the trajectory dispatcher. The
                // first qubit is input supplied by the caller; the second is a
                // coherent spectator, so both populations and phase must survive.
                auto circuit = CF::CreateCircuit({CF::CreateMeasurement({{0, 0}}), CF::CreateGate(Gate::kXGateType, 0), CF::CreateMeasurement({{0, 1}})});
                for (int repeat = 0; repeat < 2; ++repeat)
                {
                    existing->Reset();
                    existing->ApplyX(0);
                    existing->ApplyH(1);
                    existing->ApplyS(1);
                    existing->SaveState();
                    network->Execute(circuit);
                    Check(network->GetSimulator() == existing, "Execute replaced the caller's trajectory simulator");
                    Check(resultBits() == std::vector<bool>({true, false, false, false}), "Execute discarded its prepared trajectory input" + context);
                    existing->ApplySDG(1);
                    existing->ApplyH(1);
                    Check(std::abs(existing->Probability(0) - 1.) < 1e-12, "Execute changed the unmeasured input's coherence");
                    existing->RestoreState();
                    existing->ApplySDG(1);
                    existing->ApplyH(1);
                    Check(std::abs(existing->Probability(1) - 1.) < 1e-12, "Single-shot Execute overwrote the caller's checkpoint");
                }

                // RepeatedExecute must still start from zero, even for one shot.
                Check(network->RepeatedExecute(circuit, 1) == Counts{{{false, true, false, false}, 1}},
                      "RepeatedExecute retained the input of a prior Execute");

                // Prefix removal also determines the classical RNG stream IDs.
                // Reuse and fresh single-shot dispatch must agree on all streams.
                auto random = CF::CreateCircuit({CF::CreateGate(Gate::kHadamardGateType, 0), CF::CreateMeasurement({{0, 0}}), CF::CreateRandom({2, 3}),
                                                 CF::CreateGate(Gate::kXGateType, 0), CF::CreateMeasurement({{0, 1}})});
                auto measured = std::static_pointer_cast<Circuits::MeasurementOperation<>>(random->GetOperations()[1]);
                measured->SetReadout({{0.2, 0.35}});
                for (const char *seed : {"0", "1", "99", "1099511627776"})
                {
                    network->Configure("seed", seed);
                    const auto reference = network->RepeatedExecute(random, 1);
                    network->Execute(random);
                    Check(reference == Counts{{resultBits(), 1}}, "Simulator reuse changed per-shot RNG streams" + context);
                }
            }
}

// A single-shot trajectory on an unseeded network seeds only its execution.
// Neither the simulator Execute keeps nor the configuration the next run
// captures from it may turn the network's later executions deterministic.
void UnseededExecuteStaysUnseeded()
{
    auto circuit = CF::CreateCircuit();
    for (Types::qubit_t qubit = 0; qubit < 4; ++qubit)
        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, qubit));
    circuit->AddOperation(CF::CreateGate(Gate::kTGateType, 0)); // Not Clifford: keep the selected method.
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}, {2, 2}, {3, 3}}));
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kPathIntegral})
        for (bool host : {false, true})
            for (bool prepared : {false, true})
            {
                const std::string context =
                    " (method=" + std::to_string(static_cast<int>(method)) + ", host=" + std::to_string(host) + ", prepared=" + std::to_string(prepared) + ")";
                auto network = MakeNetwork(Backend::kQCSim, method, 4, 4);
                network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                network->SetOptimizeSimulator(prepared);
                if (host)
                    network->ExecuteOnHost(circuit, 0);
                else
                    network->Execute(circuit);
                Check(network->GetSimulator()->GetConfiguration("seed").empty(), "Unseeded Execute configured its trajectory seed" + context);
                const auto first = network->RepeatedExecute(circuit, 256);
                const auto second = network->RepeatedExecute(circuit, 256);
                Check(network->GetSimulator()->GetConfiguration("seed").empty(), "Network captured a trajectory seed after Execute" + context);
                Check(first != second, "Unseeded executions after Execute replayed one stream" + context);
            }
}

void DistributedSimulatorReuse()
{
    std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
    backends.push_back(Backend::kQiskitAer);
#endif
    for (auto backend : backends)
        for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kDensityMatrix, Method::kMatrixProductOperator})
        {
            if (backend != Backend::kQCSim && method == Method::kMatrixProductOperator)
                continue;
            auto network = MakeNetwork(backend, method);
            network->SetMaxSimulators(1);
            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
            network->Configure("seed", "0");
            auto flip = CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 0)});
            auto bell = CF::CreateCircuit();
            bell->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            bell->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
            for (int repeat = 0; repeat < 8; ++repeat)
            {
                const auto values = network->ExecuteExpectations(flip, {"ZI", "IZ"});
                Check(values.size() == 2 && std::abs(values[0] + 1.0) < 1e-12 && std::abs(values[1] - 1.0) < 1e-12,
                      "Distributed expectation retained the previous quantum state");
                const auto entangled = network->ExecuteExpectations(bell, {"ZI", "IZ"});
                Check(entangled.size() == 2 && std::abs(entangled[0]) < 1e-12 && std::abs(entangled[1]) < 1e-12,
                      "Distributed expectation realizations did not start from zero");
            }
        }
}

void NoiseEstimateResetStreams()
{
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(0, 1));
    noise::NoiseModel model;
    model.set_t1(1, 0.5);
    const std::vector<std::string> paulis{"ZI", "IZ"};

    for (auto method : {Method::kStatevector, Method::kMatrixProductState})
        for (bool combined : {false, true})
            for (const std::string configured : {"", "23"})
                for (const auto seed : {std::optional<unsigned int>{}, std::optional<unsigned int>{0}, std::optional<unsigned int>{5}})
                {
                    auto network = MakeNetwork(Backend::kQCSim, method);
                    network->SetOptimizeSimulator(true);
                    if (!configured.empty())
                        network->Configure("seed", configured.c_str());
                    noise::NoiseAdd add;
                    if (seed)
                        add.seed(*seed);
                    auto run = [&]() {
                        return combined ? add.full_noise_estimate(circuit, network, 0, paulis, model, 2000)
                                        : add.noisy_estimate_montecarlo(circuit, network, 0, paulis, model, 2000);
                    };

                    const auto values = run();
                    Check(values.size() == 2 && std::abs(values[0]) < 0.1 && std::abs(values[1] - 0.5) < 0.1,
                          "C++ noise estimates reused injected errors or reset outcomes");
                    if (seed)
                    {
                        add.seed(*seed);
                        Check(values == run(), "Reseeding C++ noise did not reproduce reset trajectories");
                    }
                    Check(network->GetSimulator()->GetConfiguration("seed") == configured, "Noise estimation replaced the caller's simulator seed");
                }
}

struct PoolJob
{
    std::shared_ptr<std::atomic<size_t>> runs;

    void DoWork()
    {
        ++*runs;
    }

    size_t GetJobCount() const
    {
        return 1;
    }
};

void ThreadsPoolResize()
{
    // Shrinking used to destroy the surplus workers without waking them, so
    // their join never returned.
    Utils::ThreadsPool<PoolJob> pool;
    const auto runs = std::make_shared<std::atomic<size_t>>(0);
    size_t expected = 0;
    for (size_t threads : {4, 2, 6, 1, 3})
    {
        pool.Resize(threads);
        pool.SetFinishLimit(50);
        for (int i = 0; i < 50; ++i)
            pool.AddRunJob(std::make_shared<PoolJob>(PoolJob{runs}));
        pool.WaitForFinish();
        expected += 50;
        Check(runs->load() == expected, "Thread pool lost jobs after a resize");
    }
}

struct SlowReleaseJob
{
    std::shared_ptr<std::atomic<int>> live;

    explicit SlowReleaseJob(std::shared_ptr<std::atomic<int>> counter) : live(std::move(counter))
    {
        ++*live;
    }

    ~SlowReleaseJob()
    {
        std::this_thread::sleep_for(std::chrono::microseconds(300));
        --*live;
    }

    void DoWork()
    {
    }

    size_t GetJobCount() const
    {
        return 1;
    }
};

void ThreadsPoolReleasesFinishedJobs()
{
    // Execution jobs own the network that owns the pool. A job still alive
    // after WaitForFinish() could destroy the network on its own worker.
    Utils::ThreadsPool<SlowReleaseJob> pool;
    pool.Resize(4);
    const auto live = std::make_shared<std::atomic<int>>(0);
    for (int round = 0; round < 50; ++round)
    {
        pool.SetFinishLimit(8);
        for (int i = 0; i < 8; ++i)
            pool.AddRunJob(std::make_shared<SlowReleaseJob>(live));
        pool.WaitForFinish();
        Check(live->load() == 0, "Thread pool reported jobs finished before releasing them");
    }
}

struct ThrowingPoolJob
{
    std::shared_ptr<std::atomic<size_t>> runs;
    bool fails = false;

    void DoWork()
    {
        ++*runs;
        if (fails)
            throw std::runtime_error("pool job failed");
    }

    size_t GetJobCount() const
    {
        return 1;
    }
};

void ThreadsPoolRethrowsJobFailures()
{
    // An exception escaping a worker terminated the process. The waiter
    // rethrows the first one once every job is counted (those queued after
    // it are skipped), and the next batch runs as usual.
    Utils::ThreadsPool<ThrowingPoolJob> pool;
    pool.Resize(4);
    const auto runs = std::make_shared<std::atomic<size_t>>(0);
    for (int round = 0; round < 20; ++round)
    {
        pool.SetFinishLimit(64);
        for (int i = 0; i < 64; ++i)
            pool.AddRunJob(std::make_shared<ThrowingPoolJob>(ThrowingPoolJob{runs, i % 8 == 3}));
        std::string message;
        try
        {
            pool.WaitForFinish();
        }
        catch (const std::runtime_error &error)
        {
            message = error.what();
        }
        Check(message == "pool job failed", "A pool job's exception did not reach the waiter");
        runs->store(0);
        pool.SetFinishLimit(32);
        for (int i = 0; i < 32; ++i)
            pool.AddRunJob(std::make_shared<ThrowingPoolJob>(ThrowingPoolJob{runs, false}));
        pool.WaitForFinish();
        Check(runs->load() == 32, "A failed batch kept the pool from running the next one");
    }
}

void StabilizerShotWorkersReportFailures()
{
    // A stabilizer's shot blocks meet the T gate on pool workers. The
    // rejection reaches the caller as an exception, and the network then runs
    // a Clifford circuit as usual.
    auto network = MakeNetwork(Backend::kQCSim, Method::kStabilizer, 1, 1);
    network->SetMaxSimulators(4);
    network->SetTrajectoryMinWorkerMicros(0);
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    for (size_t shots : {10, 1027})
    {
        std::string message;
        try
        {
            network->RepeatedExecuteOnHost(circuit, 0, shots);
        }
        catch (const std::runtime_error &error)
        {
            message = error.what();
        }
        Check(message.find("non-clifford") != std::string::npos, "A stabilizer's shot workers did not report the T gate: " + message);
    }
    auto clifford = CF::CreateCircuit();
    clifford->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    clifford->AddOperation(CF::CreateMeasurement({{0, 0}}));
    const auto counts = network->RepeatedExecuteOnHost(clifford, 0, 1027);
    Check(counts.size() == 1 && counts.begin()->first[0] && counts.begin()->second == 1027, "A failed stabilizer run broke the next one");
}

#ifdef MAESTRO_POOL_FORK_HANDLERS
void ForkWhileWaitingForJobs()
{
    // A thread waiting for jobs when another forks leaves a waiter, and
    // possibly a held lock, in the child's copy of the completion state.
    auto pool = std::make_unique<Utils::ThreadsPool<PoolJob>>();
    pool->Resize(2);
    pool->SetFinishLimit(1);
    std::thread waiter([&pool] { pool->WaitForFinish(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout.flush();
    const pid_t child = fork();
    if (child == 0)
    {
        alarm(120); // a hang fails the check below instead of stalling the run
        const auto runs = std::make_shared<std::atomic<size_t>>(0);
        pool->Resize(2); // as executions do; a forked idle pool starts empty
        pool->SetFinishLimit(10);
        for (int i = 0; i < 10; ++i)
            pool->AddRunJob(std::make_shared<PoolJob>(PoolJob{runs}));
        pool->WaitForFinish();
        pool.reset();
        _exit(runs->load() == 10 ? 0 : 2);
    }
    Check(child > 0, "fork failed");
    int status = 0;
    waitpid(child, &status, 0);
    const auto runs = std::make_shared<std::atomic<size_t>>(0);
    pool->AddRunJob(std::make_shared<PoolJob>(PoolJob{runs}));
    waiter.join();
    Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "Forked child inherited the pool's completion waiters");
}
#endif

std::shared_ptr<Circuit> MidCircuitTrajectory(size_t qubits)
{
    auto circuit = CF::CreateCircuit();
    for (size_t q = 0; q < qubits; ++q)
    {
        circuit->AddOperation(CF::CreateGate(Gate::kRyGateType, q, 0, 0, .29 + .07 * q));
        if (q)
            circuit->AddOperation(std::make_shared<Circuits::CXGate<>>(q - 1, q));
    }
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    circuit->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, qubits - 1), 0));
    circuit->AddOperation(CF::CreateReset({0}));
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 1}, {qubits - 1, 2}}));
    return circuit;
}

void PersistentWorkers()
{
    // Workers stay alive between executions. Reusing them with fewer or more
    // workers must give the counts of a fresh network.
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(8);
#endif
    const auto circuit = MidCircuitTrajectory(6);
    for (auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductState})
    {
        auto reused = MakeNetwork(Backend::kQCSim, method, 3, 6);
        reused->SetTrajectoryMinWorkerMicros(0);
        reused->Configure("seed", "77");
        reused->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        for (size_t workers : {8, 3, 8, 2, 5})
        {
            auto fresh = MakeNetwork(Backend::kQCSim, method, 3, 6);
            fresh->SetTrajectoryMinWorkerMicros(0);
            fresh->Configure("seed", "77");
            fresh->SetMaxSimulators(workers);
            fresh->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
            reused->SetMaxSimulators(workers);
            Check(reused->RepeatedExecute(circuit, 300) == fresh->RepeatedExecute(circuit, 300), "Reused workers changed counts");
        }
    }
    // The work limit only removes workers; per-shot streams keep the counts.
    for (auto method : {Method::kStatevector, Method::kDensityMatrix})
    {
        Counts reference;
        for (double micros : {0., 100., 1e9})
        {
            auto network = MakeNetwork(Backend::kQCSim, method, 3, 6);
            network->SetMaxSimulators(8);
            network->SetTrajectoryMinWorkerMicros(micros);
            network->Configure("seed", "78");
            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
            const auto counts = network->RepeatedExecute(circuit, 300);
            if (reference.empty())
                reference = counts;
            Check(counts == reference, "Trajectory work limit changed counts");
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
}

#ifdef MAESTRO_POOL_FORK_HANDLERS
void ForkWithIdleWorkers()
{
    // A forked child has none of the parent's threads; its pools must start
    // empty instead of waiting for workers that do not exist.
#ifdef _OPENMP
    const int previous = omp_get_max_threads();
    omp_set_num_threads(8);
#endif
    const auto circuit = MidCircuitTrajectory(6);
    auto network = MakeNetwork(Backend::kQCSim, Method::kStatevector, 3, 6);
    network->SetTrajectoryMinWorkerMicros(0);
    network->SetMaxSimulators(4);
    network->Configure("seed", "79");
    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
    const auto before = network->RepeatedExecute(circuit, 300);
    std::cout.flush();
    const pid_t child = fork();
    if (child == 0)
    {
        alarm(120); // a hang fails the check below instead of stalling the run
        _exit(network->RepeatedExecute(circuit, 300) == before ? 0 : 2);
    }
    Check(child > 0, "fork failed");
    int status = 0;
    waitpid(child, &status, 0);
    Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "Forked child could not execute on the network's workers");
    Check(network->RepeatedExecute(circuit, 300) == before, "Parent lost its workers across fork");
#ifdef _OPENMP
    omp_set_num_threads(previous);
#endif
}
#endif

} // namespace

// Bell pair preparation that is not a Clifford circuit (T and its adjoint keep
// the state), so selection does not move it to a listed stabilizer.
std::shared_ptr<Circuit> NonCliffordBell()
{
    auto bell = CF::CreateCircuit();
    bell->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    bell->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 1));
    bell->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
    bell->AddOperation(CF::CreateGate(Gate::kTdgGateType, 0));
    return bell;
}

void QuietNetwork(Net &network)
{
    network.GetController()->SetOptimizeCircuit(false);
    network.SetInitialQubitsMapOptimization(false);
    network.SetMPSOptimizeSwaps(false);
    network.SetMaxSimulators(1);
}

void ConfiguredCandidatesDecide()
{
    using Pair = std::pair<Backend, Method>;
    const std::vector<std::string> xx{"XX"};
    {
        // Without a cost model a configured list decides: a bond-1 MPS
        // placeholder outside {statevector, stabilizer}, which cannot hold a
        // Bell pair, must not execute it, not even through seeded noise.
        auto network = std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{2});
        QuietNetwork(*network);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kStatevector);
        network->AddOptimizationSimulator(Backend::kQCSim, Method::kStabilizer);
        network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
        network->Configure("matrix_product_state_max_bond_dimension", "1");
        network->SetOptimizeSimulator(true);
        Check(network->GetOptimizationCandidates() == std::vector<Pair>{{Backend::kQCSim, Method::kStatevector}, {Backend::kQCSim, Method::kStabilizer}},
              "The configured candidates lost their order");
        noise::NoiseAdd add;
        add.seed(17);
        const auto noisy = add.noisy_estimate_montecarlo(NonCliffordBell(), network, 0, xx, noise::NoiseModel{}, 1);
        Check(noisy.size() == 1 && std::abs(noisy[0] - 1) < 1e-9, "Seeded noise pinned a placeholder outside the candidate list");
        Check(add.last_execution().valid && add.last_execution().method == Method::kStatevector, "The noise helper misreported the backend it ran");
        const auto values = network->ExecuteOnHostExpectations(NonCliffordBell(), 0, xx);
        Check(values.size() == 1 && std::abs(values[0] - 1) < 1e-9 && network->GetLastSimulationType() == Method::kStatevector,
              "A placeholder outside the candidate list executed");
    }
    {
        // The constructor's default list is no request: the simulator's backend runs.
        auto network = std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{2});
        QuietNetwork(*network);
        network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
        network->SetOptimizeSimulator(true);
        Check(network->GetOptimizationCandidates().empty(), "The default candidate list counted as configured");
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        circuit->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 16);
        Check(counts.size() == 1 && counts.begin()->second == 16 && network->GetLastSimulationType() == Method::kMatrixProductOperator,
              "The default candidate list replaced the network's backend");
    }
    {
        // The first listed candidate whose register holds the circuit as
        // execution maps it: a density matrix for one active qubit of a
        // 40-qubit host, the listed MPO for all 40.
        auto network = std::make_shared<Net>(Types::qubits_vector{40}, std::vector<size_t>{40});
        QuietNetwork(*network);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kDensityMatrix);
        network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
        network->CreateSimulator(Backend::kQCSim, Method::kDensityMatrix, Network::INetwork<>::kBackendPlaceholderQubits);
        network->SetOptimizeSimulator(true);
        for (size_t width : {size_t{1}, size_t{40}})
        {
            auto circuit = CF::CreateCircuit();
            std::vector<std::pair<Types::qubit_t, size_t>> measured;
            for (size_t q = 0; q < width; ++q)
            {
                circuit->AddOperation(CF::CreateGate(Gate::kXGateType, q));
                measured.emplace_back(q, q);
            }
            circuit->AddOperation(CF::CreateMeasurement(measured));
            const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 8);
            Check(counts.size() == 1 && counts.begin()->second == 8, "Mapped-width candidate selection lost shots");
            Check(network->GetLastSimulationType() == (width == 1 ? Method::kDensityMatrix : Method::kMatrixProductOperator),
                  "Candidates were not tried in order on the register execution maps");
        }
    }
}

class ExposedNetwork : public Net
{
  public:
    using Net::Net;

    bool Recreates() const
    {
        return recreateIfNeeded;
    }
};

void NetworkSimulatorLifecycle()
{
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    const std::vector<std::string> z{"Z"};
    {
        // A network without a simulator does not simulate (a derived one may
        // only translate), and executing it must not create one.
        auto network = std::make_shared<Net>(Types::qubits_vector{1}, std::vector<size_t>{1});
        network->Execute(circuit);
        network->ExecuteOnHost(circuit, 0);
        network->ExecuteExpectations(circuit, z);
        // A host expectation query needs a state, so it reports the missing one.
        bool threw = false;
        try
        {
            network->ExecuteOnHostExpectations(circuit, 0, z);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        Check(threw, "A host expectation query ran without a simulator");
        Check(!network->GetSimulator(), "Executing a network without a simulator created one");
        Check(network->RepeatedExecute(circuit, 4).empty(), "A network without a simulator simulated");
    }
    {
        // A host execution that throws must restore recreation.
        auto network = std::make_shared<ExposedNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
        network->SetOptimizeSimulator(false);
        network->CreateSimulator(Backend::kQCSim, Method::kStatevector, Network::INetwork<>::kBackendPlaceholderQubits);
        auto wide = CF::CreateCircuit();
        for (Types::qubit_t q = 0; q < 4; ++q)
            wide->AddOperation(CF::CreateGate(Gate::kXGateType, q));
        bool threw = false;
        try
        {
            network->ExecuteOnHost(wide, 0);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        Check(threw && network->Recreates(), "A failed host execution left recreation disabled");
    }
    {
        // A whole-network run that cannot allocate its register (a density
        // matrix of 33 qubits) throws. It must clear the observables it
        // installed, restore recreation and keep the backend template.
        auto network = std::make_shared<ExposedNetwork>(Types::qubits_vector{32}, std::vector<size_t>{32});
        network->SetOptimizeSimulator(false);
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        network->CreateSimulator(Backend::kQCSim, Method::kDensityMatrix, Network::INetwork<>::kBackendPlaceholderQubits);
        const std::vector<std::string> wideObservables{std::string(40, 'Z')};
        bool threw = false;
        try
        {
            network->ExecuteExpectations(circuit, wideObservables);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        Check(threw, "A whole-network run of a 33-qubit density matrix did not fail");
        Check(network->Recreates(), "A failed expectation run left recreation disabled");
        Check(bool(network->GetSimulator()), "A failed run lost the network's backend template");
        // Observables left installed would widen this mapping past the host.
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 8);
        Check(counts.size() == 1 && counts.begin()->second == 8 && counts.begin()->first[0], "A failed expectation run left its observables installed");
    }
}

void ClonesCarryTheBackend()
{
    const std::vector<std::string> xx{"XX"};
    {
        // A setting made only on the network's simulator reaches clones, so the
        // noise helpers, which run on clones, use it as ordinary execution does.
        auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState);
        network->GetSimulator()->Configure("matrix_product_state_max_bond_dimension", "1");
        Check(network->Clone()->GetSimulator()->GetConfiguration("matrix_product_state_max_bond_dimension") == "1",
              "A clone dropped a setting made on the network's simulator");
        noise::NoiseAdd add;
        add.seed(17);
        const auto noisy = add.noisy_estimate_montecarlo(NonCliffordBell(), network, 0, xx, noise::NoiseModel{}, 1);
        const auto direct = network->ExecuteOnHostExpectations(NonCliffordBell(), 0, xx);
        Check(noisy.size() == 1 && direct.size() == 1 && std::abs(direct[0]) < 1e-9 && std::abs(noisy[0]) < 1e-9,
              "Noisy estimation ran without a setting made on the network's simulator");
    }
    {
        // An execution clone never copies the source's register; Clone() keeps
        // the creation width.
        auto network = std::make_shared<Net>(Types::qubits_vector{10}, std::vector<size_t>{10});
        network->CreateSimulator(Backend::kQCSim, Method::kStatevector, 10);
        const auto pristine = network->Clone();
        const auto execution = network->CloneForExecution(Network::INetwork<>::kBackendPlaceholderQubits);
        Check(pristine->GetSimulatorQubits() == 10 && pristine->GetSimulator()->GetNumberOfQubits() == 10, "Clone() changed the creation width");
        Check(execution->GetSimulatorQubits() == Network::INetwork<>::kBackendPlaceholderQubits &&
                  execution->GetSimulator()->GetNumberOfQubits() == Network::INetwork<>::kBackendPlaceholderQubits &&
                  execution->GetSimulator()->GetSimulationType() == Method::kStatevector,
              "An execution clone allocated the source's register");
    }
}

// A cost model that estimates nothing it is offered: selection must still keep
// to a configured list.
class NoChoiceCostModel : public Estimators::SimulatorsEstimatorInterface<>
{
  public:
    bool IsInitialized() const override
    {
        return true;
    }

    std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(const std::vector<std::pair<Backend, Method>> &, std::shared_ptr<Circuit> &, size_t &, size_t,
                                                                size_t, size_t, Backend &, Method &, std::vector<bool> &, const long long int, const double,
                                                                const std::string &, const std::string &, size_t, const std::vector<std::string> *,
                                                                bool) const override
    {
        return nullptr;
    }
};

// A cost model that chooses the first candidate it is offered, and records
// every offer.
class FirstChoiceCostModel : public Estimators::SimulatorsEstimatorInterface<>
{
  public:
    explicit FirstChoiceCostModel(std::shared_ptr<std::vector<std::pair<Backend, Method>>> offers) : offers(std::move(offers))
    {
    }

    bool IsInitialized() const override
    {
        return true;
    }

    std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(const std::vector<std::pair<Backend, Method>> &candidates, std::shared_ptr<Circuit> &,
                                                                size_t &, size_t, size_t, size_t, Backend &type, Method &method, std::vector<bool> &,
                                                                const long long int, const double, const std::string &, const std::string &, size_t,
                                                                const std::vector<std::string> *, bool) const override
    {
        offers->insert(offers->end(), candidates.begin(), candidates.end());
        if (candidates.empty())
            return nullptr;
        type = candidates.front().first;
        method = candidates.front().second;
        return Simulators::SimulatorsFactory::CreateSimulator(type, method);
    }

  private:
    std::shared_ptr<std::vector<std::pair<Backend, Method>>> offers;
};

class CostModelNetwork : public Net
{
  public:
    using Net::Net;

    void InstallCostModel(std::unique_ptr<Estimators::SimulatorsEstimatorInterface<>> model = std::make_unique<NoChoiceCostModel>())
    {
        simulatorsEstimator = std::move(model);
    }
};

void CostModelKeepsToTheList()
{
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    circuit->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    for (const bool rankable : {false, true})
    {
        // Unrankable: the cost model estimates neither a tensor network nor a
        // stabilizer (the circuit is not Clifford). Rankable: it is offered a
        // statevector and a path integral but chooses neither. Either way a
        // listed candidate runs, not the MPS placeholder outside the list.
        auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
        QuietNetwork(*network);
        const auto first = rankable ? Method::kStatevector : Method::kTensorNetwork;
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, first);
        network->AddOptimizationSimulator(Backend::kQCSim, rankable ? Method::kPathIntegral : Method::kStabilizer);
        network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
        network->SetOptimizeSimulator(true);
        network->InstallCostModel();
        Check(network->HasCostModel(), "The test cost model is not active");
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 8);
        Check(counts.size() == 1 && counts.begin()->second == 8 && counts.begin()->first[0], "Cost-model selection changed the outcome");
        Check(network->GetLastSimulationType() == first, "A cost model that chose nothing let a backend outside the list run");
    }
    {
        // Seeded noise with a cost model keeps the fallback among mixed-state
        // candidates: a 32-qubit product state is no density matrix, but the
        // listed MPO holds it.
        auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{32}, std::vector<size_t>{32});
        QuietNetwork(*network);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kDensityMatrix);
        network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
        network->CreateSimulator(Backend::kQCSim, Method::kDensityMatrix, Network::INetwork<>::kBackendPlaceholderQubits);
        network->SetOptimizeSimulator(true);
        network->InstallCostModel();
        auto wide = CF::CreateCircuit();
        std::vector<std::pair<Types::qubit_t, size_t>> measured;
        for (Types::qubit_t q = 0; q < 32; ++q)
        {
            wide->AddOperation(CF::CreateGate(Gate::kXGateType, q));
            measured.emplace_back(q, q);
        }
        wide->AddOperation(CF::CreateMeasurement(measured));
        noise::NoiseAdd add;
        add.seed(11);
        const auto counts = add.noisy_execute(wide, network, 0, noise::NoiseModel{}, 4, 1);
        Check(counts.size() == 1 && counts.begin()->second == 4, "Seeded noise lost the candidate that holds the register");
        Check(add.last_execution().valid && add.last_execution().method == Method::kMatrixProductOperator,
              "Seeded noise pinned a candidate that cannot hold the register");
    }
    {
        // Seeded noise keeps the fallback with a pure-state candidate listed
        // too: the 32-qubit product state runs on the listed MPS.
        auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{32}, std::vector<size_t>{32});
        QuietNetwork(*network);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kDensityMatrix);
        network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductState);
        network->CreateSimulator(Backend::kQCSim, Method::kDensityMatrix, Network::INetwork<>::kBackendPlaceholderQubits);
        network->SetOptimizeSimulator(true);
        network->InstallCostModel();
        auto wide = CF::CreateCircuit();
        std::vector<std::pair<Types::qubit_t, size_t>> measured;
        for (Types::qubit_t q = 0; q < 32; ++q)
        {
            wide->AddOperation(CF::CreateGate(Gate::kXGateType, q));
            measured.emplace_back(q, q);
        }
        wide->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
        wide->AddOperation(CF::CreateMeasurement(measured));
        noise::NoiseAdd add;
        add.seed(11);
        const auto counts = add.noisy_execute(wide, network, 0, noise::NoiseModel{}, 4, 1);
        Check(counts.size() == 1 && counts.begin()->second == 4 && add.last_execution().method == Method::kMatrixProductState,
              "Seeded noise dropped the listed MPS that holds the register");
    }
    for (const bool costModel : {false, true})
    {
        // The one candidate a cost model ranks (QuEST) may not be loaded: the
        // other listed one runs, not the MPS placeholder.
        auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
        QuietNetwork(*network);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQuestSim, Method::kStatevector);
        network->AddOptimizationSimulator(Backend::kQCSim, Method::kExtendedStabilizer);
        network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
        network->SetOptimizeSimulator(true);
        if (costModel)
            network->InstallCostModel();
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 4);
        const auto ran = std::make_pair(network->GetLastSimulatorType(), network->GetLastSimulationType());
        Check(counts.size() == 1 && counts.begin()->second == 4, "The listed fallback lost shots");
        Check(ran == std::make_pair(Backend::kQuestSim, Method::kStatevector) || ran == std::make_pair(Backend::kQCSim, Method::kExtendedStabilizer),
              "An unloadable ranked candidate let a backend outside the list run");
    }
    for (const auto attachment : {Method::kMatrixProductState, Method::kStabilizer})
        for (const bool costModel : {false, true})
        {
            // A listed stabilizer cannot run a non-Clifford circuit: the next
            // listed candidate runs it, also when the network's simulator is a
            // stabilizer, which does not make the circuit Clifford.
            auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
            QuietNetwork(*network);
            network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kStabilizer);
            network->AddOptimizationSimulator(Backend::kQCSim, Method::kTensorNetwork);
            network->CreateSimulator(Backend::kQCSim, attachment, Network::INetwork<>::kBackendPlaceholderQubits);
            network->SetOptimizeSimulator(true);
            if (costModel)
                network->InstallCostModel();
            auto nonClifford = CF::CreateCircuit();
            nonClifford->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            nonClifford->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
            nonClifford->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            nonClifford->AddOperation(CF::CreateMeasurement({{0, 0}}));
            const auto counts = network->RepeatedExecuteOnHost(nonClifford, 0, 4);
            size_t total = 0;
            for (const auto &entry : counts)
                total += entry.second;
            Check(total == 4 && network->GetLastSimulationType() == Method::kTensorNetwork, "A listed stabilizer ran a non-Clifford circuit");
        }
    {
        // No statevector of 62 qubits can be allocated (Eigen refuses 2^62
        // amplitudes), so the candidate the cost model ranks fails before it
        // runs. The configured list then falls back to its other candidates:
        // past the one ranked statevector to the listed MPO, and past the
        // statevector the cost model chose to the listed MPS.
        auto wide = CF::CreateCircuit();
        std::vector<std::pair<Types::qubit_t, size_t>> measured;
        for (Types::qubit_t q = 0; q < 62; ++q)
        {
            wide->AddOperation(CF::CreateGate(Gate::kXGateType, q));
            measured.emplace_back(q, q);
        }
        wide->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
        wide->AddOperation(CF::CreateMeasurement(measured));
        if (!Simulators::SimulatorsFactory::CreateSimulator(Backend::kQuestSim, Method::kStatevector))
        {
            // A stabilizer cannot run the non-Clifford circuit and QuEST cannot
            // be created, in either order and with or without a cost model:
            // the stabilizer's failure is reported, and the MPS network
            // simulator outside the list never runs.
            auto nonClifford = CF::CreateCircuit();
            nonClifford->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            nonClifford->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
            nonClifford->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            nonClifford->AddOperation(CF::CreateMeasurement({{0, 0}}));
            for (const bool questFirst : {false, true})
                for (const bool costModel : {false, true})
                    for (const bool whole : {false, true})
                {
                    auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
                    QuietNetwork(*network);
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                    const auto quest = std::make_pair(Backend::kQuestSim, Method::kStatevector);
                    const auto stabilizer = std::make_pair(Backend::kQCSim, Method::kStabilizer);
                    const auto first = questFirst ? quest : stabilizer;
                    const auto second = questFirst ? stabilizer : quest;
                    network->RemoveAllOptimizationSimulatorsAndAdd(first.first, first.second);
                    network->AddOptimizationSimulator(second.first, second.second);
                    network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
                    network->SetOptimizeSimulator(true);
                    if (costModel)
                        network->InstallCostModel();
                    const auto attached = network->GetSimulator();
                    bool reported = false;
                    try
                    {
                        if (whole)
                            network->RepeatedExecute(nonClifford, 4);
                        else
                            network->RepeatedExecuteOnHost(nonClifford, 0, 4);
                    }
                    catch (const std::exception &)
                    {
                        reported = true;
                    }
                    Check(reported, "A stabilizer list that could not run fell back to a simulator outside it");
                    // The rejected run leaves the network's simulator as it was.
                    const auto kept = network->GetSimulator();
                    Check(kept && kept->GetSimulationType() == Method::kMatrixProductState &&
                              kept->GetNumberOfQubits() == Network::INetwork<>::kBackendPlaceholderQubits,
                          "A rejected run changed the network's simulator");
                    // A whole-network run without a cost model does not clear
                    // the simulator before selection, so the rejection leaves
                    // the same one, not a re-created copy.
                    if (whole && !costModel)
                        Check(kept == attached, "A run that did not touch the network's simulator re-created it");
                    network->SetOptimizeSimulator(false);
                    size_t total = 0;
                    for (const auto &entry : network->RepeatedExecuteOnHost(nonClifford, 0, 4))
                        total += entry.second;
                    Check(total == 4, "The network's own backend could not run after a rejected run");
                }
            {
                // When several candidates fail, the first one's failure is
                // reported: the density matrix's, not that of the stabilizer
                // tried last for the non-Clifford circuit.
                auto network = std::make_shared<Net>(Types::qubits_vector{32}, std::vector<size_t>{32});
                QuietNetwork(*network);
                network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kStabilizer);
                network->AddOptimizationSimulator(Backend::kQCSim, Method::kDensityMatrix);
                network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
                network->SetOptimizeSimulator(true);
                auto dense = CF::CreateCircuit();
                std::vector<std::pair<Types::qubit_t, size_t>> all;
                for (Types::qubit_t q = 0; q < 32; ++q)
                {
                    dense->AddOperation(CF::CreateGate(Gate::kXGateType, q));
                    all.emplace_back(q, q);
                }
                dense->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
                dense->AddOperation(CF::CreateMeasurement(all));
                std::string message;
                try
                {
                    network->RepeatedExecuteOnHost(dense, 0, 4);
                }
                catch (const std::exception &error)
                {
                    message = error.what();
                }
                Check(message.find("Density-matrix") != std::string::npos, "A later candidate's failure hid the first one's");
            }
            for (const bool questFirst : {false, true})
            {
                // QuEST is not loaded. A density matrix cannot hold the
                // 32-qubit register and QuEST cannot be created, in either
                // order: the density matrix's failure is reported, and the MPS
                // network simulator outside the list never runs.
                auto network = std::make_shared<Net>(Types::qubits_vector{32}, std::vector<size_t>{32});
                QuietNetwork(*network);
                const auto first = questFirst ? std::make_pair(Backend::kQuestSim, Method::kStatevector) : std::make_pair(Backend::kQCSim, Method::kDensityMatrix);
                const auto second = questFirst ? std::make_pair(Backend::kQCSim, Method::kDensityMatrix) : std::make_pair(Backend::kQuestSim, Method::kStatevector);
                network->RemoveAllOptimizationSimulatorsAndAdd(first.first, first.second);
                network->AddOptimizationSimulator(second.first, second.second);
                network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
                network->SetOptimizeSimulator(true);
                auto dense = CF::CreateCircuit();
                std::vector<std::pair<Types::qubit_t, size_t>> all;
                for (Types::qubit_t q = 0; q < 32; ++q)
                {
                    dense->AddOperation(CF::CreateGate(Gate::kXGateType, q));
                    all.emplace_back(q, q);
                }
                dense->AddOperation(CF::CreateGate(Gate::kTGateType, 0));
                dense->AddOperation(CF::CreateMeasurement(all));
                bool reported = false;
                try
                {
                    network->RepeatedExecuteOnHost(dense, 0, 4);
                }
                catch (const std::exception &)
                {
                    reported = true;
                }
                Check(reported, "A list whose candidates could not run fell back to a simulator outside it");
            }
        }
        for (const bool chosen : {false, true})
        {
            auto network = std::make_shared<CostModelNetwork>(Types::qubits_vector{62}, std::vector<size_t>{62});
            QuietNetwork(*network);
            network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kStatevector);
            if (chosen)
            {
                // A bond cap lets the cost model rank the MPS too.
                network->Configure("matrix_product_state_max_bond_dimension", "1");
                network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductState);
            }
            network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
            network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductState, Network::INetwork<>::kBackendPlaceholderQubits);
            network->SetOptimizeSimulator(true);
            auto offers = std::make_shared<std::vector<std::pair<Backend, Method>>>();
            if (chosen)
                network->InstallCostModel(std::make_unique<FirstChoiceCostModel>(offers));
            else
                network->InstallCostModel();
            const auto counts = network->RepeatedExecuteOnHost(wide, 0, 4);
            Check(counts.size() == 1 && counts.begin()->second == 4 &&
                      std::all_of(counts.begin()->first.begin(), counts.begin()->first.end(), [](bool bit) { return bit; }),
                  "The fallback after a failed ranked candidate changed the outcome");
            Check(network->GetLastSimulationType() == (chosen ? Method::kMatrixProductState : Method::kMatrixProductOperator),
                  "A ranked candidate that could not be allocated skipped the configured list");
            if (chosen)
                Check(!offers->empty() && offers->front() == std::make_pair(Backend::kQCSim, Method::kStatevector),
                      "The cost model was not offered the statevector");
        }
    }
}

// A zero-shot repeated execution runs nothing: the network's simulator keeps
// its register and the state prepared on it.
void ZeroShotsLeaveTheSimulator()
{
    auto x = CF::CreateCircuit();
    x->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    x->AddOperation(CF::CreateMeasurement({{0, 0}}));
    for (const bool automatic : {false, true})
    {
        auto network = std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{2});
        QuietNetwork(*network);
        network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
        network->CreateSimulator(Backend::kQCSim, Method::kStatevector);
        network->SetOptimizeSimulator(automatic);
        const auto simulator = network->GetSimulator();
        simulator->ApplyX(0);
        Check(network->RepeatedExecute(x, 0).empty() && network->RepeatedExecuteOnHost(x, 0, 0).empty(), "A zero-shot execution returned results");
        Check(network->GetSimulator() == simulator && std::abs(simulator->Probability(1) - 1.0) < 1e-12,
              "A zero-shot execution reset the network's simulator");
    }
}

void NoiseRecordsAndOptionRemoval()
{
    auto x = CF::CreateCircuit();
    x->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    x->AddOperation(CF::CreateMeasurement({{0, 0}}));
    {
        // Only runs that executed a network leave an execution record.
        auto network = MakeNetwork(Backend::kQCSim, Method::kStatevector);
        noise::NoiseAdd add;
        add.seed(5);
        add.noisy_execute(x, network, 0, noise::NoiseModel{}, 4, 1);
        Check(add.last_execution().valid, "A noisy run left no execution record");
        add.noisy_execute(x, network, 0, noise::NoiseModel{}, 0, 1);
        Check(!add.last_execution().valid, "A zero-shot call kept the previous execution record");
        add.noisy_execute(x, network, 0, noise::NoiseModel{}, 4, 1);
        const std::vector<std::string> identity{"II"};
        add.noisy_estimate_montecarlo(x, network, 0, identity, noise::NoiseModel{}, 1);
        Check(!add.last_execution().valid, "An estimate that planning resolved without a network reported a backend");
        // A network without a simulator does not simulate, and records no run.
        auto translating = std::make_shared<Net>(Types::qubits_vector{1}, std::vector<size_t>{1});
        translating->SetMaxSimulators(1);
        Check(add.noisy_execute(x, translating, 0, noise::NoiseModel{}, 4, 1).empty() && !translating->GetSimulator() && !add.last_execution().valid,
              "A network that did not simulate left an execution record");
    }
    {
        // Removing an option re-creates the simulator with the settings made
        // directly on it, not only those of the network.
        auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState);
        network->GetSimulator()->Configure("matrix_product_state_max_bond_dimension", "1");
        network->Configure("seed", "3");
        network->RemoveConfiguration("seed");
        Check(network->GetSimulator()->GetConfiguration("matrix_product_state_max_bond_dimension") == "1",
              "Removing an option dropped a setting made on the network's simulator");
    }
}

int main()
try
{
    NestedCircuitJobs();
    NestedClassicalDataflow();
    NestedShotBoundaries();
    ConditionalBitHelpers();
    HostSimulatorReuse();
    ExecuteTrajectoryPreservesState();
    UnseededExecuteStaysUnseeded();
    DistributedSimulatorReuse();
    RandomJobs();
    RandomSeedPolicy();
    MeasurementJobs();
    NoiseClonePolicy();
    ConditionalOutputMapping();
    QCSimWorkerRandomStreams();
    AdaptiveWorkerCounts();
    LargeTrajectoryWorkers();
    LargeTrajectoryThreadBudget();
    SamplingJobThreading();
#if defined(_OPENMP) && !defined(NO_QISKIT_AER)
    WorkerThreadsFollowFlag();
#endif
    TerminalDensityDispatch();
    TerminalMpoDispatch();
    NestedTerminalMeasurementsShareThePrefix();
    KeptMixedCandidateIsPrepared();
    ClearedMpoStillSamplesInParallel();
    MultinomialTerminalDispatch();
    ClassicalBitIdsDoNotDeferWires();
    MixedStateCandidates();
    ConfiguredCandidatesDecide();
    NetworkSimulatorLifecycle();
    ClonesCarryTheBackend();
    CostModelKeepsToTheList();
    ZeroShotsLeaveTheSimulator();
    NoiseRecordsAndOptionRemoval();
    SmallTrajectoryBudgets();
    IndependentShotStreams();
    QCSimSamplingRandomStreams(Method::kStabilizer);
    QCSimSamplingRandomStreams(Method::kExtendedStabilizer);
    SharedReset();
    TrailingReset();
    NoiseEstimateResetStreams();
    ThreadsPoolResize();
    ThreadsPoolReleasesFinishedJobs();
    ThreadsPoolRethrowsJobFailures();
    StabilizerShotWorkersReportFailures();
#ifdef MAESTRO_POOL_FORK_HANDLERS
    ForkWhileWaitingForJobs();
#endif
    PersistentWorkers();
#ifdef MAESTRO_POOL_FORK_HANDLERS
    ForkWithIdleWorkers();
#endif
    std::cout << checks << " network job checks passed\n";
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << "Network job regression: " << error.what() << '\n';
    return 1;
}
