// Regression tests for worker ownership, terminal measurements and noise
// clones.
#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Noise/NoiseAdd.h"

#include <array>
#include <atomic>
#include <iostream>
#include <optional>
#include <thread>

namespace {
using Backend = Simulators::SimulatorType;
using Method = Simulators::SimulationType;
using Circuit = Circuits::Circuit<>;
using CF = Circuits::CircuitFactory<>;
using Gate = Circuits::QuantumGateType;
using Net = Network::SimpleDisconnectedNetwork<>;
using Counts = Circuit::ExecuteResults;
size_t checks = 0;

void Check(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

std::shared_ptr<Net> MakeNetwork(Backend backend, Method method,
                                 size_t bits = 2) {
  auto network =
      std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{bits});
  network->SetOptimizeSimulator(false);
  network->GetController()->SetOptimizeCircuit(false);
  network->SetInitialQubitsMapOptimization(false);
  network->SetMPSOptimizeSwaps(false);
  network->SetMaxSimulators(2);
  network->RemoveAllOptimizationSimulatorsAndAdd(backend, method);
  network->CreateSimulator(backend, method);
  return network;
}

class CountedX : public Circuits::XGate<> {
 public:
  explicit CountedX(std::shared_ptr<std::atomic<size_t>> calls)
      : XGate(0), calls(std::move(calls)) {}
  void Execute(const std::shared_ptr<Simulators::ISimulator>& sim,
               Circuits::OperationState& state) const override {
    ++*calls;
    XGate::Execute(sim, state);
  }
  std::shared_ptr<Circuits::IOperation<>> Clone() const override {
    return std::make_shared<CountedX>(*this);
  }

 private:
  std::shared_ptr<std::atomic<size_t>> calls;
};

class SingleHostRemapper : public Distribution::IRemapper<> {
 public:
  std::shared_ptr<Circuit> Remap(
      const std::shared_ptr<Network::INetwork<>>&,
      const std::shared_ptr<Circuit>& circuit) override {
    return circuit;
  }
  unsigned int GetNumberOfOperationsForDistribution() const override {
    return 0;
  }
  unsigned int GetNumberOfDistributions() const override { return 0; }
  Distribution::RemapperType GetType() const override {
    return Distribution::RemapperType::kLayersRemapper;
  }
};

std::shared_ptr<Circuit> RandomCircuit(bool conditional,
                                       std::optional<size_t> seed = 123) {
  auto circuit = CF::CreateCircuit();
  std::vector<size_t> bits(64);
  std::iota(bits.begin(), bits.end(), 1);
  auto random = std::static_pointer_cast<Circuits::Random<>>(
      seed ? CF::CreateRandom(bits, *seed) : CF::CreateRandom(bits));
  if (conditional)
    circuit->AddOperation(CF::CreateConditionalRandomGen(
        random, std::make_shared<Circuits::EqualCondition>(
                    std::vector<size_t>{0}, std::vector<bool>{false})));
  else
    circuit->AddOperation(random);
  circuit->AddOperation(
      CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 0), 1));
  circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
  return circuit;
}

std::array<Counts, 2> RunRandomJobs(const std::shared_ptr<Circuit>& circuit,
                                    bool locked, bool reuse, bool seeded,
                                    uint64_t executionSeed = 42) {
  std::array<Counts, 2> results;
  std::array<std::exception_ptr, 2> errors;
  std::array<std::thread, 2> threads;
  std::mutex mutex;
  auto network = MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 65);
  std::atomic<size_t> ready{0};
  for (size_t worker = 0; worker < threads.size(); ++worker)
    threads[worker] = std::thread([&, worker] {
      try {
        Network::ExecuteJob<> job(circuit, results[worker], 128, 2, 65, 65,
                                  Backend::kQCSim, Method::kMatrixProductState,
                                  mutex);
        job.network = network;
        job.optimiseMultipleShotsExecution = reuse;
        job.randomStream = worker;
        if (seeded)
          job.config.SetConfiguration(
              "seed", std::to_string(Simulators::IState::DeriveSeed(
                          executionSeed, worker)));
        ++ready;
        while (ready.load() != threads.size()) std::this_thread::yield();
        if (locked)
          job.DoWork();
        else
          job.DoWorkNoLock();
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  for (auto& thread : threads) thread.join();
  for (const auto& error : errors)
    if (error) std::rethrow_exception(error);
  return results;
}

void RandomJobs() {
  for (bool conditional : {false, true})
    for (bool locked : {false, true})
      for (bool reuse : {false, true})
        for (bool seeded : {false, true}) {
          const auto circuit = RandomCircuit(conditional);
          const auto counts = RunRandomJobs(circuit, locked, reuse, seeded);
          Check(counts == RunRandomJobs(circuit, locked, reuse, seeded),
                "Classical random results depend on worker scheduling/history");
          for (const auto& worker : counts) {
            Check(worker.size() == 128,
                  "Classical RNG was rewound between shots");
            for (const auto& [bits, count] : worker)
              Check(bits[0] == bits[1] && count == 1,
                    "Random-controlled gate or per-shot reset failed");
          }
          // The chance that independent 64-bit streams overlap here is tiny;
          // identical seeds/clones instead produce the same 128 outcomes.
          for (const auto& [bits, count] : counts[0])
            Check(counts[1].count(bits) == 0,
                  "Workers share a classical RNG stream");
        }

  // Exercise both network dispatchers, including the uneven final job and
  // cloned planner simulators. No-seed jobs also need different streams.
  for (bool onHost : {false, true})
    for (bool optimize : {false, true})
      for (bool seeded : {false, true}) {
        auto network =
            MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 65);
        network->SetOptimizeSimulator(optimize);
        if (seeded) network->Configure("seed", "42");
        if (!onHost)
          network->GetController()->SetRemapper(
              std::make_shared<SingleHostRemapper>());
        const auto circuit = RandomCircuit(true);
        const auto run = [&] {
          return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 257)
                        : network->RepeatedExecute(circuit, 257);
        };
        const auto counts = run();
        if (counts.size() != 257)
          std::cerr << "Random dispatch: onHost=" << onHost
                    << " optimize=" << optimize << " seeded=" << seeded
                    << " outcomes=" << counts.size() << " width="
                    << (counts.empty() ? 0 : counts.begin()->first.size())
                    << '\n';
        Check(counts.size() == 257,
              "Network dispatcher reused a classical stream for multiple jobs");
        Check(counts == run(),
              "Network classical randomness is not reproducible");
      }

  // Check nested generators too, and that job seeding never mutates the input.
  auto nested = CF::CreateCircuit();
  nested->AddOperation(RandomCircuit(true));
  auto first = nested->CloneForExecution(42);
  auto second = nested->CloneForExecution(43);
  auto repeat = nested->CloneForExecution(42);
  auto generator = [](const std::shared_ptr<Circuit>& outer) {
    auto inner = std::static_pointer_cast<Circuit>(outer->GetOperation(0));
    auto conditional =
        std::static_pointer_cast<Circuits::ConditionalRandomGen<>>(
            inner->GetOperation(0));
    return conditional->GetOperation();
  };
  Check(generator(first) != generator(second) &&
            generator(first) != generator(nested),
        "Nested conditional generator is still shared");
  Circuits::OperationState a(65), b(65), c(65), original(65), reference(65);
  generator(first)->Execute(nullptr, a);
  generator(second)->Execute(nullptr, b);
  generator(repeat)->Execute(nullptr, c);
  generator(nested)->Execute(nullptr, original);
  generator(nested)->Clone()->Execute(nullptr, reference);
  Check(a.GetAllBits() == c.GetAllBits() && a.GetAllBits() != b.GetAllBits(),
        "Nested generator streams are not reproducible and distinct");
  Check(original.GetAllBits() == reference.GetAllBits(),
        "Executing job copies advanced the caller's generator");
}

void RandomSeedPolicy() {
  for (bool conditional : {false, true})
    for (bool locked : {false, true})
      for (bool reuse : {false, true}) {
        const auto unseeded = RandomCircuit(conditional, std::nullopt);
        const auto fresh = RunRandomJobs(unseeded, locked, reuse, false);
        Check(fresh != RunRandomJobs(unseeded, locked, reuse, false),
              "Unseeded jobs repeated the same classical random sequence");
        Check(fresh[0] != fresh[1], "Unseeded workers reused a random stream");
        const auto seeded = RunRandomJobs(unseeded, locked, reuse, true, 0);
        Check(seeded == RunRandomJobs(unseeded, locked, reuse, true, 0) &&
                  seeded ==
                      RunRandomJobs(RandomCircuit(conditional, std::nullopt),
                                    locked, reuse, true, 0),
              "Execution seed zero did not control default-seeded generators");
        Check(seeded != RunRandomJobs(unseeded, locked, reuse, true, 1),
              "Classical random operations ignored the execution seed");
        const auto explicitZero = RandomCircuit(conditional, 0);
        Check(RunRandomJobs(explicitZero, locked, reuse, false) ==
                  RunRandomJobs(RandomCircuit(conditional, 0), locked, reuse,
                                false),
              "Explicit instruction seed zero was treated as omitted");
      }

  // Both dispatchers remap/clone circuits before creating jobs. Those copies
  // must preserve whether each instruction's seed was explicitly supplied.
  for (bool onHost : {false, true})
    for (bool optimize : {false, true})
      for (size_t workers : {1, 2}) {
        auto network =
            MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 65);
        network->SetOptimizeSimulator(optimize);
        network->SetMaxSimulators(workers);
        if (!onHost)
          network->GetController()->SetRemapper(
              std::make_shared<SingleHostRemapper>());
        const auto run = [&](const std::shared_ptr<Circuit>& circuit) {
          return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 257)
                        : network->RepeatedExecute(circuit, 257);
        };
        const auto circuit = RandomCircuit(true, std::nullopt);
        const auto unseeded = run(circuit);
        Check(unseeded.size() == 257 && unseeded != run(circuit),
              "Network execution reused an unseeded classical stream");
        Check(run(RandomCircuit(true, 0)) == run(RandomCircuit(true, 0)),
              "Network remapping lost an explicit instruction seed of zero");
        network->Configure("seed", "0");
        const auto seeded = run(circuit);
        Check(seeded == run(circuit) &&
                  seeded == run(RandomCircuit(true, std::nullopt)),
              "Network seed zero did not control unseeded instructions");
      }

  std::vector<size_t> bits(64);
  std::iota(bits.begin(), bits.end(), 0);
  const auto sample = [](const std::shared_ptr<Circuits::IOperation<>>& op) {
    Circuits::OperationState state(64);
    op->Execute(nullptr, state);
    return state.GetAllBits();
  };
  Check(sample(CF::CreateRandom(bits)) != sample(CF::CreateRandom(bits)),
        "Direct unseeded generators still default to the same seed");
  Check(sample(CF::CreateRandom(bits, 0)) == sample(CF::CreateRandom(bits, 0)),
        "Direct generators ignored an explicit seed of zero");
  auto manual = std::make_shared<Circuits::Random<>>(bits);
  manual->Seed(0);
  Check(manual->HasExplicitSeed() &&
            sample(manual) == sample(CF::CreateRandom(bits, 0)),
        "Calling Seed(0) did not mark an explicitly seeded generator");

  // Entropy for unseeded instructions must not affect explicitly seeded ones,
  // even when they coexist in a nested circuit.
  auto mixed = CF::CreateCircuit();
  mixed->AddOperation(CF::CreateRandom(bits, 0));
  mixed->AddOperation(CF::CreateRandom(bits));
  auto nested = CF::CreateCircuit({mixed});
  Check(nested->HasUnseededRandomOperations(),
        "Nested unseeded generators were not detected");
  const auto first = std::static_pointer_cast<Circuit>(
      nested->CloneForExecution(0, 123)->GetOperation(0));
  const auto second = std::static_pointer_cast<Circuit>(
      nested->CloneForExecution(0, 456)->GetOperation(0));
  Check(sample(first->GetOperation(0)) == sample(second->GetOperation(0)),
        "Fresh entropy affected an explicitly seeded instruction");
  Check(sample(first->GetOperation(1)) != sample(second->GetOperation(1)),
        "Fresh entropy did not reach a nested unseeded instruction");
}

void MeasurementJobs() {
  std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
  backends.push_back(Backend::kQiskitAer);
#endif
  for (auto backend : backends)
    for (auto method :
         {Method::kStatevector, Method::kMatrixProductState,
          Method::kDensityMatrix, Method::kMatrixProductOperator}) {
      if (backend != Backend::kQCSim &&
          method == Method::kMatrixProductOperator)
        continue;
      for (bool locked : {false, true})
        for (bool reuse : {false, true})
          for (bool prepared : {false, true})
            for (int noise : {0, 1, 2, 3}) {
              if (prepared && !reuse) continue;
              const auto circuit = CF::CreateCircuit();
              circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
              auto earlier = std::make_shared<Circuits::MeasurementOperation<>>(
                  std::vector<std::pair<Types::qubit_t, size_t>>{{1, 0}});
              auto last = std::make_shared<Circuits::MeasurementOperation<>>(
                  std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}});
              if (noise == 1 || noise == 2) earlier->SetReadout({{1, 0}});
              if (noise == 1) last->SetReadout({{0, 1}});
              if (noise == 3) last->SetReadout({{0, 0.5}});
              circuit->AddOperation(earlier);
              circuit->AddOperation(last);
              auto network = MakeNetwork(backend, method);
              Counts counts;
              std::mutex mutex;
              Network::ExecuteJob<> job(circuit, counts, 128, 2, 1, 1, backend,
                                        method, mutex);
              job.network = network;
              job.config.SetConfiguration("seed", "123");
              job.optimiseMultipleShotsExecution = reuse;
              if (prepared) {
                job.optSim = Simulators::SimulatorsFactory::CreateSimulator(
                    backend, method);
                job.optSim->SetSeed(123);
                job.optSim->AllocateQubits(2);
                job.optSim->Initialize();
                Circuits::OperationState state(1);
                job.executedGates =
                    circuit->ExecuteNonMeasurements(job.optSim, state);
              }
              if (locked)
                job.DoWork();
              else
                job.DoWorkNoLock();
              if (noise == 3) {
                Check(counts.size() == 2, "Readout flips must vary per shot");
                Check(counts.at({false}) + counts.at({true}) == 128,
                      "Readout sampling lost shots");
              } else {
                const bool expected = noise != 1;
                Check(counts.size() == 1 && counts.at({expected}) == 128,
                      "Terminal sampling changed the last classical "
                      "write/readout");
              }
            }
    }
}

void NoiseClonePolicy() {
  for (bool optimizeCircuit : {false, true})
    for (bool optimizeSimulator : {false, true}) {
      auto network = MakeNetwork(Backend::kQCSim, Method::kStatevector);
      network->GetController()->SetOptimizeCircuit(optimizeCircuit);
      network->GetController()->SetOptimizeRotationGates(!optimizeCircuit);
      network->SetOptimizeSimulator(optimizeSimulator);
      network->SetInitialQubitsMapOptimization(!optimizeSimulator);
      auto clone = network->Clone();
      Check(clone->GetController()->GetOptimizeCircuit() == optimizeCircuit,
            "Network clone lost circuit-optimization policy");
      Check(clone->GetOptimizeSimulator() == optimizeSimulator,
            "Network clone lost backend-selection policy");
      Check(clone->GetController()->GetOptimizeRotationGates() ==
                !optimizeCircuit,
            "Network clone lost rotation-optimization policy");
      Check(clone->GetInitialQubitsMapOptimization() == !optimizeSimulator,
            "Network clone lost initial-qubit-map optimization policy");
    }

  // The cancelling X pair must still execute when circuit optimization is off.
  // Also verify that it executes once per realization, not once per shot.
  for (auto method : {Method::kStatevector, Method::kMatrixProductState,
                      Method::kDensityMatrix, Method::kMatrixProductOperator})
    for (size_t workers : {size_t{1}, size_t{2}}) {
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
      Check(counts.size() == 1 && counts.begin()->second == 56 &&
                !counts.begin()->first[0],
            "Noise realization batching changed the output");
      const size_t jobs = method == Method::kStatevector ? 1 : workers;
      Check(calls->load() == 2 * 7 * jobs,
            "Noise clone optimized away gates or replayed the prefix per shot");
    }
}

void ConditionalOutputMapping() {
  for (size_t workers : {size_t{1}, size_t{2}})
    for (bool readout : {false, true}) {
      auto network =
          MakeNetwork(Backend::kQCSim, Method::kMatrixProductState, 20);
      network->SetMaxSimulators(workers);
      auto circuit = CF::CreateCircuit();
      circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
      auto measurement = std::make_shared<Circuits::MeasurementOperation<>>(
          std::vector<std::pair<Types::qubit_t, size_t>>{{0, 17}});
      if (readout) measurement->SetReadout({{0, 1}});
      circuit->AddOperation(
          std::make_shared<Circuits::ConditionalMeasurement<>>(
              measurement,
              std::make_shared<Circuits::EqualCondition>(
                  std::vector<size_t>{3}, std::vector<bool>{false})));
      const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 128);
      std::vector<bool> expected(20, false);
      expected[17] = !readout;
      Check(counts.size() == 1 && counts.at(expected) == 128,
            "Host mapping lost a conditional measurement destination/readout");
    }
}

void NestedCircuitJobs() {
  // Keep the flat circuit as an independent execution reference. Include gates
  // after nested measurements, and classical data crossing a circuit boundary.
  for (int scenario = 0; scenario < 4; ++scenario) {
    auto inner = CF::CreateCircuit();
    auto outer = CF::CreateCircuit();
    auto flat = CF::CreateCircuit();
    if (scenario == 2) {
      auto prepare = CF::CreateGate(Gate::kXGateType, 1);
      auto measure = CF::CreateMeasurement({{1, 1}});
      outer->AddOperations({prepare, measure});
      flat->AddOperations({prepare, measure});
      inner->AddOperation(CF::CreateSimpleConditionalGate(
          CF::CreateGate(Gate::kXGateType, 0), 1));
    } else if (scenario == 3) {
      inner->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
      inner->AddOperation(CF::CreateMeasurement({{0, 0}}));
      inner->AddOperation(CF::CreateSimpleConditionalGate(
          CF::CreateGate(Gate::kXGateType, 1), 0));
      inner->AddOperation(CF::CreateReset({0}));
    } else {
      inner->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    }
    if (scenario == 0 || scenario == 2)
      inner->AddOperation(CF::CreateMeasurement({{0, 0}}));
    outer->AddOperation(CF::CreateCircuit({inner}));
    flat->AddCircuit(inner);
    if (scenario == 1 || scenario == 3) {
      const auto measurement = CF::CreateMeasurement({{0, 0}, {1, 1}});
      outer->AddOperation(measurement);
      flat->AddOperation(measurement);
    }

    for (auto method : {Method::kStatevector, Method::kMatrixProductState,
                        Method::kDensityMatrix, Method::kMatrixProductOperator})
      for (bool locked : {false, true}) {
        auto network = MakeNetwork(Backend::kQCSim, method);
        network->SetInitialQubitsMapOptimization(true);
        network->SetMPSOptimizeSwaps(true);
        network->Configure("matrix_product_state_max_bond_dimension", "16");
        network->SetMPSOptimizationQubitsNumberThreshold(0);
        network->SetMPSOptimizationBondDimensionThreshold(0);
        network->SetLookaheadDepth(0);
        const auto run = [&](const std::shared_ptr<Circuit>& circuit,
                             bool reuse) {
          Counts counts;
          std::mutex mutex;
          Network::ExecuteJob<> job(circuit, counts, 128, 2, 2, 2,
                                    Backend::kQCSim, method, mutex);
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
        Check(run(outer, true) == expected,
              "Nested multishot execution differs from the flat circuit");
        Check(run(outer, false) == expected,
              "Nested execution cleared the enclosing classical state");
      }

    for (bool onHost : {false, true})
      for (size_t workers : {size_t{1}, size_t{2}})
        for (bool optimize : {false, true}) {
          auto network =
              MakeNetwork(Backend::kQCSim, Method::kMatrixProductState);
          network->SetMaxSimulators(workers);
          network->SetOptimizeSimulator(optimize);
          network->Configure("seed", "123");
          network->GetController()->SetRemapper(
              std::make_shared<SingleHostRemapper>());
          const auto run = [&](const std::shared_ptr<Circuit>& circuit) {
            return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 128)
                          : network->RepeatedExecute(circuit, 128);
          };
          Check(run(outer) == run(flat),
                "Network dispatcher skipped or misexecuted a nested circuit");
        }
  }

  // A nested random stream must advance per shot, with both boundaries
  // retaining the classical state. CloneForExecution must still isolate its
  // generators.
  auto nested = CF::CreateCircuit({CF::CreateCircuit({RandomCircuit(true)})});
  const auto counts = RunRandomJobs(nested, true, true, true);
  Check(counts == RunRandomJobs(nested, true, true, true),
        "Nested random execution is not reproducible");
  for (const auto& worker : counts) {
    Check(worker.size() == 128, "Nested random generator was not run per shot");
    for (const auto& [bits, count] : worker)
      Check(bits[0] == bits[1] && count == 1,
            "Nested random-controlled gate lost its classical input");
  }

  // A composite containing only classical operations has no qubit queue for
  // MoveMeasurementsAndResets. The composite backend must still complete it.
  std::vector<size_t> bits(64);
  std::iota(bits.begin(), bits.end(), 0);
  const auto classical =
      CF::CreateCircuit({CF::CreateCircuit({CF::CreateRandom(bits, 123)})});
  for (bool onHost : {false, true}) {
    auto network =
        MakeNetwork(Backend::kCompositeQCSim, Method::kStatevector, 64);
    network->SetMaxSimulators(1);
    network->GetController()->SetRemapper(
        std::make_shared<SingleHostRemapper>());
    const auto counts = onHost
                            ? network->RepeatedExecuteOnHost(classical, 0, 128)
                            : network->RepeatedExecute(classical, 128);
    Check(counts.size() == 128,
          "Classical-only composite was not executed on the composite backend");
  }
}

void NestedClassicalDataflow() {
  // A Boolean oracle keeps these checks independent of the flat executor.
  // Exercise both outcomes of each condition, readout flips, reset-to-one,
  // overwritten predicate bits, sparse destinations and empty composites.
  std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
  backends.push_back(Backend::kQiskitAer);
#endif
  for (unsigned scenario = 0; scenario < 16; ++scenario) {
    const bool initial0 = scenario & 1, initial1 = scenario & 2;
    const bool flip = scenario & 4, resetToOne = scenario & 8;
    auto flat = CF::CreateCircuit();
    if (initial0) flat->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    if (initial1) flat->AddOperation(CF::CreateGate(Gate::kXGateType, 1));
    auto first = std::static_pointer_cast<Circuits::MeasurementOperation<>>(
        CF::CreateMeasurement({{0, 17}}));
    if (flip) first->SetReadout({{1, 1}});
    flat->AddOperation(first);
    flat->AddOperation(CF::CreateSimpleConditionalGate(
        CF::CreateGate(Gate::kXGateType, 1), 17));
    flat->AddOperation(CF::CreateMeasurement({{1, 3}}));
    flat->AddOperation(CF::CreateConditionalMeasurement(
        std::static_pointer_cast<Circuits::MeasurementOperation<>>(
            CF::CreateMeasurement({{0, 0}})),
        std::make_shared<Circuits::EqualCondition>(std::vector<size_t>{3},
                                                   std::vector<bool>{true})));
    flat->AddOperation(std::make_shared<Circuits::Reset<>>(
        Types::qubits_vector{0}, 0, std::vector<bool>{resetToOne}));
    flat->AddOperation(CF::CreateMeasurement({{0, 3}}));
    flat->AddOperation(CF::CreateSimpleConditionalGate(
        CF::CreateGate(Gate::kXGateType, 1), 0));
    flat->AddOperation(CF::CreateMeasurement({{1, 1}}));

    std::vector<bool> expected(18, false);
    expected[17] = initial0 != flip;
    const bool measured1 = initial1 != expected[17];
    expected[0] = measured1 && initial0;
    expected[3] = resetToOne;
    expected[1] = measured1 != expected[0];
    const auto& ops = flat->GetOperations();
    for (size_t cut : {size_t{0}, ops.size() / 2, ops.size()}) {
      auto nested = CF::CreateCircuit(
          {CF::CreateCircuit(
               Circuit::OperationsVector(ops.begin(), ops.begin() + cut)),
           CF::CreateCircuit({CF::CreateCircuit(
               Circuit::OperationsVector(ops.begin() + cut, ops.end()))})});
      for (auto backend : backends)
        for (auto method :
             {Method::kStatevector, Method::kMatrixProductState,
              Method::kDensityMatrix, Method::kMatrixProductOperator}) {
          if (backend != Backend::kQCSim &&
              method == Method::kMatrixProductOperator)
            continue;
          auto network = MakeNetwork(backend, method, 18);
          for (bool locked : {false, true})
            for (bool reuse : {false, true}) {
              Counts counts;
              std::mutex mutex;
              Network::ExecuteJob<> job(nested, counts, 17, 2, 18, 18, backend,
                                        method, mutex);
              job.network = network;
              job.config.SetConfiguration("seed", "0");
              job.optimiseMultipleShotsExecution = reuse;
              if (locked)
                job.DoWork();
              else
                job.DoWorkNoLock();
              Check(
                  counts == Counts{{expected, 17}},
                  "Nested classical dataflow differs from the Boolean oracle");
            }
        }
    }
  }
}

void NestedShotBoundaries() {
  auto circuit = CF::CreateCircuit({CF::CreateCircuit(
      {CF::CreateGate(Gate::kXGateType, 0), CF::CreateMeasurement({{0, 0}})})});
  circuit->AddOperation(
      CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 1), 0));
  circuit->AddOperation(CF::CreateMeasurement({{1, 1}}));
  for (bool onHost : {false, true})
    for (bool optimize : {false, true})
      for (size_t workers : {size_t{1}, size_t{3}, size_t{8}}) {
        auto network =
            MakeNetwork(Backend::kQCSim, Method::kMatrixProductState);
        network->SetMaxSimulators(workers);
        network->SetOptimizeSimulator(optimize);
        network->GetController()->SetRemapper(
            std::make_shared<SingleHostRemapper>());
        for (size_t shots : {size_t{0}, size_t{1}, size_t{2}, size_t{5},
                             size_t{7}, size_t{257}}) {
          const auto counts =
              onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots)
                     : network->RepeatedExecute(circuit, shots);
          const Counts expected =
              shots ? Counts{{{true, true}, shots}} : Counts{};
          Check(counts == expected,
                "Nested worker dispatch lost or duplicated shots");
        }
      }
}

class MappingNetwork : public Net {
 public:
  using Net::MapCircuitOnHost;
  using Net::Net;
};

void ConditionalBitHelpers() {
  for (bool random : {false, true})
    for (bool nested : {false, true}) {
      auto circuit = CF::CreateCircuit();
      auto condition = std::make_shared<Circuits::EqualCondition>(
          std::vector<size_t>{3}, std::vector<bool>{false});
      std::shared_ptr<Circuits::IOperation<>> conditional;
      if (random)
        conditional = CF::CreateConditionalRandomGen(
            std::static_pointer_cast<Circuits::Random<>>(
                CF::CreateRandom({0, 17}, 123)),
            condition);
      else
        conditional = CF::CreateConditionalMeasurement(
            std::static_pointer_cast<Circuits::MeasurementOperation<>>(
                CF::CreateMeasurement({{0, 0}, {1, 17}})),
            condition);
      circuit->AddOperation(conditional);
      if (nested) circuit = CF::CreateCircuit({circuit});
      Check(conditional->AffectedBits() == std::vector<size_t>{3},
            "Conditional predicate bits changed meaning");
      Check(circuit->GetBits() == std::set<size_t>{0, 3, 17} &&
                circuit->AffectedBits() == std::vector<size_t>({0, 3, 17}) &&
                circuit->GetMinCbitIndex() == 0 &&
                circuit->GetMaxCbitIndex() == 17,
            "Circuit bit helpers omitted conditional destinations");
      Circuit::BitMapping qubits, reverseBits;
      size_t nq = 0, nc = 0;
      const auto mapped =
          circuit->RemapToContinuous(qubits, reverseBits, nq, nc);
      Check(nc == 3 && mapped->GetBits() == std::set<size_t>{0, 1, 2},
            "Mapping omitted a nested conditional destination");
      auto network = std::make_shared<MappingNetwork>(Types::qubits_vector{2},
                                                      std::vector<size_t>{20});
      network->MapCircuitOnHost(circuit, 0, nq, nc, false);
      Check(nc == 18, "Shared host mapping omitted conditional destinations");
    }
}

class ObservedX : public Circuits::XGate<> {
 public:
  explicit ObservedX(std::shared_ptr<Simulators::ISimulator>& observed)
      : XGate(0), observed(observed) {}
  void Execute(const std::shared_ptr<Simulators::ISimulator>& sim,
               Circuits::OperationState& state) const override {
    observed = sim;
    XGate::Execute(sim, state);
  }
  std::shared_ptr<Circuits::IOperation<>> Clone() const override {
    return std::make_shared<ObservedX>(*this);
  }

 private:
  std::shared_ptr<Simulators::ISimulator>& observed;
};

void HostSimulatorReuse() {
  std::vector<Backend> backends{Backend::kQCSim};
#ifndef NO_QISKIT_AER
  backends.push_back(Backend::kQiskitAer);
#endif
  for (auto backend : backends)
    for (auto method :
         {Method::kStatevector, Method::kMatrixProductState,
          Method::kDensityMatrix, Method::kMatrixProductOperator}) {
      if (backend != Backend::kQCSim &&
          method == Method::kMatrixProductOperator)
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
      for (size_t shots : {size_t{8}, size_t{8}, size_t{1}}) {
        const auto existing = network->GetSimulator();
        existing->ApplyX(0);
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
        Check(observed == existing,
              "Single-worker host execution replaced its simulator");
        Check(counts == Counts{{{true, false}, shots}},
              "Reused host simulator retained a previous quantum state");
      }
      // ExecuteOnHost retains a simulator with the caller's register width.
      network->CreateSimulator(backend, method, 2);
      // Exercise consecutive same-sized calls, then a change of register width.
      for (int repeat = 0; repeat < 2; ++repeat) {
        const auto existing = network->GetSimulator();
        network->ExecuteOnHost(circuit, 0);
        Check(observed == existing && network->GetState().GetAllBits() ==
                                          std::vector<bool>({true, false}),
              "ExecuteOnHost failed to reset/reuse its simulator");
      }
      auto smaller = CF::CreateCircuit();
      smaller->AddOperation(std::make_shared<ObservedX>(observed));
      smaller->AddOperation(CF::CreateMeasurement({{0, 0}}));
      const auto existing = network->GetSimulator();
      const auto counts = network->RepeatedExecuteOnHost(smaller, 0, 8);
      Check(observed == existing && counts == Counts{{{true, false}, 8}},
            "Host simulator reuse failed after changing the register width");

      auto random = CF::CreateCircuit();
      random->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
      random->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
      network->CreateSimulator(backend, method, 2);
      const auto seeded = network->RepeatedExecuteOnHost(random, 0, 128);
      Check(seeded == network->RepeatedExecuteOnHost(random, 0, 128),
            "Simulator reuse changed the configured sampling seed");

      // Readout has its own RNG. Retaining a measured state must preserve
      // neither that RNG's position nor quantum/classical state on a new call.
      const auto makeSeeded = [&] {
        auto result = MakeNetwork(backend, method);
        result->SetMaxSimulators(1);
        result->Configure("seed", "0");
        result->CreateSimulator(backend, method, 2);
        return result;
      };
      auto measured =
          std::static_pointer_cast<Circuits::MeasurementOperation<>>(
              CF::CreateMeasurement({{0, 0}, {1, 1}}));
      measured->SetReadout({{0.2, 0.35}, {0.4, 0.1}});
      auto noisy =
          CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 0), measured});
      const auto fresh = makeSeeded()->RepeatedExecuteOnHost(noisy, 0, 257);
      auto reused = makeSeeded();
      reused->ExecuteOnHost(
          CF::CreateCircuit({CF::CreateGate(Gate::kXGateType, 1), measured}),
          0);
      Check(fresh.size() == 4 &&
                reused->RepeatedExecuteOnHost(noisy, 0, 257) == fresh,
            "Host reuse leaked retained state or readout RNG position");
    }
}

void SharedReset() {
  // Reset descriptions are shared by jobs. Resetting several targets to |1>
  // must not mutate an instruction-owned X gate's target between threads.
  Circuits::Reset<> reset({0, 1}, 0, {true, true});
  std::array<std::thread, 2> threads;
  std::array<std::exception_ptr, 2> errors;
  std::atomic<size_t> ready{0}, wrong{0};
  for (size_t worker = 0; worker < threads.size(); ++worker)
    threads[worker] = std::thread([&, worker] {
      try {
        auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
            Backend::kQCSim, Method::kStatevector);
        simulator->AllocateQubits(2);
        simulator->Initialize();
        Circuits::OperationState state;
        ++ready;
        while (ready.load() != threads.size()) std::this_thread::yield();
        for (size_t shot = 0; shot < 1024; ++shot) {
          reset.Execute(simulator, state);
          if (simulator->Probability(3) < 1.0 - 1e-12) ++wrong;
        }
      } catch (...) {
        errors[worker] = std::current_exception();
      }
    });
  for (auto& thread : threads) thread.join();
  for (const auto& error : errors)
    if (error) std::rethrow_exception(error);
  Check(wrong.load() == 0, "Shared reset changed another worker's X target");
}
}  // namespace

int main() try {
  NestedCircuitJobs();
  NestedClassicalDataflow();
  NestedShotBoundaries();
  ConditionalBitHelpers();
  HostSimulatorReuse();
  RandomJobs();
  RandomSeedPolicy();
  MeasurementJobs();
  NoiseClonePolicy();
  ConditionalOutputMapping();
  SharedReset();
  std::cout << checks << " network job checks passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "Network job regression: " << error.what() << '\n';
  return 1;
}
