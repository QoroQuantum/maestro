// Shared request/configuration checks require no MPI or GPU hardware.
#include "../Circuit/Factory.h"
#include "../Execution/Options.h"
#include "../Execution/SampleRecords.h"
#include <algorithm>
#include <atomic>
#include <complex>
#include <iostream>

void Check(bool, const char *);

namespace
{
// Count actual gate execution through cloning, mapping and worker jobs. This
// catches repeated evolution without a wall-clock performance assertion.
class CountedX : public Circuits::XGate<>
{
  public:
    CountedX(size_t qubit, std::shared_ptr<std::atomic<size_t>> calls) : XGate(qubit), calls(std::move(calls))
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

// RepeatedExecute uses a distribution remapper; a single host needs no
// remapping, while still exercising the production network job dispatch.
class SingleHostRemapper : public Distribution::IRemapper<>
{
  public:
    std::shared_ptr<Circuits::Circuit<>> Remap(const std::shared_ptr<Network::INetwork<>> &, const std::shared_ptr<Circuits::Circuit<>> &circuit) override
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

void TestPreparedNestedResetAfterMeasurement()
{
    using namespace MaestroExecution;
    using CF = Circuits::CircuitFactory<>;
    using Gate = Circuits::QuantumGateType;
    constexpr size_t shots = 4096;
    for (const auto method : {Method::kDensityMatrix, Method::kMatrixProductOperator})
        for (const bool fusion : {false, true})
        {
            auto nestedCalls = std::make_shared<std::atomic<size_t>>(0);
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
            circuit->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 5));
            circuit->AddOperation(CF::CreateMeasurement({{0, 3}}));
            auto inner = CF::CreateCircuit();
            inner->AddOperation(CF::CreateReset({5}));
            inner->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 5));
            inner->AddOperation(std::make_shared<CountedX>(2, nestedCalls));
            inner->AddOperation(CF::CreateGate(Gate::kCZGateType, 2, 5));
            auto outer = CF::CreateCircuit();
            outer->AddOperation(inner);
            circuit->AddOperation(outer);
            circuit->AddOperation(CF::CreateMeasurement({{5, 0}, {2, 4}}));
            Check(!circuit->HasOpsAfterMeasurements(true), "Disjoint deterministic composite prevented terminal mixed-state sampling");

            auto baseline = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQCSim, method);
            baseline->Configure("gate_fusion", fusion ? "true" : "false");
            baseline->SetMultithreading(false);
            baseline->AllocateQubits(6);
            baseline->Initialize();
            baseline->SetLookaheadDepth(4);
            baseline->SetSeed(1123);
            Circuits::Circuit<>::ExecuteResults repeated;
            Circuits::OperationState state(5);
            for (size_t shot = 0; shot < shots; ++shot)
            {
                baseline->Reset();
                baseline->SetGatesCounter(0);
                baseline->SetUpcomingGates(circuit->GetOperations());
                circuit->ExecuteBD(baseline, state);
                ++repeated[state.GetAllBits()];
            }
            Check(*nestedCalls == shots, "Repeated reference did not execute the nested circuit each shot");

            for (const bool onHost : {false, true})
            {
                const unsigned long handle = CreateSimpleSimulator(6);
                try
                {
                    SimulatorConfig config;
                    config.simulation_type = method;
                    config.fixed_backend = true;
                    config.optimize_circuit = false;
                    config.max_simulators = 4;
                    config.gate_fusion = fusion;
                    config.lookahead_depth = 4;
                    config.seed = 1123;
                    auto network = ConfigureNetwork(handle, config);
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                    *nestedCalls = 0;
                    const auto prepared = onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots) : network->RepeatedExecute(circuit, shots);
                    Check(*nestedCalls == 1, "Nested deterministic reset circuit was evolved more than once");
                    Check(prepared.size() == 4 && repeated.size() == 4, "Nested circuit lost an earlier measurement record");
                    for (const auto &[bits, count] : repeated)
                    {
                        // Both network paths retain the allocated classical
                        // register when restoring the circuit's original mapping.
                        auto expectedBits = bits;
                        expectedBits.resize(6, false);
                        const auto found = prepared.find(expectedBits);
                        Check(bits[4] && !bits[1] && !bits[2] && found != prepared.end(), "Nested circuit changed classical bit mapping");
                        Check(std::abs(double(count) / shots - 0.25) < 0.06 && std::abs(double(found->second) / shots - double(count) / shots) < 0.08,
                              "Prepared mixed state disagrees with repeated nested measurement/reset evolution");
                    }
                }
                catch (...)
                {
                    DestroySimpleSimulator(handle);
                    throw;
                }
                DestroySimpleSimulator(handle);
            }
        }
}
} // namespace

void TestFixedBackendShotReuse()
{
    using namespace MaestroExecution;
    using CF = Circuits::CircuitFactory<>;
    using Gate = Circuits::QuantumGateType;
    GetMaestroObjectWithMute();
    TestPreparedNestedResetAfterMeasurement();
    for (const char *method : {"statevector", "matrix_product_state", "density_matrix", "matrix_product_operator"})
        for (bool onHost : {true, false})
            for (size_t workers : {size_t{1}, size_t{4}})
                for (size_t shots : {size_t{128}, size_t{131}})
                    for (int scenario : {0, 1, 2, 3, 4, 5, 6})
                    {
                        struct NetworkHandle
                        {
                            unsigned long handle = CreateSimpleSimulator(2);

                            ~NetworkHandle()
                            {
                                DestroySimpleSimulator(handle);
                            }
                        } owner;

                        Check(owner.handle != 0, "Cannot create network for shot reuse");
                        // Exercise the native parser's default fixed selection and the same
                        // ConfigureNetwork path used by native requests.
                        auto config = ParseConfig(json::object{{"backend", "qcsim"}, {"method", method}});
                        config.seed = 123;
                        config.optimize_circuit = false;
                        auto network = ConfigureNetwork(owner.handle, config);
                        Check(network && !network->GetOptimizeSimulator(), "Fixed selection unexpectedly enabled backend optimization");
                        network->SetMaxSimulators(workers);
                        if (!onHost)
                            network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());

                        auto prefix = std::make_shared<std::atomic<size_t>>(0);
                        auto suffix = std::make_shared<std::atomic<size_t>>(0);
                        auto circuit = CF::CreateCircuit();
                        circuit->AddOperation(std::make_shared<CountedX>(0, prefix));
                        if (scenario == 1 || scenario == 3 || scenario == 6)
                        {
                            auto body = scenario == 6 ? CF::CreateCircuit() : circuit;
                            if (scenario == 1 || scenario == 6)
                            {
                                body->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                                body->AddOperation(CF::CreateMeasurement({{0, 0}}));
                            }
                            else
                            {
                                circuit->AddOperation(CF::CreateRandom({0}, 123));
                            }
                            body->AddOperation(CF::CreateSimpleConditionalGate(CF::CreateGate(Gate::kXGateType, 1), 0));
                            body->AddOperation(std::make_shared<CountedX>(1, suffix));
                            body->AddOperation(CF::CreateMeasurement({{1, 1}}));
                            if (scenario == 6)
                                circuit->AddOperation(body);
                        }
                        else
                        {
                            if (scenario == 2 || scenario == 5)
                            {
                                auto body = scenario == 5 ? CF::CreateCircuit() : circuit;
                                body->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
                                body->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 1));
                                body->AddOperation(CF::CreateReset({0}));
                                body->AddOperation(std::make_shared<CountedX>(0, suffix));
                                if (scenario == 5)
                                    circuit->AddOperation(body);
                            }
                            else if (scenario == 4)
                            {
                                auto inner = CF::CreateCircuit();
                                inner->AddOperation(std::make_shared<CountedX>(1, suffix));
                                inner->AddOperation(CF::CreateGate(Gate::kXGateType, 1));
                                auto outer = CF::CreateCircuit();
                                outer->AddOperation(inner);
                                circuit->AddOperation(outer);
                            }
                            circuit->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
                        }

                        // Mixed-state resets and deterministic nested circuits belong
                        // to the shared evolution. Genuine classical branches still
                        // replay their suffix per shot.
                        const bool exactChannels = config.simulation_type == Method::kDensityMatrix || config.simulation_type == Method::kMatrixProductOperator;
                        const bool terminal = scenario == 0 || scenario == 4 || (exactChannels && (scenario == 2 || scenario == 5));
                        const std::string context = std::string(method) + " scenario=" + std::to_string(scenario) + " workers=" + std::to_string(workers) +
                                                    " shots=" + std::to_string(shots) + " on_host=" + std::to_string(onHost);
                        Check(circuit->HasOpsAfterMeasurements(exactChannels) != terminal, "Terminal classification disagrees with backend reset semantics");
                        // Terminal workers share one prepared prefix. Dynamic
                        // logical jobs prepare once per active worker here.
                        size_t jobs = terminal ? 1 : network->GetMaxSimulators();
#ifdef _OPENMP
                        if (!terminal)
                            jobs = std::min(jobs, static_cast<size_t>(std::max(1, omp_get_max_threads())));
#endif
                        for (int repeat = 0; repeat < 2; ++repeat)
                        {
                            *prefix = 0;
                            *suffix = 0;
                            const auto counts = onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots) : network->RepeatedExecute(circuit, shots);
                            Check(prefix->load() == jobs,
                                  (context + ": expected " + std::to_string(jobs) + " prefix executions, got " + std::to_string(prefix->load())).c_str());
                            Check(suffix->load() == (scenario == 0 ? 0
                                                     : terminal    ? jobs
                                                                   : shots),
                                  (context + ": shared evolution or per-shot suffix execution count changed").c_str());
                            Check(network->GetLastSimulatorType() == Backend::kQCSim && network->GetLastSimulationType() == config.simulation_type,
                                  "Shot reuse changed the fixed backend or method");
                            size_t total = 0;
                            for (const auto &[bits, count] : counts)
                            {
                                Check(bits.size() == 2, "Shot reuse changed classical width");
                                Check((scenario == 1 || scenario == 3 || scenario == 6) ? bits[0] != bits[1]
                                                                                        : bits[0] && (scenario == 2 || scenario == 5 || !bits[1]),
                                      "Shot reuse changed measurement/reset/conditional results");
                                total += count;
                            }
                            Check(total == shots, "Shot reuse lost counts");
                            Check(counts.size() == (scenario == 0 || scenario == 4 ? 1 : 2), "Independent shots reused one measurement/reset outcome");
                        }
                    }
}

void TestRequestSeedParsing()
{
    for (const char *backend : {
             "qcsim",
#ifdef __linux__
             "distributed_mpi_gpu",
#endif
         })
    {
        MaestroExecution::json::object simulator{{"backend", backend}};
        Check(!MaestroExecution::ParseConfig(simulator).seed.has_value(), "Parsing an omitted seed must preserve the unset state");
        for (uint64_t seed : {uint64_t{0}, UINT64_MAX})
        {
            simulator["options"] = MaestroExecution::json::object{{"seed", seed}};
            const auto config = MaestroExecution::ParseConfig(simulator);
            Check(config.seed.has_value() && *config.seed == seed, "Parsing changed an explicit seed");
        }
    }
}

void TestNetworkBondDefaults()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    const char *const bondKey = "matrix_product_state_max_bond_dimension";
    const auto checkNetwork = [bondKey](const SimulatorConfig &config, const char *expected) {
        struct NetworkHandle
        {
            unsigned long handle = CreateSimpleSimulator(2);
            ~NetworkHandle()
            {
                DestroySimpleSimulator(handle);
            }
        } owner;
        Check(owner.handle != 0, "Cannot create network for bond configuration");
        const auto network = ConfigureNetwork(owner.handle, config);
        Check(bool(network), "Cannot configure network bond dimension");
        // Automatic selection first creates a CPU placeholder from the same
        // network configuration the initial-layout planner reads. No GPU is needed.
        Check(network->GetSimulator()->GetConfiguration(bondKey) == expected, "Network setup used the wrong bond dimension");
        network->CreateSimulator();
        Check(network->GetSimulator()->GetConfiguration(bondKey) == expected, "Network recreation lost the bond dimension");
    };

    for (const auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        SimulatorConfig config;
        config.simulator_type = Backend::kGpuSim;
        config.simulation_type = method;
        checkNetwork(config, "128");
        Check(!config.max_bond_dimension, "Resolving the default mutated the caller's optional setting");
        config.max_bond_dimension = 128;
        checkNetwork(config, "128");
        config.max_bond_dimension = 64;
        checkNetwork(config, "64");
        config.max_bond_dimension.reset();
        config.native_options[bondKey] = "32";
        checkNetwork(config, "32");
        config.max_bond_dimension = 64;
        checkNetwork(config, "64");

        config.max_bond_dimension.reset();
        config.native_options.clear();
        config.simulator_type = Backend::kQCSim;
        checkNetwork(config, "");
    }
    for (const auto method : {Method::kStatevector, Method::kTensorNetwork})
    {
        SimulatorConfig config;
        config.simulator_type = Backend::kGpuSim;
        config.simulation_type = method;
        checkNetwork(config, "");
    }
}

void TestPauliPropagatorDedupDefault()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    const char *const key = "pauli_propagator_num_gates_between_deduplications";
    const auto cadence = [key](const SimulatorConfig &config) {
        struct NetworkHandle
        {
            unsigned long handle = CreateSimpleSimulator(2);
            ~NetworkHandle()
            {
                DestroySimpleSimulator(handle);
            }
        } owner;
        Check(owner.handle != 0, "Cannot create network for cadence configuration");
        const auto network = ConfigureNetwork(owner.handle, config);
        Check(bool(network), "Cannot configure network cadence");
        return network->GetSimulator()->GetConfiguration(key);
    };

    SimulatorConfig config;
    config.simulation_type = Method::kPauliPropagator;
    Check(cadence(config) == "10", "PauliPropagator did not get the default deduplication cadence");
    config.pp_gates_between_deduplications = 3;
    Check(cadence(config) == "3", "An explicit cadence was replaced");
    config.pp_gates_between_deduplications.reset();
    config.native_options[key] = "4";
    Check(cadence(config) == "4", "A native-option cadence was replaced");
    Check(cadence(SimulatorConfig{}).empty(), "A statevector simulation got a deduplication cadence");
}

void TestPauliTuningForwarding()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    const std::pair<std::optional<uint64_t> SimulatorConfig::*, const char *> fields[] = {
        {&SimulatorConfig::pp_workers, "pauli_propagator_workers"},
        {&SimulatorConfig::pp_max_fused_gates, "pauli_propagator_max_fused_gates"},
        {&SimulatorConfig::pp_memory_limit_bytes, "pauli_propagator_memory_limit_bytes"},
        {&SimulatorConfig::pp_sampling_cache_nodes, "pauli_propagator_sampling_cache_nodes"}};
    const auto stored = [&](const SimulatorConfig &config) {
        struct NetworkHandle
        {
            unsigned long handle = CreateSimpleSimulator(2);
            ~NetworkHandle()
            {
                DestroySimpleSimulator(handle);
            }
        } owner;
        Check(owner.handle != 0, "Cannot create network for Pauli tuning");
        const auto network = ConfigureNetwork(owner.handle, config);
        Check(bool(network) && network->GetSimulator(), "Cannot configure Pauli tuning network");
        std::vector<std::string> values;
        for (const auto &field : fields)
            values.push_back(network->GetSimulator()->GetConfiguration(field.second));
        return values;
    };

    SimulatorConfig config;
    config.simulation_type = Method::kPauliPropagator;
    config.fixed_backend = true;
    for (const auto &value : stored(config))
        Check(value.empty(), "An unset Pauli tuning option reached the backend");

    // Zero is a valid cache size (caching off) and must not be dropped as unset.
    config.pp_workers = 3;
    config.pp_max_fused_gates = 17;
    config.pp_memory_limit_bytes = 1ull << 33;
    config.pp_sampling_cache_nodes = 0;
    Check(stored(config) == std::vector<std::string>{"3", "17", "8589934592", "0"}, "Typed Pauli tuning options did not reach the backend");
}

// The extended stabilizer settings as typed fields and as native keys: one
// rule for a value, which backend a setting belongs to, and a handle that is
// configured again.
void TestExtendedStabilizerConfiguration()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    const char *tolerance = "extended_stabilizer_amplitude_tolerance";
    const char *cap = "extended_stabilizer_max_components";
    const char *error = "extended_stabilizer_approximation_error";
    const auto rejected = [](const SimulatorConfig &config) {
        try
        {
            config.Validate();
        }
        catch (const std::invalid_argument &)
        {
            return true;
        }
        return false;
    };
    SimulatorConfig extended;
    extended.simulation_type = Method::kExtendedStabilizer;
    extended.fixed_backend = true;

    // A native value is checked as the typed field is, and as the simulator
    // checks it: no value passes here and fails in Configure.
    for (const auto &[key, value] : std::vector<std::pair<const char *, const char *>>{{tolerance, "nan"},
                                                                                       {tolerance, "-0.5"},
                                                                                       {tolerance, "1"},
                                                                                       {tolerance, "0.1x"},
                                                                                       {tolerance, "abc"},
                                                                                       {cap, "0"},
                                                                                       {cap, "-1"},
                                                                                       {cap, "many"},
                                                                                       {error, "0"},
                                                                                       {error, "2"},
                                                                                       {"extended_stabilizer_sampling_method", "gibbs"},
                                                                                       {"extended_stabilizer_metropolis_mixing_time", "0"},
                                                                                       {"extended_stabilizer_norm_estimation_samples", "2147483648"}})
    {
        SimulatorConfig config; // a statevector: neither extended stabilizer may run, so only the value can be refused
        config.native_options[key] = value;
        Check(rejected(config), "An invalid native extended stabilizer value passed validation");
    }
    {
        SimulatorConfig config = extended;
        config.extended_stabilizer_amplitude_tolerance = 1.;
        Check(rejected(config), "A typed tolerance of 1 passed validation");
        config.extended_stabilizer_amplitude_tolerance = std::numeric_limits<double>::quiet_NaN();
        Check(rejected(config), "A typed NaN tolerance passed validation");
        config.extended_stabilizer_amplitude_tolerance = 0.;
        config.extended_stabilizer_max_components = 0;
        Check(rejected(config), "A typed cap of zero passed validation");
        config = SimulatorConfig{};
        config.extended_stabilizer_approximation_error = 0.;
        Check(rejected(config), "A typed approximation error of zero passed validation");
        config.extended_stabilizer_approximation_error = 0.5;
        config.extended_stabilizer_sampling_method = "gibbs";
        Check(rejected(config), "An unknown typed sampling method passed validation");
        config.extended_stabilizer_sampling_method = "metropolis";
        config.extended_stabilizer_metropolis_mixing_time = uint64_t{1} << 31;
        Check(rejected(config), "A typed mixing time beyond Aer's int passed validation");
        config.extended_stabilizer_metropolis_mixing_time = 10;
        config.extended_stabilizer_norm_estimation_samples = 0;
        Check(rejected(config), "Zero typed norm estimation samples passed validation");
        config.extended_stabilizer_norm_estimation_samples = 20;
        config.extended_stabilizer_norm_estimation_repetitions = 0;
        Check(rejected(config), "Zero typed norm estimation repetitions passed validation");
        config.extended_stabilizer_norm_estimation_repetitions = 3;
        Check(!rejected(config), "Valid Aer extended stabilizer settings were refused where neither extended stabilizer runs");
    }

    // Pruning means a positive tolerance or a cap. The typed field decides
    // over the native key, as ConfigureNetwork applies it last.
    {
        SimulatorConfig config = extended;
        Check(!SetsExtendedStabilizerApproximation(config) && !PrunesExtendedStabilizer(config), "An unset configuration prunes");
        config.extended_stabilizer_amplitude_tolerance = 0.;
        Check(!SetsExtendedStabilizerApproximation(config), "A zero tolerance prunes");
        config.native_options[tolerance] = "0.1";
        Check(!SetsExtendedStabilizerApproximation(config), "A native tolerance overrode the typed zero");
        config.extended_stabilizer_amplitude_tolerance.reset();
        Check(SetsExtendedStabilizerApproximation(config) && PrunesExtendedStabilizer(config), "A native tolerance does not prune");
        config.native_options[tolerance] = "0";
        Check(!SetsExtendedStabilizerApproximation(config), "A native zero tolerance prunes");
        config.native_options[cap] = "7";
        Check(SetsExtendedStabilizerApproximation(config), "A native cap does not prune");
        config.native_options.clear();
        config.extended_stabilizer_max_components = 7;
        Check(SetsExtendedStabilizerApproximation(config), "A typed cap does not prune");
        config.simulation_type = Method::kStatevector;
        Check(!PrunesExtendedStabilizer(config), "A statevector prunes");
    }

    // Which backend may run: the requested pair, or the candidates of an
    // automatic list, which replace it.
    {
        SimulatorConfig config;
        config.simulation_type = Method::kExtendedStabilizer;
        Check(config.MayRun(Backend::kQCSim, Method::kExtendedStabilizer) && !config.MayRun(Backend::kQCSim, Method::kStatevector),
              "MayRun misreads the requested pair");
        config.optimization_candidates = {{Backend::kQCSim, Method::kStatevector}};
        Check(!config.MayRun(Backend::kQCSim, Method::kExtendedStabilizer) && config.MayRun(Backend::kQCSim, Method::kStatevector),
              "A candidate list did not replace the requested pair");
        config.fixed_backend = true;
        Check(config.MayRun(Backend::kQCSim, Method::kExtendedStabilizer) && !config.MayRun(Backend::kQCSim, Method::kStatevector),
              "A fixed backend followed its candidate list");
    }

    // Each backend ignores the settings of the other: refused where that
    // backend may run, as typed fields and as native keys.
    {
        SimulatorConfig config = extended;
        config.extended_stabilizer_approximation_error = 0.1;
        Check(rejected(config), "The QCSim extended stabilizer took Aer's typed approximation error");
        config.extended_stabilizer_approximation_error.reset();
        config.native_options["extended_stabilizer_sampling_method"] = "metropolis";
        Check(rejected(config), "The QCSim extended stabilizer took Aer's native sampling method");
        // A list without the QCSim extended stabilizer replaces the requested pair.
        config.fixed_backend = false;
        config.optimization_candidates = {{Backend::kQCSim, Method::kStatevector}};
        Check(!rejected(config), "An Aer setting was refused though the QCSim extended stabilizer cannot run");
        config.optimization_candidates.emplace_back(Backend::kQCSim, Method::kExtendedStabilizer);
        Check(rejected(config), "An Aer setting was accepted with the QCSim extended stabilizer as a candidate");
#ifndef NO_QISKIT_AER
        SimulatorConfig aer;
        aer.simulator_type = Backend::kQiskitAer;
        aer.simulation_type = Method::kExtendedStabilizer;
        aer.fixed_backend = true;
        aer.extended_stabilizer_approximation_error = 0.1;
        aer.extended_stabilizer_sampling_method = "norm_estimation";
        Check(!rejected(aer), "Aer's extended stabilizer refused its own settings");
        aer.native_options[cap] = "8";
        Check(rejected(aer), "Aer's extended stabilizer took QCSim's native cap");
        aer.native_options.clear();
        aer.extended_stabilizer_amplitude_tolerance = 0.1;
        Check(rejected(aer), "Aer's extended stabilizer took QCSim's typed tolerance");
        // Requested on Aer, but the list names only QCSim's: the pruning applies there.
        aer.extended_stabilizer_approximation_error.reset();
        aer.extended_stabilizer_sampling_method.reset();
        aer.fixed_backend = false;
        aer.optimization_candidates = {{Backend::kQCSim, Method::kExtendedStabilizer}};
        Check(!rejected(aer), "Pruning was refused though only the QCSim extended stabilizer can run");
        // A list with both takes neither family.
        aer.optimization_candidates.emplace_back(Backend::kQiskitAer, Method::kExtendedStabilizer);
        Check(rejected(aer), "Pruning was accepted with Aer's extended stabilizer as a candidate");
        aer.extended_stabilizer_amplitude_tolerance.reset();
        Check(!rejected(aer), "A list with both extended stabilizers and no setting was refused");
        aer.extended_stabilizer_approximation_error = 0.1;
        Check(rejected(aer), "Aer's error was accepted with QCSim's extended stabilizer as a candidate");
#endif
    }

    // The settings reach the simulator, and a handle that is configured again
    // without them goes back to the default: it does not keep the pruning (or
    // Aer's settings) of the configuration before.
    struct NetworkHandle
    {
        unsigned long handle = CreateSimpleSimulator(2);

        ~NetworkHandle()
        {
            DestroySimpleSimulator(handle);
        }
    } owner;

    Check(owner.handle != 0, "Cannot create a network for the extended stabilizer settings");
    const auto stored = [&](const SimulatorConfig &config, const char *key) {
        const auto network = ConfigureNetwork(owner.handle, config);
        Check(bool(network) && network->GetSimulator(), "Cannot configure an extended stabilizer network");
        return network->GetSimulator()->GetConfiguration(key);
    };
    const auto probability = [&](const SimulatorConfig &config) {
        const auto network = ConfigureNetwork(owner.handle, config, 1);
        Check(bool(network) && network->GetSimulator(), "Cannot configure an extended stabilizer register");
        const auto simulator = network->GetSimulator();
        simulator->ApplyRy(0, 0.2);
        return simulator->Probability(1);
    };
    const double excited = std::sin(0.1) * std::sin(0.1);
    {
        SimulatorConfig config = extended;
        Check(stored(config, tolerance).empty() && stored(config, cap).empty(), "An unset pruning setting reached the backend");
        Check(std::abs(probability(config) - excited) < 1e-9, "An unset configuration pruned");
        config.extended_stabilizer_amplitude_tolerance = 0.2;
        config.extended_stabilizer_max_components = 9;
        Check(std::stod(stored(config, tolerance)) == 0.2 && stored(config, cap) == "9", "Typed pruning settings did not reach the backend");
        Check(probability(config) < 1e-12, "A typed tolerance did not prune");
        Check(stored(extended, tolerance).empty() && stored(extended, cap).empty(), "A handle kept the pruning of an earlier configuration");
        Check(std::abs(probability(extended) - excited) < 1e-9, "A handle kept pruning after a configuration without it");
        SimulatorConfig native = extended;
        native.native_options[tolerance] = "0.2";
        Check(stored(native, tolerance) == "0.2" && probability(native) < 1e-12, "A native tolerance did not prune");
        Check(std::abs(probability(extended) - excited) < 1e-9, "A handle kept a native tolerance after a configuration without it");
        // A zero tolerance is the exact simulation, not an error.
        config = extended;
        config.extended_stabilizer_amplitude_tolerance = 0.;
        Check(std::abs(probability(config) - excited) < 1e-9, "A zero tolerance did not run exactly");
        // The network refuses a bad value before it keeps it: one it kept
        // would be replayed to the simulator it creates next, which throws.
        const auto network = ConfigureNetwork(owner.handle, extended);
        bool refused = false;
        try
        {
            network->Configure(tolerance, "2");
        }
        catch (const std::invalid_argument &)
        {
            refused = true;
        }
        network->CreateSimulator(Backend::kQCSim, Method::kExtendedStabilizer, 1);
        Check(refused && network->GetSimulator() && network->GetSimulator()->GetConfiguration(tolerance).empty(),
              "A network kept a tolerance its simulator refuses");
    }
#ifndef NO_QISKIT_AER
    {
        SimulatorConfig aer;
        aer.simulator_type = Backend::kQiskitAer;
        aer.simulation_type = Method::kExtendedStabilizer;
        aer.fixed_backend = true;
        Check(stored(aer, error).empty(), "An unset Aer approximation error reached the backend");
        aer.extended_stabilizer_approximation_error = 0.1;
        aer.extended_stabilizer_sampling_method = "metropolis";
        aer.extended_stabilizer_metropolis_mixing_time = 77;
        aer.extended_stabilizer_norm_estimation_samples = 31;
        aer.extended_stabilizer_norm_estimation_repetitions = 2;
        Check(std::stod(stored(aer, error)) == 0.1 && stored(aer, "extended_stabilizer_sampling_method") == "metropolis" &&
                  stored(aer, "extended_stabilizer_metropolis_mixing_time") == "77" && stored(aer, "extended_stabilizer_norm_estimation_samples") == "31" &&
                  stored(aer, "extended_stabilizer_norm_estimation_repetitions") == "2",
              "Typed Aer extended stabilizer settings did not reach the backend");
        SimulatorConfig plain = aer;
        plain.extended_stabilizer_approximation_error.reset();
        plain.extended_stabilizer_sampling_method.reset();
        plain.extended_stabilizer_metropolis_mixing_time.reset();
        plain.extended_stabilizer_norm_estimation_samples.reset();
        plain.extended_stabilizer_norm_estimation_repetitions.reset();
        Check(stored(plain, error).empty() && stored(plain, "extended_stabilizer_sampling_method").empty() &&
                  stored(plain, "extended_stabilizer_metropolis_mixing_time").empty(),
              "A handle kept Aer's settings of an earlier configuration");
    }
#endif
}

void TestAutomaticGpuMixedStateFallback()
{
    using namespace MaestroExecution;
    using CF = Circuits::CircuitFactory<>;
    GetMaestroObjectWithMute();

    for (const auto method : {Method::kDensityMatrix, Method::kMatrixProductOperator})
        for (size_t workers : {size_t{1}, size_t{2}})
        {
            struct NetworkHandle
            {
                unsigned long handle = CreateSimpleSimulator(1);

                ~NetworkHandle()
                {
                    DestroySimpleSimulator(handle);
                }
            } owner;

            SimulatorConfig config;
            config.simulator_type = Backend::kGpuSim;
            config.simulation_type = method;
            config.seed = 123;
            config.optimize_circuit = false;
            const auto network = ConfigureNetwork(owner.handle, config);
            Check(network && network->GetSimulator()->GetType() == Backend::kQCSim &&
                      network->GetSimulator()->GetSimulationType() == Method::kMatrixProductOperator,
                  "Automatic GPU mixed-state execution needs a channel-capable CPU "
                  "fallback");

            // Exercise the CPU placeholder even on machines with a working GPU.
            // An unavailable automatic candidate leaves this same simulator in use.
            network->SetOptimizeSimulator(false);
            network->SetMaxSimulators(workers);
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(CF::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
            circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0},
                                                                                        Simulators::QuantumChannel::GeneralizedAmplitudeDamping(1.0, 0.0)));
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
            for (size_t shots : {size_t{1}, size_t{64}})
            {
                const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
                Check(counts == Circuits::Circuit<>::ExecuteResults{{{false}, shots}}, "GPU CPU fallback did not execute the exact noise channel");
                Check(network->GetLastSimulatorType() == Backend::kQCSim && network->GetLastSimulationType() == Method::kMatrixProductOperator,
                      "GPU fallback changed the mixed-state representation");
            }
        }
}

void TestTypedTrajectoryConfiguration()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    for (const bool typed : {false, true})
    {
        struct NetworkHandle
        {
            unsigned long handle = CreateSimpleSimulator(2);

            ~NetworkHandle()
            {
                DestroySimpleSimulator(handle);
            }
        } owner;

        // JSON options and Python's typed fields share ConfigureNetwork.
        auto config =
            ParseConfig(json::object{{"backend", "qcsim"},
                                     {"method", "statevector"},
                                     {"options", json::object{{"trajectory_policy", "block_v1"}, {"trajectory_max_memory_mb", 64}, {"max_simulators", 1}}}});
        if (typed)
        {
            config.trajectory_policy = "shot_v1";
            config.trajectory_max_memory_mb = 1;
            config.max_simulators = 2;
        }
        const auto network = ConfigureNetwork(owner.handle, config);
        Check(bool(network), "Cannot configure trajectory network");
        // The network caps the limit at the hardware thread count.
        const size_t expectedWorkers = std::min<size_t>(typed ? 2 : 1, QC::QubitRegisterCalculator<>::GetNumberOfThreads());
        Check(network->GetMaxSimulators() == expectedWorkers, "Typed worker limit was ignored");
        Check(network->GetSimulator()->GetConfiguration("trajectory_policy") == (typed ? "shot_v1" : "block_v1"),
              "Trajectory policy did not reach the network");
        Check(network->GetSimulator()->GetConfiguration("trajectory_max_memory_mb") == (typed ? "1" : "64"),
              "Trajectory memory budget did not reach the network");
    }
}

void TestFrameCountPolicy()
{
    using namespace MaestroExecution;
    SimulatorConfig configured;
    for (const auto seed : {0, 123})
    {
        configured.seed = seed;
        Check(PreferMultinomialCounts(configured) && PreferMultinomialCounts(configured, seed), "A typed simulator or noise seed disabled adaptive counts");
        configured.seed.reset();
        configured.native_options["seed"] = std::to_string(seed);
        Check(EffectiveSamplingSeed(configured) == seed && PreferMultinomialCounts(configured), "A native seed disabled adaptive counts");
    }
    configured.counts_policy = "legacy";
    configured.native_options["counts_policy"] = "auto";
    Check(PreferMultinomialCounts(configured), "Native auto did not override the typed counts policy");
    for (const char *policy : {"legacy", "frame_v1", "multinomial_v1"})
    {
        configured.native_options["counts_policy"] = policy;
        Check(PreferMultinomialCounts(configured) == (std::string(policy) == "multinomial_v1"), "A native count policy was ignored");
    }
    configured.native_options["sampling_policy"] = "reproducible_v1";
    Check(PreferMultinomialCounts(configured), "Explicit reproducible sampling disabled multinomial counts");
    configured.native_options["sampling_policy"] = "legacy";
    Check(!PreferMultinomialCounts(configured), "Explicit legacy sampling did not override multinomial counts");
    SetCountsPolicy(configured, "legacy");
    Check(EffectiveCountsPolicy(configured) == "legacy" && !configured.native_options.count("counts_policy"), "A resolved counts policy was overridden");
    for (const char *policy : {"auto", "legacy", "frame_v1"})
    {
        const auto config = ParseConfig(json::object{{"backend", "qcsim"}, {"method", "stabilizer"}, {"options", json::object{{"counts_policy", policy}}}});
        Check(config.counts_policy == policy, "Counts policy parser lost the option");
        auto seeded = config;
        seeded.seed = 0;
        Check(PreferFrameCounts(seeded) == (std::string(policy) != "legacy"), "Seed zero disabled eligible frame counts");
        Check(PreferFrameCounts(config) == (std::string(policy) != "legacy"), "Unseeded counts policy selection");
        // auto keeps a requested non-Stabilizer method unless the automatic
        // candidates admit the stabilizer; frame_v1 needs automatic selection.
        for (const char *selection : {"fixed", "automatic", "candidates"})
        {
            const std::string mode(selection);
            json::object simulator{{"backend", "qcsim"},
                                   {"method", "statevector"},
                                   {"selection", mode == "fixed" ? "fixed" : "automatic"},
                                   {"options", json::object{{"counts_policy", policy}}}};
            if (mode == "candidates")
                simulator["candidates"] =
                    json::array{json::object{{"backend", "qcsim"}, {"method", "statevector"}}, json::object{{"backend", "qcsim"}, {"method", "stabilizer"}}};
            const auto statevector = ParseConfig(simulator);
            const bool automatic = mode != "fixed";
            const bool expected = std::string(policy) == "frame_v1" ? automatic : std::string(policy) == "auto" && mode == "candidates";
            Check(PreferFrameCounts(statevector) == expected, "Unseeded non-Stabilizer counts policy selection");
            auto seededStatevector = statevector;
            seededStatevector.seed = 0;
            Check(PreferFrameCounts(seededStatevector) == expected, "Seeded non-Stabilizer counts policy selection");
        }
        // An explicit candidate list names every allowed backend. It replaces a
        // requested Stabilizer method too, and binds frame_v1 as well as auto.
        for (const char *method : {"stabilizer", "statevector"})
            for (const bool listed : {false, true})
            {
                json::array candidates{json::object{{"backend", "qcsim"}, {"method", "statevector"}}};
                if (listed)
                    candidates.push_back(json::object{{"backend", "qcsim"}, {"method", "stabilizer"}});
                const auto restricted = ParseConfig(json::object{{"backend", "qcsim"},
                                                                 {"method", method},
                                                                 {"selection", "automatic"},
                                                                 {"candidates", candidates},
                                                                 {"options", json::object{{"counts_policy", policy}}}});
                const bool expected = listed && std::string(policy) != "legacy";
                Check(PreferFrameCounts(restricted) == expected, "A frame policy bypassed the request's candidate list");
            }
    }
    for (const char *invalid : {"", "frame", "FRAME_V1"})
    {
        auto config = SimulatorConfig{};
        config.counts_policy = invalid;
        bool rejected = false;
        try
        {
            config.Validate();
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        Check(rejected, "Unknown counts policy accepted");
    }
}

void TestGateFusionConfiguration()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    for (const bool fusion : {false, true})
    {
        const auto config =
            ParseConfig(json::object{{"backend", "qcsim"}, {"method", "matrix_product_state"}, {"options", json::object{{"gate_fusion", fusion}}}});
        Check(config.gate_fusion == fusion, "Fusion option parser");

        struct NetworkHandle
        {
            unsigned long handle = CreateSimpleSimulator(2);

            ~NetworkHandle()
            {
                DestroySimpleSimulator(handle);
            }
        } owner;

        const auto network = ConfigureNetwork(owner.handle, config);
        Check(network && network->GetSimulator()->IsGateFusionEnabled() == fusion, "Fusion option did not reach the simulator");
    }
    bool rejected = false;
    try
    {
        ParseConfig(json::object{{"backend", "qcsim"}, {"options", json::object{{"gate_fusion", "false"}}}});
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    Check(rejected, "Non-boolean fusion option accepted");
}

void TestFusionPublicInterfaces()
{
    using namespace MaestroExecution;
    auto *maestro = static_cast<Maestro *>(GetMaestroObjectWithMute());
    const auto backend = static_cast<int>(Simulators::SimulatorType::kQCSim);
    const char *circuit = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; rz(0.17) q[0]; measure "
                          "q->c;";
    for (const bool estimate : {false, true})
    {
        const auto handle = CreateSimpleSimulator(2);
        RemoveAllOptimizationSimulatorsAndAdd(handle, backend, 0);
        const auto network = maestro->GetSimpleSimulator(handle);
        network->SetOptimizeSimulator(false);
        for (const char *setting : {"false", "true", "false"})
        {
            const std::string config = std::string("{\"shots\":2,\"gate_fusion\":") + setting + "}";
            char *output = estimate ? SimpleEstimate(handle, circuit, "ZI", config.c_str()) : SimpleExecute(handle, circuit, config.c_str());
            Check(output != nullptr, "Legacy fusion execution failed");
            FreeResult(output);
            Check(network->GetSimulator()->IsGateFusionEnabled() == (std::string(setting) == "true"), "Legacy JSON fusion setting was ignored");
        }
        char *invalid =
            estimate ? SimpleEstimate(handle, circuit, "ZI", "{\"gate_fusion\":\"false\"}") : SimpleExecute(handle, circuit, "{\"gate_fusion\":\"false\"}");
        Check(!invalid, "Legacy API accepted a nonboolean fusion setting");
        DestroySimpleSimulator(handle);
    }
    for (int width : {1, 2, 3})
    {
        const auto handle = CreateSimulator(backend, 0);
        void *sim = GetSimulator(handle);
        AllocateQubits(sim, 3);
        InitializeSimulator(sim);
        // A three-qubit statevector is below the default fusion threshold; an
        // explicit setting turns fusion on.
        Check(GetGateFusionMaxQubits(sim) == 3 && !IsGateFusionEnabled(sim), "C default fusion threshold");
        Check(ConfigureSimulator(sim, "gate_fusion", "true") && IsGateFusionEnabled(sim), "C fusion capability");
        ApplyH(sim, 2);
        ApplyH(sim, 2);
        MaestroGateFusionStatistics stats{};
        Check(GetGateFusionStatistics(sim, &stats) && stats.submittedGates == 2 && stats.backendGates == 0, "C statistics query must not flush");
        double matrix[128]{};
        const auto dim = 1 << width;
        for (int col = 0; col < dim; ++col)
            matrix[2 * (((col + 1) % dim) * dim + col) + 1] = 1;
        const int ok = width == 1   ? ApplyGenericOneQubitGate(sim, 2, matrix)
                       : width == 2 ? ApplyGenericTwoQubitGate(sim, 2, 0, matrix)
                                    : ApplyGenericThreeQubitGate(sim, 2, 0, 1, matrix);
        Check(ok == 1, "C matrix submission");
        matrix[2 * dim + 1] = 0;
        double *amplitude = Amplitude(sim, 4);
        Check(amplitude && std::abs(amplitude[0]) < 1e-12 && std::abs(amplitude[1] - 1) < 1e-12, "C matrix layout/order/ownership");
        FreeDoubleVector(amplitude);
        Check(!ApplyGenericOneQubitGate(sim, 3, matrix), "C invalid target");
        Check(!ApplyGenericTwoQubitGate(sim, 0, 0, matrix), "C repeated target");
        Check(!ApplyGenericOneQubitGate(sim, 0, nullptr), "C null matrix");
        matrix[0] = std::numeric_limits<double>::quiet_NaN();
        Check(!ApplyGenericOneQubitGate(sim, 0, matrix), "C nonfinite matrix");
        Check(ConfigureSimulator(sim, "gate_fusion", "false") && !IsGateFusionEnabled(sim), "C fusion toggle");
        DestroySimulator(handle);
    }
}

void TestVirtualRoutingConfiguration()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    Simulators::Configuration native;
    Check(Simulators::RoutingConfig::GetStrategy(native) == Simulators::RoutingConfig::Strategy::Legacy, "Wrong default routing strategy");
    native.SetConfiguration("mps_routing_strategy", "legacy");
    Check(Simulators::RoutingConfig::GetStrategy(native) == Simulators::RoutingConfig::Strategy::Legacy, "Legacy strategy parsed incorrectly");
    for (const char *invalid : {"", "typo", "Combined", "current"})
    {
        bool rejected = false;
        try
        {
            native.SetConfiguration("mps_routing_strategy", invalid);
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        Check(rejected && native.GetConfiguration("mps_routing_strategy") == "legacy", "Invalid strategy changed native configuration");
    }
    char *catalogText = MaestroGetCapabilitiesJson();
    Check(catalogText != nullptr, "Cannot inspect routing capabilities");
    const auto catalog = json::parse(catalogText);
    FreeResult(catalogText);
    bool strategyFound = false;
    for (const auto &option : catalog.at("options").as_array())
        if (option.at("name") == "mps_routing_strategy")
        {
            strategyFound = true;
            Check(option.at("enum") == json::array{"combined", "legacy"} && option.at("default") == "legacy", "Incorrect routing strategy discovery");
            const auto &supported = option.at("supported_configurations").as_array();
            Check(supported.size() == 2, "Routing strategy advertised unsupported configurations");
            for (const auto &item : supported)
                Check(item.at("backend") == "qcsim" && (item.at("method") == "matrix_product_state" || item.at("method") == "matrix_product_operator"),
                      "Routing strategy advertised an untested backend");
        }
    Check(strategyFound, "Routing strategy missing from capabilities");
    for (const char *method : {"matrix_product_state", "matrix_product_operator"})
    {
        auto config = ParseConfig(json::object{{"backend", "qcsim"},
                                               {"method", method},
                                               {"options", json::object{{"mps_routing_virtual_bonds", true},
                                                                        {"mps_routing_truncation_weight", 0.5},
                                                                        {"mps_routing_legacy_cap", false},
                                                                        {"mps_routing_strategy", "legacy"}}}});
        const auto handle = CreateSimpleSimulator(4);
        const auto network = ConfigureNetwork(handle, config);
        Check(network && network->GetSimulator()->GetConfiguration("mps_routing_virtual_bonds") == "true", "JSON virtual routing setting lost");
        Check(std::stod(network->GetSimulator()->GetConfiguration("mps_routing_truncation_weight")) == 0.5, "JSON routing weight lost");
        Check(network->GetSimulator()->GetConfiguration("mps_routing_strategy") == "legacy", "JSON routing strategy lost");
        config.mps_routing_strategy = "combined";
        config.mps_routing_virtual_bonds = false;
        config.mps_routing_truncation_weight = 0.;
        const auto updated = ConfigureNetwork(handle, config);
        Check(updated && updated->GetSimulator()->GetConfiguration("mps_routing_virtual_bonds") == "false", "Typed virtual routing override lost");
        Check(std::stod(updated->GetSimulator()->GetConfiguration("mps_routing_truncation_weight")) == 0., "Typed zero routing weight lost");
        Check(updated->GetSimulator()->GetConfiguration("mps_routing_strategy") == "combined", "Typed routing strategy override lost");
        for (bool estimate : {false, true})
        {
            const char *circuit = "OPENQASM 2.0; qreg q[4]; creg c[4]; h q[0]; measure q->c;";
            const char *options = "{\"shots\":2,\"mps_routing_virtual_bonds\":true,\"mps_routing_truncation_weight\":0.25,"
                                  "\"mps_routing_strategy\":\"legacy\"}";
            char *result = estimate ? SimpleEstimate(handle, circuit, "ZIII", options) : SimpleExecute(handle, circuit, options);
            Check(result != nullptr, "Legacy routing configuration failed");
            FreeResult(result);
            Check(updated->GetSimulator()->GetConfiguration("mps_routing_virtual_bonds") == "true", "Legacy routing flag was ignored");
            Check(std::stod(updated->GetSimulator()->GetConfiguration("mps_routing_truncation_weight")) == 0.25, "Legacy routing weight was ignored");
            Check(updated->GetSimulator()->GetConfiguration("mps_routing_strategy") == "legacy", "Legacy routing strategy was ignored");
            for (const char *invalid : {"{\"mps_routing_strategy\":\"typo\"}", "{\"mps_routing_strategy\":\"current\"}", "{\"mps_routing_strategy\":true}"})
            {
                char *bad = estimate ? SimpleEstimate(handle, circuit, "ZIII", invalid) : SimpleExecute(handle, circuit, invalid);
                const bool rejected = bad == nullptr;
                if (bad)
                    FreeResult(bad);
                Check(rejected && updated->GetSimulator()->GetConfiguration("mps_routing_strategy") == "legacy",
                      "Invalid legacy strategy changed routing configuration");
            }
            updated->GetSimulator()->Configure("mps_routing_strategy", "combined");
            result = estimate ? SimpleEstimate(handle, circuit, "ZIII", "{\"shots\":2}") : SimpleExecute(handle, circuit, "{\"shots\":2}");
            Check(result != nullptr, "Legacy routing default failed");
            FreeResult(result);
            Check(updated->GetSimulator()->GetConfiguration("mps_routing_strategy") == "legacy", "Missing routing strategy did not restore legacy default");
        }
        DestroySimpleSimulator(handle);
    }
    for (const auto &options :
         {json::object{{"mps_routing_virtual_bonds", "true"}}, json::object{{"mps_routing_truncation_weight", -1.}},
          json::object{{"mps_routing_strategy", "typo"}}, json::object{{"mps_routing_strategy", "current"}}, json::object{{"mps_routing_strategy", true}}})
    {
        bool rejected = false;
        try
        {
            ParseConfig(json::object{{"backend", "qcsim"}, {"method", "matrix_product_state"}, {"options", options}});
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        Check(rejected, "Invalid routing setting accepted");
    }
    for (const auto &backendMethod : {std::pair{"qcsim", "statevector"}, std::pair{"gpu", "matrix_product_state"}, std::pair{"gpu", "matrix_product_operator"}})
    {
        bool rejected = false;
        try
        {
            ParseConfig(json::object{
                {"backend", backendMethod.first}, {"method", backendMethod.second}, {"options", json::object{{"mps_routing_strategy", "combined"}}}});
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        Check(rejected, "Routing strategy accepted for an unsupported backend/method");
    }
    for (const char *invalid : {"typo", "current"})
    {
        SimulatorConfig invalidStrategy;
        invalidStrategy.mps_routing_strategy = invalid;
        bool strategyRejected = false;
        try
        {
            invalidStrategy.Validate();
        }
        catch (const std::invalid_argument &)
        {
            strategyRejected = true;
        }
        Check(strategyRejected, "Invalid typed routing strategy accepted");
    }
    for (double weight : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        SimulatorConfig config;
        config.mps_routing_truncation_weight = weight;
        bool rejected = false;
        try
        {
            config.Validate();
        }
        catch (const std::invalid_argument &)
        {
            rejected = true;
        }
        Check(rejected, "Invalid typed routing weight accepted");
    }
}

// The network simulator only selects the backend when the network executes
// (INetwork::CreateSimulator). Configuring, executing, re-creating, cloning
// and dropping options must not allocate a dense register of the whole network
// that execution then discards. A caller-sized register keeps its exact width.
void TestCandidateWorkerBudget()
{
    using namespace MaestroExecution;
    // A lone density-matrix candidate needs 256 MiB at 12 qubits, the
    // requested statevector 64 KiB: realization workers are sized for the
    // backend that may run.
    SimulatorConfig config;
    config.simulator_type = Backend::kQCSim;
    config.simulation_type = Method::kStatevector;
    config.fixed_backend = false;
    config.optimization_candidates = {{Backend::kQCSim, Method::kDensityMatrix}};
    config.trajectory_max_memory_mb = 600;
    config.max_simulators = 8;
    Check(SamplingWorkers(config, 8, 12) == 1, "Realization workers were sized for the requested method, not the candidate");
    config.optimization_candidates.clear();
    if (Utils::RealizationWorkerCount(8, 8) > 1)
        Check(SamplingWorkers(config, 8, 12) > 1, "A small statevector lost its realization workers");
}

void TestNetworkSimulatorPlaceholders()
{
    using namespace MaestroExecution;
    using Backend = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
    using CF = Circuits::CircuitFactory<>;
    using Gate = Circuits::QuantumGateType;
    constexpr size_t placeholder = Network::INetwork<>::kBackendPlaceholderQubits;
    constexpr int width = 6;
    GetMaestroObjectWithMute();

    struct NetworkHandle
    {
        unsigned long handle;

        explicit NetworkHandle(int qubits) : handle(CreateSimpleSimulator(qubits))
        {
        }

        ~NetworkHandle()
        {
            DestroySimpleSimulator(handle);
        }
    };

    const auto measured = [] {
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
        circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
        circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 4));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}, {4, 2}}));
        return circuit;
    };
    const auto unmeasured = [] {
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 2));
        return circuit;
    };
    const auto checkCounts = [](const Circuits::Circuit<>::ExecuteResults &counts, size_t shots, const char *message) {
        size_t total = 0;
        for (const auto &[bits, count] : counts)
        {
            Check(bits.size() >= 3 && bits[0] && !bits[1] && bits[2], message);
            total += count;
        }
        Check(total == shots, message);
    };
    const auto keeps = [](const std::shared_ptr<Network::INetwork<>> &network, Backend type, Method method, size_t qubits, const char *message) {
        const auto simulator = network->GetSimulator();
        Check(simulator && simulator->GetType() == type && simulator->GetSimulationType() == method && simulator->GetNumberOfQubits() == qubits, message);
    };

    for (const auto method : {Method::kStatevector, Method::kDensityMatrix})
        for (size_t workers : {size_t{1}, size_t{4}})
        {
            SimulatorConfig config;
            config.simulator_type = Backend::kQCSim;
            config.simulation_type = method;
            config.fixed_backend = true;
            config.seed = 7;
            config.max_simulators = workers;

            // A fixed backend for network execution: a placeholder, kept by
            // every execution and re-creation that follows.
            NetworkHandle owner(width);
            const auto network = ConfigureNetwork(owner.handle, config);
            keeps(network, Backend::kQCSim, method, placeholder, "A fixed backend allocated a register for network execution");
            checkCounts(network->RepeatedExecuteOnHost(measured(), 0, 64), 64, "Placeholder network sampled wrong counts");
            keeps(network, Backend::kQCSim, method, placeholder, "Sampling re-created a full-width simulator");
            const auto values = network->ExecuteOnHostExpectations(unmeasured(), 0, {"Z", "IZ", "IIZ"});
            Check(values.size() == 3 && std::abs(values[0] + 1) < 1e-12 && std::abs(values[1] - 1) < 1e-12 && std::abs(values[2] + 1) < 1e-12,
                  "Placeholder network returned wrong expectations");
            keeps(network, Backend::kQCSim, method, placeholder, "An estimate re-created a full-width simulator");
            if (method == Method::kStatevector)
            {
                const auto amplitudes = network->ExecuteOnHostAmplitudes(unmeasured(), 0);
                Check(amplitudes.size() == 4 && std::abs(amplitudes[3] - std::complex<double>(1)) < 1e-12, "Placeholder network returned wrong amplitudes");
                keeps(network, Backend::kQCSim, method, placeholder, "An amplitude query re-created a full-width simulator");
            }
            keeps(network->Clone(), Backend::kQCSim, method, placeholder, "A clone of a placeholder network allocated a register");

            // A caller-sized register stays exactly that wide, without the
            // network's entanglement qubit, when executions re-create it.
            NetworkHandle exact(width);
            const auto direct = ConfigureNetwork(exact.handle, config, width);
            keeps(direct, Backend::kQCSim, method, width, "A caller-sized register has the wrong width");
            direct->ExecuteOnHostExpectations(unmeasured(), 0, {"Z"});
            keeps(direct, Backend::kQCSim, method, width, "An estimate re-created a caller-sized register at another width");
            checkCounts(direct->RepeatedExecuteOnHost(measured(), 0, 16), 16, "Caller-sized network sampled wrong counts");
            keeps(direct, Backend::kQCSim, method, width, "Sampling re-created a caller-sized register at another width");
            direct->Configure("trajectory_max_memory_mb", "64");
            direct->RemoveConfiguration("trajectory_max_memory_mb");
            keeps(direct, Backend::kQCSim, method, width, "Dropping an option re-created a caller-sized register at another width");
        }

    // Clone() keeps whatever width the simulator was created with.
    {
        const auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{width}, std::vector<size_t>{width});
        network->CreateSimulator(Backend::kQCSim, Method::kDensityMatrix, placeholder);
        Check(network->GetSimulatorQubits() == placeholder, "The network forgot its simulator's requested width");
        keeps(network->Clone(), Backend::kQCSim, Method::kDensityMatrix, placeholder, "Clone() grew a placeholder");
        network->CreateSimulator(Backend::kQCSim, Method::kStatevector, width);
        keeps(network->Clone(), Backend::kQCSim, Method::kStatevector, width, "Clone() changed a caller-sized register");
        network->CreateSimulator();
        keeps(network->Clone(), Backend::kQCSim, Method::kMatrixProductState, width + 1, "Clone() changed the default simulator's width");
    }

    // Candidate lists: one candidate needs no probe; several are probed at the
    // host's register (not the entanglement qubit) and leave a placeholder of
    // the first that fits, which then executes.
    using Candidates = std::vector<std::pair<Backend, Method>>;
    for (const auto &candidates : {Candidates{{Backend::kQCSim, Method::kDensityMatrix}},
                                   Candidates{{Backend::kQCSim, Method::kDensityMatrix}, {Backend::kQCSim, Method::kMatrixProductOperator}},
                                   Candidates{{Backend::kQCSim, Method::kStatevector}, {Backend::kQCSim, Method::kMatrixProductState}}})
    {
        SimulatorConfig config;
        config.optimization_candidates = candidates;
        config.seed = 11;
        NetworkHandle owner(width);
        const auto network = ConfigureNetwork(owner.handle, config);
        const auto [type, method] = candidates.front();
        keeps(network, type, method, placeholder, "A candidate list allocated a register for network execution");
        checkCounts(network->RepeatedExecuteOnHost(measured(), 0, 32), 32, "Candidate network sampled wrong counts");
        // Composer's cost model ranks lists with pure-state candidates; one
        // candidate, or only mixed-state ones, are resolved without it.
#ifdef MAESTRO_ENABLE_COMPOSER
        const bool ranked = std::any_of(candidates.begin(), candidates.end(),
                                        [](const auto &candidate) {
                                            return candidate.second != Method::kDensityMatrix && candidate.second != Method::kMatrixProductOperator;
                                        }) &&
                            candidates.size() > 1;
#else
        const bool ranked = false;
#endif
        Check(ranked || network->GetLastSimulationType() == method, "Candidate network did not run the first fitting candidate");
        NetworkHandle exact(width);
        keeps(ConfigureNetwork(exact.handle, config, width), type, method, width, "A caller-sized candidate has the wrong width");
    }

    // Every CPU backend runs a one-qubit circuit (the register a one-qubit
    // circuit maps to), through the network and directly.
    Candidates backends{{Backend::kQCSim, Method::kStatevector},     {Backend::kQCSim, Method::kMatrixProductState},
                        {Backend::kQCSim, Method::kStabilizer},      {Backend::kQCSim, Method::kTensorNetwork},
                        {Backend::kQCSim, Method::kPauliPropagator}, {Backend::kQCSim, Method::kExtendedStabilizer},
                        {Backend::kQCSim, Method::kDensityMatrix},   {Backend::kQCSim, Method::kMatrixProductOperator},
                        {Backend::kQCSim, Method::kPathIntegral},    {Backend::kCompositeQCSim, Method::kStatevector}};
#ifndef NO_QISKIT_AER
    for (const auto method :
         {Method::kStatevector, Method::kMatrixProductState, Method::kStabilizer, Method::kTensorNetwork, Method::kExtendedStabilizer, Method::kDensityMatrix})
        backends.emplace_back(Backend::kQiskitAer, method);
    backends.emplace_back(Backend::kCompositeQiskitAer, Method::kStatevector);
#endif
    std::string failures;
    for (const auto &[type, method] : backends)
    {
        const auto name = std::to_string(static_cast<int>(type)) + "/" + std::to_string(static_cast<int>(method));
        try
        {
            SimulatorConfig config;
            config.simulator_type = type;
            config.simulation_type = method;
            config.fixed_backend = true;
            config.seed = 5;
            NetworkHandle owner(1);
            const auto network = ConfigureNetwork(owner.handle, config);
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
            const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 8);
            if (counts.size() != 1 || counts.begin()->second != 8 || !counts.begin()->first[0])
                failures += " " + name + ":counts";
            auto x = CF::CreateCircuit();
            x->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
            const auto values = network->ExecuteOnHostExpectations(x, 0, {"Z"});
            if (values.size() != 1 || std::abs(values[0] + 1) > 1e-9)
                failures += " " + name + ":expectation";

            auto simulator = Simulators::SimulatorsFactory::CreateSimulator(type, method);
            simulator->AllocateQubits(1);
            simulator->Initialize();
            simulator->ApplyX(0);
            if (std::abs(simulator->ExpectationValue("Z") + 1) > 1e-9)
                failures += " " + name + ":direct";
        }
        catch (const std::exception &e)
        {
            failures += " " + name + ":" + e.what();
        }
    }
    if (!failures.empty())
        std::cerr << "One-qubit backend failures:" << failures << '\n';
    Check(failures.empty(), "A CPU backend rejects a one-qubit register");
}
