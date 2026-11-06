// Count actual evolution work, independently of sampling metadata or timing.
#include "../Circuit/Factory.h"
#include "../Execution/NoisyEstimation.h"
#include "../Execution/PrefixCheckpoint.h"
#include "../Execution/TerminalSampling.h"

#include <atomic>
#include <cmath>

void Check(bool, const char *);

namespace
{
using CF = Circuits::CircuitFactory<>;
using Gate = Circuits::QuantumGateType;
using Method = Simulators::SimulationType;
using MaestroExecution::RecordCircuitPtr;

class CountedX : public Circuits::XGate<>
{
  public:
    CountedX(size_t qubit, std::shared_ptr<std::atomic<size_t>> calls) : XGate(qubit), calls(std::move(calls))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
    {
        ++*calls;
        XGate::Execute(simulator, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<CountedX>(*this);
    }

  private:
    std::shared_ptr<std::atomic<size_t>> calls;
};

MaestroExecution::SimulatorConfig Config(Method method)
{
    MaestroExecution::SimulatorConfig config;
    config.simulation_type = method;
    config.fixed_backend = true;
    config.optimize_circuit = false;
    config.gate_fusion = false;
    config.max_simulators = 1;
    config.max_bond_dimension = 16;
    config.seed = 123;
    return config;
}

std::shared_ptr<Simulators::ISimulator> Simulator(Method method)
{
    auto simulator = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, method);
    Check(simulator != nullptr, "Cannot create mixed-state simulator");
    simulator->Configure("gate_fusion", "false");
    simulator->AllocateQubits(2);
    simulator->Initialize();
    simulator->ApplyX(1);
    simulator->SetSeed(123);
    return simulator;
}

void PreparedStateContract(Method method)
{
    auto simulator = Simulator(method);
    simulator->SaveState();
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto terminal = CF::CreateCircuit();
    auto nested = CF::CreateCircuit();
    nested->AddOperation(std::make_shared<CountedX>(0, calls));
    terminal->AddOperation(nested);
    auto readout = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 3}, {1, 0}});
    readout->SetReadout({{0.0, 1.0}, {0.0, 0.0}});
    terminal->AddOperation(readout);
    terminal->AddOperation(CF::CreateMeasurement({{0, 2}}));
    Circuits::OperationState state(5);
    state.SetBit(1);
    state.SetBit(4);
    std::vector<bool> seen(67, false);
    const bool sampled =
        MaestroExecution::SamplePreparedTerminal(*terminal, simulator, seen.size(), state, 19, [&](size_t row, const Circuits::OperationState &sample) {
            Check(row < seen.size() && !seen[row], "Terminal sampler emitted duplicate or invalid row indices");
            seen[row] = true;
            Check(sample.GetAllBits() == std::vector<bool>({true, true, true, false, true}),
                  "Terminal sampler lost initial bits, mapping, or per-occurrence readout");
        });
    Check(sampled && calls->load() == 1, "Prepared-state sampler evolved the circuit more than once");
    Check(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "Terminal sampler omitted rows");
    simulator->RestoreState();
    Check(std::abs(simulator->Probability(2) - 1.0) < 1e-10, "Terminal sampler overwrote the saved prefix");

    auto dynamic = CF::CreateCircuit();
    dynamic->AddOperation(std::make_shared<CountedX>(0, calls));
    dynamic->AddOperation(CF::CreateMeasurement({{0, 0}}));
    dynamic->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    const auto before = state.GetAllBits();
    simulator->SetSeed(123);
    simulator->SetGatesCounter(17);
    const auto reference = Simulator(method);
    size_t writes = 0;
    Check(!MaestroExecution::SamplePreparedTerminal(*dynamic, simulator, 67, state, 19, [&](size_t, const Circuits::OperationState &) { ++writes; }),
          "Dynamic circuit incorrectly accepted terminal sampling");
    Check(writes == 0 && calls->load() == 1 && state.GetAllBits() == before && simulator->GetGatesCounter() == 17,
          "Rejected terminal sampling changed circuit or classical execution state");
    Check(std::abs(simulator->Probability(2) - 1.0) < 1e-10 && simulator->RandomUniform() == reference->RandomUniform(),
          "Rejected terminal sampling changed the quantum state or RNG");
}

void CheckpointEvolutionCounts(Method method)
{
    auto prefix_calls = std::make_shared<std::atomic<size_t>>(0);
    auto suffix_calls = std::make_shared<std::atomic<size_t>>(0);
    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(std::make_shared<CountedX>(1, prefix_calls));
    MaestroExecution::PrefixCheckpoint checkpoint(prefix, 2, Config(method));
    Check(prefix_calls->load() == 1, "Checkpoint prefix was not prepared exactly once");
    auto suffix = CF::CreateCircuit();
    suffix->AddOperation(std::make_shared<CountedX>(0, suffix_calls));
    suffix->AddOperation(CF::CreateMeasurement({{0, 2}, {1, 0}}));
    noise::NoiseModel model;
    model.set_bit_flip(0, 0.25);
    for (const size_t shots : {size_t{17}, size_t{521}})
        for (bool noisy : {false, true})
        {
            suffix_calls->store(0);
            const auto result = checkpoint.ExecuteSuffix(suffix, shots, noisy ? &model : nullptr, std::nullopt, 11, 0, true);
            Check(prefix_calls->load() == 1 && suffix_calls->load() == 1, "Checkpoint terminal work grew with shots");
            Check(result.realizations == (noisy ? 1 : 0) && result.width == 3, "Checkpoint automatic policy or classical width is wrong");
            for (size_t shot = 0; shot < shots; ++shot)
                Check(result.rows[3 * shot] && !result.rows[3 * shot + 1] && (noisy || result.rows[3 * shot + 2]),
                      "Repeated checkpoint calls lost the prefix or classical mapping");
        }
    suffix_calls->store(0);
    const auto explicit_count = checkpoint.ExecuteSuffix(suffix, 73, &model, 7, 11, 0, true);
    Check(suffix_calls->load() == 7 && explicit_count.realizations == 7, "Checkpoint ignored explicit realization count");

    // A reset inside a mixed-state suffix is deterministic, even after a gate.
    auto reset = CF::CreateCircuit();
    reset->AddOperation(std::make_shared<CountedX>(0, suffix_calls));
    reset->AddOperation(CF::CreateReset({0}));
    reset->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    suffix_calls->store(0);
    const auto reset_result = checkpoint.ExecuteSuffix(reset, 71, nullptr, std::nullopt, 11, 0, true);
    Check(suffix_calls->load() == 1, "Mixed-state checkpoint reset caused shot replay");
    for (size_t shot = 0; shot < 71; ++shot)
        Check(!reset_result.rows[2 * shot] && reset_result.rows[2 * shot + 1], "Checkpoint reset changed the unreset prefix qubit");

    auto after_measurement = std::make_shared<std::atomic<size_t>>(0);
    auto dynamic = CF::CreateCircuit();
    dynamic->AddOperation(std::make_shared<CountedX>(0, suffix_calls));
    dynamic->AddOperation(CF::CreateMeasurement({{0, 0}}));
    dynamic->AddOperation(std::make_shared<CountedX>(0, after_measurement));
    dynamic->AddOperation(CF::CreateMeasurement({{0, 1}}));
    suffix_calls->store(0);
    const auto dynamic_result = checkpoint.ExecuteSuffix(dynamic, 71, nullptr, std::nullopt, 11, 0, true);
    // The deterministic head evolves once on a copy; the branch runs per shot.
    Check(prefix_calls->load() == 1 && suffix_calls->load() == 1 && after_measurement->load() == 71,
          "Dynamic checkpoint suffix did not share its head and replay its branch per shot");
    for (size_t shot = 0; shot < 71; ++shot)
        Check(dynamic_result.rows[2 * shot] && !dynamic_result.rows[2 * shot + 1], "Dynamic checkpoint records lost measurement order");
    // The copy leaves the saved prefix intact for the next suffix.
    suffix_calls->store(0);
    const auto after_dynamic = checkpoint.ExecuteSuffix(suffix, 33, nullptr, std::nullopt, 11, 0, true);
    for (size_t shot = 0; shot < 33; ++shot)
        Check(after_dynamic.rows[3 * shot] && after_dynamic.rows[3 * shot + 2], "Shared suffix head replaced the saved prefix");
    // Seeded calls replay: the copy draws the stream the replay would have.
    const auto replay = checkpoint.ExecuteSuffix(dynamic, 71, nullptr, std::nullopt, 11, 0, true);
    Check(replay.rows == dynamic_result.rows, "Shared suffix head changed seeded rows");
}

void RawEvolutionCounts(Method method)
{
    auto prefix_calls = std::make_shared<std::atomic<size_t>>(0);
    auto suffix_calls = std::make_shared<std::atomic<size_t>>(0);
    const auto config = Config(method);
    for (const size_t shots : {size_t{19}, size_t{139}})
        for (bool reset : {false, true})
        {
            auto circuit = CF::CreateCircuit();
            circuit->AddOperation(std::make_shared<CountedX>(0, prefix_calls));
            if (reset)
                circuit->AddOperation(CF::CreateReset({0}));
            circuit->AddOperation(CF::CreateMeasurement({{0, 1}}));
            prefix_calls->store(0);
            const auto result = MaestroExecution::SampleCircuitRows(circuit, config, shots);
            Check(prefix_calls->load() == 1, "Terminal raw sampling repeated mixed-state evolution");
            for (size_t shot = 0; shot < shots; ++shot)
                Check(!result.rows[2 * shot] && result.rows[2 * shot + 1] == !reset, "Raw terminal mapping or reset is wrong");

            noise::NoiseModel noise;
            noise.set_bit_flip(0, 0.25);
            MaestroExecution::NoisyExecutionOptions options;
            options.rows = true;
            options.layer = MaestroExecution::NoiseLayer::kCombined;
            prefix_calls->store(0);
            const auto noisy = MaestroExecution::ExecuteNoisy(circuit, noise, config, shots, std::nullopt, 13, options);
            Check(prefix_calls->load() == 1 && noisy.realizations == 1, "Automatic noisy raw sampling repeated exact evolution");
        }

    auto dynamic = CF::CreateCircuit();
    dynamic->AddOperation(std::make_shared<CountedX>(0, prefix_calls));
    dynamic->AddOperation(CF::CreateMeasurement({{0, 0}}));
    dynamic->AddOperation(std::make_shared<CountedX>(0, suffix_calls));
    dynamic->AddOperation(CF::CreateMeasurement({{0, 1}}));
    prefix_calls->store(0);
    const auto result = MaestroExecution::SampleCircuitRows(dynamic, config, 79);
    Check(prefix_calls->load() == 1 && suffix_calls->load() == 79, "Dynamic raw sampling did not reuse only its deterministic prefix");
    for (size_t shot = 0; shot < 79; ++shot)
        Check(result.rows[2 * shot] && !result.rows[2 * shot + 1], "Dynamic raw sampling lost ordered measurement records");
}

void ArbitraryPureCheckpointReset()
{
    auto simulator = Simulator(Method::kStatevector);
    simulator->Reset();
    simulator->ApplyH(0);
    simulator->ApplyCX(0, 1);
    auto suffix = CF::CreateCircuit();
    suffix->AddOperation(CF::CreateReset({0}));
    suffix->AddOperation(CF::CreateMeasurement({{1, 0}}));
    Circuits::OperationState state(2);
    size_t writes = 0;
    Check(!MaestroExecution::SamplePreparedTerminal(*suffix, simulator, 64, state, 17, [&](size_t, const Circuits::OperationState &) { ++writes; }),
          "Initial suffix reset was incorrectly treated as deterministic on an arbitrary pure state");
    Check(writes == 0 && std::abs(simulator->Probability(0) - 0.5) < 1e-10 && std::abs(simulator->Probability(3) - 0.5) < 1e-10,
          "Rejecting pure-state suffix reset changed its entangled input");

    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    prefix->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 1));
    MaestroExecution::PrefixCheckpoint checkpoint(prefix, 2, Config(Method::kStatevector));
    const auto result = checkpoint.ExecuteSuffix(suffix, 1024, nullptr, std::nullopt, 17, 0, true);
    size_t ones = 0;
    for (size_t shot = 0; shot < 1024; ++shot)
        ones += result.rows[2 * shot];
    Check(ones > 400 && ones < 624, "Pure-state checkpoint reused one sampled reset branch for every shot");
}

void PureCheckpointTerminalSampling(Method method)
{
    // Statevector, stabilizer and MPS samplers leave the save slot alone, so a
    // terminal suffix is prepared once per realization there too.
    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    prefix->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 1));
    MaestroExecution::PrefixCheckpoint checkpoint(prefix, 2, Config(method));
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto suffix = CF::CreateCircuit();
    suffix->AddOperation(std::make_shared<CountedX>(0, calls));
    suffix->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    const auto result = checkpoint.ExecuteSuffix(suffix, 1024, nullptr, std::nullopt, 17, 0, true);
    Check(calls->load() == 1 && result.sampling_strategy == "terminal_v1", "Pure-state terminal checkpoint suffix replayed shots");
    size_t ones = 0;
    for (size_t shot = 0; shot < 1024; ++shot)
    {
        Check(result.rows[2 * shot] != result.rows[2 * shot + 1], "Pure-state terminal checkpoint lost the prefix correlations");
        ones += result.rows[2 * shot];
    }
    Check(ones > 400 && ones < 624, "Pure-state terminal checkpoint sampled one branch");
    // The saved Bell pair survives: undo it and every shot reads 00.
    auto undo = CF::CreateCircuit();
    undo->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 1));
    undo->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    undo->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    const auto clean = checkpoint.ExecuteSuffix(undo, 64, nullptr, std::nullopt, 17, 0, false);
    Check(clean.counts.size() == 1 && clean.counts.front().second == 64 && !clean.counts.front().first[0] && !clean.counts.front().first[1],
          "Pure-state terminal sampling overwrote the saved prefix");
}

void NestedTerminalMeasurementsAreSampled(Method method)
{
    // A nested circuit that ends by measuring is terminal once inlined.
    auto simulator = Simulator(method);
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto nested = CF::CreateCircuit();
    nested->AddOperation(std::make_shared<CountedX>(0, calls));
    nested->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    auto terminal = CF::CreateCircuit();
    terminal->AddOperation(nested);
    Circuits::OperationState state(2);
    size_t rows = 0;
    const bool sampled = MaestroExecution::SamplePreparedTerminal(*terminal, simulator, 41, state, 3, [&](size_t, const Circuits::OperationState &sample) {
        ++rows;
        Check(sample.GetAllBits() == std::vector<bool>({true, true}), "Inlined nested measurements read the wrong outcome");
    });
    Check(sampled && rows == 41 && calls->load() == 1, "A nested terminal measurement forced shot replay");
    Check(nested->GetOperations().size() == 2 && terminal->GetOperations().size() == 1, "Inlining modified the input circuit");
}

void FidelityPreparesResetTargets()
{
    // A reset to |1> on the fresh register prepares |1>: the ideal state too.
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateReset({0, 1}, {true, false}));
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 1));
    for (const auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kMatrixProductOperator})
    {
        const auto result = MaestroExecution::NoisyFidelity(circuit, noise::NoiseModel{}, std::nullopt, Config(method), 7);
        Check(std::abs(result.fidelity - 1.0) < 1e-10, "A reset to |1> made the noiseless fidelity zero");
    }
}

void CandidateListsResolveOneBackend()
{
    // Statevector requested with a sole density-matrix candidate: the explicit
    // simulator, the exact-channel policy and the metadata use the candidate.
    auto config = Config(Method::kStatevector);
    config.fixed_backend = false;
    config.optimization_candidates = {{Simulators::SimulatorType::kQCSim, Method::kDensityMatrix}};
    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    MaestroExecution::PrefixCheckpoint checkpoint(prefix, 1, config);
    auto suffix = CF::CreateCircuit();
    suffix->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(1.)));
    suffix->AddOperation(CF::CreateMeasurement({{0, 0}}));
    const auto result = checkpoint.ExecuteSuffix(suffix, 64, nullptr, std::nullopt, 5, 0, false);
    Check(result.method == Method::kDensityMatrix && result.counts.size() == 1 && !result.counts.front().first[0], "Checkpoint ignored its candidate list");
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    noise::NoiseModel model;
    model.set_bit_flip(0, 0.25);
    const auto fidelity = MaestroExecution::NoisyFidelity(circuit, model, std::nullopt, config, 5);
    Check(fidelity.realizations == 1 && std::abs(fidelity.fidelity - 0.75) < 1e-10, "Noisy fidelity ignored its candidate list");

    // Several candidates: the first listed that can hold the register.
    config.optimization_candidates = {{Simulators::SimulatorType::kQCSim, Method::kDensityMatrix},
                                      {Simulators::SimulatorType::kQCSim, Method::kMatrixProductOperator}};
    Check(MaestroExecution::ResolveCandidateBackend(config, 2).simulation_type == Method::kDensityMatrix, "A fitting first candidate was not chosen");
    Check(MaestroExecution::ResolveCandidateBackend(config, 40).simulation_type == Method::kMatrixProductOperator,
          "An impossible density matrix was chosen over a listed MPO");
}

void DeterministicPrefixNoise()
{
    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    noise::NoiseModel model;
    model.set_bit_flip(0, 0.25);
    MaestroExecution::PrefixCheckpoint checkpoint(prefix, 1, Config(Method::kDensityMatrix), &model);
    auto suffix = CF::CreateCircuit();
    suffix->AddOperation(CF::CreateMeasurement({{0, 0}}));
    const auto result = checkpoint.ExecuteSuffix(suffix, 4000, nullptr, std::nullopt, 9, 0, true);
    size_t ones = 0;
    for (size_t shot = 0; shot < 4000; ++shot)
        ones += result.rows[shot];
    Check(ones > 2850 && ones < 3150, "Exact prefix noise was not applied once");
    bool rejected = false;
    try
    {
        MaestroExecution::PrefixCheckpoint pure(prefix, 1, Config(Method::kStatevector), &model);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Check(rejected, "Sampled prefix noise was frozen into a checkpoint");

    // Crosstalk spectators beyond the register stay in |0> (a global phase),
    // as for suffixes: the prefix noise is trimmed, not rejected.
    noise::NoiseModel spectators;
    spectators.set_bit_flip(0, 0.1);
    spectators.set_crosstalk(0, 4, 0.3);
    for (const auto method : {Method::kDensityMatrix, Method::kMatrixProductOperator})
    {
        MaestroExecution::PrefixCheckpoint trimmed(prefix, 1, Config(method), &spectators);
        const auto counts = trimmed.ExecuteSuffix(suffix, 4000, nullptr, std::nullopt, 9, 0, true);
        size_t set = 0;
        for (size_t shot = 0; shot < 4000; ++shot)
            set += counts.rows[shot];
        Check(set > 3480 && set < 3720, "Prefix noise with outside spectators lost its channel");
    }
}

void HeadCopyCountsTheCheckpoint()
{
    // The checkpoint's live state and saved prefix count, as well as the copy
    // and its saved head: 4 density matrices of 11 qubits (256 MiB) exceed
    // 128 MiB, 4 of 10 qubits (64 MiB) fit.
    auto config = Config(Method::kDensityMatrix);
    config.trajectory_max_memory_mb = 128;
    Check(!MaestroExecution::HeadCopyFits(config, 11), "Head copy ignored the checkpoint's own states");
    Check(MaestroExecution::HeadCopyFits(config, 10), "Head copy rejected a register that fits");
}

void NestedChannelsLeaveTheIdealBaseline()
{
    auto inner = CF::CreateCircuit();
    inner->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    inner->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(1.)));
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(inner);
    const auto ideal = MaestroExecution::WithoutCircuitChannels(circuit);
    Check(circuit->HasQuantumChannels() && !ideal->HasQuantumChannels(), "Nested channels survived in the ideal circuit");
    Check(ideal->GetOperations().size() == 1 && ideal->GetOperations()[0]->GetType() == Circuits::OperationType::kComposite &&
              std::static_pointer_cast<Circuits::Circuit<>>(ideal->GetOperations()[0])->GetOperations().size() == 1,
          "Stripping nested channels lost the nested gate");
    Check(inner->GetOperations().size() == 2, "Stripping channels modified the input circuit");
    auto config = Config(Method::kDensityMatrix);
    config.auto_reduce = false; // this checks channel stripping, not cone reduction
    const auto estimate =
        MaestroExecution::EstimateNoisy(circuit, {"Z"}, noise::NoiseModel{}, std::nullopt, config, 5, MaestroExecution::NoiseLayer::kCombined);
    Check(std::abs(estimate.values[0] - 1.0) < 1e-10 && std::abs(estimate.ideal_values[0] + 1.0) < 1e-10,
          "Nested channel noise leaked into the ideal baseline");
}
} // namespace

void TestMixedStateExecution()
{
    for (const auto method : {Method::kDensityMatrix, Method::kMatrixProductOperator})
    {
        PreparedStateContract(method);
        CheckpointEvolutionCounts(method);
        RawEvolutionCounts(method);
    }
    ArbitraryPureCheckpointReset();
    for (const auto method : {Method::kStatevector, Method::kStabilizer, Method::kMatrixProductState})
        PureCheckpointTerminalSampling(method);
    for (const auto method : {Method::kStatevector, Method::kDensityMatrix})
        NestedTerminalMeasurementsAreSampled(method);
    NestedChannelsLeaveTheIdealBaseline();
    FidelityPreparesResetTargets();
    CandidateListsResolveOneBackend();
    DeterministicPrefixNoise();
    HeadCopyCountsTheCheckpoint();
}
