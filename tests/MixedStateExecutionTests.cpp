// Count actual evolution work, independently of sampling metadata or timing.
#include "../Circuit/Factory.h"
#include "../Execution/CircuitHelpers.h"
#include "../Execution/NoisyEstimation.h"
#include "../Execution/PrefixCheckpoint.h"
#include "../Execution/TerminalExpectation.h"
#include "../Execution/TerminalSampling.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <map>

void Check(bool, const char *);

namespace
{
using CF = Circuits::CircuitFactory<>;
using Gate = Circuits::QuantumGateType;
using Method = Simulators::SimulationType;
using MaestroExecution::RecordCircuitPtr;

void CheckUnitaryPreparation()
{
    struct UnsupportedGate : Circuits::XGate<>
    {
        Gate GetGateType() const override
        {
            return Gate::kNone;
        }
    };

    const auto x = CF::CreateGate(Gate::kXGateType, 0);
    const auto h = CF::CreateGate(Gate::kHadamardGateType, 1);
    const auto rejects = [&](const MaestroExecution::OperationPtr &op, const char *kind) {
        try
        {
            MaestroExecution::UnitaryPreparation(CF::CreateCircuit({x, op, h}), "inner_product");
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            Check(message.find("inner_product") != std::string::npos && message.find(kind) != std::string::npos,
                  "Preparation error did not identify the query and offending operation");
            return;
        }
        Check(false, "Unitary preparation accepted an unsupported operation between valid gates");
    };
    rejects(nullptr, "null operation");
    rejects(std::make_shared<UnsupportedGate>(), "gate");
    rejects(CF::CreateCircuit({x}), "composite");

    const auto circuit = CF::CreateCircuit({x, CF::CreateMeasurement({{0, 4}}), CF::CreateDelay(0, 1.0), CF::CreateNoOp(), h});
    const auto prepared = MaestroExecution::UnitaryPreparation(circuit, "mirror_fidelity");
    Check(Circuits::ComparableCircuit<>(*prepared) == *CF::CreateCircuit({x, h}),
          "Preparation changed gates around a terminal measurement and idle operations");
}

void CheckPreparationGateAdjoints()
{
    for (int i = 0; i < static_cast<int>(Gate::kNone); ++i)
    {
        const auto gate = CF::CreateGate(static_cast<Gate>(i), 2, 0, 1, 0.4, -0.7, 0.9, 0.3);
        const auto prepared = MaestroExecution::UnitaryPreparation(CF::CreateCircuit({gate}), "inner_product");
        Check(prepared->size() == 1, "Preparation dropped a reversible gate");
        const auto inverse = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(MaestroExecution::adjoint_gate((*prepared)[0]));
        Check(inverse && inverse->AffectedQubits() == gate->AffectedQubits(), "Adjoint changed the gate targets");
        Check(inverse->GetMatrix().isApprox(gate->GetMatrix().adjoint(), 1e-12), "Adjoint gate matrix differs from the conjugate transpose");
    }
}

// The entries of a count list come in no particular order: workers merge
// theirs as they finish, so two runs with equal counts can list them differently.
bool SameCounts(std::vector<std::pair<std::vector<bool>, size_t>> a, std::vector<std::pair<std::vector<bool>, size_t>> b)
{
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a == b;
}

class CountedX : public Circuits::XGate<>
{
  public:
    CountedX(size_t qubit, std::shared_ptr<std::atomic<size_t>> calls, std::shared_ptr<std::atomic<size_t>> routed = {})
        : XGate(qubit), calls(std::move(calls)), routed(std::move(routed))
    {
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state) const override
    {
        ++*calls;
        if (routed && simulator->IsRoutingLookaheadEnabled())
            ++*routed;
        XGate::Execute(simulator, state);
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<CountedX>(*this);
    }

  private:
    std::shared_ptr<std::atomic<size_t>> calls;
    std::shared_ptr<std::atomic<size_t>> routed;
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

void CheckRealizationFidelityIdeal()
{
    const auto config = Config(Method::kStatevector);
    const auto x = CF::CreateGate(Gate::kXGateType, 0);
    const auto noisy = CF::CreateCircuit({x});
    const auto readout = CF::CreateMeasurement({{0, 0}});
    const auto h = CF::CreateGate(Gate::kHadamardGateType, 0);
    const auto ideal = CF::CreateCircuit({CF::CreateCircuit({h, readout})});
    Check(std::abs(MaestroExecution::RealizationFidelity(ideal, ideal, config) - 1.) < 1e-12, "Nested terminal readout collapsed a fidelity operand");
    Check(std::abs(MaestroExecution::NoisyFidelity(ideal, noise::NoiseModel{}, 1, config, 5).fidelity - 1.) < 1e-12,
          "Noisy fidelity lost nested terminal readout");
    const auto channel = std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(1.));
    for (const auto &invalid : {CF::CreateCircuit({x, CF::CreateReset({0})}), CF::CreateCircuit({x, readout, x}), CF::CreateCircuit({channel}),
                                CF::CreateCircuit({CF::CreateSimpleConditionalGate(x, 0)}), CF::CreateCircuit({CF::CreateCircuit({nullptr})})})
    {
        try
        {
            MaestroExecution::RealizationFidelity(invalid, noisy, config);
        }
        catch (const std::invalid_argument &)
        {
            continue;
        }
        Check(false, "RealizationFidelity silently discarded a nonunitary operation from the ideal");
    }
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

void PathIntegralKrausBranches()
{
    auto simulator = Simulators::SimulatorsFactory::CreateSimulatorUnique(Simulators::SimulatorType::kQCSim, Method::kPathIntegral);
    simulator->AllocateQubits(7);
    simulator->Initialize();
    simulator->SetSeed(42);
    Check(simulator->SamplesQuantumChannels() && !simulator->SupportsQuantumChannels(), "Path integral reported the wrong channel capability");

    // Target order is {q6,q1}: local |01> becomes |10> on a jump.
    // q4 is entangled with q6 and remains set on the jumped branch.
    Eigen::Matrix4cd stay = Eigen::Matrix4cd::Identity(), jump = Eigen::Matrix4cd::Zero();
    stay(1, 1) = std::complex<double>(0., .6);
    jump(2, 1) = .8;
    const Simulators::QuantumChannel channel(Simulators::QuantumChannel::KrausOperators{stay, jump});
    simulator->ApplyH(4);
    simulator->ApplyCX(4, 6);
    simulator->SaveState();
    size_t jumps = 0;
    for (size_t shot = 0; shot < 1024; ++shot)
    {
        simulator->RestoreState();
        simulator->ApplyQuantumChannel({6, 1}, channel);
        Check(std::abs(simulator->TotalProbability() - 1.) < 1e-12, "A Kraus trajectory lost normalization");
        if (simulator->Probability(18) > .5)
        {
            ++jumps;
            Check(std::abs(simulator->Probability(18) - 1.) < 1e-12, "The jump lost the entangled spectator");
        }
        else
        {
            const auto zero = simulator->Amplitude(0);
            Check(std::abs(std::norm(zero) - 1. / 1.36) < 1e-12, "The surviving amplitudes have the wrong weights");
            Check(std::abs(simulator->Amplitude(80) - std::complex<double>(0., .6) * zero) < 1e-12, "The no-jump branch lost its relative phase");
        }
    }
    Check(std::abs(jumps / 1024. - .32) < .07, "Kraus jumps did not follow the incoming state");

    simulator->Reset();
    simulator->ApplyX(6);
    Eigen::Matrix4cd controlled = Eigen::Matrix4cd::Identity();
    controlled.row(1).swap(controlled.row(3));
    simulator->ApplyQuantumChannel({6, 1}, Simulators::QuantumChannel::UnitaryMixture({1.}, {controlled}));
    Check(std::abs(simulator->Probability(66) - 1.) < 1e-12, "A two-qubit unitary channel reversed its targets");

    simulator->Reset();
    simulator->Configure("path_integral_threshold", "0.3");
    simulator->ApplyRy(0, std::acos(-1.) / 3.);
    simulator->ApplyX(0);
    Check(std::abs(simulator->TotalProbability() - .75) < 1e-12, "Kraus pruning fixture lost its retained mass");
    simulator->SaveState();
    const auto damping = Simulators::QuantumChannel::AmplitudeDamping(.125);
    jumps = 0;
    for (size_t shot = 0; shot < 512; ++shot)
    {
        simulator->RestoreState();
        simulator->ApplyQuantumChannel({0}, damping);
        Check(std::abs(simulator->TotalProbability() - 1.) < 1e-12, "A sampled Kraus branch kept the pruned input norm");
        jumps += simulator->Probability(0) > .5;
    }
    Check(std::abs(jumps / 512. - .125) < .055, "A Kraus branch below the gate threshold was dropped");
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
    const std::vector<bool> expected{true, true, true, false, true};
    const bool sampled =
        MaestroExecution::SamplePreparedTerminal(*terminal, simulator, seen.size(), state, 19, [&](size_t row, const Circuits::OperationState &sample) {
            Check(row < seen.size() && !seen[row], "Terminal sampler emitted duplicate or invalid row indices");
            seen[row] = true;
            Check(sample.GetAllBits() == expected, "Terminal sampler lost initial bits, mapping, or per-occurrence readout");
        });
    Check(sampled && calls->load() == 1, "Prepared-state sampler evolved the circuit more than once");
    Check(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }), "Terminal sampler omitted rows");
    simulator->RestoreState();
    Check(std::abs(simulator->Probability(2) - 1.0) < 1e-10, "Terminal sampler overwrote the saved prefix");
    // A count sink takes the place of the rows. Readout errors are drawn per
    // shot, so it then receives the shots one at a time, never aggregated.
    size_t counted = 0;
    Check(MaestroExecution::SamplePreparedTerminal(
              *terminal, simulator, seen.size(), state, 19, [](size_t, const auto &) { Check(false, "A count sink still expanded rows"); }, nullptr, false,
              [&](size_t count, const Circuits::OperationState &sample) {
                  Check(count == 1, "Readout errors were aggregated across shots");
                  Check(sample.GetAllBits() == expected, "Counted shots lost initial bits, mapping, or per-occurrence readout");
                  ++counted;
              }) &&
              counted == seen.size() && calls->load() == 2,
          "A count sink did not receive every shot of a readout measurement once");
    simulator->RestoreState();

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
    Check(writes == 0 && calls->load() == 2 && state.GetAllBits() == before && simulator->GetGatesCounter() == 17,
          "Rejected terminal sampling changed circuit or classical execution state");
    Check(std::abs(simulator->Probability(2) - 1.0) < 1e-10 && simulator->RandomUniform() == reference->RandomUniform(),
          "Rejected terminal sampling changed the quantum state or RNG");
}

// A count sink skips the permutation of the shots, not the readout draws: with
// readout errors it gives the histogram of the rows for the same seed.
void CountSinkKeepsReadoutDraws(Method method)
{
    constexpr size_t shots = 3000;
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    auto readout = std::make_shared<Circuits::MeasurementOperation<>>(std::vector<std::pair<Types::qubit_t, size_t>>{{0, 0}, {1, 1}});
    readout->SetReadout({{0.2, 0.1}, {0.05, 0.3}});
    circuit->AddOperation(readout);
    const auto histogram = [&](bool counts) {
        auto simulator = Simulator(method);
        Circuits::OperationState state(2);
        std::map<std::vector<bool>, size_t> result;
        size_t rows = 0, shotsCounted = 0;
        MaestroExecution::TerminalCountSink sink;
        if (counts)
            sink = [&](size_t count, const Circuits::OperationState &sample) {
                Check(count == 1, "Readout errors were aggregated across shots");
                ++result[sample.GetAllBits()];
                ++shotsCounted;
            };
        Check(MaestroExecution::SamplePreparedTerminal(
                  *circuit, simulator, shots, state, 42,
                  [&](size_t, const Circuits::OperationState &sample) {
                      ++result[sample.GetAllBits()];
                      ++rows;
                  },
                  nullptr, false, sink),
              "A terminal readout measurement was not sampled from the prepared state");
        Check(counts ? rows == 0 && shotsCounted == shots : rows == shots, "A terminal readout measurement used the wrong sink");
        return result;
    };
    const auto rows = histogram(false);
    Check(rows.size() == 4 && rows == histogram(true), "A count sink changed the seeded readout histogram");
}

void CheckpointMultinomialCounts()
{
    constexpr size_t shots = 4096;
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(CF::CreateMeasurement({{0, 4}, {0, 1}}));
    auto simulator = Simulator(Method::kStatevector);
    simulator->Configure("counts_policy", "multinomial_v1");
    Circuits::OperationState state(6);
    state.SetBit(3); // unwritten bits keep their initial values
    size_t total = 0, bins = 0;
    Check(MaestroExecution::SamplePreparedTerminal(
              *circuit, simulator, shots, state, 42, [](size_t, const auto &) { Check(false, "Count-only terminal measurement expanded individual rows"); },
              nullptr, false,
              [&](size_t count, const Circuits::OperationState &sample) {
                  const auto &bits = sample.GetAllBits();
                  Check(bits.size() == 6 && bits[3] && !bits[0] && !bits[2] && !bits[5] && bits[1] == bits[4], "Aggregated terminal bit mapping");
                  total += count;
                  ++bins;
              }),
          "Terminal measurements could not be aggregated");
    Check(total == shots && bins == 2, "Expected one write per distinct terminal outcome");

    auto config = Config(Method::kStatevector);
    config.seed.reset();
    const auto prefix = CF::CreateCircuit();
    const auto run = [&](MaestroExecution::PrefixCheckpoint &checkpoint, std::optional<uint64_t> seed, bool rows) {
        auto result = checkpoint.ExecuteSuffix(circuit, shots, nullptr, std::nullopt, seed, 6, rows);
        std::sort(result.counts.begin(), result.counts.end());
        return result;
    };
    config.counts_policy = "multinomial_v1";
    MaestroExecution::PrefixCheckpoint adaptive(prefix, 2, config);
    const auto expectedCounts = run(adaptive, 42, false).counts;
    const auto expectedRows = run(adaptive, 42, true).rows;
    for (const std::string policy : {"auto", "multinomial_v1"})
    {
        config.counts_policy = policy;
        MaestroExecution::PrefixCheckpoint checkpoint(prefix, 2, config);
        const auto first = run(checkpoint, std::nullopt, false);
        Check(first.width == 6 && first.counts.size() == 2 && first.rows.empty(), "Checkpoint count padding or storage changed");
        const auto seeded = run(checkpoint, 42, false).counts;
        Check(SameCounts(seeded, expectedCounts), "A prior unseeded checkpoint call changed the seeded adaptive histogram");
        Check(run(checkpoint, 42, true).rows == expectedRows, "Adaptive checkpoint raw rows did not replay");
        Check(SameCounts(run(checkpoint, 42, false).counts, seeded), "Count replay changed after a raw checkpoint call");
    }
    config.counts_policy = "auto";
    config.seed = 0;
    MaestroExecution::PrefixCheckpoint typedSeed(prefix, 2, config);
    config.seed.reset();
    config.native_options["seed"] = "0";
    config.native_options["counts_policy"] = "auto";
    MaestroExecution::PrefixCheckpoint nativeSeed(prefix, 2, config);
    Check(SameCounts(run(typedSeed, std::nullopt, false).counts, run(nativeSeed, std::nullopt, false).counts),
          "Native seed zero lost its checkpoint count stream");
    Check(run(typedSeed, std::nullopt, true).rows == run(nativeSeed, std::nullopt, true).rows, "Native seed zero lost its checkpoint row stream");
    // The small batches use the native fallback kernels. Both they and the
    // multinomial kernels must leave the same saved prefix for the next suffix.
    auto x = CF::CreateCircuit();
    x->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    auto measure = CF::CreateCircuit();
    measure->AddOperation(CF::CreateMeasurement({{0, 0}}));
    auto undo = CF::CreateCircuit();
    undo->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    undo->AddOperation(CF::CreateMeasurement({{0, 0}}));
    for (auto method : {Method::kPathIntegral, Method::kExtendedStabilizer})
    {
        auto fixed = Config(method);
        fixed.counts_policy = "multinomial_v1";
        MaestroExecution::PrefixCheckpoint checkpoint(x, 1, fixed);
        for (const size_t batch : {size_t{17}, shots})
            for (const bool inverted : {true, false})
            {
                const auto result = checkpoint.ExecuteSuffix(inverted ? undo : measure, batch, nullptr, std::nullopt, 42, 1, false);
                Check(result.sampling_strategy == "terminal_v1" && result.counts.size() == 1 &&
                          result.counts.front() == std::make_pair(std::vector<bool>{!inverted}, batch),
                      "Prepared count sampling overwrote its checkpoint or replayed the suffix");
            }
        Check(checkpoint.ExecuteSuffix(measure, 17, nullptr, std::nullopt, 42, 1, true).sampling_strategy == "terminal_v1",
              "Safe checkpoint raw records replayed the terminal suffix");
    }
}

// sampling_policy is a native option, not a Python SimulatorConfig property.
// Exercise PrefixCheckpoint itself: specifying its default must keep adaptive
// terminal sampling for both counts and ordered rows on the same checkpoint.
void ExplicitSamplingPolicyKeepsAdaptiveCheckpoint(Method method)
{
    constexpr size_t shots = 4096;
    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    auto suffix = CF::CreateCircuit();
    suffix->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
    auto config = Config(method);
    config.counts_policy = "auto";
    config.seed = 0;
    MaestroExecution::PrefixCheckpoint implicit(prefix, 2, config);
    config.native_options["sampling_policy"] = "reproducible_v1";
    Check(MaestroExecution::PreferMultinomialCounts(config), "Explicit reproducible sampling disabled adaptive counts");
    MaestroExecution::PrefixCheckpoint explicitDefault(prefix, 2, config);
    const auto run = [&](MaestroExecution::PrefixCheckpoint &checkpoint, bool rows) {
        auto result = checkpoint.ExecuteSuffix(suffix, shots, nullptr, std::nullopt, 0, 2, rows);
        Check(result.sampling_strategy == "terminal_v1", "Explicit reproducible checkpoint replayed its terminal suffix");
        std::sort(result.counts.begin(), result.counts.end());
        return result;
    };
    const auto expectedCounts = run(implicit, false).counts;
    const auto expectedRows = run(implicit, true).rows;
    for (size_t repeat = 0; repeat < 2; ++repeat)
    {
        Check(SameCounts(run(explicitDefault, false).counts, expectedCounts), "Explicit reproducible checkpoint changed the adaptive histogram");
        Check(run(explicitDefault, true).rows == expectedRows, "Explicit reproducible checkpoint changed adaptive ordered rows");
    }
}

void CheckpointTensorRouting(Method method, const Types::qubits_vector &path)
{
    const auto spare = path.size(); // first used by a suffix
    const size_t width = spare + 1;
    auto edges = CF::CreateCircuit();
    for (size_t parity : {0, 1})
        for (size_t i = parity; i + 1 < path.size(); i += 2)
            edges->AddOperation(CF::CreateGate(Gate::kCZGateType, path[i], path[i + 1]));
    auto rotations = CF::CreateCircuit();
    for (const auto q : path)
        rotations->AddOperation(CF::CreateGate(Gate::kHadamardGateType, q));
    auto prefix = CF::CreateCircuit();
    prefix->AddOperation(CF::CreateGate(Gate::kXGateType, path.front()));
    prefix->AddOperation(rotations);
    prefix->AddOperation(edges);
    auto end = CF::CreateCircuit();
    end->AddOperation(CF::CreateMeasurement({{path.front(), 0}}));
    prefix->AddOperation(end); // The saved graph must keep the coherence this terminal measurement would destroy.

    auto undo = CF::CreateCircuit();
    undo->AddOperation(edges); // The CZs commute and each is its own inverse.
    undo->AddOperation(rotations);
    auto terminal = CF::CreateCircuit();
    terminal->AddOperation(undo);
    for (size_t i = 0; i < 10; ++i) // A longer suffix qualifies for automatic lookahead.
        terminal->AddOperation(CF::CreateGate(Gate::kCXGateType, path.front(), path.back()));
    auto calls = std::make_shared<std::atomic<size_t>>(0);
    auto routed = std::make_shared<std::atomic<size_t>>(0);
    terminal->AddOperation(std::make_shared<CountedX>(spare, calls, routed));
    terminal->AddOperation(CF::CreateMeasurement({{spare, 0}, {path.back(), 1}, {path.front(), 2}}));

    auto dynamic = CF::CreateCircuit();
    dynamic->AddOperation(undo);
    dynamic->AddOperation(CF::CreateMeasurement({{path.front(), 3}}));
    dynamic->AddOperation(CF::CreateGate(Gate::kCXGateType, path.front(), spare));
    dynamic->AddOperation(CF::CreateMeasurement({{path.back(), 1}, {spare, 4}}));
    noise::NoiseModel readout;
    readout.set_readout_error(static_cast<int>(path.back()), 1., 1.);

    for (bool disabled : {false, true})
    {
        auto config = Config(method);
        config.max_bond_dimension = 64;
        config.max_simulators = 2;
        config.disable_optimized_swapping = disabled;
        MaestroExecution::PrefixCheckpoint checkpoint(prefix, static_cast<int>(width), config);
        // Placing the graph along the chain leaves one CZ across each cut. Its projector has the square of the pure-state rank.
        const size_t lineRank = method == Method::kMatrixProductState ? 2 : 4;
        Check(disabled ? checkpoint.MaxBondDimension() > lineRank : checkpoint.MaxBondDimension() == lineRank,
              "Nested checkpoint prefix ignored the configured initial chain layout");
        const auto checkTerminal = [&] {
            calls->store(0);
            routed->store(0);
            const auto result = checkpoint.ExecuteSuffix(terminal, 3, nullptr, std::nullopt, 7, 0, false);
            std::vector<bool> expected(width, false);
            expected[0] = expected[2] = true;
            Check(result.sampling_strategy == "terminal_v1" && result.counts.size() == 1 && result.counts.front() == std::make_pair(expected, size_t{3}),
                  "Terminal checkpoint sampling lost logical qubits after restoring a routed prefix");
            Check(calls->load() == 1 && routed->load() == (disabled ? 0u : 1u), "Nested suffix used the wrong automatic swap lookahead");
        };
        checkTerminal();
        const auto rows = checkpoint.ExecuteSuffix(dynamic, 8, &readout, 4, 7, 0, true);
        Check(rows.realizations == 4 && rows.width == width && rows.rows.size() == 8 * width && rows.sampling_strategy == "trajectory_v1",
              "Dynamic checkpoint test did not sample four noisy realizations");
        for (size_t i = 0; i < rows.rows.size(); ++i)
        {
            const auto bit = i % width;
            Check(rows.rows[i] == (bit == 1 || bit == 3 || bit == 4), "A checkpoint worker lost a logical qubit or readout target across restore");
        }
        checkTerminal(); // The worker saves and the different suffix must leave the original prefix intact.
    }
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

void TerminalStatevectorExpectations(Simulators::SimulatorType backend)
{
    using Channel = Simulators::QuantumChannel;
    using namespace MaestroExecution;
    auto config = Config(Method::kStatevector);
    config.simulator_type = backend;
    const auto estimate = [&](const RecordCircuitPtr &circuit, const std::vector<std::string> &words, const noise::NoiseModel &model,
                              std::optional<size_t> requested = std::nullopt) {
        return EstimateNoisy(circuit, words, model, requested, config, 24, NoiseLayer::kCombined);
    };
    auto x = CF::CreateCircuit();
    x->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    noise::NoiseModel t1;
    t1.set_t1(0, 0.3);
    for (const std::optional<size_t> requested : {std::optional<size_t>{}, std::optional<size_t>{7}})
    {
        const auto result = estimate(x, {"Z", "z", "I"}, t1, requested);
        Check(result.terminal_channels && !result.policy.stochastic_noise && result.realizations == requested.value_or(1) && result.realization_workers == 1,
              "Terminal T1 did not resolve on one deterministic preparation");
        Check(std::abs(result.values[0] + 0.4) < 1e-12 && result.values[1] == result.values[0] && result.values[2] == 1. && result.ideal_values[0] == -1. &&
                  result.standard_errors[0] == 0.,
              "Terminal T1 lost its affine shift, observable order or ideal baseline");
    }

    // A state-independent noise model is normally drawn into gates. When it
    // acts only at the end the estimator can instead average the full channel.
    noise::NoiseModel flip;
    flip.set_bit_flip(0, 0.25);
    Check(std::abs(estimate(x, {"Z"}, flip).values[0] + 0.5) < 1e-12, "Terminal bit-flip model was drawn into one trajectory");

    // Non-Pauli Kraus on two asymmetrically ordered targets, followed by a
    // noncommuting channel. Short words must grow to include the other target.
    auto two = CF::CreateCircuit();
    two->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 2));
    two->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    Eigen::Matrix4cd stay = Eigen::Matrix4cd::Identity(), jump = Eigen::Matrix4cd::Zero();
    stay(3, 3) = 0.8;
    jump(1, 3) = 0.6;
    two->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{2, 0}, Channel(Channel::KrausOperators{stay, jump})));
    two->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{2}, Channel::GeneralizedAmplitudeDamping(0.31, 0.23)));
    const std::vector<std::string> words{"Z", "IIX", "IIZ", "ZIX", "XIZ", "III", "Z"};
    const auto reference = EstimateNoisy(two, words, {}, std::nullopt, Config(Method::kDensityMatrix), 91, NoiseLayer::kCombined);
    const auto result = estimate(two, words, {});
    Check(result.terminal_channels && result.realizations == 1, "Two-qubit terminal Kraus channel was sampled");
    for (size_t i = 0; i < words.size(); ++i)
        Check(std::abs(result.values[i] - reference.values[i]) < 1e-12, "Terminal Kraus adjoint disagreed with the density matrix");

    // Here the first damping is still sampled. The last one's conditional
    // result has two possible values; its standard error must be calculated
    // after combining the terms in Z -> (1-gamma) Z + gamma I.
    auto mixed = CF::CreateCircuit();
    mixed->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    mixed->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Channel::AmplitudeDamping(0.5)));
    mixed->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    mixed->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Channel::AmplitudeDamping(0.5)));
    const size_t samples = 40;
    const auto partial = estimate(mixed, {"Z"}, {}, samples);
    const double mean = partial.values[0];
    Check(partial.terminal_channels && partial.realizations == samples && mean > 0. && mean < 1. && partial.policy.stochastic_execution,
          "A sampled statevector prefix was treated as deterministic");
    Check(std::abs(partial.standard_errors[0] - std::sqrt(mean * (1. - mean) / (samples - 1))) < 1e-12,
          "Terminal conditional expectation has the wrong trajectory standard error");
    const auto repeated = estimate(mixed, {"Z"}, {}, samples);
    Check(repeated.values == partial.values && repeated.standard_errors == partial.standard_errors, "A seeded sampled prefix did not replay");
    const auto trajectories = EstimateNoisy(x, {"Z"}, t1, samples, config, 24, NoiseLayer::kCombined, false);
    Check(!trajectories.terminal_channels && trajectories.standard_errors[0] > 0., "Explicit trajectories were averaged at the terminal channel");

    // A rare Hadamard leaves an X expectation below the Pauli propagator's
    // coefficient snapping threshold. It is the entire signal here.
    const double rare = 4e-15;
    Eigen::Matrix2cd h;
    h << 1., 1., 1., -1.;
    h /= std::sqrt(2.);
    auto little = CF::CreateCircuit();
    const auto rare_h = Channel::UnitaryMixture({1. - rare, rare}, {Eigen::Matrix2cd::Identity(), h});
    little->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, rare_h.Then(Channel::PhaseFlip(0.25))));
    const auto weak = estimate(little, {"X"}, {});
    Check(weak.terminal_channels && std::abs(weak.values[0] - 0.5 * rare) < 1e-28, "A weak nonzero terminal expectation was discarded");

    // A phased X accepted as a Kraus operator can have squared norm 1 + one ulp.
    // In a composition its recognized Pauli factor keeps that rounding.
    Eigen::Matrix2cd phased = Eigen::Matrix2cd::Zero();
    phased(0, 1) = phased(1, 0) = std::complex<double>(1., std::ldexp(1., -26));
    const Channel phased_x(Channel::KrausOperators{phased});
    Check(phased_x.GetFactors().front().kind == Channel::FactorKind::kPauli, "A global phase prevented Pauli classification");
    auto phased_circuit = CF::CreateCircuit();
    phased_circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    phased_circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Channel::AmplitudeDamping(.25).Then(phased_x)));
    const auto phased_result = estimate(phased_circuit, {"Z"}, {});
    Check(phased_result.realizations == 1 && std::abs(phased_result.values[0] + .25) < 1e-14, "A roundoff-sized phased Pauli factor changed the estimate");

    // Placement preserves the given Kraus operators. A weak coherent rotation can
    // look like an identity Pauli to the factor classifier, but it produces X signal.
    constexpr double angle = 1e-12;
    Eigen::Matrix2cd ry;
    ry << std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2);
    auto placed = CF::CreateCircuit();
    placed->AddOperation(
        std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0, 1}, Channel::OnQubit(Channel(Channel::KrausOperators{ry}), 1)));
    placed->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{1}, Channel::PhaseFlip(.25)));
    const auto rotated = estimate(placed, {"IX"}, {});
    const auto rotated_dm = EstimateNoisy(placed, {"IX"}, {}, std::nullopt, Config(Method::kDensityMatrix), 24, NoiseLayer::kCombined);
    Check(rotated.realizations == 1 && std::abs(rotated.values[0] - .5 * std::sin(angle)) < 1e-24 && std::abs(rotated.values[0] - rotated_dm.values[0]) < 1e-24,
          "Placed raw Kraus classification discarded a coherent terminal signal");

    auto composed = CF::CreateCircuit();
    composed->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0},
                                                                                 Channel(Channel::KrausOperators{ry}).Then(Channel::AmplitudeDamping(.2))));
    const auto coherent = estimate(composed, {"X"}, {});
    const auto coherent_dm = EstimateNoisy(composed, {"X"}, {}, std::nullopt, Config(Method::kDensityMatrix), 24, NoiseLayer::kCombined);
    Check(std::abs(coherent.values[0] - std::sqrt(.8) * std::sin(angle)) < 1e-24 && std::abs(coherent.values[0] - coherent_dm.values[0]) < 1e-24,
          "Composition classified a weak coherent rotation as a Pauli mixture");

    constexpr double weak_probability = 1e-17;
    Eigen::Matrix2cd reset_plus = Eigen::Matrix2cd::Zero();
    reset_plus(0, 0) = reset_plus(1, 0) = std::sqrt(.5);
    for (bool unitary : {false, true})
    {
        const Channel raw(Channel::KrausOperators{Eigen::Matrix2cd::Identity(), std::sqrt(weak_probability) * (unitary ? h : reset_plus)});
        Check(raw.GetFactors().front().kind == (unitary ? Channel::FactorKind::kUnitaryMixture : Channel::FactorKind::kKraus),
              "A rare Kraus branch was classified without considering its own norm");
        auto rare_input = CF::CreateCircuit();
        rare_input->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, raw.Then(Channel::PhaseFlip(.25))));
        const auto rare_value = estimate(rare_input, {"X"}, {});
        Check(std::abs(rare_value.values[0] - .5 * weak_probability) < 1e-31, "Composition lost a nonzero raw Kraus branch");
    }

    auto large = CF::CreateCircuit();
    for (size_t q = 0; q < 9; ++q)
        large->AddOperation(CF::CreateGate(Gate::kXGateType, q));
    for (size_t q = 0; q < 9; ++q)
        large->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{q}, Channel::AmplitudeDamping(.2)));
    const auto seven = estimate(large, {"ZZZZZZZ"}, {});
    Check(seven.realizations == 1 && std::abs(seven.values[0] - std::pow(-.6, 7)) < 1e-12, "A modest deterministic damping expansion was sampled");
    const auto hybrid = estimate(large, {"Z", "ZZZZZZZZZ", "Z"}, {}, 15);
    const auto sampled = EstimateNoisy(large, {"ZZZZZZZZZ"}, {}, 15, config, 24, NoiseLayer::kCombined, false);
    Check(hybrid.terminal_channels && hybrid.realizations == 15 && std::abs(hybrid.values[0] + .6) < 1e-14 && hybrid.standard_errors[0] == 0. &&
              hybrid.values[2] == hybrid.values[0] && hybrid.standard_errors[2] == 0. && hybrid.values[1] == sampled.values[0] &&
              hybrid.standard_errors[1] == sampled.standard_errors[0],
          "A wide fallback row disturbed an exact damping estimate or its seeded fallback");

    auto stochastic = CF::CreateCircuit();
    auto prefix_calls = std::make_shared<std::atomic<size_t>>(0);
    stochastic->AddOperation(std::make_shared<CountedX>(0, prefix_calls));
    for (size_t q = 1; q < 3; ++q)
        stochastic->AddOperation(CF::CreateGate(Gate::kXGateType, q));
    stochastic->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Channel::AmplitudeDamping(.5)));
    stochastic->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
    for (size_t q = 0; q < 3; ++q)
        stochastic->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{q}, Channel::AmplitudeDamping(.5)));
    const auto hybrid_prefix = estimate(stochastic, {"Z", "ZZZ"}, {}, samples);
    const auto all_sampled = EstimateNoisy(stochastic, {"ZZZ"}, {}, samples, config, 24, NoiseLayer::kCombined, false);
    const double hybrid_mean = hybrid_prefix.values[0];
    Check(hybrid_prefix.terminal_channels && hybrid_prefix.realizations == samples && hybrid_mean > 0. && hybrid_mean < 1. &&
              std::abs(hybrid_prefix.standard_errors[0] - std::sqrt(hybrid_mean * (1. - hybrid_mean) / (samples - 1))) < 1e-12 &&
              hybrid_prefix.values[1] == all_sampled.values[0] && hybrid_prefix.standard_errors[1] == all_sampled.standard_errors[0],
          "A sampled prefix lost the conditional expectation or reproducibility of a mixed request");

    const auto together = PrepareTerminalExpectation(stochastic, {"Z", "ZZZ"}, config);
    if (backend == Simulators::SimulatorType::kQCSim
#ifndef NO_QISKIT_AER
        || backend == Simulators::SimulatorType::kQiskitAer
#endif
    )
    {
        Check(together && together->RequiresTrajectories(), "A mixed terminal request lost its sampled rows");
        for (bool fixed : {false, true})
        {
            auto scheduled = config;
            scheduled.fixed_backend = fixed;
            const auto joint = PrepareTerminalExpectation(stochastic, {"Z", "ZZZ"}, scheduled);
            for (size_t seed = 0; seed < 8; ++seed)
            {
                scheduled.seed = seed;
                const auto prefix = EstimateCircuitTrajectory(joint->prefix, joint->prefix_observables, scheduled);
                const auto full = EstimateCircuitTrajectory(stochastic, joint->sampled_observables, scheduled);
                prefix_calls->store(0);
                const auto actual = EstimateTerminalExpectation(*joint, scheduled);
                Check(prefix_calls->load() == 1 && actual.values == joint->Combine(prefix.values, full.values),
                      "Mixed rows repeated the sampled prefix or changed its seeded branches");
            }
            scheduled.seed.reset();
            prefix_calls->store(0);
            EstimateTerminalExpectation(*joint, scheduled);
            Check(prefix_calls->load() == 1, "An unseeded mixed request ran its prefix twice");
        }
        // The q2 damping is still terminal on its wire, but moving its physical
        // draw past the interior q0 damping would exchange their random numbers.
        auto reordered = stochastic->GetOperations();
        std::rotate(reordered.begin() + 3, reordered.end() - 1, reordered.end());
        auto interleaved = CF::CreateCircuit(reordered);
        const auto split = PrepareTerminalExpectation(interleaved, {"Z", "ZZZ"}, config);
        Check(split && !split->prefix_read_position, "Interleaved channel draws were reordered to share a trajectory");
        const auto full = EstimateCircuitTrajectory(interleaved, split->sampled_observables, config);
        Check(EstimateTerminalExpectation(*split, config).values[1] == full.values[0], "Per-wire folding changed an interleaved trajectory's seed");
    }

    if (backend == Simulators::SimulatorType::kQCSim)
    {
        // Above the default fusion threshold, the interior damping is resolved
        // before the last gate layer. Local X/Y/Z rows can read that layer
        // without flushing the gates the full trajectory will fuse with damping.
        constexpr size_t width = 11;
        auto fused = config;
        fused.gate_fusion.reset();
        fused.enable_causal_cone_reduction = false;
        fused.auto_reduce = false;
        auto layered = CF::CreateCircuit();
        layered->AddOperation(std::make_shared<CountedX>(0, prefix_calls));
        for (size_t q = 0; q < width; ++q)
            layered->AddOperation(CF::CreateGate(Gate::kRyGateType, q, 0, 0, .3 + .1 * q));
        layered->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{1}, Channel::AmplitudeDamping(.4)));
        for (size_t layer = 0; layer < 4; ++layer)
        {
            for (size_t q = layer % 2; q + 1 < width; q += 2)
                layered->AddOperation(CF::CreateGate(Gate::kCXGateType, q + 1, q));
            for (size_t q = 0; q < width; ++q)
                layered->AddOperation(CF::CreateGate(Gate::kRxGateType, q, 0, 0, .2 + .05 * q + .1 * layer));
        }
        for (size_t q = 0; q < width; ++q)
            layered->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{q}, Channel::AmplitudeDamping(.2)));
        const auto joint = PrepareTerminalExpectation(layered, {"Z", "IY", "IIX", "IIIY", "IXY", std::string(width, 'Z')}, fused);
        Check(joint && joint->prefix_read_position && joint->sampled_indices.size() == 1, "Missing mixed fused-prefix fixture");
        for (size_t seed = 0; seed < 8; ++seed)
        {
            fused.seed = seed;
            const auto prefix = EstimateCircuitTrajectory(joint->prefix, joint->prefix_observables, fused);
            const auto full = EstimateCircuitTrajectory(layered, joint->sampled_observables, fused);
            prefix_calls->store(0);
            const auto actual = EstimateTerminalExpectation(*joint, fused);
            const auto expected = joint->Combine(prefix.values, full.values);
            Check(actual.gate_fusion && prefix_calls->load() == 1, "Default QCSim fusion repeated a resolved stochastic prefix");
            Check(actual.values.back() == expected.back(), "Observing a fused prefix changed a terminal branch");
            for (size_t i = 0; i + 1 < expected.size(); ++i)
                Check(std::abs(actual.values[i] - expected[i]) < 1e-13, "A fused prefix Pauli differs from its standalone expectation");
        }

        const auto wide_prefix = PrepareTerminalExpectation(layered, {"XXXX", std::string(width, 'Z')}, fused);
        Check(wide_prefix && wide_prefix->prefix_read_position, "Missing wide fused-prefix fixture");
        prefix_calls->store(0);
        EstimateTerminalExpectation(*wide_prefix, fused);
        Check(prefix_calls->load() == 2, "A wide prefix exceeded the three-qubit reduced-density limit");

        // This shorter prefix still holds a state-dependent branch at the read.
        // A second pass preserves the independent resolution order in each run.
        fused.gate_fusion = true;
        const auto pending = PrepareTerminalExpectation(stochastic, {"Z", "ZZZ"}, fused);
        Check(pending && pending->prefix_read_position, "Missing unresolved fused-prefix fixture");
        const auto separate_prefix = EstimateCircuitTrajectory(pending->prefix, pending->prefix_observables, fused);
        const auto separate_full = EstimateCircuitTrajectory(stochastic, pending->sampled_observables, fused);
        prefix_calls->store(0);
        const auto pending_result = EstimateTerminalExpectation(*pending, fused);
        Check(prefix_calls->load() == 2 && pending_result.values == pending->Combine(separate_prefix.values, separate_full.values),
              "An unresolved fused prefix changed its seeded branch order");

        // Explicit marginalization permits separate sampled cones. A row with
        // two nonconstant terms needs a common trajectory, but merging all
        // these disjoint cones into one 70-qubit dense register is infeasible.
        auto clustered = config;
        clustered.allow_reset_marginalization = true;
        clustered.max_cluster_qubits = 1;
        clustered.auto_reduce = false;
        auto cones = CF::CreateCircuit();
        std::vector<std::string> singles;
        for (size_t q = 0; q < 70; ++q)
        {
            cones->AddOperation(CF::CreateGate(Gate::kRyGateType, q, 0, 0, .4));
            singles.push_back(std::string(q, 'I') + 'Z');
        }
        cones->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Channel::AmplitudeDamping(0.5)));
        cones->AddOperation(CF::CreateGate(Gate::kXGateType, 0));
        cones->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0},
                                                                                  Channel::UnitaryMixture({0.75, 0.25}, {Eigen::Matrix2cd::Identity(), h})));
        const auto separated = PrepareTerminalExpectation(cones, singles, clustered);
        Check(separated && separated->sampled_indices == std::vector<size_t>{0}, "A sampled adjoint mixed the independently eligible observable cones");
        const auto split = EstimateNoisy(cones, singles, {}, 2, clustered, 24, NoiseLayer::kCombined);
        Check(split.terminal_channels && split.qubits_after == 1 && std::abs(split.values.back() - std::cos(.4)) < 1e-12,
              "Explicit marginalization lost its compact sampled cones");
        Check(PrepareTerminalExpectation(mixed, {"Z"}, clustered).has_value(), "Marginalized affine Z and identity unnecessarily fell back to trajectories");
    }

    auto wide = CF::CreateCircuit();
    for (size_t q = 0; q < 12; ++q)
        wide->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{q}, Channel::AmplitudeDamping(0.2)));
    Check(!PrepareTerminalExpectation(wide, {std::string(12, 'Z')}, config), "Wide damping expansion did not switch back to trajectories");
    Check(PrepareTerminalExpectation(wide, {std::string(12, 'X')}, config).has_value(), "A wide diagonal observable was needlessly sampled");
    wide->AddOperation(CF::CreateGate(Gate::kXGateType, 15));
    Check(PrepareTerminalExpectation(wide, {std::string(8, 'Z')}, config).has_value(), "A 256-term deterministic row was sampled on 16 qubits");
    wide->AddOperation(CF::CreateGate(Gate::kXGateType, 19));
    Check(PrepareTerminalExpectation(wide, {std::string(7, 'Z')}, config).has_value(), "A 128-term deterministic row was sampled on 20 qubits");
    Check(!PrepareTerminalExpectation(wide, {std::string(8, 'Z')}, config), "A wide dense deterministic row exceeded its bounded scan budget");

    if (backend == Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
        || backend == Simulators::SimulatorType::kCompositeQiskitAer
#endif
    )
    {
        // Each qubit stays independent. A single global X observable touches
        // every one, so its expectation must stay a product across groups
        // even when a correlated final channel acts on the endpoints.
        auto separate = CF::CreateCircuit();
        constexpr size_t width = 70;
        constexpr double angle = 0.06;
        for (size_t q = 0; q < width; ++q)
        {
            separate->AddOperation(CF::CreateGate(Gate::kHadamardGateType, q));
            separate->AddOperation(CF::CreateGate(Gate::kRzGateType, q, 0, 0, angle));
        }
        separate->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{width - 1, 0}, Channel::TwoQubitDepolarizing(0.25)));
        config.enable_causal_cone_reduction = false;
        config.auto_reduce = false;
        const auto factored = estimate(separate, {std::string(width, 'X')}, {});
        Check(factored.terminal_channels && factored.realizations == 1 && factored.ideal.simulator == backend &&
                  std::abs(factored.values[0] - (1. - 16. * 0.25 / 15.) * std::pow(std::cos(angle), width)) < 1e-12,
              "A terminal estimate on independent statevector components lost its factorized expectation");
    }
}
} // namespace

void TestMixedStateExecution()
{
    PathIntegralKrausBranches();
    CheckUnitaryPreparation();
    CheckPreparationGateAdjoints();
    CheckRealizationFidelityIdeal();
    CheckpointMultinomialCounts();
    CheckpointTensorRouting(Method::kMatrixProductState, {1, 9, 8, 5, 10, 2, 3, 7, 4, 0, 11, 6});
    CheckpointTensorRouting(Method::kMatrixProductOperator, {4, 2, 1, 0, 5, 3});
    for (const auto method : {Method::kStatevector, Method::kExtendedStabilizer})
        ExplicitSamplingPolicyKeepsAdaptiveCheckpoint(method);
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
    for (const auto method : {Method::kStatevector, Method::kDensityMatrix, Method::kPauliPropagator})
        CountSinkKeepsReadoutDraws(method);
    NestedChannelsLeaveTheIdealBaseline();
    for (const auto backend : {Simulators::SimulatorType::kQCSim, Simulators::SimulatorType::kCompositeQCSim
#ifndef NO_QISKIT_AER
                               ,
                               Simulators::SimulatorType::kQiskitAer, Simulators::SimulatorType::kCompositeQiskitAer
#endif
         })
        TerminalStatevectorExpectations(backend);
    FidelityPreparesResetTargets();
    CandidateListsResolveOneBackend();
    DeterministicPrefixNoise();
    HeadCopyCountsTheCheckpoint();
}
