// Terminal sampling from an existing state, shared by raw rows and checkpoints.
#pragma once

#include "SampleRecords.h"

#include <algorithm>
#include <limits>

namespace MaestroExecution
{
inline bool HasResetOperation(const Circuits::Circuit<double> &circuit)
{
    for (const auto &op : circuit.GetOperations())
        if (op->GetType() == Circuits::OperationType::kReset ||
            (op->GetType() == Circuits::OperationType::kComposite && HasResetOperation(*std::static_pointer_cast<Circuits::Circuit<double>>(op))))
            return true;
    return false;
}

// Whether terminal sampling leaves a saved checkpoint intact. Mixed-state
// adapters and the QCSim statevector, stabilizer and MPS adapters draw shots
// without the save slot; tensor-network sampling, for one, saves and restores
// its own state between draws, so checkpoints keep replaying shots there.
inline bool PreservesSavedStateWhileSampling(const Simulators::ISimulator &simulator)
{
    if (simulator.SupportsQuantumChannels())
        return true;
    using Method = Simulators::SimulationType;
    const auto method = simulator.GetSimulationType();
    return simulator.GetType() == Simulators::SimulatorType::kQCSim &&
           (method == Method::kStatevector || method == Method::kStabilizer || method == Method::kMatrixProductState);
}

// Nested circuits are inlined before analysis (Circuit::InlinedComposites):
// one that ends in measurements is then terminal like the enclosing circuit.
inline std::shared_ptr<Circuits::Circuit<double>> InlinedForSampling(const Circuits::Circuit<double> &circuit)
{
    return circuit.HasCompositeOperations() ? circuit.InlinedComposites() : nullptr;
}

inline bool CanSampleTerminal(const Circuits::Circuit<double> &input, const Simulators::ISimulator &simulator, size_t width,
                              bool known_zero_initial_state = false)
{
    const auto inlined = InlinedForSampling(input);
    const auto &circuit = inlined ? *inlined : input;
    const bool exact = simulator.SupportsQuantumChannels();
    // Initial resets are safe for a fresh zero register, but can collapse an
    // entangled arbitrary checkpoint on pure-state methods.
    if (!exact && !known_zero_initial_state && HasResetOperation(circuit))
        return false;
    if (circuit.HasOpsAfterMeasurements(exact))
        return false;
    Circuits::OperationState dry(width);
    const auto executed = circuit.ExecuteNonMeasurements(nullptr, dry, nullptr, exact);
    const auto &operations = circuit.GetOperations();
    const size_t first = operations.size() - executed.size();
    for (size_t i = first; i < operations.size(); ++i)
        if (!executed[i - first] && operations[i]->GetType() != Circuits::OperationType::kMeasurement)
            return false;
    return true;
}

// On success, evolves the deterministic part exactly once from the CURRENT
// simulator state, then emits (row index, classical state) for every sample.
// Never resets the simulator or overwrites its saved checkpoint. A false result
// leaves both quantum and classical state untouched, for the trajectory fallback.
// Unwritten classical bits keep their input values. Readout is independent per
// measurement occurrence and shot; a seeded permutation makes raw rows exchangeable.
template <class RowSink>
bool SamplePreparedTerminal(const Circuits::Circuit<double> &input, const std::shared_ptr<Simulators::ISimulator> &simulator, size_t shots,
                            Circuits::OperationState &state, std::optional<uint64_t> seed, RowSink &&write, size_t *maxBondDimension = nullptr,
                            bool known_zero_initial_state = false)
{
    const auto inlined = InlinedForSampling(input);
    const auto &circuit = inlined ? *inlined : input;
    if (!simulator || !CanSampleTerminal(circuit, *simulator, state.GetAllBits().size(), known_zero_initial_state))
        return false;
    // Lookahead routing (MPS/MPO swap planning, fusion) follows the circuit
    // that runs now, not one left over from a previous call.
    simulator->SetGatesCounter(0);
    simulator->SetUpcomingGates(circuit.GetOperations());
    const auto executed = circuit.ExecuteNonMeasurements(simulator, state, maxBondDimension);
    const auto measurements = circuit.GetLastMeasurements(executed, false);
    const auto &qubits = measurements->GetQubits();
    if (qubits.empty())
    {
        for (size_t row = 0; row < shots; ++row)
            write(row, state);
        return true;
    }
    const Circuits::OperationState initial(state);
    const auto order = SampleRowOrder::Permutation(shots, seed, SampleRowOrder::kTerminalStream);
    const auto histogram = simulator->SampleCountsMany(qubits, shots);
    using Outcome = std::pair<const std::vector<bool>, Types::qubit_t>;
    std::vector<const Outcome *> outcomes;
    outcomes.reserve(histogram.size());
    for (const auto &outcome : histogram)
        outcomes.push_back(&outcome);
    std::sort(outcomes.begin(), outcomes.end(), [](const Outcome *a, const Outcome *b) { return a->first < b->first; });
    auto *readout = measurements->HasReadout() ? simulator.get() : nullptr;
    size_t row = 0;
    for (const auto *outcome : outcomes)
        for (Types::qubit_t copy = 0; copy < outcome->second; ++copy)
        {
            if (row >= shots)
                throw std::runtime_error("Sampling returned too many shots");
            state = initial;
            measurements->SetStateFromSample(outcome->first, state, readout);
            write(order[row++], state);
        }
    if (row != shots)
        throw std::runtime_error("Sampling returned an incomplete shot matrix");
    return true;
}

// Whether a dynamic circuit's deterministic head may be held on a copy of the
// simulator: every retained state within the realization memory budget
// (trajectory_max_memory_mb), as RealizationWorkers counts them. That is the
// checkpoint's live state and saved prefix plus the copy and its saved head.
// Unbounded states (an MPS/MPO without a bond cap, tensor networks) keep
// replaying.
inline constexpr size_t kHeadCopyRetainedStates = 4;

inline bool HeadCopyFits(const SimulatorConfig &config, size_t qubits)
{
    const long double bytes = MaxRealizationBytes(config, qubits);
    const size_t budget = RealizationBudgetMB(config);
    return bytes > 0 && budget <= (std::numeric_limits<size_t>::max() >> 20) && kHeadCopyRetainedStates * bytes <= static_cast<long double>(budget << 20);
}

// A dynamic circuit samples a branch per shot, but its deterministic head (the
// operations ExecuteNonMeasurements shares) needs evolving only once. From the
// CURRENT simulator state, evolve that head on a copy of the simulator, save it
// there and replay only the rest for every shot, emitting (shot, classical
// state) in shot order. The caller's simulator and its saved checkpoint stay
// untouched. Returns false, doing nothing, when no operation can be shared,
// when a pure-state reset could branch (on an arbitrary state even a leading
// reset may), or when the backend cannot be copied. A given seed seeds the
// copy as the caller seeded its simulator: the head draws no randomness, so
// the shots consume the same measurement stream as replaying the circuit.
template <class RowSink>
bool SampleSharedHead(const Circuits::Circuit<double> &input, const std::shared_ptr<Simulators::ISimulator> &simulator, size_t shots,
                      Circuits::OperationState &state, std::optional<uint64_t> seed, RowSink &&write, size_t *maxBondDimension = nullptr)
{
    if (!simulator || shots < 2)
        return false;
    const auto inlined = InlinedForSampling(input);
    const auto &circuit = inlined ? *inlined : input;
    const bool exact = simulator->SupportsQuantumChannels();
    if (!exact && HasResetOperation(circuit))
        return false;
    Circuits::OperationState dry(state);
    const auto mask = circuit.ExecuteNonMeasurements(nullptr, dry, nullptr, exact);
    const size_t shared = circuit.GetOperations().size() - mask.size() + static_cast<size_t>(std::count(mask.begin(), mask.end(), true));
    if (shared == 0)
        return false;
    std::shared_ptr<Simulators::ISimulator> head(simulator->Clone());
    if (!head)
        return false;
    if (seed)
        head->SetSeed(*seed);
    head->SetGatesCounter(0);
    head->SetUpcomingGates(circuit.GetOperations());
    auto executed = circuit.ExecuteNonMeasurements(head, state, maxBondDimension);
    const auto rest = circuit.RemoveExecutedOperations(executed);
    head->SaveState();
    for (size_t shot = 0; shot < shots; ++shot)
    {
        if (shot > 0)
            head->RestoreState();
        head->SetGatesCounter(0);
        head->SetUpcomingGates(rest->GetOperations());
        rest->ExecuteBD(head, state, maxBondDimension);
        write(shot, state);
    }
    return true;
}
} // namespace MaestroExecution
