#pragma once

#include "../Circuit/Circuit.h"

namespace MaestroExecution
{
// The initial circuit followed by N copies of one step is one program. Remove
// final readout with the same backwards scan as estimate, carrying dependencies
// across the boundaries. Once the sets stop changing, every earlier iteration
// has the same operations, even for a request with billions of steps.
class IncrementalEvolution
{
    using Circuit = Circuits::Circuit<>;
    using CircuitPtr = std::shared_ptr<Circuit>;

  public:
    IncrementalEvolution(const CircuitPtr &initial, const CircuitPtr &step, size_t repetitions) : repetitions_(repetitions)
    {
        std::unordered_set<Types::qubit_t> activeQubits;
        std::unordered_set<size_t> neededBits;
        const auto stepCircuit = step->HasCompositeOperations() ? step->InlinedComposites() : step;
        for (size_t i = 0; i < repetitions; ++i)
        {
            const auto laterQubits = activeQubits;
            const auto laterBits = neededBits;
            stepsFromEnd_.push_back(stepCircuit->RemoveEndMeasurementsBefore(activeQubits, neededBits));
            if (activeQubits == laterQubits && neededBits == laterBits)
                break;
        }
        const auto initialCircuit = initial->HasCompositeOperations() ? initial->InlinedComposites() : initial;
        initial_ = initialCircuit->RemoveEndMeasurementsBefore(activeQubits, neededBits);
    }

    void Start(const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state, size_t &maxBond) const
    {
        Apply(initial_, simulator, state, maxBond);
    }

    void Step(size_t index, const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state, size_t &maxBond) const
    {
        const auto fromEnd = std::min(repetitions_ - index, stepsFromEnd_.size());
        Apply(stepsFromEnd_[fromEnd - 1], simulator, state, maxBond);
    }

  private:
    static void Apply(const CircuitPtr &circuit, const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state, size_t &maxBond)
    {
        // Installing a routing window restarts its gate counter and synchronizes
        // the backend. Ordinary evolution can continue directly into the fragment.
        if (simulator->IsRoutingLookaheadEnabled())
            simulator->SetUpcomingGates(circuit->GetOperations());
        circuit->ExecuteContinuation(simulator, state, &maxBond);
    }

    size_t repetitions_;
    CircuitPtr initial_;
    std::vector<CircuitPtr> stepsFromEnd_;
};
} // namespace MaestroExecution
