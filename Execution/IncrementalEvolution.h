#pragma once

#include "../Circuit/Circuit.h"

namespace MaestroExecution
{
// An observation sees the end of the current fragment before its terminal readout.
// Only the classical results used by the repeated step cross a boundary.
class IncrementalEvolution
{
    using Circuit = Circuits::Circuit<>;
    using CircuitPtr = std::shared_ptr<Circuit>;
    using Op = Circuit::OperationPtr;
    using Type = Circuits::OperationType;
    using Bits = std::unordered_set<size_t>;
    using Wires = std::unordered_set<Types::qubit_t>;
    using State = Circuits::OperationState;
    using Simulator = std::shared_ptr<Simulators::ISimulator>;

    static void Apply(const CircuitPtr &circuit, const Simulator &sim, State &state, size_t &maxBond)
    {
        if (circuit->GetOperations().empty())
            return;
        if (sim->IsRoutingLookaheadEnabled())
            sim->SetUpcomingGates(circuit->GetOperations());
        circuit->ExecutePhysicalOperations(sim, state, &maxBond);
    }

    struct Event
    {
        // Execute the body through this operation before the observation. At
        // the next boundary, replay only its classical writes and deferred readout.
        CircuitPtr body;
        std::shared_ptr<Circuits::ICondition> condition;
        CircuitPtr readout;
        std::vector<size_t> writes;
    };

    struct Fragment
    {
        CircuitPtr body;
        std::vector<Event> events;
        CircuitPtr tail;

        Fragment() = default;

        Fragment(const CircuitPtr &circuit, Bits neededAfter)
        {
            Bits localBits;
            Circuit::MeasurementMasks local, full;
            body = circuit->RemoveEndMeasurementsBefore(localBits, &local);
            if (neededAfter.empty())
                return;
            circuit->RemoveEndMeasurementsBefore(neededAfter, &full);
            if (local == full)
                return;

            // Keep the classical history of this fragment. For example, a
            // deferred conditional measurement must see its original predicate,
            // even when another operation overwrites that bit before observation.
            tail = std::make_shared<Circuit>();
            const auto &ops = circuit->GetOperations();
            for (size_t i = 0; i < ops.size(); ++i)
            {
                const auto &op = ops[i];
                const auto condition = op->IsConditional() ? std::static_pointer_cast<Circuits::IConditionalOperation<>>(op) : nullptr;
                const auto nested = condition ? condition->GetOperation() : op;
                if (!nested || (condition && !condition->GetCondition()))
                    continue;
                const bool measurement = op->GetType() == Type::kMeasurement || op->GetType() == Type::kConditionalMeasurement;
                const auto retained = measurement ? Circuit::SelectMeasurementPairs(op, local[i]) : op;
                if (retained)
                    tail->AddOperation(retained);

                Event event{tail, condition ? condition->GetCondition() : nullptr, nullptr, {}};
                if (measurement)
                {
                    auto deferred = full[i];
                    for (size_t j = 0; j < deferred.size(); ++j)
                        deferred[j] = deferred[j] && !local[i][j];
                    if (const auto read = Circuit::SelectMeasurementPairs(nested, deferred))
                        event.readout = std::make_shared<Circuit>(Circuit::OperationsVector{read});

                    // A grouped measurement may write a destination more than
                    // once. Its last retained pair supplies the final value.
                    const auto &bits = std::static_pointer_cast<Circuits::MeasurementOperation<>>(nested)->GetBitsIndices();
                    Bits seen;
                    for (size_t j = full[i].size(); j-- > 0;)
                        if (full[i][j] && seen.insert(bits[j]).second && local[i][j])
                            event.writes.push_back(bits[j]);
                }
                else
                    event.writes = nested->AffectedBits();
                if (event.readout || !event.writes.empty())
                {
                    events.emplace_back(std::move(event));
                    tail = std::make_shared<Circuit>();
                }
            }
        }
    };

  public:
    IncrementalEvolution(const CircuitPtr &initial, const CircuitPtr &step, size_t repetitions) : repetitions_(repetitions)
    {
        const auto initialCircuit = initial->HasCompositeOperations() ? initial->InlinedComposites() : initial;
        const auto stepCircuit = step->HasCompositeOperations() ? step->InlinedComposites() : step;
        classicalBits_ = std::max(initialCircuit->GetMaxCbitIndex(), stepCircuit->GetMaxCbitIndex()) + 1;
        // Compute the bits a step can consume before overwriting them. Using
        // the fixed point makes every observation independent of future requests;
        // only classical dependencies flow across fragment boundaries.
        Bits needed;
        for (;;)
        {
            const auto previous = needed;
            stepCircuit->RemoveEndMeasurementsBefore(needed);
            if (needed == previous)
                break;
        }
        initial_ = Fragment(initialCircuit, needed);
        step_ = Fragment(stepCircuit, needed);
    }

    void Start(const Simulator &sim, State &state, size_t &maxBond)
    {
        Prepare(initial_, sim, state, maxBond);
    }

    void Step(size_t index, const Simulator &sim, State &state, size_t &maxBond)
    {
        // Delaying readout until entry also preserves the step-zero observation.
        const auto &previous = index == 0 ? initial_ : step_;
        if (!previous.events.empty())
        {
            size_t pos = 0;
            for (const auto &event : previous.events)
            {
                if (!event.condition || event.condition->IsConditionMet(boundary_))
                {
                    if (event.readout)
                        Apply(event.readout, sim, boundary_, maxBond);
                    for (size_t j = 0; j < event.writes.size(); ++j)
                        boundary_.SetBit(event.writes[j], writes_[pos + j]);
                }
                pos += event.writes.size();
            }
            state = boundary_;
        }
        Prepare(step_, sim, state, maxBond);
    }

    bool IsStochastic(bool exactChannels) const
    {
        Wires touched;
        State zeros(classicalBits_);
        // Until a measurement or a random source runs, every predicate reads
        // the initially zero classical register.
        const auto draws = [&](const Fragment &fragment, bool includeReadout) {
            for (const auto &op : fragment.body->GetOperations())
            {
                const auto conditional = op->IsConditional() ? std::static_pointer_cast<Circuits::IConditionalOperation<>>(op) : nullptr;
                if (conditional && (!conditional->GetCondition() || !conditional->GetCondition()->IsConditionMet(zeros)))
                    continue;
                const auto nested = conditional ? conditional->GetOperation() : op;
                if (!nested)
                    continue;
                const auto type = nested->GetType();
                if (type == Type::kDelay || type == Type::kNoOp)
                    continue;
                if (type == Type::kMeasurement || type == Type::kRandomGen || (!exactChannels && Circuit::IsDrawnChannel(nested, Circuit::ChannelDraws::kAll)))
                    return true;
                for (auto q : nested->AffectedQubits())
                {
                    if (type == Type::kReset)
                    {
                        if (touched.erase(q) && !exactChannels)
                            return true;
                    }
                    else
                        touched.insert(q);
                }
            }
            if (includeReadout)
                for (const auto &event : fragment.events)
                    if (event.readout && (!event.condition || event.condition->IsConditionMet(zeros)))
                        return true;
            return false;
        };
        return draws(initial_, repetitions_ != 0) || (repetitions_ && draws(step_, repetitions_ > 1)) || (repetitions_ > 1 && draws(step_, false));
    }

  private:
    void Prepare(const Fragment &fragment, const Simulator &sim, State &state, size_t &maxBond)
    {
        if (fragment.events.empty())
        {
            Apply(fragment.body, sim, state, maxBond);
            return;
        }
        boundary_ = state;
        writes_.clear();
        for (const auto &event : fragment.events)
        {
            Apply(event.body, sim, state, maxBond);
            for (const auto bit : event.writes)
                writes_.push_back(state.GetBit(bit));
        }
        Apply(fragment.tail, sim, state, maxBond);
    }

    size_t repetitions_;
    size_t classicalBits_;
    Fragment initial_, step_;
    State boundary_;
    std::vector<bool> writes_;
};
} // namespace MaestroExecution
