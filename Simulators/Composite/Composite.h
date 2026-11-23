#pragma once
#ifdef INCLUDED_BY_FACTORY
#include "../Fusion/FusionSimulator.h"
#include "ImmediateComposite.h"

namespace Simulators::Private
{
class CompositeSimulator : public FusionSimulator<FusionState>
{
  public:
    explicit CompositeSimulator(SimulatorType type = SimulatorType::kQCSim)
        : FusionSimulator(std::make_shared<ImmediateCompositeSimulator>(type)), childType_(type)
    {
    }

    unsigned GetGateFusionMaxQubits() const override
    {
        return childType_ == SimulatorType::kQCSim ? 3 : 0;
    }

#ifndef NO_QISKIT_AER
    auto SampleAerComponents(const Types::qubits_vector &qubits, size_t shots)
    {
        Flush();
        return static_cast<ImmediateCompositeSimulator &>(*immediate_).SampleAerComponents<true>(qubits, shots);
    }
#endif

    std::unique_ptr<ISimulator> Clone() override
    {
        auto copy = std::make_unique<CompositeSimulator>(childType_);
        CloneInto(*copy);
        return copy;
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        auto copy = std::make_unique<CompositeSimulator>(childType_);
        CloneInto(*copy, &seed);
        return copy;
    }

  protected:
    // The composite keeps its separate cloned snapshot when rebuilding children.
    bool InitializationPreservesSnapshots() const override
    {
        return true;
    }

  private:
    SimulatorType childType_;
};
} // namespace Simulators::Private
#endif
