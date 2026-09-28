#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "ImmediateGpuSimulator.h"
#include "FusionState.h"
namespace Simulators::Private {
class GpuState : public FusionState {
 public:
  GpuState() : FusionState(std::make_shared<ImmediateGpuSimulator>()) {}
  unsigned GetGateFusionMaxQubits() const override {
    switch (GetSimulationType()) {
      case SimulationType::kStatevector:
        return GpuLibrary::GetInstance()->HasStatevectorMatrixAPI() ? 3 : 0;
      case SimulationType::kDensityMatrix:
      case SimulationType::kMatrixProductState:
      case SimulationType::kMatrixProductOperator:
        return 2;
      default:
        return 0;
    }
  }

 protected:
  bool PreserveStructuredGates() const override {
    return GetSimulationType() == SimulationType::kStatevector;
  }
  bool UsesDestructiveStateStorage() const override {
    return GetSimulationType() == SimulationType::kStatevector;
  }
};
}  // namespace Simulators::Private
#endif
