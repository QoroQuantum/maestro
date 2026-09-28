#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "ImmediateQCSimSimulator.h"
#include "FusionState.h"
namespace Simulators::Private {
class QCSimState : public FusionState {
 public:
  QCSimState() : FusionState(std::make_shared<ImmediateQCSimSimulator>()) {}
  unsigned GetGateFusionMaxQubits() const override {
    switch (GetSimulationType()) {
      case SimulationType::kStatevector:
      case SimulationType::kDensityMatrix:
        return 3;
      case SimulationType::kMatrixProductState:
      case SimulationType::kMatrixProductOperator:
      case SimulationType::kTensorNetwork:
        return 2;
      default:
        return 0;
    }
  }
};
}  // namespace Simulators::Private
#endif
