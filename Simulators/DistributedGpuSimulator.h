// Gate bridge; native calls check errors before notifying observers.
#pragma once
#if defined(__linux__) && defined(INCLUDED_BY_FACTORY)
#include "DistributedGpuState.h"
namespace Simulators::Private {
template <class State>
class DistributedGpuSimulatorBase : public State {
 public:
  void ApplyGenericOneQubitGate(Types::qubit_t q,
                                const Eigen::Matrix2cd& gate) override {
    this->QubitIndex(q);
    this->Native().ApplyOneQubitMatrixWithLayout(
        q, reinterpret_cast<const double*>(gate.data()), 1);
    this->NotifyObservers({q});
  }
  void ApplyGenericTwoQubitGate(Types::qubit_t q0, Types::qubit_t q1,
                                const Eigen::Matrix4cd& gate) override {
    this->QubitIndex(q0);
    this->QubitIndex(q1);
    this->Native().ApplyTwoQubitMatrixWithLayout(
        q0, q1, reinterpret_cast<const double*>(gate.data()), 1);
    this->NotifyObservers({q0, q1});
  }
  void ApplyP(Types::qubit_t qubit, double lambda) override {
    this->QubitIndex(qubit);
    this->Native().ApplyP(qubit, lambda);
    this->NotifyObservers({qubit});
  }
  void ApplyX(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyX(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyY(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyY(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyZ(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyZ(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyH(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyH(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyS(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyS(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplySDG(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplySDG(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyT(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyT(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyTDG(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyTDG(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplySx(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplySX(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplySxDAG(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplySXDG(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyK(Types::qubit_t qubit) override {
    this->QubitIndex(qubit);
    this->Native().ApplyK(qubit);
    this->NotifyObservers({qubit});
  }
  void ApplyRx(Types::qubit_t qubit, double theta) override {
    this->QubitIndex(qubit);
    this->Native().ApplyRx(qubit, theta);
    this->NotifyObservers({qubit});
  }
  void ApplyRy(Types::qubit_t qubit, double theta) override {
    this->QubitIndex(qubit);
    this->Native().ApplyRy(qubit, theta);
    this->NotifyObservers({qubit});
  }
  void ApplyRz(Types::qubit_t qubit, double theta) override {
    this->QubitIndex(qubit);
    this->Native().ApplyRz(qubit, theta);
    this->NotifyObservers({qubit});
  }
  void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda,
              double gamma) override {
    this->QubitIndex(qubit);
    this->Native().ApplyU(qubit, theta, phi, lambda, gamma);
    this->NotifyObservers({qubit});
  }
  void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCX(ctrl_qubit, tgt_qubit);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCY(ctrl_qubit, tgt_qubit);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCZ(ctrl_qubit, tgt_qubit);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
               double lambda) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCP(ctrl_qubit, tgt_qubit, lambda);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCRx(ctrl_qubit, tgt_qubit, theta);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCRy(ctrl_qubit, tgt_qubit, theta);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCRz(ctrl_qubit, tgt_qubit, theta);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCH(ctrl_qubit, tgt_qubit);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCSX(ctrl_qubit, tgt_qubit);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyCSxDAG(Types::qubit_t ctrl_qubit,
                   Types::qubit_t tgt_qubit) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCSXDG(ctrl_qubit, tgt_qubit);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override {
    this->QubitIndex(qubit0);
    this->QubitIndex(qubit1);
    this->Native().ApplySwap(qubit0, qubit1);
    this->NotifyObservers({qubit0, qubit1});
  }
  void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1,
                Types::qubit_t qubit2) override {
    this->QubitIndex(qubit0);
    this->QubitIndex(qubit1);
    this->QubitIndex(qubit2);
    this->Native().ApplyCCX(qubit0, qubit1, qubit2);
    this->NotifyObservers({qubit0, qubit1, qubit2});
  }
  void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0,
                  Types::qubit_t qubit1) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(qubit0);
    this->QubitIndex(qubit1);
    this->Native().ApplyCSwap(ctrl_qubit, qubit0, qubit1);
    this->NotifyObservers({ctrl_qubit, qubit0, qubit1});
  }
  void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
               double theta, double phi, double lambda, double gamma) override {
    this->QubitIndex(ctrl_qubit);
    this->QubitIndex(tgt_qubit);
    this->Native().ApplyCU(ctrl_qubit, tgt_qubit, theta, phi, lambda, gamma);
    this->NotifyObservers({ctrl_qubit, tgt_qubit});
  }
  void ApplyNop() override { this->NotifyObservers({}); }
  std::unique_ptr<ISimulator> Clone() override {
    auto copy = std::make_unique<DistributedGpuSimulatorBase<State>>();
    copy->configuration = this->configuration;
    copy->nrQubits = this->nrQubits;
    if (this->state) copy->state = this->state->Clone();
    return copy;
  }
};
using DistributedGpuSimulator =
    DistributedGpuSimulatorBase<DistributedGpuState>;
using DistributedMpiGpuSimulator =
    DistributedGpuSimulatorBase<DistributedMpiGpuState>;
}  // namespace Simulators::Private
#endif
