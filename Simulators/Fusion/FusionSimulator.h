#pragma once
#include "FusionState.h"
namespace Simulators::Private {
// Gate submission shared by QCSim, GPU, distributed and composite adapters.
template <class State>
class FusionSimulator : public State {
 public:
  using State::State;
  void ApplyP(Types::qubit_t q0, double p0) override {
    this->SubmitGate({FusionGate::Kind::kPhaseGateType, {q0}, {p0}, {}});
  }
  void ApplyX(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kXGateType, {q0}, {}, {}});
  }
  void ApplyY(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kYGateType, {q0}, {}, {}});
  }
  void ApplyZ(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kZGateType, {q0}, {}, {}});
  }
  void ApplyH(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kHadamardGateType, {q0}, {}, {}});
  }
  void ApplyS(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kSGateType, {q0}, {}, {}});
  }
  void ApplySDG(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kSdgGateType, {q0}, {}, {}});
  }
  void ApplyT(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kTGateType, {q0}, {}, {}});
  }
  void ApplyTDG(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kTdgGateType, {q0}, {}, {}});
  }
  void ApplySx(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kSxGateType, {q0}, {}, {}});
  }
  void ApplySxDAG(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kSxDagGateType, {q0}, {}, {}});
  }
  void ApplyK(Types::qubit_t q0) override {
    this->SubmitGate({FusionGate::Kind::kKGateType, {q0}, {}, {}});
  }
  void ApplyRx(Types::qubit_t q0, double p0) override {
    this->SubmitGate({FusionGate::Kind::kRxGateType, {q0}, {p0}, {}});
  }
  void ApplyRy(Types::qubit_t q0, double p0) override {
    this->SubmitGate({FusionGate::Kind::kRyGateType, {q0}, {p0}, {}});
  }
  void ApplyRz(Types::qubit_t q0, double p0) override {
    this->SubmitGate({FusionGate::Kind::kRzGateType, {q0}, {p0}, {}});
  }
  void ApplyU(Types::qubit_t q0, double p0, double p1, double p2,
              double p3) override {
    this->SubmitGate(
        {FusionGate::Kind::kUGateType, {q0}, {p0, p1, p2, p3}, {}});
  }
  void ApplySwap(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kSwapGateType, {q0, q1}, {}, {}});
  }
  void ApplyCX(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kCXGateType, {q0, q1}, {}, {}});
  }
  void ApplyCY(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kCYGateType, {q0, q1}, {}, {}});
  }
  void ApplyCZ(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kCZGateType, {q0, q1}, {}, {}});
  }
  void ApplyCP(Types::qubit_t q0, Types::qubit_t q1, double p0) override {
    this->SubmitGate({FusionGate::Kind::kCPGateType, {q0, q1}, {p0}, {}});
  }
  void ApplyCRx(Types::qubit_t q0, Types::qubit_t q1, double p0) override {
    this->SubmitGate({FusionGate::Kind::kCRxGateType, {q0, q1}, {p0}, {}});
  }
  void ApplyCRy(Types::qubit_t q0, Types::qubit_t q1, double p0) override {
    this->SubmitGate({FusionGate::Kind::kCRyGateType, {q0, q1}, {p0}, {}});
  }
  void ApplyCRz(Types::qubit_t q0, Types::qubit_t q1, double p0) override {
    this->SubmitGate({FusionGate::Kind::kCRzGateType, {q0, q1}, {p0}, {}});
  }
  void ApplyCH(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kCHGateType, {q0, q1}, {}, {}});
  }
  void ApplyCSx(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kCSxGateType, {q0, q1}, {}, {}});
  }
  void ApplyCSxDAG(Types::qubit_t q0, Types::qubit_t q1) override {
    this->SubmitGate({FusionGate::Kind::kCSxDagGateType, {q0, q1}, {}, {}});
  }
  void ApplyCU(Types::qubit_t q0, Types::qubit_t q1, double p0, double p1,
               double p2, double p3) override {
    this->SubmitGate(
        {FusionGate::Kind::kCUGateType, {q0, q1}, {p0, p1, p2, p3}, {}});
  }
  void ApplyCSwap(Types::qubit_t q0, Types::qubit_t q1,
                  Types::qubit_t q2) override {
    this->SubmitGate({FusionGate::Kind::kCSwapGateType, {q0, q1, q2}, {}, {}});
  }
  void ApplyCCX(Types::qubit_t q0, Types::qubit_t q1,
                Types::qubit_t q2) override {
    this->SubmitGate({FusionGate::Kind::kCCXGateType, {q0, q1, q2}, {}, {}});
  }
  void ApplyGenericOneQubitGate(Types::qubit_t q,
                                const Eigen::Matrix2cd& m) override {
    this->SubmitGate({FusionGate::Kind::kNone, {q}, {}, m});
  }
  void ApplyGenericTwoQubitGate(Types::qubit_t a, Types::qubit_t b,
                                const Eigen::Matrix4cd& m) override {
    this->SubmitGate({FusionGate::Kind::kNone, {a, b}, {}, m});
  }
  void ApplyGenericThreeQubitGate(Types::qubit_t a, Types::qubit_t b,
                                  Types::qubit_t c,
                                  const ISimulator::Matrix8cd& m) override {
    this->SubmitGate({FusionGate::Kind::kNone, {a, b, c}, {}, m});
  }
  void ApplyNop() override {
    this->FlushPendingGates();
    this->immediate_->ApplyNop();
    this->ConsumeBoundary(Circuits::OperationType::kNoOp);
    this->NotifyObservers({});
  }
};
}  // namespace Simulators::Private
