#pragma once
#include <array>
#include "../Circuit/Factory.h"
#include "GateFusion.h"

namespace Simulators {
// Named gates retain their native entry point until another operation fuses
// with them. Matrices are constructed lazily; routing preparation needs none.
struct FusionGate {
  using Kind = Circuits::QuantumGateType;
  Kind kind = Kind::kNone;
  Types::qubits_vector qubits;
  std::array<double, 4> params{};
  Eigen::MatrixXcd generic;

  Types::qubits_vector Targets() const {
    auto order = qubits;
    if (kind != Kind::kNone && order.size() > 1 && kind != Kind::kSwapGateType)
      std::reverse(order.begin(), order.end());
    return order;
  }
  Eigen::MatrixXcd Matrix() const {
    using namespace Circuits;
    const auto q1 = qubits[0], q2 = qubits.size() > 1 ? qubits[1] : 0,
               q3 = qubits.size() > 2 ? qubits[2] : 0;
    const auto param1 = params[0], param2 = params[1], param3 = params[2],
               param4 = params[3];
    // Construct named gates on the stack: merging needs only their matrix.
    switch (kind) {
      case Kind::kNone:
        return generic;
      case Kind::kPhaseGateType:
        return PhaseGate<>(q1, param1).GetMatrix();
      case Kind::kXGateType:
        return XGate<>(q1).GetMatrix();
      case Kind::kYGateType:
        return YGate<>(q1).GetMatrix();
      case Kind::kZGateType:
        return ZGate<>(q1).GetMatrix();
      case Kind::kHadamardGateType:
        return HadamardGate<>(q1).GetMatrix();
      case Kind::kSGateType:
        return SGate<>(q1).GetMatrix();
      case Kind::kSdgGateType:
        return SdgGate<>(q1).GetMatrix();
      case Kind::kTGateType:
        return TGate<>(q1).GetMatrix();
      case Kind::kTdgGateType:
        return TdgGate<>(q1).GetMatrix();
      case Kind::kSxGateType:
        return SxGate<>(q1).GetMatrix();
      case Kind::kSxDagGateType:
        return SxDagGate<>(q1).GetMatrix();
      case Kind::kKGateType:
        return KGate<>(q1).GetMatrix();
      case Kind::kRxGateType:
        return RxGate<>(q1, param1).GetMatrix();
      case Kind::kRyGateType:
        return RyGate<>(q1, param1).GetMatrix();
      case Kind::kRzGateType:
        return RzGate<>(q1, param1).GetMatrix();
      case Kind::kUGateType:
        return UGate<>(q1, param1, param2, param3, param4).GetMatrix();
      case Kind::kSwapGateType:
        return SwapGate<>(q1, q2).GetMatrix();
      case Kind::kCXGateType:
        return CXGate<>(q1, q2).GetMatrix();
      case Kind::kCYGateType:
        return CYGate<>(q1, q2).GetMatrix();
      case Kind::kCZGateType:
        return CZGate<>(q1, q2).GetMatrix();
      case Kind::kCPGateType:
        return CPGate<>(q1, q2, param1).GetMatrix();
      case Kind::kCRxGateType:
        return CRxGate<>(q1, q2, param1).GetMatrix();
      case Kind::kCRyGateType:
        return CRyGate<>(q1, q2, param1).GetMatrix();
      case Kind::kCRzGateType:
        return CRzGate<>(q1, q2, param1).GetMatrix();
      case Kind::kCHGateType:
        return CHGate<>(q1, q2).GetMatrix();
      case Kind::kCSxGateType:
        return CSxGate<>(q1, q2).GetMatrix();
      case Kind::kCSxDagGateType:
        return CSxDagGate<>(q1, q2).GetMatrix();
      case Kind::kCUGateType:
        return CUGate<>(q1, q2, param1, param2, param3, param4).GetMatrix();
      case Kind::kCSwapGateType:
        return CSwapGate<>(q1, q2, q3).GetMatrix();
      case Kind::kCCXGateType:
        return CCXGate<>(q1, q2, q3).GetMatrix();
    }
    throw std::logic_error("Unknown fusion gate");
  }
  static FusionGate FromCircuit(const Circuits::IQuantumGate<>& gate) {
    FusionGate result;
    result.kind = gate.GetGateType();
    if (result.kind == Kind::kNone) result.generic = gate.GetMatrix();
    for (size_t i = 0; i < gate.GetNumQubits(); ++i)
      result.qubits.push_back(gate.GetQubit(i));
    const auto p = gate.GetParams();
    std::copy_n(p.begin(), std::min(p.size(), result.params.size()),
                result.params.begin());
    return result;
  }
  bool Matches(const FusionGate& other) const {
    return kind == other.kind && qubits == other.qubits &&
           params == other.params &&
           (kind != Kind::kNone ||
            (generic.rows() == other.generic.rows() &&
             generic.cols() == other.generic.cols() &&
             (generic.array() == other.generic.array()).all()));
  }
  bool IsStructured() const {
    switch (kind) {
      case Kind::kPhaseGateType:
      case Kind::kXGateType:
      case Kind::kYGateType:
      case Kind::kZGateType:
      case Kind::kSGateType:
      case Kind::kSdgGateType:
      case Kind::kTGateType:
      case Kind::kTdgGateType:
      case Kind::kRzGateType:
      case Kind::kSwapGateType:
      case Kind::kCXGateType:
      case Kind::kCYGateType:
      case Kind::kCZGateType:
      case Kind::kCPGateType:
      case Kind::kCRzGateType:
      case Kind::kCSwapGateType:
      case Kind::kCCXGateType:
        return true;
      default:
        // Generic matrices already use the dense API; preserving their
        // individual calls cannot recover a specialized native gate path.
        return false;
    }
  }
  void Apply(ISimulator& sim) const {
    switch (kind) {
      case Kind::kPhaseGateType:
        sim.ApplyP(qubits[0], params[0]);
        return;
      case Kind::kXGateType:
        sim.ApplyX(qubits[0]);
        return;
      case Kind::kYGateType:
        sim.ApplyY(qubits[0]);
        return;
      case Kind::kZGateType:
        sim.ApplyZ(qubits[0]);
        return;
      case Kind::kHadamardGateType:
        sim.ApplyH(qubits[0]);
        return;
      case Kind::kSGateType:
        sim.ApplyS(qubits[0]);
        return;
      case Kind::kSdgGateType:
        sim.ApplySDG(qubits[0]);
        return;
      case Kind::kTGateType:
        sim.ApplyT(qubits[0]);
        return;
      case Kind::kTdgGateType:
        sim.ApplyTDG(qubits[0]);
        return;
      case Kind::kSxGateType:
        sim.ApplySx(qubits[0]);
        return;
      case Kind::kSxDagGateType:
        sim.ApplySxDAG(qubits[0]);
        return;
      case Kind::kKGateType:
        sim.ApplyK(qubits[0]);
        return;
      case Kind::kRxGateType:
        sim.ApplyRx(qubits[0], params[0]);
        return;
      case Kind::kRyGateType:
        sim.ApplyRy(qubits[0], params[0]);
        return;
      case Kind::kRzGateType:
        sim.ApplyRz(qubits[0], params[0]);
        return;
      case Kind::kUGateType:
        sim.ApplyU(qubits[0], params[0], params[1], params[2], params[3]);
        return;
      case Kind::kSwapGateType:
        sim.ApplySwap(qubits[0], qubits[1]);
        return;
      case Kind::kCXGateType:
        sim.ApplyCX(qubits[0], qubits[1]);
        return;
      case Kind::kCYGateType:
        sim.ApplyCY(qubits[0], qubits[1]);
        return;
      case Kind::kCZGateType:
        sim.ApplyCZ(qubits[0], qubits[1]);
        return;
      case Kind::kCPGateType:
        sim.ApplyCP(qubits[0], qubits[1], params[0]);
        return;
      case Kind::kCRxGateType:
        sim.ApplyCRx(qubits[0], qubits[1], params[0]);
        return;
      case Kind::kCRyGateType:
        sim.ApplyCRy(qubits[0], qubits[1], params[0]);
        return;
      case Kind::kCRzGateType:
        sim.ApplyCRz(qubits[0], qubits[1], params[0]);
        return;
      case Kind::kCHGateType:
        sim.ApplyCH(qubits[0], qubits[1]);
        return;
      case Kind::kCSxGateType:
        sim.ApplyCSx(qubits[0], qubits[1]);
        return;
      case Kind::kCSxDagGateType:
        sim.ApplyCSxDAG(qubits[0], qubits[1]);
        return;
      case Kind::kCUGateType:
        sim.ApplyCU(qubits[0], qubits[1], params[0], params[1], params[2],
                    params[3]);
        return;
      case Kind::kCSwapGateType:
        sim.ApplyCSwap(qubits[0], qubits[1], qubits[2]);
        return;
      case Kind::kCCXGateType:
        sim.ApplyCCX(qubits[0], qubits[1], qubits[2]);
        return;
      case Kind::kNone:
        if (qubits.size() == 1)
          sim.ApplyGenericOneQubitGate(qubits[0], generic);
        else if (qubits.size() == 2)
          sim.ApplyGenericTwoQubitGate(qubits[0], qubits[1], generic);
        else
          sim.ApplyGenericThreeQubitGate(qubits[0], qubits[1], qubits[2],
                                         generic);
        return;
    }
    throw std::logic_error("Unknown fusion gate");
  }
  std::vector<FusionGate> Expand(unsigned width) const {
    if (qubits.size() <= width) return {*this};
    if (width != 2 ||
        (kind != Kind::kCCXGateType && kind != Kind::kCSwapGateType))
      throw std::invalid_argument("Unsupported generic gate width");
    auto ccx = [](Types::qubit_t a, Types::qubit_t b, Types::qubit_t c) {
      return std::vector<FusionGate>{{Kind::kCSxGateType, {b, c}, {}, {}},
                                     {Kind::kCXGateType, {a, b}, {}, {}},
                                     {Kind::kCSxDagGateType, {b, c}, {}, {}},
                                     {Kind::kCXGateType, {a, b}, {}, {}},
                                     {Kind::kCSxGateType, {a, c}, {}, {}}};
    };
    if (kind == Kind::kCCXGateType) return ccx(qubits[0], qubits[1], qubits[2]);
    auto result = ccx(qubits[0], qubits[1], qubits[2]);
    FusionGate cx{Kind::kCXGateType, {qubits[2], qubits[1]}, {}, {}};
    result.insert(result.begin(), cx);
    result.push_back(cx);
    return result;
  }
};

// Routing-only operation. It cannot accidentally be executed as a quantum gate.
class FusionRoutingOperation : public Circuits::IOperation<> {
 public:
  explicit FusionRoutingOperation(Types::qubits_vector targets)
      : Circuits::IOperation<>(0), targets_(std::move(targets)) {}
  Types::qubits_vector AffectedQubits() const override { return targets_; }
  Circuits::OperationType GetType() const override {
    return Circuits::OperationType::kGate;
  }
  void Execute(const std::shared_ptr<ISimulator>&,
               Circuits::OperationState&) const override {
    throw std::logic_error("A routing operation cannot be executed");
  }
  std::shared_ptr<Circuits::IOperation<>> Clone() const override {
    return std::make_shared<FusionRoutingOperation>(targets_);
  }
  std::shared_ptr<Circuits::IOperation<>> Remap(
      const std::unordered_map<Types::qubit_t, Types::qubit_t>& map,
      const std::unordered_map<Types::qubit_t, Types::qubit_t>& = {})
      const override {
    auto targets = targets_;
    for (auto& q : targets) q = map.at(q);
    return std::make_shared<FusionRoutingOperation>(std::move(targets));
  }

 private:
  Types::qubits_vector targets_;
};
}  // namespace Simulators
