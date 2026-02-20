/**
 * @file QuestSimulator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The quest simulator class.
 *
 * Should not be used directly, create an instance with the factory and use the
 * generic interface instead.
 */

#pragma once

#ifndef _QUESTSIMULATOR_H
#define _QUESTSIMULATOR_H

#ifdef INCLUDED_BY_FACTORY

#include "QuestState.h"

namespace Simulators {

namespace Private {
/**
 * @class QuestSimulator
 * @brief Quest simulator class.
 *
 * This is the implementation for the quest simulator.
 * Do not use this class directly, use the factory to create an instance.
 * Only the interface should be exposed.
 * @sa QuestState
 * @sa ISimulator
 * @sa IState
 */
class QuestSimulator : public QuestState {
 public:
  QuestSimulator() = default;

  // allow no copy or assignment
  QuestSimulator(const QuestSimulator &) = delete;
  QuestSimulator &operator=(const QuestSimulator &) = delete;

  // but allow moving
  QuestSimulator(QuestSimulator &&other) = default;
  QuestSimulator &operator=(QuestSimulator &&other) = default;

  /**
   * @brief Applies a phase shift gate to the qubit
   *
   * Applies a specified phase shift gate to the qubit
   * @param qubit The qubit to apply the gate to.
   * @param lambda The phase shift angle.
   */
  void ApplyP(Types::qubit_t qubit, double lambda) override {
    questLib->ApplyP(sim, static_cast<int>(qubit), lambda);
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a not gate to the qubit
   *
   * Applies a not (X) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyX(Types::qubit_t qubit) override {
    questLib->ApplyX(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Y gate to the qubit
   *
   * Applies a not (Y) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyY(Types::qubit_t qubit) override {
    questLib->ApplyY(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Z gate to the qubit
   *
   * Applies a not (Z) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyZ(Types::qubit_t qubit) override {
    questLib->ApplyZ(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Hadamard gate to the qubit
   *
   * Applies a Hadamard gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyH(Types::qubit_t qubit) override {
    questLib->ApplyH(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a S gate to the qubit
   *
   * Applies a S gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyS(Types::qubit_t qubit) override {
    questLib->ApplyS(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a S dagger gate to the qubit
   *
   * Applies a S dagger gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplySDG(Types::qubit_t qubit) override {
    questLib->ApplySdg(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a T gate to the qubit
   *
   * Applies a T gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyT(Types::qubit_t qubit) override {
    questLib->ApplyT(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a T dagger gate to the qubit
   *
   * Applies a T dagger gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyTDG(Types::qubit_t qubit) override {
    questLib->ApplyTdg(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Sx gate to the qubit
   *
   * Applies a Sx gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplySx(Types::qubit_t qubit) override {
    questLib->ApplySx(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Sx dagger gate to the qubit
   *
   * Applies a Sx dagger gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplySxDAG(Types::qubit_t qubit) override {
    questLib->ApplySxDg(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a K gate to the qubit
   *
   * Applies a K (Hy) gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   */
  void ApplyK(Types::qubit_t qubit) override {
    questLib->ApplyK(sim, static_cast<int>(qubit));
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Rx gate to the qubit
   *
   * Applies an x rotation gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   */
  void ApplyRx(Types::qubit_t qubit, double theta) override {
    questLib->ApplyRx(sim, static_cast<int>(qubit), theta);
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Ry gate to the qubit
   *
   * Applies a y rotation gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   */
  void ApplyRy(Types::qubit_t qubit, double theta) override {
    questLib->ApplyRy(sim, static_cast<int>(qubit), theta);
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a Rz gate to the qubit
   *
   * Applies a z rotation gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   */
  void ApplyRz(Types::qubit_t qubit, double theta) override {
    questLib->ApplyRz(sim, static_cast<int>(qubit), theta);
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a U gate to the qubit
   *
   * Applies a U gate to the specified qubit
   * @param qubit The qubit to apply the gate to.
   * @param theta The rotation angle.
   * @param phi The phase angle.
   * @param lambda The lambda angle.
   * @param gamma The gamma angle.
   */
  void ApplyU(Types::qubit_t qubit, double theta, double phi, double lambda,
              double gamma) override {
    questLib->ApplyU(sim, static_cast<int>(qubit), theta, phi, lambda, gamma);
    NotifyObservers({qubit});
  }

  /**
   * @brief Applies a CX gate to the qubits
   *
   * Applies a controlled X gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCX(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    questLib->ApplyCX(sim, static_cast<int>(ctrl_qubit),
                      static_cast<int>(tgt_qubit));
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CY gate to the qubits
   *
   * Applies a controlled Y gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCY(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    questLib->ApplyCY(sim, static_cast<int>(ctrl_qubit),
                      static_cast<int>(tgt_qubit));
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CZ gate to the qubits
   *
   * Applies a controlled Z gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCZ(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    questLib->ApplyCZ(sim, static_cast<int>(ctrl_qubit),
                      static_cast<int>(tgt_qubit));
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CP gate to the qubits
   *
   * Applies a controlled phase gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param lambda The phase shift angle.
   */
  void ApplyCP(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
               double lambda) override {
    questLib->ApplyCP(sim, static_cast<int>(ctrl_qubit),
                      static_cast<int>(tgt_qubit), lambda);
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CRx gate to the qubits
   *
   * Applies a controlled x rotation gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta The rotation angle.
   */
  void ApplyCRx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    questLib->ApplyCRx(sim, static_cast<int>(ctrl_qubit),
                       static_cast<int>(tgt_qubit), theta);
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CRy gate to the qubits
   *
   * Applies a controlled y rotation gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta The rotation angle.
   */
  void ApplyCRy(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    questLib->ApplyCRy(sim, static_cast<int>(ctrl_qubit),
                       static_cast<int>(tgt_qubit), theta);
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CRz gate to the qubits
   *
   * Applies a controlled z rotation gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta The rotation angle.
   */
  void ApplyCRz(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
                double theta) override {
    questLib->ApplyCRz(sim, static_cast<int>(ctrl_qubit),
                       static_cast<int>(tgt_qubit), theta);
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CH gate to the qubits
   *
   * Applies a controlled Hadamard gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCH(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    questLib->ApplyCH(sim, static_cast<int>(ctrl_qubit),
                      static_cast<int>(tgt_qubit));
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CSx gate to the qubits
   *
   * Applies a controlled squared root not gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCSx(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit) override {
    questLib->ApplyCSx(sim, static_cast<int>(ctrl_qubit),
                       static_cast<int>(tgt_qubit));
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a CSx dagger gate to the qubits
   *
   * Applies a controlled squared root not dagger gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   */
  void ApplyCSxDAG(Types::qubit_t ctrl_qubit,
                   Types::qubit_t tgt_qubit) override {
    questLib->ApplyCSxDg(sim, static_cast<int>(ctrl_qubit),
                         static_cast<int>(tgt_qubit));
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a swap gate to the qubits
   *
   * Applies a swap gate to the specified qubits
   * @param qubit0 The first qubit
   * @param qubit1 The second qubit
   */
  void ApplySwap(Types::qubit_t qubit0, Types::qubit_t qubit1) override {
    questLib->ApplySwap(sim, static_cast<int>(qubit0),
                        static_cast<int>(qubit1));
    NotifyObservers({qubit1, qubit0});
  }

  /**
   * @brief Applies a controlled controlled not gate to the qubits
   *
   * Applies a controlled controlled not gate to the specified qubits
   * @param qubit0 The first control qubit
   * @param qubit1 The second control qubit
   * @param qubit2 The target qubit
   */
  void ApplyCCX(Types::qubit_t qubit0, Types::qubit_t qubit1,
                Types::qubit_t qubit2) override {
    questLib->ApplyCCX(sim, static_cast<int>(qubit0),
                       static_cast<int>(qubit1), static_cast<int>(qubit2));
    NotifyObservers({qubit0, qubit1, qubit2});
  }

  /**
   * @brief Applies a controlled swap gate to the qubits
   *
   * Applies a controlled swap gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param qubit0 The first qubit
   * @param qubit1 The second qubit
   */
  void ApplyCSwap(Types::qubit_t ctrl_qubit, Types::qubit_t qubit0,
                  Types::qubit_t qubit1) override {
    questLib->ApplyCSwap(sim, static_cast<int>(ctrl_qubit),
                         static_cast<int>(qubit0), static_cast<int>(qubit1));
    NotifyObservers({qubit1, qubit0, ctrl_qubit});
  }

  /**
   * @brief Applies a controlled U gate to the qubits
   *
   * Applies a controlled U gate to the specified qubits
   * @param ctrl_qubit The control qubit
   * @param tgt_qubit The target qubit
   * @param theta Theta parameter for the U gate
   * @param phi Phi parameter for the U gate
   * @param lambda Lambda parameter for the U gate
   * @param gamma Gamma parameter for the U gate
   */
  void ApplyCU(Types::qubit_t ctrl_qubit, Types::qubit_t tgt_qubit,
               double theta, double phi, double lambda, double gamma) override {
    questLib->ApplyCU(sim, static_cast<int>(ctrl_qubit),
                      static_cast<int>(tgt_qubit), theta, phi, lambda, gamma);
    NotifyObservers({tgt_qubit, ctrl_qubit});
  }

  /**
   * @brief Applies a nop
   *
   * Applies a nop (no operation).
   * Typically does (almost) nothing. Equivalent to an identity.
   */
  void ApplyNop() override {
    // do nothing
  }

  /**
   * @brief Clones the simulator.
   *
   * Clones the simulator, including the state, the configuration and the
   * internally saved state, if any. Does not copy the observers. Should be used
   * mainly internally, to optimise multiple shots execution, copying the state
   * from the simulator used for timing.
   *
   * @return A unique pointer to the cloned simulator.
   */
  std::unique_ptr<ISimulator> Clone() override {
    auto cloned = std::make_unique<QuestSimulator>();
    cloned->questLib = questLib;
    cloned->nrQubits = nrQubits;

    if (sim) {
      cloned->simHandle = questLib->CloneSimulator(sim);
      cloned->sim = questLib->GetSimulator(cloned->simHandle);
    }

    return cloned;
  }
};
}  // namespace Private
}  // namespace Simulators

#endif

#endif  // _QUESTSIMULATOR_H
