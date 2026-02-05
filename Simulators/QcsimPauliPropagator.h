/**
 * @file QcsimPauliPropagator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The qcsim pauli propagator class.
 *
 * The main role is to extend the qcsim pauli propagator with more gates.
 */

#pragma once

#ifndef _QCSIM_PAULI_PROPAGATOR_H
#define _QCSIM_PAULI_PROPAGATOR_H 1

#include "PauliPropagator.h"


namespace Simulators {

class QcsimPauliPropagator : public QC::PauliPropagator {
 public:
  void ApplyP(int qubit, double lambda) { ApplyRZ(qubit, lambda); }

  void ApplyT(int qubit) { ApplyRZ(qubit, M_PI / 4); }

  void ApplyTDG(int qubit) { ApplyRZ(qubit, -M_PI / 4); }

  void ApplyU(int qubit, double theta, double phi, double lambda) {
    ApplyRZ(qubit, lambda);
    ApplyRY(qubit, theta);
    ApplyRZ(qubit, phi);
  }

  void ApplyCH(int controlQubit, int targetQubit) {
    ApplyH(targetQubit);
    ApplySDG(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyH(targetQubit);
    ApplyT(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyT(targetQubit);
    ApplyH(targetQubit);
    ApplyS(targetQubit);
    ApplyX(targetQubit);
    ApplyS(controlQubit);
  }

  void ApplyCU(int controlQubit, int targetQubit, double theta, double phi,
               double lambda) {
    ApplyRZ(targetQubit, (lambda - phi) * 0.5);
    ApplyCX(controlQubit, targetQubit);

    ApplyRY(targetQubit, -theta * 0.5);
    ApplyRZ(targetQubit, -(phi + lambda) * 0.5);

    ApplyCX(controlQubit, targetQubit);

    ApplyRZ(targetQubit, phi);
    ApplyRY(targetQubit, theta * 0.5);
  }

  void ApplyCRX(int controlQubit, int targetQubit, double angle) {
    const double halfAngle = angle * 0.5;
    ApplyRX(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRX(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);

  }

  void ApplyCRY(int controlQubit, int targetQubit, double angle) {
    const double halfAngle = angle * 0.5;
    ApplyRY(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRY(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCRZ(int controlQubit, int targetQubit, double angle) {
    const double halfAngle = angle * 0.5;
    ApplyRZ(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
  }


  void ApplyCP(int controlQubit, int targetQubit, double lambda) {
    ApplyCRZ(controlQubit, targetQubit, lambda);
  }

  void ApplyCSX(int controlQubit, int targetQubit) {
    ApplyCRX(controlQubit, targetQubit, M_PI_2);
  }

  void ApplyCSXDAG(int controlQubit, int targetQubit) {
    ApplyCRX(controlQubit, targetQubit, -M_PI_2);
  }

  void ApplyCSwap(int controlQubit, int targetQubit1, int targetQubit2) {
    // TODO: implement this gate
  }

  void ApplyCCX(int controlQubit1, int controlQubit2, int targetQubit) {
    // TODO: implement this gate
  }
};

}  // namespace Simulators

#endif  // _QCSIM_PAULI_PROPAGATOR_H
