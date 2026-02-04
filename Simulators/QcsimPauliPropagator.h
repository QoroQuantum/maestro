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
};

}  // namespace Simulators

#endif  // _QCSIM_PAULI_PROPAGATOR_H
