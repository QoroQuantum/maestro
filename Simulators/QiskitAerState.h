/**
 * @file QiskitAerState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The qiskit aer state class, derived from AER::AerState.
 *
 * Should not be used directly, this should allow exposing more from AER::AerState than it's available through the public interface, if needed.
 */

#pragma once

#ifndef _QISKIT_AER_STATE_H_
#define _QISKIT_AER_STATE_H_

#ifndef NO_QISKIT_AER

#ifdef INCLUDED_BY_FACTORY

#define private protected
#include "controllers/state_controller.hpp"
#undef private

namespace Simulators {
	// TODO: Maybe use the pimpl idiom https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation for good
	// but during development this should be good enough
	namespace Private {

		/**
         * @class QiskitAerState
         * @brief Class for the qiskit aer simulator.
         *
         * Derived from AER::AerState, could add more functionality and expose more than it's available through the base class public interface.
         * Do not use this class directly.
         */
		class QiskitAerState : public AER::AerState
		{
		public:
			double expval_pauli(const reg_t& qubits, const std::string& pauli)
			{
				if (!state_) return 0.0;

				return state_->expval_pauli(qubits, pauli);
			}
		};

	}
}

#endif
#endif
#endif
