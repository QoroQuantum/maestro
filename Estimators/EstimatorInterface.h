/**
 * @file EstimatorInterface.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Interface class for the execution estimators, exposes a common interface.
 */

#pragma once

#ifndef __ESTIMATOR_INTERFACE_H_
#define __ESTIMATOR_INTERFACE_H_

#include "Simulators/Simulator.h"

namespace Estimators {

	class EstimatorInterface {
	public:
		virtual ~EstimatorInterface() = default;

		virtual double EstimateTime(Simulators::SimulatorType type, Simulators::SimulationType method) const = 0;
	};
}

#endif // !__ESTIMATOR_INTERFACE_H_
