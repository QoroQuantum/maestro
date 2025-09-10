/*******************************************************

Copyright (C) 2023 2639731 ONTARIO INC. <joe.diadamo.dryrock@gmail.com>

The files in this repository make up the Codebase.

All files in this Codebase are owned by 2639731 ONTARIO INC..

Any files within this Codebase can not be copied and/or distributed without the express permission of 2639731 ONTARIO INC.

*******************************************************/

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
