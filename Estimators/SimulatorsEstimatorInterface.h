/*******************************************************

Copyright (C) 2023 2639731 ONTARIO INC. <joe.diadamo.dryrock@gmail.com>

The files in this repository make up the Codebase.

All files in this Codebase are owned by 2639731 ONTARIO INC..

Any files within this Codebase can not be copied and/or distributed without the express permission of 2639731 ONTARIO INC.

*******************************************************/

/**
 * @file SimulatorsEstimatorInterface.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Interface class for the simulators estimators, exposes a common interface.
 */

#pragma once

#ifndef __SIMULATORS_ESTIMATOR_INTERFACE_H_
#define __SIMULATORS_ESTIMATOR_INTERFACE_H_

#include "Simulators/Simulator.h"
#include "Circuit/Circuit.h"

namespace Estimators {

	template<typename Time = Types::time_type> class SimulatorsEstimatorInterface {
	public:
		virtual ~SimulatorsEstimatorInterface() = default;

		virtual std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(const std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>>& simulatorTypes, const std::shared_ptr<Circuits::Circuit<Time>>& dcirc, 
			size_t& counts, size_t nrQubits, size_t nrCbits, size_t nrResultCbits, 
			Simulators::SimulatorType& simType, Simulators::SimulationType& method, std::vector<bool>& executed, 
			const std::string& maxBondDim, const std::string& singularValueThreshold, const std::string& mpsSample,
			size_t maxSimulators,
			bool multithreading = false) const = 0;
	};

}

#endif // !__SIMULATORS_ESTIMATOR_INTERFACE_H_
