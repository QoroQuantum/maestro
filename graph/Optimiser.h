/*******************************************************

Copyright (C) 2023 2639731 ONTARIO INC. <joe.diadamo.dryrock@gmail.com>

The files in this repository make up the Codebase.

All files in this Codebase are owned by 2639731 ONTARIO INC..

Any files within this Codebase can not be copied and/or distributed without the express permission of 2639731 ONTARIO INC.

*******************************************************/

/**
 * @file Optimiser.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The optimiser interface for graph optimisers.
 */

#pragma once

#ifndef _OPTIMISER_H_
#define _OPTIMISER_H_

#include "Network/Network.h"

namespace Graphs {

	enum class OptimiserType {
		kMonteCarlo,
		kGreedy,
		kOptimal,
		kClifford,
		kNone
	};

	template<typename Time = Types::time_type> class IOptimiser : public std::enable_shared_from_this<IOptimiser<Time>>
	{
	public:
		virtual ~IOptimiser() = default;

		virtual void SetNetworkAndCircuit(const std::shared_ptr<Network::INetwork<Time>>& network, const std::shared_ptr<Circuits::Circuit<Time>>& circuit) = 0;

		virtual size_t GetNumCuts() const = 0;

		virtual size_t Optimise(size_t numSteps = 10000) = 0;


		virtual const std::unordered_map<Types::qubit_t, Types::qubit_t>& GetQubitsMap() const = 0;

		virtual const std::unordered_map<Types::qubit_t, Types::qubit_t>& GetReverseQubitsMap() const = 0;

		virtual Types::qubit_t TranslateQubitToOriginal(Types::qubit_t qubit) const = 0;

		virtual Types::qubit_t TranslateQubitFromOriginal(Types::qubit_t qubit) const = 0;

		virtual size_t TranslateStateToOriginal(size_t state) const = 0;

		virtual size_t TranslateStateFromOriginal(size_t state) const = 0;

		/**
		 * @brief Get a shared pointer to this object.
		 *
		 * Returns a shared pointer to this object.
		 * The object needs to be already wrapped in a shared pointer.
		 * @return A shared pointer to this object.
		 */
		std::shared_ptr<IOptimiser<Time>> getptr()
		{
			return std::enable_shared_from_this<IOptimiser<Time>>::shared_from_this();
		}
	};

}


#endif // _OPTIMISER_H_
