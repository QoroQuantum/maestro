/**
 * @file PathIntegralSimulator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The path integral simulator class.
 */

#pragma once

#ifndef _PATH_INTEGRAL_SIMULATOR_H_
#define _PATH_INTEGRAL_SIMULATOR_H_

#include "PathIntegral.h"

namespace Simulators {

	class PathIntegralSimulator {

          void SetTrimValue(double val) { simulator.SetTrimValue(val); }

          double GetTrimValue() const { return simulator.GetTrimValue(); }

          void SetMaxDoublingsForBackwardPaths(size_t doublings) {
            simulator.SetMaxDoublingsForBackwardPaths(doublings);
          }

          size_t GetMaxDoublingsForBackwardPaths() const { return simulator.GetMaxDoublingsForBackwardPaths(); }

		private:
          QC::PathIntegral::PathIntegralSimulator simulator;

          QC::Gates::PhaseShiftGate<> pgate;
          QC::Gates::PauliXGate<> xgate;
          QC::Gates::PauliYGate<> ygate;
          QC::Gates::PauliZGate<> zgate;
          QC::Gates::HadamardGate<> h;
          // QC::Gates::UGate<> ugate;
          QC::Gates::SGate<> sgate;
          QC::Gates::SDGGate<> sdggate;
          QC::Gates::TGate<> tgate;
          QC::Gates::TDGGate<> tdggate;
          QC::Gates::SquareRootNOTGate<> sxgate;
          QC::Gates::SquareRootNOTDagGate<> sxdaggate;
          QC::Gates::HyGate<> k;
          QC::Gates::RxGate<> rxgate;
          QC::Gates::RyGate<> rygate;
          QC::Gates::RzGate<> rzgate;
          QC::Gates::UGate<> ugate;
          QC::Gates::CNOTGate<> cxgate;
          QC::Gates::ControlledYGate<> cygate;
          QC::Gates::ControlledZGate<> czgate;
          QC::Gates::ControlledPhaseShiftGate<> cpgate;
          QC::Gates::ControlledRxGate<> crxgate;
          QC::Gates::ControlledRyGate<> crygate;
          QC::Gates::ControlledRzGate<> crzgate;
          QC::Gates::ControlledHadamardGate<> ch;
          QC::Gates::ControlledSquareRootNOTGate<> csx;
          QC::Gates::ControlledSquareRootNOTDagGate<> csxdag;
          QC::Gates::SwapGate<> swapgate;
          QC::Gates::ToffoliGate<> ccxgate;
          QC::Gates::FredkinGate<> cswapgate;
          QC::Gates::ControlledUGate<> cugate;
	};

}

#endif  // _PATH_INTEGRAL_SIMULATOR_H_

