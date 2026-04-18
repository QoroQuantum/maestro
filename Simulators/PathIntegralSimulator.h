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
#include "../Circuit/Circuit.h"

namespace Simulators {

	class PathIntegralSimulator {

          void SetTrimValue(double val) { simulator.SetTrimValue(val); }

          double GetTrimValue() const { return simulator.GetTrimValue(); }

          void SetMaxDoublingsForBackwardPaths(size_t doublings) {
            simulator.SetMaxDoublingsForBackwardPaths(doublings);
          }

          size_t GetMaxDoublingsForBackwardPaths() const { return simulator.GetMaxDoublingsForBackwardPaths(); }

          std::vector<QC::Gates::AppliedGate<>> ConvertCircuit(const std::shared_ptr<Circuits::Circuit<>>& circuit)
          {
            std::vector<QC::Gates::AppliedGate<>> result;
            if (!circuit) return result;

            const auto& operations = circuit->GetOperations();
            result.reserve(operations.size());

            for (const auto& op : operations)
            {
              if (!op || op->GetType() != Circuits::OperationType::kGate)
                continue;

              const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<>>(op);

              switch (gate->GetGateType())
              {
                // single qubit gates
              case Circuits::QuantumGateType::kPhaseGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::PhaseGate<>>(gate);
                pgate.SetPhaseShift(g->GetLambda());
                result.emplace_back(pgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kXGateType:
                result.emplace_back(xgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kYGateType:
                result.emplace_back(ygate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kZGateType:
                result.emplace_back(zgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kHadamardGateType:
                result.emplace_back(h.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kSGateType:
                result.emplace_back(sgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kSdgGateType:
                result.emplace_back(sdggate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kTGateType:
                result.emplace_back(tgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kTdgGateType:
                result.emplace_back(tdggate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kSxGateType:
                result.emplace_back(sxgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kSxDagGateType:
                result.emplace_back(sxdaggate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kKGateType:
                result.emplace_back(k.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kRxGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::RxGate<>>(gate);
                rxgate.SetTheta(g->GetTheta());
                result.emplace_back(rxgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kRyGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::RyGate<>>(gate);
                rygate.SetTheta(g->GetTheta());
                result.emplace_back(rygate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kRzGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::RzGate<>>(gate);
                rzgate.SetTheta(g->GetTheta());
                result.emplace_back(rzgate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kUGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::UGate<>>(gate);
                ugate.SetParams(g->GetTheta(), g->GetPhi(), g->GetLambda(), g->GetGamma());
                result.emplace_back(ugate.getRawOperatorMatrix(), gate->GetQubit(0));
                break;
              }

              // two qubit gates
              case Circuits::QuantumGateType::kSwapGateType:
                result.emplace_back(swapgate.getRawOperatorMatrix(), gate->GetQubit(0), gate->GetQubit(1));
                break;
              case Circuits::QuantumGateType::kCXGateType:
                result.emplace_back(cxgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCYGateType:
                result.emplace_back(cygate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCZGateType:
                result.emplace_back(czgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCPGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::CPGate<>>(gate);
                cpgate.SetPhaseShift(g->GetLambda());
                result.emplace_back(cpgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kCRxGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::CRxGate<>>(gate);
                crxgate.SetTheta(g->GetTheta());
                result.emplace_back(crxgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kCRyGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::CRyGate<>>(gate);
                crygate.SetTheta(g->GetTheta());
                result.emplace_back(crygate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kCRzGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::CRzGate<>>(gate);
                crzgate.SetTheta(g->GetTheta());
                result.emplace_back(crzgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              }
              case Circuits::QuantumGateType::kCHGateType:
                result.emplace_back(ch.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCSxGateType:
                result.emplace_back(csx.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCSxDagGateType:
                result.emplace_back(csxdag.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCUGateType:
              {
                const auto g = std::static_pointer_cast<Circuits::CUGate<>>(gate);
                cugate.SetParams(g->GetTheta(), g->GetPhi(), g->GetLambda(), g->GetGamma());
                result.emplace_back(cugate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
                break;
              }

              // three qubit gates
              case Circuits::QuantumGateType::kCCXGateType:
                result.emplace_back(ccxgate.getRawOperatorMatrix(), gate->GetQubit(2), gate->GetQubit(1), gate->GetQubit(0));
                break;
              case Circuits::QuantumGateType::kCSwapGateType:
                result.emplace_back(cswapgate.getRawOperatorMatrix(), gate->GetQubit(2), gate->GetQubit(1), gate->GetQubit(0));
                break;

              default:
                break;
              }
            }

            return result;
          }

          static size_t GetBranching(const std::vector<QC::Gates::AppliedGate<>>& circuit) {
            size_t doublings = 0;
            for (const auto& gate : circuit) 
              if (gate.isBranching()) ++doublings;

            return doublings;
          }

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

