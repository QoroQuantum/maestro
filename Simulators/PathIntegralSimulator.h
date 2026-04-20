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

#include <optional>

#include "PathIntegral.h"
#include "../Circuit/Circuit.h"

namespace Simulators {

	class PathIntegralSimulator {
		public:
		  void SetTrimValue(double val) { simulator.SetTrimValue(val); }

          double GetTrimValue() const { return simulator.GetTrimValue(); }

          void SetMaxDoublingsForBackwardPaths(size_t doublings) {
            simulator.SetMaxDoublingsForBackwardPaths(doublings);
          }

          size_t GetMaxDoublingsForBackwardPaths() const { return simulator.GetMaxDoublingsForBackwardPaths(); }

          bool SetCircuit(const std::shared_ptr<Circuits::Circuit<>>& circuit)
          {
            const auto convertedCircuit = ConvertCircuit(circuit);
            if (convertedCircuit.empty()) return false;
            simulator.SetCircuit(convertedCircuit);
            return true;
          }

          std::complex<double> AmplitudeFromZero(const std::vector<bool>& endState)
          {
            return simulator.Propagate(endState);
          }

          std::complex<double> Amplitude(const std::vector<bool>& startState, const std::vector<bool>& endState)
          {
            return simulator.Propagate(startState, endState);
          }

          std::optional<QC::Gates::AppliedGate<>> ConvertGate(const std::shared_ptr<Circuits::IQuantumGate<>>& gate)
          {
            if (!gate) return std::nullopt;

            switch (gate->GetGateType())
            {
              // single qubit gates
            case Circuits::QuantumGateType::kPhaseGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::PhaseGate<>>(gate);
              pgate.SetPhaseShift(g->GetLambda());
              return QC::Gates::AppliedGate<>(pgate.getRawOperatorMatrix(), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kXGateType:
              return QC::Gates::AppliedGate<>(xgate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kYGateType:
              return QC::Gates::AppliedGate<>(ygate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kZGateType:
              return QC::Gates::AppliedGate<>(zgate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kHadamardGateType:
              return QC::Gates::AppliedGate<>(h.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kSGateType:
              return QC::Gates::AppliedGate<>(sgate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kSdgGateType:
              return QC::Gates::AppliedGate<>(sdggate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kTGateType:
              return QC::Gates::AppliedGate<>(tgate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kTdgGateType:
              return QC::Gates::AppliedGate<>(tdggate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kSxGateType:
              return QC::Gates::AppliedGate<>(sxgate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kSxDagGateType:
              return QC::Gates::AppliedGate<>(sxdaggate.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kKGateType:
              return QC::Gates::AppliedGate<>(k.getRawOperatorMatrix(), gate->GetQubit(0));
            case Circuits::QuantumGateType::kRxGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::RxGate<>>(gate);
              rxgate.SetTheta(g->GetTheta());
              return QC::Gates::AppliedGate<>(rxgate.getRawOperatorMatrix(), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kRyGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::RyGate<>>(gate);
              rygate.SetTheta(g->GetTheta());
              return QC::Gates::AppliedGate<>(rygate.getRawOperatorMatrix(), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kRzGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::RzGate<>>(gate);
              rzgate.SetTheta(g->GetTheta());
              return QC::Gates::AppliedGate<>(rzgate.getRawOperatorMatrix(), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kUGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::UGate<>>(gate);
              ugate.SetParams(g->GetTheta(), g->GetPhi(), g->GetLambda(), g->GetGamma());
              return QC::Gates::AppliedGate<>(ugate.getRawOperatorMatrix(), gate->GetQubit(0));
            }

            // two qubit gates
            case Circuits::QuantumGateType::kSwapGateType:
              return QC::Gates::AppliedGate<>(swapgate.getRawOperatorMatrix(), gate->GetQubit(0), gate->GetQubit(1));
            case Circuits::QuantumGateType::kCXGateType:
              return QC::Gates::AppliedGate<>(cxgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCYGateType:
              return QC::Gates::AppliedGate<>(cygate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCZGateType:
              return QC::Gates::AppliedGate<>(czgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCPGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::CPGate<>>(gate);
              cpgate.SetPhaseShift(g->GetLambda());
              return QC::Gates::AppliedGate<>(cpgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kCRxGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::CRxGate<>>(gate);
              crxgate.SetTheta(g->GetTheta());
              return QC::Gates::AppliedGate<>(crxgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kCRyGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::CRyGate<>>(gate);
              crygate.SetTheta(g->GetTheta());
              return QC::Gates::AppliedGate<>(crygate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kCRzGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::CRzGate<>>(gate);
              crzgate.SetTheta(g->GetTheta());
              return QC::Gates::AppliedGate<>(crzgate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            }
            case Circuits::QuantumGateType::kCHGateType:
              return QC::Gates::AppliedGate<>(ch.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCSxGateType:
              return QC::Gates::AppliedGate<>(csx.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCSxDagGateType:
              return QC::Gates::AppliedGate<>(csxdag.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCUGateType:
            {
              const auto g = std::static_pointer_cast<Circuits::CUGate<>>(gate);
              cugate.SetParams(g->GetTheta(), g->GetPhi(), g->GetLambda(), g->GetGamma());
              return QC::Gates::AppliedGate<>(cugate.getRawOperatorMatrix(), gate->GetQubit(1), gate->GetQubit(0));
            }

            // three qubit gates
            case Circuits::QuantumGateType::kCCXGateType:
              return QC::Gates::AppliedGate<>(ccxgate.getRawOperatorMatrix(), gate->GetQubit(2), gate->GetQubit(1), gate->GetQubit(0));
            case Circuits::QuantumGateType::kCSwapGateType:
              return QC::Gates::AppliedGate<>(cswapgate.getRawOperatorMatrix(), gate->GetQubit(2), gate->GetQubit(1), gate->GetQubit(0));

            default:
              return std::nullopt;
            }
          }

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
              if (auto applied = ConvertGate(gate))
                result.emplace_back(std::move(*applied));
            }

            return result;
          }

          static size_t GetBranchingForQcsimCircuit(const std::vector<QC::Gates::AppliedGate<>>& circuit) {
            size_t doublings = 0;
            for (const auto& gate : circuit) 
              if (gate.isBranching()) ++doublings;

            return doublings;
          }

          size_t GetBranchingForMaestroCircuit(const std::shared_ptr<Circuits::Circuit<>>& circuit) {
            const auto circ = ConvertCircuit(circuit);

            return GetBranchingForQcsimCircuit(circ);
          }

          double QubitProbability(size_t qubit, bool value = true) {
            return simulator.QubitProbability(qubit, value);
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

