#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "ImmediateQCSimSimulator.h"
#include "QuantumGate.h"

namespace Simulators::Private
{
// Gate storage for backends that consume QCSim gate matrices. Backends with
// native named-gate APIs derive directly from ImmediateQCSimSimulator instead.
class QCSimGateSimulator : public ImmediateQCSimSimulator
{
  protected:
    // These gates are never modified, and their matrix accessors return const
    // references. Sharing them avoids allocation on construction and cloning.
    inline static const QC::Gates::PauliXGate<> xgate;
    inline static const QC::Gates::PauliYGate<> ygate;
    inline static const QC::Gates::PauliZGate<> zgate;
    inline static const QC::Gates::HadamardGate<> h;
    inline static const QC::Gates::SGate<> sgate;
    inline static const QC::Gates::SDGGate<> sdggate;
    inline static const QC::Gates::TGate<> tgate;
    inline static const QC::Gates::TDGGate<> tdggate;
    inline static const QC::Gates::SquareRootNOTGate<> sxgate;
    inline static const QC::Gates::SquareRootNOTDagGate<> sxdaggate;
    inline static const QC::Gates::HyGate<> k;
    inline static const QC::Gates::CNOTGate<> cxgate;
    inline static const QC::Gates::ControlledYGate<> cygate;
    inline static const QC::Gates::ControlledZGate<> czgate;
    inline static const QC::Gates::ControlledHadamardGate<> ch;
    inline static const QC::Gates::ControlledSquareRootNOTGate<> csx;
    inline static const QC::Gates::ControlledSquareRootNOTDagGate<> csxdag;
    inline static const QC::Gates::SwapGate<> swapgate;
    inline static const QC::Gates::ToffoliGate<> ccxgate;
    inline static const QC::Gates::FredkinGate<> cswapgate;

    // Setters update these matrices before each application. Keep them local
    // to each simulator so independent instances and clones cannot interfere.
    QC::Gates::PhaseShiftGate<> pgate;
    QC::Gates::RxGate<> rxgate;
    QC::Gates::RyGate<> rygate;
    QC::Gates::RzGate<> rzgate;
    QC::Gates::UGate<> ugate;
    QC::Gates::ControlledPhaseShiftGate<> cpgate;
    QC::Gates::ControlledRxGate<> crxgate;
    QC::Gates::ControlledRyGate<> crygate;
    QC::Gates::ControlledRzGate<> crzgate;
    QC::Gates::ControlledUGate<> cugate;
};
} // namespace Simulators::Private
#endif
