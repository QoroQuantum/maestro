/**
 * @file qcsimextendedstabilizertests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the Maestro QCSim extended stabilizer wrapper.
 */

#include <boost/test/unit_test.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../Simulators/Factory.h"
#include "../Simulators/QCSimExtendedStabilizer.h"

namespace {

constexpr size_t kNrQubits = 3;
constexpr double kTolerance = 1e-9;
constexpr double kPi = 3.141592653589793238462643383279502884;

enum class Gate {
  kX,
  kY,
  kZ,
  kH,
  kS,
  kSDG,
  kSX,
  kSXDG,
  kK,
  kP,
  kT,
  kTDG,
  kRX,
  kRY,
  kRZ,
  kU,
  kCX,
  kCY,
  kCZ,
  kSwap,
  kISwap,
  kISwapDG,
  kCH,
  kCRX,
  kCRY,
  kCRZ,
  kCP,
  kCS,
  kCSDAG,
  kCSX,
  kCSXDAG,
  kCU,
  kCCX,
  kCSwap,
};

struct GateCase {
  Gate gate;
  const char* name;
};

const std::array<GateCase, 34> kGateCases{{
    {Gate::kX, "X"},           {Gate::kY, "Y"},
    {Gate::kZ, "Z"},           {Gate::kH, "H"},
    {Gate::kS, "S"},           {Gate::kSDG, "SDG"},
    {Gate::kSX, "SX"},         {Gate::kSXDG, "SXDG"},
    {Gate::kK, "K"},           {Gate::kP, "P"},
    {Gate::kT, "T"},           {Gate::kTDG, "TDG"},
    {Gate::kRX, "RX"},         {Gate::kRY, "RY"},
    {Gate::kRZ, "RZ"},         {Gate::kU, "U"},
    {Gate::kCX, "CX"},         {Gate::kCY, "CY"},
    {Gate::kCZ, "CZ"},         {Gate::kSwap, "SWAP"},
    {Gate::kISwap, "ISWAP"},   {Gate::kISwapDG, "ISWAPDG"},
    {Gate::kCH, "CH"},         {Gate::kCRX, "CRX"},
    {Gate::kCRY, "CRY"},       {Gate::kCRZ, "CRZ"},
    {Gate::kCP, "CP"},         {Gate::kCS, "CS"},
    {Gate::kCSDAG, "CSDAG"},   {Gate::kCSX, "CSX"},
    {Gate::kCSXDAG, "CSXDAG"}, {Gate::kCU, "CU"},
    {Gate::kCCX, "CCX"},       {Gate::kCSwap, "CSWAP"},
}};

std::shared_ptr<Simulators::ISimulator> CreateStatevector() {
  auto statevector = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  statevector->AllocateQubits(kNrQubits);
  statevector->Initialize();
  return statevector;
}

void PrepareReferenceState(
    Simulators::QCSimExtendedStabilizer& extendedStabilizer,
    const std::shared_ptr<Simulators::ISimulator>& statevector) {
  extendedStabilizer.ApplyH(0);
  statevector->ApplyH(0);

  extendedStabilizer.ApplyRY(1, 0.37);
  statevector->ApplyRy(1, 0.37);

  extendedStabilizer.ApplyRX(2, -0.29);
  statevector->ApplyRx(2, -0.29);

  extendedStabilizer.ApplyCX(0, 1);
  statevector->ApplyCX(0, 1);

  extendedStabilizer.ApplyRZ(0, 0.21);
  statevector->ApplyRz(0, 0.21);

  extendedStabilizer.ApplyCX(1, 2);
  statevector->ApplyCX(1, 2);
}

void ApplyGate(Simulators::QCSimExtendedStabilizer& simulator, Gate gate) {
  switch (gate) {
    case Gate::kX:
      simulator.ApplyX(0);
      break;
    case Gate::kY:
      simulator.ApplyY(0);
      break;
    case Gate::kZ:
      simulator.ApplyZ(0);
      break;
    case Gate::kH:
      simulator.ApplyH(0);
      break;
    case Gate::kS:
      simulator.ApplyS(0);
      break;
    case Gate::kSDG:
      simulator.ApplySDG(0);
      break;
    case Gate::kSX:
      simulator.ApplySX(0);
      break;
    case Gate::kSXDG:
      simulator.ApplySXDG(0);
      break;
    case Gate::kK:
      simulator.ApplyK(0);
      break;
    case Gate::kP:
      simulator.ApplyP(0, 0.43);
      break;
    case Gate::kT:
      simulator.ApplyT(0);
      break;
    case Gate::kTDG:
      simulator.ApplyTDG(0);
      break;
    case Gate::kRX:
      simulator.ApplyRX(0, -0.31);
      break;
    case Gate::kRY:
      simulator.ApplyRY(0, 0.47);
      break;
    case Gate::kRZ:
      simulator.ApplyRZ(0, -0.53);
      break;
    case Gate::kU:
      simulator.ApplyU(0, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCX:
      simulator.ApplyCX(0, 1);
      break;
    case Gate::kCY:
      simulator.ApplyCY(0, 1);
      break;
    case Gate::kCZ:
      simulator.ApplyCZ(0, 1);
      break;
    case Gate::kSwap:
      simulator.ApplySWAP(0, 1);
      break;
    case Gate::kISwap:
      simulator.ApplyISWAP(0, 1);
      break;
    case Gate::kISwapDG:
      simulator.ApplyISWAPDG(0, 1);
      break;
    case Gate::kCH:
      simulator.ApplyCH(0, 1);
      break;
    case Gate::kCRX:
      simulator.ApplyCRX(0, 1, -0.31);
      break;
    case Gate::kCRY:
      simulator.ApplyCRY(0, 1, 0.47);
      break;
    case Gate::kCRZ:
      simulator.ApplyCRZ(0, 1, -0.53);
      break;
    case Gate::kCP:
      simulator.ApplyCP(0, 1, 0.43);
      break;
    case Gate::kCS:
      simulator.ApplyCS(0, 1);
      break;
    case Gate::kCSDAG:
      simulator.ApplyCSDAG(0, 1);
      break;
    case Gate::kCSX:
      simulator.ApplyCSX(0, 1);
      break;
    case Gate::kCSXDAG:
      simulator.ApplyCSXDAG(0, 1);
      break;
    case Gate::kCU:
      simulator.ApplyCU(0, 1, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCCX:
      simulator.ApplyCCX(0, 1, 2);
      break;
    case Gate::kCSwap:
      simulator.ApplyCSwap(0, 1, 2);
      break;
  }
}

void ApplyGate(const std::shared_ptr<Simulators::ISimulator>& simulator,
               Gate gate) {
  switch (gate) {
    case Gate::kX:
      simulator->ApplyX(0);
      break;
    case Gate::kY:
      simulator->ApplyY(0);
      break;
    case Gate::kZ:
      simulator->ApplyZ(0);
      break;
    case Gate::kH:
      simulator->ApplyH(0);
      break;
    case Gate::kS:
      simulator->ApplyS(0);
      break;
    case Gate::kSDG:
      simulator->ApplySDG(0);
      break;
    case Gate::kSX:
      simulator->ApplySx(0);
      break;
    case Gate::kSXDG:
      simulator->ApplySxDAG(0);
      break;
    case Gate::kK:
      simulator->ApplyK(0);
      break;
    case Gate::kP:
      simulator->ApplyP(0, 0.43);
      break;
    case Gate::kT:
      simulator->ApplyT(0);
      break;
    case Gate::kTDG:
      simulator->ApplyTDG(0);
      break;
    case Gate::kRX:
      simulator->ApplyRx(0, -0.31);
      break;
    case Gate::kRY:
      simulator->ApplyRy(0, 0.47);
      break;
    case Gate::kRZ:
      simulator->ApplyRz(0, -0.53);
      break;
    case Gate::kU:
      simulator->ApplyU(0, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCX:
      simulator->ApplyCX(0, 1);
      break;
    case Gate::kCY:
      simulator->ApplyCY(0, 1);
      break;
    case Gate::kCZ:
      simulator->ApplyCZ(0, 1);
      break;
    case Gate::kSwap:
      simulator->ApplySwap(0, 1);
      break;
    case Gate::kISwap: {
      const Eigen::Matrix4cd matrix =
          QC::Gates::iSwapGate<>().getRawOperatorMatrix();
      simulator->ApplyGenericTwoQubitGate(0, 1, matrix);
      break;
    }
    case Gate::kISwapDG: {
      const Eigen::Matrix4cd matrix =
          QC::Gates::iSwapDagGate<>().getRawOperatorMatrix();
      simulator->ApplyGenericTwoQubitGate(0, 1, matrix);
      break;
    }
    case Gate::kCH:
      simulator->ApplyCH(0, 1);
      break;
    case Gate::kCRX:
      simulator->ApplyCRx(0, 1, -0.31);
      break;
    case Gate::kCRY:
      simulator->ApplyCRy(0, 1, 0.47);
      break;
    case Gate::kCRZ:
      simulator->ApplyCRz(0, 1, -0.53);
      break;
    case Gate::kCP:
      simulator->ApplyCP(0, 1, 0.43);
      break;
    case Gate::kCS:
      simulator->ApplyCP(0, 1, kPi / 2.0);
      break;
    case Gate::kCSDAG:
      simulator->ApplyCP(0, 1, -kPi / 2.0);
      break;
    case Gate::kCSX:
      simulator->ApplyCSx(0, 1);
      break;
    case Gate::kCSXDAG:
      simulator->ApplyCSxDAG(0, 1);
      break;
    case Gate::kCU:
      simulator->ApplyCU(0, 1, 0.31, -0.23, 0.41, 0.17);
      break;
    case Gate::kCCX:
      simulator->ApplyCCX(0, 1, 2);
      break;
    case Gate::kCSwap:
      simulator->ApplyCSwap(0, 1, 2);
      break;
  }
}

std::string DecodePauliString(size_t encoded) {
  static constexpr std::array<char, 4> paulis{'I', 'X', 'Y', 'Z'};
  std::string pauli(kNrQubits, 'I');
  for (size_t qubit = 0; qubit < kNrQubits; ++qubit)
    pauli[qubit] = paulis[(encoded >> (2 * qubit)) & 3ULL];
  return pauli;
}

double ExtendedBasisProbability(
    const Simulators::QCSimExtendedStabilizer& simulator, size_t outcome) {
  double probability = 0.0;
  const size_t nrTerms = 1ULL << kNrQubits;
  for (size_t mask = 0; mask < nrTerms; ++mask) {
    std::string pauli(kNrQubits, 'I');
    double sign = 1.0;
    for (size_t qubit = 0; qubit < kNrQubits; ++qubit) {
      if (((mask >> qubit) & 1ULL) == 0) continue;
      pauli[qubit] = 'Z';
      if (((outcome >> qubit) & 1ULL) != 0) sign = -sign;
    }
    probability += sign * simulator.ExpectationValue(pauli);
  }
  return probability / static_cast<double>(nrTerms);
}

void CheckStatesMatch(
    const Simulators::QCSimExtendedStabilizer& extendedStabilizer,
    const std::shared_ptr<Simulators::ISimulator>& statevector,
    const std::string& context) {
  BOOST_TEST_CONTEXT(context) {
    const size_t nrPauliStrings = 1ULL << (2 * kNrQubits);
    for (size_t encoded = 0; encoded < nrPauliStrings; ++encoded) {
      const std::string pauli = DecodePauliString(encoded);
      const double expected = statevector->ExpectationValue(pauli);
      const double actual = extendedStabilizer.ExpectationValue(pauli);
      BOOST_TEST(std::abs(expected - actual) < kTolerance,
                 "Expectation mismatch for "
                     << pauli << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }

    const size_t nrStates = 1ULL << kNrQubits;
    for (size_t outcome = 0; outcome < nrStates; ++outcome) {
      const double expected = statevector->Probability(outcome);
      const double actual =
          ExtendedBasisProbability(extendedStabilizer, outcome);
      BOOST_TEST(std::abs(expected - actual) < kTolerance,
                 "Probability mismatch for outcome "
                     << outcome << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }

    for (size_t qubit = 0; qubit < kNrQubits; ++qubit) {
      double expected = 0.0;
      for (size_t outcome = 0; outcome < nrStates; ++outcome)
        if (((outcome >> qubit) & 1ULL) != 0)
          expected += statevector->Probability(outcome);
      const double actual = extendedStabilizer.GetQubitProbability(qubit);
      BOOST_TEST(std::abs(expected - actual) < kTolerance,
                 "Qubit probability mismatch for qubit "
                     << qubit << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }
  }
}

std::unordered_map<size_t, size_t> SampleExtendedStabilizer(
    Simulators::QCSimExtendedStabilizer& simulator, size_t shots) {
  const std::array<std::array<size_t, kNrQubits>, 3> measurementOrders{{
      {0, 1, 2},
      {2, 0, 1},
      {1, 2, 0},
  }};

  std::unordered_map<size_t, size_t> counts;
  simulator.SaveState();
  for (size_t shot = 0; shot < shots; ++shot) {
    simulator.RestoreState();
    size_t outcome = 0;
    for (const size_t qubit :
         measurementOrders[shot % measurementOrders.size()])
      if (simulator.Measure(qubit)) outcome |= 1ULL << qubit;
    ++counts[outcome];
  }
  simulator.RestoreState();
  return counts;
}

}  // namespace

BOOST_AUTO_TEST_SUITE(qcsim_extended_stabilizer_wrapper_tests)

BOOST_AUTO_TEST_CASE(control_target_order_matches_maestro) {
  Simulators::QCSimExtendedStabilizer simulator(2);

  simulator.ApplyX(0);
  simulator.ApplyCX(0, 1);

  BOOST_CHECK_SMALL(1.0 - simulator.GetQubitProbability(0), 1e-12);
  BOOST_CHECK_SMALL(1.0 - simulator.GetQubitProbability(1), 1e-12);
}

BOOST_AUTO_TEST_CASE(all_gates_match_qcsim_statevector) {
  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);

  for (const auto& gateCase : kGateCases) {
    statevector->Reset();
    extendedStabilizer.Reset(kNrQubits);
    PrepareReferenceState(extendedStabilizer, statevector);

    ApplyGate(extendedStabilizer, gateCase.gate);
    ApplyGate(statevector, gateCase.gate);

    CheckStatesMatch(extendedStabilizer, statevector,
                     std::string("Gate ") + gateCase.name);
  }
}

BOOST_AUTO_TEST_CASE(mixed_circuit_matches_qcsim_statevector) {
  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);

  PrepareReferenceState(extendedStabilizer, statevector);
  const std::array<Gate, 18> circuit{
      Gate::kT,   Gate::kCRY, Gate::kH,  Gate::kCU,  Gate::kSDG, Gate::kCP,
      Gate::kRX,  Gate::kCY,  Gate::kU,  Gate::kCSX, Gate::kRZ,  Gate::kCCX,
      Gate::kTDG, Gate::kCH,  Gate::kRY, Gate::kCRZ, Gate::kCZ,  Gate::kCSwap,
  };

  for (const Gate gate : circuit) {
    ApplyGate(extendedStabilizer, gate);
    ApplyGate(statevector, gate);
  }

  CheckStatesMatch(extendedStabilizer, statevector, "Mixed circuit");
}

BOOST_AUTO_TEST_CASE(measurement_sampling_matches_qcsim_statevector) {
  constexpr size_t shots = 8192;
  constexpr double sampleTolerance = 0.04;

  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);
  extendedStabilizer.SetRandomSeed(0x5A17B1E3U);
  PrepareReferenceState(extendedStabilizer, statevector);
  ApplyGate(extendedStabilizer, Gate::kT);
  ApplyGate(statevector, Gate::kT);
  ApplyGate(extendedStabilizer, Gate::kCRY);
  ApplyGate(statevector, Gate::kCRY);
  ApplyGate(extendedStabilizer, Gate::kCP);
  ApplyGate(statevector, Gate::kCP);

  Types::qubits_vector qubits(kNrQubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  const auto statevectorCounts = statevector->SampleCounts(qubits, shots);
  const auto extendedCounts =
      SampleExtendedStabilizer(extendedStabilizer, shots);

  for (size_t outcome = 0; outcome < (1ULL << kNrQubits); ++outcome) {
    const double expected = statevector->Probability(outcome);
    const auto statevectorCount = statevectorCounts.find(outcome);
    const auto extendedCount = extendedCounts.find(outcome);
    const double statevectorFrequency =
        statevectorCount == statevectorCounts.end()
            ? 0.0
            : static_cast<double>(statevectorCount->second) / shots;
    const double extendedFrequency =
        extendedCount == extendedCounts.end()
            ? 0.0
            : static_cast<double>(extendedCount->second) / shots;

    BOOST_TEST(std::abs(statevectorFrequency - expected) < sampleTolerance,
               "Statevector sample mismatch for outcome "
                   << outcome << ": exact " << expected << ", sampled "
                   << statevectorFrequency);
    BOOST_TEST(std::abs(extendedFrequency - expected) < sampleTolerance,
               "Extended stabilizer sample mismatch for outcome "
                   << outcome << ": statevector exact " << expected
                   << ", sampled " << extendedFrequency);
    BOOST_TEST(std::abs(extendedFrequency - statevectorFrequency) <
                   2.0 * sampleTolerance,
               "Sample distribution mismatch for outcome "
                   << outcome << ": statevector " << statevectorFrequency
                   << ", extended stabilizer " << extendedFrequency);
  }

  // Sampling restores the state after every shot and leaves it unchanged.
  CheckStatesMatch(extendedStabilizer, statevector,
                   "State after measurement sampling");
}

BOOST_AUTO_TEST_CASE(save_restore_and_reset_match_qcsim_statevector) {
  auto statevector = CreateStatevector();
  Simulators::QCSimExtendedStabilizer extendedStabilizer(kNrQubits);
  PrepareReferenceState(extendedStabilizer, statevector);

  extendedStabilizer.SaveState();
  statevector->SaveState();

  ApplyGate(extendedStabilizer, Gate::kCU);
  ApplyGate(statevector, Gate::kCU);
  ApplyGate(extendedStabilizer, Gate::kCCX);
  ApplyGate(statevector, Gate::kCCX);
  CheckStatesMatch(extendedStabilizer, statevector, "Mutated saved state");

  extendedStabilizer.RestoreState();
  statevector->RestoreState();
  CheckStatesMatch(extendedStabilizer, statevector, "Restored state");

  extendedStabilizer.Reset(kNrQubits);
  statevector->Reset();
  CheckStatesMatch(extendedStabilizer, statevector, "Reset state");
}

BOOST_AUTO_TEST_SUITE_END()
