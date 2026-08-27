/** Checks that InitializeToBasisState works across every simulator backend:
 * either via a backend-specific native primitive, or via the generic
 * ISimulator fallback (reset to |0...0>, then apply X on every set bit). */

#include <boost/test/unit_test.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "../Simulators/Factory.h"

namespace {

constexpr unsigned int kNumQubits = 3;
constexpr double kTolerance = 1e-9;

void CheckBasisStateProbabilities(Simulators::ISimulator& sim,
                                  Types::qubit_t basisState) {
  // The Pauli propagator's Probability(outcome) is nan/-nan for most
  // outcomes even on a freshly Initialize()'d simulator - a pre-existing
  // property of that method's Probability() unrelated to basis-state
  // initialization (reproduced identically with no gates applied at all, on
  // both the QCSim and gpu backends). Only check the initialized outcome for
  // it; every other backend is checked exhaustively.
  const bool skipOtherOutcomes =
      sim.GetSimulationType() == Simulators::SimulationType::kPauliPropagator;

  // Query outcomes one at a time via Probability() rather than
  // AllProbabilities(): some backends (e.g. QCSim's tensor network) refuse
  // AllProbabilities() by design as prohibitively expensive, but still
  // support querying individual outcomes.
  for (Types::qubit_t outcome = 0; outcome < (1ULL << kNumQubits); ++outcome) {
    if (skipOtherOutcomes && outcome != basisState) continue;
    const double expected = (outcome == basisState) ? 1.0 : 0.0;
    const double actual = sim.Probability(outcome);
    BOOST_TEST(std::abs(actual - expected) < kTolerance,
               "backend method '" << sim.GetConfiguration("method")
                                  << "': outcome " << outcome
                                  << " probability " << actual
                                  << " (expected " << expected << ")");
  }
}

void CheckAllBasisStates(std::shared_ptr<Simulators::ISimulator> sim) {
  if (!sim) return;
  for (Types::qubit_t basisState = 0; basisState < (1ULL << kNumQubits);
       ++basisState) {
    sim->InitializeToBasisState(kNumQubits, basisState);
    CheckBasisStateProbabilities(*sim, basisState);
  }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(initialize_to_basis_state_tests)

BOOST_AUTO_TEST_CASE(qcsim_backends_support_basis_state_initialization) {
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kDensityMatrix));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductOperator));
  // The next ones have no dedicated basis-state primitive in QCSim; they
  // must rely on the generic ISimulator fallback (reset + apply X).
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStabilizer));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kTensorNetwork));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kExtendedStabilizer));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kPathIntegral));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kPauliPropagator));
}

#ifdef __linux__
BOOST_AUTO_TEST_CASE(gpu_backends_support_basis_state_initialization) {
  Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
  if (!Simulators::SimulatorsFactory::IsGpuLibraryAvailable()) {
    BOOST_TEST_MESSAGE("GPU library is unavailable; skipping");
    return;
  }

  // Statevector, tensor network and Pauli propagator have no dedicated
  // basis-state primitive in the gpu library either; they must rely on the
  // same generic ISimulator fallback as their CPU counterparts.
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kStatevector));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kDensityMatrix));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kMatrixProductState));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kMatrixProductOperator));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kTensorNetwork));
  CheckAllBasisStates(Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kGpuSim,
      Simulators::SimulationType::kPauliPropagator));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
