// Regression tests for caller-ordered sampling, including mapped internal states.
#include "../Simulators/Factory.h"
#define INCLUDED_BY_FACTORY
#ifndef NO_QISKIT_AER
#include "../Simulators/AerSimulator.h"
#endif
#include "../Simulators/QCSimSimulator.h"
#include "../Simulators/Individual.h"
#include <array>
#include <iostream>
#include <stdexcept>

using namespace Simulators;
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// Test every computational basis state so symmetric bit patterns cannot hide
// permutations. Include full permutations, subsets and single-qubit queries.
void CheckSamples(ISimulator& sim, const std::vector<Types::qubit_t>& ids) {
  sim.AllocateQubits(3);
  sim.Initialize();
  unsigned previous = 0;
  for (unsigned basis = 0; basis < 8; ++basis) {
    for (unsigned bit = 0; bit < 3; ++bit)
      if (((previous ^ basis) >> bit) & 1) sim.ApplyX(ids[bit]);
    previous = basis;
    const std::vector<std::vector<unsigned>> orders{
        {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0},
        {2,0}, {1,2}, {2}, {0}};
    for (const auto& order : orders) {
      Types::qubits_vector qubits;
      std::vector<bool> expected;
      Types::qubit_t packed = 0;
      for (size_t i = 0; i < order.size(); ++i) {
        qubits.push_back(ids[order[i]]);
        expected.push_back((basis >> order[i]) & 1);
        if (expected.back()) packed |= Types::qubit_t(1) << i;
      }
      for (size_t shots : {size_t(1), size_t(16)}) {
        const auto smallv = sim.SampleCounts(qubits, shots);
        const auto wide = sim.SampleCountsMany(qubits, shots);
        const std::string context = "basis=" + std::to_string(basis) +
            " first_qubit=" + std::to_string(qubits.front()) +
            " width=" + std::to_string(qubits.size()) + " shots=" + std::to_string(shots);
        Require(smallv.size() == 1 && smallv.count(packed) && smallv.at(packed) == shots,
                "packed sampling: " + context);
        Require(wide.size() == 1 && wide.count(expected) && wide.at(expected) == shots,
                "vector sampling: " + context);
      }
    }
  }
  Require(sim.SampleCounts({ids[0]}, 0).empty() && sim.SampleCountsMany({ids[0]}, 0).empty(), "zero shots");
}

void Backend(SimulatorType type, SimulationType method, const char* algorithm = nullptr) {
  auto sim = SimulatorsFactory::CreateSimulator(type, method);
  Require(bool(sim), "simulator unavailable");
  if (algorithm) sim->Configure("mps_sample_measure_algorithm", algorithm);
  CheckSamples(*sim, {0,1,2});
}

void CliffordWideSampling() {
  auto sim = SimulatorsFactory::CreateSimulator(SimulatorType::kQCSim,
                                               SimulationType::kStabilizer);
  sim->AllocateQubits(65);
  sim->Initialize();
  for (unsigned bit : {32, 63}) {
    const Types::qubit_t outcome = Types::qubit_t(1) << bit;
    Require(sim->Probability(outcome) == 0.0, "high outcome bits must not alias zero");
    sim->ApplyX(bit);
    Require(sim->Probability(outcome) == 1.0 && sim->Probability(0) == 0.0,
            "high outcome bits must reach the Clifford backend");
    sim->ApplyX(bit);
  }
  sim->SaveState();
  sim->ApplyX(64); sim->ApplyH(0); sim->ApplyCX(0, 63);
  const Types::qubits_vector selected{64, 63, 0, 63};
  sim->SetSeed(891);
  const auto packed = sim->SampleCounts(selected, 128);
  Require(packed.size() == 2 && packed.count(1) && packed.count(15) &&
          packed.at(1) + packed.at(15) == 128, "wide ordered marginal counts");
  sim->SetSeed(891);
  Require(sim->SampleCounts(selected, 128) == packed, "warm seeded Clifford counts");
  sim->SetSeed(891);
  const auto many = sim->SampleCountsMany(selected, 128);
  Require(many.size() == 2 && many.at(std::vector<bool>{true,false,false,false}) == packed.at(1) &&
          many.at(std::vector<bool>{true,true,true,true}) == packed.at(15), "packed/vector counts agree");
  bool rejected = false;
  try { sim->SampleCounts(Types::qubits_vector(65, 0), 1); }
  catch (const std::invalid_argument&) { rejected = true; }
  Require(rejected, "reject packed outcomes wider than size_t");
  rejected = false;
  try { sim->SampleCountsMany({65}, 1); }
  catch (const std::out_of_range&) { rejected = true; }
  Require(rejected, "reject invalid measured qubits");
  sim->RestoreState();
  Require(sim->Probability(0) == 1.0, "sampling must preserve the caller's saved state");
}

void Individual(SimulatorType type) {
  for (const std::vector<Types::qubit_t>& ids :
       {std::vector<Types::qubit_t>{0,1,2}, {5,7,9}, {9,5,7}}) {
    Private::IndividualSimulator sim(type);
    for (size_t i = 0; i < ids.size(); ++i) sim.GetQubitsMap()[ids[i]] = i;
    CheckSamples(sim, ids);
  }
}
}

int main(int argc, char** argv) {
  int failed = 0, passed = 0;
  auto run = [&](const std::string& name, auto test) {
    try { test(); ++passed; std::cout << "PASS " << name << std::endl; }
    catch (const std::exception& error) {
      ++failed; std::cerr << "FAIL " << name << ": " << error.what() << std::endl;
    }
  };
  const bool gpuOnly = argc == 2 && std::string(argv[1]) == "--gpu";
  if (gpuOnly) {
#ifdef __linux__
    const int count = SimulatorsFactory::GetGpuDeviceCount();
    if (count < 0) { std::cerr << "CUDA discovery failed\n"; return 1; }
    if (count == 0) { std::cout << "SKIP: GPU plugin/device unavailable\n"; return 77; }
    for (const auto method : {SimulationType::kStatevector, SimulationType::kMatrixProductState,
                             SimulationType::kTensorNetwork, SimulationType::kDensityMatrix,
                             SimulationType::kMatrixProductOperator, SimulationType::kPauliPropagator})
      run("GPU method " + std::to_string(int(method)), [&] { Backend(SimulatorType::kGpuSim, method); });
#else
    return 77;
#endif
  } else {
    for (const auto method : {SimulationType::kStatevector, SimulationType::kMatrixProductState,
                             SimulationType::kTensorNetwork, SimulationType::kDensityMatrix,
                             SimulationType::kMatrixProductOperator, SimulationType::kStabilizer,
                             SimulationType::kExtendedStabilizer, SimulationType::kPauliPropagator,
                             SimulationType::kPathIntegral})
      run("QCSim method " + std::to_string(int(method)), [&] { Backend(SimulatorType::kQCSim, method); });
    run("QCSim Clifford wide probabilities and marginal sampling", CliffordWideSampling);
    run("QCSim MPS collapse sampler", [&] { Backend(SimulatorType::kQCSim, SimulationType::kMatrixProductState, "mps_apply_measure"); });
    run("Composite QCSim", [&] { Backend(SimulatorType::kCompositeQCSim, SimulationType::kStatevector); });
    run("Individual QCSim mappings", [&] { Individual(SimulatorType::kQCSim); });
#ifndef NO_QISKIT_AER
    for (const auto method : {SimulationType::kStatevector, SimulationType::kDensityMatrix,
                             SimulationType::kStabilizer, SimulationType::kExtendedStabilizer})
      run("Aer method " + std::to_string(int(method)), [&] { Backend(SimulatorType::kQiskitAer, method); });
    for (const char* algorithm : {"mps_probabilities", "mps_apply_measure"})
      run(std::string("Aer MPS ") + algorithm, [&] { Backend(SimulatorType::kQiskitAer, SimulationType::kMatrixProductState, algorithm); });
    run("Composite Aer", [&] { Backend(SimulatorType::kCompositeQiskitAer, SimulationType::kStatevector); });
    run("Individual Aer mappings", [&] { Individual(SimulatorType::kQiskitAer); });
#else
    std::cout << "Aer coverage disabled in this build\n";
#endif
  }
  std::cout << passed << " passed, " << failed << " failed\n";
  return failed ? 1 : 0;
}
