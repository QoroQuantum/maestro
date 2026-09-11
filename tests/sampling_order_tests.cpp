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
        const auto small = sim.SampleCounts(qubits, shots);
        const auto wide = sim.SampleCountsMany(qubits, shots);
        const std::string context = "basis=" + std::to_string(basis) +
            " first_qubit=" + std::to_string(qubits.front()) +
            " width=" + std::to_string(qubits.size()) + " shots=" + std::to_string(shots);
        Require(small.size() == 1 && small.count(packed) && small.at(packed) == shots,
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
