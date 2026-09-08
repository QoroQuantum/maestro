// Real hardware regression: two live GPU instances, independent CPU references.
#include "../../Simulators/Factory.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace Simulators;
using Factory = SimulatorsFactory;

namespace {
constexpr unsigned kQubits = 3;
constexpr double kTolerance = 1e-6;

std::shared_ptr<ISimulator> Create(SimulatorType type, SimulationType method,
                                   int device) {
  auto sim = Factory::CreateSimulator(type, method);
  if (!sim) throw std::runtime_error("simulator creation failed");
  if (type == SimulatorType::kGpuSim) {
    sim->Configure("gpu_device", std::to_string(device).c_str());
    sim->Configure("use_double_precision", "true");
  }
  sim->AllocateQubits(kQubits);
  sim->Initialize();
  if (type == SimulatorType::kGpuSim && sim->GetGpuDevice() != device)
    throw std::runtime_error("native simulator placement differs from gpu_device");
  return sim;
}

void Step(ISimulator& sim, unsigned round, unsigned device) {
  const unsigned q = (round + device) % kQubits;
  const double angle = (device + 1) * (round + 1) * 0.173;
  sim.ApplyH(q);
  sim.ApplyRx(q, angle);
  sim.ApplyRy((q + 1) % kQubits, -0.7 * angle);
  sim.ApplyCX(q, (q + 1) % kQubits);
  sim.ApplyRz((q + 2) % kQubits, 0.3 + angle);
  sim.ApplyCX((q + 1) % kQubits, (q + 2) % kQubits);
}

void Check(ISimulator& gpu, ISimulator& cpu, const std::string& context) {
  // Complete Pauli basis detects phase and probability errors for all backends.
  for (unsigned word = 0; word < (1u << (2 * kQubits)); ++word) {
    std::string pauli(kQubits, 'I');
    for (unsigned q = 0; q < kQubits; ++q)
      pauli[q] = "IXYZ"[(word >> (2 * q)) & 3];
    const double expected = cpu.ExpectationValue(pauli);
    const double actual = gpu.ExpectationValue(pauli);
    if (!std::isfinite(expected) || !std::isfinite(actual) ||
        std::abs(actual - expected) > kTolerance)
      throw std::runtime_error(context + " Pauli " + pauli +
                               ": GPU=" + std::to_string(actual) +
                               " CPU=" + std::to_string(expected));
  }
}
}  // namespace

int main() {
  try {
    const int deviceCount = Factory::GetGpuDeviceCount();
    if (deviceCount < 0) throw std::runtime_error("CUDA device discovery failed");
    if (deviceCount < 2) {
      std::cout << "SKIP: two visible CUDA GPUs and the real GPU plugin "
                   "are required (check CUDA_VISIBLE_DEVICES).\n";
      return 77;
    }
    if (!Factory::InitGpuLibraryWithMute())
      throw std::runtime_error("GPU initialization failed");
    if (Factory::GetGpuLibrary()->GetFunction("MockInitializations"))
      throw std::runtime_error("Real GPU plugin required; mock plugin loaded");

    for (const auto& entry :
         {std::make_pair(SimulationType::kStatevector, "statevector"),
          std::make_pair(SimulationType::kDensityMatrix, "density matrix"),
          std::make_pair(SimulationType::kMatrixProductState, "MPS"),
          std::make_pair(SimulationType::kMatrixProductOperator, "MPO"),
          std::make_pair(SimulationType::kTensorNetwork, "tensor network"),
          std::make_pair(SimulationType::kPauliPropagator,
                         "Pauli propagator")}) {
      std::cout << "Testing " << entry.second << std::endl;
      std::array<std::shared_ptr<ISimulator>, 2> gpu, cpu;
      for (int device = 0; device < 2; ++device) {
        gpu[device] = Create(SimulatorType::kGpuSim, entry.first, device);
        gpu[device]->ApplyX(2);
        const auto samples = gpu[device]->SampleCounts({2, 0, 1}, 16);
        const auto wideSamples = gpu[device]->SampleCountsMany({2, 0, 1}, 16);
        if (samples.size() != 1 || samples.at(1) != 16 ||
            wideSamples.size() != 1 || wideSamples.at(std::vector<bool>{true, false, false}) != 16)
          throw std::runtime_error(std::string(entry.second) + " lost sampling bit order");
        gpu[device]->ApplyX(2);
        cpu[device] =
            Create(SimulatorType::kQCSim, SimulationType::kStatevector, device);
      }
      for (unsigned round = 0; round < 6; ++round) {
        // Reverse order each round; updates must leave the peer untouched.
        for (unsigned turn = 0; turn < 2; ++turn) {
          const unsigned device = (turn + round) % 2;
          Factory::SelectGpuDevice(1 - device);
          Step(*gpu[device], round, device);
          Step(*cpu[device], round, device);
          for (unsigned check = 0; check < 2; ++check)
            Check(*gpu[check], *cpu[check],
                  std::string(entry.second) +
                      " device=" + std::to_string(check) +
                      " round=" + std::to_string(round) +
                      " updated=" + std::to_string(device));
        }
      }
      // Destroy one instance while the other remains usable.
      gpu[0].reset();
      Step(*gpu[1], 6, 1);
      Step(*cpu[1], 6, 1);
      Check(*gpu[1], *cpu[1],
            std::string(entry.second) + " after peer destruction");
      Factory::SelectGpuDevice(0);
    }
    std::cout << "Two-GPU CPU-statevector comparisons passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
