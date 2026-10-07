// Manual benchmark: no wall-clock assertions. Always includes the final flush.
#include "Simulators/Core/Factory.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace Simulators;
using Clock = std::chrono::steady_clock;

void Run(SimulationType method, const char* name, size_t qubits,
         SimulatorType backend, bool sparse) {
  std::vector<double> reference;
  for (bool fusion : {false, true}) {
    auto sim = SimulatorsFactory::CreateSimulator(backend, method);
    sim->Configure("gate_fusion", fusion ? "true" : "false");
    sim->Configure("use_double_precision", "true");
    sim->Configure("matrix_product_state_max_bond_dimension", "256");
    sim->Configure("matrix_product_state_truncation_threshold", "0");
    sim->SetMultithreading(false);
    sim->AllocateQubits(qubits);
    sim->Initialize();
    const auto start = Clock::now();
    for (size_t layer = 0; layer < 8; ++layer)
      for (size_t a = layer % 2; a + 1 < qubits; a += 2) {
        const auto b = a + 1;
        if (sparse) {
          sim->ApplyCX(a, (a + 1 + layer % (qubits - 1)) % qubits);
        } else {
          sim->ApplyH(a);
          sim->ApplyRz(a, .11);
          sim->ApplyH(b);
          sim->ApplyCX(a, b);
          sim->ApplyRy(a, .07);
          sim->ApplyCP(b, a, .05);
        }
      }
    sim->Flush();
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    const auto stats = sim->GetGateFusionStatistics();
    // Probe a fixed set of outcomes. Enumerating an entire MPS state on the
    // GPU can cost far more than the circuit and is outside the timed region.
    std::vector<double> probabilities;
    const size_t outcomes = size_t{1} << qubits;
    for (size_t i = 0; i < std::min(outcomes, size_t{64}); ++i)
      probabilities.push_back(sim->Probability((i * 2654435761ULL) % outcomes));
    double error = 0;
    if (!fusion)
      reference = probabilities;
    else
      for (size_t i = 0; i < reference.size(); ++i)
        error = std::max(error, std::abs(reference[i] - probabilities[i]));
    std::cout << name << ',' << qubits << ','
              << (sparse ? "sparse" : "repeated_pair") << ',' << fusion << ','
              << stats.submittedGates << ',' << stats.backendGates << ','
              << std::setprecision(7) << ms << ',' << error << '\n';
    if (error > 1e-8) throw std::runtime_error("Benchmark result mismatch");
  }
}

int main(int argc, char** argv) {
  try {
    const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    const auto backend = gpu ? SimulatorType::kGpuSim : SimulatorType::kQCSim;
#ifdef __linux__
    if (gpu && !SimulatorsFactory::InitGpuLibrary()) return 1;
#else
    if (gpu) throw std::runtime_error("GPU backend requires Linux");
#endif
    std::cout << "method,qubits,pattern,fusion,submitted,backend,milliseconds,"
                 "max_sampled_probability_error\n";
    for (bool sparse : {false, true}) {
      if (!gpu)
        Run(SimulationType::kStatevector, "statevector", 20, backend, sparse);
      Run(SimulationType::kDensityMatrix, "density_matrix", 8, backend, sparse);
      Run(SimulationType::kMatrixProductState, "mps", 14, backend, sparse);
      Run(SimulationType::kMatrixProductOperator, "mpo", 8, backend, sparse);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
