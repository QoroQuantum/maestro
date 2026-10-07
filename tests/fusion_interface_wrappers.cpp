#include "../maestroexe/Simulator.hpp"
#include "../Simulators/Interfaces/State.h"
#include "../Simulators/Core/Factory.h"
#include <cmath>
#include <stdexcept>

void Check(bool ok) {
  if (!ok) throw std::runtime_error("Dynamic fusion interface regression");
}
int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) return 2;
  const bool gpu = argc == 3;
#ifdef __linux__
  if (gpu && !Simulators::SimulatorsFactory::GetGpuDeviceCount()) return 77;
#endif
  const auto tensorBackend = gpu ? Simulators::SimulatorType::kGpuSim
                                 : Simulators::SimulatorType::kQCSim;
  Simulator sim;
  Check(sim.Init(argv[1]));
  Check(sim.CreateSimulator(static_cast<int>(Simulators::SimulatorType::kQCSim),
                            0));
  sim.AllocateQubits(3);
  Check(sim.InitializeSimulator());
  // Three qubits are below the default statevector fusion threshold.
  Check(sim.GetGateFusionMaxQubits() == 3 && !sim.IsGateFusionEnabled());
  Check(sim.ConfigureSimulator("gate_fusion", "true"));
  Check(sim.IsGateFusionEnabled());
  double one[8] = {0, 0, 1, 0, 1, 0, 0, 0};
  Check(sim.ApplyGenericOneQubitGate(0, one));
  double two[32]{};
  for (int c = 0; c < 4; ++c) two[2 * ((c ^ 1) * 4 + c)] = 1;
  Check(sim.ApplyGenericTwoQubitGate(1, 0, two));
  double three[128]{};
  for (int c = 0; c < 8; ++c) three[2 * ((c ^ 1) * 8 + c)] = 1;
  Check(sim.ApplyGenericThreeQubitGate(2, 0, 1, three));
  Check(std::abs(sim.Probability(7) - 1) < 1e-12);
  MaestroGateFusionStatistics stats{};
  Check(sim.GetGateFusionStatistics(&stats));
  Check(stats.submittedGates == 3 && stats.backendGates == 1 &&
        stats.fusedBlocks == 1);
  Check(sim.ConfigureSimulator("gate_fusion", "false"));
  Check(!sim.IsGateFusionEnabled());
  const auto values = sim.ExpectationValues({"ZZI", "ZII", "ZZI"});
  Check(values.size() == 3 && values[0] == 1 && values[1] == -1 && values[2] == 1);
  Check(sim.ExpectationValues({}).empty());
  const auto state = sim.GetStateVector();
  Check(state.size() == 8 && std::abs(state[7] - 1.) < 1e-12);
  Simulator mps;
  Check(mps.Init(argv[1]));
  Check(mps.CreateSimulator(
      static_cast<int>(tensorBackend),
      static_cast<int>(Simulators::SimulationType::kMatrixProductState)));
  Check(mps.ConfigureSimulator("precision", "double"));
  mps.AllocateQubits(2);
  Check(mps.InitializeSimulator());
  Check(mps.ConfigureSimulator("gate_fusion", "true"));
  mps.ApplyX(0);
  mps.MoveAtBeginningOfChain({});
  mps.MoveAtBeginningOfChain({1});
  const std::array<std::complex<double>, 4> x{0., 1., 1., 0.};
  const std::array<std::complex<double>, 4> y{0., std::complex<double>(0., -1.),
                                              std::complex<double>(0., 1.), 0.};
  Check(std::abs(mps.ExpectationValueOperators({0, 0}, {x, y}) -
                 std::complex<double>(0., 1.)) < 1e-10);
  Check(std::abs(mps.GetStateVector()[1] - 1.) < 1e-10);
  Simulator mpo;
  Check(mpo.Init(argv[1]));
  Check(mpo.CreateSimulator(
      static_cast<int>(tensorBackend),
      static_cast<int>(Simulators::SimulationType::kMatrixProductOperator)));
  Check(mpo.ConfigureSimulator("precision", "double"));
  mpo.AllocateQubits(2);
  Check(mpo.InitializeSimulator());
  mpo.ApplyH(0);
  mpo.ApplyCX(0, 1);
  mpo.MoveAtBeginningOfChain({});
  const auto complex = mpo.ExpectationValuesComplex({"XX", "YY", "ZZ"}, false);
  Check(complex.size() == 3 && std::abs(complex[0] - 1.) < 1e-10 &&
        std::abs(complex[1] + 1.) < 1e-10 && std::abs(complex[2] - 1.) < 1e-10);
  const std::string request =
      std::string(R"({"schema_version":2,"operation":"diagnostics",
      "circuit":{"num_qubits":1,"source":"OPENQASM 2.0; qreg q[1];"},
      "simulator":{"backend":")") +
      (gpu ? "gpu" : "qcsim") + R"(","method":"matrix_product_state"},
      "diagnostics":["operator_expectation"],"expectation_operators":[]})";
  Check(mps.ValidateRequestJson(request).find("\"ok\":true") !=
        std::string::npos);
  Check(mps.RunRequestJson(request).find("\"operator_expectation\":[1") !=
        std::string::npos);
  return 0;
}
