#include "../maestroexe/Simulator.hpp"
#include "../Simulators/State.h"
#include <cmath>
#include <stdexcept>

void Check(bool ok) {
  if (!ok) throw std::runtime_error("Dynamic fusion interface regression");
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
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
  Simulator mpo;
  Check(mpo.Init(argv[1]));
  Check(mpo.CreateSimulator(static_cast<int>(Simulators::SimulatorType::kQCSim),
      static_cast<int>(Simulators::SimulationType::kMatrixProductOperator)));
  mpo.AllocateQubits(2);
  Check(mpo.InitializeSimulator());
  mpo.ApplyH(0);
  mpo.ApplyCX(0, 1);
  const auto complex = mpo.ExpectationValuesComplex({"XX", "YY", "ZZ"}, false);
  Check(complex.size() == 3 && std::abs(complex[0] - 1.) < 1e-10 &&
        std::abs(complex[1] + 1.) < 1e-10 && std::abs(complex[2] - 1.) < 1e-10);
  return 0;
}
