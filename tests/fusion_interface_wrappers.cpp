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
  Check(sim.GetGateFusionMaxQubits() == 3 && sim.IsGateFusionEnabled());
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
  return 0;
}
