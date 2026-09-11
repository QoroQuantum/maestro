#include "../../Simulators/Factory.h"
#include "../../Network/SimpleDisconnectedNetwork.h"
#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cmath>
#ifdef MAESTRO_MPI_GPU_TESTS
#include <mpi.h>
#endif
using namespace Simulators;
using Factory = SimulatorsFactory;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F>
void Reject(F&& action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "Expected an exception");
}
void Compare(ISimulator& actual, ISimulator& reference, double eps) {
  for (size_t i = 0; i < (size_t{1} << actual.GetNumberOfQubits()); ++i)
    Require(std::abs(actual.Amplitude(i) - reference.Amplitude(i)) < eps,
            "CPU amplitude mismatch");
  for (const std::string pauli : {"IIII", "XIII", "YIZI", "IXYZ", "ZZZZ"})
    Require(std::abs(actual.ExpectationValue(pauli) -
                     reference.ExpectationValue(pauli)) < eps * 8,
            "CPU Pauli expectation mismatch");
}
void MixedGates(ISimulator& sim) {
  sim.ApplyH(0);
  sim.ApplyH(1);
  sim.ApplyH(3);
  sim.ApplyX(2);
  sim.ApplyY(1);
  sim.ApplyZ(0);
  sim.ApplyS(0);
  sim.ApplySDG(3);
  sim.ApplyT(1);
  sim.ApplyTDG(2);
  sim.ApplySx(0);
  sim.ApplySxDAG(2);
  sim.ApplyK(3);
  sim.ApplyP(0, .17);
  sim.ApplyRx(1, .32);
  sim.ApplyRy(0, -.23);
  sim.ApplyRz(2, .41);
  sim.ApplyU(3, .12, .23, .34, .45);
  sim.ApplyCX(0, 2);
  sim.ApplyCY(1, 0);
  sim.ApplyCZ(0, 3);
  sim.ApplyCH(2, 1);
  sim.ApplyCSx(1, 0);
  sim.ApplyCSxDAG(0, 3);
  sim.ApplyCP(0, 1, .2);
  sim.ApplyCRx(2, 0, .3);
  sim.ApplyCRy(0, 3, .4);
  sim.ApplyCRz(1, 0, .5);
  sim.ApplyCCX(0, 2, 1);
  sim.ApplySwap(0, 3);
  sim.ApplyCSwap(2, 0, 1);
  sim.ApplyCU(0, 2, .2, .3, .4, .5);
  Eigen::Matrix2cd one;
  const double c = std::cos(.23), s = std::sin(.23);
  one << c, std::complex<double>(0, s), std::complex<double>(0, s), c;
  sim.ApplyGenericOneQubitGate(0, one);
  // Asymmetric, non-real unitary detects transposes and reversed target order.
  Eigen::Matrix4cd two = Eigen::Matrix4cd::Zero();
  two(2, 0) = 1.;
  two(0, 1) = std::complex<double>(0, 1);
  two(3, 2) = -1.;
  two(1, 3) = std::complex<double>(0, -1);
  sim.ApplyGenericTwoQubitGate(0, 2, two);
  sim.Flush();
}
int main(int argc, char** argv) {
  bool mpi = false, shared = false, two = false, configOnly = false,
       p2p = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    mpi |= arg == "--mpi";
    p2p |= arg == "--p2p";
    shared |= arg == "--shared";
    two |= arg == "--two-gpus";
    configOnly |= arg == "--config-only";
  }
  if (argc == 2 && std::string(argv[1]) == "--missing-mpi-plugin") {
    auto sim = Factory::CreateSimulator(SimulatorType::kDistMpiGpuSim,
                                        SimulationType::kStatevector);
    sim->AllocateQubits(4);
    try {
      sim->Initialize();
    } catch (const std::runtime_error& e) {
      Require(
          std::string(e.what()).find("Unable to load distributed GPU plugin") !=
              std::string::npos,
          "Expected runtime loading diagnostic, without MPI initialization");
      std::cout << "Unavailable MPI plugin rejected at runtime\n";
      return 0;
    }
    throw std::runtime_error("Missing MPI plugin was accepted");
  }
  int code = 0;
#ifdef MAESTRO_MPI_GPU_TESTS
  if (mpi) MPI_Init(&argc, &argv);
#else
  if (mpi) return 77;
#endif
  try {
    const auto type =
        mpi ? SimulatorType::kDistMpiGpuSim : SimulatorType::kDistGpuSim;
    auto sim = Factory::CreateSimulator(type, SimulationType::kStatevector);
    Require(bool(sim), "Missing simulator");
    Require(bool(Factory::CreateSimulator(SimulatorType::kDistMpiGpuSim,
                                          SimulationType::kStatevector)),
            "MPI simulator must be available in every build");
    if (p2p) sim->Configure("mpi_p2p_bits", "1");
    Reject([&] {
      Factory::CreateSimulator(type, SimulationType::kDensityMatrix);
    });
    Reject([&] { sim->Configure("method", "matrix_product_state"); });
    Reject([&] { sim->Configure("distributed_devices", "0,-1"); });
    Reject(
        [&] { sim->Configure("distributed_max_queued_gates", "4294967296"); });
    Reject([&] { sim->Configure("distributed_flags", "4294967296"); });
    Reject([&] { sim->Configure("seed", "-1"); });
    Reject([&] { sim->AllocateQubits(63); });
    Require(sim->GetType() == type, "Wrong simulator type");
    auto unique =
        Factory::CreateSimulatorUnique(type, SimulationType::kStatevector);
    Require(unique && unique->Clone()->GetType() == type,
            "Unique factory/empty clone");
    if (!configOnly) {
      auto lib = mpi ? std::static_pointer_cast<DistributedGpuLibrary>(
                           Factory::GetDistributedMpiGpuLibrary())
                     : Factory::GetDistributedGpuLibrary();
      int available = lib->Load() ? lib->GetGpuDeviceCount() : 0;
#ifdef MAESTRO_MPI_GPU_TESTS
      if (mpi) {
        int minimum;
        MPI_Allreduce(&available, &minimum, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        available = minimum;
      }
#endif
      if (available <= 0 || (two && available < 2)) {
        std::cout << "SKIP: required plugin/GPU devices unavailable\n";
        code = 77;
      } else {
        for (const char* precision : {"single", "double"}) {
          for (const char* flags : {"0", "2", "8"}) {
            sim->Clear();
            sim->Configure("precision", precision);
            sim->Configure("distributed_snapshot_storage",
                           std::string(precision) == "double" ? "host" : "gpu");
            sim->Configure(
                "distributed_flags",
                shared ? std::to_string(std::stoi(flags) | 1).c_str() : flags);
            if (shared) {
              sim->Configure("distributed_devices", "0,0");
              if (!mpi) sim->Configure("distributed_backend", "conventional");
            }
            sim->SetSeed(42);
            sim->AllocateQubits(4);
            sim->Initialize();
#ifdef MAESTRO_MPI_GPU_TESTS
            if (mpi)
              Reject([&] { Factory::FinalizeDistributedMpiGpuBackend(); });
#endif
            const std::string devices =
                sim->GetConfiguration("distributed_shard_devices");
            if (two || shared || mpi)
              Require(sim->GetConfiguration(
                          "distributed_configured_global_qubits") == "0",
                      "Default globals must start at zero");
            if (two)
              Require(devices == "0,1",
                      "Expected two physical shards by default");
            if (two || mpi)
              Require(sim->GetGpuDevice() == -1,
                      "Distributed placement must not report one GPU");
            Reject([&] {
              sim->Configure("precision", std::string(precision) == "single"
                                              ? "double"
                                              : "single");
            });
            auto cpu = Factory::CreateSimulator(SimulatorType::kQCSim,
                                                SimulationType::kStatevector);
            cpu->AllocateQubits(4);
            cpu->Initialize();
            MixedGates(*sim);
            MixedGates(*cpu);
            const double eps =
                std::string(precision) == "double" ? 1e-10 : 2e-5;
            Compare(*sim, *cpu, eps);
            sim->SaveState();
            sim->ApplyX(0);
            sim->RestoreState();
            Compare(*sim, *cpu, eps);
            auto clone = sim->Clone();
            Compare(*clone, *cpu, eps);
            sim->ApplyX(1);
            Compare(*clone, *cpu, eps);
            sim->RestoreState();
            sim->SaveStateToInternalDestructive();
            Reject([&] { sim->Amplitude(0); });
            sim->RestoreInternalDestructiveSavedState();
            Compare(*sim, *cpu, eps);
            clone.reset();
            sim->Reset();
            sim->ApplyX(0);
            sim->ApplyX(3);
            const auto counts = sim->SampleCounts({3, 1, 0}, 32);
            Require(counts.size() == 1 && counts.at(5) == 32,
                    "Sample output bit order");
            auto many = sim->SampleCountsMany({3, 1, 0}, 32);
            Require(many.at({true, false, true}) == 32,
                    "SampleCountsMany output order");
            Reject([&] { sim->SampleCounts({4}, 1); });
            Require(sim->Measure({3, 1, 0}) == 5, "Collapse output bit order");
            sim->ApplyReset({0, 3});
            Require(std::abs(sim->Probability(0) - 1.) < eps, "Reset qubits");
            sim->Reset();
            sim->ApplyH(0);
            sim->ApplyCX(0, 3);
            auto bell = sim->SampleCounts({0, 3}, 1000);
            Require(bell.size() == 2 && bell.count(0) && bell.count(3),
                    "Distributed Bell samples");
            Require(bell[0] > 350 && bell[0] < 650,
                    "Bell sampling distribution");
            auto measured = sim->Measure({3, 0});
            Require(measured == 0 || measured == 3, "Joint Bell collapse");
            Require(std::abs(sim->Probability(measured ? 9 : 0) - 1.) < eps,
                    "Collapse normalization");
            sim->Clear();
            std::vector<std::complex<double>> initial(16);
            initial[5] = std::complex<double>(0, 1);
            sim->InitializeState(4, initial);
            Require(std::abs(sim->Amplitude(5) - initial[5]) < eps,
                    "State import");
          }
        }
        sim->Clear();
        // Explicit network selection must survive optimization and recreation.
        Network::SimpleDisconnectedNetwork<> network({4}, {4});
        network.Configure("max_simulators", "4");
        if (shared) {
          network.Configure("distributed_devices", "0,0");
          network.Configure("distributed_flags", "1");
          if (!mpi) network.Configure("distributed_backend", "conventional");
        }
        network.CreateSimulator(type, SimulationType::kStatevector);
        Require(network.GetSimulator()->GetType() == type,
                "Network factory type");
        network.GetSimulator()->ApplyX(0);
        Require(network.GetSimulator()->SampleCounts({0}, 4).at(1) == 4,
                "Network configuration replay");
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    code = 1;
  }
#ifdef MAESTRO_MPI_GPU_TESTS
  if (mpi) {
    try {
      Factory::FinalizeDistributedMpiGpuBackend();
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      code = 1;
    }
    MPI_Finalize();
  }
#endif
  if (code == 0) std::cout << "Distributed GPU integration passed\n";
  return code;
}
