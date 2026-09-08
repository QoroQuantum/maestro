/** GPU SVD wrapper/configuration coverage; --config-only uses the mock plugin.
 */
#include "../../Simulators/Factory.h"
#include "../../Simulators/Configuration.h"
#include "../../Network/SimpleDisconnectedNetwork.h"
#include <cmath>
#include <iostream>

using namespace Simulators;
using Factory = SimulatorsFactory;
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void Reject(F action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid/unsupported SVD setting accepted");
}

void CheckConfiguration() {
  for (const auto* prefix :
       {"matrix_product_state_use_gesvd", "matrix_product_operator_use_gesvd",
        "tensor_network_use_gesvd"}) {
    const std::string group(prefix);
    Configuration config;
    config.SetConfiguration(group + 'j', "true");
    config.SetConfiguration(group + 'p', "1");
    Require(config.GetConfiguration(group + 'j') == "false",
            "P did not clear J");
    config.SetConfiguration(group + 'r', "true");
    Require(config.GetConfiguration(group + 'p') == "false",
            "R did not clear P");
    Reject([&] { config.SetConfiguration(group + 'p', "invalid"); });
    Require(config.GetConfiguration(group + 'r') == "true",
            "invalid flag changed selection");
    Configuration replay;
    replay.ApplyConfigurationFromMap(config.GetConfigMap());
    Require(replay.GetConfigMap() == config.GetConfigMap(),
            "unordered replay changed selection");
    config.SetConfiguration(group + 'r', "false");
    Require(config.GetConfiguration(group + 'r') == "false",
            "clearing R failed");
  }
}

template <class Backend>
void Select(Backend& backend, int algorithm) {
  if (algorithm == 0) {
    Require(backend.SetGesvdJ(false) && backend.SetGesvdP(false) &&
                backend.SetGesvdR(false),
            "reset SVD failed");
  } else if (algorithm == 1)
    Require(backend.SetGesvdJ(true), "set J failed");
  else if (algorithm == 2)
    Require(backend.SetGesvdP(true), "set P failed");
  else
    Require(backend.SetGesvdR(true), "set R failed");
  Require(backend.GetGesvdJ() == (algorithm == 1), "J flag mismatch");
  Require(backend.GetGesvdP() == (algorithm == 2), "P flag mismatch");
  Require(backend.GetGesvdR() == (algorithm == 3), "R flag mismatch");
}

template <class Backend>
void CheckSplits(const std::shared_ptr<GpuLibrary>& lib) {
  for (int precision : {0, 1}) {
    for (int algorithm : {0, 1, 2, 3}) {
      Backend backend(lib);
      backend.SetDataType(precision);
      backend.SetMaxExtent(8);
      Select(backend, algorithm);
      Require(backend.Create(4), "SVD backend creation failed");
      Require(backend.GetLastSvdAlgo() == -1,
              "split reported before first gate");
      backend.ApplyH(0);
      backend.ApplyCX(0, 1);
      Require(backend.GetLastSvdAlgo() == algorithm,
              "requested SVD was not executed");
      // Selection remains live after allocation and is retained by native
      // clones.
      const int next = (algorithm + 1) % 4;
      Select(backend, next);
      backend.ApplyCX(1, 2);
      Require(backend.GetLastSvdAlgo() == next,
              "dynamic SVD switch not executed");
      auto clone = backend.Clone();
      Require(bool(clone), "clone failed");
      Require(clone->GetGesvdP() == (next == 2) &&
                  clone->GetGesvdR() == (next == 3),
              "clone lost SVD selection");
    }
  }
}

void CheckSimulatorConfiguration() {
  for (const auto& entry :
       {std::make_pair(SimulationType::kMatrixProductState,
                       "matrix_product_state_use_gesvd"),
        std::make_pair(SimulationType::kMatrixProductOperator,
                       "matrix_product_operator_use_gesvd"),
        std::make_pair(SimulationType::kTensorNetwork,
                       "tensor_network_use_gesvd")}) {
    const std::string group(entry.second);
    auto sim = Factory::CreateSimulator(SimulatorType::kGpuSim, entry.first);
    Require(bool(sim), "GPU simulator unavailable");
    sim->Configure((group + 'j').c_str(), "true");
    sim->Configure((group + 'p').c_str(), "true");
    sim->Configure("matrix_product_state_max_bond_dimension", "8");
    sim->AllocateQubits(4);
    sim->Initialize();
    // GetConfiguration reads the live native flags once initialized.
    Require(sim->GetConfiguration((group + 'p').c_str()) == "true",
            "pre-init P not applied");
    Require(sim->GetConfiguration((group + 'j').c_str()) == "false",
            "stale J replayed");
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    sim->Configure((group + 'r').c_str(), "true");
    Require(sim->GetConfiguration((group + 'r').c_str()) == "true",
            "live R not applied");
    Require(sim->GetConfiguration((group + 'p').c_str()) == "false",
            "live R did not clear P");
    Reject([&] { sim->Configure((group + 'j').c_str(), "invalid"); });
    if (entry.first != SimulationType::kTensorNetwork) {
      auto clone = sim->Clone();
      Require(clone->GetConfiguration((group + 'r').c_str()) == "true",
              "simulator clone lost R");
    }
    sim->Clear();
    sim->AllocateQubits(4);
    sim->Initialize();
    Require(sim->GetConfiguration((group + 'r').c_str()) == "true",
            "recreation lost R");
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    Require(std::abs(sim->ExpectationValue("ZZII") - 1.) < 1e-3,
            "SVD evolution changed Bell correlation");

    Network::SimpleDisconnectedNetwork<> network({4}, {4});
    network.Configure((group + 'j').c_str(), "true");
    network.Configure((group + 'p').c_str(), "true");
    network.CreateSimulator(SimulatorType::kGpuSim, entry.first);
    network.Configure((group + 'r').c_str(), "true");
    network.CreateSimulator(SimulatorType::kGpuSim, entry.first);
    Require(network.GetSimulator()->GetConfiguration((group + 'r').c_str()) ==
                "true",
            "network replay lost R");
    Require(network.GetSimulator()->GetConfiguration((group + 'p').c_str()) ==
                "false",
            "network replay restored stale P");
  }
}

int main(int argc, char** argv) {
  try {
    Require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--config-only"),
            "usage: gpu_svd_tests [--config-only (mock plugin required)]");
    CheckConfiguration();
    Require(Factory::InitGpuLibraryWithMute(), "GPU plugin unavailable");
    auto lib = Factory::GetGpuLibrary();
    if (argc > 1) {
      Require(lib->GetFunction("MockInitializations") != nullptr &&
                  !lib->GetFunction("MPSSetGesvdP") &&
                  !lib->GetFunction("MPOSetGesvdR") &&
                  !lib->GetFunction("TNSetGesvdP"),
              "--config-only requires the legacy mock GPU plugin");
      // The legacy mock has no P/R API: existing use must remain valid, while
      // explicitly requesting a missing SVD API fails without a null call.
      GpuDeviceContext context(lib);
      void* obj = context->CreateStateVector();
      Require(obj != nullptr, "mock state creation failed");
      Require(!context->MPSSetGesvdP(obj, 1) &&
                  !context->MPOSetGesvdR(obj, 1) &&
                  !context->TNSetGesvdP(obj, 1),
              "missing API claimed success");
      Reject([&] { context->MPSGetGesvdR(obj); });
      Reject([&] { context->MPOGetLastSvdAlgo(obj); });
      context->DestroyStateVector(obj);
    } else {
      CheckSplits<GpuLibMPSSim>(lib);
      CheckSplits<GpuMPO>(lib);
      for (int precision : {0, 1}) {
        for (int algorithm : {0, 1, 2, 3}) {
          GpuLibTNSim tn(lib);
          tn.SetDataType(precision);
          tn.SetMaxExtent(4);
          Select(tn, algorithm);
          Require(tn.Create(2), "TN creation failed");
          tn.ApplyH(0);
          tn.ApplyCX(0, 1);
          Require(std::abs(tn.ExpectationValue("ZZ") - 1.) < 1e-3,
                  "TN SVD Bell correlation failed");
        }
      }
      CheckSimulatorConfiguration();
    }
    std::cout << "GPU SVD tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
