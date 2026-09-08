#include "../../Simulators/GpuLibraryRegistry.h"
#include "../../Simulators/GpuLibStateVectorSim.h"
#include "../../Simulators/GpuPauliPropagator.h"
#include "../../Simulators/GpuStabilizer.h"
#include <future>
#include <iostream>
#include <vector>

using namespace Simulators;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::unique_ptr<GpuLibStateVectorSim> Create(GpuLibraryRegistry& registry, int device) {
  auto lock = GpuLibrary::GetInstance()->LockInitialization();
  auto library = registry.Acquire(device, true);
  Require(bool(library), "device initialization failed");
  auto state = std::make_unique<GpuLibStateVectorSim>(library);
  Require(state->Create(2), "state creation failed");
  return state;
}

int main(int argc, char** argv) {
  try {
    Require(argc == 2 || (argc == 3 && std::string(argv[2]) == "--real"),
            "usage: gpu_registry_tests plugin-path [--real]");
    const bool real = argc == 3;
    GpuLibraryRegistry registry(argv[1]);
    Require(registry.DeviceCount() >= 1, "no visible GPU devices");
    auto first = registry.Acquire(0, true);
    Require(bool(first), "device 0 initialization failed");
    const int secondDevice = registry.DeviceCount() > 1 ? 1 : 0;
    auto second = registry.Acquire(secondDevice, true);
    Require(bool(second), "second device selection failed");
    Require(first == second && first == GpuLibrary::GetInstance(),
            "devices did not share the singleton");
    GpuLibraryRegistry anotherRegistry(argv[1]);
    Require(anotherRegistry.Acquire(0, true) == first,
            "another registry created another GPU library");

    std::vector<std::future<std::shared_ptr<GpuLibrary>>> requests;
    for (int i = 0; i < 8; ++i)
      requests.push_back(std::async(std::launch::async, [&, i] {
        return registry.Acquire(i % 2 ? secondDevice : 0, true);
      }));
    for (auto& request : requests)
      Require(request.get() == first, "concurrent acquisition lost singleton identity");
    if (!real) {
      auto count = reinterpret_cast<int (*)()>(first->GetFunction("MockInitializations"));
      auto selected = reinterpret_cast<int (*)()>(first->GetFunction("MockSelectedDevice"));
      Require(count && selected, "mock inspection API missing");
      Require(count() == 1, "InitLib must run once, not once per device");
      registry.Acquire(1, true);
      Require(selected() == 1, "device selection was not reapplied");
      registry.Acquire(0, true);
      Require(selected() == 0, "returning to device 0 did not select it");
      Require(!registry.Acquire(2, true), "failed selection accepted");
      Require(!registry.Acquire(2, true), "failed selection cached as successful");
      Require(count() == 1, "device selection reinitialized the plugin");
    }
    Require(!registry.Acquire(registry.DeviceCount(), true), "invalid device accepted");
    Require(registry.Acquire(0, true) == first, "invalid request poisoned singleton");

    auto zero = Create(registry, 0);
    auto one = Create(registry, secondDevice);
    if (!real) {
      auto creations = reinterpret_cast<int (*)(int)>(first->GetFunction("MockCreationsOnDevice"));
      Require(creations && creations(0) == 1 && creations(1) == 1,
              "native objects were not created on their requested devices");
    }
    Require(zero->GetGpuDevice() == 0 && one->GetGpuDevice() == secondDevice,
            "native device queries disagree with requested placement");
    // Wrappers with delayed native creation must remember their own selection.
    {
      Require(first->SetGpuDevice(0), "select delayed wrapper default");
      GpuPauliPropagator pauli(first);
      GpuStabilizer stabilizer(first, secondDevice);
      Require(first->SetGpuDevice(secondDevice), "change selection before delayed creation");
      Require(pauli.CreateSimulator(1) && pauli.GetGpuDevice() == 0,
              "delayed Pauli creation followed a later selection");
      Require(first->SetGpuDevice(0), "change selection before stabilizer creation");
      Require(stabilizer.CreateSimulator(2, 4, 2, 0) &&
                  stabilizer.GetGpuDevice() == secondDevice,
              "delayed stabilizer creation lost explicit device");
      Require(pauli.CreateSimulator(2) && pauli.GetGpuDevice() == 0,
              "Pauli recreation changed GPU");
      stabilizer.Clear();
      Require(stabilizer.GetGpuDevice() == -1 &&
                  stabilizer.CreateSimulator(2, 4, 2, 0) &&
                  stabilizer.GetGpuDevice() == secondDevice,
              "stabilizer clear/recreation lost GPU");
      GpuLibStateVectorSim explicitDevice(first, secondDevice);
      Require(explicitDevice.GetGpuDevice() == secondDevice,
              "direct wrapper constructor ignored explicit device");
    }
    // Calls do not select a device in Maestro. The plugin must retain each
    // object's device, including when objects are interleaved or cloned.
    zero->ApplyX(0);
    one->ApplyX(1);
    double values[4]{};
    Require(zero->AllProbabilities(values) && values[1] > 0.99999,
            "first instance lost its state/device");
    Require(one->AllProbabilities(values) && values[2] > 0.99999,
            "second instance lost its state/device");
    auto clone = zero->Clone();
    Require(bool(clone), "clone failed");
    Require(clone->GetGpuDevice() == 0, "clone migrated to the selected GPU");
    clone->ApplyX(1);
    Require(clone->AllProbabilities(values) && values[3] > 0.99999, "clone lost state/device");
    zero.reset();
    Require(one->AllProbabilities(values) && values[2] > 0.99999,
            "peer destruction invalidated singleton");

    // A native object's device belongs to the plugin, not its calling thread.
    std::async(std::launch::async, [&] { one->ApplyX(0); }).get();
    Require(one->AllProbabilities(values) && values[3] > 0.99999,
            "worker call lost state/device");
    std::async(std::launch::async, [&] { one.reset(); }).get();
    std::unique_ptr<GpuLibStateVectorSim> survivor;
    {
      GpuLibraryRegistry temporary(argv[1]);
      survivor = Create(temporary, 0);
    }
    survivor->ApplyX(0);
    Require(survivor->AllProbabilities(values) && values[1] > 0.99999,
            "registry destruction invalidated singleton");
    std::cout << (real ? "Real GPU singleton" : "Mock GPU singleton")
              << " tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
