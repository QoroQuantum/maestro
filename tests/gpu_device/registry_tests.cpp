#include "../../Simulators/GpuLibraryRegistry.h"
#include "../../Simulators/GpuLibStateVectorSim.h"
#include "../../Simulators/GpuDensityMatrix.h"
#include "../../Simulators/GpuMPO.h"
#include <future>
#include <iostream>
#include <vector>

using namespace Simulators;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void Exercise(const std::shared_ptr<GpuLibrary>& library) {
  GpuLibStateVectorSim state(library);
  Require(state.Create(2), "create failed");
  state.ApplyX(1);
  double values[4]{};
  Require(state.AllProbabilities(values), "probability query failed");
  Require(values[2] > 0.99999, "incorrect state / device");
  auto clone = state.Clone();
  Require(bool(clone), "clone failed");
  clone->ApplyX(1);
  Require(clone->AllProbabilities(values), "clone query failed");
  Require(values[0] > 0.99999, "incorrect clone");
}
int main(int argc, char** argv) {
  try {
    Require(argc >= 2, "usage: gpu_registry_tests plugin-path [--real]");
    const bool real = argc > 2 && std::string(argv[2]) == "--real";
    GpuLibraryRegistry registry(argv[1]);
    Require(registry.DeviceCount() >= 1, "no visible GPU devices");
    auto first = registry.Acquire(0, true);
    Require(bool(first), "device 0 initialization failed");
    Require(first == registry.Acquire(0, true), "device 0 was not cached");
    std::vector<std::future<std::shared_ptr<GpuLibrary>>> requests;
    // Race the first request for a second device in the mock case.
    for (int i = 0; i < 8; ++i)
      requests.push_back(std::async(std::launch::async, [&] {
        return registry.Acquire(real ? 0 : 1, true);
      }));
    auto second = requests.front().get();
    Require(bool(second), "second device initialization failed");
    for (size_t i = 1; i < requests.size(); ++i)
      Require(requests[i].get() == second,
              "concurrent acquisition duplicated a library");
    if (!real) {
      Require(first != second && first->GetHandle() != second->GetHandle(),
              "namespaces not isolated");
      for (auto lib : {first, second}) {
        auto count = reinterpret_cast<int (*)()>(
            lib->GetFunction("MockInitializations"));
        Require(count() == 1, "library initialized more than once");
      }
      // Device activation must restore the caller's previous device even
      // when several simulators are interleaved on one host thread.
      auto setDevice =
          reinterpret_cast<int (*)(int)>(first->GetFunction("cudaSetDevice"));
      auto getDevice =
          reinterpret_cast<int (*)(int*)>(first->GetFunction("cudaGetDevice"));
      Require(setDevice(2) == 0, "mock device setup failed");
      Exercise(first);
      int restored = -1;
      Require(getDevice(&restored) == 0 && restored == 2,
              "caller device not restored");
      Require(!registry.Acquire(2, true), "failed init published in cache");
      Require(!registry.Acquire(2, true), "failed init cached as successful");
    } else if (registry.DeviceCount() > 1) {
      second = registry.Acquire(1, true);
      Require(bool(second), "real device 1 initialization failed");
    } else {
      // Deliberately bypass the cache to exercise two real CUDA namespaces on
      // a single physical GPU. Production creates only one per device.
      second = std::make_shared<GpuLibrary>();
      second->SetMute(true);
      second->SetRequestedGpuDevice(0);
      Require(second->Init(argv[1]),
              "second real namespace initialization failed");
    }
    Require(!registry.Acquire(registry.DeviceCount(), true),
            "invalid device accepted");
    Require(registry.Acquire(0, true) == first,
            "invalid request poisoned cache");
    for (int i = 0; i < 3; ++i) {
      Exercise(first);
      Exercise(second);
    }
    // Create, operate and destroy on different host threads.
    auto state = std::async(std::launch::async, [&] {
                   auto result = std::make_unique<GpuLibStateVectorSim>(first);
                   Require(result->Create(2), "worker create failed");
                   return result;
                 }).get();
    std::async(std::launch::async, [&] { state->ApplyX(0); }).get();
    double values[4]{};
    Require(state->AllProbabilities(values) && values[1] > 0.99999,
            "worker device selection failed");
    std::async(std::launch::async, [&] { state.reset(); }).get();
    // Separate simulators/namespaces run concurrently (the mock live-state
    // counter is intentionally used only by one thread per namespace here).
    auto a = std::async(std::launch::async, [&] { Exercise(first); });
    auto b = std::async(std::launch::async, [&] { Exercise(second); });
    a.get();
    b.get();
    if (real) {
      if (first->HasDensityMatrixAPI() && second->HasDensityMatrixAPI()) {
        GpuDensityMatrix x(first), y(second);
        Require(x.Create(2) && y.Create(2), "density matrix creation failed");
        bool rejected = false;
        try {
          x.HilbertSchmidtOverlap(y);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        Require(rejected, "cross-context density overlap accepted");
        Require(std::abs(x.HilbertSchmidtOverlap(x).real() - 1.) < 1e-5,
                "same-context density overlap failed");
      }
      if (first->HasMPOAPI() && second->HasMPOAPI()) {
        GpuMPO x(first), y(second);
        Require(x.Create(2) && y.Create(2), "MPO creation failed");
        bool rejected = false;
        try {
          x.HilbertSchmidtOverlap(y);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        Require(rejected, "cross-context MPO overlap accepted");
        Require(std::abs(x.HilbertSchmidtOverlap(x).real() - 1.) < 1e-5,
                "same-context MPO overlap failed");
      }
    }
    // Outstanding wrappers own their library even if the registry goes away.
    std::unique_ptr<GpuLibStateVectorSim> survivor;
    {
      GpuLibraryRegistry temporary(argv[1]);
      survivor =
          std::make_unique<GpuLibStateVectorSim>(temporary.Acquire(0, true));
      Require(survivor->Create(1), "survivor creation failed");
    }
    survivor->ApplyX(0);
    Require(survivor->AllProbabilities(values) && values[1] > 0.99999,
            "library lifetime ended too early");
    survivor.reset();
    second.reset();
    Exercise(first);
    std::cout << (real ? "Real GPU namespace" : "Mock GPU registry")
              << " tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
