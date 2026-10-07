// MPI remains behind the plugin C ABI; the application owns its lifecycle.
#pragma once
#ifdef __linux__
#include "DistributedGpuLibrary.h"
#include <vector>
namespace Simulators {
class DistributedMpiGpuLibrary : public DistributedGpuLibrary {
 public:
  using Communicator = DistributedGpuApi::MgdMpiCommunicator;
  using RuntimeInfo = DistributedGpuApi::MgdMpiRuntimeInfo;
  static std::shared_ptr<DistributedMpiGpuLibrary> GetInstance();
  void* CreateNative(int, int) override {
    throw std::logic_error("MPI GPU states require a communicator");
  }
  RuntimeInfo GetRuntimeInfo(const Communicator* comm) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    RequireRuntime();
    RuntimeInfo info{sizeof(RuntimeInfo), 0, 0, 0};
    Check(getInfo(comm, &info), "GetMpiRuntimeInfo");
    return info;
  }
  void GatherDevices(const Communicator* comm, int32_t device,
                     std::vector<int32_t>& devices) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    RequireRuntime();
    Check(gather(comm, device, devices.data(), devices.size()),
          "GatherMpiDevices");
  }
  uint64_t GenerateSeed(const Communicator* comm) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    RequireRuntime();
    // Resolve on demand so explicit-seed execution also supports older plugins.
    if (!generateSeed)
      generateSeed = Resolve<decltype(generateSeed)>("GenerateMpiSeed");
    uint64_t seed = 0;
    Check(generateSeed(comm, &seed), "GenerateMpiSeed");
    return seed;
  }
  void* CreateMpiNative(const Communicator* comm, int device,
                        unsigned p2pBits) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    RequireRuntime();
    if (!initializationAttempted) {
      initializationAttempted = true;
      if (validate(comm, std::getenv("MAESTRO_LICENSE_KEY")) != 1) {
        const char* detail = GetLastError();
        initializationError = detail && *detail ? detail : "license validation failed";
        throw std::runtime_error("MPI GPU backend unavailable: " + initializationError);
      }
      context = InitLib();
      if (!context) {
        const char* detail = GetLastError();
        initializationError = detail && *detail ? detail : "InitLib failed";
      }
    }
    // Every rank enters, even when InitLib returned null. This agrees on
    // library availability before any vendor collective construction.
    auto obj = create(context, comm, device, p2pBits);
    if (!obj) {
      if (!initializationError.empty())
        throw std::runtime_error("MPI GPU backend unavailable: " + initializationError);
      Fail("CreateMpiStateVectorRuntime");
    }
    ++liveStates;
    return obj;
  }
  // Explicit terminal shutdown, after all states and before MPI_Finalize.
  void FinalizeBackend() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (liveStates != 0)
      throw std::logic_error(
          "Destroy all MPI GPU states before finalizing the backend");
    if (finalized) return;
    if (context) {
      RequireRuntime();
      Check(finalize(), "FinalizeMpiBackend");
    }
    finalized = true;
  }

 private:
  DistributedMpiGpuLibrary() : DistributedGpuLibrary(true) {}
  template <typename T>
  T Resolve(const char* name) {
    auto fn = reinterpret_cast<T>(GetFunction(name));
    if (!fn)
      throw std::runtime_error(std::string("MPI GPU plugin missing ") + name +
                               "; update maestro-gpu-distributed");
    return fn;
  }
  void RequireRuntime() {
    if (finalized) throw std::runtime_error("MPI GPU backend is finalized");
    RequireLoaded();
    if (runtimeLoaded) return;
    getInfo = Resolve<decltype(getInfo)>("GetMpiRuntimeInfo");
    gather = Resolve<decltype(gather)>("GatherMpiDevices");
    validate = Resolve<decltype(validate)>("ValidateMpiLicenseRuntime");
    create = Resolve<decltype(create)>("CreateMpiStateVectorRuntime");
    finalize = Resolve<decltype(finalize)>("FinalizeMpiBackend");
    runtimeLoaded = true;
  }
  int (*getInfo)(const Communicator*, RuntimeInfo*) = nullptr;
  int (*gather)(const Communicator*, int32_t, int32_t*, uint32_t) = nullptr;
  int (*generateSeed)(const Communicator*, uint64_t*) = nullptr;
  int (*validate)(const Communicator*, const char*) = nullptr;
  void* (*create)(void*, const Communicator*, int32_t, uint32_t) = nullptr;
  int (*finalize)() = nullptr;
  bool finalized = false, runtimeLoaded = false;
  bool initializationAttempted = false;
  std::string initializationError;
};
}  // namespace Simulators
#endif
