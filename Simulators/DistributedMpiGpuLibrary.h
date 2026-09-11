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
  static std::shared_ptr<DistributedMpiGpuLibrary> GetInstance() {
    static auto lib = std::shared_ptr<DistributedMpiGpuLibrary>(
        new DistributedMpiGpuLibrary());
    return lib;
  }
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
  void* CreateMpiNative(const Communicator* comm, int device,
                        unsigned p2pBits) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    RequireRuntime();
    Check(validate(comm, std::getenv("MAESTRO_LICENSE_KEY")),
          "ValidateMpiLicenseRuntime");
    if (!context) context = InitLib();
    // The plugin coordinates invalid context/license admission across ranks.
    auto obj = create(context, comm, device, p2pBits);
    if (!obj) Fail("CreateMpiStateVectorRuntime");
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
  int (*validate)(const Communicator*, const char*) = nullptr;
  void* (*create)(void*, const Communicator*, int32_t, uint32_t) = nullptr;
  int (*finalize)() = nullptr;
  bool finalized = false, runtimeLoaded = false;
};
}  // namespace Simulators
#endif
