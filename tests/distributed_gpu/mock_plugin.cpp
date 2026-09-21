// CPU-only loader fixture. Unused numerical entry points abort so accidental
// execution cannot silently pass. Deliberately does not export CheckLicense.
#include "../gpu_device/mock_plugin.cpp"
#include "../../Simulators/DistributedGpuApi.h"
using namespace Simulators::DistributedGpuApi;
extern "C" {
uint32_t GetApiVersion() { return 1; }
uint64_t GetCapabilities() { return 31; }
const char* GetLastError() { return GetLicenseError(); }
void* CreateStateVectorWithBackend(void* lib, int) {
  return lib ? CreateStateVector(lib) : nullptr;
}
int ValidateMpiLicenseRuntime(const MgdMpiCommunicator*, const char* key) {
  return ValidateLicense(key);
}
void* CreateMpiStateVectorRuntime(void* lib, const MgdMpiCommunicator*, int,
                                  unsigned) {
  return lib ? CreateStateVector(lib) : nullptr;
}
int GetMpiRuntimeInfo(const MgdMpiCommunicator*, MgdMpiRuntimeInfo* info) {
  *info = {sizeof(*info), 1, 0, 0};
  return 1;
}
int GatherMpiDevices(const MgdMpiCommunicator*, int32_t device,
                     int32_t* devices, uint32_t) {
  *devices = device;
  return 1;
}
int FinalizeMpiBackend() { return 1; }

#define UNUSED(name) \
  void name() { std::abort(); }
UNUSED(GetBackend)
UNUSED(SetExExecutionConfig)
UNUSED(SetSeed)
UNUSED(CreateWithState)
UNUSED(Reset)
UNUSED(MeasureQubitCollapse)
UNUSED(MeasureQubitNoCollapse)
UNUSED(MeasureQubitsCollapse)
UNUSED(MeasureQubitsNoCollapse)
UNUSED(MeasureAllQubitsCollapse)
UNUSED(MeasureAllQubitsNoCollapse)
UNUSED(SaveState)
UNUSED(SaveStateToHost)
UNUSED(SaveStateDestructive)
UNUSED(RestoreStateFreeSaved)
UNUSED(RestoreStateNoFreeSaved)
UNUSED(FreeSavedState)
UNUSED(Sample)
UNUSED(SampleAll)
UNUSED(Amplitude)
UNUSED(Probability)
UNUSED(BasisStateProbability)
UNUSED(ExpectationValue)
UNUSED(ApplyY)
UNUSED(ApplyZ)
UNUSED(ApplyH)
UNUSED(ApplyS)
UNUSED(ApplySDG)
UNUSED(ApplyT)
UNUSED(ApplyTDG)
UNUSED(ApplySX)
UNUSED(ApplySXDG)
UNUSED(ApplyK)
UNUSED(ApplyP)
UNUSED(ApplyRx)
UNUSED(ApplyRy)
UNUSED(ApplyRz)
UNUSED(ApplyU)
UNUSED(ApplyCX)
UNUSED(ApplyCY)
UNUSED(ApplyCZ)
UNUSED(ApplyCH)
UNUSED(ApplyCSX)
UNUSED(ApplyCSXDG)
UNUSED(ApplyCP)
UNUSED(ApplyCRx)
UNUSED(ApplyCRy)
UNUSED(ApplyCRz)
UNUSED(ApplyCCX)
UNUSED(ApplySwap)
UNUSED(ApplyCSwap)
UNUSED(ApplyCU)
UNUSED(ConfigureDistribution)
UNUSED(GetShardDevices)
UNUSED(GetGlobalQubits)
UNUSED(GetQubitLayout)
UNUSED(Redistribute)
UNUSED(SwapGlobalLocalQubits)
UNUSED(Synchronize)
UNUSED(GetStateRange)
UNUSED(SetStateRange)
UNUSED(GetLocalStateBounds)
UNUSED(GetLocalStateRange)
UNUSED(SetLocalStateRange)
UNUSED(CreateWithBasisState)
UNUSED(ApplyOneQubitMatrix)
UNUSED(ApplyOneQubitMatrixWithLayout)
UNUSED(ApplyTwoQubitMatrix)
UNUSED(ApplyTwoQubitMatrixWithLayout)
#undef UNUSED
}
