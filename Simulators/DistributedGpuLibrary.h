// Dynamic loader for the local distributed statevector plugin.
// The C ABI boundary is covered by the Maestro Plugin Linking Exception.
#pragma once
#ifdef __linux__
#include "../Utils/Library.h"
#include "DistributedGpuApi.h"
#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace Simulators {
class DistributedGpuLibrary : public Utils::Library {
 public:
  static std::shared_ptr<DistributedGpuLibrary> GetInstance() {
    static auto lib =
        std::shared_ptr<DistributedGpuLibrary>(new DistributedGpuLibrary());
    return lib;
  }
  ~DistributedGpuLibrary() override {
    if (context && FreeLib) FreeLib();
  }
  bool Load() {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (loaded) return true;
    const char* path = std::getenv(mpi ? "MAESTRO_DIST_MPI_GPU_LIBRARY"
                                       : "MAESTRO_DIST_GPU_LIBRARY");
    if (!path || !*path)
      path = mpi ? "libmaestro_gpu_distributed_mpi.so"
                 : "libmaestro_gpu_distributed.so";
    SetMute(true);
    if (!Utils::Library::Init(path)) return false;
    // Resolve every required entry before permitting any state construction.
    SetGpuDevice = reinterpret_cast<DistributedGpuApi::SetGpuDeviceFn>(
        GetFunction("SetGpuDevice"));
    if (!SetGpuDevice)
      throw std::runtime_error("Distributed GPU plugin missing SetGpuDevice");
    GetGpuDeviceCount =
        reinterpret_cast<DistributedGpuApi::GetGpuDeviceCountFn>(
            GetFunction("GetGpuDeviceCount"));
    if (!GetGpuDeviceCount)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetGpuDeviceCount");
    ValidateLicense = reinterpret_cast<DistributedGpuApi::ValidateLicenseFn>(
        GetFunction("ValidateLicense"));
    if (!ValidateLicense)
      throw std::runtime_error(
          "Distributed GPU plugin missing ValidateLicense");
    CheckLicense = reinterpret_cast<DistributedGpuApi::CheckLicenseFn>(
        GetFunction("CheckLicense"));
    if (!CheckLicense)
      throw std::runtime_error("Distributed GPU plugin missing CheckLicense");
    InitLib =
        reinterpret_cast<DistributedGpuApi::InitLibFn>(GetFunction("InitLib"));
    if (!InitLib)
      throw std::runtime_error("Distributed GPU plugin missing InitLib");
    FreeLib =
        reinterpret_cast<DistributedGpuApi::FreeLibFn>(GetFunction("FreeLib"));
    if (!FreeLib)
      throw std::runtime_error("Distributed GPU plugin missing FreeLib");
    CreateStateVector =
        reinterpret_cast<DistributedGpuApi::CreateStateVectorFn>(
            GetFunction("CreateStateVector"));
    if (!CreateStateVector)
      throw std::runtime_error(
          "Distributed GPU plugin missing CreateStateVector");
    CreateStateVectorWithBackend =
        reinterpret_cast<DistributedGpuApi::CreateStateVectorWithBackendFn>(
            GetFunction("CreateStateVectorWithBackend"));
    if (!CreateStateVectorWithBackend)
      throw std::runtime_error(
          "Distributed GPU plugin missing CreateStateVectorWithBackend");
    GetBackend = reinterpret_cast<DistributedGpuApi::GetBackendFn>(
        GetFunction("GetBackend"));
    if (!GetBackend)
      throw std::runtime_error("Distributed GPU plugin missing GetBackend");
    SetExExecutionConfig =
        reinterpret_cast<DistributedGpuApi::SetExExecutionConfigFn>(
            GetFunction("SetExExecutionConfig"));
    if (!SetExExecutionConfig)
      throw std::runtime_error(
          "Distributed GPU plugin missing SetExExecutionConfig");
    DestroyStateVector =
        reinterpret_cast<DistributedGpuApi::DestroyStateVectorFn>(
            GetFunction("DestroyStateVector"));
    if (!DestroyStateVector)
      throw std::runtime_error(
          "Distributed GPU plugin missing DestroyStateVector");
    SetDataType = reinterpret_cast<DistributedGpuApi::SetDataTypeFn>(
        GetFunction("SetDataType"));
    if (!SetDataType)
      throw std::runtime_error("Distributed GPU plugin missing SetDataType");
    SetSeed =
        reinterpret_cast<DistributedGpuApi::SetSeedFn>(GetFunction("SetSeed"));
    if (!SetSeed)
      throw std::runtime_error("Distributed GPU plugin missing SetSeed");
    IsDoublePrecision =
        reinterpret_cast<DistributedGpuApi::IsDoublePrecisionFn>(
            GetFunction("IsDoublePrecision"));
    if (!IsDoublePrecision)
      throw std::runtime_error(
          "Distributed GPU plugin missing IsDoublePrecision");
    GetNrQubits = reinterpret_cast<DistributedGpuApi::GetNrQubitsFn>(
        GetFunction("GetNrQubits"));
    if (!GetNrQubits)
      throw std::runtime_error("Distributed GPU plugin missing GetNrQubits");
    GetStateVectorGpuId =
        reinterpret_cast<DistributedGpuApi::GetStateVectorGpuIdFn>(
            GetFunction("GetStateVectorGpuId"));
    if (!GetStateVectorGpuId)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetStateVectorGpuId");
    Create =
        reinterpret_cast<DistributedGpuApi::CreateFn>(GetFunction("Create"));
    if (!Create)
      throw std::runtime_error("Distributed GPU plugin missing Create");
    CreateWithState = reinterpret_cast<DistributedGpuApi::CreateWithStateFn>(
        GetFunction("CreateWithState"));
    if (!CreateWithState)
      throw std::runtime_error(
          "Distributed GPU plugin missing CreateWithState");
    Reset = reinterpret_cast<DistributedGpuApi::ResetFn>(GetFunction("Reset"));
    if (!Reset)
      throw std::runtime_error("Distributed GPU plugin missing Reset");
    MeasureQubitCollapse =
        reinterpret_cast<DistributedGpuApi::MeasureQubitCollapseFn>(
            GetFunction("MeasureQubitCollapse"));
    if (!MeasureQubitCollapse)
      throw std::runtime_error(
          "Distributed GPU plugin missing MeasureQubitCollapse");
    MeasureQubitNoCollapse =
        reinterpret_cast<DistributedGpuApi::MeasureQubitNoCollapseFn>(
            GetFunction("MeasureQubitNoCollapse"));
    if (!MeasureQubitNoCollapse)
      throw std::runtime_error(
          "Distributed GPU plugin missing MeasureQubitNoCollapse");
    MeasureQubitsCollapse =
        reinterpret_cast<DistributedGpuApi::MeasureQubitsCollapseFn>(
            GetFunction("MeasureQubitsCollapse"));
    if (!MeasureQubitsCollapse)
      throw std::runtime_error(
          "Distributed GPU plugin missing MeasureQubitsCollapse");
    MeasureQubitsNoCollapse =
        reinterpret_cast<DistributedGpuApi::MeasureQubitsNoCollapseFn>(
            GetFunction("MeasureQubitsNoCollapse"));
    if (!MeasureQubitsNoCollapse)
      throw std::runtime_error(
          "Distributed GPU plugin missing MeasureQubitsNoCollapse");
    MeasureAllQubitsCollapse =
        reinterpret_cast<DistributedGpuApi::MeasureAllQubitsCollapseFn>(
            GetFunction("MeasureAllQubitsCollapse"));
    if (!MeasureAllQubitsCollapse)
      throw std::runtime_error(
          "Distributed GPU plugin missing MeasureAllQubitsCollapse");
    MeasureAllQubitsNoCollapse =
        reinterpret_cast<DistributedGpuApi::MeasureAllQubitsNoCollapseFn>(
            GetFunction("MeasureAllQubitsNoCollapse"));
    if (!MeasureAllQubitsNoCollapse)
      throw std::runtime_error(
          "Distributed GPU plugin missing MeasureAllQubitsNoCollapse");
    SaveState = reinterpret_cast<DistributedGpuApi::SaveStateFn>(
        GetFunction("SaveState"));
    if (!SaveState)
      throw std::runtime_error("Distributed GPU plugin missing SaveState");
    SaveStateToHost = reinterpret_cast<DistributedGpuApi::SaveStateToHostFn>(
        GetFunction("SaveStateToHost"));
    if (!SaveStateToHost)
      throw std::runtime_error(
          "Distributed GPU plugin missing SaveStateToHost");
    SaveStateDestructive =
        reinterpret_cast<DistributedGpuApi::SaveStateDestructiveFn>(
            GetFunction("SaveStateDestructive"));
    if (!SaveStateDestructive)
      throw std::runtime_error(
          "Distributed GPU plugin missing SaveStateDestructive");
    RestoreStateFreeSaved =
        reinterpret_cast<DistributedGpuApi::RestoreStateFreeSavedFn>(
            GetFunction("RestoreStateFreeSaved"));
    if (!RestoreStateFreeSaved)
      throw std::runtime_error(
          "Distributed GPU plugin missing RestoreStateFreeSaved");
    RestoreStateNoFreeSaved =
        reinterpret_cast<DistributedGpuApi::RestoreStateNoFreeSavedFn>(
            GetFunction("RestoreStateNoFreeSaved"));
    if (!RestoreStateNoFreeSaved)
      throw std::runtime_error(
          "Distributed GPU plugin missing RestoreStateNoFreeSaved");
    FreeSavedState = reinterpret_cast<DistributedGpuApi::FreeSavedStateFn>(
        GetFunction("FreeSavedState"));
    if (!FreeSavedState)
      throw std::runtime_error("Distributed GPU plugin missing FreeSavedState");
    Clone = reinterpret_cast<DistributedGpuApi::CloneFn>(GetFunction("Clone"));
    if (!Clone)
      throw std::runtime_error("Distributed GPU plugin missing Clone");
    Sample =
        reinterpret_cast<DistributedGpuApi::SampleFn>(GetFunction("Sample"));
    if (!Sample)
      throw std::runtime_error("Distributed GPU plugin missing Sample");
    SampleAll = reinterpret_cast<DistributedGpuApi::SampleAllFn>(
        GetFunction("SampleAll"));
    if (!SampleAll)
      throw std::runtime_error("Distributed GPU plugin missing SampleAll");
    Amplitude = reinterpret_cast<DistributedGpuApi::AmplitudeFn>(
        GetFunction("Amplitude"));
    if (!Amplitude)
      throw std::runtime_error("Distributed GPU plugin missing Amplitude");
    Probability = reinterpret_cast<DistributedGpuApi::ProbabilityFn>(
        GetFunction("Probability"));
    if (!Probability)
      throw std::runtime_error("Distributed GPU plugin missing Probability");
    BasisStateProbability =
        reinterpret_cast<DistributedGpuApi::BasisStateProbabilityFn>(
            GetFunction("BasisStateProbability"));
    if (!BasisStateProbability)
      throw std::runtime_error(
          "Distributed GPU plugin missing BasisStateProbability");
    AllProbabilities = reinterpret_cast<DistributedGpuApi::AllProbabilitiesFn>(
        GetFunction("AllProbabilities"));
    if (!AllProbabilities)
      throw std::runtime_error(
          "Distributed GPU plugin missing AllProbabilities");
    ExpectationValue = reinterpret_cast<DistributedGpuApi::ExpectationValueFn>(
        GetFunction("ExpectationValue"));
    if (!ExpectationValue)
      throw std::runtime_error(
          "Distributed GPU plugin missing ExpectationValue");
    ApplyX =
        reinterpret_cast<DistributedGpuApi::ApplyXFn>(GetFunction("ApplyX"));
    if (!ApplyX)
      throw std::runtime_error("Distributed GPU plugin missing ApplyX");
    ApplyY =
        reinterpret_cast<DistributedGpuApi::ApplyYFn>(GetFunction("ApplyY"));
    if (!ApplyY)
      throw std::runtime_error("Distributed GPU plugin missing ApplyY");
    ApplyZ =
        reinterpret_cast<DistributedGpuApi::ApplyZFn>(GetFunction("ApplyZ"));
    if (!ApplyZ)
      throw std::runtime_error("Distributed GPU plugin missing ApplyZ");
    ApplyH =
        reinterpret_cast<DistributedGpuApi::ApplyHFn>(GetFunction("ApplyH"));
    if (!ApplyH)
      throw std::runtime_error("Distributed GPU plugin missing ApplyH");
    ApplyS =
        reinterpret_cast<DistributedGpuApi::ApplySFn>(GetFunction("ApplyS"));
    if (!ApplyS)
      throw std::runtime_error("Distributed GPU plugin missing ApplyS");
    ApplySDG = reinterpret_cast<DistributedGpuApi::ApplySDGFn>(
        GetFunction("ApplySDG"));
    if (!ApplySDG)
      throw std::runtime_error("Distributed GPU plugin missing ApplySDG");
    ApplyT =
        reinterpret_cast<DistributedGpuApi::ApplyTFn>(GetFunction("ApplyT"));
    if (!ApplyT)
      throw std::runtime_error("Distributed GPU plugin missing ApplyT");
    ApplyTDG = reinterpret_cast<DistributedGpuApi::ApplyTDGFn>(
        GetFunction("ApplyTDG"));
    if (!ApplyTDG)
      throw std::runtime_error("Distributed GPU plugin missing ApplyTDG");
    ApplySX =
        reinterpret_cast<DistributedGpuApi::ApplySXFn>(GetFunction("ApplySX"));
    if (!ApplySX)
      throw std::runtime_error("Distributed GPU plugin missing ApplySX");
    ApplySXDG = reinterpret_cast<DistributedGpuApi::ApplySXDGFn>(
        GetFunction("ApplySXDG"));
    if (!ApplySXDG)
      throw std::runtime_error("Distributed GPU plugin missing ApplySXDG");
    ApplyK =
        reinterpret_cast<DistributedGpuApi::ApplyKFn>(GetFunction("ApplyK"));
    if (!ApplyK)
      throw std::runtime_error("Distributed GPU plugin missing ApplyK");
    ApplyP =
        reinterpret_cast<DistributedGpuApi::ApplyPFn>(GetFunction("ApplyP"));
    if (!ApplyP)
      throw std::runtime_error("Distributed GPU plugin missing ApplyP");
    ApplyRx =
        reinterpret_cast<DistributedGpuApi::ApplyRxFn>(GetFunction("ApplyRx"));
    if (!ApplyRx)
      throw std::runtime_error("Distributed GPU plugin missing ApplyRx");
    ApplyRy =
        reinterpret_cast<DistributedGpuApi::ApplyRyFn>(GetFunction("ApplyRy"));
    if (!ApplyRy)
      throw std::runtime_error("Distributed GPU plugin missing ApplyRy");
    ApplyRz =
        reinterpret_cast<DistributedGpuApi::ApplyRzFn>(GetFunction("ApplyRz"));
    if (!ApplyRz)
      throw std::runtime_error("Distributed GPU plugin missing ApplyRz");
    ApplyU =
        reinterpret_cast<DistributedGpuApi::ApplyUFn>(GetFunction("ApplyU"));
    if (!ApplyU)
      throw std::runtime_error("Distributed GPU plugin missing ApplyU");
    ApplyCX =
        reinterpret_cast<DistributedGpuApi::ApplyCXFn>(GetFunction("ApplyCX"));
    if (!ApplyCX)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCX");
    ApplyCY =
        reinterpret_cast<DistributedGpuApi::ApplyCYFn>(GetFunction("ApplyCY"));
    if (!ApplyCY)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCY");
    ApplyCZ =
        reinterpret_cast<DistributedGpuApi::ApplyCZFn>(GetFunction("ApplyCZ"));
    if (!ApplyCZ)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCZ");
    ApplyCH =
        reinterpret_cast<DistributedGpuApi::ApplyCHFn>(GetFunction("ApplyCH"));
    if (!ApplyCH)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCH");
    ApplyCSX = reinterpret_cast<DistributedGpuApi::ApplyCSXFn>(
        GetFunction("ApplyCSX"));
    if (!ApplyCSX)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCSX");
    ApplyCSXDG = reinterpret_cast<DistributedGpuApi::ApplyCSXDGFn>(
        GetFunction("ApplyCSXDG"));
    if (!ApplyCSXDG)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCSXDG");
    ApplyCP =
        reinterpret_cast<DistributedGpuApi::ApplyCPFn>(GetFunction("ApplyCP"));
    if (!ApplyCP)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCP");
    ApplyCRx = reinterpret_cast<DistributedGpuApi::ApplyCRxFn>(
        GetFunction("ApplyCRx"));
    if (!ApplyCRx)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCRx");
    ApplyCRy = reinterpret_cast<DistributedGpuApi::ApplyCRyFn>(
        GetFunction("ApplyCRy"));
    if (!ApplyCRy)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCRy");
    ApplyCRz = reinterpret_cast<DistributedGpuApi::ApplyCRzFn>(
        GetFunction("ApplyCRz"));
    if (!ApplyCRz)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCRz");
    ApplyCCX = reinterpret_cast<DistributedGpuApi::ApplyCCXFn>(
        GetFunction("ApplyCCX"));
    if (!ApplyCCX)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCCX");
    ApplySwap = reinterpret_cast<DistributedGpuApi::ApplySwapFn>(
        GetFunction("ApplySwap"));
    if (!ApplySwap)
      throw std::runtime_error("Distributed GPU plugin missing ApplySwap");
    ApplyCSwap = reinterpret_cast<DistributedGpuApi::ApplyCSwapFn>(
        GetFunction("ApplyCSwap"));
    if (!ApplyCSwap)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCSwap");
    ApplyCU =
        reinterpret_cast<DistributedGpuApi::ApplyCUFn>(GetFunction("ApplyCU"));
    if (!ApplyCU)
      throw std::runtime_error("Distributed GPU plugin missing ApplyCU");
    GetApiVersion = reinterpret_cast<DistributedGpuApi::GetApiVersionFn>(
        GetFunction("GetApiVersion"));
    if (!GetApiVersion)
      throw std::runtime_error("Distributed GPU plugin missing GetApiVersion");
    GetCapabilities = reinterpret_cast<DistributedGpuApi::GetCapabilitiesFn>(
        GetFunction("GetCapabilities"));
    if (!GetCapabilities)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetCapabilities");
    GetLastError = reinterpret_cast<DistributedGpuApi::GetLastErrorFn>(
        GetFunction("GetLastError"));
    if (!GetLastError)
      throw std::runtime_error("Distributed GPU plugin missing GetLastError");
    ConfigureDistribution =
        reinterpret_cast<DistributedGpuApi::ConfigureDistributionFn>(
            GetFunction("ConfigureDistribution"));
    if (!ConfigureDistribution)
      throw std::runtime_error(
          "Distributed GPU plugin missing ConfigureDistribution");
    GetShardDevices = reinterpret_cast<DistributedGpuApi::GetShardDevicesFn>(
        GetFunction("GetShardDevices"));
    if (!GetShardDevices)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetShardDevices");
    GetGlobalQubits = reinterpret_cast<DistributedGpuApi::GetGlobalQubitsFn>(
        GetFunction("GetGlobalQubits"));
    if (!GetGlobalQubits)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetGlobalQubits");
    GetQubitLayout = reinterpret_cast<DistributedGpuApi::GetQubitLayoutFn>(
        GetFunction("GetQubitLayout"));
    if (!GetQubitLayout)
      throw std::runtime_error("Distributed GPU plugin missing GetQubitLayout");
    Redistribute = reinterpret_cast<DistributedGpuApi::RedistributeFn>(
        GetFunction("Redistribute"));
    if (!Redistribute)
      throw std::runtime_error("Distributed GPU plugin missing Redistribute");
    SwapGlobalLocalQubits =
        reinterpret_cast<DistributedGpuApi::SwapGlobalLocalQubitsFn>(
            GetFunction("SwapGlobalLocalQubits"));
    if (!SwapGlobalLocalQubits)
      throw std::runtime_error(
          "Distributed GPU plugin missing SwapGlobalLocalQubits");
    Synchronize = reinterpret_cast<DistributedGpuApi::SynchronizeFn>(
        GetFunction("Synchronize"));
    if (!Synchronize)
      throw std::runtime_error("Distributed GPU plugin missing Synchronize");
    GetStateRange = reinterpret_cast<DistributedGpuApi::GetStateRangeFn>(
        GetFunction("GetStateRange"));
    if (!GetStateRange)
      throw std::runtime_error("Distributed GPU plugin missing GetStateRange");
    SetStateRange = reinterpret_cast<DistributedGpuApi::SetStateRangeFn>(
        GetFunction("SetStateRange"));
    if (!SetStateRange)
      throw std::runtime_error("Distributed GPU plugin missing SetStateRange");
    GetLocalStateBounds =
        reinterpret_cast<DistributedGpuApi::GetLocalStateBoundsFn>(
            GetFunction("GetLocalStateBounds"));
    if (!GetLocalStateBounds)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetLocalStateBounds");
    GetLocalStateRange =
        reinterpret_cast<DistributedGpuApi::GetLocalStateRangeFn>(
            GetFunction("GetLocalStateRange"));
    if (!GetLocalStateRange)
      throw std::runtime_error(
          "Distributed GPU plugin missing GetLocalStateRange");
    SetLocalStateRange =
        reinterpret_cast<DistributedGpuApi::SetLocalStateRangeFn>(
            GetFunction("SetLocalStateRange"));
    if (!SetLocalStateRange)
      throw std::runtime_error(
          "Distributed GPU plugin missing SetLocalStateRange");
    CreateWithBasisState =
        reinterpret_cast<DistributedGpuApi::CreateWithBasisStateFn>(
            GetFunction("CreateWithBasisState"));
    if (!CreateWithBasisState)
      throw std::runtime_error(
          "Distributed GPU plugin missing CreateWithBasisState");
    ApplyOneQubitMatrix =
        reinterpret_cast<DistributedGpuApi::ApplyOneQubitMatrixFn>(
            GetFunction("ApplyOneQubitMatrix"));
    if (!ApplyOneQubitMatrix)
      throw std::runtime_error(
          "Distributed GPU plugin missing ApplyOneQubitMatrix");
    ApplyOneQubitMatrixWithLayout =
        reinterpret_cast<DistributedGpuApi::ApplyOneQubitMatrixWithLayoutFn>(
            GetFunction("ApplyOneQubitMatrixWithLayout"));
    if (!ApplyOneQubitMatrixWithLayout)
      throw std::runtime_error(
          "Distributed GPU plugin missing ApplyOneQubitMatrixWithLayout");
    ApplyTwoQubitMatrix =
        reinterpret_cast<DistributedGpuApi::ApplyTwoQubitMatrixFn>(
            GetFunction("ApplyTwoQubitMatrix"));
    if (!ApplyTwoQubitMatrix)
      throw std::runtime_error(
          "Distributed GPU plugin missing ApplyTwoQubitMatrix");
    ApplyTwoQubitMatrixWithLayout =
        reinterpret_cast<DistributedGpuApi::ApplyTwoQubitMatrixWithLayoutFn>(
            GetFunction("ApplyTwoQubitMatrixWithLayout"));
    if (!ApplyTwoQubitMatrixWithLayout)
      throw std::runtime_error(
          "Distributed GPU plugin missing ApplyTwoQubitMatrixWithLayout");
    if (GetApiVersion() != 1)
      throw std::runtime_error("Unsupported distributed GPU API version");
    loaded = true;
    return true;
  }
  virtual void* CreateNative(int device, int backend) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    RequireLoaded();
    if (!context) {
      Check(ValidateLicense(std::getenv("MAESTRO_LICENSE_KEY")),
            "ValidateLicense");
      context = InitLib();
      if (!context) Fail("InitLib");
    }
    Check(SetGpuDevice(device), "SetGpuDevice");
    auto obj = CreateStateVectorWithBackend(context, backend);
    if (!obj) Fail("CreateStateVectorWithBackend");
    ++liveStates;
    return obj;
  }
  void DestroyNative(void* obj) noexcept {
    if (obj) {
      DestroyStateVector(obj);
      --liveStates;
    }
  }
  void* CloneNative(void* obj) {
    auto copy = Clone(obj);
    if (!copy) Fail("Clone");
    ++liveStates;
    return copy;
  }
  void RequireLoaded() {
    if (!Load())
      throw std::runtime_error(
          "Unable to load distributed GPU plugin; set MAESTRO_DIST_GPU_LIBRARY "
          "or MAESTRO_DIST_MPI_GPU_LIBRARY");
  }
  [[noreturn]] void Fail(const char* operation) const {
    const char* error = GetLastError ? GetLastError() : nullptr;
    throw std::runtime_error(std::string("Distributed GPU ") + operation +
                             ": " +
                             (error && *error ? error : "operation failed"));
  }
  void Check(int status, const char* operation) const {
    if (status != 1) Fail(operation);
  }
  DistributedGpuApi::SetGpuDeviceFn SetGpuDevice = nullptr;
  DistributedGpuApi::GetGpuDeviceCountFn GetGpuDeviceCount = nullptr;
  DistributedGpuApi::ValidateLicenseFn ValidateLicense = nullptr;
  DistributedGpuApi::CheckLicenseFn CheckLicense = nullptr;
  DistributedGpuApi::InitLibFn InitLib = nullptr;
  DistributedGpuApi::FreeLibFn FreeLib = nullptr;
  DistributedGpuApi::CreateStateVectorFn CreateStateVector = nullptr;
  DistributedGpuApi::CreateStateVectorWithBackendFn
      CreateStateVectorWithBackend = nullptr;
  DistributedGpuApi::GetBackendFn GetBackend = nullptr;
  DistributedGpuApi::SetExExecutionConfigFn SetExExecutionConfig = nullptr;
  DistributedGpuApi::DestroyStateVectorFn DestroyStateVector = nullptr;
  DistributedGpuApi::SetDataTypeFn SetDataType = nullptr;
  DistributedGpuApi::SetSeedFn SetSeed = nullptr;
  DistributedGpuApi::IsDoublePrecisionFn IsDoublePrecision = nullptr;
  DistributedGpuApi::GetNrQubitsFn GetNrQubits = nullptr;
  DistributedGpuApi::GetStateVectorGpuIdFn GetStateVectorGpuId = nullptr;
  DistributedGpuApi::CreateFn Create = nullptr;
  DistributedGpuApi::CreateWithStateFn CreateWithState = nullptr;
  DistributedGpuApi::ResetFn Reset = nullptr;
  DistributedGpuApi::MeasureQubitCollapseFn MeasureQubitCollapse = nullptr;
  DistributedGpuApi::MeasureQubitNoCollapseFn MeasureQubitNoCollapse = nullptr;
  DistributedGpuApi::MeasureQubitsCollapseFn MeasureQubitsCollapse = nullptr;
  DistributedGpuApi::MeasureQubitsNoCollapseFn MeasureQubitsNoCollapse =
      nullptr;
  DistributedGpuApi::MeasureAllQubitsCollapseFn MeasureAllQubitsCollapse =
      nullptr;
  DistributedGpuApi::MeasureAllQubitsNoCollapseFn MeasureAllQubitsNoCollapse =
      nullptr;
  DistributedGpuApi::SaveStateFn SaveState = nullptr;
  DistributedGpuApi::SaveStateToHostFn SaveStateToHost = nullptr;
  DistributedGpuApi::SaveStateDestructiveFn SaveStateDestructive = nullptr;
  DistributedGpuApi::RestoreStateFreeSavedFn RestoreStateFreeSaved = nullptr;
  DistributedGpuApi::RestoreStateNoFreeSavedFn RestoreStateNoFreeSaved =
      nullptr;
  DistributedGpuApi::FreeSavedStateFn FreeSavedState = nullptr;
  DistributedGpuApi::CloneFn Clone = nullptr;
  DistributedGpuApi::SampleFn Sample = nullptr;
  DistributedGpuApi::SampleAllFn SampleAll = nullptr;
  DistributedGpuApi::AmplitudeFn Amplitude = nullptr;
  DistributedGpuApi::ProbabilityFn Probability = nullptr;
  DistributedGpuApi::BasisStateProbabilityFn BasisStateProbability = nullptr;
  DistributedGpuApi::AllProbabilitiesFn AllProbabilities = nullptr;
  DistributedGpuApi::ExpectationValueFn ExpectationValue = nullptr;
  DistributedGpuApi::ApplyXFn ApplyX = nullptr;
  DistributedGpuApi::ApplyYFn ApplyY = nullptr;
  DistributedGpuApi::ApplyZFn ApplyZ = nullptr;
  DistributedGpuApi::ApplyHFn ApplyH = nullptr;
  DistributedGpuApi::ApplySFn ApplyS = nullptr;
  DistributedGpuApi::ApplySDGFn ApplySDG = nullptr;
  DistributedGpuApi::ApplyTFn ApplyT = nullptr;
  DistributedGpuApi::ApplyTDGFn ApplyTDG = nullptr;
  DistributedGpuApi::ApplySXFn ApplySX = nullptr;
  DistributedGpuApi::ApplySXDGFn ApplySXDG = nullptr;
  DistributedGpuApi::ApplyKFn ApplyK = nullptr;
  DistributedGpuApi::ApplyPFn ApplyP = nullptr;
  DistributedGpuApi::ApplyRxFn ApplyRx = nullptr;
  DistributedGpuApi::ApplyRyFn ApplyRy = nullptr;
  DistributedGpuApi::ApplyRzFn ApplyRz = nullptr;
  DistributedGpuApi::ApplyUFn ApplyU = nullptr;
  DistributedGpuApi::ApplyCXFn ApplyCX = nullptr;
  DistributedGpuApi::ApplyCYFn ApplyCY = nullptr;
  DistributedGpuApi::ApplyCZFn ApplyCZ = nullptr;
  DistributedGpuApi::ApplyCHFn ApplyCH = nullptr;
  DistributedGpuApi::ApplyCSXFn ApplyCSX = nullptr;
  DistributedGpuApi::ApplyCSXDGFn ApplyCSXDG = nullptr;
  DistributedGpuApi::ApplyCPFn ApplyCP = nullptr;
  DistributedGpuApi::ApplyCRxFn ApplyCRx = nullptr;
  DistributedGpuApi::ApplyCRyFn ApplyCRy = nullptr;
  DistributedGpuApi::ApplyCRzFn ApplyCRz = nullptr;
  DistributedGpuApi::ApplyCCXFn ApplyCCX = nullptr;
  DistributedGpuApi::ApplySwapFn ApplySwap = nullptr;
  DistributedGpuApi::ApplyCSwapFn ApplyCSwap = nullptr;
  DistributedGpuApi::ApplyCUFn ApplyCU = nullptr;
  DistributedGpuApi::GetApiVersionFn GetApiVersion = nullptr;
  DistributedGpuApi::GetCapabilitiesFn GetCapabilities = nullptr;
  DistributedGpuApi::GetLastErrorFn GetLastError = nullptr;
  DistributedGpuApi::ConfigureDistributionFn ConfigureDistribution = nullptr;
  DistributedGpuApi::GetShardDevicesFn GetShardDevices = nullptr;
  DistributedGpuApi::GetGlobalQubitsFn GetGlobalQubits = nullptr;
  DistributedGpuApi::GetQubitLayoutFn GetQubitLayout = nullptr;
  DistributedGpuApi::RedistributeFn Redistribute = nullptr;
  DistributedGpuApi::SwapGlobalLocalQubitsFn SwapGlobalLocalQubits = nullptr;
  DistributedGpuApi::SynchronizeFn Synchronize = nullptr;
  DistributedGpuApi::GetStateRangeFn GetStateRange = nullptr;
  DistributedGpuApi::SetStateRangeFn SetStateRange = nullptr;
  DistributedGpuApi::GetLocalStateBoundsFn GetLocalStateBounds = nullptr;
  DistributedGpuApi::GetLocalStateRangeFn GetLocalStateRange = nullptr;
  DistributedGpuApi::SetLocalStateRangeFn SetLocalStateRange = nullptr;
  DistributedGpuApi::CreateWithBasisStateFn CreateWithBasisState = nullptr;
  DistributedGpuApi::ApplyOneQubitMatrixFn ApplyOneQubitMatrix = nullptr;
  DistributedGpuApi::ApplyOneQubitMatrixWithLayoutFn
      ApplyOneQubitMatrixWithLayout = nullptr;
  DistributedGpuApi::ApplyTwoQubitMatrixFn ApplyTwoQubitMatrix = nullptr;
  DistributedGpuApi::ApplyTwoQubitMatrixWithLayoutFn
      ApplyTwoQubitMatrixWithLayout = nullptr;

 protected:
  explicit DistributedGpuLibrary(bool mpi = false) : mpi(mpi) {}
  std::recursive_mutex mutex;
  std::atomic_size_t liveStates{0};
  void* context = nullptr;
  bool mpi;
  bool loaded = false;
};
}  // namespace Simulators
#endif
