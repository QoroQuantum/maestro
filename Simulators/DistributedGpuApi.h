// C ABI version 1 of maestro-gpu-distributed. No CUDA headers or link
// dependency. Keep these declarations synchronized with that project's
// include/gpusim.h. This is part of the C-language plugin boundary covered by
// the linking exception in LICENSE, like GpuLibrary.h.
#pragma once
#include <cstdint>
namespace Simulators::DistributedGpuApi {
// MPI-independent declarations from include/gpusim_mpi_runtime.h.
struct MgdMpiCommunicator {
  uint32_t struct_size;
  uint32_t reserved;
  int64_t fortran_handle;
};
struct MgdMpiRuntimeInfo {
  uint32_t struct_size;
  int32_t size;
  int32_t rank;
  int32_t default_device;
};
struct MgdExExecutionConfig {
  uint32_t struct_size;
  uint32_t max_queued_gates;
  uint64_t transfer_workspace_bytes;
};
struct MgdDistributionConfig {
  uint32_t struct_size;
  uint32_t flags;
  uint32_t num_global_qubits;
  const int32_t *global_qubits;
  uint32_t num_shards;
  const int32_t *shard_devices;
};
using SetGpuDeviceFn = int (*)(int deviceId);
using GetGpuDeviceCountFn = int (*)(void);
using ValidateLicenseFn = int (*)(const char *licenseKey);
using CheckLicenseFn = int (*)(void *state);
using InitLibFn = void *(*)(void);
using FreeLibFn = void (*)(void);
using CreateStateVectorFn = void *(*)(void *lib);
using CreateStateVectorWithBackendFn = void *(*)(void *lib, int backend);
using GetBackendFn = int (*)(void *obj);
using SetExExecutionConfigFn = int (*)(void *obj,
                                       const MgdExExecutionConfig *config);
using DestroyStateVectorFn = void (*)(void *obj);
using SetDataTypeFn = int (*)(void *obj, int useDoublePrecision);
using SetSeedFn = int (*)(void *obj, unsigned long long seed);
using IsDoublePrecisionFn = int (*)(void *obj);
using GetNrQubitsFn = int (*)(void *obj);
using GetStateVectorGpuIdFn = int (*)(void *obj);
using CreateFn = int (*)(void *obj, unsigned int nrQubits);
using CreateWithStateFn = int (*)(void *obj, unsigned int nrQubits,
                                  const double *state);
using ResetFn = int (*)(void *obj);
using MeasureQubitCollapseFn = int (*)(void *obj, int qubitIndex);
using MeasureQubitNoCollapseFn = int (*)(void *obj, int qubitIndex);
using MeasureQubitsCollapseFn = int (*)(void *obj, int *qubits, int *bitstring,
                                        int bitstringLen);
using MeasureQubitsNoCollapseFn = int (*)(void *obj, int *qubits,
                                          int *bitstring, int bitstringLen);
using MeasureAllQubitsCollapseFn = unsigned long long (*)(void *obj);
using MeasureAllQubitsNoCollapseFn = unsigned long long (*)(void *obj);
using SaveStateFn = int (*)(void *obj);
using SaveStateToHostFn = int (*)(void *obj);
using SaveStateDestructiveFn = int (*)(void *obj);
using RestoreStateFreeSavedFn = int (*)(void *obj);
using RestoreStateNoFreeSavedFn = int (*)(void *obj);
using FreeSavedStateFn = void (*)(void *obj);
using CloneFn = void *(*)(void *obj);
using SampleFn = int (*)(void *obj, unsigned int nSamples, long int *samples,
                         unsigned int nBits, int *bitOrdering);
using SampleAllFn = int (*)(void *obj, unsigned int nSamples,
                            long int *samples);
using AmplitudeFn = int (*)(void *obj, long long int state, double *real,
                            double *imaginary);
using ProbabilityFn = double (*)(void *obj, int *qubits, int *mask, int len);
using BasisStateProbabilityFn = double (*)(void *obj, long long int state);
using AllProbabilitiesFn = int (*)(void *obj, double *probabilities);
using ExpectationValueFn = double (*)(void *obj, const char *pauliString,
                                      int len);
using ApplyXFn = int (*)(void *obj, int qubit);
using ApplyYFn = int (*)(void *obj, int qubit);
using ApplyZFn = int (*)(void *obj, int qubit);
using ApplyHFn = int (*)(void *obj, int qubit);
using ApplySFn = int (*)(void *obj, int qubit);
using ApplySDGFn = int (*)(void *obj, int qubit);
using ApplyTFn = int (*)(void *obj, int qubit);
using ApplyTDGFn = int (*)(void *obj, int qubit);
using ApplySXFn = int (*)(void *obj, int qubit);
using ApplySXDGFn = int (*)(void *obj, int qubit);
using ApplyKFn = int (*)(void *obj, int qubit);
using ApplyPFn = int (*)(void *obj, int qubit, double theta);
using ApplyRxFn = int (*)(void *obj, int qubit, double theta);
using ApplyRyFn = int (*)(void *obj, int qubit, double theta);
using ApplyRzFn = int (*)(void *obj, int qubit, double theta);
using ApplyUFn = int (*)(void *obj, int qubit, double theta, double phi,
                         double lambda, double gamma);
using ApplyCXFn = int (*)(void *obj, int controlQubit, int targetQubit);
using ApplyCYFn = int (*)(void *obj, int controlQubit, int targetQubit);
using ApplyCZFn = int (*)(void *obj, int controlQubit, int targetQubit);
using ApplyCHFn = int (*)(void *obj, int controlQubit, int targetQubit);
using ApplyCSXFn = int (*)(void *obj, int controlQubit, int targetQubit);
using ApplyCSXDGFn = int (*)(void *obj, int controlQubit, int targetQubit);
using ApplyCPFn = int (*)(void *obj, int controlQubit, int targetQubit,
                          double theta);
using ApplyCRxFn = int (*)(void *obj, int controlQubit, int targetQubit,
                           double theta);
using ApplyCRyFn = int (*)(void *obj, int controlQubit, int targetQubit,
                           double theta);
using ApplyCRzFn = int (*)(void *obj, int controlQubit, int targetQubit,
                           double theta);
using ApplyCCXFn = int (*)(void *obj, int controlQubit1, int controlQubit2,
                           int targetQubit);
using ApplySwapFn = int (*)(void *obj, int qubit1, int qubit2);
using ApplyCSwapFn = int (*)(void *obj, int controlQubit, int qubit1,
                             int qubit2);
using ApplyCUFn = int (*)(void *obj, int controlQubit, int targetQubit,
                          double theta, double phi, double lambda,
                          double gamma);
using GetApiVersionFn = uint32_t (*)(void);
using GetCapabilitiesFn = uint64_t (*)(void);
using GetLastErrorFn = const char *(*)(void);
using ConfigureDistributionFn = int (*)(void *obj,
                                        const MgdDistributionConfig *config);
using GetShardDevicesFn = int (*)(void *obj, int32_t *devices,
                                  uint32_t capacity);
using GetGlobalQubitsFn = int (*)(void *obj, int32_t *qubits,
                                  uint32_t capacity);
using GetQubitLayoutFn = int (*)(void *obj, int32_t *wires, uint32_t capacity);
using RedistributeFn = int (*)(void *obj, const int32_t *global_qubits,
                               uint32_t count);
using SwapGlobalLocalQubitsFn = int (*)(void *obj, const int32_t *global_qubits,
                                        const int32_t *local_qubits,
                                        uint32_t count);
using SynchronizeFn = int (*)(void *obj);
using GetStateRangeFn = int (*)(void *obj, uint64_t begin, uint64_t end,
                                double *output);
using SetStateRangeFn = int (*)(void *obj, uint64_t begin, uint64_t end,
                                const double *input);
using GetLocalStateBoundsFn = int (*)(void *obj, uint64_t *begin,
                                      uint64_t *end);
using GetLocalStateRangeFn = int (*)(void *obj, uint64_t begin, uint64_t end,
                                     double *output);
using SetLocalStateRangeFn = int (*)(void *obj, uint64_t begin, uint64_t end,
                                     const double *input);
using CreateWithBasisStateFn = int (*)(void *obj, uint32_t nrQubits,
                                       uint64_t basis);
using ApplyOneQubitMatrixFn = int (*)(void *obj, int qubit,
                                      const double *matrix);
using ApplyOneQubitMatrixWithLayoutFn = int (*)(void *obj, int qubit,
                                                const double *matrix,
                                                int layout);
using ApplyTwoQubitMatrixFn = int (*)(void *obj, int qubit0, int qubit1,
                                      const double *matrix);
using ApplyTwoQubitMatrixWithLayoutFn = int (*)(void *obj, int qubit0,
                                                int qubit1,
                                                const double *matrix,
                                                int layout);
}  // namespace Simulators::DistributedGpuApi
