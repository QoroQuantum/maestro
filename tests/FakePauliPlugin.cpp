// Minimal plugin for adapter compatibility tests. No CUDA dependency or device allocations.
#include <cmath>
#include <cstddef>
#include <limits>

namespace
{
struct State
{
    int qubits;
    bool sampling = false, allocated = false;
    unsigned long long gates = 0, saved = 0;
};

int failAfter = -1;
State *lastCreated = nullptr;
#ifdef PAULI_CHECKPOINT_PLUGIN
// Arguments of the most recent tuning and batch calls; -1 until called.
long long lastMemoryBytes = -1, lastMaxFusedGates = -1, lastCacheNodes = -1, lastBatchWorkers = -1, lastBatchCount = -1;
#endif

int Add(void *p)
{
    if (failAfter == 0)
    {
        failAfter = -1;
        return 0;
    }
    if (failAfter > 0)
        --failAfter;
    ++static_cast<State *>(p)->gates;
    return 1;
}
} // namespace

extern "C"
{
    int ValidateLicense(const char *)
    {
        return 1;
    }

    const char *GetLicenseError()
    {
        return "";
    }

    int GetGpuDeviceCount()
    {
        return 1;
    }

    int SetGpuDevice(int device)
    {
        return device == 0;
    }

    void *InitLib()
    {
        static int initialized;
        return &initialized;
    }

    void FreeLib()
    {
    }

    void *CreatePauliPropSimulator(int qubits)
    {
        lastCreated = new State{qubits};
        return lastCreated;
    }

    void DestroyPauliPropSimulator(void *p)
    {
        delete static_cast<State *>(p);
    }

    int PauliPropGetNrQubits(void *p)
    {
        return static_cast<State *>(p)->qubits;
    }

    int PauliPropGetGpuId(void *)
    {
        return 0;
    }

    int PauliPropSetSeed(void *, unsigned long long)
    {
        return 1;
    }

    // Network configuration sets the default deduplication cadence.
    void PauliPropSetNumGatesBetweenDeduplications(void *, int)
    {
    }

    int PauliPropSetWillUseSampling(void *p, int enabled)
    {
        static_cast<State *>(p)->sampling = enabled;
        return 1;
    }

    int PauliPropGetWillUseSampling(void *p)
    {
        return static_cast<State *>(p)->sampling;
    }

    int PauliPropAllocateMemory(void *p, double)
    {
        auto &state = *static_cast<State *>(p);
        // Models the older plugin's need for the hint BEFORE allocating its expansion/workspace buffers.
        state.allocated = state.sampling;
        return state.allocated;
    }

    int PauliPropClearOperators(void *p)
    {
        static_cast<State *>(p)->gates = 0;
        return 1;
    }

    int PauliPropApplyRZ(void *p, int, double)
    {
        return Add(p);
    }

    int PauliPropApplyRY(void *p, int, double)
    {
        return Add(p);
    }

    int PauliPropApplyH(void *p, int)
    {
        return Add(p);
    }

    int PauliPropApplyX(void *p, int)
    {
        return Add(p);
    }

    int PauliPropApplyZ(void *p, int)
    {
        return Add(p);
    }

    int PauliPropApplyS(void *p, int)
    {
        return Add(p);
    }

    int PauliPropApplyCX(void *p, int, int)
    {
        return Add(p);
    }

    void PauliPropSaveState(void *p)
    {
        auto &s = *static_cast<State *>(p);
        s.saved = s.gates;
    }

    void PauliPropRestoreState(void *p)
    {
        auto &s = *static_cast<State *>(p);
        s.gates = s.saved;
    }

    unsigned char *PauliPropSampleQubits(void *p, const int *, int count)
    {
        if (!static_cast<State *>(p)->allocated)
            return nullptr;
        return new unsigned char[(count + 7) / 8]{};
    }

    void PauliPropFreeSampledQubits(unsigned char *p)
    {
        delete[] p;
    }

    unsigned long long PauliTestLastOperatorCount()
    {
        return lastCreated->gates;
    }

    void PauliTestFailAfter(int count)
    {
        failAfter = count;
    }

    unsigned long long PauliTestOperatorCount(void *p)
    {
        return static_cast<State *>(p)->gates;
    }
#ifdef PAULI_CHECKPOINT_PLUGIN
    unsigned long long PauliPropGetOperatorCount(void *p)
    {
        return PauliTestOperatorCount(p);
    }

    int PauliPropTruncateOperators(void *p, unsigned long long count)
    {
        auto &s = *static_cast<State *>(p);
        if (count > s.gates)
            return 0;
        s.gates = count;
        return 1;
    }

    // The tuning and batch symbols of newer plugins, recorded for the tests.
    int PauliPropAllocateMemoryBytes(void *p, unsigned long long bytes)
    {
        lastMemoryBytes = static_cast<long long>(bytes);
        return PauliPropAllocateMemory(p, 0);
    }

    void PauliPropSetMaxFusedGates(void *, int count)
    {
        lastMaxFusedGates = count;
    }

    void PauliPropSetSamplingCacheMaxNodes(void *, unsigned long long count)
    {
        lastCacheNodes = static_cast<long long>(count);
    }

    int PauliPropExpectationValues(void *, int count, const char *const *, double *values, int workers)
    {
        lastBatchCount = count;
        lastBatchWorkers = workers;
        for (int i = 0; i < count; ++i)
            values[i] = 1.0;
        return 1;
    }

    // Last memory budget, fused-gate limit, cache size, batch worker count and batch size.
    void PauliTestLastTuning(long long *out)
    {
        out[0] = lastMemoryBytes;
        out[1] = lastMaxFusedGates;
        out[2] = lastCacheNodes;
        out[3] = lastBatchWorkers;
        out[4] = lastBatchCount;
    }
#endif
}
