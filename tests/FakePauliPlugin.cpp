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
#endif
}
