// Minimal |00...0> MPS with counters and injected failures. Build once with the
// new exports and once without them to exercise dynamic capability discovery.
#include "MockPlugin.cpp"
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace
{
struct Tensor : State
{
    void *context = nullptr;
    void (*summary)(void *, int64_t) = nullptr;
    void (*full)(void *, const int64_t *) = nullptr;
};

int calls[10]{};

bool Fails(const char *name)
{
    const auto *failure = std::getenv("MAESTRO_TEST_TENSOR_FAILURE");
    return failure && std::strcmp(failure, name) == 0;
}

double Pauli(const char *p, int n)
{
    for (int i = 0; i < n; ++i)
        if (p[i] == 'X' || p[i] == 'Y')
            return 0.;
    return 1.;
}

using Histogram = std::unordered_map<std::vector<bool>, int64_t>;
} // namespace

extern "C"
{
    int MockTensorCalls(int index)
    {
        return calls[index];
    }

    void *CreateMPS(void *)
    {
        ++live;
        auto *t = new Tensor;
        t->device = selected;
        t->qubits = 0;
        t->bits = 0;
        return t;
    }

    void DestroyMPS(void *obj)
    {
        delete static_cast<Tensor *>(obj);
        --live;
    }

    int MPSCreate(void *obj, unsigned n)
    {
        auto *t = static_cast<Tensor *>(obj);
        t->qubits = n;
        if (t->summary)
            t->summary(t->context, 3);
        if (t->full)
        {
            std::vector<int64_t> dims(n - 1, 3);
            t->full(t->context, dims.data());
        }
        return 1;
    }

    int MPSGetNrQubits(void *obj)
    {
        return static_cast<Tensor *>(obj)->qubits;
    }

    int MPSIsCreated(void *obj)
    {
        return MPSGetNrQubits(obj) != 0;
    }

    int MPSSetMaxExtent(void *, long)
    {
        return 1;
    }

    int MPSSetDataType(void *, int)
    {
        return 1;
    }

    int MPSIsDoublePrecision(void *)
    {
        return 1;
    }

    int MPSSetUseOptimalMeetingPosition(void *, int)
    {
        return 1;
    }

    int MPSSetCallbackContext(void *obj, void *context)
    {
        static_cast<Tensor *>(obj)->context = context;
        return 1;
    }

    int MPSSetBondDimensionsCallback(void *obj, void (*callback)(void *, const int64_t *))
    {
        ++calls[5];
        static_cast<Tensor *>(obj)->full = callback;
        return 1;
    }

    int MPSGetBondDimensions(void *obj, long long *dims)
    {
        std::fill_n(dims, MPSGetNrQubits(obj) - 1, 1);
        return 1;
    }

    double MPSExpectationValue(void *, const char *pauli, int length)
    {
        ++calls[0];
        return Pauli(pauli, length);
    }

    int MPSAmplitude(void *, long n, long *bits, double *re, double *im)
    {
        ++calls[7];
        *re = 1.;
        *im = 0.;
        for (long q = 0; q < n; ++q)
            if (bits[q])
                *re = 0.;
        return !Fails("amplitude");
    }

    void *MPSGetMapForSample()
    {
        ++calls[6];
        return new Histogram;
    }

    int MPSFreeMapForSample(void *obj)
    {
        --calls[6];
        delete static_cast<Histogram *>(obj);
        return 1;
    }

    int MPSSample(void *, long shots, long width, unsigned *, void *result)
    {
        if (Fails("sample"))
            return 0;
        auto &map = *static_cast<Histogram *>(result);
        map[std::vector<bool>(width)] = Fails("partial") ? shots - 1 : shots;
        return 1;
    }
#ifndef MAESTRO_OLD_TENSOR_PLUGIN
    int MPSExpectationValues(void *, int count, const char *const *paulis, const int *lengths, double *values)
    {
        ++calls[1];
        if (Fails("batch"))
            return 0;
        for (int i = 0; i < count; ++i)
            values[i] = Pauli(paulis[i], lengths[i]);
        return 1;
    }

    int MPSGetStateVector(void *obj, double *raw)
    {
        ++calls[2];
        if (Fails("dense"))
            return 0;
        std::fill_n(raw, 2 << MPSGetNrQubits(obj), 0.);
        raw[0] = 1.;
        return 1;
    }

    int MPSAllProbabilities(void *obj, double *raw)
    {
        ++calls[3];
        if (Fails("probabilities"))
            return 0;
        std::fill_n(raw, 1 << MPSGetNrQubits(obj), 0.);
        raw[0] = 1.;
        return 1;
    }

    int MPSSetBondDimensionSummaryCallback(void *obj, void (*callback)(void *, int64_t))
    {
        ++calls[4];
        static_cast<Tensor *>(obj)->summary = callback;
        return 1;
    }

    long long MPSGetMaxBondDimension(void *)
    {
        return 1;
    }

    int MPSMoveAtBeginningOfChain(void *, const int *, int)
    {
        ++calls[8];
        return !Fails("move");
    }

    int MPSExpectationValueOperators(void *, int, const int *, const double *, double *re, double *im)
    {
        ++calls[9];
        if (Fails("operators"))
            return 0;
        *re = 1.;
        *im = 0.;
        return 1;
    }
#endif
}
