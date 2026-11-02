#include "../Simulators/Core/Factory.h"
#include "LoadedMaestro.h"
#define INCLUDED_BY_FACTORY
#include "../Simulators/Gpu/GpuPauliPropagatorSimulator.h"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace Simulators;

void Check(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class F> void Reject(F &&f)
{
    bool rejected = false;
    try { f(); }
    catch (const std::exception &) { rejected = true; }
    Check(rejected, "uninitialized Pauli wrapper accepted a result");
}

void UninitializedResults(const std::shared_ptr<GpuLibrary> &library)
{
    for (auto loaded : {std::shared_ptr<GpuLibrary>{}, library})
    {
        GpuPauliPropagator uninitialized(loaded);
        Reject([&] { uninitialized.QubitProbability0(0); });
        Reject([&] { uninitialized.Probability(0); });
        Reject([&] { uninitialized.MeasureQubit(0); });
    }
    Reject([&] { library->PauliPropQubitProbability0(nullptr, 0); });
    Reject([&] { library->PauliPropProbability(nullptr, 0); });
    Reject([&] { library->PauliPropMeasureQubit(nullptr, 0); });
}

int main(int argc, char **)
{
    try
    {
        CheckLoadedMaestro();
        Check(SimulatorsFactory::InitGpuLibraryWithMute(), "load fake plugin");
        auto library = SimulatorsFactory::GetGpuLibrary();
        UninitializedResults(library);
        Private::GpuPauliPropagatorSimulator state;
        state.AllocateQubits(2);
        state.Initialize(); // Fails on the old API if the sampling hint follows allocation.
        Check(state.SampleCounts({0, 1}, 3).at(0) == 3, "older plugin samples after initialization");
        GpuPauliPropagator pp(library);
        Check(pp.CreateSimulator(2), "create composite recording fixture");
        auto failAfter = reinterpret_cast<void (*)(int)>(library->GetFunction("PauliTestFailAfter"));
        auto count = reinterpret_cast<unsigned long long (*)()>(library->GetFunction("PauliTestLastOperatorCount"));
        // Invalid arguments are rejected before recording, including with older plugins.
        Check(!pp.ApplyU(0, .3, std::numeric_limits<double>::quiet_NaN(), .2), "reject nonfinite composite");
        Check(!pp.ApplyCH(0, 0) && !pp.ApplyCRX(0, 2, .3), "reject overlapping/out-of-range composite qubits");
        Check(count() == 0, "invalid composites leave no partial recording");
        if (argc > 1)
        {
            Check(library->HasPauliPropOperatorCheckpoints(), "checkpoint plugin extensions are visible");
            Check(pp.ApplyX(0), "record initial gate");
            pp.SaveState();
            for (int position : {0, 1, 2})
            {
                failAfter(position);
                Check(!pp.ApplyU(0, .3, .4, .5), "injected subgate failure");
                Check(count() == 1, "failed composite rolled back all recorded subgates");
                // A successful gate followed by RestoreState must still restore the caller's checkpoint.
                Check(pp.ApplyH(1), "recover after composite failure");
                pp.RestoreState();
                Check(count() == 1, "rollback preserves caller saved state");
            }
            failAfter(4);
            Check(!pp.ApplyCH(0, 1) && count() == 1, "nested composite rollback");
        }
        else
        {
            Check(!library->HasPauliPropOperatorCheckpoints(), "older plugin has no new symbols");
        }
        std::cout << "Pauli plugin compatibility checks passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
