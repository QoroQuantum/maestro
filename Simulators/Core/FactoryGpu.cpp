/**
 * @file FactoryGpu.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory part for the gpu and distributed gpu backends, including
 * the management of the gpu libraries.
 *
 * The gpu backends are available only on linux, so on other platforms this
 * translation unit is empty.
 */

#ifdef __linux__

#include "../Gpu/GpuLibraryRegistry.h"
#include "FactoryBackends.h"

#define INCLUDED_BY_FACTORY
#include "../DistributedGpu/DistributedGpuSimulator.h"
#include "../Gpu/GpuDensityMatrixSimulator.h"
#include "../Gpu/GpuMPOSimulator.h"
#include "../Gpu/GpuMPSSimulator.h"
#include "../Gpu/GpuPauliPropagatorSimulator.h"
#include "../Gpu/GpuSimulator.h"
#include "../Gpu/GpuStatevectorSimulator.h"
#include "../Gpu/GpuTensorNetworkSimulator.h"

namespace Simulators
{

std::atomic_int SimulatorsFactory::requestedGpuDeviceId{0};
thread_local int SimulatorsFactory::scopedGpuDeviceId = -1;

namespace
{
GpuLibraryRegistry &GpuLibraries()
{
    static GpuLibraryRegistry registry;
    return registry;
}
} // namespace

std::shared_ptr<DistributedGpuLibrary> SimulatorsFactory::GetDistributedGpuLibrary()
{
    return DistributedGpuLibrary::GetInstance();
}

std::shared_ptr<DistributedGpuLibrary> DistributedGpuLibrary::GetInstance()
{
    static auto lib = std::shared_ptr<DistributedGpuLibrary>(new DistributedGpuLibrary());
    return lib;
}

std::shared_ptr<DistributedMpiGpuLibrary> DistributedMpiGpuLibrary::GetInstance()
{
    static auto lib = std::shared_ptr<DistributedMpiGpuLibrary>(new DistributedMpiGpuLibrary());
    return lib;
}

bool SimulatorsFactory::IsDistributedGpuAvailable() noexcept
{
    try
    {
        auto lib = GetDistributedGpuLibrary();
        return lib->Load() && lib->GetGpuDeviceCount() > 0;
    }
    catch (...)
    {
        // Initialization still exposes the full diagnostic; discovery is a probe.
        return false;
    }
}

std::shared_ptr<DistributedMpiGpuLibrary> SimulatorsFactory::GetDistributedMpiGpuLibrary()
{
    return DistributedMpiGpuLibrary::GetInstance();
}

void SimulatorsFactory::FinalizeDistributedMpiGpuBackend()
{
    DistributedMpiGpuLibrary::GetInstance()->FinalizeBackend();
}

void SimulatorsFactory::SelectGpuDevice(int deviceId)
{
    if (deviceId < 0)
        throw std::invalid_argument("gpu_device must be nonnegative");
    requestedGpuDeviceId = deviceId;
}

int SimulatorsFactory::ResolveGpuDevice(int deviceId)
{
    if (deviceId >= 0)
        return deviceId;
    if (scopedGpuDeviceId >= 0)
        return scopedGpuDeviceId;
    return requestedGpuDeviceId.load();
}

SimulatorsFactory::ScopedGpuDevice::ScopedGpuDevice(int deviceId) : previous(scopedGpuDeviceId)
{
    scopedGpuDeviceId = ResolveGpuDevice(deviceId);
}

SimulatorsFactory::ScopedGpuDevice::~ScopedGpuDevice()
{
    scopedGpuDeviceId = previous;
}

std::shared_ptr<GpuLibrary> SimulatorsFactory::GetGpuLibrary(int deviceId)
{
    return GpuLibraries().Acquire(ResolveGpuDevice(deviceId));
}

bool SimulatorsFactory::InitGpuLibrary()
{
    return bool(GetGpuLibrary());
}

bool SimulatorsFactory::InitGpuLibraryWithMute()
{
    return bool(GpuLibraries().Acquire(ResolveGpuDevice(), true));
}

int SimulatorsFactory::GetGpuDeviceCount()
{
    return GpuLibraries().DeviceCount();
}

bool SimulatorsFactory::IsGpuLibraryAvailable(int deviceId)
{
    return bool(GpuLibraries().Acquire(ResolveGpuDevice(deviceId), true));
}

namespace FactoryBackends
{

std::unique_ptr<ISimulator> CreateGpuImmediate(SimulationType method)
{
    std::unique_ptr<ISimulator> sim;
    switch (method)
    {
    case SimulationType::kStatevector:
        sim = std::make_unique<Private::GpuStatevectorSimulator>();
        break;
    case SimulationType::kMatrixProductState:
        sim = std::make_unique<Private::GpuMPSSimulator>();
        break;
    case SimulationType::kMatrixProductOperator:
        sim = std::make_unique<Private::GpuMPOSimulator>();
        break;
    case SimulationType::kTensorNetwork:
        sim = std::make_unique<Private::GpuTensorNetworkSimulator>();
        break;
    case SimulationType::kPauliPropagator:
        sim = std::make_unique<Private::GpuPauliPropagatorSimulator>();
        break;
    case SimulationType::kDensityMatrix:
        sim = std::make_unique<Private::GpuDensityMatrixSimulator>();
        break;
    default:
        return nullptr;
    }
    sim->Configure("gpu_device", std::to_string(SimulatorsFactory::ResolveGpuDevice()).c_str());
    return sim;
}

std::unique_ptr<ISimulator> CreateGpuSimulator(std::unique_ptr<ISimulator> immediate)
{
    return std::make_unique<Private::GpuSimulator>(std::move(immediate));
}

std::unique_ptr<ISimulator> CreateDistributedGpuSimulator()
{
    return std::make_unique<Private::DistributedGpuSimulator>();
}

std::unique_ptr<ISimulator> CreateDistributedMpiGpuSimulator()
{
    return std::make_unique<Private::DistributedMpiGpuSimulator>();
}

} // namespace FactoryBackends
} // namespace Simulators

#endif // __linux__
