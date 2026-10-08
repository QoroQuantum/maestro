/**
 * @file Factory.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory implementation
 *
 * Call CreateSimulator with the desired simulator type to create a simulator
 * returned as a shared pointer. Currently only two simulators are supported:
 * qiskit aer and qcsim. Can be esily extended to support more simulators. Just
 * implement the interface for another simulator, add its type to the enum and
 * add another case to the switch statement.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#ifndef NO_QISKIT_AER
#ifndef __APPLE__
#ifndef _QV_AVX2_IMPL
#define _QV_AVX2_IMPL
#pragma warning(push)
#pragma warning(disable : 4789)
#include "simulators/statevector/qv_avx2.cpp"
#pragma warning(pop)
#endif
#endif
#endif

#include "../Gpu/GpuLibraryRegistry.h"
#include "Factory.h"

#define INCLUDED_BY_FACTORY
#ifndef NO_QISKIT_AER
#include "../Aer/AerSimulator.h"
#endif
#include "../Composite/Composite.h"
#include "../DistributedGpu/DistributedGpuSimulator.h"
#include "../Gpu/GpuDensityMatrixSimulator.h"
#include "../Gpu/GpuMPOSimulator.h"
#include "../Gpu/GpuMPSSimulator.h"
#include "../Gpu/GpuPauliPropagatorSimulator.h"
#include "../Gpu/GpuSimulator.h"
#include "../Gpu/GpuStatevectorSimulator.h"
#include "../Gpu/GpuTensorNetworkSimulator.h"
#include "../QCSim/QCSimDensityMatrixSimulator.h"
#include "../QCSim/QCSimExtendedStabilizerSimulator.h"
#include "../QCSim/QCSimMPOSimulator.h"
#include "../QCSim/QCSimMPSSimulator.h"
#include "../QCSim/QCSimPathIntegralSimulator.h"
#include "../QCSim/QCSimPauliPropagatorSimulator.h"
#include "../QCSim/QCSimSimulator.h"
#include "../QCSim/QCSimStabilizerSimulator.h"
#include "../QCSim/QCSimStatevectorSimulator.h"
#include "../QCSim/QCSimTensorNetworkSimulator.h"
#include "../Quest/QuestSimulator.h"

namespace Simulators
{

std::unique_ptr<ISimulator> SimulatorsFactory::CreateImmediateSimulatorUnique(SimulatorType type, SimulationType method)
{
    switch (type)
    {
    case SimulatorType::kQCSim:
        switch (method)
        {
        case SimulationType::kStatevector:
            return std::make_unique<Private::QCSimStatevectorSimulator>();
        case SimulationType::kMatrixProductState:
            return std::make_unique<Private::QCSimMPSSimulator>();
        case SimulationType::kMatrixProductOperator:
            return std::make_unique<Private::QCSimMPOSimulator>();
        case SimulationType::kStabilizer:
            return std::make_unique<Private::QCSimStabilizerSimulator>();
        case SimulationType::kTensorNetwork:
            return std::make_unique<Private::QCSimTensorNetworkSimulator>();
        case SimulationType::kPauliPropagator:
            return std::make_unique<Private::QCSimPauliPropagatorSimulator>();
        case SimulationType::kPathIntegral:
            return std::make_unique<Private::QCSimPathIntegralSimulator>();
        case SimulationType::kDensityMatrix:
            return std::make_unique<Private::QCSimDensityMatrixSimulator>();
        case SimulationType::kExtendedStabilizer:
            return std::make_unique<Private::QCSimExtendedStabilizerSimulator>();
        default:
            throw std::invalid_argument("Simulation Type not supported for QCSim");
        }
#ifndef NO_QISKIT_AER
    case SimulatorType::kQiskitAer:
        // Aer owns its method selection and does not use the fusion adapter.
        return CreateSimulatorUnique(type, method);
#endif
#ifdef __linux__
    case SimulatorType::kGpuSim: {
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
        sim->Configure("gpu_device", std::to_string(ResolveGpuDevice()).c_str());
        return sim;
    }
#endif
    default:
        throw std::invalid_argument("Unsupported immediate backend");
    }
}

#ifdef __linux__
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

#endif

std::shared_ptr<QuestLibSim> SimulatorsFactory::questLibrary = nullptr;
std::atomic_bool SimulatorsFactory::firstTimeQuest = true;

bool SimulatorsFactory::InitQuestLibrary()
{
    if (!questLibrary)
    {
        questLibrary = std::make_shared<QuestLibSim>();
        if (!firstTimeQuest.exchange(false))
            questLibrary->SetMute(true);
        if (questLibrary->Init(
#ifdef _WIN32
                "maestroquest.dll"
#elif defined(__APPLE__)
                "libmaestroquest.dylib"
#else
                "libmaestroquest.so"
#endif
                ))
            return true;
        else
            questLibrary = nullptr;
    }
    return false;
}

bool SimulatorsFactory::IsQuestLibraryAvailable()
{
    return questLibrary && questLibrary->IsValid();
}

std::shared_ptr<QuestLibSim> SimulatorsFactory::GetQuestLibrary()
{
    if (!questLibrary || !questLibrary->IsValid())
        return nullptr;
    return questLibrary;
}

bool SimulatorsFactory::InitQuestLibraryWithMute()
{
    if (!questLibrary)
    {
        questLibrary = std::make_shared<QuestLibSim>();
        firstTimeQuest = false;
        questLibrary->SetMute(true);
        if (questLibrary->Init(
#ifdef _WIN32
                "maestroquest.dll"
#elif defined(__APPLE__)
                "libmaestroquest.dylib"
#else
                "libmaestroquest.so"
#endif
                ))
            return true;
        else
            questLibrary = nullptr;
    }
    return false;
}

std::shared_ptr<ISimulator> SimulatorsFactory::CreateSimulator(SimulatorType t, SimulationType m)
{
    return CreateSimulatorUnique(t, m);
}

std::unique_ptr<ISimulator> SimulatorsFactory::CreateSimulatorUnique(SimulatorType t, SimulationType m)
{
    switch (t)
    {
    case SimulatorType::kQCSim:
        return std::make_unique<Private::QCSimSimulator>(CreateImmediateSimulatorUnique(t, m));
#ifndef NO_QISKIT_AER
    case SimulatorType::kQiskitAer: {
        auto sim = std::make_unique<Private::AerSimulator>();
        if (m == SimulationType::kMatrixProductState)
            sim->Configure("method", "matrix_product_state");
        else if (m == SimulationType::kStabilizer)
            sim->Configure("method", "stabilizer");
        else if (m == SimulationType::kTensorNetwork)
            sim->Configure("method", "tensor_network");
        else if (m == SimulationType::kExtendedStabilizer)
            sim->Configure("method", "extended_stabilizer");
        else if (m == SimulationType::kDensityMatrix)
            sim->Configure("method", "density_matrix");
        else if (m == SimulationType::kStatevector)
            sim->Configure("method", "statevector");
        else
            throw std::invalid_argument("Simulation Type not supported for Qiskit Aer");

        return sim;
    }
    case SimulatorType::kCompositeQiskitAer:
        return std::make_unique<Private::ImmediateCompositeSimulator>(SimulatorType::kQiskitAer);
#endif
    case SimulatorType::kCompositeQCSim:
        return std::make_unique<Private::CompositeSimulator>(SimulatorType::kQCSim);
#ifdef __linux__
    case SimulatorType::kDistGpuSim:
        if (m != SimulationType::kStatevector)
            throw std::invalid_argument("Distributed GPU supports only statevector");
        if (!IsDistributedGpuAvailable())
            return nullptr;
        return std::make_unique<Private::DistributedGpuSimulator>();
    case SimulatorType::kDistMpiGpuSim:
        if (m != SimulationType::kStatevector)
            throw std::invalid_argument("Distributed MPI GPU supports only statevector");
        return std::make_unique<Private::DistributedMpiGpuSimulator>();

    case SimulatorType::kGpuSim:
        // Library initialization is checked before advertising the backend;
        // device resources remain lazy and configuration follows creation.
        if (GetGpuDeviceCount() <= 0)
            return nullptr;
        if (auto immediate = CreateImmediateSimulatorUnique(t, m))
            return std::make_unique<Private::GpuSimulator>(std::move(immediate));
        return nullptr;
#endif
    case SimulatorType::kQuestSim:
        if (m != SimulationType::kStatevector)
            throw std::invalid_argument("Simulation Type not supported for Quest Simulator");
        else if (questLibrary && questLibrary->IsValid())
        {
            return std::make_unique<Private::QuestSimulator>();
        }
        return nullptr;
    default:
        break;
    }

    throw std::invalid_argument("Simulator Type not supported");

    return nullptr; // keep compillers happy
}
} // namespace Simulators
