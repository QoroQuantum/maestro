#include "Simulators/Core/Factory.h"

#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <typeindex>

using namespace Simulators;

namespace
{
struct Method
{
    SimulationType type;
    const char *name;
};

const Method methods[] = {{SimulationType::kStatevector, "statevector"},
                          {SimulationType::kMatrixProductState, "matrix_product_state"},
                          {SimulationType::kMatrixProductOperator, "matrix_product_operator"},
                          {SimulationType::kStabilizer, "stabilizer"},
                          {SimulationType::kTensorNetwork, "tensor_network"},
                          {SimulationType::kPauliPropagator, "pauli_propagator"},
                          {SimulationType::kPathIntegral, "path_integral"},
                          {SimulationType::kDensityMatrix, "density_matrix"},
                          {SimulationType::kExtendedStabilizer, "extended_stabilizer"}};

void Check(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void CheckState(ISimulator &sim)
{
    Check(std::abs(sim.Probability(2) - 0.5) < 1e-9, "backend changed Bell-state probability");
    Check(std::abs(sim.Probability(5) - 0.5) < 1e-9, "backend changed entangled-state probability");
    Check(std::abs(sim.ExpectationValue("ZIZ") - 1.) < 1e-9, "backend changed Pauli expectation");
}

void Exercise(std::unique_ptr<ISimulator> sim, const Method &method)
{
    Check(sim->GetSimulationType() == method.type, "factory selected the wrong method");
    Check(sim->GetConfiguration("method") == method.name, "method configuration disagrees with concrete type");
    Check(sim->GetConfiguration("precision") == "double", "CPU precision query failed before initialization");
    sim->Reset();                          // No native state exists yet.
    sim->Configure("method", method.name); // Configuration replay is still valid.
    sim->SetMultithreading(false);
    sim->SetSeed(1234);
    sim->AllocateQubits(3);
    sim->Initialize();
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    sim->ApplyCX(1, 2);
    sim->ApplyX(1);
    CheckState(*sim);

    sim->SaveState();
    auto copy = sim->Clone();
    Check(typeid(*copy) == typeid(*sim), "clone lost its concrete backend type");
    Check(copy->GetSimulationType() == method.type, "clone lost its method");
    CheckState(*copy);
    sim.reset();
    copy->ApplyX(1);
    copy->RestoreState();
    CheckState(*copy);

    const auto samples = copy->SampleCounts({2, 0}, 32);
    size_t shots = 0;
    for (const auto &[outcome, count] : samples)
    {
        Check(outcome == 0 || outcome == 3, "backend lost measurement correlations");
        shots += count;
    }
    Check(shots == 32, "backend lost samples");
    copy->Clear();
    copy->AllocateQubits(2);
    copy->Initialize();
    Check(std::abs(copy->Probability(0) - 1.) < 1e-9, "clear/reinitialize changed the backend state");
}

void CpuBackends(bool publicOnly)
{
    std::set<std::type_index> types;
    for (const auto &method : methods)
    {
        std::cout << "Checking QCSim " << method.name << std::endl;
        if (!publicOnly)
        {
            auto immediate = SimulatorsFactory::CreateImmediateSimulatorUnique(SimulatorType::kQCSim, method.type);
            Check(types.insert(typeid(*immediate)).second, "QCSim methods still share a multiplexer implementation");
            bool rejected = false;
            try
            {
                immediate->Configure("method", method.type == SimulationType::kStatevector ? "matrix_product_state" : "statevector");
            }
            catch (const std::invalid_argument &)
            {
                rejected = true;
            }
            Check(rejected && immediate->GetConfiguration("method") == method.name, "method mutation corrupted a concrete backend");
            Exercise(std::move(immediate), method);
        }
        Exercise(SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, method.type), method);
        auto shared = SimulatorsFactory::CreateSimulator(SimulatorType::kQCSim, method.type);
        Check(shared->GetSimulationType() == method.type, "shared and unique factories disagree");
        std::cout << "PASS QCSim " << method.name << '\n';
    }
}

void ApplyGateSequence(ISimulator &sim, double angle, bool threeQubitGates)
{
    sim.ApplyH(0);
    sim.ApplyH(1);
    sim.ApplyH(2);
    sim.ApplyX(0);
    sim.ApplyY(1);
    sim.ApplyZ(2);
    sim.ApplyS(0);
    sim.ApplySDG(1);
    sim.ApplyT(2);
    sim.ApplyTDG(0);
    sim.ApplySx(1);
    sim.ApplySxDAG(2);
    sim.ApplyK(0);
    sim.ApplyCX(0, 1);
    sim.ApplyCY(1, 2);
    sim.ApplyCZ(2, 0);
    sim.ApplyCH(0, 2);
    sim.ApplyCSx(1, 0);
    sim.ApplyCSxDAG(2, 1);
    sim.ApplySwap(0, 2);
    if (threeQubitGates)
    {
        sim.ApplyCCX(0, 1, 2);
        sim.ApplyCSwap(1, 0, 2);
    }
    sim.ApplyP(0, angle);
    sim.ApplyRx(1, angle + .1);
    sim.ApplyRy(2, angle - .2);
    sim.ApplyRz(0, angle + .3);
    sim.ApplyU(1, angle, -.4, .7, .2);
    sim.ApplyCP(0, 2, angle - .5);
    sim.ApplyCRx(2, 1, angle + .6);
    sim.ApplyCRy(1, 0, angle - .7);
    sim.ApplyCRz(0, 1, angle + .8);
    sim.ApplyCU(2, 0, angle, .3, -.2, .4);
}

std::array<double, 16> GateSequenceProbabilities(const Method &method, double angle, bool fused)
{
    auto sim = fused ? SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, method.type)
                     : SimulatorsFactory::CreateImmediateSimulatorUnique(SimulatorType::kQCSim, method.type);
    sim->SetMultithreading(false);
    sim->SetSeed(1234);
    sim->AllocateQubits(3);
    sim->Initialize();
    const bool threeQubitGates =
        method.type == SimulationType::kStatevector || method.type == SimulationType::kDensityMatrix || method.type == SimulationType::kPathIntegral;
    ApplyGateSequence(*sim, angle, threeQubitGates);
    sim->Flush();
    auto clone = sim->Clone();
    ApplyGateSequence(*sim, angle + .17, threeQubitGates);
    ApplyGateSequence(*clone, angle - .23, threeQubitGates);

    std::array<double, 16> result;
    for (Types::qubit_t outcome = 0; outcome < 8; ++outcome)
    {
        result[outcome] = sim->Probability(outcome);
        result[8 + outcome] = clone->Probability(outcome);
    }
    return result;
}

void ConcurrentGateApplications(bool publicOnly)
{
    // Compare independent instances and clones running concurrently with their
    // serial results. Every fixed gate and every parameterized cache is used.
    struct Run
    {
        Method method;
        double angle;
        std::array<double, 16> expected;
    };

    for (const bool fused : {false, true})
    {
        if (publicOnly && !fused)
            continue;
        std::vector<Run> runs;
        for (const auto &method : methods)
        {
            if (method.type == SimulationType::kStabilizer || method.type == SimulationType::kExtendedStabilizer ||
                method.type == SimulationType::kPauliPropagator)
                continue;
            const double angle = .13 * (runs.size() + 1);
            runs.push_back({method, angle, GateSequenceProbabilities(method, angle, fused)});
        }
        std::vector<std::future<void>> workers;
        for (const auto &run : runs)
            workers.push_back(std::async(std::launch::async, [run, fused]() {
                for (size_t repeat = 0; repeat < 4; ++repeat)
                {
                    const auto actual = GateSequenceProbabilities(run.method, run.angle, fused);
                    for (size_t i = 0; i < actual.size(); ++i)
                        Check(std::abs(actual[i] - run.expected[i]) < 1e-9, "concurrent gate applications changed a simulator or clone");
                }
            }));
        for (auto &worker : workers)
            worker.get();
    }
    std::cout << "PASS concurrent gate applications and clones\n";
}

void GpuSelection()
{
#ifdef __linux__
    // Construction is lazy: check every concrete GPU type even on CPU-only hosts.
    std::set<std::type_index> types;
    for (const auto &method : methods)
    {
        if (method.type == SimulationType::kStabilizer || method.type == SimulationType::kExtendedStabilizer || method.type == SimulationType::kPathIntegral)
            continue;
        auto sim = SimulatorsFactory::CreateImmediateSimulatorUnique(SimulatorType::kGpuSim, method.type);
        Check(types.insert(typeid(*sim)).second, "GPU methods still share a multiplexer implementation");
        Check(sim->GetSimulationType() == method.type, "GPU factory selected the wrong method");
        Check(sim->GetConfiguration("method") == method.name, "GPU method configuration disagrees with its type");
        Check(sim->GetNumberOfQubits() == 0, "GPU factory allocated native state eagerly");
        sim->GetConfiguration("precision");
        sim->Reset();
    }
    Check(!SimulatorsFactory::CreateImmediateSimulatorUnique(SimulatorType::kGpuSim, SimulationType::kStabilizer),
          "unsupported GPU methods must remain unavailable");
#endif
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        const bool publicOnly = argc > 1 && std::string(argv[1]) == "--public-only";
        CpuBackends(publicOnly);
        ConcurrentGateApplications(publicOnly);
        if (!publicOnly)
            GpuSelection();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
