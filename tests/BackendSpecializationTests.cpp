#include "Simulators/Core/Factory.h"

#include <array>
#include <cmath>
#include <future>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <typeindex>
#ifdef _OPENMP
#include <omp.h>
#endif

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

void ReproducibleTrajectoryArithmetic()
{
#ifdef _OPENMP
    const int previousThreads = omp_get_max_threads();
#endif
    for (auto method : {SimulationType::kStatevector, SimulationType::kDensityMatrix})
        for (bool fused : {false, true})
            for (uint64_t seed : {uint64_t{0}, uint64_t{1} << 40})
                for (int team : {1, 3, 8})
                {
#ifdef _OPENMP
                    omp_set_num_threads(team);
#endif
                    const size_t qubits = method == SimulationType::kStatevector ? 15 : 8;
                    auto create = [&](bool parallel) {
                        auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, method);
                        sim->Configure("gate_fusion", fused ? "true" : "false");
                        // The serial native legacy path is the pre-change arithmetic
                        // reference, including its measurement RNG conversion.
                        sim->Configure("sampling_policy", parallel ? "reproducible_v1" : "legacy");
                        sim->Configure("reproducible_trajectory", "true");
                        sim->SetMultithreading(parallel);
                        sim->SetSeed(seed);
                        sim->AllocateQubits(qubits);
                        sim->Initialize();
                        for (size_t q = 0; q < qubits; ++q)
                        {
                            sim->ApplyRy(q, .137 + .039 * q);
                            sim->ApplyRz(q, .09 * q);
                            if (q) sim->ApplyCX(q - 1, q);
                        }
                        sim->SaveState();
                        return sim;
                    };
                    auto serial = create(false), parallel = create(true);
                    Types::qubits_vector allQubits(qubits);
                    for (size_t q = 0; q < qubits; ++q) allQubits[q] = q;
                    const char *phase = "initial preparation";
                    auto sameState = [&] {
                        if (method == SimulationType::kStatevector)
                        {
                            const auto a = serial->GetStateVector(), b = parallel->GetStateVector();
                            if (a != b)
                            {
                                std::cerr << "State mismatch: phase=" << phase << " fused=" << fused << " seed=" << seed << " team=" << team << '\n';
                                for (size_t i = 0; i < a.size(); ++i)
                                    if (a[i] != b[i]) { std::cerr << "First mismatch at " << i << ": " << std::hexfloat << a[i] << " vs " << b[i] << std::defaultfloat << '\n'; break; }
                            }
                            Check(a == b, "Parallel trajectory changed exact amplitudes");
                        }
                        else
                        {
                            const auto a = serial->PartialTrace(allQubits), b = parallel->PartialTrace(allQubits);
                            if (a != b)
                                std::cerr << "Density mismatch: phase=" << phase << " fused=" << fused << " seed=" << seed << " team=" << team << '\n';
                            Check(a == b, "Parallel trajectory changed exact density matrix");
                        }
                    };
                    sameState();
                    for (size_t shot = 0; shot < 8; ++shot)
                    {
                        serial->RestoreState();
                        parallel->RestoreState();
                        const Types::qubits_vector measured{qubits - 1, 0, qubits / 2, 0};
                        Check(serial->MeasureMany(measured) == parallel->MeasureMany(measured), "Parallel collapse changed native seeded outcomes");
                        phase = "collapse";
                        sameState();
                        serial->ApplyReset({0, qubits / 2});
                        parallel->ApplyReset({0, qubits / 2});
                        phase = "reset";
                        sameState();
                        const double angle = .231 + shot * .017;
                        const std::vector<std::pair<const char *, std::function<void(ISimulator &)>>> gates{
                            {"sim.ApplyH", [&](ISimulator &sim) { sim.ApplyH(0); }},
                            {"sim.ApplyH", [&](ISimulator &sim) { sim.ApplyH(1); }},
                            {"sim.ApplyH", [&](ISimulator &sim) { sim.ApplyH(2); }},
                            {"sim.ApplyX", [&](ISimulator &sim) { sim.ApplyX(0); }},
                            {"sim.ApplyY", [&](ISimulator &sim) { sim.ApplyY(1); }},
                            {"sim.ApplyZ", [&](ISimulator &sim) { sim.ApplyZ(2); }},
                            {"sim.ApplyS", [&](ISimulator &sim) { sim.ApplyS(0); }},
                            {"sim.ApplySDG", [&](ISimulator &sim) { sim.ApplySDG(1); }},
                            {"sim.ApplyT", [&](ISimulator &sim) { sim.ApplyT(2); }},
                            {"sim.ApplyTDG", [&](ISimulator &sim) { sim.ApplyTDG(0); }},
                            {"sim.ApplySx", [&](ISimulator &sim) { sim.ApplySx(1); }},
                            {"sim.ApplySxDAG", [&](ISimulator &sim) { sim.ApplySxDAG(2); }},
                            {"sim.ApplyK", [&](ISimulator &sim) { sim.ApplyK(0); }},
                            {"sim.ApplyCX", [&](ISimulator &sim) { sim.ApplyCX(0, 1); }},
                            {"sim.ApplyCY", [&](ISimulator &sim) { sim.ApplyCY(1, 2); }},
                            {"sim.ApplyCZ", [&](ISimulator &sim) { sim.ApplyCZ(2, 0); }},
                            {"sim.ApplyCH", [&](ISimulator &sim) { sim.ApplyCH(0, 2); }},
                            {"sim.ApplyCSx", [&](ISimulator &sim) { sim.ApplyCSx(1, 0); }},
                            {"sim.ApplyCSxDAG", [&](ISimulator &sim) { sim.ApplyCSxDAG(2, 1); }},
                            {"sim.ApplySwap", [&](ISimulator &sim) { sim.ApplySwap(0, 2); }},
                            {"sim.ApplyCCX", [&](ISimulator &sim) { sim.ApplyCCX(0, 1, 2); }},
                            {"sim.ApplyCSwap", [&](ISimulator &sim) { sim.ApplyCSwap(1, 0, 2); }},
                            {"sim.ApplyP", [&](ISimulator &sim) { sim.ApplyP(0, angle); }},
                            {"sim.ApplyRx", [&](ISimulator &sim) { sim.ApplyRx(1, angle + .1); }},
                            {"sim.ApplyRy", [&](ISimulator &sim) { sim.ApplyRy(2, angle - .2); }},
                            {"sim.ApplyRz", [&](ISimulator &sim) { sim.ApplyRz(0, angle + .3); }},
                            {"sim.ApplyU", [&](ISimulator &sim) { sim.ApplyU(1, angle, -.4, .7, .2); }},
                            {"sim.ApplyCP", [&](ISimulator &sim) { sim.ApplyCP(0, 2, angle - .5); }},
                            {"sim.ApplyCRx", [&](ISimulator &sim) { sim.ApplyCRx(2, 1, angle + .6); }},
                            {"sim.ApplyCRy", [&](ISimulator &sim) { sim.ApplyCRy(1, 0, angle - .7); }},
                            {"sim.ApplyCRz", [&](ISimulator &sim) { sim.ApplyCRz(0, 1, angle + .8); }},
                            {"sim.ApplyCU", [&](ISimulator &sim) { sim.ApplyCU(2, 0, angle, .3, -.2, .4); }},
                        };
                        for (const auto &[name, apply] : gates)
                        {
                            apply(*serial);
                            apply(*parallel);
                            phase = name;
                            sameState();
                        }
                        ApplyGateSequence(*serial, angle + .09, true);
                        ApplyGateSequence(*parallel, angle + .09, true);
                        phase = "uninterrupted gate burst";
                        sameState();
                        if (method == SimulationType::kDensityMatrix)
                        {
                            Eigen::MatrixXcd unitary(2, 2);
                            unitary << std::cos(.37), std::complex<double>(0., -std::sin(.37)),
                                       std::complex<double>(0., -std::sin(.37)), std::cos(.37);
                            for (auto *sim : {serial.get(), parallel.get()})
                            {
                                sim->ApplyAmplitudeDamping(0, .23);
                                sim->ApplyCorrelatedPhaseFlipNoise(0, qubits - 1, .17);
                                sim->ApplyKrausChannel({qubits - 1}, {std::sqrt(.6) * Eigen::MatrixXcd::Identity(2, 2), std::sqrt(.4) * unitary});
                            }
                            phase = "density channels";
                            sameState();
                        }
                        Check(serial->Measure({1, qubits - 1, 0}) == parallel->Measure({1, qubits - 1, 0}), "Reset or packed measurement changed RNG continuation");
                        phase = "packed collapse";
                        sameState();
                    }
                }
#ifdef _OPENMP
    omp_set_num_threads(previousThreads);
#endif
    std::cout << "PASS exact serial/parallel trajectory arithmetic and RNG continuation\n";
}

void TerminalDensityArithmetic()
{
#ifdef _OPENMP
    const int previousThreads = omp_get_max_threads();
    omp_set_num_threads(3);
#endif
    std::vector<std::complex<double>> input(256);
    double norm = 0.;
    for (size_t i = 0; i < input.size(); ++i)
    {
        input[i] = {std::sin(.17 * i), std::cos(.03 * i)};
        norm += std::norm(input[i]);
    }
    for (auto &value : input) value /= std::sqrt(norm);
    auto create = [&](const char *policy) {
        auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, SimulationType::kDensityMatrix);
        sim->Configure("gate_fusion", "false");
        sim->Configure("sampling_policy", policy);
        sim->Configure("reproducible_trajectory", "true");
        sim->Configure("reproducible_trajectory", "false");
        sim->SetMultithreading(true);
        sim->InitializeState(8, input);
        sim->ApplyT(2);
        sim->ApplyCP(0, 2, .231);
        return sim;
    };
    auto old = create("legacy"), current = create("reproducible_v1");
    Check(old->PartialTrace({0, 1, 2, 3, 4, 5, 6, 7}) == current->PartialTrace({0, 1, 2, 3, 4, 5, 6, 7}),
          "Terminal density preparation retained the trajectory-only serial diagonal path");
#ifdef _OPENMP
    omp_set_num_threads(previousThreads);
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
        ReproducibleTrajectoryArithmetic();
        TerminalDensityArithmetic();
        if (!publicOnly)
            GpuSelection();
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
