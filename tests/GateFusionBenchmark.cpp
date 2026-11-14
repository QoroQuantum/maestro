// Manual benchmark: no wall-clock assertions. Always includes the final flush.
#include "Simulators/Core/Factory.h"
#include "Simulators/Core/QuantumChannel.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace Simulators;
using Clock = std::chrono::steady_clock;

void Run(SimulationType method, const char *name, size_t qubits, SimulatorType backend, bool sparse)
{
    std::vector<double> reference;
    for (bool fusion : {false, true})
    {
        auto sim = SimulatorsFactory::CreateSimulator(backend, method);
        sim->Configure("gate_fusion", fusion ? "true" : "false");
        sim->Configure("use_double_precision", "true");
        sim->Configure("matrix_product_state_max_bond_dimension", "256");
        sim->Configure("matrix_product_state_truncation_threshold", "0");
        sim->SetMultithreading(false);
        sim->AllocateQubits(qubits);
        sim->Initialize();
        const auto start = Clock::now();
        for (size_t layer = 0; layer < 8; ++layer)
            for (size_t a = layer % 2; a + 1 < qubits; a += 2)
            {
                const auto b = a + 1;
                if (sparse)
                {
                    sim->ApplyCX(a, (a + 1 + layer % (qubits - 1)) % qubits);
                }
                else
                {
                    sim->ApplyH(a);
                    sim->ApplyRz(a, .11);
                    sim->ApplyH(b);
                    sim->ApplyCX(a, b);
                    sim->ApplyRy(a, .07);
                    sim->ApplyCP(b, a, .05);
                }
            }
        sim->Flush();
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        const auto stats = sim->GetGateFusionStatistics();
        // Probe a fixed set of outcomes. Enumerating an entire MPS state on the
        // GPU can cost far more than the circuit and is outside the timed region.
        std::vector<double> probabilities;
        const size_t outcomes = size_t{1} << qubits;
        for (size_t i = 0; i < std::min(outcomes, size_t{64}); ++i)
            probabilities.push_back(sim->Probability((i * 2654435761ULL) % outcomes));
        double error = 0;
        if (!fusion)
            reference = probabilities;
        else
            for (size_t i = 0; i < reference.size(); ++i)
                error = std::max(error, std::abs(reference[i] - probabilities[i]));
        std::cout << name << ',' << qubits << ',' << (sparse ? "sparse" : "repeated_pair") << ',' << fusion << ',' << stats.submittedGates << ','
                  << stats.backendGates << ',' << std::setprecision(7) << ms << ',' << error << '\n';
        if (error > 1e-8)
            throw std::runtime_error("Benchmark result mismatch");
    }
}

// Trajectories of a noisy circuit on the statevector: layers of one- and
// two-qubit gates on a line, each gate followed by its error as a channel
// (depolarizing, then relaxation with T2 > T1, which is drawn from the state).
// A block of gates and channels is either applied first, its norm telling
// whether a channel did anything, or resolved from the reduced density matrix
// first (trajectory_blocks_read_first). Both draw the same branches, so the
// probabilities probed after the last trajectory must agree.
void RunTrajectories(size_t qubits, size_t trajectories, double scale, bool threads)
{
    const QuantumChannel relax1 = QuantumChannel::ThermalRelaxation(30e-9 * scale, 100e-6, 120e-6);
    const QuantumChannel relax2 = QuantumChannel::ThermalRelaxation(300e-9 * scale, 100e-6, 120e-6);
    const QuantumChannel error1 = QuantumChannel::Depolarizing(std::min(1e-3 * scale, .5)).Then(relax1);
    const QuantumChannel error2 = QuantumChannel::TwoQubitDepolarizing(std::min(1e-2 * scale, .5)).Then(relax2, {0}).Then(relax2, {1});
    std::vector<double> reference;
    double readFirstMs = 0;
    for (bool readFirst : {true, false})
    {
        auto sim = SimulatorsFactory::CreateSimulator(SimulatorType::kQCSim, SimulationType::kStatevector);
        sim->Configure("gate_fusion", "true");
        sim->Configure("trajectory_blocks_read_first", readFirst ? "true" : "false");
        sim->SetMultithreading(threads);
        sim->AllocateQubits(qubits);
        sim->Initialize();
        const auto start = Clock::now();
        for (size_t trajectory = 0; trajectory < trajectories; ++trajectory)
        {
            sim->Reset();
            sim->SetSeed(1000 + trajectory);
            for (size_t layer = 0; layer < 8; ++layer)
            {
                for (size_t q = 0; q < qubits; ++q)
                {
                    sim->ApplyRy(q, .3 + .1 * static_cast<double>(layer + q));
                    sim->ApplyQuantumChannel({q}, error1);
                }
                for (size_t a = layer % 2; a + 1 < qubits; a += 2)
                {
                    sim->ApplyCX(a, a + 1);
                    sim->ApplyQuantumChannel({a, a + 1}, error2);
                }
            }
            sim->Flush();
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / static_cast<double>(trajectories);
        const auto stats = sim->GetGateFusionStatistics();
        std::vector<double> probabilities;
        const size_t outcomes = size_t{1} << qubits;
        for (size_t i = 0; i < std::min(outcomes, size_t{64}); ++i)
            probabilities.push_back(sim->Probability((i * 2654435761ULL) % outcomes));
        double error = 0;
        if (readFirst)
        {
            reference = probabilities;
            readFirstMs = ms;
        }
        else
            for (size_t i = 0; i < reference.size(); ++i)
                error = std::max(error, std::abs(reference[i] - probabilities[i]));
        std::cout << qubits << ',' << scale << ',' << (threads ? "threads" : "serial") << ',' << (readFirst ? "read_first" : "applied_first") << ','
                  << trajectories << ',' << stats.backendGates / trajectories << ',' << stats.unreadBlocks / trajectories << ','
                  << static_cast<double>(stats.rereadBlocks) / static_cast<double>(trajectories) << ',' << std::setprecision(5) << ms << ','
                  << (readFirst ? 1. : readFirstMs / ms) << ',' << error << '\n';
        if (error > 1e-8)
            throw std::runtime_error("Trajectory benchmark mismatch");
    }
}

int main(int argc, char **argv)
{
    try
    {
        if (argc > 1 && std::string(argv[1]) == "--trajectories")
        {
            std::cout << "qubits,noise_scale,threading,blocks,trajectories,writes,unread_blocks,reread_blocks,ms_per_trajectory,speedup,"
                         "max_sampled_probability_error\n";
            // Calibrated noise, and noise thirty times stronger, where more blocks have to be read after all.
            for (const double scale : {1., 30.})
            {
                RunTrajectories(12, 2000, scale, false);
                RunTrajectories(16, 200, scale, false);
                RunTrajectories(20, 12, scale, false);
                RunTrajectories(20, 12, scale, true);
            }
            return 0;
        }
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
        const auto backend = gpu ? SimulatorType::kGpuSim : SimulatorType::kQCSim;
#ifdef __linux__
        if (gpu && !SimulatorsFactory::InitGpuLibrary())
            return 1;
#else
        if (gpu)
            throw std::runtime_error("GPU backend requires Linux");
#endif
        std::cout << "method,qubits,pattern,fusion,submitted,backend,milliseconds,"
                     "max_sampled_probability_error\n";
        for (bool sparse : {false, true})
        {
            if (!gpu)
                Run(SimulationType::kStatevector, "statevector", 20, backend, sparse);
            Run(SimulationType::kDensityMatrix, "density_matrix", 8, backend, sparse);
            Run(SimulationType::kMatrixProductState, "mps", 14, backend, sparse);
            Run(SimulationType::kMatrixProductOperator, "mpo", 8, backend, sparse);
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
