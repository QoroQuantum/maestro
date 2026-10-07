#include "../../Simulators/Core/Factory.h"
#include "../../maestrolib/Interface.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace
{
void Check(bool ok)
{
    if (!ok)
        throw std::runtime_error("GPU tensor dispatch/failure contract failed");
}

template <class F> void Reject(F &&f)
{
    bool rejected = false;
    try
    {
        f();
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    Check(rejected);
}
} // namespace

int main(int argc, char **)
{
    try
    {
        const bool old = argc > 1;
        Check(Simulators::SimulatorsFactory::InitGpuLibraryWithMute());
        auto lib = Simulators::SimulatorsFactory::GetGpuLibrary();
        Check(bool(lib));
        auto calls = reinterpret_cast<int (*)(int)>(lib->GetFunction("MockTensorCalls"));
        Check(calls != nullptr);
        auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
        sim->Configure("gate_fusion", "false");
        sim->AllocateQubits(2);
        sim->Initialize();
        Check(lib->HasMPSExpectationValues() != old);
        Check(calls(4) == (old ? 0 : 1) && calls(5) == (old ? 1 : 0));
        Check(sim->GetCurrentMaxBondDimension() == 3); // callback, not getter (one)
        const auto values = sim->ExpectationValues({"ZI", "XI", "ZI"});
        Check(values == std::vector<double>({1., 0., 1.}));
        Check(calls(1) == (old ? 0 : 1) && calls(0) == (old ? 3 : 0));
        const auto state = sim->GetStateVector();
        Check(state.size() == 4 && state[0] == 1. && state[3] == 0.);
        const auto probs = sim->AllProbabilities();
        Check(probs == std::vector<double>({1., 0., 0., 0.}));
        Check(calls(2) == (old ? 0 : 1) && calls(3) == (old ? 0 : 1));
        Check(calls(7) == (old ? 8 : 0));
        if (old)
        {
            Reject([&] { sim->MoveAtBeginningOfChain({1}); });
            Reject([&] { sim->ExpectationValueOperators({}, {}); });
        }
        else
        {
            sim->MoveAtBeginningOfChain({1});
            Check(sim->ExpectationValueOperators({}, {}) == 1.);
            Check(calls(8) == 1 && calls(9) == 1);
            setenv("MAESTRO_TEST_TENSOR_FAILURE", "batch", 1);
            const char *paulis[]{"ZI", "XI"};
            double output[]{7., 8.};
            Check(!MaestroExpectationValues(sim.get(), paulis, 2, output, 2));
            Check(output[0] == 7. && output[1] == 8. && calls(0) == 0);
            setenv("MAESTRO_TEST_TENSOR_FAILURE", "dense", 1);
            double raw[8]{9.};
            Check(!MaestroGetStateVector(sim.get(), raw, 4));
            Check(raw[0] == 9.);
            setenv("MAESTRO_TEST_TENSOR_FAILURE", "probabilities", 1);
            Reject([&] { sim->AllProbabilities(); });
            Check(AllProbabilities(sim.get()) == nullptr);
            setenv("MAESTRO_TEST_TENSOR_FAILURE", "operators", 1);
            double re = 7., im = 8.;
            Check(!MaestroExpectationValueOperators(sim.get(), nullptr, 0, nullptr, &re, &im));
            Check(re == 7. && im == 8.);
            setenv("MAESTRO_TEST_TENSOR_FAILURE", "move", 1);
            Reject([&] { sim->MoveAtBeginningOfChain({1}); });
        }
        for (const char *failure : {"sample", "partial"})
        {
            setenv("MAESTRO_TEST_TENSOR_FAILURE", failure, 1);
            Reject([&] { sim->SampleCounts({0, 1}, 10); });
            Reject([&] { sim->SampleCountsMany({0, 1}, 10); });
            const unsigned long long targets[]{0, 1};
            Check(SampleCounts(sim.get(), targets, 2, 10) == nullptr);
            unsigned char bits[2]{};
            unsigned long long counts[1]{};
            const unsigned long wideTargets[]{0, 1};
            size_t written = 0;
            Check(!MaestroSampleCountsBits(sim.get(), wideTargets, 2, 10, bits, counts, 1, &written));
            Check(calls(6) == 0); // failure must release the histogram
        }
        unsetenv("MAESTRO_TEST_TENSOR_FAILURE");
        Check(sim->SampleCounts({0, 1}, 10).at(0) == 10);
        Check(calls(6) == 0);
        auto wide = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
        wide->Configure("gate_fusion", "false");
        wide->AllocateQubits(64);
        wide->Initialize();
        Reject([&] { wide->GetStateVector(); });
        Reject([&] { wide->AllProbabilities(); });
        double untouched[]{9.};
        Check(!MaestroGetStateVector(wide.get(), untouched, 1));
        Check(untouched[0] == 9.);
        std::cout << (old ? "Legacy fallbacks" : "Native dispatch") << " and failure propagation passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
