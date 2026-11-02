#include "LoadedMaestro.h"
// Public adapter regressions for bounded GPU Pauli propagation.
#include "../Execution/Capabilities.h"
#include "../Execution/Options.h"
#include "../Execution/Request.h"
#include "../Simulators/Core/Factory.h"
#include <functional>
#define INCLUDED_BY_FACTORY
#include "../Simulators/Gpu/GpuPauliPropagatorSimulator.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace Simulators;

namespace
{
void Check(bool ok, const char *message)
{
    if (!ok)
        throw std::runtime_error(message);
}

void Near(double actual, double expected, double tolerance = 1e-9)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error("expected " + std::to_string(expected) + ", got " + std::to_string(actual));
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
    Check(rejected, "failure was accepted as a valid result");
}

auto Create(size_t qubits, SimulatorType backend = SimulatorType::kGpuSim)
{
    auto sim = SimulatorsFactory::CreateSimulator(backend, backend == SimulatorType::kGpuSim ? SimulationType::kPauliPropagator : SimulationType::kStatevector);
    Check(bool(sim), "create simulator");
    if (backend == SimulatorType::kGpuSim)
    {
        sim->Configure("precision", "double");
        sim->Configure("pauli_propagator_memory_limit_bytes", "16777216");
        sim->Configure("pauli_propagator_num_gates_between_deduplications", "10");
        sim->Configure("pauli_propagator_workers", "2");
    }
    sim->AllocateQubits(qubits);
    sim->Initialize();
    return sim;
}

void FailureAndPrecision()
{
    auto pp = SimulatorsFactory::CreateGpuPauliPropagatorSimulatorUnique();
    Check(pp && pp->CreateSimulator(1) && pp->AllocateMemoryBytes(1 << 20), "create wrapper");
    Reject([&] { pp->GetExpectationValue(); });
    Reject([&] { pp->ExpectationValue("?"); });
    Reject([&] { pp->ExpectationValueMultiple({"I", "Z"}, {1.}); });
    Near(pp->ExpectationValue("Z"), 1.);
    Near(pp->QubitProbability0(0), 1.);
    Near(pp->Probability(1), 0.);
    Check(!pp->MeasureQubit(0), "valid zero measurement remains distinguishable from failure");
    Check(pp->SetUseDoublePrecision(true), "configure actual double precision");
    Near(pp->ExpectationValueMultiple({"I"}, {1. + 1e-10}), 1. + 1e-10, 1e-13);
    Check(pp->SetUseDoublePrecision(false), "configure actual single precision");
    Near(pp->ExpectationValueMultiple({"I"}, {1. + 1e-10}), 1., 1e-13);
    auto limited = Create(20);
    limited->Configure("pauli_propagator_memory_limit_bytes", "65536");
    for (int q = 0; q < 20; ++q)
        limited->ApplyRx(q, .7853981633974483);
    Reject([&] { limited->ExpectationValue(std::string(20, 'Z')); });
    limited->Reset();
    Near(limited->ExpectationValue("Z"), 1.);
    Reject([&] { limited->SampleCounts({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, 1); });
    Near(limited->Probability(1), 0.);
}

void ConfigurationAndReference()
{
    auto gpu = Create(3), cpu = Create(3, SimulatorType::kQCSim);
    Check(gpu->GetConfiguration("precision") == "double", "reported precision matches native storage");
    gpu->Configure("pauli_propagator_steps_between_trims", "7");
    gpu->Configure("pauli_propagator_num_gates_between_deduplications", "5");
    Check(gpu->GetConfiguration("pauli_propagator_steps_between_trims") == "7", "deduplication changed independent trim interval");
    for (auto key : {"pauli_propagator_steps_between_trims", "pauli_propagator_num_gates_between_deduplications", "pauli_propagator_workers",
                     "pauli_propagator_memory_limit_bytes", "pauli_propagator_max_fused_gates"})
        for (auto value : {"0", "-1", "nan", "2junk"})
            Reject([&] { gpu->Configure(key, value); });
    Reject([&] { gpu->Configure("pauli_propagator_workers", "65"); });
    Reject([&] { gpu->Configure("pauli_propagator_max_fused_gates", "257"); });
    Reject([&] { gpu->Configure("pauli_propagator_coefficient_threshold", "nan"); });
    Reject([&] { gpu->Configure("pauli_propagator_pauli_weight_threshold", "1.5"); });
    for (auto sim : {gpu, cpu})
    {
        for (int i = 0; i < 12; ++i)
        {
            sim->ApplyH(i % 3);
            sim->ApplyS((i + 1) % 3);
            sim->ApplyCX(i % 3, (i + 1) % 3);
            sim->ApplyRx(i % 3, .13 * (i + 1));
            sim->ApplyRy((i + 1) % 3, .17);
            sim->ApplyRz((i + 2) % 3, -.21);
        }
    }
    std::vector<std::string> observables;
    for (int i = 0; i < 64; ++i)
    {
        std::string p(3, 'I');
        for (int q = 0; q < 3; ++q)
            p[q] = "IXYZ"[(i >> (2 * q)) & 3];
        observables.push_back(p);
    }
    for (int workers : {1, 2, 4})
    {
        gpu->Configure("pauli_propagator_workers", std::to_string(workers).c_str());
        auto values = gpu->ExpectationValues(observables);
        Check(values.size() == observables.size(), "observable batch length");
        for (size_t i = 0; i < values.size(); ++i)
            Near(values[i], cpu->ExpectationValue(observables[i]));
    }
    for (int i = 0; i < 8; ++i)
        Near(gpu->Probability(i), cpu->Probability(i));
    const double before = gpu->ExpectationValue("ZZZ");
    gpu->SaveState();
    gpu->ApplyH(0);
    gpu->Measure({1});
    gpu->RestoreState();
    Near(gpu->ExpectationValue("ZZZ"), before);
    Reject([&] { gpu->ExpectationValues({"III", "?II", "ZZZ"}); });
    Near(gpu->ExpectationValue("ZZZ"), before);
}

void Sampling()
{
    auto gpu = Create(3);
    gpu->ApplyRy(0, .7);
    gpu->ApplyCX(0, 1);
    gpu->ApplyH(2);
    gpu->SetSeed(975);
    const auto cached = gpu->SampleCounts({0, 1, 2}, 1024);
    gpu->Configure("pauli_propagator_sampling_cache_nodes", "0");
    gpu->SetSeed(975);
    Check(cached == gpu->SampleCounts({0, 1, 2}, 1024), "cache changes seeded shots");
    size_t ones = 0;
    for (const auto &[bits, count] : cached)
    {
        Check((bits & 1) == ((bits >> 1) & 1), "sampling lost Bell correlation");
        if (bits & 1)
            ones += count;
    }
    Near(double(ones) / 1024, std::pow(std::sin(.35), 2), .04);
    gpu->Reset();
    gpu->ApplyX(2);
    Check(gpu->SampleCounts({2, 1, 0}, 5).at(1) == 5, "sampling cache survived circuit reset");
    Check(gpu->SampleCounts({2, 0, 2}, 5).at(5) == 5, "repeated sampling qubits used independent draws");
    Check(gpu->SampleCountsMany(Types::qubits_vector(70, 2), 2).at(std::vector<bool>(70, true)) == 2, "wide sampling result");
    Check(gpu->SampleCounts(Types::qubits_vector(64, 2), 2).at(std::numeric_limits<Types::qubit_t>::max()) == 2, "64-bit packed boundary");
    Reject([&] { gpu->SampleCounts(Types::qubits_vector(65, 2), 1); });
    Reject([&] { gpu->SampleCounts({Types::qubit_t{1} << 32}, 1); });
}

void SamplingBudgetRequests()
{
    namespace j = boost::json;
    j::array source;
    for (int q = 0; q < 14; ++q)
        source.push_back(j::object{{"name", "ry"}, {"qubits", j::array{q}}, {"params", j::array{.21 + .03 * q}}});
    for (int q = 0; q < 13; ++q)
        source.push_back(j::object{{"name", "cx"}, {"qubits", j::array{q + 1, q}}});
    for (int q = 0; q < 14; ++q)
        source.push_back(j::object{{"name", "rx"}, {"qubits", j::array{q}}, {"params", j::array{.17 + .01 * q}}});
    for (int q = 0; q < 14; ++q)
        source.push_back(j::object{{"name", "measure"}, {"qubits", j::array{q}}, {"clbits", j::array{q}}});
    for (bool precision : {false, true})
    {
        for (int weight : {4, 5})
        {
            j::object request{{"schema_version", 2},
                              {"operation", "execute"},
                              {"circuit", j::object{{"format", "instructions"}, {"num_qubits", 14}, {"num_clbits", 14}, {"source", source}}},
                              {"simulator", j::object{{"backend", "gpu"},
                                                      {"method", "pauli_propagator"},
                                                      {"options", j::object{{"precision", precision ? "double" : "single"},
                                                                            {"pp_max_pauli_weight", weight},
                                                                            {"pp_gates_between_trims", 1},
                                                                            {"pp_gates_between_deduplications", 10},
                                                                            {"optimize_circuit", false},
                                                                            {"auto_reduce", false},
                                                                            {"enable_sampling_reduction", false}}}}},
                              {"execution", j::object{{"shots", 3}, {"seed", 183}}}};
            const auto run = [&](size_t kib, size_t cache) {
                auto &options = request.at("simulator").as_object().at("options").as_object();
                options["pp_memory_limit_bytes"] = kib * 1024;
                options["pp_sampling_cache_nodes"] = cache;
                const auto input = j::serialize(request);
                char *raw = MaestroRunRequestJson(input.c_str());
                Check(raw, "null sampling request response");
                const std::string output(raw);
                FreeResult(raw);
                auto result = j::parse(output).as_object();
                if (!result.at("ok").as_bool())
                    throw std::runtime_error(input + "\n" + output);
                const auto &metadata = result.at("execution_metadata").as_object();
                Check(metadata.at("backend") == "gpu" && metadata.at("method") == "pauli_propagator", "sampling request changed backend");
                Check(metadata.at("configured_options").at("pauli_propagator_memory_limit_bytes") == j::value(std::to_string(kib * 1024)),
                      "sampling request ignored the budget");
                return result.at("counts");
            };
            const auto expected = run(32768, 0);
            // Pin successful budgets inside and above the reported gaps, in both precisions.
            const std::vector<size_t> budgets = weight == 4
                                                    ? (precision ? std::vector<size_t>{768, 1024} : std::vector<size_t>{602, 614, 626, 640, 768})
                                                    : (precision ? std::vector<size_t>{3502, 3513, 3584, 4096}
                                                                 : std::vector<size_t>{2980, 3072, 3430, 3462, 3502, 3513, 4096});
            for (size_t cache : {0, 8192})
                for (size_t kib : budgets)
                    Check(run(kib, cache) == expected, "sampling budget or cache changed seeded request counts");
        }
    }
}

void QubitValidation()
{
    auto sim = Create(3);
    sim->Configure("pauli_propagator_workers", "1");
    sim->ApplyH(0);
    sim->ApplyX(1);
    const auto unchanged = [&] {
        Near(sim->ExpectationValue("XII"), 1.);
        Near(sim->ExpectationValue("IZI"), -1.);
        Near(sim->ExpectationValue("IIZ"), 1.);
    };
    using Q = Types::qubit_t;
    const std::function<void(Q)> gates[]{
        [&](Q q) { sim->ApplyP(q, .3); }, [&](Q q) { sim->ApplyX(q); }, [&](Q q) { sim->ApplyY(q); },
        [&](Q q) { sim->ApplyZ(q); }, [&](Q q) { sim->ApplyH(q); }, [&](Q q) { sim->ApplyS(q); },
        [&](Q q) { sim->ApplySDG(q); }, [&](Q q) { sim->ApplyT(q); }, [&](Q q) { sim->ApplyTDG(q); },
        [&](Q q) { sim->ApplySx(q); }, [&](Q q) { sim->ApplySxDAG(q); }, [&](Q q) { sim->ApplyK(q); },
        [&](Q q) { sim->ApplyRx(q, .3); }, [&](Q q) { sim->ApplyRy(q, .3); }, [&](Q q) { sim->ApplyRz(q, .3); },
        [&](Q q) { sim->ApplyU(q, .3, .4, .5, .6); },
        [&](Q q) { sim->ApplyCX(q, 2); }, [&](Q q) { sim->ApplyCX(2, q); },
        [&](Q q) { sim->ApplyCY(q, 2); }, [&](Q q) { sim->ApplyCY(2, q); },
        [&](Q q) { sim->ApplyCZ(q, 2); }, [&](Q q) { sim->ApplyCZ(2, q); },
        [&](Q q) { sim->ApplyCP(q, 2, .3); }, [&](Q q) { sim->ApplyCP(2, q, .3); },
        [&](Q q) { sim->ApplyCRx(q, 2, .3); }, [&](Q q) { sim->ApplyCRx(2, q, .3); },
        [&](Q q) { sim->ApplyCRy(q, 2, .3); }, [&](Q q) { sim->ApplyCRy(2, q, .3); },
        [&](Q q) { sim->ApplyCRz(q, 2, .3); }, [&](Q q) { sim->ApplyCRz(2, q, .3); },
        [&](Q q) { sim->ApplyCH(q, 2); }, [&](Q q) { sim->ApplyCH(2, q); },
        [&](Q q) { sim->ApplyCSx(q, 2); }, [&](Q q) { sim->ApplyCSx(2, q); },
        [&](Q q) { sim->ApplyCSxDAG(q, 2); }, [&](Q q) { sim->ApplyCSxDAG(2, q); },
        [&](Q q) { sim->ApplySwap(q, 2); }, [&](Q q) { sim->ApplySwap(2, q); },
        [&](Q q) { sim->ApplyCU(q, 2, .3, .4, .5, .6); }, [&](Q q) { sim->ApplyCU(2, q, .3, .4, .5, .6); },
        [&](Q q) { sim->ApplyCCX(q, 1, 2); }, [&](Q q) { sim->ApplyCCX(0, q, 2); }, [&](Q q) { sim->ApplyCCX(0, 1, q); },
        [&](Q q) { sim->ApplyCSwap(q, 1, 2); }, [&](Q q) { sim->ApplyCSwap(0, q, 2); }, [&](Q q) { sim->ApplyCSwap(0, 1, q); }};
    sim->SetSeed(975);
    const auto expected = sim->SampleCounts({0, 1, 2}, 24);
    for (Q invalid : {Q{3}, Q{INT_MAX} + 1, Q{1} << 32, (Q{1} << 32) + 1, std::numeric_limits<Q>::max()})
    {
        for (const auto &gate : gates)
        {
            Reject([&] { gate(invalid); });
            unchanged();
        }
        // The valid first qubit must neither collapse/reset nor consume a draw
        // when a later index is invalid.
        for (int operation : {0, 1, 2})
        {
            sim->SetSeed(975);
            Reject([&] {
                if (operation == 0) sim->Measure({0, invalid});
                else if (operation == 1) sim->MeasureMany({0, invalid});
                else sim->ApplyReset({1, invalid});
            });
            unchanged();
            Check(sim->SampleCounts({0, 1, 2}, 24) == expected, "invalid measurement/reset advanced the RNG");
        }
    }
    Private::GpuPauliPropagatorSimulator allocation;
    Check(allocation.AllocateQubits(1) == 0, "first qubit allocation");
    Reject([&] { allocation.AllocateQubits(size_t{1} << 32); });
    Reject([&] { allocation.AllocateQubits(static_cast<size_t>(INT_MAX)); });
    Check(allocation.AllocateQubits(1) == 1, "rejected qubit count preserved allocation state");
}

void RequestOptions()
{
    namespace j = boost::json;
    using MaestroExecution::ParseConfig;
    j::object config{
        {"backend", "gpu"},
        {"method", "pauli_propagator"},
        {"options", j::object{{"pp_workers", 2}, {"pp_memory_limit_bytes", 16777216}, {"pp_max_fused_gates", 32}, {"pp_sampling_cache_nodes", 8192}}}};
    auto parsed = ParseConfig(config);
    Check(parsed.native_options.at("pauli_propagator_workers") == "2", "worker request option not forwarded");
    for (auto pair : {std::pair{"pp_workers", 0}, {"pp_workers", 65}, {"pp_memory_limit_bytes", 0}, {"pp_max_fused_gates", 257}})
    {
        auto bad = config;
        bad.at("options").as_object()[pair.first] = pair.second;
        Reject([&] { ParseConfig(bad); });
    }
    config["backend"] = "qcsim";
    Reject([&] { ParseConfig(config); });
}

void FailedOperationsPreserveState()
{
    struct Observer : ISimulatorObserver
    {
        size_t calls = 0;

        void Update(const Types::qubits_vector &) override
        {
            ++calls;
        }
    };

    struct Inspected : Private::GpuPauliPropagatorSimulator
    {
        using IState::DontNotify;
        using IState::Notify;
    };

    auto sim = std::make_shared<Inspected>();
    sim->Configure("pauli_propagator_memory_limit_bytes", "16777216");
    sim->AllocateQubits(2);
    sim->Initialize();
    auto observer = std::make_shared<Observer>();
    sim->RegisterObserver(observer);
    const std::function<void()> failures[]{[&] { sim->SampleCounts({2}, 1); }, [&] { sim->SampleCountsMany({2}, 1); }, [&] { sim->Measure({2}); },
                                           [&] { sim->MeasureMany({2}); }, [&] { sim->ApplyReset({2}); }};
    for (const auto &fail : failures)
    {
        Reject(fail);
        observer->calls = 0;
        sim->ApplyX(0);
        Check(observer->calls == 1, "failed operation left notifications disabled");
    }
    sim->DontNotify();
    Reject([&] { sim->SampleCounts({2}, 1); });
    observer->calls = 0;
    sim->ApplyX(0);
    Check(observer->calls == 0, "notification guard overwrote previously disabled notifications");
    sim->Notify();
    sim->Reset();
    Reject([&] { sim->Configure("pauli_propagator_memory_limit_bytes", "40"); });
    Check(sim->GetConfiguration("pauli_propagator_memory_limit_bytes") == "16777216", "rejected budget changed reported configuration");
    Near(sim->ExpectationValue("Z"), 1.);
    Reject([&] { sim->ApplyU(0, 1., std::numeric_limits<double>::quiet_NaN(), 0., 0.); });
    Near(sim->ExpectationValue("Z"), 1.);
    auto pp = SimulatorsFactory::CreateGpuPauliPropagatorSimulatorUnique();
    Check(pp && pp->CreateSimulator(2) && pp->AllocateMemoryBytes(1 << 20), "composite validation fixture");
    Check(!pp->ApplyCU(0, 1, 1., 0., 0., INFINITY) && !pp->ApplyCH(0, 0) && !pp->ApplyCRX(0, 2, .3) && !pp->ApplyCSwap(0, 1, 1) && !pp->ApplyCCX(0, 0, 1),
          "reject all composite arguments before recording");
    Near(pp->ExpectationValue("ZZ"), 1.);
}

void RequestDefaultsAndCapabilities()
{
    using namespace MaestroExecution;
    GetMaestroObjectWithMute();
    for (auto backend : {SimulatorType::kGpuSim, SimulatorType::kQCSim})
    {
        for (int scenario = 0; scenario < 5; ++scenario)
        {
            struct Owner
            {
                unsigned long handle = CreateSimpleSimulator(2);

                ~Owner()
                {
                    DestroySimpleSimulator(handle);
                }
            } owner;

            SimulatorConfig config;
            config.simulator_type = backend;
            config.simulation_type = SimulationType::kPauliPropagator;
            config.fixed_backend = true;
            if (backend == SimulatorType::kGpuSim)
                config.native_options["pauli_propagator_memory_limit_bytes"] = "16777216";
            if (scenario == 1 || scenario == 2 || scenario == 3)
                config.pp_coefficient_threshold = .001;
            if (scenario == 2)
                config.pp_gates_between_trims = 3;
            if (scenario == 3)
                config.native_options["pauli_propagator_steps_between_trims"] = "17";
            if (scenario == 4)
                config.native_options["pauli_propagator_pauli_weight_threshold"] = "1";
            config.pp_gates_between_deduplications = 7;
            const auto network = ConfigureNetwork(owner.handle, config);
            Check(network && network->GetSimulator(), "configure threshold request");
            const auto sim = network->GetSimulator();
            const auto trim = sim->GetConfiguration("pauli_propagator_steps_between_trims");
            Check(trim == (scenario == 0   ? ""
                           : scenario == 2 ? "3"
                           : scenario == 3 ? "17"
                                           : "7"),
                  "threshold default or explicit/native trim precedence changed");
            Check(sim->GetConfiguration("pauli_propagator_num_gates_between_deduplications") == "7", "trim default changes deduplication configuration");
        }
    }
    for (const auto &option : Options())
    {
        const std::string name = option.name;
        if (name == "pp_workers")
        {
            const auto cap = OptionCapability(option);
            Check(cap.at("maximum").to_number<int>() == 1024, "CPU worker bound changed");
            const auto &gpu = cap.at("constraints").as_array().at(0).as_object();
            Check(gpu.at("backend") == "gpu" && gpu.at("minimum").to_number<int>() == 1 && gpu.at("maximum").to_number<int>() == 64,
                  "GPU capabilities must match worker parser limits");
        }
        if (name == "pp_max_fused_gates")
            Check(OptionCapability(option).at("maximum").to_number<int>() == 256, "missing fused block maximum");
    }
}

void ThresholdCadenceParity()
{
    using namespace MaestroExecution;

    struct Owner
    {
        unsigned long handle = CreateSimpleSimulator(4);

        ~Owner()
        {
            DestroySimpleSimulator(handle);
        }
    };

    for (bool native : {false, true})
    {
        for (bool weight : {false, true})
        {
            for (int cadence : {0, 1, 3, 7, 25})
            {
                std::vector<double> reference;
                for (auto backend : {SimulatorType::kQCSim, SimulatorType::kGpuSim})
                {
                    Owner owner;
                    SimulatorConfig config;
                    config.simulator_type = backend;
                    config.simulation_type = SimulationType::kPauliPropagator;
                    config.fixed_backend = true;
                    if (backend == SimulatorType::kGpuSim)
                    {
                        config.precision = "double";
                        config.native_options["pauli_propagator_memory_limit_bytes"] = "16777216";
                    }
                    if (native)
                    {
                        config.native_options[weight ? "pauli_propagator_pauli_weight_threshold" : "pauli_propagator_coefficient_threshold"] =
                            weight ? "2" : "0.05";
                        if (cadence)
                            config.native_options["pauli_propagator_num_gates_between_deduplications"] = std::to_string(cadence);
                    }
                    else
                    {
                        if (weight)
                            config.pp_max_pauli_weight = 2;
                        else
                            config.pp_coefficient_threshold = .05;
                        if (cadence)
                            config.pp_gates_between_deduplications = cadence;
                    }
                    auto network = ConfigureNetwork(owner.handle, config);
                    Check(network && network->GetSimulator(), "create cadence parity simulator");
                    auto sim = network->GetSimulator();
                    Check(sim->GetConfiguration("pauli_propagator_steps_between_trims") == std::to_string(cadence ? cadence : 10),
                          "implicit trim cadence must follow effective deduplication cadence");
                    for (int layer = 0; layer < 4; ++layer)
                    {
                        for (int q = 0; q < 4; ++q)
                        {
                            sim->ApplyRx(q, .13 + .17 * ((q + 3 * layer) % 5));
                            sim->ApplyRy(q, .23 + .11 * ((2 * q + layer) % 7));
                        }
                        for (int q = 0; q < 3; ++q)
                            sim->ApplyCX(q, q + 1);
                    }
                    const auto values = sim->ExpectationValues({"ZIII", "IXII", "IIYZ", "XYZI", "ZZZZ"});
                    if (backend == SimulatorType::kQCSim)
                        reference = values;
                    else
                    {
                        Check(values.size() == reference.size(), "cadence parity batch size");
                        for (size_t i = 0; i < values.size(); ++i)
                            Near(values[i], reference[i], 1e-10);
                    }
                }
            }
        }
    }
    // Native keys and typed values can coexist in C++ callers; typed values win.
    for (auto backend : {SimulatorType::kQCSim, SimulatorType::kGpuSim})
    {
        for (int scenario = 0; scenario < 4; ++scenario)
        {
            Owner owner;
            SimulatorConfig config;
            config.simulator_type = backend;
            config.simulation_type = SimulationType::kPauliPropagator;
            config.fixed_backend = true;
            config.pp_coefficient_threshold = .05;
            config.native_options["pauli_propagator_num_gates_between_deduplications"] = scenario == 3 ? "2147483647" : "7";
            if (backend == SimulatorType::kGpuSim)
                config.native_options["pauli_propagator_memory_limit_bytes"] = "16777216";
            if (scenario == 0)
                config.pp_gates_between_deduplications = 3;
            if (scenario == 1 || scenario == 2)
                config.native_options["pauli_propagator_steps_between_trims"] = "17";
            if (scenario == 2)
                config.pp_gates_between_trims = 5;
            auto network = ConfigureNetwork(owner.handle, config);
            Check(network && network->GetSimulator(), "mixed configuration simulator");
            Check(network->GetSimulator()->GetConfiguration("pauli_propagator_steps_between_trims") == (scenario == 0   ? "3"
                                                                                                        : scenario == 1 ? "17"
                                                                                                        : scenario == 2 ? "5"
                                                                                                                        : "2147483647"),
                  "typed/native precedence and disabled cadence are preserved");
        }
    }
}

} // namespace

int main()
{
    try
    {
        CheckLoadedMaestro();
        if (!SimulatorsFactory::GetGpuLibrary())
            return 77;
        FailedOperationsPreserveState();
        RequestDefaultsAndCapabilities();
        ThresholdCadenceParity();
        FailureAndPrecision();
        ConfigurationAndReference();
        Sampling();
        QubitValidation();
        SamplingBudgetRequests();
        RequestOptions();
        std::cout << "GPU Pauli public adapter, QCSim parity, sampling and configuration tests passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
