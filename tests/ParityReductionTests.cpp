#include "Circuit/Factory.h"
#include "Execution/SimulatorConfig.h"
#include "Execution/Transforms/ParityReductionTransform.h"
#include "Execution/Transforms/TransformPipeline.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

using Circuit = Circuits::Circuit<double>;
using Factory = Circuits::CircuitFactory<double>;
using Gate = Circuits::QuantumGateType;
using namespace MaestroExecution;

void Check(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

int main()
{
    try
    {
        auto source = std::make_shared<Circuit>();
        for (size_t q = 0; q < 37; ++q)
            source->AddOperation(Factory::CreateGate(Gate::kHadamardGateType, q));
        std::vector<std::string> original_observables;
        for (size_t q = 0; q < 18; ++q)
        {
            source->AddOperation(Factory::CreateGate(Gate::kCXGateType, 2 * q, 36));
            source->AddOperation(Factory::CreateGate(Gate::kCXGateType, 2 * q + 1, 36));
            source->AddOperation(Factory::CreateGate(Gate::kRzGateType, 36, 0, 0, 0.37));
            source->AddOperation(Factory::CreateGate(Gate::kCXGateType, 2 * q + 1, 36));
            source->AddOperation(Factory::CreateGate(Gate::kCXGateType, 2 * q, 36));
            std::string observable(37, 'I');
            observable[2 * q] = observable[2 * q + 1] = observable[36] = 'Z';
            original_observables.push_back(observable);
        }
        for (size_t q = 0; q < 37; ++q)
            source->AddOperation(Factory::CreateGate(Gate::kRxGateType, q, 0, 0, -0.25));
        SimulatorConfig config;
        config.auto_reduce = true;
        config.enable_causal_cone_reduction = false;
        auto circuit = source;
        auto observables = original_observables;
        TransformContext ctx{circuit, observables, config.simulator_type, config};
        const auto start = std::chrono::steady_clock::now();
        TransformPipeline::Run(ctx);
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        Check(ctx.auto_reduced && ctx.qubits_before == 37 && ctx.qubits_after == 18, "37-to-18 reduction failed");
        Check(source != circuit && source->GetMaxQubitIndex() == 36, "Source was mutated");
        Check(ctx.expectation_factors == std::vector<double>(18, 1.0), "Check factors incorrect");
        std::cout << "37-to-18 preflight (including circuit emission): " << elapsed << " ms\n";

        for (auto backend : {Simulators::SimulatorType::kDistGpuSim, Simulators::SimulatorType::kDistMpiGpuSim})
        {
            circuit = source;
            observables = original_observables;
            ctx.simulator_type = backend;
            TransformPipeline::Run(ctx);
            Check(!ctx.auto_reduced && circuit == source, "Distributed layout was changed");
        }
        ctx.simulator_type = config.simulator_type;
        for (const auto &operation : {Factory::CreateReset({0}), Factory::CreateMeasurement({{0, 0}}),
                                      std::shared_ptr<Circuits::IOperation<double>>(std::make_shared<Circuits::QuantumChannelOperation<double>>(
                                          Types::qubits_vector{0}, Simulators::QuantumChannel({Eigen::Matrix2cd::Identity()})))})
        {
            auto noisy = std::make_shared<Circuit>(*source);
            noisy->AddOperation(operation);
            circuit = noisy;
            observables = original_observables;
            TransformPipeline::Run(ctx);
            Check(!ctx.auto_reduced && circuit == noisy && observables == original_observables, "Nonunitary fallback changed input");
        }
        circuit = source;
        observables = {"bad observable"};
        TransformPipeline::Run(ctx);
        Check(!ctx.auto_reduced && circuit == source, "Invalid observable was reduced");

        // Resource guards must return before any exponential allocation.
        for (size_t width : {size_t{24}, size_t{65}})
        {
            auto oversized = std::make_shared<Circuit>();
            for (size_t q = 0; q < width; ++q)
                oversized->AddOperation(Factory::CreateGate(Gate::kHadamardGateType, q));
            for (size_t q = 0; q < width - 1; ++q)
                oversized->AddOperation(Factory::CreateGate(Gate::kRzGateType, q, 0, 0, 0.4));
            circuit = oversized;
            observables = {std::string(width, 'Z')};
            TransformPipeline::Run(ctx);
            Check(!ctx.auto_reduced && circuit == oversized, "Resource cap failed");
        }
        config.auto_reduce = false;
        circuit = source;
        observables = original_observables;
        TransformPipeline::Run(ctx);
        Check(!ctx.auto_reduced && circuit == source && ctx.applied_transforms.empty(), "Disabled reduction changed circuit");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
