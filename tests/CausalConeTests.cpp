#include "Circuit/Factory.h"
#include "Execution/SimulatorConfig.h"
#include "Execution/Transforms/TransformPipeline.h"

#include <iostream>
#include <stdexcept>

using Circuit = Circuits::Circuit<double>;
using Factory = Circuits::CircuitFactory<double>;
using Type = Simulators::SimulatorType;

void Require(bool ok, const char *message)
{
    if (!ok)
        throw std::runtime_error(message);
}

int main()
{
    try
    {
        auto source = std::make_shared<Circuit>();
        source->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
        source->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 2));
        // A two-shard layout with global qubit 2 is valid at width three.
        // Reducing either support to one qubit would invalidate the shard count;
        // remapping qubit 2 to zero would also invalidate the explicit layout.
        // Identity-only requests must retain the original width as well.
        for (auto backend : {Type::kDistGpuSim, Type::kDistMpiGpuSim})
            for (const auto &observable : {"ZII", "IIZ", "III"})
            {
                auto circuit = source;
                std::vector<std::string> paulis{observable};
                MaestroExecution::SimulatorConfig config;
                MaestroExecution::TransformContext ctx{circuit, paulis, backend, config};
                MaestroExecution::TransformPipeline::Run(ctx);
                Require(ctx.qubits_before == 3 && ctx.qubits_after == 3 && ctx.applied_transforms.empty(), "Distributed diagnostics changed");
                Require(circuit == source, "Distributed reduction replaced the source circuit");
                Require(circuit->GetMaxQubitIndex() == 2, "Distributed reduction changed register width");
                Require(paulis == std::vector<std::string>{observable}, "Distributed reduction remapped the observable");
            }

        // The fallback must not disable reduction for ordinary CPU/GPU backends.
        for (auto backend : {Type::kQCSim, Type::kGpuSim})
        {
            auto circuit = source;
            std::vector<std::string> paulis{"IIZ"};
            MaestroExecution::SimulatorConfig config;
            MaestroExecution::TransformContext ctx{circuit, paulis, backend, config};
            MaestroExecution::TransformPipeline::Run(ctx);
            Require(ctx.qubits_before == 3 && ctx.qubits_after == 1, "Local diagnostics did not track register width");
            Require(ctx.applied_transforms == std::vector<std::string>{"CausalCone"}, "Pipeline did not record the pass");
            Require(circuit != source, "Local reduction did not create a reduced circuit");
            Require(circuit->GetMaxQubitIndex() == 0, "Local reduction did not compact qubits");
            Require(circuit->GetOperations().size() == 1, "Local reduction retained spectator gates");
            Require(paulis == std::vector<std::string>{"Z"}, "Local reduction did not remap the observable");
            Require(source->GetMaxQubitIndex() == 2 && source->GetOperations().size() == 2, "Reduction mutated the source");
        }

        auto circuit = source;
        std::vector<std::string> paulis{"IIIIIZ"};
        MaestroExecution::SimulatorConfig config;
        config.enable_causal_cone_reduction = false;
        MaestroExecution::TransformContext ctx{circuit, paulis, Type::kQCSim, config};
        MaestroExecution::TransformPipeline::Run(ctx);
        Require(circuit == source && paulis == std::vector<std::string>{"IIIIIZ"}, "Disabled pipeline changed inputs");
        Require(ctx.qubits_before == 6 && ctx.qubits_after == 6 && ctx.applied_transforms.empty(), "Disabled diagnostics ignored observables");

        config.enable_causal_cone_reduction = true;
        MaestroExecution::TransformPipeline::Run(ctx);
        Require(circuit->GetOperations().empty() && paulis == std::vector<std::string>{"Z"}, "Idle observable reduction failed");
        Require(ctx.qubits_before == 6 && ctx.qubits_after == 1, "Idle observable width is incorrect");
        MaestroExecution::TransformPipeline::Run(ctx);
        Require(ctx.applied_transforms.size() == 1 && ctx.qubits_before == 1, "Pipeline diagnostics accumulated across runs");

        circuit = source;
        paulis = {"III"};
        MaestroExecution::TransformPipeline::Run(ctx);
        Require(circuit->GetOperations().empty() && paulis == std::vector<std::string>{"I"} && ctx.qubits_after == 1,
                "Identity reduction did not retain the minimum register width");

        circuit = source;
        paulis = {"I?I"};
        MaestroExecution::TransformPipeline::Run(ctx);
        Require(circuit == source && paulis == std::vector<std::string>{"I?I"}, "Malformed observable did not fall back");

        for (const auto &operation : {Factory::CreateReset({2}), Factory::CreateMeasurement({{2, 0}})})
        {
            auto nonunitary = std::make_shared<Circuit>();
            nonunitary->AddOperation(operation);
            circuit = nonunitary;
            paulis = {"ZII"};
            MaestroExecution::TransformPipeline::Run(ctx);
            Require(circuit == nonunitary && paulis == std::vector<std::string>{"ZII"}, "Nonunitary operation did not fall back");
            Require(ctx.qubits_before == 3 && ctx.qubits_after == 3, "Fallback diagnostics changed register width");
        }

        circuit.reset();
        bool rejected_null = false;
        try
        {
            MaestroExecution::TransformPipeline::Run(ctx);
        }
        catch (const std::invalid_argument &)
        {
            rejected_null = true;
        }
        Require(rejected_null, "Pipeline accepted a null circuit");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
