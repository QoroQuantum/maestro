#include "Circuit/Factory.h"
#include "Execution/SimulatorConfig.h"
#include "Execution/Transforms/ParityReductionTransform.h"
#include "Execution/Transforms/TransformPipeline.h"
#include "Simulators/Core/Factory.h"

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

void CheckCliffordParityReduction(bool dual, bool inverseCheck, bool inverseMixer)
{
    auto source = std::make_shared<Circuit>();
    if (!dual)
        for (size_t q = 0; q < 3; ++q)
            source->AddOperation(Factory::CreateGate(Gate::kHadamardGateType, q));

    const auto positiveCheck = dual ? Gate::kSxGateType : Gate::kSGateType;
    const auto negativeCheck = dual ? Gate::kSxDagGateType : Gate::kSdgGateType;
    const auto positiveMixer = dual ? Gate::kSGateType : Gate::kSxGateType;
    const auto negativeMixer = dual ? Gate::kSdgGateType : Gate::kSxDagGateType;
    const auto cnot = Factory::CreateGate(Gate::kCXGateType, dual ? 1 : 0, dual ? 0 : 1);
    // Choose independent checks {0} and {0,1}. The {1} check and the mixer
    // on physical wire 0 then each act on BOTH logical wires.
    source->AddOperation(Factory::CreateGate(positiveCheck, 0));
    source->AddOperation(cnot);
    source->AddOperation(Factory::CreateGate(negativeCheck, 1));
    source->AddOperation(cnot);
    source->AddOperation(Factory::CreateGate(inverseCheck ? negativeCheck : positiveCheck, 1));
    source->AddOperation(Factory::CreateGate(inverseMixer ? negativeMixer : positiveMixer, 0));
    source->AddOperation(Factory::CreateGate(Gate::kCZGateType, 0, 1));

    std::vector<std::string> physicalObservables{"IIZ"};
    for (const char a : std::string("IXYZ"))
        for (const char b : std::string("IXYZ"))
            physicalObservables.push_back(std::string{a, b, 'I'});
    auto observables = physicalObservables;
    auto circuit = source;
    SimulatorConfig config;
    config.simulation_type = Simulators::SimulationType::kStabilizer;
    config.fixed_backend = true;
    config.enable_causal_cone_reduction = false;
    TransformContext ctx{circuit, observables, config.simulator_type, config};
    TransformPipeline::Run(ctx);
    Check(ctx.auto_reduced && ctx.qubits_before == 3 && ctx.qubits_after == 2, "Clifford circuit should reduce to two qubits");
    Check(circuit->IsClifford(), "Parity reduction introduced a non-Clifford gate into a Clifford circuit");

    const auto execute = [&](const auto &input, size_t width, Simulators::SimulationType method, const auto &paulis) {
        auto simulator = Simulators::SimulatorsFactory::CreateSimulator(config.simulator_type, method);
        simulator->AllocateQubits(width);
        simulator->Initialize();
        Circuits::OperationState bits;
        input->Execute(simulator, bits);
        return simulator->ExpectationValues(paulis);
    };
    const auto expected = execute(source, 3, Simulators::SimulationType::kStatevector, physicalObservables);
    const auto actual = execute(circuit, 2, config.simulation_type, observables);
    for (size_t i = 0; i < observables.size(); ++i)
        Check(std::abs(actual[i] * ctx.expectation_factors[i] - expected[i]) < 1e-12, "Projected Clifford expectations differ from the full statevector");
}

void CheckNearQuarterTurn()
{
    const double angle = std::nextafter(std::acos(-1.) / 2., 2.);
    auto circuit = std::make_shared<Circuit>();
    circuit->AddOperation(Factory::CreateGate(Gate::kHadamardGateType, 0));
    circuit->AddOperation(Factory::CreateGate(Gate::kHadamardGateType, 1));
    circuit->AddOperation(Factory::CreateGate(Gate::kRzGateType, 0, 0, 0, angle));
    const auto source = circuit;
    std::vector<std::string> observables{"XI"};
    SimulatorConfig config;
    config.enable_causal_cone_reduction = false;
    TransformContext ctx{circuit, observables, config.simulator_type, config};
    TransformPipeline::Run(ctx);
    Check(ctx.auto_reduced && ctx.qubits_after == 1, "Generic phase rotation should reduce to one qubit");
    Check(!circuit->IsClifford() && circuit->GetOperations().size() == 1, "Generic rotation was rounded to a Clifford gate");
    auto rotation = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(circuit->GetOperations().front());
    Check(rotation && rotation->GetGateType() == Gate::kRxGateType && rotation->GetParams() == std::vector<double>{angle}, "Generic rotation angle changed");

    // P's existing stabilizer eligibility allows a small tolerance around
    // pi/2. The transform must keep that original gate for the backend.
    source->ReplaceOperation(2, Factory::CreateGate(Gate::kPhaseGateType, 0, 0, 0, angle));
    circuit = source;
    observables = {"XI"};
    config.simulation_type = Simulators::SimulationType::kStabilizer;
    config.fixed_backend = true;
    TransformPipeline::Run(ctx);
    Check(!ctx.auto_reduced && circuit == source, "Near-Clifford phase gate was rewritten for a stabilizer");
}

int main()
{
    try
    {
        for (const bool dual : {false, true})
            for (const bool inverseCheck : {false, true})
                for (const bool inverseMixer : {false, true})
                    CheckCliffordParityReduction(dual, inverseCheck, inverseMixer);
        CheckNearQuarterTurn();
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
        Check(ctx.applied_transforms == std::vector<std::string>{"ParityReduction"}, "applied_transforms incorrect");
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
        config.auto_reduce = true;
        config.enable_causal_cone_reduction = true;
        circuit = source;
        observables = original_observables;
        TransformPipeline::Run(ctx);
        Check(ctx.auto_reduced, "Composite reduction failed");
        Check(std::find(ctx.applied_transforms.begin(), ctx.applied_transforms.end(), "ParityReduction") != ctx.applied_transforms.end(),
              "ParityReduction missing from applied_transforms in composite run");

        // Out-of-span observable must be projected with factor 0.
        config.enable_causal_cone_reduction = false;
        circuit = source;
        std::string out_of_span_obs(37, 'I');
        out_of_span_obs[0] = 'Z';
        observables = {out_of_span_obs};
        TransformPipeline::Run(ctx);
        Check(ctx.auto_reduced, "Out-of-span observable reduction failed");
        Check(ctx.expectation_factors.size() == 1 && ctx.expectation_factors[0] == 0.0, "Out-of-span factor must be 0.0");

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
