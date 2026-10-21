#include "Circuit/Factory.h"
#include "Execution/Estimation.h"
#include "Execution/Sampling.h"
#include "Execution/SimulatorConfig.h"
#include "Execution/Transforms/BoundedPauli.h"
#include "Execution/Transforms/ObservableClustering.h"
#include "Execution/Transforms/TransformPipeline.h"
#include "Noise/NoiseAdd.h"

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
        Require(ctx.applied_transforms.empty() && ctx.qubits_before == 1, "Pipeline diagnostics accumulated across runs");

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
            nonunitary->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 0, 2));
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

        // Observable clustering: disjoint lightcones split, shared ones merge.
        {
            using Groups = std::vector<std::vector<size_t>>;
            auto chain = std::make_shared<Circuit>();
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 0));
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 0, 1));
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 4));
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 4, 5));
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 1, 2));

            // ZIIIII and IZIIII share qubits {0,1}; IIIIZI sits on {4,5}; identity is separate.
            const std::vector<std::string> obs{"ZIIIII", "IIIIZI", "IZIIII", "IIIIII", "IIIIIZ"};
            const auto groups = MaestroExecution::ClusterObservables(chain, obs);
            Require(groups == Groups({{0, 2}, {1, 4}, {3}}), "Clustering split incorrect");

            // A gate bridging both regions merges everything into one cluster.
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 2, 4));
            chain->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 5, 3));
            const auto merged = MaestroExecution::ClusterObservables(chain, {"ZIIIII", "IIIIZI"});
            Require(merged.size() == 1 && merged[0].size() == 2, "Entangled observables did not form one cluster");

            Require(!MaestroExecution::BenefitsFromObservableClustering(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kPauliPropagator),
                    "GPU Pauli propagation should keep one group");
            Require(MaestroExecution::BenefitsFromObservableClustering(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kPauliPropagator) &&
                        MaestroExecution::BenefitsFromObservableClustering(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kStatevector),
                    "Only GPU Pauli propagation should skip clustering");

            // Bounded packing on a register past 64 qubits. Cones: {70..73}, {72..75},
            // {70,71}, {74,75}, {120}, then an identity.
            auto wide = std::make_shared<Circuit>();
            wide->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 70, 71));
            wide->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 72, 73));
            wide->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 74, 75));
            const auto word = [](std::vector<size_t> zs) {
                std::string pauli(121, 'I');
                for (auto q : zs)
                    pauli[q] = 'Z';
                return pauli;
            };
            const std::vector<std::string> wide_obs{word({70, 72}), word({73, 74}), word({71}), word({75}), word({120}), word({})};
            Require(MaestroExecution::ClusterObservables(wide, wide_obs) == Groups({{0, 1, 2, 3}, {4}, {5}}), "Wide connected components incorrect");
            Require(MaestroExecution::ClusterObservables(wide, wide_obs, 4) == Groups({{0, 2}, {1, 3}, {4}, {5}}), "Budget split incorrect");
            // Cones larger than the budget keep their own group and still absorb contained cones.
            Require(MaestroExecution::ClusterObservables(wide, wide_obs, 1) == Groups({{0, 2}, {1, 3}, {4}, {5}}), "Soft budget incorrect");
            // A budget covering the union reproduces the components; disjoint cones never merge.
            Require(MaestroExecution::ClusterObservables(wide, wide_obs, 6) == Groups({{0, 1, 2, 3}, {4}, {5}}), "Fitting budget split cones");
            Require(MaestroExecution::ClusterObservables(wide, wide_obs, 1000) == Groups({{0, 1, 2, 3}, {4}, {5}}), "Large budget merged disjoint cones");

            // Exercise both observable bit batches (>64 observables) and qubit
            // words (>64 qubits), including repeated/contained and empty cones.
            std::vector<std::string> many;
            for (size_t i = 0; i < 130; ++i)
                many.push_back(word({i % 2 ? size_t{120} : size_t{71}}));
            many.push_back(word({}));
            const auto packed = MaestroExecution::ClusterObservables(wide, many, 2);
            Require(packed.size() == 3 && packed[0].size() == 65 && packed[1].size() == 65 && packed[2] == std::vector<size_t>{130},
                    "Bit-parallel batches lost observable indices");
            Require(MaestroExecution::ClusterObservables(wide, {}).empty(), "Empty requests must have zero groups");

            // Nonunitary circuits and malformed observables fall back to one group.
            auto dynamic = std::make_shared<Circuit>();
            dynamic->AddOperation(Factory::CreateMeasurement({{2, 0}}));
            Require(MaestroExecution::ClusterObservables(dynamic, {"ZII", "IIZ"}).size() == 1, "Nonunitary fallback failed");
            Require(MaestroExecution::ClusterObservables(source, {"ZII", "I?Z"}).size() == 1, "Malformed fallback failed");
        }
        {
            using namespace MaestroExecution;
            SimulatorConfig cfg;
            cfg.auto_reduce = false;
            auto unchanged = std::make_shared<Circuit>();
            unchanged->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 0));
            auto original = unchanged;
            std::vector<std::string> obs{"Z"};
            TransformContext no_op{unchanged, obs, cfg.simulator_type, cfg};
            TransformPipeline::Run(no_op);
            Require(unchanged == original && no_op.applied_transforms.empty(), "No-op cone cloned its circuit");

            SimulatorConfig fixed_gpu;
            fixed_gpu.simulator_type = Simulators::SimulatorType::kGpuSim;
            fixed_gpu.simulation_type = Simulators::SimulationType::kTensorNetwork;
            fixed_gpu.fixed_backend = true;
            const auto padded = PrepareEstimation(original, {"Z"}, fixed_gpu);
            Require(padded.qubits_after == 2 && padded.groups[0].observables == std::vector<std::string>{"ZI"},
                    "Fixed GPU tensor network lost its minimum register width");
            EstimationCache gpu_cache;
            const auto cached_pad = PrepareEstimation(original, {"Z"}, fixed_gpu, &gpu_cache);
            Require(cached_pad.qubits_after == 2 && cached_pad.groups[0].observables == std::vector<std::string>{"ZI"},
                    "Cached fixed GPU tensor network lost its minimum register width");
            auto sampled = std::make_shared<Circuit>(*original);
            sampled->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 3));
            sampled->AddOperation(Factory::CreateMeasurement({{0, 0}, {3, 1}}));
            fixed_gpu.enable_sampling_reduction = true;
            Require(!SampleDisjointCones(sampled, fixed_gpu, 10), "Fixed GPU tensor-network sampling attempted a one-qubit component");
            fixed_gpu.fixed_backend = false;
            Require(!SampleDisjointCones(sampled, fixed_gpu, 10), "Automatic GPU tensor-network sampling attempted a one-qubit component");

            EstimationCache cache;
            auto first = std::make_shared<Circuit>();
            first->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
            first->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 3));
            const auto a = EstimateCircuit(first, {"ZIII"}, cfg, &cache);
            Require(a.values == std::vector<double>{-1} && cache.hits == 0, "Initial cached estimate incorrect");
            auto second = std::make_shared<Circuit>(*first);
            second->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
            const auto b = EstimateCircuit(second, {"ZIII"}, cfg, &cache);
            Require(b.values == std::vector<double>{1} && cache.hits == 1, "Cache reused gate values instead of topology");
            second->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kCXGateType, 3, 0));
            const auto c = EstimateCircuit(second, {"ZIII"}, cfg, &cache);
            Require(std::abs(c.values[0]) < 1e-12 && cache.hits == 1 && c.plan.qubits_after == 2, "Topology change did not invalidate cache");
            second->AddOperation(Factory::CreateReset({3}));
            const auto fallback = PrepareEstimation(second, {"ZIII"}, cfg, &cache);
            Require(fallback.qubits_after == 4 && cache.hits == 1, "Trajectory reset incorrectly reused cache");

            // Backward branching has a work cap and leaves ordinary simulation
            // available when no bounded expansion can be returned.
            auto branching = std::make_shared<Circuit>();
            for (size_t q = 0; q < 8; ++q)
            {
                branching->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kRxGateType, q, 0, 0, 0.7));
                branching->AddOperation(Factory::CreateGate(Circuits::QuantumGateType::kRyGateType, q, 0, 0, 0.4));
            }
            branching->AddOperation(
                std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{0}, Simulators::QuantumChannel::PhaseFlip(0.01)));
            Require(!EvaluateBoundedPauli(branching, "ZZZZZZZZ", 0.001), "Pauli expansion exceeded its work cap");

            // NoiseAdd must use the same preparation path with an already
            // configured network, including exact channels on spectators.
            GetMaestroObjectWithMute();
            const auto handle = CreateSimpleSimulator(1);
            Require(handle != 0, "Cannot create native noise test network");

            struct Cleanup
            {
                unsigned long handle;

                ~Cleanup()
                {
                    DestroySimpleSimulator(handle);
                }
            } cleanup{handle};

            cfg.simulation_type = Simulators::SimulationType::kDensityMatrix;
            cfg.fixed_backend = true;
            auto network = ConfigureNetwork(handle, cfg);
            noise::NoiseModel model;
            model.set_depolarizing(3, 0.2);
            noise::NoiseAdd add;
            add.set_exact_channels(true);
            add.seed(17);
            const auto native = add.noisy_estimate_montecarlo(first, network, 0, {"ZIII"}, model, 3);
            Require(native.size() == 1 && std::abs(native[0] + 1) < 1e-12, "Native NoiseAdd reduction changed the expectation");
            Require(add.noisy_estimate_montecarlo(first, network, 0, {}, model, 3).empty(), "Native empty estimate did work");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
