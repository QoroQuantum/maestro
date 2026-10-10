#include "Circuit/Factory.h"
#include "python/bindings/causal_cone.h"

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
                maestro_bindings::ReduceCausalCone(circuit, paulis, backend);
                Require(circuit == source, "Distributed reduction replaced the source circuit");
                Require(circuit->GetMaxQubitIndex() == 2, "Distributed reduction changed register width");
                Require(paulis == std::vector<std::string>{observable}, "Distributed reduction remapped the observable");
            }

        // The fallback must not disable reduction for ordinary CPU/GPU backends.
        for (auto backend : {Type::kQCSim, Type::kGpuSim})
        {
            auto circuit = source;
            std::vector<std::string> paulis{"IIZ"};
            maestro_bindings::ReduceCausalCone(circuit, paulis, backend);
            Require(circuit != source, "Local reduction did not create a reduced circuit");
            Require(circuit->GetMaxQubitIndex() == 0, "Local reduction did not compact qubits");
            Require(circuit->GetOperations().size() == 1, "Local reduction retained spectator gates");
            Require(paulis == std::vector<std::string>{"Z"}, "Local reduction did not remap the observable");
            Require(source->GetMaxQubitIndex() == 2 && source->GetOperations().size() == 2, "Reduction mutated the source");
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
