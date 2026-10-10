#include "core.h"
#include "runtime.h"

namespace maestro_bindings
{

// Helper: Create the adjoint (inverse) of a single quantum gate operation.
// Non-gate operations (measurements, resets, etc.) return nullptr and are
// skipped when building the mirror circuit.
using OperationPtr = std::shared_ptr<Circuits::IOperation<double>>;

OperationPtr adjoint_gate(const OperationPtr &op)
{
    if (op->GetType() != Circuits::OperationType::kGate)
        return nullptr;

    auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
    if (!gate)
        return nullptr;

    const auto gt = gate->GetGateType();
    const auto params = gate->GetParams();

    switch (gt)
    {
    // ---- Self-inverse (Hermitian) gates ----
    case Circuits::QuantumGateType::kXGateType:
    case Circuits::QuantumGateType::kYGateType:
    case Circuits::QuantumGateType::kZGateType:
    case Circuits::QuantumGateType::kHadamardGateType:
    case Circuits::QuantumGateType::kKGateType:
    case Circuits::QuantumGateType::kCXGateType:
    case Circuits::QuantumGateType::kCYGateType:
    case Circuits::QuantumGateType::kCZGateType:
    case Circuits::QuantumGateType::kCHGateType:
    case Circuits::QuantumGateType::kSwapGateType:
    case Circuits::QuantumGateType::kCCXGateType:
    case Circuits::QuantumGateType::kCSwapGateType:
        return op->Clone();

    // ---- Paired gates ----
    case Circuits::QuantumGateType::kSGateType:
        return std::make_shared<Circuits::SdgGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSdgGateType:
        return std::make_shared<Circuits::SGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kTGateType:
        return std::make_shared<Circuits::TdgGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kTdgGateType:
        return std::make_shared<Circuits::TGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSxGateType:
        return std::make_shared<Circuits::SxDagGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSxDagGateType:
        return std::make_shared<Circuits::SxGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kCSxGateType:
        return std::make_shared<Circuits::CSxDagGate<>>(gate->GetQubit(0), gate->GetQubit(1));
    case Circuits::QuantumGateType::kCSxDagGateType:
        return std::make_shared<Circuits::CSxGate<>>(gate->GetQubit(0), gate->GetQubit(1));

    // ---- Parametric single-qubit: negate angle ----
    case Circuits::QuantumGateType::kPhaseGateType:
        return std::make_shared<Circuits::PhaseGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRxGateType:
        return std::make_shared<Circuits::RxGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRyGateType:
        return std::make_shared<Circuits::RyGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRzGateType:
        return std::make_shared<Circuits::RzGate<>>(gate->GetQubit(), -params[0]);

    // ---- U gate: U†(θ,φ,λ,γ) = U(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kUGateType:
        return std::make_shared<Circuits::UGate<>>(gate->GetQubit(), -params[0], -params[2], -params[1], -params[3]);

    // ---- Controlled parametric: negate angle ----
    case Circuits::QuantumGateType::kCPGateType:
        return std::make_shared<Circuits::CPGate<>>(gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRxGateType:
        return std::make_shared<Circuits::CRxGate<>>(gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRyGateType:
        return std::make_shared<Circuits::CRyGate<>>(gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRzGateType:
        return std::make_shared<Circuits::CRzGate<>>(gate->GetQubit(0), gate->GetQubit(1), -params[0]);

    // ---- CU gate: CU†(θ,φ,λ,γ) = CU(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kCUGateType:
        return std::make_shared<Circuits::CUGate<>>(gate->GetQubit(0), gate->GetQubit(1), -params[0], -params[2], -params[1], -params[3]);

    default:
        return op->Clone(); // Fallback: clone as-is
    }
}

// Core Mirror Fidelity Logic
// Builds circuit + adjoint(circuit) in reverse, returns P(|0...0>).
// By default uses shot-based sampling. Set full_amplitude=true for exact
// statevector computation (only feasible for small qubit counts).
double mirror_fidelity_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots, bool full_amplitude)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");

    // Build the mirror circuit: forward gates + adjoint gates in reverse
    auto mirror = std::make_shared<Circuits::Circuit<double>>();
    const auto &ops = circuit->GetOperations();

    // Forward pass: add only gate operations (skip measurements)
    for (const auto &op : ops)
    {
        if (op->GetType() == Circuits::OperationType::kGate)
        {
            mirror->AddOperation(op->Clone());
        }
    }

    // Reverse pass: iterate backward and add adjoint of each gate operation only
    // (skip measurements and other non-gate ops — they have no adjoint)
    for (auto it = ops.rbegin(); it != ops.rend(); ++it)
    {
        if ((*it)->GetType() != Circuits::OperationType::kGate)
            continue;
        auto adj = adjoint_gate(*it);
        if (adj)
            mirror->AddOperation(adj);
    }

    // Helper lambda for the shot-based path
    auto run_shot_based = [&]() -> double {
        // Need a fresh mirror circuit since measurements mutate it
        auto mirror_copy = std::make_shared<Circuits::Circuit<double>>();
        for (const auto &op : mirror->GetOperations())
        {
            mirror_copy->AddOperation(op->Clone());
        }

        size_t n = std::max(1, static_cast<int>(mirror_copy->GetMaxQubitIndex()) + 1);
        std::vector<std::pair<Types::qubit_t, size_t>> pairs;
        pairs.reserve(n);
        for (size_t i = 0; i < n; ++i)
            pairs.emplace_back(static_cast<Types::qubit_t>(i), i);
        mirror_copy->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(pairs));

        // Build network directly (like execute_core) but with circuit optimization
        // disabled. The mirror circuit's paired gate/adjoint sequences must not be
        // cancelled by the optimizer: e.g. ry(-θ) + ry(θ) → ry(0), followed by
        // s + ry(0) + sdg, which the optimizer would incorrectly simplify further.
        int num_qubits = std::max(1, static_cast<int>(mirror_copy->GetMaxQubitIndex()) + 1);
        ScopedSimulator sim(num_qubits);
        if (sim.handle == 0)
            throw std::runtime_error("mirror_fidelity: failed to create simulator.");
        auto network = ConfigureNetwork(sim.handle, config);
        if (!network)
            throw std::runtime_error("mirror_fidelity: failed to configure network.");
        // Disable circuit optimization: the mirror's gate/adjoint pairs must not
        // be cancelled or merged by the optimizer.
        network->GetController()->SetOptimizeCircuit(false);
        // Disable MPS swap optimization: MPSDummySimulator used for swap-cost
        // estimation throws on multi-qubit measurement operations.
        network->SetInitialQubitsMapOptimization(false);
        network->SetMPSOptimizeSwaps(false);

        Network::INetwork<double>::ExecuteResults raw_results;
        {
            nb::gil_scoped_release release;
            raw_results = network->RepeatedExecuteOnHost(mirror_copy, 0, (size_t)shots);
        }

        if (raw_results.empty() && shots > 0)
        {
            throw std::runtime_error("mirror_fidelity: Simulation failed to return measurement samples.");
        }

        // Convert results to counts dict and look up all-zeros bitstring
        std::string zeros(n, '0');
        size_t zero_count = 0;
        size_t total_shots = 0;
        for (const auto &pair : raw_results)
        {
            total_shots += pair.second;
            const auto &bool_vec = pair.first;
            std::string bitstring(bool_vec.size(), '0');
            for (size_t i = 0; i < bool_vec.size(); ++i)
                if (bool_vec[i])
                    bitstring[i] = '1';
            if (bitstring == zeros)
                zero_count += pair.second;
        }
        if (total_shots == 0 && shots > 0)
        {
            throw std::runtime_error("mirror_fidelity: Simulation produced 0 total measurement shots.");
        }
        return static_cast<double>(zero_count) / static_cast<double>(shots);
    };

    if (full_amplitude)
    {
        // Try exact statevector with circuit optimization disabled
        try
        {
            int num_qubits = std::max(1, static_cast<int>(mirror->GetMaxQubitIndex()) + 1);
            ScopedSimulator sim(num_qubits);
            if (sim.handle != 0)
            {
                auto network = ConfigureNetwork(sim.handle, config);
                if (network)
                {
                    network->GetController()->SetOptimizeCircuit(false);
                    network->SetInitialQubitsMapOptimization(false);
                    network->SetMPSOptimizeSwaps(false);
                    std::vector<std::complex<double>> amplitudes;
                    {
                        nb::gil_scoped_release release;
                        amplitudes = network->ExecuteOnHostAmplitudes(mirror, 0);
                    }
                    if (!amplitudes.empty())
                        return std::norm(amplitudes[0]);
                }
            }
        }
        catch (...)
        {
            // Statevector not available for this backend — fall back to shots
        }
        // Issue a Python warning so the user knows we fell back
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "full_amplitude mode not supported by this simulator/simulation "
                     "type. Falling back to shot-based sampling.",
                     1);
        return run_shot_based();
    }
    else
    {
        return run_shot_based();
    }
}

// Core Inner Product Logic
// Computes <psi_1|psi_2> = <0|U1† U2|0> via ProjectOnZero.
std::complex<double> inner_product_core(const std::shared_ptr<Circuits::Circuit<double>> &circuit_1,
                                        const std::shared_ptr<Circuits::Circuit<double>> &circuit_2, const SimulatorConfig &config)
{
    if (!circuit_1)
        throw nb::value_error("circuit_1 is null.");
    if (!circuit_2)
        throw nb::value_error("circuit_2 is null.");

    // Build combined circuit for <0| U1† U2 |0>.
    // Circuit gates are applied left-to-right, so we place U2's gates first
    // (they act on |0> first), then U1†'s gates (applied last = leftmost in
    // the matrix product).
    auto combined = std::make_shared<Circuits::Circuit<double>>();
    const auto &ops1 = circuit_1->GetOperations();
    const auto &ops2 = circuit_2->GetOperations();

    // Forward pass of circuit_2: gate operations only
    for (const auto &op : ops2)
    {
        if (op->GetType() == Circuits::OperationType::kGate)
        {
            combined->AddOperation(op->Clone());
        }
    }

    // Adjoint of circuit_1: reverse order, each gate adjointed
    for (auto it = ops1.rbegin(); it != ops1.rend(); ++it)
    {
        auto adj = adjoint_gate(*it);
        if (adj)
            combined->AddOperation(adj);
    }

    int num_qubits = std::max(1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
    ScopedSimulator sim(num_qubits);
    if (sim.handle == 0)
        throw std::runtime_error("Failed to create simulator handle.");

    auto network = ConfigureNetwork(sim.handle, config);
    if (!network)
        throw std::runtime_error("Failed to configure network.");

    std::complex<double> result;
    {
        nb::gil_scoped_release release;
        result = network->ExecuteOnHostProjectOnZero(combined, 0);
    }
    return result;
}

// Fidelity to the ideal unitary circuit's pure state. Unlike inner_product,
// retain the noisy forward resets/channels and read a probability, which is
// supported by both pure-state and density-matrix/MPO backends.
double noisy_fidelity_core(const std::shared_ptr<Circuits::Circuit<double>> &ideal, const std::shared_ptr<Circuits::Circuit<double>> &noisy,
                           const SimulatorConfig &config)
{
    auto combined = std::make_shared<Circuits::Circuit<double>>();
    // Fidelity is evaluated before terminal readout, as in inner_product.
    for (const auto &op : noisy->GetOperations())
    {
        if (op->GetType() != Circuits::OperationType::kMeasurement)
            combined->AddOperation(op->Clone());
    }
    const auto &ideal_ops = ideal->GetOperations();
    for (auto it = ideal_ops.rbegin(); it != ideal_ops.rend(); ++it)
    {
        auto adj = adjoint_gate(*it);
        if (adj)
            combined->AddOperation(adj);
    }

    const int num_qubits = std::max(1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
    ScopedSimulator sim(num_qubits);
    if (sim.handle == 0)
        throw std::runtime_error("Failed to create simulator handle.");
    auto network = ConfigureNetwork(sim.handle, config);
    if (!network)
        throw std::runtime_error("Failed to configure network.");
    network->CreateSimulator(config.simulator_type, config.simulation_type);
    auto simulator = network->GetSimulator();
    if (!simulator)
        throw std::runtime_error("noisy_fidelity: requested backend is unavailable.");
    if (config.seed)
        simulator->SetSeed(*config.seed);
    Circuits::OperationState state(num_qubits);
    nb::gil_scoped_release release;
    combined->ExecuteBD(simulator, state);
    return simulator->Probability(0);
}

} // namespace maestro_bindings
