#include "Execution/CircuitHelpers.h"
#include "core.h"
#include "runtime.h"

namespace maestro_bindings
{

// Apply the ket's preparation first, then the inverse of the bra's preparation.
static CircuitPtr build_overlap_circuit(const CircuitPtr &bra, const CircuitPtr &ket)
{
    auto combined = std::static_pointer_cast<Circuits::Circuit<double>>(ket->Clone());
    combined->SetGlobalPhase(ket->GetGlobalPhase() - bra->GetGlobalPhase());
    const auto &ops = bra->GetOperations();
    for (auto it = ops.rbegin(); it != ops.rend(); ++it)
        combined->AddOperation(MaestroExecution::adjoint_gate(*it));
    return combined;
}

// P(|0...0>) of the mirror circuit. By default uses shot-based sampling. Set
// full_amplitude=true for exact statevector computation (only feasible for
// small qubit counts).
static double mirror_fidelity_projection(const CircuitPtr &mirror, const SimulatorConfig &config, int shots, bool full_amplitude)
{
    // Helper lambda for the shot-based path
    auto run_shot_based = [&]() -> double {
        // Need a fresh mirror circuit since measurements mutate it
        auto mirror_copy = std::make_shared<Circuits::Circuit<double>>();
        for (const auto &op : mirror->GetOperations())
        {
            mirror_copy->AddOperation(op->Clone());
        }

        // Every qubit up to the highest index is measured, so the register
        // holds them all.
        const int num_qubits = RegisterQubits(CircuitWidth(*mirror_copy));
        const size_t n = static_cast<size_t>(num_qubits);
        std::vector<std::pair<Types::qubit_t, size_t>> pairs;
        pairs.reserve(n);
        for (size_t i = 0; i < n; ++i)
            pairs.emplace_back(static_cast<Types::qubit_t>(i), i);
        mirror_copy->AddOperation(std::make_shared<Circuits::MeasurementOperation<>>(pairs));

        // Build network directly (like execute_core) but with circuit optimization
        // disabled. The mirror circuit's paired gate/adjoint sequences must not be
        // cancelled by the optimizer: e.g. ry(-θ) + ry(θ) → ry(0), followed by
        // s + ry(0) + sdg, which the optimizer would incorrectly simplify further.
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
            const int num_qubits = HandleQubits(*mirror);
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

// Core Mirror Fidelity Logic
// Builds circuit + adjoint(circuit) in reverse and returns P(|0...0>). With
// observables, returns <0|M^dag O M|0> for each Pauli O instead, which is 1 for
// Z-type strings under exact evolution and degrades with truncation.
nb::object mirror_fidelity_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots, bool full_amplitude,
                                const std::optional<nb::object> &observables)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");
    const auto preparation = MaestroExecution::UnitaryPreparation(circuit, "mirror_fidelity");
    const auto mirror = build_overlap_circuit(preparation, preparation);
    if (!observables || observables->is_none())
        return nb::float_(mirror_fidelity_projection(mirror, config, shots, full_amplitude));

    const auto paulis = ParseObservables(*observables);
    if (paulis.empty())
        throw nb::value_error("observables must contain at least one Pauli string.");
    const bool single = nb::isinstance<nb::str>(*observables) && paulis.size() == 1 && nb::cast<std::string>(*observables).find(';') == std::string::npos;

    // Preserve the gate sequence: even exact estimator reductions can hide
    // truncation error by changing which operations the backend executes.
    SimulatorConfig mirror_config = config;
    mirror_config.optimize_circuit = false;
    mirror_config.native_options.erase("optimize_circuit");
    mirror_config.auto_reduce = false;
    mirror_config.enable_causal_cone_reduction = false;
    MaestroExecution::EstimationResult result;
    {
        nb::gil_scoped_release release;
        result = MaestroExecution::EstimateCircuit(mirror, paulis, mirror_config, nullptr);
    }
    if (single)
        return nb::float_(result.values.at(0));
    return nb::cast(result.values);
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

    const auto bra = MaestroExecution::UnitaryPreparation(circuit_1, "inner_product circuit_1");
    const auto ket = MaestroExecution::UnitaryPreparation(circuit_2, "inner_product circuit_2");
    const auto combined = build_overlap_circuit(bra, ket);

    const int num_qubits = HandleQubits(*combined);
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

} // namespace maestro_bindings
