// Versioned native computational API. No Python types or runtime are used.
#include "Request.h"

#include "AdjointDifferentiation.h"
#include "Capabilities.h"
#include "CircuitHelpers.h"
#include "Estimation.h"
#include "ReferenceFrameSampler.h"
#include "SampleRecords.h"
#include "Sampling.h"
#include "TerminalSampling.h"
#include "maestrolib/Interface.h"
#include "maestrolib/Json.h"
#include "qasm/QasmCirc.h"
#include "Json/NoiseJson.h"
#include <chrono>
#include <climits>
#include <complex>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <random>

namespace MaestroExecution
{
using Circuit = Circuits::Circuit<double>;
using CircuitPtr = std::shared_ptr<Circuit>;
using Clock = std::chrono::steady_clock;

namespace
{
const std::set<std::string> operations{"execute",
                                       "estimate",
                                       "statevector",
                                       "probabilities",
                                       "amplitudes",
                                       "state_probability",
                                       "mirror_fidelity",
                                       "inner_product",
                                       "noisy_fidelity",
                                       "diagnostics",
                                       "incremental_evolve",
                                       "checkpoint_batch",
                                       "batch",
                                       "validate",
                                       "adjoint_expectation_values",
                                       "adjoint_jacobian",
                                       "adjoint_vjp",
                                       "density_matrix_overlap",
                                       "fidelity_with_statevector"};

bool AdjointOperation(const std::string &operation)
{
    return operation == "adjoint_expectation_values" || operation == "adjoint_jacobian" || operation == "adjoint_vjp";
}

// Mixed-state counterparts of inner_product and fidelity, which read the
// prepared density matrix or MPO directly (noisy_fidelity applies to mixed
// states itself: it reads P(0...0) after the ideal circuit's adjoint).
bool MixedStateQuery(const std::string &operation)
{
    return operation == "density_matrix_overlap" || operation == "fidelity_with_statevector";
}

// Per-measurement readout rates: a symmetric flip probability, or the two
// asymmetric rates named like the readout noise channel.
Circuits::ReadoutRates ParseReadoutRates(const json::value &value)
{
    const auto probability = [](double p, const char *name) {
        Require(p >= 0 && p <= 1, std::string(name) + " must be in [0, 1]");
        return p;
    };
    if (value.is_number())
    {
        const double p = probability(Number(value), "readout_error");
        return {p, p};
    }
    const auto &object = Object(value);
    Keys(object, {"p_meas1_prep0", "p_meas0_prep1"});
    return {probability(Number(Field(object, "p_meas1_prep0")), "p_meas1_prep0"), probability(Number(Field(object, "p_meas0_prep1")), "p_meas0_prep1")};
}

// qelib1.inc's two-qubit rotations have no gate type; expand them exactly as
// the QASM parser does: cx; rz; cx (between Hadamards for rxx).
json::array ExpandInstruction(const json::object &instruction)
{
    const auto name = String(Field(instruction, "name"));
    if (name != "rzz" && name != "rxx")
        return {instruction};
    const auto &qubits = Array(Field(instruction, "qubits"));
    Require(qubits.size() == 2, "Gate " + name + " requires exactly 2 qubits");
    const auto gate = [&](const char *gateName, json::array targets, const json::value *params = nullptr) {
        json::object expanded{{"name", gateName}, {"qubits", std::move(targets)}};
        if (params)
            expanded["params"] = *params;
        if (const auto *condition = instruction.if_contains("conditional_reg"))
            expanded["conditional_reg"] = *condition;
        return expanded;
    };
    json::array result;
    const bool xBasis = name == "rxx";
    if (xBasis)
    {
        result.emplace_back(gate("h", {qubits[0]}));
        result.emplace_back(gate("h", {qubits[1]}));
    }
    result.emplace_back(gate("cx", {qubits[0], qubits[1]}));
    result.emplace_back(gate("rz", {qubits[1]}, &instruction.at("params")));
    result.emplace_back(gate("cx", {qubits[0], qubits[1]}));
    if (xBasis)
    {
        result.emplace_back(gate("h", {qubits[0]}));
        result.emplace_back(gate("h", {qubits[1]}));
    }
    return result;
}

struct ParsedCircuit
{
    CircuitPtr circuit;
    size_t qubits, clbits;
};

ParsedCircuit ParseCircuit(const json::object &object)
{
    Keys(object, {"format", "source", "num_qubits", "num_clbits", "parameters"});
    const size_t qubits = UInt(Field(object, "num_qubits"));
    const size_t clbits = UInt(object, "num_clbits", qubits);
    Require(qubits > 0 && qubits <= 1000000, "num_qubits must be in [1,1000000]");
    Require(clbits <= 1000000, "num_clbits exceeds the result limit");
    const auto format = String(object, "format", "openqasm");
    CircuitPtr circuit;
    if (format == "openqasm")
    {
        const auto source = String(Field(object, "source"));
        Require(!source.empty(), "Circuit source is empty");
        std::unordered_map<std::string, double> params;
        for (const auto &entry : Sub(object, "parameters"))
            params.emplace(std::string(entry.key()), Number(entry.value()));
        qasm::QasmToCirc<> parser;
        circuit = parser.ParseAndTranslateWithParams(source, params);
        Require(circuit && !parser.Failed(), "QASM parse error: " + parser.GetErrorMessage());
    }
    else if (format == "instructions")
    {
        Require(!object.contains("parameters"), "Instruction arrays do not accept parameter bindings");
        const auto &array = Array(Field(object, "source"));
        circuit = std::make_shared<Circuit>();
        Json::JsonParserMaestro<> parser;
        for (const auto &value : array)
        {
            auto instruction = Object(value);
            const auto name = String(Field(instruction, "name"));
            Keys(instruction,
                 {"name", "qubits", "clbits", "params", "conditional_reg", "duration", "operators", "probabilities", "noiseless", "readout_error"});
            Types::qubits_vector targets;
            std::set<uint64_t> unique;
            for (const auto &q : Array(Field(instruction, "qubits")))
            {
                Require(UInt(q) < qubits && unique.insert(UInt(q)).second, "Invalid instruction qubit");
                targets.push_back(UInt(q));
            }
            Require(!targets.empty(), "Instruction requires qubits");
            Require(name != "measure" || !instruction.contains("noiseless"), "noiseless applies only to gates");
            Require(name == "measure" || !instruction.contains("readout_error"), "readout_error applies only to measurement");
            Require(name == "pauli_channel" || !instruction.contains("probabilities"), "probabilities applies only to Pauli channel instructions");
            if (name == "reset" || name == "delay" || name == "kraus" || name == "pauli_channel")
            {
                Require(!instruction.contains("conditional_reg") && !instruction.contains("params") && !instruction.contains("clbits") &&
                            !instruction.contains("noiseless"),
                        "Unsupported fields on native channel/reset/delay");
                Require(name == "delay" || !instruction.contains("duration"), "duration applies only to delay instructions");
                Require(name == "kraus" || !instruction.contains("operators"), "operators applies only to Kraus instructions");
                if (name == "reset")
                    circuit->AddOperation(Circuits::CircuitFactory<>::CreateReset(targets));
                else if (name == "delay")
                {
                    const auto duration = Number(Field(instruction, "duration"));
                    Require(duration >= 0, "Negative delay");
                    for (auto q : targets)
                        circuit->AddOperation(Circuits::CircuitFactory<>::CreateDelay(q, duration));
                }
                else if (name == "pauli_channel")
                {
                    // Base-4 digit k of an index selects I, X, Y or Z on qubits[k].
                    Require(targets.size() <= 2, "Pauli channel instructions require one or two qubits");
                    std::vector<double> probabilities;
                    for (const auto &value : Array(Field(instruction, "probabilities")))
                        probabilities.push_back(Number(value));
                    Require(probabilities.size() == (size_t{1} << (2 * targets.size())), "A Pauli channel needs 4**len(qubits) probabilities");
                    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(targets, Simulators::QuantumChannel::Pauli(probabilities)));
                }
                else
                {
                    Require(targets.size() <= 2, "Kraus instructions require one or two qubits");
                    const size_t dim = size_t{1} << targets.size();
                    Simulators::QuantumChannel::KrausOperators operators;
                    for (const auto &matrix : Array(Field(instruction, "operators")))
                    {
                        const auto &entries = Array(matrix);
                        Require(entries.size() == dim * dim, "Kraus matrices must be flat row-major arrays");
                        Eigen::MatrixXcd op(dim, dim);
                        for (size_t r = 0; r < dim; ++r)
                            for (size_t c = 0; c < dim; ++c)
                                op(r, c) = Complex(entries[r * dim + c]);
                        operators.push_back(std::move(op));
                    }
                    circuit->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<>>(targets, Simulators::QuantumChannel(operators)));
                }
            }
            else
            {
                Require(!instruction.contains("duration") && !instruction.contains("operators"), "Unsupported gate/measurement fields");
                if (const auto *params = instruction.if_contains("params"))
                    for (const auto &param : Array(*params))
                        Number(param);
                const std::map<std::string, size_t> arities{{"p", 1},  {"rx", 1},  {"ry", 1},  {"rz", 1},  {"u", 3},  {"u1", 1},  {"u2", 2}, {"u3", 3},
                                                            {"cp", 1}, {"crx", 1}, {"cry", 1}, {"crz", 1}, {"cu", 4}, {"rxx", 1}, {"rzz", 1}};
                const auto params = instruction.if_contains("params") ? Array(instruction.at("params")) : json::array{};
                Require(params.size() == (arities.count(name) ? arities.at(name) : 0), "Incorrect gate parameter count");
                Require(name == "measure" || !instruction.contains("clbits"), "clbits applies only to measurement");
                const bool noiseless = instruction.contains("noiseless") && Boolean(instruction.at("noiseless"));
                std::optional<Circuits::ReadoutRates> readout;
                if (const auto *value = instruction.if_contains("readout_error"))
                    readout = ParseReadoutRates(*value);
                instruction.erase("noiseless");
                instruction.erase("readout_error");
                if (const auto *condition = instruction.if_contains("conditional_reg"))
                {
                    const auto bit = condition->is_array()
                                         ? (Require(Array(*condition).size() == 1, "Conditional register requires one bit"), UInt(Array(*condition)[0]))
                                         : UInt(*condition);
                    Require(bit < clbits && name != "measure", "Invalid conditional bit");
                }
                if (name == "id")
                {
                    instruction["name"] = "p";
                    instruction["params"] = json::array{0};
                }
                else if (name == "u1")
                {
                    instruction["name"] = "p";
                }
                else if (name == "u2")
                {
                    instruction["name"] = "u";
                    instruction["params"] = json::array{std::acos(-1.0) / 2, params[0], params[1]};
                }
                else if (name == "u3")
                {
                    instruction["name"] = "u";
                }
                const auto expanded = ExpandInstruction(instruction);
                CircuitPtr parsed;
                try
                {
                    parsed = parser.ParseCircuit(json::serialize(expanded).c_str());
                }
                catch (const std::runtime_error &error)
                {
                    // The legacy instruction parser uses runtime_error for malformed
                    // gates/measurements. Translate only this parsing boundary, never
                    // exceptions from simulator execution or resource allocation.
                    throw Error("invalid_input", "Invalid instruction: " + std::string(error.what()));
                }
                Require(parsed && parsed->GetOperations().size() == expanded.size(), "Invalid instruction");
                for (const auto &operation : parsed->GetOperations())
                {
                    // Noise models add no gate noise after a noiseless gate; an
                    // expanded rxx/rzz is noiseless as a whole.
                    if (noiseless)
                        operation->SetNoiseless();
                    if (readout && (readout->p_meas1_prep0 > 0 || readout->p_meas0_prep1 > 0))
                    {
                        auto measurement = std::dynamic_pointer_cast<Circuits::MeasurementOperation<>>(operation);
                        Require(measurement != nullptr, "readout_error applies only to measurement");
                        measurement->SetReadout(std::vector<Circuits::ReadoutRates>(measurement->GetQubits().size(), *readout));
                    }
                    circuit->AddOperation(operation);
                }
            }
        }
    }
    else
        throw Error("unsupported_capability", "Unknown circuit format: " + format);
    for (const auto q : circuit->GetQubits())
        Require(q < qubits, "Circuit qubit exceeds num_qubits");
    for (const auto c : circuit->GetBits())
        Require(c < clbits, "Circuit bit exceeds num_clbits");
    return {std::move(circuit), qubits, clbits};
}

CircuitPtr Unmeasured(const CircuitPtr &circuit, bool unitary = false, const std::string &context = "Overlap/fidelity")
{
    auto result = std::make_shared<Circuit>();
    bool measured = false;
    for (const auto &op : circuit->GetOperations())
    {
        const auto type = op->GetType();
        if (type == Circuits::OperationType::kMeasurement)
        {
            measured = true;
            continue;
        }
        Supported(!measured, "This query requires terminal measurements only");
        if (unitary)
            Supported(type == Circuits::OperationType::kGate || type == Circuits::OperationType::kDelay, context + " requires a unitary circuit");
        result->AddOperation(op->Clone());
    }
    return result;
}

std::vector<std::string> Observables(const json::object &request, size_t qubits)
{
    std::vector<std::string> result;
    if (const auto *values = request.if_contains("observables"))
    {
        for (const auto &item : Array(*values))
        {
            const auto pauli = String(item);
            Require(pauli.size() == qubits && pauli.find_first_not_of("IXYZ") == std::string::npos,
                    "Observables must be I/X/Y/Z strings of num_qubits characters "
                    "(q0 first)");
            result.push_back(pauli);
        }
    }
    Require(result.size() <= 100000, "Too many observables");
    return result;
}

struct Context
{
    unsigned long handle = 0;
    std::shared_ptr<Network::INetwork<double>> network;

    explicit Context(const ParsedCircuit &circuit, const SimulatorConfig &config, bool exactRegister = false)
    {
        auto *maestro = static_cast<Maestro *>(GetMaestroObjectWithMute());
        if (!maestro)
            throw Error("native_failure", "Cannot initialize Maestro");
        handle = maestro->CreateSimpleSimulator(circuit.qubits, circuit.clbits);
        try
        {
            // exactRegister: the caller drives the simulator itself. A single-host
            // network reserves an internal entanglement qubit, which direct
            // queries must not include; allocate exactly the declared register
            // from the start (the ancilla would double a statevector and
            // quadruple a density matrix, and might not fit). Otherwise the
            // network executes on its own register and its simulator only
            // selects the backend. A fixed bond-1 MPO costs nothing at full
            // width, and single-worker execution then reuses it. Automatic
            // selection prepares its own register, whatever the requested
            // method: a candidate list may not even hold the declared width.
            const bool exactWidth =
                exactRegister || (config.fixed_backend && config.simulation_type == Simulators::SimulationType::kMatrixProductOperator);
            network = ConfigureNetwork(handle, config, exactWidth ? circuit.qubits : 0);
            if (!network || !network->GetSimulator())
                throw Error("backend_unavailable", "Requested backend could not be created");
            // Automatic selection's placeholder may still differ; replace it.
            // Copy its backend first: a pointer alive across the call would keep
            // the old register allocated next to the new one.
            if (exactWidth && network->GetSimulator()->GetNumberOfQubits() != circuit.qubits)
            {
                const auto type = network->GetSimulator()->GetType();
                const auto method = network->GetSimulator()->GetSimulationType();
                network->CreateSimulator(type, method, circuit.qubits);
            }
            if (config.fixed_backend &&
                (network->GetSimulator()->GetType() != config.simulator_type || network->GetSimulator()->GetSimulationType() != config.simulation_type))
                throw Error("backend_unavailable", "Requested fixed backend/method is unavailable; fallback "
                                                   "is disabled");
            // Distributed placement is validated during backend initialization;
            // MPI and multi-device states report -1 instead of one GPU ordinal.
            // Only a fixed backend runs on this simulator: automatic selection
            // creates its GPU backend on the configured device itself (or fails),
            // and Metadata checks the device that ran.
            if (config.fixed_backend && !Simulators::IsDistributedGpuSimulator(config.simulator_type) && config.native_options.count("gpu_device") &&
                network->GetSimulator()->GetGpuDevice() != std::stoi(config.native_options.at("gpu_device")))
                throw Error("backend_unavailable", "Requested GPU device was not selected");
        }
        catch (...)
        {
            if (handle)
                DestroySimpleSimulator(handle);
            throw;
        }
    }

    ~Context()
    {
        network.reset();
        if (handle)
            DestroySimpleSimulator(handle);
    }

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    auto simulator() const
    {
        return network->GetSimulator();
    }
};

CircuitPtr Inject(const CircuitPtr &circuit, const NoiseConfig &noise, std::mt19937 &rng)
{
    if (!noise.enabled || noise.mode == "analytical")
        return circuit;
    CircuitPtr noisy;
    if (noise.mode == "coherent")
        noisy = noise.exact ? noise::inject_coherent_noise_exact(circuit, noise.model, rng) : noise::inject_coherent_noise(circuit, noise.model, rng);
    else if (noise.mode == "pauli")
        noisy = noise.exact ? noise::inject_exact_noise(circuit, noise.model) : noise::inject_noise(circuit, noise.model, rng);
    else
        noisy = noise.exact ? noise::inject_combined_noise_exact(circuit, noise.model, rng) : noise::inject_combined_noise(circuit, noise.model, rng);
    // Use the same measurement-time path as Python, including conditional
    // measurements and readout results consumed by subsequent classical control.
    noise::attach_readout_error(noisy, noise.model);
    return noisy;
}

// Inject(plan.circuit(), noise, rng) for a plan built from noise.model, without
// per-realization model lookups. Realization loops build the plan once.
CircuitPtr Inject(const noise::NoiseInjectionPlan &plan, const NoiseConfig &noise, std::mt19937 &rng)
{
    if (!noise.enabled || noise.mode == "analytical")
        return plan.circuit();
    CircuitPtr noisy;
    if (noise.mode == "coherent")
        noisy = plan.Coherent(rng, noise.exact);
    else if (noise.mode == "pauli")
        noisy = noise.exact ? plan.ExactPauli() : plan.Pauli(rng);
    else
        noisy = plan.Combined(rng, noise.exact);
    plan.AttachReadout(noisy);
    return noisy;
}

void ValidateNoisyCircuit(const CircuitPtr &circuit)
{
    for (const auto &op : circuit->GetOperations())
    {
        const auto type = op->GetType();
        Supported(type == Circuits::OperationType::kGate || type == Circuits::OperationType::kConditionalGate ||
                      type == Circuits::OperationType::kMeasurement || type == Circuits::OperationType::kConditionalMeasurement ||
                      type == Circuits::OperationType::kReset || type == Circuits::OperationType::kDelay || type == Circuits::OperationType::kQuantumChannel,
                  "Noise injection requires a flattened circuit of gates, "
                  "measurements, resets, delays and channels");
    }
}

bool HasChannels(const CircuitPtr &circuit)
{
    return circuit->HasQuantumChannels();
}

bool HasMeasurementReadout(const CircuitPtr &circuit)
{
    for (const auto &op : circuit->GetOperations())
        if (const auto measurement = std::dynamic_pointer_cast<Circuits::MeasurementOperation<>>(op); measurement && measurement->HasReadout())
            return true;
    return false;
}

// Mixed-state backends apply circuit channels exactly unless trajectories were
// requested. Otherwise each noise realization samples one Pauli per channel,
// which only Pauli channels and the noise injectors support.
void CheckChannels(const CircuitPtr &circuit, const SimulatorConfig &config, const NoiseConfig &noise)
{
    const bool sampled = noise.enabled && !noise.exact && noise.mode != "analytical";
    for (const auto &op : circuit->GetOperations())
    {
        // Nested channels execute as part of their circuit; they need the same
        // backend support as top-level ones.
        if (op->GetType() == Circuits::OperationType::kComposite)
        {
            CheckChannels(std::static_pointer_cast<Circuits::Circuit<>>(op), config, noise);
            continue;
        }
        if (op->GetType() != Circuits::OperationType::kQuantumChannel)
            continue;
        if (Mixed(config) && !sampled)
            continue;
        const auto channel = std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(op);
        Supported(!channel->GetChannel().GetPauliProbabilities().empty(),
                  "Explicit Kraus instructions require density_matrix or MPO with exact evaluation; only Pauli channels can be sampled");
        Supported(sampled, "Pauli channel instructions on pure-state backends require a noise block with trajectory evaluation; "
                           "set noise.realizations to the shot count for independent draws");
    }
}

using Counts = std::map<std::string, uint64_t>;

json::object EncodeCounts(const Counts &counts)
{
    json::object result;
    for (const auto &entry : counts)
        result[entry.first] = entry.second;
    return result;
}

std::string Bits(const std::vector<bool> &values, size_t size)
{
    std::string bits(size, '0');
    for (size_t i = 0; i < values.size() && i < size; ++i)
        if (values[i])
            bits[i] = '1';
    return bits;
}

json::array Expectations(const std::shared_ptr<Simulators::ISimulator> &simulator, const std::vector<std::string> &observables)
{
    json::array result;
    const auto values = simulator->ExpectationValues(observables);
    result.reserve(values.size());
    for (double value : values)
        result.emplace_back(value);
    return result;
}

void Prepare(Context &context, const CircuitPtr &circuit, size_t bits)
{
    Circuits::OperationState state(bits);
    auto simulator = context.simulator();
    simulator->Reset();
    simulator->SetGatesCounter(0);
    simulator->SetUpcomingGates(circuit->GetOperations());
    size_t bond = 0;
    circuit->ExecuteBD(simulator, state, &bond);
}

json::array Basis(const json::object &request, size_t qubits, size_t limit)
{
    Supported(qubits < 63, "Basis-index queries support fewer than 63 qubits");
    json::array states;
    if (const auto *values = request.if_contains("basis_states"))
        states = Array(*values);
    else
    {
        const auto count = uint64_t{1} << qubits;
        Require(count <= limit, "Full state output exceeds max_output_elements; request selected "
                                "basis_states");
        for (uint64_t i = 0; i < count; ++i)
            states.emplace_back(i);
    }
    Require(states.size() <= limit, "Selected state output exceeds max_output_elements");
    for (const auto &value : states)
        Require(UInt(value) < (uint64_t{1} << qubits), "Basis index out of range");
    return states;
}

void SameSize(const ParsedCircuit &first, const ParsedCircuit &other)
{
    Require(first.qubits == other.qubits && first.clbits == other.clbits, "Circuits in this operation must have identical register sizes");
}

template <class Result> json::object ReductionMetadata(const Result &execution, const SimulatorConfig &config)
{
    json::object options;
    for (const auto &[key, value] : config.native_options)
        options[key] = value;
    for (const auto &[key, value] : TypedNativeOptions(config))
        options[key] = value;
    options["seed"] = config.seed.value_or(0);
    if (config.gate_fusion)
        options["gate_fusion"] = *config.gate_fusion;
    options["auto_reduce"] = config.auto_reduce;
    options["enable_causal_cone_reduction"] = config.enable_causal_cone_reduction;
    options["enable_observable_clustering"] = config.enable_observable_clustering;
    options["allow_reset_marginalization"] = config.allow_reset_marginalization;
    options["enable_sampling_reduction"] = config.enable_sampling_reduction;
    options["counts_policy"] = config.counts_policy;
    if (config.max_cluster_qubits)
        options["max_cluster_qubits"] = *config.max_cluster_qubits;
    if (config.min_cluster_qubits)
        options["min_cluster_qubits"] = *config.min_cluster_qubits;
    if (config.observable_error_tolerance)
        options["observable_error_tolerance"] = *config.observable_error_tolerance;
    json::object metadata{{"backend", BackendName(execution.simulator)},
                          {"method", MethodName(execution.method)},
                          {"selection", config.fixed_backend ? "fixed" : "automatic"},
                          {"configured_options", std::move(options)},
                          {"gate_fusion", json::object{{"requested", config.gate_fusion ? json::value(*config.gate_fusion) : json::value()},
                                                       {"enabled", execution.gate_fusion},
                                                       {"max_qubits", execution.gate_fusion_max_qubits}}}};
    if (execution.gpu_device >= 0)
        metadata["gpu_device"] = execution.gpu_device;
    return metadata;
}

json::object Metadata(Context &context, const SimulatorConfig &config, bool executed)
{
    auto simulator = context.simulator();
    json::object result{{"backend", BackendName(executed ? context.network->GetLastSimulatorType() : simulator->GetType())},
                        {"method", MethodName(executed ? context.network->GetLastSimulationType() : simulator->GetSimulationType())},
                        {"selection", config.fixed_backend ? "fixed" : "automatic"}};
    json::object options;
    for (const auto &entry : config.native_options)
        options[entry.first] = entry.second;
    for (const auto &entry : config.distributed_options)
        options[entry.first] = entry.second;
    for (const auto &[key, value] : TypedNativeOptions(config))
        options[key] = value;
    options["seed"] = config.seed.value_or(0);
    options["counts_policy"] = config.counts_policy;
    const json::value requestedFusion = config.gate_fusion ? json::value(*config.gate_fusion) : json::value();
    if (config.gate_fusion)
        options["gate_fusion"] = *config.gate_fusion;
    result["gate_fusion"] = json::object{{"requested", requestedFusion},
                                         {"enabled", executed ? context.network->WasGateFusionEnabled() : simulator->IsGateFusionEnabled()},
                                         {"max_qubits", executed ? context.network->GetLastGateFusionMaxQubits() : simulator->GetGateFusionMaxQubits()}};
    result["configured_options"] = std::move(options);
    // After an execution the simulator is the restored backend placeholder,
    // which may not even be a GPU one: report the device that ran.
    if (executed)
    {
        const auto ran = context.network->GetLastSimulatorType();
        const int device = context.network->GetLastGpuDevice();
        if (ran == Simulators::SimulatorType::kGpuSim && config.native_options.count("gpu_device") &&
            device != std::stoi(config.native_options.at("gpu_device")))
            throw Error("backend_unavailable", "Requested GPU device was not selected");
        if (Simulators::IsGpuSimulator(ran))
            result["gpu_device"] = device;
    }
    else if (Simulators::IsGpuSimulator(simulator->GetType()))
        result["gpu_device"] = simulator->GetGpuDevice();
    if (Simulators::IsDistributedGpuSimulator(simulator->GetType()))
    {
        json::object placement;
        for (const char *key : Network::INetwork<double>::kDistributedPlacementKeys)
        {
            // After an execution, where the register that ran was placed: the
            // network's re-created simulator may have another width.
            const auto executedValue = executed ? context.network->GetLastDistributedPlacement(key) : std::optional<std::string>();
            placement[key] = executedValue ? *executedValue : simulator->GetConfiguration(key);
        }
        result["distribution"] = std::move(placement);
    }
    return result;
}

// Overflow-checked element count for output bounds.
uint64_t Elements(std::initializer_list<uint64_t> factors)
{
    uint64_t result = 1;
    for (const auto factor : factors)
    {
        Require(factor == 0 || result <= UINT64_MAX / factor, "Adjoint output size overflows");
        result *= factor;
    }
    return result;
}

json::array Matrix(const std::vector<double> &values, size_t rows, size_t cols)
{
    json::array result;
    result.reserve(rows);
    for (size_t row = 0; row < rows; ++row)
    {
        json::array line;
        line.reserve(cols);
        for (size_t col = 0; col < cols; ++col)
            line.emplace_back(values[row * cols + col]);
        result.emplace_back(std::move(line));
    }
    return result;
}

// Adjoint differentiation of a unitary circuit on QCSim's CPU statevector, as
// AdjointProgram in Python. Readouts are Pauli strings, weighted Pauli sums
// ({"terms": [[coefficient, pauli], ...]}) or basis-state probabilities of some
// qubits ({"probabilities": [qubits]}, first listed qubit least significant).
// The differentiable parameters are the gate angles in circuit order: one for
// p, rx, ry, rz, cp, crx, cry and crz; theta, phi, lambda for u; four for cu.
json::object RunAdjoint(const json::object &request, const std::string &operation, const SimulatorConfig &config, const ParsedCircuit &input,
                        const NoiseConfig &noise, bool validate, uint64_t limit)
{
    Supported(config.simulation_type == Method::kStatevector && (config.simulator_type == Backend::kQCSim || config.simulator_type == Backend::kCompositeQCSim),
              "Adjoint differentiation requires the qcsim or composite_qcsim statevector");
    Supported(!noise.enabled, "Adjoint differentiation requires a noiseless unitary circuit");
    Supported(input.qubits <= Adjoint::kMaxQubits, "Adjoint differentiation supports at most 63 qubits");
    // Terminal measurements are ignored, as for estimate; anything else non-unitary is rejected.
    const auto circuit = input.circuit->RemoveEndMeasurements();
    std::unique_ptr<Adjoint::Program> program;
    try
    {
        program = std::make_unique<Adjoint::Program>(*circuit);
    }
    catch (const std::invalid_argument &error)
    {
        throw Error("invalid_input", error.what());
    }

    const auto pauli = [&](const json::value &value) {
        const auto text = String(value);
        Require(text.size() == input.qubits && text.find_first_not_of("IXYZ") == std::string::npos,
                "Observables must be I/X/Y/Z strings of num_qubits characters (q0 first)");
        return text;
    };
    std::vector<Adjoint::Readout> readouts;
    const auto &items = Array(Field(request, "observables"));
    Require(!items.empty() && items.size() <= 100000, "Adjoint operations require 1..100000 observables");
    for (const auto &item : items)
    {
        if (item.is_string())
        {
            readouts.push_back(Adjoint::Readout::Expectation({{1., pauli(item)}}));
            continue;
        }
        const auto &readout = Object(item);
        Keys(readout, {"terms", "probabilities"});
        Require(readout.size() == 1, "An observable object needs exactly one of terms or probabilities");
        if (const auto *terms = readout.if_contains("terms"))
        {
            Adjoint::Observable observable;
            for (const auto &term : Array(*terms))
            {
                const auto &pair = Array(term);
                Require(pair.size() == 2, "Observable terms must be [coefficient, pauli] pairs");
                observable.emplace_back(Number(pair[0]), pauli(pair[1]));
            }
            Require(!observable.empty() && observable.size() <= 100000, "An observable needs 1..100000 terms");
            readouts.push_back(Adjoint::Readout::Expectation(std::move(observable)));
        }
        else
        {
            std::vector<Types::qubit_t> qubits;
            for (const auto &q : Array(readout.at("probabilities")))
            {
                Require(UInt(q) < input.qubits, "Probabilities qubit exceeds num_qubits");
                qubits.push_back(static_cast<Types::qubit_t>(UInt(q)));
            }
            readouts.push_back(Adjoint::Readout::Probabilities(std::move(qubits)));
        }
    }
    const size_t numValues = Adjoint::Columns(readouts);
    const size_t numParams = program->NumParams();

    size_t numRows = 1;
    std::vector<double> rows;
    if (const auto *value = request.if_contains("parameter_rows"))
    {
        const auto &values = Array(*value);
        Require(!values.empty() && values.size() <= limit, "parameter_rows needs 1..max_output_elements rows");
        numRows = values.size();
        rows.reserve(Elements({numRows, numParams}));
        for (const auto &row : values)
        {
            const auto &entries = Array(row);
            Require(entries.size() == numParams, "Each parameter row needs num_parameters (" + std::to_string(numParams) + ") values");
            for (const auto &entry : entries)
                rows.push_back(Number(entry));
        }
    }
    else
        rows = program->DefaultParams();

    uint64_t outputs = Elements({numRows, numValues});
    if (operation == "adjoint_jacobian")
        outputs += Elements({numRows, numValues, numParams});
    if (operation == "adjoint_vjp")
        outputs += Elements({numRows, numParams});
    Require(outputs <= limit, "Adjoint output exceeds max_output_elements");

    std::vector<double> cotangents;
    if (operation == "adjoint_vjp")
    {
        const auto &values = Array(Field(request, "cotangents"));
        Require(values.size() == numRows, "cotangents needs one row per parameter row");
        cotangents.reserve(outputs);
        for (const auto &row : values)
        {
            const auto &entries = Array(row);
            Require(entries.size() == numValues, "Each cotangent row needs num_values (" + std::to_string(numValues) + ") values");
            for (const auto &entry : entries)
                cotangents.push_back(Number(entry));
        }
    }

    json::object result{{"schema_version", SchemaVersion},
                        {"operation", operation},
                        {"backend", BackendName(config.simulator_type)},
                        {"method", MethodName(config.simulation_type)},
                        {"num_qubits", input.qubits},
                        {"num_parameters", numParams},
                        {"num_values", numValues}};
    if (validate)
    {
        result["valid"] = true;
        return result;
    }
    const auto start = Clock::now();
    const bool cone = config.enable_causal_cone_reduction;
    std::vector<double> values, jacobian, vjp;
    if (operation == "adjoint_expectation_values")
        values = program->Values(readouts, rows, numRows, cone);
    else if (operation == "adjoint_jacobian")
        program->Jacobian(readouts, rows, numRows, cone, values, jacobian);
    else
        program->VectorJacobianProduct(readouts, cotangents, rows, numRows, cone, values, vjp);

    json::array defaults;
    for (const double value : program->DefaultParams())
        defaults.emplace_back(value);
    result.erase("backend");
    result.erase("method");
    result["num_clbits"] = input.clbits;
    result["circuit_parameters"] = std::move(defaults);
    result["expectation_values"] = Matrix(values, numRows, numValues);
    if (operation == "adjoint_jacobian")
    {
        json::array batch;
        for (size_t row = 0; row < numRows; ++row)
        {
            std::vector<double> block(jacobian.begin() + static_cast<std::ptrdiff_t>(row * numValues * numParams),
                                      jacobian.begin() + static_cast<std::ptrdiff_t>((row + 1) * numValues * numParams));
            batch.emplace_back(Matrix(block, numValues, numParams));
        }
        result["jacobian"] = std::move(batch);
    }
    if (operation == "adjoint_vjp")
        result["vjp"] = Matrix(vjp, numRows, numParams);
    result["execution_metadata"] = json::object{{"backend", BackendName(config.simulator_type)},
                                                {"method", MethodName(config.simulation_type)},
                                                {"selection", "fixed"},
                                                {"differentiation", "adjoint"},
                                                {"causal_cone_reduction", cone}};
    result["time_taken"] = std::chrono::duration<double>(Clock::now() - start).count();
    result["probability_order"] = "first_listed_qubit_least_significant";
    Require(json::serialize(result).size() <= MaxResultBytes, "Result exceeds native result limit");
    return result;
}
} // namespace

json::object Capabilities()
{
    return NativeCapabilities(operations);
}

json::object Run(const json::object &request, bool validate, unsigned depth, ShotRows *rows)
{
    Require(depth <= 4, "Batch nesting exceeds four levels");
    Require(UInt(Field(request, "schema_version")) == SchemaVersion, "Unsupported native schema_version");
    Keys(request, {"schema_version", "operation",    "circuit",      "simulator",     "execution",           "noise",      "observables",
                   "outputs",        "basis_states", "target_state", "other_circuit", "step_circuit",        "steps",      "suffixes",
                   "requests",       "diagnostics",  "maintenance",  "keep_qubits",   "max_output_elements", "operators",  "expectation_operators",
                   "move_qubits",    "pauli",        "row_state",    "col_state",     "parameter_rows",      "cotangents", "target_statevector"});
    const auto operation = String(Field(request, "operation"));
    Supported(operations.count(operation), "Unknown operation: " + operation);
    // A known field for another operation must not bypass nested validation or
    // appear to configure a computation which never reads it.
    const std::map<std::string, std::set<std::string>> operationFields{
        {"observables", {"estimate", "incremental_evolve", "diagnostics", "mirror_fidelity", "adjoint_expectation_values", "adjoint_jacobian", "adjoint_vjp"}},
        {"basis_states", {"statevector", "amplitudes", "probabilities"}},
        {"target_state", {"state_probability"}},
        {"other_circuit", {"inner_product", "density_matrix_overlap"}},
        {"target_statevector", {"fidelity_with_statevector"}},
        {"step_circuit", {"incremental_evolve"}},
        {"steps", {"incremental_evolve"}},
        {"suffixes", {"checkpoint_batch"}},
        {"requests", {"batch"}},
        {"diagnostics", {"diagnostics"}},
        {"maintenance", {"diagnostics"}},
        {"keep_qubits", {"diagnostics"}},
        {"operators", {"diagnostics"}},
        {"expectation_operators", {"diagnostics"}},
        {"move_qubits", {"diagnostics"}},
        {"pauli", {"diagnostics"}},
        {"row_state", {"diagnostics"}},
        {"col_state", {"diagnostics"}},
        {"parameter_rows", {"adjoint_expectation_values", "adjoint_jacobian", "adjoint_vjp"}},
        {"cotangents", {"adjoint_vjp"}},
        {"max_output_elements",
         {"statevector", "amplitudes", "probabilities", "diagnostics", "incremental_evolve", "adjoint_expectation_values", "adjoint_jacobian", "adjoint_vjp"}}};
    for (const auto &entry : request)
    {
        const auto field = std::string(entry.key());
        const auto rule = operationFields.find(field);
        Require(rule == operationFields.end() || rule->second.count(operation), "Field " + field + " does not apply to " + operation);
    }
    if (operation == "batch")
    {
        Keys(request, {"schema_version", "operation", "requests"});
        const auto &requests = Array(Field(request, "requests"));
        Require(!requests.empty() && requests.size() <= 256, "A batch needs 1..256 requests");
        // Validate every member before starting any numerical work.
        for (const auto &item : requests)
            Run(Object(item), true, depth + 1);
        json::array results;
        if (!validate)
            for (const auto &item : requests)
            {
                results.emplace_back(Run(Object(item), false, depth + 1));
                Require(json::serialize(results).size() <= MaxResultBytes, "Batch results exceed native result limit");
            }
        return {{"schema_version", SchemaVersion}, {"operation", operation}, {"results", results}};
    }
    auto config = ParseConfig(Sub(request, "simulator"));
    const auto &execution = Sub(request, "execution");
    Keys(execution, {"shots", "seed", "return_samples"});
    Require(!execution.contains("shots") || operation == "execute" || operation == "checkpoint_batch", "execution.shots does not apply to " + operation);
    const size_t shots = UInt(execution, "shots", 1024);
    Require(shots > 0 && shots <= 1000000000, "shots must be in [1,1000000000]");
    Require(!execution.contains("return_samples") || operation == "execute", "execution.return_samples does not apply to " + operation);
    const bool return_samples = Boolean(execution, "return_samples", false);
    Require(!rows || return_samples, "Packed shot rows require an execute request with return_samples");
    if (execution.contains("seed"))
    {
        Require(!Sub(Sub(request, "simulator"), "options").contains("seed"), "Specify seed only once");
        config.seed = UInt(Field(execution, "seed"));
    }
    // An explicit simulator seed wins; otherwise the public noise seed also
    // controls measurement/readout randomness, as in Python's noisy execution.
    if (!execution.contains("seed") && !Sub(Sub(request, "simulator"), "options").contains("seed") && Sub(request, "noise").contains("seed"))
        config.seed = UInt(Field(Sub(request, "noise"), "seed"));
    const auto input = ParseCircuit(Object(Field(request, "circuit")));
    Supported(!Simulators::IsDistributedGpuSimulator(config.simulator_type) || input.qubits < 63, "Distributed GPU requires fewer than 63 qubits");
    Supported(config.fixed_backend || operation == "execute" || operation == "estimate" || operation == "validate",
              "State/query workflows require fixed selection");
    const auto &distribution = Sub(Sub(request, "simulator"), "distribution");
    if (const auto *globals = distribution.if_contains("global_qubits"))
        for (const auto &q : Array(*globals))
            Require(UInt(q) < input.qubits, "Global qubit exceeds register");
    if (const auto *devices = distribution.if_contains("devices"))
        Require(Array(*devices).size() < (uint64_t{1} << input.qubits), "Distribution requires at least one local qubit");
    auto noise = ParseNoise(Sub(request, "noise"), config, input.qubits);
    const auto limit = UInt(request, "max_output_elements", 65536);
    Require(limit > 0 && limit <= 1048576, "max_output_elements must be in [1,1048576]");
    Require(!request.contains("outputs"), "Use operation-specific queries instead of outputs");
    // Measurement readout_error changes recorded counts only. Check every parsed
    // circuit before its terminal measurements are removed, so operations that
    // never sample reject it instead of dropping it silently.
    const auto rejectReadout = [&](const CircuitPtr &parsed) {
        Supported(operation == "execute" || operation == "checkpoint_batch" || !HasMeasurementReadout(parsed),
                  "Measurement readout_error applies to measurement counts only");
    };
    rejectReadout(input.circuit);
    if (AdjointOperation(operation))
        return RunAdjoint(request, operation, config, input, noise, validate, limit);
    CheckChannels(input.circuit, config, noise);
    const auto observables = Observables(request, input.qubits);
    if (operation == "estimate" || operation == "incremental_evolve")
        Require(!observables.empty(), "This operation requires observables");
    Supported(!noise.enabled || config.fixed_backend, "Noise requires fixed backend selection");
    Supported(noise.mode != "analytical" || operation == "estimate", "Analytical noise is an estimator approximation");
    Supported(!noise.prefix || operation == "checkpoint_batch", "noise.prefix applies to checkpoint_batch prefixes only");
    Supported(!noise.model.has_readout_error() || operation == "execute" || operation == "checkpoint_batch",
              "Readout noise applies to measurement counts only");
    if (noise.enabled)
        ValidateNoisyCircuit(input.circuit);
    auto circuit = operation == "execute" || operation == "checkpoint_batch" ? input.circuit
                   : operation == "estimate"
                       ? input.circuit->RemoveEndMeasurements()
                       : Unmeasured(input.circuit, operation == "inner_product" || operation == "mirror_fidelity" || operation == "noisy_fidelity");
    const bool stochastic_estimation = operation == "estimate" && HasStochasticEvolution(circuit, UsesExactQuantumChannels(config));
    if (noise.enabled)
    {
        const auto layer = noise.mode == "pauli" ? NoiseLayer::kPauli : noise.mode == "coherent" ? NoiseLayer::kCoherent : NoiseLayer::kCombined;
        const bool estimator = operation == "estimate" || operation == "noisy_fidelity";
        noise.policy =
            ResolveNoiseRealizations(noise.requested_realizations, noise.model, config, layer, estimator ? 100 : 64, circuit, estimator, noise.exact);
        noise.realizations = noise.mode == "analytical" ? 1 : noise.policy.realizations;
        if (noise.policy.stochastic_noise && noise.mode != "analytical")
            AddApproximation(noise, "finite_noise_realizations");
        Supported(noise.mode != "analytical" || !stochastic_estimation,
                  "Analytical damping requires deterministic evolution; use combined noise to average measurement branches");
        // Averaging finitely many branches approximates an ensemble expectation.
        // Counts draw each shot's branch from its exact distribution instead:
        // stochastic_execution reports that, but it is no approximation.
        if (stochastic_estimation)
            AddApproximation(noise, "finite_measurement_trajectories");
    }
    std::vector<ParsedCircuit> additional;
    if (operation == "inner_product" || operation == "density_matrix_overlap")
    {
        additional.push_back(ParseCircuit(Object(Field(request, "other_circuit"))));
        SameSize(input, additional.back());
        rejectReadout(additional.back().circuit);
        // A mixed-state overlap prepares each state on its own, so channels and
        // resets are allowed; the pure-state inner product needs U(first)^dagger.
        additional.back().circuit = Unmeasured(additional.back().circuit, operation == "inner_product");
        if (operation == "density_matrix_overlap" && noise.enabled)
            ValidateNoisyCircuit(additional.back().circuit);
    }
    if (operation == "incremental_evolve")
    {
        Supported(!noise.enabled, "Incremental noisy evolution must be supplied as explicit "
                                  "circuit channels");
        additional.push_back(ParseCircuit(Object(Field(request, "step_circuit"))));
        SameSize(input, additional.back());
        rejectReadout(additional.back().circuit);
        additional.back().circuit = Unmeasured(additional.back().circuit);
        const auto &steps = Array(Field(request, "steps"));
        Require(!steps.empty() && steps.size() <= limit, "Invalid observation steps");
        uint64_t previous = 0;
        for (const auto &value : steps)
        {
            const auto step = UInt(value);
            Require(step >= previous && step <= 1000000000, "steps must be ordered nonnegative integers");
            previous = step;
        }
    }
    if (operation == "checkpoint_batch")
    {
        // Only suffix measurements are sampled. The prefix's terminal measurements
        // are removed below, so readout on them would be dropped silently.
        Supported(!HasMeasurementReadout(input.circuit), "Measurement readout_error applies to checkpoint suffixes only; "
                                                         "prefix measurements are removed");
        circuit = Unmeasured(input.circuit, true, "checkpoint_batch prefix");
        Supported(noise.mode != "analytical", "Checkpoint suffixes require explicit noise evolution");
        const auto &suffixes = Array(Field(request, "suffixes"));
        Require(!suffixes.empty() && suffixes.size() <= 256, "Invalid suffix count");
        for (const auto &suffix : suffixes)
        {
            additional.push_back(ParseCircuit(Object(suffix)));
            SameSize(input, additional.back());
            if (noise.enabled)
                ValidateNoisyCircuit(additional.back().circuit);
        }
        if (noise.enabled && noise.prefix)
        {
            // The saved prefix evolves once, so noise applied to it must be
            // deterministic there: exact channels without sampled layers.
            const auto layer = noise.mode == "pauli" ? NoiseLayer::kPauli : noise.mode == "coherent" ? NoiseLayer::kCoherent : NoiseLayer::kCombined;
            const auto prefixPolicy = ResolveNoiseRealizations(std::nullopt, noise.model, config, layer, 64, circuit, false, noise.exact);
            Supported(noise.exact && !prefixPolicy.stochastic_noise,
                      "noise.prefix requires deterministic exact evaluation: density_matrix or matrix_product_operator without coherent or "
                      "correlated noise");
        }
        if (noise.enabled)
        {
            // The policy above saw only the unitary prefix. The suffixes hold the
            // sampled channels and the branches (reported per suffix below).
            noise.policy.stochastic_execution = false;
            if (!noise.exact)
                for (const auto &suffix : additional)
                    noise.policy.stochastic_noise = noise.policy.stochastic_noise || suffix.circuit->HasQuantumChannels();
            if (noise.policy.stochastic_noise)
                AddApproximation(noise, "finite_noise_realizations");
        }
    }
    bool circuitChannels = HasChannels(input.circuit);
    for (const auto &other : additional)
    {
        CheckChannels(other.circuit, config, noise);
        circuitChannels = circuitChannels || HasChannels(other.circuit);
    }
    Require(!noise.enabled || noise.channel_count > 0 || circuitChannels,
            "noise.channels may be empty only when the circuit's channel instructions supply the noise");
    const bool amplitudes = operation == "statevector" || operation == "amplitudes";
    const bool query = amplitudes || operation == "probabilities" || operation == "state_probability" || operation == "diagnostics";
    // Observable mirror evaluation only takes expectations of the executed mirror
    // state, which mixed-state backends provide; it needs no pure-state overlap.
    const bool mirrorObservables = operation == "mirror_fidelity" && !observables.empty();
    if (amplitudes || operation == "inner_product" || (operation == "mirror_fidelity" && !mirrorObservables))
        Supported(!Mixed(config), "A mixed state has no unique statevector or pure-state overlap");
    if (MixedStateQuery(operation))
        Supported(MixedStateQueryApplies(config), operation + " requires the qcsim or gpu backend with density_matrix or matrix_product_operator");
    Supported(!noise.enabled || operation == "execute" || operation == "estimate" ||
                  ((query || MixedStateQuery(operation)) && noise.exact && noise.realizations == 1) || operation == "noisy_fidelity" ||
                  operation == "checkpoint_batch",
              "This operation requires a single exact noisy state or an explicit "
              "batch");
    // <psi|rho|psi> for every noise layer: sampled Pauli/T1/coherent
    // trajectories on pure states, exact channels on density matrices and MPOs
    // (as Python noisy_fidelity). Analytical damping is rejected above.
    if (operation == "noisy_fidelity")
        Supported(noise.enabled, "Noisy fidelity requires a noise block");
    json::array basis;
    if (amplitudes || operation == "probabilities")
        basis = Basis(request, input.qubits, limit);
    const bool nativeMpo = (config.simulator_type == Simulators::SimulatorType::kGpuSim || config.simulator_type == Simulators::SimulatorType::kQCSim) &&
                           config.simulation_type == Simulators::SimulationType::kMatrixProductOperator;
    const bool nativeMps = (config.simulator_type == Simulators::SimulatorType::kGpuSim || config.simulator_type == Simulators::SimulatorType::kQCSim) &&
                           config.simulation_type == Simulators::SimulationType::kMatrixProductState;
    auto readBits = [&](const char *field) {
        const auto value = String(Field(request, field));
        Require(value.size() == input.qubits && value.find_first_not_of("01") == std::string::npos,
                std::string(field) + " must be a q0-first bitstring of num_qubits characters");
        std::vector<bool> bits;
        for (char bit : value)
            bits.push_back(bit == '1');
        return bits;
    };
    uint64_t target = 0;
    std::vector<bool> targetBits;

    struct LocalOperator
    {
        Types::qubits_vector qubits;
        Eigen::MatrixXcd matrix;
        bool normalize;
    };

    std::vector<LocalOperator> localOperators;
    Types::qubits_vector moveQubits;
    Types::qubits_vector expectationQubits;
    std::vector<Eigen::MatrixXcd> expectationOperators;
    if (operation == "state_probability")
    {
        Supported(nativeMpo || input.qubits < 63, "Wide state probability requires CPU or GPU MPO");
        const auto state = String(Field(request, "target_state"));
        Require(state.size() == input.qubits && state.find_first_not_of("01") == std::string::npos,
                "target_state must be a q0-first bitstring of num_qubits characters");
        targetBits = readBits("target_state");
        for (size_t q = 0; q < std::min(size_t{64}, state.size()); ++q)
            if (state[q] == '1')
                target |= uint64_t{1} << q;
    }
    Eigen::VectorXcd targetVector;
    if (operation == "fidelity_with_statevector")
    {
        Supported(input.qubits <= MaxTargetStatevectorQubits,
                  "fidelity_with_statevector supports at most " + std::to_string(MaxTargetStatevectorQubits) + " qubits");
        const auto &entries = Array(Field(request, "target_statevector"));
        const size_t dimension = size_t{1} << input.qubits;
        Require(entries.size() == dimension, "target_statevector needs 2^num_qubits [real, imaginary] amplitudes");
        targetVector.resize(static_cast<Eigen::Index>(dimension));
        double norm = 0;
        for (size_t i = 0; i < dimension; ++i)
        {
            const auto amplitude = Complex(entries[i]);
            targetVector[static_cast<Eigen::Index>(i)] = amplitude;
            norm += std::norm(amplitude);
        }
        // Maestro computes <psi|rho|psi>/Tr(rho) and does not rescale psi.
        Require(std::abs(norm - 1) <= 1e-6, "target_statevector must be normalized");
    }
    if (operation == "diagnostics")
    {
        auto readQubits = [&](const json::value &value, bool allowEmpty = false) {
            Types::qubits_vector qubits;
            std::set<uint64_t> unique;
            for (const auto &q : Array(value))
            {
                const auto index = UInt(q);
                Require(index < input.qubits && unique.insert(index).second, "Invalid tensor target qubit");
                qubits.push_back(index);
            }
            Require(allowEmpty || !qubits.empty(), "Tensor operation requires qubits");
            return qubits;
        };
        if (const auto *value = request.if_contains("move_qubits"))
        {
            Supported(nativeMpo || nativeMps, "Chain routing requires CPU or GPU MPS/MPO");
            moveQubits = readQubits(*value, true);
        }
        if (const auto *values = request.if_contains("operators"))
        {
            Supported(nativeMpo, "Local operators require CPU or GPU MPO");
            for (const auto &value : Array(*values))
            {
                const auto &op = Object(value);
                Keys(op, {"qubits", "matrix", "normalize"});
                auto qubits = readQubits(Field(op, "qubits"));
                Require(qubits.size() <= 2, "Operators require one or two qubits");
                const size_t dim = size_t{1} << qubits.size();
                const auto &entries = Array(Field(op, "matrix"));
                Require(entries.size() == dim * dim, "Operator matrix must be flat row-major");
                Eigen::MatrixXcd matrix(dim, dim);
                for (size_t r = 0; r < dim; ++r)
                    for (size_t c = 0; c < dim; ++c)
                        matrix(r, c) = Complex(entries[r * dim + c]);
                bool normalize = false;
                if (const auto *flag = op.if_contains("normalize"))
                {
                    Require(flag->is_bool(), "normalize must be boolean");
                    normalize = flag->as_bool();
                }
                localOperators.push_back({std::move(qubits), std::move(matrix), normalize});
            }
        }
        if (const auto *values = request.if_contains("expectation_operators"))
        {
            Supported(nativeMps, "Operator expectations require CPU or GPU MPS");
            const auto &operators = Array(*values);
            Require(operators.size() <= limit, "Operator list exceeds output bound");
            for (const auto &value : operators)
            {
                const auto &op = Object(value);
                Keys(op, {"qubit", "matrix"});
                const auto qubit = UInt(Field(op, "qubit"));
                Require(qubit < input.qubits, "Operator target qubit is out of range");
                const auto &entries = Array(Field(op, "matrix"));
                Require(entries.size() == 4, "Expectation operator must be a flat row-major 2x2 matrix");
                Eigen::MatrixXcd matrix(2, 2);
                for (size_t r = 0; r < 2; ++r)
                    for (size_t c = 0; c < 2; ++c)
                        matrix(r, c) = Complex(entries[2 * r + c]);
                expectationQubits.push_back(qubit);
                expectationOperators.push_back(std::move(matrix));
            }
        }
        Supported(Mixed(config) || MatrixProductChain(config), "Diagnostics require density_matrix, MPS or MPO");
        if (request.contains("keep_qubits"))
        {
            bool partial = false;
            if (const auto *values = request.if_contains("diagnostics"))
                for (const auto &value : Array(*values))
                    partial |= String(value) == "partial_trace";
            Require(partial, "keep_qubits requires the partial_trace diagnostic");
        }
        std::set<std::string> requestedDiagnostics;
        if (const auto *values = request.if_contains("diagnostics"))
            for (const auto &value : Array(*values))
                requestedDiagnostics.insert(String(value));
        for (const char *field : {"row_state", "col_state"})
            Require(!request.contains(field) || requestedDiagnostics.count("element"), std::string(field) + " requires the element diagnostic");
        Require(!request.contains("pauli") || requestedDiagnostics.count("expectation_complex") || requestedDiagnostics.count("unnormalized_expectation"),
                "pauli requires a complex expectation diagnostic");
        Require(!request.contains("observables") || requestedDiagnostics.count("expectations_complex") ||
                    requestedDiagnostics.count("unnormalized_expectations"),
                "observables requires a batch complex expectation diagnostic");
        Require(!request.contains("expectation_operators") || requestedDiagnostics.count("operator_expectation"),
                "expectation_operators requires the operator_expectation diagnostic");
        if (const auto *values = request.if_contains("diagnostics"))
            for (const auto &value : Array(*values))
            {
                const auto name = String(value);
                Require(DiagnosticNames().count(name), "Unknown diagnostic");
                if (name == "operator_expectation")
                {
                    Supported(nativeMps, "Operator expectations require CPU or GPU MPS");
                    Require(request.contains("expectation_operators"), "operator_expectation requires expectation_operators");
                }
                else
                {
                    Supported(Mixed(config), "Mixed-state diagnostics require density_matrix or MPO");
                }
                if (name == "density_matrix" || name == "unnormalized_density_matrix")
                {
                    Supported(nativeMpo, "Dense MPO output requires CPU or GPU MPO");
                    Require(input.qubits <= 13 && (uint64_t{1} << (2 * input.qubits)) <= limit, "Dense matrix exceeds output bound");
                }
                else if (name == "element")
                {
                    Supported(nativeMpo, "Bit-vector matrix elements require CPU or GPU MPO");
                    readBits("row_state");
                    readBits("col_state");
                }
                else if (name == "expectation_complex" || name == "unnormalized_expectation")
                {
                    Supported(nativeMpo, "Complex expectations require CPU or GPU MPO");
                    const auto pauli = String(Field(request, "pauli"));
                    Require(pauli.size() == input.qubits && pauli.find_first_not_of("IXYZ") == std::string::npos, "pauli must contain one I/X/Y/Z per qubit");
                }
                else if (name == "expectations_complex" || name == "unnormalized_expectations")
                {
                    Supported(nativeMpo, "Complex expectations require CPU or GPU MPO");
                    Require(request.contains("observables"), "Batch complex expectations require observables");
                    Require(observables.size() <= limit, "Complex expectation batch exceeds output bound");
                }
                if (name == "partial_trace")
                {
                    const auto &keep = Array(Field(request, "keep_qubits"));
                    Require(keep.size() < 32 && (uint64_t{1} << (2 * keep.size())) <= limit, "Reduced matrix exceeds output bound");
                    std::set<uint64_t> unique;
                    for (const auto &q : keep)
                        Require(UInt(q) < input.qubits && unique.insert(UInt(q)).second, "Invalid reduced-state qubit");
                }
            }
        if (const auto *actions = request.if_contains("maintenance"))
            for (const auto &action : Array(*actions))
            {
                const auto name = String(action);
                Require(name == "restore_trace" || name == "hermitize" || name == "trim" || name == "recanonicalize", "Unknown maintenance action");
                if (name == "trim" || name == "recanonicalize")
                    Supported(MatrixProductChain(config), "This maintenance action requires MPS or MPO");
                else
                    Supported(Mixed(config), "This maintenance action requires density_matrix or MPO");
            }
    }
    if (return_samples)
    {
        Supported(!Simulators::IsDistributedGpuSimulator(config.simulator_type), "return_samples is not available for distributed GPU execution");
        // A JSON row is a quoted bitstring and a separator; packed rows have no such bound.
        Require(rows || input.clbits + 3 <= MaxResultBytes / shots,
                "Shot rows would exceed the native result limit; use MaestroRunRequestSamples for packed rows");
    }
    if (validate || operation == "validate")
        return {{"schema_version", SchemaVersion},
                {"operation", operation},
                {"valid", true},
                {"backend", BackendName(config.simulator_type)},
                {"method", MethodName(config.simulation_type)}};
    const auto start = Clock::now();
    // Resolve an omitted seed only after validation. MPI execution obtains one
    // shared random seed, while validation needs neither entropy nor collectives.
    const bool prefer_frame_counts = PreferFrameCounts(config);
    if (!config.seed)
    {
        config.seed = Simulators::GenerateRandomSeed(config.simulator_type, config.distributed_options);
        noise.seed = static_cast<uint32_t>(*config.seed);
    }
    if (operation == "execute" && return_samples)
    {
        // The row samplers shared with Python and C++ (SampleRecords.h).
        if (!noise.thermal_approximation_warning.empty())
            std::cerr << "Warning: " << noise.thermal_approximation_warning << '\n';
        ShotRecords records;
        if (noise.enabled)
        {
            NoisyExecutionOptions options;
            options.layer = noise.mode == "pauli" ? NoiseLayer::kPauli : noise.mode == "coherent" ? NoiseLayer::kCoherent : NoiseLayer::kCombined;
            options.rows = true;
            options.exact_channels = noise.exact;
            records = ExecuteNoisy(circuit, noise.model, config, shots, noise.realizations, noise.seed, options);
        }
        else
            records = SampleCircuitRows(circuit, config, shots);
        if (records.shots != shots || records.rows.size() != SampleStorageSize(records.shots, records.width))
            throw Error("native_failure", "Backend did not return the requested shot rows");
        // Columns follow the declared classical register, as count keys do.
        const size_t columns = std::min(records.width, input.clbits);
        json::object result{{"schema_version", SchemaVersion}, {"operation", operation}, {"num_qubits", input.qubits}, {"num_clbits", input.clbits}};
        if (rows)
        {
            rows->shots = shots;
            rows->width = input.clbits;
            rows->data.assign(SampleStorageSize(shots, input.clbits), 0);
            for (size_t shot = 0; shot < shots; ++shot)
                std::copy_n(records.rows.data() + shot * records.width, columns, rows->data.data() + shot * input.clbits);
        }
        else
        {
            json::array samples;
            samples.reserve(shots);
            std::string row(input.clbits, '0');
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const uint8_t *record = records.rows.data() + shot * records.width;
                for (size_t bit = 0; bit < columns; ++bit)
                    row[bit] = record[bit] ? '1' : '0';
                samples.emplace_back(row);
            }
            result["samples"] = std::move(samples);
        }
        SamplingResult executed;
        executed.simulator = records.simulator;
        executed.method = records.method;
        executed.gpu_device = records.gpu_device;
        executed.gate_fusion = records.gate_fusion;
        executed.gate_fusion_max_qubits = records.gate_fusion_max_qubits;
        auto metadata = ReductionMetadata(executed, config);
        metadata["sampling_strategy"] = records.sampling_strategy;
        result["shots"] = shots;
        result["sampling_components"] = records.components;
        result["largest_component_qubits"] = records.largest_component;
        result["seed"] = *config.seed;
        result["execution_metadata"] = std::move(metadata);
        result["time_taken"] = std::chrono::duration<double>(Clock::now() - start).count();
        result["count_order"] = "classical_bit_0_first";
        result["basis_order"] = "qubit_0_least_significant";
        if (records.max_bond_dimension)
            result["max_bond_dim_reached"] = records.max_bond_dimension;
        if (noise.enabled)
            result["noise"] = NoiseMetadata(noise, records.realizations);
        Require(json::serialize(result).size() <= MaxResultBytes, "Result exceeds native result limit");
        return result;
    }
    if (operation == "execute" && prefer_frame_counts && (!noise.enabled || noise.realizations >= shots) && noise.mode != "coherent" &&
        noise.model.inject_noise_covers_model())
    {
        auto channels = noise::inject_exact_noise(circuit, noise.model);
        noise::attach_readout_error(channels, noise.model);
        if (auto plan = CompileReferenceSampler(channels, input.qubits, input.clbits))
        {
            // This path returns before the ordinary executor's warning below.
            if (!noise.thermal_approximation_warning.empty())
                std::cerr << "Warning: " << noise.thermal_approximation_warning << '\n';
            const auto seed = noise.enabled ? Simulators::IState::DeriveSeed(*config.seed, noise.seed) : *config.seed;
            Counts counts;
            for (const auto &[bits, count] : plan->SampleCounts(shots, seed))
                counts[Bits(bits, input.clbits)] += count;
            SamplingResult frame;
            frame.simulator = Simulators::SimulatorType::kQCSim;
            frame.method = Simulators::SimulationType::kStabilizer;
            frame.components = 1;
            frame.largest_component = input.qubits;
            auto metadata = ReductionMetadata(frame, config);
            metadata["sampling_strategy"] = "reference_frame_v1";
            // Replaying the returned seed also requires the resolved policy.
            metadata["configured_options"].as_object()["counts_policy"] = "frame_v1";
            json::object result{{"schema_version", SchemaVersion},
                                {"operation", operation},
                                {"num_qubits", input.qubits},
                                {"num_clbits", input.clbits},
                                {"counts", EncodeCounts(counts)},
                                {"shots", shots},
                                {"sampling_components", 1},
                                {"largest_component_qubits", input.qubits},
                                {"seed", *config.seed},
                                {"execution_metadata", std::move(metadata)},
                                {"time_taken", std::chrono::duration<double>(Clock::now() - start).count()},
                                {"count_order", "classical_bit_0_first"},
                                {"basis_order", "qubit_0_least_significant"}};
            if (noise.enabled)
                result["noise"] = NoiseMetadata(noise, std::min(shots, noise.realizations));
            Require(json::serialize(result).size() <= MaxResultBytes, "Result exceeds native result limit");
            return result;
        }
    }
    if (operation == "estimate" && !Simulators::IsDistributedGpuSimulator(config.simulator_type))
    {
        if (!noise.thermal_approximation_warning.empty())
            std::cerr << "Warning: " << noise.thermal_approximation_warning << '\n';
        std::mt19937 rng(noise.seed);
        const size_t realizations = noise.enabled && noise.mode != "analytical" ? noise.realizations : stochastic_estimation ? kMeasurementTrajectories : 1;
        std::vector<double> sums(observables.size()), squares(observables.size()), bounds(observables.size());
        size_t bounded = 0;
        EstimationCache cache;
        EstimationResult last;
        const noise::NoiseInjectionPlan plan(circuit, noise.model);
        size_t largest = 0, clusters = 0, total_qubits = 0, total_clusters = 0;
        double reduction_time = 0;
        bool reduced = false;
        for (size_t r = 0; r < realizations; ++r)
        {
            auto execution_config = config;
            execution_config.seed = Simulators::IState::DeriveSeed(*config.seed, r);
            if (noise.enabled && noise.mode != "analytical")
                execution_config.auto_reduce = false;
            last = EstimateCircuit(Inject(plan, noise, rng), observables, execution_config, noise.enabled && noise.mode != "analytical" ? &cache : nullptr);
            largest = std::max(largest, last.plan.qubits_after);
            clusters = std::max(clusters, last.plan.groups.size());
            total_qubits += last.plan.total_cluster_qubits;
            total_clusters += last.plan.groups.size();
            reduction_time += last.plan.reduction_time;
            reduced = reduced || last.plan.auto_reduced;
            bounded = std::max(bounded, last.plan.bounded_observables);
            for (size_t i = 0; i < sums.size(); ++i)
            {
                bounds[i] += last.plan.error_bounds[i] / realizations;
                const double value = last.values[i] * (noise.mode == "analytical" ? noise.model.compute_damping(observables[i]) : 1);
                sums[i] += value;
                squares[i] += value * value;
            }
        }
        json::array values, errors, methods, error_bounds;
        for (auto bound : bounds)
            error_bounds.emplace_back(bound);
        for (size_t i = 0; i < sums.size(); ++i)
        {
            const double mean = sums[i] / realizations;
            values.emplace_back(mean);
            errors.emplace_back(realizations > 1 ? std::sqrt(std::max(0.0, (squares[i] - realizations * mean * mean) / (realizations - 1) / realizations)) : 0);
        }
        for (auto method : last.cluster_methods)
            methods.emplace_back(MethodName(method));
        auto metadata = ReductionMetadata(last, config);
        metadata["stochastic_execution"] = stochastic_estimation;
        metadata["measurement_trajectories"] = stochastic_estimation ? realizations : size_t{0};
        metadata["estimate_evaluation"] = stochastic_estimation || noise.policy.stochastic_noise ? "trajectory_average" : "deterministic";
        json::object result{{"schema_version", SchemaVersion},
                            {"operation", operation},
                            {"num_qubits", input.qubits},
                            {"num_clbits", input.clbits},
                            {"expectation_values", std::move(values)},
                            {"realization_standard_errors", std::move(errors)},
                            {"auto_reduced", reduced},
                            {"expectation_error_bounds", std::move(error_bounds)},
                            {"bounded_observables", bounded},
                            {"cone_cache_hits", cache.hits},
                            {"qubits_before", input.qubits},
                            {"qubits_after", largest},
                            {"clusters_count", clusters},
                            {"largest_cluster_qubits", largest},
                            {"mean_cluster_qubits", total_clusters ? static_cast<double>(total_qubits) / total_clusters : 0},
                            {"reduction_time", reduction_time},
                            {"cluster_methods", std::move(methods)},
                            {"execution_metadata", std::move(metadata)},
                            {"seed", *config.seed},
                            {"time_taken", std::chrono::duration<double>(Clock::now() - start).count()},
                            {"count_order", "classical_bit_0_first"},
                            {"basis_order", "qubit_0_least_significant"}};
        if (noise.enabled)
            result["noise"] = NoiseMetadata(noise, realizations);
        if (realizations == 1 && (stochastic_estimation || noise.policy.stochastic_noise))
            result["realization_standard_errors"] = nullptr;
        Require(json::serialize(result).size() <= MaxResultBytes, "Result exceeds native result limit");
        return result;
    }
    if (operation == "execute" && config.enable_sampling_reduction && !Simulators::IsDistributedGpuSimulator(config.simulator_type))
    {
        if (!noise.thermal_approximation_warning.empty())
            std::cerr << "Warning: " << noise.thermal_approximation_warning << '\n';
        std::mt19937 rng(noise.seed);
        const auto realizations = std::min(shots, noise.enabled && noise.mode != "analytical" ? noise.realizations : size_t{1});
        Counts counts;
        SamplingResult last;
        const noise::NoiseInjectionPlan plan(circuit, noise.model);
        size_t largest = 0, components = 0;
        for (size_t r = 0; r < realizations; ++r)
        {
            auto local_config = config;
            local_config.seed = Simulators::IState::DeriveSeed(*config.seed, r);
            last = SampleCircuit(Inject(plan, noise, rng), local_config, shots / realizations + (r < shots % realizations));
            largest = std::max(largest, last.largest_component);
            components = std::max(components, last.components);
            for (const auto &[outcome, count] : last.counts)
                counts[Bits(outcome, input.clbits)] += count;
        }
        json::object result{{"schema_version", SchemaVersion},
                            {"operation", operation},
                            {"num_qubits", input.qubits},
                            {"num_clbits", input.clbits},
                            {"counts", EncodeCounts(counts)},
                            {"shots", shots},
                            {"sampling_components", components},
                            {"largest_component_qubits", largest},
                            {"seed", *config.seed},
                            {"execution_metadata", ReductionMetadata(last, config)},
                            {"time_taken", std::chrono::duration<double>(Clock::now() - start).count()},
                            {"count_order", "classical_bit_0_first"},
                            {"basis_order", "qubit_0_least_significant"}};
        if (!last.sampling_strategy.empty())
            result["execution_metadata"].as_object()["sampling_strategy"] = last.sampling_strategy;
        if (noise.enabled)
            result["noise"] = NoiseMetadata(noise, realizations);
        Require(json::serialize(result).size() <= MaxResultBytes, "Result exceeds native result limit");
        return result;
    }
    // Only execute and estimate run on the network, which allocates its own
    // register; the network simulator is then just a backend placeholder.
    // Every other operation drives the simulator directly and needs exactly the
    // declared register: mixed-state queries compare whole registers, and the
    // network's entanglement ancilla would double statevector and quadruple
    // density-matrix work (checkpoints, overlaps, mirrors, evolution).
    const bool networkExecuted = operation == "execute" || operation == "estimate";
    Context context(input, config, !networkExecuted);
    // Execution metadata describes the network that ran last (parallel execute).
    Context *metadata_context = &context;
    std::vector<std::unique_ptr<Context>> metadata_contexts;
    // Only direct operations drive the simulator. Holding it through network
    // executions would keep the register the first realization grew alive
    // after the network installs its placeholder again.
    auto simulator = networkExecuted ? nullptr : context.simulator();
    // Validation remains silent; warn once per execution, not per realization.
    if (!noise.thermal_approximation_warning.empty())
        std::cerr << "Warning: " << noise.thermal_approximation_warning << '\n';
    std::mt19937 rng(noise.seed);
    const noise::NoiseInjectionPlan plan(circuit, noise.model);
    json::object result{{"schema_version", SchemaVersion}, {"operation", operation}, {"num_qubits", input.qubits}, {"num_clbits", input.clbits}};
    // As in the non-distributed estimate, a dynamic estimate without sampled
    // noise still averages measurement branches.
    size_t realizations = noise.enabled && noise.mode != "analytical" ? noise.realizations : stochastic_estimation ? kMeasurementTrajectories : 1;
    if (operation == "execute" || operation == "checkpoint_batch")
        realizations = std::min(realizations, shots);
    if (operation == "execute")
    {
        Counts counts;
        // One shot per realization makes each realization's result depend only
        // on its own seed, so CPU QCSim realizations may run on several networks
        // without changing seeded counts. Realizations with several shots keep
        // the shared network: their inner shot workers set seeded partitions.
        size_t workers = 1;
        if (realizations > 1 && realizations == shots && RunsOnlyQCSim(config))
        {
            // Each worker's network maps the circuit to its own register (at most
            // the declared qubits; the network's ancilla is never allocated),
            // for whichever backend selection runs.
            if (const long double bytes = MaxRealizationBytes(config, input.qubits); bytes > 0)
                workers = Utils::LimitRealizationWorkersByMemory(Utils::RealizationWorkerCount(realizations, config.max_simulators.value_or(0)),
                                                                 static_cast<size_t>(bytes), RealizationBudgetMB(config), 1, 2);
        }
        if (workers <= 1)
            for (size_t r = 0; r < realizations; ++r)
            {
                auto noisy = Inject(plan, noise, rng);
                const auto count = shots / realizations + (r < shots % realizations ? 1 : 0);
                context.network->Configure("seed", std::to_string(Simulators::IState::DeriveSeed(*config.seed, r)).c_str());
                const auto raw = context.network->RepeatedExecuteOnHost(noisy, 0, count);
                for (const auto &entry : raw)
                    counts[Bits(entry.first, input.clbits)] += entry.second;
            }
        else
        {
            // Worker 0 uses the request's network; the others get their own,
            // created here, serially. Metadata then describes the network that
            // ran the last realization, as after the serial loop.
            std::vector<std::unique_ptr<Context>> extra;
            std::vector<Context *> contexts{&context};
            for (size_t worker = 1; worker < workers; ++worker)
            {
                extra.push_back(std::make_unique<Context>(input, config));
                contexts.push_back(extra.back().get());
            }
            std::vector<Counts> local(workers);
            std::vector<size_t> latest(workers, 0);
            std::vector<char> ran(workers, 0);
            Utils::RunRealizations(
                realizations, workers, [&](size_t) { return Inject(plan, noise, rng); },
                [&](size_t worker, size_t r, CircuitPtr noisy) {
                    auto &network = contexts[worker]->network;
                    network->Configure("seed", std::to_string(Simulators::IState::DeriveSeed(*config.seed, r)).c_str());
                    for (const auto &entry : network->RepeatedExecuteOnHost(noisy, 0, 1))
                        local[worker][Bits(entry.first, input.clbits)] += entry.second;
                    latest[worker] = std::max(latest[worker], r);
                    ran[worker] = 1;
                });
            size_t last = 0;
            for (size_t worker = 0; worker < workers; ++worker)
            {
                for (const auto &[bits, count] : local[worker])
                    counts[bits] += count;
                if (ran[worker] && latest[worker] >= latest[last])
                    last = worker;
            }
            metadata_context = contexts[last];
            metadata_contexts = std::move(extra);
        }
        uint64_t total = 0;
        for (const auto &entry : counts)
            total += entry.second;
        if (total != shots)
            throw Error("native_failure", "Backend did not return the requested number of shots");
        result["counts"] = EncodeCounts(counts);
        result["shots"] = total;
    }
    else if (operation == "estimate")
    {
        std::vector<double> sums(observables.size(), 0), squares(observables.size(), 0);
        for (size_t r = 0; r < realizations; ++r)
        {
            const auto noisy = Inject(plan, noise, rng);
            context.network->Configure("seed", std::to_string(Simulators::IState::DeriveSeed(*config.seed, r)).c_str());
            const auto values = context.network->ExecuteOnHostExpectations(noisy, 0, observables);
            Require(values.size() == sums.size(), "Backend returned an invalid expectation vector");
            for (size_t i = 0; i < values.size(); ++i)
            {
                const double value = values[i] * (noise.mode == "analytical" ? noise.model.compute_damping(observables[i]) : 1);
                sums[i] += value;
                squares[i] += value * value;
            }
        }
        json::array values, errors;
        for (size_t i = 0; i < sums.size(); ++i)
        {
            const auto mean = sums[i] / realizations;
            values.emplace_back(mean);
            errors.emplace_back(realizations > 1 ? std::sqrt(std::max(0.0, (squares[i] - realizations * mean * mean) / (realizations - 1) / realizations))
                                                 : 0.0);
        }
        result["expectation_values"] = std::move(values);
        // One sample of a stochastic estimate cannot estimate its uncertainty.
        if (realizations == 1 && (stochastic_estimation || noise.policy.stochastic_noise))
            result["realization_standard_errors"] = nullptr;
        else
            result["realization_standard_errors"] = std::move(errors);
    }
    else if (operation == "fidelity_with_statevector")
    {
        Prepare(context, Inject(circuit, noise, rng), input.clbits);
        result["fidelity"] = simulator->FidelityWithStatevector(targetVector);
    }
    else if (operation == "density_matrix_overlap")
    {
        // Both states use the same backend and noise model; the overlap needs two
        // registers of one simulator type.
        Prepare(context, Inject(circuit, noise, rng), input.clbits);
        Context other(additional[0], config, true);
        Prepare(other, Inject(additional[0].circuit, noise, rng), input.clbits);
        result["overlap"] = Complex(simulator->DensityMatrixOverlap(*other.simulator()));
    }
    else if (query)
    {
        Prepare(context, Inject(circuit, noise, rng), input.clbits);
        if (operation == "state_probability")
            result["probability"] = nativeMpo ? simulator->ProbabilityBits(targetBits) : simulator->Probability(target);
        else if (operation == "diagnostics")
        {
            for (const auto &op : localOperators)
                simulator->ApplyOperator(op.qubits, op.matrix, op.normalize);
            if (!moveQubits.empty())
                simulator->MoveAtBeginningOfChain(moveQubits);
            Supported(Mixed(config) || MatrixProductChain(config), "Diagnostics require density_matrix, MPS or MPO");
            if (const auto *actions = request.if_contains("maintenance"))
                for (const auto &action : Array(*actions))
                {
                    const auto name = String(action);
                    if (name == "restore_trace")
                        simulator->RestoreDensityMatrixTrace();
                    else if (name == "hermitize")
                        simulator->HermitizeDensityMatrix();
                    else if (name == "trim")
                        simulator->Trim();
                    else if (name == "recanonicalize")
                        simulator->ReCanonicalize();
                    else
                        throw Error("invalid_input", "Unknown maintenance action");
                }
            // a pure MPS has no mixed-state diagnostics to report by default
            json::array defaults;
            if (Mixed(config))
                defaults = {"trace", "purity"};
            const auto *queries = request.if_contains("diagnostics");
            for (const auto &value : queries ? Array(*queries) : defaults)
            {
                const auto name = String(value);
                if (name == "operator_expectation")
                    result[name] = Complex(simulator->ExpectationValueOperators(expectationQubits, expectationOperators));
                else if (name == "trace")
                    result[name] = Complex(simulator->DensityMatrixTrace());
                else if (name == "purity")
                    result[name] = simulator->DensityMatrixPurity();
                else if (name == "trace_of_square")
                    result[name] = Complex(simulator->DensityMatrixTraceOfSquare());
                else if (name == "hermiticity_residual")
                    result[name] = simulator->DensityMatrixHermiticityResidual();
                else if (name == "is_hermitian")
                    result[name] = simulator->IsDensityMatrixHermitian();
                else if (name == "element")
                    result[name] = Complex(simulator->DensityMatrixElementBits(readBits("row_state"), readBits("col_state")));
                else if (name == "expectation_complex" || name == "unnormalized_expectation")
                    result[name] = Complex(simulator->ExpectationValueComplex(String(Field(request, "pauli")), name == "expectation_complex"));
                else if (name == "expectations_complex" || name == "unnormalized_expectations")
                {
                    const auto values = simulator->ExpectationValuesComplex(observables, name == "expectations_complex");
                    json::array encoded;
                    encoded.reserve(values.size());
                    for (const auto &value : values)
                        encoded.emplace_back(Complex(value));
                    result[name] = std::move(encoded);
                }
                else if (name == "density_matrix" || name == "unnormalized_density_matrix")
                {
                    const auto matrix = simulator->GetDensityMatrix(name == "density_matrix");
                    json::array entries;
                    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
                        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
                            entries.emplace_back(Complex(matrix(row, col)));
                    result[name] = json::object{{"dimension", matrix.rows()}, {"row_major", entries}};
                }
                else if (name == "partial_trace")
                {
                    Types::qubits_vector keep;
                    std::set<uint64_t> unique;
                    for (const auto &q : Array(Field(request, "keep_qubits")))
                    {
                        const auto index = UInt(q);
                        Require(index < input.qubits && unique.insert(index).second, "Invalid reduced-state qubit");
                        keep.push_back(index);
                    }
                    Require(keep.size() < 32 && (uint64_t{1} << (2 * keep.size())) <= limit, "Reduced matrix exceeds output bound");
                    const auto matrix = simulator->PartialTrace(keep);
                    json::array entries;
                    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
                        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
                            entries.emplace_back(Complex(matrix(row, col)));
                    result[name] = json::object{{"dimension", matrix.rows()}, {"row_major", entries}};
                }
                else
                    throw Error("invalid_input", "Unknown diagnostic");
            }
        }
        else
        {
            json::array values;
            // Tensor-network backends intentionally do not provide efficient bulk
            // probabilities. An explicitly requested full output still uses the
            // bounded per-index path, just as selected basis states do.
            const bool scalarProbabilities = !amplitudes && simulator->GetSimulationType() == Simulators::SimulationType::kTensorNetwork;
            if (!request.contains("basis_states") && !scalarProbabilities)
            {
                if (amplitudes)
                {
                    for (const auto &value : simulator->GetStateVector())
                        values.emplace_back(Complex(value));
                }
                else
                {
                    for (double value : simulator->AllProbabilities())
                        values.emplace_back(value);
                }
                Require(values.size() == basis.size(), "Backend returned an invalid full state output");
            }
            else
            {
                for (const auto &index : basis)
                    if (amplitudes)
                        values.emplace_back(Complex(simulator->Amplitude(UInt(index))));
                    else
                        values.emplace_back(simulator->Probability(UInt(index)));
            }
            result[amplitudes ? "amplitudes" : "probabilities"] = std::move(values);
            result["basis_states"] = std::move(basis);
        }
    }
    else if (operation == "incremental_evolve")
    {
        Prepare(context, circuit, input.clbits);
        Circuits::OperationState state(input.clbits);
        json::array values;
        uint64_t current = 0;
        size_t bond = 0;
        for (const auto &value : Array(Field(request, "steps")))
        {
            const auto step = UInt(value);
            while (current < step)
            {
                additional[0].circuit->ExecuteBD(simulator, state, &bond);
                ++current;
            }
            values.emplace_back(Expectations(simulator, observables));
        }
        result["steps"] = Field(request, "steps");
        result["expectation_values"] = std::move(values);
        result["max_bond_dim_reached"] = bond;
    }
    else if (operation == "checkpoint_batch")
    {
        // noise.prefix: the prefix's noise evolves once, as exact channels. It
        // draws from its own generator, so the suffixes' noise stream is the
        // same as with an ideal prefix.
        auto prefixCircuit = circuit;
        if (noise.enabled && noise.prefix)
        {
            std::mt19937 prefixRng(noise.seed ^ 0x70726566u);
            prefixCircuit = Inject(noise::NoiseInjectionPlan(circuit, noise.model), noise, prefixRng);
        }
        Prepare(context, prefixCircuit, input.clbits);
        simulator->SaveState();
        json::array outputs;
        size_t bond = 0;
        // Dynamic suffixes evolve their deterministic head once on a copy of the
        // restored prefix when two more states fit the realization budget. Each
        // copy draws from its own stream derived from the request seed.
        const bool reuse_heads = HeadCopyFits(config, input.qubits);
        const uint64_t head_streams = Simulators::IState::DeriveSeed(*config.seed, 0x6865616473ULL);
        uint64_t head_stream = 0;
        for (const auto &suffix : additional)
        {
            // A suffix continues the saved prefix, so even a leading pure-state
            // reset may branch. Its shots sample those branches exactly.
            const bool stochastic_suffix = HasStochasticEvolution(suffix.circuit, simulator->SupportsQuantumChannels(), false);
            noise.policy.stochastic_execution = noise.policy.stochastic_execution || stochastic_suffix;
            const noise::NoiseInjectionPlan suffixPlan(suffix.circuit, noise.model);
            Counts counts;
            size_t terminal_batches = 0, trajectory_batches = 0;
            const size_t samples = std::min(realizations, shots);
            for (size_t r = 0; r < samples; ++r)
            {
                const auto noisy = Inject(suffixPlan, noise, rng);
                const size_t repetitions = shots / samples + (r < shots % samples ? 1 : 0);
                simulator->RestoreState();
                simulator->SetGatesCounter(0);
                simulator->SetUpcomingGates(noisy->GetOperations());
                Circuits::OperationState terminal_state(input.clbits);
                if (repetitions > 1 && PreservesSavedStateWhileSampling(*simulator) &&
                    SamplePreparedTerminal(
                        *noisy, simulator, repetitions, terminal_state, Simulators::IState::DeriveSeed(*config.seed, r),
                        [&](size_t, const Circuits::OperationState &state) { ++counts[Bits(state.GetAllBits(), input.clbits)]; }, &bond))
                {
                    ++terminal_batches;
                    continue;
                }
                ++trajectory_batches;
                Circuits::OperationState head_state(input.clbits);
                if (reuse_heads && SampleSharedHead(
                                       *noisy, simulator, repetitions, head_state, Simulators::IState::DeriveSeed(head_streams, head_stream++),
                                       [&](size_t, const Circuits::OperationState &state) { ++counts[Bits(state.GetAllBits(), input.clbits)]; }, &bond))
                    continue;
                for (size_t shot = 0; shot < repetitions; ++shot)
                {
                    if (shot > 0)
                        simulator->RestoreState();
                    simulator->SetGatesCounter(0);
                    simulator->SetUpcomingGates(noisy->GetOperations());
                    Circuits::OperationState state(input.clbits);
                    noisy->ExecuteBD(simulator, state, &bond);
                    ++counts[Bits(state.GetAllBits(), input.clbits)];
                }
            }
            outputs.emplace_back(json::object{{"counts", EncodeCounts(counts)},
                                              {"shots", shots},
                                              {"sampling_strategy", terminal_batches && !trajectory_batches ? "terminal_v1"
                                                                    : !terminal_batches                     ? "trajectory_v1"
                                                                                                            : "mixed_v1"},
                                              {"stochastic_execution", stochastic_suffix},
                                              {"realizations", samples}});
        }
        result["results"] = std::move(outputs);
        if (noise.enabled)
            result["noise_scope"] = noise.prefix ? "prefix_and_suffix" : "suffix_only_ideal_prefix";
    }
    else
    {
        // Build U(first)^dagger U(second). Every nonmeasurement operation was
        // checked above; unsupported adjoints fail instead of silently cloning.
        // With observables, mirror_fidelity returns <0|M^dag O M|0> for each Pauli
        // O of the executed mirror M instead of P(|0...0>): 1 for Z-type strings
        // under exact evolution, and lower where the backend truncated.
        double sum = 0, squares = 0;
        std::complex<double> overlap;
        for (size_t r = 0; r < realizations; ++r)
        {
            const auto second = operation == "inner_product" ? additional[0].circuit : Inject(plan, noise, rng);
            auto combined = std::make_shared<Circuit>();
            for (const auto &op : second->GetOperations())
                combined->AddOperation(op->Clone());
            for (auto it = circuit->GetOperations().rbegin(); it != circuit->GetOperations().rend(); ++it)
                if (auto adjoint = adjoint_gate(*it))
                    combined->AddOperation(adjoint);
            Prepare(context, combined, input.clbits);
            if (mirrorObservables)
            {
                result["expectation_values"] = Expectations(simulator, observables);
                continue;
            }
            // A mixed state's fidelity to U|0> is P(0...0) after U^dagger. Exact
            // channels need no projection; a pure state's overlap is reported too.
            double fidelity = 0;
            if (operation == "noisy_fidelity" && Mixed(config))
                fidelity = simulator->Probability(0);
            else
            {
                overlap = simulator->ProjectOnZero();
                fidelity = std::norm(overlap);
            }
            sum += fidelity;
            squares += fidelity * fidelity;
        }
        if (operation == "inner_product")
            result["inner_product"] = Complex(overlap);
        else if (!mirrorObservables)
        {
            const auto mean = sum / realizations;
            result["fidelity"] = mean;
            // As in Python noisy_fidelity: one sampled disturbance has no
            // estimable uncertainty, rather than none.
            if (realizations == 1 && noise.enabled && (noise.policy.stochastic_noise || noise.policy.stochastic_execution))
                result["standard_error"] = nullptr;
            else
                result["standard_error"] =
                    realizations > 1 ? std::sqrt(std::max(0.0, (squares - realizations * mean * mean) / (realizations - 1) / realizations)) : 0;
        }
    }
    result["execution_metadata"] = Metadata(*metadata_context, config, operation == "execute" || operation == "estimate");
    if (operation == "estimate")
    {
        // Distributed estimates report their evaluation like the local path.
        auto &metadata = result["execution_metadata"].as_object();
        metadata["stochastic_execution"] = stochastic_estimation;
        metadata["measurement_trajectories"] = stochastic_estimation ? realizations : size_t{0};
        metadata["estimate_evaluation"] = stochastic_estimation || noise.policy.stochastic_noise ? "trajectory_average" : "deterministic";
    }
    result["seed"] = *config.seed;
    result["time_taken"] = std::chrono::duration<double>(Clock::now() - start).count();
    result["count_order"] = "classical_bit_0_first";
    result["basis_order"] = "qubit_0_least_significant";
    if (noise.enabled)
        result["noise"] = NoiseMetadata(noise, realizations);
    Require(json::serialize(result).size() <= MaxResultBytes, "Result exceeds native result limit");
    return result;
}
} // namespace MaestroExecution

namespace
{
template <typename Function> char *NativeBoundary(Function function) noexcept
{
    namespace json = boost::json;
    try
    {
        json::object envelope;
        try
        {
            envelope = function();
            envelope["ok"] = true;
        }
        catch (const MaestroExecution::Error &error)
        {
            envelope = {{"ok", false}, {"error", json::object{{"code", error.code}, {"message", error.what()}}}};
        }
        catch (const std::invalid_argument &error)
        {
            envelope = {{"ok", false}, {"error", json::object{{"code", "invalid_input"}, {"message", error.what()}}}};
        }
        catch (const MaestroExecution::Adjoint::MemoryLimitError &error)
        {
            // A refused allocation estimate, not an exhausted heap: keep its reason.
            envelope = {{"ok", false}, {"error", json::object{{"code", "insufficient_resources"}, {"message", error.what()}}}};
        }
        catch (const std::bad_alloc &)
        {
            // Even allocating an error can fail: the outer serialization fallback is
            // null.
            try
            {
                envelope = {{"ok", false}, {"error", json::object{{"code", "insufficient_resources"}, {"message", "Allocation failed"}}}};
            }
            catch (...)
            {
                return nullptr;
            }
        }
        catch (const std::exception &error)
        {
            envelope = {{"ok", false}, {"error", json::object{{"code", "native_failure"}, {"message", error.what()}}}};
        }
        catch (...)
        {
            try
            {
                envelope = {{"ok", false}, {"error", json::object{{"code", "native_failure"}, {"message", "Unknown native exception"}}}};
            }
            catch (...)
            {
                return nullptr;
            }
        }
        try
        {
            envelope["schema_version"] = MaestroExecution::SchemaVersion;
            auto value = json::serialize(envelope);
            // max_result_bytes bounds the complete document, envelope included,
            // so a transport can size its frames from the advertised limit alone.
            if (value.size() > MaestroExecution::MaxResultBytes)
            {
                envelope = {{"ok", false},
                            {"error", json::object{{"code", "invalid_input"}, {"message", "Result exceeds native result limit"}}},
                            {"schema_version", MaestroExecution::SchemaVersion}};
                value = json::serialize(envelope);
            }
            auto *output = new char[value.size() + 1];
            std::memcpy(output, value.c_str(), value.size() + 1);
            return output;
        }
        catch (...)
        {
            return nullptr;
        }
    }
    catch (...)
    {
        return nullptr;
    }
}

boost::json::object ParseRequest(const char *request)
{
    MaestroExecution::Require(request != nullptr, "Request is null");
    const auto length = std::strlen(request);
    MaestroExecution::Require(length <= MaestroExecution::MaxRequestBytes, "Request exceeds native size limit");
    boost::system::error_code error;
    auto value = boost::json::parse(boost::json::string_view(request, length), error);
    MaestroExecution::Require(!error, "Invalid JSON: " + error.message());
    return MaestroExecution::Object(value);
}

// Packed shot rows: ceil(width / 64) words per row, classical bit b at bit
// b % 64 of the row's word b / 64.
size_t RowWords(size_t width)
{
    return (width + 63) / 64;
}

size_t PackedWords(size_t shots, size_t width)
{
    const size_t words = RowWords(width);
    MaestroExecution::Require(!words || shots <= std::numeric_limits<size_t>::max() / words, "Packed shot rows overflow");
    return shots * words;
}

void PackRows(const uint8_t *bytes, size_t shots, size_t width, unsigned long long *out)
{
    const size_t words = RowWords(width);
    std::fill(out, out + shots * words, 0ULL);
    for (size_t shot = 0; shot < shots; ++shot)
    {
        const uint8_t *row = bytes + shot * width;
        unsigned long long *target = out + shot * words;
        for (size_t bit = 0; bit < width; ++bit)
            if (row[bit])
                target[bit / 64] |= 1ULL << (bit % 64);
    }
}

// An execute request with return_samples, its rows packed into the caller's
// buffer instead of JSON. Without a buffer, only validates and sizes it.
boost::json::object RunSamples(const boost::json::object &request, unsigned long long *rows, size_t capacity)
{
    using namespace MaestroExecution;
    Require(request.contains("operation") && String(request.at("operation")) == "execute", "Packed shot rows require operation execute");
    auto copy = request;
    if (!copy.contains("execution"))
        copy["execution"] = boost::json::object{};
    Require(copy["execution"].is_object(), "execution must be an object");
    copy["execution"].as_object()["return_samples"] = true;
    const auto input = ParseCircuit(Object(Field(copy, "circuit")));
    const size_t shots = UInt(copy["execution"].as_object(), "shots", 1024);
    const size_t words = PackedWords(shots, input.clbits);
    ShotRows sink;
    if (!rows)
    {
        Run(copy, true, 0, &sink);
        return {{"schema_version", SchemaVersion},     {"operation", "execute"}, {"valid", true}, {"shots", shots}, {"num_clbits", input.clbits},
                {"row_words", RowWords(input.clbits)}, {"required_words", words}};
    }
    Require(capacity >= words, "capacity must be at least shots * row_words words");
    auto result = Run(copy, false, 0, &sink);
    if (sink.shots != shots || sink.width != input.clbits || sink.data.size() != shots * input.clbits)
        throw Error("native_failure", "Backend did not return the requested shot rows");
    PackRows(sink.data.data(), sink.shots, sink.width, rows);
    result["row_words"] = RowWords(input.clbits);
    return result;
}

// A noisy Clifford circuit compiled once to reference frames: "circuit", and
// optionally "simulator" and "noise" (incoherent Pauli-mixture channels) as in
// an execute request.
std::unique_ptr<MaestroExecution::ReferenceFrameSampler> CompileFrameSampler(const boost::json::object &request)
{
    using namespace MaestroExecution;
    Keys(request, {"schema_version", "circuit", "simulator", "noise"});
    Require(UInt(Field(request, "schema_version")) == SchemaVersion, "Unsupported native schema_version");
    // Accept exactly what an execute request accepts: the same circuit,
    // simulator and noise validation. Frames run on the QCSim stabilizer, so
    // an omitted simulator means that one rather than execute's statevector.
    boost::json::object execute{{"schema_version", SchemaVersion},
                                {"operation", "execute"},
                                {"circuit", Field(request, "circuit")},
                                {"simulator", boost::json::object{{"backend", "qcsim"}, {"method", "stabilizer"}}}};
    for (const char *key : {"simulator", "noise"})
        if (const auto *value = request.if_contains(key))
            execute[key] = *value;
    Run(execute, true);
    const auto config = ParseConfig(Sub(execute, "simulator"));
    const auto input = ParseCircuit(Object(Field(request, "circuit")));
    const auto noise = ParseNoise(Sub(request, "noise"), config, input.qubits);
    Supported(AdmitsReferenceFrames(config),
              "Reference frames require the qcsim stabilizer: a stabilizer method, or automatic selection whose candidates include it");
    Supported(!noise.enabled || noise.mode == "pauli" || noise.mode == "combined", "Reference frames apply pauli or combined noise only");
    Supported(noise.model.inject_noise_covers_model(), "Reference frames cannot represent coherent, correlated or crosstalk noise");
    auto channels = noise::inject_exact_noise(input.circuit, noise.model);
    noise::attach_readout_error(channels, noise.model);
    auto plan = CompileReferenceSampler(channels, input.qubits, input.clbits);
    Supported(plan.has_value(), "Reference frames do not apply: they need measurements, Clifford gates and Pauli-mixture channels, "
                                "without classical control");
    return std::make_unique<ReferenceFrameSampler>(std::move(*plan));
}
} // namespace

extern "C"
{
    char *MaestroRunRequestJson(const char *request)
    {
        return NativeBoundary([&] { return MaestroExecution::Run(ParseRequest(request)); });
    }

    char *MaestroValidateRequestJson(const char *request)
    {
        return NativeBoundary([&] { return MaestroExecution::Run(ParseRequest(request), true); });
    }

    char *MaestroRunRequestSamples(const char *request, unsigned long long *rows, size_t capacity)
    {
        return NativeBoundary([&] { return RunSamples(ParseRequest(request), rows, capacity); });
    }

    void *MaestroCreateReferenceFrameSampler(const char *request, char **error)
    {
        if (error)
            *error = nullptr;
        std::unique_ptr<MaestroExecution::ReferenceFrameSampler> created;
        char *envelope = NativeBoundary([&] {
            created = CompileFrameSampler(ParseRequest(request));
            return boost::json::object{{"num_qubits", created->Qubits()}, {"num_clbits", created->Width()}};
        });
        if (created)
        {
            FreeResult(envelope);
            return created.release();
        }
        if (error)
            *error = envelope;
        else
            FreeResult(envelope);
        return nullptr;
    }

    void MaestroDestroyReferenceFrameSampler(void *sampler)
    {
        delete static_cast<MaestroExecution::ReferenceFrameSampler *>(sampler);
    }

    int MaestroReferenceFrameSamplerShape(void *sampler, size_t *qubits, size_t *clbits)
    {
        if (!sampler || !qubits || !clbits)
            return 0;
        const auto &plan = *static_cast<const MaestroExecution::ReferenceFrameSampler *>(sampler);
        *qubits = plan.Qubits();
        *clbits = plan.Width();
        return 1;
    }

    int MaestroReferenceFrameSampleBits(void *sampler, size_t shots, unsigned long long seed, unsigned long long *rows, size_t capacity)
    {
        if (!sampler)
            return 0;
        try
        {
            const auto &plan = *static_cast<const MaestroExecution::ReferenceFrameSampler *>(sampler);
            if (!shots)
                return 1;
            if (!rows || capacity < PackedWords(shots, plan.Width()))
                return 0;
            const auto bytes = plan.Sample(shots, seed);
            PackRows(bytes.data(), shots, plan.Width(), rows);
            return 1;
        }
        catch (...)
        {
            return 0;
        }
    }

    int MaestroReferenceFrameSampleCountsBits(void *sampler, size_t shots, unsigned long long seed, unsigned long long *rows, unsigned long long *counts,
                                              size_t capacity, size_t *written)
    {
        if (!sampler || !written)
            return 0;
        try
        {
            *written = 0;
            const auto &plan = *static_cast<const MaestroExecution::ReferenceFrameSampler *>(sampler);
            if (!shots)
                return 1;
            const size_t words = RowWords(plan.Width());
            if (!rows || !counts || capacity < shots || (words && capacity > std::numeric_limits<size_t>::max() / words))
                return 0;
            const auto outcomes = plan.SampleCounts(shots, seed);
            for (const auto &[bits, count] : outcomes)
            {
                unsigned long long *target = rows + *written * words;
                std::fill(target, target + words, 0ULL);
                for (size_t bit = 0; bit < bits.size(); ++bit)
                    if (bits[bit])
                        target[bit / 64] |= 1ULL << (bit % 64);
                counts[*written] = count;
                ++*written;
            }
            return 1;
        }
        catch (...)
        {
            *written = 0;
            return 0;
        }
    }

    char *MaestroGetCapabilitiesJson()
    {
        return NativeBoundary([] { return MaestroExecution::Capabilities(); });
    }

    char *MaestroFinalizeDistributedMpiGpuJson()
    {
        return NativeBoundary([] {
#ifdef __linux__
            Simulators::SimulatorsFactory::FinalizeDistributedMpiGpuBackend();
#endif
            return boost::json::object{};
        });
    }
}
