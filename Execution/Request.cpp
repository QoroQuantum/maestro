// Versioned native computational API. No Python types or runtime are used.
#include <complex>
#include <climits>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include "Request.h"
#include "maestrolib/Interface.h"
#include "NoiseJson.h"
#include "CircuitHelpers.h"
#include "maestrolib/Json.h"
#include "qasm/QasmCirc.h"

namespace MaestroExecution {
using Circuit = Circuits::Circuit<double>;
using CircuitPtr = std::shared_ptr<Circuit>;
using Clock = std::chrono::steady_clock;

namespace {
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
                                       "validate"};
struct ParsedCircuit {
  CircuitPtr circuit;
  size_t qubits, clbits;
};
ParsedCircuit ParseCircuit(const json::object& object) {
  Keys(object, {"format", "source", "num_qubits", "num_clbits", "parameters"});
  const size_t qubits = UInt(Field(object, "num_qubits"));
  const size_t clbits = UInt(object, "num_clbits", qubits);
  Require(qubits > 0 && qubits <= 1000000, "num_qubits must be in [1,1000000]");
  Require(clbits <= 1000000, "num_clbits exceeds the result limit");
  const auto format = String(object, "format", "openqasm");
  CircuitPtr circuit;
  if (format == "openqasm") {
    const auto source = String(Field(object, "source"));
    Require(!source.empty(), "Circuit source is empty");
    std::unordered_map<std::string, double> params;
    for (const auto& entry : Sub(object, "parameters"))
      params.emplace(std::string(entry.key()), Number(entry.value()));
    qasm::QasmToCirc<> parser;
    circuit = parser.ParseAndTranslateWithParams(source, params);
    Require(circuit && !parser.Failed(),
            "QASM parse error: " + parser.GetErrorMessage());
  } else if (format == "instructions") {
    Require(!object.contains("parameters"),
            "Instruction arrays do not accept parameter bindings");
    const auto& array = Array(Field(object, "source"));
    circuit = std::make_shared<Circuit>();
    Json::JsonParserMaestro<> parser;
    for (const auto& value : array) {
      auto instruction = Object(value);
      const auto name = String(Field(instruction, "name"));
      Keys(instruction, {"name", "qubits", "clbits", "params",
                         "conditional_reg", "duration", "operators"});
      Types::qubits_vector targets;
      std::set<uint64_t> unique;
      for (const auto& q : Array(Field(instruction, "qubits"))) {
        Require(UInt(q) < qubits && unique.insert(UInt(q)).second,
                "Invalid instruction qubit");
        targets.push_back(UInt(q));
      }
      Require(!targets.empty(), "Instruction requires qubits");
      if (name == "reset" || name == "delay" || name == "kraus") {
        Require(!instruction.contains("conditional_reg") &&
                    !instruction.contains("params") &&
                    !instruction.contains("clbits"),
                "Unsupported fields on native channel/reset/delay");
        Require(name == "delay" || !instruction.contains("duration"),
                "duration applies only to delay instructions");
        Require(name == "kraus" || !instruction.contains("operators"),
                "operators applies only to Kraus instructions");
        if (name == "reset")
          circuit->AddOperation(
              Circuits::CircuitFactory<>::CreateReset(targets));
        else if (name == "delay") {
          const auto duration = Number(Field(instruction, "duration"));
          Require(duration >= 0, "Negative delay");
          for (auto q : targets)
            circuit->AddOperation(
                Circuits::CircuitFactory<>::CreateDelay(q, duration));
        } else {
          Require(targets.size() <= 2,
                  "Kraus instructions require one or two qubits");
          const size_t dim = size_t{1} << targets.size();
          Simulators::QuantumChannel::KrausOperators operators;
          for (const auto& matrix : Array(Field(instruction, "operators"))) {
            const auto& entries = Array(matrix);
            Require(entries.size() == dim * dim,
                    "Kraus matrices must be flat row-major arrays");
            Eigen::MatrixXcd op(dim, dim);
            for (size_t r = 0; r < dim; ++r)
              for (size_t c = 0; c < dim; ++c)
                op(r, c) = Complex(entries[r * dim + c]);
            operators.push_back(std::move(op));
          }
          circuit->AddOperation(
              std::make_shared<Circuits::QuantumChannelOperation<>>(
                  targets, Simulators::QuantumChannel(operators)));
        }
      } else {
        Require(!instruction.contains("duration") &&
                    !instruction.contains("operators"),
                "Unsupported gate/measurement fields");
        if (const auto* params = instruction.if_contains("params"))
          for (const auto& param : Array(*params)) Number(param);
        const std::map<std::string, size_t> arities{
            {"p", 1},   {"rx", 1},  {"ry", 1}, {"rz", 1}, {"u", 3},
            {"u1", 1},  {"u2", 2},  {"u3", 3}, {"cp", 1}, {"crx", 1},
            {"cry", 1}, {"crz", 1}, {"cu", 4}};
        const auto params = instruction.if_contains("params")
                                ? Array(instruction.at("params"))
                                : json::array{};
        Require(params.size() == (arities.count(name) ? arities.at(name) : 0),
                "Incorrect gate parameter count");
        Require(name == "measure" || !instruction.contains("clbits"),
                "clbits applies only to measurement");
        if (const auto* condition =
                instruction.if_contains("conditional_reg")) {
          const auto bit =
              condition->is_array()
                  ? (Require(Array(*condition).size() == 1,
                             "Conditional register requires one bit"),
                     UInt(Array(*condition)[0]))
                  : UInt(*condition);
          Require(bit < clbits && name != "measure", "Invalid conditional bit");
        }
        if (name == "id") {
          instruction["name"] = "p";
          instruction["params"] = json::array{0};
        } else if (name == "u1") {
          instruction["name"] = "p";
        } else if (name == "u2") {
          instruction["name"] = "u";
          instruction["params"] =
              json::array{std::acos(-1.0) / 2, params[0], params[1]};
        } else if (name == "u3") {
          instruction["name"] = "u";
        }
        CircuitPtr parsed;
        try {
          parsed = parser.ParseCircuit(
              json::serialize(json::array{instruction}).c_str());
        } catch (const std::runtime_error& error) {
          // The legacy instruction parser uses runtime_error for malformed
          // gates/measurements. Translate only this parsing boundary, never
          // exceptions from simulator execution or resource allocation.
          throw Error("invalid_input",
                      "Invalid instruction: " + std::string(error.what()));
        }
        Require(parsed && parsed->GetOperations().size() == 1,
                "Invalid instruction");
        circuit->AddOperation(parsed->GetOperations().front());
      }
    }
  } else
    throw Error("unsupported_capability", "Unknown circuit format: " + format);
  for (const auto q : circuit->GetQubits())
    Require(q < qubits, "Circuit qubit exceeds num_qubits");
  for (const auto c : circuit->GetBits())
    Require(c < clbits, "Circuit bit exceeds num_clbits");
  return {std::move(circuit), qubits, clbits};
}
CircuitPtr Unmeasured(const CircuitPtr& circuit, bool unitary = false,
                      const std::string& context = "Overlap/fidelity") {
  auto result = std::make_shared<Circuit>();
  bool measured = false;
  for (const auto& op : circuit->GetOperations()) {
    const auto type = op->GetType();
    if (type == Circuits::OperationType::kMeasurement) {
      measured = true;
      continue;
    }
    Supported(!measured, "This query requires terminal measurements only");
    if (unitary)
      Supported(type == Circuits::OperationType::kGate ||
                    type == Circuits::OperationType::kDelay,
                context + " requires a unitary circuit");
    result->AddOperation(op->Clone());
  }
  return result;
}
std::vector<std::string> Observables(const json::object& request,
                                     size_t qubits) {
  std::vector<std::string> result;
  if (const auto* values = request.if_contains("observables")) {
    for (const auto& item : Array(*values)) {
      const auto pauli = String(item);
      Require(pauli.size() == qubits &&
                  pauli.find_first_not_of("IXYZ") == std::string::npos,
              "Observables must be I/X/Y/Z strings of num_qubits characters "
              "(q0 first)");
      result.push_back(pauli);
    }
  }
  Require(result.size() <= 100000, "Too many observables");
  return result;
}
struct Context {
  unsigned long handle = 0;
  std::shared_ptr<Network::INetwork<double>> network;
  explicit Context(const ParsedCircuit& circuit,
                   const SimulatorConfig& config) {
    auto* maestro = static_cast<Maestro*>(GetMaestroObjectWithMute());
    if (!maestro) throw Error("native_failure", "Cannot initialize Maestro");
    handle = maestro->CreateSimpleSimulator(circuit.qubits, circuit.clbits);
    try {
      network = ConfigureNetwork(handle, config);
      if (!network || !network->GetSimulator())
        throw Error("backend_unavailable",
                    "Requested backend could not be created");
      if (config.fixed_backend &&
          (network->GetSimulator()->GetType() != config.simulator_type ||
           network->GetSimulator()->GetSimulationType() !=
               config.simulation_type))
        throw Error("backend_unavailable",
                    "Requested fixed backend/method is unavailable; fallback "
                    "is disabled");
      if (config.native_options.count("gpu_device") &&
          network->GetSimulator()->GetGpuDevice() !=
              std::stoi(config.native_options.at("gpu_device")))
        throw Error("backend_unavailable",
                    "Requested GPU device was not selected");
    } catch (...) {
      if (handle) DestroySimpleSimulator(handle);
      throw;
    }
  }
  ~Context() {
    network.reset();
    if (handle) DestroySimpleSimulator(handle);
  }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  auto simulator() const { return network->GetSimulator(); }
};
CircuitPtr Inject(const CircuitPtr& circuit, const NoiseConfig& noise,
                  std::mt19937& rng) {
  if (!noise.enabled || noise.mode == "analytical") return circuit;
  CircuitPtr noisy;
  if (noise.mode == "coherent")
    noisy = noise::inject_coherent_noise(circuit, noise.model, rng);
  else if (noise.mode == "pauli")
    noisy = noise.exact ? noise::inject_exact_noise(circuit, noise.model)
                        : noise::inject_noise(circuit, noise.model, rng);
  else
    noisy = noise.exact
                ? noise::inject_combined_noise_exact(circuit, noise.model, rng)
                : noise::inject_combined_noise(circuit, noise.model, rng);
  // Use the same measurement-time path as Python, including conditional
  // measurements and readout results consumed by subsequent classical control.
  noise::attach_readout_error(noisy, noise.model);
  return noisy;
}
void ValidateNoisyCircuit(const CircuitPtr& circuit) {
  for (const auto& op : circuit->GetOperations()) {
    const auto type = op->GetType();
    Supported(type == Circuits::OperationType::kGate ||
                  type == Circuits::OperationType::kConditionalGate ||
                  type == Circuits::OperationType::kMeasurement ||
                  type == Circuits::OperationType::kConditionalMeasurement ||
                  type == Circuits::OperationType::kReset ||
                  type == Circuits::OperationType::kDelay,
              "Noise injection requires a flattened circuit of gates, "
              "measurements, resets and delays");
  }
}
using Counts = std::map<std::string, uint64_t>;
json::object EncodeCounts(const Counts& counts) {
  json::object result;
  for (const auto& entry : counts) result[entry.first] = entry.second;
  return result;
}
std::string Bits(const std::vector<bool>& values, size_t size) {
  std::string bits(size, '0');
  for (size_t i = 0; i < values.size() && i < size; ++i)
    if (values[i]) bits[i] = '1';
  return bits;
}
json::array Expectations(
    const std::shared_ptr<Simulators::ISimulator>& simulator,
    const std::vector<std::string>& observables) {
  json::array result;
  for (const auto& pauli : observables)
    result.emplace_back(simulator->ExpectationValue(pauli));
  return result;
}
void Prepare(Context& context, const CircuitPtr& circuit, size_t bits) {
  Circuits::OperationState state(bits);
  auto simulator = context.simulator();
  simulator->Reset();
  simulator->SetGatesCounter(0);
  simulator->SetUpcomingGates(circuit->GetOperations());
  size_t bond = 0;
  circuit->ExecuteBD(simulator, state, &bond);
}
json::array Basis(const json::object& request, size_t qubits, size_t limit) {
  Supported(qubits < 63, "Basis-index queries support fewer than 63 qubits");
  json::array states;
  if (const auto* values = request.if_contains("basis_states"))
    states = Array(*values);
  else {
    const auto count = uint64_t{1} << qubits;
    Require(count <= limit,
            "Full state output exceeds max_output_elements; request selected "
            "basis_states");
    for (uint64_t i = 0; i < count; ++i) states.emplace_back(i);
  }
  Require(states.size() <= limit,
          "Selected state output exceeds max_output_elements");
  for (const auto& value : states)
    Require(UInt(value) < (uint64_t{1} << qubits), "Basis index out of range");
  return states;
}
void SameSize(const ParsedCircuit& first, const ParsedCircuit& other) {
  Require(first.qubits == other.qubits && first.clbits == other.clbits,
          "Circuits in this operation must have identical register sizes");
}
json::object Metadata(Context& context, const SimulatorConfig& config,
                      bool executed) {
  auto simulator = context.simulator();
  json::object result{
      {"backend", BackendName(executed ? context.network->GetLastSimulatorType()
                                       : simulator->GetType())},
      {"method", MethodName(executed ? context.network->GetLastSimulationType()
                                     : simulator->GetSimulationType())},
      {"selection", config.fixed_backend ? "fixed" : "automatic"}};
  json::object options;
  for (const auto& entry : config.native_options)
    options[entry.first] = entry.second;
  for (const auto& entry : config.distributed_options)
    options[entry.first] = entry.second;
  options["seed"] = config.seed.value_or(0);
  result["configured_options"] = std::move(options);
  if (Simulators::IsGpuSimulator(simulator->GetType()))
    result["gpu_device"] = simulator->GetGpuDevice();
  if (Simulators::IsDistributedGpuSimulator(simulator->GetType())) {
    json::object placement;
    for (const char* key :
         {"distributed_shard_devices", "distributed_configured_global_qubits",
          "distributed_qubit_layout", "distributed_backend", "precision"})
      placement[key] = simulator->GetConfiguration(key);
    result["distribution"] = std::move(placement);
  }
  return result;
}
}  // namespace

json::object Capabilities() {
  json::array backends, options, noiseKinds, names;
  for (const auto& entry : Backends()) {
    json::array methods;
    for (const auto& method : Methods())
      if (Accepts(entry.second, method.second))
        methods.emplace_back(method.first);
    backends.emplace_back(json::object{
        {"name", entry.first},
        {"legacy_id", static_cast<int>(entry.second)},
        {"methods", methods},
        {"compiled", true},
        {"readiness",
         entry.first == "qcsim" || entry.first == "composite_qcsim" ||
                 entry.first == "aer" || entry.first == "composite_aer"
             ? "available"
             : "requires_runtime_probe"}});
  }
  for (const auto& option : Options())
    options.emplace_back(json::object{{"name", option.name},
                                      {"native_name", option.native_name},
                                      {"type", option.type},
                                      {"family", option.family}});
  for (const auto& entry : NoiseKinds()) noiseKinds.emplace_back(entry.first);
  for (const auto& operation : operations) names.emplace_back(operation);
  return {{"schema_version", SchemaVersion},
          {"api", "maestro.native.request"},
          {"operations", names},
          {"backends", backends},
          {"options", options},
          {"noise_channels", noiseKinds},
          {"circuit_formats", json::array{"openqasm", "instructions"}},
          {"python_required", false},
          {"max_request_bytes", MaxRequestBytes},
          {"max_result_bytes", MaxResultBytes},
          {"max_output_elements", 1048576},
          {"count_order", "classical_bit_0_first"},
          {"basis_order", "qubit_0_least_significant"},
          {"complex_encoding", "[real, imaginary]"},
          {"mpi_lifecycle", "externally_initialized_collective"}};
}

json::object Run(const json::object& request, bool validate, unsigned depth) {
  Require(depth <= 4, "Batch nesting exceeds four levels");
  Require(UInt(Field(request, "schema_version")) == SchemaVersion,
          "Unsupported native schema_version");
  Keys(request,
       {"schema_version", "operation", "circuit", "simulator", "execution",
        "noise", "observables", "outputs", "basis_states", "target_state",
        "other_circuit", "step_circuit", "steps", "suffixes", "requests",
        "diagnostics", "maintenance", "keep_qubits", "max_output_elements"});
  const auto operation = String(Field(request, "operation"));
  Supported(operations.count(operation), "Unknown operation: " + operation);
  // A known field for another operation must not bypass nested validation or
  // appear to configure a computation which never reads it.
  const std::map<std::string, std::set<std::string>> operationFields{
      {"observables", {"estimate", "incremental_evolve"}},
      {"basis_states", {"statevector", "amplitudes", "probabilities"}},
      {"target_state", {"state_probability"}},
      {"other_circuit", {"inner_product"}},
      {"step_circuit", {"incremental_evolve"}},
      {"steps", {"incremental_evolve"}},
      {"suffixes", {"checkpoint_batch"}},
      {"requests", {"batch"}},
      {"diagnostics", {"diagnostics"}},
      {"maintenance", {"diagnostics"}},
      {"keep_qubits", {"diagnostics"}},
      {"max_output_elements",
       {"statevector", "amplitudes", "probabilities", "diagnostics",
        "incremental_evolve"}}};
  for (const auto& entry : request) {
    const auto field = std::string(entry.key());
    const auto rule = operationFields.find(field);
    Require(rule == operationFields.end() || rule->second.count(operation),
            "Field " + field + " does not apply to " + operation);
  }
  if (operation == "batch") {
    Keys(request, {"schema_version", "operation", "requests"});
    const auto& requests = Array(Field(request, "requests"));
    Require(!requests.empty() && requests.size() <= 256,
            "A batch needs 1..256 requests");
    // Validate every member before starting any numerical work.
    for (const auto& item : requests) Run(Object(item), true, depth + 1);
    json::array results;
    if (!validate)
      for (const auto& item : requests) {
        results.emplace_back(Run(Object(item), false, depth + 1));
        Require(json::serialize(results).size() <= MaxResultBytes,
                "Batch results exceed native result limit");
      }
    return {{"schema_version", SchemaVersion},
            {"operation", operation},
            {"results", results}};
  }
  auto config = ParseConfig(Sub(request, "simulator"));
  const auto& execution = Sub(request, "execution");
  Keys(execution, {"shots", "seed"});
  Require(!execution.contains("shots") || operation == "execute" ||
              operation == "checkpoint_batch",
          "execution.shots does not apply to " + operation);
  const size_t shots = UInt(execution, "shots", 1024);
  Require(shots > 0 && shots <= 1000000000, "shots must be in [1,1000000000]");
  if (execution.contains("seed")) {
    Require(!Sub(Sub(request, "simulator"), "options").contains("seed"),
            "Specify seed only once");
    config.seed = UInt(Field(execution, "seed"));
  }
  // An explicit simulator seed wins; otherwise the public noise seed also
  // controls measurement/readout randomness, as in Python's noisy execution.
  if (!execution.contains("seed") &&
      !Sub(Sub(request, "simulator"), "options").contains("seed") &&
      Sub(request, "noise").contains("seed"))
    config.seed = UInt(Field(Sub(request, "noise"), "seed"));
  if (!config.seed) config.seed = 0;
  const auto input = ParseCircuit(Object(Field(request, "circuit")));
  Supported(!Simulators::IsDistributedGpuSimulator(config.simulator_type) ||
                input.qubits < 63,
            "Distributed GPU requires fewer than 63 qubits");
  Supported(config.fixed_backend || operation == "execute" ||
                operation == "estimate" || operation == "validate",
            "State/query workflows require fixed selection");
  const auto& distribution = Sub(Sub(request, "simulator"), "distribution");
  if (const auto* globals = distribution.if_contains("global_qubits"))
    for (const auto& q : Array(*globals))
      Require(UInt(q) < input.qubits, "Global qubit exceeds register");
  if (const auto* devices = distribution.if_contains("devices"))
    Require(Array(*devices).size() < (uint64_t{1} << input.qubits),
            "Distribution requires at least one local qubit");
  const auto noise = ParseNoise(Sub(request, "noise"), config, input.qubits);
  for (const auto& op : input.circuit->GetOperations())
    Supported(op->GetType() != Circuits::OperationType::kQuantumChannel ||
                  Mixed(config),
              "Explicit Kraus instructions require density_matrix or MPO");
  const auto observables = Observables(request, input.qubits);
  const auto limit = UInt(request, "max_output_elements", 65536);
  Require(limit > 0 && limit <= 1048576,
          "max_output_elements must be in [1,1048576]");
  Require(!request.contains("outputs"),
          "Use operation-specific queries instead of outputs");
  if (operation == "estimate" || operation == "incremental_evolve")
    Require(!observables.empty(), "This operation requires observables");
  Supported(!noise.enabled || config.fixed_backend,
            "Noise requires fixed backend selection");
  Supported(noise.mode != "analytical" || operation == "estimate",
            "Analytical noise is an estimator approximation");
  Supported(!noise.model.has_readout_error() || operation == "execute" ||
                operation == "checkpoint_batch",
            "Readout noise applies to measurement counts only");
  if (noise.enabled) ValidateNoisyCircuit(input.circuit);
  auto circuit =
      operation == "execute" || operation == "checkpoint_batch"
          ? input.circuit
          : Unmeasured(input.circuit, operation == "inner_product" ||
                                          operation == "mirror_fidelity" ||
                                          operation == "noisy_fidelity");
  std::vector<ParsedCircuit> additional;
  if (operation == "inner_product") {
    additional.push_back(ParseCircuit(Object(Field(request, "other_circuit"))));
    SameSize(input, additional.back());
    additional.back().circuit = Unmeasured(additional.back().circuit, true);
  }
  if (operation == "incremental_evolve") {
    Supported(!noise.enabled,
              "Incremental noisy evolution must be supplied as explicit "
              "circuit channels");
    additional.push_back(ParseCircuit(Object(Field(request, "step_circuit"))));
    SameSize(input, additional.back());
    additional.back().circuit = Unmeasured(additional.back().circuit);
    const auto& steps = Array(Field(request, "steps"));
    Require(!steps.empty() && steps.size() <= limit,
            "Invalid observation steps");
    uint64_t previous = 0;
    for (const auto& value : steps) {
      const auto step = UInt(value);
      Require(step >= previous && step <= 1000000000,
              "steps must be ordered nonnegative integers");
      previous = step;
    }
  }
  if (operation == "checkpoint_batch") {
    circuit = Unmeasured(input.circuit, true, "checkpoint_batch prefix");
    Supported(noise.mode != "analytical",
              "Checkpoint suffixes require explicit noise evolution");
    const auto& suffixes = Array(Field(request, "suffixes"));
    Require(!suffixes.empty() && suffixes.size() <= 256,
            "Invalid suffix count");
    for (const auto& suffix : suffixes) {
      additional.push_back(ParseCircuit(Object(suffix)));
      SameSize(input, additional.back());
      if (noise.enabled) ValidateNoisyCircuit(additional.back().circuit);
    }
  }
  for (const auto& other : additional)
    for (const auto& op : other.circuit->GetOperations())
      Supported(op->GetType() != Circuits::OperationType::kQuantumChannel ||
                    Mixed(config),
                "Explicit Kraus instructions require density_matrix or MPO");
  const bool amplitudes =
      operation == "statevector" || operation == "amplitudes";
  const bool query = amplitudes || operation == "probabilities" ||
                     operation == "state_probability" ||
                     operation == "diagnostics";
  if (amplitudes || operation == "inner_product" ||
      operation == "mirror_fidelity" || operation == "noisy_fidelity")
    Supported(!Mixed(config),
              "A mixed state has no unique statevector or pure-state overlap");
  Supported(
      !noise.enabled || operation == "execute" || operation == "estimate" ||
          (query && noise.exact && noise.realizations == 1) ||
          operation == "noisy_fidelity" || operation == "checkpoint_batch",
      "This operation requires a single exact noisy state or an explicit "
      "batch");
  if (operation == "noisy_fidelity")
    Supported(noise.enabled && noise.mode == "coherent",
              "Noisy fidelity supports unitary coherent noise only");
  json::array basis;
  if (amplitudes || operation == "probabilities")
    basis = Basis(request, input.qubits, limit);
  uint64_t target = 0;
  if (operation == "state_probability") {
    Supported(input.qubits < 63,
              "Basis-index state probability supports fewer than 63 qubits");
    const auto state = String(Field(request, "target_state"));
    Require(
        state.size() == input.qubits &&
            state.find_first_not_of("01") == std::string::npos,
        "target_state must be a q0-first bitstring of num_qubits characters");
    for (size_t q = 0; q < state.size(); ++q)
      if (state[q] == '1') target |= uint64_t{1} << q;
  }
  if (operation == "diagnostics") {
    Supported(Mixed(config),
              "Mixed-state diagnostics require density_matrix or MPO");
    if (request.contains("keep_qubits")) {
      bool partial = false;
      if (const auto* values = request.if_contains("diagnostics"))
        for (const auto& value : Array(*values))
          partial |= String(value) == "partial_trace";
      Require(partial, "keep_qubits requires the partial_trace diagnostic");
    }
    if (const auto* values = request.if_contains("diagnostics"))
      for (const auto& value : Array(*values)) {
        const auto name = String(value);
        Require(std::set<std::string>{"trace", "purity", "trace_of_square",
                                      "hermiticity_residual", "is_hermitian",
                                      "partial_trace"}
                    .count(name),
                "Unknown diagnostic");
        if (name == "partial_trace") {
          const auto& keep = Array(Field(request, "keep_qubits"));
          Require(
              keep.size() < 32 && (uint64_t{1} << (2 * keep.size())) <= limit,
              "Reduced matrix exceeds output bound");
          std::set<uint64_t> unique;
          for (const auto& q : keep)
            Require(UInt(q) < input.qubits && unique.insert(UInt(q)).second,
                    "Invalid reduced-state qubit");
        }
      }
    if (const auto* actions = request.if_contains("maintenance"))
      for (const auto& action : Array(*actions)) {
        const auto name = String(action);
        Require(name == "restore_trace" || name == "hermitize" ||
                    name == "trim" || name == "recanonicalize",
                "Unknown maintenance action");
        Supported((name != "trim" && name != "recanonicalize") ||
                      config.simulation_type == Method::kMatrixProductOperator,
                  "This maintenance action requires MPO");
      }
  }
  if (validate || operation == "validate")
    return {{"schema_version", SchemaVersion},
            {"operation", operation},
            {"valid", true},
            {"backend", BackendName(config.simulator_type)},
            {"method", MethodName(config.simulation_type)}};
  const auto start = Clock::now();
  Context context(input, config);
  auto simulator = context.simulator();
  // Validation remains silent; warn once per execution, not per realization.
  if (!noise.thermal_approximation_warning.empty())
    std::cerr << "Warning: " << noise.thermal_approximation_warning << '\n';
  std::mt19937 rng(noise.seed);
  json::object result{{"schema_version", SchemaVersion},
                      {"operation", operation},
                      {"num_qubits", input.qubits},
                      {"num_clbits", input.clbits}};
  size_t realizations =
      noise.enabled && noise.mode != "analytical" ? noise.realizations : 1;
  if (operation == "execute" || operation == "checkpoint_batch")
    realizations = std::min(realizations, shots);
  if (operation == "execute") {
    Counts counts;
    for (size_t r = 0; r < realizations; ++r) {
      auto noisy = Inject(circuit, noise, rng);
      const auto count =
          shots / realizations + (r < shots % realizations ? 1 : 0);
      context.network->Configure(
          "seed",
          std::to_string(Simulators::IState::DeriveSeed(*config.seed, r))
              .c_str());
      const auto raw = context.network->RepeatedExecuteOnHost(noisy, 0, count);
      for (const auto& entry : raw)
        counts[Bits(entry.first, input.clbits)] += entry.second;
    }
    uint64_t total = 0;
    for (const auto& entry : counts) total += entry.second;
    if (total != shots)
      throw Error("native_failure",
                  "Backend did not return the requested number of shots");
    result["counts"] = EncodeCounts(counts);
    result["shots"] = total;
  } else if (operation == "estimate") {
    std::vector<double> sums(observables.size(), 0),
        squares(observables.size(), 0);
    for (size_t r = 0; r < realizations; ++r) {
      const auto noisy = Inject(circuit, noise, rng);
      context.network->Configure(
          "seed",
          std::to_string(Simulators::IState::DeriveSeed(*config.seed, r))
              .c_str());
      const auto values =
          context.network->ExecuteOnHostExpectations(noisy, 0, observables);
      Require(values.size() == sums.size(),
              "Backend returned an invalid expectation vector");
      for (size_t i = 0; i < values.size(); ++i) {
        const double value =
            values[i] * (noise.mode == "analytical"
                             ? noise.model.compute_damping(observables[i])
                             : 1);
        sums[i] += value;
        squares[i] += value * value;
      }
    }
    json::array values, errors;
    for (size_t i = 0; i < sums.size(); ++i) {
      const auto mean = sums[i] / realizations;
      values.emplace_back(mean);
      errors.emplace_back(
          realizations > 1
              ? std::sqrt(
                    std::max(0.0, (squares[i] - realizations * mean * mean) /
                                      (realizations - 1) / realizations))
              : 0.0);
    }
    result["expectation_values"] = std::move(values);
    result["realization_standard_errors"] = std::move(errors);
  } else if (query) {
    Prepare(context, Inject(circuit, noise, rng), input.clbits);
    if (operation == "state_probability")
      result["probability"] = simulator->Probability(target);
    else if (operation == "diagnostics") {
      Supported(Mixed(config),
                "Mixed-state diagnostics require density_matrix or MPO");
      if (const auto* actions = request.if_contains("maintenance"))
        for (const auto& action : Array(*actions)) {
          const auto name = String(action);
          if (name == "restore_trace")
            simulator->RestoreDensityMatrixTrace();
          else if (name == "hermitize")
            simulator->HermitizeDensityMatrix();
          else if (name == "trim")
            simulator->TrimMatrixProductOperator();
          else if (name == "recanonicalize")
            simulator->ReCanonicalizeMatrixProductOperator();
          else
            throw Error("invalid_input", "Unknown maintenance action");
        }
      json::array defaults{"trace", "purity"};
      const auto* queries = request.if_contains("diagnostics");
      for (const auto& value : queries ? Array(*queries) : defaults) {
        const auto name = String(value);
        if (name == "trace")
          result[name] = Complex(simulator->DensityMatrixTrace());
        else if (name == "purity")
          result[name] = simulator->DensityMatrixPurity();
        else if (name == "trace_of_square")
          result[name] = Complex(simulator->DensityMatrixTraceOfSquare());
        else if (name == "hermiticity_residual")
          result[name] = simulator->DensityMatrixHermiticityResidual();
        else if (name == "is_hermitian")
          result[name] = simulator->IsDensityMatrixHermitian();
        else if (name == "partial_trace") {
          Types::qubits_vector keep;
          std::set<uint64_t> unique;
          for (const auto& q : Array(Field(request, "keep_qubits"))) {
            const auto index = UInt(q);
            Require(index < input.qubits && unique.insert(index).second,
                    "Invalid reduced-state qubit");
            keep.push_back(index);
          }
          Require(
              keep.size() < 32 && (uint64_t{1} << (2 * keep.size())) <= limit,
              "Reduced matrix exceeds output bound");
          const auto matrix = simulator->PartialTrace(keep);
          json::array entries;
          for (Eigen::Index row = 0; row < matrix.rows(); ++row)
            for (Eigen::Index col = 0; col < matrix.cols(); ++col)
              entries.emplace_back(Complex(matrix(row, col)));
          result[name] = json::object{{"dimension", matrix.rows()},
                                      {"row_major", entries}};
        } else
          throw Error("invalid_input", "Unknown diagnostic");
      }
    } else {
      json::array values;
      for (const auto& index : basis)
        if (amplitudes)
          values.emplace_back(Complex(simulator->Amplitude(UInt(index))));
        else
          values.emplace_back(simulator->Probability(UInt(index)));
      result[amplitudes ? "amplitudes" : "probabilities"] = std::move(values);
      result["basis_states"] = std::move(basis);
    }
  } else if (operation == "incremental_evolve") {
    Prepare(context, circuit, input.clbits);
    Circuits::OperationState state(input.clbits);
    json::array values;
    uint64_t current = 0;
    size_t bond = 0;
    for (const auto& value : Array(Field(request, "steps"))) {
      const auto step = UInt(value);
      while (current < step) {
        additional[0].circuit->ExecuteBD(simulator, state, &bond);
        ++current;
      }
      values.emplace_back(Expectations(simulator, observables));
    }
    result["steps"] = Field(request, "steps");
    result["expectation_values"] = std::move(values);
    result["max_bond_dim_reached"] = bond;
  } else if (operation == "checkpoint_batch") {
    Prepare(context, circuit, input.clbits);
    simulator->SaveState();
    json::array outputs;
    size_t bond = 0;
    for (const auto& suffix : additional) {
      Counts counts;
      const size_t samples = std::min(realizations, shots);
      for (size_t r = 0; r < samples; ++r) {
        const auto noisy = Inject(suffix.circuit, noise, rng);
        const size_t repetitions =
            shots / samples + (r < shots % samples ? 1 : 0);
        for (size_t shot = 0; shot < repetitions; ++shot) {
          simulator->RestoreState();
          simulator->SetGatesCounter(0);
          simulator->SetUpcomingGates(noisy->GetOperations());
          Circuits::OperationState state(input.clbits);
          noisy->ExecuteBD(simulator, state, &bond);
          ++counts[Bits(state.GetAllBits(), input.clbits)];
        }
      }
      outputs.emplace_back(
          json::object{{"counts", EncodeCounts(counts)}, {"shots", shots}});
    }
    result["results"] = std::move(outputs);
    if (noise.enabled) result["noise_scope"] = "suffix_only_ideal_prefix";
  } else {
    // Build U(first)^dagger U(second). Every nonmeasurement operation was
    // checked above; unsupported adjoints fail instead of silently cloning.
    double sum = 0, squares = 0;
    std::complex<double> overlap;
    for (size_t r = 0; r < realizations; ++r) {
      const auto second = operation == "inner_product"
                              ? additional[0].circuit
                              : Inject(circuit, noise, rng);
      auto combined = std::make_shared<Circuit>();
      for (const auto& op : second->GetOperations())
        combined->AddOperation(op->Clone());
      for (auto it = circuit->GetOperations().rbegin();
           it != circuit->GetOperations().rend(); ++it)
        if (auto adjoint = adjoint_gate(*it)) combined->AddOperation(adjoint);
      Prepare(context, combined, input.clbits);
      overlap = simulator->ProjectOnZero();
      const auto fidelity = std::norm(overlap);
      sum += fidelity;
      squares += fidelity * fidelity;
    }
    if (operation == "inner_product")
      result["inner_product"] = Complex(overlap);
    else {
      const auto mean = sum / realizations;
      result["fidelity"] = mean;
      result["standard_error"] =
          realizations > 1
              ? std::sqrt(std::max(0.0, (squares - realizations * mean * mean) /
                                            (realizations - 1) / realizations))
              : 0;
    }
  }
  result["execution_metadata"] = Metadata(
      context, config, operation == "execute" || operation == "estimate");
  result["seed"] = *config.seed;
  result["time_taken"] =
      std::chrono::duration<double>(Clock::now() - start).count();
  result["count_order"] = "classical_bit_0_first";
  result["basis_order"] = "qubit_0_least_significant";
  if (noise.enabled)
    result["noise"] = json::object{
        {"mode", noise.mode},
        {"evaluation", noise.mode == "analytical" ? "analytical_approximation"
                       : noise.exact              ? "exact_channels"
                                                  : "trajectories"},
        {"seed", noise.seed},
        {"realizations", realizations},
        {"approximations", noise.approximations}};
  Require(json::serialize(result).size() <= MaxResultBytes,
          "Result exceeds native result limit");
  return result;
}
}  // namespace MaestroExecution

namespace {
template <typename Function>
char* NativeBoundary(Function function) noexcept {
  namespace json = boost::json;
  try {
    json::object envelope;
    try {
      envelope = function();
      envelope["ok"] = true;
    } catch (const MaestroExecution::Error& error) {
      envelope = {{"ok", false},
                  {"error", json::object{{"code", error.code},
                                         {"message", error.what()}}}};
    } catch (const std::invalid_argument& error) {
      envelope = {{"ok", false},
                  {"error", json::object{{"code", "invalid_input"},
                                         {"message", error.what()}}}};
    } catch (const std::bad_alloc&) {
      // Even allocating an error can fail: the outer serialization fallback is
      // null.
      try {
        envelope = {{"ok", false},
                    {"error", json::object{{"code", "insufficient_resources"},
                                           {"message", "Allocation failed"}}}};
      } catch (...) {
        return nullptr;
      }
    } catch (const std::exception& error) {
      envelope = {{"ok", false},
                  {"error", json::object{{"code", "native_failure"},
                                         {"message", error.what()}}}};
    } catch (...) {
      try {
        envelope = {
            {"ok", false},
            {"error", json::object{{"code", "native_failure"},
                                   {"message", "Unknown native exception"}}}};
      } catch (...) {
        return nullptr;
      }
    }
    try {
      envelope["schema_version"] = MaestroExecution::SchemaVersion;
      const auto value = json::serialize(envelope);
      auto* output = new char[value.size() + 1];
      std::memcpy(output, value.c_str(), value.size() + 1);
      return output;
    } catch (...) {
      return nullptr;
    }
  } catch (...) {
    return nullptr;
  }
}
boost::json::object ParseRequest(const char* request) {
  MaestroExecution::Require(request != nullptr, "Request is null");
  const auto length = std::strlen(request);
  MaestroExecution::Require(length <= MaestroExecution::MaxRequestBytes,
                            "Request exceeds native size limit");
  boost::system::error_code error;
  auto value =
      boost::json::parse(boost::json::string_view(request, length), error);
  MaestroExecution::Require(!error, "Invalid JSON: " + error.message());
  return MaestroExecution::Object(value);
}
}  // namespace
extern "C" {
char* MaestroRunRequestJson(const char* request) {
  return NativeBoundary(
      [&] { return MaestroExecution::Run(ParseRequest(request)); });
}
char* MaestroValidateRequestJson(const char* request) {
  return NativeBoundary(
      [&] { return MaestroExecution::Run(ParseRequest(request), true); });
}
char* MaestroGetCapabilitiesJson() {
  return NativeBoundary([] { return MaestroExecution::Capabilities(); });
}
char* MaestroFinalizeDistributedMpiGpuJson() {
  return NativeBoundary([] {
#ifdef __linux__
    Simulators::SimulatorsFactory::FinalizeDistributedMpiGpuBackend();
#endif
    return boost::json::object{};
  });
}
}
