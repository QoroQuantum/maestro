// Small, independent checks of the public C ABI; no Python or GPU is required.
#include "../maestrolib/Interface.h"
#include <boost/json.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
namespace j = boost::json;
extern "C" int maestro_request_c_header_test(void);
static unsigned checks = 0;
void TestRequestNoiseAndOptions();
void TestRequestSeedParsing();
void TestNetworkBondDefaults();
void TestAutomaticGpuMixedStateFallback();
void TestFixedBackendShotReuse();
void Check(bool condition, const char* message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
j::object Request(const char* operation, size_t n, const std::string& body,
                  const char* method = "statevector", size_t bits = 0) {
  j::object execution{{"seed", 123}};
  if (std::string(operation) == "execute" ||
      std::string(operation) == "checkpoint_batch")
    execution["shots"] = 80;
  return {{"schema_version", 2},
          {"operation", operation},
          {"circuit",
           j::object{{"format", "openqasm"},
                     {"num_qubits", n},
                     {"num_clbits", bits ? bits : n},
                     {"source",
                      "OPENQASM 2.0;\n// this newline must survive\nqreg q[" +
                          std::to_string(n) + "]; creg c[" +
                          std::to_string(bits ? bits : n) + "];\n" + body}}},
          {"simulator", j::object{{"backend", "qcsim"}, {"method", method}}},
          {"execution", execution}};
}
j::object Call(const j::object& request, bool success = true,
               bool validate = false) {
  const auto input = j::serialize(request);
  char* result = validate ? MaestroValidateRequestJson(input.c_str())
                          : MaestroRunRequestJson(input.c_str());
  Check(result != nullptr, "Null native result");
  const std::string output(result);
  FreeResult(result);
  auto parsed = j::parse(output).as_object();
  if (parsed.at("ok").as_bool() != success) throw std::runtime_error(output);
  ++checks;
  return parsed;
}
double Real(const j::value& value) { return value.to_number<double>(); }
void Near(double actual, double expected) {
  Check(std::abs(actual - expected) < 1e-9, "Numerical mismatch");
}

void TestLegacySeeds() {
  GetMaestroObjectWithMute();
  auto* capabilities = MaestroGetCapabilitiesJson();
  Check(capabilities != nullptr, "Cannot discover legacy backend ID");
  const auto catalog = j::parse(capabilities);
  FreeResult(capabilities);
  int backend = -1;
  for (const auto& entry : catalog.at("backends").as_array())
    if (entry.at("name") == "qcsim")
      backend = entry.at("legacy_id").to_number<int>();
  Check(backend >= 0, "Missing QCSim backend");
  const char* circuit =
      "OPENQASM 2.0; qreg q[2]; creg c[2]; "
      "h q[0]; h q[1]; measure q->c;";
  for (int method : {0, 1}) {
    struct Simulator {
      unsigned long handle = CreateSimpleSimulator(2);
      ~Simulator() { DestroySimpleSimulator(handle); }
    } simulator;
    Check(simulator.handle != 0, "Cannot create legacy simulator");
    Check(RemoveAllOptimizationSimulatorsAndAdd(simulator.handle, backend,
                                                method) == 1,
          "Cannot configure legacy simulator");
    auto run = [&](const j::value& seed) {
      const auto options =
          j::serialize(j::object{{"shots", 512}, {"seed", seed}});
      auto* raw = SimpleExecute(simulator.handle, circuit, options.c_str());
      Check(raw != nullptr, "Legacy seeded execution failed");
      const auto result = j::parse(raw);
      FreeResult(raw);
      return result.at("counts");
    };
    for (uint64_t seed : {uint64_t{0}, uint64_t{123}, UINT64_MAX}) {
      const auto counts = run(seed);
      Check(counts == run(seed), "Legacy execution ignored config.seed");
      Check(counts != run(seed ^ 1),
            "Different legacy seeds reused one stream");
      Check(counts == run(seed),
            "Legacy reseeding failed on an existing handle");
    }
    for (const auto& seed : j::array{-1, 1.5, "123", true, nullptr}) {
      const auto options =
          j::serialize(j::object{{"shots", 10}, {"seed", seed}});
      auto* result = SimpleExecute(simulator.handle, circuit, options.c_str());
      const bool rejected = result == nullptr;
      if (result) FreeResult(result);
      Check(rejected, "Malformed legacy seed was accepted");
      result = SimpleEstimate(simulator.handle, circuit, "ZI", options.c_str());
      const bool estimateRejected = result == nullptr;
      if (result) FreeResult(result);
      Check(estimateRejected, "Legacy estimator accepted a malformed seed");
    }
    run(123);
  }
}

void TestNativeRandomSeeds() {
  auto request =
      Request("execute", 4, "h q[0]; h q[1]; h q[2]; h q[3]; measure q->c;");
  request["execution"] = j::object{{"shots", 4096}};
  for (bool noisy : {false, true}) {
    if (noisy)
      request["noise"] =
          j::object{{"realizations", 16},
                    {"channels", j::array{j::object{{"kind", "readout"},
                                                    {"targets", j::array{0}},
                                                    {"probability", 0.3}}}}};
    const auto first = Call(request);
    const auto second = Call(request);
    Check(first.at("seed") != second.at("seed"),
          "Omitted native seeds reused a fixed seed");
    Check(first.at("counts") != second.at("counts"),
          "Independent unseeded requests reused their measurement stream");
    if (noisy)
      Check(first.at("noise").at("seed").to_number<uint32_t>() ==
                static_cast<uint32_t>(first.at("seed").to_number<uint64_t>()),
            "Default noise seed did not follow the generated execution seed");
    request["execution"].as_object()["seed"] = first.at("seed");
    Check(Call(request).at("counts") == first.at("counts"),
          "Returned random seed cannot reproduce native sampling/readout");
    for (uint64_t seed : {uint64_t{0}, UINT64_MAX}) {
      request["execution"].as_object()["seed"] = seed;
      const auto seeded = Call(request);
      Check(seeded.at("seed").to_number<uint64_t>() == seed,
            "Explicit native seed was replaced");
      Check(seeded.at("counts") == Call(request).at("counts"),
            "Explicit native seed no longer reproduces results");
    }
    request["execution"].as_object().erase("seed");
  }
  const auto batch = Call(j::object{{"schema_version", 2},
                                    {"operation", "batch"},
                                    {"requests", j::array{request, request}}});
  const auto& children = batch.at("results").as_array();
  Check(children[0].at("seed") != children[1].at("seed"),
        "Batch members must resolve omitted seeds independently");

#ifdef __linux__
  // Unseeded MPI validation is usable without loading/initializing MPI.
  request.erase("noise");
  request["simulator"].as_object()["backend"] = "distributed_mpi_gpu";
  Check(Call(request, true, true).at("valid").as_bool(),
        "Unseeded MPI validation attempted execution");
#endif
}

int main() try {
  Check(maestro_request_c_header_test(), "C header/ABI ownership check failed");
  TestFixedBackendShotReuse();
  char* capabilities = MaestroGetCapabilitiesJson();
  Check(capabilities != nullptr, "Missing capabilities");
  auto caps = j::parse(capabilities).as_object();
  FreeResult(capabilities);
  Check(!caps.at("python_required").as_bool(),
        "Native API must not require Python");
  Check(caps.at("schema_version").as_int64() == 2, "Wrong native schema");

  // Seeds remain meaningful across operations; shots configure sampling only.
  for (const char* operation :
       {"estimate", "statevector", "amplitudes", "probabilities",
        "state_probability", "inner_product", "mirror_fidelity",
        "noisy_fidelity", "diagnostics", "incremental_evolve", "validate"}) {
    auto document =
        Request(operation, 1, "",
                std::string(operation) == "diagnostics" ? "density_matrix"
                                                        : "statevector");
    const std::string kind(operation);
    if (kind == "estimate" || kind == "incremental_evolve")
      document["observables"] = j::array{"Z"};
    if (kind == "state_probability") document["target_state"] = "0";
    if (kind == "inner_product")
      document["other_circuit"] = document.at("circuit");
    if (kind == "incremental_evolve") {
      document["step_circuit"] = document.at("circuit");
      document["steps"] = j::array{0, 1};
    }
    if (kind == "noisy_fidelity")
      document["noise"] = j::object{
          {"mode", "coherent"},
          {"channels", j::array{j::object{{"kind", "coherent_rotation"},
                                          {"targets", j::array{0}},
                                          {"rx", 0},
                                          {"ry", 0},
                                          {"rz", 0.1}}}}};
    Call(document, true, true);
    document["execution"].as_object()["shots"] = 999;
    for (bool validate : {false, true}) {
      const auto error =
          Call(document, false, validate).at("error").as_object();
      Check(error.at("code") == "invalid_input" &&
                std::string(error.at("message").as_string().c_str()) ==
                    "execution.shots does not apply to " + kind,
            "Irrelevant shots were not rejected with operation context");
    }
  }

  auto request = Request("execute", 2, "x q[0]; measure q -> c;");
  auto result = Call(request);
  Near(Real(result.at("counts").at("10")), 80);
  Check(result.at("execution_metadata").at("method") == "statevector",
        "Fixed backend method changed");
  request["operation"] = "statevector";
  request["execution"].as_object().erase("shots");
  result = Call(request);
  Near(Real(result.at("amplitudes").at(1).at(0)), 1);
  Near(Real(result.at("amplitudes").at(2).at(0)), 0);
  request["operation"] = "probabilities";
  result = Call(request);
  Near(Real(result.at("probabilities").at(1)), 1);
  request["operation"] = "state_probability";
  request["target_state"] = "10";
  Near(Real(Call(request).at("probability")), 1);

  request =
      Request("execute", 2, "x q[1]; measure q[1] -> c[3];", "statevector", 4);
  result = Call(request);
  Near(Real(result.at("counts").at("0001")), 80);
  request["noise"] =
      j::object{{"channels", j::array{j::object{{"kind", "readout"},
                                                {"targets", j::array{1}},
                                                {"probability", 1}}}}};
  result = Call(request);
  Near(Real(result.at("counts").at("0000")), 80);

  request = Request("estimate", 1, "h q[0];", "density_matrix");
  request["observables"] = j::array{"X", "Z"};
  request["noise"] = j::object{
      {"evaluation", "exact"},
      {"channels",
       j::array{j::object{
           {"kind", "t1"}, {"targets", j::array{0}}, {"gamma", 0.36}}}}};
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), 0.8);
  Near(Real(result.at("expectation_values").at(1)), 0.36);
  request["operation"] = "diagnostics";
  request.erase("observables");
  result = Call(request);
  Near(Real(result.at("trace").at(0)), 1);
  Near(Real(result.at("purity")), (1 + 0.8 * 0.8 + 0.36 * 0.36) / 2);
  request["operation"] = "probabilities";
  result = Call(request);
  Near(Real(result.at("probabilities").at(0)), 0.68);
  request["operation"] = "statevector";
  Call(request, false);
  request["operation"] = "estimate";
  request["observables"] = j::array{"X", "Z"};
  request["simulator"].as_object()["method"] = "matrix_product_operator";
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), 0.8);

  request = Request("estimate", 1, "h q[0];", "density_matrix");
  request["observables"] = j::array{"X"};
  request["noise"] =
      j::object{{"evaluation", "exact"},
                {"channels", j::array{j::object{{"kind", "pauli"},
                                                {"targets", j::array{0}},
                                                {"px", 0},
                                                {"py", 0},
                                                {"pz", 0.25}}}}};
  Near(Real(Call(request).at("expectation_values").at(0)), 0.5);
  request["simulator"].as_object()["method"] = "statevector";
  request["noise"].as_object()["evaluation"] = "trajectories";
  request["noise"].as_object()["realizations"] = 10000;
  auto sampled = Call(request);
  Check(std::abs(Real(sampled.at("expectation_values").at(0)) - 0.5) <
            5 * std::sqrt(0.75 / 10000),
        "Sampled noise does not converge");
  Near(Real(Call(request).at("expectation_values").at(0)),
       Real(sampled.at("expectation_values").at(0)));

  request = Request("inner_product", 2, "h q[0]; cx q[0],q[1];");
  request["other_circuit"] = request.at("circuit");
  Near(Real(Call(request).at("inner_product").at(0)), 1);
  request["operation"] = "mirror_fidelity";
  request.erase("other_circuit");
  Near(Real(Call(request).at("fidelity")), 1);
  request = Request("incremental_evolve", 1, "");
  request["step_circuit"] = Request("execute", 1, "x q[0];").at("circuit");
  request["steps"] = j::array{0, 1, 2};
  request["observables"] = j::array{"Z"};
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0).at(0)), 1);
  Near(Real(result.at("expectation_values").at(1).at(0)), -1);
  Near(Real(result.at("expectation_values").at(2).at(0)), 1);

  request = Request("checkpoint_batch", 1, "x q[0];");
  request["suffixes"] =
      j::array{Request("execute", 1, "measure q->c;").at("circuit"),
               Request("execute", 1, "x q[0]; measure q->c;").at("circuit")};
  result = Call(request);
  Near(Real(result.at("results").at(0).at("counts").at("1")), 80);
  Near(Real(result.at("results").at(1).at("counts").at("0")), 80);
  for (const auto& launch :
       j::array{nullptr, j::object{{"profile", "local"}, {"ranks", 2}}}) {
    auto nested = request;
    nested["suffixes"].as_array()[0].as_object()["launch"] = launch;
    for (bool validate : {false, true}) {
      const auto error = Call(nested, false, validate).at("error").as_object();
      Check(error.at("code") == "invalid_input" &&
                error.at("message") == "Unknown field: launch",
            "Checkpoint suffix accepted launch metadata");
    }
  }

  request = Request("execute", 1, "x q[0]; measure q->c;");
  request["execution"].as_object()["shots"] = 0;
  Call(request, false, true);
  request["execution"].as_object()["shots"] = 1;
  request["simulator"].as_object()["options"] =
      j::object{{"not_an_option", true}};
  Call(request, false, true);
  request["simulator"].as_object()["options"] =
      j::object{{"mpo_hermitize_after_truncation", true}};
  Call(request, false, true);
  request["simulator"].as_object()["options"] = j::object{};
  request["simulator"].as_object()["backend"] = "distributed_gpu";
  request["simulator"].as_object()["method"] = "density_matrix";
  Call(request, false, true);
  request = Request("statevector", 30, "x q[0];");
  Call(request, false, true);
  request = Request("execute", 1, "not valid QASM;");
  Call(request, false, true);

  // Native instructions retain identity gates so configured noise is applied.
  request = Request("estimate", 1, "", "density_matrix");
  request["circuit"] = j::object{
      {"format", "instructions"},
      {"num_qubits", 1},
      {"source", j::array{j::object{{"name", "id"}, {"qubits", j::array{0}}}}}};
  request["observables"] = j::array{"Z"};
  request["noise"] =
      j::object{{"channels", j::array{j::object{{"kind", "bit_flip"},
                                                {"targets", j::array{0}},
                                                {"probability", 1}}}}};
  Near(Real(Call(request).at("expectation_values").at(0)), -1);
  request.erase("noise");
  request["operation"] = "incremental_evolve";
  // An explicit value keeps the singleton outer array from copying the matrix.
  request["step_circuit"] = j::object{
      {"format", "instructions"},
      {"num_qubits", 1},
      {"source",
       j::array{j::object{
           {"name", "kraus"},
           {"qubits", j::array{0}},
           {"operators", j::array{j::value(j::array{0, 1, 1, 0})}}}}}};
  request["steps"] = j::array{0, 1, 2};
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0).at(0)), 1);
  Near(Real(result.at("expectation_values").at(1).at(0)), -1);
  Near(Real(result.at("expectation_values").at(2).at(0)), 1);
  request["simulator"].as_object()["method"] = "statevector";
  Call(request, false, true);

  request = Request("checkpoint_batch", 1, "x q[0];", "density_matrix");
  request["suffixes"] =
      j::array{Request("execute", 1, "measure q->c;").at("circuit"),
               Request("execute", 1, "x q[0]; measure q->c;").at("circuit")};
  request["noise"] =
      j::object{{"channels", j::array{j::object{{"kind", "bit_flip"},
                                                {"targets", j::array{0}},
                                                {"probability", 1}}}}};
  result = Call(request);
  Check(result.at("noise_scope") == "suffix_only_ideal_prefix",
        "Checkpoint noise scope missing");
  Near(Real(result.at("results").at(0).at("counts").at("1")), 80);
  Near(Real(result.at("results").at(1).at("counts").at("1")), 80);

  request =
      Request("diagnostics", 2, "h q[0]; cx q[0],q[1];", "density_matrix");
  request["diagnostics"] = j::array{"partial_trace"};
  request["keep_qubits"] = j::array{0};
  result = Call(request);
  Near(Real(result.at("partial_trace").at("row_major").at(0).at(0)), 0.5);
  Near(Real(result.at("partial_trace").at("row_major").at(3).at(0)), 0.5);
  request["keep_qubits"] = j::array{2};
  Call(request, false, true);
  request["diagnostics"] = j::array{"unknown"};
  Call(request, false, true);

  request = Request("estimate", 1, "x q[0];");
  request["observables"] = j::array{"Z"};
  request["simulator"].as_object()["selection"] = "automatic";
  request["simulator"].as_object()["candidates"] = j::array{
      j::object{{"backend", "qcsim"}, {"method", "matrix_product_state"}}};
  result = Call(request);
  Near(Real(result.at("expectation_values").at(0)), -1);
  Check(result.at("execution_metadata").at("method") == "matrix_product_state",
        "Automatic metadata must describe executed backend");

  for (auto distribution : {j::object{{"devices", j::array{0, 1, 2}}},
                            j::object{{"devices", j::array{0, 0}}},
                            j::object{{"max_queued_gates", 0}},
                            j::object{{"global_qubits", j::array{3}}},
                            j::object{{"flags", "wrong"}},
                            j::object{{"snapshot_storage", "wrong"}}}) {
    request = Request("execute", 2, "measure q->c;");
    request["simulator"] = j::object{{"backend", "distributed_gpu"},
                                     {"distribution", distribution}};
    Call(request, false, true);
  }
  request["simulator"] =
      j::object{{"backend", "distributed_gpu"},
                {"distribution", j::object{{"devices", j::array{0, 0}},
                                           {"global_qubits", j::array{1}},
                                           {"flags", 1},
                                           {"backend", "conventional"}}}};
#ifdef __linux__
  Call(request, true, true);  // Logical shared-device validation needs no GPU.
#else
  Check(Call(request, false, true).at("error").at("code") ==
            "unsupported_capability",
        "Uncompiled distributed backend should be rejected");
#endif

  request = Request("probabilities", 1, "");
  request["circuit"] = j::object{
      {"format", "instructions"},
      {"num_qubits", 1},
      {"source",
       j::array{
           j::object{{"name", "x"}, {"qubits", j::array{0}}},
           j::object{
               {"name", "delay"}, {"qubits", j::array{0}}, {"duration", 0.001}},
           j::object{{"name", "reset"}, {"qubits", j::array{0}}}}}};
  Near(Real(Call(request).at("probabilities").at(0)), 1);
  request["circuit"].as_object()["source"].as_array().push_back(
      j::object{{"name", "unknown"}, {"qubits", j::array{0}}});
  Call(request, false, true);
  request = Request("execute", 1, "x q[0]; measure q->c;");
  request["simulator"] = j::object{
      {"backend", "gpu"}, {"options", j::object{{"gpu_device", 2147483647}}}};
  result = Call(request, false);
  Check(result.at("error").at("code") != "",
        "Backend selection failures must be explicit");
  request = Request("statevector", 1, "");
  request["circuit"] = j::object{
      {"format", "instructions"},
      {"num_qubits", 1},
      {"source",
       j::array{j::object{{"name", "u2"},
                          {"qubits", j::array{0}},
                          {"params", j::array{0, std::acos(-1.0)}}}}}};
  result = Call(request);
  Near(Real(result.at("amplitudes").at(0).at(0)), std::sqrt(0.5));
  Near(Real(result.at("amplitudes").at(1).at(0)), std::sqrt(0.5));
  request["circuit"].as_object()["source"].as_array()[0].as_object()["params"] =
      j::array{};
  Call(request, false, true);
  request = Request("noisy_fidelity", 1, "h q[0];");
  request["noise"] =
      j::object{{"mode", "coherent"},
                {"realizations", 2},
                {"channels", j::array{j::object{{"kind", "coherent_rotation"},
                                                {"targets", j::array{0}},
                                                {"rx", 0},
                                                {"ry", 0},
                                                {"rz", 0}}}}};
  Near(Real(Call(request).at("fidelity")), 1);
  auto parameterized = Request("estimate", 1, "");
  parameterized["circuit"] = j::object{
      {"num_qubits", 1},
      {"source", "OPENQASM 3.0; input float theta; qubit q; rx(theta) q;"},
      {"parameters", j::object{{"theta", std::acos(-1.0)}}}};
  parameterized["observables"] = j::array{"Z"};
  Near(Real(Call(parameterized).at("expectation_values").at(0)), -1);
  auto batch = j::object{{"schema_version", 2},
                         {"operation", "batch"},
                         {"requests", j::array{request, parameterized}}};
  result = Call(batch);
  Near(Real(result.at("results").at(0).at("fidelity")), 1);
  Near(Real(result.at("results").at(1).at("expectation_values").at(0)), -1);
  auto invalid = Request("diagnostics", 1, "", "matrix_product_operator");
  invalid["simulator"].as_object()["options"] =
      j::object{{"mpo_kraus_completeness_check", "typo"}};
  Call(invalid, false, true);
  batch["requests"].as_array().push_back(invalid);
  Call(batch, false, true);
  batch["requests"] = j::array{parameterized};
  batch["simulator"] = j::object{{"backend", "qcsim"}};
  Call(batch, false,
       true);  // Batch members must carry their own configuration.

  // Parser failures are input errors in both validation and execution.
  for (const auto& instruction :
       j::array{j::object{{"name", "cx"}, {"qubits", j::array{0}}},
                j::object{{"name", "x"}, {"qubits", j::array{0, 1}}},
                j::object{{"name", "ccx"}, {"qubits", j::array{0, 1}}},
                j::object{{"name", "unknown"}, {"qubits", j::array{0}}},
                j::object{{"name", "measure"},
                          {"qubits", j::array{0, 1}},
                          {"clbits", j::array{0}}}}) {
    request = Request("execute", 3, "");
    request["circuit"] = j::object{{"format", "instructions"},
                                   {"num_qubits", 3},
                                   {"source", j::array{instruction}}};
    for (bool validate : {false, true})
      Check(Call(request, false, validate).at("error").at("code") ==
                "invalid_input",
            "Instruction errors must be invalid_input");
  }

  for (const auto& operation :
       j::array{j::object{{"name", "reset"}, {"qubits", j::array{0}}},
                j::object{
                    {"name", "kraus"},
                    {"qubits", j::array{0}},
                    {"operators", j::array{j::value(j::array{0, 1, 1, 0})}}}}) {
    request = Request("checkpoint_batch", 1, "", "density_matrix");
    request["circuit"] = j::object{{"format", "instructions"},
                                   {"num_qubits", 1},
                                   {"source", j::array{operation}}};
    request["suffixes"] =
        j::array{Request("execute", 1, "measure q->c;").at("circuit")};
    const auto error = Call(request, false, true).at("error").as_object();
    Check(error.at("code") == "unsupported_capability" &&
              error.at("message") ==
                  "checkpoint_batch prefix requires a unitary circuit",
          "Checkpoint prefix error lost operation context");
  }

  for (const auto* field :
       {"other_circuit", "step_circuit", "suffixes", "steps", "requests",
        "diagnostics", "maintenance", "keep_qubits", "basis_states",
        "target_state", "observables", "max_output_elements"}) {
    request = Request("execute", 1, "");
    request[field] = j::object{{"unknown_nested_field", true}};
    Check(Call(request, false, true).at("error").at("code") == "invalid_input",
          "Irrelevant operation field was accepted");
  }
  request = Request("diagnostics", 1, "", "density_matrix");
  request["keep_qubits"] = j::array{0};
  Call(request, false, true);
  for (const auto& instruction : j::array{
           j::object{
               {"name", "reset"}, {"qubits", j::array{0}}, {"duration", 1}},
           j::object{{"name", "delay"},
                     {"qubits", j::array{0}},
                     {"duration", 1},
                     {"operators", j::array{}}}}) {
    request = Request("execute", 1, "");
    request["circuit"] = j::object{{"format", "instructions"},
                                   {"num_qubits", 1},
                                   {"source", j::array{instruction}}};
    Call(request, false, true);
  }

  // Checkpoint seeds promise repeatability of the entire ordered request, not
  // invariant per-branch samples after branches are moved or changed.
  request = Request("checkpoint_batch", 1, "");
  request["suffixes"] =
      j::array{Request("execute", 1, "x q[0]; measure q->c;").at("circuit"),
               Request("execute", 1, "h q[0]; measure q->c;").at("circuit")};
  request["noise"] =
      j::object{{"mode", "pauli"},
                {"seed", 123},
                {"realizations", 40},
                {"channels", j::array{j::object{{"kind", "bit_flip"},
                                                {"targets", j::array{0}},
                                                {"probability", 0.3}},
                                      j::object{{"kind", "readout"},
                                                {"targets", j::array{0}},
                                                {"probability", 0.1}}}}};
  const auto checkpoint = Call(request).at("results");
  Check(checkpoint == Call(request).at("results"),
        "Seeded checkpoint changed across repeats");
  for (const auto& suffix : checkpoint.as_array()) {
    int64_t count = 0;
    for (const auto& entry : suffix.at("counts").as_object())
      count += entry.value().to_number<int64_t>();
    Check(count == 80, "Checkpoint lost shots");
  }

  request = Request("noisy_fidelity", 1, "");
  request["circuit"] = j::object{
      {"format", "instructions"},
      {"num_qubits", 1},
      {"source", j::array{j::object{{"name", "h"}, {"qubits", j::array{0}}}}}};
  request["noise"] =
      j::object{{"mode", "coherent"},
                {"seed", 12},
                {"realizations", 8},
                {"channels", j::array{j::object{{"kind", "coherent_rotation"},
                                                {"targets", j::array{0}},
                                                {"rx", 0.2},
                                                {"ry", 0.1},
                                                {"rz", 0.3}}}}};
  const auto fidelity = Real(Call(request).at("fidelity"));
  request["circuit"].as_object()["source"].as_array().push_back(j::object{
      {"name", "delay"}, {"qubits", j::array{0}}, {"duration", 0.001}});
  Near(Real(Call(request).at("fidelity")), fidelity);
  request.erase("noise");
  request["operation"] = "mirror_fidelity";
  Near(Real(Call(request).at("fidelity")), 1);

  request = Request("execute", 1, "");
  request["launch"] = j::object{{"profile", "missing"}, {"ranks", 999}};
  Check(Call(request, false, true).at("error").at("code") == "invalid_input",
        "Native validation ignored launch metadata");
  Call(request, false);
  auto nested = j::object{{"schema_version", 2},
                          {"operation", "batch"},
                          {"requests", j::array{request}}};
  Call(nested, false, true);
  TestRequestNoiseAndOptions();
  TestLegacySeeds();
  TestNativeRandomSeeds();
  TestRequestSeedParsing();
  TestNetworkBondDefaults();
  TestAutomaticGpuMixedStateFallback();

  std::cout << "Native request API: " << checks << " checks passed\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << "Native request API failure after " << checks
            << " checks: " << error.what() << '\n';
  return 1;
}
