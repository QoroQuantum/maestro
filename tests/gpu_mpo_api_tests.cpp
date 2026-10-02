// Hardware contract tests for the Hastings MPO plugin and its public adapters.
#include "../Simulators/Factory.h"
#include "../maestrolib/Interface.h"
#include <boost/json.hpp>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace Simulators;
namespace j = boost::json;
namespace {
SimulatorType backend = SimulatorType::kGpuSim;
void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void Near(std::complex<double> actual, std::complex<double> expected) {
  Check(std::abs(actual - expected) < 1e-7, "numerical mismatch");
}
template <class F>
void Reject(F&& f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  Check(rejected, "invalid input was accepted");
}
auto Create(size_t width, bool fusion) {
  auto sim = SimulatorsFactory::CreateSimulator(
      backend, SimulationType::kMatrixProductOperator);
  Check(bool(sim), "MPO creation failed");
  sim->Configure("gate_fusion", fusion ? "true" : "false");
  sim->Configure("use_double_precision", "true");
  sim->AllocateQubits(width);
  sim->Initialize();
  return sim;
}
j::object Request(size_t width, const char* operation) {
  j::object request{
      {"schema_version", 2},
      {"operation", operation},
      {"circuit", j::object{{"format", "instructions"},
                            {"num_qubits", width},
                            {"source", j::array{}}}},
      {"simulator",
       j::object{
           {"backend", backend == SimulatorType::kGpuSim ? "gpu" : "qcsim"},
           {"method", "matrix_product_operator"},
           {"options",
            j::object{{"precision", "double"}, {"gate_fusion", true}}}}}};
  if (backend == SimulatorType::kQCSim)
    request.at("simulator")
        .as_object()
        .at("options")
        .as_object()
        .erase("precision");
  return request;
}
j::object Run(const j::object& request, bool validate = false,
              bool success = true) {
  const auto input = j::serialize(request);
  char* text = validate ? MaestroValidateRequestJson(input.c_str())
                        : MaestroRunRequestJson(input.c_str());
  Check(text, "null request response");
  const std::string output(text);
  FreeResult(text);
  auto result = j::parse(output).as_object();
  if (result.at("ok").as_bool() != success) throw std::runtime_error(output);
  return result;
}
void Wide(bool fusion) {
  constexpr size_t width = 70;
  auto sim = Create(width, fusion);
  sim->ApplyX(69);
  sim->ApplyX(63);
  sim->ApplyX(2);
  std::vector<bool> bits(width);
  bits[69] = bits[63] = bits[2] = true;
  Types::qubits_vector all(width);
  std::iota(all.begin(), all.end(), 0);
  auto counts = sim->SampleCountsMany(all, 11);
  Check(counts.size() == 1 && counts.at(bits) == 11,
        "wide histogram lost high bits");
  Check(sim->SampleCounts({69, 0, 63, 2, 69}, 7).at(29) == 7,
        "ordered subset/duplicates failed");
  Types::qubits_vector packedQubits(64);
  std::iota(packedQubits.begin(), packedQubits.end(), 0);
  Check(
      sim->SampleCounts(packedQubits, 2).at((Types::qubit_t{1} << 63) | 4) == 2,
      "64-bit packed boundary failed");
  Check(sim->MeasureNoCollapseMany() == bits,
        "wide no-collapse measurement failed");
  Near(sim->ProbabilityBits(bits), 1.);
  Near(sim->DensityMatrixElementBits(bits, bits), 1.);
  Check(sim->SampleCountsMany(all, 0).empty(), "zero shots failed");
  Reject([&] { sim->SampleCounts(all, 1); });
  Reject([&] { sim->SampleCountsMany({width}, 1); });
  Reject([&] { sim->MeasureNoCollapse(); });
  Reject([&] { sim->ProbabilityBits({true}); });
  Reject([&] { sim->GetDensityMatrix(); });
  Reject([&] { sim->PartialTrace(all); });
  sim->SaveState();
  sim->ApplyH(0);
  sim->ApplyCX(0, 68);
  sim->SetSeed(721);
  const auto first = sim->SampleCounts({68, 0, 69}, 64);
  Check(first.size() == 2 && first.count(4) && first.count(7),
        "entangled wide sample failed");
  sim->SetSeed(721);
  Check(sim->SampleCounts({68, 0, 69}, 64) == first, "seed replay failed");
  sim->RestoreState();
  Near(sim->ProbabilityBits(bits), 1.);
  std::vector<unsigned long> qs(all.begin(), all.end());
  std::vector<unsigned char> output(width * 3);
  unsigned long long totals[3];
  size_t written = 0;
  Check(MaestroSampleCountsBits(sim.get(), qs.data(), width, 3, output.data(),
                                totals, 3, &written),
        "C wide sample failed");
  Check(written == 1 && totals[0] == 3, "C histogram counts failed");
  for (size_t q = 0; q < width; ++q)
    Check(bool(output[q]) == bits[q], "C histogram bits failed");
  double probability = 0.;
  Check(MaestroProbabilityBits(sim.get(), output.data(), width, &probability),
        "C bit probability failed");
  Near(probability, 1.);
  Check(!MaestroSampleCountsBits(sim.get(), qs.data(), width, 3, output.data(),
                                 totals, 1, &written),
        "C short buffer accepted");
  Check(MaestroMeasureBits(sim.get(), qs.data(), width, output.data()),
        "C wide collapse failed");
  for (size_t q = 0; q < width; ++q)
    Check(bool(output[q]) == bits[q], "C collapse bits failed");
  Check(sim->MeasureMany(all) == bits, "wide collapse failed");
}
void Diagnostics(bool fusion) {
  auto sim = Create(2, fusion);
  sim->Configure("use_double_precision", "true");
  if (backend == SimulatorType::kGpuSim)
    Reject([&] { sim->Configure("use_double_precision", "false"); });
  sim->Configure("matrix_product_operator_restore_trace_after_truncation",
                 "true");
  sim->Configure("matrix_product_operator_hermitize_after_truncation", "true");
  Check(sim->GetConfiguration(
            "matrix_product_operator_restore_trace_after_truncation") == "true",
        "trace option ignored");
  Check(sim->GetConfiguration(
            "matrix_product_operator_hermitize_after_truncation") == "true",
        "hermitize option ignored");
  sim->Configure("matrix_product_operator_max_bond_dimension", "0");
  sim->ApplyH(0);
  sim->ApplyCX(0, 1);
  Eigen::Matrix2cd projector = Eigen::Matrix2cd::Zero();
  projector(1, 1) = 1.;
  sim->ApplyOperator({0}, projector);
  Near(sim->DensityMatrixTrace(), .5);
  Near(sim->GetDensityMatrix(false)(3, 3), .5);
  Near(sim->GetDensityMatrix()(3, 3), 1.);
  Near(sim->PartialTrace({1})(1, 1), 1.);
  Near(sim->ExpectationValueComplex("ZI"), -1.);
  Near(sim->ExpectationValueComplex("ZI", false), -.5);
  Near(sim->DensityMatrixElementBits({true, true}, {true, true}), .5);
  auto clone = sim->Clone();
  sim->MoveAtBeginningOfChain({1});
  Near(sim->ProbabilityBits({true, true}), 1.);
  Near(sim->DensityMatrixOverlap(*clone), 1.);
  sim->ReCanonicalize();
  sim->Trim();
  sim->ApplyOperator({0}, projector, true);
  Near(sim->DensityMatrixTrace(), 1.);
  const auto before = sim->GetDensityMatrix(false);
  Reject([&] { sim->ApplyOperator({0}, Eigen::Matrix2cd::Zero(), true); });
  Check((before - sim->GetDensityMatrix(false)).norm() < 1e-7,
        "failed postselection changed state");
  double dense[32], re = 0., im = 0.;
  Check(MaestroGetDensityMatrix(sim.get(), 1, dense, 32),
        "C dense query failed");
  Near(std::complex<double>(dense[30], dense[31]), 1.);
  Check(!MaestroGetDensityMatrix(sim.get(), 1, dense, 2),
        "C dense short buffer accepted");
  Check(MaestroExpectationValueComplex(sim.get(), "ZI", 0, &re, &im),
        "C complex expectation failed");
  Near(std::complex<double>(re, im), -1.);
  unsigned long q = 0;
  const double op[] = {0, 0, 0, 0, 0, .5, 0, 0};
  Check(MaestroApplyOperator(sim.get(), &q, 1, op, 0), "C operator failed");
  Check(MaestroDensityMatrixTrace(sim.get(), &re, &im), "C trace failed");
  Near(std::complex<double>(re, im), .25);
  Near(sim->GetDensityMatrix(false)(2, 2), .25);
  const unsigned char basis[] = {0, 1};
  Check(MaestroDensityMatrixElementBits(sim.get(), basis, basis, 2, &re, &im),
        "C element failed");
  Near(std::complex<double>(re, im), .25);
  Eigen::Matrix4cd two = Eigen::Matrix4cd::Zero();
  two(2, 1) = std::complex<double>(0., 2.);
  sim->ApplyOperator({1, 0}, two);
  Near(sim->GetDensityMatrix(false)(1, 1), 1.);
}
void CpuParity() {
  auto gpu = Create(4, true);
  auto cpu = SimulatorsFactory::CreateSimulator(SimulatorType::kQCSim,
                                                SimulationType::kDensityMatrix);
  cpu->AllocateQubits(4);
  cpu->Initialize();
  gpu->Configure("matrix_product_operator_truncation_threshold", "0");
  gpu->Configure("matrix_product_operator_max_bond_dimension", "0");
  for (unsigned round = 0; round < 5; ++round) {
    for (auto* sim : {gpu.get(), cpu.get()}) {
      sim->ApplyH(round % 4);
      sim->ApplyCX(3, 0);
      sim->ApplyRy(1, .17 * (round + 1));
      sim->ApplyCX(0, 2);
      sim->ApplyRz(3, -.21 * (round + 1));
      sim->ApplyAmplitudeDamping(2, .07);
    }
    const auto expected = cpu->PartialTrace({0, 1, 2, 3});
    Check((gpu->GetDensityMatrix() - expected).norm() < 1e-7,
          "GPU/CPU noisy density matrix parity failed");
    Near(gpu->ExpectationValueComplex("XYZI"), cpu->ExpectationValue("XYZI"));
  }
}
void Precision(SimulatorType type, SimulationType method, bool useDouble,
               bool fusion) {
  auto sim = SimulatorsFactory::CreateSimulator(type, method);
  Check(bool(sim), "precision test backend unavailable");
  const char* precision = useDouble ? "double" : "single";
  std::cout << "precision " << static_cast<int>(type) << ":"
            << static_cast<int>(method) << " " << precision
            << " fusion=" << fusion << std::endl;
  sim->Configure("gate_fusion", fusion ? "true" : "false");
  if (type == SimulatorType::kDistGpuSim)
    sim->Configure("distributed_devices", "0");
  // Last alias wins; unordered configuration replay must not reverse it.
  sim->Configure("precision", useDouble ? "single" : "double");
  sim->Configure("use_double_precision", useDouble ? "true" : "false");
  Check(sim->GetConfiguration("precision") == precision,
        "precision alias conflict");
  Reject([&] { sim->Configure("precision", "half"); });
  Reject([&] { sim->Configure("use_double_precision", "invalid"); });
  sim->AllocateQubits(3);
  sim->Initialize();
  Check(sim->GetConfiguration("precision") == precision,
        "native dtype mismatch");
  Check(ConfigureSimulator(sim.get(), "precision", precision) == 1,
        "C precision reapplication failed");
  Check(ConfigureSimulator(sim.get(), "precision",
                           useDouble ? "single" : "double") == 0,
        "C live precision change was accepted");
  Check(ConfigureSimulator(sim.get(), "precision", "invalid") == 0,
        "C invalid precision was accepted");
  sim->Configure("use_double_precision", useDouble ? "1" : "0");
  Reject([&] { sim->Configure("precision", useDouble ? "single" : "double"); });
  Check(sim->GetConfiguration("precision") == precision,
        "failed change poisoned dtype");
  const double theta = .713;
  sim->ApplyRy(0, theta);
  sim->ApplyCX(0, 2);
  const double expected = std::pow(std::sin(theta / 2), 2);
  const double tolerance = useDouble ? 1e-10 : 2e-5;
  Check(std::abs(sim->Probability(5) - expected) < tolerance,
        "precision computation failed");
  sim->SaveState();
  sim->ApplyX(1);
  sim->RestoreState();
  Check(std::abs(sim->Probability(5) - expected) < tolerance,
        "precision restore failed");
  if (method != SimulationType::kTensorNetwork) {
    auto clone = sim->Clone();
    Check(clone->GetConfiguration("precision") == precision,
          "precision clone dtype failed");
    Check(std::abs(clone->Probability(5) - expected) < tolerance,
          "precision clone failed");
  }
  // Check imports for backends supporting statevector initialization.
  bool supportsImport = method != SimulationType::kTensorNetwork &&
                        method != SimulationType::kMatrixProductState;
#ifndef NO_QISKIT_AER
  supportsImport &= type != SimulatorType::kCompositeQiskitAer;
#endif
  if (supportsImport) {
    std::vector<std::complex<double>> amplitudes(8);
    amplitudes[0] = std::cos(theta / 2);
    amplitudes[5] = std::complex<double>(0., std::sin(theta / 2));
    sim->Clear();
    sim->InitializeState(3, amplitudes);
    Check(sim->GetConfiguration("precision") == precision,
          "import changed dtype");
    Check(std::abs(sim->Probability(5) - expected) < tolerance,
          "precision import failed");
  }
  sim->Clear();
  sim->Configure("precision", useDouble ? "single" : "double");
  sim->AllocateQubits(2);
  sim->Initialize();
  Check(
      sim->GetConfiguration("use_double_precision") == (useDouble ? "0" : "1"),
      "clear/reinitialize failed to change dtype");
}
void PrecisionRequests() {
#ifdef NO_QISKIT_AER
  if (backend != SimulatorType::kGpuSim) return;
#endif
  const std::vector<std::string> methods =
      backend == SimulatorType::kGpuSim
          ? std::vector<std::string>{"statevector", "matrix_product_state",
                                     "tensor_network", "density_matrix",
                                     "matrix_product_operator"}
          : std::vector<std::string>{"statevector", "density_matrix"};
  for (const auto& method : methods)
    for (const char* precision : {"single", "double"}) {
      auto request = Request(2, "state_probability");
      request["simulator"] = j::object{
          {"backend", backend == SimulatorType::kGpuSim ? "gpu" : "aer"},
          {"method", method},
          {"options", j::object{{"precision", precision}}}};
      request["target_state"] = "11";
      request["circuit"] = j::object{
          {"format", "openqasm"},
          {"num_qubits", 2},
          {"source", "OPENQASM 2.0; qreg q[2]; ry(0.713) q[0]; cx q[0],q[1];"}};
      const double value = Run(request).at("probability").to_number<double>();
      Check(std::abs(value - std::pow(std::sin(.713 / 2), 2)) <
                (std::string(precision) == "double" ? 1e-10 : 2e-5),
            "JSON precision forwarding failed");
    }
}
void Precisions() {
  for (bool useDouble : {false, true})
    for (bool fusion : {false, true}) {
      if (backend == SimulatorType::kGpuSim) {
        for (auto method :
             {SimulationType::kStatevector, SimulationType::kMatrixProductState,
              SimulationType::kTensorNetwork, SimulationType::kDensityMatrix,
              SimulationType::kMatrixProductOperator})
          Precision(backend, method, useDouble, fusion);
        if (!fusion && SimulatorsFactory::IsDistributedGpuAvailable())
          Precision(SimulatorType::kDistGpuSim, SimulationType::kStatevector,
                    useDouble, false);
      } else {
#ifndef NO_QISKIT_AER
        for (auto method :
             {SimulationType::kStatevector, SimulationType::kDensityMatrix})
          Precision(SimulatorType::kQiskitAer, method, useDouble, fusion);
        Precision(SimulatorType::kCompositeQiskitAer,
                  SimulationType::kStatevector, useDouble, fusion);
#endif
        auto cpu = Create(2, fusion);
        cpu->Configure("precision", "single");
        Check(cpu->GetConfiguration("precision") == "double",
              "QCSim must remain double");
      }
    }
}
void Json() {
  auto request = Request(70, "state_probability");
  request["target_state"] = std::string(70, '0');
  Run(request, true);
  Near(Run(request).at("probability").to_number<double>(), 1.);
  request = Request(70, "execute");
  request["circuit"] = j::object{
      {"format", "openqasm"},
      {"num_qubits", 70},
      {"source",
       "OPENQASM 2.0; qreg q[70]; creg c[70]; x q[69]; measure q -> c;"}};
  request["execution"] = j::object{{"shots", 5}, {"seed", 42}};
  const auto counts = Run(request).at("counts").as_object();
  Check(counts.size() == 1 &&
            counts.at(std::string(69, '0') + '1').to_number<int>() == 5,
        "JSON wide execute lost bits");
  request = Request(2, "diagnostics");
  request["diagnostics"] =
      j::array{"trace",   "density_matrix",      "unnormalized_density_matrix",
               "element", "expectation_complex", "unnormalized_expectation"};
  request["row_state"] = "00";
  request["col_state"] = "00";
  request["pauli"] = "ZI";
  request["operators"] =
      j::array{j::object{{"qubits", j::array{0}},
                         {"matrix", j::array{j::array{.5, 0}, j::array{0, 0},
                                             j::array{0, 0}, j::array{1, 0}}}}};
  request["move_qubits"] = j::array{1};
  auto& options = request.at("simulator").as_object().at("options").as_object();
  options["mpo_restore_trace_after_truncation"] = true;
  options["mpo_hermitize_after_truncation"] = true;
  Run(request, true);
  const auto response = Run(request);
  Near(response.at("trace").as_array()[0].to_number<double>(), .25);
  Near(response.at("element").as_array()[0].to_number<double>(), .25);
  Near(response.at("expectation_complex").as_array()[0].to_number<double>(),
       1.);
  Near(
      response.at("unnormalized_expectation").as_array()[0].to_number<double>(),
      .25);
  request["max_output_elements"] = 2;
  Run(request, true, false);
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--cpu")
      backend = SimulatorType::kQCSim;
    if (backend == SimulatorType::kGpuSim) {
      if (SimulatorsFactory::GetGpuDeviceCount() == 0) {
        std::cout
            << "SKIP: GPU plugin/device unavailable (check plugin and CUDA "
               "library paths)\n";
        return 77;
      }
      Check(SimulatorsFactory::InitGpuLibraryWithMute(),
            "GPU library unavailable");
      auto lib = SimulatorsFactory::GetGpuLibrary();
      Check(lib->HasMPOAPI(), "updated MPO API unavailable");
      GpuMPO direct(lib);
      Check(direct.Create(70), "direct MPO creation failed");
      direct.ApplyX(69);
      const auto bits = direct.SampleBits(2, {69, 0});
      Check(bits == std::vector<unsigned char>({1, 0, 1, 0}),
            "raw shot bits failed");
      std::vector<int> all(70);
      std::iota(all.begin(), all.end(), 0);
      const auto full = direct.SampleBits(1, all);
      Check(full.size() == 70 && full[69] == 1 && full[63] == 0,
            "wide raw shot bits failed");
      direct.SetMaxExtent(0);
      Check(direct.GetMaxExtent() == 0, "unlimited bond cap failed");
    }
    for (bool fusion : {false, true}) {
      Wide(fusion);
      Diagnostics(fusion);
    }
    CpuParity();
    Json();
    Precisions();
    PrecisionRequests();
    std::cout << "PASS MPO API, wide sampling, fusion, C and JSON\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
