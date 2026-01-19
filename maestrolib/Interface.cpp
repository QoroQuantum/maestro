/**
 * @file interface.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 * C interface implementation for the maestro library.
 */

#include "Interface.h"

#ifdef COMPOSER
#include "../../composer/composer/Estimators/ExecutionEstimator.h"
#endif

#include "../Simulators/Factory.h"

#include "Maestro.h"

#include "Json.h"

#include <boost/json/src.hpp>

#include <atomic>
#include <memory>

#include "../Utils/LogFile.h"
#include "../qasm/QasmCirc.h"

static std::atomic_bool isInitialized{false};
static std::unique_ptr<Maestro> maestroInstance;

extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
    void *GetMaestroObject() {
  if (!isInitialized.exchange(true)) {
#ifdef __linux__
    Simulators::SimulatorsFactory::InitGpuLibrary();
#endif

#ifdef COMPOSER
    Estimators::ExecutionEstimator<>::InitializeRegressors();
#endif

    maestroInstance = std::make_unique<Maestro>();
  }

  return (void *)maestroInstance.get();
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void *GetMaestroObjectWithMute() {
  if (!isInitialized.exchange(true)) {
#ifdef __linux__
    Simulators::SimulatorsFactory::InitGpuLibraryWithMute();
#endif

#ifdef COMPOSER
    Estimators::ExecutionEstimator<>::InitializeRegressors();
#endif

    maestroInstance = std::make_unique<Maestro>();
  }

  return (void *)maestroInstance.get();
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int CreateSimpleSimulator(int nrQubits) {
  if (!maestroInstance) return 0;

  return maestroInstance->CreateSimpleSimulator(nrQubits);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void DestroySimpleSimulator(unsigned long int simHandle) {
  if (!maestroInstance || simHandle == 0) return;

  maestroInstance->DestroySimpleSimulator(simHandle);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int RemoveAllOptimizationSimulatorsAndAdd(unsigned long int simHandle,
                                              int simType, int simExecType) {
  if (!maestroInstance || simHandle == 0) return 0;

  return maestroInstance->RemoveAllOptimizationSimulatorsAndAdd(
      simHandle, static_cast<Simulators::SimulatorType>(simType),
      static_cast<Simulators::SimulationType>(simExecType));
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int AddOptimizationSimulator(unsigned long int simHandle, int simType,
                                 int simExecType) {
  if (!maestroInstance || simHandle == 0) return 0;

  return maestroInstance->AddOptimizationSimulator(
      simHandle, static_cast<Simulators::SimulatorType>(simType),
      static_cast<Simulators::SimulationType>(simExecType));
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    char *SimpleExecute(unsigned long int simpleSim, const char *circuitStr,
                        const char *jsonConfig) {
  if (simpleSim == 0 || !circuitStr || !jsonConfig || !maestroInstance)
    return nullptr;

  auto network = maestroInstance->GetSimpleSimulator(simpleSim);

  // step 1: Parse the JSON circuit and configuration strings
  // convert the JSON circuit into a Circuit object

  // I'm unsure here on how it deals with the classical registers, more
  // precisely with stuff like "other_measure_name" and "meas" (see below) since
  // in the example it seems to just use the cbit number

  // This is the json format:
  // {"instructions":
  // [{"name": "h", "qubits": [0], "params": []},
  // {"name": "cx", "qubits": [0, 1], "params": []},
  // {"name": "rx", "qubits": [0], "params": [0.39528385768119634]},
  // {"name": "measure", "qubits": [0], "memory": [0]}],
  //
  // "num_qubits": 2, "num_clbits": 4,
  // "quantum_registers": {"q": [0, 1]},
  // "classical_registers": {"c": [0, 1], "other_measure_name": [2], "meas":
  // [3]}}

  std::shared_ptr<Circuits::Circuit<>> circuit;

  if (circuitStr[0] == '{' || circuitStr[0] == '[') {
    // assume JSON format only if either object or array
    Json::JsonParserMaestro<> jsonParser;
    circuit = jsonParser.ParseCircuit(circuitStr);
  } else {
    // QASM 2.0 format
    qasm::QasmToCirc<> parser;
    std::string qasmInput(circuitStr);
    circuit = parser.ParseAndTranslate(qasmInput);
    if (parser.Failed()) return nullptr;
  }

  // check if the circuit has measurements only at the end

  // get the number of shots from the configuration
  size_t nrShots = 1;  // default value

  const auto configJson = Json::JsonParserMaestro<>::ParseString(jsonConfig);

  if (configJson.is_object()) {
    const auto configObject = configJson.as_object();
    // get whatever else is needed from the configuration
    // maybe simulator type, allowed simulator types, bond dimension limit, etc.

    // execute the circuit in the network object
    if (configObject.contains("shots") &&
        configObject.at("shots").is_number()) {
      auto number = configObject.at("shots");
      nrShots = number.is_int64() ? (size_t)number.as_int64()
                                  : (size_t)number.as_uint64();
    }
  }

  bool configured = false;

  const std::string maxBondDim = Json::JsonParserMaestro<>::GetConfigString(
      "matrix_product_state_max_bond_dimension", configJson);
  if (!maxBondDim.empty()) {
    configured = true;
    if (network->GetSimulator()) network->GetSimulator()->Clear();
    network->Configure("matrix_product_state_max_bond_dimension",
                       maxBondDim.c_str());
  }

  const std::string singularValueThreshold =
      Json::JsonParserMaestro<>::GetConfigString(
          "matrix_product_state_truncation_threshold", configJson);
  if (!singularValueThreshold.empty()) {
    configured = true;
    if (network->GetSimulator()) network->GetSimulator()->Clear();
    network->Configure("matrix_product_state_truncation_threshold",
                       singularValueThreshold.c_str());
  }

  const std::string mpsSample = Json::JsonParserMaestro<>::GetConfigString(
      "mps_sample_measure_algorithm", configJson);
  if (!mpsSample.empty()) {
    configured = true;
    if (network->GetSimulator()) network->GetSimulator()->Clear();
    network->Configure("mps_sample_measure_algorithm", mpsSample.c_str());
  }

  if (configured || !network->GetSimulator()) network->CreateSimulator();

  // TODO: get from config the allowed simulators types and so on, if set
  auto start = std::chrono::high_resolution_clock::now();
  auto results = network->RepeatedExecuteOnHost(circuit, 0, nrShots);
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  double time_taken = duration.count();
  std::string timeStr = std::to_string(time_taken);

  // convert the results into a JSON string
  // allocate memory for the result string and copy the JSON result into it
  // return the result string

  boost::json::object jsonResult;
  jsonResult.reserve(results.size());

  for (auto &result : results) {
    boost::json::string bits;
    bits.reserve(result.first.size());
    for (const auto bit : result.first) bits.append(bit ? "1" : "0");

    jsonResult.emplace(std::move(bits), std::move(result.second));
  }

  boost::json::object response;
  response.reserve(4);

  response.emplace("counts", std::move(jsonResult));
  response.emplace("time_taken", timeStr);

  auto simulatorType = network->GetLastSimulatorType();

  switch (simulatorType) {
#ifndef NO_QISKIT_AER
    case Simulators::SimulatorType::kQiskitAer:
      response.emplace("simulator", "aer");
      break;
#endif
    case Simulators::SimulatorType::kQCSim:
      response.emplace("simulator", "qcsim");
      break;
#ifndef NO_QISKIT_AER
    case Simulators::SimulatorType::kCompositeQiskitAer:
      response.emplace("simulator", "composite_aer");
      break;
#endif
    case Simulators::SimulatorType::kCompositeQCSim:
      response.emplace("simulator", "composite_qcsim");
      break;
#ifdef __linux__
    case Simulators::SimulatorType::kGpuSim:
      response.emplace("simulator", "gpu_simulator");
      break;
#endif
    default:
      response.emplace("simulator", "unknown");
      break;
  }

  auto simulationType = network->GetLastSimulationType();
  switch (simulationType) {
    case Simulators::SimulationType::kStatevector:
      response.emplace("method", "statevector");
      break;
    case Simulators::SimulationType::kMatrixProductState:
      response.emplace("method", "matrix_product_state");
      break;
    case Simulators::SimulationType::kStabilizer:
      response.emplace("method", "stabilizer");
      break;
    case Simulators::SimulationType::kTensorNetwork:
      response.emplace("method", "tensor_network");
      break;
    default:
      response.emplace("method", "unknown");
      break;
  }

  const std::string responseStr = boost::json::serialize(response);
  const size_t responseSize = responseStr.length();
  char *result = new char[responseSize + 1];

  const char *responseData = responseStr.c_str();
  std::copy(responseData, responseData + responseSize, result);

  result[responseSize] = 0;  // ensure null-termination

  return result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FreeResult(char *result) {
  if (result) delete[] result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int CreateSimulator(int simType, int simExecType) {
  if (!maestroInstance) return 0;

  return maestroInstance->CreateSimulator(
      static_cast<Simulators::SimulatorType>(simType),
      static_cast<Simulators::SimulationType>(simExecType));
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void *GetSimulator(unsigned long int simHandle) {
  if (!maestroInstance || simHandle == 0) return nullptr;
  return maestroInstance->GetSimulator(simHandle);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void DestroySimulator(unsigned long int simHandle) {
  if (!maestroInstance || simHandle == 0) return;
  maestroInstance->DestroySimulator(simHandle);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyX(void *sim, int qubit) {
  if (!sim) return 0;

  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyX(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyY(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyY(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyZ(void *sim, int qubit) {
  if (!sim) return 0;

  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyZ(qubit);
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyH(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyH(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyS(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyS(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySDG(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplySDG(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyT(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyT(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyTDG(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyTDG(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySX(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplySx(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySXDG(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplySxDAG(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyK(void *sim, int qubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyK(qubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyP(void *sim, int qubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyP(qubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyRx(void *sim, int qubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyRx(qubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyRy(void *sim, int qubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyRy(qubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyRz(void *sim, int qubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyRz(qubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyU(void *sim, int qubit, double theta, double phi, double lambda,
               double gamma) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyU(qubit, theta, phi, lambda, gamma);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCX(void *sim, int controlQubit, int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCX(controlQubit, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCY(void *sim, int controlQubit, int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCY(controlQubit, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCZ(void *sim, int controlQubit, int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCZ(controlQubit, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCH(void *sim, int controlQubit, int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCH(controlQubit, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCSX(void *sim, int controlQubit, int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCSx(controlQubit, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCSXDG(void *sim, int controlQubit, int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCSxDAG(controlQubit, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCP(void *sim, int controlQubit, int targetQubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCP(controlQubit, targetQubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCRx(void *sim, int controlQubit, int targetQubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCRx(controlQubit, targetQubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCRy(void *sim, int controlQubit, int targetQubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCRy(controlQubit, targetQubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCRz(void *sim, int controlQubit, int targetQubit, double theta) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCRz(controlQubit, targetQubit, theta);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCCX(void *sim, int controlQubit1, int controlQubit2,
                 int targetQubit) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCCX(controlQubit1, controlQubit2, targetQubit);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySwap(void *sim, int qubit1, int qubit2) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplySwap(qubit1, qubit2);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCSwap(void *sim, int controlQubit, int qubit1, int qubit2) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCSwap(controlQubit, qubit1, qubit2);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCU(void *sim, int controlQubit, int targetQubit, double theta,
                double phi, double lambda, double gamma) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->ApplyCU(controlQubit, targetQubit, theta, phi, lambda, gamma);

  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int InitializeSimulator(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->Initialize();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ResetSimulator(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->Reset();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ConfigureSimulator(void *sim, const char *key, const char *value) {
  if (!sim || !key || !value) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->Configure(key, value);
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    char *GetConfiguration(void *sim, const char *key) {
  if (!sim || !key) return nullptr;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  std::string value = simulator->GetConfiguration(key);
  if (value.empty()) return nullptr;
  // allocate memory for the result string and copy the configuration value into
  // it
  const size_t valueSize = value.length();
  char *result = new char[valueSize + 1];
  std::copy(value.c_str(), value.c_str() + valueSize, result);
  result[valueSize] = 0;  // ensure null-termination
  return result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int AllocateQubits(void *sim, unsigned long int nrQubits) {
  if (!sim || nrQubits == 0) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  const size_t res = simulator->AllocateQubits(nrQubits);

  return static_cast<unsigned long int>(res);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int GetNumberOfQubits(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  const size_t res = simulator->GetNumberOfQubits();
  return static_cast<unsigned long int>(res);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ClearSimulator(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->Clear();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long long int Measure(void *sim, const unsigned long int *qubits,
                                   unsigned long int nrQubits) {
  if (!sim || !qubits || nrQubits == 0) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  Types::qubits_vector qubitVector(qubits, qubits + nrQubits);
  const size_t res = simulator->Measure(qubitVector);
  return static_cast<unsigned long long int>(res);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyReset(void *sim, const unsigned long int *qubits,
                   unsigned long int nrQubits) {
  if (!sim || !qubits || nrQubits == 0) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  Types::qubits_vector qubitVector(qubits, qubits + nrQubits);
  simulator->ApplyReset(qubitVector);
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    double Probability(void *sim, unsigned long long int outcome) {
  if (!sim) return 0.0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  const double res = simulator->Probability(outcome);
  return res;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FreeDoubleVector(double *vec) {
  if (vec) delete[] vec;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    void FreeULLIVector(unsigned long long int *vec) {
  if (vec) delete[] vec;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    double *Amplitude(void *sim, unsigned long long int outcome) {
  if (!sim) return nullptr;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  const std::complex<double> amp = simulator->Amplitude(outcome);

  double *result = new double[2];
  result[0] = amp.real();
  result[1] = amp.imag();
  return result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    double *AllProbabilities(void *sim) {
  if (!sim) return nullptr;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  const auto probabilities = simulator->AllProbabilities();

  double *result = new double[probabilities.size()];
  std::copy(probabilities.begin(), probabilities.end(), result);
  return result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    double *Probabilities(void *sim, const unsigned long long int *qubits,
                          unsigned long int nrQubits) {
  if (!sim || !qubits || nrQubits == 0) return nullptr;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  Types::qubits_vector qubitVector(qubits, qubits + nrQubits);
  const auto probabilities = simulator->Probabilities(qubitVector);

  double *result = new double[probabilities.size()];
  std::copy(probabilities.begin(), probabilities.end(), result);
  return result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long long int *SampleCounts(void *sim,
                                         const unsigned long long int *qubits,
                                         unsigned long int nrQubits,
                                         unsigned long int shots) {
  if (!sim || !qubits || nrQubits == 0 || shots == 0) return nullptr;

  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  Types::qubits_vector qubitVector(qubits, qubits + nrQubits);
  const auto counts = simulator->SampleCounts(qubitVector, shots);

  unsigned long long int *result =
      new unsigned long long int[counts.size() * 2];
  size_t index = 0;
  for (const auto &count : counts) {
    result[index] = count.first;  // outcome
    ++index;
    result[index] = count.second;  // count
    ++index;
  }
  return result;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetSimulatorType(void *sim) {
  if (!sim) return -1;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  return static_cast<int>(simulator->GetType());
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetSimulationType(void *sim) {
  if (!sim) return -1;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  return static_cast<int>(simulator->GetSimulationType());
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int FlushSimulator(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->Flush();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int SaveStateToInternalDestructive(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->SaveStateToInternalDestructive();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int RestoreInternalDestructiveSavedState(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->RestoreInternalDestructiveSavedState();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int SaveState(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->SaveState();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int RestoreState(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->RestoreState();
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int SetMultithreading(void *sim, int multithreading) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  simulator->SetMultithreading(multithreading != 0);
  return 1;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetMultithreading(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  return simulator->GetMultithreading() ? 1 : 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int IsQcsim(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  return simulator->IsQcsim() ? 1 : 0;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long long int MeasureNoCollapse(void *sim) {
  if (!sim) return 0;
  auto simulator = static_cast<Simulators::ISimulator *>(sim);
  return static_cast<unsigned long long int>(simulator->MeasureNoCollapse());
}
}
