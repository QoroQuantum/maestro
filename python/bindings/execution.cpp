#include "core.h"
#include "runtime.h"

namespace maestro_bindings {

// Helper to parse observables from String (";" sep) or List[str]
std::vector<std::string> ParseObservables(const nb::object& observables) {
  std::vector<std::string> paulis;

  if (nb::isinstance<nb::str>(observables)) {
    std::string obsStr = nb::cast<std::string>(observables);
    std::stringstream ss(obsStr);
    std::string item;
    while (std::getline(ss, item, ';')) {
      // Trim whitespace if necessary, usually safe to skip empty
      if (!item.empty()) paulis.push_back(item);
    }
  } else if (nb::isinstance<nb::list>(observables)) {
    paulis = nb::cast<std::vector<std::string>>(observables);
  } else {
    throw nb::type_error(
        "Observables must be a ';'-separated string or a list of strings.");
  }
  return paulis;
}

// Core Execution Logic
nb::dict execute_core(std::shared_ptr<Circuits::Circuit<double>> circuit,
                      const SimulatorConfig& config, int shots) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
  if (!network) throw std::runtime_error("Failed to configure network.");

  Network::INetwork<double>::ExecuteResults raw_results;

  // Release GIL for heavy computation
  auto start = std::chrono::high_resolution_clock::now();
  {
    nb::gil_scoped_release release;
    raw_results = network->RepeatedExecuteOnHost(circuit, 0, (size_t)shots);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Process results back in Python land
  nb::dict counts;
  for (const auto& pair : raw_results) {
    // Optimization: Pre-allocate string to avoid repeated reallocation
    const auto& bool_vec = pair.first;
    std::string bitstring(bool_vec.size(), '0');
    for (size_t i = 0; i < bool_vec.size(); ++i) {
      if (bool_vec[i]) bitstring[i] = '1';
    }
    counts[bitstring.c_str()] = pair.second;
  }

  nb::dict py_result;
  py_result["counts"] = counts;
  py_result["time_taken"] = std::chrono::duration<double>(end - start).count();
  py_result["simulator"] = (int)network->GetLastSimulatorType();
  py_result["method"] = (int)network->GetLastSimulationType();
  if (network->GetLastGpuDevice() >= 0)
    py_result["gpu_device"] = network->GetLastGpuDevice();

  size_t max_bond_dim = network->GetCurrentMaxBondDimension();
  if (max_bond_dim > 0) py_result["max_bond_dim_reached"] = max_bond_dim;

  return py_result;
}

// Core Estimation Logic
nb::dict estimate_core(std::shared_ptr<Circuits::Circuit<double>> circuit,
                       const std::vector<std::string>& observables,
                       const SimulatorConfig& config) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  const auto& paulis = observables;

  int num_qubits = static_cast<int>(circuit->GetMaxQubitIndex()) + 1;
  for (const auto& p : paulis)
    num_qubits = std::max(num_qubits, (int)p.length());

  ScopedSimulator sim(std::max(1, num_qubits));
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
  if (!network) throw std::runtime_error("Failed to configure network.");

  std::vector<double> expectations;

  // Release GIL
  auto start = std::chrono::high_resolution_clock::now();
  {
    nb::gil_scoped_release release;
    expectations = network->ExecuteOnHostExpectations(circuit, 0, paulis);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Convert to Python list
  nb::list exp_vals;
  for (double val : expectations) exp_vals.append(val);

  nb::dict py_result;
  py_result["expectation_values"] = exp_vals;
  py_result["time_taken"] = std::chrono::duration<double>(end - start).count();
  py_result["simulator"] = (int)network->GetLastSimulatorType();
  py_result["method"] = (int)network->GetLastSimulationType();
  if (network->GetLastGpuDevice() >= 0)
    py_result["gpu_device"] = network->GetLastGpuDevice();

  size_t max_bond_dim = network->GetCurrentMaxBondDimension();
  if (max_bond_dim > 0) py_result["max_bond_dim_reached"] = max_bond_dim;

  return py_result;
}

// Core Statevector Logic
std::vector<std::complex<double>> statevector_core(
    std::shared_ptr<Circuits::Circuit<double>> circuit,
    const SimulatorConfig& config) {
  if (!circuit) throw nb::value_error("Circuit is null.");

  int num_qubits =
      std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
  ScopedSimulator sim(num_qubits);
  if (sim.handle == 0)
    throw std::runtime_error("Failed to create simulator handle.");

  auto network = ConfigureNetwork(sim.handle, config);
  if (!network) throw std::runtime_error("Failed to configure network.");

  std::vector<std::complex<double>> amplitudes;
  {
    nb::gil_scoped_release release;
    amplitudes = network->ExecuteOnHostAmplitudes(circuit, 0);
  }
  return amplitudes;
}

}  // namespace maestro_bindings
