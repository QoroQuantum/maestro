#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>

#include "Circuit/Circuit.h"
#include "Interface.h"
#include "Maestro.h"
#include "Simulators/Factory.h"
#include "Simulators/Simulator.h"
#include "qasm/QasmCirc.h"

#include <boost/json.hpp>

namespace nb = nanobind;

NB_MODULE(maestro, m) {
    m.doc() = "Python bindings for Maestro Quantum Simulator";

    // Bind SimulatorType enum
    nb::enum_<Simulators::SimulatorType>(m, "SimulatorType")
        .value("QCSim", Simulators::SimulatorType::kQCSim)
#ifndef NO_QISKIT_AER
        .value("QiskitAer", Simulators::SimulatorType::kQiskitAer)
        .value("CompositeQiskitAer", Simulators::SimulatorType::kCompositeQiskitAer)
#endif
        .value("CompositeQCSim", Simulators::SimulatorType::kCompositeQCSim)
        .export_values();

  // Bind SimulationType enum
  nb::enum_<Simulators::SimulationType>(m, "SimulationType")
      .value("Statevector", Simulators::SimulationType::kStatevector)
      .value("MatrixProductState",
             Simulators::SimulationType::kMatrixProductState)
      .value("Stabilizer", Simulators::SimulationType::kStabilizer)
      .value("TensorNetwork", Simulators::SimulationType::kTensorNetwork)
      .export_values();

  // Bind ISimulator interface
  nb::class_<Simulators::ISimulator>(m, "ISimulator")
      .def("ApplyP", &Simulators::ISimulator::ApplyP)
      .def("ApplyX", &Simulators::ISimulator::ApplyX)
      .def("ApplyY", &Simulators::ISimulator::ApplyY)
      .def("ApplyZ", &Simulators::ISimulator::ApplyZ)
      .def("ApplyH", &Simulators::ISimulator::ApplyH)
      .def("ApplyS", &Simulators::ISimulator::ApplyS)
      .def("ApplySDG", &Simulators::ISimulator::ApplySDG)
      .def("ApplyT", &Simulators::ISimulator::ApplyT)
      .def("ApplyTDG", &Simulators::ISimulator::ApplyTDG)
      .def("ApplySx", &Simulators::ISimulator::ApplySx)
      .def("ApplySxDAG", &Simulators::ISimulator::ApplySxDAG)
      .def("ApplyK", &Simulators::ISimulator::ApplyK)
      .def("ApplyRx", &Simulators::ISimulator::ApplyRx)
      .def("ApplyRy", &Simulators::ISimulator::ApplyRy)
      .def("ApplyRz", &Simulators::ISimulator::ApplyRz)
      .def("ApplyU", &Simulators::ISimulator::ApplyU)
      .def("ApplyCX", &Simulators::ISimulator::ApplyCX)
      .def("ApplyCY", &Simulators::ISimulator::ApplyCY)
      .def("ApplyCZ", &Simulators::ISimulator::ApplyCZ)
      .def("ApplyCP", &Simulators::ISimulator::ApplyCP)
      .def("ApplyCRx", &Simulators::ISimulator::ApplyCRx)
      .def("ApplyCRy", &Simulators::ISimulator::ApplyCRy)
      .def("ApplyCRz", &Simulators::ISimulator::ApplyCRz)
      .def("ApplyCH", &Simulators::ISimulator::ApplyCH)
      .def("ApplyCSx", &Simulators::ISimulator::ApplyCSx)
      .def("ApplyCSxDAG", &Simulators::ISimulator::ApplyCSxDAG)
      .def("ApplySwap", &Simulators::ISimulator::ApplySwap)
      .def("ApplyCCX", &Simulators::ISimulator::ApplyCCX)
      .def("ApplyCSwap", &Simulators::ISimulator::ApplyCSwap)
      .def("ApplyCU", &Simulators::ISimulator::ApplyCU)
      .def("ApplyNop", &Simulators::ISimulator::ApplyNop)
      .def("AllocateQubits", &Simulators::ISimulator::AllocateQubits)
      .def("GetNumberOfQubits", &Simulators::ISimulator::GetNumberOfQubits)
      .def("Measure", &Simulators::ISimulator::Measure)
      .def("SampleCounts", &Simulators::ISimulator::SampleCounts,
           nb::arg("qubits"), nb::arg("shots") = 1000)
      .def("Initialize", &Simulators::ISimulator::Initialize)
      .def("Reset", &Simulators::ISimulator::Reset);

  // Bind Maestro class
  nb::class_<Maestro>(m, "Maestro")
      .def(nb::init<>())
      .def("CreateSimulator", &Maestro::CreateSimulator,
           nb::arg("simType") = Simulators::SimulatorType::kQCSim,
           nb::arg("simExecType") =
               Simulators::SimulationType::kMatrixProductState)
      .def(
          "GetSimulator",
          [](Maestro &self, unsigned long int simHandle) {
            return static_cast<Simulators::ISimulator *>(
                self.GetSimulator(simHandle));
          },
          nb::rv_policy::reference_internal)
      .def("DestroySimulator", &Maestro::DestroySimulator);

  // Bind Circuit class
  nb::class_<Circuits::Circuit<double>>(m, "Circuit")
      .def("GetMaxQubitIndex", &Circuits::Circuit<double>::GetMaxQubitIndex);

  // Bind QasmToCirc class
  nb::class_<qasm::QasmToCirc<double>>(m, "QasmToCirc")
      .def(nb::init<>())
      .def("ParseAndTranslate", &qasm::QasmToCirc<double>::ParseAndTranslate);

  // Bind simple_execute convenience function
  m.def(
      "simple_execute",
      [](const std::string &qasm_circuit, Simulators::SimulatorType sim_type,
         Simulators::SimulationType sim_exec_type, int shots,
         std::optional<size_t> max_bond_dimension,
         std::optional<double> singular_value_threshold) -> nb::object {
        // Initialize Maestro instance if needed
        GetMaestroObjectWithMute();

        // Parse the QASM circuit first to determine the number of qubits
        qasm::QasmToCirc<> parser;
        auto circuit = parser.ParseAndTranslate(qasm_circuit);
        if (parser.Failed() || !circuit) {
          return nb::none();
        }

        // Get the number of qubits from the circuit
        int num_qubits = static_cast<int>(circuit->GetMaxQubitIndex()) + 1;

        // Create a simple simulator with the correct number of qubits
        unsigned long int sim_handle = CreateSimpleSimulator(num_qubits);
        if (sim_handle == 0) {
          return nb::none();
        }

        // Add optimization simulator with the desired types
        int result = RemoveAllOptimizationSimulatorsAndAdd(
            sim_handle, static_cast<int>(sim_type),
            static_cast<int>(sim_exec_type));

        if (result == 0) {
          DestroySimpleSimulator(sim_handle);
          return nb::none();
        }

        // Build JSON configuration
        boost::json::object config;
        config["shots"] = shots;

        // --- NEW CONFIGURATION VARIABLES ---
        if (max_bond_dimension.has_value()) {
          config["matrix_product_state_max_bond_dimension"] =
              *max_bond_dimension;
        }

        if (singular_value_threshold.has_value()) {
          config["matrix_product_state_truncation_threshold"] =
              *singular_value_threshold;
        }
        // -----------------------------------

        std::string json_config = boost::json::serialize(config);

        // Execute the circuit
        char *result_str = SimpleExecute(sim_handle, qasm_circuit.c_str(),
                                         json_config.c_str());

        // Clean up simulator
        DestroySimpleSimulator(sim_handle);

        if (result_str == nullptr) {
          return nb::none();
        }

        // Parse JSON result
        try {
          boost::json::value json_result = boost::json::parse(result_str);
          FreeResult(result_str);

          if (!json_result.is_object()) {
            return nb::none();
          }

          auto result_obj = json_result.as_object();

          // Convert to Python dictionary
          nb::dict py_result;

          // Convert counts
          if (result_obj.contains("counts") &&
              result_obj.at("counts").is_object()) {
            nb::dict counts;
            auto counts_obj = result_obj.at("counts").as_object();
            for (const auto &pair : counts_obj) {
              std::string key(pair.key());
              if (pair.value().is_int64()) {
                counts[key.c_str()] = pair.value().as_int64();
              } else if (pair.value().is_uint64()) {
                counts[key.c_str()] =
                    static_cast<int64_t>(pair.value().as_uint64());
              }
            }
            py_result["counts"] = counts;
          }

          // Add other metadata
          if (result_obj.contains("simulator")) {
            py_result["simulator"] =
                std::string(result_obj.at("simulator").as_string());
          }
          if (result_obj.contains("method")) {
            py_result["method"] =
                std::string(result_obj.at("method").as_string());
          }
          if (result_obj.contains("time_taken")) {
            auto time_str =
                std::string(result_obj.at("time_taken").as_string());
            py_result["time_taken"] = std::stod(time_str);
          }

          return py_result;

        } catch (const std::exception &e) {
          FreeResult(result_str);
          return nb::none();
        }
      },
      nb::arg("qasm_circuit"),
      nb::arg("simulator_type") = Simulators::SimulatorType::kQCSim,
      nb::arg("simulation_type") = Simulators::SimulationType::kStatevector,
      nb::arg("shots") = 1024, nb::arg("max_bond_dimension") = 2,
      nb::arg("singular_value_threshold") = 1e-8,
      "Execute a QASM circuit and return measurement results.\n\n"
      "Args:\n"
      "    qasm_circuit: QASM 2.0 quantum circuit as a string\n"
      "    simulator_type: Type of simulator to use (default: QCSim)\n"
      "    simulation_type: Simulation method to use (default: Statevector)\n"
      "    shots: Number of measurement shots (default: 1024)\n"
      "    max_bond_dimension: Max bond dimension for MPS (default: 2)\n"
      "    singular_value_threshold: Truncation threshold for MPS (default: "
      "1e-8)\n\n"
      "Returns:\n"
      "    Dictionary with keys 'counts', 'simulator', 'method', and "
      "'time_taken',\n"
      "    or None if execution failed");
}
