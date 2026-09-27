#include "core.h"
#include "module.h"

namespace maestro_bindings {

void bind_simulator(nb::module_& m) {
  nb::class_<Simulators::ISimulator>(m, "Simulator")
      // Low-level operations from Interface.h, using Python-owned results.
      .def("InitializeSimulator", &Simulators::ISimulator::Initialize)
      .def("Initialize", &Simulators::ISimulator::Initialize)
      .def("ResetSimulator", &Simulators::ISimulator::Reset)
      .def("Reset", &Simulators::ISimulator::Reset)
      .def("ConfigureSimulator", &Simulators::ISimulator::Configure, "key"_a,
           "value"_a)
      .def("Configure", &Simulators::ISimulator::Configure, "key"_a, "value"_a)
      .def("GetConfiguration", &Simulators::ISimulator::GetConfiguration,
           "key"_a)
      .def("AllocateQubits", &Simulators::ISimulator::AllocateQubits,
           "num_qubits"_a)
      .def("GetNumberOfQubits", &Simulators::ISimulator::GetNumberOfQubits)
      .def("ClearSimulator", &Simulators::ISimulator::Clear)
      .def("Clear", &Simulators::ISimulator::Clear)
      .def("Measure", &Simulators::ISimulator::Measure, "qubits"_a,
           "Measure and collapse the selected qubits; the first listed qubit "
           "is the least-significant result bit.")
      .def("ApplyReset", &Simulators::ISimulator::ApplyReset, "qubits"_a)
      .def("Probability", &Simulators::ISimulator::Probability, "outcome"_a)
      .def("Amplitude", &Simulators::ISimulator::Amplitude, "outcome"_a)
      .def("AllProbabilities", &Simulators::ISimulator::AllProbabilities)
      .def("Probabilities", &Simulators::ISimulator::Probabilities,
           "outcomes"_a,
           "Return probabilities for the given basis-state indices.")
      .def(
          "SampleCounts", &Simulators::ISimulator::SampleCounts, "qubits"_a,
          "shots"_a = 1000,
          "Sample without collapsing the state, returning {integer_outcome: "
          "count}; the first listed qubit is the least-significant result bit.")
      .def("GetSimulatorType", &Simulators::ISimulator::GetType)
      .def("GetSimulationType", &Simulators::ISimulator::GetSimulationType)
      .def("FlushSimulator", &Simulators::ISimulator::Flush)
      .def("Flush", &Simulators::ISimulator::Flush)
      .def("SaveStateToInternalDestructive",
           &Simulators::ISimulator::SaveStateToInternalDestructive)
      .def("RestoreInternalDestructiveSavedState",
           &Simulators::ISimulator::RestoreInternalDestructiveSavedState)
      .def("SaveState", &Simulators::ISimulator::SaveState)
      .def("RestoreState", &Simulators::ISimulator::RestoreState)
      .def("SetMultithreading", &Simulators::ISimulator::SetMultithreading,
           "multithreading"_a = true)
      .def("GetMultithreading", &Simulators::ISimulator::GetMultithreading)
      .def("IsQcsim", &Simulators::ISimulator::IsQcsim)
      .def("MeasureNoCollapse", &Simulators::ISimulator::MeasureNoCollapse)
      .def("ApplyX", &Simulators::ISimulator::ApplyX, "qubit"_a)
      .def("ApplyY", &Simulators::ISimulator::ApplyY, "qubit"_a)
      .def("ApplyZ", &Simulators::ISimulator::ApplyZ, "qubit"_a)
      .def("ApplyH", &Simulators::ISimulator::ApplyH, "qubit"_a)
      .def("ApplyS", &Simulators::ISimulator::ApplyS, "qubit"_a)
      .def("ApplySDG", &Simulators::ISimulator::ApplySDG, "qubit"_a)
      .def("ApplyT", &Simulators::ISimulator::ApplyT, "qubit"_a)
      .def("ApplyTDG", &Simulators::ISimulator::ApplyTDG, "qubit"_a)
      .def("ApplySX", &Simulators::ISimulator::ApplySx, "qubit"_a)
      .def("ApplySXDG", &Simulators::ISimulator::ApplySxDAG, "qubit"_a)
      .def("ApplyK", &Simulators::ISimulator::ApplyK, "qubit"_a)
      .def("ApplyP", &Simulators::ISimulator::ApplyP, "qubit"_a, "theta"_a)
      .def("ApplyRx", &Simulators::ISimulator::ApplyRx, "qubit"_a, "theta"_a)
      .def("ApplyRy", &Simulators::ISimulator::ApplyRy, "qubit"_a, "theta"_a)
      .def("ApplyRz", &Simulators::ISimulator::ApplyRz, "qubit"_a, "theta"_a)
      .def("ApplyU", &Simulators::ISimulator::ApplyU, "qubit"_a, "theta"_a,
           "phi"_a, "lambda_"_a, "gamma"_a = 0.0)
      .def("ApplyCX", &Simulators::ISimulator::ApplyCX, "control_qubit"_a,
           "target_qubit"_a)
      .def("ApplyCY", &Simulators::ISimulator::ApplyCY, "control_qubit"_a,
           "target_qubit"_a)
      .def("ApplyCZ", &Simulators::ISimulator::ApplyCZ, "control_qubit"_a,
           "target_qubit"_a)
      .def("ApplyCH", &Simulators::ISimulator::ApplyCH, "control_qubit"_a,
           "target_qubit"_a)
      .def("ApplyCSX", &Simulators::ISimulator::ApplyCSx, "control_qubit"_a,
           "target_qubit"_a)
      .def("ApplyCSXDG", &Simulators::ISimulator::ApplyCSxDAG,
           "control_qubit"_a, "target_qubit"_a)
      .def("ApplyCP", &Simulators::ISimulator::ApplyCP, "control_qubit"_a,
           "target_qubit"_a, "theta"_a)
      .def("ApplyCRx", &Simulators::ISimulator::ApplyCRx, "control_qubit"_a,
           "target_qubit"_a, "theta"_a)
      .def("ApplyCRy", &Simulators::ISimulator::ApplyCRy, "control_qubit"_a,
           "target_qubit"_a, "theta"_a)
      .def("ApplyCRz", &Simulators::ISimulator::ApplyCRz, "control_qubit"_a,
           "target_qubit"_a, "theta"_a)
      .def("ApplyCCX", &Simulators::ISimulator::ApplyCCX, "control_qubit1"_a,
           "control_qubit2"_a, "target_qubit"_a)
      .def("ApplySwap", &Simulators::ISimulator::ApplySwap, "qubit1"_a,
           "qubit2"_a)
      .def("ApplyCSwap", &Simulators::ISimulator::ApplyCSwap, "control_qubit"_a,
           "qubit1"_a, "qubit2"_a)
      .def("ApplyCU", &Simulators::ISimulator::ApplyCU, "control_qubit"_a,
           "target_qubit"_a, "theta"_a, "phi"_a, "lambda_"_a, "gamma"_a = 0.0)
      .def("set_seed", &Simulators::ISimulator::SetSeed, "seed"_a)
      .def("density_matrix_trace", &Simulators::ISimulator::DensityMatrixTrace)
      .def("density_matrix_purity",
           &Simulators::ISimulator::DensityMatrixPurity)
      .def("density_matrix_trace_of_square",
           &Simulators::ISimulator::DensityMatrixTraceOfSquare)
      .def("density_matrix_overlap",
           &Simulators::ISimulator::DensityMatrixOverlap, "other"_a)
      .def("density_matrix_hermiticity_residual",
           &Simulators::ISimulator::DensityMatrixHermiticityResidual)
      .def("is_density_matrix_hermitian",
           &Simulators::ISimulator::IsDensityMatrixHermitian, "eps"_a = 1e-10)
      .def("partial_trace", &Simulators::ISimulator::PartialTrace, "qubits"_a)
      .def("fidelity_with_statevector",
           &Simulators::ISimulator::FidelityWithStatevector, "statevector"_a)
      .def("restore_density_matrix_trace",
           &Simulators::ISimulator::RestoreDensityMatrixTrace)
      .def("hermitize_density_matrix",
           &Simulators::ISimulator::HermitizeDensityMatrix)
      .def("trim_mpo", &Simulators::ISimulator::TrimMatrixProductOperator)
      .def("recanonicalize_mpo",
           &Simulators::ISimulator::ReCanonicalizeMatrixProductOperator);

  // --- Maestro Class ---
  nb::class_<Maestro>(m, "Maestro")
      .def(nb::init<>())
      .def("create_simulator", &Maestro::CreateSimulator,
           "sim_type"_a = Simulators::SimulatorType::kQCSim,
           "sim_exec_type"_a = Simulators::SimulationType::kMatrixProductState)
      .def(
          "get_simulator",
          [](Maestro& self, unsigned long int h) {
            return static_cast<Simulators::ISimulator*>(self.GetSimulator(h));
          },
          nb::rv_policy::reference_internal)
      .def("destroy_simulator", &Maestro::DestroySimulator);
}

}  // namespace maestro_bindings
