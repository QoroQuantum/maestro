#include "Simulators/Core/GenericGateValidation.h"
#include "core.h"
#include "module.h"

namespace maestro_bindings
{

template <int Dimension>
static Eigen::Matrix<std::complex<double>, Dimension, Dimension> PythonGateMatrix(const std::vector<std::vector<std::complex<double>>> &values)
{
    if (values.size() != Dimension)
        throw std::invalid_argument("Generic gate matrix has the wrong dimensions");
    Eigen::Matrix<std::complex<double>, Dimension, Dimension> matrix;
    for (int row = 0; row < Dimension; ++row)
    {
        if (values[row].size() != Dimension)
            throw std::invalid_argument("Generic gate matrix has the wrong dimensions");
        for (int col = 0; col < Dimension; ++col)
            matrix(row, col) = values[row][col];
    }
    return matrix;
}

void bind_simulator(nb::module_ &m)
{
    nb::class_<Simulators::GateFusionStatistics>(m, "GateFusionStatistics")
        .def_ro("submittedGates", &Simulators::GateFusionStatistics::submittedGates)
        .def_ro("backendGates", &Simulators::GateFusionStatistics::backendGates)
        .def_ro("fusedBlocks", &Simulators::GateFusionStatistics::fusedBlocks);

    nb::class_<Simulators::ISimulator>(m, "Simulator")
        .def("GetGateFusionMaxQubits", &Simulators::ISimulator::GetGateFusionMaxQubits)
        .def("IsGateFusionEnabled", &Simulators::ISimulator::IsGateFusionEnabled)
        .def("GetGateFusionStatistics", &Simulators::ISimulator::GetGateFusionStatistics)
        .def(
            "ApplyGenericOneQubitGate",
            [](Simulators::ISimulator &sim, Types::qubit_t q0, const std::vector<std::vector<std::complex<double>>> &values) {
                const auto matrix = PythonGateMatrix<2>(values);
                Simulators::ValidateGenericGate(sim, {q0}, matrix);
                sim.ApplyGenericOneQubitGate(q0, matrix);
            },
            "qubit0"_a, "matrix"_a,
            "Apply a 2x2 matrix; the first target is the least-significant local "
            "bit.")
        .def(
            "ApplyGenericTwoQubitGate",
            [](Simulators::ISimulator &sim, Types::qubit_t q0, Types::qubit_t q1, const std::vector<std::vector<std::complex<double>>> &values) {
                const auto matrix = PythonGateMatrix<4>(values);
                Simulators::ValidateGenericGate(sim, {q0, q1}, matrix);
                sim.ApplyGenericTwoQubitGate(q0, q1, matrix);
            },
            "qubit0"_a, "qubit1"_a, "matrix"_a,
            "Apply a 4x4 matrix; the first target is the least-significant local "
            "bit.")
        .def(
            "ApplyGenericThreeQubitGate",
            [](Simulators::ISimulator &sim, Types::qubit_t q0, Types::qubit_t q1, Types::qubit_t q2,
               const std::vector<std::vector<std::complex<double>>> &values) {
                const auto matrix = PythonGateMatrix<8>(values);
                Simulators::ValidateGenericGate(sim, {q0, q1, q2}, matrix);
                sim.ApplyGenericThreeQubitGate(q0, q1, q2, matrix);
            },
            "qubit0"_a, "qubit1"_a, "qubit2"_a, "matrix"_a,
            "Apply a 8x8 matrix; the first target is the least-significant local "
            "bit.")
        // Low-level operations from Interface.h, using Python-owned results.
        .def("InitializeSimulator", &Simulators::ISimulator::Initialize)
        .def("Initialize", &Simulators::ISimulator::Initialize)
        .def("ResetSimulator", &Simulators::ISimulator::Reset)
        .def("Reset", &Simulators::ISimulator::Reset)
        .def("ConfigureSimulator", &Simulators::ISimulator::Configure, "key"_a, "value"_a)
        .def("Configure", &Simulators::ISimulator::Configure, "key"_a, "value"_a)
        .def("GetConfiguration", &Simulators::ISimulator::GetConfiguration, "key"_a)
        .def("AllocateQubits", &Simulators::ISimulator::AllocateQubits, "num_qubits"_a)
        .def("GetNumberOfQubits", &Simulators::ISimulator::GetNumberOfQubits)
        .def("ClearSimulator", &Simulators::ISimulator::Clear)
        .def("Clear", &Simulators::ISimulator::Clear)
        .def("Measure", &Simulators::ISimulator::Measure, "qubits"_a,
             "Measure and collapse the selected qubits; the first listed qubit "
             "is the least-significant result bit.")
        .def("MeasureMany", &Simulators::ISimulator::MeasureMany, "qubits"_a)
        .def("MeasureNoCollapseMany", &Simulators::ISimulator::MeasureNoCollapseMany)
        .def(
            "SampleCountsMany",
            [](Simulators::ISimulator &sim, const Types::qubits_vector &qubits, size_t shots) {
                nb::dict result;
                for (const auto &entry : sim.SampleCountsMany(qubits, shots))
                {
                    nb::list bits;
                    for (bool bit : entry.first)
                        bits.append(bit);
                    result[nb::tuple(bits)] = entry.second;
                }
                return result;
            },
            "qubits"_a, "shots"_a = 1000)
        .def("ApplyReset", &Simulators::ISimulator::ApplyReset, "qubits"_a)
        .def("Probability", &Simulators::ISimulator::Probability, "outcome"_a)
        .def("Amplitude", &Simulators::ISimulator::Amplitude, "outcome"_a)
        .def("AllProbabilities", &Simulators::ISimulator::AllProbabilities)
        .def("GetStateVector", &Simulators::ISimulator::GetStateVector, nb::call_guard<nb::gil_scoped_release>())
        .def("get_statevector", &Simulators::ISimulator::GetStateVector, nb::call_guard<nb::gil_scoped_release>(),
             "Full pure-state amplitudes in logical basis order (q0 is the low "
             "bit).")
        .def("ExpectationValueOperators", &Simulators::ISimulator::ExpectationValueOperators, "qubits"_a, "matrices"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("expectation_value_operators", &Simulators::ISimulator::ExpectationValueOperators, "qubits"_a, "matrices"_a,
             nb::call_guard<nb::gil_scoped_release>(),
             "MPS expectation of ordered one-qubit operators; repeated targets "
             "are allowed.")
        .def("Probabilities", &Simulators::ISimulator::Probabilities, "outcomes"_a, "Return probabilities for the given basis-state indices.")
        .def("SampleCounts", &Simulators::ISimulator::SampleCounts, "qubits"_a, "shots"_a = 1000,
             "Sample without collapsing the state, returning {integer_outcome: "
             "count}; the first listed qubit is the least-significant result bit.")
        .def("GetSimulatorType", &Simulators::ISimulator::GetType)
        .def("GetSimulationType", &Simulators::ISimulator::GetSimulationType)
        .def("FlushSimulator", &Simulators::ISimulator::Flush)
        .def("Flush", &Simulators::ISimulator::Flush)
        .def("SaveStateToInternalDestructive", &Simulators::ISimulator::SaveStateToInternalDestructive)
        .def("RestoreInternalDestructiveSavedState", &Simulators::ISimulator::RestoreInternalDestructiveSavedState)
        .def("SaveState", &Simulators::ISimulator::SaveState)
        .def("RestoreState", &Simulators::ISimulator::RestoreState)
        .def("SetMultithreading", &Simulators::ISimulator::SetMultithreading, "multithreading"_a = true)
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
        .def("ApplyU", &Simulators::ISimulator::ApplyU, "qubit"_a, "theta"_a, "phi"_a, "lambda_"_a, "gamma"_a = 0.0)
        .def("ApplyCX", &Simulators::ISimulator::ApplyCX, "control_qubit"_a, "target_qubit"_a)
        .def("ApplyCY", &Simulators::ISimulator::ApplyCY, "control_qubit"_a, "target_qubit"_a)
        .def("ApplyCZ", &Simulators::ISimulator::ApplyCZ, "control_qubit"_a, "target_qubit"_a)
        .def("ApplyCH", &Simulators::ISimulator::ApplyCH, "control_qubit"_a, "target_qubit"_a)
        .def("ApplyCSX", &Simulators::ISimulator::ApplyCSx, "control_qubit"_a, "target_qubit"_a)
        .def("ApplyCSXDG", &Simulators::ISimulator::ApplyCSxDAG, "control_qubit"_a, "target_qubit"_a)
        .def("ApplyCP", &Simulators::ISimulator::ApplyCP, "control_qubit"_a, "target_qubit"_a, "theta"_a)
        .def("ApplyCRx", &Simulators::ISimulator::ApplyCRx, "control_qubit"_a, "target_qubit"_a, "theta"_a)
        .def("ApplyCRy", &Simulators::ISimulator::ApplyCRy, "control_qubit"_a, "target_qubit"_a, "theta"_a)
        .def("ApplyCRz", &Simulators::ISimulator::ApplyCRz, "control_qubit"_a, "target_qubit"_a, "theta"_a)
        .def("ApplyCCX", &Simulators::ISimulator::ApplyCCX, "control_qubit1"_a, "control_qubit2"_a, "target_qubit"_a)
        .def("ApplySwap", &Simulators::ISimulator::ApplySwap, "qubit1"_a, "qubit2"_a)
        .def("ApplyCSwap", &Simulators::ISimulator::ApplyCSwap, "control_qubit"_a, "qubit1"_a, "qubit2"_a)
        .def("ApplyCU", &Simulators::ISimulator::ApplyCU, "control_qubit"_a, "target_qubit"_a, "theta"_a, "phi"_a, "lambda_"_a, "gamma"_a = 0.0)
        .def("set_seed", &Simulators::ISimulator::SetSeed, "seed"_a)
        .def("probability_bits", &Simulators::ISimulator::ProbabilityBits, "bits"_a)
        .def("density_matrix_element_bits", &Simulators::ISimulator::DensityMatrixElementBits, "row"_a, "col"_a)
        .def("get_density_matrix", &Simulators::ISimulator::GetDensityMatrix, "normalized"_a = true)
        .def("expectation_value_complex", &Simulators::ISimulator::ExpectationValueComplex, "pauli"_a, "normalized"_a = true)
        .def("ExpectationValues", &Simulators::ISimulator::ExpectationValues, "paulis"_a, nb::call_guard<nb::gil_scoped_release>(),
             "Evaluate Pauli strings together on the current state, in input "
             "order.")
        .def("expectation_values", &Simulators::ISimulator::ExpectationValues, "paulis"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("expectation_values_complex", &Simulators::ISimulator::ExpectationValuesComplex, "paulis"_a, "normalized"_a = true,
             nb::call_guard<nb::gil_scoped_release>(), "Batch MPO expectations; normalized=False returns raw traces.")
        .def("ExpectationValuesComplex", &Simulators::ISimulator::ExpectationValuesComplex, "paulis"_a, "normalized"_a = true,
             nb::call_guard<nb::gil_scoped_release>())
        .def("apply_operator", &Simulators::ISimulator::ApplyOperator, "qubits"_a, "matrix"_a, "normalize"_a = false)
        .def("move_at_beginning_of_chain", &Simulators::ISimulator::MoveAtBeginningOfChain, "qubits"_a)
        .def("density_matrix_trace", &Simulators::ISimulator::DensityMatrixTrace)
        .def("density_matrix_purity", &Simulators::ISimulator::DensityMatrixPurity)
        .def("density_matrix_trace_of_square", &Simulators::ISimulator::DensityMatrixTraceOfSquare)
        .def("density_matrix_overlap", &Simulators::ISimulator::DensityMatrixOverlap, "other"_a)
        .def("density_matrix_hermiticity_residual", &Simulators::ISimulator::DensityMatrixHermiticityResidual)
        .def("is_density_matrix_hermitian", &Simulators::ISimulator::IsDensityMatrixHermitian, "eps"_a = 1e-10)
        .def("partial_trace", &Simulators::ISimulator::PartialTrace, "qubits"_a)
        .def("fidelity_with_statevector", &Simulators::ISimulator::FidelityWithStatevector, "statevector"_a)
        .def("restore_density_matrix_trace", &Simulators::ISimulator::RestoreDensityMatrixTrace)
        .def("hermitize_density_matrix", &Simulators::ISimulator::HermitizeDensityMatrix)
        .def("trim", &Simulators::ISimulator::Trim)
        .def("recanonicalize", &Simulators::ISimulator::ReCanonicalize);

    // --- Maestro Class ---
    nb::class_<Maestro>(m, "Maestro")
        .def(nb::init<>())
        .def("create_simulator", &Maestro::CreateSimulator, "sim_type"_a = Simulators::SimulatorType::kQCSim,
             "sim_exec_type"_a = Simulators::SimulationType::kMatrixProductState)
        .def(
            "get_simulator", [](Maestro &self, unsigned long int h) { return static_cast<Simulators::ISimulator *>(self.GetSimulator(h)); },
            nb::rv_policy::reference_internal)
        .def("destroy_simulator", &Maestro::DestroySimulator);
}

} // namespace maestro_bindings
