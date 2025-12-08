#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>

#include "Maestro.h"
#include "Simulators/Simulator.h"
#include "Simulators/Factory.h"
#include "Circuit/Circuit.h"
#include "qasm/QasmCirc.h"

namespace nb = nanobind;

NB_MODULE(maestro_py, m) {
    m.doc() = "Python bindings for Maestro Quantum Simulator";

    // Bind SimulatorType enum
    nb::enum_<Simulators::SimulatorType>(m, "SimulatorType")
        .value("QCSim", Simulators::SimulatorType::kQCSim)
        .value("QiskitAer", Simulators::SimulatorType::kQiskitAer)
        .value("CompositeQiskitAer", Simulators::SimulatorType::kCompositeQiskitAer)
        .value("CompositeQCSim", Simulators::SimulatorType::kCompositeQCSim)
        .export_values();

    // Bind SimulationType enum
    nb::enum_<Simulators::SimulationType>(m, "SimulationType")
        .value("Statevector", Simulators::SimulationType::kStatevector)
        .value("MatrixProductState", Simulators::SimulationType::kMatrixProductState)
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
        .def("SampleCounts", &Simulators::ISimulator::SampleCounts, nb::arg("qubits"), nb::arg("shots") = 1000)
        .def("Initialize", &Simulators::ISimulator::Initialize)
        .def("Reset", &Simulators::ISimulator::Reset);

    // Bind Maestro class
    nb::class_<Maestro>(m, "Maestro")
        .def(nb::init<>())
        .def("CreateSimulator", &Maestro::CreateSimulator, 
             nb::arg("simType") = Simulators::SimulatorType::kQCSim,
             nb::arg("simExecType") = Simulators::SimulationType::kMatrixProductState)
        .def("GetSimulator", [](Maestro& self, unsigned long int simHandle) {
            return static_cast<Simulators::ISimulator*>(self.GetSimulator(simHandle));
        }, nb::rv_policy::reference_internal)
        .def("DestroySimulator", &Maestro::DestroySimulator);

    // Bind Circuit class
    nb::class_<Circuits::Circuit<double>>(m, "Circuit")
        .def("GetMaxQubitIndex", &Circuits::Circuit<double>::GetMaxQubitIndex);

    // Bind QasmToCirc class
    nb::class_<qasm::QasmToCirc<double>>(m, "QasmToCirc")
        .def(nb::init<>())
        .def("ParseAndTranslate", &qasm::QasmToCirc<double>::ParseAndTranslate);
}
