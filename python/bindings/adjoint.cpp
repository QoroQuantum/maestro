#include <nanobind/ndarray.h>

#include "Execution/AdjointDifferentiation.h"
#include "core.h"
#include "module.h"

namespace maestro_bindings
{

using MaestroExecution::Adjoint::Observable;
using MaestroExecution::Adjoint::Program;
using MaestroExecution::Adjoint::Readout;
using NumpyArray = nb::ndarray<nb::numpy, double>;

/** Python-facing request for basis-state probabilities of some qubits. */
struct ProbabilitiesReadout
{
    std::vector<Types::qubit_t> qubits;
};

namespace
{

// Accepts a Pauli string, a ';'-separated string of them, or a list whose
// items are Pauli strings, lists of (coefficient, Pauli string) pairs, or
// maestro.Probabilities readouts.
std::vector<Readout> ParseReadouts(const nb::handle &observables)
{
    std::vector<Readout> out;
    if (nb::isinstance<nb::str>(observables))
    {
        for (const auto &pauli : ParseObservables(nb::borrow(observables)))
            out.push_back(Readout::Expectation({{1., pauli}}));
        return out;
    }
    if (!nb::isinstance<nb::sequence>(observables))
        throw nb::type_error("Observables must be a Pauli string or a list of Pauli strings, (coefficient, Pauli string) term lists or "
                             "maestro.Probabilities.");
    for (const auto &item : observables)
    {
        if (nb::isinstance<nb::str>(item))
        {
            out.push_back(Readout::Expectation({{1., nb::cast<std::string>(item)}}));
            continue;
        }
        if (nb::isinstance<ProbabilitiesReadout>(item))
        {
            out.push_back(Readout::Probabilities(nb::cast<const ProbabilitiesReadout &>(item).qubits));
            continue;
        }
        Observable obs;
        for (const auto &term : item)
        {
            std::pair<double, std::string> parsed;
            if (!nb::try_cast(term, parsed))
                throw nb::type_error("Observable terms must be (coefficient, Pauli string) pairs.");
            obs.push_back(std::move(parsed));
        }
        out.push_back(Readout::Expectation(std::move(obs)));
    }
    return out;
}

// Returns the parameter rows as a flat row-major copy and their count. None
// means one row holding the circuit's own parameters.
std::vector<double> ParseRows(const Program &program, const nb::handle &params, size_t &numRows)
{
    const size_t nParams = program.NumParams();
    if (params.is_none())
    {
        numRows = 1;
        return program.DefaultParams();
    }
    nb::object np = nb::module_::import_("numpy");
    nb::object arr = np.attr("ascontiguousarray")(params, "dtype"_a = np.attr("float64"));
    const auto ndim = nb::cast<size_t>(arr.attr("ndim"));
    const auto shape = nb::cast<std::vector<size_t>>(arr.attr("shape"));
    if (ndim == 1 && shape[0] == nParams)
        numRows = 1;
    else if (ndim == 2 && shape[1] == nParams)
        numRows = shape[0];
    else
        throw nb::value_error(("params must have shape (num_params,) or (batch, num_params) with num_params = " + std::to_string(nParams)).c_str());
    auto flat = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(arr.attr("reshape")(-1));
    const double *data = flat.data();
    return std::vector<double>(data, data + numRows * nParams);
}

NumpyArray ToNumpy(std::vector<double> &&values, std::vector<size_t> shape)
{
    auto *buffer = new std::vector<double>(std::move(values));
    nb::capsule owner(buffer, [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    return NumpyArray(buffer->data(), shape.size(), shape.data(), owner);
}

// The program always runs on QCSim's CPU statevector kernels; refuse other
// backends rather than silently ignore the choice (a GPU-sized register would
// otherwise be allocated in host memory).
void RequireStatevector(const SimulatorConfig &config)
{
    if (config.simulation_type != Simulators::SimulationType::kStatevector)
        throw nb::value_error("Adjoint differentiation requires simulation_type=Statevector.");
    if (config.simulator_type != Simulators::SimulatorType::kQCSim && config.simulator_type != Simulators::SimulatorType::kCompositeQCSim)
        throw nb::value_error("Adjoint differentiation runs on the CPU statevector and requires simulator_type=QCSim.");
}

nb::dict Expectations(const Program &program, const nb::handle &observables, const nb::handle &params, const SimulatorConfig &config)
{
    RequireStatevector(config);
    const auto obs = ParseReadouts(observables);
    const size_t nCols = MaestroExecution::Adjoint::Columns(obs);
    size_t numRows = 0;
    const auto rows = ParseRows(program, params, numRows);
    std::vector<double> values;
    {
        nb::gil_scoped_release release;
        values = program.Values(obs, rows, numRows, config.enable_causal_cone_reduction);
    }
    nb::dict out;
    out["expectation_values"] = ToNumpy(std::move(values), {numRows, nCols});
    return out;
}

nb::dict Jacobian(const Program &program, const nb::handle &observables, const nb::handle &params, const SimulatorConfig &config)
{
    RequireStatevector(config);
    const auto obs = ParseReadouts(observables);
    const size_t nCols = MaestroExecution::Adjoint::Columns(obs);
    size_t numRows = 0;
    const auto rows = ParseRows(program, params, numRows);
    std::vector<double> values, jac;
    {
        nb::gil_scoped_release release;
        program.Jacobian(obs, rows, numRows, config.enable_causal_cone_reduction, values, jac);
    }
    nb::dict out;
    out["expectation_values"] = ToNumpy(std::move(values), {numRows, nCols});
    out["jacobian"] = ToNumpy(std::move(jac), {numRows, nCols, program.NumParams()});
    return out;
}

nb::dict VectorJacobianProduct(const Program &program, const nb::handle &observables, const nb::handle &cotangents, const nb::handle &params,
                               const SimulatorConfig &config)
{
    RequireStatevector(config);
    const auto obs = ParseReadouts(observables);
    const size_t nCols = MaestroExecution::Adjoint::Columns(obs);
    size_t numRows = 0;
    const auto rows = ParseRows(program, params, numRows);

    nb::object np = nb::module_::import_("numpy");
    nb::object dy = np.attr("ascontiguousarray")(cotangents, "dtype"_a = np.attr("float64")).attr("reshape")(-1);
    auto dyArr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(dy);
    if (dyArr.size() != numRows * nCols)
        throw nb::value_error("cotangents must have shape (batch, num_values).");
    const std::vector<double> dyVec(dyArr.data(), dyArr.data() + dyArr.size());

    std::vector<double> values, vjp;
    {
        nb::gil_scoped_release release;
        program.VectorJacobianProduct(obs, dyVec, rows, numRows, config.enable_causal_cone_reduction, values, vjp);
    }
    nb::dict out;
    out["expectation_values"] = ToNumpy(std::move(values), {numRows, nCols});
    out["vjp"] = ToNumpy(std::move(vjp), {numRows, program.NumParams()});
    return out;
}

Program Compile(const std::shared_ptr<Circuits::Circuit<double>> &circuit)
{
    require_circuit(circuit);
    try
    {
        return Program(*circuit);
    }
    catch (const std::invalid_argument &e)
    {
        throw nb::value_error(e.what());
    }
}

} // namespace

void bind_adjoint(nb::module_ &m)
{
    nb::class_<ProbabilitiesReadout>(m, "Probabilities",
                                     "Readout of the basis-state probabilities of some qubits, for\n"
                                     "AdjointProgram. It produces 2**len(qubits) values; the first listed\n"
                                     "qubit is the least significant bit of the outcome index.")
        .def(nb::init<std::vector<Types::qubit_t>>(), "qubits"_a)
        .def_ro("qubits", &ProbabilitiesReadout::qubits)
        .def("__repr__", [](const ProbabilitiesReadout &r) {
            std::string out = "Probabilities([";
            for (size_t i = 0; i < r.qubits.size(); ++i)
                out += (i ? ", " : "") + std::to_string(r.qubits[i]);
            return out + "])";
        });

    nb::class_<Program>(m, "AdjointProgram",
                        "A unitary circuit compiled for batched Pauli expectation values and\n"
                        "adjoint derivatives on the CPU statevector.\n\n"
                        "Compile once, then evaluate many parameter rows. The differentiable\n"
                        "parameters, in circuit order, are one angle for p, rx, ry, rz, cp, crx,\n"
                        "cry and crz; theta, phi, lambda for u; theta, phi, lambda, gamma for cu.\n"
                        "Each row of ``params`` replaces all of them.\n\n"
                        "Observables are Pauli strings (character q acts on qubit q), lists of\n"
                        "(coefficient, Pauli string) terms, or maestro.Probabilities(qubits).\n"
                        "An observable gives one value and Probabilities gives 2**k values;\n"
                        "results list them side by side in order (num_values columns). Gates\n"
                        "outside a readout's causal cone are skipped when\n"
                        "config.enable_causal_cone_reduction is set.")
        .def(
            "__init__", [](Program *self, const std::shared_ptr<Circuits::Circuit<double>> &circuit) { new (self) Program(Compile(circuit)); }, "circuit"_a)
        .def_prop_ro("num_params", &Program::NumParams)
        .def_prop_ro("num_qubits", &Program::NumQubits)
        .def_prop_ro(
            "parameters", [](const Program &p) { return ToNumpy(std::vector<double>(p.DefaultParams()), {p.NumParams()}); }, nb::rv_policy::move,
            "The circuit's own parameter values, in program order.")
        .def("expectation_values", &Expectations, "observables"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
             "Values of every readout for each parameter row.\n\n"
             "Returns a dict with 'expectation_values' of shape (batch, num_values).")
        .def("jacobian", &Jacobian, "observables"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
             "Values and their adjoint Jacobian for each parameter row, one backward\n"
             "sweep per value.\n\n"
             "Returns a dict with 'expectation_values' (batch, num_values) and\n"
             "'jacobian' (batch, num_values, num_params).")
        .def("vjp", &VectorJacobianProduct, "observables"_a, "cotangents"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
             "Vector-Jacobian product, one backward sweep per row for all readouts.\n\n"
             "cotangents has shape (batch, num_values). Returns a dict with\n"
             "'expectation_values' (batch, num_values) and 'vjp' (batch, num_params).");

    m.def(
        "adjoint_jacobian",
        [](const std::shared_ptr<Circuits::Circuit<double>> &circuit, const nb::handle &observables, const nb::handle &params, const SimulatorConfig &config) {
            return Jacobian(Compile(circuit), observables, params, config);
        },
        "circuit"_a, "observables"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
        "Adjoint Jacobian of expectation values and probabilities; see AdjointProgram.jacobian.");
}

} // namespace maestro_bindings
