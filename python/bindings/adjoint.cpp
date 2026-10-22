#include <nanobind/ndarray.h>

#include "Execution/AdjointDifferentiation.h"
#include "core.h"
#include "module.h"

namespace maestro_bindings
{

using MaestroExecution::Adjoint::Observable;
using MaestroExecution::Adjoint::Program;
using NumpyArray = nb::ndarray<nb::numpy, double>;

namespace
{

// Accepts a Pauli string, a ';'-separated string of them, or a list whose
// items are Pauli strings or lists of (coefficient, Pauli string) pairs.
std::vector<Observable> ParseWeightedObservables(const nb::handle &observables)
{
    std::vector<Observable> out;
    if (nb::isinstance<nb::str>(observables))
    {
        for (const auto &pauli : ParseObservables(nb::borrow(observables)))
            out.push_back({{1., pauli}});
        return out;
    }
    if (!nb::isinstance<nb::sequence>(observables))
        throw nb::type_error("Observables must be a Pauli string or a list of Pauli strings or (coefficient, Pauli string) term lists.");
    for (const auto &item : observables)
    {
        if (nb::isinstance<nb::str>(item))
        {
            out.push_back({{1., nb::cast<std::string>(item)}});
            continue;
        }
        Observable obs;
        for (const auto &term : item)
        {
            const auto pair = nb::cast<std::pair<double, std::string>>(term);
            obs.push_back(pair);
        }
        out.push_back(std::move(obs));
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

void RequireStatevector(const SimulatorConfig &config)
{
    if (config.simulation_type != Simulators::SimulationType::kStatevector)
        throw nb::value_error("Adjoint differentiation requires simulation_type=Statevector.");
}

nb::dict Expectations(const Program &program, const nb::handle &observables, const nb::handle &params, const SimulatorConfig &config)
{
    RequireStatevector(config);
    const auto obs = ParseWeightedObservables(observables);
    size_t numRows = 0;
    const auto rows = ParseRows(program, params, numRows);
    std::vector<double> values;
    {
        nb::gil_scoped_release release;
        values = program.Expectations(obs, rows, numRows, config.enable_causal_cone_reduction);
    }
    nb::dict out;
    out["expectation_values"] = ToNumpy(std::move(values), {numRows, obs.size()});
    return out;
}

nb::dict Jacobian(const Program &program, const nb::handle &observables, const nb::handle &params, const SimulatorConfig &config)
{
    RequireStatevector(config);
    const auto obs = ParseWeightedObservables(observables);
    size_t numRows = 0;
    const auto rows = ParseRows(program, params, numRows);
    std::vector<double> values, jac;
    {
        nb::gil_scoped_release release;
        program.Jacobian(obs, rows, numRows, config.enable_causal_cone_reduction, values, jac);
    }
    nb::dict out;
    out["expectation_values"] = ToNumpy(std::move(values), {numRows, obs.size()});
    out["jacobian"] = ToNumpy(std::move(jac), {numRows, obs.size(), program.NumParams()});
    return out;
}

nb::dict VectorJacobianProduct(const Program &program, const nb::handle &observables, const nb::handle &cotangents, const nb::handle &params,
                               const SimulatorConfig &config)
{
    RequireStatevector(config);
    const auto obs = ParseWeightedObservables(observables);
    size_t numRows = 0;
    const auto rows = ParseRows(program, params, numRows);

    nb::object np = nb::module_::import_("numpy");
    nb::object dy = np.attr("ascontiguousarray")(cotangents, "dtype"_a = np.attr("float64")).attr("reshape")(-1);
    auto dyArr = nb::cast<nb::ndarray<const double, nb::c_contig, nb::device::cpu>>(dy);
    if (dyArr.size() != numRows * obs.size())
        throw nb::value_error("cotangents must have shape (batch, num_observables).");
    const std::vector<double> dyVec(dyArr.data(), dyArr.data() + dyArr.size());

    std::vector<double> values, vjp;
    {
        nb::gil_scoped_release release;
        program.VectorJacobianProduct(obs, dyVec, rows, numRows, config.enable_causal_cone_reduction, values, vjp);
    }
    nb::dict out;
    out["expectation_values"] = ToNumpy(std::move(values), {numRows, obs.size()});
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
    nb::class_<Program>(m, "AdjointProgram",
                        "A unitary circuit compiled for batched Pauli expectation values and\n"
                        "adjoint derivatives on the CPU statevector.\n\n"
                        "Compile once, then evaluate many parameter rows. The differentiable\n"
                        "parameters, in circuit order, are one angle for p, rx, ry, rz, cp, crx,\n"
                        "cry and crz; theta, phi, lambda for u; theta, phi, lambda, gamma for cu.\n"
                        "Each row of ``params`` replaces all of them.\n\n"
                        "Observables are Pauli strings (character q acts on qubit q) or lists\n"
                        "of (coefficient, Pauli string) terms. Gates outside an observable's\n"
                        "causal cone are skipped when config.enable_causal_cone_reduction is set.")
        .def(
            "__init__", [](Program *self, const std::shared_ptr<Circuits::Circuit<double>> &circuit) { new (self) Program(Compile(circuit)); }, "circuit"_a)
        .def_prop_ro("num_params", &Program::NumParams)
        .def_prop_ro("num_qubits", &Program::NumQubits)
        .def_prop_ro(
            "parameters", [](const Program &p) { return ToNumpy(std::vector<double>(p.DefaultParams()), {p.NumParams()}); }, nb::rv_policy::move,
            "The circuit's own parameter values, in program order.")
        .def("expectation_values", &Expectations, "observables"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
             "Expectation values for each parameter row.\n\n"
             "Returns a dict with 'expectation_values' of shape (batch, num_observables).")
        .def("jacobian", &Jacobian, "observables"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
             "Expectation values and their adjoint Jacobian for each parameter row.\n\n"
             "Returns a dict with 'expectation_values' (batch, num_observables) and\n"
             "'jacobian' (batch, num_observables, num_params).")
        .def("vjp", &VectorJacobianProduct, "observables"_a, "cotangents"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
             "Vector-Jacobian product, one backward sweep per row.\n\n"
             "cotangents has shape (batch, num_observables). Returns a dict with\n"
             "'expectation_values' (batch, num_observables) and 'vjp' (batch, num_params).");

    m.def(
        "adjoint_jacobian",
        [](const std::shared_ptr<Circuits::Circuit<double>> &circuit, const nb::handle &observables, const nb::handle &params, const SimulatorConfig &config) {
            return Jacobian(Compile(circuit), observables, params, config);
        },
        "circuit"_a, "observables"_a, "params"_a = nb::none(), "config"_a = SimulatorConfig{},
        "Adjoint Jacobian of Pauli expectation values; see AdjointProgram.jacobian.");
}

} // namespace maestro_bindings
