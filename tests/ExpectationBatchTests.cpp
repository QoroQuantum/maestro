#include "Network/SimpleDisconnectedNetwork.h"
#include "Simulators/Core/Factory.h"
#include "Simulators/Fusion/FusionSimulator.h"
#include "maestrolib/Interface.h"
#include <boost/json.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace
{
using Method = Simulators::SimulationType;
using Backend = Simulators::SimulatorType;
using CF = Circuits::CircuitFactory<>;
using Gate = Circuits::QuantumGateType;
using Circuit = Circuits::Circuit<>;
using Net = Network::SimpleDisconnectedNetwork<>;
namespace j = boost::json;
size_t checks = 0;
Backend backend = Backend::kQCSim;
bool single = false;

void Check(bool ok, const char *message)
{
    ++checks;
    if (!ok)
        throw std::runtime_error(message);
}

void Near(std::complex<double> a, std::complex<double> b)
{
    if (std::abs(a - b) >= (single ? 5e-5 : 1e-9))
        std::cerr << "Check " << checks << ": " << std::setprecision(17) << a << " != " << b << '\n';
    Check(std::abs(a - b) < (single ? 5e-5 : 1e-9), "Expectation mismatch");
}

auto Make(Method method, size_t n = 4)
{
    auto sim = Simulators::SimulatorsFactory::CreateSimulator(backend, method);
    sim->SetMultithreading(false);
    sim->Configure("precision", single ? "single" : "double");
    sim->AllocateQubits(n);
    sim->Initialize();
    return sim;
}

void RealBatches()
{
    const std::vector<std::string> paulis{"XIZY", "ZZZZ", "IIII", "XIZY", "YIIY", "ZIIZ"};
    for (auto method : {Method::kStatevector, Method::kMatrixProductState, Method::kMatrixProductOperator, Method::kDensityMatrix})
        for (bool fusion : {false, true})
        {
            auto sim = Make(method);
            sim->Configure("gate_fusion", fusion ? "true" : "false");
            if (method == Method::kMatrixProductState || method == Method::kMatrixProductOperator)
                sim->SetInitialQubitsMap({3, 1, 0, 2});
            sim->ApplyH(0);
            sim->ApplyRy(2, .31);
            sim->ApplyCX(0, 3);
            sim->ApplyRz(3, .17);
            const auto values = sim->ExpectationValues(paulis); // flush pending gates
            Check(values.size() == paulis.size(), "Batch changed length");
            for (size_t i = 0; i < paulis.size(); ++i)
                Near(values[i], sim->ExpectationValue(paulis[i]));
            Check(sim->ExpectationValues({}).empty(), "Empty real batch");
            const std::vector<std::string> padded{"", "Z", "zi", "IIIIZ", "IIIIX"};
            const auto padding = sim->ExpectationValues(padded);
            for (size_t i = 0; i < padded.size(); ++i)
                Near(padding[i], sim->ExpectationValue(padded[i]));
            sim->SaveState();
            sim->ApplyX(0);
            const auto changed = sim->ExpectationValues(paulis);
            for (size_t i = 0; i < paulis.size(); ++i)
                Near(changed[i], sim->ExpectationValue(paulis[i]));
            sim->RestoreState();
            const auto restored = sim->ExpectationValues(paulis);
            for (size_t i = 0; i < values.size(); ++i)
                Near(values[i], restored[i]);
        }
}

void ComplexAndC()
{
    auto sim = Make(Method::kMatrixProductOperator, 2);
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    Eigen::Matrix2cd scale = 2. * Eigen::Matrix2cd::Identity();
    sim->ApplyOperator({0}, scale, false); // trace four, not one
    sim->MoveAtBeginningOfChain({1});
    const std::vector<std::string> paulis{"XX", "YY", "ZZ", "II", "XX"};
    for (bool normalized : {false, true})
    {
        const auto values = sim->ExpectationValuesComplex(paulis, normalized);
        for (size_t i = 0; i < paulis.size(); ++i)
            Near(values[i], sim->ExpectationValueComplex(paulis[i], normalized));
        Near(values[0], normalized ? 1. : 4.);
        Near(values[1], normalized ? -1. : -4.);
    }
    Check(sim->ExpectationValuesComplex({}).empty(), "Empty complex batch");
    const char *input[]{"XX", "YY", "II"};
    double real[4]{9, 9, 9, 9}, imag[4]{9, 9, 9, 9};
    Check(MaestroExpectationValues(sim.get(), input, 3, real, 4), "C real batch failed");
    Near(real[0], 1);
    Near(real[1], -1);
    Near(real[2], 1);
    Near(real[3], 9);
    Check(MaestroExpectationValuesComplex(sim.get(), input, 3, 0, real, imag, 4), "C complex batch failed");
    Near(real[0], 4);
    Near(real[1], -4);
    Near(imag[0], 0);
    Near(imag[3], 9);
    const char *invalid[]{"XX", "bad"};
    Check(!MaestroExpectationValuesComplex(sim.get(), invalid, 2, 1, real, imag, 4), "C invalid Pauli accepted");
    Near(real[0], 4);
    Near(imag[0], 0);
    Check(!MaestroExpectationValues(sim.get(), input, 3, real, 2), "C capacity ignored");
    Near(real[0], 4);
    const char *nullString[]{"XX", nullptr};
    Check(!MaestroExpectationValues(sim.get(), nullString, 2, real, 4), "C null string accepted");
    Check(!MaestroExpectationValues(sim.get(), nullptr, 1, real, 4), "C null input accepted");
    Check(!MaestroExpectationValues(sim.get(), input, 1, nullptr, 4), "C null output accepted");
    Check(MaestroExpectationValues(sim.get(), nullptr, 0, nullptr, 0), "C empty real batch failed");
    Check(MaestroExpectationValuesComplex(sim.get(), nullptr, 0, 1, nullptr, nullptr, 0), "C empty complex batch failed");
    auto unsupported = Make(Method::kStatevector, 2);
    Check(!MaestroExpectationValuesComplex(unsupported.get(), input, 3, 1, real, imag, 4), "Unsupported complex query accepted");
    // Cross the native MPO batch tile size, retaining duplicate ordering.
    std::vector<std::string> many(300, "XX");
    for (size_t i = 1; i < many.size(); i += 2)
        many[i] = "YY";
    const auto tiled = sim->ExpectationValuesComplex(many, false);
    for (size_t i = 0; i < tiled.size(); ++i)
        Near(tiled[i], i % 2 ? -4. : 4.);
    sim->ApplyOperator({0}, Eigen::MatrixXcd::Zero(2, 2), false);
    for (const auto &value : sim->ExpectationValuesComplex(paulis, false))
        Near(value, 0.);
    real[0] = 17.;
    imag[0] = 18.;
    Check(!MaestroExpectationValuesComplex(sim.get(), input, 3, 1, real, imag, 4), "Zero-trace normalized batch accepted");
    Near(real[0], 17.);
    Near(imag[0], 18.);
}

struct Recording : Simulators::Private::FusionSimulator<Simulators::Private::FusionState>
{
    using Base = Simulators::Private::FusionSimulator<Simulators::Private::FusionState>;

    explicit Recording(std::shared_ptr<Simulators::ISimulator> sim) : Base(std::move(sim))
    {
    }

    size_t batches = 0, singles = 0;
    size_t dense = 0;

    std::vector<std::complex<double>> GetStateVector() override
    {
        ++dense;
        return Base::GetStateVector();
    }

    std::vector<double> ExpectationValues(const std::vector<std::string> &p) override
    {
        ++batches;
        return Base::ExpectationValues(p);
    }

    double ExpectationValue(const std::string &p) override
    {
        ++singles;
        return Base::ExpectationValue(p);
    }

    std::unique_ptr<Simulators::ISimulator> Clone() override
    {
        throw std::logic_error("Unexpected clone");
    }
};

struct QueryNetwork : Net
{
    explicit QueryNetwork(const std::shared_ptr<Recording> &sim) : Net({4}, {4})
    {
        simulator = sim;
        recreateIfNeeded = false;
    }

    ExecuteResults RepeatedExecute(const std::shared_ptr<Circuit> &, size_t) override
    {
        return {};
    }

    ExecuteResults RepeatedExecuteOnHost(const std::shared_ptr<Circuit> &, size_t, size_t) override
    {
        return {};
    }
};

void NetworkDispatch()
{
    auto record = std::make_shared<Recording>(Make(Method::kMatrixProductState));
    record->ApplyX(3);
    QueryNetwork query(record);
    auto circuit = CF::CreateCircuit();
    for (auto values : {query.ExecuteExpectations(circuit, {"IIIZ", "ZIII"}), query.ExecuteOnHostExpectations(circuit, 0, {"IIIZ", "ZIII"})})
    {
        Near(values[0], -1);
        Near(values[1], 1);
    }
    Check(record->batches == 2 && record->singles == 0, "Network bypassed batch dispatch");
    const auto state = query.ExecuteOnHostAmplitudes(circuit, 0);
    Check(record->dense == 1 && state.size() == 16, "Network bypassed bulk state dispatch");
    Near(state[8], 1.);
    circuit->AddOperation(CF::CreateGate(Gate::kXGateType, 3));
    circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
    auto network = std::make_shared<Net>(Types::qubits_vector{4}, std::vector<size_t>{4});
    network->SetOptimizeSimulator(false);
    network->Configure("precision", single ? "single" : "double");
    network->RemoveAllOptimizationSimulatorsAndAdd(backend, Method::kMatrixProductState);
    network->CreateSimulator(backend, Method::kMatrixProductState);
    const auto mapped = network->ExecuteOnHostExpectations(circuit, 0, {"IIIZ", "XIII", "IXII", "XIII"});
    Near(mapped[0], -1);
    Near(mapped[1], 1);
    Near(mapped[2], 0);
    Near(mapped[3], 1);
}

j::object Request(const char *operation, const char *method)
{
    j::object result{{"schema_version", 2},
                     {"operation", operation},
                     {"circuit", j::object{{"num_qubits", 2}, {"source", "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"}}},
                     {"simulator", j::object{{"backend", backend == Backend::kGpuSim ? "gpu" : "qcsim"},
                                             {"method", method},
                                             {"options", j::object{{"precision", single ? "single" : "double"}, {"gate_fusion", true}}}}}};
    if (backend == Backend::kQCSim)
        result.at("simulator").as_object().at("options").as_object().erase("precision");
    return result;
}

j::object Call(const j::object &request, bool valid = true, bool validate = false)
{
    const auto input = j::serialize(request);
    auto *raw = validate ? MaestroValidateRequestJson(input.c_str()) : MaestroRunRequestJson(input.c_str());
    Check(raw != nullptr, "Null JSON response");
    const auto result = j::parse(raw).as_object();
    FreeResult(raw);
    if (result.at("ok").as_bool() != valid)
        throw std::runtime_error(j::serialize(result));
    return result;
}

void JsonBatches()
{
    if (backend == Backend::kQCSim)
    {
        auto probabilities = Request("probabilities", "tensor_network");
        const auto full = Call(probabilities).at("probabilities").as_array();
        probabilities["basis_states"] = j::array{0, 1, 2, 3};
        const auto selected = Call(probabilities).at("probabilities").as_array();
        Check(full == selected, "Tensor-network full probabilities differ from selected output");
        Near(full[0].to_number<double>(), .5);
        Near(full[1].to_number<double>(), 0.);
        Near(full[2].to_number<double>(), 0.);
        Near(full[3].to_number<double>(), .5);
#ifndef NO_QISKIT_AER
        auto diagnostic = Request("diagnostics", "matrix_product_state");
        diagnostic["simulator"].as_object()["backend"] = "aer";
        diagnostic["diagnostics"] = j::array{"operator_expectation"};
        const auto error = Call(diagnostic, false, true).at("error").as_object();
        Check(error.at("message").as_string() == "Operator expectations require CPU or GPU MPS", "Incorrect unsupported-operator diagnostic");
#endif
    }
    for (const char *method : {"matrix_product_state", "matrix_product_operator"})
    {
        auto emptyMove = Request("diagnostics", method);
        emptyMove["move_qubits"] = j::array{};
        emptyMove["diagnostics"] = j::array{};
        Call(emptyMove);
        auto request = Request("estimate", method);
        request["observables"] = j::array{"XX", "YY", "ZZ", "XX"};
        const auto values = Call(request).at("expectation_values").as_array();
        Near(values[0].to_number<double>(), 1);
        Near(values[1].to_number<double>(), -1);
        Near(values[2].to_number<double>(), 1);
        Near(values[3].to_number<double>(), 1);
        request["operation"] = "incremental_evolve";
        request["step_circuit"] = j::object{{"num_qubits", 2}, {"source", "OPENQASM 2.0; qreg q[2]; x q[0];"}};
        request["steps"] = j::array{0, 1, 2};
        const auto rows = Call(request).at("expectation_values").as_array();
        for (size_t r = 0; r < 3; ++r)
        {
            Near(rows[r].at(0).to_number<double>(), 1);
            Near(rows[r].at(2).to_number<double>(), r == 1 ? -1. : 1.);
        }
    }
    auto request = Request("diagnostics", "matrix_product_operator");
    request["observables"] = j::array{"XX", "YY", "XX"};
    request["diagnostics"] = j::array{"expectations_complex", "unnormalized_expectations"};
    request["operators"] = j::array{j::object{{"qubits", j::array{0}}, {"matrix", j::array{j::array{2, 0}, j::array{0, 0}, j::array{0, 0}, j::array{2, 0}}}}};
    const auto result = Call(request);
    Near(result.at("expectations_complex").at(0).at(0).to_number<double>(), 1);
    Near(result.at("unnormalized_expectations").at(1).at(0).to_number<double>(), -4);
    request["max_output_elements"] = 2;
    Call(request, false, true);
    request.erase("max_output_elements");
    request["observables"] = j::array{"wrong"};
    Call(request, false, true);
    request["observables"] = j::array{};
    Check(Call(request).at("expectations_complex").as_array().empty(), "JSON empty batch");
    request.erase("observables");
    Call(request, false, true);
}

template <class F> void Reject(F &&f)
{
    bool rejected = false;
    try
    {
        f();
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    Check(rejected, "Invalid tensor query accepted");
}

void TensorQueries()
{
    Eigen::MatrixXcd x(2, 2), y(2, 2), a(2, 2);
    x << 0., 1., 1., 0.;
    y << 0., std::complex<double>(0., -1.), std::complex<double>(0., 1.), 0.;
    a << std::complex<double>(1., .3), std::complex<double>(.2, -.7), std::complex<double>(-.4, .1), std::complex<double>(.9, .2);
    for (bool fusion : {false, true})
    {
        auto sim = Make(Method::kMatrixProductState, 3);
        sim->Configure("gate_fusion", fusion ? "true" : "false");
        sim->SetInitialQubitsMap({2, 0, 1});
        sim->ApplyRy(0, .37);
        sim->ApplyH(2);
        sim->ApplyCX(2, 1);
        sim->ApplyRz(0, .42);
        const auto before = sim->GetStateVector();
        Check(before.size() == 8, "Wrong statevector size");
        const auto probs = sim->AllProbabilities();
        for (size_t i = 0; i < before.size(); ++i)
        {
            Near(before[i], sim->Amplitude(i));
            Near(probs[i], std::norm(before[i]));
        }
        auto applied = before;
        const Types::qubits_vector targets{0, 2, 0};
        const std::vector<unsigned long> cTargets(targets.begin(), targets.end());
        const std::vector<Eigen::MatrixXcd> matrices{x, a, y};
        for (size_t op = 0; op < targets.size(); ++op)
        {
            const size_t bit = size_t{1} << targets[op];
            for (size_t i = 0; i < applied.size(); ++i)
                if (!(i & bit))
                {
                    const auto lo = applied[i], hi = applied[i | bit];
                    applied[i] = matrices[op](0, 0) * lo + matrices[op](0, 1) * hi;
                    applied[i | bit] = matrices[op](1, 0) * lo + matrices[op](1, 1) * hi;
                }
        }
        std::complex<double> expected = 0.;
        for (size_t i = 0; i < before.size(); ++i)
            expected += std::conj(before[i]) * applied[i];
        Near(sim->ExpectationValueOperators(targets, matrices), expected);
        Near(sim->ExpectationValueOperators({}, {}), 1.);
        // Complex, nonsymmetric matrices expose row/column-major transposition
        // bugs.
        std::vector<double> raw;
        for (const auto &matrix : matrices)
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c)
                {
                    raw.push_back(matrix(r, c).real());
                    raw.push_back(matrix(r, c).imag());
                }
        double re = 7., im = 8.;
        Check(MaestroExpectationValueOperators(sim.get(), cTargets.data(), cTargets.size(), raw.data(), &re, &im), "C operator expectation failed");
        Near({re, im}, expected);
        re = 7.;
        im = 8.;
        const unsigned long invalid[]{3};
        Check(!MaestroExpectationValueOperators(sim.get(), invalid, 1, raw.data(), &re, &im), "C target validation failed");
        Near(re, 7.);
        Near(im, 8.);
        raw[0] = std::numeric_limits<double>::quiet_NaN();
        Check(!MaestroExpectationValueOperators(sim.get(), cTargets.data(), cTargets.size(), raw.data(), &re, &im), "C finite validation failed");
        Near(re, 7.);
        Near(im, 8.);
        Check(MaestroExpectationValueOperators(sim.get(), nullptr, 0, nullptr, &re, &im), "C identity failed");
        Near({re, im}, 1.);
        Reject([&] { sim->ExpectationValueOperators({0}, {}); });
        Reject([&] { sim->ExpectationValueOperators({0}, {Eigen::MatrixXcd::Identity(4, 4)}); });
        std::vector<double> state(18, 9.);
        Check(!MaestroGetStateVector(sim.get(), state.data(), 7), "C dense capacity ignored");
        for (double v : state)
            Near(v, 9.);
        Check(!MaestroGetStateVector(sim.get(), nullptr, 8), "C null output accepted");
        Check(MaestroGetStateVector(sim.get(), state.data(), 9), "C dense query failed");
        for (size_t i = 0; i < before.size(); ++i)
            Near({state[2 * i], state[2 * i + 1]}, before[i]);
        Near(state[16], 9.);
        Near(state[17], 9.);
        sim->SaveState();
        Check(MaestroMoveAtBeginningOfChain(sim.get(), nullptr, 0), "C empty MPS movement failed");
        Check(!MaestroMoveAtBeginningOfChain(sim.get(), nullptr, 1), "C null nonempty movement accepted");
        const unsigned long move[]{2, 0};
        Check(MaestroMoveAtBeginningOfChain(sim.get(), move, 2), "C MPS movement failed");
        for (size_t i = 0; i < before.size(); ++i)
            Near(sim->GetStateVector()[i], before[i]);
        Reject([&] { sim->MoveAtBeginningOfChain({0, 0}); });
        Reject([&] { sim->MoveAtBeginningOfChain({3}); });
        sim->ApplyX(1);
        sim->RestoreState();
        const auto restored = sim->GetStateVector();
        for (size_t i = 0; i < before.size(); ++i)
            Near(restored[i], before[i]);
        auto clone = sim->Clone();
        clone->ApplyX(0);
        Near(clone->ExpectationValueOperators({0, 0}, {x, y}), -sim->ExpectationValueOperators({0, 0}, {x, y}));
        Reject([&] { sim->SampleCountsMany({3}, 10); });
        auto counts = sim->SampleCountsMany({2, 0, 2}, 31);
        size_t total = 0;
        for (const auto &entry : counts)
        {
            Check(entry.first[0] == entry.first[2], "Repeated sampled bit disagrees");
            total += entry.second;
        }
        Check(total == 31, "Sampling lost shots");
    }
    // Peak statistics must not be replaced by the new instantaneous getter.
    for (auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator})
    {
        auto sim = Make(method, 4);
        sim->ApplyH(0);
        sim->ApplyCX(0, 1);
        const auto peak = sim->GetCurrentMaxBondDimension();
        Check(peak >= 2, "Bond summary missed entanglement");
        auto clone = sim->Clone();
        clone->ApplyH(2);
        clone->ApplyCX(2, 3);
        clone->ApplySwap(1, 2);
        Check(clone->GetCurrentMaxBondDimension() > peak, "Clone callback missed bond growth");
        Check(sim->GetCurrentMaxBondDimension() == peak, "Clone callback changed original");
        sim->Measure({0, 1});
        Check(sim->GetCurrentMaxBondDimension() >= peak, "Measurement erased peak");
    }
    auto mixed = Make(Method::kMatrixProductOperator, 2);
    Reject([&] { mixed->GetStateVector(); });
    Reject([&] { mixed->ExpectationValueOperators({}, {}); });
    double untouched[8]{9};
    Check(!MaestroGetStateVector(mixed.get(), untouched, 4), "Mixed state vector accepted");
    Near(untouched[0], 9.);
}

void JsonTensorQueries()
{
    auto request = Request("diagnostics", "matrix_product_state");
    request["move_qubits"] = j::array{1};
    request["diagnostics"] = j::array{"operator_expectation"};
    const j::array x{0, 1, 1, 0};
    request["expectation_operators"] = j::array{j::object{{"qubit", 0}, {"matrix", x}}, j::object{{"qubit", 1}, {"matrix", x}}};
    Call(request, true, true);
    Near(Call(request).at("operator_expectation").at(0).to_number<double>(), 1.);
    // Ordered repeated targets: YX = -iZ; on |00> this is -i.
    request.at("circuit").as_object()["source"] = "OPENQASM 2.0; qreg q[2];";
    request["expectation_operators"] =
        j::array{j::object{{"qubit", 0}, {"matrix", x}}, j::object{{"qubit", 0}, {"matrix", j::array{0, j::array{0, -1}, j::array{0, 1}, 0}}}};
    Near(Call(request).at("operator_expectation").at(1).to_number<double>(), -1.);
    request["expectation_operators"] = j::array{};
    Near(Call(request).at("operator_expectation").at(0).to_number<double>(), 1.);
    request["expectation_operators"] = j::array{j::object{{"qubit", 2}, {"matrix", x}}};
    Call(request, false, true);
    request["expectation_operators"] = j::array{};
    request["diagnostics"] = j::array{};
    Call(request, false, true);
    request.erase("expectation_operators");
    Call(request);
    request = Request("statevector", "matrix_product_state");
    const auto full = Call(request).at("amplitudes").as_array();
    request["basis_states"] = j::array{3, 0, 3};
    const auto selected = Call(request).at("amplitudes").as_array();
    for (size_t i = 0; i < selected.size(); ++i)
        Near(selected[i].at(0).to_number<double>(), full[i == 1 ? 0 : 3].at(0).to_number<double>());
    request.erase("basis_states");
    request["operation"] = "probabilities";
    const auto probs = Call(request).at("probabilities").as_array();
    Near(probs[0].to_number<double>(), .5);
    Near(probs[3].to_number<double>(), .5);
    request["max_output_elements"] = 2;
    Call(request, false, true);
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        if (argc > 1)
            backend = Backend::kGpuSim;
        if (argc > 2)
            single = true;
#ifdef __linux__
        if (backend == Backend::kGpuSim)
        {
            if (!Simulators::SimulatorsFactory::GetGpuDeviceCount())
                return 77;
            const auto lib = Simulators::SimulatorsFactory::GetGpuLibrary();
            Check(lib && lib->HasMPSExpectationValues() && lib->HasMPOExpectationValues() && lib->HasMPOExpectationValuesComplex() &&
                      lib->HasMPSGetStateVector() && lib->HasMPSAllProbabilities() && lib->HasMPSBondSummary() && lib->HasMPOBondSummary() &&
                      lib->HasMPSExpectationValueOperators() && lib->HasMPSMoveAtBeginningOfChain(),
                  "GPU tensor extensions unavailable");
        }
#endif
        RealBatches();
        ComplexAndC();
        NetworkDispatch();
        JsonBatches();
        TensorQueries();
        JsonTensorQueries();
        std::cout << checks << " expectation batch checks passed\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
