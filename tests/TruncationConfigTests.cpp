#include "Circuit/Factory.h"
#include "Execution/SimulatorConfig.h"
#include "Simulators/Core/Factory.h"

#include <boost/json.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
using Backend = Simulators::SimulatorType;
using Method = Simulators::SimulationType;
using Net = Network::SimpleDisconnectedNetwork<>;
using Circuit = Circuits::Circuit<>;
using CF = Circuits::CircuitFactory<>;
constexpr const char *kStateThreshold = "matrix_product_state_truncation_threshold";
constexpr const char *kOperatorThreshold = "matrix_product_operator_truncation_threshold";

void Check(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void CheckThreshold(Simulators::ISimulator &sim, double expected)
{
    Simulators::Configuration copied;
    copied.ApplyConfigurationFromMap(sim.GetConfigMap());
    for (const char *key : {kStateThreshold, kOperatorThreshold})
    {
        Check(!sim.GetConfiguration(key).empty(), "threshold getter omitted the effective value");
        Check(std::stod(sim.GetConfiguration(key)) == expected, "threshold getter returned a stale value");
        Check(copied.IsSet(key), "configuration map omitted the effective threshold");
        Check(copied.GetConfigurationAsDouble(key) == expected, "copied configuration lost the effective threshold");
        char *value = ::GetConfiguration(&sim, key);
        Check(value != nullptr, "C configuration getter omitted the effective threshold");
        const double threshold = std::stod(value);
        FreeResult(value);
        Check(threshold == expected, "C configuration getter returned a stale threshold");
    }
}

void CheckTruncation(Simulators::ISimulator &sim, double theta, bool truncated)
{
    sim.Reset();
    sim.ApplyRy(0, theta);
    sim.ApplyCX(0, 1);
    // Reading a probability flushes gates queued by the public fusion wrapper.
    Check(sim.Probability(0) > 0.99, "test circuit lost its dominant component");
    Check((sim.GetCurrentMaxBondDimension() == 1) == truncated, "effective threshold did not control bond growth");
}

void CheckLifecycle(Method method, bool fused)
{
    auto sim = fused ? Simulators::SimulatorsFactory::CreateSimulatorUnique(Backend::kQCSim, method)
                     : Simulators::SimulatorsFactory::CreateImmediateSimulatorUnique(Backend::kQCSim, method);
    CheckThreshold(*sim, 1e-15);
    CheckThreshold(*sim->Clone(), 1e-15);
    Check(!sim->GetExplicitConfigMap().count(kStateThreshold), "implicit default became an explicit setting");
    sim->AllocateQubits(2);
    sim->Initialize();
    CheckThreshold(*sim, 1e-15);
    // A 1e-8 Schmidt coefficient is above Eigen's numerical rank cutoff,
    // but its squared weight is below Maestro's explicit default.
    CheckTruncation(*sim, 2e-8, true);
    auto clone = sim->Clone();
    CheckThreshold(*clone, 1e-15);
    Check(!clone->GetExplicitConfigMap().count(kStateThreshold), "cloning promoted an implicit default");
    CheckTruncation(*clone, 2e-8, true);

    for (const char *mode : {"discarded_weight", "relative_max"})
    {
        sim->Configure("matrix_product_state_truncation_mode", mode);
        for (const char *key : {kStateThreshold, kOperatorThreshold})
        {
            // Alternate aliases on both methods, including zero, then exercise the same
            // settings after both cloning and native-state recreation.
            for (const char *value : {"0.1", "0", "1e-18"})
            {
                sim->Configure(key, value);
                const double threshold = std::stod(value);
                CheckThreshold(*sim, threshold);
                CheckTruncation(*sim, 0.1, threshold == 0.1);
                clone = sim->Clone();
                CheckThreshold(*clone, threshold);
                CheckTruncation(*clone, 0.1, threshold == 0.1);
                sim->Clear();
                sim->AllocateQubits(2);
                sim->Initialize();
                CheckThreshold(*sim, threshold);
                CheckTruncation(*sim, 0.1, threshold == 0.1);
            }
        }
    }
    sim->Configure("matrix_product_state_truncation_mode", "discarded_weight");
    // A smaller explicit value must preserve a component the default drops.
    CheckTruncation(*sim, 2e-8, false);
    sim->Configure(kOperatorThreshold, "1e-15");
    Check(sim->GetExplicitConfigMap().count(kStateThreshold), "explicit value equal to the default was lost");
    Check(sim->Clone()->GetExplicitConfigMap().count(kStateThreshold), "clone lost an explicit value equal to the default");
}

void CheckNetwork(Method method)
{
    auto &owner = *static_cast<Maestro *>(GetMaestroObject());
    const auto handle = owner.CreateSimpleSimulator(2, 2);
    MaestroExecution::SimulatorConfig config;
    config.simulation_type = method;
    config.fixed_backend = true;
    auto network = MaestroExecution::ConfigureNetwork(handle, config, 2);
    CheckThreshold(*network->GetSimulator(), 1e-15);
    CheckTruncation(*network->GetSimulator(), 2e-8, true);

    const char *nativeKey = method == Method::kMatrixProductOperator ? kOperatorThreshold : kStateThreshold;
    config.native_options[nativeKey] = "0.1";
    network = MaestroExecution::ConfigureNetwork(handle, config, 2);
    CheckThreshold(*network->GetSimulator(), 0.1);
    CheckTruncation(*network->GetSimulator(), 0.1, true);

    for (const double threshold : {1e-18, 0.})
    {
        config.singular_value_threshold = threshold;
        network = MaestroExecution::ConfigureNetwork(handle, config, 2);
        CheckThreshold(*network->GetSimulator(), threshold);
        CheckTruncation(*network->GetSimulator(), 2e-8, false);
    }

    // Both threshold spellings are one setting. Removing either after a
    // capture must restore the default instead of replaying a stale alias.
    for (const char *key : {kStateThreshold, kOperatorThreshold})
    {
        network->Configure(nativeKey, "0.1");
        network->Configure(kStateThreshold, "0");
        auto cloned = network->Clone();
        CheckThreshold(*cloned->GetSimulator(), 0.);
        CheckTruncation(*cloned->GetSimulator(), 2e-8, false);
        network->CaptureSimulatorSettings();
        network->RemoveConfiguration(key);
        CheckThreshold(*network->GetSimulator(), 1e-15);
        CheckTruncation(*network->GetSimulator(), 2e-8, true);
    }

    config.singular_value_threshold.reset();
    network = MaestroExecution::ConfigureNetwork(handle, config, 2);
    CheckThreshold(*network->GetSimulator(), 0.1); // Native option still applies.
    config.native_options.clear();
    config.singular_value_threshold = 1e-18;
    network = MaestroExecution::ConfigureNetwork(handle, config, 2);
    CheckThreshold(*network->GetSimulator(), 1e-18);
    config.singular_value_threshold.reset();
    network = MaestroExecution::ConfigureNetwork(handle, config, 2);
    CheckThreshold(*network->GetSimulator(), 1e-15);
    Check(!network->GetSimulator()->GetExplicitConfigMap().count(kStateThreshold), "reused handle retained its old explicit threshold");
    CheckTruncation(*network->GetSimulator(), 2e-8, true);
    owner.DestroySimpleSimulator(handle);
}

boost::json::value ReadJsonResult(char *raw)
{
    Check(raw != nullptr, "JSON execution returned no result");
    const std::string result(raw);
    FreeResult(raw);
    return boost::json::parse(result);
}

void CheckJsonThresholds(Method method)
{
    namespace j = boost::json;
    const char *methodName = method == Method::kMatrixProductState ? "matrix_product_state" : "matrix_product_operator";
    const char *circuit = "OPENQASM 2.0; qreg q[2]; creg c[2]; ry(0.00000002) q[0]; cx q[0],q[1];";
    for (const char *selection : {"fixed", "automatic"})
        for (const char *key : {"singular_value_threshold", kStateThreshold})
            for (const auto &threshold : j::array{nullptr, 0.1, 0, 1e-18, 1e-15})
            {
                j::object options{{"auto_reduce", false}, {"enable_causal_cone_reduction", false}, {"enable_observable_clustering", false}};
                if (!threshold.is_null())
                    options[key] = threshold;
                const j::object request{{"schema_version", 2},
                                        {"operation", "estimate"},
                                        {"circuit", j::object{{"format", "openqasm"}, {"num_qubits", 2}, {"num_clbits", 2}, {"source", circuit}}},
                                        {"simulator", j::object{{"backend", "qcsim"}, {"method", methodName}, {"selection", selection}, {"options", options}}},
                                        {"observables", j::array{"XX"}}};
                const auto result = ReadJsonResult(MaestroRunRequestJson(j::serialize(request).c_str()));
                Check(result.at("ok").as_bool(), "Native JSON threshold request failed");
                Check(result.at("execution_metadata").at("method") == methodName, "Native JSON threshold test selected another method");
                if (!threshold.is_null())
                    Check(std::stod(result.at("execution_metadata").at("configured_options").at(kStateThreshold).as_string().c_str()) ==
                              threshold.to_number<double>(),
                          "Native JSON rounded away the threshold");
                const double expected = !threshold.is_null() && threshold.to_number<double>() < 1e-15 ? std::sin(2e-8) : 0.;
                Check(std::abs(result.at("expectation_values").as_array()[0].to_number<double>() - expected) < 1e-12,
                      "Native JSON threshold did not control truncation");
            }

    // Legacy JSON accepts numeric strings as well as numbers and reuses a handle.
    // Exercise both entry points because each forwards the cutoff separately.
    auto &owner = *static_cast<Maestro *>(GetMaestroObject());
    for (const bool estimate : {false, true})
    {
        const auto handle = owner.CreateSimpleSimulator(2, 2);
        const auto network = owner.GetSimpleSimulator(handle);
        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, method);
        for (const auto &threshold : j::array{nullptr, 0.1, 0, 1e-18, 1e-15, "0.1", "0", "1e-18"})
        {
            j::object options{{"shots", 8}, {"seed", 73}};
            if (!threshold.is_null())
                options[kStateThreshold] = threshold;
            const auto input = j::serialize(options);
            const std::string measured = std::string(circuit) + " measure q -> c;";
            const auto result =
                ReadJsonResult(estimate ? SimpleEstimate(handle, circuit, "XX", input.c_str()) : SimpleExecute(handle, measured.c_str(), input.c_str()));
            const double cutoff = threshold.is_null()     ? 1e-15
                                  : threshold.is_string() ? std::stod(threshold.as_string().c_str())
                                                          : threshold.to_number<double>();
            Check(network->GetLastSimulationType() == method, "Legacy JSON threshold test selected another method");
            CheckThreshold(*network->GetSimulator(), cutoff);
            if (estimate)
                Check(std::abs(result.at("expectation_values").as_array()[0].to_number<double>() - (cutoff < 1e-15 ? std::sin(2e-8) : 0.)) < 1e-12,
                      "Legacy JSON threshold did not control truncation");
            else
                CheckTruncation(*network->GetSimulator(), 2e-8, cutoff >= 1e-15);
        }
        owner.DestroySimpleSimulator(handle);
    }
}

// Exercise the QCSim placeholder used by automatic backend selection, both
// directly and through network cloning. An explicit value equal to the default
// must propagate, whereas an implicit default must stay with its backend.
void CheckBackendSelection(Backend backend, Method method)
{
    auto &owner = *static_cast<Maestro *>(GetMaestroObject());
    const auto handle = owner.CreateSimpleSimulator(2, 2);
    MaestroExecution::SimulatorConfig config;
    config.simulator_type = backend;
    config.simulation_type = method;
    auto network = MaestroExecution::ConfigureNetwork(handle, config, 2);
    CheckThreshold(*network->GetSimulator(), 1e-15);
    // The GPU mock supplies device discovery, not tensor evolution. Selecting
    // an empty register exercises configuration transfer without a CUDA state.
    const size_t width = backend == Backend::kGpuSim ? 0 : 2;
    auto baseline = Simulators::SimulatorsFactory::CreateSimulator(backend, method);
    baseline->AllocateQubits(width);
    baseline->Initialize();
    const auto backendDefault = baseline->GetConfiguration(kStateThreshold);
    for (const char *threshold : {"", "1e-15", "0", "0.1"})
    {
        if (*threshold)
            network->GetSimulator()->Configure(kOperatorThreshold, threshold);
        for (const auto &candidate : {network->Clone(), network})
        {
            candidate->CaptureSimulatorSettings();
            auto circuit = CF::CreateCircuit();
            size_t shots = 256;
            auto type = Backend::kQCSim;
            auto selectedMethod = candidate->GetSimulator()->GetSimulationType();
            std::vector<bool> executed;
            auto selected = candidate->ChooseBestSimulator(circuit, shots, width, width, width, type, selectedMethod, executed, false, true);
            Check(selected && type == backend && selectedMethod == method, "automatic selection did not choose the requested backend");
            const auto value = selected->GetConfiguration(kStateThreshold);
            Check(*threshold ? (!value.empty() && std::stod(value) == std::stod(threshold)) : value == backendDefault,
                  "placeholder capture changed another backend's threshold");
        }
    }
    owner.DestroySimpleSimulator(handle);
}

class ProbeNetwork : public Net
{
  public:
    using Net::Net;
    using Net::UsesFullTerminalBatch;

    void InstallCostModel(std::unique_ptr<Estimators::SimulatorsEstimatorInterface<>> model)
    {
        simulatorsEstimator = std::move(model);
    }
};

class SingleHostRemapper : public Distribution::IRemapper<>
{
  public:
    std::shared_ptr<Circuit> Remap(const std::shared_ptr<Network::INetwork<>> &, const std::shared_ptr<Circuit> &circuit) override
    {
        return circuit;
    }

    unsigned int GetNumberOfOperationsForDistribution() const override
    {
        return 0;
    }

    unsigned int GetNumberOfDistributions() const override
    {
        return 0;
    }

    Distribution::RemapperType GetType() const override
    {
        return Distribution::RemapperType::kLayersRemapper;
    }
};

class NoChoiceCostModel : public Estimators::SimulatorsEstimatorInterface<>
{
  public:
    explicit NoChoiceCostModel(size_t &calls) : calls(calls)
    {
    }

    bool IsInitialized() const override
    {
        return true;
    }

    std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(const std::vector<std::pair<Backend, Method>> &, std::shared_ptr<Circuit> &, size_t &, size_t,
                                                                size_t, size_t, Backend &, Method &, std::vector<bool> &, const long long int, const double,
                                                                const std::string &, const std::string &, size_t, const std::vector<std::string> *,
                                                                bool) const override
    {
        ++calls;
        return nullptr;
    }

  private:
    size_t &calls;
};

class ThresholdCostModel : public Estimators::SimulatorsEstimatorInterface<>
{
  public:
    explicit ThresholdCostModel(double &observed) : observed(observed)
    {
    }

    bool IsInitialized() const override
    {
        return true;
    }

    std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(const std::vector<std::pair<Backend, Method>> &candidates, std::shared_ptr<Circuit> &, size_t &,
                                                                size_t, size_t, size_t, Backend &type, Method &method, std::vector<bool> &, const long long int,
                                                                const double threshold, const std::string &, const std::string &, size_t,
                                                                const std::vector<std::string> *, bool) const override
    {
        observed = threshold;
        type = candidates.front().first;
        method = candidates.front().second;
        return Simulators::SimulatorsFactory::CreateSimulator(type, method);
    }

  private:
    double &observed;
};

void CheckAliasBeforeCapture()
{
    Simulators::Configuration settings;
    settings.SetConfiguration(kOperatorThreshold, "0.1");
    Check(settings.GetConfigurationAsDouble(kStateThreshold) == 0.1, "fresh configuration lost the canonical threshold");
    settings.RemoveConfiguration(kStateThreshold);
    Check(!settings.IsSet(kOperatorThreshold), "canonical removal retained the alias");

    auto network = std::make_shared<ProbeNetwork>(Types::qubits_vector{2}, std::vector<size_t>{2});
    network->CreateSimulator();
    network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kStatevector);
    network->AddOptimizationSimulator(Backend::kQCSim, Method::kMatrixProductState);
    network->SetOptimizeSimulator(true);
    network->SetCostModelEnabled(true);
    double observed = -1.;
    network->InstallCostModel(std::make_unique<ThresholdCostModel>(observed));
    network->Configure(kOperatorThreshold, "0.1");
    auto circuit = CF::CreateCircuit();
    size_t shots = 256;
    auto type = Backend::kQCSim;
    auto method = Method::kMatrixProductState;
    std::vector<bool> executed;
    auto selected = network->ChooseBestSimulator(circuit, shots, 2, 2, 2, type, method, executed, false, true);
    Check(selected && observed == 0.1, "cost model missed the alias before simulator capture");
}

void CheckMpoHistory(bool fused)
{
    auto sim = fused ? Simulators::SimulatorsFactory::CreateSimulatorUnique(Backend::kQCSim, Method::kMatrixProductOperator)
                     : Simulators::SimulatorsFactory::CreateImmediateSimulatorUnique(Backend::kQCSim, Method::kMatrixProductOperator);
    sim->AllocateQubits(2);
    sim->Initialize();
    Check(sim->CanSampleSingleQubitMarginal(), "initializing the default marked the state truncated");
    sim->Configure(kOperatorThreshold, "0");
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    Check(sim->CanSampleSingleQubitMarginal(), "late zero override disabled marginal sampling");
    sim->SetSeed(73);
    const auto counts = sim->SampleCounts({0}, 4096);
    Check(counts.size() == 2 && counts.at(0) + counts.at(1) == 4096, "marginal sampling lost Bell-state shots");
    sim->Configure("counts_policy", "legacy");
    sim->SetSeed(73);
    Check(counts != sim->SampleCounts({0}, 4096), "late zero override still used per-shot native sampling");
    sim->Configure("counts_policy", "auto");
    sim->SaveState();
    sim->Configure(kStateThreshold, "0.1");
    Check(sim->CanSampleSingleQubitMarginal(), "configuring a limit changed state history");
    sim->ApplyCX(0, 1);
    Check(!sim->CanSampleSingleQubitMarginal(), "a potentially truncating gate lost its history");
    sim->Configure(kStateThreshold, "0");
    Check(!sim->CanSampleSingleQubitMarginal(), "zero override erased existing truncation history");
    sim->RestoreState();
    Check(sim->CanSampleSingleQubitMarginal(), "restoring an exact snapshot retained unrelated history");
    sim->SetSeed(73);
    Check(counts == sim->SampleCounts({0}, 4096), "restoring an exact snapshot changed marginal sampling");

    sim->Configure(kStateThreshold, "0.1");
    sim->Trim();
    Check(!sim->CanSampleSingleQubitMarginal(), "explicit compression lost its truncation history");
    sim->SaveState();
    sim->Configure(kStateThreshold, "0");
    auto clone = sim->Clone();
    for (auto *state : {sim.get(), clone.get()})
    {
        Check(!state->CanSampleSingleQubitMarginal(), "clone lost current truncation history");
        state->Reset();
        Check(state->CanSampleSingleQubitMarginal(), "reset retained truncation history");
        state->RestoreState();
        Check(!state->CanSampleSingleQubitMarginal(), "reset erased a saved snapshot's truncation history");
        state->Clear();
        state->AllocateQubits(2);
        state->Initialize();
        state->RestoreState();
        Check(state->CanSampleSingleQubitMarginal(), "reinitialization retained a cleared snapshot's history");
    }

    sim->Configure("matrix_product_state_max_bond_dimension", "1");
    Check(sim->CanSampleSingleQubitMarginal(), "configuring a bond cap changed state history");
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    sim->Configure("matrix_product_operator_max_bond_dimension", "0");
    Check(!sim->CanSampleSingleQubitMarginal(), "removing a bond cap erased state history");
    sim->Reset();
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    Check(sim->CanSampleSingleQubitMarginal(), "removed bond cap still disabled marginal sampling after reset");
}

void CheckMpoBatchDispatch()
{
    ProbeNetwork network(Types::qubits_vector{2}, std::vector<size_t>{2});
    network.CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
    const auto fullBatch = [&](bool startsFromZero, const std::shared_ptr<Simulators::ISimulator> &prepared = nullptr) {
        return network.UsesFullTerminalBatch(Backend::kQCSim, Method::kMatrixProductOperator, 4096, 2, 1, startsFromZero, prepared);
    };
    Check(!fullBatch(true), "implicit positive cutoff was ignored when scheduling a pending circuit");
    network.Configure(kOperatorThreshold, "0");
    Check(fullBatch(false), "late zero override split an exact MPO marginal batch");
    auto sim = network.GetSimulator();
    network.Configure(kStateThreshold, "0.1");
    sim->ApplyH(0);
    sim->ApplyCX(0, 1);
    network.Configure(kOperatorThreshold, "0");
    Check(!fullBatch(false), "preserved truncated MPO was scheduled for an unavailable marginal sampler");
    Check(fullBatch(true), "stale history disabled batching before a reset");
    std::shared_ptr<Simulators::ISimulator> prepared = sim->Clone();
    Check(!fullBatch(true, prepared), "prepared truncated MPO lost its history when execution starts from zero");
    sim->Reset();
    Check(fullBatch(false), "reset MPO still disabled a full marginal batch");
    sim->Clear();
    Check(fullBatch(true), "cleared MPO disabled batching before reallocation");
    Check(!fullBatch(false), "cleared MPO was considered an existing sampleable state");
    network.Configure("matrix_product_operator_max_bond_dimension", "2");
    Check(!fullBatch(true), "reset execution ignored its configured bond cap");
    network.Configure("matrix_product_operator_max_bond_dimension", "0");
    network.Configure(kStateThreshold, "1e-15");
    Check(!fullBatch(true), "cleared MPO lost its configured positive cutoff");

    // Preparing a fresh execution resets the old history. That must not make
    // the logical shot partition depend on how many workers prepare/sample it.
    const auto run = [](size_t workers) {
        auto net = std::make_shared<Net>(Types::qubits_vector{2}, std::vector<size_t>{1});
        net->SetOptimizeSimulator(false);
        net->GetController()->SetOptimizeCircuit(false);
        net->SetMaxSimulators(workers);
        net->CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
        net->Configure(kStateThreshold, "0.1");
        net->GetSimulator()->ApplyH(0);
        net->GetSimulator()->ApplyCX(0, 1);
        net->Configure(kOperatorThreshold, "0");
        net->Configure("seed", "73");
        auto circuit = CF::CreateCircuit();
        circuit->AddOperation(CF::CreateGate(Circuits::QuantumGateType::kHadamardGateType, 0));
        circuit->AddOperation(CF::CreateGate(Circuits::QuantumGateType::kCXGateType, 0, 1));
        circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
        return net->RepeatedExecuteOnHost(circuit, 0, 4096);
    };
    Check(run(1) == run(4), "resetting MPO history made seeded counts depend on worker count");
}

void CheckMpoExecutionReplay()
{
    // A non-Clifford circuit keeps automatic selection from switching to a
    // stabilizer before the MPO batching decision we want to exercise.
    auto circuit = CF::CreateCircuit();
    circuit->AddOperation(CF::CreateGate(Circuits::QuantumGateType::kRyGateType, 0, 0, 0, 0.7));
    circuit->AddOperation(CF::CreateGate(Circuits::QuantumGateType::kCXGateType, 0, 1));
    circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
    std::string failures;
    for (bool onHost : {false, true})
        for (bool fusion : {false, true})
        {
            // A direct 4096-shot marginal is the reference, so making every
            // execution use smaller blocks cannot accidentally satisfy replay.
            auto expected = Simulators::SimulatorsFactory::CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
            expected->Configure("gate_fusion", fusion ? "true" : "false");
            expected->Configure(kStateThreshold, "0");
            expected->SetSeed(73);
            expected->AllocateQubits(2);
            expected->Initialize();
            Circuits::OperationState state;
            state.AllocateBits(1);
            circuit->ExecuteNonMeasurements(expected, state);
            const auto counts = expected->SampleCountsMany({0}, 4096);
            const Net::ExecuteResults reference(counts.begin(), counts.end());
            for (const std::string setup : {"fresh", "stale", "cleared", "default_candidates", "declining_estimator", "prepared"})
                for (const size_t workers : {1u, 4u})
                {
                    auto network = std::make_shared<ProbeNetwork>(Types::qubits_vector{2}, std::vector<size_t>{1});
                    network->GetController()->SetOptimizeCircuit(false);
                    network->GetController()->SetRemapper(std::make_shared<SingleHostRemapper>());
                    network->SetOptimizeSimulator(setup == "default_candidates" || setup == "declining_estimator" || setup == "prepared");
                    network->SetMaxSimulators(workers);
                    network->Configure("gate_fusion", fusion ? "true" : "false");
                    network->CreateSimulator(Backend::kQCSim, Method::kMatrixProductOperator);
                    if (setup == "stale")
                    {
                        network->Configure(kStateThreshold, "0.1");
                        network->GetSimulator()->ApplyH(0);
                        network->GetSimulator()->ApplyCX(0, 1);
                    }
                    network->Configure(kOperatorThreshold, "0");
                    if (setup == "cleared")
                        network->GetSimulator()->Clear();
                    size_t estimatorCalls = 0;
                    if (setup == "declining_estimator")
                        network->InstallCostModel(std::make_unique<NoChoiceCostModel>(estimatorCalls));
                    if (setup == "prepared")
                        network->RemoveAllOptimizationSimulatorsAndAdd(Backend::kQCSim, Method::kMatrixProductOperator);
                    network->Configure("seed", "73");
                    const auto run = [&] { return onHost ? network->RepeatedExecuteOnHost(circuit, 0, 4096) : network->RepeatedExecute(circuit, 4096); };
                    const auto first = run();
                    const auto second = run();
                    Check(first.size() == 2 && first.at({false}) + first.at({true}) == 4096, "MPO replay test lost measured shots");
                    Check(network->GetLastSimulationType() == Method::kMatrixProductOperator, "MPO replay test switched methods");
                    if (setup == "declining_estimator")
                        Check(estimatorCalls == 2, "MPO replay test did not exercise an estimator returning nullptr");
                    const std::string context =
                        setup + (onHost ? " host" : " network") + (fusion ? " fused" : " immediate") + " workers=" + std::to_string(workers);
                    if (first != second)
                        failures += context + ": identical executions did not replay\n";
                    if (first != reference)
                        failures += context + ": counts differed from a full 4096-shot marginal batch\n";
                }
        }
    if (!failures.empty())
        throw std::runtime_error(failures);
}

void CheckDoubleConfiguration()
{
    Simulators::Configuration config;
    Network::Configuration<> networkConfig;
    for (const double threshold : {1e-15, 1e-12, 1e-18, 0.})
    {
        config.SetConfiguration(kStateThreshold, threshold);
        networkConfig.SetConfiguration(kStateThreshold, threshold);
        Check(config.GetConfigurationAsDouble(kStateThreshold) == threshold, "double setter rounded away the threshold");
        Check(networkConfig.GetConfigurationAsDouble(kStateThreshold) == threshold, "network double setter rounded away the threshold");
    }
}
} // namespace

int main(int argc, char **argv)
{
    try
    {
        for (const auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator})
        {
            for (bool fused : {false, true})
                CheckLifecycle(method, fused);
            CheckNetwork(method);
            CheckJsonThresholds(method);
        }
        CheckDoubleConfiguration();
        CheckAliasBeforeCapture();
        CheckBackendSelection(Backend::kQCSim, Method::kStatevector);
        for (bool fused : {false, true})
            CheckMpoHistory(fused);
        CheckMpoBatchDispatch();
        CheckMpoExecutionReplay();
#ifndef NO_QISKIT_AER
        CheckBackendSelection(Backend::kQiskitAer, Method::kMatrixProductState);
#endif
#ifdef __linux__
        if (argc > 1 && std::string(argv[1]) == "--mock-gpu")
            for (const auto method : {Method::kMatrixProductState, Method::kMatrixProductOperator, Method::kDensityMatrix, Method::kTensorNetwork})
                CheckBackendSelection(Backend::kGpuSim, method);
#endif
        auto statevector = Simulators::SimulatorsFactory::CreateSimulatorUnique(Backend::kQCSim, Method::kStatevector);
        Check(statevector->GetConfiguration(kStateThreshold).empty(), "tensor-chain default leaked into a statevector");
        std::cout << "Truncation configuration tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
