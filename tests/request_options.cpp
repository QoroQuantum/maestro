// Shared request/configuration checks require no MPI or GPU hardware.
#include "../Execution/Options.h"
#include "../Circuit/Factory.h"
#include <atomic>

void Check(bool, const char*);

namespace {
// Count actual gate execution through cloning, mapping and worker jobs. This
// catches repeated evolution without a wall-clock performance assertion.
class CountedX : public Circuits::XGate<> {
 public:
  CountedX(size_t qubit, std::shared_ptr<std::atomic<size_t>> calls)
      : XGate(qubit), calls(std::move(calls)) {}

  void Execute(const std::shared_ptr<Simulators::ISimulator>& sim,
               Circuits::OperationState& state) const override {
    ++*calls;
    XGate::Execute(sim, state);
  }

  std::shared_ptr<Circuits::IOperation<>> Clone() const override {
    return std::make_shared<CountedX>(*this);
  }

 private:
  std::shared_ptr<std::atomic<size_t>> calls;
};

// RepeatedExecute uses a distribution remapper; a single host needs no
// remapping, while still exercising the production network job dispatch.
class SingleHostRemapper : public Distribution::IRemapper<> {
 public:
  std::shared_ptr<Circuits::Circuit<>> Remap(
      const std::shared_ptr<Network::INetwork<>>&,
      const std::shared_ptr<Circuits::Circuit<>>& circuit) override {
    return circuit;
  }
  unsigned int GetNumberOfOperationsForDistribution() const override {
    return 0;
  }
  unsigned int GetNumberOfDistributions() const override { return 0; }
  Distribution::RemapperType GetType() const override {
    return Distribution::RemapperType::kLayersRemapper;
  }
};
}  // namespace

void TestFixedBackendShotReuse() {
  using namespace MaestroExecution;
  using CF = Circuits::CircuitFactory<>;
  using Gate = Circuits::QuantumGateType;
  GetMaestroObjectWithMute();
  constexpr size_t shots = 128;

  for (const char* method : {"statevector", "matrix_product_state",
                             "density_matrix", "matrix_product_operator"})
    for (bool onHost : {true, false})
      for (size_t workers : {size_t{1}, size_t{2}})
        for (int scenario : {0, 1, 2, 3}) {
          struct NetworkHandle {
            unsigned long handle = CreateSimpleSimulator(2);
            ~NetworkHandle() { DestroySimpleSimulator(handle); }
          } owner;
          Check(owner.handle != 0, "Cannot create network for shot reuse");
          // Exercise the native parser's default fixed selection and the same
          // ConfigureNetwork path used by native requests.
          auto config = ParseConfig(
              json::object{{"backend", "qcsim"}, {"method", method}});
          config.seed = 123;
          config.optimize_circuit = false;
          auto network = ConfigureNetwork(owner.handle, config);
          Check(network && !network->GetOptimizeSimulator(),
                "Fixed selection unexpectedly enabled backend optimization");
          network->SetMaxSimulators(workers);
          if (!onHost)
            network->GetController()->SetRemapper(
                std::make_shared<SingleHostRemapper>());

          auto prefix = std::make_shared<std::atomic<size_t>>(0);
          auto suffix = std::make_shared<std::atomic<size_t>>(0);
          auto circuit = CF::CreateCircuit();
          circuit->AddOperation(std::make_shared<CountedX>(0, prefix));
          if (scenario == 1 || scenario == 3) {
            if (scenario == 1) {
              circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
              circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
            } else {
              circuit->AddOperation(CF::CreateRandom({0}, 123));
            }
            circuit->AddOperation(CF::CreateSimpleConditionalGate(
                CF::CreateGate(Gate::kXGateType, 1), 0));
            circuit->AddOperation(std::make_shared<CountedX>(1, suffix));
            circuit->AddOperation(CF::CreateMeasurement({{1, 1}}));
          } else {
            if (scenario == 2) {
              circuit->AddOperation(CF::CreateGate(Gate::kHadamardGateType, 0));
              circuit->AddOperation(CF::CreateGate(Gate::kCXGateType, 0, 1));
              circuit->AddOperation(CF::CreateReset({0}));
              circuit->AddOperation(std::make_shared<CountedX>(0, suffix));
            }
            circuit->AddOperation(CF::CreateMeasurement({{0, 0}, {1, 1}}));
          }

          const size_t jobs =
              scenario == 0 && config.simulation_type == Method::kStatevector
                  ? 1
                  : network->GetMaxSimulators();
          for (int repeat = 0; repeat < 2; ++repeat) {
            *prefix = 0;
            *suffix = 0;
            const auto counts =
                onHost ? network->RepeatedExecuteOnHost(circuit, 0, shots)
                       : network->RepeatedExecute(circuit, shots);
            Check(prefix->load() == jobs,
                  "Fixed backend re-executed the common prefix per shot");
            Check(
                suffix->load() == (scenario ? shots : 0),
                "Measurement-dependent operations were not executed per shot");
            Check(
                network->GetLastSimulatorType() == Backend::kQCSim &&
                    network->GetLastSimulationType() == config.simulation_type,
                "Shot reuse changed the fixed backend or method");
            size_t total = 0;
            for (const auto& [bits, count] : counts) {
              Check(bits.size() == 2, "Shot reuse changed classical width");
              Check((scenario == 1 || scenario == 3)
                        ? bits[0] != bits[1]
                        : bits[0] && (scenario == 2 || !bits[1]),
                    "Shot reuse changed measurement/reset/conditional results");
              total += count;
            }
            Check(total == shots, "Shot reuse lost counts");
            Check(counts.size() == (scenario ? 2 : 1),
                  "Independent shots reused one measurement/reset outcome");
          }
        }
}

void TestRequestSeedParsing() {
  for (const char* backend : {
           "qcsim",
#ifdef __linux__
           "distributed_mpi_gpu",
#endif
       }) {
    MaestroExecution::json::object simulator{{"backend", backend}};
    Check(!MaestroExecution::ParseConfig(simulator).seed.has_value(),
          "Parsing an omitted seed must preserve the unset state");
    for (uint64_t seed : {uint64_t{0}, UINT64_MAX}) {
      simulator["options"] = MaestroExecution::json::object{{"seed", seed}};
      const auto config = MaestroExecution::ParseConfig(simulator);
      Check(config.seed.has_value() && *config.seed == seed,
            "Parsing changed an explicit seed");
    }
  }
}

void TestNetworkBondDefaults() {
  using namespace MaestroExecution;
  GetMaestroObjectWithMute();
  const char* const bondKey = "matrix_product_state_max_bond_dimension";
  const auto checkNetwork = [bondKey](const SimulatorConfig& config,
                                     const char* expected) {
    struct NetworkHandle {
      unsigned long handle = CreateSimpleSimulator(2);
      ~NetworkHandle() { DestroySimpleSimulator(handle); }
    } owner;
    Check(owner.handle != 0, "Cannot create network for bond configuration");
    const auto network = ConfigureNetwork(owner.handle, config);
    Check(bool(network), "Cannot configure network bond dimension");
    // Automatic selection first creates a CPU placeholder from the same
    // network configuration the initial-layout planner reads. No GPU is needed.
    Check(network->GetSimulator()->GetConfiguration(bondKey) == expected,
          "Network setup used the wrong bond dimension");
    network->CreateSimulator();
    Check(network->GetSimulator()->GetConfiguration(bondKey) == expected,
          "Network recreation lost the bond dimension");
  };

  for (const auto method : {Method::kMatrixProductState,
                            Method::kMatrixProductOperator}) {
    SimulatorConfig config;
    config.simulator_type = Backend::kGpuSim;
    config.simulation_type = method;
    checkNetwork(config, "128");
    Check(!config.max_bond_dimension,
          "Resolving the default mutated the caller's optional setting");
    config.max_bond_dimension = 128;
    checkNetwork(config, "128");
    config.max_bond_dimension = 64;
    checkNetwork(config, "64");
    config.max_bond_dimension.reset();
    config.native_options[bondKey] = "32";
    checkNetwork(config, "32");
    config.max_bond_dimension = 64;
    checkNetwork(config, "64");

    config.max_bond_dimension.reset();
    config.native_options.clear();
    config.simulator_type = Backend::kQCSim;
    checkNetwork(config, "");
  }
  for (const auto method : {Method::kStatevector, Method::kTensorNetwork}) {
    SimulatorConfig config;
    config.simulator_type = Backend::kGpuSim;
    config.simulation_type = method;
    checkNetwork(config, "");
  }
}

void TestAutomaticGpuMixedStateFallback() {
  using namespace MaestroExecution;
  using CF = Circuits::CircuitFactory<>;
  GetMaestroObjectWithMute();

  for (const auto method :
       {Method::kDensityMatrix, Method::kMatrixProductOperator})
    for (size_t workers : {size_t{1}, size_t{2}}) {
      struct NetworkHandle {
        unsigned long handle = CreateSimpleSimulator(1);
        ~NetworkHandle() { DestroySimpleSimulator(handle); }
      } owner;
      SimulatorConfig config;
      config.simulator_type = Backend::kGpuSim;
      config.simulation_type = method;
      config.seed = 123;
      config.optimize_circuit = false;
      const auto network = ConfigureNetwork(owner.handle, config);
      Check(network && network->GetSimulator()->GetType() == Backend::kQCSim &&
                network->GetSimulator()->GetSimulationType() ==
                    Method::kMatrixProductOperator,
            "Automatic GPU mixed-state execution needs a channel-capable CPU "
            "fallback");

      // Exercise the CPU placeholder even on machines with a working GPU.
      // An unavailable automatic candidate leaves this same simulator in use.
      network->SetOptimizeSimulator(false);
      network->SetMaxSimulators(workers);
      auto circuit = CF::CreateCircuit();
      circuit->AddOperation(
          CF::CreateGate(Circuits::QuantumGateType::kXGateType, 0));
      circuit->AddOperation(
          std::make_shared<Circuits::QuantumChannelOperation<>>(
              Types::qubits_vector{0},
              Simulators::QuantumChannel::GeneralizedAmplitudeDamping(1.0,
                                                                      0.0)));
      circuit->AddOperation(CF::CreateMeasurement({{0, 0}}));
      for (size_t shots : {size_t{1}, size_t{64}}) {
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, shots);
        Check(counts == Circuits::Circuit<>::ExecuteResults{{{false}, shots}},
              "GPU CPU fallback did not execute the exact noise channel");
        Check(network->GetLastSimulatorType() == Backend::kQCSim &&
                  network->GetLastSimulationType() ==
                      Method::kMatrixProductOperator,
              "GPU fallback changed the mixed-state representation");
      }
    }
}
