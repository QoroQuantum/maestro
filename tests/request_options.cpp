// Shared request/configuration checks require no MPI or GPU hardware.
#include "../Execution/Options.h"

void Check(bool, const char*);

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
