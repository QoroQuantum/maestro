// Seed parsing must remain independent of entropy, MPI and GPU availability.
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
