// Opt-in native C ABI checks: every MPI rank executes the same request.
#include "../maestrolib/Interface.h"
#include <boost/json.hpp>
#include <mpi.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace j = boost::json;

void RequireEveryRank(bool condition, const char* message) {
  int local = condition, all = 0;
  MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!all) throw std::runtime_error(message);
}

j::object Decode(char* pointer) {
  std::unique_ptr<char, decltype(&FreeResult)> buffer(pointer, FreeResult);
  if (!buffer) throw std::runtime_error("Null native result");
  auto result = j::parse(buffer.get()).as_object();
  if (!result.at("ok").as_bool())
    throw std::runtime_error(j::serialize(result));
  return result;
}

void CheckRankAgreement(const j::object& result) {
  if (result.at("operation") == "batch") {
    for (const auto& child : result.at("results").as_array())
      CheckRankAgreement(child.as_object());
    return;
  }
  RequireEveryRank(
      result.at("execution_metadata").at("backend") == "distributed_mpi_gpu",
      "Native request did not use the MPI GPU backend");
  j::object comparable{{"seed", result.at("seed")},
                       {"counts", result.at("counts")}};
  if (const auto* noise = result.if_contains("noise"))
    comparable["noise_seed"] = noise->at("seed");
  auto root = j::serialize(comparable);
  uint64_t length = root.size();
  MPI_Bcast(&length, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  root.resize(length);
  MPI_Bcast(root.data(), static_cast<int>(length), MPI_CHAR, 0, MPI_COMM_WORLD);
  RequireEveryRank(j::value(comparable) == j::parse(root),
                   "Native MPI seeds or sampling/readout differ across ranks");
}

j::object Run(const j::object& request) {
  const auto input = j::serialize(request);
  auto result = Decode(MaestroRunRequestJson(input.c_str()));
  CheckRankAgreement(result);
  return result;
}

int main(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
      MPI_SUCCESS)
    return 1;
  try {
    RequireEveryRank(provided >= MPI_THREAD_FUNNELED,
                     "MPI must provide funneled threading");
    j::object request{
        {"schema_version", 2},
        {"operation", "execute"},
        {"simulator",
         j::object{{"backend", "distributed_mpi_gpu"},
                   {"distribution", j::object{{"backend", "ex"}}}}},
        {"circuit",
         j::object{{"num_qubits", 4},
                   {"source",
                    "OPENQASM 2.0; qreg q[4]; creg c[4]; "
                    "h q[0]; h q[1]; h q[2]; h q[3]; measure q->c;"}}},
        {"execution", j::object{{"shots", 4096}}}};
    for (bool noisy : {false, true}) {
      if (noisy)
        request["noise"] =
            j::object{{"realizations", 8},
                      {"channels", j::array{j::object{{"kind", "bit_flip"},
                                                      {"targets", j::array{0}},
                                                      {"probability", 0.2}},
                                            j::object{{"kind", "readout"},
                                                      {"targets", j::array{1}},
                                                      {"probability", 0.3}}}}};
      const auto first = Run(request), second = Run(request);
      RequireEveryRank(first.at("seed") != second.at("seed"),
                       "Omitted native MPI seeds reused a fixed default");
      if (noisy)
        RequireEveryRank(
            first.at("noise").at("seed").to_number<uint32_t>() ==
                static_cast<uint32_t>(first.at("seed").to_number<uint64_t>()),
            "Default MPI noise seed differs from the execution seed");
      request["execution"].as_object()["seed"] = first.at("seed");
      RequireEveryRank(Run(request).at("counts") == first.at("counts"),
                       "Generated MPI seed did not replay sampling/readout");
      for (uint64_t seed : {uint64_t{0}, UINT64_MAX}) {
        request["execution"].as_object()["seed"] = seed;
        const auto seeded = Run(request);
        RequireEveryRank(seeded.at("seed").to_number<uint64_t>() == seed,
                         "Native MPI replaced an explicit seed");
        RequireEveryRank(Run(request).at("counts") == seeded.at("counts"),
                         "Explicit MPI seed did not replay sampling/readout");
      }
      request["execution"].as_object().erase("seed");
      const auto batch =
          Run(j::object{{"schema_version", 2},
                        {"operation", "batch"},
                        {"requests", j::array{request, request}}});
      RequireEveryRank(batch.at("results").at(0).at("seed") !=
                           batch.at("results").at(1).at("seed"),
                       "Native MPI batch children reused a fixed seed");
    }
    Decode(MaestroFinalizeDistributedMpiGpuJson());
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) std::cout << "Native MPI seed checks passed\n";
    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}
