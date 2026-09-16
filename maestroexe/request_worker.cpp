// Native request runner. The optional MPI target initializes application MPI
// and executes identical native calls on every rank. No Python dependency.
#include "../maestrolib/Interface.h"
#include <boost/json.hpp>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#ifdef MAESTRO_REQUEST_MPI
#include <mpi.h>
#endif
namespace json = boost::json;
using Buffer = std::unique_ptr<char, decltype(&FreeResult)>;
constexpr size_t MaxRequest = 16 * 1024 * 1024;
void Output(const json::object& result, bool framed) {
  if (framed) std::cout << "MAESTRO_RESULT_V2 ";
  std::cout << json::serialize(result) << '\n' << std::flush;
}
json::object Decode(char* pointer) {
  Buffer buffer(pointer, FreeResult);
  if (!buffer) throw std::runtime_error("Native API returned a null buffer");
  return json::parse(buffer.get()).as_object();
}
bool Success(const json::object& value) {
  const auto* ok = value.if_contains("ok");
  return ok && ok->is_bool() && ok->as_bool();
}
int main(int argc, char** argv) {
  int rank = 0;
#ifdef MAESTRO_REQUEST_MPI
  int ranks = 1;
#endif
  bool framed = false, capabilities = false, validate = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--framed-result")
      framed = true;
    else if (arg == "--capabilities")
      capabilities = true;
    else if (arg == "--validate")
      validate = true;
    else {
      std::cerr << "Unknown argument: " << arg << '\n';
      return 2;
    }
  }
#ifdef MAESTRO_REQUEST_MPI
  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) !=
      MPI_SUCCESS)
    return 3;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  if (provided < MPI_THREAD_FUNNELED) {
    MPI_Abort(MPI_COMM_WORLD, 3);
    return 3;
  }
#endif
  try {
    std::string input;
    if (!capabilities && rank == 0) {
      char buffer[8192];
      while (std::cin.read(buffer, sizeof(buffer)) || std::cin.gcount()) {
        input.append(buffer, static_cast<size_t>(std::cin.gcount()));
        if (input.size() > MaxRequest)
          throw std::invalid_argument("Request exceeds 16 MiB");
      }
    }
#ifdef MAESTRO_REQUEST_MPI
    if (rank == 0 && framed) {
      const auto* job = std::getenv("SLURM_JOB_ID");
      const auto* step = std::getenv("SLURM_STEP_ID");
      if (job && step)
        std::cout << "MAESTRO_STEP_V2 " << job << '.' << step << '\n'
                  << std::flush;
    }
    uint64_t length = input.size();
    MPI_Bcast(&length, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    if (length > MaxRequest)
      throw std::invalid_argument("Oversized collective request");
    input.resize(length);
    MPI_Bcast(input.data(), static_cast<int>(length), MPI_BYTE, 0,
              MPI_COMM_WORLD);
#endif
    auto result = capabilities
                      ? Decode(MaestroGetCapabilitiesJson())
                      : Decode(MaestroValidateRequestJson(input.c_str()));
    int valid = Success(result) ? 1 : 0;
#ifdef MAESTRO_REQUEST_MPI
    int allValid = 0;
    MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    valid = allValid;
    if (!valid && Success(result))
      result = json::object{
          {"ok", false},
          {"error", json::object{{"code", "collective_validation_failure"},
                                 {"message", "A peer rejected the request"}}}};
#endif
    if (valid && !capabilities && !validate)
      result = Decode(MaestroRunRequestJson(input.c_str()));
    if (!Success(result)) {
      if (rank == 0)
        Output(result, framed);
      else
        std::cerr << json::serialize(result) << '\n';
#ifdef MAESTRO_REQUEST_MPI
      // A peer may still be inside a backend collective. Abort instead of
      // waiting in another collective and deadlocking the entire worker group.
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      return 1;
    }
#ifdef MAESTRO_REQUEST_MPI
    MPI_Barrier(MPI_COMM_WORLD);  // All Run calls destroyed their states.
    const auto shutdown = Decode(MaestroFinalizeDistributedMpiGpuJson());
    if (!Success(shutdown)) throw std::runtime_error(json::serialize(shutdown));
    result["mpi"] = json::object{{"ranks", ranks},
                                 {"result_rank", 0},
                                 {"aggregation", "one_replicated_result"}};
    MPI_Finalize();
#endif
    if (rank == 0) Output(result, framed);
    return 0;
  } catch (const std::exception& error) {
    json::object result{{"schema_version", 2},
                        {"ok", false},
                        {"error", json::object{{"code", "worker_failure"},
                                               {"message", error.what()}}}};
    if (rank == 0)
      Output(result, framed);
    else
      std::cerr << json::serialize(result) << '\n';
#ifdef MAESTRO_REQUEST_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }
}
