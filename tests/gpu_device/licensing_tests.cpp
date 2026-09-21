#include "../../Simulators/Factory.h"
#include "../../Simulators/GpuLibraryRegistry.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace Simulators;
using Factory = SimulatorsFactory;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

int main(int argc, char** argv) {
  try {
    Require(argc == 3,
            "usage: licensing_tests gpu|distributed|mpi valid|load|init");
    const std::string backend = argv[1], failure = argv[2];
    const bool valid = failure == "valid";
    setenv("MAESTRO_TEST_LICENSE_FAILURE", valid ? "" : failure.c_str(), 1);
    std::ostringstream diagnostics;
    auto* previous = std::cerr.rdbuf(diagnostics.rdbuf());
    struct Restore {
      std::streambuf* previous;
      ~Restore() { std::cerr.rdbuf(previous); }
    } restore{previous};
    Utils::Library* library = nullptr;
    if (backend == "gpu") {
      auto lib = GpuLibrary::GetInstance();
      library = lib.get();
      for (int n = 0; n < 3; ++n) {
        Require((Factory::GetGpuDeviceCount() > 0) == valid,
                "GPU discovery ignored licensing");
        Require(bool(Factory::GetGpuLibrary()) == valid,
                "GPU acquisition ignored licensing");
        Require(
            bool(Factory::CreateSimulator(
                SimulatorType::kGpuSim, SimulationType::kStatevector)) == valid,
            "GPU factory availability mismatch");
        Require(
            bool(Factory::CreateSimulatorUnique(
                SimulatorType::kGpuSim, SimulationType::kStatevector)) == valid,
            "GPU unique factory availability mismatch");
      }
    } else if (backend == "distributed") {
      auto lib = Factory::GetDistributedGpuLibrary();
      library = lib.get();
      for (int n = 0; n < 3; ++n) {
        Require(Factory::IsDistributedGpuAvailable() == valid,
                "distributed discovery ignored licensing");
        Require(bool(Factory::CreateSimulator(SimulatorType::kDistGpuSim,
                                              SimulationType::kStatevector)) ==
                    valid,
                "distributed factory availability mismatch");
        Require(bool(Factory::CreateSimulatorUnique(
                    SimulatorType::kDistGpuSim,
                    SimulationType::kStatevector)) == valid,
                "distributed unique factory availability mismatch");
        if (valid) {
          auto* obj = lib->CreateNative(0, 0);
          Require(obj, "native construction failed");
          lib->DestroyNative(obj);
        } else {
          try {
            lib->RequireLoaded();
            throw std::logic_error("unavailable library accepted");
          } catch (const std::runtime_error& e) {
            Require(std::string(e.what()).find("license expired") !=
                        std::string::npos,
                    "loader lost licensing diagnostic");
          }
        }
      }
    } else {
      auto lib = Factory::GetDistributedMpiGpuLibrary();
      library = lib.get();
      for (int n = 0; n < 3; ++n) {
        if (valid) {
          auto* obj = lib->CreateMpiNative(nullptr, 0, 0);
          Require(obj, "MPI native construction failed");
          lib->DestroyNative(obj);
        } else {
          try {
            lib->CreateMpiNative(nullptr, 0, 0);
            throw std::logic_error("unavailable MPI library accepted");
          } catch (const std::runtime_error& e) {
            Require(std::string(e.what()).find("license expired") !=
                        std::string::npos,
                    "MPI loader lost licensing diagnostic");
          }
        }
      }
      lib->FinalizeBackend();
    }
    auto count = [&](const char* symbol) {
      auto fn = reinterpret_cast<int (*)()>(library->GetFunction(symbol));
      Require(fn, "mock counter missing");
      return fn();
    };
    Require(count("MockValidations") == 1, "licensing repeated after loading");
    Require(count("MockInitializations") == (failure == "load" ? 0 : 1),
            "library initialization repeated or ran after rejected license");
    if (!valid && backend != "mpi")
      Require(diagnostics.str().find("license expired") != std::string::npos,
              "fallback log lost licensing diagnostic");
    std::cout << backend << " " << failure << " licensing boundary passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
