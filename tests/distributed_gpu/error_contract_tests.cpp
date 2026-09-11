#include "../../Simulators/DistributedGpuLibStateVectorSim.h"
#include <iostream>
#include <limits>
using namespace Simulators;
namespace {
int destroyed = 0;
const char* LastError() { return "injected plugin failure"; }
class FakeLibrary : public DistributedGpuLibrary {
 public:
  FakeLibrary() {
    GetLastError = LastError;
    DestroyStateVector = [](void*) { ++destroyed; };
    liveStates = 1;
    ApplyX = [](void*, int) { return 0; };
    MeasureQubitCollapse = [](void*, int) { return -1; };
    MeasureQubitNoCollapse = [](void*, int) { return -1; };
    MeasureAllQubitsCollapse = [](void*) -> unsigned long long {
      return UINT64_MAX;
    };
    MeasureAllQubitsNoCollapse = [](void*) -> unsigned long long {
      return UINT64_MAX;
    };
    BasisStateProbability = [](void*, long long) {
      return std::numeric_limits<double>::quiet_NaN();
    };
    ExpectationValue = [](void*, const char*, int) {
      return std::numeric_limits<double>::quiet_NaN();
    };
    Clone = [](void*) -> void* { return nullptr; };
  }
};
template <class F>
void Reject(F&& action) {
  try {
    action();
  } catch (const std::runtime_error& e) {
    if (std::string(e.what()).find("injected plugin failure") !=
        std::string::npos)
      return;
    throw;
  }
  throw std::runtime_error("Error sentinel was silently accepted");
}
}  // namespace
int main() {
  try {
    auto lib = std::make_shared<FakeLibrary>();
    {
      DistributedGpuLibStateVectorSim original(lib, reinterpret_cast<void*>(1));
      DistributedGpuLibStateVectorSim state(std::move(original));
      Reject([&] { state.ApplyX(0); });
      Reject([&] { state.MeasureQubitCollapse(0); });
      Reject([&] { state.MeasureQubitNoCollapse(0); });
      Reject([&] { state.MeasureAllQubitsCollapse(); });
      Reject([&] { state.MeasureAllQubitsNoCollapse(); });
      Reject([&] { state.BasisStateProbability(0); });
      Reject([&] { state.ExpectationValue("X", 1); });
      Reject([&] { state.Clone(); });
      // Zero is a measurement outcome, not a failed status.
      lib->MeasureQubitCollapse = [](void*, int) { return 0; };
      if (state.MeasureQubitCollapse(0) != 0) return 1;
      lib->GetLastError = []() -> const char* { return ""; };
      lib->GetStateVectorGpuId = [](void*) { return -1; };
      if (state.GetStateVectorGpuId() != -1) return 1;
    }
    if (destroyed != 1)
      throw std::runtime_error("Move double-freed the native handle");
    std::cout << "Distributed GPU error contract passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
