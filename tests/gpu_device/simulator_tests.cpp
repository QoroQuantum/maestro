#include "../../Simulators/Factory.h"
#include "../../Network/SimpleDisconnectedNetwork.h"
#include <future>
#include <iostream>

using Factory = Simulators::SimulatorsFactory;
constexpr auto gpu = Simulators::SimulatorType::kGpuSim;
constexpr auto sv = Simulators::SimulationType::kStatevector;
void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class F>
void Reject(F&& action) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid device/configuration accepted");
}
// Exercise the unchanged estimator interface: a synchronous estimator must
// see the network's device before it creates or initializes its candidate.
class DeviceEstimator : public Estimators::SimulatorsEstimatorInterface<> {
 public:
  bool IsInitialized() const override { return true; }
  std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(
      const std::vector<
          std::pair<Simulators::SimulatorType, Simulators::SimulationType>>&,
      std::shared_ptr<Circuits::Circuit<>>&, size_t&, size_t, size_t, size_t,
      Simulators::SimulatorType& type, Simulators::SimulationType& method,
      std::vector<bool>&, long long, double, const std::string&,
      const std::string&, size_t, const std::vector<std::string>*,
      bool) const override {
    Require(Factory::ResolveGpuDevice() == 1,
            "estimator used the process default");
    type = gpu;
    method = sv;
    return Factory::CreateSimulator(gpu, sv);
  }
};
class EstimatedNetwork : public Network::SimpleDisconnectedNetwork<> {
 public:
  EstimatedNetwork() : SimpleDisconnectedNetwork({2}, {2}) {
    simulatorsEstimator = std::make_unique<DeviceEstimator>();
  }
};

int main() {
  try {
    // This executable uses the mock via LD_LIBRARY_PATH; fail if misconfigured.
    Require(Factory::GetGpuDeviceCount() == 3, "mock GPU plugin required");
    Require(Factory::InitGpuLibraryWithMute(), "mock initialization failed");
    Factory::SelectGpuDevice(1);
    Require(Factory::InitGpuLibraryWithMute(),
            "device 1 initialization failed");
    auto one = Factory::CreateSimulator(gpu, sv);
    auto zero = Factory::CreateSimulatorUnique(gpu, sv);
    zero->Configure("gpu_device", "0");
    for (auto* sim : {one.get(), zero.get()}) {
      sim->AllocateQubits(2);
      sim->Initialize();
      sim->ApplyX(0);
      Require(sim->AllProbabilities()[1] == 1, "wrong simulator state");
    }
    Require(one->GetGpuDevice() == 1 && zero->GetGpuDevice() == 0,
            "configuration did not reach native GPU objects");
    auto clone = one->Clone();
    Require(clone->GetGpuDevice() == 1, "cloned native state moved GPU");
    Require(clone->GetConfiguration("gpu_device") == "1", "clone lost device");
    Require(clone->AllProbabilities()[1] == 1, "clone lost state");
    one->Configure("gpu_device", "1");  // idempotent reapplication is allowed
    Reject([&] { one->Configure("gpu_device", "0"); });
    for (auto value : {"-1", "1junk", "", "2147483648", "1.0"})
      Reject([&] { zero->Configure("gpu_device", value); });
    Require(one->GetConfiguration("gpu_device") == "1",
            "failed update mutated device");
    one->Clear();
    Require(one->GetGpuDevice() == -1, "cleared simulator reports a live GPU");
    one->Configure("gpu_device", "0");
    one->AllocateQubits(1);
    one->Initialize();
    Require(one->AllProbabilities()[0] == 1, "clear/reinitialize failed");
    auto invalid = Factory::CreateSimulator(gpu, sv);
    invalid->Configure("gpu_device", "3");
    invalid->AllocateQubits(1);
    Reject([&] { invalid->Initialize(); });

    Network::SimpleDisconnectedNetwork<> network({2}, {2});
    network.Configure("gpu_device", "1");
    network.CreateSimulator(gpu, sv);
    Require(network.GetGpuDevice() == 1, "network did not expose native placement");
    Require(network.GetSimulator()->GetConfiguration("gpu_device") == "1",
            "network lost device");
    Reject([&] { network.Configure("gpu_device", "0"); });
    network.CreateSimulator(gpu, sv);
    Require(network.GetSimulator()->GetConfiguration("gpu_device") == "1",
            "network mutation was not transactional");
    network.GetSimulator()->Clear();
    network.Configure("gpu_device", "0");
    network.CreateSimulator(gpu, sv);
    Require(network.GetSimulator()->GetConfiguration("gpu_device") == "0",
            "network recreation lost device");

    // Scoped defaults used by legacy synchronous estimators cannot affect other
    // threads, nor can they override a simulator's explicit configuration.
    Factory::SelectGpuDevice(0);
    {
      Factory::ScopedGpuDevice scope(1);
      Require(Factory::CreateSimulator(gpu, sv)->GetConfiguration(
                  "gpu_device") == "1",
              "scoped estimator device lost");
      auto other = std::async(std::launch::async,
                              [] { return Factory::ResolveGpuDevice(); });
      Require(other.get() == 0, "scoped device leaked to another thread");
    }
    Require(Factory::ResolveGpuDevice() == 0, "scoped default not restored");

    Network::SimpleDisconnectedNetwork<> defaultNetwork({2}, {2});
    defaultNetwork.CreateSimulator(gpu, sv);
    Factory::SelectGpuDevice(1);
    auto clonedNetwork = defaultNetwork.Clone();
    Require(clonedNetwork->GetSimulator()->GetConfiguration("gpu_device") == "0",
            "network clone followed a changed process default");
    defaultNetwork.CreateSimulator(gpu, sv);
    Require(defaultNetwork.GetSimulator()->GetConfiguration("gpu_device") == "0",
            "network recreation followed a changed process default");
    Factory::SelectGpuDevice(0);

    EstimatedNetwork estimated;
    estimated.SetOptimizeSimulator(true);
    estimated.Configure("gpu_device", "1");
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    size_t shots = 1;
    auto type = gpu;
    auto method = sv;
    std::vector<bool> executed;
    auto chosen = estimated.ChooseBestSimulator(circuit, shots, 2, 2, 2, type,
                                                method, executed, false, true);
    Require(chosen && chosen->GetConfiguration("gpu_device") == "1",
            "network estimator lost requested device");
    Require(chosen->AllProbabilities()[0] == 1,
            "estimated simulator not initialized");
    Require(Factory::ResolveGpuDevice() == 0,
            "network estimator changed global default");
    std::cout << "GPU simulator and network configuration tests passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
