// Statevector bridge shared by the local and MPI distributed plugins.
#pragma once
#if defined(__linux__) && defined(INCLUDED_BY_FACTORY)
#include "Configuration.h"
#include "DistributedGpuLibStateVectorSim.h"
#include "DistributedMpiGpuLibStateVectorSim.h"
#include <charconv>
#include <limits>
#include <numeric>
#include <sstream>

namespace Simulators::Private {
class DistributedGpuState : public ISimulator {
 public:
  explicit DistributedGpuState(bool mpi = false) : mpi(mpi) {}
  void Initialize() override { InitializeData(nrQubits, nullptr); }
  void InitializeState(size_t n,
                       std::vector<std::complex<double>>& values) override {
    InitializeVector(n, values);
  }
  void InitializeState(size_t n, Eigen::VectorXcd& values) override {
    InitializeVector(n, values);
  }
#ifndef NO_QISKIT_AER
  void InitializeState(size_t n,
                       AER::Vector<std::complex<double>>& values) override {
    InitializeVector(n, values);
  }
#endif
  void Configure(const char* key, const char* value) override {
    if (!key || !value)
      throw std::invalid_argument("Null distributed GPU configuration");
    const std::string k(key), v(value);
    if (configuration.WasApplied(k, v)) return;
    static const std::unordered_set<std::string> distributionKeys{
        "distributed_devices",
        "distributed_global_qubits",
        "distributed_flags",
        "distributed_backend",
        "distributed_max_queued_gates",
        "distributed_transfer_workspace_bytes",
        "distributed_snapshot_storage",
        "mpi_communicator",
        "mpi_p2p_bits"};
    if ((k.compare(0, 12, "distributed_") == 0 ||
         k.compare(0, 4, "mpi_") == 0) &&
        !distributionKeys.count(k))
      throw std::invalid_argument("Unknown distributed GPU option: " + k);
    if (!mpi && k.compare(0, 4, "mpi_") == 0)
      throw std::invalid_argument("MPI options require DistributedMpiGpu");
    if (k == "method" && v != "statevector")
      throw std::invalid_argument(
          "Distributed GPU simulators currently support only statevector");
    const bool allocationOption =
        k == "gpu_device" || k == "precision" || k == "use_double_precision" ||
        k.compare(0, 12, "distributed_") == 0 || k.compare(0, 4, "mpi_") == 0;
    if (state && allocationOption)
      throw std::logic_error(
          "Clear the distributed GPU state before changing allocation "
          "settings");
    if (k == "gpu_device") Configuration::ParseGpuDevice(v);
    if (k == "seed") {
      auto seed = ParseUnsigned(v);
      if (state) state->SetSeed(seed);
    }
    if (k == "precision" && v != "single" && v != "double")
      throw std::invalid_argument("precision must be single or double");
    if (k == "use_double_precision" && v != "0" && v != "1" && v != "false" &&
        v != "true")
      throw std::invalid_argument("use_double_precision must be a boolean");
    if (k == "distributed_devices" || k == "distributed_global_qubits")
      ParseList(v);
    if (k == "distributed_flags" || k == "distributed_max_queued_gates" ||
        k == "distributed_transfer_workspace_bytes" || k == "mpi_p2p_bits")
      ParseUnsigned(v);
    if (k == "mpi_communicator") ParseCommunicator(v);
    if (k == "distributed_snapshot_storage" && v != "host" && v != "gpu")
      throw std::invalid_argument(
          "distributed_snapshot_storage must be host or gpu");
    if (k == "distributed_flags" && ParseUnsigned(v) > 15)
      throw std::invalid_argument("Unsupported distribution flags");
    if (k == "distributed_max_queued_gates" &&
        (ParseUnsigned(v) < 1 || ParseUnsigned(v) > 65536))
      throw std::invalid_argument("Queue size must be 1..65536");
    if (k == "mpi_p2p_bits" && ParseUnsigned(v) > 5)
      throw std::invalid_argument("mpi_p2p_bits must be 0..5");
    if (k == "distributed_backend" && v != "ex" && v != "conventional")
      throw std::invalid_argument(
          "distributed_backend must be ex or conventional");
    if (mpi && k == "distributed_backend" && v != "ex")
      throw std::invalid_argument("MPI requires the Ex backend");
    configuration.SetConfiguration(k, v);
  }
  std::string GetConfiguration(const char* key) const override {
    const std::string k(key);
    if (k == "method") return "statevector";
    if (state && (k == "distributed_shard_devices" ||
                  k == "distributed_configured_global_qubits" ||
                  k == "distributed_qubit_layout")) {
      std::vector<int32_t> values(k == "distributed_qubit_layout" ? nrQubits
                                                                  : 32);
      int count = k == "distributed_shard_devices"
                      ? state->GetShardDevices(values.data(), values.size())
                  : k == "distributed_configured_global_qubits"
                      ? state->GetGlobalQubits(values.data(), values.size())
                      : state->GetQubitLayout(values.data(), values.size());
      values.resize(count);
      return Join(values);
    }
    return configuration.GetConfiguration(key);
  }
  const std::unordered_map<std::string, std::string>& GetConfigMap()
      const override {
    return configuration.GetConfigMap();
  }
  size_t AllocateQubits(size_t count) override {
    if (state)
      throw std::logic_error("Clear before allocating distributed GPU qubits");
    if (count >= 63 || nrQubits >= 63 - count)
      throw std::invalid_argument(
          "Distributed GPU requires fewer than 63 qubits");
    auto old = nrQubits;
    nrQubits += count;
    return old;
  }
  size_t GetNumberOfQubits() const override { return nrQubits; }
  void Clear() override {
    state.reset();
    nrQubits = 0;
  }
  void Reset() override { Native().Reset(); }
  size_t Measure(const Types::qubits_vector& qubits) override {
    auto bits = MeasureMany(qubits);
    size_t result = 0;
    for (size_t i = 0; i < bits.size(); ++i)
      if (bits[i]) result |= size_t{1} << i;
    return result;
  }
  std::vector<bool> MeasureMany(const Types::qubits_vector& qubits) override {
    auto qb = Qubits(qubits);
    if (qb.empty()) return {};
    std::vector<int> values(qb.size());
    Native().MeasureQubitsCollapse(qb.data(), values.data(), values.size());
    NotifyObservers(qubits);
    return std::vector<bool>(values.begin(), values.end());
  }
  void ApplyReset(const Types::qubits_vector& qubits) override {
    auto bits = MeasureMany(qubits);
    for (size_t i = 0; i < qubits.size(); ++i)
      if (bits[i]) ApplyX(qubits[i]);
  }
  double Probability(Types::qubit_t outcome) override {
    return Native().BasisStateProbability(outcome);
  }
  std::complex<double> Amplitude(Types::qubit_t outcome) override {
    double real, imag;
    Native().Amplitude(outcome, &real, &imag);
    return {real, imag};
  }
  std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override {
    return Amplitude(outcome);
  }
  std::complex<double> ProjectOnZero() override { return Amplitude(0); }
  std::vector<double> AllProbabilities() override {
    std::vector<double> values(size_t{1} << nrQubits);
    Native().AllProbabilities(values.data());
    return values;
  }
  std::vector<double> Probabilities(
      const Types::qubits_vector& outcomes) override {
    std::vector<double> values;
    for (auto outcome : outcomes) values.push_back(Probability(outcome));
    return values;
  }
  std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(
      const Types::qubits_vector& qubits, size_t shots = 1000) override {
    auto qb = Qubits(qubits);
    std::unordered_map<Types::qubit_t, Types::qubit_t> result;
    if (qb.empty() || !shots) return result;
    if (shots > std::numeric_limits<unsigned>::max())
      throw std::invalid_argument("Too many GPU samples");
    std::vector<long> samples(shots);
    Native().Sample(shots, samples.data(), qb.size(), qb.data());
    for (auto sample : samples) ++result[static_cast<Types::qubit_t>(sample)];
    return result;
  }
  std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(
      const Types::qubits_vector& qubits, size_t shots = 1000) override {
    std::unordered_map<std::vector<bool>, Types::qubit_t> result;
    for (const auto& [outcome, count] : SampleCounts(qubits, shots)) {
      std::vector<bool> bits(qubits.size());
      for (size_t i = 0; i < bits.size(); ++i) bits[i] = (outcome >> i) & 1;
      result[bits] += count;
    }
    return result;
  }
  double ExpectationValue(const std::string& pauli) override {
    return Native().ExpectationValue(pauli.c_str(), pauli.size());
  }
  SimulatorType GetType() const override {
    return mpi ? SimulatorType::kDistMpiGpuSim : SimulatorType::kDistGpuSim;
  }
  SimulationType GetSimulationType() const override {
    return SimulationType::kStatevector;
  }
  int GetGpuDevice() const override {
    return state ? state->GetStateVectorGpuId() : -1;
  }
  void Flush() override { Native().Synchronize(); }
  void SaveStateToInternalDestructive() override {
    Native().SaveStateDestructive();
  }
  void RestoreInternalDestructiveSavedState() override {
    Native().RestoreStateFreeSaved();
  }
  void SaveState() override {
    if (configuration.GetConfiguration("distributed_snapshot_storage") ==
        "host")
      Native().SaveStateToHost();
    else
      Native().SaveState();
  }
  void RestoreState() override { Native().RestoreStateNoFreeSaved(); }
  void SetMultithreading(bool = true) override {}
  bool GetMultithreading() const override { return !mpi; }
  bool IsQcsim() const override { return false; }
  Types::qubit_t MeasureNoCollapse() override {
    return Native().MeasureAllQubitsNoCollapse();
  }
  std::vector<bool> MeasureNoCollapseMany() override {
    auto outcome = MeasureNoCollapse();
    std::vector<bool> bits(nrQubits);
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = (outcome >> i) & 1;
    return bits;
  }

 protected:
  int QubitIndex(Types::qubit_t q) const {
    if (q >= nrQubits)
      throw std::out_of_range("Distributed GPU qubit is out of range");
    return static_cast<int>(q);
  }
  DistributedGpuLibStateVectorSim& Native() const {
    if (!state)
      throw std::logic_error("Distributed GPU state is not initialized");
    return *state;
  }
  static uint64_t ParseUnsigned(const std::string& v) {
    uint64_t result = 0;
    auto parsed = std::from_chars(v.data(), v.data() + v.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != v.data() + v.size())
      throw std::invalid_argument("Expected an unsigned integer: " + v);
    return result;
  }
  static int64_t ParseCommunicator(const std::string& value) {
    size_t used = 0;
    const auto handle = std::stoll(value, &used);
    if (used != value.size())
      throw std::invalid_argument("Invalid MPI communicator handle");
    return handle;
  }
  static int ParseInt(const std::string& v) {
    int result = 0;
    auto parsed = std::from_chars(v.data(), v.data() + v.size(), result);
    if (parsed.ec != std::errc() || parsed.ptr != v.data() + v.size())
      throw std::invalid_argument("Expected an integer: " + v);
    return result;
  }
  static std::vector<int32_t> ParseList(const std::string& value) {
    std::vector<int32_t> values;
    if (value.empty()) return values;
    size_t begin = 0;
    do {
      const auto end = value.find(',', begin);
      int item = ParseInt(
          value.substr(begin, end == std::string::npos ? end : end - begin));
      if (item < 0)
        throw std::invalid_argument("Distributed indices must be nonnegative");
      values.push_back(item);
      if (end == std::string::npos) break;
      begin = end + 1;
    } while (true);
    return values;
  }
  static std::string Join(const std::vector<int32_t>& values) {
    std::string result;
    for (auto v : values) {
      if (!result.empty()) result += ',';
      result += std::to_string(v);
    }
    return result;
  }
  std::vector<int> Qubits(const Types::qubits_vector& qubits) const {
    if (qubits.size() > nrQubits)
      throw std::invalid_argument("Too many measured qubits");
    std::vector<int> values;
    std::unordered_set<Types::qubit_t> seen;
    for (auto q : qubits) {
      if (q >= nrQubits || !seen.insert(q).second)
        throw std::invalid_argument("Invalid or duplicate qubit");
      values.push_back(static_cast<int>(q));
    }
    return values;
  }
  uint64_t Option(const char* key, uint64_t fallback) const {
    return configuration.IsSet(key)
               ? ParseUnsigned(configuration.GetConfiguration(key))
               : fallback;
  }
  template <class V>
  void InitializeVector(size_t n, V& values) {
    if (!n || n >= 63 || static_cast<size_t>(values.size()) != (size_t{1} << n))
      throw std::invalid_argument("Statevector length must equal 2^num_qubits");
    InitializeData(n, reinterpret_cast<const double*>(values.data()));
  }
  void InitializeData(size_t n, const double* values) {
    if (state)
      throw std::logic_error(
          "Clear before initializing a distributed GPU state again");
    if (!n || n >= 63)
      throw std::invalid_argument("Distributed GPU requires 1..62 qubits");
    int device = configuration.IsSet("gpu_device")
                     ? Configuration::ParseGpuDevice(
                           configuration.GetConfiguration("gpu_device"))
                     : 0;
    auto devices =
        ParseList(configuration.GetConfiguration("distributed_devices"));
    auto globals =
        ParseList(configuration.GetConfiguration("distributed_global_qubits"));
    std::unique_ptr<DistributedGpuLibStateVectorSim> next;
    if (mpi) {
      auto lib = DistributedMpiGpuLibrary::GetInstance();
      DistributedMpiGpuLibrary::Communicator descriptor{
          sizeof(DistributedMpiGpuLibrary::Communicator), 0, 0};
      const DistributedMpiGpuLibrary::Communicator* comm = nullptr;
      if (configuration.IsSet("mpi_communicator")) {
        descriptor.fortran_handle = ParseCommunicator(
            configuration.GetConfiguration("mpi_communicator"));
        comm = &descriptor;
      }
      const auto info = lib->GetRuntimeInfo(comm);
      if (!devices.empty() && devices.size() != static_cast<size_t>(info.size))
        throw std::invalid_argument(
            "MPI distributed_devices must contain one ordinal per rank");
      if (!configuration.IsSet("gpu_device"))
        device = devices.empty() ? info.default_device : devices[info.rank];
      if (devices.empty()) {
        devices.resize(info.size);
        lib->GatherDevices(comm, device, devices);
      }
      next = std::make_unique<DistributedMpiGpuLibStateVectorSim>(
          lib, comm, device, Option("mpi_p2p_bits", 0));
    } else {
      auto lib = DistributedGpuLibrary::GetInstance();
      if (devices.empty()) {
        if (configuration.IsSet("gpu_device"))
          devices.push_back(device);
        else {
          lib->RequireLoaded();
          const int count = lib->GetGpuDeviceCount();
          if (count <= 0) lib->Fail("GetGpuDeviceCount");
          int shards = 1;
          while (shards < 32 && shards * 2 <= count &&
                 static_cast<uint64_t>(shards * 2) <= (uint64_t{1} << (n - 1)))
            shards *= 2;
          for (int i = 0; i < shards; ++i) devices.push_back(i);
        }
      }
      const int backend = configuration.GetConfiguration(
                              "distributed_backend") == "conventional"
                              ? 0
                              : 1;
      next = std::make_unique<DistributedGpuLibStateVectorSim>(
          lib, lib->CreateNative(device, backend));
    }
    if (devices.empty() || devices.size() > 32 ||
        (devices.size() & (devices.size() - 1)))
      throw std::invalid_argument(
          "distributed_devices must contain a power of two shards (1..32)");
    size_t globalBits = 0;
    while ((size_t{1} << globalBits) < devices.size()) ++globalBits;
    if (globalBits >= n)
      throw std::invalid_argument(
          "Distribution requires at least one local qubit");
    if (!configuration.IsSet("distributed_global_qubits"))
      for (size_t i = 0; i < globalBits; ++i) globals.push_back(i);
    DistributedGpuApi::MgdDistributionConfig dist{
        sizeof(dist),
        static_cast<uint32_t>(Option("distributed_flags", 0)),
        static_cast<uint32_t>(globals.size()),
        globals.data(),
        static_cast<uint32_t>(devices.size()),
        devices.data()};
    next->ConfigureDistribution(&dist);
    const bool useDouble =
        configuration.IsSet("precision")
            ? configuration.GetConfiguration("precision") == "double"
            : (configuration.GetConfiguration("use_double_precision") == "1" ||
               configuration.GetConfiguration("use_double_precision") ==
                   "true");
    next->SetDataType(useDouble ? 1 : 0);
    if (configuration.IsSet("distributed_max_queued_gates") ||
        configuration.IsSet("distributed_transfer_workspace_bytes")) {
      DistributedGpuApi::MgdExExecutionConfig execution{
          sizeof(execution),
          static_cast<uint32_t>(Option("distributed_max_queued_gates", 1024)),
          Option("distributed_transfer_workspace_bytes", 16ULL * 1024 * 1024)};
      next->SetExExecutionConfig(&execution);
    }
    // Identical explicit/default seeds keep MPI rank streams aligned.
    if (configuration.IsSet("seed") || mpi) next->SetSeed(Option("seed", 0));
    if (values)
      next->CreateWithState(n, values);
    else
      next->Create(n);
    nrQubits = n;
    state = std::move(next);
  }
  bool mpi;
  size_t nrQubits = 0;
  Configuration configuration;
  std::unique_ptr<DistributedGpuLibStateVectorSim> state;
};
class DistributedMpiGpuState : public DistributedGpuState {
 public:
  DistributedMpiGpuState() : DistributedGpuState(true) {}
};
}  // namespace Simulators::Private
#endif
