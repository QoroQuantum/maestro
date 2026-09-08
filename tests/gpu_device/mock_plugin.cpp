// A C ABI plugin with process-global initialization and thread-local device
// state. No CUDA installation or hardware is needed for registry tests.
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {
bool initialized = false;
int initializations = 0;
int cleanups = 0;
std::atomic<int> live{0};
std::atomic<int> creations[3]{};
thread_local int current = 0;
thread_local int selected = 0;
struct State {
  int device;
  unsigned qubits;
  unsigned bits;
};
State* Checked(void* obj) {
  auto* state = static_cast<State*>(obj);
  if (!state || !initialized) std::abort();
  // The plugin restores the native object's device on each operation.
  current = state->device;
  return state;
}
}  // namespace
extern "C" {
int ValidateLicense(const char*) { return 1; }
int GetGpuDeviceCount() { return 3; }
int cudaGetDevice(int* device) {
  *device = current;
  return 0;
}
int cudaSetDevice(int device) {
  if (device < 0 || device >= 3) return 1;
  current = device;
  return 0;
}
int SetGpuDevice(int device) {
  // Device 2 deliberately rejects selection; discovery still reports it.
  if (device == 2 || cudaSetDevice(device) != 0) return 0;
  selected = device;
  return 1;
}
void* InitLib() {
  ++initializations;
  initialized = true;
  return &initialized;
}
void FreeLib() {
  if (live) std::abort();
  ++cleanups;
  initializations = 0;
  initialized = false;
}
int MockCreationsOnDevice(int device) { return creations[device]; }
int MockSelectedDevice() { return selected; }
int MockCleanups() { return cleanups; }
int MockInitializations() { return initializations; }
int MockLiveStates() { return live; }
void* CreateStateVector(void*) {
  if (!initialized) std::abort();
  ++live;
  ++creations[selected];
  return new State{selected, 0, 0};
}
void DestroyStateVector(void* obj) {
  delete Checked(obj);
  --live;
}
int Create(void* obj, unsigned qubits) {
  auto* state = Checked(obj);
  state->qubits = qubits;
  state->bits = 0;
  return 1;
}
int ApplyX(void* obj, int qubit) {
  Checked(obj)->bits ^= 1U << qubit;
  return 1;
}
int AllProbabilities(void* obj, double* result) {
  auto* state = Checked(obj);
  for (unsigned i = 0; i < (1U << state->qubits); ++i)
    result[i] = i == state->bits;
  return 1;
}
int GetStateVectorGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int MPSGetGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int TNGetGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int DMGetGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int MPOGetGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int GetStabilizerGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int PauliPropGetGpuId(void* obj) { return obj ? static_cast<State*>(obj)->device : -1; }
int GetNrQubits(void* obj) { return Checked(obj)->qubits; }
int SetDataType(void* obj, int) {
  Checked(obj);
  return 1;
}
int IsDoublePrecision(void* obj) {
  Checked(obj);
  return 0;
}
void* Clone(void* obj) {
  ++live;
  return new State(*Checked(obj));
}

void* CreatePauliPropSimulator(int qubits) {
  void* obj = CreateStateVector(nullptr);
  Create(obj, qubits);
  return obj;
}
void DestroyPauliPropSimulator(void* obj) { DestroyStateVector(obj); }
void* CreateStabilizerSimulator(long long qubits, long long, long long, long long) {
  return CreatePauliPropSimulator(static_cast<int>(qubits));
}
void DestroyStabilizerSimulator(void* obj) { DestroyStateVector(obj); }
}
