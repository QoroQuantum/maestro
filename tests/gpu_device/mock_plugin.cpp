// A C ABI plugin with namespace-global initialization and thread-local device
// state. No CUDA installation or hardware is needed for registry tests.
#include <cstdlib>
#include <cstring>

namespace {
int selected = -1;
int initializations = 0;
int live = 0;
thread_local int current = 0;
struct State {
  int device;
  unsigned qubits;
  unsigned bits;
};
State* Checked(void* obj) {
  auto* state = static_cast<State*>(obj);
  if (!state || state->device != current) std::abort();
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
  if (selected >= 0 || cudaSetDevice(device)) return 0;
  selected = device;
  return 1;
}
void* InitLib() {
  ++initializations;
  // Device 2 deliberately fails, so failed acquisitions must never be cached.
  return selected == 2 ? nullptr : &selected;
}
void FreeLib() {
  if (live || current != selected) std::abort();
}
int MockInitializations() { return initializations; }
int MockLiveStates() { return live; }
void* CreateStateVector(void*) {
  if (current != selected) std::abort();
  ++live;
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
}
