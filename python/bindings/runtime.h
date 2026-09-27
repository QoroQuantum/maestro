#pragma once
#include "common.h"

namespace maestro_bindings {
// RAII Wrapper to ensure the simulator handle is destroyed strictly
struct ScopedSimulator {
  unsigned long int handle;

  explicit ScopedSimulator(int num_qubits) {
    GetMaestroObjectWithMute();
    handle = CreateSimpleSimulator(num_qubits);
  }

  ~ScopedSimulator() {
    if (handle != 0) DestroySimpleSimulator(handle);
  }

  // Disable copying to prevent double-free
  ScopedSimulator(const ScopedSimulator&) = delete;
  ScopedSimulator& operator=(const ScopedSimulator&) = delete;
};

}  // namespace maestro_bindings
