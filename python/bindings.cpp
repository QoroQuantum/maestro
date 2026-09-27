#include "bindings/module.h"

NB_MODULE(maestro, m) {
  m.doc() = "Python bindings for Maestro Quantum Simulator";
  // Config and class registrations must precede functions using their defaults.
  maestro_bindings::bind_config(m);
  maestro_bindings::bind_simulator(m);
  maestro_bindings::bind_circuit(m);
  maestro_bindings::bind_api(m);
  maestro_bindings::bind_noise(m);
  maestro_bindings::bind_checkpoint(m);
}
