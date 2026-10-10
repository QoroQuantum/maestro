# Python binding layout

`../Bindings.cpp` is the nanobind module entry point. CMake lists each translation
unit explicitly; normal CMake and scikit-build builds use the same sources.

- `module.h`: registration entry points; config/enums register before functions
  with `SimulatorConfig` defaults.
- `config.cpp`, `simulator.cpp`, `circuit.cpp`, `noise.cpp`: Python class bindings.
- `api.cpp`: QASM parsing and module-level simulation functions.
- `execution.cpp`, `fidelity.cpp`, `evolution.cpp`: shared executor implementations.
- `noise_execution.cpp`: noise injection, seeding, and realization averaging.
- `checkpoint.cpp`: persistent prefix simulator and its Python registration.
- `core.h`, `common.h`, `runtime.h`: internal declarations, shared binding types,
  and simulator-handle lifetime management.
