# Python binding layout

`../bindings.cpp` is the nanobind module entry point. CMake lists each translation
unit explicitly; normal CMake and scikit-build builds use the same sources.

- `module.h`: registration entry points; config/enums register before functions
  with `SimulatorConfig` defaults.
- `config.cpp`, `simulator.cpp`, `circuit.cpp`, `noise.cpp`: Python class bindings.
- `api.cpp`: QASM parsing and module-level simulation functions.
- `execution.cpp`, `fidelity.cpp`, `evolution.cpp`: shared executor implementations.
- `noise_execution.cpp`: noise injection, seeding, and realization averaging.
- `checkpoint.cpp`: persistent prefix simulator and its Python registration.
- `causal_cone.cpp`: optional estimation preprocessing, independent of the
  network and backend implementations.
- `core.h`, `common.h`, `runtime.h`: internal declarations, shared binding types,
  and simulator-handle lifetime management.

Causal-cone reduction is opt-in via SimulatorConfig.enable_causal_cone_reduction.
It traces the union of observable supports backward through unitary operations,
then clones/remaps the retained operations and Pauli strings to compact indices.
All observables run together through the usual configured network. Unsupported
operations retain full execution; no backend policy or network implementation
is changed by the reducer.
