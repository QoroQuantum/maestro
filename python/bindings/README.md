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
- Pre-execution circuit transforms (causal-cone pruning, symmetry reduction) are
  hosted in native C++ under `Execution/Transforms/` and invoked via `TransformPipeline`.
- `core.h`, `common.h`, `runtime.h`: internal declarations, shared binding types,
  and simulator-handle lifetime management.

Causal-cone reduction is enabled by default via SimulatorConfig.enable_causal_cone_reduction (set to False to disable).
It traces the union of observable supports backward through unitary operations,
then clones/remaps the retained operations and Pauli strings to compact indices.
All observables run together through the usual configured network. Unsupported
operations retain full execution. Distributed GPU and MPI backends also retain
full execution to preserve explicit qubit layouts and the minimum register width
required by their shard counts.

## Correctness invariant

Starting with the union of the observable supports, walk the gates in reverse
execution order. If a unitary gate's qubits are disjoint from the current support,
`G† O G = O`, so it can be dropped. Otherwise keep the gate and add all its qubits
(including controls) to the support. This maintains a conservative superset of
the Heisenberg-evolved support for every observable. It need not be the smallest
cone: commuting gates and algebraic cancellations can leave extra gates retained.
Restore forward order, clone the retained gates, and apply the same one-to-one
qubit map to gates and observables. The removed qubits contribute an identity
factor, whose expectation is one in the all-zero product input used by these
execution APIs. Noise is injected before this pass; channels and classical or
measurement operations trigger full execution. Approximate simulator truncation
can change after reduction, with no error bound supplied by the reducer.
