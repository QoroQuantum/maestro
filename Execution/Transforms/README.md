# Pre-Execution Transforms Pipeline

The `Execution/Transforms` module provides an extensible, native C++ pipeline for analyzing and reducing circuits and observables before simulator allocation or network configuration.

Running reductions before simulator allocation allows all simulation backends (Statevector, MPS, TensorNetwork, Aer, GPU) to automatically simulate a smaller effective register without needing backend-specific modifications.

---

## Architecture

* **`TransformContext.h` (`TransformContext`)**:
  Shared state passed across the pipeline. Contains mutable references to the circuit, observables, backend `SimulatorType`, and `SimulatorConfig`. Also tracks diagnostic metrics (`qubits_before`, `qubits_after`, `applied_transforms`).
* **`ITransform.h` (`ITransform`)**:
  Abstract base class for all pre-execution passes:
  - `Name()`: Returns the string identifier of the pass (e.g. `"CausalCone"`).
  - `IsApplicable(const TransformContext &ctx)`: Predicate checking whether the transform should run based on configuration, circuit operations, or backend compatibility.
  - `Apply(TransformContext &ctx)`: Executes the transformation on the context in-place.
* **`TransformPipeline.h` / `TransformPipeline.cpp` (`TransformPipeline`)**:
  Orchestrator that executes applicable passes in deterministic order. Instantiates per-run transform objects to ensure thread safety across concurrent calls.
* **`Export.h`**:
  Manages cross-platform shared library symbol exports (`MAESTRO_TRANSFORM_API`).

---

## Active Transforms

1. **`CausalConeTransform` (`"CausalCone"`)**:
   - Traces the union of requested observable supports backward through unitary operations in reverse execution order.
   - Discards unitary gates outside the backward lightcone (`G† O G = O`).
   - Remaps retained operations and Pauli strings to compact indices.
   - Conservatively falls back to full execution for non-unitary operations (measurements, resets, channels) and distributed GPU layouts.

---

## Adding a New Transform

To add a new transformation (e.g., `SymmetryReductionTransform`):

1. **Create the transform class**:
   Inherit from `ITransform` in `Execution/Transforms/`:
   ```cpp
   class MAESTRO_TRANSFORM_API MyTransform : public ITransform
   {
     public:
       std::string_view Name() const override { return "MyTransform"; }
       bool IsApplicable(const TransformContext &ctx) const override;
       void Apply(TransformContext &ctx) override;
   };
   ```

2. **Register in `TransformPipeline.cpp`**:
   Add the transform to the ordered `transforms` array in `TransformPipeline::Run`:
   ```cpp
   MyTransform my_transform;
   CausalConeTransform causal_cone;
   const std::array<ITransform *, 2> transforms{&my_transform, &causal_cone};
   ```

3. **Add files to `CMakeLists.txt`**:
   Add the new `.h` and `.cpp` files to `MAESTROSRC`.

4. **Add unit tests**:
   Add tests in `tests/` verifying pass behavior and pipeline diagnostics (`ctx.qubits_after`, `ctx.applied_transforms`).

## Automatic parity reduction

`ParityReduction` runs before causal-cone reduction in Python expectation APIs.
It is enabled by default (`SimulatorConfig.auto_reduce = true`). Circuits whose
initial operations do not match bail out in nanoseconds via fast gate signature
checks; when rejection occurs later in a long circuit, the time required to fall
back depends on the length of the scanned prefix.
A successful reduction executes the projected effective circuit using the caller's
configured simulator backend and options.

The detector recognizes a uniform `H` preparation of `|+>^N`, CNOT parity
ladders containing `Rz` checks, and physical `Rx` mixers after each ladder has
been uncomputed. Both star and chain ladders work. It also accepts the global
Hadamard dual: `|0>^N`, X checks and Z mixers. All CNOT frames must be uncomputed
at the end. Other gates, partial product preparations, measurements, resets,
channels, classical control and distributed GPU layouts conservatively fall
back. Noisy estimation keeps the requested backend for its noisy realizations.
This pass does not discover arbitrary Clifford stabilizer symmetries.

Binary row elimination selects independent *original* checks `B_j`. The effective
basis is `|b> = product_j Z(B_j)^b_j |+>^N`. Its dimension is `2^rank(B)`:

- An independent check becomes `X_j`; a dependent check becomes a product of Xs
  determined by its GF(2) coordinates.
- Physical `X_q` becomes the diagonal product of `Z_j` for checks containing q.
- Pauli observables are projected with their signs preserved. An observable
  connecting to another check coset has expectation zero.

The rank here counts check generators, **not** conserved symmetry generators
(which would leave `N - rank` qubits). Rank alone does not establish invariance:
the preparation and all operations must also pass the detector. Effective
rotations are emitted as ordinary Maestro gates. `TransformContext` carries
observable factors and `auto_reduced`; callers must apply both. The pipeline
resets these outputs on each run, and never mutates the source circuit.

This implementation supports up to 64 physical qubits and rank 1 through 22,
strictly below the physical width. Larger registers/ranks fall back, matching
the reference adapter's 22-qubit allocation cap (64 MiB for effective amplitudes).
Statevector-return and shot APIs retain their physical-register contracts and
are not reduced. General stabilizer state preparation and arbitrary tapering
Cliffords are outside this detector's scope.

Python estimate results report `auto_reduced`, `qubits_before`, `qubits_after`,
and `reduction_time` in seconds (including causal-cone processing). Native
estimate APIs do not currently report these diagnostic fields. Python `time_taken`
includes the preflight pass and execution, following the existing estimate timing
convention; network setup is excluded. Performance depends on circuit depth and
hardware.
`parity_reduction_tests` prints the 37-to-18 preflight time, including effective
circuit emission, without imposing a flaky wall-clock assertion.

The check-subspace construction follows `evolve_check_subspace` in the local
qec-primitives-benchmark adapter. Related background:
[Bravyi et al., tapering off qubits (2017)](https://arxiv.org/abs/1701.08213) and
[Farhi et al., QAOA (2014)](https://arxiv.org/abs/1411.4028).

Local Release-build validation on the distance-7, 37-qubit colour code from
`qecbench.codes.color_code(7)` (18 independent checks, linear-ramp QAOA,
`delta=0.5`) gave the following representative timings. Wall time is the minimum
of three complete Python estimate calls; these are measurements, not guarantees.

| QAOA depth | Preflight including causal cone | Estimate wall time | Energy error vs NumPy reference |
| --- | --- | --- | --- |
| 1 | 0.078 ms | 9.3 ms | 3.0e-14 |
| 3 | 0.186 ms | 22.8 ms | 1.8e-15 |
| 10 | 0.564 ms | 69.9 ms | 7.1e-15 |

`tests/python/test_parity_reduction.py` retains the colour-code checks and
reference energies so the numerical comparison is reproducible without the
external benchmark package.
