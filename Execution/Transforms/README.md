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
