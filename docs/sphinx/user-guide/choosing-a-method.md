---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Choosing a simulation method

Pick the simulation method that fits your circuit, then select it with one
`SimulatorConfig` field.

```{code-cell} python
import maestro
from maestro.circuits import QuantumCircuit

ghz = QuantumCircuit()
ghz.h(0)
for q in range(99):
    ghz.cx(q, q + 1)
ghz.measure_all()

config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.MatrixProductState)
counts = ghz.execute(shots=1000, config=config)["counts"]
print({bits[:5] + "…": n for bits, n in counts.items()})
```

A 100-qubit statevector would need 2¹⁰⁰ amplitudes. The MPS above handles it
because a GHZ state has little entanglement. The same circuit runs on any
method that supports its gates.

(py_decision_table)=
## Decision table

Every row is backed by a measurement further down this page, run when these docs
were built.

| Problem shape | Method | Runs on | Caveat |
|---------------|--------|---------|--------|
| Any gates, up to about 30 qubits; shots or expectation values | `Statevector` | CPU; GPU¹ | Time and memory double with every qubit: 16 GiB at 30 qubits in double precision. {ref}`Measured <evidence-statevector>`. |
| Many qubits with limited entanglement, such as shallow or 1D circuits; shots or expectation values | `MatrixProductState` | CPU; GPU¹ | Approximate once the bond dimension truncates. Check with mirror fidelity. {ref}`Measured <evidence-mps>`. |
| Clifford gates only, any number of qubits; shots | `Stabilizer` | CPU | Raises an error on any non-Clifford gate. {ref}`Measured <evidence-stabilizer>`. |
| Mostly Clifford with a few T gates; shots | `ExtendedStabilizer` | CPU | Approximate sampling. {ref}`Measured <evidence-extended-stabilizer>`. |
| Expectation values on circuits too wide for a statevector; no shots | `PauliPropagator` | CPU; GPU¹ | Truncation is set by the `pp_*` options. {ref}`Measured <evidence-pauli-propagation>`. |
| The exact mixed state under noise, with no sampling error, for any channel (amplitude damping, Kraus channels); about 10 qubits or fewer | `DensityMatrix`, or `MatrixProductOperator` for more qubits | CPU; GPU¹ | Density-matrix cost grows steeply with qubit count. `Statevector` and `MatrixProductState` sample the same channels, one trajectory per shot (see {ref}`py_noise_sampled_channels`). {ref}`Measured <evidence-density-matrix>`. |
| Error-correction circuits written in Stim, decoded with Sinter | `maestro.sinter` samplers | CPU | Clifford circuits with Pauli noise use native reference frames; other circuits and noise models run as trajectories. {ref}`Measured <evidence-qec>`. |

¹ Requires GPU access; see {doc}`/getting-started/gpu-access`. GPU support per
method follows the {ref}`compatibility table <py_backend_table>`.

% The GPU column comes from Simulators/Core/FactoryGpu.cpp (the methods Maestro
% routes to the GPU plugin), not from a run: the docs build has no GPU.
% TODO(engineering): rows removed for lack of a runnable measurement:
% DistributedGpu/DistributedMpiGpu (statevectors larger than one GPU), QuEST
% (checked by hand against a local libmaestroquest build, but the docs build has
% no plugin), TensorNetwork and PathIntegral. Add each back with a measured example.

(py_key_options)=
## The options you'll actually set

| Option | What it does |
|--------|--------------|
| `simulation_type` | The method: a `SimulationType` such as `Statevector` or `MatrixProductState` |
| `simulator_type` | The backend: `QCSim` (default, CPU), `QiskitAer`, or `Gpu` with GPU access |
| `max_bond_dimension` | Largest MPS/MPO bond dimension; higher is more exact and slower |
| `singular_value_threshold` | MPS/MPO truncation threshold, read according to `truncation_mode` |
| `pp_max_pauli_weight`, `pp_coefficient_threshold` | Pauli propagation truncation |
| `seed` | Makes sampling reproducible |
| `gpu_device` | Which CUDA device to use (GPU access only) |

Every option is keyword-only in the constructor, and fields can also be set
after construction. {ref}`All SimulatorConfig options <simulatorconfig-options>`
lists the rest.

## The measurements behind this table

Timings come from the machine that built these docs. Compare the trends, not the
absolute numbers. The measurements share two helpers: a random brickwork circuit
and a timer.

(evidence-statevector)=
### Statevector

Simulation time with every reduction off, so the full register is simulated:

```{code-cell} python
import math
import random
import time


def brickwork(num_qubits, depth, seed=1):
    """Random rotations and alternating CNOT layers."""
    rng = random.Random(seed)
    circuit = QuantumCircuit()
    for layer in range(depth):
        for q in range(num_qubits):
            circuit.ry(q, rng.uniform(0, math.pi))
            circuit.rz(q, rng.uniform(0, math.pi))
        for q in range(layer % 2, num_qubits - 1, 2):
            circuit.cx(q, q + 1)
    return circuit


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


no_reductions = dict(
    enable_causal_cone_reduction=False, enable_observable_clustering=False, auto_reduce=False
)

config = maestro.SimulatorConfig(**no_reductions)
for n in [16, 20, 24]:
    circuit = brickwork(n, 6)
    _, seconds = timed(lambda: circuit.estimate(["Z" + "I" * (n - 1)], config=config))
    print(f"{n} qubits: {seconds:.3f} s, {16 * 2**n / 2**20:.0f} MiB of amplitudes")
```

(evidence-mps)=
### Matrix product state

Mirror fidelity is 1.0 for an exact simulation. On a 20-qubit, depth-12
brickwork circuit, a small bond dimension visibly truncates:

```{code-cell} python
circuit = brickwork(20, 12)
for chi in [4, 16, 64]:
    config = maestro.SimulatorConfig(
        simulation_type=maestro.SimulationType.MatrixProductState, max_bond_dimension=chi
    )
    print(f"max_bond_dimension={chi}: mirror fidelity {circuit.mirror_fidelity(config=config, full_amplitude=True):.4f}")
```

(evidence-stabilizer)=
### Stabilizer

A 1,000-qubit GHZ state samples instantly, and a single T gate is rejected:

```{code-cell} python
stabilizer = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Stabilizer)

big = QuantumCircuit()
big.h(0)
for q in range(999):
    big.cx(q, q + 1)
big.measure_all()
counts, seconds = timed(lambda: big.execute(shots=1000, config=stabilizer)["counts"])
print(f"1000 qubits: {len(counts)} distinct outcomes in {seconds:.3f} s")

t_gate = QuantumCircuit()
t_gate.h(0)
t_gate.t(0)
t_gate.measure_all()
try:
    t_gate.execute(shots=10, config=stabilizer)
except RuntimeError as error:
    print("T gate:", error)
```

(evidence-extended-stabilizer)=
### Extended stabilizer

H·T·H gives P(0) = cos²(π/8) ≈ 0.854. The extended stabilizer samples it
approximately; the exact statevector is the reference:

```{code-cell} python
ht = QuantumCircuit()
ht.h(0)
ht.t(0)
ht.h(0)
ht.measure_all()
for method in ["ExtendedStabilizer", "Statevector"]:
    config = maestro.SimulatorConfig(simulation_type=getattr(maestro.SimulationType, method))
    counts = ht.execute(shots=2000, config=config)["counts"]
    print(f"{method}: P(0) = {counts.get('0', 0) / 2000:.3f}")
print(f"exact: {math.cos(math.pi / 8) ** 2:.3f}")
```

(evidence-pauli-propagation)=
### Pauli propagation

On a 60-qubit circuit, Pauli propagation agrees with an MPS reference:

```{code-cell} python
circuit = brickwork(60, 4)
observables = ["I" * 29 + "ZZ" + "I" * 29, "I" * 10 + "X" + "I" * 49]
for method in ["PauliPropagator", "MatrixProductState"]:
    config = maestro.SimulatorConfig(simulation_type=getattr(maestro.SimulationType, method))
    values, seconds = timed(lambda: circuit.estimate(observables, config=config)["expectation_values"])
    print(f"{method}: {[round(v, 6) for v in values]} in {seconds:.3f} s")
```

(evidence-density-matrix)=
### Density matrix

With depolarizing noise after every gate and reductions off, the cost grows
much faster than a statevector's:

```{code-cell} python
config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.DensityMatrix, **no_reductions)
for n in [4, 6, 8]:
    circuit = brickwork(n, 4)
    noise = maestro.NoiseModel()
    noise.set_all_depolarizing(num_qubits=n, p=0.01)
    _, seconds = timed(lambda: circuit.full_noise_estimate(["Z" + "I" * (n - 1)], noise, config=config))
    print(f"{n} qubits: {seconds:.3f} s")
```

When we measured, 10 qubits took about 8 s and 12 qubits about 4 minutes, so the
docs build stops at 8. For exact channels themselves, see {ref}`py_noise_cptp`.

% TODO(engineering): 12-qubit density matrix with depth-4 depolarizing noise
% took 259 s (10 qubits: 7.9 s) on an Apple Silicon laptop. Is that expected?

(evidence-qec)=
### Error correction

{doc}`qec-sinter` samples a noisy repetition code through Maestro and compares
the detection-event rate with Stim's own sampler.

(py_backends_overview)=
## Backends and methods

`simulator_type` picks the engine and `simulation_type` picks the method. The
default is QCSim, Maestro's CPU engine, with a statevector.

Gate fusion is on by default for supported QCSim and GPU methods, except CPU
statevectors below 11 qubits and CPU density matrices below 5, where it costs
more than it saves. `gate_fusion=False` or `True` forces it. MPS, MPO and CPU
tensor-network fusion is limited to two qubits; QCSim statevector, density
matrix and composite statevector fusion can use three. Fusing MPS/MPO gates
changes where truncation happens, so results at a finite bond dimension or a
nonzero cutoff can change.

(py_statevector)=
### Statevector

A full 2ⁿ-amplitude statevector, exact for any gate set.

(py_mps)=
### Matrix product state

MPS simulates many qubits when entanglement is bounded:

```{code-cell} python
mps_config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
    singular_value_threshold=1e-10,
    lookahead_depth=3,
)
result = ghz.execute(shots=1000, config=mps_config)
print(result["method"])
```

(py_mps_truncation_mode)=
#### Truncation mode

`singular_value_threshold` is read according to `truncation_mode`:

- `"discarded_weight"` (the default on every backend) discards the smallest
  singular values until their cumulative squared weight, normalized by the
  total, would reach the threshold. Qiskit Aer and ITensor use the same
  convention.
- `"relative_max"` discards singular values below `threshold * sigma_max`, the
  largest singular value at that bond. This was the QCSim and GPU default before
  `discarded_weight`.

Only QCSim and the GPU backend support `"relative_max"`. The Aer backend raises
`ValueError` if you request it.

```{code-cell} python
relative = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    singular_value_threshold=1e-10,
    truncation_mode="relative_max",
)
counts = ghz.execute(shots=100, config=relative)["counts"]
print({bits[:5] + "…": n for bits, n in counts.items()})
```

(py_mps_gpu_svd)=
#### GPU SVD algorithm

:::{admonition} Requires GPU access
:class: note
These options only affect the GPU backend. See {doc}`/getting-started/gpu-access`.
:::

For GPU MPS, MPO and tensor-network simulations, `mps_svd_solver`,
`mpo_svd_solver` and `tensor_network_svd_solver` select the cuSOLVER SVD
algorithm: `"gesvd"` (QR-based, the plugin's default), `"gesvdj"` (Jacobi, good
for small to medium bond dimensions), `"gesvdp"` (polar decomposition) or
`"gesvdr"` (randomized, for fixed-rank truncation; tensor networks fall back to
GESVD). `None` leaves the choice to the installed plugin.

```python
# Not executed in the docs build: requires GPU access.
gpu_config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.MatrixProductState,
    mps_svd_solver="gesvdj",
)
```

(py_stabilizer)=
### Stabilizer

Polynomial-time simulation of circuits made only of Clifford gates (H, S, CX,
CZ, X, Y, Z). See the {ref}`measurement above <evidence-stabilizer>`.

(py_backend_table)=
### Compatibility table

Which `SimulationType` each backend supports:

| SimulationType | QCSim | Qiskit Aer | GPU¹ | DistributedGpu¹ | DistributedMpiGpu¹ | QuEST² |
|----------------|-------|------------|------|-----------------|--------------------|--------|
| `Statevector` | Yes | Yes | Yes | Yes | Yes | Yes |
| `MatrixProductState` | Yes | Yes | Yes | No | No | No |
| `MatrixProductOperator` | Yes | No | Yes | No | No | No |
| `DensityMatrix` | Yes | Yes | Yes | No | No | No |
| `Stabilizer` | Yes | Yes | No | No | No | No |
| `TensorNetwork` | Yes | Yes | Yes | No | No | No |
| `PauliPropagator` | Yes | No | Yes | No | No | No |
| `PathIntegral` | Yes | No | No | No | No | No |
| `ExtendedStabilizer` | Yes | Yes | No | No | No | No |

¹ Requires GPU access; see {doc}`/getting-started/gpu-access`.
² Requires the QuEST plugin, which is not part of the pip package; see {ref}`py_quest`.

The QCSim and Aer columns were checked by running each method on both backends.
The GPU and QuEST columns follow the methods Maestro's source routes to those
plugins.

`CompositeQCSim` and `CompositeQiskitAer` run partitioned (p-block) circuits on
the CPU. For statevectors spread across GPUs or MPI ranks, see {doc}`scaling`.

(py_quest)=
### QuEST

:::{admonition} Requires the QuEST plugin
:class: note
`SimulatorType.QuestSim` loads the QuEST plugin at run time. It isn't included
in the pip package. It is open source (GPL-3.0): build it from
[maestro-quest-interface](https://github.com/QoroQuantum/maestro-quest-interface)
with CMake, which fetches QuEST itself. It builds on Linux, macOS and Windows,
and has MPI and GPU build options for clusters. Put the resulting
`libmaestroquest.so` (`.dylib` on macOS, `maestroquest.dll` on Windows) on the
library search path.
:::

`init_quest()` loads the plugin and returns `True` on success.
`is_quest_available()` only reports `True` after that call, so call
`init_quest()` first:

```python
# Not executed in the docs build: requires the QuEST plugin.
if not maestro.init_quest():
    raise RuntimeError("libmaestroquest was not found on the library search path")

quest_config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QuestSim,
    simulation_type=maestro.SimulationType.Statevector,
)
result = ghz.execute(shots=1024, config=quest_config)
```

QuEST supports only `SimulationType.Statevector`; other methods raise an
exception.

(py_advanced)=
## Low-level simulator control

For full control of the simulation lifecycle, drive a simulator through the
`Maestro` engine directly:

```{code-cell} python
engine = maestro.Maestro()
handle = engine.create_simulator(maestro.SimulatorType.QCSim, maestro.SimulationType.Statevector)

sim = engine.get_simulator(handle)
sim.AllocateQubits(2)
sim.InitializeSimulator()
sim.ApplyH(0)
sim.ApplyCX(0, 1)
print(sim.SampleCounts([0, 1], shots=1024))  # integer outcomes: 0 is |00>, 3 is |11>

engine.destroy_simulator(handle)
```

These methods act on the simulator directly, without the network that
`simple_execute` and `QuantumCircuit.execute` use. Use the returned object only
while its handle is alive.

`Simulator` exposes the operations of `maestrolib/Interface.h`:

| Category | Methods |
|----------|---------|
| Lifecycle | `AllocateQubits`, `GetNumberOfQubits`, `InitializeSimulator`, `ResetSimulator`, `ClearSimulator`, `FlushSimulator` |
| Configuration | `ConfigureSimulator`, `GetConfiguration`, `GetSimulatorType`, `GetSimulationType`, `SetMultithreading`, `GetMultithreading`, `IsQcsim` |
| Measurement and state queries | `Measure`, `MeasureNoCollapse`, `ApplyReset`, `Probability`, `Amplitude`, `AllProbabilities`, `Probabilities`, `SampleCounts` |
| Saved state | `SaveState`, `RestoreState`, `SaveStateToInternalDestructive`, `RestoreInternalDestructiveSavedState` |
| Single-qubit gates | `ApplyX`, `ApplyY`, `ApplyZ`, `ApplyH`, `ApplyS`, `ApplySDG`, `ApplyT`, `ApplyTDG`, `ApplySX`, `ApplySXDG`, `ApplyK`, `ApplyP`, `ApplyRx`, `ApplyRy`, `ApplyRz`, `ApplyU` |
| Controlled gates | `ApplyCX`, `ApplyCY`, `ApplyCZ`, `ApplyCH`, `ApplyCSX`, `ApplyCSXDG`, `ApplyCP`, `ApplyCRx`, `ApplyCRy`, `ApplyCRz`, `ApplyCU`, `ApplyCCX` |
| Swaps | `ApplySwap`, `ApplyCSwap` |

`Initialize`, `Reset`, `Configure`, `Clear` and `Flush` are shorter aliases for
the methods ending in `Simulator`. Pass Python sequences for qubits or outcomes.
`Amplitude` returns a complex number, `AllProbabilities` and `Probabilities`
return lists, and `SampleCounts` returns a dictionary from integer outcomes to
counts (`shots=1000` by default). `Probabilities` takes basis-state indices,
while `Measure`, `ApplyReset` and `SampleCounts` take qubit indices. Packed
outcomes use the first listed qubit as the least significant bit; for
`MeasureNoCollapse`, qubit 0 is the least significant bit. `ApplyU` and
`ApplyCU` take `theta`, `phi`, `lambda_` and an optional `gamma=0.0`.

The destructive save/restore pair supports internal-state sampling on Aer:
flush pending gates, save, call `MeasureNoCollapse`, then restore before
applying more gates.
