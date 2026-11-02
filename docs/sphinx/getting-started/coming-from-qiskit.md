---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_coming_from_qiskit)=
# Coming from Qiskit Aer

Run existing Qiskit code on Maestro by changing an import. The qiskit-maestro
package gives Qiskit a Maestro backend, Sampler and Estimator:

```bash
pip install qiskit-maestro
```

| Qiskit | qiskit-maestro |
|---|---|
| `AerSimulator()` | `MaestroBackend()` |
| `StatevectorSampler()` or Aer's `SamplerV2()` | `MaestroSampler()` |
| `StatevectorEstimator()` or Aer's `EstimatorV2()` | `MaestroEstimator()` |

```{code-cell} python
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_maestro import MaestroBackend, MaestroEstimator, MaestroSampler

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
measured = qc.measure_all(inplace=False)

backend = MaestroBackend(seed_simulator=7)  # was AerSimulator(seed_simulator=7)
print(backend.run(transpile(measured, backend), shots=1000).result().get_counts())

sampler = MaestroSampler(seed=7)  # was StatevectorSampler(seed=7)
print(sampler.run([measured], shots=1000).result()[0].data.meas.get_counts())

estimator = MaestroEstimator()  # was StatevectorEstimator()
observables = [
    SparsePauliOp("IIZ"),
    SparsePauliOp("XXX"),
    SparsePauliOp(["IZZ", "XXX"], coeffs=[0.5, 0.3]),
]
print(estimator.run([(qc, observables)]).result()[0].data.evs)
```

The plugin is a separate package,
[qiskit-maestro](https://github.com/QoroQuantum/qiskit-maestro), which installs
`qoro-maestro` for you. It needs Python 3.10 or later and works with Qiskit 1.1
and later, including 2.x. Everything stays in Qiskit's conventions: bit order,
`SparsePauliOp` observables with coefficients, classical registers, PUBs and
broadcasting. `transpile` and `generate_preset_pass_manager` target Maestro's
native gates, and Qiskit's `BackendSamplerV2` works with `MaestroBackend`.

If you call Maestro directly rather than through Qiskit,
{ref}`qiskit-translate` maps Aer code onto Maestro's own API.

## Parameter sets and VQE

Parameterised circuits take parameter values as in any `EstimatorV2`, with
Qiskit's broadcasting rules. qiskit-maestro converts the circuit once and runs
the parameter sets in parallel, so scanning a VQE ansatz is one call:

```{code-cell} python
import numpy as np
from qiskit.circuit.library import efficient_su2

n = 6
ansatz = efficient_su2(n, reps=2, entanglement="linear")
hamiltonian = SparsePauliOp.from_sparse_list(
    [("ZZ", [q, q + 1], -1.0) for q in range(n - 1)] + [("X", [q], -0.5) for q in range(n)],
    num_qubits=n,
)
parameter_values = np.random.default_rng(0).uniform(0, np.pi, (8, ansatz.num_parameters))
result = estimator.run([(ansatz, hamiltonian, parameter_values)]).result()
print(result[0].data.evs)  # One energy per parameter set.
```

The [example notebook](https://github.com/QoroQuantum/qiskit-maestro/blob/main/examples/vqe_aer_vs_maestro.ipynb)
runs a whole VQE loop on a 100-qubit Ising chain on Aer and on Maestro, both
with matrix product states.

## Choose the simulation method

Options take Aer's names, and Aer's longer MPS option names are accepted too.
An MPS handles many qubits when entanglement is limited:

```{code-cell} python
mps_estimator = MaestroEstimator(
    method="matrix_product_state",
    matrix_product_state_max_bond_dimension=64,
    matrix_product_state_truncation_threshold=1e-10,
)
wide = QuantumCircuit(120)
wide.h(0)
for q in range(119):
    wide.cx(q, q + 1)
zz = SparsePauliOp.from_sparse_list([("ZZ", [0, 119], 1.0)], 120)
print(mps_estimator.run([(wide, zz)]).result()[0].data.evs)
```

| Option | Values |
|---|---|
| `method` | `"statevector"` (default), `"matrix_product_state"` (or `"mps"`), `"stabilizer"`, `"density_matrix"`, `"tensor_network"`, `"pauli_propagation"` |
| `device` | `"CPU"` (default), or `"GPU"` with {doc}`GPU access <gpu-access>` |
| `max_bond_dimension`, `truncation_threshold` | MPS truncation; Aer's `matrix_product_state_*` names work too |
| `precision` | `"double"` (default) or `"single"` |
| `seed_simulator` (`seed` on the Sampler and Estimator) | Reproducible results |
| `max_parallel_experiments` | Threads for circuits and parameter sets; `0`, the default, chooses |
| `maestro_options` | A dictionary of any other {doc}`SimulatorConfig </api/configuration>` field |

Pass the options to `MaestroBackend`, `MaestroSampler` or `MaestroEstimator`.
The primitives also accept the `options={"backend_options": ...}` dictionary of
Aer's primitives unchanged. Maestro may run a circuit on a faster engine than
the one you asked for, such as a Clifford circuit on the stabilizer engine;
results are unaffected, and each result's metadata records what ran.

How MPS expectation values compare with Aer's on 100+ qubits is on the
{ref}`benchmarks page <bench-mps-estimator>`: 6 to 7.5 times faster for a VQE
energy evaluation.

## Time evolution

When the PUBs of one call are one circuit per time point of the same Trotter
step, `MaestroEstimator` simulates the longest circuit once with
{ref}`incremental evolution <py_incremental_evolve>` and reads every time point
on the way. Measuring after each of N steps then costs N steps instead of about
N²/2. `MaestroEstimator.run_evolution` takes the initial circuit and the step
directly:

```{code-cell} python
n = 40
step = QuantumCircuit(n)  # One Trotter step of an Ising chain.
for start in (0, 1):
    for q in range(start, n - 1, 2):
        step.rzz(-0.2, q, q + 1)
step.rx(-0.14, range(n))
magnetisation = SparsePauliOp.from_sparse_list([("Z", [q], 1 / n) for q in range(n)], n)

# One circuit per time point, as for any estimator.
circuits, circuit = [], QuantumCircuit(n)
for k in range(11):
    circuits.append(circuit.copy())
    circuit.compose(step, inplace=True)
results = mps_estimator.run([(c, magnetisation) for c in circuits]).result()
print("incremental:", results[0].metadata["maestro"]["incremental_evolution"])
print(np.array([r.data.evs for r in results]).round(4))

# The same evolution, given as an initial circuit and a step.
evolution = mps_estimator.run_evolution(QuantumCircuit(n), step, magnetisation, steps=range(11))
print(evolution.result()[0].data.evs.round(4))
```

`PauliEvolutionGate` circuits synthesised with `LieTrotter(reps=k)` are
recognised too. On the benchmarks page, measuring every step this way is 4.9
times faster than Aer at 120 qubits and 7 times faster at 200. Incremental
evolution isn't used for PUBs with arrays of parameter values or for circuits
that reset qubits, and `incremental_evolution=False` turns it off.

## Run on a GPU

:::{admonition} Requires GPU access
:class: note
`device="GPU"` needs the GPU plugin and a licence. See {doc}`gpu-access`.
:::

```python
# Not executed in the docs build: requires GPU access.
gpu_backend = MaestroBackend(device="GPU")
gpu_mps = MaestroEstimator(device="GPU", method="matrix_product_state", max_bond_dimension=256)
```

The GPU runs the statevector, MPS, tensor network and Pauli propagation
methods. Maestro reads the licence key from `MAESTRO_LICENSE_KEY` itself. If no
GPU is available or the licence isn't accepted, qiskit-maestro raises
`MaestroGpuError` rather than running on the CPU.

## What isn't supported yet

qiskit-maestro doesn't yet translate Aer noise models, `for` and `while` loops,
`switch`, or Aer's `save_*` instructions, and Estimator circuits can't contain
measurements, as in `StatevectorEstimator`. Its
[README](https://github.com/QoroQuantum/qiskit-maestro#whats-supported-and-what-isnt-yet)
lists everything that isn't supported yet.

(qiskit-translate)=
## Translate to Maestro's API

The rest of this page maps Aer code onto Maestro's own API, for code that calls
Maestro directly.

(qiskit-bit-order)=
:::{warning}
Maestro puts qubit 0 **first**, on the left, in bitstrings and Pauli strings.
Qiskit puts it last, on the right. Flipping only qubit 0 of three reads `"100"`
in Maestro and `"001"` in Qiskit, and Z on qubit 0 is `"ZII"` in Maestro and
`"IIZ"` in a Qiskit `SparsePauliOp`.
:::

```{code-cell} python
from maestro.circuits import QuantumCircuit

flip = QuantumCircuit()
flip.x(0)
flip.measure([(0, 0), (1, 1), (2, 2)])
counts = flip.execute(shots=10)["counts"]
print("Maestro order:", counts)
print("Qiskit order: ", {bits[::-1]: n for bits, n in counts.items()})
```

Each section below shows the Aer code next to its Maestro equivalent. The Aer
code was checked against Qiskit 2.6 and Qiskit Aer 0.18 but isn't run in the docs
build; the Maestro code is.

(qiskit-statevector)=
### Statevector

`AerSimulator(method="statevector")` with `save_statevector()` becomes `get_statevector()`. Both use the same amplitude order, with qubit 0 as the least significant bit of the index.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
**Qiskit Aer**

```{literalinclude} qiskit/statevector_aer.py
:language: python
```
:::

:::{grid-item}
**Maestro**

```{literalinclude} qiskit/statevector_maestro.py
:language: python
```
:::
::::

Output of the Maestro code, run when these docs were built:

```{code-cell} python
:tags: [remove-input]

%run qiskit/statevector_maestro.py
```

(qiskit-mps)=
### Matrix product state

Aer's `matrix_product_state_*` options become `SimulatorConfig` fields. Both default to the discarded-weight truncation convention, so thresholds mean the same thing.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
**Qiskit Aer**

```{literalinclude} qiskit/mps_aer.py
:language: python
```
:::

:::{grid-item}
**Maestro**

```{literalinclude} qiskit/mps_maestro.py
:language: python
```
:::
::::

Output of the Maestro code, run when these docs were built:

```{code-cell} python
:tags: [remove-input]

%run qiskit/mps_maestro.py
```

(qiskit-shots)=
### Shots and counts

Shots go to the call in both. Maestro needs no `transpile` step, and `seed_simulator` becomes `seed`. The counts below are the same Bell outcomes, but compare {ref}`bit order <qiskit-bit-order>` before comparing longer bitstrings.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
**Qiskit Aer**

```{literalinclude} qiskit/shots_aer.py
:language: python
```
:::

:::{grid-item}
**Maestro**

```{literalinclude} qiskit/shots_maestro.py
:language: python
```
:::
::::

Output of the Maestro code, run when these docs were built:

```{code-cell} python
:tags: [remove-input]

%run qiskit/shots_maestro.py
```

(qiskit-estimator)=
### Expectation values (Estimator)

`EstimatorV2` becomes `estimate`. Pauli labels are reversed, and a weighted sum such as a `SparsePauliOp` with coefficients is estimated term by term.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
**Qiskit Aer**

```{literalinclude} qiskit/estimator_aer.py
:language: python
```
:::

:::{grid-item}
**Maestro**

```{literalinclude} qiskit/estimator_maestro.py
:language: python
```
:::
::::

Output of the Maestro code, run when these docs were built:

```{code-cell} python
:tags: [remove-input]

%run qiskit/estimator_maestro.py
```

(qiskit-noise)=
### Noise models

Aer attaches errors to gate names; Maestro attaches them to qubits, after every gate that touches the qubit. With a Pauli channel on both qubits of each CX, the two models below are the same.

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item}
**Qiskit Aer**

```{literalinclude} qiskit/noise_aer.py
:language: python
```
:::

:::{grid-item}
**Maestro**

```{literalinclude} qiskit/noise_maestro.py
:language: python
```
:::
::::

Output of the Maestro code, run when these docs were built:

```{code-cell} python
:tags: [remove-input]

%run qiskit/noise_maestro.py
```

### Run an existing Qiskit circuit

`MaestroBackend` and the qiskit-maestro primitives take Qiskit circuits as they
are. Without the package, export the circuit
to OpenQASM 2.0 with `qiskit.qasm2.dumps(qc)` and run the text with
`maestro.simple_execute`. Qiskit's
`measure_all()` output, barrier included, runs as is:

```{code-cell} python
import maestro

exported = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg meas[3];
h q[0];
cx q[0],q[1];
rz(0.3) q[2];
barrier q[0],q[1],q[2];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
"""  # qasm2.dumps() output for a Qiskit circuit
print(maestro.simple_execute(exported, shots=1000)["counts"])
```

Qiskit would print the `"110"` outcome above as `"011"`.

(qiskit-options)=
### Simulator options

| Qiskit Aer | Maestro `SimulatorConfig` |
|------------|---------------------------|
| `method="statevector"` | `simulation_type=SimulationType.Statevector` |
| `method="matrix_product_state"` | `simulation_type=SimulationType.MatrixProductState` |
| `method="stabilizer"` | `simulation_type=SimulationType.Stabilizer` |
| `method="extended_stabilizer"` | `simulation_type=SimulationType.ExtendedStabilizer`, with `simulator_type=SimulatorType.QiskitAer` |
| `method="density_matrix"` | `simulation_type=SimulationType.DensityMatrix` |
| `matrix_product_state_max_bond_dimension` | `max_bond_dimension` |
| `matrix_product_state_truncation_threshold` | `singular_value_threshold` |
| `precision` | `precision` |
| `seed_simulator` | `seed` |
| `device="GPU"` | `simulator_type=SimulatorType.Gpu`, which requires {doc}`GPU access <gpu-access>` |

To keep Aer's engine while using Maestro's interface, set
`simulator_type=SimulatorType.QiskitAer`.

(qiskit-differences)=
### What doesn't map directly

These differences apply to Maestro's own API. qiskit-maestro takes care of the
bit order, weighted observables, primitives, registers and `if_test` for you.

- **Bit and qubit order.** Counts and Pauli labels are reversed; see the warning
  at the top of this page.
- **Weighted observables.** `estimate` takes Pauli strings only. Estimate each
  term of a `SparsePauliOp` and weight them yourself, as in
  {ref}`qiskit-estimator`. `AdjointProgram` does accept `(coefficient, string)`
  terms.
- **Statevector width.** `get_statevector` leaves out qubits that no gate
  touches, so a 3-qubit circuit that never uses qubit 1 returns 4 amplitudes, and
  qubit 2 becomes bit 1 of the index.
- **Depolarizing strength.** Maestro's `p` is the total Pauli error probability,
  p/3 each for X, Y and Z. Aer's `depolarizing_error(λ, 1)` corresponds to
  `p = 3λ/4`.
- **Noise sampling in `noisy_execute`.** Aer draws noise for every shot.
  `noisy_execute` draws `noise_realizations` noise samples (64 by default) and
  splits the shots among them, so counts vary more between runs unless you set
  `noise_realizations=shots`.
- **No primitives or `save_*` instructions.** Call `execute`, `estimate` or
  `get_statevector` on the circuit. Results are plain dictionaries; see
  {ref}`py_result_dict`.
- **No registers.** `measure` takes (qubit, classical bit) pairs, and counts are
  single bitstrings with no spaces between registers.
- **Classical control.** `QuantumCircuit` has no `if_test` or `c_if`. Write
  feed-forward in OpenQASM 2.0 and run it with `simple_execute`; see
  {ref}`py_mid_circuit`.
