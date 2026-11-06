---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_sampling)=
# Sampling and shots

Run a circuit for a number of shots and read the measurement counts.

```{code-cell} python
import maestro
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure([(0, 0), (1, 1)])

result = qc.execute(shots=1000)
print(result["counts"])
```

`counts` maps each measured bitstring to the number of shots that produced it.

(py_bit_order)=
:::{warning}
Maestro writes classical bit 0 first, on the **left**. Qiskit writes it last, on
the right. The same outcome reads `"100"` in Maestro and `"001"` in Qiskit.
:::

```{code-cell} python
flip = QuantumCircuit()
flip.x(0)  # flip only qubit 0 of three
flip.measure([(0, 0), (1, 1), (2, 2)])

counts = flip.execute(shots=10)["counts"]
print("Maestro order:", counts)
print("Qiskit order: ", {bits[::-1]: n for bits, n in counts.items()})
```

(py_result_dict)=
## The result dictionary

`execute` returns a dictionary:

```{code-cell} python
print(result)
```

| Key | Type | Meaning |
|-----|------|---------|
| `counts` | `dict[str, int]` | Bitstring to number of shots |
| `time_taken` | `float` | Wall-clock execution time in seconds |
| `simulator` | `SimulatorType` | Backend that ran the circuit |
| `method` | `SimulationType` | Simulation method that ran it |
| `sampling_components` | `int` | Number of independent parts sampled separately; 1 unless `enable_sampling_reduction=True` splits the circuit |
| `largest_component_qubits` | `int` | Qubits in the largest of those parts |

`simulator` and `method` are enum members, so you can compare them directly:

```{code-cell} python
print(result["simulator"], result["method"])
print(result["method"] == maestro.SimulationType.Statevector)
```

They also compare equal to the integers that releases up to 0.3.10 returned, so
existing checks such as `result["method"] == 0` keep working.

(py_raw_samples)=
## Raw shot records

Pass `return_samples=True` to get each shot's classical record instead of a
histogram. The result then has `samples`, a `numpy.uint8` array of shape
`(shots, clbits)`, in place of `counts`; column `k` is classical bit `k`, the
same position as character `k` of a count key. `execute`, `noisy_execute`,
`coherent_execute`, `full_noise_execute` and
`PrefixCheckpointedSimulator.execute_suffix` all accept it.

```{code-cell} python
bell = QuantumCircuit()
bell.h(0)
bell.cx(0, 1)
bell.measure_all()
rows = bell.execute(shots=5, return_samples=True)
print(rows["samples"], rows["sampling_strategy"])
```

`sampling_strategy` says how the shots were drawn:

| Value | Meaning |
|-------|---------|
| `reference_frame_v1` | Noisy-Clifford reference frames: one stabilizer simulation, then Pauli frames per shot |
| `terminal_v1` | One prepared state sampled for every shot (terminal measurements only) |
| `trajectory_v1` | Each shot, or each noise realization, executed on its own |

Rows keep each shot's record together. Frames, direct trajectories and
checkpoints also keep shot order; MPS, MPO and tensor networks, and every
method with `enable_sampling_reduction=True`, sample a histogram and shuffle
complete records within each noise realization, so their rows are exchangeable
rather than chronological.

(py_reference_frame_sampler)=
### Reusing compiled reference frames

When the same noisy Clifford circuit is sampled batch after batch, as in a
decoding loop, compile it once. `compile_reference_sampler` applies the
incoherent channels of a noise model as independent per-shot channels and
returns a `ReferenceFrameSampler`, or `None` when frames do not apply. The
channels must be Pauli mixtures: Pauli and depolarizing noise, phase damping,
dephasing with infinite T1 and readout errors qualify, while amplitude damping
and finite-T1 relaxation do not. Non-Clifford gates, classical control,
coherent, correlated or crosstalk noise, and a non-QCSim backend also return
`None`:

```{code-cell} python
model = maestro.NoiseModel()
model.set_bit_flip(1, 0.1)
sampler = maestro.compile_reference_sampler(bell, model)
batch = sampler.sample(1000, seed=7)["samples"]
counts = sampler.sample_counts(1000, seed=7)["counts"]
print(sampler.num_qubits, sampler.num_clbits, batch.shape, counts)
```

Each call draws new shots from the compiled plan; equal seeds give equal rows,
and `sample_counts` is the histogram of exactly the rows `sample` returns for the
same seed. The native request API, C++ (`Execution/SampleRecords.h`) and the C ABI
offer the same rows and compiled frames; see `docs/native-request-api.md`.

(py_counts_policy)=
## How counts are sampled

`SimulatorConfig.counts_policy` chooses how QCSim counts are drawn:

| Policy | Behavior |
|--------|----------|
| `"auto"` (default) | Unseeded requests that the stabilizer simulator could run use reference frames; unseeded noisy requests with one realization per shot otherwise run their realizations directly. Seeded requests keep their earlier sequences. |
| `"legacy"` | Always the earlier executor |
| `"frame_v1"` | Frames even with seeds, for any supported circuit and noise |

`sampling_policy` selects the lower-level statevector, density-matrix and
path-integral sampler, and `trajectory_policy` the random streams of dynamic
circuits (`"shot_v1"`, the default, gives counts independent of the worker
count; `"block_v1"` keeps earlier seeded counts). Setting `trajectory_policy`
explicitly also gives stabilizer, extended-stabilizer, MPS and MPO dynamic
circuits worker-independent streams.

(py_circuit_measure)=
## Measurement

`measure` takes (qubit, classical bit) pairs. `measure_all` measures every qubit
into the classical bit with the same index. `readout_error` flips each recorded
result with that probability, like Stim's `M(p)`:

```{code-cell} python
noisy_readout = QuantumCircuit()
noisy_readout.x(0)
noisy_readout.measure([(0, 0)], readout_error=0.1)
print(noisy_readout.execute(shots=1000)["counts"])  # about 10% read as 0
```

(py_mid_circuit)=
## Mid-circuit measurement and reset

Measurements can sit anywhere in the circuit. Here qubit 0 is measured after the
H gate, flipped, and measured again into a second bit, so the two bits always
disagree:

```{code-cell} python
mid = QuantumCircuit()
mid.h(0)
mid.measure([(0, 0)])
mid.x(0)
mid.measure([(0, 1)])
print(mid.execute(shots=1000)["counts"])
```

`reset(qubit)` and `reset_qubits([...])` return qubits to |0⟩ mid-circuit:

```{code-cell} python
reuse = QuantumCircuit()
reuse.x(0)
reuse.measure([(0, 0)])
reuse.reset(0)
reuse.measure([(0, 1)])
print(reuse.execute(shots=100)["counts"])
```

`QuantumCircuit` has no method for gates conditioned on a measurement. Write
classical feed-forward in OpenQASM 2.0 and run it with `simple_execute`. Here
qubit 1 is flipped only when qubit 0 measured 1:

```{code-cell} python
feed_forward = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
measure q[0] -> c[0];
if (c==1) x q[1];
measure q[1] -> c[1];
"""
print(maestro.simple_execute(feed_forward, shots=1000)["counts"])
```

## Reproducible results

Set `seed` on the config to make sampling repeatable. Parallel jobs derive
distinct, deterministic random streams from it.

```{code-cell} python
config = maestro.SimulatorConfig(seed=1234)
first = qc.execute(shots=1000, config=config)["counts"]
second = qc.execute(shots=1000, config=config)["counts"]
print(first, first == second)
```
