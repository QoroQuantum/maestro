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
