---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_circuits)=
# Building circuits

Build a circuit gate by gate with `QuantumCircuit`, then run it on any
simulation method.

```{code-cell} python
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.rz(1, 0.25)
qc.measure([(0, 0), (1, 1)])  # (qubit, classical bit) pairs

print("qubits:", qc.num_qubits)
print(qc.execute(shots=1000)["counts"])
```

There is no register size to declare. A circuit grows to cover every qubit you
use, and qubits and classical bits are numbered from 0. To sample the circuit or
read its counts, see {doc}`sampling-and-shots`. To compute expectation values
instead, see {doc}`expectation-values`.

(py_circuit_gates)=
## Available gates

Angles are in radians.

| Gate | Method | Parameters |
|------|--------|------------|
| Pauli-X | `qc.x(qubit)` | — |
| Pauli-Y | `qc.y(qubit)` | — |
| Pauli-Z | `qc.z(qubit)` | — |
| Hadamard | `qc.h(qubit)` | — |
| S | `qc.s(qubit)` | — |
| S† | `qc.sdg(qubit)` | — |
| T | `qc.t(qubit)` | — |
| T† | `qc.tdg(qubit)` | — |
| √X | `qc.sx(qubit)` | — |
| √X† | `qc.sxdg(qubit)` | — |
| K | `qc.k(qubit)` | — |
| Phase | `qc.p(qubit, λ)` | λ |
| Rx | `qc.rx(qubit, θ)` | θ |
| Ry | `qc.ry(qubit, θ)` | θ |
| Rz | `qc.rz(qubit, θ)` | θ |
| U | `qc.u(qubit, θ, φ, λ)` | 3 Euler angles |
| CNOT | `qc.cx(ctrl, tgt)` | — |
| CY | `qc.cy(ctrl, tgt)` | — |
| CZ | `qc.cz(ctrl, tgt)` | — |
| CH | `qc.ch(ctrl, tgt)` | — |
| CSX | `qc.csx(ctrl, tgt)` | — |
| CSX† | `qc.csxdg(ctrl, tgt)` | — |
| SWAP | `qc.swap(q1, q2)` | — |
| CP | `qc.cp(ctrl, tgt, λ)` | λ |
| CRx | `qc.crx(ctrl, tgt, θ)` | θ |
| CRy | `qc.cry(ctrl, tgt, θ)` | θ |
| CRz | `qc.crz(ctrl, tgt, θ)` | θ |
| CU | `qc.cu(ctrl, tgt, θ, φ, λ, γ)` | 4 angles |
| Toffoli | `qc.ccx(c1, c2, tgt)` | — |
| Fredkin | `qc.cswap(ctrl, q1, q2)` | — |
| Delay | `qc.delay(qubit, duration)` | duration in seconds |

The docs build applies every gate in this table with the listed arguments, so
the table stays in step with the package:

```{code-cell} python
:tags: [hide-input]

import math

gates = QuantumCircuit()
one_qubit = ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "k"]
for name in one_qubit:
    getattr(gates, name)(0)
for name in ["p", "rx", "ry", "rz"]:
    getattr(gates, name)(0, math.pi / 4)
gates.u(0, 0.1, 0.2, 0.3)
for name in ["cx", "cy", "cz", "ch", "csx", "csxdg", "swap"]:
    getattr(gates, name)(0, 1)
for name in ["cp", "crx", "cry", "crz"]:
    getattr(gates, name)(0, 1, math.pi / 4)
gates.cu(0, 1, 0.1, 0.2, 0.3, 0.4)
gates.ccx(0, 1, 2)
gates.cswap(0, 1, 2)
gates.delay(0, 1e-6)
print(f"applied 31 gate types on {gates.num_qubits} qubits")
```

(py_circuit_delay)=
## Delays and idle time

`qc.delay(qubit, duration)` schedules idle time on a qubit, in seconds:

```{code-cell} python
idle = QuantumCircuit()
idle.h(0)
idle.delay(0, 50e-6)  # 50 microseconds
idle.h(0)
idle.measure([(0, 0)])
print(idle.execute(shots=1000)["counts"])
```

Without noise, a delay does nothing. With a `NoiseModel` that sets idle noise,
it applies T1/T2 relaxation, Ornstein-Uhlenbeck phase drift and coherent
detuning for the duration; see {ref}`py_noise_idle`.

## From OpenQASM

To run an existing OpenQASM 2.0 program, pass the string to
`maestro.simple_execute` or `maestro.simple_estimate`; see
{ref}`py_quick`. OpenQASM is also the way to write classical feed-forward
(`if (c==1) x q[1];`), which `QuantumCircuit` has no method for; see
{ref}`py_mid_circuit`.
