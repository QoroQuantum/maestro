---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_quickstart)=
# Install and run your first circuit

Install Maestro, simulate a Bell state, and read its counts and expectation
values, all on your CPU in about five minutes.

```bash
pip install qoro-maestro
```

```{code-cell} python
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

result = qc.execute(shots=1024)
print("counts:", result["counts"])

estimate = qc.estimate(observables=["ZZ", "XX", "YY"])
print("expectation values:", [round(v, 6) for v in estimate["expectation_values"]])
```

The default simulator is the CPU statevector. `counts` maps each measured
bitstring to the number of shots that gave it. For a Bell state,
⟨ZZ⟩ = ⟨XX⟩ = 1 and ⟨YY⟩ = −1.

:::{warning}
Maestro writes qubit 0 first, on the **left**, in both bitstrings and Pauli
strings. Qiskit writes it last, on the right. Flipping only qubit 0 of three
reads `"100"` in Maestro and `"001"` in Qiskit:
:::

```{code-cell} python
flip = QuantumCircuit()
flip.x(0)
flip.measure([(0, 0), (1, 1), (2, 2)])

counts = flip.execute(shots=10)["counts"]
print("Maestro order:", counts)
print("Qiskit order: ", {bits[::-1]: n for bits, n in counts.items()})
```

(py_install)=
## Install options

Pre-built wheels need no compiler. They are available for Python 3.10 and newer
on these platforms:

| Platform | Architecture |
|----------|--------------|
| Linux | x86_64 |
| macOS | arm64 (Apple Silicon) |
| Windows | AMD64 |

The pip package includes every CPU simulation method. GPU simulation needs a
separate plugin; see {doc}`gpu-access`.

To sample Stim circuits through Maestro, install the Sinter extra:

```bash
pip install "qoro-maestro[sinter]"
```

To build from source, run `pip install .` from the repository root. The
[installation guide](https://github.com/QoroQuantum/maestro/blob/main/markdown/INSTALL.md)
lists the build requirements.

(py_quick)=
## Run an OpenQASM string

`maestro.simple_execute` runs an OpenQASM 2.0 program directly:

```{code-cell} python
import maestro

qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

result = maestro.simple_execute(qasm, shots=1024)
print(result["counts"])
print(result["simulator"], result["method"])
```

`result["simulator"]` and `result["method"]` say which backend and simulation
method ran; {ref}`py_result_dict` lists every key.

(py_config)=
## Choose a different simulator

Every execution and estimation function accepts a `SimulatorConfig`. Create one
and reuse it to change the simulation method without touching the circuit:

```{code-cell} python
config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
print(qc.execute(shots=1024, config=config)["counts"])
```

## Next steps

- {doc}`/user-guide/choosing-a-method` compares statevector, MPS, stabilizer and
  the other methods.
- {doc}`gpu-access` explains how to run the same code on an NVIDIA GPU.
- {doc}`coming-from-qiskit` runs your Qiskit code on Maestro with qiskit-maestro,
  or maps Aer code onto Maestro's own API.
