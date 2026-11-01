---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Maestro

One Python API for simulating quantum circuits with statevector, MPS,
stabilizer, Pauli-propagation and density-matrix methods, on your laptop or on
GPUs.

```bash
pip install qoro-maestro
```

```{code-cell} python
from maestro.circuits import QuantumCircuit
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
print(qc.execute(shots=1000)["counts"])
```

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/first-circuit
:link-type: doc

Install Maestro and run your first circuit in five minutes, on the CPU.
:::

:::{grid-item-card} Get GPU access
:link: getting-started/gpu-access
:link-type: doc

Run the same code on NVIDIA GPUs, in our cloud or on your own machines.
:::

:::{grid-item-card} Coming from Qiskit
:link: getting-started/coming-from-qiskit
:link-type: doc

Swap Aer for Maestro with one import using qiskit-maestro, or translate your Aer
code pattern by pattern.
:::
::::

% The C++ reference moves to /cpp/ when both sites deploy together; see
% .github/workflows/docs.yml.

:::{toctree}
:hidden:

getting-started/index
user-guide/index
examples/index
api/index
C++ reference <https://qoroquantum.github.io/maestro/cpp/>
benchmarks
:::
