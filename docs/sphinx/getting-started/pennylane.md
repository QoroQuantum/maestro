---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_pennylane)=
# Use Maestro from PennyLane

Run existing PennyLane code on Maestro by changing one line: the device name.

```bash
pip install pennylane-maestro
```

```{code-cell} python
import pennylane as qp

dev = qp.device("maestro.qubit", wires=2)  # instead of "default.qubit"


@qp.qnode(dev)
def circuit(theta):
    qp.RX(theta, wires=0)
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.Z(1))


print(circuit(0.5))
```

The plugin is a separate package,
[pennylane-maestro](https://github.com/QoroQuantum/pennylane-maestro), which
installs PennyLane and `qoro-maestro` for you. Like Maestro, PennyLane writes
wire 0 first in bitstrings, so counts read the same in both.

## VQE with exact gradients

On the CPU statevector, the default, the plugin computes gradients with the
adjoint method, so `qp.grad` and PennyLane's optimizers work unchanged:

```{code-cell} python
from pennylane import numpy as np

n = 4
hamiltonian = qp.Hamiltonian(
    [-1.0] * (n - 1) + [-0.5] * n,
    [qp.Z(i) @ qp.Z(i + 1) for i in range(n - 1)] + [qp.X(i) for i in range(n)],
)
dev = qp.device("maestro.qubit", wires=n)


@qp.qnode(dev)
def energy(params):
    for i in range(n):
        qp.RY(params[i], wires=i)
    for i in range(n - 1):
        qp.CNOT(wires=[i, i + 1])
    for i in range(n):
        qp.RY(params[n + i], wires=i)
    return qp.expval(hamiltonian)


params = np.array([0.1] * (2 * n), requires_grad=True)
print("gradient method:", qp.workflow.get_best_diff_method(energy)(params))

optimizer = qp.GradientDescentOptimizer(stepsize=0.2)
for step in range(41):
    params, value = optimizer.step_and_cost(energy, params)
    if step % 10 == 0:
        print(f"step {step:2d}: energy {value:.4f}")
print(f"exact ground energy: {min(qp.eigvals(hamiltonian)):.4f}")
```

The plugin evaluates every term of the Hamiltonian in one Maestro call. Finite
shots, mid-circuit measurements and methods other than the statevector fall back
to parameter-shift gradients.

## Choose the simulation method

Device keyword arguments select the method and map onto Maestro's
`SimulatorConfig`. An MPS handles many qubits when entanglement is limited:

```{code-cell} python
mps = qp.device("maestro.qubit", wires=60, simulation_type="MatrixProductState", max_bond_dimension=16)


@qp.qnode(mps)
def ghz_correlation():
    qp.Hadamard(0)
    for i in range(59):
        qp.CNOT(wires=[i, i + 1])
    return qp.expval(qp.Z(0) @ qp.Z(59))


print(ghz_correlation())
```

Finite shots with a seed:

```{code-cell} python
sampler = qp.device("maestro.qubit", wires=2, shots=1000, seed=42)


@qp.qnode(sampler)
def bell_counts():
    qp.Hadamard(0)
    qp.CNOT(wires=[0, 1])
    return qp.counts()


print(bell_counts())
```

| Argument | Values |
|----------|--------|
| `simulation_type` | `"Statevector"` (default), `"MatrixProductState"`, `"Stabilizer"`, `"TensorNetwork"`, `"PauliPropagator"`, `"DensityMatrix"`, `"MatrixProductOperator"`, … |
| `simulator_type` | `"QCSim"` (default, CPU), `"QiskitAer"`, or `"Gpu"`, `"DistributedGpu"` and `"DistributedMpiGpu"` with GPU access |
| `max_bond_dimension`, `singular_value_threshold` | MPS truncation |
| `pp_max_pauli_weight`, `pp_coefficient_threshold` | Pauli propagation truncation |
| `shots`, `seed` | Finite-shot sampling and reproducibility |

Other `SimulatorConfig` options, such as `gate_fusion` or `precision`, are
accepted as device arguments too; see the
[plugin README](https://github.com/QoroQuantum/pennylane-maestro#device-options)
for the full list.

## Run on a GPU

:::{admonition} Requires GPU access
:class: note
`simulator_type="Gpu"` needs the GPU plugin and a licence. See
{doc}`gpu-access`.
:::

```python
# Not executed in the docs build: requires GPU access.
dev = qp.device("maestro.qubit", wires=28, simulator_type="Gpu")
gpu_mps = qp.device(
    "maestro.qubit", wires=200, simulator_type="Gpu",
    simulation_type="MatrixProductState", max_bond_dimension=256,
)
```
