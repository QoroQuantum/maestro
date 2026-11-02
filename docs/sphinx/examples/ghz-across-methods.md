---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# One circuit, three simulation methods

Run the same GHZ circuit on a statevector, an MPS and a stabilizer simulator,
from 20 to 1,000 qubits.

```{code-cell} python
import time

import maestro
from maestro.circuits import QuantumCircuit


def ghz(num_qubits):
    qc = QuantumCircuit()
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    qc.measure_all()
    return qc


runs = [
    (20, maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector)),
    (100, maestro.SimulatorConfig(
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=4,
    )),
    (1000, maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Stabilizer)),
]

for num_qubits, config in runs:
    start = time.perf_counter()
    counts = ghz(num_qubits).execute(shots=1000, config=config)["counts"]
    elapsed = time.perf_counter() - start

    outcomes = {"all zeros" if set(k) == {"0"} else "all ones": v for k, v in counts.items()}
    print(f"{num_qubits:>5} qubits  {config.simulation_type.name:<20} {outcomes}  {elapsed * 1000:.1f} ms")
```

A GHZ state puts every qubit into (|00…0⟩ + |11…1⟩)/√2, so sampling it only
ever returns all zeros or all ones, in roughly equal numbers. Each method above
runs at a size the others can't reach:

- A statevector stores all 2ⁿ amplitudes, so 20 qubits is easy but 100 is out of
  reach.
- An MPS needs only bond dimension 2 for a GHZ state, however many qubits it has.
- A stabilizer simulator handles Clifford circuits like this one in polynomial
  time.

{doc}`/user-guide/choosing-a-method` covers how to pick a method for your own
circuits.
