---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Expectation values and parameter sweeps

Compute Pauli expectation values directly from the simulated state, without
sampling, and sweep a circuit's parameters with exact gradients.

```{code-cell} python
import maestro
from maestro.circuits import QuantumCircuit

ghz = QuantumCircuit()
ghz.h(0)
ghz.cx(0, 1)
ghz.cx(1, 2)

result = ghz.estimate(["ZZI", "XXX", "IZI"])
print(result["expectation_values"])
```

Observables are Pauli strings, given as a list or as one `;`-separated string.
Values come back in the order you gave them. The circuit needs no measurements.
For this GHZ state ⟨ZZI⟩ = ⟨XXX⟩ = 1, while a single ⟨Z⟩ averages to 0.

:::{warning}
In a Pauli string, the character at position *q* acts on qubit *q*, so `"ZII"`
is Z on qubit 0. Qiskit's `SparsePauliOp` uses the opposite order, where the
same operator is `"IIZ"`.
:::

(py_expectation)=
## From an OpenQASM string

`maestro.simple_estimate` takes an OpenQASM 2.0 program instead of a circuit:

```{code-cell} python
ghz_qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
"""

values = maestro.simple_estimate(ghz_qasm, "ZZZ;ZZI;XXX")["expectation_values"]
print(values)
```

⟨ZZZ⟩ is 0 here: |000⟩ and |111⟩ have opposite parity under Z⊗Z⊗Z.

(py_circuit_estimate)=
## On a different simulation method

Pass a `SimulatorConfig` to estimate on another method, for example an MPS:

```{code-cell} python
mps_config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=16,
)
print(ghz.estimate(["ZZI", "XXX"], config=mps_config)["expectation_values"])
```

(py_circuits_transforms)=
## Reductions before simulation

Before allocating a simulator, Maestro shrinks the problem:

1. **Causal-cone reduction** (`enable_causal_cone_reduction=True`, default)
   removes gates and idle qubits outside the backward lightcone of the requested
   observables.
2. **Observable clustering** (`enable_observable_clustering=True`, default)
   splits the observables into groups whose lightcones are disjoint, and reduces
   and simulates each group on its own smaller register. Observables whose
   lightcones overlap, even indirectly through other observables, stay in one
   group. `max_cluster_qubits` also splits overlapping lightcones (see below).
3. **Parity-check reduction** (`auto_reduce=True`, default) detects invariant
   check subspaces, such as QAOA parity ladders with uncomputed CNOT frames, and
   simulates an effective register of dimension 2^rank(checks) on the backend
   you configured.

The result reports what was applied. Here two independent entangled pairs give
two groups of two qubits each:

```{code-cell} python
pairs = QuantumCircuit()
pairs.h(0)
pairs.cx(0, 1)
pairs.ry(2, 0.4)
pairs.cx(2, 3)

result = pairs.estimate(["XXII", "IIXX"])
for key in ["expectation_values", "auto_reduced", "qubits_before", "qubits_after",
            "clusters_count", "largest_cluster_qubits", "cluster_methods"]:
    print(f"{key}: {result[key]}")
```

`qubits_after` reports the largest group, and `cluster_methods` lists each
group's `SimulationType`. The source circuit is unchanged. Turn the
reductions off with the matching config fields:

```{code-cell} python
config = maestro.SimulatorConfig(
    enable_causal_cone_reduction=False,
    enable_observable_clustering=False,
    auto_reduce=False,
)
result = pairs.estimate(["XXII", "IIXX"], config=config)
print(result["qubits_after"], result["clusters_count"])
```

% TODO(engineering): clusters_count is 1 (largest_cluster_qubits 2) in two cases
% where the text above implies 2: Z-only observables ["ZZII", "IIZZ"], and
% ["XXII", "IIXX"] on two identical Bell pairs (h/cx on both). Explain how
% diagonal observables and identical groups are counted, then document it here.

(py_circuits_cluster_budget)=
### Bounding group size

When many observables are spread across a device, their lightcones often chain
together into one group covering every qubit, and clustering saves nothing.
`max_cluster_qubits` caps each group's combined lightcone. Each group starts
from the largest remaining lightcone and adds the overlapping observable that
adds the fewest new qubits, while the union fits the budget. An observable whose
own lightcone is larger than the budget gets a group of its own.

Here every nearest-neighbour ⟨XX⟩ on a 12-qubit brickwork circuit overlaps the
next, so without a budget they form one 12-qubit group:

```{code-cell} python
n = 12
brickwork = QuantumCircuit()
for q in range(n):
    brickwork.ry(q, 0.1 * (q + 1))
for q in range(0, n - 1, 2):
    brickwork.cx(q, q + 1)
for q in range(1, n - 1, 2):
    brickwork.cx(q, q + 1)

neighbours = ["I" * q + "XX" + "I" * (n - 2 - q) for q in range(n - 1)]
for budget in [None, 4]:
    config = maestro.SimulatorConfig(
        simulation_type=maestro.SimulationType.PauliPropagator,
        max_cluster_qubits=budget,
    )
    result = brickwork.estimate(neighbours, config=config)
    print(f"budget={budget}: {result['clusters_count']} groups, "
          f"largest {result['largest_cluster_qubits']} qubits")
```

Smaller groups mean smaller registers but more simulator runs, so the best
budget depends on the backend. A budget helps backends whose cost grows with
register width (Pauli propagation, statevector); matrix product states pay per
group and gain little. GPU Pauli propagation, which requires
{doc}`GPU access </getting-started/gpu-access>`, ignores the budget and keeps all
observables in one run.

% TODO(engineering): the old docs cite "a 118-qubit benchmark with 179 Pauli
% observables, CPU Pauli propagation ran about 2.2x faster with budgets from 35
% to 50" and "each GPU run costs about a second to set up". Add these back only
% with a script that reproduces them.

```{note}
With `PauliPropagator` and `pp_gates_between_trims` above 1, trims land on
different gates in each group's reduced circuit, so values can change slightly
with the budget, within the truncation error. Trimming and deduplicating after
every gate (`pp_gates_between_trims=1`, `pp_gates_between_deduplications=1`)
makes values independent of the grouping. Measurements, resets, quantum
channels, classical control and distributed GPU layouts fall back to full
execution.
```

(py_sweeps)=
## Parameter sweeps and gradients

To evaluate one circuit at many parameter values, compile it once with
`AdjointProgram` and pass one row of parameters per point. The parameters are
the angles of the `p`, `rx`, `ry`, `rz`, `cp`, `crx`, `cry`, `crz`, `u` and
`cu` gates, in circuit order. `jacobian` also returns exact derivatives,
computed with the adjoint method on the CPU statevector.

```{code-cell} python
import numpy as np

ansatz = QuantumCircuit()
ansatz.ry(0, 0.0)  # parameter 0
ansatz.cx(0, 1)
ansatz.rz(1, 0.0)  # parameter 1

program = maestro.AdjointProgram(ansatz)
print("parameters:", program.num_params)

thetas = np.linspace(0, np.pi, 5)
params = np.column_stack([thetas, np.zeros_like(thetas)])

result = program.jacobian(["ZI", "XX"], params)
values = np.asarray(result["expectation_values"])  # (points, observables)
grads = np.asarray(result["jacobian"])  # (points, observables, parameters)

for theta, (zi, xx), dzi in zip(thetas, values, grads[:, 0, 0]):
    print(f"theta={theta:.2f}  <ZI>={zi:+.3f}  <XX>={xx:+.3f}  d<ZI>/dtheta={dzi:+.3f}")
```

Call `expectation_values` instead of `jacobian` when you only need values, or
`vjp` to backpropagate a loss through the circuit. `maestro.adjoint_jacobian`
does the same in one call without keeping a compiled program.

When only the end of a circuit changes between runs,
{py:class}`maestro.PrefixCheckpointedSimulator` simulates the shared prefix once
and runs each suffix from that saved state.

(py_mirror)=
## Mirror fidelity

Mirror fidelity runs a circuit, then the inverse of every gate in reverse order,
and returns the probability of getting back to |0…0⟩. An exact simulation gives
1.0, so the shortfall measures simulation error, for example from a low MPS bond
dimension.

```{code-cell} python
mirror = QuantumCircuit()
mirror.h(0)
mirror.cx(0, 1)
mirror.rx(0, np.pi / 4)

print("sampled (1024 shots):", maestro.mirror_fidelity(mirror))
print("exact amplitude:     ", maestro.mirror_fidelity(mirror, full_amplitude=True))
print("circuit method, MPS: ", mirror.mirror_fidelity(config=mps_config, shots=10000))
```

By default it samples 1024 shots. `full_amplitude=True` computes it exactly,
which only suits small circuits. Measurements in the circuit are skipped; only
unitary gates are inverted.

(py_mirror_observables)=
### Observable mirror fidelity

For wide circuits, the full-register probability is intractable for Pauli
propagation and collapses towards 0 for a truncated MPS. Pass `observables` to
evaluate Pauli strings on the mirror circuit C C† instead. Z-type strings have
ideal value 1.0, so truncation error shows up as a value below 1.0 on exactly
the observables you study.

```{code-cell} python
pp_config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.PauliPropagator,
    pp_max_pauli_weight=4,
    pp_coefficient_threshold=1e-6,
)
print(mirror.mirror_fidelity(config=pp_config, observables="ZZ"))  # one string: float
print(mirror.mirror_fidelity(config=pp_config, observables=["ZZ", "ZI"]))  # list
```

In this mode, circuit optimization is turned off so gate and inverse pairs are
not cancelled, and `shots` and `full_amplitude` are ignored.

(py_inner_product)=
## Inner products

`maestro.inner_product(c1, c2)` returns the complex overlap ⟨ψ₁|ψ₂⟩ of two
circuits' output states. It evaluates ⟨0|U₁†U₂|0⟩ directly, so with an MPS it
never builds a full statevector. `abs(overlap) ** 2` is the state fidelity, and
the phase is the relative global phase.

```{code-cell} python
bell_a = QuantumCircuit()
bell_a.h(0)
bell_a.cx(0, 1)

bell_b = QuantumCircuit()
bell_b.h(0)
bell_b.cx(0, 1)

plus = QuantumCircuit()
plus.h(0)
minus = QuantumCircuit()
minus.x(0)
minus.h(0)

print("<bell|bell> =", maestro.inner_product(bell_a, bell_b))
print("|<+|->|     =", abs(maestro.inner_product(plus, minus)))
```

(py_incremental_evolve)=
## Incremental time evolution

`maestro.incremental_evolve` applies a Trotter step repeatedly on one persistent
simulator and records expectation values at chosen steps, instead of
re-simulating from step 0 for each time point.

```{code-cell} python
init = QuantumCircuit()
init.x(0)

step = QuantumCircuit()
step.rx(0, 0.1)
step.rx(1, 0.2)
step.cx(0, 1)
step.rz(1, 0.05)
step.cx(0, 1)

result = maestro.incremental_evolve(
    init_circuit=init,
    trotter_step=step,
    measure_at_steps=[10, 20, 30, 50, 100],
    observables=["ZI", "IZ", "ZZ"],
    config=mps_config,
)
for s, evs in zip(result["steps"], result["expectation_values"]):
    print(f"step {s:3d}: <ZI>={evs[0]:+.4f}  <IZ>={evs[1]:+.4f}  <ZZ>={evs[2]:+.4f}")
```

Measuring at *n* points across *N* steps costs about *n·N/2* steps when each
point starts a fresh circuit, and *N* steps incrementally: 300 instead of about
3,000 for N = 300 and 20 points.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_circuit` | `QuantumCircuit` | Required | State preparation, run once |
| `trotter_step` | `QuantumCircuit` | Required | One Trotter step, applied repeatedly |
| `measure_at_steps` | `list[int]` | Required | Steps at which to compute expectation values |
| `observables` | `list[str]` or `str` | Required | Pauli strings to evaluate |
| `config` | `SimulatorConfig` | `SimulatorConfig()` | Backend configuration |
