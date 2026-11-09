# Benchmarks

Maestro plugs into the frameworks you already use: Qiskit through
[qiskit-maestro](https://github.com/QoroQuantum/qiskit-maestro) and PennyLane
through [pennylane-maestro](https://github.com/QoroQuantum/pennylane-maestro). These benchmarks measure what you get when you
change only the backend or device: the same circuits, observables and shots,
the same bond-dimension cap, and the same number of CPU threads.

```{image} _static/benchmarks/overview.png
:class: only-light
:alt: Bar chart of Maestro's speed-up for seven workloads. Against Qiskit Aer it ranges from 1.8 times for statevector sampling to 608 times for exact noisy simulation; against PennyLane's own devices from 11 to 638 times.
```

```{image} _static/benchmarks/overview-dark.png
:class: only-dark
:alt: Bar chart of Maestro's speed-up for seven workloads. Against Qiskit Aer it ranges from 1.8 times for statevector sampling to 608 times for exact noisy simulation; against PennyLane's own devices from 11 to 638 times.
```

Each bar is the competitor's time divided by Maestro's, at the largest size
both finished. Every Maestro result was checked against a reference simulator,
and none failed. Maestro isn't faster everywhere: see
[where Maestro is slower](#where-maestro-is-slower).

The switch is one line:

```python
# Qiskit: instead of AerSimulator(method="matrix_product_state")
from qiskit_maestro import MaestroBackend

backend = MaestroBackend(method="matrix_product_state", max_bond_dimension=64)

# PennyLane: instead of qml.device("default.tensor", wires=n, method="mps")
import pennylane as qml

dev = qml.device(
    "maestro.qubit", wires=n, simulation_type="MatrixProductState", max_bond_dimension=64
)
```

## Statevector

```{image} _static/benchmarks/statevector_sampling.png
:class: only-light
:alt: Time against qubits for statevector sampling. Both Maestro plugins are about 1.8 times faster than Aer from 24 to 32 qubits, and 19 times faster than lightning.qubit, which goes over an hour at 32 qubits.
```

```{image} _static/benchmarks/statevector_sampling-dark.png
:class: only-dark
:alt: Time against qubits for statevector sampling. Both Maestro plugins are about 1.8 times faster than Aer from 24 to 32 qubits, and 19 times faster than lightning.qubit, which goes over an hour at 32 qubits.
```

10,000 shots from a random circuit of single-qubit rotations and CNOTs, with
depth equal to the number of qubits:

| Qubits | Aer | qiskit-maestro | Speed-up | `lightning.qubit` | `maestro.qubit` | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.041 s | 0.060 s | 0.7× | 0.079 s | 0.106 s | 0.7× |
| 24 | 2.8 s | 1.7 s | 1.7× | 30 s | 1.8 s | 17× |
| 28 | 62 s | 35 s | 1.8× | 663 s | 34 s | 19× |
| 30 | 289 s | 157 s | 1.8× | 3,038 s | 157 s | 19× |
| 32 | 1,321 s | 717 s | 1.8× | over 1 h | 714 s | over 5× |

At 32 qubits `lightning.qubit` passed the 1-hour limit, so "over 5×" is a
lower bound: the 1-hour limit divided by Maestro's time. The other "over"
speed-ups on this page are bounds of the same kind.

Quantum-volume circuits give the same picture: 1.8× against Aer at 32 qubits.
`lightning.qubit` also used three times the memory (50 GB at 30 qubits,
against 17 GB for Maestro and Aer).

### Parameter sweeps

```{image} _static/benchmarks/statevector_batches.png
:class: only-light
:alt: Time against qubits for 100 parameter sets of a VQE circuit. qiskit-maestro is about 3 times faster than Aer from 20 to 28 qubits; lightning.qubit crashes at 28 qubits.
```

```{image} _static/benchmarks/statevector_batches-dark.png
:class: only-dark
:alt: Time against qubits for 100 parameter sets of a VQE circuit. qiskit-maestro is about 3 times faster than Aer from 20 to 28 qubits; lightning.qubit crashes at 28 qubits.
```

Variational algorithms evaluate one circuit at many parameter values. This is
the VQE circuit used below (`efficient_su2`, 3 layers, linear entanglement,
Ising Hamiltonian) at 100 parameter sets, sent as one batch:

| Qubits | Aer | qiskit-maestro | Speed-up | `lightning.qubit` | `maestro.qubit` |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.61 s | 0.14 s | 4.2× | 1.5 s | 0.16 s |
| 20 | 7.0 s | 2.4 s | 3.0× | 24 s | 4.9 s |
| 24 | 137 s | 46 s | 3.0× | 736 s | 147 s |
| 28 | 2,577 s | 870 s | 3.0× | crashed | 2,717 s |

`qiskit-maestro` runs several circuits at once, 15 here, one per thread, and
that costs memory: it peaked at 60 GB at 28 qubits, against 4 GB for Aer. Lower
`max_parallel_experiments` to trade speed for memory. `maestro.qubit` runs one
circuit at a time, as Aer does, so it matches Aer from 24 qubits (0.94× and
0.95×). `lightning.qubit` aborted at 28 qubits without a Python error.

### Training in PennyLane

```{image} _static/benchmarks/pennylane_training.png
:class: only-light
:alt: Time for PennyLane training steps. maestro.qubit is 7.6 to 11 times faster than lightning.qubit on a batched classifier and 3 to 6 times faster on a 50-step VQE; default.qubit fails on the classifier and is 19 to 25 times slower on the VQE.
```

```{image} _static/benchmarks/pennylane_training-dark.png
:class: only-dark
:alt: Time for PennyLane training steps. maestro.qubit is 7.6 to 11 times faster than lightning.qubit on a batched classifier and 3 to 6 times faster on a 50-step VQE; default.qubit fails on the classifier and is 19 to 25 times slower on the VQE.
```

Gradients through the adjoint method (`default.qubit` uses backpropagation),
for a `StronglyEntanglingLayers` circuit with 4 layers. The classifier step is
one loss and gradient over a batch of 32 inputs. The VQE is 50 Adam steps on an
Ising chain.

| Workload | Qubits | `default.qubit` | `lightning.qubit` | `maestro.qubit` | Speed-up over Lightning |
|---|---:|---:|---:|---:|---:|
| Classifier step | 16 | error¹ | 3.2 s | 0.30 s | 11× |
| Classifier step | 20 | error¹ | 67 s | 7.2 s | 9.3× |
| Classifier step | 22 | error¹ | 338 s | 44 s | 7.6× |
| VQE, 50 steps | 16 | 32 s | 5.3 s | 1.7 s | 3.2× |
| VQE, 50 steps | 20 | 423 s | 103 s | 17 s | 6.1× |

¹ `default.qubit` raised a `ValueError` from 12 qubits up on the batched
classifier (PennyLane 0.45.1). It reproduces with plain PennyLane, outside the
benchmark harness. Setting `device_vjp=True` on `maestro.qubit`, which computes
the whole vector-Jacobian product on the device, was slightly slower here: 0.36 s,
9.1 s and 54 s.

(bench-mps-estimator)=
## Matrix product states

MPS simulation is where Maestro's lead is largest. All MPS runs use a
truncation threshold of 10⁻¹⁰, and a bond-dimension cap of 64 unless stated.

### Expectation values on many qubits

```{image} _static/benchmarks/mps_qubits.png
:class: only-light
:alt: Time against qubits, 50 to 1,000, for one VQE energy evaluation. Maestro is 5.5 times faster than Aer at 50 qubits and 17 times at 1,000; default.tensor is slowest.
```

```{image} _static/benchmarks/mps_qubits-dark.png
:class: only-dark
:alt: Time against qubits, 50 to 1,000, for one VQE energy evaluation. Maestro is 5.5 times faster than Aer at 50 qubits and 17 times at 1,000; default.tensor is slowest.
```

One VQE energy evaluation: `efficient_su2(n, reps=3, entanglement="linear")`
with random angles, and a transverse-field Ising Hamiltonian with 2n − 1 terms.

| Qubits | Aer `EstimatorV2` | `MaestroEstimator` | Speed-up | `default.tensor` | `maestro.qubit` | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.070 s | 0.013 s | 5.5× | 0.67 s | 0.026 s | 26× |
| 100 | 0.22 s | 0.029 s | 7.6× | 2.7 s | 0.054 s | 50× |
| 200 | 0.76 s | 0.074 s | 10× | 12 s | 0.12 s | 100× |
| 500 | 4.5 s | 0.31 s | 14× | 125 s | 0.41 s | 300× |
| 1,000 | 18.6 s | 1.1 s | 17× | 836 s | 1.3 s | 640× |

### High bond dimension

```{image} _static/benchmarks/mps_bond_dimension.png
:class: only-light
:alt: Time against maximum bond dimension, 16 to 512, on 100 qubits. Maestro's lead over Aer grows from 2.4 times at bond dimension 16 to 18 times at 256.
```

```{image} _static/benchmarks/mps_bond_dimension-dark.png
:class: only-dark
:alt: Time against maximum bond dimension, 16 to 512, on 100 qubits. Maestro's lead over Aer grows from 2.4 times at bond dimension 16 to 18 times at 256.
```

100 qubits, 12 layers of random two-qubit (SU(4)) gates in a brickwork, which
quickly saturates the bond dimension, then ⟨Z⟩ on every qubit:

| Bond dimension | Aer | qiskit-maestro | Speed-up | `default.tensor` | `maestro.qubit` | Speed-up |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.72 s | 0.30 s | 2.4× | 3.5 s | 0.61 s | 5.8× |
| 64 | 13 s | 2.2 s | 6.0× | 178 s | 2.6 s | 70× |
| 128 | 83 s | 7.3 s | 11× | 222 s | 7.7 s | 29× |
| 256 | 457 s | 25 s | 18× | 271 s | 25 s | 11× |
| 512 | 796 s | 48 s | 17× | 301 s | 48 s | 6.3× |

### Accuracy against time

A bigger bond dimension buys accuracy, so the fair question is how long each
simulator takes to reach a given accuracy. On a 60-step Ising quench (h = 1),
every simulator reaches about the same accuracy at the same bond dimension, so
the times compare like for like:

| Bond dimension | 28 qubits: Aer / qiskit-maestro | Speed-up | 100 qubits: Aer / qiskit-maestro | Speed-up |
|---:|---:|---:|---:|---:|
| 32 | 2.1 s / 1.3 s | 1.6× | 15 s / 6.6 s | 2.2× |
| 64 | 9.9 s / 3.9 s | 2.5× | 82 s / 21 s | 3.8× |
| 128 | 58 s / 12 s | 4.8× | 518 s / 73 s | 7.1× |
| 256 | 391 s / 35 s | 9.9× | over 1 h / 239 s | over 15× |
| 512 | not run | not run | over 1 h / 466 s | over 7× |

Bond dimension 512 was not run at 28 qubits because 256 is already exact
there.

At 28 qubits, bond dimension 256 matches the exact statevector to 8 × 10⁻⁶ in
every ⟨Z⟩, in 35 s on Maestro against 391 s on Aer. At 100 qubits only Maestro
reached bond dimension 512. `maestro.qubit` against `default.tensor` at 100
qubits: 11× at bond dimension 32, 45× at 128 and 14× at 256.

### Sampling

```{image} _static/benchmarks/mps_sampling.png
:class: only-light
:alt: Time against shots on a 1,000-qubit circuit. Maestro is 4 times faster than Aer at 1,000 shots, 22 times at 10,000 and 39 times at 100,000.
```

```{image} _static/benchmarks/mps_sampling-dark.png
:class: only-dark
:alt: Time against shots on a 1,000-qubit circuit. Maestro is 4 times faster than Aer at 1,000 shots, 22 times at 10,000 and 39 times at 100,000.
```

Shots from the VQE circuit above. Aer's time grows in proportion to the shots,
while Maestro's grows much more slowly:

| Qubits | Shots | Aer | qiskit-maestro | Speed-up | `maestro.qubit` |
|---:|---:|---:|---:|---:|---:|
| 100 | 100,000 | 33 s | 1.2 s | 27× | 4.2 s |
| 500 | 100,000 | 165 s | 4.2 s | 40× | 19 s |
| 1,000 | 1,000 | 3.6 s | 0.85 s | 4.2× | 1.6 s |
| 1,000 | 10,000 | 34 s | 1.5 s | 22× | 4.8 s |
| 1,000 | 100,000 | 334 s | 8.6 s | 39× | 38 s |

PennyLane's `default.tensor` supports analytic results only, so it has no
column here.

:::{note}
`transpile(circuit, AerSimulator(method="matrix_product_state"))` raises
`CircuitTooWideForTarget` above 63 qubits, so both Qiskit backends ran these
circuits untranspiled. Their gates are native to both simulators.
:::

### Time evolution, measured at every step

Dynamics code measures after each Trotter step: one circuit per time point,
here 21 of them (0 to 20 steps), each with the energy and ⟨Z⟩ on every qubit:

| Qubits | Aer, 21 PUBs | `MaestroEstimator`, 21 PUBs | Speed-up | `MaestroEstimator.run_evolution` | Speed-up |
|---:|---:|---:|---:|---:|---:|
| 100 | 5.8 s | 1.2 s | 4.9× | 0.15 s | 38× |
| 200 | 18.9 s | 2.6 s | 7.3× | 0.33 s | 58× |

`MaestroEstimator` recognises that the 21 circuits are one initial state
followed by 0 to 20 repetitions of the same step. It runs them as one evolution
on a persistent simulator, using Maestro's
{ref}`incremental evolution <py_incremental_evolve>`, and reads every time
point on the way: 20 steps of simulation in total, where Aer simulates each
circuit from the start, 210 steps in all. `run_evolution` takes the initial
circuit and the step directly, so it also skips recognising them.
`maestro.qubit` with `incremental_evolve` takes 0.17 s and 0.34 s.
`default.tensor` crashed on this workload, inside cotengra while contracting
the Hamiltonian (PennyLane 0.45.1, quimb 1.15).

Incremental evolution needs the repeated step to stay recognisable: after
transpiling every circuit to `rx`, `ry`, `rz` and `cx`, it no longer triggers,
and Maestro is 3.1× and 4.0× faster than Aer.

The [example notebook](https://github.com/QoroQuantum/qiskit-maestro/blob/main/examples/vqe_aer_vs_maestro.ipynb)
runs a full VQE loop on a 100-qubit Ising chain on each estimator.

### Maestro's circuit shortcuts

Every other result on this page has Maestro's shortcuts turned off
(`enable_causal_cone_reduction`, `enable_observable_clustering` and
`auto_reduce`), so it compares simulator against simulator. They're on by
default. On a shallow 1,000-qubit circuit (depth 4, ⟨Z₀Z₁⟩), Aer takes 0.20 s
and Maestro 0.40 s with the shortcuts off, but 0.055 s with them on: 3.6×
faster than Aer.

## Noise

### Exact noisy simulation

```{image} _static/benchmarks/noise_mpo.png
:class: only-light
:alt: Time against qubits for exact noisy simulation. Density matrices stop at 15 to 16 qubits (out of memory, or over an hour for default.mixed); Maestro's MPO takes under a second at 16 qubits and 6.5 s at 100.
```

```{image} _static/benchmarks/noise_mpo-dark.png
:class: only-dark
:alt: Time against qubits for exact noisy simulation. Density matrices stop at 15 to 16 qubits (out of memory, or over an hour for default.mixed); Maestro's MPO takes under a second at 16 qubits and 6.5 s at 100.
```

A depth-6 random brickwork with amplitude damping (γ = 10⁻³) and depolarizing
noise (10⁻³ on one qubit, 10⁻² on two) after every gate, then ⟨Z⟩ on every
qubit. A density matrix of n qubits takes 16 × 4ⁿ bytes, which runs out in the
mid-teens; Maestro's matrix product operator (MPO) represents the same mixed
state and keeps going:

| Qubits | Aer density matrix | Maestro density matrix | `default.mixed` | Maestro MPO | MPO against Aer |
|---:|---:|---:|---:|---:|---:|
| 15 | 93 s, 16 GB | 621 s, 64 GB | over 1 h | 0.60 s | 3 × 10⁻⁶ |
| 16 | 404 s, 64 GB | out of memory | not run | 0.67 s | 2 × 10⁻⁶ |
| 17 | out of memory | not run | not run | 0.68 s | no reference |
| 50 | not run | not run | not run | 3.2 s | no reference |
| 100 | not run | not run | not run | 6.5 s | no reference |

Once a simulator runs out of memory or time, larger sizes are not run for it.
The last column is the largest difference in any ⟨Z⟩ between the MPO, with
its bond dimension capped at 128, and Aer's exact density matrix.

### Exact MPO against trajectories

```{image} _static/benchmarks/noise_time_to_accuracy.png
:class: only-light
:alt: Time to get every expectation value within 0.001 of the exact answer. Maestro's MPO takes 0.87 to 3.1 s in all four cases; Aer's MPS trajectories take 1.9 s to 14 minutes.
```

```{image} _static/benchmarks/noise_time_to_accuracy-dark.png
:class: only-dark
:alt: Time to get every expectation value within 0.001 of the exact answer. Maestro's MPO takes 0.87 to 3.1 s in all four cases; Aer's MPS trajectories take 1.9 s to 14 minutes.
```

The usual alternative to a density matrix is to average many noisy pure-state
trajectories, whose error shrinks only as one over the square root of their
number. The figure shows the time to get ⟨Z⟩ on every qubit within 10⁻³ (mean
absolute error) of a bond-dimension-512 MPO. At 10⁻⁴:

| Noise, qubits | Maestro MPO | Aer MPS trajectories | Maestro MPS trajectories |
|---|---:|---:|---:|
| Depolarizing, 20 | 0.87 s | 185 s | 299 s |
| Depolarizing, 50 | 3.0 s | 430 s | 692 s |
| Depolarizing and damping, 20 | 0.87 s | about 3.8 h (extrapolated) | not run |
| Depolarizing and damping, 50 | 3.1 s | about 17 h (extrapolated) | not run |

Extrapolated times assume the error keeps falling as one over the square root
of the number of trajectories; the runs themselves stopped at the 1-hour limit.
Maestro's trajectories ran with depolarizing noise only, because its pure-state
amplitude damping is a reset approximation.

### Noisy shot sampling

```{image} _static/benchmarks/noise_sampling.png
:class: only-light
:alt: Time to draw 1,000,000 noisy shots at 20, 32 and 50 qubits. Maestro's MPO takes 14, 38 and 68 seconds; Aer's MPS takes 831 and 2,032 seconds at 20 and 32 qubits and passes one hour at 50.
```

```{image} _static/benchmarks/noise_sampling-dark.png
:class: only-dark
:alt: Time to draw 1,000,000 noisy shots at 20, 32 and 50 qubits. Maestro's MPO takes 14, 38 and 68 seconds; Aer's MPS takes 831 and 2,032 seconds at 20 and 32 qubits and passes one hour at 50.
```

Sampling shots from a noisy circuit: depolarizing noise (10⁻³ on one qubit,
10⁻² on two) after every gate and a 1% readout error, on the same depth-6
random brickwork. Maestro offers three ways (MPO, MPS and statevector) and Aer
two (MPS and statevector):

| Qubits | Shots | Aer MPS | Maestro MPS | Maestro MPO | MPO against Aer MPS |
|---:|---:|---:|---:|---:|---:|
| 20 | 10,000 | 8.1 s | 1.9 s | 0.76 s | 11× |
| 20 | 1,000,000 | 831 s | 218 s | 14 s | 60× |
| 32 | 10,000 | 20 s | 3.3 s | 1.6 s | 13× |
| 32 | 1,000,000 | 2,032 s | 368 s | 38 s | 53× |
| 50 | 10,000 | 49 s | 5.4 s | 2.8 s | 17× |
| 50 | 100,000 | 491 s | 55 s | 8.7 s | 56× |
| 50 | 1,000,000 | over 1 h | 613 s | 68 s | over 50× |

Maestro's MPS is 4 to 9 times faster than Aer's. On statevectors, at 20 qubits
and 10,000 shots, Maestro took 127 s against Aer's 615 s. At 100,000 shots
Aer's statevector passed the 1-hour limit while Maestro's took 1,270 s. At 32
qubits Aer's statevector failed with `No counts for experiment "0"` and
Maestro's passed 1 hour.

## Pauli propagation

```{image} _static/benchmarks/pauli_propagation.png
:class: only-light
:alt: Error against time for Pauli propagation on a 28-qubit chain. At the smaller angle, Maestro reaches an error of 0.006 in 970 seconds; IBM's pauli-prop reaches 0.043 in 1,551 seconds. At the larger angle neither converges in an hour.
```

```{image} _static/benchmarks/pauli_propagation-dark.png
:class: only-dark
:alt: Error against time for Pauli propagation on a 28-qubit chain. At the smaller angle, Maestro reaches an error of 0.006 in 970 seconds; IBM's pauli-prop reaches 0.043 in 1,551 seconds. At the larger angle neither converges in an hour.
```

Pauli propagation evolves the observable, not the state, and drops Pauli
terms with small coefficients. Maestro's version runs through
`MaestroEstimator(method="pauli_propagation")`. The comparison is IBM's
[`pauli-prop`](https://github.com/Qiskit/pauli-prop), the Qiskit
counterpart; PennyLane has none built in. The circuit is a kicked Ising model
for 20 steps, with `rx(θ)` on every qubit and `rzz(−π/2)` on every edge, and
the observable is the average magnetization. There are two lattices: a
28-qubit chain, whose exact answer comes from Aer's statevector, and a
115-qubit heavy-hex lattice, the layout of IBM's devices, which has no exact answer.

The two libraries' thresholds don't mean the same thing: `pauli-prop` trims
after every operation, while Maestro trims on a schedule set by
`pp_gates_between_trims`. We ran Maestro with its default schedule and with
`pp_gates_between_trims=1`, which matches `pauli-prop`, and the same threshold
is not the same work either way. The figure compares accuracy against time
instead, one point per threshold.

On the chain at θ = π/8, where the exact answer is 0.7909, the time to reach a
given error:

| Method | Error ≤ 0.1 | Error ≤ 0.01 |
|---|---:|---:|
| Maestro, default trims | 426 s (error 0.044) | 2,984 s (0.0076) |
| Maestro, trimming after every gate | 118 s (0.085) | 970 s (0.0056) |
| `pauli-prop` | 1,551 s† (0.043) | not reached† |

† `pauli-prop` filled its term cap at thresholds of 10⁻⁶ and below, at 100
million terms (30 GB, as much as these machines hold), so the cap, not the
threshold, limited its accuracy there. Its best error was 0.043.

- **Harder angle (θ = π/4):** none converge within an hour. The best errors are
  0.111 for Maestro at 375 s and 0.125 for `pauli-prop` at 362 s.
- **Heavy-hex, 115 qubits:** both settle near 0.89 at θ = π/8, so which is
  closer isn't known. Maestro's values are within 0.01 of it by 6 s; `pauli-prop`
  needs 66 s, and reaches 0.8916 at 1,773 s, where Maestro had 0.8916 at 165 s.
  At θ = π/4 both settle near 0.26: Maestro in about 1 minute, `pauli-prop` in
  3.5 minutes.
- **Smallest thresholds:** Maestro's runs there either ran out of memory (the
  default schedule at 10⁻⁶ on the chain, at 68 GB) or passed the 1-hour limit.
  They are left out of the figure.

## Where Maestro is slower

- **Small statevector circuits.** At 16 qubits, Aer (0.041 s) and
  `lightning.qubit` (0.079 s) beat Maestro (0.060 s and 0.106 s). Maestro pulls
  ahead from about 20 qubits.
- **Memory in parameter sweeps.** `qiskit-maestro` runs one circuit per
  thread, so a 28-qubit sweep peaked at 60 GB against Aer's 4 GB. It is also
  the reason it is 3× faster.
- **Maestro's own density matrix** is 5 to 7 times slower than Aer's and uses
  four times the memory, running out at 16 qubits. Use the MPO for exact noisy
  simulation instead.
- **MPS trajectories.** With depolarizing noise, Maestro's MPS trajectories are
  1.2 to 1.6 times slower than Aer's. Its statevector trajectories are about 3
  times faster.
- **Loose accuracy with depolarizing noise.** If an error of 10⁻² is enough,
  trajectories (0.03 to 0.09 s) beat the exact MPO (0.3 to 3 s). With
  amplitude damping as well, the MPO is faster at every accuracy.
- **Pauli propagation on hard angles.** At θ = π/4 on the 28-qubit chain,
  Maestro and `pauli-prop` are about level: neither converges in an hour.
- **Circuits with `rzz` gates.** Aer applies `rzz` natively; qiskit-maestro
  expands each one into `cx`, `rz`, `cx`. On 50-qubit Trotter circuits Aer is
  slightly faster (0.11 s against 0.12 s); from 200 qubits Maestro leads again
  (2.3× at 1,000).
- **Shallow circuits with the shortcuts off.** Aer is 2× faster on the
  1,000-qubit, depth-4 circuit above, until Maestro's default shortcuts are on.

## Setup

- **Hardware.** Google Cloud `c2-standard-30` VMs: Intel Xeon (Cascade Lake)
  at 3.1 GHz, 15 cores with two threads each, 117 GB of memory, Debian 12. Each
  comparison, meaning every simulator at one size, ran on one VM, pinned to the
  15 physical cores, with every simulator using 15 threads. The VMs agreed on a
  check workload to within 4%. CPU only.
- **Software.** Python 3.12.16, Qiskit 2.6.2, Qiskit Aer 0.18.2, PennyLane
  0.45.1, PennyLane Lightning 0.45.0, quimb 1.15.0, qoro-maestro 0.3.14,
  qiskit-maestro 0.1.0 and pennylane-maestro 0.6.0.
- **Timing.** The median of five runs after a warm-up run (three runs for a
  case over 60 s, one over 300 s, and the warm-up alone over 900 s), timing the
  whole call, including circuit conversion. Each run had a 1-hour limit and a
  100 GB memory cap.
- **Correctness.** Every result was compared with a reference, except Pauli
  propagation, which is reported as accuracy against the exact value: exact results
  to a relative 10⁻⁸, truncated MPS results to 10⁻⁴ (or reported as accuracy
  where truncation is the point), and sampled results within statistical error.
  No Maestro result failed.

:::{note}
More benchmarks are on the way, including GPU hardware.
:::
