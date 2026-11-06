---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_noise)=
# Noise and error correction

Add a noise model to a circuit, then estimate or sample under noise, from
zero-cost analytical damping to exact density-matrix channels. For error
correction with Stim and Sinter, see {doc}`qec-sinter`.

```{code-cell} python
import math

import maestro
from maestro.circuits import QuantumCircuit

bell = QuantumCircuit()
bell.h(0)
bell.cx(0, 1)

noise = maestro.NoiseModel()
noise.set_all_depolarizing(num_qubits=2, p=0.05)

result = maestro.noisy_estimate(bell, ["ZZ", "XX", "YY"], noise)
print("noisy:", result["expectation_values"])
print("ideal:", result["ideal_expectation_values"])
```

`noisy_estimate` runs the circuit once without noise and damps each expectation
value analytically, so it costs nothing extra. The other functions trade speed
for more physical detail:

| Function | Cost | Noise mechanism | Use for |
|----------|------|-----------------|---------|
| `noisy_estimate` | One noiseless run | Analytical Pauli damping | Screening many circuits quickly |
| `noisy_estimate_montecarlo` | N noisy runs | Random Pauli errors after each gate | Noise that builds up with depth |
| `noisy_execute` | N noisy runs | Gate-by-gate Pauli errors, plus shot noise | Sampled counts under noise |
| `coherent_estimate` | N noisy runs | Systematic rotation errors | Calibration errors |
| `coherent_execute` | N noisy runs | Coherent errors, plus shot noise | Sampled counts under coherent errors |
| `full_noise_estimate` | N noisy runs | Every configured noise layer | Device-level expectation values |
| `full_noise_execute` | N noisy runs | Every configured noise layer, plus shot noise | Device-level sampling |
| `noisy_fidelity` | N inner products | Every noise layer | Fidelity to the ideal state |
| `qc.noisy_prob` | Path integral | Readout error correction | Single-outcome probabilities |

N is the effective `noise_realizations`. Its default, `None`, selects one
realization for deterministic density-matrix/MPO evolution. Sampled noise uses
64 realizations for execution and 100 for estimation/fidelity. An explicit
positive count is honored (execution caps it at `shots`). Noisy estimates also
compute an ideal baseline. {doc}`/examples/noise-modes` compares the analytical
and Monte Carlo modes on one circuit.

(py_noise_model)=
## Building a noise model

`NoiseModel` holds error rates per qubit. Pauli noise applies after every gate
that touches the qubit. Here each of four flipped qubits gets a different
channel, and ⟨Z⟩ shows which ones the channel can disturb:

```{code-cell} python
nm = maestro.NoiseModel()
nm.set_depolarizing(qubit=0, p=0.01)  # X, Y and Z each with probability p/3
nm.set_dephasing(qubit=1, p=0.02)  # Z errors, which leave <Z> alone
nm.set_bit_flip(qubit=2, p=0.01)  # X errors
nm.set_qubit_noise(qubit=3, px=0.005, py=0.002, pz=0.01)  # any Pauli channel

flipped4 = QuantumCircuit()
for q in range(4):
    flipped4.x(q)
print(maestro.noisy_estimate(flipped4, ["ZIII", "IZII", "IIZI", "IIIZ"], nm)["expectation_values"])
```

`set_all_depolarizing(num_qubits, p)` sets the same depolarizing rate on every
qubit.

(py_circuit_noise)=
## Noise at a point in the circuit

A Pauli channel can also sit at a fixed point in the circuit, like a Stim noise
instruction. `probabilities` has `4**len(qubits)` entries; base-4 digit *k* of an
index selects I, X, Y or Z on `qubits[k]`.

```{code-cell} python
p = 0.01
channel = QuantumCircuit()
channel.h(0)
channel.cx(0, 1)
channel.pauli_channel([0], [1 - p, p / 3, p / 3, p / 3])  # DEPOLARIZE1(p)
channel.pauli_channel([0, 1], [1 - p] + [p / 15] * 15)  # DEPOLARIZE2(p)
channel.measure_all()

# Statevector and MPS draw one Pauli per noise realization.
counts = channel.full_noise_execute(maestro.NoiseModel(), shots=1000, noise_realizations=1000)["counts"]
print(counts)
```

To keep a basis change that belongs to a measurement or reset out of a
NoiseModel's gate noise, pass `noiseless=True` to `h`, `s` or `sdg`.

(py_noise_mc)=
## Gate-by-gate Monte Carlo

`noisy_estimate_montecarlo` injects random Pauli errors after each gate and
averages over many noisy runs, so noise accumulates with circuit depth.
`noisy_execute` does the same for sampled counts:

```{code-cell} python
mc = maestro.noisy_estimate_montecarlo(
    bell, ["ZZ", "XX"], noise, noise_realizations=200, noise_seed=42
)
print("Monte Carlo:", mc["expectation_values"])

measured = QuantumCircuit()
measured.h(0)
measured.cx(0, 1)
measured.measure_all()
print(maestro.noisy_execute(measured, noise, shots=1024, noise_realizations=64, noise_seed=42)["counts"])
```

In the estimate functions, `noise_realizations` is the number of independent
noise samples averaged. In the execute functions, it is the number of batches
the shots are split into, so at most `shots` batches run. On QCSim,
realizations run on several worker threads at once, within `max_simulators`
and the `trajectory_max_memory_mb` budget; each realization keeps its own
random streams, so seeded results do not depend on the worker count.

The execute functions also accept `return_samples=True`, which returns each
shot's record as a `(shots, clbits)` `uint8` array under `samples` (see
{ref}`py_raw_samples`). With one realization per shot and no seed, the default
`counts_policy="auto"` samples Clifford circuits with Pauli, readout and
phase-damping noise through reference frames
(`sampling_strategy: "reference_frame_v1"`) and runs other realizations directly
(`"trajectory_v1"`); seeded calls keep their earlier sequences (see
{ref}`py_counts_policy`). {py:class}`maestro.PrefixCheckpointedSimulator`
simulates a shared noiseless prefix once and samples noisy suffixes from it; see
{doc}`/user-guide/expectation-values`.

(py_noise_coherent)=
## Coherent errors

Coherent noise applies deterministic rotations after gates instead of random
Pauli flips, modelling systematic over-rotation and phase drift:

```{code-cell} python
coherent = maestro.NoiseModel()
# Rotate by epsilon = 2*arcsin(sqrt(p)) about a body diagonal; its Pauli twirl is DEPOLARIZE1(p).
coherent.set_coherent_depolarizing(qubit=0, p=0.01)
coherent.set_coherent_dephasing(qubit=1, p=0.01)  # about Z
coherent.set_coherent_rotation(qubit=0, rx=0.01, ry=0.0, rz=0.05)  # explicit angles

print(maestro.coherent_estimate(bell, ["ZZ", "XX"], coherent, noise_realizations=100)["expectation_values"])
```

Each setter replaces only the component it names, so qubit 0 above has both the
diagonal rotation and the explicit angles.

(py_noise_thermal)=
## Thermal relaxation (T1 and T2)

Thermal relaxation models energy decay (T1) and dephasing (T2) from gate
durations:

```{code-cell} python
thermal = maestro.NoiseModel()
thermal.set_thermal_relaxation(qubit=0, gate_time_s=30e-9, t1_s=100e-6, t2_s=80e-6)  # 1-qubit gates
thermal.set_thermal_relaxation_2q(qubit=0, gate_time_s=300e-9, t1_s=100e-6, t2_s=80e-6)  # 2-qubit gates
print(thermal.has_thermal_relaxation())
```

T2 must be at most 2·T1. Density-matrix and supported MPO backends apply the
exact thermal channel. Sampled statevector and MPS runs approximate
T1 < T2 ≤ 2·T1 with an effective T2 = T1 and emit one `RuntimeWarning` naming
the affected qubits; use a density matrix for faithful evolution in that range.

(py_noise_cptp)=
## Exact channels (density matrix and MPO)

Some channels have no Pauli-trajectory form and need an exact mixed-state
backend, `DensityMatrix` or `MatrixProductOperator`:

```{code-cell} python
exact = maestro.NoiseModel()
exact.set_phase_damping(qubit=0, gamma=0.05)
exact.set_generalized_amplitude_damping(qubit=1, gamma=0.02, excited_population=0.05)
exact.set_correlated_phase_flip(q1=0, q2=1, probability=0.01, correlation=0.8)

# Any channel given as Kraus operators with sum_k E_k^† E_k = I, here a bit flip.
p_flip = 0.01
exact.set_kraus_channel([0], [
    [[math.sqrt(1 - p_flip), 0], [0, math.sqrt(1 - p_flip)]],
    [[0, math.sqrt(p_flip)], [math.sqrt(p_flip), 0]],
])

flipped = QuantumCircuit()
flipped.x(0)
flipped.cx(0, 1)
for method in ["DensityMatrix", "MatrixProductOperator"]:
    config = maestro.SimulatorConfig(simulation_type=getattr(maestro.SimulationType, method), fixed_backend=True)
    print(method, flipped.full_noise_estimate(["ZI", "IZ"], exact, config=config)["expectation_values"])
```

The statevector backend rejects these channels with a `ValueError` instead of
approximating them. `set_phase_damping_from_time(qubit, gate_time_s, t_phi_s)`
sets phase damping from a dephasing time.

With `noise_realizations=None` (the default), these Markovian channels evolve
once. Terminal measurement counts and raw rows are then sampled from the prepared
state, with independent classical readout errors. Unrecorded resets are also
deterministic channels on mixed-state backends, so they do not require shot replay.
Checkpoint suffixes use the same preparation-and-sampling path, as do terminal
checkpoint suffixes on QCSim statevector, stabilizer and MPS backends. QCSim supports
DM and MPO, Aer supports DM, and the Linux GPU backend supports DM and MPO. A CPU
MPO samples terminal shots on several workers by cloning its one prepared state,
with `fixed_backend=True` or with automatic selection.

Exact channel evaluation does not average every kind of noise. Coherent error
signs are drawn once per realization and held across the circuit; OU/1/f models
also sample disturbances. These retain multiple realizations on DM/MPO. Results
report `noise_realizations_requested`, `noise_realizations_policy`,
`noise_evaluation`, `stochastic_noise`, and `stochastic_execution` separately.
MPO truncation remains an approximation even with exact channels.

Recorded intermediate measurements and feedback require measurement branches.
Counts preserve each shot's classical history. For their expectation values,
`qc.estimate` averages 100 measurement trajectories on every backend, as native
`estimate` requests do (seeded runs derive one seed per trajectory), and
`qc.noisy_estimate_montecarlo(observables, maestro.NoiseModel(), config=config)`
averages branches with an explicit count, including the ideal baseline. The
analytical `qc.noisy_estimate` rejects such circuits: damping scales a single
deterministic expectation. Stochastic estimates report
`estimate_evaluation="trajectory_average"` and `realization_standard_errors`;
with only one realization the standard error is `None`, since it cannot be
estimated. On pure-state backends a reset that precedes every other operation on
its qubit leaves `|0>` and is not a branch; a later reset is.
`noisy_fidelity` requires a unitary ideal circuit (terminal measurements are
ignored; leading resets, including resets to `|1>`, and gate-only nested
circuits are allowed).

The current model injector does not attach gate noise to conditional gates or
recurse into nested circuits. Flatten composites before model injection, including
readout noise. Branch averaging preserves measurement and feedback behavior but
does not expand that noise-injection support.

`fixed_backend=True` pins the requested simulator and method. This is useful when
comparing DM/MPO behavior: automatic selection may otherwise use a reference-frame
sampler for eligible Clifford circuits. Sinter uses one realization for
deterministic mixed-state noise and retains independent per-shot disturbances for
its stochastic fallback paths. On DM/MPO its checkpoint also keeps in-circuit
Pauli channels and resets before the first measurement, and a deterministic
external noise model, so they evolve once per sampler. On pure-state backends the
checkpoint ends at a reset of a qubit already used, which may collapse a branch.
`PrefixCheckpointedSimulator(prefix, n, config, prefix_noise_model=model)` applies
such deterministic noise to a prefix (it raises `ValueError` otherwise); crosstalk
spectators outside the register stay in `|0>` and only contribute a phase, as for
suffixes. Dynamic checkpoint suffixes evolve their deterministic head once per
realization when all four retained states (the checkpoint's live state and saved
prefix, the copy and its saved head) fit `trajectory_max_memory_mb`, then replay
only their branches.

The C++ `noise::NoiseAdd` infers exact channels from the backend the network
will run: a fixed simulator, or the candidate an automatically selecting network
always takes (`ConfigureNetwork` with `fixed_backend=false` leaves only a
placeholder simulator beside it). Seeded or exact executions pin that candidate.

(py_noise_crosstalk)=
## Crosstalk and two-qubit errors

```{code-cell} python
pairwise = maestro.NoiseModel()
pairwise.set_crosstalk(q1=0, q2=1, strength=0.005)  # ZZ crosstalk between neighbours
pairwise.set_2q_depolarizing(q1=0, q2=1, p=1.8e-3)  # after two-qubit gates such as CX or CZ
print(pairwise.has_crosstalk(), pairwise.has_any_2q_depolarizing())
```

Crosstalk rotates neighbours of each noisy gate's qubits. A neighbour the
circuit never touches stays in `|0>`, where the rotation is a global phase:
Maestro simulates only the circuit's qubits, and count keys are padded with
zeros up to the highest spectator, as in earlier releases.

(py_noise_readout)=
## Readout error

Readout error flips the classical result when a measurement writes its bit.
Rates follow the measured qubit, and later classical conditions see the noisy
result:

```{code-cell} python
readout = maestro.NoiseModel()
readout.set_readout_error(qubit=0, p_meas1_prep0=0.003, p_meas0_prep1=0.06)  # asymmetric
readout.set_readout_error(qubit=1, p_meas1_prep0=0.003, p_meas0_prep1=0.06)

ones = QuantumCircuit()
ones.x(0)
ones.x(1)
ones.measure_all()
print(maestro.noisy_execute(ones, readout, shots=2000, noise_seed=7)["counts"])

# Probability of reading "11" once readout error is included, from the path integral
print(bell.noisy_prob("11", readout)["probability"])  # 0.5 * (1 - 0.06)**2
```

`noisy_prob` expands to first order in the readout error rates, so outcomes two
flips away from `"11"`, such as `"00"`, don't contribute.

`set_all_readout_error(num_qubits, p_error)` sets a symmetric rate on every
qubit.

The `noise_seed` argument (also accepted as `seed`, including by
`PrefixCheckpointedSimulator.execute_suffix`) is an unsigned 32-bit integer. It
seeds noise injection and, unless `SimulatorConfig.seed` is set, measurement and
readout randomness as well. An
explicit config seed takes precedence for simulator randomness. Without a seed,
each call uses fresh random streams. Each noise realization derives its own
simulator seed, so seeded runs are reproducible without repeating readout draws
across batches. `coherent_execute` follows the same rules.

(py_noise_correlated)=
## Time-correlated dephasing

Real hardware has non-Markovian phase drift from background fluctuators.
Maestro models it as Ornstein-Uhlenbeck (OU) processes, discretized as AR(1)
noise after each gate. `sigma` is the noise strength and `alpha` the
correlation time in units of the gate time:

```{code-cell} python
drift = maestro.NoiseModel()
drift.set_correlated_ou(qubit=0, sigma=15.0, alpha=0.5, gate_time=100e-9, stationary_init=True)
drift.set_all_correlated_ou(num_qubits=5, sigma=15.0, alpha=0.5, gate_time=100e-9)
print(drift.has_correlated())
```

(py_noise_multiband)=
### Several bands and 1/f noise

Superpose independent OU bands on a qubit to reproduce a 1/f spectrum over
several decades of frequency:

```{code-cell} python
spectrum = maestro.NoiseModel()
spectrum.set_correlated_ou_band(qubit=0, sigma=10.0, alpha=0.1, gate_time=100e-9)
spectrum.set_correlated_ou_band(qubit=0, sigma=5.0, alpha=1.0, gate_time=100e-9)
spectrum.set_multi_correlated_ou(qubit=1, bands=[(10.0, 0.1), (5.0, 1.0)], gate_time=100e-9)
spectrum.set_1_over_f_noise(
    qubit=2, total_power=1e-3, f_min=1e3, f_max=1e7, num_bands=6,
    gate_time=100e-9, stationary_init=True,
)
print(spectrum.has_correlated())
```

(py_noise_idle)=
## Idle noise during delays

`set_idle_noise` configures relaxation, dephasing and coherent detuning that act
during `qc.delay(qubit, duration)`. Active OU processes also evolve through the
delay:

```{code-cell} python
idle = maestro.NoiseModel()
idle.set_idle_noise(
    qubit=0,
    t1=100e-6,  # seconds
    t2=80e-6,  # at most 2 * t1
    excited_population=0.01,
    detuning_hz=5e3,
)

ramsey = QuantumCircuit()
ramsey.h(0)
ramsey.delay(0, 20e-6)
ramsey.h(0)
ramsey.measure_all()
print(ramsey.full_noise_execute(idle, shots=1024, seed=3)["counts"])
```

(py_noise_combined)=
## Every layer together

`full_noise_execute`, `full_noise_estimate` and `noisy_fidelity` apply every
configured layer in physical order:

```{code-cell} python
device = maestro.NoiseModel()
device.set_all_correlated_ou(num_qubits=2, sigma=15.0, alpha=0.5, gate_time=100e-9)
device.set_all_coherent_depolarizing(num_qubits=2, p=0.002)
device.set_all_readout_error(num_qubits=2, p_error=0.01)

print("counts:", measured.full_noise_execute(device, shots=1024, noise_realizations=64, seed=5)["counts"])
print("values:", bell.full_noise_estimate(["ZZ", "XX"], device, noise_realizations=100, seed=5)["expectation_values"])
fid = bell.noisy_fidelity(device, noise_realizations=100, seed=5)
print(f"fidelity: {fid['fidelity']:.4f} +/- {fid['std_error']:.4f}")
```

## Error correction with Stim and Sinter

To sample Stim circuits through Maestro, including from `sinter collect` with a
decoder, see {doc}`qec-sinter`.

:::{toctree}
:hidden:

qec-sinter
:::
