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
realization for deterministic density-matrix/MPO evolution, and for Pauli-type
noise on the QCSim Pauli propagator (see {ref}`py_noise_cptp`). Sampled noise uses
64 realizations for execution and 100 for estimation/fidelity. An explicit
positive count is honored (execution caps it at `shots`). Noisy estimates also
compute an ideal baseline. {doc}`/examples/noise-modes` compares the analytical
and Monte Carlo modes on one circuit.

The statevector and the matrix product state draw the noise that depends on the
state as each shot runs (see {ref}`py_noise_sampled_channels`). With such noise
every shot is a realization of its own, and `noise_realizations` reports the
shot count whatever was requested.

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
random streams, so seeded results do not depend on the worker count. The
result reports how many workers ran the realizations as `realization_workers`.

The execute functions also accept `return_samples=True`, which returns each
shot's record as a `(shots, clbits)` `uint8` array under `samples` (see
{ref}`py_raw_samples`). With one realization per shot, the default
`counts_policy="auto"` samples Clifford circuits with Pauli, readout and
phase-damping noise through reference frames
(`sampling_strategy: "reference_frame_v1"`) and runs other realizations directly
(`"trajectory_v1"`). When several shots share one realization, eligible terminal
histograms can use multinomial sampling with or without a seed (see
{ref}`py_counts_policy`). Eligible `extended_stabilizer` runs with Pauli,
readout and phase-damping noise use batched Pauli frames under any realization
schedule (`sampling_strategy: "extended_frame_v1"`). A batch of shots shares one
Clifford basis, and shots of one realization share its errors. Some runs keep
scalar trajectories:

- circuits with classical control;
- non-Pauli noise such as amplitude damping;
- thermal relaxation with T2 > T1 on a qubit the circuit relaxes;
- relaxation that sampled execution draws as resets when shots share a
  realization: T1, or thermal or idle relaxation with a finite T1, after a gate
  (or, for idle relaxation, a delay) on that qubit.

Which of these a trajectory runs exactly depends on the method: see
{ref}`py_noise_sampled_channels`.

These automatic paths support both seeded and unseeded requests.
`counts_policy="frame_v1"` explicitly requests eligible frames. As for scalar runs, `noise_seed`
drives the circuit errors and `config.seed` the measurements and readout, on
reference frames as on batched frames. Frames share `trajectory_max_memory_mb`
among their batches, and their rows depend on neither the budget nor
`max_simulators`. Like the trajectories' budget, it is soft: a shot that alone
exceeds it runs by itself beyond it, with the draws it would have had.
Reference frames keep trajectories for thermal relaxation with T2 > T1.
{py:class}`maestro.PrefixCheckpointedSimulator`
simulates a shared noiseless prefix once and samples noisy suffixes from it; see
{doc}`/user-guide/expectation-values`. On `extended_stabilizer`, after a
gate-only prefix, its noisy suffixes use the same frames.
`maestro.compile_reference_sampler` with an `ExtendedStabilizer` config returns a
reusable `ExtendedFrameSampler`, which `maestro.sinter` uses for such tasks. Its
`noise_realizations` below the shot count raises `ValueError` when the circuit's
noise includes relaxation that `noisy_execute` draws as a reset. Its memory
budget is soft, as `noisy_execute`'s. Compiled samplers apply channels exactly,
so where `noisy_execute` clamps T2 to T1 (see {ref}`py_noise_thermal`) they keep
T2. `noisy_execute` warns about the clamp only for qubits whose relaxation the
circuit draws (a gate, or for idle noise a delay, on them).

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
exact thermal channel. The statevector and the matrix product state sample it
from their state, one branch per shot, and keep the calibrated T2 (see
{ref}`py_noise_sampled_channels`).

The other pure-state methods (tensor network, path integral, stabilizer,
extended stabilizer, and the GPU and QuEST backends) run realizations made of
gates and resets only. For T2 ≤ T1 that is exact: the channel is a mixture of
nothing, a phase flip and a reset to the thermal state. They approximate
T1 < T2 ≤ 2·T1 with an effective T2 = T1 and emit one `RuntimeWarning` naming
the affected qubits; use one of the methods above for faithful evolution in
that range.

(py_noise_cptp)=
## Channels beyond Pauli noise

Some channels are no mixture of Pauli errors: amplitude damping, generalized
amplitude damping, correlated phase flips, and channels given by their Kraus
operators. The mixed-state methods, `DensityMatrix` and `MatrixProductOperator`,
apply them exactly:

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

`set_phase_damping_from_time(qubit, gate_time_s, t_phi_s)` sets phase damping
from a dephasing time.

(py_noise_sampled_channels)=
### Sampled from the state: statevector and matrix product state

A pure state holds no ensemble, so the statevector and the matrix product state
take a channel as one sampled branch, a quantum trajectory. For every channel a
shot applies one Kraus operator, chosen with the probability its state gives
it, and the shots average to the channel. Nothing is approximated: `set_t1` is
amplitude damping, thermal and idle relaxation keep their T2, and the model
above runs as it is, with no extra qubit:

```{code-cell} python
measured = QuantumCircuit()
measured.x(0)
measured.cx(0, 1)
measured.measure_all()
for method in ["Statevector", "MatrixProductState"]:
    config = maestro.SimulatorConfig(simulation_type=getattr(maestro.SimulationType, method), seed=7)
    result = measured.full_noise_execute(exact, config=config, shots=2000, noise_seed=11)
    print(method, result["noise_evaluation"], result["noise_realizations"], sorted(result["counts"].items()))
```

This holds for the QCSim and Qiskit Aer statevector and matrix product state,
and for the composite statevector, in the execute and estimate functions,
`noisy_fidelity`, checkpoint suffixes and native requests.

- **Every shot is a trajectory.** The execute functions and native `execute`
  requests report the shot count as `noise_realizations` (checkpoint suffixes
  keep their realization count: each of their shots still draws its own
  branches), and `noise_seed` and `config.seed` together determine the
  branches. A noise
  layer drawn per realization (coherent signs, OU and 1/f disturbances) keeps
  the realization schedule: each realization then holds the channels, and its
  shots draw their branches, from the two seeds together there as well.
- **What needs no state is drawn first.** Pauli and depolarizing parts, mixtures
  of unitaries, and relaxation with T2 ≤ T1 are drawn without reading the state.
  Only the rest reads it: relaxation with T2 > T1, amplitude damping, and Kraus
  channels.
- **A channel with one branch is a gate.** A mixture with a single unitary, a
  Pauli channel with one term or a single Kraus operator, such as a coherent
  over-rotation given as a channel, has nothing to draw
  (`QuantumChannel.is_deterministic`). So has a damping of zero, which is the
  identity: a Kraus operator that is zero is no branch. A circuit whose channels are all of that
  kind evolves once for all its shots, an estimate of it is one evolution, and
  the shots of a realization share it.
- **Statevector.** One pass over the state gives the reduced density matrix of
  a channel's qubits, and with it the probability of every operator. With gate
  fusion, a block of gates and channels on up to three qubits is resolved from
  one such read and applied as one matrix. Where the channels of a block most
  probably do nothing, the block is applied first and the norm of the result
  tells whether that was right, so the state is read only when something
  happened. That shortcut needs a state of known norm: after a generic matrix
  that is not unitary, applied through a simulator handle, blocks are read
  first.
- **Qiskit Aer.** Aer draws the branches that need the state itself, from the
  stream that also draws its measurement outcomes. Adding such a channel to a
  seeded circuit therefore changes the outcomes of the measurements after it,
  where on the QCSim methods the branches have a stream of their own and the
  measurements around a channel keep theirs. The statistics are the same.
- **Matrix product state.** After an operator that is not unitary the chain
  keeps the environments of its bonds explicitly instead of being swept back
  into canonical form, so a channel costs a few matrix products and a two-qubit
  gate is truncated as in the canonical form. Two-qubit Kraus channels act on
  any pair of qubits.
- **Chain order.** Those environments cost one product per site between two
  operations that read them. When every two-qubit gate and channel of a circuit
  acts on neighbouring qubits, so that nothing has to be routed, the QCSim
  matrix product state runs the operations in an order that stays close along
  the chain. Operations on different qubits commute, so it is the same circuit;
  only the order in which the channels draw their branches changes, and with it
  a seeded trajectory, not the distribution. The QCSim matrix product operator
  does the same for its exact channels: there the result is the same up to
  truncation, which acts in another order. A circuit laid out on a line of a
  device counts as well, although the indices of its qubits are not in order:
  when the two-qubit operations give every qubit at most two partners and close
  no ring, the qubits are placed along the lines they form, which needs no
  swap, and the order is taken along them (from 12 qubits on the state, 6 on
  the operator; smaller registers run as listed). Circuits that need routing
  run as listed.
- **Composite statevector.** A one-qubit channel is drawn in the component that
  holds its qubit; only a two-qubit unitary or Kraus branch joins components.

The remaining pure-state methods neither hold nor sample channels. A noise
realization draws for them every channel whose branches need no state (a
Clifford branch becomes H and S gates, which the stabilizer methods run),
approximates `set_t1` by a reset with probability gamma and relaxation with
T2 > T1 by T2 = T1, and raises `ValueError` for generalized amplitude damping,
correlated phase flips and Kraus channels.

(py_noise_channel_instructions)=
### Channels as circuit instructions

`maestro.QuantumChannel` describes one channel, and `QuantumCircuit.channel`
places it on one or two qubits. Build each distinct channel once: the circuit
shares the object. `then` composes channels and keeps their parts, so the error
of a two-qubit gate can be given as it is calibrated, a depolarizing error
followed by relaxation of each qubit:

```{code-cell} python
Channel = maestro.QuantumChannel
relaxation = Channel.thermal_relaxation(duration=300e-9, t1=100e-6, t2=120e-6)
gate_error = Channel.depolarizing(0.01, num_qubits=2).then(relaxation, [0]).then(relaxation, [1])

noisy_bell = QuantumCircuit()
noisy_bell.h(0)
noisy_bell.channel([0], Channel.amplitude_damping(0.02))
noisy_bell.cx(0, 1)
noisy_bell.channel([0, 1], gate_error)
noisy_bell.measure_all()

print([part["kind"] for part in gate_error.factors])
for method in ["DensityMatrix", "Statevector"]:
    config = maestro.SimulatorConfig(simulation_type=getattr(maestro.SimulationType, method), seed=3)
    print(method, sorted(noisy_bell.execute(config=config, shots=2000)["counts"].items()))
```

A mixed-state method applies each instruction exactly. The statevector and the
matrix product state draw a branch of it in every shot, in a plain `execute` as
under a noise model. The other pure-state methods need a noise function
(`noisy_execute`, `full_noise_execute`), whose realizations draw the
instruction, and accept only channels whose branches need no state. Other
factories are `pauli`, `unitary_mixture`, `phase_damping`,
`generalized_amplitude_damping`, `from_kraus` and `on_qubit`; for two-qubit
matrices `qubits[0]` is the least significant bit of the index.

The QCSim Pauli propagator (`SimulationType.PauliPropagator`) also evaluates
channels exactly, without a density matrix: it applies the adjoint of each
channel to the observable it propagates. One execution then gives the exact
ensemble average, and the model above runs on it too. Which models use this
automatically:

- Pauli-type noise (Pauli and depolarizing channels, dephasing, phase damping,
  correlated phase flips) is diagonal in the Pauli basis. Each channel only
  scales terms, so the default is one exact realization instead of 64 or 100
  sampled ones.
- T1, thermal and idle relaxation that acts on the circuit keeps its sampled
  realizations on the propagator. Relaxation also sends Z to the identity, and
  without a truncation threshold those branches fill the operator space of the
  observable's light cone (`4^width` terms), which costs far more than the
  trajectories.
- Relaxation counts by the channel it applies. A zero T1 probability, thermal
  or idle relaxation towards equal populations (`excited_population=0.5`), and
  relaxation on qubits the circuit never gates (or idle relaxation without a
  delay) only scale terms or add nothing, so they stay exact, with the
  calibrated T2.
- A model with generalized amplitude damping or Kraus channels, which the
  propagator cannot sample, runs exactly as a whole.

Exact here means without a positive `pp_coefficient_threshold` and without a
`pp_max_pauli_weight` below the register; with either the result is an
approximation, as any truncated propagation is.

With `noise_realizations=None` (the default), these Markovian channels evolve
once. Terminal measurement counts and raw rows are then sampled from the prepared
state, with independent classical readout errors. Unrecorded resets are also
deterministic channels on mixed-state backends and on the QCSim Pauli propagator,
so they do not require shot replay.
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
suffixes. The prefix's own channel instructions follow the same rule: the
statevector and the matrix product state would draw one branch of them for every
suffix shot to share, so a prefix that holds channel instructions with a branch
to draw raises `ValueError` there; put those channels in the suffixes. A channel
with one branch is a gate and may stay in the prefix. Only suffixes are sampled,
so a prefix runs without its end measurements: one that is a whole program, as
a circuit parsed from OpenQASM is, does not collapse the saved state. A
measurement that later operations of the prefix follow or read runs, and
collapses once for every suffix shot. Dynamic checkpoint suffixes evolve their deterministic head once per
realization when all four retained states (the checkpoint's live state and saved
prefix, the copy and its saved head) fit `trajectory_max_memory_mb`, then replay
only their branches.

The C++ `noise::NoiseAdd` infers exact channels from the backend the network
will run: a fixed simulator, or the candidate an automatically selecting network
always takes (`ConfigureNetwork` with `fixed_backend=false` leaves only a
placeholder simulator beside it). Seeded or exact executions pin that candidate.
Its `seed` and a seed configured on the network together determine the branches
of the channels a backend samples from its state, as `noise_seed` and
`config.seed` do in Python: either seed alone changes them.

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
configured layer in physical order. After each gate, configured channels act in
order: correlated dephasing, coherent rotations, crosstalk on neighbour qubits,
relaxation and damping (T1, thermal relaxation and phase damping) on each gate
qubit, single-qubit Pauli noise, exact Kraus/CPTP channels, and two-qubit
depolarizing on two-qubit gates; `noisy_execute` and `noisy_estimate_montecarlo`
apply the incoherent subset in the same order:

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
