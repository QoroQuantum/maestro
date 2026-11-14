# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `execution_metadata.realization_workers` is also reported for shot rows
  that realization workers drew. Noisy `estimate` and `noisy_fidelity`
  requests, which run their realizations one after another, report 1. The
  Python noisy execute, estimate and fidelity functions report the number as
  `realization_workers` in their result.
- `maestro.build_info()` returns the identity of the build: `version`,
  `source_revision` and `qcsim_revision`, the Git revision of the QCSim
  sources it was compiled against. `maestro.__version__` is the version. The
  native capabilities report the QCSim revision as `build.qcsim_revision`.
- Noise channels on pure states, sampled from the state. The QCSim and Qiskit
  Aer `statevector` and `matrix_product_state` and the composite statevector
  take a channel as one sampled branch, a quantum trajectory: each time the
  circuit runs, a channel applies one of its Kraus operators with the
  probability the state gives it, on the circuit's own qubits. Amplitude
  damping (`set_t1`), generalized amplitude damping, thermal and idle
  relaxation with T2 > T1, correlated phase flips and Kraus channels therefore
  run as they are on these methods, where trajectories approximated them (T1
  as a reset, T2 clamped to T1) or the call was rejected. This holds in
  `execute`, `estimate`, the noisy execute and estimate functions,
  `noisy_fidelity`, checkpoint suffixes and native requests. With such noise
  every shot is a realization of its own: results report the shot count as the
  number of realizations, and the noise seed and the simulator seed together
  determine the draws, so either alone changes them. A noise layer drawn per
  realization beside the channels (coherent signs, OU or 1/f disturbances)
  keeps the realization schedule, and the two seeds together determine the
  branches there as well. The C++ `NoiseAdd` follows the same rule: its
  `seed` joins a seed configured on the network for those branches, where the
  configured seed alone decided them. On a 24-core desktop with
  a device-like noise model, these trajectories ran 1.6 to 5.3 times as fast as
  Qiskit Aer's on the statevector at 8 to 20 qubits, and took 0.25 s, 2.6 s and
  18.6 s on the matrix product state at 8, 12 and 16 qubits (bond limit 64, 200
  shots) where Aer took 2.5 s, 25 s and 320 s.
- `maestro.QuantumChannel` describes one channel on one or two qubits (in C++,
  `Simulators::QuantumChannel`). Factories: `pauli`, `depolarizing`,
  `amplitude_damping`, `phase_damping`, `generalized_amplitude_damping`,
  `thermal_relaxation`, `unitary_mixture`, `from_kraus`, `on_qubit` and
  `placed`. `then` composes channels and keeps their parts (`factors`), so a
  trajectory draws a Pauli or unitary-mixture part without reading its state.
  `kraus_operators`, `superoperator`, `choi`, `is_state_independent`,
  `is_deterministic` and `is_approx` describe a channel.
  `QuantumCircuit.channel(qubits, channel)` places it in a circuit, which
  shares the object with every instruction that uses it.
- Native requests accept a `channel` circuit instruction: a structured channel
  given by its `factors` (`pauli`, `unitary_mixture`, `kraus`, `depolarizing`,
  `amplitude_damping`, `phase_damping`, `generalized_amplitude_damping`,
  `thermal_relaxation`), the request form of `QuantumChannel`. The
  capabilities list the instruction and its `channel_factors`.
- Simulator handles apply a channel directly: `ApplyQuantumChannel` (also
  `apply_channel`) in Python and `MaestroApplyKrausChannel` in the C ABI.
  `SupportsQuantumChannels` says whether the simulator applies it exactly and
  `SamplesQuantumChannels` whether it applies one sampled branch
  (`MaestroSupportsQuantumChannels`, `MaestroSamplesQuantumChannels`).
  `MaestroGetTrajectoryBlockStatistics` reports how many fused blocks of
  sampled channels a statevector applied before reading its state.

### Changed

- The density matrix and the MPO apply a channel as one superoperator in one
  pass, and with gate fusion a channel and the gates around it on up to two
  qubits become one superoperator. A 12-qubit density-matrix run with a
  device-like noise model (1,000 shots, 24-core desktop) took 3.6 s, against
  5.4 s for Qiskit Aer.
- With gate fusion, the statevector resolves a block of gates and sampled
  channels on up to three qubits from one read of their reduced density matrix
  and applies it as one matrix. Where the channels of a block most probably do
  nothing, the block is applied first and the state is read only when
  something happened.
- After an operator that is not unitary, the QCSim matrix product state keeps
  the environments of its bonds instead of being swept back into canonical
  form, so a channel costs a few matrix products. Two-qubit Kraus channels act
  on any pair of qubits.
- On the QCSim `matrix_product_state` and `matrix_product_operator`, a circuit
  whose two-qubit gates and channels all act on neighbouring qubits runs in an
  order that stays close along the chain once a channel has acted. It is the
  same circuit; on the matrix product state a seeded trajectory depends on
  that order. A circuit laid out on a line whose qubit indices are not in
  order, such as a line of a device, is placed along that line without swaps
  (from 12 qubits on the state and 6 on the operator).
- The pure-state methods that neither hold nor sample channels (tensor
  network, path integral, stabilizer, extended stabilizer, and the GPU and
  QuEST backends) draw, in a noise realization, every channel whose branches
  need no state: Pauli channels, one-qubit mixtures of unitaries and
  relaxation with T2 <= T1, also as circuit instructions. They still
  approximate `set_t1` and relaxation with T2 > T1, and reject generalized
  amplitude damping, correlated phase flips and Kraus channels.
- A channel with one branch only, a unitary (`QuantumChannel.is_deterministic`),
  is a gate: nothing is drawn for it, and a circuit whose channels are all of
  that kind evolves once for all its shots. A damping of zero
  (`amplitude_damping(0)`, `generalized_amplitude_damping(0, p)`) is one: it
  is the identity, and a Kraus operator that is zero is no branch.
- Channel instructions run on `statevector` and `matrix_product_state` in a
  plain `execute` or `estimate`, with or without a noise model. A query of one
  state (`get_statevector`, `get_probabilities`, and the native `statevector`,
  `amplitudes`, `probabilities` and `state_probability` requests) returns the
  state of one branch and reports `stochastic_execution`.
- `noisy_fidelity` accepts channel instructions in its circuit, as noise the
  ideal circuit does not have, and a reset before any other operation on its
  qubit.
- A checkpoint prefix that holds channel instructions with a branch to draw
  raises `ValueError` on a method that samples channels, since every suffix
  shot would share the one branch drawn: put those channels in the suffixes. A
  channel with one branch is a gate and may stay in the prefix.
- A checkpoint prefix follows one set of rules in `PrefixCheckpointedSimulator`
  and in `checkpoint_batch` requests. A request's prefix took gates only
  (`checkpoint_batch prefix requires a unitary circuit`); it now takes resets,
  channel instructions (by the rule above) and measurements that later
  operations of the prefix follow or read, which run once. The end measurements
  of a prefix are removed in both, since only suffixes are sampled: a prefix
  that is a whole program, as one parsed from OpenQASM is, no longer collapses
  the saved state in `PrefixCheckpointedSimulator`, where it did once for every
  suffix shot. Plain execution keeps its measurements.
- `noisy_execute` warns that T2 is clamped to T1 only for qubits whose
  relaxation the circuit draws.
- Seeded noisy results on the statevector, the matrix product state and the
  composite statevector differ from 0.3.14, since the noise is drawn in
  another way.

### Fixed

- `get_probabilities` answers on every simulation type. It squared the
  statevector, so it raised where a state has none to read: a density matrix,
  a matrix product operator, a tensor network, the Pauli propagator and the
  extended stabilizer, on all of which a native `probabilities` request
  answers. On the mixed-state methods it returns the distribution of the
  state that channel instructions leave. Python and the request now read the
  probabilities through one function (`Simulators::BasisProbabilities`;
  Python reaches it through the new `INetwork::ExecuteOnHostProbabilities`),
  and neither runs the circuit's end measurements.
- With sampled noise on a method that samples no channels (tensor network,
  path integral, stabilizer and the like), a one-qubit mixture of unitaries
  among the circuit's channel instructions was counted as drawn from the
  state while the circuit runs. It is drawn into a gate when the realization
  is built, as a Pauli channel is, and counts as one now: `execute`,
  `estimate` and checkpoint suffixes no longer report `stochastic_execution`
  for it, in Python and in requests, and on the path integral its realizations
  are scheduled as the prepared states they are. A relaxation still counts,
  since it may be drawn into a reset, which reads the state.


## [0.3.14] - 2026-11-11

### Added

- `SimulatorConfig` exposes the Pauli propagation tuning options that native
  requests already accept: `pp_workers`, `pp_max_fused_gates`,
  `pp_memory_limit_bytes` and `pp_sampling_cache_nodes`. Each defaults to `None`,
  which keeps the backend default. `pp_workers` lets GPU PauliPropagator estimates
  spread independent observables across up to 64 workers, each on its own CUDA
  stream; before this change, Python estimates always ran with one GPU worker.
  Out-of-range values raise `ValueError` when the config is built or a field is
  set, using the same limits as native requests.

- The QCSim `pauli_propagator` evaluates noise channels exactly in one
  execution. It applies the adjoint of each channel to the observable it
  propagates, so expectation values, probabilities and samples describe the
  noisy ensemble without trajectories. Pauli-type noise (Pauli, depolarizing,
  dephasing, phase damping, correlated phase flips) now resolves to one exact
  realization there instead of 64 or 100 sampled ones. Generalized amplitude
  damping, Kraus channels and circuit `kraus`/`pauli_channel` instructions,
  previously rejected on this method, are accepted. `noisy_fidelity` requests
  run on it up to 24 qubits, and a deterministic noisy checkpoint prefix
  (`prefix_noise_model`, `noise.prefix`) is evolved once on it.
  `noise.exact_configurations` in the capabilities lists it.
- Native `execute` results report `execution_metadata.realization_workers`,
  the number of workers that ran the noise realizations.

- Fast noisy sampling, shot rows and parallel realizations (`Execution/ReferenceFrameSampler`,
  `Execution/SampleRecords`, `Execution/NoisyEstimation`, `Execution/PrefixCheckpoint`,
  `Utils/Threads/RealizationWorkers`):
  - Reference-frame (Pauli-frame) sampler for noisy Clifford circuits, used for
    counts under `counts_policy` (`auto`, `legacy`, `frame_v1`), for shot rows,
    and as a reusable compiled sampler (`compile_reference_sampler` /
    `ReferenceFrameSampler` in Python and `MaestroCreateReferenceFrameSampler`
    in the C ABI). Pauli-mixture Kraus channels such as phase damping compile to
    frames.
  - Shot rows (`return_samples=True` in Python, `execution.return_samples` in the
    native JSON API, and `MaestroRunRequestSamples` in the C ABI) returning
    per-shot classical bit matrices.
  - Direct trajectory sessions, a precompiled `NoiseInjectionPlan`, and parallel
    realization workers sharing `trajectory_max_memory_mb`; seeded results do
    not depend on the worker count.
- Batched Pauli-frame sampling for noisy QCSim `extended_stabilizer` circuits
  (`ExtendedFrameBatch`, `ExtendedFrameSampler`, `extended_frame_v1`), exposed
  through `ExecuteNoisy`, `PrefixCheckpoint`, native `execute` and
  `checkpoint_batch`, `compile_reference_sampler`, and `maestro.sinter`.
- Adaptive multinomial terminal count sampling (`Utils/Sampling/Multinomial.h`,
  `counts_policy="multinomial_v1"`, and automatic unseeded selection under
  `counts_policy="auto"`), drawing terminal histograms directly from outcome
  probabilities via sequential binomial splits with rare-outcome grouping on
  QCSim statevectors, density matrices, path integrals, 1-qubit MPS/MPO
  marginals and up to 8-bit extended-stabilizer marginals.
- Automatic realization policy (`Execution/NoisePolicy.h`) and prepared-state
  reuse (`Execution/TerminalSampling.h`):
  - Omitted `noise_realizations` (`None` in Python; omitted, `null` or `"auto"`
    in JSON) resolves to 1 realization when exact density-matrix or MPO channels
    make evolution deterministic, and 64 (counts) or 100 (estimates and
    fidelity) otherwise.
  - Terminal measurements on mixed-state backends and eligible QCSim
    statevector, stabilizer and MPS runs sample one prepared state per
    realization (`terminal_v1` / `mixed_v1`), including checkpoint suffixes and
    exact prefix noise (`prefix_noise_model` / `noise.prefix`).
  - `SimulatorConfig.optimization_candidates` bound in Python.
- `INetwork::CaptureSimulatorSettings()` copies the settings made directly on
  the network's simulator into the network, for code that re-creates the
  simulator. Composer's parallel scheduler calls it, so a setting made only on
  the simulator (a bond cap) holds for every scheduled run, not just the
  first.

### Changed

- A seed fixes the random streams of a request and no longer selects its
  sampler. Under `counts_policy="auto"`, seeded requests take every path an
  unseeded one would: reference frames, extended-stabilizer frames, direct
  trajectory sessions, adaptive terminal sampling (for counts, shot rows and
  checkpoint suffixes) and parallel realization workers. A seeded request
  reproduces its result for the same request, seed, configuration and Maestro
  version. Seeded results differ from 0.3.13 and may change between versions.
  `counts_policy="legacy"`, `sampling_policy="legacy"` and
  `trajectory_policy="block_v1"` remain as algorithm choices, without the
  sequences of earlier releases.
- QCSim stabilizer, extended-stabilizer, MPS and MPO dynamic circuits default
  to `trajectory_policy="shot_v1"`, as the dense methods do, so their seeded
  counts no longer depend on the worker count. Their terminal sampling evolves
  the circuit once and draws fixed logical batches, also independently of the
  worker count.
- The native `sampling_policy="reproducible_v1"` option no longer disables
  adaptive multinomial counts; only `"legacy"` does.
- QCSim seeds its random engines directly instead of through `std::seed_seq`.
  Reseeding a simulator costs about 0.6 µs instead of 10 µs, which removes
  most of the per-shot cost of `shot_v1` trajectories.
- An unrecorded reset on the QCSim `pauli_propagator` is a deterministic
  channel: it draws no random number and inserts no projector. Circuits with
  mid-circuit resets no longer replay per shot or average measurement
  trajectories there, and seeded runs that contain a reset give different
  (still valid) samples than before.
- T1, thermal or idle relaxation that acts on the circuit keeps sampled
  realizations on the `pauli_propagator` under automatic evaluation, since
  exact relaxation without a truncation threshold fills the operator space of
  the observable's light cone. `evaluation: "exact"` (requests) or
  `NoiseAdd::set_exact_channels(true)` applies it exactly, with the calibrated
  T2. Relaxation is judged by the channel it applies: a zero T1 probability,
  relaxation towards equal populations (`excited_population` 0.5) and
  relaxation on qubits the circuit never gates or delays are exact. A circuit
  `kraus` instruction beside relaxation makes the whole request exact.
- `checkpoint_batch` and `PrefixCheckpointedSimulator` suffixes on the QCSim
  `pauli_propagator` sample terminal measurements from the saved prefix
  instead of replaying every shot. Under `counts_policy` `auto`, seeded counts
  with readout error and seeded raw rows therefore differ from earlier
  releases (both remain valid samples). `legacy` keeps the per-shot replay.
- `amplitudes`, `statevector`, `inner_product` and `mirror_fidelity` (without
  observables) requests on `pauli_propagator` are rejected when the request is
  validated, with `unsupported_capability`, instead of failing in the backend.
- `noise.approximations` lists `Pauli_truncation` when a `pauli_propagator`
  request runs with a positive `pp_coefficient_threshold` or a
  `pp_max_pauli_weight` below the register.
- `noisy_execute` and `full_noise_execute` keep `counts_policy="auto"` for
  shared-realization count batches when the request may run `CompositeQCSim`
  (fixed, or in a candidate list), so its components sample them adaptively,
  as native requests already did. Only a realization that could itself take
  reference frames (Clifford, with the QCSim stabilizer listed) keeps the
  per-shot sampler. Seeded `auto` histograms of such runs change.
- The help of the noisy execute functions, the Qiskit migration guide and the
  errors for channels that cannot be sampled name the QCSim `pauli_propagator`
  beside density matrix and MPO.

- A network's simulator is documented and treated as a backend template
  (`INetwork::CreateSimulator`). Executions size their own register.
  `kBackendPlaceholderQubits` (2) is the width to create a backend that only
  selects execution, and configured backends are created at that width.
- Without a cost model, a configured candidate list decides automatic
  selection. The network simulator's backend runs first when it is listed,
  then the others in the order they were added, on the register the circuit
  maps to. A simulator outside a configured list never runs. The list a
  network starts with is not configured, so its simulator still decides.
  `GetOptimizationCandidates()` returns the configured list. Either way, and
  with or without a seed, a Clifford circuit may move to a stabilizer backend
  the list admits (QCSim's or Aer's for the default list); this is now
  documented.

### Fixed

- Count-only checkpoint suffixes no longer build a permutation of all shots
  (8 bytes per shot), whatever the counts policy. With deterministic readout
  they add each occupied outcome once; with readout errors they add the shots
  one at a time, drawing the errors in the order the row path draws them.
- An explicitly seeded classical random instruction replays its bits in an
  unseeded request; only unseeded instructions draw fresh entropy.
- Realization workers honor `max_simulators` when it is given as a native
  option (`simulator.options` in native requests), as they do the typed setting.
- `noisy_fidelity` and `PrefixCheckpointedSimulator.execute_suffix` decide the
  "T2 clamped to T1" warning from the backend a candidate list resolves to.
  A list such as density matrix then statevector ran exact channels and warned
  all the same, which failed the call under `warnings.simplefilter("error")`.
  `noisy_fidelity` with a candidate list, which it resolves inside the call,
  now warns after the run; every other call still warns before it.

- An exception thrown by a network's shot worker (a stabilizer meeting a T
  gate) terminated the process. `Utils::ThreadsPool` now keeps the first one,
  skips the jobs still queued and rethrows it from `WaitForFinish()`, so
  Python raises `RuntimeError` and a native request returns `native_failure`.
  A native request for a fixed `stabilizer` method whose circuit has a
  non-Clifford gate, after the network's optimizer (`T;T` runs as `S`), is
  rejected before any shot runs with `unsupported_capability`. Estimates and
  disjoint sampling, which run only the gates in their cones, keep the
  simulator's own error.
- Candidate lists were tested at the declared width. One active qubit of a
  40-qubit circuit now runs on a listed density matrix. A requested MPO with a
  lone density-matrix candidate no longer fails with a size overflow.
- `incremental_evolve` drives, and reports, the backend a candidate list
  resolves to, not the requested pair.
- Automatic GPU requests accept an explicit `gpu_device`, and their metadata
  reports the device that ran.
- Executing a network without a simulator no longer creates one. A failed run
  restores the network's state and keeps its simulator.
- Clones, and the noise helpers that run on them, keep settings made directly
  on the network's simulator. Execution clones no longer copy a large register,
  and noisy-run worker budgets count only the registers that stay allocated.
- `noise::NoiseAdd::last_execution()` reports the backend a noisy run used
  (Composer binds it as `NoiseAdd.last_execution()`). It reports nothing for
  a run that executed no network: zero shots, or observables that planning
  resolved.
- With a cost model, a configured candidate list still runs one of its own
  candidates when the model cannot rank the list, chooses nothing, or picks a
  candidate that cannot be created, allocate the register or run the start of
  the circuit (a statevector at 70 qubits falls back to a listed MPO).
- A network simulator created as a stabilizer no longer makes the cost model
  rank stabilizer backends for a non-Clifford circuit.
- A configured list whose candidates cannot run reports the first failure, in
  any order. Before, with QuEST not loaded, `[density matrix, QuEST]` (when the
  density matrix could not hold the register) and `[stabilizer, QuEST]` (for a
  non-Clifford circuit) ran the network's own simulator, outside the list.
- A zero-shot `RepeatedExecute` or `RepeatedExecuteOnHost` runs nothing and
  leaves the network's simulator, with its register and state, as it was. It
  reset them before.
- `Schedulers::ExecuteJob` keeps what its host network threw (`error`) instead
  of terminating the worker thread, so a scheduler can report it.
- A network that has a scheduler is freed when its last reference goes. The
  scheduler, which the network owns through its controller, held the network
  strongly, so neither was ever destroyed. `IScheduler` now holds it weakly, and
  `GetNetwork()` returns null once the network is gone.
- Selection never runs a listed stabilizer for a non-Clifford circuit: it
  counts as a failed candidate, after the others. When it is the only reason
  a list cannot run, the error is a runtime error, like the stabilizer's own
  rejection of a non-Clifford gate (Python `RuntimeError`, request API
  `native_failure`). The capabilities' `automatic_selection` states this rule
  and the first-failure rule. NetQASM and discrete-event
  networks, which do not run the circuit's start while selecting, installed it
  and failed only in the shots, reporting that error instead of the first
  candidate's. Selection also moves on when a candidate cannot run the start of
  the circuit, as it does when one cannot allocate the register.
- A run that throws leaves the network's simulator with the backend and
  creation width it had. NetQASM and discrete-event networks kept the backend
  selection had installed (a retry with automatic selection off then ran on
  it), and a simple network kept a cleared register. A simulator the failed run
  did not touch stays as it is, and an MPI distributed one is re-created only
  when the run left none, since that is collective across ranks.
- Seeded or exact-channel noise turns off the cost model on its execution
  clones (`SetCostModelEnabled`) instead of pinning one backend, so a listed
  candidate that fits (an MPO or MPS where a density matrix cannot) still
  runs.
- Noisy fidelity tests candidates on the register it evaluates: a terminal
  measurement on an otherwise unused qubit no longer rejects a listed density
  matrix. Its backend probe no longer outlives the first realization.
- Realization workers are sized for the largest register a candidate list
  may need, for the crosstalk spectators injection adds, and on networks that
  run on their whole layout. Noisy expectation values (`NoiseAdd` and
  `EstimateNoisy`) also count the idle qubits an observable names, which the
  host executor allocates.
- A whole-network `Execute()` on a NetQASM or discrete-event network under
  automatic selection starts from |0>, as on a simple network: only fixed
  selection continues a prepared state.
- Removing a network option keeps the settings made directly on its
  simulator.
- Distributed request metadata describes the placement of the register that
  ran.

## [0.3.13] - 2026-11-05

### Added

- The native request API (and so the local server and QRMI) reaches the newer
  Python features:
  - `adjoint_expectation_values`, `adjoint_jacobian` and `adjoint_vjp` run
    `AdjointProgram` on the CPU statevector, with Pauli, weighted-sum and
    probability readouts, batched `parameter_rows` and VJP `cotangents`.
  - `mirror_fidelity` accepts `observables` and returns their expectations on
    the mirror state, as Python's `mirror_fidelity(observables=...)` does.
  - Instruction arrays accept `rxx` and `rzz` (expanded as in QASM), `noiseless`
    on gates, `readout_error` on measurements (symmetric or asymmetric), and
    `pauli_channel`. Mixed-state methods apply Pauli channels exactly; pure-state
    methods sample them per realization from a trajectory `noise` block, whose
    `channels` may then be omitted.
  - Capabilities list the instruction format (`instructions`) and the adjoint
    operations (`adjoint`), so clients can detect these features.
  - An adjoint request whose state would exceed the machine's memory fails with
    `insufficient_resources` and the reason.
- `NoiseModel.native_channels(num_qubits, include_readout=True)` returns a Python
  noise model as the `channels` of a native request's `noise` block, built from
  its recorded setter calls, so the same model runs through the native API, the
  local server and QRMI. Channels on qubits beyond `num_qubits` are left out, a
  later setting replaces an earlier one on the same channel and targets, and
  `include_readout=False` drops readout errors for estimates, which reject them.
  A model that re-sets T1 around thermal relaxation can depend on the order of
  those calls, which the channels do not keep, and the native request then
  rejects it.

## [0.3.12] - 2026-11-03

### Added

- GPU Pauli propagation takes `pp_memory_limit_bytes`, a memory budget (default
  90% of available GPU memory), and `pp_max_fused_gates`, the largest fused block
  (1–256, default 64; one disables fusion). `pp_workers` (1–64 on GPU, default
  one) and `pp_sampling_cache_nodes` (default 8192, zero disables caching) now
  apply to the GPU backend as well as QCSim.

### Fixed

- Unseeded calls no longer read hardware entropy for every simulator they build.
  A `simple_estimate`, `simple_execute` or `get_statevector` call constructed
  `std::random_device` about 13 times; on x86 Linux each one executes RDSEED,
  which all cores share, so a small-circuit call cost about 0.5 ms and parallel
  processes slowed each other down. Entropy is now drawn once per process, and
  again in a forked child. A small-circuit call takes about 45 µs, and 8
  concurrent processes take about 65 µs per call instead of 4.3 ms. Seeded
  results are unchanged.

## [0.3.11] - 2026-11-01

### Changed

- Result dictionaries report `simulator`, `method` and `cluster_methods` as
  `SimulatorType` and `SimulationType` members instead of bare integers, so
  `print(result["simulator"])` shows `SimulatorType.QCSim`. Both enums are now
  integer enums, so the members still compare equal to the old values and
  `result["method"] == 0` keeps working.
- The Python documentation moved from the Doxygen site to a new site at the
  documentation root, with a user guide, examples and API reference whose code
  examples run when the site is built. The C++ reference is now under `/cpp/`,
  and the old C++ and Python guide URLs redirect.
- macOS wheels need macOS 15 or later, up from 14, because GitHub retired its
  macOS 14 build machines. They remain Apple Silicon (arm64) only.

### Fixed

- Docstrings in the Python bindings and `maestro.sinter` that contained text such
  as `|0>` or `<psi|phi>` no longer produce reStructuredText warnings and errors
  in the generated API reference.

## [0.3.10] - 2026-10-30

### Added

- The QASM parser accepts qelib1.inc's `rzz` and `rxx`, expanding them into `cx`,
  `rz`, `cx` (between `h` gates for `rxx`), so the gates no longer need
  decomposing before they reach Maestro.
- `maestro.sinter` runs Stim's own noise. `DEPOLARIZE1/2`, `X/Y/Z_ERROR` and
  `PAULI_CHANNEL_1/2` translate to the new `QuantumCircuit.pauli_channel`, and
  measurement flips such as `M(p)` to the new `measure(..., readout_error=p)`, so a
  noisy Stim circuit such as SI1000 needs no `NoiseModel`. Pure-state and MPS
  backends draw one Pauli per noise realization; density-matrix and MPO backends
  apply the channel exactly. `E` and heralded noise still raise.
- `MaestroCompiledSampler.sample_measurements(shots, seed=None)` returns the raw
  measurement record.
- `MaestroSinterSampler(noise_model=...)` also accepts a picklable
  `task -> NoiseModel` callable, so one sampler serves a noise-strength sweep.
- `QuantumCircuit.h/s/sdg(..., noiseless=True)` exempts a basis change from a
  NoiseModel's gate noise.

### Changed

- **Breaking:** `set_coherent_depolarizing` (and `set_all_coherent_depolarizing`,
  `set_coherent_strength`) is now isotropic: it rotates by ε = 2·arcsin(√p) about a
  body diagonal (±1, ±1, ±1)/√3, whose Pauli twirl is exactly DEPOLARIZE1(p). It was
  a Z rotation, so Z-basis logicals never saw it. `set_coherent_dephasing` keeps the
  old behaviour.
- Coherent setters replace only the component they name, so
  `set_coherent_bit_flip` followed by `set_coherent_dephasing` keeps both axes.
- **Breaking:** `MaestroSinterSampler` raises ValueError when no decoder resolves.
  Under `sinter.collect`, `task.decoder` is the `custom_decoders` key; the key
  `"maestro"` used to mean "undecoded" silently. Pass `decoder="pymatching"`, or
  `decoder="vacuous"` for the undecoded rate.
- The basis changes `maestro.sinter` inserts for `RX`, `RY`, `MX`, `MY`, `MRX` and
  `MRY` are noiseless, so a NoiseModel adds gate noise only where Stim has gates.
  `CLEAN_CIRCUIT_API_VERSION` is now 2.
- A NoiseModel's readout error composes with a measurement's own readout error
  instead of replacing it.

### Fixed

- Noise-model call logs record only successful setters and snapshot mutable
  arguments, preserving their configured values during pickle and call-log replay.
- Conflicting T1 and thermal-relaxation settings are rejected before changing the
  noise model, including uniform T1 updates across multiple qubits.
- `NoiseModel` setters accessed on an instance keep their docstrings.
- The docstrings of `set_2q_depolarizing` (Stim's DEPOLARIZE2, after any 2-qubit
  gate on the pair) and `set_2q_gate_depolarizing` (independent single-qubit
  depolarizing) now say how they differ.

## [0.3.9] - 2026-10-28

### Added

- `mps_routing_strategy` for CPU MPS/MPO automatic swap routing. The default
  `"legacy"` preserves the existing routing behavior; opt-in `"combined"` uses
  bounded forecasts from live tensor ranks when automatic lookahead is zero.
  Combined prepares its gate table on first use and shares prepared immutable
  metadata between clones. It has measured runtime and finite-cap accuracy
  tradeoffs, so it is not selected automatically. Existing optimization thresholds,
  explicit lookahead settings, initial layouts, and GPU routing are preserved.

### Changed

- CPU MPS/MPO setup and cloning reuse valid fusion routing metadata, avoiding
  repeated preparation while keeping each clone's execution state independent.
- x86 builds with GCC and Clang pass `-ffp-contract=off`, so the compiler no longer
  fuses multiplies and adds on its own. Serial and multithreaded runs now give
  bit-identical results with GCC 15; floating-point results can differ from 0.3.8 in
  the last bits.

### Fixed

- Concurrent first calls from several threads no longer fail with
  `Failed to create simulator handle`. The library instance is now initialised once,
  and threads that arrive during initialisation wait for it. 0.3.8 exposed the race by
  creating simulator handles with the GIL released.
- Source builds find OpenBLAS on systems whose OpenBLAS CMake package lists a library
  that does not exist, such as Ubuntu 26.04, by falling back to `FindBLAS`.

## [0.3.8] - 2026-10-26

### Added

- Adjoint differentiation engine (`AdjointProgram`) supporting exact Jacobians,
  Vector-Jacobian Products (VJP), and batched expectation values for parametrized
  statevector circuits.
- Probability readouts for `AdjointProgram`.
- Observable clustering with configurable lightcone size budget (`cluster_budget`)
  to group commuting observables during expectation value evaluation.
- Added `observables` parameter to `mirror_fidelity` for expectation-based mirror
  fidelity computations, preserving state truncation.
- The QASM parser accepts qelib1.inc's `u0`, a parametrised idle that adds no operation.
- Accepted `seed` as an alias for `noise_seed` in noise execution helpers.

### Changed

- MPS/MPO swap routing improved with a multi-policy rollout portfolio, adaptive
  lookahead depth, and Schmidt-rank-based bond growth estimations derived from gate
  unitaries instead of fixed upper bounds.
- The MPS initial-layout search keeps the circuit's qubit order unless another layout
  is predicted to be at least four times cheaper (`MPSDummySimulator::setMinLayoutGain`).
  MPO layouts are unchanged.
- Circuits are compacted to continuous qubit indices in ascending order, not in order of
  first use. Seeded samples of circuits that first touch their qubits out of order can
  differ from earlier releases; their distributions do not.
- Worker threads in the shot execution pool are now retained between successive
  executions instead of being respawned each run.
- Trajectory execution thread counts for large states are capped at QCSim's team size.
- Prebuilt Windows dependencies and `abi3` wheel builds for faster CI.
- Updated `qcsim` dependency pin.

### Fixed

- Two race conditions in the persistent worker thread pool.
- MPS QFT at 64 qubits took 20 s instead of 0.2 s and truncated: the layout search
  scrambled the qubit order, and whether a run recovered depended on its random seed.
- macOS compilation errors from `qubit_t` and `size_t` type differences, and mixed-type
  template deduction in `Sampling.cpp`.
- Adjoint differentiation guarded against physical memory exhaustion and cgroup limits
  on Linux and Windows, with validation for cotangent shapes and qubit ranges.
- Potential crash during test shutdown caused by duplicate simulator factory globals.
- Single-qubit GPU tensor estimates defaulting to statevector backend.
- Missing GPU device metadata for identity-only expectation estimates.

## [0.3.7] - 2026-10-19

### Added

- `Circuit::RemoveEndMeasurements`, exposed in Python as `QuantumCircuit.remove_end_measurements`.

### Changed

- Expectation estimation strips terminal measurements before execution, across the
  Python, native request and noisy estimation paths.

### Fixed

- `get_statevector` and `get_probabilities` lost amplitudes or swapped qubit order for
  circuits that leave lower-indexed qubits untouched; `x(1)` alone returned `[0, 0]`.

## [0.3.6] - 2026-10-12

### Added

- Automatic parity reduction transform (`ParityReductionTransform` / `parity_reduction`)
  for observable estimation, reducing active qubit count by exploiting Z2 symmetries (#40).
- Automatic causal-cone reduction transform (`CausalConeReductionTransform` / `causal_cone_reduction`)
  for observable estimation, stripping non-causal operations and spectator qubits (#39).
- Pre-execution transform configuration fields in `SimulatorConfig`: `causal_cone_reduction`
  and `parity_reduction`.
- Batched tensor expectations across public C and Python APIs.
- Native diagnostic capabilities and build metadata reporting.
- Modular nanobind Python bindings split into translation units under `bindings/python/` (#38).
- Workflow input option in GitHub Actions to conditionally run the complete test suite.

### Changed

- QCSim trajectory performance improvements, defaulting to per-shot execution.
- Improved reproducible sampling across QCSim backends.
- C++ codebase organization: grouped Maestro headers into functional subfolders and
  normalized native source filenames to `UpperCamelCase`.
- Optional, path-configurable Composer integration.
- Expanded `README.md` with feature overview, Python quickstart, and HPC execution guide.
- Updated Python reference documentation for pre-execution transforms and reduction config options.

### Fixed

- Restored simulator validity retention before execution for Composer compatibility.
- Trajectory threading and terminal density sampling dispatch.

### Dependencies

- Updated QCSim pin to `217c60953795` with improved extended stabilizer and Pauli propagation
  simulators and additional non-Clifford gates.

## [0.3.5] - 2026-10-02

### Added

- Gate fusion for the QCSim and GPU backends, controlled by `gate_fusion`
  in `SimulatorConfig` and exposed through the simulator, network, C and
  native request interfaces with fusion statistics.
  `SimulatorConfig.gate_fusion` is optional: `None` means unset, and
  `GetConfiguration("gate_fusion")` then reports `"auto"`.
- Native MPO queries on the QCSim and GPU backends: bit-vector probabilities,
  density matrix elements, dense density-matrix output, complex Pauli
  expectations and local operators `A rho A†`, through `IState`, the C
  interface, the Python bindings and the native request API.
- `trim()` and `recanonicalize()` for matrix-product states, and the `trim`
  and `recanonicalize` maintenance actions for MPS in the request API.
- Swap optimization (look-ahead routing) for MPOs, and
  `disable_optimized_swapping` and `lookahead_depth` for MPO requests.

### Changed

- **Breaking:** the Python methods `trim_mpo` and `recanonicalize_mpo` are
  renamed `trim` and `recanonicalize`; they now work on MPS as well as MPO.
- When `gate_fusion` is not set, QCSim statevectors below 11 qubits and density
  matrices below 5 qubits run unfused, as fusion is slower there. An explicit
  `true` or `false` still forces fusion on or off at any size.
- MPO `max_bond_dimension` accepts zero for unlimited, and the trace-restore
  and hermitize options now apply to GPU MPO too.
- `precision` and `use_double_precision` are aliases kept in sync. Qiskit Aer
  honours single precision for initialization and state export, and the
  network no longer forces single precision.
- Faster CPU fused circuits, from the vectorized dense gate kernels and a
  cheaper fusion cache.
- The GPU diagnostics operation accepts an MPS for maintenance. An MPS reports
  no diagnostics by default.

### Fixed

- Circuit layering no longer moves a measurement ahead of an earlier write to
  the same classical bit, or ahead of a conditional gate still reading it.
  Routed MPS/MPO runs returned wrong distributions for circuits that write a
  bit twice.
- Single-shot network calls keep the shot's classical bits when the executed
  circuit is narrower than the network allocation; `ExecuteOnHost` reported all
  zeros.
- MPS look-ahead routing survives gate fusion, so fused MPS circuits with
  mid-circuit measurements no longer fall back to local routing.
- `FlushSimulator` waits for queued GPU statevector and density-matrix work, and
  fusion boundaries no longer stall the GPU.
- Multithreading flags are forwarded to QCSim MPS and MPO simulators.
- Tensor network clones are isolated from the original.
- GPU, distributed GPU, Aer and composite simulators ignore a precision set
  after initialization instead of throwing. A new precision takes effect after
  `Clear` and the next `Initialize`.
- CI build installs apt packages after refreshing the package index.

### Dependencies

- QCSim pin bumped to 1edb647: vectorized dense gate kernels, MPS
  recanonicalization fixes, MPO long-chain scaling, MPS sampling and input
  validation fixes, and Windows compile fixes.

## [0.3.4] - 2026-09-26

### Changed

- **Breaking:** `SimulatorConfig` takes keyword arguments only and has one name
  per option, in Python and in the native JSON API:
  - `use_double_precision` → `precision` (`"single"` or `"double"`, Qiskit Aer
    and the GPU backends)
  - `mps_measure_no_collapse` / `mps_sample_measure_algorithm` → `mps_sampling`
    (`"probabilities"` or `"apply_measure"`)
  - `{mps,mpo,tensor_network}_use_gesvd{,j,p,r}` → `mps_svd_solver`,
    `mpo_svd_solver`, `tensor_network_svd_solver` (`"gesvd"`, `"gesvdj"`,
    `"gesvdp"` or `"gesvdr"`)
  - `pp_pauli_weight_threshold` → `pp_max_pauli_weight`
  - `pp_steps_between_trims` / `pp_steps_between_deduplications` →
    `pp_gates_between_trims` / `pp_gates_between_deduplications`
- **Breaking:** the noisy execution and estimation functions take `noise_seed`
  instead of `seed`, and `config` now comes before `shots`/`noise_realizations`.
  `noise_seed` must fit in 32 bits; when it is unset, the low 32 bits of
  `config.seed` are used. Zero shots or zero realisations are rejected.
- **Breaking:** `NoiseModel.add_correlated_ou_band` is renamed
  `set_correlated_ou_band`.
- Pauli propagation deduplicates every 10 operations when
  `pp_gates_between_deduplications` is unset. Deduplication alone leaves
  results unchanged; an explicit cadence still takes precedence.

### Fixed

- Pickled noise models keep their correlated OU bands.
- Each noise realisation gets its own seed.
- Qiskit Aer ignores configuration settings it rejects instead of throwing, and
  no longer records them, so a cloned simulator does not replay them and they
  cannot change its reported simulation type.
- GPU Pauli propagation applies its settings after creating the native state,
  so thresholds and cadences take effect. A deduplication cadence also sets the
  truncation cadence unless one is given.

## [0.3.3] - 2026-09-24

### Fixed

- Linux wheels import again. The 0.3.2 wheels failed with
  `ImportError: libgfortran-83c28eba.so.5.0.0` because the bundled OpenBLAS
  could not locate its own bundled dependencies.
- Distributed expectation queries reset reused simulators before a new circuit,
  preventing state from leaking between Composer noise realizations when
  automatic simulator selection is disabled.
- Distributed reset splitting preserves target states and execution delays.
- Host mapping selects the current circuit even when its qubit IDs already fit
  the host, avoiding stale or missing circuits in Composer's NetQASM path.
- Python `coherent_execute` now seeds measurement execution from the public seed
  unless `SimulatorConfig.seed` overrides it, and derives a separate simulator
  stream for each realization.
- Direct simulator `Configure("seed", ...)` calls seed readout randomness as well
  as quantum sampling. Reapplying the same seed restarts both streams.
- A reset with no later measurement on its qubit is no longer dropped by the
  terminal-measurement execution path. Expectation values and state queries
  previously ignored such resets, so sampled T1/thermal-relaxation noise after
  the last gate had no effect on `noisy_estimate_montecarlo` or native
  `estimate` requests on pure-state methods.
- Python noisy execution and estimation (`noisy_execute`,
  `noisy_estimate_montecarlo`, `noisy_fidelity`, and the `coherent_*` and
  `full_noise_*` variants) fall back to `SimulatorConfig.seed` for noise injection when no `seed`
  argument is given, matching the native JSON API. Previously a configured
  simulator seed alone did not make the injected noise reproducible.
- Python `noisy_estimate_montecarlo` and `full_noise_estimate` use a separate
  simulator seed for each realization. Resets of entangled qubits now sample
  independent collapse outcomes while remaining reproducible from a public
  seed or `SimulatorConfig.seed`.

### Changed

- QCSim pinned to `3ea2ea1`, which seeds unseeded MPS/MPO simulators with
  `std::random_device` entropy as well as the clock.

### Removed

- Removed `NoiseAdd::apply_readout_error_to_counts` (`MAESTRO_NOISE_ADD_VERSION`
  is now 4). Attach readout rates with `noise::attach_readout_error` before
  execution, or use the noisy/full execution helpers. Composer must be updated
  together with Maestro to remove its bindings to this method.

## [0.3.2] - 2026-09-18

### Added
- **Native Qiskit Aer Integration**: Enabled Qiskit Aer backend by default in CMake (`ENABLE_QISKIT_AER=ON`), automatically fetching the fork from `https://github.com/QoroQuantum/qiskit-aer` and linking against OpenBLAS. Exposed `SimulatorType.QiskitAer` and `SimulatorType.CompositeQiskitAer` in the Python bindings.

### Fixed
- Native JSON requests accept sampled thermal relaxation with `T1 < T2 <= 2*T1`, using effective `T2 = T1` with a warning and approximation metadata, matching Python. Exact density-matrix/MPO execution preserves calibrated T2.
- Native readout noise applies at measurement time, including conditional measurements and noisy outcomes consumed by classical control.
- Native measurement/readout randomness follows the explicit simulator seed or falls back to `noise.seed`, independently of circuit-noise injection.
- Legacy `SimpleExecute` and `SimpleEstimate` honor `config.seed`, including zero and the full uint64 range, and reject invalid seeds.

### Changed
- Python package version bumped to `0.3.2`.

## [0.3.1] - 2026-09-10

### Added
- **Native Delay & Idle Operation**: Added first-class `Delay` instruction across Circuit IR, Factory, Python bindings (`qc.delay`), and OpenQASM 2/3 parsers with duration validation in seconds.
- **Idle Channel Decoherence**: Introduced idle channel simulation via `set_idle_noise(qubit, t1, t2, excited_population, detuning_hz)` to model thermal relaxation, phase decay, and coherent detuning rotation accumulated over idling intervals.
- **Continuous OU Noise Integration**: Integrated continuous-time Ornstein-Uhlenbeck (OU) dephasing during delays, analytically propagating correlated frequency fluctuations across arbitrary idle durations.
- **Multi-Band OU & 1/f Spectrum Noise**: Added multi-band OU dephasing support (`add_correlated_ou_band`, `set_multi_correlated_ou`, `set_all_multi_correlated_ou`), synthetic 1/f noise generation (`set_1_over_f_noise`), and stationary equilibrium initialization (`stationary_init`).
- **Sinter Integration**: Added `maestro.sinter` module with `MaestroSinterSampler` and `MaestroCompiledSampler` implementing Sinter's sampling interface for Stim-based quantum error correction (QEC) benchmarking.
- **GPU SVD Algorithm Selection**: Added configuration parameters in `SimulatorConfig` to select specialized GPU SVD solvers (`gesvd`, `gesvdj`, `gesvdp`, `gesvdr`) for MPS, MPO, and Tensor Network backends.

### Changed
- Python package version bumped to `0.3.1`.


## [0.3.0] - 2026-09-07

### Added
- **GPU Density Matrix Simulator**: Added full density matrix simulation on GPU (`SimulationType.kDensityMatrix` with `SimulatorType.kGpuSim`) for exact mixed-state quantum simulation.
- **GPU Matrix Product Operator (MPO) Simulator**: Added GPU-accelerated MPO simulation (`SimulationType.kMatrixProductOperator` with `SimulatorType.kGpuSim`), supporting non-unitary noise, truncation thresholding, trimming, and recanonicalization.
- **QCSim Extended Stabilizer Simulator**: Added extended stabilizer simulation backend support in QCSim for near-Clifford circuit execution.
- **CPTP Quantum Channels & Exact Noise Models**: Introduced `QuantumChannel` and `QuantumChannelOperation` classes supporting general Kraus representations, along with predefined channels (bit-flip, phase-flip, depolarizing, amplitude damping, phase damping, and thermal relaxation).
- **Deterministic Simulation Seeding**: Exposed `seed` parameter in `SimulatorConfig` and across simulator backends for reproducible stochastic simulations.
- **GPU Device Selection**: Added interface to select the target GPU device for GPU simulator backends.
- **OpenQASM 3 Support**: Added OpenQASM 3 import and export capabilities.
- **Documentation Rework**: Restructured Python API documentation into dedicated guides for backends, HPC execution, noise simulation, algorithms, and quickstart reference.

### Fixed
- **Bond Dimension Capping & Synchronization**: Configured default MPS and MPO bond dimension to 128 in `GpuState` and GPU plugin to prevent unbounded device memory allocation.
- **Mirror Fidelity Validation**: Added validation in `mirror_fidelity` to raise informative errors if simulation yields zero measurement samples or zero shots instead of failing silently.
- **Pauli Propagation**: Fixed GPU Pauli destructive state hooks and truncation configuration propagation.

### Changed
- Python package version bumped to `0.3.0`.


## [0.2.18] - 2026-08-13

### Added
- Bond dimension tracking per time step in `incremental_evolve`
- Simulators and networks Configuration classes and refactoring

### Fixed
- Misaligned time step guard handling in network evolution

### Changed
- Python package version bumped to `0.2.18`


## [0.2.17] - 2026-08-11

### Added
- **Bond Dimension Callback & Max Bond Dimension Tracking** — exposed maximum bond dimension reached during MPS simulation across `execute`, `estimate`, and `incremental_evolve` Python return dictionaries (`max_bond_dim_reached`)
- Integrated QCSim bond dimension callback support and GPU MPS simulator bond dimension callback
- Added simulator `Configuration` class to standardise parameter delivery across simulator backends

### Changed
- Python package version bumped to `0.2.17`


## [0.2.13] - 2026-05-02

### Added
- **Asymmetric readout error** — `set_readout_error(q, p_meas1_prep0, p_meas0_prep1)`, `set_readout_error_symmetric(q, p)`, `set_all_readout_error(n, p)` for per-qubit classical measurement error modelling. Applied as a post-measurement channel in `noisy_execute` and `full_noise_execute`
- **Two-qubit depolarizing channel** — `set_2q_depolarizing(q1, q2, p)` for correlated 15-Pauli noise applied after CX/CZ gates on specific qubit pairs
- **Gate-type-specific depolarizing** — `set_1q_gate_depolarizing(q, p)` and `set_2q_gate_depolarizing(q, p)` for separate noise rates after single-qubit vs two-qubit gates, with bulk setters `set_all_1q_gate_depolarizing` and `set_all_2q_gate_depolarizing`
- **`qc.noisy_prob(target, nm)`** — readout-corrected path integral probability using first-order analytic expansion (n+1 PI evaluations instead of Monte Carlo)
- Readout error, two-qubit depolarizing, gate-type-specific noise, and `noisy_prob` sections in `python.dox` (Doxygen) and `TUTORIAL.md`
- Complete `NoiseModel` method reference table in both documentation sources

### Changed
- Python package version bumped to `0.2.13`
- Fixed `set_pauli_channel` documentation to use correct binding name `set_qubit_noise`

## [0.2.12] - 2026-04-28

### Added
- **Path Integral simulation backend** — new exact-amplitude simulator (`kPathIntegral`) integrated into the QCSim common interface, supporting amplitudes, measurements, sampling, and Pauli-string expectation values. Exposed through the factory and the `SimulatorConfig(simulator_type="path_integral")` Python API
- **Path Integral Python bindings** — `amplitude_from_zero()` and `amplitude()` methods on `QuantumCircuit` for computing exact transition amplitudes via path integrals
- `IsBranching()` method on all quantum gate types, used by the path integral simulator to identify branching operations
- **Path Integral via Network interface** — `SimpleDisconnectedNetwork` now dispatches path integral jobs; multithreading is disabled for path integral simulators in networks when measurements are only at the end
- **Coherent noise model** — `NoiseModel` now supports deterministic rotation-gate noise injection (Rx/Ry/Rz) alongside stochastic Pauli channels. Coherent angle is derived from error probability via ε = 2·arcsin(√p) to match per-gate infidelity
- Coherent noise setters: `set_coherent_depolarizing()`, `set_coherent_dephasing()`, `set_coherent_bit_flip()`, `set_coherent_rotation()`, `set_all_coherent_depolarizing()`
- **T1 amplitude damping noise** — `set_t1()`, `set_all_t1()`, `set_t1_from_time()` for per-qubit T1 decay with physical time-constant conversion (γ = 1 − e^(−t_gate/T₁))
- **ZZ crosstalk noise** — `set_crosstalk()` for pairwise qubit coupling injection
- **`coherent_execute`** / **`coherent_estimate`** — Python bindings for executing circuits and estimating expectation values with coherent rotation noise
- **`full_noise_execute`** / **`full_noise_estimate`** — combined noise pipeline applying coherent + crosstalk + T1 + Pauli noise in a single call
- Comprehensive test suites: 600+ lines of path integral C++ tests (random circuits, amplitudes, measurements, sampling, Pauli strings), 290+ lines of network simulator tests, 1200+ lines of new Python binding tests covering all noise modes
- Noise simulation and path integral sections in `TUTORIAL.md` with usage examples

### Changed
- `NoiseModel` header expanded to support Pauli, coherent, T1, and crosstalk noise in a unified model — both modes can be configured on the same instance
- `QCSimSimulator` routes all gate types through the path integral backend when `kPathIntegral` simulation type is selected
- `QCSimState` extended with path integral simulator state management, amplitude computation, and measurement support
- Doxygen upgraded to latest version; docs CI workflow updated
- Python package version bumped to `0.2.12`

### Fixed
- Missing `public:` access specifier on `PathIntegralSimulator` class methods
- Include path issues for Linux compilation in network simulator tests
- Compile warnings in `NetworkJob.h` and `SimpleDisconnectedNetwork.h`

## [0.2.11] - 2026-04-20

### Added
- **Python 3.13 and 3.14 wheels** published to PyPI for Linux, macOS, and Windows
- `QasmToCirc.failed()` and `QasmToCirc.get_error_message()` Python bindings for inspecting parser state after a failed parse
- QASM parser: `id` gates and unrecognised no-op instructions are now silently skipped instead of raising

### Changed
- **`SimulatorConfig` struct** — all execution, estimation, and fidelity functions now accept a single `config=maestro.SimulatorConfig(...)` parameter instead of repeating `simulator_type`, `simulation_type`, `max_bond_dimension`, `singular_value_threshold`, `use_double_precision`, `disable_optimized_swapping`, `lookahead_depth`, and `mps_measure_no_collapse` as individual keyword arguments. Create a config once and reuse it across calls.
- Updated all Python examples to use the new `SimulatorConfig` API
- Updated `python.dox` and `TUTORIAL.md` documentation with `SimulatorConfig` usage
- `QasmToCirc.parse_and_translate` now raises `ValueError` carrying the parser error message when QASM input is invalid, instead of returning a bad circuit silently
- Build toolchain: `cibuildwheel` upgraded from `v2.22.0` to `v3.4.1`; default Linux manylinux image moved from `manylinux2014` to `manylinux_2_28` (wheels now require glibc 2.28+, i.e. RHEL 8 / Ubuntu 20.04 / Debian 10 or newer)
- MPS swap optimization: `growthFactorGate` heuristic tuned from `0.7` to `0.65` across QCSim, GPU, MPSDummy, and SimpleDisconnectedNetwork — may shift swap-vs-gate planning decisions on large circuits

### Fixed
- `simple_execute` (QASM variant) was not forwarding `mps_measure_no_collapse` to the simulator
- `noisy_estimate_montecarlo` was not forwarding `mps_measure_no_collapse` to noisy runs
- `mirror_fidelity` returned incorrect values because the circuit optimizer cancelled the mirror's paired gate/adjoint operations before execution. Circuit optimization and MPS swap optimization are now disabled for the mirror run, and non-gate operations (e.g. measurements) are skipped during the adjoint reverse pass
- `SampleCountsMany` on the Qiskit Aer MPS backend returned bits in ascending-qubit-index order rather than the caller-requested qubit order, silently misaligning outcomes. Bits are now remapped to match the caller order (consistent with the statevector backend)
- `singular_value_threshold` values below ~1e-4 were truncated to `0` when serialized to the MPS backend (via `std::to_string`'s default 6-digit precision), effectively disabling entanglement truncation. Serialization now uses `max_digits10` (17 significant digits)

## [0.2.10] - 2026-04-16

### Added
- **Noise simulation API** — `NoiseModel` class for per-qubit Pauli noise channels (depolarizing, dephasing, bit-flip, custom)
- **`noisy_estimate`** — analytical Pauli channel damping with zero simulation overhead for fast ansatz noise screening
- **`noisy_estimate_montecarlo`** — gate-by-gate Monte Carlo noisy estimation for accurate depth-dependent noise simulation
- **`noisy_execute`** — Monte Carlo shot-based noisy circuit execution with batched noise realizations
- Noise model helpers: `set_depolarizing`, `set_dephasing`, `set_bit_flip`, `set_pauli_channel`, `set_all_depolarizing`
- Comprehensive noise test suite (25 tests covering NoiseModel, analytical damping, Monte Carlo estimation, and execution)
- Noise simulation section in TUTORIAL.md with usage examples and parameter reference

### Changed
- Default `lookahead_depth` changed from `20` to `-1` (auto-tuning) to prevent hangs on larger MPS circuits

### Fixed
- MPS swap optimization hang on larger circuits caused by hardcoded `lookahead_depth=20` bypassing the C++ auto-tuning heuristic

## [0.2.7] - 2026-03-27

### Added
- **Inner product** — compute ⟨ψ₁|ψ₂⟩ between two circuits via `ProjectOnZero` (`maestro.inner_product(circ_1, circ_2)` and `qc.inner_product(other)`)
- Python documentation for `inner_product` in `python.dox`
- Comprehensive test suite for inner product (9 tests covering identical, orthogonal, parametric, MPS, and phase-difference cases)

## [0.2.6] - 2026-03-25

### Added
- **`ProjectOnZero`** — efficient ⟨0|ψ⟩ projection for all simulator backends (QCSim MPS, GPU MPS, statevector, composite)
- `ExecuteOnHostProjectOnZero` network-level method with qubit remapping support
- `ExecuteOnHostAmplitudes` with proper qubit remapping for statevector extraction
- **Mirror fidelity** — `maestro.mirror_fidelity()` and `qc.mirror_fidelity()` with shot-based and exact statevector modes
- **Statevector access** — `maestro.get_statevector()` and `qc.get_statevector()` Python bindings
- **Probability access** — `maestro.get_probabilities()` Python binding
- MPS swap optimization with lookahead heuristics and initial qubit mapping
- Classically controlled gate support for MPS lookahead swap optimization
- Improved QCSim MPS sampling for partial qubit measurements
- ProjectOnZero C++ tests across all simulator backends

### Fixed
- Linux GPU library compilation (missing include)
- MPS optimization test adjusted from 20 to 12 qubits for stability

## [0.2.5] - 2026-03-18

### Added
- **Windows wheel support** — pre-built wheels for Windows AMD64 (Python 3.10, 3.11, 3.12) published to PyPI
- vcpkg-based dependency management for Windows CI builds
- Dry-run Windows CI workflow

### Fixed
- Eigen type mismatch between Windows and Linux (`Eigen::Index` → `long long int`)
- OpenMP configuration on Windows
- Compile warnings in MPS simulator and QCSim

## [0.2.4] - 2026-03-12

### Changed
- Split Python version builds into separate CI matrix elements for faster Windows builds
- Code styling and minor CI workflow improvements

## [0.2.3] - 2026-03-07

### Added
- Exposed setting the initial qubits map from the Python API
- MPS swap cost optimization with improved heuristics

### Fixed
- Linux wheel repair (`auditwheel`) patching for `libmaestro` linkage
- rpath configuration (`$ORIGIN`) for reliable library resolution in wheels

## [0.2.2] - 2026-03-01

### Fixed
- Bundled excluded `libmaestro` shared library with Python wheel
- macOS build issues

## [0.2.1] - 2026-02-25

### Fixed
- PyPI package metadata issue
- Version bump for initial stable release

### Added
- **Python bindings** via nanobind with `QuantumCircuit` model and GPU support
- **PyPI distribution** with `scikit-build-core` and `cibuildwheel` for Linux & macOS
- **Pauli propagator simulator** — CPU and GPU implementations with non-Clifford gate decomposition
- **Extended stabilizer simulator** with support for > 64 qubits
- **QuEST simulator** integration with factory function and tests
- **GPU stabilizer simulator** exposed through the library API
- **Rydberg atom array** simulation examples (adiabatic preparation, phase diagrams, spatial correlations)
- MPS parameters passthrough in the `SimpleExecute` flow
- Expectation value estimation via Python (`SimpleEstimate`)
- Maestro executable support for new simulators (Pauli propagator, extended stabilizer, QuEST)
- Double-precision toggle from Python bindings
- ccache support for faster rebuilds
- Doxygen documentation GitHub Actions workflow

### Changed
- Refactored GitHub Actions CI workflow for better dependency caching
- Build system configured for PyPI distribution (rpath, install targets)
- Improved GPU precision handling with placeholder params

### Fixed
- macOS OpenMP/install build issues
- Linux GPU compilation errors
- Circuit distribution crash when no distributor is present in disconnected networks
- Build warnings suppressed for third-party dependencies

## [0.1.0] - 2026-01-18

### Added
- **Core simulation engine** with multi-backend support (QCSim statevector, MPS tensor network, Qiskit Aer, Clifford)
- **Composite simulator** for automatic circuit distribution across backends
- **GPU tensor network simulator** with statevector and stabilizer support
- Maestro shared library (`libmaestro`) with `SimpleExecute` JSON/QASM API
- Maestro standalone executable with command-line interface
- QASM 2.0 parser (string → circuit conversion) with standard gate support
- Expectation value estimation framework
- Network-based job scheduling and execution
- OpenMP parallelization with SIMD (AVX2/FMA) acceleration
- Boost serialization support
- Doxygen API documentation with GitHub Pages deployment
- GitHub Actions CI (Ubuntu build + test)
- Pre-commit hooks with clang-format code formatting
- `CITATION.cff`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `INSTALL.md`

[Unreleased]: https://github.com/QoroQuantum/maestro/compare/v0.3.14...HEAD
[0.3.14]: https://github.com/QoroQuantum/maestro/compare/v0.3.13...v0.3.14
[0.3.13]: https://github.com/QoroQuantum/maestro/compare/v0.3.12...v0.3.13
[0.3.12]: https://github.com/QoroQuantum/maestro/compare/v0.3.11...v0.3.12
[0.3.11]: https://github.com/QoroQuantum/maestro/compare/v0.3.10...v0.3.11
[0.3.10]: https://github.com/QoroQuantum/maestro/compare/v0.3.9...v0.3.10
[0.3.9]: https://github.com/QoroQuantum/maestro/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/QoroQuantum/maestro/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/QoroQuantum/maestro/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/QoroQuantum/maestro/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/QoroQuantum/maestro/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/QoroQuantum/maestro/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/QoroQuantum/maestro/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/QoroQuantum/maestro/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/QoroQuantum/maestro/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/QoroQuantum/maestro/compare/v0.2.18...v0.3.0
[0.2.18]: https://github.com/QoroQuantum/maestro/compare/v0.2.17...v0.2.18
[0.2.17]: https://github.com/QoroQuantum/maestro/compare/v0.2.13...v0.2.17
[0.2.13]: https://github.com/QoroQuantum/maestro/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/QoroQuantum/maestro/compare/v0.2.11...v0.2.12
[0.2.11]: https://github.com/QoroQuantum/maestro/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/QoroQuantum/maestro/compare/v0.2.7...v0.2.10
[0.2.7]: https://github.com/QoroQuantum/maestro/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/QoroQuantum/maestro/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/QoroQuantum/maestro/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/QoroQuantum/maestro/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/QoroQuantum/maestro/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/QoroQuantum/maestro/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/QoroQuantum/maestro/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/QoroQuantum/maestro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/QoroQuantum/maestro/releases/tag/v0.1.0
