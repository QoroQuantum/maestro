# QCSim sampling policies

The QCSim statevector, density-matrix and path-integral adapters use
`sampling_policy=reproducible_v1` by default. Set `sampling_policy=legacy` to
select the native per-shot count algorithms. Both policies reseed clones.
QCSim composite components also use the prepared sampler.

Seeded replay is guaranteed for the same Maestro version, build, configuration
and logical call sequence. Policy names do not freeze random sequences or
performance thresholds across releases. Explicit seeds, including zero, do not
disable eligible optimized execution paths.

```cpp
sim->SetSeed(123);                         // zero and all 64 seed bits are valid
sim->SetMultithreading(false);             // e.g. an outer worker owns this clone
sim->Configure("sampling_policy", "reproducible_v1");
sim->Configure("sampling_max_memory_mb", "256");
const auto counts = sim->SampleCountsMany({7, 0, 3, 0}, 1000000);
```

The memory setting is a positive integer in MiB, defaulting to 256. It budgets
sampling tables, probability/label snapshots, uniform batches, sampled rows and
private histograms. The quantum state and returned count map are outside this
budget. An alias table that does not fit falls back to CDF sampling. Probability
blocks grow to fit the budget, and path-integral sampling uses bounded sequential
CDF sweeps when its row snapshot would not fit. A composite still requires room
for each selected component's minimum scratch allowance and a shared draw batch.

## Count execution dispatch

`SimulatorConfig.counts_policy` selects the strategy for count-returning execution. Its default is `"auto"`.

| Policy | Count execution |
| --- | --- |
| `auto` | Use multinomial sampling for eligible QCSim terminal count batches with or without an explicit seed. Eligible requests can also use frames and direct trajectories, with or without seeds, as described below. |
| `legacy` | Keep the ordinary count/trajectory executor and never select the new frame-count path. |
| `frame_v1` | Request frame counts when backend selection and circuit semantics permit them. Other requested QCSim methods can use frames when backend optimization permits a stabilizer implementation. Unsupported requests fall back. |
| `multinomial_v1` | Allow the terminal sampler to draw counts by conditional binomials, including with explicit seeds. Small batches and unsupported distributions use the existing backend sampler. |

The adaptive count policy supports QCSim statevector, density matrix, retained
path-integral rows, a single MPS output bit, a single bit of an MPO without a bond
cap or truncation threshold, and small extended-stabilizer
readouts. Rows expected to receive at most two shots are grouped per block, so a
concentrated distribution with a long tail of tiny amplitudes also qualifies.
A bare simulator applies the same cost thresholds to `SampleCounts` and
`SampleCountsMany` by default, including after `SetSeed`. A QCSim composite
with one component forwards its count request to that component's sampler.
Explicit `sampling_policy="reproducible_v1"` also permits adaptive counts.
Only `sampling_policy="legacy"` overrides the multinomial choice. Seeded
`multinomial_v1` sequences use Boost.Random's binomial generator. The
[implementation report](multinomial-sampling.md) describes the measured
crossovers, scope of each backend, tensor-size costs and commands for
reproducing the benchmarks.

`auto` never replaces a method the ordinary executor would honor:

- A Stabilizer request qualifies.
- A Python request for another method (Statevector, MPS, density matrix, MPO,
  ...) keeps it. Python restricts automatic selection to the requested method,
  so its ordinary executor runs even a Clifford circuit on that method.
- A native JSON request for another method qualifies only with
  `"selection": "automatic"` and a `candidates` list that includes the QCSim
  stabilizer. A `candidates` list replaces the requested method, so an
  automatic Stabilizer request whose list omits the QCSim stabilizer does not
  qualify either.
- A native C++ `NoiseAdd` network with another method qualifies only when
  simulator optimization is enabled, its optimization candidates admit the
  QCSim stabilizer, and exact channels are off (they pin the requested
  mixed-state backend). With optimization enabled, a Stabilizer method also
  needs candidates that admit the QCSim stabilizer.

To use frames with another requested method, select `counts_policy="frame_v1"`
explicitly. `frame_v1` may replace an automatically selected method, but an
explicit native `candidates` list names every backend the request allows:
`frame_v1` uses frames only if that list includes the QCSim stabilizer. The
same holds for a C++ `NoiseAdd` network whose optimization candidates exclude
it.

Seed zero is explicit, including through the Python `seed` alias for
`noise_seed`. Repeat the same request on the same build with the same seeds
and configuration to reproduce its histogram. Counts can differ between
releases. `counts_policy="legacy"` selects the ordinary count executor; it
does not promise an older release's random sequence. Ordered shot requests can
also use eligible adaptive terminal sampling, followed by a seeded permutation
of the sampled outcomes. `counts_policy` is a writable, validated
configuration property and survives pickling.

```python
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.Stabilizer,
    counts_policy="frame_v1",
    seed=123,
)
result = circuit.full_noise_execute(
    noise_model, config=config, shots=1000, noise_realizations=1000,
    noise_seed=456,
)
counts = result["counts"]  # the count API and classical-bit ordering are unchanged
```

Frame eligibility requires a circuit and noise model the compiler supports,
as described below. Besides Pauli channels, the compiler accepts any one- or
two-qubit Kraus channel that is exactly a Pauli mixture, recognized from its
process matrix in the Pauli basis: phase damping, for example, is a Z flip with
`p = (1 - sqrt(1 - gamma)) / 2`. Amplitude damping and other non-unital channels
still fall back. Noisy count calls also require one independent realization
per shot. Shared-noise realizations, coherent or unsupported operations,
incompatible fixed backends, and sampling-reduction requests keep their
ordinary execution path, as does thermal noise in the T1 < T2 <= 2 T1 window,
which trajectories clamp to T1. As on trajectories, the noise seed drives the
circuit errors and the measurement seed the outcomes and readout, so either can
change alone. Non-Clifford circuits never use frames. A request for `frame_v1` never approximates an
unsupported channel or coherent rotation as Pauli noise. Calls that use frames
report `"sampling_strategy": "reference_frame_v1"` and the actual Stabilizer
method where execution metadata is returned.

Frame counts stream directly into a native histogram from batches of at most
1024 shots. They do not allocate the full `shots × storage_width` byte matrix.
Scratch storage is bounded by one frame batch; the returned histogram still
grows with the number and width of distinct outcomes. Counts retain the usual
classical-register padding, including unused zero columns.

When frames do not apply, seeded or unseeded `auto` noisy counts with one
independent realization per shot (`noise_realizations >= shots`) run each realization
directly on a worker-owned simulator, the executor `return_samples=True`
uses, and histogram its classical record. This skips per-realization host
mapping, backend selection and simulator construction, and reports
`"sampling_strategy": "trajectory_v1"`. It applies to QCSim Statevector,
density matrix, Stabilizer, extended stabilizer and path integral requests.
MPS and MPO keep the ordinary executor, for counts and raw samples alike:
it places their initial qubit map and lookahead swap routing per execution
(`OptimizeMPSInitialQubitsMap`), which affects truncation accuracy, so their raw
samples come from its histograms, shuffled within each realization. They honor
the requested sampler and the network's worker policy. When several realization
workers run concurrently, each uses one inner simulator to avoid multiplying
the worker and memory budgets. Requests with `enable_sampling_reduction=True` keep the
ordinary executor for raw samples as well, so that it can split disjoint
measurement cones; their rows likewise come from its histograms.
`legacy`, shared-realization and sampling-reduction counts keep their applicable
ordinary execution paths, as do circuits without
measurements: the ordinary executor keys their records by its own rule rather
than by the padded classical width used here (with no classical operation,
a single all-zero key; classical random operations can still write nonzero
bits).

Direct sessions run the same optimized circuit as the ordinary executor
wherever optimization can change results: when `optimize_circuit` is on (the
default), a copy of each circuit is optimized before execution on the
Stabilizer, extended-stabilizer and path-integral methods. There, merging
gates changes more than rounding. Path-integral pruning applies per gate, so ten
`Ry(0.3)` gates under a 0.2 threshold prune where one merged `Ry(3)` does not,
and `T;T` becomes a Clifford `S` that the stabilizer accepts. On Statevector
and density matrix, merging
only changes floating-point rounding, so direct sessions skip its cost (about
40 µs per realization of a 400-gate circuit).

Direct trajectory sessions (`return_samples=True`, noiseless or noisy, and the
counts path above) allocate only the qubits the circuit touches: like the
ordinary executor, they relabel those qubits in increasing order
(`QubitCompaction`), once per call and only when the indices are sparse.
`Circuit::Remap` keeps classical bits, so keys and sample columns retain their
padding. Allocating every index up to the highest would grow exponentially
with it on dense backends. The frame compiler compacts its tableau and frame
rows the same way (`ReferenceFrameSampler::FrameQubits`). It translates each
qubit index as it compiles and never copies an operation, so unsupported
operations fall back exactly as before; its random streams are keyed by
instruction, so seeded rows are unchanged.
`PrefixCheckpointedSimulator` allocates exactly the register it is given. A
default network simulator also allocates each host's network-entangled qubit,
which doubled statevector and quadrupled density-matrix work on these paths.

Crosstalk can rotate spectator qubits the circuit never touches, at any index
(`NoiseInjectionPlan::RegisterWidth()` reports the highest). Such a qubit stays
in `|0>` and receives only crosstalk `Rz`, a global phase. Direct sessions
drop those rotations (`noise::drop_spectator_phases_if`), and checkpoints drop
those beyond their register (`noise::drop_outside_spectator_phases`); both
reject any other operation there, and probabilities and expectations are
unchanged. A spectator the circuit touches keeps its rotations.

The ordinary executor still simulates the spectators, compacting each
realization to the circuit's qubits plus the untouched spectators; its host
spans the noisy register. Its realization workers are budgeted
for that compacted width, while direct sessions are budgeted for the circuit's
own qubits. Count keys are padded to the noisy register, so a
spectator beyond the circuit widens them with zeros; direct counts pad theirs
the same way, so seeded and unseeded calls return the same key format. A
distant spectator therefore gives correspondingly wide keys, and counts fail
when the noisy register exceeds the simulator's register limit (2^31 - 1
qubits). Raw sample columns keep the circuit's classical
width, since spectators are not classical bits, and still run there.

Realization loops (Python noisy execution, estimates and fidelity, prefix
checkpoints, native JSON requests and C++ `NoiseAdd`) build one
`noise::NoiseInjectionPlan` per call. It resolves the model's per-qubit noise
parameters, each operation's qubits, two-qubit depolarizing probabilities and
the circuit-width bookkeeping once; each realization then runs the unchanged
injector code against those cached values. Realizations contain the same
operations in circuit order and use the assigned realization noise stream.

Plan realizations share the caller's plain gates, delays, resets and channels
instead of copying them. Measurements (which receive readout rates), classical
random operations, conditionals and nested circuits are still copied per
realization. Gates have setters, so shared
realizations are read-only: the executors never edit them. The public
`noise::inject_*` functions still return independent copies that a caller may
edit.

Frame counts are returned as a list of distinct outcomes
(`MaestroExecution::FrameOutcomes`), and direct counts accumulate records
packed 64 bits per word under `FrameDetail::PackedRowHash`. MSVC's
`std::hash<std::vector<bool>>` never carries bit 7 of a byte into the low hash
bits that select a bucket, so records that differ only there shared 1/128 of
the buckets. Python noisy frame and direct counts no longer hash such keys.
`Circuit::ExecuteResults`, the count map of the network executor, `NoiseAdd`
and `SampleCircuit`, uses `Types::BitsHash`, which finishes the standard hash
with a SplitMix64 mix so that every record bit reaches the bucket bits. The
simulators' own `SampleCountsMany` maps keep the standard hash.

Both changes alter the public C++ API. `Circuit::ExecuteResults` appears in
the virtual `INetwork` interface (`RepeatedExecute`, `RepeatedExecuteOnHost`,
`ExecuteScheduled`), in exported library functions (`CombineIndependentCounts`,
`SamplingSession::Sample`, `SampleCircuit` through `SamplingResult`) and in
`NoiseAdd`, and `ReferenceFrameSampler::SampleCounts` now returns
`FrameOutcomes`. Rebuild native C++ consumers together with the library and
the Python extension: exported symbols and object layouts change. Code that
spells `std::unordered_map<std::vector<bool>, size_t>` where these results are
passed must use the `ExecuteResults` alias instead. The C and JSON request
APIs exchange only C types and strings and are unaffected.

This setting is separate from `sampling_policy`, which selects the lower-level
statevector/density-matrix/path-integral sampling algorithm, and from
`trajectory_policy`, which selects dynamic-circuit RNG streams. Choosing
`counts_policy="legacy"` does not change either of those settings. It also
does not disable frame sampling requested through `return_samples=True` or
Sinter.

The native JSON request API accepts the setting in
`simulator.options.counts_policy`. Its frame response records the resolved
`"frame_v1"` policy in
`execution_metadata.configured_options.counts_policy`, along with its returned
top-level `seed` and, when noise is enabled, `noise.seed`. Replay those seeds
and the resolved policy together on the same version. An explicit returned
seed does not disable eligible automatic frame selection. Keep the
circuit, noise configuration, backend selection and logical request unchanged.
Native C++ `NoiseAdd` helpers accept the same network configuration key; their
frame optimization is limited to supported local single-host execution. Their
frames keep the trajectories' rules: the object's noise stream drives circuit
errors and the simulator seed the measurements and readout, so either can change
alone, and thermal noise in the T1 < T2 <= 2 T1 window, which the trajectories
clamp, keeps the trajectories. These objects advance their RNG streams between
calls; reseed the object and replay the same call sequence to reproduce an
earlier run.

## Python shot matrices and noisy Clifford sampling

`QuantumCircuit.execute`, `simple_execute`, `noisy_execute`, `coherent_execute`,
`full_noise_execute`, and `PrefixCheckpointedSimulator.execute_suffix` accept
`return_samples=False`. The default continues to return `"counts"`. Opting in
returns a C-contiguous NumPy `uint8` matrix under `"samples"` instead of a
histogram:

```python
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.Stabilizer,
    seed=123,
)
result = circuit.full_noise_execute(
    noise_model, config=config, shots=1000, noise_realizations=1000,
    noise_seed=456, return_samples=True,
)
samples = result["samples"]  # one byte per outcome, shape (1000, storage_width)
```

Column `k` is classical bit `k`, the same position as character `k` in a count
key. Storage width is the larger of the circuit's qubit width (crosstalk
spectators allocated beyond it do not widen it) and the highest referenced
classical-bit index plus one; checkpoint calls also honor `num_measurements`.
Count keys use the same width, except that crosstalk spectators beyond the
circuit pad them further (see above) and a circuit without measurements keeps
the ordinary executor's keys (a single all-zero key unless a classical random
operation writes bits). Columns that no measurement or classical operation writes
remain zero. The direct path does not add physical measurements to fill those
columns. Repeated writes to a classical bit retain its final value. This is a
classical-register matrix; callers that need every intermediate measurement
must give each measurement a distinct classical destination.

For QCSim, supported calls report
`"sampling_strategy": "reference_frame_v1"` when Stabilizer is requested or
automatic backend selection permits a stabilizer implementation. The default
automatic selection can therefore use frames for a Clifford circuit requested
with MPS or statevector; a fixed backend keeps the requested method, and a
native `candidates` list (or C++ `optimization_candidates`) admits frames only
if it includes the QCSim stabilizer, and then whatever the requested backend,
since the list replaces it. Sampling reduction disables frames for rows as it
does for counts. This reference-frame selection for ordered rows is independent
of `counts_policy`; that option controls counts dispatch and eligible raw
terminal, histogram-fallback and extended-frame sampling. With such a list, rows that do not use
frames run on the network executor, which selects from it, and report the
backend it chose. Frame
results report the actual `Stabilizer` method in their metadata. Compilation
executes one reference tableau; sampling propagates independent Pauli frames
for 64 shots per packed word, in batches of 1024 shots. Each random
instruction (measurement gauge, channel, readout) draws from its own stream per
batch, and sparse errors are located by geometric skipping, so noise costs grow
with the error rate rather than with shots times channels. For a fixed seed the
first `n` rows do not depend on how many rows are requested. Supported gates are X, Y, Z, H, S, S-dagger,
square-root X and its dagger, K, CX, CY, CZ, and SWAP, together with nested
circuits, delays, Z measurements,
readout errors, and resets to zero or one. Supported channel operations are
one- or two-qubit channels that are exact Pauli mixtures: Pauli channels
carrying Pauli-probability metadata, as the Pauli channel constructors and
compatible NoiseModel injection produce, and Kraus channels recognized from
their Pauli-basis process matrix (see above). Coherent rotations, non-unital and
other non-Pauli channels, and classical conditional operations use the ordinary
trajectory fallback. Checkpoint suffix sampling
also retains its trajectory implementation.

Other native QCSim raw samples (statevector, MPS, density matrix, MPO, extended
stabilizer, path integral) depend on where the circuit measures. When every
measurement is terminal, a call with several shots prepares the state once and
draws all rows from it, reporting `"sampling_strategy": "terminal_v1"`. The
same rule applies per noise realization that supplies several shots. Readout
errors are drawn independently for each row, and rows receive a seeded random
order, so any leading slice of one prepared state's rows is a uniform subsample
of them. Realizations fill consecutive rows, so with fewer realizations than
shots a leading slice of the whole matrix draws from the first realizations
only. Mid-circuit measurements,
resets after other operations, classical control, random bits, nested circuits
and single-shot calls replay the circuit for every shot and report
`"trajectory_v1"`. A noisy call reports `"terminal_v1"` only when every
realization used the prepared-state path.

Noisy frame sampling requires at least as many `noise_realizations` as shots,
so each shot has independent noise. Fewer realizations preserve the existing
meaning: each realization supplies one injected circuit shared by several
shots. Increasing this setting changes noise correlations, not just execution
speed. Coherent errors remain coherent; the frame path does not replace them
with Pauli noise.

The raw-sample policy has its own seeded sequence. Repeating the same logical
request, configuration and explicit seeds on the same build is reproducible,
but its rows need not expand the
histogram obtained with `return_samples=False`, and the frame and trajectory
policies need not produce the same seeded samples. Eligible terminal batches
and histogram fallbacks honor the counts policy, so `auto` and `legacy` raw
requests can also return different seeded rows. Backend, policy, circuit,
noise configuration, and logical call boundaries remain part of the replay
contract. Direct trajectories, frames and checkpoint sampling produce complete
classical records. Histogram fallbacks, including unsupported QCSim methods
and other backends, sort outcomes and apply a separate seeded permutation to
complete rows within each noise realization. This preserves the sampled
histogram and every record's correlations, and each leading slice within a
realization is a uniform subsample. It does not recover the original
chronological shot order from a histogram.

`maestro.sinter` compiles and reuses a full-circuit frame plan when supported,
before allocating a prefix checkpoint. Other circuits request raw samples
and convert classical columns into Stim measurement-record order. It retains
a count-based compatibility path for older extensions. General execution
callers keep the count API unless they explicitly request samples.

For an end-to-end manual comparison, run:

```text
python tests/sampling_performance.py --threads 1 --workers 1 --repeats 3
```

The script compares current count and raw-frame execution, checkpoint counts
and raw trajectories, Sinter sampling, and Stim on generated surface codes. By
default it runs one distance-three, three-round code at `--shots`;
`--distances 3 5 7` and `--shot-counts 1000 10000` sweep a grid instead (rounds
equal the distance), and `--include-legacy-large` adds the slow legacy and
checkpoint cases above 1,000 shots. It also compares MPS coherent noise with one
shared realization and with independent realizations (`--mps-shots`; skip with
`--skip-mps`). These are distinct workloads.
Timings use external wall-clock medians after warm-up and include Python result
construction. The JSON output records seeds, thread limits, backend settings,
and optional-dependency skips. It is a current-path comparison, not a saved
historical-library baseline or a promised speedup on other workloads.

### Shot rows outside Python

The row samplers live in `libmaestro` (`Execution/SampleRecords.h`), and the
Python methods above are thin wrappers over them:

- C++: `MaestroExecution::SampleCircuitRows(circuit, config, shots)` returns a
  `ShotRecords` with a row-major `shots x ClassicalWidth(circuit)` byte matrix
  and its `sampling_strategy`. `MaestroExecution::ExecuteNoisy(circuit, model,
  config, shots, noise_realizations, noise_seed, options)` is the noisy
  executor behind `noisy_execute`, `coherent_execute` and `full_noise_execute`:
  `options.layer` selects the noise layers, `options.rows` returns rows instead
  of counts, and `options.exact_channels` overrides the backend's channel rule.
  Neither emits warnings; `ThermalApproximationWarning` returns the text.
- Native JSON: `execution.return_samples: true` on `execute` returns `samples`
  as bitstrings (see `docs/native-request-api.md`).
- C ABI: `MaestroRunRequestSamples` writes the rows of an `execute` request as
  packed 64-bit words, without the JSON result bound.
- C++ checkpoints: `MaestroExecution::PrefixCheckpoint` (`Execution/PrefixCheckpoint.h`)
  executes a prefix once and samples suffixes from the saved state as rows or
  counts, with the seeding rules of `PrefixCheckpointedSimulator`.

`maestro.compile_reference_sampler(circuit, noise_model=None, config=...)`
returns a reusable `ReferenceFrameSampler`, or `None` when frames do not apply.
Its `sample(shots, seed=None)` and `sample_counts(shots, seed=None)` draw from
the compiled plan without recompiling the tableau, so repeated QEC batches pay
for compilation once; equal seeds give equal rows, and `sample_counts` is the
histogram of exactly the rows `sample` draws. The C ABI exposes the same plan as
a handle (`MaestroCreateReferenceFrameSampler`). `_compile_reference_sampler`
remains as an alias for `maestro.sinter` and older callers.

With a QCSim `ExtendedStabilizer` config, `compile_reference_sampler` returns an
`ExtendedFrameSampler` for any circuit of that method's gates. It compiles to
batched extended-stabilizer frames (`extended_frame_v1`) behind the same
`sample`/`sample_counts` interface, plus an optional `noise_realizations` that
shares each realization's errors among its shots. When the circuit's noise
includes relaxation that `noisy_execute` realizes as a reset shared within each
realization (T1, or thermal or idle relaxation with a finite T1, on a qubit the
circuit gates or delays), it accepts only one realization per shot (`ValueError`
otherwise). Both compiled samplers apply the exact channels: in the
T1 < T2 <= 2 T1 window, where `noisy_execute` clamps T2 to T1 and warns, they
keep T2. Every shot draws from streams of its own, and a measurement draws the
shot's physical outcome, so equal seeds give equal rows whatever the batch, the
worker count or `trajectory_max_memory_mb`. That budget is soft, as for
`noisy_execute`: a shot that alone exceeds it runs by itself beyond it, with the
draws it would have had. `maestro.sinter` therefore samples an `ExtendedStabilizer`
task on frames.

Eligible seeded and unseeded noisy `extended_stabilizer` sampling uses the same
frames from every entry point:

- Python: `noisy_execute` and `full_noise_execute`, and
  `PrefixCheckpointedSimulator.execute_suffix` after a gate-only prefix.
- C++: `ExecuteNoisy` and `PrefixCheckpoint`; `SampleExtendedFrames` in
  `Execution/SampleRecords.h` is the shared entry point.
- Native requests: `execute` counts and rows, and `checkpoint_batch` suffixes.

`counts_policy="auto"` and `"frame_v1"` permit these frames for seeded calls too.
There the memory budget is soft, as for trajectories: a shot that alone exceeds it runs by itself beyond it,
with the draws it would have had, instead of a scalar retry that would draw it
again.

## Reproducibility and clones

For identical probability data, policy, memory setting, selected qubits and
logical sampling calls, physical OpenMP team sizes do not determine samples.
The algorithm family depends on logical inputs, never wall-clock timings,
thread IDs, available cores or fluctuating free RAM. Changing the number of
shots in a call, splitting a call, changing the policy or changing the memory
setting can change the seeded sequence.

CDF blocks have fixed boundaries and a fixed summation order. Serial sorted
sampling scatters results back to their original shot positions. Parallel
sampling reads immutable tables, uses private integer counters, and consumes
the same serially generated uniform sequence. The per-shot CDF and alias
samplers convert one `mt19937_64` word per shot to a 53-bit value in `[0,1)`.
Eligible multinomial batches consume draws for conditional counts and rare-row placement.
Packed and bit-vector count APIs select the same plan and consume the same RNG
stream for the same request. Reordered and duplicate selected qubits retain
their caller-specified positions.

Statevector and path-integral one-shot calls keep the native direct sampler and
native RNG stream. QCSim's path-integral sampler weighs a state with an extreme
retained total from rescaled amplitudes, still with one draw from its stream
(see [Dispatch and memory](#dispatch-and-memory)). Statevector one-shot sampling fixes the native summation path
to serial, independently of the threading flag. Direct measurement APIs still
use their native RNG streams. Under `reproducible_v1`, statevector measurement
and reset also retain the native serial scan and probability-summation order,
while allowing independent collapse updates to run in parallel. The native
one-shot conversion and quantum
evolution retain their existing numerical behavior; bitwise whole-circuit
replay across different standard libraries, compiler floating-point settings or
backends is not promised. The portable batch conversion cannot compensate for
different input probabilities or automatic backend selection.

An explicitly seeded parent assigns ordinary clones deterministic child seeds
from its root and clone ordinal. Siblings and grandchildren receive their own
streams; quantum state copying never publishes a copied RNG state. Density
matrix cloning explicitly replaces the RNG copied by its native copy
constructor. Cloning does not consume the parent's sampling draws.

Without an explicit seed, each new simulator obtains a random root. Its effective
root can be read with `GetConfiguration("sampling_seed")` and recorded for replay.
Reinitializing an unseeded native state obtains a fresh root too; `Reset` and
ordinary sampling continue the current stream.
Replaying a clone tree requires the same root and logical clone order. Ordinary
cloning of a shared mutable parent must be serialized by its caller.
`CloneForExecution(seed)` assigns an externally chosen execution stream without
consuming the ordinary clone ordinal on these adapters and their fusion wrappers.
Callers must assign distinct seeds to independent execution clones.

Composite joins, splits and internal amplitude restores preserve the surviving
component's native, batch and auxiliary RNG positions and clone ordinal.
They use state-only replacement rather than public initialization, which can
restart an explicitly seeded run. A detached component gets a separate derived
seed from a split ordinal that survives restores; copying its parent's seed
would correlate later measurements. Saved composite snapshots contain one
quantum-state snapshot, without recursively retaining earlier snapshots.

Composite count sampling delegates directly when only one component remains.
With several components it prepares component samplers once, generates bounded
batches from each component's own RNG, and combines rows at the same shot index.
Small local projection tables avoid repeating bit permutations for every shot.
When the selected joint space fits, integer histograms defer output conversion
and hashing until counts are complete. Reordered and duplicate selections are
preserved, including bit-vector outputs wider than 64 bits and packed marginals
selected from global qubit IDs above 63.

The composite memory setting covers all component probability/projection tables
and shared draw, joint-output and histogram buffers together. Components receive
table space in stable component-ID order, reserving the minimum index space for
later components and space for bounded batches. Preparation and drawing respect
the multithreading flag and avoid starting nested OpenMP teams. RNG words remain
serially assigned within each component; batching and physical thread counts do
not change their assignment to shots. Unmeasured components skip preparation but
still advance their RNG by one word per shot, preserving subsequent sampling.
Single-shot calls use each selected component's native stream with serial
summation. Unselected components consume the same native distribution draw
without scanning their state; this does not assume a particular standard
library's distribution consumes exactly one engine word.
These composite optimizations apply to `reproducible_v1`; the legacy/Aer paths
retain their existing algorithms.

Maestro terminal measurement batches use one statevector, density-matrix or
path-integral simulator, letting the sampler use permitted internal threads.
This density-matrix dispatch applies to Aer and both QCSim sampling policies.
Terminal batches do not save an unused trajectory-restoration snapshot.

With `sampling_policy=reproducible_v1`, the other supported QCSim terminal
samplers use fixed logical batches when a full-request adaptive sampler does
not apply: 256 shots each, or at most 256 larger batches for longer requests.
Stabilizer, extended-stabilizer, MPS and MPO circuits are evolved once. The
first batch samples that prepared state itself, so what its sampler sets up
once (an MPS moves the measured sites to the head of its chain) is not repeated
per batch. One worker keeps sampling the same state, and several workers sample
private copies taken after the first batch. Each batch's seed follows
its logical index, so changing the worker count changes concurrency without
changing its seeded histogram. This also preserves worker invariance for raw
rows reconstructed from those histograms with a seeded permutation.
Qualifying single-qubit MPS/MPO marginals and extended-stabilizer marginals
keep the complete request on one sampler, avoiding repeated contractions or
blocks that fall below the adaptive crossover. The choice between these paths
does not depend on the worker count. The batch size is a current implementation
choice, not a promise about sequences in future releases.

QCSim trajectory execution defaults to `trajectory_policy=shot_v1`, which
distributes individual shots across workers, including short dynamic requests.
It assigns each shot separate quantum, readout and
classical-random streams derived from the request seed and absolute shot index.
Changing job boundaries, worker counts or the trajectory memory budget does not
change its seeded counts for supported execution paths within the same build.
`trajectory_policy=block_v1` instead uses fixed logical blocks of 256 shots,
seeded by block identity independently of outer worker count. Both policies
sample the same distribution; neither promises cross-version sequences.
These controls apply to QCSim statevector, density-matrix, path-integral,
stabilizer, extended-stabilizer, MPS and MPO trajectories, except when
`sampling_policy=legacy` selects the native path. Omitting `trajectory_policy`
and explicitly setting `shot_v1` select the same scheduler. Terminal batches
use the distribution-sampling paths above under either trajectory policy.

Classical `Random` instructions with an explicit seed, including zero, retain
their reproducible instruction streams when the request seed is omitted.
Logical shot or block IDs and instruction positions distinguish those streams;
fresh request entropy is supplied only to instructions without their own seed.
An explicit request seed participates in all execution streams. Replaying an
instruction stream requires the same logical execution sequence; replaying the
whole circuit also requires seeds for its other random operations.

For the JSON request API, put these options in `simulator.options`; for the
legacy `SimpleExecute` API, put them in its configuration object:

```json
{"trajectory_policy": "shot_v1", "max_simulators": 16, "trajectory_max_memory_mb": 1024}
```

Maestro's Python API exposes the same controls on `SimulatorConfig`:

```python
config = maestro.SimulatorConfig(
    seed=12345,
    trajectory_policy="shot_v1",
    max_simulators=16,
    trajectory_max_memory_mb=1024,
)
result = maestro.simple_execute(circuit, config=config, shots=100)
```

These are also writable properties and survive configuration pickling. Each
defaults to `None`, which keeps the network default. Invalid policy names and
out-of-range limits are rejected on construction or assignment. `max_simulators`
is limited to 1–1024. The trajectory policy and memory budget affect only the
QCSim execution paths described above.

`max_simulators` and `trajectory_max_memory_mb` must be positive JSON integers.
The latter defaults to 1024 MiB and applies to dense
statevector/density-matrix trajectory concurrency and supported noisy
realization workers. Noisy workers also estimate stabilizer and bounded
MPS/MPO states; unknown or unbounded state sizes retain serial execution.
Native `NoiseAdd` includes the full network allocation, entangled qubits,
retained source/execution copies and worker workspaces in its estimate, even
when the circuit touches only a small part of that network. Automatic native
backend selection with mixed or unknown candidate sizes retains serial
realizations; a source-method estimate cannot budget a different backend.
These estimates
always allow at least one worker. It is a **soft concurrency budget**, not a cap on
process RSS: circuit storage, libraries, gate scratch, results and other
allocations are outside it, and a single state may exceed it. It is separate
from `sampling_max_memory_mb`. The legacy API resets omitted trajectory policy
and memory options to their defaults for each request.

State copies are created when jobs start, so queued jobs do not each retain a
quantum-state copy. Dense execution clones share an immutable saved checkpoint;
their live states and any replacement checkpoints remain independent. Outer
workers count against the caller's OpenMP budget. States with at least `2^20`
complex elements use at most one outer worker per four budgeted threads to
reduce memory-bandwidth contention, subject to the memory and job-count limits.
Statevector and density-matrix trajectory jobs divide the caller's OpenMP thread
limit among active outer workers, with a minimum of one thread per job. A single
block can therefore use internal threads for gates and independent collapse
updates. With thread-local OpenMP runtimes, each job restores its thread's
previous limit when it finishes. Legacy MSVC OpenMP uses a shared process-wide
budget lease across the native library, Python extension and network jobs;
overlapping calls retain the original caller budget and the final participant
restores it. Nested realization calls from worker threads run inline, and no
budget mutex is held while a callback runs or workers join. Calls inside an
existing OpenMP team and path-integral jobs remain serial
internally. Measurement scans and probability sums retain serial arithmetic,
keeping outcomes stable across outer pool and OpenMP
team sizes. Consecutive statevector measurements visit only the subspace left
by earlier collapses, retaining the native ordering of surviving summands,
normalization and RNG draws (including duplicate measurements). Gates, reset,
restore and state replacement invalidate that subspace information. A complete
measurement sequence therefore avoids repeating full-state scans for every bit.
Density diagonal gates retain serial arithmetic during reproducible
trajectories because their native OpenMP loops can round complex products
differently. The dispatcher applies this arithmetic mode to prefix preparation
and execution clones, and clears it for terminal batches; other density gates
and collapse updates can run in parallel. This guarantee does not
extend to changes in backend selection, compiler or standard-library behavior.
Readout noise is applied in sorted outcome order under the new policy so count
map insertion order cannot change its RNG assignment.

## Dispatch and memory

Let `N` be the candidate row count and `S` the shot count. For statevectors, `N`
is an exact upper bound on occupied basis states, tracked through initialization,
gates, clones and restore. Import inspects trailing amplitudes once to establish
that bound. Tiny nonzero entries are retained. For path integrals, `N` is the
number of stored rows, independent of the number of qubits.

* One-shot statevector/path-integral calls use their direct native sampler.
* Other batches prepare a normalized, fixed-block probability index. Block size
  starts at 64 when `S >= 65536` and `S >= N/16`, otherwise 256. It doubles until
  the validation index plus draw scratch fit the configured budget. Composite
  reservations use each selected component's share of the common budget.
* Preparation also counts positive rows exactly. When at most a quarter of the
  candidate rows are positive, alias tables and histograms compact those rows,
  preserving their original indices. Zero-mass blocks are skipped during table
  construction; small positive probabilities are never pruned.
* A total weight outside `[2^-900, 2^900]`, which only an unnormalized input can
  have, is shifted by an exact power of two before any table is built: a tiny
  total up into `[0.5, 1)`, a huge one down to just below `2^900`. Finite
  weights whose sum overflows in row order are first summed again scaled down
  by `2^-128`, so they are accepted. Random targets are then never rounded to
  the subnormal grid and no subtotal can overflow. The counts equal those of the
  same relative weights at ordinary scale whenever every weight stays
  representable after the shift; a row pushed below the smallest double becomes
  zero, which can change the sampler chosen and so the seeded counts, though not
  the distribution of any reachable outcome.
* Path-integral amplitudes are scaled by an exact power of two before squaring
  only when a probability would otherwise vanish or overflow in `|a|^2`. The
  batch sampler rejects such a state while validating it, before its first
  draw: when the total of the squares is below `2^-970`, or when a square
  overflows. The amplitudes are then scaled up, so that the largest component
  lies in `[0.5, 1)`, or down, so that it lies just below `2^511`, and the
  state is sampled again. Every other state keeps its squared magnitudes, as in
  earlier builds, with no extra pass. Beside a total of at least `2^-970`, a row
  whose square underflowed lies more than `2^104` below the total: its real and
  imaginary squares can each round to zero.
* One-shot path-integral calls use QCSim's direct sampler. It weighs a retained
  total below `2^-900`, or one that overflows, from amplitudes scaled the same
  way into `[0.5, 1)`, and draws one uniform from its stream either way.
  QCSim's collapsing qubit measurement does the same, and so does the alias
  table of `sampling_policy=legacy` batches, so legacy one-shots and batches
  follow the same law. Ordinary states keep their unscaled weights.
* Seeded counts for these inputs, a total weight outside `[2^-900, 2^900]` (for
  a path integral, the total of the squared amplitudes, including one that
  overflows), differ from earlier builds. Those builds sampled tiny totals with
  a bias against the first and toward the last populated row, could drop rare
  rows of huge totals, and rejected or zeroed extreme path-integral
  amplitudes. A one-shot or measurement on a state that earlier builds
  rejected before drawing now draws from QCSim's stream, which shifts later
  one-shots and measurements on that simulator. A batch on such a state now
  draws from Maestro's batch stream, one draw per shot under
  `sampling_policy=legacy`. That shifts later batches on that simulator, while
  QCSim's stream is unaffected. Reseed the simulator to replay an earlier
  sequence after such a call.
* Alias construction is selected when `S >= max(512, M/4)` and its table fits the
  explicit scratch budget. Otherwise CDF lookup is used. Small CDFs can retain
  within-block cumulative probabilities; large serial CDF batches sort uniforms
  and scan each visited block in order. `M` is the compact row count when used,
  otherwise `N`.
* For sufficiently many shots, up to `2^20` rows use integer histograms and
  project only occupied rows into the requested output format. Counter storage
  and reduction work limit the useful worker count as well as available cores.
  Private counters have padding to avoid false sharing. Tiny alias draws remain
  serial because generating their random values dominates the work.
* Density matrices read real diagonal populations, validate and normalize their
  trace, and use them directly as probabilities. They never square populations.
  A joint marginal is prepared for a proper subset with at most 12 unique qubits
  when `S >= max(256, N/16)`. This preserves correlations and duplicate outputs.
* Path-integral batches normalize retained probability mass and borrow immutable
  full-width labels only for the duration of the call. Labels are not copied
  into alias rows. Alias indices are 32-bit when the row count fits, regardless
  of label width; wider row counts retain the 64-bit path. The current
  path-integral backend continues to disable internal multithreading. When the
  probability/label snapshot cannot fit, validation and each sorted target batch
  traverse the original rows directly. This uses bounded scratch at the cost of
  another traversal per batch; no full-width labels are copied. As with the
  ordinary CDF sampler, batch boundaries do not change the seeded result.

No long-lived prepared-distribution cache is shared across mutation, restore or
clone boundaries. No sampler changes the process-wide OpenMP configuration.

## Limits and scope

- Python `noise_seed` (and its `seed` alias) is an unsigned 32-bit integer.
- Python noisy execution runs at most 16 realization workers unless
  `max_simulators` is set explicitly; the CPU budget and
  `trajectory_max_memory_mb` still apply. One estimator
  (`Utils::RealizationStateBytes`) sizes Python, checkpoint, native and C++
  `NoiseAdd` realizations.
- Direct trajectory counts (`trajectory_v1`) are a Python and C++
  (`MaestroExecution::ExecuteNoisy`) feature. Native `execute` counts run
  independent one-shot realizations in parallel on the ordinary executor and
  report `sampling_strategy` only for frames; native rows report it always.
- Legacy `SimpleExecute` accepts `counts_policy` with the meaning it has
  elsewhere: under `auto`, an eligible Clifford circuit on a network that admits
  the QCSim stabilizer is counted from reference frames, with or without seeds,
  and reports `"sampling_strategy": "reference_frame_v1"`. Explicit `legacy`
  keeps the network executor. The network's own `RepeatedExecute*` methods do not select
  frames: they execute the circuits, often single noise realizations, that the
  layers above choose for them.
- C++ `NoiseAdd` runs Monte Carlo estimates on realization workers; each
  realization keeps its seed and values are summed in realization order, so
  seeded estimates match the serial loop. Eligible multi-shot realizations
  can also run on realization workers when the network samples them from one
  prepared state (terminal measurements on a dense QCSim method), whether
  seeded or unseeded. Each realization retains its assigned shot count and
  random streams; parallelism must not pool independent noise realizations.
- Prefix checkpoints, Monte Carlo noisy estimates and noisy fidelity live in
  libmaestro (`MaestroExecution::PrefixCheckpoint`, `EstimateNoisy` and
  `NoisyFidelity`); Python's `PrefixCheckpointedSimulator`,
  `noisy_estimate_montecarlo`, `coherent_estimate`, `full_noise_estimate` and
  `noisy_fidelity` wrap them. Estimate and fidelity realizations run on
  realization workers within `max_simulators` and `trajectory_max_memory_mb`;
  each keeps its seeds and results are combined in realization order, so they
  do not depend on the worker count. Workers start from a reduction cache
  warmed by the first realization, so `cone_cache_hits` matches a serial loop
  unless the noise changes a circuit's multi-qubit topology between
  realizations.
- Native `trajectory_max_memory_mb` applies to QCSim statevector, density
  matrix, stabilizer, MPS and MPO requests. All supported methods use logical
  shot streams by default. Those workers follow the memory budget where a
  state-size estimate is available: always on the stabilizer, and on MPS and
  MPO under a bond-dimension cap. Extended-stabilizer states have no known
  bound and are not budgeted by this dispatcher. The `SimpleExecute` API
  restores an omitted trajectory policy to `shot_v1` on each call.

## Testing and benchmarking

With `COMPILE_TESTS=ON`, build and run these executables (CTest names in
parentheses): `reference_frame_sampler_tests` (`reference_frame_sampler`),
`realization_workers_tests` (`realization_workers`), `alias_sampling_tests`
(`alias_sampling`), `adaptive_sampling_tests` (`adaptive_sampling`),
`sampling_order_tests` (`sampling_order`), `network_job_tests`
(`network_jobs`), `backend_specialization_tests` (`backend_specialization`) and
`gate_fusion_tests` (`gate_fusion_tests`); `maestro_request_tests`
(`maestro_request_api`) needs `MAESTRO_BUILD_REQUEST_TESTS=ON`. The Python
sampling suites are `tests/python/test_sampling_*.py`, including
`test_sampling_replay.py` for current-version replay and policy selection, and
`test_trajectory_config.py` for implicit/explicit defaults and worker settings.
Tests compare repeated executions from the current build, not archived
histogram fixtures from an older release. A build configured
with `MAESTRO_ENABLE_OPENMP=OFF` exercises the serial implementation.

The optional `sampling_benchmarks` target is excluded from the normal build and
CTest. It compares `legacy` and `reproducible_v1` on the same state, with one
warm-up each, alternating measurement order and median timings. For example:

```text
cmake --build build --config Release --target sampling_benchmarks
sampling_benchmarks sv 24 24 10000 0 random 24 packed 5
sampling_benchmarks dm 12 12 1000000 8 random 4 many 5
sampling_benchmarks path 1024 18 10000 0 random 18 many 5
sampling_benchmarks sv 24 24 10000 0 random 24 packed 5 4
```

Arguments are backend, register width, populated low bits, shots, permitted
internal threads (`0` disables internal threading), distribution, output bits,
format, repetitions, and optional outer worker count. Concurrent workers use
independently seeded clones and require internal threading disabled. The output
is CSV: backend, qubits, stored states, shots, threads, distribution, measured
bits, format, legacy milliseconds, adaptive milliseconds, speedup, outer workers.

Timings include probability preparation, RNG, sampling, projection and count-map
construction. State initialization, import support discovery, clone construction
and correctness checks are excluded. Concurrent timings include worker startup
and report elapsed time for the whole group. These are count-API measurements,
not full circuit execution timings. Thresholds are conservative policy defaults,
not a claim that one crossover is optimal on every machine.

### Measurements on 2026-10-09

The measurements below record earlier implementations. Their descriptions of
seeded dispatch and policy defaults are historical; the policy and replay
contract above supersedes them. They are not fresh measurements of the current
seed-independent dispatch.

Windows Release build, MSVC 19.44, OpenMP, Intel Core i9-13900KS (32 logical
processors), Aer disabled. Each result is the median of five timed calls after
warm-up. The legacy baseline already includes the earlier density preparation,
count conversion and compact alias-construction fixes. These numbers measure
the additional benefit of this change, not a comparison against Aer.

| Workload | Internal threads permitted | Legacy (ms) | New (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Statevector, 16 qubits, random, 10,000 shots | Disabled | 1.956 | 2.068 | 0.95x |
| Statevector, 24 qubits, random, 1 shot | Disabled | 8.539 | 8.224 | 1.04x |
| Statevector, 24 qubits, random, 10,000 shots | Disabled | 228.405 | 29.989 | 7.62x |
| Statevector, 24 qubits, random, 10,000 shots | Up to 8 | 228.425 | 9.633 | 23.71x |
| Statevector, 24 qubits, random, 1,000,000 shots | Up to 8 | 505.710 | 271.561 | 1.86x |
| Statevector, 24 qubits, GHZ, 8,000,000 shots | Up to 8 | 207.898 | 39.314 | 5.29x |
| Density matrix, 12 qubits, random, 1,000,000 shots | Disabled | 99.079 | 13.403 | 7.39x |
| Density matrix, 12 qubits, random, 1,000,000 shots | Up to 8 | 98.136 | 6.016 | 16.31x |
| Density matrix, 12 qubits, four-bit marginal, 1,000,000 shots | Disabled | 69.768 | 4.782 | 14.59x |
| Path integral, 1,024 qubits, 262,144 stored rows, 10,000 shots | Disabled | 13.162 | 5.475 | 2.40x |
| Path integral, 1,024 qubits, 4,096 stored rows, 1,000,000 shots | Disabled | 92.655 | 11.857 | 7.81x |

With four independent simulators sampling concurrently and internal threading
disabled, elapsed time for the whole group improved from 253.840 to 33.064 ms
(7.68x) for the 24-qubit random statevector at 10,000 shots per simulator, and
from 92.259 to 13.194 ms (6.99x) for the 12-qubit random density matrix at
1,000,000 shots per simulator.

The 40-case matrix included disabled threading and limits of 8 and 32 threads,
random/uniform/basis/GHZ states, partial measurements and four concurrent clones.
Gains are workload dependent: the small 16-qubit case regressed by about 6%,
and the 24-qubit GHZ case at 10,000 shots was approximately unchanged (1.02x).
Increasing the permitted team from 8 to 32 gave little additional benefit on
several workloads. Fixed policy choices preserve seeded replay across team sizes;
they do not promise the fastest possible algorithm for every machine.

## Historical measurements

The sections below record investigations as they were made in October 2026.
The artifact directories they cite (`investigations/...`) and environment notes
such as `/usr/local/libmaestro.so` were local to those runs and are not part of
the repository. Statements about defaults describe the state at the time: for
example, the scheduling follow-up predates `shot_v1` becoming the default
`trajectory_policy`. The sections above describe current behavior.

## Composite follow-up measurements

Measured on the same i9-13900KS (32 logical processors), Windows Release,
MSVC 19.44, with OpenMP enabled and Aer disabled in the benchmark library. Baseline:
the composite implementation before the batched merge/forwarding changes,
including the earlier adaptive preparation and RNG fixes. Both versions use
`reproducible_v1`. The same benchmark executable loaded separately saved old
and new libraries. Each time is the median of five calls after warm-up, with
preparation, RNG, drawing, projection and count construction timed together.
Initialization, seeding, cloning and correctness checks are outside timing.

`A x B` means A independent components of B qubits each. States have nonuniform
dense support generated by fixed Ry rotations followed by a CX chain within
each component, except the explicitly marked GHZ case. All qubits are measured
unless stated otherwise. Outputs use `SampleCountsMany`, except the packed
cases. Zero internal threads means multithreading is disabled. Concurrent-clone
times are elapsed time for the whole group, with the stated shots per clone.

| Layout / request | Shots | Internal threads | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 x 16 | 1,000,000 | 0 | 115.871 | 10.817 | 10.71x |
| 1 x 16 | 1,000,000 | 8 | 113.383 | 9.245 | 12.26x |
| 1 x 20, 12 measured | 10,000 | 0 | 3.833 | 3.155 | 1.21x |
| 1 x 20, 12 measured | 10,000 | 8 | 2.698 | 1.923 | 1.40x |
| 2 x 8 | 1,000,000 | 0 | 135.362 | 24.467 | 5.53x |
| 2 x 8 | 1,000,000 | 8 | 140.459 | 16.080 | 8.74x |
| 8 x 2, packed | 1,000,000 | 0 | 314.480 | 80.996 | 3.88x |
| 8 x 2, packed | 1,000,000 | 8 | 318.826 | 80.800 | 3.95x |
| 16 x 1 | 1,000,000 | 0 | 535.702 | 145.814 | 3.67x |
| 16 x 1 | 1,000,000 | 8 | 540.494 | 145.568 | 3.71x |
| 2 x 18, 6 measured/component | 10,000 | 0 | 5.652 | 3.253 | 1.74x |
| 2 x 18, 6 measured/component | 10,000 | 8 | 4.599 | 2.495 | 1.84x |
| 4 x 8, 2 measured/component | 1,000,000 | 0 | 222.577 | 36.817 | 6.05x |
| 4 x 8, 2 measured/component | 1,000,000 | 8 | 220.332 | 17.727 | 12.43x |
| 2 x 8 | 1,000,000 | 32 | 137.109 | 18.842 | 7.28x |
| Four concurrent clones of 2 x 8 | 1,000,000 | 0 | 138.918 | 28.198 | 4.93x |
| 4 x 8 GHZ, 2 measured/component | 1,000,000 | 0 | 148.703 | 18.136 | 8.20x |
| 16 x 1, packed | 64 | 0 | 0.046 | 0.022 | 2.09x |

All 18 before/after count checksums matched. Independent scalar-reference tests
also compare complete count maps and subsequent RNG continuation exactly, with
threading disabled and teams of 1, 3 and 8. Additional regressions cover a shared
1 MiB budget, repeated batches, 130-qubit composites, wide/duplicate outputs,
single-component mappings, clone independence, observers and saved states.
The 31-case sampling suite passed both with and without OpenMP; seven CMake
suites and 246 selected Visual Studio cases passed, including all 42 fusion
cases.

Thread scaling remains workload dependent: the 2 x 8 case took 16.080 ms with
eight threads and 18.842 ms with 32. Tiny component alias samplers deliberately
keep their RNG/draw/projection loop serial; their gains come from cheaper
projection and joint aggregation. These measurements establish improvements on
this machine, not universal speedup factors.

The optional CMake target `composite_sampling_benchmarks` builds
`tests/CompositeSamplingBenchmarks.cpp`. Its arguments are component count,
bits per component, shots, permitted internal threads (0 disables), measured
bits per component, `packed|many`, repetitions, optional outer worker count,
and optional `spread|ghz`. For example:

```text
composite_sampling_benchmarks 2 8 1000000 8 8 many 5
composite_sampling_benchmarks 2 8 1000000 0 8 many 5 4
```

## Trajectory and terminal-density scheduling follow-up

This section records the 2026-10-09/10 implementation work and measurements.
Historical sequence comparisons are validation performed at that time, not a
cross-version replay promise. The current policy section above governs defaults.

The regression introduced in `b4c838f` was in network scheduling:
`ExecuteLogicalShotBlocks` limited concurrency to one worker per 256-shot block
and disabled internal threading in every block. Thus 100 shots used one serial
worker even with a larger worker limit; 1,024 shots could use only four workers.
The fix keeps these logical blocks and their seed streams, but assigns unused
OpenMP capacity to internal work. Native arithmetic differences discovered by
exact-state tests required the selective serial paths described above.

Terminal density-matrix sampling had a separate dispatch problem: only QCSim's
reproducible policy selected a single simulator. Aer and QCSim legacy still
split one final-state sampling request across simulator jobs. The fix extends
single-simulator dispatch to those paths and removes unused terminal snapshots.
The Aer measurements below directly test the cost of that dispatch, rather than
inferring it solely from execution-versus-sampling timing ratios.

Measured on 2026-10-09 with GCC 13.3 Release, Linux/WSL2, Intel Core i9-13900KS,
eight permitted OpenMP threads and one OpenBLAS thread. The environment's Intel
OpenMP/MKL preload was retained for both versions. These are controlled network
benchmarks, not a replay of an external circuit report. The baseline is the
regressed `9cbda78` build, compared with the scheduling/arithmetic fixes above.

The harness fixes the backend, disables gate fusion, reuses the prepared prefix,
and requests 16 simulators. Each median comes from three before/after pairs with
alternating order; every process performs an untimed warm-up. Separate saved
executables and libraries prevent rebuilding from replacing the baseline.

| Workload | Shots | Before (s) | After (s) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| QCSim statevector, 20 qubits, mid-circuit | 100 | 0.745 | 0.475 | 1.57x |
| QCSim statevector, 22 qubits, mid-circuit | 100 | 4.947 | 3.278 | 1.51x |
| QCSim statevector, 24 qubits, mid-circuit | 100 | 20.102 | 14.633 | 1.37x |
| QCSim density matrix, 10 qubits, mid-circuit | 100 | 0.682 | 0.234 | 2.91x |
| QCSim density matrix, 11 qubits, mid-circuit | 100 | 4.159 | 1.980 | 2.10x |
| QCSim density matrix, 12 qubits, mid-circuit | 100 | 17.925 | 11.612 | 1.54x |
| QCSim statevector, 20 qubits, mid-circuit | 1,024 | 4.528 | 3.512 | 1.29x |
| QCSim density matrix, 10 qubits, mid-circuit | 1,024 | 4.210 | 3.477 | 1.21x |
| Aer density matrix, 11 qubits, terminal | 10,000 | 0.966 | 0.062 | 15.51x |
| Aer density matrix, 12 qubits, terminal | 10,000 | 3.972 | 0.271 | 14.65x |
| QCSim legacy density matrix, 11 qubits, terminal | 10,000 | 0.745 | 0.188 | 3.97x |
| QCSim reproducible density matrix, 11 qubits, terminal | 10,000 | 0.206 | 0.196 | 1.05x |

Controls with one simulator and one OpenMP thread were approximately unchanged:
0.730 to 0.737 seconds for the 20-qubit statevector and 0.775 to 0.764 seconds
for the 10-qubit density matrix, both with 100 mid-circuit shots. Peak process
RSS for the 12-qubit Aer terminal case fell from 1,572 to 1,315 MiB. This is
measured process memory, including preparation and runtime allocations, not
the theoretical size of 16 simultaneously live state/snapshot pairs.

Every measured call conserved its shots and repeated its seeded histogram.
All QCSim `reproducible_v1` cases matched the baseline histograms exactly.
Serial measurement scans/sums and density diagonal gates remain a throughput
limit in reproducible trajectories; these results do not claim recovery of all
performance from before `b4c838f` or uniform speedups on other circuits.

Validation passed seven OpenMP CTest suites (`alias_sampling`,
`adaptive_sampling`, `sampling_order`, `network_jobs`, `backend_specialization`,
`gate_fusion_tests`, `fusion_routing_tests`) and both `network_jobs` and
`backend_specialization` with OpenMP and Aer disabled. The added tests compare
exact statevector amplitudes and density entries with native serial arithmetic,
including gates, fusion, measurements, resets, and density channels. Network
tests cover both entry points, prefix reuse, seed zero/high bits, readout noise,
conditional gates, shot-block boundaries, worker budgets and restoration of the
caller's OpenMP settings. Terminal tests check one preparation, correlations,
readout, preservation of an existing checkpoint, and clearing trajectory mode.

The investigation artifacts are under `investigations/sampling_regression`:
`compare_fixed.py`, `fixed_comparison.jsonl`, `fixed_summary.txt`, and
`fixed_metadata.json`, plus `large_comparison.jsonl` and `large_summary.txt`
for the 24-qubit statevector and 12-qubit density trajectories. Library paths
are selected explicitly for validation;
the environment otherwise prefers an installed library in `/usr/local`.

### Follow-up measurement and scheduling work (2026-10-10)

The follow-up starts from `9e536c1` and uses the circuits from
`investigations/aer_vs_qcsim`, with a fixed QCSim backend, automatic fusion,
16 permitted OpenMP threads and one OpenBLAS thread. Compiler, hardware and
preload settings match the saved baseline. Each fresh process performs an
untimed warm-up; measurements use explicitly selected saved libraries and check
shot conservation and repeated seeded counts. Compatibility-policy comparisons also
require identical counts between baseline and implementation.

The final small cases use two alternating pairs with two timed calls per
process. The large cases use one three-way comparison with two timed calls per
process. Each 4,096-shot variant has one full warm-up and one timed call, so its
reported speedup has less timing precision. Warm-up durations are retained in
the progress JSONL files. All timings use request seed 12345; `30c02ad` was not
rebuilt for this follow-up.

The retained changes are consecutive statevector measurement subspaces,
independent per-shot streams (now the default), shared dense checkpoints and
bounded dense-state concurrency. Two attempts to cap small-state internal teams were rejected:
both helped some workloads but slowed 16-qubit random trajectories. Simply
enabling the native parallel density diagonal kernel was also rejected because
it changed the default policy's exact arithmetic. A reproducible fixed-tile
density diagonal prototype passed correctness checks and accelerated isolated
gates, but improved end-to-end 12-qubit QAOA by only 1% (34.15 to 33.79 seconds).
Its 10-qubit gains were 3% for random trajectories and 7% for QAOA. It was left
out because the workload-level benefit did not justify a second gate kernel.

Artifacts in `investigations/runtime_followup` include `benchmark.py`, saved
libraries for each stage, paired timings and histograms in the stage JSON files,
`summary.json`, `artifact_hashes.json`, and build/test logs. `step3_v1.json`
records the rejected whole-job cap. `step3_shots_ignored_option.json` is an
invalid early run: the legacy API had ignored the new option. It is excluded
from the summary. The API now explicitly forwards trajectory controls, and the
harness verifies the returned policy before accepting per-shot measurements.

Validation includes exact amplitude comparisons against the native serial
statevector path; serial/parallel density comparisons and native numerical
agreement; clone snapshot replacement and lifetime tests; worker, prefix reuse,
host entry point, classical randomness, readout, reset, single-shot and tail-job
replay tests; and public API validation including memory-dependent scheduling.
Eight OpenMP CTest suites and three focused suites without OpenMP/Aer pass.
`distribution_check.py` checks feed-forward/reset distributions over 20 seeds
for each dense backend (163,840 shots total), including forbidden outcomes and
chi-square checks against analytic probabilities.

Selected incremental measurements (seconds; each row compares the stated
experiment with its own control, not always the original baseline):

| Experiment | Workload, 100 shots | Control | Experiment | Decision |
| --- | --- | ---: | ---: | --- |
| Per-kernel thread caps | Statevector random, 16 qubits | 0.229 | 0.368 | Rejected |
| Consecutive measurement subspaces | Statevector random, 16 qubits | 0.208 | 0.115 | Retained; identical counts |
| Whole-job small-state thread cap | Statevector random, 16 qubits | 0.108 | 0.209 | Rejected |
| Per-shot streams versus updated block policy | Statevector random, 16 qubits | 0.116 | 0.063 | Retained; now the default, new seeded sequence |
| Checkpoint sharing and dense-worker limits | Per-shot statevector random, 20 qubits | 1.978 | 1.363 | Retained; identical counts |
| Checkpoint sharing and dense-worker limits | Per-shot density QAOA, 10 qubits | 1.662 | 0.683 | Retained; identical counts |
| Fixed-tile density diagonals | Per-shot density QAOA, 12 qubits | 34.150 | 33.789 | Rejected; marginal end-to-end gain |

Final comparisons with the starting `9e536c1` build (seconds). These measurements
predate promoting `shot_v1` to the default; the saved binaries retain the earlier
default, so columns identify the explicit policy:

| Workload | Shots | Baseline | `block_v1` (replay) | `shot_v1` (now default) |
| --- | ---: | ---: | ---: | ---: |
| Statevector GHZ, 16 qubits | 100 | 0.146 | 0.049 | 0.009 |
| Statevector random, 16 qubits | 100 | 0.225 | 0.119 | 0.061 |
| Statevector random, 20 qubits | 100 | 3.228 | 1.650 | 1.339 |
| Statevector QAOA, 20 qubits | 100 | 2.659 | 1.042 | 0.838 |
| Statevector random, 24 qubits | 100 | 88.85 | 44.63 | 38.30 |
| Density QAOA, 10 qubits | 100 | 0.911 | 0.921 | 0.703 |
| Density QAOA, 12 qubits | 100 | 40.25 | 40.71 | 33.29 |
| Statevector random, 20 qubits | 4096 | 118.96 | 53.66 | 53.36 |
| Density random, 10 qubits | 4096 | 97.13 | 55.49 | 55.86 |
| Statevector terminal random, 20 qubits | 10000 | 0.043 | 0.045 | 0.043 |
| Density terminal random, 12 qubits | 10000 | 1.022 | 1.031 | 1.027 |

Every `block_v1` histogram matched the original build exactly. Terminal
histograms also matched under `shot_v1`; terminal timing ranges overlapped, so
no terminal speedup is claimed. Density QAOA with 100 shots was approximately
unchanged with `block_v1` (about 1% slower), while `shot_v1` improved it.

At 4,096 shots, peak process RSS fell from 592.5 to 144.7 MiB for statevectors
and from 592.2 to 144.3 MiB for density matrices with `block_v1` (about 76% lower).
For the 100-shot large cases, `block_v1` RSS fell from 1,072 to 816 MiB
(24-qubit statevector) and from 1,071 to 815 MiB (12-qubit density matrix).
The `shot_v1` large cases used roughly the original memory footprint to run two
live trajectories. The memory budget is a concurrency estimate, as described above.

The `block_v1` policy still leaves the 16-qubit random case slower than the
historical 0.071-second `30c02ad` result quoted in the earlier report; the `shot_v1`
result is 0.061 seconds here. Preserving the old block RNG sequence prevents
distributing a short block across independently seeded workers. These historical
numbers are context, not a fresh cross-commit comparison.

Reproduce the final suite with `python3 -B investigations/runtime_followup/final_suite.py`
once the saved libraries and existing circuit generator are present. Run it without
competing builds or benchmarks. Raw timings, counts, configurations and library
hashes are retained in `final_small.json`, `final_large.json`, and `final_scaling.json`.

The subsequent Python/default-policy follow-up exposes `trajectory_policy`,
`trajectory_max_memory_mb` and `max_simulators` through `SimulatorConfig`, with
constructor/setter validation and pickle support. `shot_v1` is now the network
and legacy-request default and is advertised by capability discovery. Explicit
`block_v1` remains available as a fixed-block algorithm. Validation at that time
passed 49 Python tests (19 hardware/optional-dependency skips), the native request suite, and the network
job suite, including omitted-versus-explicit policy checks.

With 16 OpenMP threads, the Python 16-qubit random/100-shot check measured
0.061 seconds for the new default versus 0.114 seconds for explicit `block_v1`
(one warm-up, three timed calls per setting). The default matched explicit
`shot_v1` counts exactly. Fresh-process comparisons against the saved final
library's explicit `shot_v1` also matched every histogram for statevector
random circuits at 16/20 qubits and density QAOA at 10 qubits. Repeated timings
were mixed, including roughly 5% slower at 20 qubits; this follow-up makes the
previously measured per-shot policy the default, rather than introducing a new
kernel speedup. Results are in `python_interface_benchmark.json` and
`default_policy_replay_repeated.json`, with the corresponding test logs in
`investigations/runtime_followup`. Local Python tests explicitly load
`build/libmaestro.so`; an older `/usr/local/libmaestro.so` otherwise takes
precedence in this environment.
