# QCSim sampling policies

The QCSim statevector, density-matrix and path-integral adapters use
`sampling_policy=reproducible_v1` by default. Set `sampling_policy=legacy` to
use the previous count-sampling algorithms. The clone reseeding fixes apply
to both policies. QCSim composite components also use the new prepared sampler.

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
the same serially generated uniform sequence. The new batch kernels convert
exactly one `mt19937_64` word per shot to a specified 53-bit value in `[0,1)`.
Packed and bit-vector count APIs select the same plan and consume the same RNG
stream for the same request. Reordered and duplicate selected qubits retain
their caller-specified positions.

Statevector and path-integral one-shot calls keep the native direct sampler and
native RNG stream. Statevector one-shot sampling fixes the native summation path
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
Aer and legacy QCSim batches can therefore produce different seeded histograms
from the former multi-simulator dispatch; their probability distributions are
unchanged. QCSim's reproducible terminal batches already used one simulator.
QCSim trajectory execution defaults to `trajectory_policy=shot_v1`, which
distributes individual shots across workers, including short dynamic requests.
It assigns each shot separate quantum, readout and
classical-random streams derived from the request seed and absolute shot index.
Changing job boundaries, worker counts or the trajectory memory budget does not
change its seeded counts. To replay the older seeded sequence, explicitly select
`trajectory_policy=block_v1`: fixed logical blocks of 256 shots, seeded by block
identity independently of the number of outer workers. **The new default changes
seeded counts for dynamic circuits.** Both policies sample the same distribution.
These controls apply to reproducible QCSim
statevector, density-matrix and path-integral trajectories, not terminal batches
or `sampling_policy=legacy`.

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
The latter applies to dense statevector/density-matrix trajectory concurrency
and defaults to 1024 MiB. It estimates live states and checkpoints, always
allowing at least one worker. It is a **soft concurrency budget**, not a cap on
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
updates. Each job restores its thread's previous OpenMP limit when it finishes;
calls inside an existing OpenMP team and path-integral jobs remain serial
internally. Measurement scans and probability sums retain serial arithmetic,
preserving the existing block seeds and outcomes across outer pool and OpenMP
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

## Testing and benchmarking

With `COMPILE_TESTS=ON`, build and run `alias_sampling_tests`,
`adaptive_sampling_tests`, `sampling_order_tests`, `network_job_tests`,
`backend_specialization_tests` and `gate_fusion_tests`. A build configured with
`MAESTRO_ENABLE_OPENMP=OFF` exercises the serial implementation.

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
`block_v1` remains available for replay. Validation passed 49 Python tests (19
hardware/optional-dependency skips), the native request suite, and the network
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
