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
use their native RNG streams. The native one-shot conversion and quantum
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
QCSim trajectory execution uses fixed logical blocks of 256 shots, seeded by
block identity, independently of the number of outer workers. State copies are
created when jobs start, so queued jobs do not each retain a quantum-state copy.
Trajectory blocks disable internal threading even with one outer worker, so
native collapse reductions use the same arithmetic regardless of outer pool or
OpenMP team size. Large single-worker trajectories may therefore run more slowly;
terminal count batches still permit internal threading. This guarantee does not
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
