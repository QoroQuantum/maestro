# Gate fusion

Maestro enables gate fusion by default for these simulator methods:

| Backend | Method | Maximum fused width |
|---|---|---|
| QCSim | Statevector, density matrix | 3 qubits |
| QCSim | MPS, MPO, tensor network | 2 qubits |
| Composite QCSim | Statevector | 3 qubits |
| GPU | Statevector | 3 qubits |
| GPU | Density matrix, MPS, MPO | 2 qubits |
| Distributed GPU, conventional backend | Statevector | 3 qubits |

Aer, including composite Aer, retains its own behavior. Maestro fusion is off
for QuEST, stabilizer, extended stabilizer, Pauli propagation, path integral,
and the GPU tensor-network adapter. Those adapters do not expose the generic
matrices needed by this layer. Distributed Ex local (the default) and Ex MPI
use the library's own caching/fusion: Maestro fusion is always off for these
backends, even when `gate_fusion` is requested. Their generic one-, two-, and
three-qubit gate APIs remain available.

The GPU plugin must export the new statevector matrix APIs for three-qubit
fusion. With older plugins, single-device statevector fusion stays off and
conventional distributed fusion is limited to two qubits. Calling a missing
generic gate API reports an error.

## Configuration

Direct C++ and low-level simulator interfaces accept
`Configure("gate_fusion", "false")` on fusion-aware adapters. Use `"true"` to
enable it again. Switching flushes pending gates first. The network interface
accepts the same setting and forwards it only to the appropriate adapters.
Enabling the setting never enables fusion on an unsupported method.
`GetConfiguration("gate_fusion")` reports this requested setting;
`IsGateFusionEnabled()` reports whether the selected backend can actually use it.

In Python:

```python
config = maestro.SimulatorConfig(gate_fusion=False)
```

Native requests accept `simulator.options.gate_fusion` as a boolean. This is
independent of `optimize_circuit`.

MPS/MPO fusion changes when truncation occurs and can change approximate
results at a finite bond dimension or nonzero cutoff. Disable fusion when
comparing against the previous sequence of truncations. Roundoff can also
change with matrix multiplication order on exact methods.

## Cache and state boundaries

`GateFusion<Gate>` is a reusable cache with no backend dependency. A gate
descriptor supplies its ordered targets and matrix. Cached blocks have
disjoint supports. An incoming gate absorbs overlapping blocks if their union
fits the backend width; otherwise only blocks that do not fit are emitted first.
Disjoint pending blocks can remain cached. Matrices use the first target as
the least significant local bit, with explicit permutations when targets differ.
The later matrix multiplies on the left. Single, unfused gates keep their
native backend entry points.
GPU statevectors preserve native diagonal/permutation-only groups instead of
replacing them with dense matrices. Mixed groups containing dense operations
can still fuse. This avoids unnecessary distributed qubit exchanges and keeps
specialized kernels available; it does not promise a speedup for every circuit.

`FusionState` and `FusionSimulator` own the immediate implementation and share
the cache across all gate entry points. Reads, sampling, measurements, channels,
saves, clones, configuration changes, and explicit `Flush()` synchronize pending
gates. State replacement, reset, clear, and restore discard superseded pending
work. Replacing native state also invalidates its saved fusion context; reset
retains snapshots on backends whose native reset retains them. Circuit execution
flushes Maestro fusion at completion and around non-gate operations only when
it is enabled. Aer and Ex backends retain their native buffering behavior.
Nonunitary generic operators are immediate boundaries. Three-qubit named gates
are decomposed before fusion when the backend width is two.

Composite QCSim has exactly one cache at the composite boundary. Its individual
simulators instantiate `ImmediateQCSimSimulator` directly and contain no fusion
state or gate-submission code.

`GetGateFusionMaxQubits()`, `IsGateFusionEnabled()`, and
`GetGateFusionStatistics()` expose capability, effective setting, and cumulative
submitted/backend/fused-block counts. Statistics count execution attempts that
reach the submission path; restoring state does not rewind them. Internal bond
dimension telemetry uses `GetExecutedMaxBondDimension()` so collecting it does
not flush every gate; the public current-bond query synchronizes.
Backend counts measure calls emitted by the adapter; a native call may
decompose further inside its backend.

## MPS/MPO swap optimization

`SetUpcomingGates()` preserves the source circuit. For active MPS/MPO lookahead,
it runs the same cache algorithm without matrix construction to prepare routing
operations. Initial placement can also request this view lazily. Statevectors
do not build routing plans. Source
positions and emitted routing positions are separate. Both initial placement
and meeting-position lookahead use the emitted sequence. The routing view is
never executed as a circuit.

Every meeting-position callback reads the actual backend logical-to-chain map
and current bond dimensions. It does not predict that map by replaying source
gates. Early observations or a different runtime operation invalidate an
incompatible prepared sequence and use local routing until an explicit circuit
preparation, reset, or restore prepares a new view. Lookahead stops at non-gate
boundaries, but the prepared view is reused across ordinary measurements and
classical boundaries. With fusion off, advancing the cursor does not reinstall
the source list. Saves, clones, and restores carry source context; multi-shot
execution saves after installing the remaining circuit.

QCSim needs the read-only `getQubitsMap()` accessors on MPS/MPO. Maestro's pinned
revision already includes them. External `QCSIM_INCLUDE_DIR` checkouts should
also contain those accessors; Maestro does not patch dependency sources.

The GPU plugin optionally exports `MPSGetQubitsMap` and `MPOGetQubitsMap`, which
copy the current map into caller-owned storage without CUDA work. Current
plugins include both exports. Older QCSim headers and GPU plugins remain usable,
but external circuit lookahead is unavailable without the map accessor,
including when `gate_fusion=false`. Native local routing and initial placement
remain available. `IsRoutingLookaheadEnabled()` reports effective availability.
Setting the lookahead depth does not override an explicit
`SetUseOptimalMeetingPosition(false)`.

The callbacks treat unavailable routing metadata or failed cost estimation as
a request for native local routing. Native MPS implementations must process the
fallback sentinel before asserting that the meeting position is valid, including
in debug builds.

MPO now shares MPS network initial-placement and swap-optimization settings.
These settings apply independently of fusion and may affect finite-bond
truncation; disable the corresponding network optimization when comparing
against the previous MPO routing. Generic-matrix target-order corrections for
QCSim tensor networks and GPU MPS, and the GPU clone counter fix, also apply with
fusion disabled.

## Extending and testing

To support another backend, implement immediate generic gate methods with the
documented target order, wrap the immediate simulator in `FusionSimulator`, and
override `GetGateFusionMaxQubits()` with two or three for eligible methods.
Other methods return zero. The default interface remains opt-out.
Override `PreserveStructuredGates()` where native structured kernels should be
kept for groups consisting solely of diagonal/permutation gates. Backends with
external circuit lookahead expose `IsRoutingLookaheadEnabled()` independently.

Build `gate_fusion_tests` and run it through CTest. It covers independent dense
matrix checks, two/three-qubit fusion, backend equivalence, reversed targets,
generic matrices, dynamic circuit boundaries, snapshots, sampling, observers,
and MPS/MPO routing. `gate_fusion_tests --gpu` additionally checks a real GPU
plugin, including statevector matrix layouts and fusion. Use `--distributed`
to check both conventional and Ex matrix wrappers and their fusion policy.
These runs require an available licensed device. Existing sampling, network-job,
request, Python, and backend suites cover the external execution paths.
`fusion_routing_tests` checks callback execution, preparation cost across
boundaries, backend observers, multi-shot suffix snapshots, callback fallback,
and the absence of circuit-completion flushes on unfused backends.

The optional `gate_fusion_benchmark` target prints elapsed execution time,
submitted/backend gate counts, and a numerical cross-check with fusion disabled.
It includes repeated-pair and sparse circuits, measures the final flush, and has
no timing assertions. Pass `--gpu` for GPU MPS/MPO/density-matrix measurements.
See [validation results](gate_fusion_validation.md) for the exercised backends
and the remaining hardware-validation limits.

## Public interfaces

Python `Simulator` objects expose `IsGateFusionEnabled()`,
`GetGateFusionMaxQubits()` and `GetGateFusionStatistics()`. The statistics object
has read-only `submittedGates`, `backendGates` and `fusedBlocks` fields. Queries
leave pending gates cached; call `Flush()` before inspecting final backend
counts. These queries describe Maestro's layer, not Aer or Ex's native fusion.

The C API exports the same queries. `GetGateFusionStatistics(sim, &statistics)`
fills a caller-owned `MaestroGateFusionStatistics` and returns 1 on success.
Both dynamic C++ wrappers expose these functions. Older libraries without the
new exports can still be loaded; the new wrapper calls then return zero.

Python and C also expose `ApplyGenericOneQubitGate`, `ApplyGenericTwoQubitGate`
and `ApplyGenericThreeQubitGate`. Python accepts nested complex sequences,
including NumPy matrices, sized 2x2, 4x4 or 8x8. C accepts row-major buffers of
interleaved real/imaginary doubles (8, 32 or 128 doubles respectively). The first
target is the least-significant matrix-index bit. Inputs are copied on submission.
Invalid dimensions, repeated/out-of-range targets, nonfinite matrices and
unsupported operations raise Python exceptions; the C entry points return 0.
Generic operators need not be unitary; backend support still applies.

Legacy C `SimpleExecute` and `SimpleEstimate` accept a JSON boolean
`gate_fusion`, defaulting to true on each call, and reject other value types.
Their results include `gate_fusion.enabled` and `gate_fusion.max_qubits` for the
backend actually used. The legacy CLI accepts `--gate-fusion=false` or
`--gate-fusion=true` (the default).

Native request results include `execution_metadata.gate_fusion` with `requested`,
`enabled` and `max_qubits`, and echo the boolean in
`execution_metadata.configured_options.gate_fusion`. Execution workers capture
capability before simulator recreation, including when automatic selection
chooses an unsupported method. Width describes capability independently of the
requested switch; it is zero when Maestro fusion is unsupported.
