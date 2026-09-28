# Gate-fusion validation

Validated on 2026-09-28 with the local QCSim source and an NVIDIA RTX 5090.

- The new CPU fusion suite passed with Aer enabled and with Aer disabled.
  The latter build used older QCSim headers without the map accessors, exercising
  the fallback to local routing.
- The real-GPU fusion suite passed for statevector, density matrix, MPS, and
  MPO, including arbitrary complex matrices, permutations, snapshots, and
  routing. Statevector coverage includes all six one-/two-/three-qubit matrix
  entry points, row- and column-major storage, and both precisions.
- The distributed matrix/fusion suite passed for conventional and Ex local
  statevectors. Conventional fusion produces three-qubit blocks; Ex local and
  Ex MPI keep Maestro fusion disabled while exposing generic gates.
- All 12 selected CPU/GPU/distributed integration and configuration CTests
  passed. Two-rank MPI integration passed with single and double precision and
  automatic, fixed, and pinned layouts, including generic three-qubit gates,
  snapshots, clones, sampling, and measurement.
- Selected existing CPU suites passed 383 cases and 816,173 assertions.
- Existing GPU MPS/MPO cross-checks and density-matrix/MPO suites passed 163
  cases and 6,357 assertions on the real plugin.
- Native request tests passed 3,634 checks; network jobs passed 10,676 checks.
  Targeted CTest runs also passed sampling, distributed mapping/configuration,
  GPU device/SVD configuration, and fusion tests.
- The broader Python selection passed 299 tests with four skips. The final
  configuration/truncation selection passed 33 tests with six GPU skips.
- The pinned QCSim revision already contains the routing-map accessors; the
  obsolete dependency patch and its application step have been removed.

Review-fix validation on the same machine:

- The 10 selected CPU/GPU/distributed fusion, routing, network and configuration
  CTests passed, as did two-rank MPI integration.
- Snapshot regressions cover native reinitialization, vector/basis/mixture
  imports, ordinary and destructive saves, and snapshots that survive reset or
  composite-child reconstruction. GPU MPO retains its existing error for a
  missing snapshot without losing newly submitted gates.
- Routing regressions assert that external callbacks actually execute, backend
  observers survive circuit preparation, and multi-shot snapshots contain the
  suffix, including both worker entry points and estimator-prepared states.
- Alternating gate/measurement/classical-boundary circuits with 50 and 100
  measurement boundaries perform no operation-list reinstalls after preparation,
  with fusion either on or off. Statevector execution builds no routing plan.
- Asymmetric complex two-qubit matrices with both target orders pass for QCSim
  tensor networks and GPU MPS with fusion on and off. Immediate MPS/MPO clone
  counters are independent of the original simulator and survive its destruction.
- Conventional distributed CZ/CP/Rz sequences preserve the existing layout
  with fusion on and off, avoiding the previously observed global/local exchange.
- All 20 native GPU regression tests pass, including MPS/MPO callbacks returning
  invalid meeting positions. A focused QCSim MPS test also passes with assertions
  enabled. The companion native MPS fixes validate the fallback before asserting.
- The Aer-enabled fusion, request, network, routing and sampling suites pass.
  Circuit-completion tests verify that an unfused backend receives no new flush
  calls.

MPI validation must load the same Open MPI 5.0.10 and UCX 1.20.0 libraries used
at build time. The review-fix run used `UCX_TLS=tcp,cuda_copy,self` and
`UCX_MEMTYPE_CACHE=n`, following the distributed plugin's documented setup.
Loading the system Open MPI/UCX libraries instead is not a valid test of this
build.

Distributed numerical validation used development plugin builds with license
checks disabled. The conventional shared-device and two-rank MPI cases used
one physical RTX 5090; multiple physical GPUs and multiple hosts were not
exercised. The configuration mock also checks the older-plugin fallback to
two-qubit conventional fusion when the three-qubit symbols are absent.

An MPI shutdown failure was isolated with AddressSanitizer to the distributed
plugin's `FreeLib()`: a static library-singleton destructor called it after the
thread-local error string had been destroyed. Clearing that string wrote to
freed memory. The companion distributed-plugin fix removes this unnecessary
access, since the library context owns no resources. A focused shutdown test
fails under AddressSanitizer before the fix and passes afterward. The complete
Maestro MPI integration also passes against the instrumented fixed plugin and
in three consecutive release runs. Both local and MPI shutdown regressions
pass under AddressSanitizer, and the library's two-rank generic-matrix suite
passes with the fixed release plugin.

The manual repeated-pair benchmark produced these backend submission counts:

| Method | Qubits | Fusion off | Fusion on |
|---|---:|---:|---:|
| CPU statevector | 20 | 456 | 38 |
| CPU density matrix | 8 | 168 | 14 |
| CPU MPS | 14 | 312 | 52 |
| CPU MPO | 8 | 168 | 28 |
| GPU density matrix | 8 | 168 | 28 |
| GPU MPS | 14 | 312 | 52 |
| GPU MPO | 8 | 168 | 28 |

Timing includes the final flush. The benchmark compares 64 fixed probability
probes outside the timed region; the dedicated correctness suites compare full
small states and density matrices. The GPU repeated-pair cases ran faster with
fusion, while CPU MPO timings varied between runs. Sparse cases demonstrated
that fusion overhead can outweigh the savings. These are smoke measurements,
without timing assertions or a claim of a universal speedup.

Build and run `gate_fusion_tests`, `gate_fusion_tests --gpu`,
`gate_fusion_tests --distributed`, and the optional `gate_fusion_benchmark`
target (`--gpu` for GPU timings). CTest registers the hardware fusion runs as
`gate_fusion_gpu` and `gate_fusion_distributed`. Enable
`MAESTRO_BUILD_MPI_GPU_TESTS` for `distributed_gpu_mpi`. Ensure the runtime loads
the freshly built Maestro and plugin libraries when an older installation is
also present.
