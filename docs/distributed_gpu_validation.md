# Distributed GPU integration validation — 2026-09-11

The integration was tested locally on an RTX 5090 and in an isolated checkout
on the supplied two-T4 server. Existing server source trees were preserved.

## Coverage and results

| Check | Result |
| --- | --- |
| Maestro and Python extension without MPI linking | Passed locally and on the server; no undefined MPI symbols or MPI dependencies |
| Default build without MPI discovery | Passed; MPI classes remain available and missing plugins fail at runtime |
| Distributed library full local MPI-enabled suite | 28 passed, 23 hardware/MPI skips; ABI tests pass without CUDA-aware MPI and with GPUs hidden |
| Distributed library full server suite, including benchmark smoke tests | 50 passed, 1 skipped (UCX lacks MPI_THREAD_MULTIPLE) |
| Maestro distributed CTests | Local: 5 passed, 1 hardware skip; server: all 7 passed |
| Hardware-independent configuration and C ABI error handling | Passed |
| Local Ex statevector on the RTX 5090 | Passed |
| Two conventional logical shards sharing the RTX 5090 | Passed |
| Two physical T4 GPUs, local Ex distribution | Passed |
| Two MPI ranks, one physical T4 per rank | Passed |
| MPI peer-to-peer groups (`mpi_p2p_bits=1`) on the T4 pair | Passed |
| Python local execution, estimation, state export and config pickling | Passed locally and on the server |
| Python two-rank MPI execution and replicated outcomes | Passed on both ranks |
| Existing Python configuration, enum, seed and estimation regressions | 71 passed, 2 skipped |

The C++ numerical tests compare all named gates and asymmetric generic Eigen
matrices with QCSim, in single/double precision and automatic/fixed/pinned
layouts. They cover GPU/host snapshots, destructive restore, independent
clones, initialization from amplitudes, reversed/subset sampling, Bell-state
statistics, joint collapse and reset. Tests also check default global qubit 0
for two shards, actual device placement, invalid configuration and both factory
ownership variants. Lifecycle tests reject shutdown with live MPI states and
reject new MPI execution after terminal shutdown, including across the Python
extension/core-library boundary.

Python tests exercise the network path, including operations after measurement,
idle observable qubits and full-register statevector export. The latter cases
exposed and verified fixes for qubit compaction and initial backend selection.

Physical cross-machine MPI execution was not tested. Shared-device tests do
not establish additional GPU capacity. Hardware tests skip explicitly when
the required devices/plugin are unavailable; the local two-physical-GPU case
was skipped and then executed successfully on the server.

## MPI-independent runtime interface

The MPI plugin now exposes `gpusim_mpi_runtime.h`, retaining its native MPI API.
Runtime tests cover dynamic loading from a C program with no MPI linking,
pre-initialization and post-finalization rejection, default world and reversed
Fortran communicators, automatic device discovery, collective output/context
validation, and communicator ownership after the caller frees its handle.
Maestro's runtime declarations were checked against the plugin's C structures
for size, alignment, and every field offset.

The final core library and Python extension were checked using `readelf -d` and
`nm -D --undefined-only` on both machines: neither depends on MPI or references
MPI symbols. Only the optional C++ MPI test executable links MPI. The former
`MAESTRO_ENABLE_MPI_GPU` feature gate has been removed; use
`MAESTRO_BUILD_MPI_GPU_TESTS=ON` only to build the application-side MPI tests.
The final server integration run passed all seven CTests, explicit P2P execution,
and Python MPI execution on both ranks. The Python interpreter shutdown warning
below remains reproducible.

## Server reproduction

Source/build directories:

- `/home/aroman/maestro-distributed-integration-20260911`
- `/home/aroman/maestro-distributed-plugins-20260911`

The plugins were built from the current local `maestro-gpu-distributed` source
with `ENABLE_MPI=ON`, `ENABLE_LICENSE_CHECK=OFF`, and CUDA architecture 75.
The server used Open MPI 5.0.10 and UCX 1.20.0 from its existing isolated install.
Python 3.14 used pytest and mpi4py wheels extracted into the integration
checkout's `python-deps`; system Python packages were not changed.

Source the server's existing
`/home/aroman/maestro-gpu-distributed-tests-20260910/validation/mpi-env.sh`, then
use the new Maestro build first on `LD_LIBRARY_PATH` and set both
`MAESTRO_DIST_GPU_LIBRARY` and `MAESTRO_DIST_MPI_GPU_LIBRARY` to the corresponding
new plugin `.so` paths. The launch flags were:

```sh
export UCX_TLS=^cuda_ipc
mpirun --map-by slot:OVERSUBSCRIBE --bind-to none \
  --mca pml ucx --mca pml_ucx_tls any -n 2 \
  /home/aroman/maestro-distributed-integration-20260911/build/bin/distributed_gpu_tests --mpi
```

For Python, set `MPI4PY_RC_THREAD_LEVEL=serialized` before importing mpi4py.
The installed UCX transport rejects mpi4py's default request for
`MPI_THREAD_MULTIPLE`; the same failure reproduced with mpi4py alone.
Maestro's distributed network path executes serially and works with the
serialized thread level.

The server's Python 3.14 + mpi4py process emits nanobind reference-count
warnings at interpreter shutdown. These also reproduce with only
`from mpi4py import MPI; import maestro`, without creating or executing any
simulator. Numerical tests pass, and explicit native MPI backend shutdown
succeeds after all native states are destroyed. The Python shutdown diagnostics
remain an interoperability limitation; they have not been suppressed.

Final server logs are in `/tmp/mgd-runtime-final-tests.log`,
`/tmp/maestro-runtime-final-tests.log`, `/tmp/maestro-runtime-p2p-tests.log`,
`/tmp/maestro-runtime-python-local-tests.log`,
`/tmp/maestro-runtime-python-mpi-tests.log`, and
`/tmp/maestro-runtime-dependencies.log`. The library suite was configured with
`BUILD_TESTING=ON`, `BUILD_BENCHMARKS=ON`, and `ENABLE_MPI=ON`; Maestro was
configured with `MAESTRO_BUILD_MPI_GPU_TESTS=ON`. Builds and GPU tests ran
sequentially because this server has 7.3 GiB RAM. An earlier parallel compilation
was killed for memory pressure; the final serial build succeeded.

## Reviewer follow-up

The library review identified several fixes, now applied:

- Runtime hardware tests check Open MPI CUDA awareness before state creation and
  skip unavailable prerequisites. They do not hide arbitrary factory failures
  behind a skip. A single GPU is sufficient for the `p2p_bits=0` runtime test.
- `mpi_runtime_shared_device` now permanently tests two ranks with
  `CUDA_VISIBLE_DEVICES=0`. This passed on the T4 server. The proposed prohibition
  on GPU sharing was not adopted: the single-GPU test and earlier four/eight-rank
  tests demonstrate working shared placement. Documentation distinguishes one
  shard/device selection per rank from exclusive physical GPU ownership.
- Licensed tests now exercise runtime admission with valid, invalid, cached, and
  asymmetric keys, including loss/recovery of admission and rank-specific errors.
- Non-MPI symbol checks now also reject `FinalizeMpiBackend` exports.
- Runtime readiness agreement now has operation-specific MPI diagnostics,
  independent of licensing. Documentation explicitly describes the inability to
  coordinate invalid communicator descriptors on only a subset of ranks.
- Installation/header documentation includes the runtime interface, and a new
  dynamic-loading example was checked with a standard C compiler and no MPI
  headers or linking.

The alleged split-communicator leak was not present: `MPI_Comm_rank` returns a
status that is saved, `MPI_Comm_free` runs, and only then are statuses checked
for exceptions. A comment now makes this ordering explicit.

Follow-up verification: the full local MPI-enabled 48-test suite had 26 passes
and 22 environment/hardware skips, including the corrected runtime test. The new
shared-device CTest also skips correctly with the local non-CUDA-aware MPI.
All eight targeted server tests passed (dynamic loading, licensing, runtime API,
MPI simulation and lifecycle), and the shared-device test passed separately.
Logs: `/tmp/mgd-review-local-tests.log`, `/tmp/mgd-review-local-shared-gpu.log`,
and server `/tmp/mgd-review-server-tests.log`,
`/tmp/mgd-review-server-registered-shared.log`, `/tmp/maestro-review-server-tests.log`.

## Additional runtime review

Native and MPI-independent admission/factory entry points now share internal
implementations and each has a single public error-handling boundary. Both enforce
the same MPI lifecycle and initialization-thread requirements; license admission
keeps its lifecycle queries and communicator conversion within the serialized
transaction. Tests verify native/runtime errors before MPI initialization, after
finalization, and from a worker under MPI_THREAD_FUNNELED, plus diagnostic clearing
on success and invalid-context errors without entering GPU construction.

`mpi_runtime_abi` runs without CUDA-aware MPI. `mpi_runtime_abi_no_devices` hides
all GPUs and verifies the expected runtime-info/InitLib rejections while still
exercising output validation, device gathering, Fortran communicators, admission,
and invalid-context creation. The two distributed-state tests alone retain the
CUDA-aware MPI prerequisite. Runtime-info diagnostics distinguish invalid outputs
from unavailable devices.

Shared-GPU wording and installed-header lists were corrected throughout the
public docs. `FinalizeMpiBackend` has one canonical declaration in the runtime
header, included by the native MPI header. Both headers compile together with
redundant-declaration warnings treated as errors, and Maestro's mirrored C data
layouts remain compatible. Existing untracked 2026-09-10 validation logs were
left untouched and were not staged or committed.

Final full-suite results: local 28 passed / 23 expected skips; server 50 passed /
one UCX thread-support skip. Logs: `/tmp/mgd-review2-local-complete.log` and server
`/tmp/mgd-review2-server-tests.log`. Maestro regression results are in server
`/tmp/maestro-review2-server-tests.log`.
