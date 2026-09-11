# Distributed GPU statevectors

Maestro supports two dynamically loaded plugins from `maestro-gpu-distributed`:

| C++ simulator type | Python simulator type | Plugin |
| --- | --- | --- |
| `kDistGpuSim` | `SimulatorType.DistributedGpu` | `libmaestro_gpu_distributed.so` |
| `kDistMpiGpuSim` | `SimulatorType.DistributedMpiGpu` | `libmaestro_gpu_distributed_mpi.so` |

Both currently support **statevector only**. Other simulation methods are rejected.
The existing `Gpu` backend remains independently available. CUDA/cuQuantum are
runtime dependencies of the plugins, not build dependencies of Maestro.

Put the plugins and their dependencies on the runtime library search path, or
set `MAESTRO_DIST_GPU_LIBRARY` and `MAESTRO_DIST_MPI_GPU_LIBRARY` to their absolute
paths. Each plugin has a separate process-wide function table. API version 1 and
the entry points in `Simulators/DistributedGpuApi.h` are required. Licensed builds
use `MAESTRO_LICENSE_KEY` or the plugin's cached activation; MPI admission is
collective. Discovery does not validate a license or allocate a state.

## Defaults

No distribution configuration is needed for ordinary execution:

- Local execution uses Ex and the largest power-of-two group of visible CUDA
  devices, starting at ordinal 0 (at most 32, leaving at least one local qubit).
  Set `CUDA_VISIBLE_DEVICES` to restrict the visible group. An explicit
  `gpu_device` without `distributed_devices` selects a single-device group.
- Global qubits are **0, 1, ..., log2(shards)-1**.
- Layout policy is **automatic** (`distributed_flags=0`). The plugin handles
  placement internally; Maestro does not insert explicit storage swaps.
- Precision is single; set `precision="double"` in C++, or
  `SimulatorConfig.precision=True` / `use_double_precision=True` in Python.
- Ex uses its default queue and transfer workspace unless configured.
- MPI uses `MPI_COMM_WORLD`, one shard per rank, and `p2p_bits=0`. It assigns
  devices by rank within each shared-memory host, modulo its visible device
  count. With per-rank `CUDA_VISIBLE_DEVICES`, each rank can use ordinal 0.
  More ranks than local GPUs share GPUs; this adds no physical memory capacity.
- MPI's default seed is 0. An explicit seed must match on all ranks. Python
  stochastic noise generation also uses a rank-consistent default; circuits,
  noise settings and explicit noise seeds must match.

The number of MPI ranks must be a power of two, at most 32. At least one qubit
must remain local. All backends support fewer than 63 total qubits; GPU memory
is the practical limit. Full host-state/probability export still allocates a
full host array. The low-level wrapper also exposes bounded logical and physical
local-range I/O for applications needing distributed data access.

## C++

```cpp
#include "Simulators/Factory.h"
using namespace Simulators;

auto sim = SimulatorsFactory::CreateSimulator(
    SimulatorType::kDistGpuSim, SimulationType::kStatevector);
sim->AllocateQubits(10);
sim->Initialize();
sim->ApplyH(0);
sim->ApplyCX(0, 9);
auto counts = sim->SampleCounts({0, 9}, 1000);
```

Call `Configure(key, value)` before initialization for optional overrides:

| Key | Value |
| --- | --- |
| `distributed_devices` | Comma-separated device ordinals, e.g. `0,1`; MPI entries are relative to their owning rank |
| `distributed_global_qubits` | Ordered comma-separated logical qubits, e.g. `0,1` for four shards |
| `distributed_backend` | `ex` (default), or `conventional` for local reference/testing execution |
| `distributed_flags` | Bitmask: 1 permits local shared-device testing, 2 fixed layout, 4 local fullmesh topology, 8 pinned layout; 2 and 8 are mutually exclusive |
| `distributed_max_queued_gates` | Ex queue limit, 1–65536 |
| `distributed_transfer_workspace_bytes` | Ex workspace, 1 MiB–1 GiB, multiple of 256 bytes |
| `distributed_snapshot_storage` | `gpu` (default) or `host` for ordinary `SaveState()` |
| `mpi_communicator` | Fortran communicator handle, from `MPI_Comm_c2f` / mpi4py `Comm.py2f()` |
| `mpi_p2p_bits` | Number of low rank bits identifying an intra-host peer group; default 0 |
| `precision` | `single` or `double` |
| `seed` | Unsigned 64-bit seed |

Allocation settings cannot change after initialization. `Clear()` releases the
state and permits reconfiguration. Settings survive `Clear()` and cloning.
Full-network recreation and cloning retain the resolved device group. Smaller
local host simulations use a power-of-two prefix of an automatically selected
group when necessary to leave at least one local qubit; they do not change the
full-network placement. Explicit device lists and global-qubit settings remain
constraints and incompatible smaller allocations fail. MPI groups retain their
communicator's shard count. MPI communicator handles
must remain valid for later network recreation or state initialization.

`GetConfiguration("distributed_shard_devices")` returns the actual shard devices.
`distributed_configured_global_qubits` returns the **configured** partition;
`distributed_qubit_layout` returns current physical-bit-to-logical-qubit order.
The latter flushes queued gates and is collective in MPI. `GetGpuDevice()` returns
-1 for a multi-device state and every MPI state; this is not an error.

Generic Eigen matrices are passed with explicit column-major layout and
low-bit-first target ordering. Failed gates, snapshots, observations and
measurements throw with the plugin diagnostic; measurement failure sentinels
are never converted into successful outcomes. `Flush()` synchronizes queued
execution. Destructive saves suspend observations until restoration.

## Python

```python
import maestro

config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.DistributedGpu,
    simulation_type=maestro.SimulationType.Statevector,
)
result = maestro.simple_execute(
    'OPENQASM 2.0; include "qelib1.inc"; '
    'qreg q[4]; creg c[4]; h q[0]; cx q[0],q[3]; measure q -> c;',
    shots=1000, config=config,
)
```

Optional configuration uses the same string keys and values:

```python
config.distributed_options = {
    "distributed_devices": "0,1",
    "distributed_global_qubits": "0",
    "distributed_max_queued_gates": "2048",
}
```

`distributed_options` is also a constructor keyword and survives config
pickling. `is_distributed_gpu_available()` probes local plugin/device presence;
it returns `False` without throwing for missing or incompatible plugins and
unavailable devices. Actual initialization retains detailed error diagnostics
and can still fail, for example due to license or memory.

## Qubit numbering when executing on one host

Distributed `ExecuteOnHost*` and `RepeatedExecuteOnHost` calls retain the host's
entire register, including idle wires. Set the network-only option
`network.Configure("distributed_host_qubit_indexing", "local")` or `"global"`
when the circuit's coordinate system is known. Python accepts this option in
`SimulatorConfig.distributed_options` as well. It does not apply to direct
simulator initialization or change the other backends' host mapping.

The default, `"auto"`, uses global numbering if **all affected qubits** fit the
host's global range; otherwise it accepts local numbering if all fit
`[0, host_qubits)`. Other circuits are rejected. Classification happens before
circuit optimization. It cannot infer intent when both interpretations fit:
for a host owning `[3,8)`, an isolated gate on qubit 4 means local slot 1 in
`"auto"`/`"global"`, and local slot 4 in `"local"`. Use an explicit mode for
sparse local circuits in overlapping ranges.

Returned amplitude vectors and Pauli-string positions use host-local ordering
in every mode. Pauli strings must fit the host's register. Classical bit IDs
are independent of quantum indices and measurement results retain the original
classical IDs. The indexing mode is preserved when cloning a network.

## MPI lifecycle and execution

Both distributed backends are always included in Linux Maestro builds. Maestro
and its Python extension need neither MPI headers nor MPI linking. The MPI plugin
is loaded only when used; missing libraries/dependencies or an older plugin
without `gpusim_mpi_runtime.h` entry points produce a runtime error. The application
and plugin must use compatible MPI installations. All ranks must load a compatible
plugin: Maestro cannot coordinate a loading failure on another rank without MPI.

The application initializes MPI and runs **the same Maestro program on every
rank**. Every simulator call, including construction, observation, cloning,
configuration, snapshots and destruction, must occur in matching order with
matching logical arguments. Ordinary outputs are replicated on every rank.
Only display/output code belongs inside rank-zero branches. Maestro does not
provide a rank-zero command dispatcher or initialize/finalize application MPI.

```python
# Run with: mpirun -n 2 python mpi_example.py
import mpi4py
mpi4py.rc.thread_level = "serialized"
from mpi4py import MPI
import maestro

config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.DistributedMpiGpu,
    simulation_type=maestro.SimulationType.Statevector,
)
try:
    result = maestro.simple_execute(
        'OPENQASM 2.0; include "qelib1.inc"; '
        'qreg q[4]; creg c[4]; h q[0]; cx q[0],q[3]; measure q -> c;',
        shots=1000, config=config,
    )
    if MPI.COMM_WORLD.rank == 0:
        print(result["counts"])
finally:
    # High-level execution has already destroyed its temporary states.
    maestro.finalize_distributed_mpi_gpu()
# mpi4py subsequently finalizes MPI.
```

In C++, destroy all MPI GPU simulators/networks, call
`SimulatorsFactory::FinalizeDistributedMpiGpuBackend()`, then `MPI_Finalize()`.
Backend finalization is terminal: do not create further MPI GPU states afterward.
No MPI cleanup runs implicitly from Maestro's static destructors.

Calls run on the calling thread. With `MPI_THREAD_SINGLE`/`FUNNELED`, use the MPI
initialization thread. mpi4py defaults to requesting `MPI_THREAD_MULTIPLE`;
set `MPI4PY_RC_THREAD_LEVEL=serialized` before import when the CUDA-aware MPI
transport does not support that level (as on the validated two-T4 server).
Serialize calls across states unless MPI provides
`MPI_THREAD_MULTIPLE`. Each native state owns its duplicated communicator.

Distributed backends are explicitly selected. Network execution uses one
simulator and avoids shot-worker clones, including circuits with mid-circuit
measurement. Automatic timing-based backend selection is bypassed for an
explicit distributed simulator, so ranks cannot choose different backends.
Register numbering and idle qubits are preserved so configured global qubits
retain their meaning. Saved states are used for repeated trajectories. Full physical multi-host MPI
behavior also depends on the plugin's CUDA-aware transport setup; consult the
plugin's `ADAPTER_NOTES.md` and `MPI_VALIDATION.md` for its runtime requirements.

## Tests

```sh
ctest --test-dir build -R 'distributed_gpu_(configuration|error_contract)' --output-on-failure
MAESTRO_DIST_GPU_LIBRARY=/path/to/libmaestro_gpu_distributed.so \
    ctest --test-dir build -R 'distributed_gpu_(single|shared|two_devices)' --output-on-failure
# Configure -DMAESTRO_BUILD_MPI_GPU_TESTS=ON for C++ MPI tests only;
# supply MPIEXEC_PREFLAGS at configuration if needed:
MAESTRO_DIST_MPI_GPU_LIBRARY=/path/to/libmaestro_gpu_distributed_mpi.so \
    ctest --test-dir build -R '^distributed_gpu_mpi$' --output-on-failure
python -m pytest tests/python/test_distributed_gpu.py
MPI4PY_RC_THREAD_LEVEL=serialized MAESTRO_TEST_MPI_GPU=1 mpirun -n 2 python -m pytest \
    tests/python/test_distributed_gpu.py -k 'execution and True'
```

Hardware tests return skip code 77 when the plugin or required GPUs are missing.
They compare every supported gate (including asymmetric generic matrices), both
precisions, automatic/fixed/pinned policies, snapshots, clone independence,
state import, sampling order and collapse with CPU references. Shared-device
coverage does not establish physical multi-GPU or cross-host coverage.
