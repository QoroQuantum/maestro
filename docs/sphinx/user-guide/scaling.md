---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_hpc)=
# Scaling: multi-GPU and MPI

Spread one statevector across several GPUs in a node, or across MPI ranks, when
it no longer fits in one GPU's memory.

:::{admonition} Requires GPU access
:class: note
Distributed simulation uses separate GPU plugins that are not part of the pip
package, and runs on Linux only. See {doc}`/getting-started/gpu-access`.
:::

```{code-cell} python
import maestro

available = getattr(maestro, "is_distributed_gpu_available", lambda: False)()
print("distributed GPU plugin available:", available)
```

`is_distributed_gpu_available()` checks for the plugin and devices without
allocating any state. It exists only in Linux builds, hence the `getattr`
guard above; the docs build has no GPU, so it prints `False`.

Two plugins provide the distributed backends:

| `SimulatorType` | Plugin library | Spreads the state across |
|-----------------|----------------|--------------------------|
| `DistributedGpu` | `libmaestro_gpu_distributed.so` | The GPUs of one node |
| `DistributedMpiGpu` | `libmaestro_gpu_distributed_mpi.so` | MPI ranks and their GPUs, on one or more nodes |

Both support statevector simulation only; other methods raise an exception.
For a single local GPU, see {ref}`py_gpu_local`.

Each plugin has two ways of managing the distributed state. The default, `"ex"`,
uses cuStateVec's Ex engine, which queues and fuses gates and can move qubits
between local and global positions to reduce communication. The `"conventional"`
engine applies gates one at a time and exists mainly as a reference. The MPI
plugin supports only `"ex"`.

:::{admonition} Multi-node status
:class: warning
The plugin's authors have validated MPI execution on one machine with two GPUs.
Runs across several physical machines haven't been validated yet.
:::

(py_dist_gpu_local)=
## Several GPUs in one node

```python
# Not executed in the docs build: requires GPU access.
from maestro.circuits import QuantumCircuit

ghz = QuantumCircuit()
ghz.h(0)
for q in range(33):
    ghz.cx(q, q + 1)
ghz.measure_all()

config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.DistributedGpu,
    simulation_type=maestro.SimulationType.Statevector,
    distributed_options={"distributed_devices": "0,1"},  # device ordinals
)
print(ghz.execute(shots=1000, config=config)["counts"])
```

(py_dist_gpu_prereq)=
## Prerequisites

- 64-bit Linux with CUDA Toolkit 12 or newer, a compatible driver, and
  cuStateVec 1.14 or newer. The plugin was validated with CUDA 13.3.
- Put the plugin libraries and their dependencies on the library search path
  (`LD_LIBRARY_PATH`), or set `MAESTRO_DIST_GPU_LIBRARY` and
  `MAESTRO_DIST_MPI_GPU_LIBRARY` to their absolute paths.
- MPI execution needs a CUDA-aware MPI, such as OpenMPI or MPICH built with CUDA
  support.
- MPI uses 2 to 32 ranks, in powers of two, with one shard of the state per
  rank. Ranks can share a physical GPU with `mpi_p2p_bits` set to `"0"`, which
  doesn't add memory capacity.
- Licensed builds read `MAESTRO_LICENSE_KEY`, or use the plugin's cached
  activation. Licences are bound to a machine and user; see
  {ref}`py_gpu_avail`.

(py_dist_gpu_mpi)=
## Across MPI ranks

Run the same script on every rank with an MPI launcher:

```bash
mpirun -n 2 python mpi_sim.py
```

Every rank must make the same simulator calls in the same order with the same
arguments. Results are replicated on every rank, so print from rank 0 only.

```python
# mpi_sim.py. Not executed in the docs build: requires GPU access and MPI.
import mpi4py

mpi4py.rc.thread_level = "serialized"  # before importing MPI, for CUDA-aware transports
from mpi4py import MPI

import maestro
from maestro.circuits import QuantumCircuit

ghz = QuantumCircuit()
ghz.h(0)
for q in range(33):
    ghz.cx(q, q + 1)
ghz.measure_all()

config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.DistributedMpiGpu,
    simulation_type=maestro.SimulationType.Statevector,
)
try:
    result = ghz.execute(shots=1000, config=config)
    if MPI.COMM_WORLD.rank == 0:
        print("counts:", result["counts"])
finally:
    # Shut the backend down after the state is destroyed and before MPI finalizes.
    maestro.finalize_distributed_mpi_gpu()
```

(py_dist_gpu_options)=
## `distributed_options`

Optional overrides, as string key-value pairs in
`SimulatorConfig.distributed_options`:

| Key | Description | Example values |
|-----|-------------|----------------|
| `distributed_devices` | Comma-separated GPU ordinals. For MPI, relative to the owning rank. | `"0,1"`, `"0"` |
| `distributed_global_qubits` | Ordered, comma-separated global logical qubits | `"0"`, `"0,1"` |
| `distributed_backend` | Distribution engine. `"conventional"` is local only. | `"ex"` (default), `"conventional"` |
| `distributed_flags` | Bitmask. 1: several shards on one GPU, for testing (conventional engine only). 2: restore the configured layout after every gate. 4: assume a full-mesh GPU topology (local only). 8: keep the layout pinned and skip the Ex gate queue. 16: force host-staged transfers (conventional engine only). 2 and 8 are mutually exclusive. | `"0"` (default), `"8"` |
| `distributed_max_queued_gates` | Gate queue limit for the Ex engine, 1–65536. Larger values let Ex optimize longer circuit regions but delay error reporting. | `"2048"` (default `"1024"`) |
| `distributed_transfer_workspace_bytes` | Ex MPI transfer workspace, 1 MiB–1 GiB in multiples of 256 bytes. Default 16 MiB. | `"67108864"` |
| `distributed_snapshot_storage` | Where snapshots are stored | `"gpu"` (default), `"host"` |
| `mpi_communicator` | Fortran communicator handle from mpi4py's `Comm.py2f()` | `str(MPI.COMM_WORLD.py2f())` |
| `mpi_p2p_bits` | Number of low rank bits that identify peers on the same host | `"0"` (default) |
| `precision` | Precision override | `"single"` (default), `"double"` |
| `seed` | Explicit 64-bit random seed; without one, ranks generate one collectively | `"42"` |
