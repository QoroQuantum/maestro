---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Configuration

`SimulatorConfig` bundles every simulator option into one reusable value. Every
execution and estimation function takes one as `config`; without it, a default
`SimulatorConfig()` runs QCSim with a statevector.

```{code-cell} python
import maestro

config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
config.seed = 7  # fields can also be set after construction
print(config.simulation_type, config.max_bond_dimension, config.seed)
```

The constructor takes every option by keyword only.
{ref}`py_key_options` lists the handful most people set.

(simulatorconfig-options)=
## All SimulatorConfig options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fixed_backend` | `bool` | `False` | Pin the requested backend and method; disable automatic replacement, including reference-frame sampling. |
| `optimization_candidates` | `list[tuple[SimulatorType, SimulationType]]` | `[]` | With `fixed_backend=False`, the backend/method pairs automatic selection may use instead of the requested pair. Without a cost model the first listed candidate that can hold the register execution maps (the qubits the circuit uses) runs; Clifford circuits move to a stabilizer only as a listed pair, and a listed stabilizer never runs a non-Clifford circuit; a list whose candidates cannot be created raises. When no candidate can run the circuit, the first one's error is raised (`RuntimeError` for a stabilizer asked to run a non-Clifford circuit, as when the stabilizer itself rejects the gate). Exact noise channels apply only when every candidate supports them. Distributed GPU backends cannot be candidates. |
| `simulator_type` | `SimulatorType` | `QCSim` | Simulation backend. `Gpu`, `DistributedGpu` and `DistributedMpiGpu` require GPU access; see {doc}`/getting-started/gpu-access`. |
| `simulation_type` | `SimulationType` | `Statevector` | Simulation method |
| `max_bond_dimension` | `int` or `None` | `None` | Bond dimension limit for MPS, MPO and GPU tensor networks. For GPU MPS/MPO, `None` uses 128. Other backends keep their own defaults. |
| `singular_value_threshold` | `float` or `None` | `None` | Truncation threshold for MPS, MPO and GPU tensor networks, read according to `truncation_mode` |
| `truncation_mode` | `str` or `None` | `None` (`"discarded_weight"`) | `"discarded_weight"` (Qiskit Aer and ITensor convention: discard the smallest singular values until their cumulative squared weight reaches the threshold) or `"relative_max"` (discard singular values below `threshold * sigma_max`). The first compares squared weights, so the same threshold truncates much harder. Only QCSim and GPU support `"relative_max"`; Aer raises `ValueError`. |
| `precision` | `str` or `None` | `None` | `"single"` or `"double"` for Qiskit Aer and the GPU simulators; `None` keeps each backend's default. QCSim always uses double precision. |
| `seed` | `int` or `None` | `None` | Seeds all backend and Maestro sampling random streams. Parallel jobs derive deterministic, distinct child streams. |
| `gpu_device` | `int` or `None` | `None` | CUDA device for single-GPU execution (GPU access only) |
| `distributed_options` | `dict[str, str]` | `{}` | Settings for `DistributedGpu` and `DistributedMpiGpu`, such as `distributed_devices`, `distributed_global_qubits`, `mpi_communicator` and `mpi_p2p_bits` (GPU access only). See {doc}`/user-guide/scaling`. |
| `gate_fusion` | `bool` or `None` | `None` | Fuse compatible gates. `None` uses each backend's default: on, except CPU statevectors below 11 qubits and CPU density matrices below 5. Truncated MPS/MPO results can change. |
| `disable_optimized_swapping` | `bool` | `False` | Turn off swap-cost and initial qubit-map optimization |
| `lookahead_depth` | `int` | `-1` | Swap optimization lookahead depth; `-1` uses the default |
| `mps_routing_strategy` | `str` or `None` | `None` | CPU MPS/MPO routing: `"legacy"` (default) or opt-in `"combined"`. An explicit lookahead overrides it. |
| `mps_routing_virtual_bonds` | `bool` or `None` | `None` | Opt into bounded virtual-bond routing on QCSim and GPU MPS and MPO; off by default |
| `mps_routing_truncation_weight` | `float` or `None` | `None` | Non-negative overflow penalty relative to capped runtime cost. Default 1; 0 charges runtime alone. |
| `mps_sampling` | `str` | `"probabilities"` | MPS shot sampling: `"probabilities"` samples without collapsing the state, `"apply_measure"` measures and restores it per shot |
| `mps_svd_solver`, `mpo_svd_solver`, `tensor_network_svd_solver` | `str` or `None` | `None` | GPU SVD solver: `"gesvd"`, `"gesvdj"`, `"gesvdp"` or `"gesvdr"`; see {ref}`py_mps_gpu_svd` (GPU access only) |
| `mpo_kraus_completeness_check` | `str` or `None` | `None` | MPO handling of incomplete Kraus sets: `"ignore"`, `"warn"` or `"strict"` |
| `mpo_restore_trace_after_truncation`, `mpo_hermitize_after_truncation` | `bool` | `False` | Renormalize the MPO (CPU or GPU) after each truncation |
| `pp_coefficient_threshold` | `float` or `None` | `None` | Pauli propagation: drop strings whose coefficient magnitude is at most this |
| `pp_max_pauli_weight` | `int` or `None` | `None` | Pauli propagation: drop strings acting on more qubits than this |
| `pp_gates_between_trims`, `pp_gates_between_deduplications` | `int` or `None` | `None` | Pauli propagation: how often, in primitive operations, to apply the two thresholds; deduplication also merges repeated strings. Unset, PauliPropagator deduplicates every 10 operations. |
| `path_integral_threshold` | `float` or `None` | `None` | PathIntegral trim threshold |
| `enable_causal_cone_reduction` | `bool` | `True` | Drop gates and idle qubits outside the backward lightcone of the requested observables |
| `enable_observable_clustering` | `bool` | `True` | Simulate groups of observables with disjoint lightcones separately, each on its own register |
| `max_cluster_qubits` | `int` or `None` | `None` | Also split overlapping lightcones into groups of at most this many qubits; see {ref}`py_circuits_cluster_budget` |
| `min_cluster_qubits` | `int` or `None` | `None` | Batch disjoint small lightcones together up to this width, within the maximum |
| `auto_reduce` | `bool` | `True` | Detect invariant check subspaces and simulate the reduced register on the configured backend |
| `observable_error_tolerance` | `float` or `None` | `None` | Opt-in absolute error budget for bounded backward evaluation of exact noisy channels |
| `enable_sampling_reduction` | `bool` | `False` | Sample disjoint measurement lightcones independently and exactly; changes seeded random streams |
| `allow_reset_marginalization` | `bool` | `False` | Allow ensemble-equivalent reset marginalization; individual trajectories can change |
| `trajectory_policy` | `str` or `None` | `None` | QCSim dynamic circuits: `"shot_v1"` gives streams independent of worker count; `"block_v1"` keeps earlier seeded counts. Statevector, density-matrix and path-integral runs default to `"shot_v1"`. Stabilizer, extended-stabilizer, MPS and MPO runs use a policy only when it is set; `None` keeps their earlier, worker-dependent seeded sequences. Terminal sampling is unaffected. |
| `trajectory_max_memory_mb` | `int` or `None` | `None` | Soft budget in MiB for concurrent QCSim work: dense trajectory shots (and, under an explicit `trajectory_policy`, stabilizer and bounded MPS/MPO shots), and the parallel noise realizations (of executions, Monte Carlo estimates and fidelities) and checkpoint suffixes of statevector, density-matrix, stabilizer and bounded MPS/MPO runs. `None` uses 1024. At least one worker always runs. |
| `counts_policy` | `str` | `"auto"` | Eligible terminal counts use adaptive multinomial sampling with or without a seed. Unseeded requests may also use reference frames or direct trajectories. Repeating the same seeded request on the same build reproduces its histogram. `"legacy"` uses the earlier executor, `"frame_v1"` selects eligible frame sampling, and `"multinomial_v1"` selects adaptive terminal count sampling. |
| `max_simulators` | `int` or `None` | `None` | Maximum concurrent simulators, from 1 to 1024. `None` keeps the default; memory budgets may lower it. |

```{code-cell} python
:tags: [remove-cell]

# Fail the docs build if SimulatorConfig gains an option this table doesn't list.
import pathlib
import re

table = pathlib.Path("configuration.md").read_text().split("(simulatorconfig-options)=")[1]
listed = set(re.findall(r"`([a-z0-9_]+)`", table.split("```")[0]))
fields = {n for n in dir(maestro.SimulatorConfig)
          if not n.startswith("_") and isinstance(getattr(maestro.SimulatorConfig, n), property)}
assert fields <= listed, sorted(fields - listed)
```

## Reference

```{eval-rst}
.. currentmodule:: maestro

.. autoclass:: SimulatorConfig

.. autoclass:: SimulatorType

.. autoclass:: SimulationType
```
