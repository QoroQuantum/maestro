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
| `singular_value_threshold` | `float` or `None` | `None` | Truncation threshold for MPS, MPO and GPU tensor networks, read according to `truncation_mode`. `None` uses the backend default: `1e-15` for QCSim MPS/MPO. An explicit value, including `0`, replaces that default. |
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
| `pp_workers` | `int` or `None` | `None` | Pauli propagation workers. GPU PauliPropagator estimates parallelise independent observables across 1 to 64 workers, each on its own CUDA stream; CPU PauliPropagator uses 0 to 1024 threads, where 0 picks a count from the hardware. The best count depends on the workload; more workers can add overhead on small batches. `None` uses the backend default, one worker on the GPU. |
| `pp_max_fused_gates` | `int` or `None` | `None` | GPU Pauli propagation: most gates fused into one block, from 1 to 256; 1 disables fusion. `None` uses 64. (GPU access only) |
| `pp_memory_limit_bytes` | `int` or `None` | `None` | GPU Pauli propagation: positive device memory budget in bytes. `None` uses 90% of available memory. (GPU access only) |
| `pp_sampling_cache_nodes` | `int` or `None` | `None` | Pauli propagation: most nodes in the shot-sampling cache; 0 disables caching. `None` uses the backend default (8192 on the GPU). |
| `path_integral_threshold` | `float` or `None` | `None` | PathIntegral trim threshold |
| `extended_stabilizer_amplitude_tolerance` | `float` or `None` | `None` | QCSim ExtendedStabilizer: after each non-Clifford gate and measurement, drop components whose amplitude is at most this fraction of the state's norm, in [0, 1). With `extended_stabilizer_max_components` also `None` the simulation is exact. Pruned runs use scalar trajectories, not batched extended frames. |
| `extended_stabilizer_max_components` | `int` or `None` | `None` | QCSim ExtendedStabilizer: keep at most this many of the largest components after each non-Clifford gate and measurement. Must be positive. |
| `extended_stabilizer_approximation_error` | `float` or `None` | `None` | Qiskit Aer ExtendedStabilizer: the error bound of Aer's stabilizer-rank decomposition, in (0, 1]. Aer samples about extent / error² terms, so a smaller error costs more time and memory. `None` keeps Aer's default, 0.05. Aer has no exact mode on a non-Clifford circuit. |
| `extended_stabilizer_sampling_method` | `str` or `None` | `None` | Qiskit Aer ExtendedStabilizer: `'resampled_metropolis'` (Aer's default: a fresh Metropolis walk for each shot), `'metropolis'` (one walk for all the shots) or `'norm_estimation'` (qubit by qubit, from estimated norms). |
| `extended_stabilizer_metropolis_mixing_time` | `int` or `None` | `None` | Qiskit Aer ExtendedStabilizer: Metropolis steps taken before a sample is read. Positive; `None` keeps Aer's default, 5000. |
| `extended_stabilizer_norm_estimation_samples` | `int` or `None` | `None` | Qiskit Aer ExtendedStabilizer: random stabilizer states used in one norm estimate, which expectation values, probabilities and `'norm_estimation'` sampling rest on. Positive; `None` keeps Aer's default, 100. |
| `extended_stabilizer_norm_estimation_repetitions` | `int` or `None` | `None` | Qiskit Aer ExtendedStabilizer: norm estimates taken, of which Aer keeps the median. Positive; `None` keeps Aer's default, 3. |
| `enable_causal_cone_reduction` | `bool` | `True` | Drop gates and idle qubits outside the backward lightcone of the requested observables |
| `enable_observable_clustering` | `bool` | `True` | Simulate groups of observables with disjoint lightcones separately, each on its own register |
| `max_cluster_qubits` | `int` or `None` | `None` | Also split overlapping lightcones into groups of at most this many qubits; see {ref}`py_circuits_cluster_budget` |
| `min_cluster_qubits` | `int` or `None` | `None` | Batch disjoint small lightcones together up to this width, within the maximum |
| `auto_reduce` | `bool` | `True` | Detect invariant check subspaces and simulate the reduced register on the configured backend |
| `observable_error_tolerance` | `float` or `None` | `None` | Opt-in absolute error budget for bounded backward evaluation of exact noisy channels |
| `enable_sampling_reduction` | `bool` | `False` | Sample disjoint measurement lightcones independently and exactly; repeated seeded requests remain reproducible |
| `allow_reset_marginalization` | `bool` | `False` | Allow ensemble-equivalent reset marginalization; individual trajectories can change |
| `trajectory_policy` | `str` or `None` | `None` | QCSim dynamic circuits: `"shot_v1"` schedules individual shots with streams independent of worker count; `"block_v1"` schedules fixed logical blocks. Supported statevector, density-matrix, path-integral, stabilizer, extended-stabilizer, MPS and MPO trajectories default to `"shot_v1"`. Terminal sampling is unaffected. |
| `trajectory_max_memory_mb` | `int` or `None` | `None` | Soft budget in MiB for concurrent trajectory work (QCSim, and the CPU Qiskit Aer statevector and MPS): dense, stabilizer and bounded MPS/MPO trajectory shots, parallel noise realizations (executions, Monte Carlo estimates and fidelities), and checkpoint suffixes. `None` uses 1024, or half the machine's memory on the Aer statevector and MPS, which Aer itself fills with parallel shots. An Aer MPS without a bond limit has no known size: with a budget set its shots run on one worker, without one they are not limited. At least one worker always runs. |
| `counts_policy` | `str` | `"auto"` | Eligible adaptive terminal counts, reference frames and direct trajectories are available with or without a seed. The same request, seed and configuration reproduce its histogram within one Maestro version; sequences may change between versions. `"legacy"` selects the ordinary executor, `"frame_v1"` selects eligible frame sampling, and `"multinomial_v1"` selects adaptive terminal count sampling. |
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

## Bond limits on the Qiskit Aer matrix product state

A `max_bond_dimension` that the circuit reaches (or a truncation threshold
that discards weight) makes a matrix product state approximate on every
backend. On the Qiskit Aer MPS it has two further effects. Both come from Aer
itself: Aer run from Qiskit (qiskit-aer 0.17.2) behaves the same way, and
gives the same numbers for the small case below.

**Reads of the same state can disagree.** After it truncates, Aer does not
restore the canonical form of the chain, which its expectation values rely
on. An expectation value and the same quantity computed from the state's
probabilities then differ, by about as much as the truncation changed the
result in the first place. For 60 random circuits of 60 one- and two-qubit
gates on 10 qubits, whose largest bond is 16:

| Bond limit | Circuits truncated | ⟨Z⟩ error against the exact state (mean, max) | ⟨Z⟩ against the state's own probabilities (mean, max) |
|---|---|---|---|
| 2 | 59 | 0.51, 1.0 | 0.17, 1.0 |
| 4 | 46 | 0.12, 0.72 | 0.058, 0.69 |
| 8 | 22 | 0.020, 0.23 | 0.016, 0.23 |
| 16 or more, or none | 0 | 2e-11, 1.5e-10 | 1.3e-11, 1.5e-10 |

A small case: `H(0), H(1), CX(1,2), CX(2,3), CCX(0,1,2)` under a limit of 2
gives ⟨Z₃⟩ = 0 as an expectation value and +1/3 from the probabilities; under
a limit of 4 both give 0. So with a limit that bites, `estimate` and the
counts or probabilities of the same circuit may describe different states.

**An operation on qubits that are apart changes other qubits.** To apply a
two-qubit gate or a two-qubit channel, the chain first moves the two qubits
next to each other, and under the limit that move truncates: qubits the
operation does not act on, and the ones entangled with them, change too.
Maestro sends a channel only to the targets it acts on, so a channel given on
two targets that leaves one alone does not move that one.

Neither applies as long as nothing is discarded: no bond limit, or one the
circuit does not reach (no bond of an n-qubit register exceeds 2^(n/2)), and a
truncation threshold that drops no weight that matters. The Aer MPS then
agrees with the statevector to about 1e-10 in the table above. Both settings
count: the small case above under a bond limit of 64 but a truncation
threshold of 0.25 again gives 0 as an expectation value and +1/3 from the
probabilities. QCSim's MPS records when a truncation has left the canonical
form approximate and then computes what a read needs from the tensors, so its
expectation values agree with its probabilities under any limit or threshold;
they are still approximations of the exact state.

## Reference

```{eval-rst}
.. currentmodule:: maestro

.. autoclass:: SimulatorConfig

.. autoclass:: SimulatorType

.. autoclass:: SimulationType
```
