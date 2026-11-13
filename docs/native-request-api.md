# Native request API, schema 2

Maestro exposes batch computation through a C ABI implemented entirely in C++.
Python bindings remain optional. This API does not initialize or invoke Python.

## Entry points and ownership

Include `maestrolib/Interface.h` and link `libmaestro`:

```c
char *MaestroGetCapabilitiesJson(void);
char *MaestroValidateRequestJson(const char *request);
char *MaestroRunRequestJson(const char *request);
char *MaestroFinalizeDistributedMpiGpuJson(void);
void FreeResult(char *result);

/* Packed shot rows and compiled reference frames; see "Shot rows". */
char *MaestroRunRequestSamples(const char *request, unsigned long long *rows, size_t capacity);
void *MaestroCreateReferenceFrameSampler(const char *request, char **error);
int MaestroReferenceFrameSamplerShape(void *sampler, size_t *qubits, size_t *clbits);
int MaestroReferenceFrameSampleBits(void *sampler, size_t shots, unsigned long long seed,
                                    unsigned long long *rows, size_t capacity);
int MaestroReferenceFrameSampleCountsBits(void *sampler, size_t shots, unsigned long long seed,
                                          unsigned long long *rows, unsigned long long *counts,
                                          size_t capacity, size_t *written);
void MaestroDestroyReferenceFrameSampler(void *sampler);
```

Each call returns an owned UTF-8 JSON string; release it with `FreeResult`, including
error responses. Null means that even the response buffer could not be allocated.
Inputs are NUL-terminated strings. No exception crosses these C entry points.
Success has `ok:true` and `schema_version:2`; failure has `ok:false` and
`error:{code,message}`. Validation checks the request without creating a simulator;
it does not prove plugin, license, device, memory, or MPI availability.

Use a fresh worker process per task. Existing native singletons/configuration are
not an isolated multi-tenant execution environment. The supplied local server
provides that process boundary. MPI must be initialized collectively by the native
MPI worker before invoking distributed MPI functionality.

`maestro-request` reads one JSON document from stdin and writes the result.
`--validate` and `--capabilities` select those calls. `--framed-result` prefixes the
single result with `MAESTRO_RESULT_V2 `, distinguishing it from library logs.

## Request and result conventions

```json
{
  "schema_version": 2,
  "operation": "estimate",
  "circuit": {
    "format": "openqasm",
    "source": "OPENQASM 2.0; qreg q[1]; h q[0];",
    "num_qubits": 1,
    "num_clbits": 1
  },
  "simulator": {"backend": "qcsim", "method": "density_matrix"},
  "observables": ["X", "Z"],
  "execution": {"seed": 123},
  "noise": {
    "evaluation": "exact",
    "channels": [{"kind": "t1", "targets": [0], "gamma": 0.36}]
  }
}
```

This returns expectations `[0.8,0.36]`. Backend/method defaults are
`qcsim/statevector`; selection defaults to `fixed`. Unavailable fixed selections
fail; they do not return a substitute backend's calculation. `selection:"automatic"`
and optional `candidates:[{backend,method},...]` retain optimizer selection for
ideal execute/estimate. Results identify the backend/method actually used, which
is always one of the listed candidates: without a cost model the first listed
candidate that can hold the register runs (a 40-qubit density matrix yields to a
listed MPO), and Clifford circuits move to a stabilizer only as a listed
backend/method pair. A listed stabilizer never runs a non-Clifford circuit. When
no listed candidate can be created, the request fails instead of running an
unlisted backend; this includes a one-entry list. When candidates can be created
but none can run the circuit, the request fails with the first one's error: for
a stabilizer asked to run a non-Clifford circuit that is `native_failure`, as
when the stabilizer itself rejects the gate; for a register too large for a
backend it is that backend's own error.

Fixed selection preserves multi-shot reuse: eligible circuits evolve once per
execution job and sample repeatedly. Circuits with intermediate measurements,
resets, or classical conditions reuse only their independent prefix and execute
the dependent operations for each shot. Selecting a fixed backend does not force
the whole circuit to run again for every shot.

`num_clbits` defaults to `num_qubits` and can differ. Counts list classical bit 0
first; Pauli observables and `target_state` list qubit 0 first. Integer basis
indices use qubit 0 as the least significant bit. Complex numbers are `[real,imag]`.
QASM parameters can be supplied as `circuit.parameters:{name:number}`. The native
QASM parser supports its existing OpenQASM 2/3 subset; parser errors are returned.

The alternative `format:"instructions"` takes an array as `source`. Gate and
measurement entries use the existing CUNQA names, `qubits`, `params`, `clbits`,
and an optional single `conditional_reg`. Native additions are:

```json
[
  {"name":"x", "qubits":[0], "noiseless":true},
  {"name":"rzz", "qubits":[0,1], "params":[0.7]},
  {"name":"delay", "qubits":[0], "duration":0.000001},
  {"name":"kraus", "qubits":[0], "operators":[[0,1,1,0]]},
  {"name":"pauli_channel", "qubits":[0], "probabilities":[0.97,0.01,0.01,0.01]},
  {"name":"channel", "qubits":[0,1], "factors":[
    {"kind":"depolarizing", "probability":0.01},
    {"kind":"thermal_relaxation", "qubits":[1], "duration":3e-7, "t1":5e-5, "t2":7e-5}]},
  {"name":"reset", "qubits":[0]},
  {"name":"measure", "qubits":[0], "clbits":[0], "readout_error":0.01}
]
```

Kraus matrices are flat row-major arrays with real or complex entries, on one or
two targets. A mixed-state backend and the QCSim `pauli_propagator` apply a
channel instruction (`kraus`, `pauli_channel`, `channel`) exactly. `statevector`
and `matrix_product_state` sample it from their state, one branch each time the
circuit runs: in `execute`, `estimate`, `noisy_fidelity` and `checkpoint_batch`
suffixes, with or without a noise block. A query of one state (`statevector`,
`amplitudes`, `probabilities`, `state_probability`) runs the circuit once, so
there it returns the state of one branch, as Python's `get_statevector` and
`get_probabilities` do and as a mid-circuit reset does on any pure-state method;
`execution_metadata.stochastic_execution` is then `true`, and the seed selects
the branch. Under a `candidates` list and without a noise block, one candidate
that holds channels is enough: the network tries its candidates in turn and
runs the circuit on one that can, as Python's `execute` does. A noise block has
to know how its realizations are built before a backend is selected, so there
every candidate must qualify. The other pure-state methods need a noise block with trajectory
evaluation, whose realizations draw the instruction into gates and resets, and
accept only channels whose branches need no state: Pauli channels, one-qubit
mixtures of unitaries, and relaxation with T2 <= T1. Identity instructions
remain present for noise injection. Gate parameter counts and U1/U2/U3
conventions are checked.

`channel` holds a structured channel on one or two qubits: its `factors`, applied
in order, as one instruction that keeps them apart. It is the request form of
Python's `maestro.QuantumChannel` constructors composed with `then()`, and of
`Simulators::QuantumChannel` in C++. A factor names its `kind` and parameters:

| `kind` | Parameters |
|---|---|
| `pauli` | `probabilities`: 4 or 16 entries, as for `pauli_channel` |
| `unitary_mixture` | `probabilities`, `unitaries` (flat row-major matrices) |
| `kraus` | `operators` (flat row-major matrices) |
| `depolarizing` | `probability`; one or two qubits |
| `amplitude_damping`, `phase_damping` | `gamma` |
| `generalized_amplitude_damping` | `gamma`, `excited_population` |
| `thermal_relaxation` | `duration`, `t1`, `t2` (numbers or `"infinity"`), optional `excited_population` |

A factor acts on all the instruction's qubits, in their order, unless it lists
`qubits`: positions among the instruction's qubits (`[1]` is the second one,
`[1, 0]` exchanges the two). A one-qubit factor of a two-qubit instruction must
list its position. Keeping the factors apart matters on pure-state methods: a
trajectory draws a Pauli or unitary-mixture factor without reading its state,
and the methods that hold no channels can draw relaxation with T2 <= T1 into
resets, which they cannot do for the same map given as a flat `kraus` list.

`rxx` and `rzz` take one angle and two qubits. They expand exactly as the QASM
parser expands qelib1.inc's gates: `cx; rz; cx`, between Hadamards on both
qubits for `rxx`. A `conditional_reg` applies to every expanded gate.

`noiseless: true` on a gate (including an expanded `rxx`/`rzz`) exempts it from
injected gate noise, like Python's `h/s/sdg(..., noiseless=True)`. Use it for
basis changes that belong to a non-Z measurement or reset. It is rejected on
measurements and non-gate instructions.

`readout_error` on a measurement flips each recorded result: a number gives a
symmetric probability, and `{"p_meas1_prep0": p, "p_meas0_prep1": q}` gives the
two rates of the `readout` noise channel. A noise model's readout error composes
with the measurement's own rates. Like model readout, it applies to measurement
counts only: `execute` and `checkpoint_batch` accept it, and other operations
reject it instead of silently dropping it. That includes the adjoint operations,
the nested `other_circuit` and `step_circuit`, and the `checkpoint_batch` prefix,
whose terminal measurements are otherwise removed. A checkpoint applies readout
on its suffix measurements only.

`pauli_channel` applies a Pauli channel to one or two qubits. `probabilities`
has `4**len(qubits)` entries summing to one; base-4 digit `k` of an index selects
I, X, Y or Z on `qubits[k]`, so one qubit is `[pI, pX, pY, pZ]` and two are
`II, XI, YI, ZI, IX, XX, ... ZZ`. Density-matrix and MPO methods apply it exactly,
and so does the QCSim `pauli_propagator`, which scales the Pauli terms of the
observable it propagates. `statevector` and `matrix_product_state` draw one
Pauli each time the circuit runs. The other pure-state methods sample one Pauli
per noise realization: the request needs a `noise` block with trajectory
evaluation, whose `channels` may then be omitted. Set `noise.realizations` to
the shot count for independent draws per shot, as with Python's
`noisy_execute(..., noise_realizations=shots)`. A general `kraus` instruction,
or a `channel` factor that needs the state, cannot be drawn that way: it needs
exact evaluation or one of the two methods that sample from their state.
Circuit channel instructions combine with a `noise` model: exact evaluation
keeps them in place and adds the model's channels around them, and so does
trajectory evaluation on a method that samples from its state.

Operation-specific fields are accepted only by the operations listed below.
For example, `other_circuit` on `execute` is an input error, even if that nested
circuit would otherwise be valid. `keep_qubits` requires `partial_trace`.
`execution.shots` is accepted only for `execute` and `checkpoint_batch`; estimators
and state/query operations reject it. `execution.seed` remains available for other
operations. The `validate` operation itself has no sampling semantics and rejects
shots. To validate an execute/checkpoint request, retain that operation and call
`MaestroValidateRequestJson` or `maestro-request --validate`.
`launch` is not a native field and is rejected, including inside batches.
Maestro local server accepts it as outer-task routing metadata, validates the
administrator-owned profile, then removes it before native validation/execution.
When invoking `maestro-mpi-worker` directly, select ranks through your launcher
and pass only the computational document.
Checkpoint `suffixes` contain circuit descriptions, not independent requests;
they cannot override the simulator or specify their own launch profiles.

## Operations

| Operation | Additional fields | Output/semantics |
|---|---|---|
| `execute` | `execution.shots` (default 1024), optional `execution.return_samples` | Measured counts, or `samples` (one bitstring per shot; see [Shot rows](#shot-rows)); total shots is not multiplied by realizations |
| `estimate` | `observables` | Exact observable evaluation per realization, mean and realization standard errors |
| `statevector`, `amplitudes` | optional `basis_states` | Pure-state amplitudes; these two names have the same output shape |
| `probabilities` | optional `basis_states` | Selected or full basis probabilities |
| `state_probability` | `target_state` | Probability of a q0-first bitstring |
| `diagnostics` | `diagnostics`, `maintenance`, `keep_qubits` | Mixed-state diagnostics/reduced density matrix; MPS/MPO maintenance |
| `inner_product` | `other_circuit` | Complex overlap of two unitary preparations |
| `density_matrix_overlap` | `other_circuit` | `Tr(rho sigma)` of two mixed-state preparations, as `overlap` |
| `fidelity_with_statevector` | `target_statevector` | `<psi|rho|psi> / Tr(rho)` against a normalized pure target, as `fidelity` |
| `mirror_fidelity` | optional `observables` | Fidelity of the preparation followed by its inverse; with observables, their expectations on that mirror state |
| `noisy_fidelity` | `noise` (any mode but `analytical`) | Ideal-versus-noisy fidelity `<psi|rho|psi>` across realizations; exact on DM/MPO, and on the QCSim `pauli_propagator` up to 24 qubits |
| `incremental_evolve` | `step_circuit`, ordered `steps`, `observables` | Prepare once, advance to each step, observe without reset |
| `checkpoint_batch` | `suffixes`, `execution.shots` | Save the prefix, restore for each suffix/shot; noise applies to suffixes, and with `noise.prefix` to the prefix too |
| `batch` | `requests` | Independent native requests in one worker; validate all members first |
| `validate` | circuit/configuration | Validation response only |
| `adjoint_expectation_values` | `observables`, optional `parameter_rows` | Readout values for each parameter row |
| `adjoint_jacobian` | `observables`, optional `parameter_rows` | Values and their adjoint Jacobian |
| `adjoint_vjp` | `observables`, `cotangents`, optional `parameter_rows` | Values and the vector–Jacobian product |

With `observables`, `mirror_fidelity` returns `expectation_values`, the value of
`<0|M† O M|0>` for each Pauli string `O` and the executed mirror `M`, instead of
`fidelity`. Z-type strings give 1 under exact evolution and less where the
backend truncated, so this measures truncation error per observable. The gate
sequence runs on the configured backend as given, as in Python. Because it only
takes expectations of the executed mirror state, observable mode also accepts
`density_matrix` and `matrix_product_operator`; plain mirror fidelity, a
pure-state overlap, does not.

State queries remove terminal measurements; mid-circuit measurements are rejected
for these queries. Overlap/fidelity require unitary circuits. Incremental evolution
accepts explicit Kraus instructions in the step circuit for exact noisy evolution;
a top-level stochastic noise model is rejected for this operation. State queries
with top-level noise require a single exact realization. No statevector or
pure-state overlap is advertised for density matrices/MPOs; their counterparts
are the two mixed-state queries below. Individual methods may have narrower
numerical-query support and return a native error.

`density_matrix_overlap` and `fidelity_with_statevector` read the prepared
density matrix or MPO directly, as the Python `density_matrix_overlap` and
`fidelity_with_statevector` simulator methods do. They require fixed selection of
the `qcsim` or `gpu` backend with `density_matrix` or `matrix_product_operator`.
Their circuits may contain resets and channel instructions; terminal measurements
are removed and mid-circuit measurements rejected. A top-level noise model must
use exact evaluation with one realization and applies to both overlap circuits.

`density_matrix_overlap` prepares `circuit` and `other_circuit` (identical
register sizes) on separate registers of the same backend and returns
`overlap = [real, imaginary]`, the unnormalized Hilbert-Schmidt product
`Tr(rho sigma)`; it is 1 for identical pure states. `fidelity_with_statevector`
returns `fidelity`, `<psi|rho|psi> / Tr(rho)` clamped to `[0, 1]`.
`target_statevector` lists `2^num_qubits` amplitudes as `[real, imaginary]` pairs
in `qubit_0_least_significant` order, for at most 20 qubits (and within
`max_request_bytes`). It must be normalized to within `1e-6`; Maestro does not
rescale it.

## Adjoint differentiation

The three `adjoint_*` operations expose Python's `AdjointProgram`: exact,
batched values and reverse-mode derivatives of a unitary circuit on QCSim's CPU
statevector (`qcsim` or `composite_qcsim`, method `statevector`). Noise is
rejected; terminal measurements are ignored as for `estimate`, and any other
non-unitary operation is an input error. Registers are limited to 63 qubits.

The differentiable parameters are the gate angles in circuit order: one for
`p`, `rx`, `ry`, `rz`, `cp`, `crx`, `cry` and `crz`; `theta, phi, lambda` for
`u` (its global phase stays fixed); four for `cu`. QASM `parameters` bindings
and the `u1/u2/u3`, `rxx`, `rzz` expansions contribute their resulting angles.
Results report `num_parameters` and the circuit's own values as
`circuit_parameters`, in that order.

Each `observables` entry is one readout:

- a Pauli string, one character per qubit (q0 first), giving one value;
- `{"terms": [[coefficient, pauli], ...]}`, a real weighted sum, giving one value;
- `{"probabilities": [q, ...]}`, 1–24 distinct qubits, giving `2**k` values
  with the first listed qubit as the least significant bit of the outcome index.

Values are placed side by side in readout order; `num_values` is their total.
`parameter_rows` is an array of rows of `num_parameters` angles, each replacing
all circuit parameters; omitted means one row with the circuit's own values.

```json
{
  "schema_version": 2,
  "operation": "adjoint_jacobian",
  "circuit": {"num_qubits": 2,
              "source": "OPENQASM 2.0; qreg q[2]; ry(0.4) q[0]; cx q[0],q[1]; rx(0.9) q[1];"},
  "observables": ["ZI", {"terms": [[0.5, "ZZ"], [2, "XI"]]}, {"probabilities": [1]}],
  "parameter_rows": [[0.4, 0.9], [1.2, -0.3]]
}
```

| Operation | Result fields |
|---|---|
| `adjoint_expectation_values` | `expectation_values[row][value]` |
| `adjoint_jacobian` | also `jacobian[row][value][parameter]` |
| `adjoint_vjp` | also `vjp[row][parameter]`, from `cotangents[row][value]` |

`adjoint_jacobian` runs one backward sweep per value; `adjoint_vjp` runs one per
row for all readouts, so it is the cheaper choice for a scalar loss.
`simulator.options.enable_causal_cone_reduction` (default true) drops gates
outside the readouts' joint cone; their parameters get an exactly zero
derivative. The total number of returned numbers is bounded by
`max_output_elements`. Validation reports `num_parameters` and `num_values`
without computing. A request whose state memory would exceed the machine's
physical or cgroup memory fails with `insufficient_resources` and the reason.
Results are deterministic and carry no `seed`.

Diagnostics are `trace`, `purity` (defaults), `trace_of_square`,
`hermiticity_residual`, `is_hermitian`, and `partial_trace`, for density
matrices and MPOs. The latter requires unique `keep_qubits` and returns
`{dimension,row_major}`. Maintenance accepts `restore_trace` and `hermitize`
(density matrix and MPO) and `trim`/`recanonicalize` (MPS and MPO). An MPS
reports no diagnostics by default; its `operator_expectation` query is described
below.

## Batched Pauli expectations

`estimate` evaluates all observables together on each prepared state, including
each noise realization. `incremental_evolve` does the same at each requested
step. CPU QCSim and current GPU MPS/MPO plugins batch the contractions; MPS
shares observable prefixes and MPO shares contraction work and trace normalization.
Older GPU plugins and other backends preserve their scalar
behavior. Batches preserve input order and duplicates and do not collapse the
state. They never combine different evolution steps or noise realizations.

For MPO `diagnostics`, use `expectations_complex` and/or
`unnormalized_expectations` with an `observables` array of full-width Pauli
strings. Each result is an array of `[real, imaginary]` values, in input order.
The former divides by the operator trace; the latter retains its scale.
`max_output_elements` bounds the number of values in each batch. Empty batches
return empty arrays. Scalar `pauli` diagnostics continue to work unchanged.

The same current-state queries are available directly:

```cpp
auto real = simulator->ExpectationValues({"XX", "YY", "ZZ"});
auto normalized = simulator->ExpectationValuesComplex({"XX", "YY"});
auto raw = simulator->ExpectationValuesComplex({"XX", "YY"}, false);
```

The C ABI accepts a simulator pointer returned by `GetSimulator`:

```c
const char *paulis[] = {"XX", "YY", "ZZ"};
double real[3], imag[3];
int ok = MaestroExpectationValues(sim, paulis, 3, real, 3);
ok = MaestroExpectationValuesComplex(sim, paulis, 3, 0, real, imag, 3);
```

Capacity counts entries in each output array. A zero count permits null arrays;
the simulator must still be valid. The functions return 1 on success or 0 on
failure, leave output buffers unchanged on failure, and contain C++ exceptions.
Real queries follow the existing scalar padding rules; complex MPO queries
require exactly one uppercase `I`, `X`, `Y`, or `Z` per qubit. The C++ dynamic
library wrappers expose `ExpectationValues` and `ExpectationValuesComplex`;
their new exports are optional when loading an older library and throw a clear
error only when called against that library.

Maestro Python exposes `sim.expectation_values(paulis)` and
`sim.expectation_values_complex(paulis, normalized=True)`, with matching
`ExpectationValues` / `ExpectationValuesComplex` aliases. These return Python
lists and release the GIL during native computation. Callers must serialize
queries and mutations of the same simulator. Rebuild native C++ consumers when
updating the simulator interface headers.

## Configuration

Backend names and supported methods come from `MaestroGetCapabilitiesJson`;
legacy integer IDs are build-dependent. GPU/QuEST entries can be compiled while
still requiring a runtime plugin/license/device probe. Distributed engines support
statevectors only. `mps` and `mpo` are aliases for the full method names.

Options go under `simulator.options`; the native-name alias is also accepted.
Unknown options, duplicate aliases, incompatible families, and invalid values
are rejected. Small floating-point thresholds retain their precision.

| Name | Native alias | Type | Applicability |
|---|---|---|---|
| `max_bond_dimension` | `matrix_product_state_max_bond_dimension` | positive_integer | tensor |
| `singular_value_threshold` | `matrix_product_state_truncation_threshold` | nonnegative | tensor |
| `truncation_mode` | `matrix_product_state_truncation_mode` | string | tensor |
| `precision` | `` | string | precision |
| `gpu_device` | `gpu_device` | integer | device |
| `seed` | `seed` | integer | all |
| `mps_sampling` | `` | string | mps |
| `disable_optimized_swapping` | `` | boolean | mps, mpo |
| `lookahead_depth` | `` | lookahead | mps, mpo |
| `mps_routing_strategy` | same | string | QCSim MPS and MPO |
| `mps_routing_virtual_bonds` | same | boolean | QCSim/GPU MPS and MPO |
| `mps_routing_truncation_weight` | same | nonnegative | QCSim/GPU MPS and MPO |
| `mps_routing_legacy_cap` | same | boolean | QCSim/GPU MPS and MPO |
| `optimize_circuit` | `` | boolean | all |
| `gate_fusion` | `` | boolean | all |
| `auto_reduce` | `` | boolean | all (default true) |
| `enable_causal_cone_reduction` | `` | boolean | all (default true) |
| `enable_observable_clustering` | `` | boolean | all (default true) |
| `max_cluster_qubits` | `` | positive_integer | all |
| `min_cluster_qubits` | `` | positive_integer (≤ `max_cluster_qubits`) | all |
| `observable_error_tolerance` | `` | number in (0, 1] | all |
| `allow_reset_marginalization` | `` | boolean | all (default false) |
| `enable_sampling_reduction` | `` | boolean | all (default false) |
| `max_simulators` | `max_simulators` | positive_integer | all |
| `trajectory_policy` | `trajectory_policy` | `block_v1` or `shot_v1` (default) | QCSim statevector, density matrix, path integral; opt-in for stabilizer, extended stabilizer, MPS and MPO |
| `trajectory_max_memory_mb` | `trajectory_max_memory_mb` | positive_integer | QCSim statevector, density matrix, stabilizer, MPS and MPO |
| `counts_policy` | `` | `auto` (default), `legacy`, `frame_v1` or `multinomial_v1` | all (specialized samplers run on QCSim) |
| `mpo_kraus_completeness_check` | `matrix_product_operator_kraus_completeness_check` | string | mpo |
| `mpo_restore_trace_after_truncation` | `matrix_product_operator_restore_trace_after_truncation` | boolean | mpo |
| `mpo_hermitize_after_truncation` | `matrix_product_operator_hermitize_after_truncation` | boolean | mpo |
| `mps_svd_solver` | `` | string | gpu_mps |
| `mpo_svd_solver` | `` | string | gpu_mpo |
| `tensor_network_svd_solver` | `` | string | gpu_tn |
| `pp_workers` | `pauli_propagator_workers` | integer | pp (GPU: 1–64; CPU: 0–1024) |
| `pp_sampling_cache_nodes` | `pauli_propagator_sampling_cache_nodes` | integer | pp |
| `pp_memory_limit_bytes` | `pauli_propagator_memory_limit_bytes` | positive_integer | gpu_pp |
| `pp_max_fused_gates` | `pauli_propagator_max_fused_gates` | positive_integer (1–256) | gpu_pp |
| `pp_coefficient_threshold` | `pauli_propagator_coefficient_threshold` | nonnegative | pp |
| `pp_max_pauli_weight` | `pauli_propagator_pauli_weight_threshold` | integer | pp |
| `pp_gates_between_trims` | `pauli_propagator_steps_between_trims` | positive_integer | pp |
| `pp_gates_between_deduplications` | `pauli_propagator_num_gates_between_deduplications` | positive_integer | pp |
| `path_integral_threshold` | `path_integral_threshold` | nonnegative | path |

Tensor options apply to MPS/MPO/TN; GPU solver selectors require their named
GPU method. Precision is `single` or `double` and applies to Aer and the GPU
backends. Truncation is `relative_max` or `discarded_weight`. `mps_sampling` is
`probabilities` or `apply_measure`, and each SVD solver option is `gesvd`,
`gesvdj`, `gesvdp` or `gesvdr`. Pauli propagation without a deduplication cadence
deduplicates at every tenth circuit index, including zero. When either Pauli
threshold is supplied without a trim interval, the trim interval follows the
effective deduplication interval, including an explicit native configuration key.
With a threshold and neither interval supplied, both default to ten. Explicit trim intervals take precedence. GPU
trimming and deduplication use separate intervals; QCSim additionally applies
thresholds during every deduplication pass. Explicitly different intervals can
therefore produce different approximate CPU/GPU results. GPU propagation also
keeps the final trim, including when `INT_MAX` disables periodic trimming.
Requests without thresholds retain the backend's trim default.
GPU buffers grow within the configured budget (default 90% of available memory);
selecting a budget does not allocate the expansion buffers. Independent budgets
are limits, not reservations against other GPU users. GPU workers default to one
and share that budget; memory pressure reduces concurrency automatically. Fused blocks default
to 64 gates (one disables fusion), and the sampling cache defaults to 8192 nodes
(zero disables caching). Larger worker counts can add overhead on small batches.
Omitted seeds are generated randomly for each
request. Specify an explicit simulator seed once, through `execution.seed` or
the options object, to reproduce a run; zero is a valid explicit seed.

Estimation reductions run before simulation. `enable_causal_cone_reduction`
drops operations outside the observables' cones, `auto_reduce` removes
check-generated invariant subspaces (parity reduction), and
`enable_observable_clustering` splits observables with disjoint cones into
groups simulated on compact registers. `max_cluster_qubits` caps a group's width
(unset: 20 for dense statevectors, 10 for density matrices; oversized cones stay
whole), and `min_cluster_qubits` batches small disjoint cones up to that width.
`observable_error_tolerance` opts into bounded backward Pauli evaluation of exact
noisy channels with that absolute error budget per expectation; unset keeps the
reduction exact. `allow_reset_marginalization` averages over unobserved reset
outcomes, preserving ensemble values but not individual trajectories.
`enable_sampling_reduction` samples disjoint measurement cones independently for
`execute`, counts and rows alike (neither then uses reference frames); it is
exact but opt-in pending a workload-based cost model for automatic selection.
Repeated requests with the same seed and configuration remain reproducible
within one Maestro version. Estimates report `auto_reduced`, `qubits_before`, `qubits_after`,
`clusters_count`, `largest_cluster_qubits`, `mean_cluster_qubits`,
`cluster_methods`, `expectation_error_bounds` and `bounded_observables`.

`gate_fusion` forces fusion on or off on eligible methods. When omitted, fusion
is on except for CPU statevectors below 11 qubits and CPU density matrices below
5 qubits, where it costs more than it saves. It is independent of
`optimize_circuit`. Unsupported methods and Aer retain their existing behavior.

For maintainers: `SimulatorConfig` has two active construction paths. Python
bindings populate typed option fields. The native parser populates typed network
controls and seed, while most backend settings use validated `native_options` and
`distributed_options` key/value maps. Both converge in `ConfigureNetwork`. Changes
to option names, conversion or semantics must account for both paths; the typed
fields are not dead code. The native parser avoids populating duplicate
representations of the same setting.

## Noise models

Noise channels are declarative entries with `kind`, nonempty `targets`, and the
parameters below. `channels` may be empty or omitted only when the circuit's own
`pauli_channel`/`kraus` instructions supply the noise, for example to sample a
Pauli channel on a pure-state method. Probabilities must be physical; time constants accept the string
`infinity`. Durations are seconds. Pair channels have exactly two targets; generic
Kraus channels have one or two. Correlated OU bands can be accumulated. Other
repeated settings that would replace an existing setting are rejected.

| Kind | Parameters |
|---|---|
| `pauli` | `px py pz` |
| `depolarizing` | `probability` |
| `dephasing` | `probability` |
| `bit_flip` | `probability` |
| `gate_depolarizing_1q` | `probability` |
| `gate_depolarizing_2q` | `probability` |
| `pair_depolarizing` | `probability` |
| `coherent_rotation` | `rx ry rz` |
| `coherent_depolarizing` | `probability` |
| `coherent_dephasing` | `probability` |
| `coherent_bit_flip` | `probability` |
| `t1` | `gamma duration t1` |
| `t1_2q` | `gamma duration t1` |
| `phase_damping` | `gamma duration t_phi` |
| `generalized_amplitude_damping` | `gamma excited_population` |
| `thermal_relaxation` | `duration t1 t2 excited_population` |
| `thermal_relaxation_2q` | `duration t1 t2 excited_population` |
| `idle` | `t1 t2 excited_population detuning_hz` |
| `crosstalk` | `strength` |
| `readout` | `p_meas1_prep0 p_meas0_prep1 probability` |
| `correlated_phase_flip` | `probability correlation` |
| `kraus` | `operators` |
| `correlated_ar1` | `phi sigma_eta after_1q after_2q stationary_init` |
| `correlated_ou` | `sigma alpha gate_time after_1q after_2q stationary_init` |
| `correlated_ou_band` | `sigma alpha gate_time after_1q after_2q stationary_init` |
| `multi_correlated_ou` | `bands gate_time after_1q after_2q stationary_init` |
| `one_over_f` | `total_power f_min f_max num_bands gate_time after_1q after_2q stationary_init` |

Use `noise.mode` = `combined` (default), `pauli`, `coherent`, or `analytical`.
Combined mode is needed for gate-aware, relaxation, idle, correlated, and Kraus
models. Placement follows the native model: ordinary single-qubit channels after
applicable gates, gate-specific/pair channels after their applicable gates, idle
noise on delays, and asymmetric readout errors when measurements write their
classical bits. Readout rates follow the measured qubit, including permuted,
partial, repeated and conditional measurements; subsequent classical conditions
see the noisy result. Skipped measurements do not apply readout error.
The only accepted explicit `placement` is `after_each_gate`; the channel kind
still determines applicability. Like Python's shared noise helpers, quantum-noise
insertion acts on unconditional gates and delays; conditional gates retain their
original conditional operation. Readout applies to both ordinary and conditional
measurements.

For execute/estimate, noise is injected into each realization before the resulting
circuit reaches network optimization. The optimizer therefore sees inserted noise
operations, rather than an ideal circuit awaiting later noise placement.
`simulator.options.optimize_circuit` retains its existing default. Contract tests
compare on/off results for exact relaxation, Pauli and coherent noise on supported
CPU methods. Those comparisons do not guarantee identical rounding or truncation
for every approximate method; disable optimization when making such comparisons.

`evaluation:"auto"` uses exact channels on density-matrix/MPO methods and sampled
realizations on pure-state methods. `exact` requires a mixed-state method or the
QCSim `pauli_propagator`; `trajectories` runs sampled realizations.

On `statevector` and `matrix_product_state` those are trajectories of the
channels themselves. A realization draws what needs no state (Pauli and
depolarizing errors, relaxation with T2 <= T1) and keeps every other channel for
the backend, which applies one of its Kraus operators with the probability the
state gives it each time the circuit runs. Relaxation with T2 > T1, T1 as
amplitude damping, generalized amplitude damping, correlated phase flips and
Kraus maps are therefore neither approximated nor refused there, and
`noise.approximations` lists only `finite_noise_realizations`. When a model
has such channels and no layer drawn per realization (coherent signs, OU or
1/f disturbances), `execute` builds one circuit that holds all the noise as
channels and runs every shot on it: each shot is a realization of its own,
`noise.realizations` reports the shot count whatever was requested, and
`noise.seed` and the execution seed together determine the draws. The
realizations of `estimate` and `noisy_fidelity` and the shots of a
`checkpoint_batch` suffix share such a circuit in the same way, and there too
either seed alone changes the errors drawn. A channel with one branch only, a
unitary (a `unitary_mixture` with one term, a Pauli channel with one, a single
Kraus operator), is a gate and counts as none of this: nothing is drawn for
it, and a circuit whose channels are all such evolves once for all its shots.
Otherwise
shots that share a realization share its drawn errors and draw the kept
channels each on their own, and the two seeds still determine those branches
together: with a layer drawn per realization beside the channels, on a qubit
of the circuit or on an idle one, either seed alone changes them, in every
operation. The other pure-state methods run realizations of
gates and resets only, with the approximations and limits described below.

On the QCSim `matrix_product_state` and `matrix_product_operator`, a circuit
whose two-qubit gates and channels all act on neighbouring qubits runs in an
order that keeps the chain's work local once a channel has acted. Operations
on different qubits commute, so the circuit is the same; on
`matrix_product_state` a seeded trajectory depends on that order. Circuits that
need routing run as listed, and the layout and swap planning are unaffected.

The QCSim `pauli_propagator` holds no density matrix. It applies the adjoint of
each channel to the observable it propagates, so one execution gives the exact
ensemble average, probabilities and samples, and an unrecorded reset is a
deterministic operation. It holds no amplitudes either: `amplitudes`,
`statevector`, `inner_product` and `mirror_fidelity` without observables are
rejected there when the request is validated, while `probabilities` and
`state_probability` are available. How `auto` resolves depends on the channels
the request's circuits receive:

- Every channel is diagonal in the Pauli basis (Pauli, depolarizing, dephasing,
  phase damping, correlated phase flips): exact channels. Each channel only
  scales terms, at about one extra pass per channel.
- T1, thermal or idle relaxation with a finite T1 acts on the circuits: sampled
  realizations, as before. Relaxation also sends Z to the identity. Without a
  truncation threshold those branches fill the operator space of the
  observable's light cone (`4^width` terms), which costs far more than the
  trajectories. Ask for `evaluation:"exact"` to apply relaxation exactly, with
  the calibrated T2; combine it with `pp_coefficient_threshold` for wide cones.
- Relaxation counts by the channel it applies, not by its kind. A zero T1
  probability is the identity. Thermal or idle relaxation towards equal
  populations (`excited_population: 0.5`) only scales terms, at any T2 the
  channel allows. Relaxation on a qubit that no noisy circuit of the request
  gates, and idle relaxation without a positive delay, is never injected. All
  of these stay exact.
- The model has channels that cannot be sampled (generalized amplitude damping,
  Kraus maps), or a circuit of the request holds a `kraus` instruction: exact
  channels, for the whole request, relaxation included.

Exact evaluation on the propagator is exact only without truncation thresholds;
with a positive `pp_coefficient_threshold`, or a `pp_max_pauli_weight` below the
register, it is an approximation, as any truncated propagation is, and
`noise.approximations` then lists `Pauli_truncation`. On the pure-state methods that do not sample channels,
a model with generalized amplitude damping, correlated phase flips or Kraus maps is rejected, and T1 is drawn as a
reset (`sampled_T1_reset`). Finite realizations, analytical terminal-Pauli damping, and MPO truncation are also
reported as approximations. Analytical mode is estimator-only. Exact channels do
not imply untruncated MPO evolution or remove stochastic coherent realizations.

For thermal relaxation (including two-qubit gate and idle noise) with
`T1 < T2 <= 2*T1`, sampled execution follows Python. `statevector` and
`matrix_product_state` sample the channel and keep the calibrated T2. The other
pure-state methods (tensor network, path integral, stabilizer, extended
stabilizer) use effective `T2 = T1` instead of rejecting the request. Results
identify this approximation as
`thermal_T2_clamped_to_T1` in `noise.approximations` when the request's noisy
circuits draw such relaxation (a gate, or for idle noise a positive delay, on the
qubit; a zero duration is the identity and is not clamped). Execution emits one
warning listing the affected qubits to stderr, which the local server retains in
task logs; validation alone does not emit it. Exact density-matrix/MPO execution keeps
the calibrated T2, as does `evaluation:"exact"` on the QCSim `pauli_propagator`. Unphysical `T2 > 2*T1`, and channels
that need the state on a method that neither holds nor samples them, remain errors.

An explicit simulator seed (`execution.seed` or `simulator.options.seed`,
including zero) controls measurements and readout. Otherwise, an explicit
`noise.seed` also seeds the simulator; if neither is supplied, execution generates
a fresh random uint64 seed and reports it in the result's `seed` field. Reusing
that seed with the same request and the reported
`execution_metadata.configured_options.counts_policy` reproduces the run. The
policy records the selected algorithm: under the default `auto`, an eligible
request that the stabilizer simulator could run uses noisy-Clifford
reference frames, reports `execution_metadata.sampling_strategy:
"reference_frame_v1"` and rewrites `configured_options.counts_policy` to
`"frame_v1"`. Replay that policy along with the seed to use frames again.
Seeded and unseeded QCSim count requests that resolve to adaptive terminal
sampling report `"multinomial_v1"`. A noisy extended-stabilizer checkpoint
that runs any suffix on frames keeps its requested policy, since its other
suffixes may have used another sampler, and reports each suffix's actual
strategy. In every case, replay uses the returned seed and the reported
`configured_options.counts_policy`. Replay is guaranteed within the same
Maestro version and configuration; algorithms and random sequences may change
across versions.
MPI ranks share the generated
seed through their communicator; validation alone does not generate a seed or
require MPI initialization.
`noise.prefix: true` (`checkpoint_batch` only) also applies the noise to the saved
prefix, once, as exact channels; it requires deterministic exact evaluation
(`density_matrix`, `matrix_product_operator` or the QCSim `pauli_propagator`,
without coherent or correlated noise; relaxation on the `pauli_propagator` needs
`evaluation:"exact"`) and reports `noise_scope: "prefix_and_suffix"` instead of
`"suffix_only_ideal_prefix"`. The prefix draws from its own generator, so the
suffixes' noise stream is unchanged.
`noise.seed` is a uint32 (default lower 32 bits of the simulator seed) and drives
circuit-noise injection independently of measurement/readout randomness.
Omitted, `null`, or `"auto"` `realizations` selects one realization for deterministic
exact-channel evolution. Sampled noise or residual coherent/OU/1/f disturbances
use 64 for execution and 100 for estimation/fidelity. Explicit positive counts
are honored. `noise.requested_realizations`, `realizations_policy`,
`stochastic_noise`, and `stochastic_execution` distinguish the request from the
work performed; finite disturbance sampling is reported even with exact channels.
Execute divides
its total shots between realizations; estimator standard errors describe variation
between realizations, not hardware shot error. Multi-shot reuse stays within each
injected realization. Injected Pauli/coherent errors remain fixed for that
realization. Measurement outcomes and readout flips remain per-shot operations;
DM/MPO resets are deterministic channels and run once when no recorded
measurement branch intervenes. Exact channels evolve the density matrix/MPO before
sampling, including fixed CPU MPO and eligible checkpoint suffixes. Terminal
checkpoint suffixes are also sampled from one prepared state on the QCSim
`pauli_propagator`, resets included (it replays every shot under `counts_policy:
"legacy"`), and on QCSim statevector, stabilizer and MPS backends; a suffix reset on
those pure-state backends still runs per shot, since the saved state may be entangled. Each checkpoint result reports its
`sampling_strategy` (`terminal_v1`, `trajectory_v1`, or `mixed_v1` when only some
realizations could be sampled from one state), `stochastic_execution`, and effective
`realizations`. Recorded branches in `execute` and `checkpoint_batch` set
`stochastic_execution`; each shot samples its branch from the exact distribution,
so they add no approximation. A dynamic checkpoint suffix evolves its
deterministic head once on a copy of the restored prefix when all four retained
states (the checkpoint's live state and saved prefix, the copy and its saved
head) fit `trajectory_max_memory_mb`; its branches still run per shot, each copy drawing a
measurement stream derived from the request seed. Nested circuits that end in
measurements are inlined, so they count as terminal measurements. `finite_measurement_trajectories` marks estimates
that average finitely many branches.

Dynamic estimates automatically average 100 measurement trajectories when no
count is specified, including estimates without an external noise model.
`execution_metadata.estimate_evaluation` identifies `trajectory_average` versus
`deterministic`, and `measurement_trajectories` reports branch work. A stochastic
estimate with one explicit realization returns `null` standard errors rather than
claiming zero uncertainty; `noisy_fidelity` with one sampled realization likewise
returns a `null` `standard_error`. Distributed-GPU estimates follow the same rules.
On pure-state backends a reset that precedes every other operation on its qubit
leaves `|0>` and does not make an estimate stochastic; a later reset does.
Analytical damping requires deterministic evolution.
The current noise-model injector does not attach gate noise to conditional gates;
averaging their measurement branches does not extend that injection support.
Repeating the same seeds/configuration is deterministic within the same
backend/runtime; cross-backend bitwise identity
is not promised. Checkpoint suffixes consume sequential noise, readout and simulator
measurement streams. Changing, inserting or reordering an earlier suffix may change
a later suffix's samples. The reproducibility unit is the entire ordered request;
checkpoint restore restores quantum state, not random-generator state.

Legacy `SimpleExecute` and `SimpleEstimate` also honor `config.seed`: a
nonnegative JSON integer in the uint64 range. Invalid seed types/ranges return
the legacy null-result failure. Repeating a seed reseeds an existing handle as
well as a fresh worker; omitting it leaves the simulator's seed configuration
unchanged.

`SimpleExecute` also accepts `counts_policy` (`auto`, `legacy`, `frame_v1` or `multinomial_v1`,
as in `simulator.options`). Under `auto`, an eligible Clifford circuit on a
network that admits the QCSim stabilizer is counted from reference frames, and
the result reports `"sampling_strategy": "reference_frame_v1"`, with or without a seed.
Explicit `legacy` keeps the ordinary network executor. Eligible terminal histograms use
multinomial sampling under `auto`. Supported QCSim dynamic circuits default to
`trajectory_policy="shot_v1"`, including stabilizer, extended-stabilizer, MPS and MPO.
Omitting the policy uses that default even after an earlier call supplied an override.

Fidelity operations use ideal inverse preparation. Delays are identity operations
in that inverse; `noisy_fidelity` applies its noise only to the forward
preparation: sampled Pauli/T1/coherent trajectories on pure states, or exact
channels on density-matrix/MPO backends, where it reads `P(0...0)` after the
inverse. Its circuit follows the rule of Python's `noisy_fidelity`. It may hold
channel instructions: they are noise too, present in the forward preparation and
absent from the ideal inverse. It may also reset a qubit before any other
operation on it, which only prepares that qubit. Mid-circuit measurements, later
resets and classical control have no unitary ideal and are refused. The QCSim `pauli_propagator` reads `P(0...0)` too, through one projector
per qubit; each doubles the propagated terms, so it accepts at most 24 qubits. It does not model a physical mirror experiment with noise and
elapsed-time effects on both halves.

## Distribution and lifecycle

`simulator.backend` is `distributed_gpu` or `distributed_mpi_gpu`.
`simulator.distribution` accepts `devices` (local shards), `global_qubits`,
`backend` (`ex` or `conventional`), `flags`, `max_queued_gates`,
`transfer_workspace_bytes`, `snapshot_storage` (`host`/`gpu`),
`host_qubit_indexing` (`auto`/`local`/`global`), and MPI-only `mpi_p2p_bits`.
MPI uses rank-local GPU visibility/`options.gpu_device`; requests cannot supply an
MPI communicator pointer. Explicit local shard count must be a power of two in
1..32 with at least one local qubit. Repeated local device IDs require the
conventional backend and flag 1; this adds no physical memory capacity. Flags are
1 (shared device), 2 (fixed layout), 4 (local full mesh), 8 (pinned layout), and
16 (conventional host-staged transfers). Fixed/pinned layouts are mutually
exclusive. MPI rejects shared-device, full-mesh, and host-staging flags.

Use the optional `maestro-mpi-worker` under a site MPI launcher. Rank zero reads
and broadcasts the request; all ranks validate and execute identical calls. Any
rank failure aborts the group. States are destroyed, ranks synchronize, the plugin
finalizes, and application MPI finalizes before rank zero publishes one result.
Counts are replicated, not summed across ranks. GPU plugins, valid licensing,
compatible CUDA libraries, and CUDA-aware MPI are deployment prerequisites.
The local server's administrator-owned profiles launch this executable.

## Bounds, exclusions, and tests

Requests are bounded to 16 MiB and serialized results to 64 MiB
(`max_result_bytes`). The bound covers the complete emitted document, including
the `ok`/`schema_version` envelope and a batch's wrapper; a larger result fails
with `invalid_input`. Basis outputs
have a default limit of 65,536 elements and a hard limit of 1,048,576 through
`max_output_elements`; selected indices avoid a full statevector export.
Basis-index queries require fewer than 63 qubits. Batches allow 256 members and
four nested levels. Physical memory limits still depend on the method; the worker
process is the resource/failure boundary.

This API does not add distributed density matrices/MPOs, Kraus trajectories on
pure-state methods other than `statevector` and `matrix_product_state`, Composer
network orchestration, decoder/Sinter execution,
interactive handles across tasks, or an external artifact service.

Build with `BUILD_PYTHON_BINDINGS=OFF`, `MAESTRO_BUILD_REQUEST_TESTS=ON`, and optionally
`MAESTRO_BUILD_MPI_WORKER=ON`. Run `ctest --test-dir BUILD --output-on-failure` with
the rebuilt library on the runtime search path. The public-API suite covers
thermal/idle relaxation, phase/generalized amplitude damping, correlated AR1/OU/
OU-band/multiband/one-over-f noise, crosstalk, analytical damping, seeded repeatability,
invalid parameters, optimization comparisons and MPS bond-limit effects. It also
checks `rxx`/`rzz` instructions against QASM, `noiseless` and `readout_error`
fields (including checkpoint-prefix readout rejection), exact and sampled
`pauli_channel` instructions, observable mirror fidelity, mixed-state overlap and
statevector fidelity, and adjoint values, Jacobians and VJPs against finite differences of
`estimate`, with their shape and backend rejections. A separate
C consumer calls all four request functions and `FreeResult` through the public
header and linked library. Linux CI enables these contract tests; the Windows wheel
dry-run workflow also runs the DLL consumer, API and CLI tests.

The default MPI tests execute two CPU ranks, including collective failure.
Set `MAESTRO_RUN_GPU_REQUEST_TESTS=ON` only when the
local GPU plugin/license/device are available; it tests two logical shards on one
GPU. Separately, `MAESTRO_RUN_MPI_GPU_REQUEST_TESTS=ON` requires both request tests
and the MPI worker. Its ideal, noisy and failing requests use the actual
`distributed_mpi_gpu` backend across two ranks, with no CPU fallback. It also checks
that omitted native seeds are fresh and shared across ranks, reported seeds replay
sampling and readout, explicit zero is preserved, and batch children receive
independent seeds. The parser's unset-seed regression runs in the CPU suite. Configure
rank-local device 0 visibility, the MPI GPU plugin/license and CUDA-aware MPI;
use `MPIEXEC_PREFLAGS` for site launcher/binding arguments. This opt-in suite is
not enabled on ordinary CPU CI runners. On systems where hwloc probes unavailable
graphical displays, an appropriate
site test profile can set `HWLOC_COMPONENTS=-gl,-opencl`.

Fusion status is returned under `execution_metadata.gate_fusion`:
`{"requested":true,"enabled":true,"max_qubits":3}`. `requested` is `null` when
the request did not set `gate_fusion`; an explicit boolean is also echoed in
`execution_metadata.configured_options.gate_fusion`. `enabled`
and `max_qubits` describe Maestro fusion on the backend actually executed,
including automatic method selection; they do not describe native Aer/Ex fusion.

### Counts policy and sampling strategy

`simulator.options.counts_policy` selects how QCSim `execute` counts are sampled.
`multinomial_v1` permits conditional-binomial sampling for terminal counts
when the backend can cheaply obtain a sufficiently populated distribution.
`auto` requests with or without an explicit seed can select this policy after frame eligibility is
resolved. Replay using the returned seed and the reported
`execution_metadata.configured_options.counts_policy`. The policy permits the
optimization; the local count sampler checks its cost for each batch.
`auto` (the default) uses noisy-Clifford reference frames for eligible requests
that the ordinary executor could itself run on the stabilizer simulator, with or without a seed.
Requests can also use adaptive histograms on supported backends. `legacy` selects
the ordinary executor explicitly. On the QCSim `pauli_propagator`, `checkpoint_batch`
suffixes sample their terminal measurements from the saved prefix; `legacy`
replays every shot there instead (readout errors are then drawn shot by shot).
It does not guarantee output from older releases. `frame_v1` requests frames, including for
automatically selected methods. An explicit `candidates` list names every
backend a request allows and replaces the requested one, so neither policy uses
frames unless the list includes the QCSim stabilizer, and either may when it
does, whatever the requested backend. Frames need a supported Clifford circuit, Pauli or readout
noise, Kraus channels that are exact Pauli mixtures (phase damping, for
example) and, with noise, one realization per shot; other requests fall back,
as does thermal noise in the T1 < T2 <= 2 T1 window, which the trajectories of
the stabilizer methods clamp (`thermal_T2_clamped_to_T1`). `noise.seed` drives the circuit errors and
the execution seed the measurements and readout, as on the trajectories.

When frames run, `execution_metadata.sampling_strategy` is `"reference_frame_v1"`,
the reported backend and method are `qcsim` and `stabilizer`,
`sampling_components` is 1, and `configured_options.counts_policy` is
`"frame_v1"` so that replaying the returned seeds reproduces the run (see the
seed section above). Shot rows (below) always report `sampling_strategy`;
counts from the ordinary executor omit it.

Noisy CPU QCSim `execute` requests can run their
realizations on several networks at once, within `max_simulators` and
`trajectory_max_memory_mb`. Each realization keeps its own seed, so seeded
counts replay for the same request configuration. Multi-shot realizations keep
their shared noise and execute as complete batches; outer workers use one inner
simulator each. `execution_metadata.realization_workers` reports the actual
outer concurrency for ordinary noisy execute requests, and for shot-row
requests whose realizations ran on such workers (a dedicated sampler, such as
frames, draws the rows without them and leaves it out). Noisy `estimate` and
`noisy_fidelity` requests run their realizations one after another on the
request's simulator and report 1. Checkpoint suffixes retain
their defined call order. The Python noisy execute, estimate and fidelity
functions report the number as `realization_workers` in their result, by the
same rule; there the realizations of an estimate and of a fidelity run on
several workers.

### Shot rows

`execution.return_samples: true` (`execute` only) returns `samples` instead of
`counts`: `shots` bitstrings, each `num_clbits` characters long, classical bit 0
first (`count_order`). Each row is one shot's complete record. Frames and direct
sessions keep shot order. Rows from the network executor (MPS, MPO and tensor
networks, and any method under a `candidates` list or `enable_sampling_reduction`)
are drawn from a histogram of each noise realization and shuffled within it, so
they are exchangeable rather than chronological. Classical bits that no measurement writes
read `0`. Seeded rows replay exactly. Rows drawn from one noise realization are
exchangeable, and so are the rows of a noiseless request or of one with a
realization per shot: any leading slice of those is a uniform subsample of the
run. With fewer `noise.realizations` than shots, consecutive rows share a
realization, so a leading slice draws from the first realizations only.
`execution_metadata.sampling_strategy` says how the rows were drawn:

| Strategy | Rows |
| --- | --- |
| `reference_frame_v1` | Noisy-Clifford reference frames: a Stabilizer method, or automatic selection whose `candidates`, if listed, include the QCSim stabilizer; a supported Clifford circuit and, with noise, one realization per shot |
| `extended_frame_v1` | Batched Pauli frames for an eligible noisy QCSim `extended_stabilizer` request without `candidates`, under `counts_policy: "auto"` or `"frame_v1"`, with or without a seed. Ordinary execute results report `"frame_v1"` as the resolved count policy; mixed-strategy checkpoints retain the requested policy and report per-suffix strategies. Applies to Pauli or combined noise without exact channels, with readout, measurements and resets and no classical control. Rows of one noise realization share its errors. `noise.seed` drives the circuit errors and the execution seed drives measurements and readout. Relaxation that sampled execution draws as resets (T1, or thermal or idle relaxation with a finite T1, on a qubit the circuit gates or delays) needs one realization per shot, and thermal noise with T2 > T1 on such a qubit keeps scalar trajectories. The memory budget is soft: a shot that alone exceeds it runs by itself beyond it. Noisy `execute` counts and each `checkpoint_batch` suffix after a gate-only prefix report the same strategy |
| `terminal_v1` | One prepared state sampled for every row (terminal measurements only) |
| `trajectory_v1` | Each shot, or each noise realization, executed on its own |
| `mixed_v1` | Checkpoint suffixes only: some realizations sampled from one prepared state, others shot by shot |

Rows use frames whenever they apply; `counts_policy` governs counts only. A
`candidates` list names every backend the request allows, for rows as for
counts: rows run on the backend selected from it, and the metadata reports that
backend. With noise, the shots are divided between `noise.realizations` as for counts, and
`noise.realizations` in the result reports the number used. Rows are the
algorithm of Python's `return_samples=True` and of the C++
`MaestroExecution::SampleCircuitRows` and `MaestroExecution::ExecuteNoisy`
(`Execution/SampleRecords.h`). Distributed GPU execution does not support rows.

A JSON row costs `num_clbits + 3` bytes, so `shots * (num_clbits + 3)` must fit
`max_result_bytes`; validation rejects larger requests. `MaestroRunRequestSamples`
has no such bound. It runs an `execute` request as if `return_samples` were set
and writes packed rows: each row holds `ceil(num_clbits / 64)` words
(`row_words`), and classical bit `b` is bit `b % 64` of the row's word `b / 64`.
`capacity` counts words and must be at least `shots * row_words`. With `rows`
null nothing executes: the request is validated and the result reports `shots`,
`num_clbits`, `row_words` and `required_words`, so a caller can size its buffer
first without sampling twice. Otherwise the result is the execute result without
`samples`. Errors and ownership follow `MaestroRunRequestJson`.

`MaestroCreateReferenceFrameSampler` compiles a noisy Clifford circuit once, for
repeated batches (QEC decoding loops, for example) without re-simulating its
tableau. Its request is a JSON object with `schema_version: 2`, `circuit`, and
optionally `simulator` and `noise`, validated as an `execute` request would be.
An omitted `simulator` means the QCSim `stabilizer` method (not `execute`'s
statevector default). Noise needs fixed selection of the `stabilizer` method, as
in `execute`. The simulator must admit the QCSim stabilizer: the `stabilizer`
method, or automatic selection whose `candidates`, if listed, include it. The noise must use `mode` `pauli` or
`combined` with Pauli-mixture channels: Pauli channels, phase damping and readout
errors qualify; amplitude damping, finite-T1 relaxation and coherent, correlated
or crosstalk channels do not. When frames do not apply it returns null and, if
`error` is not null, stores an error JSON (code `unsupported_capability` for an
unsupported circuit) for `FreeResult`. `MaestroReferenceFrameSamplerShape` stores
the compiled register's qubits and the classical bits of each row, which set its
`ceil(clbits / 64)` words, and returns 1 on success.
`MaestroReferenceFrameSampleBits` writes `shots` packed rows (as above; `capacity`
counts words) and returns 1 on success. Equal seeds give equal rows.
`MaestroReferenceFrameSampleCountsBits` writes the histogram of exactly those rows:
each distinct row once, packed, with its count; `capacity` counts entries and must
be at least `shots`. Sampling calls on one handle may run concurrently. Release
the handle with `MaestroDestroyReferenceFrameSampler`. Python exposes the same
compiled plan as `maestro.compile_reference_sampler`.

With `simulator` set to the QCSim `extended_stabilizer` method (fixed selection,
no `candidates`), the same handle compiles any circuit of that method's gates to
batched extended-stabilizer frames: shots in batches that share one Clifford
basis, each with independent noise and its own Pauli frame and amplitudes. The
noise rules are the same as for reference frames, and channels apply exactly:
in the T1 < T2 <= 2 T1 window, where `execute` clamps T2 to T1, the handle keeps
T2. Sampling runs at most 16 batches at once, or fewer under `max_simulators`,
within `trajectory_max_memory_mb`, which also pays for the cached noiseless
head. A batch that does not fit runs afterwards, alone, with the rest of the
budget, split down to single shots. Every shot draws from streams of its own,
and a measurement draws the shot's physical outcome, so rows depend on the seed
only, never on the batch, the budget, the worker count or the host. The budget
is soft, as for `execute` and its trajectories: a shot that alone exceeds it runs
by itself beyond it, with the draws it would have had.

Noise targets must lie in the declared register. Python's
`NoiseModel.native_channels(num_qubits)` therefore leaves out channels on
qubits beyond `num_qubits`, including crosstalk spectators there, whose
rotations on an untouched `|0>` are a global phase.


### CPU and GPU MPO (Hastings implementation)

The GPU MPO backend requires the updated plugin exporting `MPOSampleBits` and
`MPOSampleHistogram`. Maestro checks the full MPO API before creating a simulator;
older plugins cannot be used as MPO backends with this version. Rebuild C++ clients
and both Python extensions when upgrading (the generic state interface has new
virtual methods).

`execute` counts support registers wider than 64 qubits. `state_probability` accepts
an arbitrary-width, q0-first `target_state` bitstring for CPU and GPU MPO. Integer basis
queries retain their documented bounds; full state enumeration is exponential.
`max_bond_dimension: 0` removes the MPO bond cap. The two
`mpo_*_after_truncation` options now apply to GPU as well as CPU MPO.

CPU and GPU MPO `diagnostics` also accept:

- `density_matrix` and `unnormalized_density_matrix`: bounded dense output, using
  the existing `{dimension, row_major}` encoding; at most 13 qubits.
- `element`: raw matrix element, with q0-first `row_state` and `col_state` strings.
- `expectation_complex` and `unnormalized_expectation`: complex Pauli expectations,
  with `pauli` containing one `I`, `X`, `Y`, or `Z` per logical qubit.

A diagnostics request can supply `operators`, an ordered list of
`{qubits, matrix, normalize}` objects. Each matrix is a flat row-major array of
complex `[real, imaginary]` entries on one or two targets; the first target is the
local least-significant bit. Each operation computes `A rho A†`. `normalize`
defaults to false; true divides by the resulting trace and preserves the state if
postselection fails. These state-changing operators require QCSim or GPU MPO.
Optional `move_qubits` supports QCSim and GPU MPS as well as MPO, and moves those
logical qubits to the start of the chain, preserving their current physical
relative order. Routing swaps use the configured cutoff and bond cap and can
truncate the state. GPU MPS sampling does not insert these moves automatically.
Operators run after the circuit, followed by explicit routing, maintenance,
and diagnostics.

Both backends expose trace-normalized dense output, partial trace, and
Hilbert–Schmidt overlap. Raw matrix elements, unnormalized expectations, and
unnormalized dense output retain the operator's scale. `trace` preserves both
real and imaginary components.

The same C++ `IState`/`ISimulator` methods on QCSim and GPU MPO expose `ProbabilityBits`,
`DensityMatrixElementBits`, `GetDensityMatrix(normalized)`,
`ExpectationValueComplex(pauli, normalized)`, `ApplyOperator`, and
`MoveAtBeginningOfChain`. `SampleCountsMany` and `MeasureNoCollapseMany` support
arbitrary-width results; packed `SampleCounts` supports at most 64 selected bits,
including subsets of a larger register. Repeated selected qubits reproduce the
same measured bit in every requested position. Unsupported backends throw.

The plain C declarations in `maestrolib/Interface.h` expose corresponding
`Maestro*` functions plus `MaestroSampleCountsBits` and `MaestroMeasureBits`.
They return 1 on success and 0 on failure without throwing across C. C matrix
buffers use **column-major interleaved doubles**, as documented in that header;
JSON matrices use row-major arrays. Sampling buffers are caller-owned, and the
header specifies their capacities. Existing `ConfigureSimulator` exposes the
truncation controls.

Maestro Python uses `probability_bits`, `density_matrix_element_bits`,
`get_density_matrix`, `expectation_value_complex`, `apply_operator`, and
`move_at_beginning_of_chain`. `SampleCountsMany` returns tuple-of-bool keys;
`MeasureMany` and `MeasureNoCollapseMany` return lists. Composer Python exposes
the C++ names and accepts lists for the new bit-vector and routing methods;
`SampleCountsManyWithList` likewise returns tuple keys.

Precision is selected before allocation with `precision=single|double` or
`use_double_precision=0|1|false|true`. The aliases stay synchronized, with the
last setting taking effect. GPU statevector, MPS, tensor network, density matrix,
MPO, distributed GPU statevector, and Aer statevector/density matrix (including
composite Aer) retain the selected precision through initialization and saved-state
restoration, and cloning where supported. GPU tensor-network cloning remains
unsupported.
After initialization the precision is fixed: a precision setting applied then
(for example by a configuration replay) is ignored, and a new one takes effect
only after clearing the state. `GetConfiguration` reports the active native
GPU precision. QCSim and Aer MPS always compute in double precision. When omitted,
backend defaults remain GPU single precision and Aer/QCSim double precision.

## MPS operator expectations and full state output

QCSim and GPU MPS support the read-only complex query
`ExpectationValueOperators(qubits, matrices)` in native C++ and
`expectation_value_operators(qubits, matrices)` (also `ExpectationValueOperators`)
in Python. Supply one finite 2x2 matrix per target. Targets may repeat; list order
means the result is `<psi|O_(k-1)...O_0|psi>`. Matrices need not be unitary or
Hermitian. An empty list returns one. The query never applies operators to the
stored state. Unsupported backends report an error.

The corresponding JSON diagnostic is `operator_expectation`, returning
`[real, imaginary]`. Supply `expectation_operators`, an ordered list of
`{"qubit": q, "matrix": [m00, m01, m10, m11]}` objects. Entries use the existing
real-number or `[real, imaginary]` encoding. This field is distinct from the
state-changing MPO `operators` field. For example, on an MPS prepared as `|00>`:

```json
{
  "schema_version": 2,
  "operation": "diagnostics",
  "circuit": {"num_qubits": 2, "source": "OPENQASM 2.0; qreg q[2];"},
  "simulator": {"backend": "gpu", "method": "matrix_product_state"},
  "diagnostics": ["operator_expectation"],
  "expectation_operators": [
    {"qubit": 0, "matrix": [0, 1, 1, 0]},
    {"qubit": 0, "matrix": [0, [0,-1], [0,1], 0]}
  ]
}
```

The result is `[0,-1]`, since the product is `YX = -iZ`. List length is bounded
by `max_output_elements`. Missing matrices, out-of-range targets, non-finite
entries, or use without the diagnostic are rejected during request validation.

`GetStateVector()` in native C++ and `get_statevector()` / `GetStateVector()` in
Python return all pure-state amplitudes with logical qubit zero as the low bit.
Mixed states have no unique statevector. Full JSON `statevector` / `amplitudes`
and `probabilities` outputs use bulk readers where supported; tensor-network
probabilities retain per-index queries. Supplying `basis_states` also keeps
selected scalar queries. Network amplitude execution uses the shared full-state
query and retains its mapping back to caller qubit order. Full outputs remain
exponential and JSON output bounds still apply. GPU MPS dense readers support
at most 40 qubits and can fail earlier when memory is insufficient.

The plain C functions are:

```c
int MaestroGetStateVector(void *sim, double *interleaved, size_t capacity);
int MaestroExpectationValueOperators(void *sim, const unsigned long *qubits,
    size_t count, const double *matrices, double *real, double *imag);
```

Statevector capacity counts complex entries; allocate twice that many doubles.
Operator matrices are consecutive **row-major**, interleaved 2x2 matrices
(eight doubles each). Both functions return 1 on success or 0 on failure,
leaving output buffers unchanged on failure. For zero operators, input arrays
may be null. `MaestroLibrary` and `Simulator` provide C++ wrappers for both
queries and `MoveAtBeginningOfChain`; their operator matrices use
`std::array<std::complex<double>,4>` in row-major order. The direct `ISimulator`
method uses Eigen matrices. These queries flush pending fused gates before reading.
Empty chain-movement lists are no-ops on supported simulators across native C++,
C, Python, the dynamic C++ wrappers, and JSON (`"move_qubits": []`). The C pointer
may be null when the count is zero; a null pointer with a nonzero count is invalid.
GPU MPS full-state readout writes into Maestro's result buffer and uses bounded
host staging for precision conversion and qubit permutation. Probability readout
does not materialize a full host amplitude vector. GPU contraction workspace and
the requested output still grow exponentially.
`MaestroLibrary`, `Simulator`, and `SimpleSimulator` also expose
`RunRequestJson(request)` and `ValidateRequestJson(request)`, returning the
complete versioned response as a `std::string`.

New GPU query symbols are optional: older plugins retain scalar expectation and
dense-readout fallbacks and full bond-array callbacks. Explicit MPS chain moves
and arbitrary-operator expectations require an updated plugin and report an
unavailable-feature error otherwise. A failed native batch does not silently
retry as scalar queries. Bond statistics continue to mean the maximum reached
during execution, including transient routing growth, rather than the current
bond dimension. MPS sampling propagates backend failure or incomplete shot
counts as an error. Legacy C pointer-returning readers/samplers return null on
failure; `Probability` returns NaN on a backend error. C save/restore returns
zero on failure.


## Additive discovery metadata

`capabilities` keeps schema version 2 and now includes `build.version` (the package
version), `build.source_revision` (Git revision captured at CMake configuration,
with `-dirty` for tracked modifications, or `unknown` in source archives),
`build.qcsim_revision` (the same for the QCSim sources the build compiled
against, or `unknown` where they are not a Git checkout of their own), and
`capability_scope: "validation"`. These identifiers are informational; use the
catalog and request validation for feature negotiation. Python returns the
`build` object from `maestro.build_info()`, and `maestro.__version__` is its
version.

`noise` describes the noise block: its `modes`, `evaluations`, the
`exact_configurations` that apply channels exactly, how omitted, `null` or `"auto"`
`realizations` resolve (`automatic`), the reported `metadata`, and
`checkpoint_prefix` (the `prefix` field and its requirements). `estimate` lists the
`evaluations` and the number of `dynamic_measurement_trajectories`. `noisy_fidelity`
lists its modes and configurations, including the exact mixed-state ones.
`automatic_selection` states the rules for `candidates`, and `sampling_strategies`
the `sampling_strategy` values results may report.

`diagnostics` lists names and `supported_configurations` backend/method pairs.
Each option also lists `supported_configurations`, and where applicable `enum`,
`minimum` and `maximum`. Conditional `exceptions` override general bounds (MPO
accepts zero for unlimited bond dimension); `constraints` narrow choices (Aer
truncation supports only `discarded_weight`). Cross-field and size constraints
remain enforced by request validation. Catalog support does not probe optional
plugins, GPU hardware or licensing, and is not a promise of runtime readiness.

`instructions` describes the instruction-array format: `gates` lists each gate
name with its parameter count (including `rxx` and `rzz`), `operations` lists
`measure`, `reset`, `delay`, `kraus` and `pauli_channel`, and `fields` maps each
optional field (`noiseless`, `readout_error`, `probabilities`, ...) to the
instructions that accept it. `adjoint` lists the adjoint operations, their
`supported_configurations`, readout kinds, differentiable gates with their
parameter counts, `max_qubits` and `max_probability_qubits`. `mixed_state_queries`
lists `density_matrix_overlap` and `fidelity_with_statevector`, their
`supported_configurations` and `max_target_statevector_qubits`. Clients can detect
these features here without sending probe requests.

### Automatic CPU MPS/MPO routing

`mps_routing_strategy` accepts `"legacy"` (default) or opt-in `"combined"` on QCSim
MPS/MPO. `"legacy"` retains the native local/adaptive routing policy.
The former name `"current"` is rejected; use `"legacy"` to request the default
strategy explicitly or `"combined"` to select bounded routing.
`"combined"` uses bounded work estimates from the actual tensor ranks when the
network's automatic lookahead resolves to zero: a horizon of up to 24 two-qubit
operations for MPS registers larger than 10 qubits, or up to 4 for MPO registers
of at most 8 qubits. It requires virtual-bond routing to be disabled and the
default gate/SWAP growth factors. Other cases retain the native routing policy.

The existing optimization thresholds still apply. By default they require an
effective register width of at least 12 and a bond cap of at least 32, or an unlimited cap. An MPO contributes twice
its logical width to the threshold and automatic-depth calculation. Automatic
depth uses the prepared routing operations, including fusion when enabled.
This option does not change initial qubit layout. Explicit `lookahead_depth`
values of zero or greater and `disable_optimized_swapping=true` retain their
existing behavior. GPU routing is unchanged.

The same optional field is available on Python `SimulatorConfig`; `None`
retains a native request value or the default. Direct simulator callers retain
native routing until the network explicitly resolves automatic lookahead.
At finite bond caps, routing can change truncation error as well as runtime.
The combined policy has measured MPS and MPO accuracy regressions, including a
noisy MPO case that is also about 70% slower than legacy. It is therefore opt-in;
compare total execution time and accuracy for the intended workload before using it.
The 4,096-update budget for the MPS forecast is not a wall-time limit, and its
24-interaction horizon may shrink on wider registers.

### Experimental virtual-bond routing

QCSim and GPU MPS/MPO named SWAP gates exchange logical qubit labels without updating
the tensors. Initial-layout planning and lookahead model these as zero tensor
cost and unchanged bond dimensions, while following the exchanged labels for
later gates. Fusion keeps these SWAPs as named operations. Internal routing
swaps that move tensor sites still incur contraction and truncation costs.
This backend behavior applies with virtual-bond routing either enabled or disabled.
QCSim MPS/MPO exact SWAP matrices use the same relabelling path for generic
two-qubit gates and for MPO `ApplyOperator` calls with `normalize=false`. GPU adapters
query the optional `MPSSwapRelabels` / `MPOSwapRelabels` plugin capabilities at
initialization and preserve virtual bond history for named and exact matrix
SWAPs when supported. Plugins without these capabilities retain physical-SWAP
planning and fusion behavior.

`mps_routing_virtual_bonds` defaults to `false`. Setting it to `true` enables
bounded estimates above the physical bond cap for initial layout and lookahead
routing. It does not change the simulator's physical bond limit. Online history
requires optimized swapping, positive lookahead, and room above the physical cap.
The virtual limit is at most four times that cap and never exceeds the exact
central-cut rank. This allows two additional rank doublings without tying the
heuristic's strength to register size.

For an estimated dimension `d` and physical cap `c`, the routing score is
`min(d,c)^3 * (1 + weight * max(0,d/c - 1))`. The first term models capped
runtime; the second is a heuristic truncation penalty, not an error bound.
`mps_routing_truncation_weight` defaults to `1`; `0` removes the penalty.
Larger weights favor avoiding overflow and may increase execution time. This
experimental policy does not guarantee better accuracy or runtime on every circuit.
An unlimited physical cap uses the ordinary cubic model without virtual history.

Each two-qubit gate grows the estimated bond by its effective Schmidt rank: the
participation ratio of its operator Schmidt coefficients, computed from the gate
matrix (closed forms for named controlled gates). It is 2 for CX/CZ and 4 for
SWAP, and tends to 1 as a controlled rotation's angle shrinks, so the
near-identity controlled phases of a QFT barely count. Fused blocks carry the
product of their gates' ranks, capped at 4. Below rank 2 the growth estimate
interpolates linearly between an unchanged bond (rank 1) and the rank-2 result.

The initial-layout search keeps the circuit's own qubit order for an MPS unless
the cost model predicts a layout at least four times cheaper; an MPO accepts any
predicted improvement. Before simulation, qubits are compacted in ascending
order, so a circuit's numbering is the chain order the search starts from.
The search is skipped for a circuit whose two-qubit operations form lines that
do not run in that order (every qubit has at most two partners and they close
no ring, as for a circuit laid out on a line of a device): its qubits are placed
along the lines, where no operation needs a swap. That also holds under a bond
limit below the one from which a layout is searched for; a register too small
for the search is left as it is. A circuit with a SWAP gate keeps the search
when SWAP gates relabel the qubits.

The same settings are available as Python `SimulatorConfig` keyword arguments
and properties. `None` retains defaults. The compatibility flag
`mps_routing_legacy_cap=true` forces virtual routing off, including when the new
flag is true; `false` alone does not opt in.

The live accuracy/runtime benchmark is excluded from default Boost test runs.
Run it explicitly with
`--run_test=MPSSwapOptTests/VirtualRoutingLiveBenchmark --log_level=message`.
It compares real network execution at equal physical caps; timings are diagnostic,
not machine-dependent CI assertions.
