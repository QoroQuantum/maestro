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
ideal execute/estimate. Results identify the backend/method actually used.

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
  {"name":"x", "qubits":[0]},
  {"name":"delay", "qubits":[0], "duration":0.000001},
  {"name":"kraus", "qubits":[0], "operators":[[0,1,1,0]]},
  {"name":"reset", "qubits":[0]},
  {"name":"measure", "qubits":[0], "clbits":[0]}
]
```

Kraus matrices are flat row-major arrays with real or complex entries and require
one/two targets and a mixed-state backend. Identity instructions remain present
for noise injection. Gate parameter counts and U1/U2/U3 conventions are checked.

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
| `execute` | `execution.shots` (default 1024) | Measured counts; total shots is not multiplied by realizations |
| `estimate` | `observables` | Exact observable evaluation per realization, mean and realization standard errors |
| `statevector`, `amplitudes` | optional `basis_states` | Pure-state amplitudes; these two names have the same output shape |
| `probabilities` | optional `basis_states` | Selected or full basis probabilities |
| `state_probability` | `target_state` | Probability of a q0-first bitstring |
| `diagnostics` | `diagnostics`, `maintenance`, `keep_qubits` | Mixed-state diagnostics/reduced density matrix; MPS/MPO maintenance |
| `inner_product` | `other_circuit` | Complex overlap of two unitary preparations |
| `mirror_fidelity` | — | Fidelity of the preparation followed by its inverse |
| `noisy_fidelity` | coherent `noise` | Ideal-versus-noisy unitary fidelity across realizations |
| `incremental_evolve` | `step_circuit`, ordered `steps`, `observables` | Prepare once, advance to each step, observe without reset |
| `checkpoint_batch` | `suffixes`, `execution.shots` | Save ideal prefix, restore for each suffix/shot; noise applies to suffixes only |
| `batch` | `requests` | Independent native requests in one worker; validate all members first |
| `validate` | circuit/configuration | Validation response only |

State queries remove terminal measurements; mid-circuit measurements are rejected
for these queries. Overlap/fidelity require unitary circuits. Incremental evolution
accepts explicit Kraus instructions in the step circuit for exact noisy evolution;
a top-level stochastic noise model is rejected for this operation. State queries
with top-level noise require a single exact realization. No mixed-state vector or
pure-state overlap is advertised for density matrices/MPOs. Individual methods may
have narrower numerical-query support and return a native error.

Diagnostics are `trace`, `purity` (defaults), `trace_of_square`,
`hermiticity_residual`, `is_hermitian`, and `partial_trace`, for density
matrices and MPOs. The latter requires unique `keep_qubits` and returns
`{dimension,row_major}`. Maintenance accepts `restore_trace` and `hermitize`
(density matrix and MPO) and `trim`/`recanonicalize` (MPS and MPO). An MPS
accepts the operation for maintenance only and reports no diagnostics by
default.

## Batched Pauli expectations

`estimate` evaluates all observables together on each prepared state, including
each noise realization. `incremental_evolve` does the same at each requested
step. CPU QCSim MPS shares repeated observable prefixes; MPO shares contraction
environments and trace normalization. Other backends preserve their scalar
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
| `disable_optimized_swapping` | `` | boolean | mps |
| `lookahead_depth` | `` | lookahead | mps |
| `optimize_circuit` | `` | boolean | all |
| `gate_fusion` | `` | boolean | all |
| `max_simulators` | `max_simulators` | positive_integer | all |
| `mpo_kraus_completeness_check` | `matrix_product_operator_kraus_completeness_check` | string | mpo |
| `mpo_restore_trace_after_truncation` | `matrix_product_operator_restore_trace_after_truncation` | boolean | mpo |
| `mpo_hermitize_after_truncation` | `matrix_product_operator_hermitize_after_truncation` | boolean | mpo |
| `mps_svd_solver` | `` | string | gpu_mps |
| `mpo_svd_solver` | `` | string | gpu_mpo |
| `tensor_network_svd_solver` | `` | string | gpu_tn |
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
deduplicates every 10 operations. Omitted seeds are generated randomly for each
request. Specify an explicit simulator seed once, through `execution.seed` or
the options object, to reproduce a run; zero is a valid explicit seed.

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
parameters below. Probabilities must be physical; time constants accept the string
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
realizations on pure-state methods. `exact` requires a mixed-state method;
`trajectories` samples the native noise helper, not a general Kraus trajectory
engine. Unsupported non-Pauli pure-state channels are rejected. Sampled T1 reset,
finite realizations, analytical terminal-Pauli damping, and MPO truncation are
reported as approximations. Analytical mode is estimator-only. Exact channels do
not imply untruncated MPO evolution or remove stochastic coherent realizations.

For thermal relaxation (including two-qubit gate and idle noise) with
`T1 < T2 <= 2*T1`, sampled execution follows Python: it uses effective `T2 = T1`
instead of rejecting the request. Results identify this approximation as
`thermal_T2_clamped_to_T1` in `noise.approximations`. Execution emits one warning
listing the affected qubits to stderr, which the local server retains in task
logs; validation alone does not emit it. Exact density-matrix/MPO execution keeps
the calibrated T2. Unphysical `T2 > 2*T1` and exact-only channels on sampled
backends remain errors.

An explicit simulator seed (`execution.seed` or `simulator.options.seed`,
including zero) controls measurements and readout. Otherwise, an explicit
`noise.seed` also seeds the simulator; if neither is supplied, execution generates
a fresh random uint64 seed and reports it in the result's `seed` field. Reusing
that seed with the same request reproduces the run. MPI ranks share the generated
seed through their communicator; validation alone does not generate a seed or
require MPI initialization.
`noise.seed` is a uint32 (default lower 32 bits of the simulator seed) and drives
circuit-noise injection independently of measurement/readout randomness.
`realizations` defaults to 1 for exact channels and 64 otherwise. Execute divides
its total shots between realizations; estimator standard errors describe variation
between realizations, not hardware shot error. Multi-shot reuse stays within each
injected realization. Injected Pauli/coherent errors remain fixed for that
realization; resets on evolved qubits, measurement outcomes, and readout flips
remain per-shot operations. Exact channels evolve the density matrix/MPO before
sampling. Selecting a fixed backend preserves these rules.
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

Fidelity operations use ideal inverse preparation. Delays are identity operations
in that inverse; `noisy_fidelity` applies coherent gate noise only to the forward
preparation and rejects idle/relaxation channels. It does not model a physical
mirror experiment with noise and elapsed-time effects on both halves.

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

Requests are bounded to 16 MiB and serialized results to 64 MiB. Basis outputs
have a default limit of 65,536 elements and a hard limit of 1,048,576 through
`max_output_elements`; selected indices avoid a full statevector export.
Basis-index queries require fewer than 63 qubits. Batches allow 256 members and
four nested levels. Physical memory limits still depend on the method; the worker
process is the resource/failure boundary.

This API does not add distributed density matrices/MPOs, arbitrary pure-state
Kraus trajectories, Composer network orchestration, decoder/Sinter execution,
interactive handles across tasks, or an external artifact service.

Build with `BUILD_PYTHON_BINDINGS=OFF`, `MAESTRO_BUILD_REQUEST_TESTS=ON`, and optionally
`MAESTRO_BUILD_MPI_WORKER=ON`. Run `ctest --test-dir BUILD --output-on-failure` with
the rebuilt library on the runtime search path. The public-API suite covers
thermal/idle relaxation, phase/generalized amplitude damping, correlated AR1/OU/
OU-band/multiband/one-over-f noise, crosstalk, analytical damping, seeded repeatability,
invalid parameters, optimization comparisons and MPS bond-limit effects. A separate
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
postselection fails. Optional `move_qubits` moves those logical qubits to the start
of the chain. Operators run after the circuit, followed by routing, maintenance,
and diagnostics. These fields require QCSim or GPU MPO.

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

## Additive discovery metadata

`capabilities` keeps schema version 2 and now includes `build.version` (the package
version), `build.source_revision` (Git revision captured at CMake configuration,
with `-dirty` for tracked modifications, or `unknown` in source archives), and
`capability_scope: "validation"`. These identifiers are informational; use the
catalog and request validation for feature negotiation.

`diagnostics` lists names and `supported_configurations` backend/method pairs.
Each option also lists `supported_configurations`, and where applicable `enum`,
`minimum` and `maximum`. Conditional `exceptions` override general bounds (MPO
accepts zero for unlimited bond dimension); `constraints` narrow choices (Aer
truncation supports only `discarded_weight`). Cross-field and size constraints
remain enforced by request validation. Catalog support does not probe optional
plugins, GPU hardware or licensing, and is not a promise of runtime readiness.
