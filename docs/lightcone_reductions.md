# Lightcone and symmetry reductions

Python `estimate`, native JSON `estimate`, and native `NoiseAdd` estimation share
the same preparation code. Preparation happens before allocating a new simulator.
It first attempts global parity reduction, partitions observables, then tightens
each cone and retries parity on smaller circuits when the global attempt failed.
An empty Python/C++ observable list returns no groups and allocates no simulator;
the native JSON contract continues to reject an empty observable list.
Fixed or already-configured GPU tensor-network estimation pads one-qubit cones
with an idle identity wire to meet that backend's two-qubit minimum. GPU
tensor-network sampling retains full-register execution when decomposition would
violate that minimum.

## Exact reductions

Backward cones accept CPTP channels and resets for density matrices and MPOs.
Sampled resets require extra care: resetting an entangled spectator can change a
single trajectory's expectation. Pure-state requests retain the original path
when a reset shares an interaction component with a requested observable.
Disconnected spectator resets can be removed. Set
`allow_reset_marginalization=True` to allow ensemble-equivalent reductions of
pure-state trajectories too. This preserves ensemble means and shot distributions,
but may change individual trajectory values and random streams. It defaults to
false in Python and native JSON. Classical feedback still disables reduction.
Reset targets are respected: resetting to `|1>` does not establish a known-zero
control. In ensemble/marginalized analysis, resets stop backward axis propagation;
retained reset operations still prepare their specified targets.

Cone tracing tracks conservative X/Z axis sets, including branches created by
non-Clifford gates. It removes a gate only when the entire gate commutes with
every possible Pauli term, not merely with one control's factor. A forward pass
also removes diagonal gates on known zero-state inputs and controlled operations
with a known-zero control. Channels invalidate input-state facts conservatively.
Noise is injected before these transformations, so pruning a gate does not erase
the noise that was scheduled after it.

Parity reduction supports H preparation, CNOT frames, Rx/Rz, X/Z, S/Sdg, T/Tdg,
P, Sx/SxDag and CZ, including the global-H dual. CZ includes both single-qubit
rotations and the two-qubit check. Separate Z and inverse-transpose X frames allow
mixers inside unfinished CNOT ladders. Observables are conjugated through any
remaining final CNOT frame, including their Pauli signs. Trailing identity padding and zero-angle rotations do not add checks.
The detector retains its 64-physical-qubit and 22-check bounds, but can retry on
individual cones below those bounds.

## Grouping and reuse

`max_cluster_qubits=None` selects a batching budget of 20 for statevectors and 10
for density matrices (16 MiB of complex doubles before backend overhead). Other
methods use connected components. This is a soft grouping budget: one cone larger
than the budget stays intact, and may still require substantial memory. Explicit
budgets override the automatic values. Groups used for expectation estimation
may overlap; each is simulated separately.

`min_cluster_qubits` optionally combines small disjoint cones up to the supplied
width, within the maximum budget, to amortize simulator setup. It is disabled by
default. Both options are Python config fields and native JSON simulator options.

Clustering propagates 64 observables at once. Structural tracing skips single-qubit
operations; Pauli-aware tracing includes axis-changing single-qubit gates.
Contained cones are absorbed together instead of triggering repeated greedy
searches. Repeated noisy estimates cache structural groups and qubit mappings,
checking the ordered multi-qubit topology, observables and relevant configuration
before reuse. Each current realization is then tightened with its own Pauli and
input-state facts. Clean-circuit tightened mappings are never reused: an injected
X on a previously active control can activate gates and require extra wires.
Reset/classical dependencies retain their policy-dependent fallback. Full noise
injection is intentionally preserved: skipping spectator injections would change
random streams, delay schedules or spectator crosstalk. `cone_cache_hits` reports
reuse in Python and JSON noisy estimates. Identity and zero-factor expectations
are resolved before allocation. Identical compact circuits containing only known
unitary gates share a simulator run; parameters and ordered operands must match
exactly. Group diagnostics describe the runs actually required.

## Optional bounded noisy evaluation

```python
config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.DensityMatrix,
    observable_error_tolerance=1e-4,
)
result = circuit.noisy_estimate_montecarlo(
    ["ZZ"], noise_model, noise_realizations=1, config=config
)
```

`observable_error_tolerance` is an absolute error budget **per expectation** in
`(0, 1]`; the default is disabled. Exact-channel circuits can be evaluated by a
bounded backward Pauli expansion. If the current operator is `a I + sum(b_P P)`,
discarding the nonidentity remainder costs at most `sum(abs(b_P))`: every Pauli
has operator norm one, and every earlier CPTP adjoint is unital and contracts the
operator norm. The evaluator can therefore stop when that bound fits the budget,
even without visiting the earlier circuit. Small expansions reaching the input
are evaluated directly on `|0>`.

The result includes `expectation_error_bounds` and `bounded_observables`. Bounds
include explicitly discarded small coefficients and a conservative floating-point
margin. They describe this preprocessing approximation, not additional MPS/MPO
truncation error, Monte Carlo uncertainty, or backend numerical error. Averaging
realizations averages their error bounds. The evaluator uses cone-local packed Pauli keys, skips operations disjoint from
all terms, and reuses local transfer coefficients within a request. Work is capped
at 1,024 terms, 32,768 coefficient visits per observable, and 262,144 per request;
unsupported operations or excessive work fall back to ordinary simulation.
Pure-state sampled errors do not activate this evaluator.

This is a sound alternative to a blind depth cutoff, **not an implementation of
the paper's polynomial-time polymer sampler**. [arXiv:2610.00548v2](https://arxiv.org/html/2610.00548v2)
uses geometric locality, a specified per-layer depolarizing model, cluster
expansions of log marginals, and Pauli-path truncation with error analysis. It does
not justify deleting arbitrary gates older than `p^-1 log(1/epsilon)`, nor does
generic Pauli noise damp every observable: dephasing preserves Z. Maestro's
one-qubit depolarizing convention has Pauli attenuation `1 - 4p/3`, whereas the
paper's replacement-by-maximally-mixed convention has attenuation `1 - p`.

## Optional independent shot sampling

`enable_sampling_reduction=True` enables exact decomposition of terminal
measurements into **disjoint structural past cones**, in Python and native JSON
execution. Overlapping cones remain together regardless of `max_cluster_qubits`.
Classical control, intermediate measurements, duplicate classical destinations,
unsupported trajectory resets, and distributed layouts retain normal execution.

Local samples preserve measurement destinations, unwritten classical bits and
per-qubit readout errors. Python keys retain at least the original register width (or a higher classical
destination); native JSON retains its declared classical width. Sparse backward
labels construct measurement components without full-width strings per measured
qubit. A hybrid join combines component histograms without storing shots
individually: large rows use hypergeometric allocation; small rows draw from the
remaining component counts without replacement using a Fenwick tree. A row of
`c` shots then needs `O(c log m)` tree work instead of scanning all `m` component
outcomes, after an `O(m)` tree build when entering the sparse path. Both paths preserve each histogram's marginal and the distribution of
independent random permutations; concatenating sorted histograms would introduce
false correlations. Memory scales with occupied outcomes. The option is off by
default because decomposition changes seeded shot sequences. The hybrid join can
also change counts for a seed used with the previous join, while remaining
reproducible for a fixed seed, configuration and implementation. Results
report `sampling_components` and `largest_component_qubits`.

No independent sampling of overlapping or merely weakly correlated components is
performed. Such an approximation needs a joint-distribution error bound that is
not supplied by local expectation bounds.

## Regression coverage

`tests/python/test_lightcone_report.py`, `tests/python/test_lightcone_update.py`,
`tests/LightconeUpdateTests.cpp`, the existing causal-cone/parity suites,
`tests/CausalConeTests.cpp`, and `tests/RequestApi.cpp` cover the confirmed report
issues, exhaustive small-register Pauli comparisons, extended gate/frame support,
wide bitset batches, cache invalidation, exact noisy channels, approximation
bounds and fallback, native integration, and sampling correlations/readout maps.

## API and linking compatibility

Native JSON estimation enables `auto_reduce` and causal-cone reduction by default;
set their simulator options to false to retain full-register estimation. Python's
legacy automatic backend selection is preserved. The fixed-backend checks in the
shared implementation apply to native fixed selection; `fixed_backend` and
`native_options` are not Python config fields. A request resolved entirely by
known constants allocates no backend and reports no cluster methods or GPU device.

Native consumers of `NoiseAdd` estimation must link Maestro, which owns the
shared preparation implementation. The legacy CMake `tests` target now links
the Maestro library artifact as well as Boost, without inheriting blanket Boost
DLL macros that conflict with its embedded Boost.Test implementation. `Estimation.h` forward-declares the network interface
instead of including the network implementation header.

Composer's CMake and Visual Studio builds include the shared estimation, sampling,
bounded-Pauli and observable-clustering sources in the Maestro library. Their
consumer targets link that library; Visual Studio project references also ensure
the library is built before the executables and Python extension.
