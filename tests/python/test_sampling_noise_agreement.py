"""The same noisy circuit gives the same statistics whichever way a method realizes the noise.

Maestro realizes a noise model in several ways:

- sampled trajectories: pure-state methods (statevector, MPS, tensor network,
  Pauli propagation, path integral, composite) run one injected circuit per
  noise realization;
- exact channels: the density matrix and the MPO evolve the ensemble once;
- Pauli frames: noisy Clifford circuits on the stabilizer method
  (reference_frame_v1), and batched frames on the extended stabilizer
  (extended_frame_v1).

For one circuit and one noise model every method must produce the same outcome
distribution and the same expectation values, up to sampling error. The QCSim
density matrix is the reference. Counts are compared as whole distributions
(two-sample chi-square) and through their averages, correlations and spread;
estimates through the realizations' standard errors.

Every call is seeded, so the comparisons are deterministic. Their bounds sit
about five standard errors out; the negative controls show that the same
checks reject noise 1.5 times stronger than the reference's.
"""

import math

import numpy as np
import pytest

import maestro
from maestro.circuits import QuantumCircuit

S = maestro.SimulationType
B = maestro.SimulatorType

SHOTS = 20000
REFERENCE_SHOTS = 40000
REALIZATIONS = 2000
# Standard errors a statistic may differ by. Each check makes about a dozen
# comparisons, so a correct method fails one with probability below 1e-5.
Z_BOUND = 5.0

OBSERVABLES = ["XIIX", "ZXIZ", "XYZI", "XXXX", "YXIY", "IIZI", "IZZX", "YYZZ"]


# ---------------------------------------------------------------- statistics

def _chi_square(a, b):
    """Two-sample chi-square homogeneity statistic and its degrees of freedom.

    Outcomes seen fewer than ten times in both samples together share one bin,
    which keeps the chi-square approximation valid for sparse tails.
    """
    na, nb = sum(a.values()), sum(b.values())
    ka, kb = math.sqrt(nb / na), math.sqrt(na / nb)
    bins, rare = [], [0, 0]
    for key in set(a) | set(b):
        x, y = a.get(key, 0), b.get(key, 0)
        if x + y < 10:
            rare[0] += x
            rare[1] += y
        else:
            bins.append((x, y))
    if sum(rare):
        bins.append(tuple(rare))
    statistic = sum((ka * x - kb * y) ** 2 / (x + y) for x, y in bins)
    return statistic, max(1, len(bins) - 1)


def _chi_square_bound(dof, z=Z_BOUND):
    """The chi-square quantile z standard normal deviations out (Wilson-Hilferty)."""
    h = 2.0 / (9.0 * dof)
    return dof * (1.0 - h + z * math.sqrt(h)) ** 3


def _summary(counts):
    """Averages, correlations and spread of a histogram, with their standard errors.

    Keys are bit strings. Returns {name: (value, standard error)}: the
    probability that each bit is one, every pair's <Z_i Z_j> correlation, and
    the mean and variance of the Hamming weight.
    """
    keys = sorted(counts)
    weights = np.array([counts[key] for key in keys], dtype=float)
    shots = weights.sum()
    bits = np.array([[int(bit) for bit in key] for key in keys], dtype=float)
    spins = 1.0 - 2.0 * bits
    width = bits.shape[1]
    stats = {}
    for i in range(width):
        p = weights @ bits[:, i] / shots
        stats[f"P(bit {i} = 1)"] = (p, math.sqrt(p * (1 - p) / shots))
    for i in range(width):
        for j in range(i + 1, width):
            c = weights @ (spins[:, i] * spins[:, j]) / shots
            stats[f"<Z{i}Z{j}>"] = (c, math.sqrt(max(1 - c * c, 0.0) / shots))
    weight = bits.sum(axis=1)
    mean = weights @ weight / shots
    variance = weights @ (weight - mean) ** 2 / shots
    fourth = weights @ (weight - mean) ** 4 / shots
    stats["weight mean"] = (mean, math.sqrt(variance / shots))
    stats["weight variance"] = (variance, math.sqrt(max(fourth - variance ** 2, 0.0) / shots))
    return stats


def _compare_counts(a, b):
    """Every way two histograms may differ, as (failures, report)."""
    assert {len(key) for key in a} == {len(key) for key in b}, "the methods returned different record widths"
    failures, report = [], []
    statistic, dof = _chi_square(a, b)
    bound = _chi_square_bound(dof)
    report.append(f"chi2 {statistic:.1f} (dof {dof}, bound {bound:.1f})")
    if statistic > bound:
        failures.append(f"distributions differ: chi2 {statistic:.1f} > {bound:.1f} (dof {dof})")
    sa, sb = _summary(a), _summary(b)
    for name in sa:
        (x, ex), (y, ey) = sa[name], sb[name]
        error = math.hypot(ex, ey)
        z = abs(x - y) / error if error > 0 else (0.0 if x == y else math.inf)
        report.append(f"{name}: {x:.4f} vs {y:.4f} (z {z:.1f})")
        if z > Z_BOUND:
            failures.append(f"{name}: {x:.4f} vs {y:.4f}, {z:.1f} standard errors apart")
    return failures, report


def _assert_same_counts(a, b, label):
    failures, report = _compare_counts(a, b)
    assert not failures, f"{label}:\n  " + "\n  ".join(failures) + "\nall statistics:\n  " + "\n  ".join(report)


def _assert_different_counts(a, b, label):
    failures, report = _compare_counts(a, b)
    assert failures, f"{label}: stronger noise passed every check, so the comparison has no power:\n  " + "\n  ".join(report)


def _assert_estimates_match_exact(result, exact, label, observables=OBSERVABLES):
    """Trajectory means lie within Z_BOUND realization standard errors of the exact values."""
    errors = result["realization_standard_errors"]
    assert errors is not None and len(errors) == len(observables)
    for name, value, error, expected in zip(observables, result["expectation_values"], errors, exact):
        assert abs(value - expected) <= Z_BOUND * error + 1e-9, (
            f"{label} <{name}>: {value:.5f} vs exact {expected:.5f}, standard error {error:.5f}")


def _assert_estimates_agree(a, b, label, observables=OBSERVABLES):
    """Two independent trajectory averages agree within their combined standard error."""
    for name, x, ex, y, ey in zip(observables, a["expectation_values"], a["realization_standard_errors"],
                                  b["expectation_values"], b["realization_standard_errors"]):
        assert abs(x - y) <= Z_BOUND * math.hypot(ex, ey) + 1e-9, (
            f"{label} <{name}>: {x:.5f} +- {ex:.5f} vs {y:.5f} +- {ey:.5f}")


# ---------------------------------------------------------------- circuits and models

def _circuit(measure=True, delays=False):
    """A generic four-qubit circuit: rotations, T gates and entanglers."""
    c = QuantumCircuit()
    c.h(0)
    c.ry(1, 0.7)
    c.rx(2, 1.1)
    c.h(3)
    c.cx(0, 1)
    c.t(1)
    c.cz(1, 2)
    c.s(2)
    c.rz(3, 0.5)
    if delays:
        c.delay(3, 3.0)
        c.delay(0, 1.5)
    c.cx(2, 3)
    c.ry(0, 0.4)
    c.cx(3, 0)
    c.h(1)
    c.t(2)
    if delays:
        c.delay(2, 2.0)
    if measure:
        c.measure_all()
    return c


def _clifford_circuit(dynamic=False):
    """A four-qubit Clifford circuit; dynamic adds a measured, reset and reused qubit.

    Noiseless, the records are (a, 1-a, 1-a, 1) for a random bit a; dynamic
    ones are (b, 1-a, 1-a, 1-b, a). Such correlated, partly deterministic
    records make every error visible. A uniform output distribution, which a
    generic Clifford circuit tends to give, would hide the noise strength.
    """
    c = QuantumCircuit()
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.x(3)
    c.s(0)
    c.cz(0, 3)
    c.sdg(0)
    c.cy(3, 1)
    c.y(2)
    c.s(1)
    if dynamic:
        c.measure([(0, 4)])
        c.reset(0)
        c.h(0)
        c.cx(0, 3)
    c.measure([(q, q) for q in range(4)])
    return c


# Mid-circuit measurement, classical control and a reused qubit.
DYNAMIC_QASM = """OPENQASM 3.0; qubit[3] q; bit[4] c;
h q[0]; t q[0]; cx q[0], q[1]; ry(0.8) q[2];
c[0] = measure q[0];
if (c[0]) { x q[2]; }
reset q[0]; h q[0]; cz q[0], q[2]; t q[2]; h q[2]; cx q[2], q[1];
c[1] = measure q[0]; c[2] = measure q[1]; c[3] = measure q[2];"""


def _dynamic_circuit():
    return maestro.QasmToCirc().parse_and_translate(DYNAMIC_QASM)


def _dynamic_model(scale=1.0):
    model = _pauli_model(scale, qubits=3)
    for q, t1 in enumerate([20.0, 30.0, 25.0]):
        model.set_thermal_relaxation(q, 1.0 * scale, t1, 0.8 * t1, 0.05)
    return model


def _pauli_model(scale=1.0, qubits=4):
    """Pauli channels of every kind a noise model offers, plus readout error."""
    m = maestro.NoiseModel()
    m.set_all_depolarizing(qubits, 0.02 * scale)
    m.set_dephasing(1, 0.03 * scale)
    m.set_bit_flip(2, 0.02 * scale)
    m.set_qubit_noise(0, 0.01 * scale, 0.005 * scale, 0.02 * scale)
    m.set_2q_depolarizing(0, 1, 0.05 * scale)
    m.set_all_2q_gate_depolarizing(qubits, 0.01 * scale)
    m.set_phase_damping(2, 0.1 * scale)
    m.set_readout_error(qubits - 1, 0.03 * scale, 0.06 * scale)
    m.set_readout_error(0, 0.02 * scale, 0.01 * scale)
    return m


def _relaxation_model(scale=1.0, qubits=4):
    """Thermal relaxation (T2 <= T1, where sampled resets are exact), idle noise on
    delays, crosstalk, depolarizing and readout error."""
    m = maestro.NoiseModel()
    times = [(20.0, 15.0, 0.0), (30.0, 30.0, 0.1), (25.0, 12.0, 0.05), (40.0, 20.0, 0.0)]
    for q, (t1, t2, excited) in enumerate(times[:qubits]):
        m.set_thermal_relaxation(q, 1.0 * scale, t1, t2, excited)
        m.set_thermal_relaxation_2q(q, 2.5 * scale, t1, t2, excited)
    m.set_idle_noise(qubits - 1, 30.0 / scale, 20.0 / scale, 0.0, detuning_hz=0.01)
    m.set_idle_noise(0, 25.0 / scale, 25.0 / scale, 0.1)
    m.set_crosstalk(0, 1, 0.08 * scale)
    m.set_all_depolarizing(qubits, 0.005 * scale)
    m.set_readout_error(1, 0.02 * scale, 0.05 * scale)
    return m


def _config(method, backend=B.QCSim, seed=1, **options):
    if method in (S.MatrixProductState, S.MatrixProductOperator):
        options.setdefault("max_bond_dimension", 64)  # exact at these widths
    return maestro.SimulatorConfig(simulator_type=backend, simulation_type=method, fixed_backend=True, seed=seed, **options)


def _seed(backend, method, base):
    """A seed of its own for each method, so that no two compared runs share realizations."""
    return base + 97 * int(backend) + 13 * int(method)


def _skip_missing(backend):
    """Optional backends: Aer when compiled in (None otherwise), QuEST and GPU plugins when loadable."""
    if backend is None:
        pytest.skip("Qiskit Aer is not compiled in")
    if backend == B.Gpu and not maestro.is_gpu_available():
        pytest.skip("the GPU simulator is unavailable")
    if backend == B.QuestSim and not maestro.is_quest_available():
        pytest.skip("the QuEST simulator is unavailable")


def _check_ran(result, backend, method, strategy=None):
    """The requested backend ran (no silent fallback), on the expected sampler."""
    assert result["simulator"] == backend, f"ran on {result['simulator']}, not {backend}"
    assert result["method"] == method, f"ran {result['method']}, not {method}"
    if strategy is not None:
        assert result.get("sampling_strategy") == strategy, result.get("sampling_strategy")


def _execute(circuit, model, entry, backend, method, shots, seed, exact, **options):
    """Counts from one method. Trajectories draw an independent realization per shot."""
    config = _config(method, backend, seed=seed, **options)
    realizations = None if exact else shots
    result = getattr(circuit, entry)(model, config=config, shots=shots, noise_realizations=realizations, noise_seed=seed + 1000)
    assert sum(result["counts"].values()) == shots
    if exact:
        assert result["noise_evaluation"] == "exact_channels" and result["noise_realizations"] == 1
    else:
        assert result["noise_realizations"] == shots
    return result


def _reference_counts(circuit, model, entry):
    """The density matrix's exact distribution, sampled."""
    result = _execute(circuit, model, entry, B.QCSim, S.DensityMatrix, REFERENCE_SHOTS, seed=4242, exact=True)
    _check_ran(result, B.QCSim, S.DensityMatrix)
    return dict(result["counts"])


# (backend, method, exact channels, shots, extra options, expected sampler)
SLOW_SHOTS = 5000
METHODS = [
    pytest.param(B.QCSim, S.Statevector, False, SHOTS, {}, None, id="qcsim-statevector"),
    pytest.param(B.QCSim, S.MatrixProductState, False, SHOTS, {}, None, id="qcsim-mps"),
    pytest.param(B.QCSim, S.MatrixProductOperator, True, SHOTS, {}, None, id="qcsim-mpo"),
    pytest.param(B.QCSim, S.TensorNetwork, False, SLOW_SHOTS, {}, None, id="qcsim-tensor-network"),
    pytest.param(B.QCSim, S.PauliPropagator, False, 2000, {}, None, id="qcsim-pauli-propagator"),
    pytest.param(B.QCSim, S.PathIntegral, False, SHOTS, {}, None, id="qcsim-path-integral"),
    pytest.param(B.QCSim, S.ExtendedStabilizer, False, SHOTS, {"counts_policy": "frame_v1"}, "extended_frame_v1",
                 id="qcsim-extended-stabilizer-frames"),
    pytest.param(B.QCSim, S.ExtendedStabilizer, False, SHOTS, {"counts_policy": "legacy"}, None, id="qcsim-extended-stabilizer"),
    pytest.param(B.CompositeQCSim, S.Statevector, False, SLOW_SHOTS, {}, None, id="composite-qcsim-statevector"),
    pytest.param(getattr(B, "QiskitAer", None), S.Statevector, False, SLOW_SHOTS, {}, None, id="aer-statevector"),
    pytest.param(getattr(B, "QiskitAer", None), S.MatrixProductState, False, SLOW_SHOTS, {}, None, id="aer-mps"),
    pytest.param(getattr(B, "QiskitAer", None), S.DensityMatrix, True, SHOTS, {}, None, id="aer-density-matrix"),
    pytest.param(B.QuestSim, S.Statevector, False, SLOW_SHOTS, {}, None, id="quest-statevector"),
    pytest.param(B.Gpu, S.Statevector, False, SLOW_SHOTS, {"precision": "double"}, None, id="gpu-statevector"),
    pytest.param(B.Gpu, S.MatrixProductState, False, SLOW_SHOTS, {"precision": "double"}, None, id="gpu-mps"),
    pytest.param(B.Gpu, S.DensityMatrix, True, SHOTS, {"precision": "double"}, None, id="gpu-density-matrix"),
    pytest.param(B.Gpu, S.MatrixProductOperator, True, SHOTS, {"precision": "double"}, None, id="gpu-mpo"),
    pytest.param(B.Gpu, S.TensorNetwork, False, SLOW_SHOTS, {"precision": "double"}, None, id="gpu-tensor-network"),
    pytest.param(B.Gpu, S.PauliPropagator, False, 2000, {"precision": "double"}, None, id="gpu-pauli-propagator"),
]


@pytest.fixture(scope="module")
def pauli_reference():
    return _reference_counts(_circuit(), _pauli_model(), "noisy_execute")


@pytest.fixture(scope="module")
def relaxation_reference():
    return _reference_counts(_circuit(delays=True), _relaxation_model(), "full_noise_execute")


# ---------------------------------------------------------------- counts

@pytest.mark.parametrize("backend,method,exact,shots,options,strategy", METHODS)
def test_pauli_noise_counts_match_density_matrix(pauli_reference, backend, method, exact, shots, options, strategy):
    _skip_missing(backend)
    result = _execute(_circuit(), _pauli_model(), "noisy_execute", backend, method, shots, seed=_seed(backend, method, 11), exact=exact,
                      **options)
    _check_ran(result, backend, method, strategy)
    _assert_same_counts(dict(result["counts"]), pauli_reference, f"{backend.name} {method.name} vs density matrix")


# The extended stabilizer's frames run per-shot Pauli and thermal noise, but not
# this model's delays and crosstalk: that case would rerun the scalar shots.
RELAXATION_METHODS = [param for param in METHODS if param.id != "qcsim-extended-stabilizer-frames"]


@pytest.mark.parametrize("backend,method,exact,shots,options,strategy", RELAXATION_METHODS)
def test_relaxation_noise_counts_match_density_matrix(relaxation_reference, backend, method, exact, shots, options, strategy):
    """Thermal relaxation runs as sampled resets on trajectories and as exact channels on mixed states."""
    _skip_missing(backend)
    result = _execute(_circuit(delays=True), _relaxation_model(), "full_noise_execute", backend, method, shots, seed=_seed(backend, method, 13),
                      exact=exact, **options)
    _check_ran(result, backend, method)
    _assert_same_counts(dict(result["counts"]), relaxation_reference, f"{backend.name} {method.name} vs density matrix")


# Methods that execute mid-circuit measurements, resets and classical control.
DYNAMIC_METHODS = [
    pytest.param(B.QCSim, S.Statevector, False, id="qcsim-statevector"),
    pytest.param(B.QCSim, S.MatrixProductState, False, id="qcsim-mps"),
    pytest.param(B.QCSim, S.MatrixProductOperator, True, id="qcsim-mpo"),
    pytest.param(B.QCSim, S.ExtendedStabilizer, False, id="qcsim-extended-stabilizer"),
    pytest.param(getattr(B, "QiskitAer", None), S.Statevector, False, id="aer-statevector"),
    pytest.param(getattr(B, "QiskitAer", None), S.DensityMatrix, True, id="aer-density-matrix"),
]


@pytest.mark.parametrize("backend,method,exact", DYNAMIC_METHODS)
def test_dynamic_circuit_counts_match_density_matrix(backend, method, exact):
    """Each shot follows its own measured branch: sampled on trajectories, exactly on mixed states."""
    _skip_missing(backend)
    model = _dynamic_model()
    reference = _reference_counts(_dynamic_circuit(), model, "full_noise_execute")
    shots = SHOTS if backend == B.QCSim else SLOW_SHOTS
    result = _execute(_dynamic_circuit(), model, "full_noise_execute", backend, method, shots, seed=_seed(backend, method, 17),
                      exact=exact, counts_policy="legacy")
    _check_ran(result, backend, method)
    _assert_same_counts(dict(result["counts"]), reference, f"{backend.name} {method.name} dynamic circuit vs density matrix")


# Clifford circuits: the stabilizer's Pauli frames and its own trajectories, the
# extended stabilizer's frames, and the other families, against the density matrix.
CLIFFORD_METHODS = [
    pytest.param(S.Stabilizer, False, "frame_v1", "reference_frame_v1", id="stabilizer-frames"),
    pytest.param(S.Stabilizer, False, "legacy", None, id="stabilizer-trajectories"),
    pytest.param(S.ExtendedStabilizer, False, "frame_v1", "extended_frame_v1", id="extended-stabilizer-frames"),
    pytest.param(S.Statevector, False, "legacy", None, id="statevector"),
    pytest.param(S.MatrixProductState, False, "legacy", None, id="mps"),
    pytest.param(S.MatrixProductOperator, True, "legacy", None, id="mpo"),
]


@pytest.mark.parametrize("dynamic", [False, True], ids=["terminal", "dynamic"])
@pytest.mark.parametrize("method,exact,policy,strategy", CLIFFORD_METHODS)
def test_clifford_counts_match_density_matrix(method, exact, policy, strategy, dynamic):
    circuit = _clifford_circuit(dynamic)
    reference = _reference_counts(circuit, _pauli_model(), "noisy_execute")
    seed = _seed(B.QCSim, method, 19 + (5 if policy == "legacy" else 0))
    result = _execute(circuit, _pauli_model(), "noisy_execute", B.QCSim, method, SHOTS, seed=seed, exact=exact, counts_policy=policy)
    _check_ran(result, B.QCSim, method, strategy)
    if strategy is None:
        assert result.get("sampling_strategy") not in ("reference_frame_v1", "extended_frame_v1")
    _assert_same_counts(dict(result["counts"]), reference, f"{method.name} ({policy}) Clifford circuit vs density matrix")


@pytest.mark.parametrize("dynamic", [False, True], ids=["terminal", "dynamic"])
def test_stabilizer_frames_match_stabilizer_trajectories(dynamic):
    """The two stabilizer samplers, against each other rather than a third method."""
    circuit = _clifford_circuit(dynamic)
    frames = _execute(circuit, _pauli_model(), "noisy_execute", B.QCSim, S.Stabilizer, SHOTS, seed=23, exact=False, counts_policy="frame_v1")
    scalar = _execute(circuit, _pauli_model(), "noisy_execute", B.QCSim, S.Stabilizer, SHOTS, seed=29, exact=False, counts_policy="legacy")
    _check_ran(frames, B.QCSim, S.Stabilizer, "reference_frame_v1")
    assert scalar.get("sampling_strategy") != "reference_frame_v1"
    _assert_same_counts(dict(frames["counts"]), dict(scalar["counts"]), "stabilizer frames vs stabilizer trajectories")


# ---------------------------------------------------------------- estimates

@pytest.fixture(scope="module")
def exact_estimates():
    """Density-matrix expectation values: one exact evaluation per model."""
    config = _config(S.DensityMatrix)
    values = {}
    for name, circuit, model, entry in [("pauli", _circuit(False), _pauli_model(), "noisy_estimate_montecarlo"),
                                        ("relaxation", _circuit(False, delays=True), _relaxation_model(), "full_noise_estimate")]:
        result = getattr(circuit, entry)(OBSERVABLES, model, None, config, noise_seed=5)
        assert result["noise_realizations"] == 1 and result["estimate_evaluation"] == "deterministic"
        values[name] = result["expectation_values"]
    return values


ESTIMATE_METHODS = [
    pytest.param(B.QCSim, S.Statevector, False, REALIZATIONS, id="qcsim-statevector"),
    pytest.param(B.QCSim, S.MatrixProductState, False, REALIZATIONS, id="qcsim-mps"),
    pytest.param(B.QCSim, S.MatrixProductOperator, True, None, id="qcsim-mpo"),
    pytest.param(B.QCSim, S.TensorNetwork, False, REALIZATIONS, id="qcsim-tensor-network"),
    pytest.param(B.QCSim, S.PauliPropagator, False, 500, id="qcsim-pauli-propagator"),
    pytest.param(B.QCSim, S.PathIntegral, False, REALIZATIONS, id="qcsim-path-integral"),
    pytest.param(B.QCSim, S.ExtendedStabilizer, False, REALIZATIONS, id="qcsim-extended-stabilizer"),
    pytest.param(B.CompositeQCSim, S.Statevector, False, REALIZATIONS, id="composite-qcsim-statevector"),
    pytest.param(getattr(B, "QiskitAer", None), S.Statevector, False, REALIZATIONS, id="aer-statevector"),
    pytest.param(getattr(B, "QiskitAer", None), S.MatrixProductState, False, REALIZATIONS, id="aer-mps"),
    pytest.param(getattr(B, "QiskitAer", None), S.DensityMatrix, True, None, id="aer-density-matrix"),
    pytest.param(B.QuestSim, S.Statevector, False, REALIZATIONS, id="quest-statevector"),
    pytest.param(B.Gpu, S.Statevector, False, REALIZATIONS, id="gpu-statevector"),
    pytest.param(B.Gpu, S.MatrixProductState, False, REALIZATIONS, id="gpu-mps"),
    pytest.param(B.Gpu, S.DensityMatrix, True, None, id="gpu-density-matrix"),
    pytest.param(B.Gpu, S.MatrixProductOperator, True, None, id="gpu-mpo"),
    pytest.param(B.Gpu, S.TensorNetwork, False, REALIZATIONS, id="gpu-tensor-network"),
    pytest.param(B.Gpu, S.PauliPropagator, False, 500, id="gpu-pauli-propagator"),
]


@pytest.mark.parametrize("model_name", ["pauli", "relaxation"])
@pytest.mark.parametrize("backend,method,exact,realizations", ESTIMATE_METHODS)
def test_estimates_match_density_matrix(exact_estimates, backend, method, exact, realizations, model_name):
    _skip_missing(backend)
    options = {"precision": "double"} if backend == B.Gpu else {}
    seed = _seed(backend, method, 31)
    config = _config(method, backend, seed=seed, **options)
    if model_name == "pauli":
        result = _circuit(False).noisy_estimate_montecarlo(OBSERVABLES, _pauli_model(), realizations, config, noise_seed=seed + 1000)
    else:
        result = _circuit(False, delays=True).full_noise_estimate(OBSERVABLES, _relaxation_model(), realizations, config, noise_seed=seed + 1000)
    _check_ran(result, backend, method)
    label = f"{backend.name} {method.name} {model_name} estimate"
    if exact:
        assert result["noise_realizations"] == 1
        assert result["expectation_values"] == pytest.approx(exact_estimates[model_name], abs=1e-8), label
    else:
        assert result["noise_realizations"] == realizations
        _assert_estimates_match_exact(result, exact_estimates[model_name], label)


PURE_STATE_METHODS = [S.Statevector, S.MatrixProductState, S.TensorNetwork, S.PauliPropagator, S.PathIntegral, S.ExtendedStabilizer]


def test_trajectory_methods_agree_realization_by_realization():
    """Pauli noise injects the same circuits for every pure-state method under one noise seed.

    Each realization is then a noiseless simulation, so the averages must agree
    to rounding: a difference is a simulator error on a particular injected
    circuit, not sampling noise.
    """
    results = {}
    for method in PURE_STATE_METHODS:
        config = _config(method, seed=41)
        results[method] = _circuit(False).noisy_estimate_montecarlo(OBSERVABLES, _pauli_model(), 300, config, noise_seed=43)
    reference = results[S.Statevector]
    for method, result in results.items():
        assert result["expectation_values"] == pytest.approx(reference["expectation_values"], abs=1e-9), method.name
        assert result["realization_standard_errors"] == pytest.approx(reference["realization_standard_errors"], abs=1e-9), method.name


DISORDER_METHODS = [S.Statevector, S.MatrixProductState, S.MatrixProductOperator, S.DensityMatrix]


def _disorder_model(layer):
    """Noise sampled per realization on every method, plus exact-able Pauli noise."""
    m = maestro.NoiseModel()
    if layer == "coherent":
        m.set_coherent_rotation(0, 0.25, 0.0, 0.15)
        m.set_coherent_rotation(2, 0.0, 0.3, 0.0)
        m.set_all_coherent_depolarizing(4, 0.01)
    else:
        m.set_all_correlated_ou(4, 0.6, 0.5, 1.0)
        m.set_correlated_ou(1, 1.0, 0.2, 1.0)
    m.set_all_depolarizing(4, 0.01)
    m.set_dephasing(3, 0.02)
    return m


@pytest.mark.parametrize("layer", ["coherent", "correlated"])
def test_disorder_averaged_noise_agrees_across_methods(layer):
    """Coherent and time-correlated noise stay sampled even on the density matrix and MPO.

    Their realization averages are then compared pairwise, each method with its
    own noise seed, so every pair is an independent statistical comparison.
    """
    realizations = {S.Statevector: REALIZATIONS, S.MatrixProductState: REALIZATIONS, S.MatrixProductOperator: 500, S.DensityMatrix: 500}
    results = {}
    for index, method in enumerate(DISORDER_METHODS):
        config = _config(method, seed=51 + index)
        result = _circuit(False).full_noise_estimate(OBSERVABLES, _disorder_model(layer), realizations[method], config, noise_seed=61 + index)
        assert result["noise_realizations"] == realizations[method]
        assert all(error > 0 for error in result["realization_standard_errors"]), "the noise was not sampled"
        results[method] = result
    for i, a in enumerate(DISORDER_METHODS):
        for b in DISORDER_METHODS[i + 1:]:
            _assert_estimates_agree(results[a], results[b], f"{layer}: {a.name} vs {b.name}")


# ---------------------------------------------------------------- power and known differences

# (circuit, model with a noise scale, entry, method, counts policy, expected sampler)
NEGATIVE_CONTROLS = {
    "pauli": (lambda: _circuit(), _pauli_model, "noisy_execute", S.Statevector, "legacy", None),
    "relaxation": (lambda: _circuit(delays=True), _relaxation_model, "full_noise_execute", S.Statevector, "legacy", None),
    "dynamic": (_dynamic_circuit, _dynamic_model, "full_noise_execute", S.Statevector, "legacy", None),
    "clifford-frames": (lambda: _clifford_circuit(), _pauli_model, "noisy_execute", S.Stabilizer, "frame_v1", "reference_frame_v1"),
    "clifford-frames-dynamic": (lambda: _clifford_circuit(True), _pauli_model, "noisy_execute", S.Stabilizer, "frame_v1", "reference_frame_v1"),
}


@pytest.mark.parametrize("case", list(NEGATIVE_CONTROLS))
def test_count_comparison_rejects_stronger_noise(case):
    """Negative controls: each count comparison above detects 1.5x noise at the same shot counts."""
    circuit, model, entry, method, policy, strategy = NEGATIVE_CONTROLS[case]
    reference = _reference_counts(circuit(), model(), entry)
    stronger = _execute(circuit(), model(1.5), entry, B.QCSim, method, SHOTS, seed=71, exact=False, counts_policy=policy)
    _check_ran(stronger, B.QCSim, method, strategy)
    _assert_different_counts(dict(stronger["counts"]), reference, f"{case}: {method.name} with 1.5x noise")


def test_estimate_comparison_rejects_stronger_noise(exact_estimates):
    config = _config(S.Statevector, seed=83)
    stronger = _circuit(False).noisy_estimate_montecarlo(OBSERVABLES, _pauli_model(1.5), REALIZATIONS, config, noise_seed=89)
    with pytest.raises(AssertionError):
        _assert_estimates_match_exact(stronger, exact_estimates["pauli"], "statevector with 1.5x Pauli noise")


def test_set_t1_differs_from_exact_damping_only_in_coherence():
    """A documented difference, pinned so it cannot change silently.

    set_t1 is exact amplitude damping on mixed states but a reset with
    probability gamma on trajectories: the populations agree, the coherence is
    (1 - gamma) instead of sqrt(1 - gamma). The agreement tests above therefore
    use set_thermal_relaxation, which is consistent on both families.
    """
    gamma = 0.3
    model = maestro.NoiseModel()
    model.set_t1(0, gamma)
    circuit = QuantumCircuit()
    circuit.h(0)
    exact = circuit.noisy_estimate_montecarlo(["Z", "X"], model, None, _config(S.DensityMatrix))["expectation_values"]
    assert exact == pytest.approx([gamma, math.sqrt(1 - gamma)], abs=1e-10)
    for method in (S.Statevector, S.MatrixProductState):
        sampled = circuit.noisy_estimate_montecarlo(["Z", "X"], model, 4000, _config(method, seed=97), noise_seed=101)
        (z, x), (ez, ex) = sampled["expectation_values"], sampled["realization_standard_errors"]
        assert abs(z - gamma) <= Z_BOUND * ez, method.name
        assert abs(x - (1 - gamma)) <= Z_BOUND * ex, method.name
        assert x < math.sqrt(1 - gamma) - Z_BOUND * ex, method.name
