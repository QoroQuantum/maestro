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
import random
import warnings

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


def _propagates_exactly(backend, method):
    """The QCSim Pauli propagator applies channels to the observable it propagates.

    Channels diagonal in the Pauli basis (the Pauli model) are then exact in one
    realization by default. Relaxation keeps its sampled trajectories there.
    """
    return backend == B.QCSim and method == S.PauliPropagator


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
    exact = exact or _propagates_exactly(backend, method)
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
    if model_name == "pauli" and _propagates_exactly(backend, method):
        exact, realizations = True, None
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


# The Pauli propagator is absent: it evaluates this Pauli model exactly rather
# than sampling it (test_pauli_propagator_noise_evaluation). Its trajectory
# path is compared realization by realization in the C++ suite, where
# NoiseAdd.set_exact_channels(False) selects it.
PURE_STATE_METHODS = [S.Statevector, S.MatrixProductState, S.TensorNetwork, S.PathIntegral, S.ExtendedStabilizer]


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


def test_pauli_propagator_noise_evaluation(exact_estimates):
    """Which models the QCSim Pauli propagator evaluates exactly by default."""
    config = _config(S.PauliPropagator, seed=47)
    # Channels diagonal in the Pauli basis: one exact realization.
    result = _circuit(False).noisy_estimate_montecarlo(OBSERVABLES, _pauli_model(), None, config, noise_seed=49)
    assert result["noise_evaluation"] == "exact_channels" and result["noise_realizations"] == 1
    assert result["estimate_evaluation"] == "deterministic"
    assert result["expectation_values"] == pytest.approx(exact_estimates["pauli"], abs=1e-8)
    # An explicit count is honored and changes nothing: every realization is the ensemble.
    repeated = _circuit(False).noisy_estimate_montecarlo(OBSERVABLES, _pauli_model(), 3, config, noise_seed=49)
    assert repeated["noise_realizations"] == 3
    assert repeated["expectation_values"] == pytest.approx(result["expectation_values"], abs=1e-12)
    # Relaxation also sends Z to the identity: it keeps its sampled realizations.
    result = _circuit(False, delays=True).full_noise_estimate(OBSERVABLES, _relaxation_model(), None, config, noise_seed=49)
    assert result["noise_evaluation"] == "trajectories" and result["noise_realizations"] == 100
    # Channels with no sampled form, rejected on this method before, run exactly.
    damping = maestro.NoiseModel()
    damping.set_generalized_amplitude_damping(0, 0.2, 0.1)
    damping.set_correlated_phase_flip(1, 2, 0.05, 0.5)
    result = _circuit(False).full_noise_estimate(OBSERVABLES, damping, None, config, noise_seed=49)
    assert result["noise_evaluation"] == "exact_channels" and result["noise_realizations"] == 1
    reference = _circuit(False).full_noise_estimate(OBSERVABLES, damping, None, _config(S.DensityMatrix), noise_seed=49)
    assert result["expectation_values"] == pytest.approx(reference["expectation_values"], abs=1e-8)
    # Terminal counts come from the one noisy circuit as well.
    counts = _circuit().noisy_execute(_pauli_model(), config=config, shots=500, noise_seed=49)
    assert counts["noise_evaluation"] == "exact_channels" and counts["noise_realizations"] == 1


def test_pauli_propagator_judges_relaxation_by_its_channel():
    """Relaxation that only scales Pauli terms, or never acts on the circuit, is exact by default."""
    config = _config(S.PauliPropagator, seed=47)
    dense = _config(S.DensityMatrix)

    def estimate(model, circuit, on=config):
        return circuit.full_noise_estimate(OBSERVABLES, model, None, on, noise_seed=49)

    # Equal populations make thermal and idle relaxation unital. T2 above T1 is
    # kept: sampled relaxation would clamp it to T1 and warn.
    unital = maestro.NoiseModel()
    for q in range(4):
        unital.set_thermal_relaxation(q, 1.0, 20.0, 30.0, 0.5)
        unital.set_thermal_relaxation_2q(q, 2.5, 20.0, 30.0, 0.5)
    unital.set_idle_noise(3, 30.0, 40.0, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = estimate(unital, _circuit(False, delays=True))
    assert result["noise_evaluation"] == "exact_channels" and result["noise_realizations"] == 1
    assert result["estimate_evaluation"] == "deterministic"
    reference = estimate(unital, _circuit(False, delays=True), dense)
    assert result["expectation_values"] == pytest.approx(reference["expectation_values"], abs=1e-8)
    # Unequal populations send Z to the identity: sampled again, T2 clamped.
    biased = maestro.NoiseModel()
    biased.set_thermal_relaxation(0, 1.0, 20.0, 30.0, 0.4)
    with pytest.warns(RuntimeWarning, match="T2 clamped to T1"):
        result = estimate(biased, _circuit(False))
    assert result["noise_evaluation"] == "trajectories" and result["noise_realizations"] == 100

    # Relaxation on a qubit the circuit never gates, a zero T1 probability and
    # idle relaxation without a delay add no channel at all.
    spectator = maestro.NoiseModel()
    spectator.set_all_depolarizing(4, 0.01)
    spectator.set_t1(7, 0.2)
    spectator.set_t1(0, 0.0)
    spectator.set_idle_noise(1, 25.0, 25.0, 0.1)
    result = estimate(spectator, _circuit(False))
    assert result["noise_evaluation"] == "exact_channels" and result["noise_realizations"] == 1
    reference = estimate(spectator, _circuit(False), dense)
    assert result["expectation_values"] == pytest.approx(reference["expectation_values"], abs=1e-8)
    # A delay on the idle qubit draws its relaxation: trajectories again.
    waiting = _circuit(False)
    waiting.delay(1, 2.0)
    result = estimate(spectator, waiting)
    assert result["noise_evaluation"] == "trajectories" and result["noise_realizations"] == 100
    # Counts follow the same rule.
    counts = _circuit().noisy_execute(spectator, config=config, shots=500, noise_seed=49)
    assert counts["noise_evaluation"] == "exact_channels" and counts["noise_realizations"] == 1


@pytest.mark.parametrize("exact", [S.PauliPropagator, S.DensityMatrix])
def test_thermal_clamp_warning_follows_the_backend_a_candidate_list_selects(exact):
    """noisy_fidelity and a checkpoint resolve a candidate list first: they warn only if that backend samples."""
    circuit = QuantumCircuit()
    circuit.h(0)
    suffix = QuantumCircuit()
    suffix.h(0)
    suffix.measure_all()
    # T2 above T1, which sampled relaxation clamps. Equal populations keep the
    # channel diagonal, so the propagator applies it exactly by default too.
    model = maestro.NoiseModel()
    model.set_thermal_relaxation(0, 1.0, 1.0, 2.0, 0.5)

    def config(first, second):
        return maestro.SimulatorConfig(optimization_candidates=[(B.QCSim, first), (B.QCSim, second)], seed=11)

    def run_suffix(candidates):
        return maestro.PrefixCheckpointedSimulator(QuantumCircuit(), 1, candidates).execute_suffix(suffix, shots=200, noise_model=model, noise_seed=5)

    # The exact backend is listed first and runs: no clamp, so no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fidelity = circuit.noisy_fidelity(model, config=config(exact, S.Statevector), noise_seed=5)
        counts = run_suffix(config(exact, S.Statevector))
    assert fidelity["noise_evaluation"] == counts["noise_evaluation"] == "exact_channels"
    assert fidelity["fidelity"] == pytest.approx((1 + math.exp(-0.5)) / 2, abs=1e-9)
    assert counts["method"] == exact
    # The statevector is listed first. It samples the relaxation channel from its
    # state, so nothing is clamped and nothing is said: its trajectories average
    # to the calibrated fidelity, where the clamp would give (1 + exp(-1)) / 2.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fidelity = circuit.noisy_fidelity(model, noise_realizations=REALIZATIONS, config=config(S.Statevector, exact), noise_seed=5)
        counts = run_suffix(config(S.Statevector, exact))
    assert fidelity["noise_evaluation"] == counts["noise_evaluation"] == "trajectories" and counts["method"] == S.Statevector
    assert abs(fidelity["fidelity"] - (1 + math.exp(-0.5)) / 2) <= Z_BOUND * fidelity["std_error"] + 1e-9
    # A method that neither holds nor samples channels is listed first: it draws
    # the relaxation as a reset and a phase flip, with T2 clamped, and warns.
    with pytest.warns(RuntimeWarning, match="T2 clamped to T1"):
        fidelity = circuit.noisy_fidelity(model, noise_realizations=REALIZATIONS, config=config(S.TensorNetwork, exact), noise_seed=5)
    assert fidelity["noise_evaluation"] == "trajectories"
    assert abs(fidelity["fidelity"] - (1 + math.exp(-1.0)) / 2) <= Z_BOUND * fidelity["std_error"] + 1e-9
    with pytest.warns(RuntimeWarning, match="T2 clamped to T1"):
        counts = run_suffix(config(S.Stabilizer, exact))
    assert counts["noise_evaluation"] == "trajectories" and counts["method"] == S.Stabilizer


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

    set_t1 is exact amplitude damping on mixed states and on the backends that
    sample channels from their state (the statevector, the matrix product
    state). On the other trajectory backends it is a reset with probability
    gamma: the populations agree, the coherence is (1 - gamma) instead of
    sqrt(1 - gamma). The agreement tests above therefore use
    set_thermal_relaxation, which is consistent everywhere.
    """
    gamma = 0.3
    model = maestro.NoiseModel()
    model.set_t1(0, gamma)
    circuit = QuantumCircuit()
    circuit.h(0)
    exact = circuit.noisy_estimate_montecarlo(["Z", "X"], model, None, _config(S.DensityMatrix))["expectation_values"]
    assert exact == pytest.approx([gamma, math.sqrt(1 - gamma)], abs=1e-10)
    for method in (S.Statevector, S.MatrixProductState, S.TensorNetwork):
        sampled = circuit.noisy_estimate_montecarlo(["Z", "X"], model, 4000, _config(method, seed=97), noise_seed=101)
        (z, x), (ez, ex) = sampled["expectation_values"], sampled["realization_standard_errors"]
        assert abs(z - gamma) <= Z_BOUND * ez, method.name
        if method != S.TensorNetwork:
            assert abs(x - math.sqrt(1 - gamma)) <= Z_BOUND * ex, method.name
        else:
            assert abs(x - (1 - gamma)) <= Z_BOUND * ex, method.name
            assert x < math.sqrt(1 - gamma) - Z_BOUND * ex, method.name


# ---------------------------------------------------------------- channels sampled from the state

def _channel_model(scale=1.0, qubits=4):
    """Noise with no exact sampled lowering: amplitude damping, relaxation with
    T1 < T2 <= 2*T1 after gates and on delays, generalized amplitude damping and
    a complex Kraus map, with depolarizing and readout error on top."""
    m = maestro.NoiseModel()
    m.set_t1(0, 0.06 * scale)
    m.set_thermal_relaxation(1, 1.0 * scale, 20.0, 32.0)
    m.set_thermal_relaxation_2q(1, 2.5 * scale, 20.0, 32.0, 0.1)
    m.set_generalized_amplitude_damping(2, 0.08 * scale, 0.2)
    gamma = 0.07 * scale
    m.set_kraus_channel([3], [[[1.0, 0.0], [0.0, 1j * math.sqrt(1.0 - gamma)]], [[0.0, math.sqrt(gamma)], [0.0, 0.0]]])
    m.set_idle_noise(qubits - 1, 30.0 / scale, 50.0 / scale)
    m.set_all_depolarizing(qubits, 0.005 * scale)
    m.set_readout_error(1, 0.02 * scale, 0.05 * scale)
    return m


def _channel_circuit():
    """The generic circuit with channel instructions of its own: a gate error made
    of a Pauli mixture and relaxation, on reversed targets, and amplitude damping."""
    Channel = maestro.QuantumChannel
    gate_error = (Channel.depolarizing(0.04, num_qubits=2)
                  .then(Channel.thermal_relaxation(1.0, 20.0, 34.0), [0])
                  .then(Channel.thermal_relaxation(1.0, 25.0, 18.0, 0.1), [1]))
    c = QuantumCircuit()
    c.h(0)
    c.ry(1, 0.7)
    c.rx(2, 1.1)
    c.h(3)
    c.cx(0, 1)
    c.channel([1, 0], gate_error)
    c.t(1)
    c.cz(1, 2)
    c.channel([2], Channel.amplitude_damping(0.15))
    c.rz(3, 0.5)
    c.cx(2, 3)
    c.channel([2, 3], gate_error)
    c.ry(0, 0.4)
    c.cx(3, 0)
    c.h(1)
    c.measure_all()
    return c


# With gate fusion (off by default on registers this small) the statevector
# resolves whole blocks of gates and channels from one reduced density matrix.
SAMPLING_BACKENDS = [pytest.param(B.QCSim, S.Statevector, {}, id="qcsim-statevector"),
                     pytest.param(B.QCSim, S.Statevector, {"gate_fusion": True}, id="qcsim-statevector-fused"),
                     pytest.param(B.QCSim, S.MatrixProductState, {}, id="qcsim-mps"),
                     pytest.param(B.CompositeQCSim, S.Statevector, {}, id="composite-qcsim-statevector"),
                     pytest.param(getattr(B, "QiskitAer", None), S.Statevector, {}, id="aer-statevector"),
                     pytest.param(getattr(B, "QiskitAer", None), S.MatrixProductState, {}, id="aer-mps")]


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_state_sampled_channels_match_the_density_matrix(backend, method, options):
    """A backend that samples channels from its state runs the noise that has no
    exact sampled form, one trajectory per shot, without approximating it."""
    _skip_missing(backend)
    circuit, model = _circuit(delays=True), _channel_model()
    reference = _reference_counts(circuit, model, "full_noise_execute")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no thermal approximation is announced, because none is made
        result = _execute(circuit, model, "full_noise_execute", backend, method, SHOTS, seed=_seed(backend, method, 31), exact=False,
                          **options)
    _check_ran(result, backend, method)
    assert result["noise_evaluation"] == "trajectories"
    _assert_same_counts(result["counts"], reference, f"{backend} {method} state-sampled channels")

    # The same checks tell stronger noise apart.
    stronger = _execute(circuit, _channel_model(1.5), "full_noise_execute", backend, method, SHOTS, seed=_seed(backend, method, 32),
                        exact=False, **options)
    _assert_different_counts(stronger["counts"], reference, f"{backend} {method} state-sampled channels")


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_state_sampled_circuit_channels_match_the_density_matrix(backend, method, options):
    """Channel instructions in the circuit itself, under an empty noise model."""
    _skip_missing(backend)
    circuit, model = _channel_circuit(), maestro.NoiseModel()
    reference = _reference_counts(circuit, model, "full_noise_execute")
    result = _execute(circuit, model, "full_noise_execute", backend, method, SHOTS, seed=_seed(backend, method, 33), exact=False, **options)
    _check_ran(result, backend, method)
    _assert_same_counts(result["counts"], reference, f"{backend} {method} circuit channels")


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_state_sampled_channel_estimates_match_the_density_matrix(backend, method, options):
    _skip_missing(backend)
    circuit, model = _circuit(measure=False, delays=True), _channel_model()
    exact = maestro.full_noise_estimate(circuit, OBSERVABLES, model, config=_config(S.DensityMatrix))
    assert exact["noise_evaluation"] == "exact_channels"
    result = maestro.full_noise_estimate(circuit, OBSERVABLES, model, noise_realizations=REALIZATIONS,
                                         config=_config(method, backend, seed=_seed(backend, method, 34), **options), noise_seed=91)
    assert result["noise_evaluation"] == "trajectories"
    _assert_estimates_match_exact(result, exact["expectation_values"], f"{backend} {method} state-sampled estimates")


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_state_sampled_channels_replay_and_share_realizations(backend, method, options):
    """Equal seeds give equal counts. The backend draws all the noise as each shot
    runs, so every shot is a realization of its own whatever number is asked for."""
    _skip_missing(backend)
    circuit, model = _circuit(delays=True), _channel_model()

    def run(realizations, seed=17):
        return circuit.full_noise_execute(model, config=_config(method, backend, seed=5, **options), shots=600,
                                          noise_realizations=realizations, noise_seed=seed)

    one_per_shot = run(600)
    assert one_per_shot["noise_realizations"] == 600
    assert one_per_shot["counts"] == run(600)["counts"]
    assert one_per_shot["counts"] != run(600, seed=18)["counts"]
    for requested in (8, None):
        shared = run(requested)
        assert shared["noise_realizations"] == 600 and sum(shared["counts"].values()) == 600
        assert shared["counts"] == one_per_shot["counts"]


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_state_sampled_channels_in_fidelity_and_checkpoint_suffixes(backend, method, options):
    """noisy_fidelity and a checkpoint suffix take the same noise the same way: every
    realization keeps the channels no draw replaces, and the backend draws their
    branches as it runs. Nothing is clamped, so nothing is announced."""
    _skip_missing(backend)
    model = _channel_model()
    circuit = _circuit(measure=False, delays=True)
    exact = circuit.noisy_fidelity(model, config=_config(S.DensityMatrix))
    assert exact["noise_evaluation"] == "exact_channels"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sampled = circuit.noisy_fidelity(model, noise_realizations=REALIZATIONS,
                                         config=_config(method, backend, seed=_seed(backend, method, 35), **options), noise_seed=95)
    assert sampled["noise_evaluation"] == "trajectories"
    assert abs(sampled["fidelity"] - exact["fidelity"]) <= Z_BOUND * sampled["std_error"] + 1e-9, (
        f"{backend} {method}: fidelity {sampled['fidelity']:.5f} vs exact {exact['fidelity']:.5f}, standard error {sampled['std_error']:.5f}")

    # A suffix with channel instructions of its own, under the model, after a saved prefix.
    prefix = QuantumCircuit()
    prefix.ry(0, 0.3)
    prefix.cx(0, 2)
    prefix.rx(3, 0.9)
    suffix = _channel_circuit()

    def run(run_method, run_backend, seed, realizations=None, **run_options):
        checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 4, _config(run_method, run_backend, seed=seed, **run_options))
        return checkpoint.execute_suffix(suffix, shots=SHOTS, noise_model=model, noise_realizations=realizations, noise_seed=97)

    reference = run(S.DensityMatrix, B.QCSim, 3)
    assert reference["noise_evaluation"] == "exact_channels"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # One realization per shot: the shots of a realization share its depolarizing errors.
        result = run(method, backend, _seed(backend, method, 37), SHOTS, **options)
    assert result["noise_evaluation"] == "trajectories" and result["method"] == method
    _assert_same_counts(result["counts"], reference["counts"], f"{backend} {method} checkpoint suffix")


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_either_seed_changes_noise_the_backend_draws_as_the_circuit_runs(backend, method, options):
    """Amplitude damping alone is kept as channels the backend samples from its state:
    injection draws nothing, so the noise seed has no realization to draw and joins the
    simulator seed. With one of them fixed the other still changes the errors drawn, in
    noisy_fidelity as in an estimate and in counts, and the same two repeat the result."""
    _skip_missing(backend)
    model = maestro.NoiseModel()
    for qubit in range(4):
        model.set_generalized_amplitude_damping(qubit, 0.2, 0.0)

    def fidelity(seed, noise_seed):
        return _circuit(measure=False).noisy_fidelity(model, noise_realizations=20, config=_config(method, backend, seed=seed, **options),
                                                      noise_seed=noise_seed)["fidelity"]

    def estimate(seed, noise_seed):
        return tuple(maestro.noisy_estimate_montecarlo(_circuit(measure=False), OBSERVABLES[:2], model, noise_realizations=20,
                                                       config=_config(method, backend, seed=seed, **options),
                                                       noise_seed=noise_seed)["expectation_values"])

    def counts(seed, noise_seed):
        return tuple(sorted(maestro.noisy_execute(_circuit(), model, shots=60, config=_config(method, backend, seed=seed, **options),
                                                  noise_seed=noise_seed)["counts"].items()))

    for run in (fidelity, estimate, counts):
        base = run(42, 1)
        assert run(42, 1) == base, f"{backend} {method} {run.__name__}: the same two seeds gave another result"
        # Three other seeds: a result may repeat once by chance, not every time.
        assert sum(run(42, noise_seed) != base for noise_seed in (2, 3, 77)) >= 2, (
            f"{backend} {method} {run.__name__}: the noise seed changes nothing under a fixed simulator seed")
        assert sum(run(seed, 1) != base for seed in (43, 44, 45)) >= 2, (
            f"{backend} {method} {run.__name__}: the simulator seed changes nothing under a fixed noise seed")


@pytest.mark.parametrize("coherent_qubit", [0, 5], ids=["on-a-circuit-qubit", "on-an-idle-qubit"])
@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_either_seed_changes_drawn_channels_beside_a_layer_of_its_own(backend, method, options, coherent_qubit):
    """A coherent rotation is a layer of its own per realization, so the realizations
    cannot share one circuit: its signs are drawn for each, and on an idle qubit it
    injects nothing at all. The damping beside it is still drawn by the backend as
    each realization runs, and the noise seed still joins the simulator seed for
    those branches: either seed alone changes them. On the idle qubit the branches
    are all the noise seed can change."""
    _skip_missing(backend)
    model = maestro.NoiseModel()
    for qubit in range(4):
        model.set_generalized_amplitude_damping(qubit, 0.2, 0.0)
    model.set_coherent_rotation(coherent_qubit, 0.2, 0.1, 0.3)

    def fidelity(seed, noise_seed):
        return _circuit(measure=False).noisy_fidelity(model, noise_realizations=20, config=_config(method, backend, seed=seed, **options),
                                                      noise_seed=noise_seed)["fidelity"]

    def estimate(seed, noise_seed):
        return tuple(maestro.full_noise_estimate(_circuit(measure=False), OBSERVABLES[:2], model, noise_realizations=20,
                                                 config=_config(method, backend, seed=seed, **options),
                                                 noise_seed=noise_seed)["expectation_values"])

    def counts(seed, noise_seed):
        return tuple(sorted(maestro.full_noise_execute(_circuit(), model, shots=60, config=_config(method, backend, seed=seed, **options),
                                                       noise_seed=noise_seed)["counts"].items()))

    for run in (fidelity, estimate, counts):
        base = run(42, 1)
        assert run(42, 1) == base, f"{backend} {method} {run.__name__}: the same two seeds gave another result"
        assert sum(run(42, noise_seed) != base for noise_seed in (2, 3, 77)) >= 2, (
            f"{backend} {method} {run.__name__}: the noise seed changes nothing under a fixed simulator seed")
        assert sum(run(seed, 1) != base for seed in (43, 44, 45)) >= 2, (
            f"{backend} {method} {run.__name__}: the simulator seed changes nothing under a fixed noise seed")


def test_state_sampled_channels_beside_noise_drawn_per_realization():
    """Coherent noise is drawn per realization, so the shots cannot share one
    circuit: each realization then keeps as channels the noise that no Pauli draw
    replaces, and the statevector samples those as it runs."""
    circuit, model = _circuit(delays=True), _channel_model()
    model.set_coherent_rotation(0, 0.25, 0.1, 0.2)
    model.set_coherent_rotation(2, 0.1, 0.3, 0.15)

    def run(method, seed, **options):
        result = circuit.full_noise_execute(model, config=_config(method, seed=seed, **options), shots=SHOTS, noise_realizations=SHOTS,
                                            noise_seed=seed + 1000)
        assert result["noise_realizations"] == SHOTS
        return result

    reference = dict(run(S.DensityMatrix, 4242)["counts"])
    for options in ({}, {"gate_fusion": True}):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = run(S.Statevector, 311 + len(options), **options)
        assert result["noise_evaluation"] == "trajectories"
        _assert_same_counts(result["counts"], reference, f"statevector {options} beside coherent noise")
    without = _reference_counts(circuit, _channel_model(), "full_noise_execute")
    _assert_different_counts(reference, without, "coherent rotations")


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_channel_instructions_are_drawn_per_shot_in_plain_execution(backend, method, options):
    """Without any noise model, a backend that samples channels draws a branch of
    every channel instruction for each shot: a Pauli channel as much as a damping
    one, and whether or not the qubit it acts on is measured."""
    _skip_missing(backend)
    Channel = maestro.QuantumChannel
    shots = 4000

    def ones(circuit, seed):
        counts = circuit.execute(_config(method, backend, seed=seed, **options), shots)["counts"]
        assert sum(counts.values()) == shots
        return sum(n for bits, n in counts.items() if bits[0] == "1") / shots

    for name, channel, expected in (("bit flip", Channel.pauli([0.7, 0.3, 0.0, 0.0]), 0.7),
                                    ("damping", Channel.amplitude_damping(0.4), 0.6),
                                    ("relaxation", Channel.thermal_relaxation(1.0, 2.0, 1.5, 0.1), 0.1 + 0.9 * math.exp(-0.5))):
        circuit = QuantumCircuit()
        circuit.x(0)
        circuit.channel([0], channel)
        circuit.measure_all()
        assert ones(circuit, 41) == pytest.approx(expected, abs=0.04), name

    # A channel on a qubit that is never measured still conditions the one that is.
    for name, channel in (("bit flip", Channel.pauli([0.5, 0.5, 0.0, 0.0])), ("damping", Channel.amplitude_damping(0.5))):
        circuit = QuantumCircuit()
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.channel([1], channel)
        circuit.measure([(0, 0)])
        assert ones(circuit, 43) == pytest.approx(0.5, abs=0.04), name


# ---------------------------------------------------------------- channels drawn without the state

def _stateless_channel_circuit(noisy=True):
    """Channel instructions whose branches need no state: a gate error made of a
    Pauli mixture and relaxations with T2 <= T1, phase damping, a unitary mixture."""
    Channel = maestro.QuantumChannel
    s = 1.0 / math.sqrt(2.0)
    gate_error = (Channel.depolarizing(0.04, num_qubits=2)
                  .then(Channel.thermal_relaxation(1.0, 25.0, 18.0, 0.1), [0])
                  .then(Channel.thermal_relaxation(1.0, 30.0, 30.0), [1]))
    mixture = Channel.unitary_mixture([0.8, 0.2], [[[1.0, 0.0], [0.0, 1.0]], [[s, s], [s, -s]]])
    c = QuantumCircuit()

    def channel(qubits, which):
        if noisy:
            c.channel(qubits, which)

    c.h(0)
    c.ry(1, 0.7)
    c.rx(2, 1.1)
    c.h(3)
    c.cx(0, 1)
    channel([1, 0], gate_error)
    c.t(1)
    c.cz(1, 2)
    channel([2], Channel.phase_damping(0.2))
    c.rz(3, 0.5)
    c.cx(2, 3)
    channel([2, 3], gate_error)
    channel([3], mixture)
    c.ry(0, 0.4)
    c.cx(3, 0)
    channel([0], Channel.thermal_relaxation(2.0, 20.0, 15.0))
    c.h(1)
    c.measure_all()
    return c


@pytest.mark.parametrize("method, shots", [
    pytest.param(S.TensorNetwork, SLOW_SHOTS, id="qcsim-tensor-network"),
    pytest.param(S.PathIntegral, SHOTS, id="qcsim-path-integral"),
    pytest.param(S.ExtendedStabilizer, SLOW_SHOTS, id="qcsim-extended-stabilizer"),
])
def test_channel_instructions_that_need_no_state_run_on_every_pure_state_method(method, shots):
    """A backend that neither holds nor samples channels runs realizations in which
    each such instruction has been drawn: a Pauli, a unitary, a phase flip or a reset."""
    circuit, model = _stateless_channel_circuit(), maestro.NoiseModel()
    reference = _reference_counts(circuit, model, "full_noise_execute")
    result = _execute(circuit, model, "full_noise_execute", B.QCSim, method, shots, seed=_seed(B.QCSim, method, 61), exact=False)
    _check_ran(result, B.QCSim, method)
    assert result["noise_evaluation"] == "trajectories"
    _assert_same_counts(result["counts"], reference, f"{method} channels drawn without the state")
    # The same checks tell the circuit without its channels apart.
    clean = dict(_stateless_channel_circuit(noisy=False).execute(_config(S.DensityMatrix, seed=4242), REFERENCE_SHOTS)["counts"])
    _assert_different_counts(clean, reference, f"{method} channels drawn without the state")

    # A part that needs the state has no drawn form there.
    damped = QuantumCircuit()
    damped.h(0)
    damped.channel([0], maestro.QuantumChannel.amplitude_damping(0.2))
    damped.measure_all()
    with pytest.raises(ValueError, match="density-matrix or MPO"):
        damped.full_noise_execute(model, config=_config(method), shots=10, noise_realizations=10)


def test_stabilizer_runs_the_channels_whose_branches_are_clifford():
    """A Pauli, a phase flip, a reset and a Clifford unitary are all the tableau can
    apply: a channel made of such branches is drawn into H, S, Pauli and reset
    operations. A branch that is no Clifford has no place on it."""
    Channel = maestro.QuantumChannel
    s = 1.0 / math.sqrt(2.0)
    identity, hadamard, phase = [[1.0, 0.0], [0.0, 1.0]], [[s, s], [s, -s]], [[1.0, 0.0], [0.0, 1j]]
    gate_error = (Channel.depolarizing(0.05, num_qubits=2)
                  .then(Channel.thermal_relaxation(1.0, 25.0, 18.0, 0.1), [0])
                  .then(Channel.thermal_relaxation(1.0, 30.0, 30.0), [1]))
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.channel([1, 0], gate_error)
    circuit.s(1)
    circuit.cx(1, 2)
    circuit.channel([2], Channel.unitary_mixture([0.6, 0.25, 0.15], [identity, hadamard, phase]))
    circuit.h(2)
    circuit.channel([2], Channel.phase_damping(0.2))
    circuit.cx(2, 3)
    circuit.channel([2, 3], gate_error)
    circuit.h(3)
    circuit.measure_all()
    model = maestro.NoiseModel()
    reference = _reference_counts(circuit, model, "full_noise_execute")
    for method in (S.Stabilizer, S.ExtendedStabilizer):
        result = _execute(circuit, model, "full_noise_execute", B.QCSim, method, SHOTS, seed=_seed(B.QCSim, method, 71), exact=False,
                          counts_policy="legacy")
        _check_ran(result, B.QCSim, method)
        _assert_same_counts(result["counts"], reference, f"{method} channels with Clifford branches")

    t_gate = [[1.0, 0.0], [0.0, complex(s, s)]]
    rotated = QuantumCircuit()
    rotated.h(0)
    rotated.channel([0], Channel.unitary_mixture([0.5, 0.5], [hadamard, t_gate]))
    rotated.measure_all()
    with pytest.raises((ValueError, RuntimeError)):
        rotated.full_noise_execute(model, config=_config(S.Stabilizer, counts_policy="legacy"), shots=64, noise_realizations=64, noise_seed=3)


# ---------------------------------------------------------------- chain order

def _chain_circuit(qubits, measure=True, at=None):
    """Gates between neighbours only, listed in an order that jumps along the chain,
    with delays for the idle noise. `at` gives the qubit at each place of the chain
    (the place itself by default): a line that does not run in index order, as that
    of a circuit laid out on a device."""
    rng = random.Random(7)
    line = at is not None
    at = at if line else list(range(qubits))
    c = QuantumCircuit()
    for q in range(qubits):
        c.ry(at[q], 0.3 + 0.2 * q)
    if line:
        # every pair of neighbours at least once, so that the line is whole
        for a in range(qubits - 1):
            c.cz(at[a], at[a + 1])
    for _ in range(5 * qubits):
        a = rng.randrange(qubits - 1)
        kind = rng.randrange(5)
        if kind == 0:
            c.cx(at[a], at[a + 1])
        elif kind == 1:
            c.cz(at[a], at[a + 1])
        elif kind == 2:
            c.rx(at[a + 1], rng.uniform(-1.5, 1.5))
        elif kind == 3:
            c.cx(at[a + 1], at[a])
            c.delay(at[a], 2.0)
        else:
            c.t(at[a])
            c.sx(at[a])
    if measure:
        c.measure_all()
    return c


def _device_line(qubits):
    """The qubit at each place of a line through a device: the indices in steps of five."""
    return [5 * place % qubits for place in range(qubits)]


def _chain_model(qubits, scale=1.0):
    """Noise on every qubit that a matrix product state has to sample from its state:
    amplitude damping on the even ones, relaxation with T2 > T1 on the odd ones."""
    m = maestro.NoiseModel()
    for q in range(qubits):
        if q % 2:
            m.set_thermal_relaxation(q, 1.0 * scale, 20.0, 32.0)
            m.set_thermal_relaxation_2q(q, 2.0 * scale, 20.0, 32.0, 0.1)
        else:
            m.set_t1(q, 0.04 * scale)
    m.set_idle_noise(0, 30.0 / scale, 50.0 / scale)
    m.set_all_depolarizing(qubits, 0.004 * scale)
    return m


@pytest.mark.parametrize("method, exact, shots", [(S.MatrixProductState, False, 5000), (S.MatrixProductOperator, True, SHOTS)],
                         ids=["qcsim-mps", "qcsim-mpo"])
def test_chain_ordered_circuits_match_the_density_matrix(method, exact, shots):
    """The QCSim chains run a circuit of neighbouring gates in the order that keeps
    their work local. It is the same circuit: counts follow the density matrix,
    which runs it as listed."""
    qubits = 6
    circuit, model = _chain_circuit(qubits), _chain_model(qubits)
    reference = _reference_counts(circuit, model, "full_noise_execute")
    result = _execute(circuit, model, "full_noise_execute", B.QCSim, method, shots, seed=_seed(B.QCSim, method, 51), exact=exact)
    _check_ran(result, B.QCSim, method)
    _assert_same_counts(result["counts"], reference, f"{method} on a chain")
    stronger = _execute(circuit, _chain_model(qubits, 2.0), "full_noise_execute", B.QCSim, method, shots, seed=_seed(B.QCSim, method, 52),
                        exact=exact)
    _assert_different_counts(stronger["counts"], reference, f"{method} on a chain")


def test_chain_order_keeps_the_statistics_of_matrix_product_state_trajectories():
    """Twelve qubits, where the network orders the circuit for the chain: the matrix
    product state's trajectories average to what the statevector's do."""
    qubits = 12
    circuit, model = _chain_circuit(qubits, measure=False), _chain_model(qubits)
    observables = ["Z" + "I" * 11, "I" * 5 + "Z" + "I" * 6, "I" * 11 + "Z", "ZZ" + "I" * 10, "I" * 4 + "ZZ" + "I" * 6, "I" * 10 + "ZZ",
                   "XX" + "I" * 10, "I" * 6 + "XX" + "I" * 4, "I" * 3 + "YZ" + "I" * 7, "X" + "I" * 10 + "X"]
    realizations = 400

    def estimate(method, base):
        result = maestro.full_noise_estimate(circuit, observables, model, noise_realizations=realizations,
                                             config=_config(method, seed=_seed(B.QCSim, method, base)), noise_seed=base + 7)
        assert result["noise_evaluation"] == "trajectories" and result["method"] == method
        return result

    chain, dense = estimate(S.MatrixProductState, 61), estimate(S.Statevector, 62)
    _assert_estimates_agree(chain, dense, "matrix product state against statevector on a chain", observables)
    # Not trivially: the noise moves these values.
    ideal = circuit.estimate(observables, config=_config(S.Statevector))["expectation_values"]
    moved = [abs(a - b) > Z_BOUND * e for a, b, e in zip(dense["expectation_values"], ideal, dense["realization_standard_errors"])]
    assert sum(moved) >= 3


@pytest.mark.parametrize("method, exact, shots", [(S.MatrixProductState, False, 5000), (S.MatrixProductOperator, True, SHOTS)],
                         ids=["qcsim-mps", "qcsim-mpo"])
def test_circuits_on_a_line_out_of_index_order_match_the_density_matrix(method, exact, shots):
    """A circuit laid out on a device acts on a line of qubits whose indices are not
    in order. The chains place the qubits along the line and order the circuit
    there; it is the same circuit, and the counts follow the density matrix."""
    qubits = 6
    circuit, model = _chain_circuit(qubits, at=_device_line(qubits)), _chain_model(qubits)
    reference = _reference_counts(circuit, model, "full_noise_execute")
    result = _execute(circuit, model, "full_noise_execute", B.QCSim, method, shots, seed=_seed(B.QCSim, method, 53), exact=exact)
    _check_ran(result, B.QCSim, method)
    _assert_same_counts(result["counts"], reference, f"{method} on a line out of index order")
    stronger = _execute(circuit, _chain_model(qubits, 2.0), "full_noise_execute", B.QCSim, method, shots, seed=_seed(B.QCSim, method, 54),
                        exact=exact)
    _assert_different_counts(stronger["counts"], reference, f"{method} on a line out of index order")


def test_a_line_out_of_index_order_keeps_the_statistics_of_matrix_product_state_trajectories():
    """Twelve qubits on a line that does not run in index order, where the network
    places the qubits along the line: the matrix product state's trajectories average
    to what the statevector's do, on the qubits the circuit names."""
    qubits = 12
    at = _device_line(qubits)
    circuit, model = _chain_circuit(qubits, measure=False, at=at), _chain_model(qubits)

    def pauli(*terms):
        """A Pauli string with the given letters on the qubits at the given places of the line."""
        letters = ["I"] * qubits
        for place, letter in terms:
            letters[at[place]] = letter
        return "".join(letters)

    observables = [pauli((0, "Z")), pauli((5, "Z")), pauli((11, "Z")), pauli((0, "Z"), (1, "Z")), pauli((4, "Z"), (5, "Z")),
                   pauli((10, "Z"), (11, "Z")), pauli((0, "X"), (1, "X")), pauli((6, "X"), (7, "X")), pauli((3, "Y"), (4, "Z")),
                   pauli((0, "X"), (11, "X"))]
    realizations = 400

    def estimate(method, base):
        result = maestro.full_noise_estimate(circuit, observables, model, noise_realizations=realizations,
                                             config=_config(method, seed=_seed(B.QCSim, method, base)), noise_seed=base + 7)
        assert result["noise_evaluation"] == "trajectories" and result["method"] == method
        return result

    chain, dense = estimate(S.MatrixProductState, 63), estimate(S.Statevector, 64)
    _assert_estimates_agree(chain, dense, "matrix product state against statevector on a line out of index order", observables)
    ideal = circuit.estimate(observables, config=_config(S.Statevector))["expectation_values"]
    moved = [abs(a - b) > Z_BOUND * e for a, b, e in zip(dense["expectation_values"], ideal, dense["realization_standard_errors"])]
    assert sum(moved) >= 3


# ---- The same channel through every entry point ---------------------------------------
#
# A structured channel reaches Maestro as a maestro.QuantumChannel in Python,
# as a `channel` instruction in a native JSON request, and through a simulator
# handle. Each must be the same map and run the same way.


def _native_request(request):
    """The result of a native JSON request, run by the library that serves this module."""
    import ctypes
    import json
    import sys
    from pathlib import Path

    name = {"win32": "maestro.dll", "darwin": "libmaestro.dylib"}.get(sys.platform, "libmaestro.so")
    library = None
    for candidate in (Path(maestro._raw_maestro.__file__).parent / name, name):
        try:
            library = ctypes.CDLL(str(candidate))
            break
        except OSError:
            continue
    if library is None or not hasattr(library, "MaestroRunRequestJson"):
        pytest.skip("the native library is not loadable beside the Python module")
    library.MaestroRunRequestJson.restype = ctypes.c_void_p
    library.MaestroRunRequestJson.argtypes = [ctypes.c_char_p]
    library.FreeResult.argtypes = [ctypes.c_void_p]
    pointer = library.MaestroRunRequestJson(json.dumps(request).encode())
    try:
        result = json.loads(ctypes.string_at(pointer).decode())
    finally:
        library.FreeResult(pointer)
    return result


_SURFACE_FACTORS = [
    {"kind": "depolarizing", "probability": 0.1},
    {"kind": "amplitude_damping", "qubits": [1], "gamma": 0.3},
    {"kind": "thermal_relaxation", "qubits": [0], "duration": 0.3, "t1": 1.0, "t2": 1.6, "excited_population": 0.1},
]
_SURFACE_OBSERVABLES = ["ZI", "IZ", "XX", "ZZ", "YY"]


def _surface_channel():
    """_SURFACE_FACTORS as Python builds them."""
    Q = maestro.QuantumChannel
    return (Q.depolarizing(0.1, 2)
            .then(Q.amplitude_damping(0.3), [1])
            .then(Q.thermal_relaxation(0.3, 1.0, 1.6, 0.1), [0]))


def _surface_circuit(measure=False):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.channel([0, 1], _surface_channel())
    if measure:
        circuit.measure_all()
    return circuit


def test_a_line_out_of_index_order_runs_alike_in_python_and_in_a_request():
    """Six qubits on a line that does not run in index order, with channels between
    the gates, on the matrix product operator, which places the qubits along the
    line. From Python and from a native request the expectation values are those of
    the density matrix, which runs the circuit as it is listed."""
    qubits = 6
    at = _device_line(qubits)
    damping = [{"kind": "amplitude_damping", "qubits": [0], "gamma": 0.1}]
    source = [{"name": "ry", "qubits": [at[place]], "params": [0.3 + 0.2 * place]} for place in range(qubits)]
    circuit = QuantumCircuit()
    for place in range(qubits):
        circuit.ry(at[place], 0.3 + 0.2 * place)
    for layer in range(3):
        for place in range(layer % 2, qubits - 1, 2):
            source.append({"name": "cx", "qubits": [at[place], at[place + 1]]})
            source.append({"name": "channel", "qubits": [at[place]], "factors": damping})
            circuit.cx(at[place], at[place + 1])
            circuit.channel([at[place]], maestro.QuantumChannel.amplitude_damping(0.1))
        # from the far end, so that the circuit as listed jumps along the line
        for place in reversed(range(qubits)):
            source.append({"name": "rz", "qubits": [at[place]], "params": [0.2 + 0.05 * place]})
            circuit.rz(at[place], 0.2 + 0.05 * place)

    def pauli(*terms):
        letters = ["I"] * qubits
        for place, letter in terms:
            letters[at[place]] = letter
        return "".join(letters)

    observables = [pauli((0, "Z")), pauli((3, "Z")), pauli((5, "Z")), pauli((0, "Z"), (1, "Z")), pauli((2, "X"), (3, "X")),
                   pauli((4, "Y"), (5, "Z")), pauli((0, "X"), (5, "X"))]

    def request(method):
        return {"schema_version": 2, "operation": "estimate",
                "circuit": {"format": "instructions", "num_qubits": qubits, "num_clbits": qubits, "source": source},
                "simulator": {"backend": "qcsim", "method": method}, "execution": {"seed": 11}, "observables": observables}

    exact = _native_request(request("density_matrix"))
    assert exact["ok"], exact
    native = _native_request(request("matrix_product_operator"))
    assert native["ok"], native
    assert native["expectation_values"] == pytest.approx(exact["expectation_values"], abs=1e-9)
    config = maestro.SimulatorConfig(simulation_type=S.MatrixProductOperator, seed=11)
    mine = circuit.estimate(observables, config=config)["expectation_values"]
    assert mine == pytest.approx(exact["expectation_values"], abs=1e-9)
    # Not trivially: the damping and the gates leave no observable at an extreme.
    assert all(abs(value) < 0.999 for value in exact["expectation_values"])


def _surface_request(operation, method, measure=False, **fields):
    source = [{"name": "h", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]},
              {"name": "channel", "qubits": [0, 1], "factors": _SURFACE_FACTORS}]
    if measure:
        source.append({"name": "measure", "qubits": [0, 1], "clbits": [0, 1]})
    return {"schema_version": 2, "operation": operation,
            "circuit": {"format": "instructions", "num_qubits": 2, "num_clbits": 2, "source": source},
            "simulator": {"backend": "qcsim", "method": method}, "execution": {"seed": 11}, **fields}


def test_a_structured_channel_is_the_same_map_in_python_and_in_a_request():
    """Exactly, on the density matrix: expectation values and the fidelity to the ideal state."""
    config = maestro.SimulatorConfig(simulation_type=S.DensityMatrix, seed=11)
    mine = _surface_circuit().estimate(_SURFACE_OBSERVABLES, config=config)["expectation_values"]
    native = _native_request(_surface_request("estimate", "density_matrix", observables=_SURFACE_OBSERVABLES))
    assert native["ok"], native
    assert native["expectation_values"] == pytest.approx(mine, abs=1e-12)
    assert abs(mine[3] - 1) > 0.1  # The channel does something.
    # The factors stay apart on both sides.
    assert [factor["kind"] for factor in _surface_channel().factors] == ["pauli", "thermal", "thermal"]
    fidelity = maestro.noisy_fidelity(_surface_circuit(), maestro.NoiseModel(), config=config)["fidelity"]
    request = _surface_request("noisy_fidelity", "density_matrix", noise={"channels": []})
    assert _native_request(request)["fidelity"] == pytest.approx(fidelity, abs=1e-12)


@pytest.mark.parametrize("method, name", [(S.Statevector, "statevector"), (S.MatrixProductState, "matrix_product_state")])
def test_a_structured_channel_is_sampled_alike_in_python_and_in_a_request(method, name):
    """Both entry points draw one branch per shot from the state, and agree with the density matrix."""
    shots = 20000
    exact = _native_request(_surface_request("probabilities", "density_matrix"))["probabilities"]
    config = maestro.SimulatorConfig(simulation_type=method, seed=11)
    python = _surface_circuit(measure=True).execute(config=config, shots=shots)["counts"]
    request = _surface_request("execute", name, measure=True)
    request["execution"]["shots"] = shots
    native = _native_request(request)
    assert native["ok"], native
    # Count keys list classical bit 0 first; probabilities index qubit 0 as the low bit.
    for counts in (python, native["counts"]):
        assert sum(counts.values()) == shots
        for index, probability in enumerate(exact):
            key = f"{index & 1}{index >> 1}"
            error = math.sqrt(max(probability * (1 - probability), 1e-4) / shots)
            assert abs(counts.get(key, 0) / shots - probability) < 5 * error, (key, counts, exact)


def test_one_state_of_a_sampled_channel_is_one_branch_in_python_and_in_a_request():
    """get_statevector and a statevector request both run the circuit once: one branch, picked by the seed."""
    damping = maestro.QuantumChannel.amplitude_damping(0.3)
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.channel([0], damping)
    kept = 0
    for seed in range(1, 41):
        state = maestro.get_statevector(circuit, maestro.SimulatorConfig(simulation_type=S.Statevector, seed=seed))
        populations = [abs(a) ** 2 for a in state]
        assert max(populations) == pytest.approx(1.0, abs=1e-12)
        kept += populations[1] > 0.5
    assert 18 < kept < 38  # About 0.7 of them.
    kept = 0
    for seed in range(1, 41):
        request = {"schema_version": 2, "operation": "probabilities",
                   "circuit": {"format": "instructions", "num_qubits": 1, "num_clbits": 1,
                               "source": [{"name": "x", "qubits": [0]},
                                          {"name": "channel", "qubits": [0], "factors": [{"kind": "amplitude_damping", "gamma": 0.3}]}]},
                   "simulator": {"backend": "qcsim", "method": "statevector"}, "execution": {"seed": seed}}
        result = _native_request(request)
        assert result["ok"], result
        assert max(result["probabilities"]) == pytest.approx(1.0, abs=1e-12)
        assert result["execution_metadata"]["stochastic_execution"] is True
        kept += result["probabilities"][1] > 0.5
    assert 18 < kept < 38


def test_a_placed_channel_exchanges_its_qubits():
    Q = maestro.QuantumChannel
    damping = Q.amplitude_damping(0.3)
    first, second = Q.on_qubit(damping, 0, 2), Q.on_qubit(damping, 1, 2)
    assert Q.placed(damping, [1], 2).is_approx(second)
    assert Q.placed(first, [1, 0], 2).is_approx(second)
    assert Q.placed(first, [0, 1], 2).is_approx(first)
    assert not first.is_approx(second)
    with pytest.raises(ValueError):
        Q.placed(first, [0, 0], 2)
    with pytest.raises(ValueError):
        Q.placed(damping, [2], 2)


def _handle(method, qubits=2, seed=5):
    owner = maestro.Maestro()
    handle = owner.create_simulator(B.QCSim, method)
    simulator = owner.get_simulator(handle)
    simulator.AllocateQubits(qubits)
    simulator.InitializeSimulator()
    simulator.set_seed(seed)
    return owner, handle, simulator


def test_a_simulator_handle_applies_channels_as_its_method_does():
    """Exactly on the density matrix; one branch drawn from the state on the statevector and the MPS."""
    damping = maestro.QuantumChannel.amplitude_damping(0.3)
    owner, handle, simulator = _handle(S.DensityMatrix)
    try:
        assert simulator.SupportsQuantumChannels() and not simulator.SamplesQuantumChannels()
        simulator.ApplyX(0)
        simulator.ApplyQuantumChannel([0], damping)
        assert simulator.Probability(1) == pytest.approx(0.7, abs=1e-12)
        statistics = simulator.GetGateFusionStatistics()
        assert statistics.unreadBlocks == 0 and statistics.rereadBlocks == 0
    finally:
        owner.destroy_simulator(handle)
    for method in (S.Statevector, S.MatrixProductState):
        kept = 0
        for seed in range(1, 41):
            owner, handle, simulator = _handle(method, seed=seed)
            try:
                assert simulator.SamplesQuantumChannels() and not simulator.SupportsQuantumChannels()
                simulator.ApplyX(0)
                simulator.apply_channel([0], damping)
                one = simulator.Probability(1)
                assert one == pytest.approx(round(one), abs=1e-12)
                kept += one > 0.5
            finally:
                owner.destroy_simulator(handle)
        assert 18 < kept < 38


def test_a_channel_says_whether_it_is_a_gate():
    Channel = maestro.QuantumChannel
    x, identity = [[0, 1], [1, 0]], [[1, 0], [0, 1]]
    assert Channel.unitary_mixture([1.0], [x]).is_deterministic
    assert Channel.unitary_mixture([0.0, 1.0], [identity, x]).is_deterministic
    assert Channel.pauli([0.0, 1.0, 0.0, 0.0]).is_deterministic
    assert Channel.from_kraus([x]).is_deterministic
    assert Channel.pauli([0.0, 1.0, 0.0, 0.0]).then(Channel.unitary_mixture([1.0], [x])).is_deterministic
    assert not Channel.unitary_mixture([0.5, 0.5], [identity, x]).is_deterministic
    assert not Channel.depolarizing(0.1).is_deterministic
    assert not Channel.amplitude_damping(0.2).is_deterministic
    assert not Channel.pauli([0.0, 1.0, 0.0, 0.0]).then(Channel.amplitude_damping(0.2)).is_deterministic


@pytest.mark.parametrize("backend, method, options", SAMPLING_BACKENDS)
def test_a_channel_with_one_unitary_branch_runs_as_the_gate_it_is(backend, method, options):
    """A channel with one branch only, a unitary, draws nothing: a backend that samples
    channels evolves the circuit once for all its shots, as with the gate in its place,
    an estimate is one evolution, and a realization holds the shots asked of it. A
    mixture of two unitaries is drawn shot by shot as before."""
    _skip_missing(backend)
    Channel = maestro.QuantumChannel
    x, identity = [[0, 1], [1, 0]], [[1, 0], [0, 1]]
    as_channel, drawn = Channel.unitary_mixture([1.0], [x]), Channel.unitary_mixture([0.5, 0.5], [identity, x])

    def circuit(third, measure=True):
        c = QuantumCircuit()
        c.h(0)
        c.cx(0, 1)
        if third is None:
            c.x(0)
        else:
            c.channel([0], third)
        c.ry(1, 0.4)
        c.cx(1, 2)
        if measure:
            c.measure_all()
        return c

    config = _config(method, backend, seed=_seed(backend, method, 61), **options)
    gate = circuit(None).execute(config=config, shots=4000)
    channel = circuit(as_channel).execute(config=config, shots=4000)
    assert channel["counts"] == gate["counts"], f"{backend} {method}: the channel did not sample as the gate it is"
    assert circuit(drawn).execute(config=config, shots=4000)["counts"] != gate["counts"]

    observables = ["ZZI", "XIX", "IYY"]
    expected = circuit(None, measure=False).estimate(observables, config=config)
    estimated = circuit(as_channel, measure=False).estimate(observables, config=config)
    assert not estimated["stochastic_execution"] and not expected["stochastic_execution"]
    assert circuit(drawn, measure=False).estimate(observables, config=config)["stochastic_execution"]
    assert estimated["expectation_values"] == pytest.approx(expected["expectation_values"], abs=1e-12)

    readout = maestro.NoiseModel()
    readout.set_readout_error(0, 0.02, 0.03)
    shared = maestro.noisy_execute(circuit(as_channel), readout, shots=500, noise_realizations=1, config=config)
    assert shared["noise_realizations"] == 1 and sum(shared["counts"].values()) == 500
    per_shot = maestro.noisy_execute(circuit(drawn), readout, shots=500, noise_realizations=1, config=config)
    assert per_shot["noise_realizations"] == 500


def test_a_generic_gate_that_is_not_unitary_leaves_the_channels_after_it_unbiased():
    """A generic gate need not be unitary. After one that is not, the fused statevector
    no longer knows the norm of its state, which a block of channels applied before the
    state is read would take for the probability that nothing happens: such blocks are
    read first. The branch drawn is then the one fusion off draws, seed by seed."""
    damping = maestro.QuantumChannel.amplitude_damping(0.25)

    def decays(scale, fusion, where=0, read_first=False):
        outcomes, unread = [], 0
        for seed in range(1, 201):
            owner, handle, simulator = _handle(S.Statevector, seed=seed)
            try:
                simulator.Configure("gate_fusion", "true" if fusion else "false")
                if read_first:
                    simulator.Configure("trajectory_blocks_read_first", "true")
                simulator.set_seed(seed)
                simulator.ApplyX(0)
                if scale is not None:
                    simulator.ApplyGenericOneQubitGate(where, [[scale, 0.0], [0.0, scale]])
                simulator.ApplyQuantumChannel([0], damping)
                simulator.Flush()
                unread += simulator.GetGateFusionStatistics().unreadBlocks
                # |00> against |01>: qubit 0 decayed or kept, whatever norm the gate gave the state
                outcomes.append(simulator.Probability(0) > simulator.Probability(1))
            finally:
                owner.destroy_simulator(handle)
        return outcomes, unread

    reference, unread = decays(None, False)
    # 200 seeds at a quarter: 50 expected, five sigma is 31
    assert unread == 0 and 19 < sum(reference) < 81
    assert decays(None, True, read_first=True) == (reference, 0)
    fused, unread = decays(None, True)
    assert fused == reference and unread > 100
    unitary, unread = decays(1.0, True)
    assert unitary == reference and unread > 100
    for scale in (2.0, 0.5):
        for where in (0, 1):
            outcomes, unread = decays(scale, True, where)
            assert unread == 0, f"scale {scale} on qubit {where}: {unread} blocks were applied before the state was read"
            assert outcomes == reference, (
                f"scale {scale} on qubit {where}: {sum(outcomes)} decays of 200 where fusion off gives {sum(reference)}")


def test_a_checkpoint_prefix_refuses_channels_it_would_sample_once():
    """A prefix runs once: a branch drawn there would be shared by every suffix shot."""
    prefix = QuantumCircuit()
    prefix.x(0)
    prefix.channel([0], maestro.QuantumChannel.amplitude_damping(0.3))
    suffix = QuantumCircuit()
    suffix.measure_all()
    for method in (S.Statevector, S.MatrixProductState):
        config = maestro.SimulatorConfig(simulation_type=method, seed=3, fixed_backend=True)
        with pytest.raises(ValueError, match="prefix runs once"):
            maestro.PrefixCheckpointedSimulator(prefix, 1, config)
        # In the suffix every shot draws its own branch.
        moved = QuantumCircuit()
        moved.channel([0], maestro.QuantumChannel.amplitude_damping(0.3))
        moved.measure_all()
        head = QuantumCircuit()
        head.x(0)
        counts = maestro.PrefixCheckpointedSimulator(head, 1, config).execute_suffix(moved, shots=8000)["counts"]
        assert abs(counts.get("1", 0) / 8000 - 0.7) < 0.03
    # A density matrix applies the prefix's channel exactly.
    config = maestro.SimulatorConfig(simulation_type=S.DensityMatrix, seed=3, fixed_backend=True)
    counts = maestro.PrefixCheckpointedSimulator(prefix, 1, config).execute_suffix(suffix, shots=8000)["counts"]
    assert abs(counts.get("1", 0) / 8000 - 0.7) < 0.03


def test_a_damping_of_zero_is_a_gate():
    """A damping of zero changes nothing: one branch, the identity, although its Kraus form
    has an operator that is zero. A prefix takes it, as it takes any gate."""
    Q = maestro.QuantumChannel
    for still in (Q.amplitude_damping(0.0), Q.generalized_amplitude_damping(0.0, 0.3)):
        assert still.is_deterministic
        assert len(still.kraus_operators) == 1
        np.testing.assert_allclose(still.kraus_operators[0], np.eye(2), atol=1e-15)
        for method in (S.Statevector, S.MatrixProductState, S.DensityMatrix):
            prefix = QuantumCircuit()
            prefix.x(0)
            prefix.channel([0], still)
            suffix = QuantumCircuit()
            suffix.measure_all()
            config = maestro.SimulatorConfig(simulation_type=method, seed=3, fixed_backend=True)
            counts = maestro.PrefixCheckpointedSimulator(prefix, 1, config).execute_suffix(suffix, shots=200)["counts"]
            assert counts == {"1": 200}
    # An operator that is zero is no branch of the others either: at zero
    # temperature the generalized damping is the plain one, with its two operators.
    damping = Q.amplitude_damping(0.3)
    cold = Q.generalized_amplitude_damping(0.3, 0.0)
    assert not damping.is_deterministic and len(damping.kraus_operators) == 2
    assert not cold.is_deterministic and len(cold.kraus_operators) == 2
    np.testing.assert_allclose(cold.superoperator(), damping.superoperator(), atol=1e-15)
    assert len(Q.generalized_amplitude_damping(0.3, 1.0).kraus_operators) == 2
    assert len(Q.generalized_amplitude_damping(0.3, 0.4).kraus_operators) == 4


def test_a_checkpoint_prefix_takes_the_same_in_python_and_in_a_request():
    """A prefix runs once. PrefixCheckpointedSimulator and a checkpoint_batch request take the same
    resets and channel instructions in it, refuse the same ones, and prepare the same state."""
    Q = maestro.QuantumChannel
    damped = {"kind": "amplitude_damping", "gamma": 0.3}
    # Each case: what follows an X on the qubit, in Python and as a request instruction, and the
    # probability of reading 1 where the prefix runs.
    cases = {
        "reset": (lambda c: c.reset(0), {"name": "reset", "qubits": [0]}, 0.0),
        "one unitary": (lambda c: c.channel([0], Q.unitary_mixture([1.0], [np.array([[0, 1], [1, 0]], dtype=complex)])),
                        {"name": "channel", "qubits": [0],
                         "factors": [{"kind": "unitary_mixture", "probabilities": [1.0], "unitaries": [[0, 1, 1, 0]]}]}, 0.0),
        "damping of zero": (lambda c: c.channel([0], Q.amplitude_damping(0.0)),
                            {"name": "channel", "qubits": [0], "factors": [{"kind": "amplitude_damping", "gamma": 0.0}]}, 1.0),
        "damping": (lambda c: c.channel([0], Q.amplitude_damping(0.3)), {"name": "channel", "qubits": [0], "factors": [damped]}, 0.7),
        "depolarizing": (lambda c: c.channel([0], Q.depolarizing(0.3)),
                         {"name": "channel", "qubits": [0], "factors": [{"kind": "depolarizing", "probability": 0.3}]}, 0.8),
    }
    exact = {"density_matrix": S.DensityMatrix, "matrix_product_operator": S.MatrixProductOperator}
    sampling = {"statevector": S.Statevector, "matrix_product_state": S.MatrixProductState}
    neither = {"tensor_network": S.TensorNetwork}
    one_branch = ("one unitary", "damping of zero")
    shots = 4000
    for case, (build, instruction, ones) in cases.items():
        for name, method in {**exact, **sampling, **neither}.items():
            runs = case == "reset" or name in exact or (name in sampling and case in one_branch)
            prefix = QuantumCircuit()
            prefix.x(0)
            build(prefix)
            suffix = QuantumCircuit()
            suffix.measure_all()
            config = maestro.SimulatorConfig(
                simulator_type=maestro.SimulatorType.QCSim, simulation_type=method, seed=3, fixed_backend=True)
            request = {
                "schema_version": 2, "operation": "checkpoint_batch",
                "circuit": {"format": "instructions", "num_qubits": 1, "num_clbits": 1,
                            "source": [{"name": "x", "qubits": [0]}, instruction]},
                "suffixes": [{"format": "instructions", "num_qubits": 1, "num_clbits": 1,
                              "source": [{"name": "measure", "qubits": [0], "clbits": [0]}]}],
                "simulator": {"backend": "qcsim", "method": name, "selection": "fixed"},
                "execution": {"shots": shots, "seed": 3},
            }
            native = _native_request(request)
            if not runs:
                with pytest.raises((ValueError, RuntimeError)):
                    maestro.PrefixCheckpointedSimulator(prefix, 1, config)
                assert not native["ok"] and native["error"]["code"] == "unsupported_capability", (case, name, native)
                continue
            assert native["ok"], (case, name, native)
            counts = maestro.PrefixCheckpointedSimulator(prefix, 1, config).execute_suffix(suffix, shots=shots)["counts"]
            for label, read in (("Python", counts.get("1", 0)), ("request", native["results"][0]["counts"].get("1", 0))):
                assert abs(read / shots - ones) < 0.04, f"{case} on {name}: {label} read 1 in {read} of {shots} shots, expected {ones}"
