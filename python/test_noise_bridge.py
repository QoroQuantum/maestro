"""Tests for noise_bridge.py.

Covers:
- ``calculate_noise_params``: per-qubit T1/T2 → Pauli probabilities,
  coherent residual, clamping, 2Q depolarizing 4/3 scaling, crosstalk
  angle conversion, warning paths.
- ``read_backend_data`` / ``maestro_noise_from_backend`` against a fake
  V2 backend.
- ``transpile_to_backend`` for str / qiskit / maestro circuit inputs.
- End-to-end ``execute_backend_noise_model`` and
  ``estimate_backend_noise_model`` runs on a fake backend, compared
  against the noiseless simulation of the same transpiled circuit via
  ``maestro.simple_execute`` / ``maestro.simple_estimate``.

Run with: pytest python/test_noise_bridge.py
"""

from __future__ import annotations

import math
import os
import sys
import warnings

import pandas as pd
import pytest

# Allow running directly from the repo without installing the package: prepend
# the locally built extension and this directory so ``import maestro`` and
# ``import ibm_noise_bridge`` resolve to the in-tree versions.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_BUILD = os.path.join(_REPO, "build", "cp312-cp312-macosx_15_0_x86_64")
for _p in (_BUILD, _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import maestro  # noqa: E402
import qiskit  # noqa: E402
import qiskit.qasm2  # noqa: E402

import noise_bridge as br  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_backend():
    """A deterministic fake IBM backend (7 qubits)."""
    from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
    return FakeJakartaV2()


def _ghz_qasm(n: int, measure: bool = True) -> str:
    """Build a GHZ-n QASM circuit on a fresh n-qubit register."""
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f"qreg q[{n}];",
    ]
    if measure:
        lines.append(f"creg c[{n}];")
    lines.append("h q[0];")
    for i in range(n - 1):
        lines.append(f"cx q[{i}], q[{i+1}];")
    if measure:
        for i in range(n):
            lines.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(lines)


@pytest.fixture
def synthetic_df():
    """Hand-built DataFrame so we can check the math exactly."""
    rows = [
        {
            "T1": 100e-6,
            "T2": 50e-6,
            "prob_meas0_prep1": 0.03,
            "prob_meas1_prep0": 0.01,
            "gate_error_1q": 1e-3,
            "gate_length_1q": 50e-9,
        },
        {
            "T1": 10e-6,
            "T2": 100e-6,   # > 2*T1: must be capped to 2*T1 = 20e-6
            "prob_meas0_prep1": 0.0,
            "prob_meas1_prep0": 0.0,
            "gate_error_1q": 5e-4,
            "gate_length_1q": 35e-9,
        },
        {
            # gate_error sits well below the decoherence floor → expect clamp + warning
            "T1": 1e-6,
            "T2": 2e-6,
            "prob_meas0_prep1": 0.0,
            "prob_meas1_prep0": 0.0,
            "gate_error_1q": 1e-8,
            "gate_length_1q": 100e-9,
        },
    ]
    df = pd.DataFrame(rows)
    df.index.name = "qubit"
    df.attrs["gates_2q"] = {
        (0, 1): {"cx_error": 1e-2, "cx_length": 300e-9},
        (1, 2): {"ecr_error": 5e-3, "ecr_length": 250e-9},
        # pair with no recognizable "*_error" key — should be skipped + warn
        (2, 0): {"junk_length": 0.0},
    }
    df.attrs["crosstalk"] = {(0, 1): 0.25}  # 2*asin(sqrt(0.25)) = pi/3
    return df


# ---------------------------------------------------------------------------
# calculate_noise_params: structure and math
# ---------------------------------------------------------------------------


class TestCalculateNoiseParams:
    def test_returns_three_sections(self, synthetic_df):
        params = br.calculate_noise_params(synthetic_df)
        assert set(params) == {"qubits", "gates_2q", "crosstalk"}

    def test_per_qubit_keys(self, synthetic_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = br.calculate_noise_params(synthetic_df)
        expected_keys = {
            "t1_time", "gate_time_1q", "p_dephasing",
            "p_1q_residual", "delta_2q_per_q",
            "p_meas1_prep0", "p_meas0_prep1",
        }
        for q, entry in params["qubits"].items():
            assert set(entry) == expected_keys, f"qubit {q}: {set(entry)}"

    def test_q0_dephasing_and_residual_match_formulas(self, synthetic_df):
        """The 1Q dephasing probability follows the pure-T2 decay rate
        γ_φ = max(0, 1/T2_eff − 1/T1) at the 1Q gate time, and the 1Q
        residual depolarizing equals (3/2)·max(0, gate_error − thermal_infid)
        per maestro's Pauli-mixing convention with dim=2."""
        params = br.calculate_noise_params(synthetic_df)
        q0 = params["qubits"][0]

        T1, T2, t = 100e-6, 50e-6, 50e-9
        T2_eff = min(T2, 2.0 * T1)
        inv_gap = max(0.0, 1.0 / T2_eff - 1.0 / T1)
        p_z_expected = (1.0 - math.exp(-t * inv_gap)) / 2.0
        p_reset = 1.0 - math.exp(-t / T1)
        infid_thermal = (2.0/3.0) * (1.0 - p_reset) * p_z_expected + 0.5 * p_reset
        p_1q_expected = (1e-3 - infid_thermal) * 3.0 / 2.0

        assert q0["p_dephasing"] == pytest.approx(p_z_expected, rel=1e-12)
        assert q0["t1_time"] == pytest.approx(T1)
        assert q0["gate_time_1q"] == pytest.approx(t)
        assert q0["p_1q_residual"] == pytest.approx(p_1q_expected, rel=1e-10)
        # Readout pass-through
        assert q0["p_meas0_prep1"] == 0.03
        assert q0["p_meas1_prep0"] == 0.01

    def test_t2_capped_at_2T1_kills_pure_dephasing(self, synthetic_df):
        """At the physical bound T2 = 2·T1 the pure dephasing rate
        γ_φ = 1/T2_eff − 1/(2·T1) is zero, so the bridge applies no
        explicit Pauli-Z channel (all decoherence flows through T1
        amplitude damping)."""
        params = br.calculate_noise_params(synthetic_df)
        q1 = params["qubits"][1]
        # q1 has T2 > 2·T1 in synthetic_df, so T2_eff caps at 2·T1.
        assert q1["p_dephasing"] == pytest.approx(0.0, abs=1e-15)

    def test_stale_calibration_clamps_and_warns(self, synthetic_df):
        """When gate_error < thermal_infid the 1Q residual would go
        negative; clamp to 0 and warn."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = br.calculate_noise_params(synthetic_df)
        assert params["qubits"][2]["p_1q_residual"] == 0.0
        msgs = [str(w.message) for w in caught]
        assert any("below the T1/T2 decoherence floor" in m for m in msgs), msgs

    def test_2q_depolarizing_subtracts_thermal_and_uses_5_over_4_factor(self, synthetic_df):
        """For 2Q gates: ``p_2q = (5/4) · max(0, gate_error_2q − thermal_2q_pair)``.
        The 5/4 factor is the maestro Pauli-mixing convention for d=4
        (``p = err_avg · (d+1)/d``); thermal_2q_pair is the sum of
        per-qubit thermal infidelities over the CX duration."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = br.calculate_noise_params(synthetic_df)

        def _thermal(T1, T2, t):
            if T1 <= 0 or t <= 0:
                return 0.0
            p_reset = 1.0 - math.exp(-t / T1)
            if T2 <= 0:
                return 0.5 * p_reset
            T2_eff = min(T2, 2.0 * T1)
            inv_gap = max(0.0, 1.0 / T2_eff - 1.0 / T1)
            p_z = (1.0 - math.exp(-t * inv_gap)) / 2.0
            return (2.0/3.0) * (1.0 - p_reset) * p_z + 0.5 * p_reset

        # cx pair (0,1) — cx_length = 300e-9
        infid_q0 = _thermal(100e-6, 50e-6, 300e-9)
        infid_q1 = _thermal(10e-6, 100e-6, 300e-9)
        expected_01 = max(0.0, 1e-2 - (infid_q0 + infid_q1)) * 5.0 / 4.0
        expected_01 = min(1.0, expected_01)
        assert params["gates_2q"][(0, 1)]["p"] == pytest.approx(expected_01, rel=1e-10)
        assert params["gates_2q"][(0, 1)]["gate_time"] == pytest.approx(300e-9)

        # ecr pair (1,2) — ecr_length = 250e-9
        infid_q1_at_ecr = _thermal(10e-6, 100e-6, 250e-9)
        infid_q2_at_ecr = _thermal(1e-6, 2e-6, 250e-9)
        expected_12 = max(0.0, 5e-3 - (infid_q1_at_ecr + infid_q2_at_ecr)) * 5.0 / 4.0
        expected_12 = min(1.0, expected_12)
        assert params["gates_2q"][(1, 2)]["p"] == pytest.approx(expected_12, rel=1e-10)
        assert params["gates_2q"][(1, 2)]["gate_time"] == pytest.approx(250e-9)

    def test_unrecognized_2q_gate_warns_and_skips(self, synthetic_df):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            params = br.calculate_noise_params(synthetic_df)
        assert (2, 0) not in params["gates_2q"]
        msgs = [str(w.message) for w in caught]
        assert any("No recognized 2Q gate for pair (2, 0)" in m for m in msgs), msgs

    def test_crosstalk_angle_is_2_asin_sqrt_p(self, synthetic_df):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = br.calculate_noise_params(synthetic_df)
        # p=0.25 -> 2*asin(0.5) = pi/3
        assert params["crosstalk"][(0, 1)] == pytest.approx(math.pi / 3.0)

    def test_2q_depolarizing_clamped_at_1(self):
        """When the residual times the 5/4 factor exceeds 1, clamp."""
        df = pd.DataFrame([{
            "T1": 100e-6, "T2": 50e-6,
            "prob_meas0_prep1": 0.0, "prob_meas1_prep0": 0.0,
            "gate_error_1q": 0.0, "gate_length_1q": 50e-9,
        }])
        df.index.name = "qubit"
        df.attrs["gates_2q"] = {(0, 1): {"cx_error": 0.95, "cx_length": 0.0}}
        df.attrs["crosstalk"] = {}
        params = br.calculate_noise_params(df)
        # gate_time=0 → no thermal subtraction → (5/4)·0.95 = 1.1875 → clamp 1.0
        assert params["gates_2q"][(0, 1)]["p"] == 1.0

    def test_p_1q_residual_clamped_at_1(self):
        """When the per-qubit residual times 3/2 exceeds 1, clamp."""
        df = pd.DataFrame([{
            "T1": 100e-6, "T2": 50e-6,
            "prob_meas0_prep1": 0.0, "prob_meas1_prep0": 0.0,
            "gate_error_1q": 5.0,  # absurd, just to exercise clamp
            "gate_length_1q": 50e-9,
        }])
        df.index.name = "qubit"
        df.attrs["gates_2q"] = {}
        df.attrs["crosstalk"] = {}
        params = br.calculate_noise_params(df)
        assert params["qubits"][0]["p_1q_residual"] == 1.0

    def test_zero_t1_t2_zero_thermal_means_full_residual(self):
        """With no T1/T2 data the thermal floor is 0, so the entire
        gate_error goes into the residual depolarizing channel
        (scaled by 3/2 per the Pauli-mixing convention for dim=2)."""
        df = pd.DataFrame([{
            "T1": 0.0, "T2": 0.0,
            "prob_meas0_prep1": 0.0, "prob_meas1_prep0": 0.0,
            "gate_error_1q": 1e-4, "gate_length_1q": 50e-9,
        }])
        df.index.name = "qubit"
        df.attrs["gates_2q"] = {}
        df.attrs["crosstalk"] = {}
        params = br.calculate_noise_params(df)
        q = params["qubits"][0]
        assert q["p_dephasing"] == 0.0
        assert q["delta_2q_per_q"] == 0.0
        assert q["p_1q_residual"] == pytest.approx(1e-4 * 3.0 / 2.0)

    def test_delta_2q_thermal_present_when_2q_gate_longer(self):
        """A qubit that participates in a 2Q gate longer than its 1Q
        gates gets a positive delta_2q_per_q (extra CX-time thermal)."""
        df = pd.DataFrame([{
            "T1": 50e-6, "T2": 30e-6,
            "prob_meas0_prep1": 0.0, "prob_meas1_prep0": 0.0,
            "gate_error_1q": 1e-3, "gate_length_1q": 35e-9,
        }, {
            "T1": 80e-6, "T2": 60e-6,
            "prob_meas0_prep1": 0.0, "prob_meas1_prep0": 0.0,
            "gate_error_1q": 1e-3, "gate_length_1q": 35e-9,
        }])
        df.index.name = "qubit"
        df.attrs["gates_2q"] = {(0, 1): {"cx_error": 1e-2, "cx_length": 400e-9}}
        df.attrs["crosstalk"] = {}
        params = br.calculate_noise_params(df)
        # Both qubits should see a positive 2Q delta (longer CX → more decay)
        assert params["qubits"][0]["delta_2q_per_q"] > 0.0
        assert params["qubits"][1]["delta_2q_per_q"] > 0.0

    def test_delta_2q_zero_when_no_2q_gates(self):
        """A qubit not involved in any 2Q gate has zero delta thermal."""
        df = pd.DataFrame([{
            "T1": 100e-6, "T2": 50e-6,
            "prob_meas0_prep1": 0.0, "prob_meas1_prep0": 0.0,
            "gate_error_1q": 1e-4, "gate_length_1q": 50e-9,
        }])
        df.index.name = "qubit"
        df.attrs["gates_2q"] = {}
        df.attrs["crosstalk"] = {}
        params = br.calculate_noise_params(df)
        assert params["qubits"][0]["delta_2q_per_q"] == 0.0


# ---------------------------------------------------------------------------
# read_backend_data on a fake backend
# ---------------------------------------------------------------------------


class TestReadBackendData:
    def test_dataframe_shape(self, fake_backend):
        df = br.read_backend_data(fake_backend)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == fake_backend.num_qubits
        assert df.index.name == "qubit"
        for col in ("T1", "T2", "prob_meas0_prep1", "prob_meas1_prep0",
                    "gate_error_1q", "gate_length_1q"):
            assert col in df.columns

    def test_attrs_have_2q_dicts(self, fake_backend):
        df = br.read_backend_data(fake_backend)
        assert isinstance(df.attrs.get("gates_2q"), dict)
        assert isinstance(df.attrs.get("crosstalk"), dict)
        # Manila has cx pairs.
        assert len(df.attrs["gates_2q"]) > 0
        for pair, entry in df.attrs["gates_2q"].items():
            assert isinstance(pair, tuple) and len(pair) == 2
            assert any(k.endswith("_error") for k in entry)

    def test_t1_t2_positive(self, fake_backend):
        df = br.read_backend_data(fake_backend)
        assert (df["T1"] > 0).all()
        assert (df["T2"] > 0).all()


# ---------------------------------------------------------------------------
# maestro_noise_from_backend
# ---------------------------------------------------------------------------


class TestMaestroNoiseFromBackend:
    def test_returns_noise_model_with_some_noise(self, fake_backend):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nm = br.maestro_noise_from_backend(fake_backend)
        assert isinstance(nm, maestro.NoiseModel)
        assert nm.has_any()

    def test_individual_channels_present(self, fake_backend):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nm = br.maestro_noise_from_backend(fake_backend)
        # T1 damping, readout error, correlated 2Q depolarizing on the
        # CX pairs, and 1Q-gate-specific depolarizing (the new incoherent
        # residual channel) should all be configured.
        assert nm.has_t1()
        assert nm.has_readout_error()
        assert nm.has_any_2q_depolarizing()
        assert nm.has_1q_gate_noise()
        # 2Q-gate-only delta thermal also populates the noise_2q_ map
        assert nm.has_2q_gate_noise()
        # The bridge no longer attaches coherent noise from RB residuals
        # (only set_crosstalk uses a coherent rotation, and fake backends
        # don't expose rzz characterization).
        assert not nm.has_coherent()


# ---------------------------------------------------------------------------
# transpile_to_backend
# ---------------------------------------------------------------------------


_BELL_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


class TestTranspileToBackend:
    def test_accepts_qasm_string(self, fake_backend):
        out = br.transpile_to_backend(_BELL_QASM, fake_backend)
        assert isinstance(out, qiskit.QuantumCircuit)
        assert out.num_qubits == fake_backend.num_qubits

    def test_accepts_qiskit_circuit(self, fake_backend):
        qc = qiskit.qasm2.loads(_BELL_QASM)
        out = br.transpile_to_backend(qc, fake_backend)
        assert isinstance(out, qiskit.QuantumCircuit)
        assert out.num_qubits == fake_backend.num_qubits

    def test_accepts_maestro_circuit(self, fake_backend):
        mcirc = maestro.QasmToCirc().parse_and_translate(_BELL_QASM)
        out = br.transpile_to_backend(mcirc, fake_backend)
        assert isinstance(out, qiskit.QuantumCircuit)
        assert out.num_qubits == fake_backend.num_qubits

    def test_rejects_unknown_type(self, fake_backend):
        with pytest.raises(TypeError):
            br.transpile_to_backend(12345, fake_backend)


# ---------------------------------------------------------------------------
# bridge_core
# ---------------------------------------------------------------------------


class TestBridgeCore:
    def test_returns_tuple(self, fake_backend):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            circ, nm = br.bridge_core(_BELL_QASM, fake_backend)
        assert isinstance(circ, maestro.circuits.QuantumCircuit)
        assert isinstance(nm, maestro.NoiseModel)

    def test_transpiled_circuit_matches_backend_width(self, fake_backend):
        # Use a GHZ-N circuit that touches every qubit, so the parsed
        # maestro circuit reports the full backend width. (Maestro's
        # QasmToCirc infers num_qubits from the highest qubit actually
        # used by a gate, not from the declared qreg.)
        qasm = _ghz_qasm(fake_backend.num_qubits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            circ, _ = br.bridge_core(qasm, fake_backend)
        assert circ.num_qubits == fake_backend.num_qubits


# ---------------------------------------------------------------------------
# End-to-end: fake-backend execution vs. exact (no-noise) simulation
# ---------------------------------------------------------------------------


def _transpiled_ghz_qasm(backend) -> str:
    """Transpile a backend-width GHZ the same way the bridge does, then
    dump it back to QASM so simple_execute / simple_estimate can run on
    the very same circuit."""
    qasm = _ghz_qasm(backend.num_qubits)
    tr = qiskit.transpile(qiskit.qasm2.loads(qasm), backend=backend,
                          seed_transpiler=42)
    return qiskit.qasm2.dumps(tr)


class TestExecuteBackendNoiseModelVsExact:
    def test_total_shots_preserved(self, fake_backend):
        qasm = _ghz_qasm(fake_backend.num_qubits)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = br.execute_backend_noise_model(qasm, fake_backend, shots=500)
        assert sum(res["counts"].values()) == 500

    def test_dominant_outcomes_match_exact(self, fake_backend):
        """The noisy distribution should still be dominated by the GHZ
        outcomes (|0...0> and |1...1>). We verify that by simulating the
        *same* transpiled circuit with no noise via maestro.simple_execute
        and checking that the two most likely bitstrings from the noisy
        run lie in the support of the ideal run."""
        n = fake_backend.num_qubits
        qasm = _ghz_qasm(n)
        tr_qasm = _transpiled_ghz_qasm(fake_backend)

        ideal = maestro.simple_execute(tr_qasm, shots=4000)
        ideal_support = {bs for bs, c in ideal["counts"].items() if c > 0}
        # GHZ is a stabilizer state — exact run should be supported only on
        # the two all-zero / all-one outcomes.
        assert ideal_support == {"0" * n, "1" * n}, ideal_support

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noisy = br.execute_backend_noise_model(qasm, fake_backend,
                                                   shots=4000)
        top_noisy = sorted(noisy["counts"].items(),
                           key=lambda kv: kv[1], reverse=True)[:2]
        top_keys = {k for k, _ in top_noisy}
        assert top_keys.issubset(ideal_support), (top_keys, ideal_support)

    def test_noisy_distribution_close_to_exact(self, fake_backend):
        """Total variation distance between the noisy distribution and the
        ideal distribution of the same transpiled GHZ-N circuit. GHZ-N
        has N-1 CX gates so the budget grows with N; on FakeJakartaV2
        (7q, ~1% CX error) the TVD should still stay under ~0.4."""
        qasm = _ghz_qasm(fake_backend.num_qubits)
        tr_qasm = _transpiled_ghz_qasm(fake_backend)
        shots = 8000
        ideal = maestro.simple_execute(tr_qasm, shots=shots)["counts"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noisy = br.execute_backend_noise_model(qasm, fake_backend,
                                                   shots=shots)["counts"]
        keys = set(ideal) | set(noisy)
        tvd = 0.5 * sum(abs(ideal.get(k, 0) - noisy.get(k, 0)) for k in keys) / shots
        assert tvd < 0.4, f"TVD too large: {tvd}"


def _first_cx_pair(transpiled: qiskit.QuantumCircuit) -> tuple[int, int]:
    """Return the first (control, target) physical-qubit pair in the
    transpiled circuit. Used to locate a guaranteed-correlated qubit pair
    for Pauli observables."""
    for inst in transpiled.data:
        if inst.operation.name == "cx":
            return (transpiled.find_bit(inst.qubits[0]).index,
                    transpiled.find_bit(inst.qubits[1]).index)
    raise AssertionError("no CX after transpilation")


class TestEstimateBackendNoiseModelVsExact:
    def _ghz_no_measure(self, n: int) -> str:
        return _ghz_qasm(n, measure=False)

    def test_ideal_field_matches_exact(self, fake_backend):
        """The 'ideal_expectation_values' field returned by noisy_estimate
        must match a noiseless simple_estimate on the same transpiled
        circuit."""
        n = fake_backend.num_qubits
        qasm = self._ghz_no_measure(n)

        tr = qiskit.transpile(qiskit.qasm2.loads(qasm), backend=fake_backend,
                              seed_transpiler=42)
        a, b = _first_cx_pair(tr)
        zz = ["I"] * n
        zz[a] = "Z"
        zz[b] = "Z"
        zz = "".join(zz)  # big-endian: position 0 == qubit 0

        tr_qasm = qiskit.qasm2.dumps(tr)
        exact = maestro.simple_estimate(tr_qasm, [zz])["expectation_values"][0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = br.estimate_backend_noise_model(qasm, [zz], fake_backend)
        assert res["ideal_expectation_values"][0] == pytest.approx(exact, abs=1e-9)

    def test_noisy_magnitude_not_above_ideal(self, fake_backend):
        """noisy_estimate applies analytical Pauli damping, which can only
        shrink magnitudes. We exercise this across a mix of single-Z
        (ideal 0 → tautologically passes), GHZ-pair ZZ (ideal +1),
        all-X parity (ideal +1 — most sensitive to dephasing)."""
        n = fake_backend.num_qubits
        qasm = self._ghz_no_measure(n)
        tr = qiskit.transpile(qiskit.qasm2.loads(qasm), backend=fake_backend,
                              seed_transpiler=42)
        a, b = _first_cx_pair(tr)

        obs_list: list[str] = []
        # Z on each single qubit → 0 for GHZ
        for q in range(n):
            s = ["I"] * n
            s[q] = "Z"
            obs_list.append("".join(s))
        # ZZ on the GHZ pair → +1
        zz = ["I"] * n
        zz[a] = "Z"
        zz[b] = "Z"
        obs_list.append("".join(zz))
        # All-X parity (the GHZ stabilizer X⊗X⊗…⊗X) → +1 in the ideal
        obs_list.append("X" * n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = br.estimate_backend_noise_model(qasm, obs_list, fake_backend)

        for noisy, ideal in zip(res["expectation_values"],
                                res["ideal_expectation_values"]):
            assert abs(noisy) <= abs(ideal) + 1e-9, (noisy, ideal)

        # The X^n stabilizer is the most dephasing-sensitive observable in
        # the list — it should be visibly damped below the ideal +1.
        all_x_noisy = res["expectation_values"][-1]
        all_x_ideal = res["ideal_expectation_values"][-1]
        assert all_x_ideal == pytest.approx(1.0, abs=1e-9)
        assert all_x_noisy < all_x_ideal


class TestEstimateMonteCarloVsExact:
    def test_ghz_zz_montecarlo_matches_exact(self, fake_backend):
        """Compare gate-by-gate Monte Carlo to the exact (no-noise)
        simulation of the same transpiled GHZ-N circuit. We pick ZZ on
        the first CX pair so the observable equals +1 in the ideal."""
        n = fake_backend.num_qubits
        qasm = _ghz_qasm(n, measure=False)
        tr = qiskit.transpile(qiskit.qasm2.loads(qasm), backend=fake_backend,
                              seed_transpiler=42)
        a, b = _first_cx_pair(tr)
        obs = ["I"] * n
        obs[a] = "Z"
        obs[b] = "Z"
        obs = "".join(obs)

        tr_qasm = qiskit.qasm2.dumps(tr)
        exact = maestro.simple_estimate(tr_qasm, [obs])["expectation_values"][0]
        assert exact == pytest.approx(1.0, abs=1e-9)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = br.estimate_backend_noise_model_montecarlo(
                qasm, [obs], fake_backend, noise_realizations=64, seed=7,
            )
        assert res["ideal_expectation_values"][0] == pytest.approx(exact, abs=1e-9)
        # GHZ-N noise on Jakarta is small but compounded across CX gates →
        # MC <ZZ> should stay clearly positive.
        assert res["expectation_values"][0] > 0.5
        assert res["noise_realizations"] == 64


def _measure_map(transpiled: qiskit.QuantumCircuit) -> tuple[dict[int, int], int]:
    """Parse a transpiled circuit's measure instructions into a
    {physical_qubit: classical_bit} dict, plus the creg width."""
    mapping: dict[int, int] = {}
    for inst in transpiled.data:
        if inst.operation.name == "measure":
            q = transpiled.find_bit(inst.qubits[0]).index
            c = transpiled.find_bit(inst.clbits[0]).index
            mapping[q] = c
    return mapping, transpiled.num_clbits


def _marginalize(maestro_bitstring: str, measure_map: dict[int, int],
                 creg_w: int) -> str:
    """Project a maestro full-register bitstring onto a qiskit-style
    classical-register bitstring.

    Maestro emits bits in **qubit-0-LEFT** order (the leftmost char is
    qubit 0). Qiskit's classical-register output uses **c[0]-RIGHT**
    (the rightmost char is c[0]). So qubit ``q``'s sample sits at
    ``maestro_bitstring[q]`` and we place it at ``out[creg_w - 1 - c]``.
    """
    out = ["0"] * creg_w
    for q, c in measure_map.items():
        out[creg_w - 1 - c] = maestro_bitstring[q]
    return "".join(out)


def _marginalize_counts(counts: dict[str, int], measure_map: dict[int, int],
                        creg_w: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for bits, c in counts.items():
        k = _marginalize(bits, measure_map, creg_w)
        out[k] = out.get(k, 0) + c
    return out


def _tvd(a: dict[str, int], b: dict[str, int], shots: int) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


class TestBridgeVsQiskitAerOnFakeBackend:
    """Compare maestro's bridge-built noise simulation against qiskit's
    own aer-based simulation of the same fake backend. Both consume the
    same device properties, so the count distributions should agree at
    the few-percent level on simple stabilizer states."""

    @pytest.fixture(autouse=True)
    def _aer(self):
        pytest.importorskip("qiskit_aer")

    def test_ghz_distribution_close_to_qiskit_aer(self, fake_backend):
        """GHZ-N circuit on all N backend qubits: ideal support is
        {0…0, 1…1}. Both noise simulators should agree on the marginal
        distribution within the budget set by accumulated CX error."""
        n = fake_backend.num_qubits
        shots = 16000
        qasm = _ghz_qasm(n)
        qc = qiskit.qasm2.loads(qasm)
        tr = qiskit.transpile(qc, backend=fake_backend, seed_transpiler=42)
        measure_map, creg_w = _measure_map(tr)
        assert creg_w == n and len(measure_map) == n

        qiskit_counts = fake_backend.run(
            tr, shots=shots, seed_simulator=123,
        ).result().get_counts()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bridge_full = br.execute_backend_noise_model(
                qasm, fake_backend, shots=shots,
            )["counts"]
        bridge_counts = _marginalize_counts(bridge_full, measure_map, creg_w)

        # Ideal mass should be similarly concentrated for both backends.
        zero = "0" * n
        one = "1" * n
        ideal_mass_q = sum(qiskit_counts.get(s, 0) for s in (zero, one)) / shots
        ideal_mass_b = sum(bridge_counts.get(s, 0) for s in (zero, one)) / shots
        assert abs(ideal_mass_q - ideal_mass_b) < 0.15, (
            ideal_mass_q, ideal_mass_b,
        )

        tvd = _tvd(qiskit_counts, bridge_counts, shots)
        # GHZ-N has N-1 CX gates, so total noise grows with N. Use a
        # bound that's loose enough to cover both 5q and 7q-class backends.
        assert tvd < 0.25, f"GHZ-{n} TVD bridge vs qiskit-aer: {tvd}"

    def test_render_bridge_vs_qiskit_aer_plot(self, fake_backend, tmp_path):
        """Visual companion: GHZ-N distribution under the maestro bridge
        vs. qiskit-aer's noisy simulation of the same fake backend."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = fake_backend.num_qubits
        shots = 100000
        qasm = _ghz_qasm(n)
        qc = qiskit.qasm2.loads(qasm)
        tr = qiskit.transpile(qc, backend=fake_backend, seed_transpiler=42)
        measure_map, creg_w = _measure_map(tr)

        qiskit_counts = fake_backend.run(
            tr, shots=shots, seed_simulator=123,
        ).result().get_counts()

        tr_qasm = qiskit.qasm2.dumps(tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bridge_full = br.execute_backend_noise_model(
                qasm, fake_backend, shots=shots,
            )["counts"]
            # Run the *same transpiled* circuit noiselessly so bit
            # positions line up with bridge_full before marginalization.
            exact_full = maestro.simple_execute(tr_qasm, shots=shots)["counts"]

        bridge_counts = _marginalize_counts(bridge_full, measure_map, creg_w)
        exact_counts = _marginalize_counts(exact_full, measure_map, creg_w)


        # Plot only the top-20 outcomes by combined frequency — the long
        # tail of single-shot outcomes drowns the figure otherwise.
        scored = sorted(
            set(qiskit_counts) | set(bridge_counts) | set(exact_counts),
            key=lambda b: -(qiskit_counts.get(b, 0) + bridge_counts.get(b, 0) + exact_counts.get(b, 0)),
        )
        keys = scored[:20]
        q_freq = [qiskit_counts.get(k, 0) / shots for k in keys]
        b_freq = [bridge_counts.get(k, 0) / shots for k in keys]
        e_freq = [exact_counts.get(k, 0) / shots for k in keys]
        tvd_bridge = _tvd(qiskit_counts, bridge_counts, shots)
        tvd_exact = _tvd(qiskit_counts, exact_counts, shots)

        fig, ax = plt.subplots(figsize=(13, 5))
        x = range(len(keys))
        w = 0.28  # three bars per outcome
        ax.bar([i - w for i in x], e_freq, width=w,
               label="exact (noise-free)", color="#1f77b4")
        ax.bar(list(x), q_freq, width=w,
               label="qiskit-aer (fake backend)", color="#2ca02c")
        ax.bar([i + w for i in x], b_freq, width=w,
               label="maestro bridge", color="#d62728", alpha=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(keys, fontfamily="monospace", fontsize=8,
                           rotation=60, ha="right")
        ax.set_ylabel("probability")
        ax.set_title(
            f"GHZ-{n} on {type(fake_backend).__name__} — exact vs. "
            f"qiskit-aer vs. maestro bridge\n"
            f"TVD(aer, bridge) = {tvd_bridge:.3f}; "
            f"TVD(aer, exact) = {tvd_exact:.3f}; "
            f"top 20 / {shots} shots"
        )
        ax.legend()
        fig.tight_layout()

        out_dir = os.path.join(_HERE, "_plots")
        os.makedirs(out_dir, exist_ok=True)
        repo_path = os.path.join(out_dir, "bridge_vs_qiskit_aer.png")
        tmp_artifact = tmp_path / "bridge_vs_qiskit_aer.png"
        fig.savefig(repo_path, dpi=120)
        fig.savefig(tmp_artifact, dpi=120)
        plt.close(fig)

        assert os.path.getsize(repo_path) > 0
        assert tmp_artifact.exists() and tmp_artifact.stat().st_size > 0


class TestExactVsNoisyPlot:
    """Render a side-by-side visualization of the exact vs. noisy Bell
    state distribution on a fake backend, and a bar plot of exact vs.
    noisy expectation values across a Pauli observable basis. The test
    asserts the figure was produced (file exists, non-empty) so it stays
    part of the suite, but its real purpose is the artifact under
    ``python/_plots/`` for human inspection."""

    def test_render_bell_distribution_and_expectations(self, fake_backend, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- (1) Counts distribution: GHZ-N on the bridge vs. the same
        # transpiled circuit run noiselessly via simple_execute ---
        n = fake_backend.num_qubits
        shots = 8000
        qasm = _ghz_qasm(n)
        tr_qasm = _transpiled_ghz_qasm(fake_backend)
        ideal_counts = maestro.simple_execute(tr_qasm, shots=shots)["counts"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            noisy_counts = br.execute_backend_noise_model(
                qasm, fake_backend, shots=shots,
            )["counts"]

        # Show top-20 bitstrings by combined frequency — long tails would
        # otherwise drown the dominant peaks.
        scored = sorted(
            set(ideal_counts) | set(noisy_counts),
            key=lambda b: -(ideal_counts.get(b, 0) + noisy_counts.get(b, 0)),
        )
        all_bits = scored[:20]
        ideal_freq = [ideal_counts.get(b, 0) / shots for b in all_bits]
        noisy_freq = [noisy_counts.get(b, 0) / shots for b in all_bits]

        # --- (2) Expectation values that distinguish GHZ-N noise behaviour ---
        ghz_no_measure = _ghz_qasm(n, measure=False)
        tr = qiskit.transpile(qiskit.qasm2.loads(ghz_no_measure),
                              backend=fake_backend, seed_transpiler=42)
        a, b = _first_cx_pair(tr)

        def _set_at(string_pauli: dict[int, str]) -> str:
            s = ["I"] * n
            for q, p in string_pauli.items():
                s[q] = p
            return "".join(s)

        obs_list = [
            (_set_at({a: "Z", b: "Z"}), "ZZ (pair)"),
            (_set_at({0: "Z"}), "Z_0"),
            ("Z" * n, f"Z^{n} (parity)"),
            ("X" * n, f"X^{n} (GHZ stabilizer)"),
        ]
        obs_strings = [o for o, _ in obs_list]
        labels = [lbl for _, lbl in obs_list]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = br.estimate_backend_noise_model(ghz_no_measure, obs_strings,
                                                  fake_backend)
        ideal_ev = est["ideal_expectation_values"]
        noisy_ev = est["expectation_values"]

        # --- Build the figure ---
        fig, (ax_counts, ax_ev) = plt.subplots(1, 2, figsize=(14, 5))

        x = range(len(all_bits))
        w = 0.4
        ax_counts.bar([i - w / 2 for i in x], ideal_freq, width=w,
                      label="exact (simple_execute)", color="#1f77b4")
        ax_counts.bar([i + w / 2 for i in x], noisy_freq, width=w,
                      label="noisy (bridge)", color="#d62728", alpha=0.85)
        ax_counts.set_xticks(list(x))
        ax_counts.set_xticklabels(all_bits, rotation=60, ha="right",
                                  fontfamily="monospace", fontsize=8)
        ax_counts.set_ylabel("probability")
        ax_counts.set_title(
            f"GHZ-{n} on {type(fake_backend).__name__} — "
            f"shot distribution (top 20 / {shots} shots)"
        )
        ax_counts.legend()

        xe = range(len(labels))
        ax_ev.bar([i - w / 2 for i in xe], ideal_ev, width=w,
                  label="exact", color="#1f77b4")
        ax_ev.bar([i + w / 2 for i in xe], noisy_ev, width=w,
                  label="noisy (analytical)", color="#d62728", alpha=0.85)
        ax_ev.set_xticks(list(xe))
        ax_ev.set_xticklabels(labels)
        ax_ev.set_ylim(-1.05, 1.05)
        ax_ev.axhline(0, color="black", linewidth=0.5)
        ax_ev.set_ylabel("<O>")
        ax_ev.set_title(f"GHZ-{n} expectation values")
        ax_ev.legend()

        fig.tight_layout()

        # Save into the repo (so the user can inspect it) and into the
        # test's tmp_path (so the test itself is hermetic).
        out_dir = os.path.join(_HERE, "_plots")
        os.makedirs(out_dir, exist_ok=True)
        repo_path = os.path.join(out_dir, "exact_vs_noisy.png")
        tmp_artifact = tmp_path / "exact_vs_noisy.png"
        fig.savefig(repo_path, dpi=120)
        fig.savefig(tmp_artifact, dpi=120)
        plt.close(fig)

        assert os.path.getsize(repo_path) > 0
        assert tmp_artifact.exists() and tmp_artifact.stat().st_size > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
