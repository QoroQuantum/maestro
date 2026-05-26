"""Unified bridge between quantum backends and maestro.

Dispatches automatically to the IBM or IQM extraction path based on the
backend object passed in.  The downstream noise math is vendor-agnostic and
lives in ``_noise_math.py``.

IBM path
--------
Reads device properties (T1, T2, gate errors, readout errors, optional ZZ
crosstalk) from a ``FakeBackendV2`` or live ``IBMBackend`` via
``backend.properties()``.  See the IBM section of ``read_backend_data`` for
the full physical model.

IQM path
--------
Accepts an ``IQMFakeBackend`` (via its ``error_profile``) or a live
``IQMBackend`` (via ``IQMClient`` calibration APIs).  IQM uses string qubit
labels (``'QB1'``, ``'QB12'``, ``'CR1'``); this bridge maps them to 0-based
integer indices for maestro.  IQM fake profiles report times in nanoseconds;
live backends use seconds — both are normalised to seconds before being
passed to the shared math.

Noise model (both vendors)
--------------------------
The bridge implements the most physically-honest model attainable from the
information each vendor exposes:

1. **T1 amplitude damping** — stochastic Reset → |0⟩ with probability
   γ = 1 − exp(−t / T1), sized to the 1Q gate time.
2. **T2 pure dephasing** — Pauli Z with probability
   p_z = (1 − exp(−t·(1/T2_eff − 1/T1))) / 2, T2_eff = min(T2, 2·T1).
3. **RB residual control error** — incoherent depolarizing channel.
4. **2Q-gate-only delta thermal** — extra T1/T2 infidelity during the
   longer 2Q gate duration.
5. **ZZ crosstalk** (IBM live backends only) — coherent Rz spectator
   rotation via ``set_crosstalk``.
6. **Readout error** — classical confusion matrix.

What's *not* modelled: idle-spectator decoherence, leakage to higher levels,
time-correlated drift, coherent control residuals.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import maestro

from _noise_math import (
    CalibrationData,
    calculate_noise_params,
    _thermal_relax_avg_infid,
    _depol_from_avg_error,
)

MaestroCircuit = maestro.circuits.QuantumCircuit

# ---------------------------------------------------------------------------
# Optional IBM imports (absent when only IQM is installed)
# ---------------------------------------------------------------------------

try:
    from qiskit_ibm_runtime import IBMBackend as _IBMBackend
    from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2 as _FakeBackendV2
    from qiskit_ibm_runtime.models.exceptions import BackendPropertyError as _BackendPropertyError
    _IBM_AVAILABLE = True
except ImportError:
    _IBMBackend = type(None)
    _FakeBackendV2 = type(None)
    _BackendPropertyError = Exception
    _IBM_AVAILABLE = False

# IQM types are detected heuristically (module-based) so no hard import is
# needed here; qiskit-iqm may not be installed for IBM-only users.

# Unit conversion: IQM fake error profiles report times in nanoseconds.
_NS_TO_S = 1e-9


# ===========================================================================
# File-based calibration backend
# ===========================================================================


class _FileBackend:
    """Calibration data loaded from a JSON/YAML file.

    Acts as a backend anywhere the bridge accepts one.  Wraps a
    ``CalibrationData`` object and a ``GenericBackendV2`` built from the
    file's coupling map and basis gates, used exclusively for
    ``qiskit.transpile``.
    """

    def __init__(self, data: CalibrationData, transpile_backend: Any) -> None:
        self._calibration_data = data
        self._transpile_backend = transpile_backend

    @property
    def num_qubits(self) -> int:
        return self._calibration_data.num_qubits


# ===========================================================================
# Vendor detection
# ===========================================================================


def _is_ibm_backend(obj: Any) -> bool:
    return _IBM_AVAILABLE and isinstance(obj, (_FakeBackendV2, _IBMBackend))


def _is_iqm_fake_backend(obj: Any) -> bool:
    return hasattr(obj, "error_profile") and type(obj).__module__.startswith("iqm.")


def _is_iqm_live_backend(obj: Any) -> bool:
    return (
        (hasattr(obj, "_client") or hasattr(obj, "client"))
        and type(obj).__module__.startswith("iqm.")
        and not _is_iqm_fake_backend(obj)
    )


def _detect_vendor(backend: Any) -> str:
    """Return ``'ibm'``, ``'iqm'``, or ``'file'``, or raise an informative error."""
    if isinstance(backend, _FileBackend):
        return "file"
    if _is_ibm_backend(backend):
        return "ibm"
    if _is_iqm_fake_backend(backend) or _is_iqm_live_backend(backend):
        return "iqm"
    # SDK not installed but backend type is still recognisable by module path.
    module = type(backend).__module__ or ""
    if module.startswith("qiskit_ibm_runtime"):
        raise ImportError(
            f"{type(backend).__name__!r} is a qiskit-ibm-runtime backend but "
            "qiskit-ibm-runtime is not installed. "
            "Run: pip install qiskit qiskit-ibm-runtime"
        )
    if module.startswith("iqm."):
        raise ImportError(
            f"{type(backend).__name__!r} is an IQM backend but the IQM SDKs "
            "are not installed. Run: pip install iqm-client qiskit-iqm"
        )
    raise TypeError(
        f"Unsupported backend type {type(backend).__name__!r}. "
        "Expected a qiskit-ibm-runtime FakeBackendV2 / IBMBackend, or an "
        "IQMFakeBackend / IQMBackend."
    )


# ===========================================================================
# IQM-specific helpers
# ===========================================================================


def _name_to_index(name: str, resonator_offset: int = 1000) -> int:
    """Convert an IQM component label (``'QB1'``, ``'CR1'``) to a 0-based int.

    * ``'QB<N>'`` → ``N - 1``
    * ``'CR<N>'`` → ``resonator_offset + N - 1``
    """
    digits = "".join(c for c in name if c.isdigit())
    if not digits:
        raise ValueError(f"Cannot extract index from IQM label {name!r}")
    n = int(digits)
    if name.startswith("CR"):
        return resonator_offset + n - 1
    return n - 1


def _is_resonator(name: str) -> bool:
    return name.startswith("CR")


def _iqm_client_from_backend(backend: Any):
    client = getattr(backend, "client", None) or getattr(backend, "_client", None)
    if client is None:
        raise RuntimeError(
            "Live IQMBackend has no .client or ._client attribute — "
            "cannot fetch calibration data."
        )
    return client


# ===========================================================================
# Calibration extraction — IBM
# ===========================================================================


def _read_ibm_backend(backend: Any) -> CalibrationData:
    """Extract per-qubit noise data from a FakeBackendV2 or live IBMBackend.

    Returns a ``CalibrationData`` with:

    - ``gates_2q`` — dict keyed by qubit pair, mapping
      ``"{gate}_error"`` / ``"{gate}_length"`` for each non-crosstalk 2Q
      basis gate the backend reports.
    - ``crosstalk`` — dict keyed by qubit pair with the
      RB-measured ``rzz``/``zz`` error probabilities (empty for fakes).
    """
    props = backend.properties()
    n = backend.num_qubits

    rows = []
    missing_t1t2: list[int] = []
    for q in range(n):
        try:
            t1 = props.t1(q)
        except (KeyError, AttributeError, _BackendPropertyError):
            t1 = float("nan")
            missing_t1t2.append(q)
        try:
            t2 = props.t2(q)
        except (KeyError, AttributeError, _BackendPropertyError):
            t2 = float("nan")
            if q not in missing_t1t2:
                missing_t1t2.append(q)
        row = {"T1": t1, "T2": t2}
        try:
            row["prob_meas0_prep1"] = props.qubit_property(q, "prob_meas0_prep1")[0]
        except (KeyError, AttributeError, _BackendPropertyError):
            row["prob_meas0_prep1"] = 0.0
        try:
            row["prob_meas1_prep0"] = props.qubit_property(q, "prob_meas1_prep0")[0]
        except (KeyError, AttributeError, _BackendPropertyError):
            row["prob_meas1_prep0"] = 0.0

        gate_errors_1q, gate_lengths_1q = [], []
        for gp in props.gates:
            if len(gp.qubits) == 1 and gp.qubits[0] == q:
                try:
                    gate_errors_1q.append(props.gate_error(gp.gate, q))
                    gate_lengths_1q.append(props.gate_length(gp.gate, q))
                except (AttributeError, KeyError, _BackendPropertyError):
                    continue
        row["gate_error_1q"] = (
            sum(gate_errors_1q) / len(gate_errors_1q) if gate_errors_1q else 0.0
        )
        row["gate_length_1q"] = (
            sum(gate_lengths_1q) / len(gate_lengths_1q) if gate_lengths_1q else 0.0
        )
        rows.append(row)

    if missing_t1t2:
        warnings.warn(
            f"Backend reports no T1/T2 for {len(missing_t1t2)} qubit(s) "
            f"(e.g. {missing_t1t2[:5]}); treating as NaN (zero thermal "
            "contribution). These are likely flagged unusable by the device.",
            stacklevel=3,
        )

    gates_2q: dict[tuple[int, int], dict] = {}
    crosstalk: dict[tuple[int, int], float] = {}
    for gp in props.gates:
        if len(gp.qubits) != 2:
            continue
        pair: tuple[int, int] = (gp.qubits[0], gp.qubits[1])
        try:
            err = props.gate_error(gp.gate, list(pair))
            length = props.gate_length(gp.gate, list(pair))
        except (AttributeError, KeyError, _BackendPropertyError):
            continue
        if gp.gate.lower() in ("rzz", "zz"):
            crosstalk[pair] = err
        else:
            entry = gates_2q.setdefault(pair, {})
            entry[f"{gp.gate}_error"] = err
            entry[f"{gp.gate}_length"] = length
    return CalibrationData(
        num_qubits=n,
        qubits={q: rows[q] for q in range(n)},
        gates_2q=gates_2q,
        crosstalk=crosstalk,
    )


# ===========================================================================
# Calibration extraction — IQM fake backend
# ===========================================================================


def _read_iqm_fake(backend: Any) -> CalibrationData:
    """Build calibration data from an ``IQMFakeBackend.error_profile``.

    Skips computational resonators (``CR*``) and ``move`` gates.  T1/T2 and
    gate durations are converted from nanoseconds to seconds.
    """
    ep = backend.error_profile

    qubit_names = [n for n in ep.t1s.keys() if not _is_resonator(n)]
    name_to_idx = {n: _name_to_index(n) for n in qubit_names}

    rows = []
    for name in qubit_names:
        t1 = float(ep.t1s.get(name, 0.0)) * _NS_TO_S
        t2 = float(ep.t2s.get(name, 0.0)) * _NS_TO_S

        # IQM convention: key '0' → prob_meas1_prep0, key '1' → prob_meas0_prep1.
        re_dict = ep.readout_errors.get(name, {})
        p_meas1_prep0 = float(re_dict.get("0", 0.0))
        p_meas0_prep1 = float(re_dict.get("1", 0.0))

        gate_errs, gate_durs = [], []
        for gname, qmap in ep.single_qubit_gate_depolarizing_error_parameters.items():
            if name not in qmap:
                continue
            gate_errs.append(float(qmap[name]))
            dur_ns = ep.single_qubit_gate_durations.get(gname)
            if dur_ns is not None:
                gate_durs.append(float(dur_ns) * _NS_TO_S)

        rows.append({
            "T1": t1,
            "T2": t2,
            "prob_meas0_prep1": p_meas0_prep1,
            "prob_meas1_prep0": p_meas1_prep0,
            "gate_error_1q": sum(gate_errs) / len(gate_errs) if gate_errs else 0.0,
            "gate_length_1q": sum(gate_durs) / len(gate_durs) if gate_durs else 0.0,
        })

    qubits = {name_to_idx[n]: rows[i] for i, n in enumerate(qubit_names)}

    gates_2q: dict[tuple[int, int], dict] = {}
    for gname, pair_dict in ep.two_qubit_gate_depolarizing_error_parameters.items():
        if gname.lower() == "move":
            continue
        dur_ns = ep.two_qubit_gate_durations.get(gname)
        for pair_names, err in pair_dict.items():
            if any(_is_resonator(n) for n in pair_names):
                continue
            idxs: tuple[int, int] = (name_to_idx[pair_names[0]], name_to_idx[pair_names[1]])
            entry = gates_2q.setdefault(idxs, {})
            entry[f"{gname}_error"] = float(err)
            if dur_ns is not None:
                entry[f"{gname}_length"] = float(dur_ns) * _NS_TO_S

    return CalibrationData(
        num_qubits=len(qubits),
        qubits=qubits,
        gates_2q=gates_2q,
        crosstalk={},
    )


# ===========================================================================
# Calibration extraction — IQM live backend
# ===========================================================================


def _read_iqm_live(backend: Any) -> CalibrationData:
    """Build calibration data from a live ``IQMBackend``'s current calibration.

    Uses ``IQMClient.get_dynamic_quantum_architecture()`` for the gate
    inventory and ``IQMClient.get_calibration_quality_metrics()`` for
    per-gate fidelities/durations and per-qubit coherence times.  Live IQM
    data is already in SI units — no conversion needed.
    """
    client = _iqm_client_from_backend(backend)
    dqa = client.get_dynamic_quantum_architecture()
    obs = client.get_calibration_quality_metrics()

    qubit_names = [n for n in dqa.qubits if not _is_resonator(n)]
    name_to_idx = {n: _name_to_index(n) for n in qubit_names}

    try:
        t1_map, t2_map = obs.get_coherence_times(qubit_names)
    except Exception as exc:
        warnings.warn(
            f"Could not fetch coherence times from IQM: {exc}; using zeros.",
            stacklevel=3,
        )
        t1_map, t2_map = {}, {}

    one_q_gates: list[tuple[str, str, tuple[str, ...]]] = []
    two_q_gates: list[tuple[str, str, tuple[str, ...]]] = []
    measure_gates: list[tuple[str, str, tuple[str, ...]]] = []
    for gate_name, gate_info in dqa.gates.items():
        is_measure = "measur" in gate_name.lower()
        for impl_name, impl_info in gate_info.implementations.items():
            for locus in impl_info.loci:
                tup = (gate_name, impl_name, tuple(locus))
                if is_measure:
                    measure_gates.append(tup)
                elif len(locus) == 1:
                    one_q_gates.append(tup)
                elif len(locus) == 2:
                    two_q_gates.append(tup)

    rows = []
    for name in qubit_names:
        t1 = float(t1_map.get(name, 0.0))
        t2 = float(t2_map.get(name, 0.0))

        p_meas1_prep0 = 0.0
        p_meas0_prep1 = 0.0
        for (g, impl, locus) in measure_gates:
            if name not in locus:
                continue
            res = obs.get_measure_errors(g, impl, locus)
            if res is None:
                continue
            e0, e1 = res
            p_meas1_prep0 = max(p_meas1_prep0, float(e0))
            p_meas0_prep1 = max(p_meas0_prep1, float(e1))

        gate_errs, gate_durs = [], []
        for (g, impl, locus) in one_q_gates:
            if name not in locus:
                continue
            fid = obs.get_gate_fidelity(g, impl, locus)
            if fid is not None:
                gate_errs.append(1.0 - float(fid))
            dur = obs.get_gate_duration(g, impl, locus)
            if dur is not None:
                gate_durs.append(float(dur))

        rows.append({
            "T1": t1,
            "T2": t2,
            "prob_meas0_prep1": p_meas0_prep1,
            "prob_meas1_prep0": p_meas1_prep0,
            "gate_error_1q": sum(gate_errs) / len(gate_errs) if gate_errs else 0.0,
            "gate_length_1q": sum(gate_durs) / len(gate_durs) if gate_durs else 0.0,
        })

    qubits = {name_to_idx[n]: rows[i] for i, n in enumerate(qubit_names)}

    gates_2q: dict[tuple[int, int], dict] = {}
    for (g, impl, locus) in two_q_gates:
        if g.lower() == "move":
            continue
        if any(_is_resonator(n) for n in locus):
            continue
        fid = obs.get_gate_fidelity(g, impl, locus)
        if fid is None:
            continue
        idxs: tuple[int, int] = (name_to_idx[locus[0]], name_to_idx[locus[1]])
        entry = gates_2q.setdefault(idxs, {})
        entry[f"{g}_error"] = max(0.0, 1.0 - float(fid))
        dur = obs.get_gate_duration(g, impl, locus)
        if dur is not None:
            entry[f"{g}_length"] = float(dur)

    return CalibrationData(
        num_qubits=len(qubits),
        qubits=qubits,
        gates_2q=gates_2q,
        crosstalk={},
    )


# ===========================================================================
# Public API
# ===========================================================================


def read_backend_data(backend: Any) -> CalibrationData:
    """Extract per-qubit noise data from a backend or calibration file.

    Dispatches to the appropriate vendor path automatically.  The returned
    ``CalibrationData`` always has the same schema (per-qubit keys: T1, T2,
    prob_meas0_prep1, prob_meas1_prep0, gate_error_1q, gate_length_1q;
    plus ``gates_2q`` and ``crosstalk`` dicts) so the shared
    ``calculate_noise_params`` logic applies unchanged for all vendors and
    file-based inputs.
    """
    if isinstance(backend, _FileBackend):
        return backend._calibration_data
    vendor = _detect_vendor(backend)
    if vendor == "ibm":
        return _read_ibm_backend(backend)
    # IQM — dispatch to fake vs. live sub-reader.
    if _is_iqm_fake_backend(backend):
        return _read_iqm_fake(backend)
    return _read_iqm_live(backend)


def maestro_noise_from_backend(backend: Any) -> maestro.NoiseModel:
    """Build a maestro NoiseModel from any supported IBM or IQM backend.

    Per qubit: T1 amplitude damping, T2 pure dephasing, 1Q incoherent
    depolarizing residual, 2Q-gate-only delta thermal, asymmetric readout
    error.  Per 2Q pair: correlated depolarizing on the native 2Q gate.
    Per crosstalk-characterised pair (IBM live only): coherent Rz spectator
    rotation.
    """
    backend_df = read_backend_data(backend)
    params = calculate_noise_params(backend_df)

    nm = maestro.NoiseModel()
    for q, p in params["qubits"].items():
        if p["t1_time"] > 0 and p["gate_time_1q"] > 0:
            nm.set_t1_from_time(q, p["gate_time_1q"], p["t1_time"])
        if p["p_dephasing"] > 0:
            nm.set_dephasing(q, p["p_dephasing"])
        if p["p_1q_residual"] > 0:
            nm.set_1q_gate_depolarizing(q, p["p_1q_residual"])
        if p["delta_2q_per_q"] > 0:
            nm.set_2q_gate_depolarizing(q, p["delta_2q_per_q"])
        if p["p_meas1_prep0"] > 0 or p["p_meas0_prep1"] > 0:
            nm.set_readout_error(q, p["p_meas1_prep0"], p["p_meas0_prep1"])
    for (q1, q2), gp in params["gates_2q"].items():
        if gp["p"] > 0:
            nm.set_2q_depolarizing(q1, q2, gp["p"])
    for (q1, q2), strength in params["crosstalk"].items():
        if strength > 0:
            nm.set_crosstalk(q1, q2, strength)
    return nm


def _transpile_target(backend: Any) -> Any:
    """Return the object to pass to ``qiskit.transpile``.

    For a ``_FileBackend`` this is the inner ``GenericBackendV2``; for all
    other backends it is the backend itself.
    """
    if isinstance(backend, _FileBackend):
        return backend._transpile_backend
    return backend


def transpile_to_backend(
    circuit: "str | Any | MaestroCircuit",
    backend: Any,
) -> "Any":
    """Transpile a circuit to the backend's coupling map and basis gates.

    IQM backends require ``LEGACY_CUSTOM_INSTRUCTIONS`` when loading QASM
    strings; IBM backends use the standard loader.

    Returns a ``qiskit.QuantumCircuit``.  Qiskit is imported lazily so that
    ``import noise_bridge`` does not require qiskit to be installed.
    """
    import qiskit
    import qiskit.qasm2

    vendor = _detect_vendor(backend)
    use_legacy = vendor == "iqm"

    if isinstance(circuit, str):
        qasm_str = circuit
    elif isinstance(circuit, MaestroCircuit):
        qasm_str = circuit.to_qasm()
    elif isinstance(circuit, qiskit.QuantumCircuit):
        return qiskit.transpile(circuit, backend=_transpile_target(backend))
    else:
        raise TypeError(f"Unsupported circuit type: {type(circuit)}")

    if use_legacy:
        qc = qiskit.qasm2.loads(
            qasm_str,
            custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
        )
    else:
        qc = qiskit.qasm2.loads(qasm_str)
    return qiskit.transpile(qc, backend=_transpile_target(backend))


def bridge_core(
    circuit: Any, backend: Any,
) -> tuple[MaestroCircuit, maestro.NoiseModel]:
    """Transpile the circuit and build the noise model in one step.

    The transpiled circuit must have ``num_qubits == backend.num_qubits`` so
    QASM qubit indices align with the noise model's physical-qubit keys.
    Execution happens locally on maestro — the backend is only consulted for
    calibration data.
    """
    transpiled_qiskit = transpile_to_backend(circuit, backend)
    if transpiled_qiskit.num_qubits != backend.num_qubits:
        raise ValueError(
            f"Transpiled circuit has {transpiled_qiskit.num_qubits} qubits "
            f"but backend has {backend.num_qubits}. QASM qubit indices won't "
            f"align with the noise model's physical-qubit keys. Re-transpile "
            f"with the backend's full layout."
        )
    import qiskit.qasm2
    qasm_str = qiskit.qasm2.dumps(transpiled_qiskit)
    transpiled_maestro = maestro.QasmToCirc().parse_and_translate(qasm_str)
    noise_model = maestro_noise_from_backend(backend)
    return transpiled_maestro, noise_model


# ===========================================================================
# Wrappers for maestro execute/estimate calls
# ===========================================================================


def execute_backend_noise_model(
    circuit,
    backend: Any,
    config: maestro.SimulatorConfig | None = None,
    shots: int = 1000,
) -> dict:
    """Run ``maestro.noisy_execute`` with the backend's noise model."""
    transpiled_circuit, noise_model = bridge_core(circuit, backend)
    return maestro.noisy_execute(
        transpiled_circuit,
        noise_model,
        config or maestro.SimulatorConfig(),
        shots=shots,
    )


def estimate_backend_noise_model(
    circuit,
    observables,
    backend: Any,
    config: maestro.SimulatorConfig | None = None,
) -> dict:
    """Run ``maestro.noisy_estimate`` with the backend's noise model."""
    transpiled_circuit, noise_model = bridge_core(circuit, backend)
    return maestro.noisy_estimate(
        transpiled_circuit,
        observables,
        noise_model,
        config or maestro.SimulatorConfig(),
    )


def estimate_backend_noise_model_montecarlo(
    circuit,
    observables,
    backend: Any,
    config: maestro.SimulatorConfig | None = None,
    noise_realizations: int = 100,
    seed: int = 42,
) -> dict:
    """Run ``maestro.noisy_estimate_montecarlo`` with the backend's noise model."""
    transpiled_circuit, noise_model = bridge_core(circuit, backend)
    return maestro.noisy_estimate_montecarlo(
        transpiled_circuit,
        observables,
        noise_model,
        noise_realizations=noise_realizations,
        config=config or maestro.SimulatorConfig(),
        seed=seed,
    )


def load_calibration_file(path: str) -> _FileBackend:
    """Load a JSON or YAML calibration file and return a backend-compatible
    object usable anywhere the bridge accepts a backend.

    The file must contain ``num_qubits`` and a ``qubits`` section.  All other
    sections (``coupling_map``, ``basis_gates``, ``gates_2q``, ``crosstalk``)
    are optional.  When ``coupling_map`` is absent, a fully-connected topology
    is assumed.

    Schema (JSON shown; YAML is identical in structure)::

        {
          "num_qubits": 5,
          "coupling_map": [[0,1],[1,0],[1,2],[2,1]],   // optional
          "basis_gates": ["cx","rz","sx","x","measure"], // optional
          "qubits": {
            "0": {"T1": 1e-4, "T2": 5e-5,
                  "gate_error_1q": 1e-3, "gate_length_1q": 5e-8,
                  "prob_meas0_prep1": 0.01, "prob_meas1_prep0": 0.02}
          },
          "gates_2q": {                                 // optional
            "0,1": {"cx_error": 5e-3, "cx_length": 2e-7}
            // or agnostic: "0,1": {"error": 5e-3, "length": 2e-7}
          },
          "crosstalk": {"0,1": 0.001}                  // optional
        }

    Usage::

        backend = br.load_calibration_file("my_device.json")
        result  = br.execute_backend_noise_model(qc, backend, shots=4000)
        nm      = br.maestro_noise_from_backend(backend)
    """
    import json
    import os

    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path) as fh:
            data = json.load(fh)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required to load YAML calibration files.  "
                "Run: pip install pyyaml"
            )
        with open(path) as fh:
            data = yaml.safe_load(fh)
    else:
        raise ValueError(
            f"Unsupported calibration file extension {ext!r}.  "
            "Use .json, .yaml, or .yml."
        )

    # ── Validate required fields ──────────────────────────────────────────
    if "num_qubits" not in data:
        raise ValueError("Calibration file must contain 'num_qubits'.")
    if "qubits" not in data or not data["qubits"]:
        raise ValueError("Calibration file must contain a non-empty 'qubits' section.")

    num_qubits: int = int(data["num_qubits"])
    if num_qubits <= 0:
        raise ValueError(f"'num_qubits' must be positive, got {num_qubits}.")

    # ── Per-qubit DataFrame ───────────────────────────────────────────────
    _QUBIT_COLS = ("T1", "T2", "gate_error_1q", "gate_length_1q",
                   "prob_meas0_prep1", "prob_meas1_prep0")
    rows: dict[int, dict] = {}
    for key, vals in data["qubits"].items():
        try:
            q = int(key)
        except (ValueError, TypeError):
            raise ValueError(
                f"Qubit key {key!r} in 'qubits' must be an integer index."
            )
        if not (0 <= q < num_qubits):
            raise ValueError(
                f"Qubit index {q} is out of range for num_qubits={num_qubits}."
            )
        rows[q] = {col: float(vals[col]) for col in _QUBIT_COLS if col in vals}

    _nan = float("nan")
    qubits: dict[int, dict[str, float]] = {
        q: {col: rows.get(q, {}).get(col, _nan) for col in _QUBIT_COLS}
        for q in range(num_qubits)
    }

    # ── 2Q gate data ──────────────────────────────────────────────────────
    gates_2q: dict[tuple[int, int], dict] = {}
    for pair_str, vals in (data.get("gates_2q") or {}).items():
        try:
            a, b = (int(x.strip()) for x in pair_str.split(","))
        except (ValueError, TypeError):
            raise ValueError(
                f"'gates_2q' key {pair_str!r} must be in 'q0,q1' format."
            )
        entry: dict = {}
        for k, v in vals.items():
            # Normalise agnostic keys: "error" → "gate_error", "length" → "gate_length"
            if k == "error":
                entry["gate_error"] = float(v)
            elif k == "length":
                entry["gate_length"] = float(v)
            else:
                entry[k] = float(v)
        gates_2q[(a, b)] = entry

    # ── Crosstalk ─────────────────────────────────────────────────────────
    crosstalk: dict[tuple[int, int], float] = {}
    for pair_str, strength in (data.get("crosstalk") or {}).items():
        try:
            a, b = (int(x.strip()) for x in pair_str.split(","))
        except (ValueError, TypeError):
            raise ValueError(
                f"'crosstalk' key {pair_str!r} must be in 'q0,q1' format."
            )
        crosstalk[(a, b)] = float(strength)

    # ── Transpilation backend (GenericBackendV2) ──────────────────────────
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    raw_cm = data.get("coupling_map")
    if raw_cm is not None:
        coupling_map = CouplingMap(raw_cm)
    else:
        coupling_map = CouplingMap.from_full(num_qubits)

    basis_gates = data.get("basis_gates", ["cx", "rz", "sx", "x", "measure"])

    # GenericBackendV2 only accepts gates from the standard library.
    # Control-flow and timing meta-ops (present on live IBM backends) must be
    # stripped — they are not physical gates used during transpilation.
    _NON_PHYSICAL = frozenset({
        "for_loop", "if_else", "switch_case", "while_loop",
        "break_loop", "continue_loop", "delay",
    })
    transpile_gates = [g for g in basis_gates if g not in _NON_PHYSICAL] or [
        "cx", "rz", "sx", "x", "measure"
    ]

    transpile_backend = GenericBackendV2(
        num_qubits=num_qubits,
        basis_gates=transpile_gates,
        coupling_map=coupling_map,
    )

    cal = CalibrationData(
        num_qubits=num_qubits,
        qubits=qubits,
        gates_2q=gates_2q,
        crosstalk=crosstalk,
    )
    return _FileBackend(cal, transpile_backend)


def _backend_to_dict(backend: Any) -> dict:
    """Serialise a real backend's calibration data to the file schema dict.

    Reads the backend via ``read_backend_data`` and packages the result into
    the same JSON/YAML structure produced by a hand-written calibration file.
    """
    cal = read_backend_data(backend)
    num_qubits: int = backend.num_qubits

    # coupling map — works for IBM FakeBackendV2 and IQM backends alike
    try:
        coupling_map: list | None = [list(e) for e in backend.coupling_map.get_edges()]
    except Exception:
        coupling_map = None

    # basis gates
    try:
        basis_gates: list = sorted(backend.operation_names)
    except Exception:
        basis_gates = ["cx", "rz", "sx", "x", "measure"]

    # per-qubit data — skip NaN values so the file stays compact
    qubits: dict = {}
    for q, row in cal.qubits.items():
        entry = {col: v for col, v in row.items() if not math.isnan(v)}
        if entry:
            qubits[str(q)] = entry

    # 2Q gate data
    gates_2q: dict = {}
    for (q1, q2), gdata in cal.gates_2q.items():
        gates_2q[f"{q1},{q2}"] = {k: float(v) for k, v in gdata.items()}

    # crosstalk
    crosstalk: dict = {}
    for (q1, q2), strength in cal.crosstalk.items():
        crosstalk[f"{q1},{q2}"] = float(strength)

    result: dict = {"num_qubits": num_qubits, "qubits": qubits}
    if coupling_map is not None:
        result["coupling_map"] = coupling_map
    if basis_gates:
        result["basis_gates"] = basis_gates
    if gates_2q:
        result["gates_2q"] = gates_2q
    if crosstalk:
        result["crosstalk"] = crosstalk
    return result


def save_calibration_file(backend: Any, path: str) -> None:
    """Save a backend's calibration data to a JSON or YAML file.

    The saved file can be reloaded with :func:`load_calibration_file` and
    used as a drop-in replacement for the backend object, eliminating
    repeated API queries for the same device characterisation.

    No-op when ``backend`` is already a ``_FileBackend`` (nothing to save).

    Args:
        backend: Any IBM or IQM backend supported by the bridge.
        path: Destination file path.  Extension determines format: ``.json``,
              ``.yaml``, or ``.yml``.

    Example::

        # Save once (queries the IBM API)
        br.save_calibration_file(ibm_backend, "ibm_eagle_2026-05-26.json")

        # Reload later (no API query)
        backend = br.load_calibration_file("ibm_eagle_2026-05-26.json")
        result  = br.execute_backend_noise_model(qc, backend, shots=4000)
    """
    if isinstance(backend, _FileBackend):
        return

    import json
    import os

    path = os.fspath(path)
    ext = os.path.splitext(path)[1].lower()
    data = _backend_to_dict(backend)

    if ext == ".json":
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "pyyaml is required to save YAML calibration files.  "
                "Run: pip install pyyaml"
            )
        with open(path, "w") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False)
    else:
        raise ValueError(
            f"Unsupported calibration file extension {ext!r}.  "
            "Use .json, .yaml, or .yml."
        )