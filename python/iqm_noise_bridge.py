"""Bridge between IQM backends and maestro.

Mirrors ``ibm_noise_bridge.py``: reads device properties (T1, T2, gate
errors/durations, readout errors) from either an **offline IQM fake
backend** (``IQMFakeApollo``, ``IQMFakeAdonis``, etc.) or a **live IQM
backend** obtained from ``IQMProvider``, builds an equivalent
``maestro.NoiseModel``, transpiles circuits to the backend's coupling
map / basis gates via ``qiskit-iqm``'s transpiler, and exposes the same
wrappers around ``maestro.noisy_execute`` / ``maestro.noisy_estimate``.

The actual noise math (T1/T2 → Pauli probabilities, RB-residual
allocation, etc.) is **shared with the IBM bridge** — only
``read_backend_data`` differs, because the upstream data source has a
different shape.

Differences from IBM
--------------------

* **Qubit labels.** IQM uses string labels like ``'QB1'``, ``'QB12'``;
  Star-architecture devices (Deneb) also expose computational
  resonators (``'CR1'``). Maestro's noise model uses integer qubit
  indices. The bridge maps ``'QB<N>'`` → ``N − 1`` and ``'CR<N>'`` →
  a high index above the qubits (so the resonator's calibration data
  doesn't collide with qubit indices). MOVE gates are dropped from the
  2Q gate set since they're not modelled as Pauli channels.
* **Units.** IQM's *fake* error profile is in **nanoseconds**; the live
  ObservationFinder reports durations in **seconds**. ``read_backend_data``
  normalizes to seconds (the unit ``calculate_noise_params`` expects).
* **No ZZ crosstalk.** Neither the fake error profile nor the live
  observation set exposes ``rzz``/``zz`` characterization. The
  crosstalk dict is empty; ``set_crosstalk`` is never called.
* **1Q gate set.** IQM's native single-qubit gate is ``prx``
  (parametric R) rather than IBM's ``sx``/``rz`` decomposition. The
  bridge's loop already averages over all 1Q gates touching each
  qubit, so ``prx`` works without code changes. The 2Q gate is
  ``cz`` (instead of IBM's ``cx``); the existing 2Q-gate iteration
  picks it up via its ``*_error`` key.

Authentication
--------------

Live IQM access is via ``iqm.qiskit_iqm.IQMProvider``. The user creates
the backend (URL + auth) and passes it in; the bridge doesn't handle
authentication itself, mirroring the IBM bridge.
"""

from __future__ import annotations

import warnings
from typing import Any

import qiskit
import qiskit.qasm2
import pandas as pd
import maestro

# Shared, vendor-agnostic noise math (no IBM-specific imports).
from _noise_math import calculate_noise_params

MaestroCircuit = maestro.circuits.QuantumCircuit

# We import IQM types lazily inside the helpers so this module can still
# be imported even if iqm-client / qiskit-iqm isn't installed (for users
# who only want the IBM bridge).

# Unit conversion: IQM fake error profiles use ns; maestro math is in s.
_NS_TO_S = 1e-9


# ---------------------------------------------------------------------------
# Backend type detection
# ---------------------------------------------------------------------------


def _is_iqm_fake_backend(obj: Any) -> bool:
    """Heuristic for ``IQMFakeBackend`` subclasses."""
    return hasattr(obj, "error_profile") and type(obj).__module__.startswith(
        "iqm."
    )


def _is_iqm_live_backend(obj: Any) -> bool:
    """Heuristic for live ``IQMBackend`` (from ``iqm.qiskit_iqm``).

    Distinguishes from the fake by presence of a ``client`` attribute
    (set in ``IQMBackend.__init__``).
    """
    return (hasattr(obj, "_client") or hasattr(obj, "client")) and \
        type(obj).__module__.startswith("iqm.") and \
        not _is_iqm_fake_backend(obj)


def _iqm_client_from_backend(backend: Any):
    """Pull the underlying ``IQMClient`` from a live IQMBackend."""
    client = getattr(backend, "client", None) or getattr(backend, "_client",
                                                         None)
    if client is None:
        raise RuntimeError(
            "Live IQMBackend has no .client or ._client attribute — "
            "cannot fetch calibration data."
        )
    return client


# ---------------------------------------------------------------------------
# Qubit-name ↔ integer-index mapping
# ---------------------------------------------------------------------------


def _name_to_index(name: str, resonator_offset: int = 1000) -> int:
    """Convert an IQM component label (``'QB1'``, ``'CR1'``) to a
    0-based integer index suitable for maestro's noise model.

    * ``'QB<N>'`` → ``N - 1``
    * ``'CR<N>'`` → ``resonator_offset + N - 1`` (kept far above qubit
      indices so the two index spaces never collide).
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


# ---------------------------------------------------------------------------
# Calibration extraction — fake backend
# ---------------------------------------------------------------------------


def _read_iqm_fake(backend: Any) -> pd.DataFrame:
    """Build a DataFrame from an ``IQMFakeBackend.error_profile``.

    Skips computational resonators (Deneb-style ``CR*``) and ``move``
    gates. T1/T2 and gate durations are converted from nanoseconds to
    seconds.
    """
    ep = backend.error_profile

    # Qubit labels appearing in t1s (excluding CR* resonators).
    qubit_names = [n for n in ep.t1s.keys() if not _is_resonator(n)]
    name_to_idx = {n: _name_to_index(n) for n in qubit_names}

    rows = []
    for name in qubit_names:
        t1 = float(ep.t1s.get(name, 0.0)) * _NS_TO_S
        t2 = float(ep.t2s.get(name, 0.0)) * _NS_TO_S

        # readout_errors[qubit] = {'0': p_meas1_prep0, '1': p_meas0_prep1}.
        # IQM's documented convention (see qiskit-iqm fake_backends docs):
        # key '0' is the probability of error when the qubit was prepared
        # in |0⟩ (i.e., prob_meas1_prep0), and key '1' is the probability
        # of error when prepared in |1⟩ (i.e., prob_meas0_prep1).
        re_dict = ep.readout_errors.get(name, {})
        p_meas1_prep0 = float(re_dict.get("0", 0.0))
        p_meas0_prep1 = float(re_dict.get("1", 0.0))

        # 1Q gate error / duration: average over all 1Q gates that touch
        # this qubit (IQM's native is ``prx``; some backends may add
        # others). Durations in the fake profile are uniform per gate,
        # not per qubit.
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
            "gate_error_1q": (sum(gate_errs) / len(gate_errs)
                              if gate_errs else 0.0),
            "gate_length_1q": (sum(gate_durs) / len(gate_durs)
                               if gate_durs else 0.0),
        })

    df = pd.DataFrame(rows, index=[name_to_idx[n] for n in qubit_names])
    df.index.name = "qubit"
    df.sort_index(inplace=True)

    # 2Q gate data: keyed by integer-index pair. Skip MOVE (not a
    # Pauli-channel-modelable gate) and resonator-coupled entries.
    gates_2q: dict[tuple[int, int], dict] = {}
    for gname, pair_dict in ep.two_qubit_gate_depolarizing_error_parameters.items():
        if gname.lower() == "move":
            continue
        dur_ns = ep.two_qubit_gate_durations.get(gname)
        for pair_names, err in pair_dict.items():
            # pair_names is a tuple of qubit-label strings.
            if any(_is_resonator(n) for n in pair_names):
                continue
            idxs = tuple(name_to_idx[n] for n in pair_names)
            entry = gates_2q.setdefault(idxs, {})
            entry[f"{gname}_error"] = float(err)
            if dur_ns is not None:
                entry[f"{gname}_length"] = float(dur_ns) * _NS_TO_S

    df.attrs["gates_2q"] = gates_2q
    df.attrs["crosstalk"] = {}   # IQM fake doesn't expose ZZ crosstalk
    return df


# ---------------------------------------------------------------------------
# Calibration extraction — live backend
# ---------------------------------------------------------------------------


def _read_iqm_live(backend: Any) -> pd.DataFrame:
    """Build a DataFrame from a live ``IQMBackend``'s current calibration.

    Uses ``IQMClient.get_dynamic_quantum_architecture()`` for the gate
    inventory and ``IQMClient.get_calibration_quality_metrics()`` (which
    returns an ``ObservationFinder``) for per-gate fidelities/durations
    and per-qubit coherence times. Live IQM data is already in SI units
    (seconds, dimensionless) — no unit conversion needed.
    """
    client = _iqm_client_from_backend(backend)
    dqa = client.get_dynamic_quantum_architecture()
    obs = client.get_calibration_quality_metrics()

    qubit_names = [n for n in dqa.qubits if not _is_resonator(n)]
    name_to_idx = {n: _name_to_index(n) for n in qubit_names}

    # Coherence times come back as two dicts keyed by qubit label.
    try:
        t1_map, t2_map = obs.get_coherence_times(qubit_names)
    except Exception as exc:  # pragma: no cover - depends on remote data
        warnings.warn(
            f"Could not fetch coherence times from IQM: {exc}; "
            "using zeros.", stacklevel=2,
        )
        t1_map, t2_map = {}, {}

    # Iterate over the DQA's gate inventory once to bin gates by arity.
    one_q_gates: list[tuple[str, str, tuple[str, ...]]] = []   # (gate, impl, locus)
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

        # Readout errors: query the measure gate(s) for this qubit.
        # get_measure_errors returns (prob_meas1_prep0, prob_meas0_prep1)
        # — same key ordering as the fake's '0'/'1' convention.
        p_meas1_prep0 = 0.0
        p_meas0_prep1 = 0.0
        for (g, impl, locus) in measure_gates:
            if name not in locus:
                continue
            res = obs.get_measure_errors(g, impl, locus)
            if res is None:
                continue
            e0, e1 = res
            # Take the worst (highest) reading across implementations.
            p_meas1_prep0 = max(p_meas1_prep0, float(e0))
            p_meas0_prep1 = max(p_meas0_prep1, float(e1))

        # 1Q gate error/duration: average over all 1Q gates on this qubit.
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
            "gate_error_1q": (sum(gate_errs) / len(gate_errs)
                              if gate_errs else 0.0),
            "gate_length_1q": (sum(gate_durs) / len(gate_durs)
                               if gate_durs else 0.0),
        })

    df = pd.DataFrame(rows, index=[name_to_idx[n] for n in qubit_names])
    df.index.name = "qubit"
    df.sort_index(inplace=True)

    # 2Q gate data
    gates_2q: dict[tuple[int, int], dict] = {}
    for (g, impl, locus) in two_q_gates:
        if g.lower() == "move":
            continue
        if any(_is_resonator(n) for n in locus):
            continue
        fid = obs.get_gate_fidelity(g, impl, locus)
        if fid is None:
            continue
        idxs = tuple(name_to_idx[n] for n in locus)
        entry = gates_2q.setdefault(idxs, {})
        entry[f"{g}_error"] = max(0.0, 1.0 - float(fid))
        dur = obs.get_gate_duration(g, impl, locus)
        if dur is not None:
            entry[f"{g}_length"] = float(dur)

    df.attrs["gates_2q"] = gates_2q
    df.attrs["crosstalk"] = {}
    return df


# ---------------------------------------------------------------------------
# Public API — mirrors ibm_noise_bridge.py
# ---------------------------------------------------------------------------


def read_backend_data(backend: Any) -> pd.DataFrame:
    """Extract per-qubit noise data from an IQM backend into a DataFrame.

    Accepts both fake (``IQMFakeApollo``, ``IQMFakeAdonis``, …) and live
    (``IQMBackend`` from ``IQMProvider``) instances. The returned
    DataFrame has the same shape as the IBM bridge's, so the shared
    ``calculate_noise_params`` logic applies unchanged.
    """
    if _is_iqm_fake_backend(backend):
        return _read_iqm_fake(backend)
    if _is_iqm_live_backend(backend):
        return _read_iqm_live(backend)
    raise TypeError(
        f"Unsupported backend type for IQM bridge: "
        f"{type(backend).__name__!r}. Expected an IQMFakeBackend or live "
        f"IQMBackend."
    )


def maestro_noise_from_backend(backend: Any) -> maestro.NoiseModel:
    """Build a maestro NoiseModel from an IQM fake or live backend.

    Applies, per qubit: T1 amplitude damping, T2 pure dephasing, 1Q
    incoherent depolarizing residual, 2Q-gate-only delta thermal, and
    asymmetric readout error. Per 2Q pair: correlated depolarizing on
    the native ``cz`` gate. (IQM doesn't expose ZZ crosstalk; that
    channel stays empty.)
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


# ----------------------------------------------------------
# Transpilation + bridge_core
# ----------------------------------------------------------


def transpile_to_backend(
    circuit: "str | qiskit.QuantumCircuit | MaestroCircuit",
    backend: Any,
) -> qiskit.QuantumCircuit:
    """Transpile a circuit to the IQM backend's coupling map and native
    basis (``prx`` + ``cz``, no ``cx``/``sx``/``rz``).

    Uses ``qiskit.transpile`` against the IQM backend's target, which
    qiskit-iqm wires up so the native gate set is applied automatically.
    """
    if isinstance(circuit, str):
        qc = qiskit.qasm2.loads(
            circuit,
            custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
        )
    elif isinstance(circuit, qiskit.QuantumCircuit):
        qc = circuit
    elif isinstance(circuit, MaestroCircuit):
        qc = qiskit.qasm2.loads(
            circuit.to_qasm(),
            custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
        )
    else:
        raise TypeError(f"Unsupported circuit type: {type(circuit)}")
    return qiskit.transpile(qc, backend=backend)


def bridge_core(
    circuit: Any, backend: Any,
) -> tuple[MaestroCircuit, maestro.NoiseModel]:
    """Transpile the circuit and build the IQM noise model in one step.

    The transpiled circuit must have ``num_qubits == backend.num_qubits``,
    matching the IBM bridge's invariant — qiskit-iqm's transpiler
    respects this by default.
    """
    transpiled_qiskit = transpile_to_backend(circuit, backend)
    if transpiled_qiskit.num_qubits != backend.num_qubits:
        raise ValueError(
            f"Transpiled circuit has {transpiled_qiskit.num_qubits} qubits "
            f"but backend has {backend.num_qubits}. QASM qubit indices "
            f"won't align with the noise model's physical-qubit keys."
        )
    qasm_str = qiskit.qasm2.dumps(transpiled_qiskit)
    transpiled_maestro = maestro.QasmToCirc().parse_and_translate(qasm_str)
    noise_model = maestro_noise_from_backend(backend)
    return transpiled_maestro, noise_model


# ----------------------------------------------------------
# Wrappers — same shape as ibm_noise_bridge.py
# ----------------------------------------------------------


def execute_backend_noise_model(
    circuit,
    backend: Any,
    config: maestro.SimulatorConfig | None = None,
    shots: int = 1000,
) -> dict:
    """Wraps ``maestro.noisy_execute`` with the IQM bridge's noise model."""
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
    """Wraps ``maestro.noisy_estimate``."""
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
    """Wraps ``maestro.noisy_estimate_montecarlo``."""
    transpiled_circuit, noise_model = bridge_core(circuit, backend)
    return maestro.noisy_estimate_montecarlo(
        transpiled_circuit,
        observables,
        noise_model,
        noise_realizations=noise_realizations,
        config=config or maestro.SimulatorConfig(),
        seed=seed,
    )
