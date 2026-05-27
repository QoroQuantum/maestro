"""Shared noise-model math used by both the IBM and IQM bridges.

The functions in this module operate on a ``CalibrationData`` object that
carries per-qubit scalar noise parameters plus two metadata dicts
(``gates_2q``, ``crosstalk``).  There is no dependency on which backend
vendor produced the data.

Both bridges call ``calculate_noise_params(data)`` on their extracted
``CalibrationData`` to produce the structured noise-params dict that
``maestro_noise_from_backend`` consumes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field


@dataclass
class CalibrationData:
    """Per-device noise characterisation used throughout the bridge.

    ``qubits`` maps integer qubit index → dict of per-qubit scalars.
    Missing fields are stored as ``float('nan')`` (zero contribution in
    the noise model).  All qubits 0..num_qubits-1 must have an entry;
    use an empty dict for qubits with no characterisation data.

    Per-qubit keys: ``T1``, ``T2`` (seconds), ``gate_error_1q`` (RB
    infidelity), ``gate_length_1q`` (seconds), ``prob_meas0_prep1``,
    ``prob_meas1_prep0``.

    ``gates_2q`` maps ``(q1, q2)`` → ``{"<gate>_error": float,
    "<gate>_length": float}``.

    ``crosstalk`` maps ``(q1, q2)`` → rzz_error (live IBM backends only).
    """

    num_qubits: int
    qubits: dict[int, dict[str, float]]
    gates_2q: dict[tuple[int, int], dict[str, float]] = field(default_factory=dict)
    crosstalk: dict[tuple[int, int], float] = field(default_factory=dict)


def _thermal_relax_avg_infid(t1_s: float, t2_s: float,
                             gate_time_s: float) -> float:
    """Average gate infidelity of the bridge's thermal-relaxation channel.

    Stochastic mixture matching qiskit-aer's ``thermal_relaxation_error``
    for T2 ≤ T1: Reset → |0⟩ with prob ``p_reset = 1 − exp(−t/T1)``, and
    Pauli Z with conditional prob ``p_z = (1 − exp(−t·(1/T2_eff − 1/T1)))/2``
    on the surviving trajectories (T2_eff = min(T2, 2·T1)). Returns the
    corresponding 1 − F_avg.
    """
    if t1_s <= 0 or gate_time_s <= 0:
        return 0.0
    p_reset = 1.0 - math.exp(-gate_time_s / t1_s)
    if t2_s <= 0:
        return 0.5 * p_reset
    t2_eff = min(t2_s, 2.0 * t1_s)
    inv_gap = (1.0 / t2_eff) - (1.0 / t1_s)
    if inv_gap < 0.0:
        inv_gap = 0.0
    p_z = (1.0 - math.exp(-gate_time_s * inv_gap)) / 2.0
    return (2.0 / 3.0) * (1.0 - p_reset) * p_z + 0.5 * p_reset


def _depol_from_avg_error(err_avg: float, dim: int) -> float:
    """Convert an average gate error to maestro's depolarizing parameter.

    Maestro's Pauli-mixing depolarizing channel sets the total
    non-identity probability ``p``; for it,
    ``err_avg = p · d/(d+1)``, so ``p = err_avg · (d+1)/d``. Clamped to
    ``[0, 1]``. (Distinct from the textbook parameterization
    ``E(ρ) = (1−λ)ρ + λ·I/d`` whose λ = ``d/(d−1) · err``.)
    """
    if err_avg <= 0.0 or dim < 2:
        return 0.0
    return max(0.0, min(1.0, err_avg * (dim + 1) / dim))


def _nan(v: float) -> bool:
    return math.isnan(v)


def calculate_noise_params(data: CalibrationData) -> dict:
    """Convert a ``CalibrationData`` object into a structured maestro
    noise-params dict.

    See the IBM bridge module docstring for the full physical model.

    Returns a dict with three sections:

    * ``"qubits"``  — ``{q: {t1_time, gate_time_1q, p_dephasing,
      p_1q_residual, delta_2q_per_q, p_meas1_prep0, p_meas0_prep1}}``.
    * ``"gates_2q"`` — ``{(q1, q2): {p, gate_time}}``.
    * ``"crosstalk"`` — ``{(q1, q2): angle_rad}``.
    """
    # First pass: longest 2Q gate time each qubit participates in.
    t_2q_per_q: dict[int, float] = {}
    for pair, entry in data.gates_2q.items():
        for key, val in entry.items():
            if not key.endswith("_length"):
                continue
            length = float(val)
            for q in pair:
                t_2q_per_q[int(q)] = max(t_2q_per_q.get(int(q), 0.0), length)

    qubit_params: dict[int, dict] = {}
    stale_qubits: list[int] = []
    for q, row in data.qubits.items():
        _nan_f = float("nan")
        gate_error_raw = row.get("gate_error_1q", _nan_f)
        gate_error = gate_error_raw if not _nan(gate_error_raw) else 0.0
        t_1q = row.get("gate_length_1q", _nan_f)
        t1_raw = row.get("T1", _nan_f)
        t2_raw = row.get("T2", _nan_f)
        t1 = t1_raw if not _nan(t1_raw) else 0.0
        t2 = t2_raw if not _nan(t2_raw) else 0.0
        t_2q = t_2q_per_q.get(int(q), 0.0)

        if t1 > 0 and t2 > 0 and t_1q > 0:
            t2_eff = min(t2, 2.0 * t1)
            inv_gap = (1.0 / t2_eff) - (1.0 / t1)
            if inv_gap < 0.0:
                inv_gap = 0.0
            p_dephasing = (1.0 - math.exp(-t_1q * inv_gap)) / 2.0
        else:
            p_dephasing = 0.0

        infid_thermal_1q = _thermal_relax_avg_infid(t1, t2, t_1q)
        infid_thermal_2q = (_thermal_relax_avg_infid(t1, t2, t_2q)
                            if t_2q > 0 else 0.0)

        residual_1q_err = gate_error - infid_thermal_1q
        if residual_1q_err < 0.0:
            stale_qubits.append(int(q))
            residual_1q_err = 0.0
        p_1q_residual = _depol_from_avg_error(residual_1q_err, dim=2)

        delta_2q_err = max(0.0, infid_thermal_2q - infid_thermal_1q)
        delta_2q_per_q = _depol_from_avg_error(delta_2q_err, dim=2)

        _nan_f = float("nan")
        qubit_params[int(q)] = {
            "t1_time": t1,
            "gate_time_1q": t_1q,
            "p_dephasing": p_dephasing,
            "p_1q_residual": p_1q_residual,
            "delta_2q_per_q": delta_2q_per_q,
            "p_meas1_prep0": row.get("prob_meas1_prep0", _nan_f),
            "p_meas0_prep1": row.get("prob_meas0_prep1", _nan_f),
        }

    if stale_qubits:
        sample = stale_qubits[:5]
        warnings.warn(
            f"RB gate_error is below the T1/T2 decoherence floor on "
            f"{len(stale_qubits)} qubit(s) (e.g. {sample}); 1Q residual "
            f"clamped to 0. Likely stale calibration data.",
            stacklevel=2,
        )

    gates_2q_params: dict[tuple[int, int], dict] = {}
    for pair, entry in data.gates_2q.items():
        chosen_err: float | None = None
        chosen_length = 0.0
        for key, val in entry.items():
            if not key.endswith("_error"):
                continue
            stem = key[: -len("_error")]
            chosen_err = float(val)
            chosen_length = float(entry.get(f"{stem}_length", 0.0))
            if chosen_length == 0.0 and f"{stem}_length" not in entry:
                warnings.warn(
                    f"No gate duration for pair {pair} gate '{stem}'; "
                    f"thermal contribution set to zero.",
                    stacklevel=2,
                )
            break
        if chosen_err is None:
            if entry:
                warnings.warn(
                    f"No recognized 2Q gate for pair {pair}: "
                    f"{list(entry.keys())}",
                    stacklevel=2,
                )
            continue

        def _qubit_thermal(qubit: int, gate_time: float) -> float:
            if qubit not in data.qubits:
                return 0.0
            qrow = data.qubits[qubit]
            _nan_f = float("nan")
            t1v = qrow.get("T1", _nan_f)
            t2v = qrow.get("T2", _nan_f)
            return _thermal_relax_avg_infid(
                t1v if not _nan(t1v) else 0.0,
                t2v if not _nan(t2v) else 0.0,
                gate_time,
            )

        q1, q2 = int(pair[0]), int(pair[1])
        infid_2q_q1 = _qubit_thermal(q1, chosen_length)
        infid_2q_q2 = _qubit_thermal(q2, chosen_length)
        residual_2q_err = max(0.0, chosen_err - (infid_2q_q1 + infid_2q_q2))
        p_2q_residual = _depol_from_avg_error(residual_2q_err, dim=4)
        gates_2q_params[pair] = {
            "p": p_2q_residual,
            "gate_time": chosen_length,
        }

    crosstalk_params: dict[tuple[int, int], float] = {}
    saturated_pairs: list[tuple[int, int]] = []
    CROSSTALK_SATURATION = 0.5
    for pair, p in data.crosstalk.items():
        p_raw = float(p)
        if p_raw >= CROSSTALK_SATURATION:
            saturated_pairs.append(pair)
            continue
        p = max(0.0, min(1.0, p_raw))
        if p > 0:
            crosstalk_params[pair] = 2.0 * math.asin(math.sqrt(p))
    if saturated_pairs:
        sample = saturated_pairs[:5]
        warnings.warn(
            f"Dropped {len(saturated_pairs)} crosstalk entries with "
            f"rzz_error ≥ {CROSSTALK_SATURATION} (e.g. {sample}); these "
            f"are saturation/failed-characterization sentinels, not "
            f"physical 100% Z errors.",
            stacklevel=2,
        )

    return {
        "qubits": qubit_params,
        "gates_2q": gates_2q_params,
        "crosstalk": crosstalk_params,
    }