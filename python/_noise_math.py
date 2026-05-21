"""Shared noise-model math used by both the IBM and IQM bridges.

The functions in this module operate purely on a pandas DataFrame of
per-qubit calibration data plus a few attrs (``gates_2q``, ``crosstalk``)
— there's no dependency on which backend vendor produced the data. This
keeps the IQM bridge importable in environments that don't have
qiskit-ibm-runtime, and vice versa.

Both bridges call ``calculate_noise_params(df)`` on their own
extracted DataFrame to produce the structured noise-params dict
``maestro_noise_from_backend`` consumes.
"""

from __future__ import annotations

import math
import warnings

import pandas as pd


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


def calculate_noise_params(error_df: pd.DataFrame) -> dict:
    """Convert a per-qubit calibration DataFrame into a structured
    maestro noise-params dict.

    See the IBM bridge module docstring for the full physical model.
    Expected DataFrame shape:

    * Index = integer qubit indices.
    * Columns: ``T1``, ``T2`` (seconds), ``prob_meas0_prep1``,
      ``prob_meas1_prep0``, ``gate_error_1q`` (RB infidelity),
      ``gate_length_1q`` (seconds).
    * ``df.attrs["gates_2q"]`` — ``{(q1, q2): {"<gate>_error": float,
      "<gate>_length": float}}``.
    * ``df.attrs["crosstalk"]`` — ``{(q1, q2): rzz_error}`` (live
      backends only; empty otherwise).

    Returns a dict with three sections (``"qubits"``, ``"gates_2q"``,
    ``"crosstalk"``) — see ibm_noise_bridge for field semantics.
    """
    # First pass: longest 2Q gate time each qubit participates in.
    t_2q_per_q: dict[int, float] = {}
    for pair, entry in error_df.attrs.get("gates_2q", {}).items():
        for key, val in entry.items():
            if not key.endswith("_length"):
                continue
            length = float(val)
            for q in pair:
                t_2q_per_q[int(q)] = max(t_2q_per_q.get(int(q), 0.0), length)

    qubit_params: dict[int, dict] = {}
    stale_qubits: list[int] = []
    for q, row in error_df.iterrows():
        gate_error = float(row["gate_error_1q"])
        t_1q = float(row["gate_length_1q"])
        t1 = float(row["T1"]) if pd.notna(row["T1"]) else 0.0
        t2 = float(row["T2"]) if pd.notna(row["T2"]) else 0.0
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

        qubit_params[int(q)] = {
            "t1_time": t1,
            "gate_time_1q": t_1q,
            "p_dephasing": p_dephasing,
            "p_1q_residual": p_1q_residual,
            "delta_2q_per_q": delta_2q_per_q,
            "p_meas1_prep0": float(row["prob_meas1_prep0"]),
            "p_meas0_prep1": float(row["prob_meas0_prep1"]),
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
    for pair, entry in error_df.attrs.get("gates_2q", {}).items():
        chosen_err: float | None = None
        chosen_length = 0.0
        for key, val in entry.items():
            if not key.endswith("_error"):
                continue
            stem = key[: -len("_error")]
            chosen_err = float(val)
            chosen_length = float(entry.get(f"{stem}_length", 0.0))
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
            if qubit not in error_df.index:
                return 0.0
            t1 = error_df.at[qubit, "T1"]
            t2 = error_df.at[qubit, "T2"]
            return _thermal_relax_avg_infid(
                float(t1) if pd.notna(t1) else 0.0,
                float(t2) if pd.notna(t2) else 0.0,
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
    for pair, p in error_df.attrs.get("crosstalk", {}).items():
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