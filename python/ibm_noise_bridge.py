"""Bridge between qiskit-ibm-runtime backends and maestro.

Reads device properties (T1, T2, gate errors, readout errors, optional ZZ
crosstalk) from a fake or live backend, builds an equivalent
``maestro.NoiseModel``, transpiles circuits to the backend's coupling map /
basis gates, and exposes thin wrappers around ``maestro.noisy_execute`` and
``maestro.noisy_estimate``.

Noise model
-----------

The bridge implements the most physically-honest model attainable from the
information IBM exposes in ``backend.properties()``:

1. **T1 amplitude damping** — stochastic Reset → |0⟩ with probability
   ``γ = 1 − exp(−t / T1)`` (``set_t1_from_time``), sized to the 1Q gate
   time. The same channel fires on every gate (1Q and 2Q) via maestro's
   global ``t1_`` map; the extra decoherence accumulated during the longer
   CX duration is added separately as a 2Q-gate-only channel (item 4).
2. **T2 pure dephasing** — Pauli Z with probability
   ``p_z = (1 − exp(−t · (1/T2_eff − 1/T1))) / 2``, ``T2_eff = min(T2, 2·T1)``
   (``set_dephasing``), again sized to the 1Q gate time.
3. **RB residual control error** — what remains of ``gate_error`` after
   subtracting the thermal infidelity. Allocated to an **incoherent
   depolarizing** channel: Clifford-RB Pauli-twirls any input error, so a
   single RB number cannot distinguish coherent vs. incoherent residuals,
   and depolarizing is the maximum-entropy (least-presumptuous) choice.
   Applied via ``set_1q_gate_depolarizing`` (1Q gates only) and
   ``set_2q_depolarizing`` (correlated channel on the CX pair).
4. **2Q-gate-only delta thermal** — the extra T1/T2 infidelity the involved
   qubits accumulate during the (~10×) longer CX duration. Modeled as a
   Pauli-mixing depolarizing channel via ``set_2q_gate_depolarizing`` on
   each involved qubit, sized to the average-error delta between 2Q-time
   and 1Q-time thermal infidelity.
5. **ZZ crosstalk** (live backends only) — IBM exposes this as an
   ``rzz``/``zz`` characterization. Modeled as a coherent Rz spectator
   rotation through ``set_crosstalk`` (a first-class maestro channel).
6. **Readout error** — classical confusion matrix
   (``set_readout_error``).

What's *not* modeled (out of scope for this bridge):

- Idle-spectator decoherence during a neighbor's long gate. Both maestro
  and qiskit-aer skip this without explicit ``Delay`` scheduling.
- Leakage to non-computational levels.
- Time-correlated drift between recalibrations.
- Coherent control errors of the residual. The maestro ``set_coherent_*``
  API remains available for users who have specific characterization data
  (Ramsey, calibration drift, pulse miscalibration) justifying a coherent
  model — they build a ``maestro.NoiseModel`` directly. The bridge cannot
  infer this from ``backend.properties()`` alone.

All channel parameters use maestro's "Pauli-mixing" depolarizing
parameterization: for a dim-d channel, the input ``p`` is the total
probability of any non-identity Pauli being applied, related to the
average gate error by ``err_avg = p · d / (d+1)``.
"""

from __future__ import annotations

import math
import warnings

import qiskit
import qiskit.qasm2
from _noise_math import (
    calculate_noise_params,
    _thermal_relax_avg_infid,
    _depol_from_avg_error,
)
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit_ibm_runtime.models.exceptions import BackendPropertyError
import pandas as pd
import maestro

MaestroCircuit = maestro.circuits.QuantumCircuit

Backend = FakeBackendV2 | IBMBackend


def read_backend_data(backend: Backend) -> pd.DataFrame:
    """Extract per-qubit noise data from a backend into a DataFrame.

    Accepts both ``FakeBackendV2`` and live ``IBMBackend``. Returns a
    DataFrame indexed by physical qubit, plus two attributes:

    - ``df.attrs['gates_2q']`` — dict keyed by qubit pair, mapping
      ``"{gate}_error"`` / ``"{gate}_length"`` for each non-crosstalk 2Q
      basis gate the backend reports (cx, cz, ecr, iswap, …).
    - ``df.attrs['crosstalk']`` — dict keyed by qubit pair, mapping the
      RB-measured ``rzz``/``zz`` error probabilities. Empty for fake
      backends that don't characterize ZZ.
    """
    props = backend.properties()
    n = backend.num_qubits

    rows = []
    missing_t1t2: list[int] = []
    for q in range(n):
        # Live backends sometimes mark qubits as unusable; their T1/T2 (and
        # downstream gate properties) simply aren't reported. Use NaN so
        # downstream ``pd.notna`` checks treat them as "no thermal data";
        # those qubits then contribute zero thermal infidelity and a
        # gate_error-only residual — fine since they're not normally in
        # the active circuit anyway.
        try:
            t1 = props.t1(q)
        except (KeyError, AttributeError, BackendPropertyError):
            t1 = float("nan")
            missing_t1t2.append(q)
        try:
            t2 = props.t2(q)
        except (KeyError, AttributeError, BackendPropertyError):
            t2 = float("nan")
            if q not in missing_t1t2:
                missing_t1t2.append(q)
        row = {"T1": t1, "T2": t2}
        try:
            row["prob_meas0_prep1"] = props.qubit_property(q, "prob_meas0_prep1")[0]
        except (KeyError, AttributeError, BackendPropertyError):
            row["prob_meas0_prep1"] = 0.0
        try:
            row["prob_meas1_prep0"] = props.qubit_property(q, "prob_meas1_prep0")[0]
        except (KeyError, AttributeError, BackendPropertyError):
            row["prob_meas1_prep0"] = 0.0

        gate_errors_1q, gate_lengths_1q = [], []
        for gp in props.gates:
            if len(gp.qubits) == 1 and gp.qubits[0] == q:
                try:
                    gate_errors_1q.append(props.gate_error(gp.gate, q))
                    gate_lengths_1q.append(props.gate_length(gp.gate, q))
                except (AttributeError, KeyError, BackendPropertyError):
                    continue
        row["gate_error_1q"] = (
            sum(gate_errors_1q) / len(gate_errors_1q) if gate_errors_1q else 0.0
        )
        row["gate_length_1q"] = (
            sum(gate_lengths_1q) / len(gate_lengths_1q) if gate_lengths_1q else 0.0
        )
        rows.append(row)

    if missing_t1t2:
        sample = missing_t1t2[:5]
        warnings.warn(
            f"Backend reports no T1/T2 for {len(missing_t1t2)} qubit(s) "
            f"(e.g. {sample}); treating as NaN (zero thermal contribution). "
            f"These are likely flagged unusable by the device.",
            stacklevel=2,
        )

    df = pd.DataFrame(rows)
    df.index.name = "qubit"

    gates_2q: dict[tuple[int, int], dict] = {}
    crosstalk: dict[tuple[int, int], float] = {}
    for gp in props.gates:
        if len(gp.qubits) != 2:
            continue
        pair = tuple(gp.qubits)
        try:
            err = props.gate_error(gp.gate, list(pair))
            length = props.gate_length(gp.gate, list(pair))
        except (AttributeError, KeyError, BackendPropertyError):
            continue
        if gp.gate.lower() in ("rzz", "zz"):
            crosstalk[pair] = err
        else:
            entry = gates_2q.setdefault(pair, {})
            entry[f"{gp.gate}_error"] = err
            entry[f"{gp.gate}_length"] = length
    df.attrs["gates_2q"] = gates_2q
    df.attrs["crosstalk"] = crosstalk
    return df


def maestro_noise_from_backend(backend: Backend) -> maestro.NoiseModel:
    """Build a maestro NoiseModel from a fake or live IBM backend.

    See the module docstring for the full physical model. Per qubit, this
    function attaches:

    * T1 amplitude damping (``set_t1_from_time``) — stochastic Reset
      sized to the 1Q gate time; fires on every gate.
    * T2 pure dephasing (``set_dephasing``) — Pauli Z sized to the 1Q
      gate time; fires on every gate.
    * 1Q RB-residual depolarizing (``set_1q_gate_depolarizing``) — only
      fires on single-qubit gates.
    * 2Q-only delta thermal (``set_2q_gate_depolarizing``) — only fires
      on two-qubit gates, capturing the extra CX-time decoherence that
      the 1Q-time-sized T1/dephasing channels above don't cover.
    * Asymmetric readout error (``set_readout_error``).

    Per 2Q pair:

    * Correlated 2Q depolarizing for the RB residual after thermal
      subtraction (``set_2q_depolarizing``).

    Per crosstalk-characterized pair (live backends only):

    * Rz spectator rotation (``set_crosstalk``) at angle
      ε = 2·arcsin(√p) derived from the RB-measured ZZ error probability.
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
# Bridge core: transpile + build the noise model
# ----------------------------------------------------------


def transpile_to_backend(
    circuit: "str | qiskit.QuantumCircuit | MaestroCircuit",
    backend: Backend,
) -> qiskit.QuantumCircuit:
    """Transpile a circuit to the backend's coupling map and basis gates."""
    if isinstance(circuit, str):
        qc = qiskit.qasm2.loads(circuit)
    elif isinstance(circuit, qiskit.QuantumCircuit):
        qc = circuit
    elif isinstance(circuit, MaestroCircuit):
        qc = qiskit.qasm2.loads(circuit.to_qasm())
    else:
        raise TypeError(f"Unsupported circuit type: {type(circuit)}")
    return qiskit.transpile(qc, backend=backend)


def bridge_core(
    circuit, backend: Backend
) -> tuple[MaestroCircuit, maestro.NoiseModel]:
    """Transpile the circuit and build the noise model in one step.

    The transpiled circuit must have ``num_qubits == backend.num_qubits``;
    otherwise QASM qubit indices won't align with the physical-qubit keys
    on the noise model. This is the default qiskit behavior when
    transpiling against a backend.

    Execution still happens locally on Maestro — the backend is only
    consulted for calibration data.
    """
    transpiled_qiskit = transpile_to_backend(circuit, backend)
    if transpiled_qiskit.num_qubits != backend.num_qubits:
        raise ValueError(
            f"Transpiled circuit has {transpiled_qiskit.num_qubits} qubits "
            f"but backend has {backend.num_qubits}. QASM qubit indices won't "
            f"align with the noise model's physical-qubit keys. Re-transpile "
            f"with the backend's full layout (do not pass a smaller "
            f"initial_layout that drops qubits)."
        )
    qasm_str = qiskit.qasm2.dumps(transpiled_qiskit)
    transpiled_maestro = maestro.QasmToCirc().parse_and_translate(qasm_str)
    noise_model = maestro_noise_from_backend(backend)
    return transpiled_maestro, noise_model


# ----------------------------------------------------------
# Wrappers for Maestro execute/simulate calls
# Maestro core calls (C++ library) exposed in bindings.cpp
# ----------------------------------------------------------


def execute_backend_noise_model(
    circuit,
    backend: Backend,
    config: maestro.SimulatorConfig | None = None,
    shots: int = 1000,
) -> dict:
    """Wraps around maestro's noisy_execute."""
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
    backend: Backend,
    config: maestro.SimulatorConfig | None = None,
) -> dict:
    """Wraps around maestro's noisy_estimate."""
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
    backend: Backend,
    config: maestro.SimulatorConfig | None = None,
    noise_realizations: int = 100,
    seed: int = 42,
) -> dict:
    """Wraps around maestro's noisy_estimate_montecarlo."""
    transpiled_circuit, noise_model = bridge_core(circuit, backend)
    return maestro.noisy_estimate_montecarlo(
        transpiled_circuit,
        observables,
        noise_model,
        noise_realizations=noise_realizations,
        config=config or maestro.SimulatorConfig(),
        seed=seed,
    )
