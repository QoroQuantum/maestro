"""Sinter interface for Maestro quantum circuit simulator.

This module provides MaestroSinterSampler and MaestroCompiledSampler,
enabling integration between the Stim/Sinter QEC framework and Maestro's
tensor-network/GPU simulation engine.
"""

from __future__ import annotations

import collections
import copy
import math
import numbers
import re
import time
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import sinter
    import stim
except ImportError:
    raise ImportError("Install qoro-maestro[sinter] to use the Sinter sampler.")

import maestro

# The documented entry points; the remaining helpers are internal.
__all__ = ["MaestroCompiledSampler", "MaestroSinterSampler", "translate_stim_to_maestro"]


def validate_shots(shots: int) -> int:
    """Validate that shots is a positive integer."""
    if isinstance(shots, bool) or not isinstance(shots, numbers.Integral) or shots <= 0:
        raise ValueError("shots must be a positive integer")
    return int(shots)


def counts_to_measurements(
    counts: dict[str, int],
    num_measurements: int,
    shots: int,
    *,
    storage_width: int | None = None,
) -> np.ndarray:
    """Strict Maestro bit order: character k is classical bit k."""
    if not counts:
        raise ValueError("Maestro returned an empty measurement histogram")
    rows, weights = [], []
    for bits, count in counts.items():
        bits = bits.replace(" ", "")
        if storage_width is not None and len(bits) == storage_width and storage_width > num_measurements:
            if set(bits[num_measurements:]) - {"0"}:
                raise ValueError("Nonzero data outside the declared measurement record")
            bits = bits[:num_measurements]
        if len(bits) != num_measurements or set(bits) - {"0", "1"}:
            raise ValueError(f"Maestro measurement width/content mismatch: expected {num_measurements}")
        if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
            raise ValueError("Maestro counts must be nonnegative integers")
        rows.append([b == "1" for b in bits])
        weights.append(count)
    if sum(weights) != shots:
        raise ValueError(f"Maestro returned {sum(weights)} shots, expected {shots}")
    return np.repeat(np.asarray(rows, dtype=bool).reshape(len(rows), num_measurements), weights, axis=0)


def samples_to_measurements(
    samples: np.ndarray,
    num_measurements: int,
    shots: int,
    *,
    storage_width: int | None = None,
) -> np.ndarray:
    """Validate raw shot rows; column k is classical measurement bit k."""
    rows = np.asarray(samples)
    if rows.ndim != 2 or rows.shape[0] != shots:
        raise ValueError("Maestro sample matrix shape does not match the requested shots")
    if rows.dtype not in (np.dtype(bool), np.dtype(np.uint8)) or np.any(rows > 1):
        raise ValueError("Maestro samples must contain boolean or uint8 values in {0, 1}")
    if storage_width is not None and rows.shape[1] == storage_width and storage_width > num_measurements:
        if np.any(rows[:, num_measurements:]):
            raise ValueError("Nonzero data outside the declared measurement record")
        rows = rows[:, :num_measurements]
    if rows.shape[1] != num_measurements:
        raise ValueError(f"Maestro measurement width/content mismatch: expected {num_measurements}")
    # Both accepted dtypes use one byte. Values are already canonical 0/1,
    # so a bool view retains the native allocation without another matrix copy.
    return rows.view(np.bool_)


def _sample_result(execute, *args, **kwargs):
    """Prefer raw rows while retaining support for older native extensions."""
    try:
        return execute(*args, return_samples=True, **kwargs)
    except TypeError as error:
        message = str(error)
        if "return_samples" not in message or not (
            "unexpected keyword" in message or "incompatible function arguments" in message
        ):
            raise
        return execute(*args, **kwargs)


def _noise_realizations(circuit, noise_model, config, shots):
    """Integrate exact channels once while preserving independent trajectory shots."""
    resolve = getattr(maestro, "_resolve_noise_realizations", None)
    if resolve is None:
        return shots
    return resolve(circuit, noise_model, config, shots)


CLEAN_CIRCUIT_API_VERSION = 2
LOCAL_NOISE = {'DEPOLARIZE1', 'DEPOLARIZE2', 'X_ERROR', 'Y_ERROR', 'Z_ERROR',
               'PAULI_CHANNEL_1', 'PAULI_CHANNEL_2'}
MEASUREMENTS = {'M', 'MZ', 'MX', 'MY', 'MR', 'MRZ', 'MRX', 'MRY'}

# Stim's other names for its unitary gates (ZCX for CX, SWAPCZ for CZSWAP,
# H_XZ for H, ...). A stim.Circuit reports canonical names already; text and
# instruction tuples may use either. Measurement and reset aliases stay as
# written, because RZ with an argument is Maestro's rotation.
STIM_GATE_ALIASES = {alias: gate.name for gate in stim.gate_data().values() if gate.is_unitary
                     for alias in gate.aliases if alias != gate.name}

# Stim gates that Maestro builds from its own gates: every step but the last
# is noiseless, and the last is one ordinary gate on the same qubits. A
# NoiseModel adds gate noise after noisy gates only, so the Stim gate gets it
# once, after the whole gate, as a native gate would: the two-qubit layers on
# its pair and the per-qubit layers on each of its qubits. Noise before
# trailing local gates would be equivalent only for noise those gates leave
# unchanged, such as depolarizing noise: an X error before an H becomes a Z
# error after it, and coherent rotations and T1 change as well. Each step is
# a QuantumCircuit method and the positions of its qubits among the Stim
# gate's targets.
#
# The entries come from a search for the fewest noiseless entangling gates,
# then the fewest gates; the swap family keeps its short forms (ISWAP is S on
# each qubit, CZ, then SWAP).
# Maestro's two-qubit gates have Z on at least one side, so a gate whose last
# interaction is XX, YY, XY or YX takes a noiseless entangling gate as well
# as the noisy one. The NoiseModel looks up its pair layers by qubit set
# (two-qubit depolarizing and correlated phase flips are stored both ways;
# Kraus channels apply in their configured order), so a last gate may take
# its pair in reverse: XCZ a b is exactly CX b a. Tests check every entry
# against stim.Tableau.from_named_gate.
ONE_QUBIT_DECOMPOSITIONS = {
    # Ry(±π/2) would be exact, but the stabilizer simulator rejects Ry.
    "SQRT_Y": (("h", 0), ("x", 0)),
    "SQRT_Y_DAG": (("h", 0), ("z", 0)),
    "C_XYZ": (("sdg", 0), ("h", 0)),
    "C_ZYX": (("h", 0), ("s", 0)),
    "C_NXYZ": (("s", 0), ("h", 0), ("y", 0)),
    "C_XNYZ": (("s", 0), ("h", 0)),
    "C_XYNZ": (("sdg", 0), ("h", 0), ("y", 0)),
    "C_NZYX": (("sdg", 0), ("sx", 0)),
    "C_ZNYX": (("h", 0), ("sdg", 0)),
    "C_ZYNX": (("s", 0), ("sxdg", 0)),
    "H_XY": (("sdg", 0), ("x", 0)),
    "H_YZ": (("sdg", 0), ("h", 0), ("s", 0)),
    "H_NXY": (("s", 0), ("x", 0)),
    "H_NXZ": (("h", 0), ("y", 0)),
    "H_NYZ": (("s", 0), ("h", 0), ("sdg", 0)),
}
TWO_QUBIT_DECOMPOSITIONS = {
    "ISWAP": (("s", 0), ("s", 1), ("cz", 0, 1), ("swap", 0, 1)),
    "ISWAP_DAG": (("sdg", 0), ("sdg", 1), ("cz", 0, 1), ("swap", 0, 1)),
    "CXSWAP": (("cx", 0, 1), ("swap", 0, 1)),
    "CZSWAP": (("cz", 0, 1), ("swap", 0, 1)),
    "SWAPCX": (("swap", 0, 1), ("cx", 0, 1)),
    "SQRT_ZZ": (("s", 0), ("s", 1), ("cz", 0, 1)),
    "SQRT_ZZ_DAG": (("sdg", 0), ("sdg", 1), ("cz", 0, 1)),
    "SQRT_XX": (("sdg", 0), ("cx", 0, 1), ("h", 0), ("sdg", 0), ("cx", 0, 1)),
    "SQRT_XX_DAG": (("s", 0), ("cx", 0, 1), ("h", 0), ("s", 0), ("cx", 0, 1)),
    "SQRT_YY": (("s", 1), ("cx", 0, 1), ("h", 0), ("sdg", 1), ("cy", 0, 1)),
    "SQRT_YY_DAG": (("sdg", 1), ("cx", 0, 1), ("h", 0), ("s", 0), ("s", 0), ("s", 1), ("cy", 0, 1)),
    "XCX": (("h", 1), ("s", 0), ("sdg", 1), ("cx", 1, 0), ("h", 1), ("sdg", 0), ("cx", 0, 1)),
    "XCY": (("h", 0), ("sdg", 0), ("cx", 0, 1), ("h", 0), ("cx", 1, 0)),
    "XCZ": (("cx", 1, 0),),
    "YCX": (("h", 1), ("sdg", 1), ("cx", 1, 0), ("h", 1), ("cx", 0, 1)),
    "YCY": (("h", 0), ("h", 1), ("s", 1), ("cx", 0, 1), ("h", 0), ("s", 1), ("cy", 0, 1)),
    "YCZ": (("cy", 1, 0),),
}
DECOMPOSED_GATES = {**ONE_QUBIT_DECOMPOSITIONS, **TWO_QUBIT_DECOMPOSITIONS}
FIXED_CLIFFORD_GATES = {"I", "II", "X", "Y", "Z", "H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
                        "CX", "CY", "CZ", "SWAP", *DECOMPOSED_GATES}
# Instructions that act on pairs of targets.
PAIR_INSTRUCTIONS = {"CX", "CNOT", "CY", "CZ", "SWAP", "II", "DEPOLARIZE2", "PAULI_CHANNEL_2",
                     *TWO_QUBIT_DECOMPOSITIONS}

if TYPE_CHECKING:
    import sinter

# Sinter does not export classify_discards_and_errors on its top-level module in all versions.
# We import it from the internal decoder module with an inline fallback.
try:
    from sinter._decoding._stim_then_decode_sampler import classify_discards_and_errors
except Exception:  # pragma: no cover
    def classify_discards_and_errors(
        *,
        actual_obs: np.ndarray,
        predictions: np.ndarray,
        postselected_observables_mask: np.ndarray | None,
        out_count_observable_error_combos: Any | None,
        num_obs: int,
    ) -> tuple[int, int]:
        num_discards = 0

        # Added bytes in predictions are used for signalling discards
        if predictions.shape[1] == actual_obs.shape[1] + 1:
            discard_mask = predictions[:, -1] != 0
            predictions = predictions[:, :-1]
            num_discards += int(np.count_nonzero(discard_mask))
            discard_mask ^= True
            actual_obs = actual_obs[discard_mask]
            predictions = predictions[discard_mask]

        # Mispredicted observables can be used for signalling discards
        if postselected_observables_mask is not None:
            discard_mask = np.any((actual_obs ^ predictions) & postselected_observables_mask, axis=1)
            num_discards += int(np.count_nonzero(discard_mask))
            discard_mask ^= True
            actual_obs = actual_obs[discard_mask]
            predictions = predictions[discard_mask]

        fail_mask = np.any(actual_obs != predictions, axis=1)
        if out_count_observable_error_combos is not None:
            for k in np.flatnonzero(fail_mask):
                mistakes = np.unpackbits(actual_obs[k] ^ predictions[k], count=num_obs, bitorder="little")
                err_key = "obs_mistake_mask=" + "".join("_E"[b] for b in mistakes)
                out_count_observable_error_combos[err_key] += 1

        num_errors = int(np.count_nonzero(fail_mask))
        return int(num_discards), int(num_errors)


def _parse_instruction_line(line: str) -> tuple[str, list[float], list[int]]:
    """Parse a single line of Stim-like circuit format."""
    line = line.strip()
    if not line or line.startswith("#"):
        return "", [], []

    # Match GATE(arg1, arg2...) target1 target2 ...
    m = re.match(r"^([A-Za-z0-9_]+)(?:\(([^)]*)\))?(.*)$", line)
    if not m:
        raise ValueError(f"Malformed instruction: {line!r}")

    name = m.group(1).upper()
    name = STIM_GATE_ALIASES.get(name, name)
    arg_str = m.group(2)
    targets_str = m.group(3)

    args = []
    if arg_str:
        args = [float(x.strip()) for x in arg_str.split(",") if x.strip()]

    targets = []
    if targets_str:
        for token in targets_str.strip().split():
            # Qubit indices are non-negative integers
            if token.isdigit():
                targets.append(int(token))
            elif name not in STIM_ANNOTATIONS:
                raise ValueError(f"Unsupported target {token!r} in {name}")

    return name, args, targets


def _stim_instructions(circuit: stim.Circuit | str | Any) -> list[tuple[str, list[float], list[int]]]:
    """(name, arguments, qubit targets) for each instruction of a Stim-like circuit."""
    if isinstance(circuit, list):
        return [(STIM_GATE_ALIASES.get(name, name), args, targets) for name, args, targets in circuit]
    if isinstance(circuit, str) or hasattr(circuit, "circuit_str"):
        text = circuit if isinstance(circuit, str) else circuit.circuit_str
        instructions = []
        for line in text.strip().splitlines():
            name, args, targets = _parse_instruction_line(line)
            if name:
                instructions.append((name, args, targets))
        return instructions

    instructions = []
    for inst in circuit.flattened():
        name = inst.name
        args = inst.gate_args_copy()
        raw_targets = inst.targets_copy()
        if name not in STIM_ANNOTATIONS and any(
            not t.is_qubit_target or t.is_inverted_result_target for t in raw_targets
        ):
            raise ValueError("Special Stim targets are unsupported by clean translation")
        targets = [t.qubit_value for t in raw_targets if t.is_qubit_target]
        instructions.append((name, args, targets))
    return instructions


# Parens arguments each Stim Pauli noise channel takes, as Stim requires.
NOISE_ARGUMENT_COUNTS = {"DEPOLARIZE1": 1, "DEPOLARIZE2": 1, "X_ERROR": 1, "Y_ERROR": 1,
                         "Z_ERROR": 1, "PAULI_CHANNEL_1": 3, "PAULI_CHANNEL_2": 15}


def _check_probabilities(name: str, args: list[float]) -> None:
    """Reject arguments of a Stim noise instruction that are not probabilities."""
    for p in args:
        if (isinstance(p, bool) or not isinstance(p, numbers.Real)
                or not math.isfinite(p) or not 0.0 <= p <= 1.0):
            raise ValueError(f"{name} arguments must be finite probabilities in [0, 1], got {p!r}")


def _measurement_flip(name: str, args: list[float]) -> float:
    """Stim's M(p) result-flip probability, validated as Stim does."""
    if len(args) > 1:
        raise ValueError(f"{name} takes 0 or 1 arguments, got {len(args)}")
    _check_probabilities(name, args)
    return float(args[0]) if args else 0.0


def _pauli_channel_probabilities(name: str, args: list[float]) -> tuple[int, list[float]]:
    """Arity and QuantumCircuit.pauli_channel probabilities of a Stim Pauli noise channel.

    Base-4 digit k of a Maestro index is the Pauli (I, X, Y, Z) on target k.
    Stim lists PAULI_CHANNEL_2 arguments as IX, IY, ..., ZZ, the first letter
    acting on the first target. Arguments are validated as Stim validates them,
    so a string or instruction list that Stim would reject raises ValueError
    here too, including zero-probability channels that are later dropped.
    """
    expected = NOISE_ARGUMENT_COUNTS[name]
    if len(args) != expected:
        raise ValueError(f"{name} takes {expected} argument{'s' if expected > 1 else ''}, got {len(args)}")
    _check_probabilities(name, args)
    if name.startswith("PAULI_CHANNEL") and math.fsum(args) > 1.0 + 1e-12:
        raise ValueError(f"{name} probabilities must sum to at most 1, got {math.fsum(args)!r}")

    if name == "DEPOLARIZE1":
        arity, errors = 1, {1: args[0] / 3, 2: args[0] / 3, 3: args[0] / 3}
    elif name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        arity, errors = 1, {"XYZ".index(name[0]) + 1: args[0]}
    elif name == "PAULI_CHANNEL_1":
        arity, errors = 1, {1: args[0], 2: args[1], 3: args[2]}
    elif name == "DEPOLARIZE2":
        arity, errors = 2, {index: args[0] / 15 for index in range(1, 16)}
    else:  # PAULI_CHANNEL_2
        arity, errors = 2, {}
        for k, p in enumerate(args):
            first, second = divmod(k + 1, 4)
            errors[first + 4 * second] = p

    probabilities = [0.0] * 4**arity
    for index, p in errors.items():
        probabilities[index] = p
    probabilities[0] = max(0.0, 1.0 - sum(errors.values()))
    return arity, probabilities


def translate_stim_to_maestro(
    circuit: stim.Circuit | str | Any,
    *, qubit_order=None,
) -> tuple[maestro.circuits.QuantumCircuit, maestro.NoiseModel, int]:
    """Translate a Stim circuit or circuit string into a Maestro QuantumCircuit and NoiseModel.

    Pauli noise channels (DEPOLARIZE1/2, X/Y/Z_ERROR, PAULI_CHANNEL_1/2) become
    QuantumCircuit.pauli_channel operations and measurement flip probabilities
    such as M(p) become readout errors, so the circuit carries Stim's noise
    exactly. The basis changes inserted for RX, RY, MX, MY, MRX and MRY are
    marked noiseless, so a NoiseModel adds gate noise only where Stim has gates.
    Stim's other one- and two-qubit Clifford gates (ISWAP, CXSWAP, SQRT_XX,
    XCY, C_XYZ, H_YZ, ...) are built from Maestro gates whose leading steps are
    noiseless, so a NoiseModel adds a gate's noise once, after the whole gate,
    as it does for a native gate.

    Args:
        circuit: A stim.Circuit, extended gate string, or list of
            (name, arguments, integer targets) tuples. Correlated (E) and
            heralded noise, inverted measurements and record-controlled
            operations reject.
        qubit_order: Optional complete qubit allocation order. Explicit order
            initializes these qubits to zero before the supplied instructions.

    Returns:
        A tuple of (QuantumCircuit, NoiseModel, num_measurements).
    """
    qc = maestro.circuits.QuantumCircuit()
    nm = maestro.NoiseModel()
    meas_idx = 0
    active_qubits: set[int] = set()
    instructions = _stim_instructions(circuit)

    used_qubits = {q for name, _, targets in instructions
                   if name not in STIM_ANNOTATIONS for q in targets}
    if any(isinstance(q, bool) or not isinstance(q, numbers.Integral) or q < 0
           for q in used_qubits):
        raise ValueError("Qubit targets must be nonnegative integers")
    if qubit_order is not None:
        order = tuple(qubit_order)
        if (any(isinstance(q, bool) or not isinstance(q, numbers.Integral) or q < 0
                for q in order) or len(set(order)) != len(order)
                or not used_qubits.issubset(order)):
            raise ValueError("qubit_order must contain every used qubit exactly once")
        for q in order:
            qc.reset(q)
            active_qubits.add(q)

    for name, args, targets in instructions:
        if len(args) and name in FIXED_CLIFFORD_GATES:
            raise ValueError(f"{name} takes 0 arguments, got {len(args)}")
        if name in PAIR_INSTRUCTIONS and len(targets) % 2:
            raise ValueError(f"{name} requires pairs of qubit targets")
        if name in PAIR_INSTRUCTIONS and any(targets[i] == targets[i + 1] for i in range(0, len(targets), 2)):
            raise ValueError(f"{name} requires two different qubits in each pair")
        # Stim's M(p) flips the recorded result, which is what readout error does.
        flip = _measurement_flip(name, args) if name in MEASUREMENTS else 0.0
        # --- Single-qubit Clifford gates ---
        if name in ("I", "II"):
            for q in targets:
                # A zero-duration idle is Clifford-safe and preserves allocation.
                qc.delay(q, 0.0)
                active_qubits.add(q)
        elif name == "X":
            for q in targets:
                qc.x(q)
                active_qubits.add(q)
        elif name == "Y":
            for q in targets:
                qc.y(q)
                active_qubits.add(q)
        elif name == "Z":
            for q in targets:
                qc.z(q)
                active_qubits.add(q)
        elif name == "H":
            for q in targets:
                qc.h(q)
                active_qubits.add(q)
        elif name in ("S", "SQRT_Z"):
            for q in targets:
                qc.s(q)
                active_qubits.add(q)
        elif name in ("S_DAG", "SQRT_Z_DAG"):
            for q in targets:
                qc.sdg(q)
                active_qubits.add(q)
        elif name == "SQRT_X":
            for q in targets:
                qc.sx(q)
                active_qubits.add(q)
        elif name == "SQRT_X_DAG":
            for q in targets:
                qc.sxdg(q)
                active_qubits.add(q)
        # Other Clifford gates, equal up to global phase. Only the last step
        # is noisy, so a NoiseModel adds gate noise once, after the gate.
        elif name in DECOMPOSED_GATES:
            *leading, last = DECOMPOSED_GATES[name]
            width = 2 if name in TWO_QUBIT_DECOMPOSITIONS else 1
            for i in range(0, len(targets), width):
                group = targets[i:i + width]
                for method, *positions in leading:
                    getattr(qc, method)(*(group[p] for p in positions), noiseless=True)
                method, *positions = last
                getattr(qc, method)(*(group[p] for p in positions))
                active_qubits.update(group)

        # --- Parameterized single-qubit rotations & non-Clifford gates ---
        elif name == "T":
            for q in targets:
                qc.t(q)
                active_qubits.add(q)
        elif name == "T_DAG":
            for q in targets:
                qc.tdg(q)
                active_qubits.add(q)
        elif name == "RX" and args:
            angle = args[0]
            for q in targets:
                qc.rx(q, angle)
                active_qubits.add(q)
        elif name == "RY" and args:
            angle = args[0]
            for q in targets:
                qc.ry(q, angle)
                active_qubits.add(q)
        elif name == "RZ" and args:
            angle = args[0]
            for q in targets:
                qc.rz(q, angle)
                active_qubits.add(q)

        # --- Two-qubit Clifford gates ---
        elif name in ("CX", "CNOT"):
            for i in range(0, len(targets), 2):
                qc.cx(targets[i], targets[i + 1])
                active_qubits.add(targets[i])
                active_qubits.add(targets[i + 1])
        elif name == "CY":
            for i in range(0, len(targets), 2):
                qc.cy(targets[i], targets[i + 1])
                active_qubits.add(targets[i])
                active_qubits.add(targets[i + 1])
        elif name == "CZ":
            for i in range(0, len(targets), 2):
                qc.cz(targets[i], targets[i + 1])
                active_qubits.add(targets[i])
                active_qubits.add(targets[i + 1])
        elif name == "SWAP":
            for i in range(0, len(targets), 2):
                qc.swap(targets[i], targets[i + 1])
                active_qubits.add(targets[i])
                active_qubits.add(targets[i + 1])

        # --- Resets (unparameterized) ---
        # Basis changes are part of the reset or measurement, so they are
        # noiseless: a NoiseModel adds gate noise only where Stim has gates.
        elif name in ("R", "RZ"):
            for q in targets:
                qc.reset(q)
        elif name == "RX":
            for q in targets:
                qc.reset(q)
                qc.h(q, noiseless=True)
                active_qubits.add(q)
        elif name == "RY":
            for q in targets:
                qc.reset(q)
                qc.h(q, noiseless=True)
                qc.s(q, noiseless=True)
                active_qubits.add(q)

        # --- Measurements and Measure-and-Resets ---
        elif name in ("M", "MZ"):
            for q in targets:
                qc.measure([(q, meas_idx)], readout_error=flip)
                meas_idx += 1
                active_qubits.add(q)
        elif name in ("MR", "MRZ"):
            for q in targets:
                qc.measure([(q, meas_idx)], readout_error=flip)
                qc.reset(q)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MX":
            for q in targets:
                qc.h(q, noiseless=True)
                qc.measure([(q, meas_idx)], readout_error=flip)
                qc.h(q, noiseless=True)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MRX":
            for q in targets:
                qc.h(q, noiseless=True)
                qc.measure([(q, meas_idx)], readout_error=flip)
                qc.reset(q)
                qc.h(q, noiseless=True)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MY":
            for q in targets:
                qc.sdg(q, noiseless=True)
                qc.h(q, noiseless=True)
                qc.measure([(q, meas_idx)], readout_error=flip)
                qc.h(q, noiseless=True)
                qc.s(q, noiseless=True)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MRY":
            for q in targets:
                qc.sdg(q, noiseless=True)
                qc.h(q, noiseless=True)
                qc.measure([(q, meas_idx)], readout_error=flip)
                qc.reset(q)
                qc.h(q, noiseless=True)
                qc.s(q, noiseless=True)
                meas_idx += 1
                active_qubits.add(q)

        # --- Pauli noise channels, carried exactly as circuit operations ---
        elif name in LOCAL_NOISE:
            arity, probabilities = _pauli_channel_probabilities(name, args)
            if probabilities[0] < 1.0:
                for i in range(0, len(targets), arity):
                    qc.pauli_channel(targets[i:i + arity], probabilities)
                    active_qubits.update(targets[i:i + arity])

        # --- Annotations (skipped for simulation) ---
        elif name in (
            "TICK",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "QUBIT_COORDS",
            "SHIFT_COORDS",
        ):
            continue
        else:
            raise ValueError(f"Unsupported Stim instruction '{name}' for Maestro translation.")

    return qc, nm, meas_idx


STIM_NOISE_INSTRUCTIONS = {
    "DEPOLARIZE1",
    "DEPOLARIZE2",
    "X_ERROR",
    "Y_ERROR",
    "Z_ERROR",
    "PAULI_CHANNEL_1",
    "PAULI_CHANNEL_2",
    "E",
    "ELSE_CORRELATED_ERROR",
}

STIM_MEASURE_INSTRUCTIONS = {
    "M",
    "MZ",
    "MX",
    "MY",
    "MR",
    "MRZ",
    "MRX",
    "MRY",
    "MPP",
}

STIM_RESET_INSTRUCTIONS = {"R", "RZ", "RX", "RY"}

STIM_ANNOTATIONS = {
    "TICK",
    "QUBIT_COORDS",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS",
}


def split_stim_prefix_suffix(
    circuit: stim.Circuit | str | Any,
    has_external_noise: bool = False,
    deterministic_noise: bool = False,
    exact_resets: bool = False,
) -> tuple[Any, Any, int]:
    """Split a Stim circuit into a deterministic prefix and suffix.

    The prefix contains all deterministic operations prior to the first
    noise channel, stochastic measurement or collapsing reset.

    Args:
        circuit: A stim.Circuit, circuit string, or wrapper object.
        has_external_noise: Whether an external noise model applies to gates
            that the prefix does not carry.
        deterministic_noise: Whether in-circuit Pauli channels evolve as exact
            channels (a density-matrix or MPO backend). They are then part of
            the deterministic prefix, which ends at the first measurement.
        exact_resets: Whether resets evolve as deterministic channels (a
            density-matrix or MPO backend). Otherwise a reset of a qubit that
            an earlier instruction used may collapse a superposition, so the
            prefix ends there: a saved prefix would replay one random branch
            for every shot. Resets of untouched qubits leave |0> and stay.

    Returns:
        A tuple of (prefix_circuit, suffix_circuit, cut_index).
    """
    used: set[int] = set()

    def cuts(name, targets):
        if name in STIM_MEASURE_INSTRUCTIONS:
            return True
        if name in STIM_NOISE_INSTRUCTIONS and not (deterministic_noise and name in LOCAL_NOISE):
            return True
        if name in STIM_RESET_INSTRUCTIONS and not exact_resets and used.intersection(targets):
            return True
        if name not in STIM_ANNOTATIONS:
            used.update(targets)
        return False

    if isinstance(circuit, str):
        lines = circuit.strip().splitlines()
        parsed = []
        for line in lines:
            name, args, targets = _parse_instruction_line(line)
            if name:
                parsed.append((name, args, targets, line))
        has_in_circuit_noise = any(n in STIM_NOISE_INSTRUCTIONS for n, _, _, _ in parsed)
        has_noise = has_in_circuit_noise or has_external_noise
        cut_idx = len(parsed)
        for idx, (name, args, targets, _) in enumerate(parsed):
            if cuts(name, targets):
                cut_idx = idx
                break
            if has_external_noise and name not in STIM_ANNOTATIONS and not (name in ("R", "RZ") and not args):
                cut_idx = idx
                break
        prefix_str = "\n".join(p[3] for p in parsed[:cut_idx])
        suffix_str = "\n".join(p[3] for p in parsed[cut_idx:])
        return prefix_str, suffix_str, cut_idx

    elif hasattr(circuit, "circuit_str"):
        prefix_str, suffix_str, cut_idx = split_stim_prefix_suffix(
            circuit.circuit_str, has_external_noise=has_external_noise,
            deterministic_noise=deterministic_noise, exact_resets=exact_resets,
        )
        return circuit.__class__(prefix_str), circuit.__class__(suffix_str), cut_idx

    else:
        flat = list(circuit.flattened())
        cut_idx = len(flat)
        for idx, inst in enumerate(flat):
            if cuts(inst.name, [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]):
                cut_idx = idx
                break
            if has_external_noise and inst.name not in STIM_ANNOTATIONS and not (inst.name in ("R", "RZ") and not inst.gate_args_copy()):
                cut_idx = idx
                break
        prefix_c = stim.Circuit()
        for inst in flat[:cut_idx]:
            prefix_c.append(inst)
        suffix_c = stim.Circuit()
        for inst in flat[cut_idx:]:
            suffix_c.append(inst)
        return prefix_c, suffix_c, cut_idx


def count_nontrivial_prefix_gates(circuit: stim.Circuit | str | Any) -> tuple[int, int]:
    """Count non-trivial (two-qubit, single-qubit) operations in a prefix circuit.

    Trivial operations include resets to ``|0>`` (R, RZ), coordinates (QUBIT_COORDS,
    SHIFT_COORDS), and ticks (TICK). Non-trivial operations include 2-qubit entangling
    gates and 1-qubit gates/rotations or non-Z basis resets (RX, RY).

    Returns:
        A tuple of (num_2q_gates, num_1q_gates).
    """
    num_2q = 0
    num_1q = 0

    TWO_QUBIT_GATES = {"CX", "CNOT", "CY", "CZ", "SWAP", *TWO_QUBIT_DECOMPOSITIONS}
    SINGLE_QUBIT_GATES = {
        "H", "X", "Y", "Z", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG",
        "T", "T_DAG", "RX", "RY", "RZ", *ONE_QUBIT_DECOMPOSITIONS,
    }

    if isinstance(circuit, str):
        lines = circuit.strip().splitlines()
        for line in lines:
            name, args, targets = _parse_instruction_line(line)
            if not name:
                continue
            if name in TWO_QUBIT_GATES:
                num_2q += len(targets) // 2
            elif name in ("RX", "RY") and not args:
                # RX and RY resets initialize to |+> and |i>, non-trivial basis prep
                num_1q += len(targets)
            elif name == "RZ" and not args:
                # RZ without args is reset to |0>, trivial
                continue
            elif name in SINGLE_QUBIT_GATES:
                num_1q += len(targets)
    elif hasattr(circuit, "circuit_str"):
        return count_nontrivial_prefix_gates(circuit.circuit_str)
    else:
        for inst in circuit.flattened():
            name = inst.name
            targets = inst.targets_copy()
            args = inst.gate_args_copy()
            if name in TWO_QUBIT_GATES:
                num_2q += len(targets) // 2
            elif name in ("RX", "RY") and not args:
                num_1q += len(targets)
            elif name == "RZ" and not args:
                continue
            elif name in SINGLE_QUBIT_GATES:
                num_1q += len(targets)

    return num_2q, num_1q


@dataclass(frozen=True)
class _MissingDetectorErrorModel:
    """Dimensions for Sinter task validation when a physical prior is unavailable."""

    num_detectors: int
    num_observables: int

    def __str__(self) -> str:
        # Sinter hashes this text before selecting a sampler. Keep it stable and
        # distinct from the serialization of every real Stim error model.
        return "Maestro: no detector error model supplied"


class NonCliffordTaskCircuit:
    """Stim-like Sinter input with non-Clifford gates and a Maestro noiseless reference."""

    def __init__(self, circuit_str: str):
        self.circuit_str = circuit_str
        annotation_lines = []
        self._has_non_clifford_gates = False
        for line in circuit_str.strip().splitlines():
            name, args, targets = _parse_instruction_line(line)
            if name in ("T", "T_DAG") or (name in ("RX", "RY", "RZ") and args):
                self._has_non_clifford_gates = True
                # Keep the qubit indices, including wires touched only by these gates.
                annotation_lines.append("I " + " ".join(map(str, targets)))
            elif name:
                annotation_lines.append(line)

        self._annotation_circuit = stim.Circuit("\n".join(annotation_lines))
        self.num_detectors = self._annotation_circuit.num_detectors
        self.num_observables = self._annotation_circuit.num_observables
        self.num_measurements = self._annotation_circuit.num_measurements

    @property
    def num_qubits(self) -> int:
        return self._annotation_circuit.num_qubits

    def compile_m2d_converter(self, *, config: maestro.SimulatorConfig | None = None,
                              qubit_order=None) -> stim.CompiledMeasurementsToDetectionEventsConverter:
        if not self._has_non_clifford_gates or not self.num_measurements or not (self.num_detectors or self.num_observables):
            return self._annotation_circuit.compile_m2d_converter()

        # Any possible noiseless record has the correct deterministic parities.
        # A fixed seed also gives gauge detectors a reproducible reference branch.
        clean = [
            (name, [] if name in MEASUREMENTS else args, targets)
            for name, args, targets in self.flattened()
            if name not in LOCAL_NOISE
        ]
        circuit, _, width = translate_stim_to_maestro(clean, qubit_order=qubit_order)
        # Preserve configured dense backends and uncapped QCSim tensor methods.
        # A truncated reference could flip every logical comparison. When no
        # configured method can supply it, QCSim MPS supports unbounded bonds.
        method = maestro.SimulationType
        fallback = (maestro.SimulatorType.QCSim, method.MatrixProductState)
        candidates = []
        if config is not None:
            requested = [(config.simulator_type, config.simulation_type)]
            if not config.fixed_backend and config.optimization_candidates:
                requested = config.optimization_candidates
            candidates = [(backend, sim) for backend, sim in requested
                          if sim in (method.Statevector, method.DensityMatrix) or
                          (backend == maestro.SimulatorType.QCSim and
                           sim in (method.MatrixProductState, method.MatrixProductOperator))]
        backend, sim = candidates[0] if candidates else fallback
        reference_config = maestro.SimulatorConfig(
            simulator_type=backend,
            simulation_type=sim,
            fixed_backend=len(candidates) < 2,
            optimization_candidates=candidates if len(candidates) > 1 else [],
            gpu_device=config.gpu_device if candidates else None,
            distributed_options=config.distributed_options if candidates else {},
            measurement_order=config.measurement_order if config is not None else "auto",
            optimize_circuit=config.optimize_circuit if config is not None else True,
            precision="double",
            singular_value_threshold=0.0,
            seed=0,
        )
        result = circuit.execute(reference_config, shots=1, return_samples=True)
        reference = samples_to_measurements(
            result["samples"], width, 1, storage_width=max(circuit.num_qubits, width)
        )[0]
        # Supply that record to Stim at the original measurement positions. MPAD
        # supplies fixed measurement results and preserves every rec[-k] offset.
        annotations = stim.Circuit()
        offset = 0
        for instruction in self._annotation_circuit:
            if instruction.name in MEASUREMENTS:
                stop = offset + len(instruction.targets_copy())
                annotations.append("MPAD", reference[offset:stop].astype(int).tolist())
                offset = stop
            elif instruction.name in STIM_ANNOTATIONS:
                annotations.append(instruction)
        return annotations.compile_m2d_converter()

    def compile_detector_sampler(self):
        # Explicitly fails because of non-Clifford gates
        raise ValueError("Circuit contains non-Clifford gates; cannot compile Stim detector sampler.")

    def detector_error_model(self, *args, **kwargs) -> stim.DetectorErrorModel | _MissingDetectorErrorModel:
        """Supply a Stim prior for Clifford input or only task dimensions for extended input."""
        if self._has_non_clifford_gates:
            # collect asks the circuit for a prior before it knows the decoder,
            # including for vacuous decoding. Supply no iterable or decodable
            # error mechanisms: only a caller-provided DEM can guide inference.
            return _MissingDetectorErrorModel(self.num_detectors, self.num_observables)
        return self._annotation_circuit.detector_error_model(*args, **kwargs)

    def flattened(self):
        instructions = []
        for line in self.circuit_str.strip().splitlines():
            name, args, targets = _parse_instruction_line(line)
            if name:
                instructions.append((name, args, targets))
        return instructions

    def __str__(self):
        return self.circuit_str


class MaestroSinterSampler(sinter.Sampler):
    """Sinter Sampler backed by Maestro's simulation engine."""

    def __init__(
        self,
        chi: int = 32,
        use_gpu: bool = False,
        noise_model: maestro.NoiseModel | None = None,
        device: str | None = None,
        config: maestro.SimulatorConfig | None = None,
        decoder: str | sinter.Decoder | None = None,
        seed: int | None = None,
        enable_checkpoint: bool = True,
        min_single_qubit_gates: int = 1,
        qubit_order: list[int] | tuple[int, ...] | None = None,
    ):
        """Initialize the Maestro Sinter sampler.

        Args:
            chi: Maximum bond dimension for Matrix Product State (MPS) simulation.
            use_gpu: Whether to use GPU acceleration.
            noise_model: Optional pre-calibrated Maestro NoiseModel, or a callable
                ``task -> NoiseModel | None`` for per-task noise (for example read
                from ``task.json_metadata``). With ``sinter.collect(num_workers>1)``
                the callable must be picklable, e.g. a module-level function.
            device: Reserved; currently unused. Select a GPU through ``config``
                (``gpu_device``) or ``use_gpu`` instead.
            config: Optional pre-configured Maestro SimulatorConfig.
            decoder: Decoder for tasks, e.g. 'pymatching', or 'vacuous' for the
                undecoded logical error rate. Required under ``sinter.collect``
                unless this sampler is registered under a built-in decoder name,
                because there ``task.decoder`` is the ``custom_decoders`` key.
            seed: Must be None for the factory. Use MaestroCompiledSampler for
                reproducible direct sampling; collect worker identities are not
                available here to partition seeded streams safely.
            enable_checkpoint: Whether to use prefix state checkpointing.
            min_single_qubit_gates: Minimum non-trivial single-qubit gates required in prefix
                to enable checkpointing (2-qubit gates always qualify).
        """
        if seed is not None or (config is not None and config.seed is not None):
            raise ValueError(
                "Seeded sampling requires MaestroCompiledSampler directly; "
                "seeded sinter.collect factory streams are not supported.")
        self.chi = chi
        self.use_gpu = use_gpu
        self.noise_model = noise_model
        self.device = device
        self.config = config
        self.decoder = decoder
        self.seed = seed
        self.enable_checkpoint = enable_checkpoint
        self.min_single_qubit_gates = min_single_qubit_gates
        self.qubit_order = tuple(qubit_order) if qubit_order is not None else None

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        """Create a compiled sampler configured for the given task."""
        noise_model = self.noise_model(task) if callable(self.noise_model) else self.noise_model
        return MaestroCompiledSampler(
            task=task,
            chi=self.chi,
            use_gpu=self.use_gpu,
            noise_model=noise_model,
            device=self.device,
            config=self.config,
            decoder=self.decoder,
            seed=self.seed,
            enable_checkpoint=self.enable_checkpoint,
            min_single_qubit_gates=self.min_single_qubit_gates,
            qubit_order=self.qubit_order,
        )


class MaestroCompiledSampler(sinter.CompiledSampler):
    """Compiled task sampler that executes shots on Maestro."""

    def __init__(
        self,
        task: sinter.Task | stim.Circuit | str,
        chi: int = 32,
        use_gpu: bool = False,
        noise_model: maestro.NoiseModel | None = None,
        device: str | None = None,
        config: maestro.SimulatorConfig | None = None,
        decoder: str | sinter.Decoder | None = None,
        seed: int | None = None,
        enable_checkpoint: bool = True,
        min_single_qubit_gates: int = 1,
        qubit_order: list[int] | tuple[int, ...] | None = None,
    ):
        if not isinstance(task, sinter.Task):
            # If passed a circuit or circuit string directly
            if isinstance(task, str):
                circuit = stim.Circuit(task)
            else:
                circuit = task
            task = sinter.Task(circuit=circuit)

        self.task = task
        self.chi = chi
        self.use_gpu = use_gpu
        self.external_noise_model = noise_model
        self.device = device
        self.decoder = decoder
        self.seed = seed
        self.enable_checkpoint = enable_checkpoint
        self.min_single_qubit_gates = min_single_qubit_gates
        self.qubit_order = tuple(qubit_order) if qubit_order is not None else None

        self.num_measurements = task.circuit.num_measurements
        self.num_qubits = getattr(task.circuit, "num_qubits", 0)

        if self.qubit_order is not None and (
            len(self.qubit_order) != self.num_qubits or
            set(self.qubit_order) != set(range(self.num_qubits))
        ):
            raise ValueError("qubit_order must be a permutation of all circuit qubit indices")
        stream_seed = seed if seed is not None else (config.seed if config is not None else None)
        self._rng = np.random.default_rng(stream_seed)
        self.qc, _, self.num_circuit_meas = translate_stim_to_maestro(
            task.circuit, qubit_order=self.qubit_order)
        if self.num_circuit_meas != self.num_measurements:
            raise ValueError("Translated measurement count differs from the detector circuit")
        self.noise_model = noise_model
        # Pauli channels in the circuit need a fresh draw per shot, like a NoiseModel.
        # Translation drops zero-probability and targetless channels, so they don't count.
        self.has_circuit_noise = any(
            name in LOCAL_NOISE and targets and _pauli_channel_probabilities(name, args)[1][0] < 1.0
            for name, args, targets in _stim_instructions(task.circuit))

        # Configure simulator
        if config is not None:
            self.config = copy.copy(config)
        else:
            self.config = maestro.SimulatorConfig()
            self.config.simulation_type = maestro.SimulationType.MatrixProductState
            self.config.max_bond_dimension = self.chi
            if self.use_gpu:
                self.config.simulator_type = maestro.SimulatorType.Gpu

        if self.seed is not None:
            self.config.seed = self.seed

        # A full-circuit frame plan also covers clean Clifford suffixes. Compile
        # it before allocating a checkpoint, whose requested MPS backend would
        # otherwise replay those suffixes for every shot.
        self._reference_sampler = None
        compile_reference = getattr(maestro, "_compile_reference_sampler", None)
        if compile_reference is not None:
            self._reference_sampler = compile_reference(
                self.qc, self.noise_model if self.noise_model is not None else maestro.NoiseModel(), self.config)

        # Prefix state checkpointing setup
        self.checkpoint_sim = None
        self.checkpoint_cut_index = None
        self.prefix_qc = None
        self.suffix_qc = None
        self.suffix_noise_model = None

        if self.enable_checkpoint and self._reference_sampler is None:
            has_ext_noise = (
                self.external_noise_model is not None and self.external_noise_model.has_any()
            )
            # On a backend that applies channels exactly (one automatic
            # realization of an empty model), the circuit's Pauli channels and
            # resets need no per-shot draw, so the checkpoint can retain them
            # and end only at the first measurement. A pure-state prefix ends
            # at a reset that may collapse. An external model joins the prefix
            # when it is deterministic there too; otherwise it is injected per
            # suffix gate, so the prefix ends at the first gate.
            resolve = getattr(maestro, "_resolve_noise_realizations", None)
            exact_backend = resolve is not None and resolve(self.qc, maestro.NoiseModel(), self.config, 2) == 1
            prefix_noise_model = None
            if has_ext_noise and exact_backend and resolve(self.qc, self.external_noise_model, self.config, 2) == 1:
                prefix_noise_model = self.external_noise_model
            prefix_c, suffix_c, cut_idx = split_stim_prefix_suffix(
                task.circuit, has_external_noise=has_ext_noise and prefix_noise_model is None,
                deterministic_noise=exact_backend, exact_resets=exact_backend,
            )
            self.checkpoint_cut_index = cut_idx
            self.prefix_qc, _, _ = translate_stim_to_maestro(prefix_c, qubit_order=self.qubit_order)
            self.suffix_qc, suffix_in_circuit_noise, _ = translate_stim_to_maestro(suffix_c)

            if self.external_noise_model is not None:
                self.suffix_noise_model = self.external_noise_model
            else:
                self.suffix_noise_model = suffix_in_circuit_noise

            # Only enable checkpointing if prefix contains actual non-trivial gates
            # (e.g., 2-qubit gates or more than N single-qubit rotations/gates).
            # If the prefix is trivial (only initial resets or coordinates), fall back to standard direct execution.
            num_2q, num_1q = count_nontrivial_prefix_gates(prefix_c)
            has_nontrivial_prefix = (num_2q > 0) or (num_1q >= self.min_single_qubit_gates)

            prefix_num_q = getattr(self.prefix_qc, "num_qubits", 0)
            suffix_num_q = getattr(self.suffix_qc, "num_qubits", 0)
            if has_nontrivial_prefix and cut_idx > 0 and prefix_num_q > 0:
                try:
                    qubits_for_sim = max(self.num_qubits, prefix_num_q, suffix_num_q, 1)
                    self.checkpoint_sim = maestro.PrefixCheckpointedSimulator(
                        self.prefix_qc, qubits_for_sim, self.config,
                        prefix_noise_model=prefix_noise_model,
                    )
                except Exception:
                    self.checkpoint_sim = None

        # Resolve decoder for task. Under sinter.collect, task.decoder is the
        # custom_decoders key this sampler is registered under, which names a
        # decoder only when it is one of sinter's built-ins.
        decoder_to_use = self.decoder
        if decoder_to_use is None and task.decoder is not None:
            if task.decoder not in sinter.BUILT_IN_DECODERS:
                raise ValueError(
                    f"No decoder configured for task decoder {task.decoder!r}. Under "
                    "sinter.collect that is the custom_decoders key, not a decoder: "
                    "pass MaestroSinterSampler(decoder='pymatching'), or "
                    "decoder='vacuous' for the undecoded logical error rate.")
            decoder_to_use = task.decoder
        self.compiled_decoder = None
        if decoder_to_use is not None:
            decoder_obj = None
            if isinstance(decoder_to_use, sinter.Decoder):
                decoder_obj = decoder_to_use
            elif decoder_to_use in sinter.BUILT_IN_DECODERS:
                decoder_obj = sinter.BUILT_IN_DECODERS[decoder_to_use]
            if decoder_obj is None:
                raise ValueError(f"Unknown decoder {decoder_to_use!r}")
            if hasattr(decoder_obj, "compile_decoder_for_dem"):
                dem = task.detector_error_model
                if dem is None and hasattr(task.circuit, "detector_error_model"):
                    try:
                        dem = task.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
                    except ValueError:
                        try:
                            dem = task.circuit.detector_error_model(approximate_disjoint_errors=True)
                        except ValueError:
                            dem = task.circuit.detector_error_model(approximate_disjoint_errors=True, flatten_loops=True)
                if isinstance(dem, _MissingDetectorErrorModel):
                    if type(decoder_obj) is not type(sinter.BUILT_IN_DECODERS["vacuous"]):
                        raise ValueError("Stim cannot derive an error model for non-Clifford gates. "
                                         "Supply sinter.Task(detector_error_model=...) for decoding.")
                    # sample() already predicts all zeros without a compiled decoder.
                elif dem is not None:
                    self.compiled_decoder = decoder_obj.compile_decoder_for_dem(dem=dem)

    @cached_property
    def converter(self) -> stim.CompiledMeasurementsToDetectionEventsConverter:
        # Raw measurement sampling never needs a detector reference.
        if isinstance(self.task.circuit, NonCliffordTaskCircuit):
            return self.task.circuit.compile_m2d_converter(config=self.config, qubit_order=self.qubit_order)
        return self.task.circuit.compile_m2d_converter()

    def _measurements(self, shots, seed=None):
        shots = validate_shots(shots)
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        run_seed = int(rng.integers(0, 2**31))
        self.config.seed = run_seed
        if self._reference_sampler is not None:
            res = self._reference_sampler.sample(shots, run_seed)
        elif self.checkpoint_sim is not None:
            res = _sample_result(self.checkpoint_sim.execute_suffix,
                self.suffix_qc, shots=shots, noise_model=self.suffix_noise_model,
                noise_realizations=_noise_realizations(
                    self.suffix_qc, self.suffix_noise_model, self.config, shots), noise_seed=run_seed,
                num_measurements=self.num_measurements)
        elif self.has_circuit_noise or (self.noise_model is not None and self.noise_model.has_any()):
            noise_model = self.noise_model if self.noise_model is not None else maestro.NoiseModel()
            res = _sample_result(self.qc.full_noise_execute,
                noise_model, self.config, shots=shots,
                noise_realizations=_noise_realizations(self.qc, noise_model, self.config, shots), noise_seed=run_seed)
        else:
            res = _sample_result(self.qc.execute, self.config, shots=shots)
        storage_width = max(self.num_qubits if self.checkpoint_sim is not None else self.qc.num_qubits,
                            self.num_measurements)
        if "samples" in res:
            return samples_to_measurements(res["samples"], self.num_measurements, shots, storage_width=storage_width)
        return counts_to_measurements(res.get("counts", {}), self.num_measurements, shots, storage_width=storage_width)

    def sample(
        self, suggested_shots: int = 1, shots: int | None = None, seed: int | None = None
    ) -> sinter.AnonTaskStats:
        """Sample shots on Maestro and return Sinter task statistics."""
        if shots is not None:
            suggested_shots = shots
        shots_to_run = validate_shots(suggested_shots)
        t0 = time.perf_counter()
        meas_matrix = self._measurements(shots_to_run, seed)

        # Native C++ bit-packed conversion to detection events and actual observables
        dets, actual_obs = self.converter.convert(
            measurements=meas_matrix,
            bit_packed=True,
            separate_observables=True,
        )

        # Discard any shots that contain a postselected detection event
        if self.task.postselection_mask is not None:
            discarded_flags = np.any(dets & self.task.postselection_mask, axis=1)
            num_discards_1 = int(np.count_nonzero(discarded_flags))
            if num_discards_1:
                dets = dets[~discarded_flags, :]
                actual_obs = actual_obs[~discarded_flags, :]
        else:
            num_discards_1 = 0

        # Classify discards and errors
        if self.compiled_decoder is not None:
            predictions = self.compiled_decoder.decode_shots_bit_packed(
                bit_packed_detection_event_data=dets
            )
        else:
            # Uncorrected check: baseline prediction is all zeros (no correction)
            predictions = np.zeros_like(actual_obs)

        num_discards_2, num_errors = classify_discards_and_errors(
            actual_obs=actual_obs,
            predictions=predictions,
            num_obs=self.task.circuit.num_observables,
            postselected_observables_mask=self.task.postselected_observables_mask,
            out_count_observable_error_combos=None,
        )
        total_discards = num_discards_1 + num_discards_2

        t1 = time.perf_counter()
        return sinter.AnonTaskStats(
            shots=shots_to_run,
            errors=int(num_errors),
            discards=int(total_discards),
            seconds=t1 - t0,
        )

    def sample_measurements(self, shots: int = 1000, seed: int | None = None) -> np.ndarray:
        """Sample the raw measurement record.

        Args:
            shots: Number of shots to sample.
            seed: Optional random seed for simulation.

        Returns:
            A boolean numpy array of shape (shots, num_measurements), with
            columns in Stim measurement-record order. Frames, QCSim direct
            trajectories and checkpoints keep each shot's record together. The
            histogram fallbacks (MPS, MPO and tensor networks) shuffle complete
            records within each noise realization, so rows are exchangeable
            rather than chronological.
        """
        return self._measurements(shots, seed)

    def sample_detection_events(
        self, shots: int = 1000, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detection events and actual observables directly.

        Args:
            shots: Number of shots to sample.
            seed: Optional random seed for simulation.

        Returns:
            A tuple of (detection_events, actual_observables) as boolean numpy arrays.
            detection_events shape: (shots, num_detectors)
            actual_observables shape: (shots, num_observables)
        """
        meas_matrix = self.sample_measurements(shots, seed)

        dets, actual_obs = self.converter.convert(
            measurements=meas_matrix,
            bit_packed=False,
            separate_observables=True,
        )
        return dets, actual_obs
