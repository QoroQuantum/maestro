"""Sinter interface for Maestro quantum circuit simulator.

This module provides MaestroSinterSampler and MaestroCompiledSampler,
enabling integration between the Stim/Sinter QEC framework and Maestro's
tensor-network/GPU simulation engine.
"""

from __future__ import annotations

import collections
import math
import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import sinter
    import stim
except ImportError:
    raise ImportError("Install qoro-maestro[sinter] to use the Sinter sampler.")

import maestro

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
        return "", [], []

    name = m.group(1).upper()
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

    return name, args, targets


def translate_stim_to_maestro(
    circuit: stim.Circuit | str | Any,
) -> tuple[maestro.circuits.QuantumCircuit, maestro.NoiseModel, int]:
    """Translate a Stim circuit or circuit string into a Maestro QuantumCircuit and NoiseModel.

    Args:
        circuit: A stim.Circuit or string representation.

    Returns:
        A tuple of (QuantumCircuit, NoiseModel, num_measurements).
    """
    qc = maestro.circuits.QuantumCircuit()
    nm = maestro.NoiseModel()
    meas_idx = 0
    active_qubits: set[int] = set()

    if isinstance(circuit, str):
        lines = circuit.strip().splitlines()
        instructions = []
        for line in lines:
            name, args, targets = _parse_instruction_line(line)
            if name:
                instructions.append((name, args, targets))
    elif hasattr(circuit, "circuit_str"):
        lines = circuit.circuit_str.strip().splitlines()
        instructions = []
        for line in lines:
            name, args, targets = _parse_instruction_line(line)
            if name:
                instructions.append((name, args, targets))
    elif isinstance(circuit, list):
        instructions = circuit
    else:
        instructions = []
        for inst in circuit.flattened():
            name = inst.name
            args = inst.gate_args_copy()
            targets = [t.qubit_value for t in inst.targets_copy() if t.is_qubit_target]
            instructions.append((name, args, targets))

    for name, args, targets in instructions:
        # --- Single-qubit Clifford gates ---
        if name == "I":
            for q in targets:
                qc.rz(q, 0.0)
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
        elif name == "SQRT_Y":
            for q in targets:
                qc.ry(q, math.pi / 2.0)
                active_qubits.add(q)
        elif name == "SQRT_Y_DAG":
            for q in targets:
                qc.ry(q, -math.pi / 2.0)
                active_qubits.add(q)

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
        elif name in ("R", "RZ"):
            for q in targets:
                if q in active_qubits:
                    qc.reset(q)
        elif name == "RX":
            for q in targets:
                if q in active_qubits:
                    qc.reset(q)
                qc.h(q)
                active_qubits.add(q)
        elif name == "RY":
            for q in targets:
                if q in active_qubits:
                    qc.reset(q)
                qc.h(q)
                qc.s(q)
                active_qubits.add(q)

        # --- Measurements and Measure-and-Resets ---
        elif name in ("M", "MZ"):
            for q in targets:
                qc.measure([(q, meas_idx)])
                meas_idx += 1
                active_qubits.add(q)
        elif name in ("MR", "MRZ"):
            for q in targets:
                qc.measure([(q, meas_idx)])
                qc.reset(q)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MX":
            for q in targets:
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.h(q)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MRX":
            for q in targets:
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.reset(q)
                qc.h(q)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MY":
            for q in targets:
                qc.sdg(q)
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.h(q)
                qc.s(q)
                meas_idx += 1
                active_qubits.add(q)
        elif name == "MRY":
            for q in targets:
                qc.sdg(q)
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.reset(q)
                qc.h(q)
                qc.s(q)
                meas_idx += 1
                active_qubits.add(q)

        # --- In-circuit Pauli noise channels ---
        elif name == "DEPOLARIZE1":
            p = args[0] if args else 0.0
            if p > 0:
                for q in targets:
                    nm.set_depolarizing(q, p)
        elif name == "DEPOLARIZE2":
            p = args[0] if args else 0.0
            if p > 0:
                for i in range(0, len(targets), 2):
                    nm.set_2q_depolarizing(targets[i], targets[i + 1], p)
        elif name == "X_ERROR":
            p = args[0] if args else 0.0
            if p > 0:
                for q in targets:
                    nm.set_bit_flip(q, p)
        elif name == "Z_ERROR":
            p = args[0] if args else 0.0
            if p > 0:
                for q in targets:
                    nm.set_dephasing(q, p)
        elif name == "Y_ERROR":
            p = args[0] if args else 0.0
            if p > 0:
                e0 = [[math.sqrt(1.0 - p), 0.0], [0.0, math.sqrt(1.0 - p)]]
                e1 = [[0.0, -1j * math.sqrt(p)], [1j * math.sqrt(p), 0.0]]
                for q in targets:
                    nm.set_kraus_channel([q], [e0, e1])

        # --- Annotations (skipped for simulation) ---
        elif name in (
            "TICK",
            "DETECTOR",
            "OBSERVABLE_INCLUDE",
            "QUBIT_COORDS",
            "SHIFT_COORDS",
            "MPAD",
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

STIM_ANNOTATIONS = {
    "TICK",
    "QUBIT_COORDS",
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS",
    "MPAD",
}


def split_stim_prefix_suffix(
    circuit: stim.Circuit | str | Any,
    has_external_noise: bool = False,
) -> tuple[Any, Any, int]:
    """Split a Stim circuit into a deterministic prefix and suffix.

    The prefix contains all deterministic operations prior to the first
    noise channel or stochastic measurement.

    Args:
        circuit: A stim.Circuit, circuit string, or wrapper object.
        has_external_noise: Whether an external noise model applies to gates.

    Returns:
        A tuple of (prefix_circuit, suffix_circuit, cut_index).
    """
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
        for idx, (name, _, _, _) in enumerate(parsed):
            if name in STIM_MEASURE_INSTRUCTIONS or name in STIM_NOISE_INSTRUCTIONS:
                cut_idx = idx
                break
            if has_noise and name not in STIM_ANNOTATIONS and name not in ("R", "RZ"):
                cut_idx = idx
                break
        prefix_str = "\n".join(p[3] for p in parsed[:cut_idx])
        suffix_str = "\n".join(p[3] for p in parsed[cut_idx:])
        return prefix_str, suffix_str, cut_idx

    elif hasattr(circuit, "circuit_str"):
        prefix_str, suffix_str, cut_idx = split_stim_prefix_suffix(
            circuit.circuit_str, has_external_noise=has_external_noise
        )
        return circuit.__class__(prefix_str), circuit.__class__(suffix_str), cut_idx

    else:
        flat = list(circuit.flattened())
        has_in_circuit_noise = any(inst.name in STIM_NOISE_INSTRUCTIONS for inst in flat)
        has_noise = has_in_circuit_noise or has_external_noise
        cut_idx = len(flat)
        for idx, inst in enumerate(flat):
            if inst.name in STIM_MEASURE_INSTRUCTIONS or inst.name in STIM_NOISE_INSTRUCTIONS:
                cut_idx = idx
                break
            if has_noise and inst.name not in STIM_ANNOTATIONS and inst.name not in ("R", "RZ"):
                cut_idx = idx
                break
        prefix_c = stim.Circuit()
        for inst in flat[:cut_idx]:
            prefix_c.append(inst)
        suffix_c = stim.Circuit()
        for inst in flat[cut_idx:]:
            suffix_c.append(inst)
        return prefix_c, suffix_c, cut_idx


class NonCliffordTaskCircuit:
    """Wrapper around a Stim-like circuit that supports non-Clifford gates for Sinter tasks."""

    def __init__(self, circuit_str: str):
        self.circuit_str = circuit_str
        self._clifford_lines = []
        num_dets = 0
        num_obs = 0
        num_meas = 0

        for line in circuit_str.strip().splitlines():
            name, args, targets = _parse_instruction_line(line)
            # Skip non-Clifford gates (T, T_DAG, or parameterized rotations) in Clifford baseline representation
            if name in ("T", "T_DAG") or (name in ("RX", "RY", "RZ") and args):
                continue
            if name:
                self._clifford_lines.append(line)
                if name == "DETECTOR":
                    num_dets += 1
                elif name == "OBSERVABLE_INCLUDE":
                    num_obs += 1
                elif name in ("M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"):
                    num_meas += len(targets)

        self._clifford_circuit = stim.Circuit("\n".join(self._clifford_lines))
        self.num_detectors = num_dets
        self.num_observables = num_obs
        self.num_measurements = num_meas

    @property
    def num_qubits(self) -> int:
        return self._clifford_circuit.num_qubits

    def compile_m2d_converter(self) -> stim.CompiledMeasurementTracker:
        return self._clifford_circuit.compile_m2d_converter()

    def compile_detector_sampler(self):
        # Explicitly fails because of non-Clifford gates
        raise ValueError("Circuit contains non-Clifford gates; cannot compile Stim detector sampler.")

    def detector_error_model(self, *args, **kwargs) -> stim.DetectorErrorModel:
        return self._clifford_circuit.detector_error_model(*args, **kwargs)

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
    ):
        """Initialize the Maestro Sinter sampler.

        Args:
            chi: Maximum bond dimension for Matrix Product State (MPS) simulation.
            use_gpu: Whether to use GPU acceleration.
            noise_model: Optional pre-calibrated Maestro NoiseModel.
            device: Optional target device string.
            config: Optional pre-configured Maestro SimulatorConfig.
            decoder: Optional default decoder (e.g. 'pymatching') to use for tasks.
            seed: Optional default random seed for reproducible sampling.
            enable_checkpoint: Whether to use prefix state checkpointing.
        """
        self.chi = chi
        self.use_gpu = use_gpu
        self.noise_model = noise_model
        self.device = device
        self.config = config
        self.decoder = decoder
        self.seed = seed
        self.enable_checkpoint = enable_checkpoint

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        """Create a compiled sampler configured for the given task."""
        return MaestroCompiledSampler(
            task=task,
            chi=self.chi,
            use_gpu=self.use_gpu,
            noise_model=self.noise_model,
            device=self.device,
            config=self.config,
            decoder=self.decoder,
            seed=self.seed,
            enable_checkpoint=self.enable_checkpoint,
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

        # Compile measurement-to-detector converter
        self.converter = task.circuit.compile_m2d_converter()
        self.num_measurements = task.circuit.num_measurements
        self.num_qubits = getattr(task.circuit, "num_qubits", 0)

        # Translate circuit to Maestro
        self.qc, in_circuit_noise, self.num_circuit_meas = translate_stim_to_maestro(task.circuit)

        # Merge noise models: external model takes precedence if provided,
        # otherwise use in-circuit noise
        if self.external_noise_model is not None:
            self.noise_model = self.external_noise_model
        else:
            self.noise_model = in_circuit_noise

        # Configure simulator
        if config is not None:
            self.config = config
        else:
            self.config = maestro.SimulatorConfig()
            self.config.simulation_type = maestro.SimulationType.MatrixProductState
            self.config.max_bond_dimension = self.chi
            if self.use_gpu:
                self.config.simulator_type = maestro.SimulatorType.Gpu

        if self.seed is not None:
            self.config.seed = self.seed

        # Prefix state checkpointing setup
        self.checkpoint_sim = None
        self.prefix_qc = None
        self.suffix_qc = None
        self.suffix_noise_model = None

        if self.enable_checkpoint:
            has_ext_noise = (
                self.external_noise_model is not None and self.external_noise_model.has_any()
            )
            prefix_c, suffix_c, cut_idx = split_stim_prefix_suffix(
                task.circuit, has_external_noise=has_ext_noise
            )
            self.prefix_qc, _, _ = translate_stim_to_maestro(prefix_c)
            self.suffix_qc, suffix_in_circuit_noise, _ = translate_stim_to_maestro(suffix_c)

            if self.external_noise_model is not None:
                self.suffix_noise_model = self.external_noise_model
            else:
                self.suffix_noise_model = suffix_in_circuit_noise

            # Determine if prefix has any operations
            prefix_num_q = getattr(self.prefix_qc, "num_qubits", 0)
            suffix_num_q = getattr(self.suffix_qc, "num_qubits", 0)
            has_prefix_ops = cut_idx > 0 and prefix_num_q > 0
            if has_prefix_ops:
                try:
                    qubits_for_sim = max(self.num_qubits, prefix_num_q, suffix_num_q, 1)
                    self.checkpoint_sim = maestro.PrefixCheckpointedSimulator(
                        self.prefix_qc, qubits_for_sim, self.config
                    )
                except Exception:
                    self.checkpoint_sim = None

        # Resolve decoder for task
        decoder_to_use = self.decoder if self.decoder is not None else task.decoder
        self.compiled_decoder = None
        if decoder_to_use is not None and decoder_to_use != "maestro":
            decoder_obj = None
            if isinstance(decoder_to_use, sinter.Decoder):
                decoder_obj = decoder_to_use
            elif decoder_to_use in sinter.BUILT_IN_DECODERS:
                decoder_obj = sinter.BUILT_IN_DECODERS[decoder_to_use]
            if decoder_obj is not None and hasattr(decoder_obj, "compile_decoder_for_dem"):
                dem = task.detector_error_model
                if dem is None and hasattr(task.circuit, "detector_error_model"):
                    try:
                        dem = task.circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
                    except ValueError:
                        try:
                            dem = task.circuit.detector_error_model(approximate_disjoint_errors=True)
                        except ValueError:
                            dem = task.circuit.detector_error_model(approximate_disjoint_errors=True, flatten_loops=True)
                if dem is not None:
                    self.compiled_decoder = decoder_obj.compile_decoder_for_dem(dem=dem)

    def sample(
        self, suggested_shots: int = 1, shots: int | None = None, seed: int | None = None
    ) -> sinter.AnonTaskStats:
        """Sample shots on Maestro and return Sinter task statistics."""
        if shots is not None:
            suggested_shots = shots
        shots_to_run = max(1, suggested_shots)
        t0 = time.monotonic()
        run_seed = seed if seed is not None else self.seed

        # Execute simulation on Maestro
        if self.checkpoint_sim is not None:
            res = self.checkpoint_sim.execute_suffix(
                self.suffix_qc,
                shots=shots_to_run,
                noise_model=self.suffix_noise_model,
                noise_realizations=shots_to_run,
                seed=run_seed,
                num_measurements=self.num_measurements,
            )
        elif self.noise_model is not None and self.noise_model.has_any():
            res = self.qc.full_noise_execute(
                self.noise_model,
                self.config,
                shots=shots_to_run,
                noise_realizations=shots_to_run,
                seed=run_seed,
            )
        else:
            res = self.qc.execute(self.config, shots=shots_to_run)

        counts = res.get("counts", {})
        if not counts:
            meas_matrix = np.zeros((shots_to_run, self.num_measurements), dtype=np.bool_)
        else:
            keys = list(counts.keys())
            weights = [counts[k] for k in keys]
            # Convert bitstrings to 2D boolean array (unpacked format for m2d converter)
            # In Maestro, classical bit k is at index k of bitstring
            unique_matrix = np.array(
                [[int(c) for c in k[: self.num_measurements]] for k in keys],
                dtype=np.bool_,
            )
            # Vectorized expansion across all shots
            meas_matrix = np.repeat(unique_matrix, weights, axis=0)

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

        t1 = time.monotonic()
        return sinter.AnonTaskStats(
            shots=shots_to_run,
            errors=num_errors,
            discards=total_discards,
            seconds=t1 - t0,
        )

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
        shots_to_run = max(1, shots)
        run_seed = seed if seed is not None else self.seed
        if self.checkpoint_sim is not None:
            res = self.checkpoint_sim.execute_suffix(
                self.suffix_qc,
                shots=shots_to_run,
                noise_model=self.suffix_noise_model,
                noise_realizations=shots_to_run,
                seed=run_seed,
                num_measurements=self.num_measurements,
            )
        elif self.noise_model is not None and self.noise_model.has_any():
            res = self.qc.full_noise_execute(
                self.noise_model,
                self.config,
                shots=shots_to_run,
                noise_realizations=shots_to_run,
                seed=run_seed,
            )
        else:
            res = self.qc.execute(self.config, shots=shots_to_run)

        counts = res.get("counts", {})
        if not counts:
            meas_matrix = np.zeros((shots_to_run, self.num_measurements), dtype=np.bool_)
        else:
            keys = list(counts.keys())
            weights = [counts[k] for k in keys]
            unique_matrix = np.array(
                [[int(c) for c in k[: self.num_measurements]] for k in keys],
                dtype=np.bool_,
            )
            meas_matrix = np.repeat(unique_matrix, weights, axis=0)

        dets, actual_obs = self.converter.convert(
            measurements=meas_matrix,
            bit_packed=False,
            separate_observables=True,
        )
        return dets, actual_obs
