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
        elif name == "X":
            for q in targets:
                qc.x(q)
        elif name == "Y":
            for q in targets:
                qc.y(q)
        elif name == "Z":
            for q in targets:
                qc.z(q)
        elif name == "H":
            for q in targets:
                qc.h(q)
        elif name in ("S", "SQRT_Z"):
            for q in targets:
                qc.s(q)
        elif name in ("S_DAG", "SQRT_Z_DAG"):
            for q in targets:
                qc.sdg(q)
        elif name == "SQRT_X":
            for q in targets:
                qc.sx(q)
        elif name == "SQRT_X_DAG":
            for q in targets:
                qc.sxdg(q)
        elif name == "SQRT_Y":
            for q in targets:
                qc.ry(q, math.pi / 2.0)
        elif name == "SQRT_Y_DAG":
            for q in targets:
                qc.ry(q, -math.pi / 2.0)

        # --- Parameterized single-qubit rotations & non-Clifford gates ---
        elif name == "T":
            for q in targets:
                qc.t(q)
        elif name == "T_DAG":
            for q in targets:
                qc.tdg(q)
        elif name == "RX" and args:
            angle = args[0]
            for q in targets:
                qc.rx(q, angle)
        elif name == "RY" and args:
            angle = args[0]
            for q in targets:
                qc.ry(q, angle)
        elif name == "RZ" and args:
            angle = args[0]
            for q in targets:
                qc.rz(q, angle)

        # --- Two-qubit Clifford gates ---
        elif name in ("CX", "CNOT"):
            for i in range(0, len(targets), 2):
                qc.cx(targets[i], targets[i + 1])
        elif name == "CY":
            for i in range(0, len(targets), 2):
                qc.cy(targets[i], targets[i + 1])
        elif name == "CZ":
            for i in range(0, len(targets), 2):
                qc.cz(targets[i], targets[i + 1])
        elif name == "SWAP":
            for i in range(0, len(targets), 2):
                qc.swap(targets[i], targets[i + 1])

        # --- Resets (unparameterized) ---
        elif name in ("R", "RZ"):
            for q in targets:
                qc.reset(q)
        elif name == "RX":
            for q in targets:
                qc.reset(q)
                qc.h(q)
        elif name == "RY":
            for q in targets:
                qc.reset(q)
                qc.h(q)
                qc.s(q)

        # --- Measurements and Measure-and-Resets ---
        elif name in ("M", "MZ"):
            for q in targets:
                qc.measure([(q, meas_idx)])
                meas_idx += 1
        elif name in ("MR", "MRZ"):
            for q in targets:
                qc.measure([(q, meas_idx)])
                qc.reset(q)
                meas_idx += 1
        elif name == "MX":
            for q in targets:
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.h(q)
                meas_idx += 1
        elif name == "MRX":
            for q in targets:
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.reset(q)
                qc.h(q)
                meas_idx += 1
        elif name == "MY":
            for q in targets:
                qc.sdg(q)
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.h(q)
                qc.s(q)
                meas_idx += 1
        elif name == "MRY":
            for q in targets:
                qc.sdg(q)
                qc.h(q)
                qc.measure([(q, meas_idx)])
                qc.reset(q)
                qc.h(q)
                qc.s(q)
                meas_idx += 1

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
            if name in ("T", "T_DAG") or (name in ("RX", "RY", "RZ") and len(args) > 0):
                continue
            self._clifford_lines.append(line)
            if name in ("M", "MZ", "MR", "MRZ", "MX", "MRX", "MY", "MRY"):
                num_meas += len(targets)
            elif name == "DETECTOR":
                num_dets += 1
            elif name == "OBSERVABLE_INCLUDE":
                num_obs = max(num_obs, 1)

        self._clifford_circuit = stim.Circuit("\n".join(self._clifford_lines))
        self.num_detectors = self._clifford_circuit.num_detectors
        self.num_observables = self._clifford_circuit.num_observables
        self.num_measurements = self._clifford_circuit.num_measurements

    def compile_m2d_converter(self):
        return self._clifford_circuit.compile_m2d_converter()

    def compile_detector_sampler(self):
        # Explicitly fails because of non-Clifford gates
        raise ValueError("Circuit contains non-Clifford gates; cannot compile Stim detector sampler.")

    def flattened(self):
        # Parse for translation
        lines = self.circuit_str.strip().splitlines()
        instructions = []
        for line in lines:
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
    ):
        """Initialize the Maestro Sinter sampler.

        Args:
            chi: Maximum bond dimension for Matrix Product State (MPS) simulation.
            use_gpu: Whether to use GPU acceleration.
            noise_model: Optional pre-calibrated Maestro NoiseModel.
            device: Optional target device string.
            config: Optional pre-configured Maestro SimulatorConfig.
            decoder: Optional default decoder (e.g. 'pymatching') to use for tasks.
        """
        self.chi = chi
        self.use_gpu = use_gpu
        self.noise_model = noise_model
        self.device = device
        self.config = config
        self.decoder = decoder

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

        # Compile measurement-to-detector converter
        self.converter = task.circuit.compile_m2d_converter()
        self.num_measurements = task.circuit.num_measurements

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

    def sample(self, suggested_shots: int = 1, shots: int | None = None) -> sinter.AnonTaskStats:
        """Sample shots on Maestro and return Sinter task statistics."""
        if shots is not None:
            suggested_shots = shots
        shots_to_run = max(1, suggested_shots)
        t0 = time.monotonic()

        # Execute simulation on Maestro
        if self.noise_model is not None and self.noise_model.has_any():
            res = self.qc.full_noise_execute(
                self.noise_model,
                self.config,
                shots=shots_to_run,
                noise_realizations=shots_to_run,
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
        self, shots: int = 1000
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detection events and actual observables directly.

        Args:
            shots: Number of shots to sample.

        Returns:
            A tuple of (detection_events, actual_observables) as boolean numpy arrays.
            detection_events shape: (shots, num_detectors)
            actual_observables shape: (shots, num_observables)
        """
        shots_to_run = max(1, shots)
        if self.noise_model is not None and self.noise_model.has_any():
            res = self.qc.full_noise_execute(
                self.noise_model,
                self.config,
                shots=shots_to_run,
                noise_realizations=shots_to_run,
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
