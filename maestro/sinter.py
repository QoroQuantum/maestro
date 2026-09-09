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

        # --- Non-Clifford single-qubit gates ---
        elif name == "T":
            for q in targets:
                qc.t(q)
        elif name == "T_DAG":
            for q in targets:
                qc.tdg(q)
        elif name == "RX":
            angle = args[0] if args else 0.0
            for q in targets:
                qc.rx(q, angle)
        elif name == "RY":
            angle = args[0] if args else 0.0
            for q in targets:
                qc.ry(q, angle)
        elif name == "RZ":
            angle = args[0] if args else 0.0
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

        # --- Resets ---
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
                qc.rx(q, math.pi / 2.0)

        # --- Measurements ---
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
            pass

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
            name, _, targets = _parse_instruction_line(line)
            if name in ("T", "T_DAG", "RX", "RY", "RZ") and name != "R":
                # Skip non-Clifford gate in Clifford baseline representation
                continue
            self._clifford_lines.append(line)
            if name in ("M", "MZ", "MR", "MRZ", "MX", "MRX"):
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
    ):
        """Initialize the Maestro Sinter sampler.

        Args:
            chi: Maximum bond dimension for Matrix Product State (MPS) simulation.
            use_gpu: Whether to use GPU acceleration.
            noise_model: Optional pre-calibrated Maestro NoiseModel.
            device: Optional target device string.
        """
        self.chi = chi
        self.use_gpu = use_gpu
        self.noise_model = noise_model
        self.device = device

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        """Create a compiled sampler configured for the given task."""
        return MaestroCompiledSampler(
            task=task,
            chi=self.chi,
            use_gpu=self.use_gpu,
            noise_model=self.noise_model,
            device=self.device,
        )


class MaestroCompiledSampler(sinter.CompiledSampler):
    """Compiled task sampler that executes shots on Maestro."""

    def __init__(
        self,
        task: sinter.Task,
        chi: int = 32,
        use_gpu: bool = False,
        noise_model: maestro.NoiseModel | None = None,
        device: str | None = None,
    ):
        self.task = task
        self.chi = chi
        self.use_gpu = use_gpu
        self.external_noise_model = noise_model
        self.device = device

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
        self.config = maestro.SimulatorConfig()
        self.config.simulation_type = maestro.SimulationType.MatrixProductState
        self.config.max_bond_dimension = self.chi
        if self.use_gpu:
            self.config.simulator_type = maestro.SimulatorType.Gpu

        # Resolve decoder for task
        self.compiled_decoder = None
        if task.decoder is not None:
            decoder_obj = None
            if task.decoder in sinter.BUILT_IN_DECODERS:
                decoder_obj = sinter.BUILT_IN_DECODERS[task.decoder]
            if decoder_obj is not None and hasattr(decoder_obj, "compile_decoder_for_dem"):
                try:
                    self.compiled_decoder = decoder_obj.compile_decoder_for_dem(
                        dem=task.detector_error_model
                    )
                except Exception:
                    self.compiled_decoder = None

    def sample(self, suggested_shots: int) -> sinter.AnonTaskStats:
        """Sample shots on Maestro and return Sinter task statistics."""
        shots = max(1, suggested_shots)
        t0 = time.monotonic()

        # Execute simulation on Maestro
        if self.noise_model is not None and self.noise_model.has_any():
            res = self.qc.full_noise_execute(
                self.noise_model,
                self.config,
                shots=shots,
                noise_realizations=shots,
            )
        else:
            res = self.qc.execute(self.config, shots=shots)

        counts = res.get("counts", {})
        if not counts:
            meas_matrix = np.zeros((shots, self.num_measurements), dtype=np.bool_)
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

        # Classify discards and errors
        num_discards = 0
        if self.compiled_decoder is not None:
            predictions = self.compiled_decoder.decode_shots_bit_packed(
                bit_packed_detection_event_data=dets
            )
            num_discards, num_errors = sinter.classify_discards_and_errors(
                actual_obs=actual_obs,
                predictions=predictions,
                num_obs=self.task.circuit.num_observables,
                postselected_observables_mask=self.task.postselected_observables_mask,
                out_count_observable_error_combos=None,
            )
        else:
            # Perfectionist / uncorrected check: any actual observable flip is an error
            num_errors = int(np.count_nonzero(np.any(actual_obs, axis=1)))
            num_discards = 0

        t1 = time.monotonic()
        return sinter.AnonTaskStats(
            shots=shots,
            errors=num_errors,
            discards=num_discards,
            seconds=t1 - t0,
        )
