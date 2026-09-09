"""Tests for Sinter integration, circuit translation, and sampling."""

import math
import pytest
import sinter
import stim

import maestro
from maestro.sinter import (
    MaestroCompiledSampler,
    MaestroSinterSampler,
    NonCliffordTaskCircuit,
    translate_stim_to_maestro,
)


# =============================================================================
# Circuit Translation Tests
# =============================================================================


def test_translate_clifford_gates():
    circuit = stim.Circuit("""
        H 0
        X 1
        Y 0
        Z 1
        S 0
        S_DAG 1
        SQRT_X 0
        SQRT_X_DAG 1
        SQRT_Y 0
        SQRT_Y_DAG 1
        CX 0 1
        CY 1 0
        CZ 0 1
        SWAP 0 1
        M 0 1
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert num_meas == 2
    assert qc.num_qubits >= 2


def test_translate_non_clifford_gates():
    circuit_str = """
        T 0
        T_DAG 1
        RX(0.785) 0
        RY(1.57) 1
        RZ(0.5) 0
        M 0 1
    """
    qc, nm, num_meas = translate_stim_to_maestro(circuit_str)
    assert num_meas == 2


def test_translate_resets_and_measurements():
    circuit = stim.Circuit("""
        R 0 1
        X 0
        MR 0
        MZ 1
        MX 0
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert num_meas == 3


def test_translate_in_circuit_noise():
    circuit = stim.Circuit("""
        H 0
        DEPOLARIZE1(0.01) 0
        CX 0 1
        DEPOLARIZE2(0.02) 0 1
        X_ERROR(0.005) 0
        Z_ERROR(0.005) 1
        M 0 1
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert nm.has_any()
    assert nm.has_any_2q_depolarizing()
    assert num_meas == 2


def test_annotations_ignored():
    circuit = stim.Circuit("""
        QUBIT_COORDS(1, 2) 0
        TICK
        H 0
        TICK
        M 0
        DETECTOR(1, 0) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert num_meas == 1


# =============================================================================
# Sampler Unit Tests
# =============================================================================


def test_sinter_sampler_instantiation():
    sampler = MaestroSinterSampler(chi=32, use_gpu=False)
    assert isinstance(sampler, sinter.Sampler)
    assert sampler.chi == 32
    assert sampler.use_gpu is False


def test_compiled_sampler_repetition_code():
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    task = sinter.Task(circuit=circuit)
    sampler = MaestroSinterSampler(chi=16)
    compiled = sampler.compiled_sampler_for_task(task)
    stats = compiled.sample(suggested_shots=50)

    assert isinstance(stats, sinter.AnonTaskStats)
    assert stats.shots == 50
    assert stats.errors >= 0
    assert stats.seconds > 0


def test_compiled_sampler_with_external_noise_model():
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 0 1
        DETECTOR rec[-1] rec[-2]
    """)
    nm = maestro.NoiseModel()
    nm.set_depolarizing(0, 0.01)
    nm.set_depolarizing(1, 0.01)

    sampler = MaestroSinterSampler(chi=16, noise_model=nm)
    task = sinter.Task(circuit=circuit)
    compiled = sampler.compiled_sampler_for_task(task)
    stats = compiled.sample(suggested_shots=30)

    assert isinstance(stats, sinter.AnonTaskStats)
    assert stats.shots == 30
    assert stats.seconds > 0


def test_sampler_without_decoder_uncorrected():
    circuit = stim.Circuit("""
        I 0
        X_ERROR(1.0) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    # When task.decoder is None, any observable flip is an uncorrected error
    task = sinter.Task(circuit=circuit)
    sampler = MaestroSinterSampler(chi=16)
    compiled = sampler.compiled_sampler_for_task(task)
    stats = compiled.sample(suggested_shots=20)

    assert stats.shots == 20
    assert stats.errors == 20


# =============================================================================
# Sinter Integration & Collect Tests
# =============================================================================


def test_sinter_collect_clifford_baseline_repetition_code():
    """Verify that sinter.collect() with Maestro reproduces Stim error rate within 1-2 sigma on repetition code."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.02,
    )
    task = sinter.Task(circuit=circuit)

    sampler = MaestroSinterSampler(chi=16)
    shots = 500

    # Sample with Maestro
    stats_maestro = sinter.collect(
        num_workers=1,
        max_shots=shots,
        tasks=[task],
        custom_decoders={"maestro": sampler},
        decoders=["maestro"],
    )[0]

    # Sample with Stim
    stats_stim = sinter.collect(
        num_workers=1,
        max_shots=shots,
        tasks=[task],
        decoders=["vacuous"],
    )[0]

    assert stats_maestro.shots == shots
    assert stats_stim.shots == shots

    p_m = stats_maestro.errors / stats_maestro.shots
    p_s = stats_stim.errors / stats_stim.shots
    # Standard deviation of the difference between two binomial proportions
    sigma = math.sqrt(p_s * (1 - p_s) / shots + p_m * (1 - p_m) / shots)
    tol = max(2.5 * sigma, 0.04)
    assert abs(p_m - p_s) <= tol


def test_sinter_collect_clifford_baseline_surface_code():
    """Verify that sinter.collect() with Maestro reproduces Stim error rate on rotated surface code."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=3,
        rounds=1,
        after_clifford_depolarization=0.01,
    )
    task = sinter.Task(circuit=circuit)

    sampler = MaestroSinterSampler(chi=32)
    shots = 200

    # Sample with Maestro
    stats_maestro = sinter.collect(
        num_workers=1,
        max_shots=shots,
        tasks=[task],
        custom_decoders={"maestro": sampler},
        decoders=["maestro"],
    )[0]

    # Sample with Stim
    stats_stim = sinter.collect(
        num_workers=1,
        max_shots=shots,
        tasks=[task],
        decoders=["vacuous"],
    )[0]

    assert stats_maestro.shots == shots
    assert stats_stim.shots == shots

    p_m = stats_maestro.errors / stats_maestro.shots
    p_s = stats_stim.errors / stats_stim.shots
    sigma = math.sqrt(p_s * (1 - p_s) / shots + p_m * (1 - p_m) / shots)
    tol = max(2.5 * sigma, 0.05)
    assert abs(p_m - p_s) <= tol


def test_non_clifford_t_gate_fails_on_stim_succeeds_on_maestro():
    """Verify that running a surface code task with an injected T-gate fails on Stim but executes on Maestro."""
    base_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=1)

    # Inject T gate
    non_clifford_str = "T 0\n" + str(base_circuit)

    # Verify Stim fails to parse or compile detector sampler for non-Clifford T gate
    with pytest.raises(Exception):
        stim.Circuit(non_clifford_str).compile_detector_sampler()

    # Wrap in NonCliffordTaskCircuit for Sinter Task
    task_circuit = NonCliffordTaskCircuit(non_clifford_str)
    task = sinter.Task(circuit=task_circuit)

    # Verify Maestro executes it successfully via MPS and produces valid stats
    sampler = MaestroSinterSampler(chi=32)
    compiled = sampler.compiled_sampler_for_task(task)
    stats = compiled.sample(suggested_shots=50)

    assert stats.shots == 50
    assert stats.seconds > 0
    assert isinstance(stats, sinter.AnonTaskStats)


def test_sinter_collect_multiprocess_with_hardware_noise():
    """Verify that sinter.collect() works across multiprocessing workers with an external NoiseModel."""
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    task = sinter.Task(circuit=circuit)

    nm = maestro.NoiseModel()
    nm.set_all_t1(3, 0.0001)
    nm.set_all_dephasing(3, 0.005)

    sampler = MaestroSinterSampler(chi=16, noise_model=nm)
    stats = sinter.collect(
        num_workers=2,
        max_shots=100,
        tasks=[task],
        decoders=["maestro"],
        custom_decoders={"maestro": sampler},
    )[0]

    assert stats.shots == 100
    assert isinstance(stats, sinter.TaskStats)
