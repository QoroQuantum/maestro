"""Tests for Sinter integration, circuit translation, and sampling."""

import math
import numpy as np
import pytest
import sinter
import stim

import maestro
from maestro.sinter import (
    MaestroCompiledSampler,
    MaestroSinterSampler,
    NonCliffordTaskCircuit,
    count_nontrivial_prefix_gates,
    split_stim_prefix_suffix,
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


def test_translate_rx_ry_resets_functional():
    """Verify that RX and RY resets initialize to |+> and |i> respectively."""
    circuit = stim.Circuit("""
        RX 0
        MX 0
        RY 1
        MY 1
        RX 2
        Z 2
        MX 2
        RY 3
        Z 3
        MY 3
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert num_meas == 4

    cfg = maestro.SimulatorConfig()
    res = qc.execute(cfg, shots=20)
    counts = res.get("counts", {})
    # Bit 0 is 0 (RX then MX)
    # Bit 1 is 0 (RY then MY)
    # Bit 2 is 1 (RX then Z then MX)
    # Bit 3 is 1 (RY then Z then MY)
    assert counts == {"0011": 20}


def test_translate_my_and_mry():
    """Verify MY and MRY operations measure and reset in the Y basis."""
    circuit = stim.Circuit("""
        MRX 0
        MX 0
        MRY 1
        MY 1
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert num_meas == 4

    cfg = maestro.SimulatorConfig()
    res = qc.execute(cfg, shots=30)
    for bitstring in res.get("counts", {}).keys():
        # Bit 1 (MX after MRX) and Bit 3 (MY after MRY) must deterministically be '0'
        assert bitstring[1] == "0"
        assert bitstring[3] == "0"


def test_translate_y_error():
    circuit = stim.Circuit("""
        H 0
        Y_ERROR(0.05) 0
        M 0
    """)
    qc, nm, num_meas = translate_stim_to_maestro(circuit)
    assert nm.has_any()
    assert num_meas == 1


def test_translate_unsupported_instruction_raises():
    with pytest.raises(ValueError, match="Unsupported Stim instruction"):
        translate_stim_to_maestro("UNSUPPORTED_OP 0")


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


def test_compiled_sampler_with_pymatching_decoder():
    """Verify that a task with decoder='pymatching' compiles DEM and decodes without errors."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    task = sinter.Task(circuit=circuit, decoder="pymatching")
    sampler = MaestroSinterSampler(chi=16)
    compiled = sampler.compiled_sampler_for_task(task)
    assert compiled.compiled_decoder is not None

    stats = compiled.sample(suggested_shots=50)
    assert isinstance(stats, sinter.AnonTaskStats)
    assert stats.shots == 50
    assert stats.errors >= 0
    assert stats.seconds > 0


def test_compiled_sampler_postselection():
    """Verify that postselection masks count discards properly."""
    circuit = stim.Circuit("""
        I 0
        X_ERROR(1.0) 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    mask = np.array([1], dtype=np.uint8)
    task = sinter.Task(circuit=circuit, postselected_observables_mask=mask)
    sampler = MaestroSinterSampler(chi=16)
    compiled = sampler.compiled_sampler_for_task(task)
    stats = compiled.sample(suggested_shots=20)

    assert stats.shots == 20
    assert stats.discards == 20
    assert stats.errors == 0


def test_non_clifford_task_circuit_preserves_rx_ry_resets():
    circuit_str = """
        RX 0
        RY 1
        T 0
        RX(0.5) 1
        M 0 1
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
    """
    task_circuit = NonCliffordTaskCircuit(circuit_str)
    assert "RX 0" in str(task_circuit._clifford_circuit)
    assert "RY 1" in str(task_circuit._clifford_circuit)
    assert "T 0" not in str(task_circuit._clifford_circuit)
    assert "RX(0.5)" not in str(task_circuit._clifford_circuit)


def test_sampler_accepts_simulator_config():
    cfg = maestro.SimulatorConfig()
    cfg.simulation_type = maestro.SimulationType.MatrixProductState
    cfg.max_bond_dimension = 16

    sampler = MaestroSinterSampler(config=cfg)
    assert sampler.config is cfg
    task = sinter.Task(circuit=stim.Circuit("H 0\nM 0"))
    compiled = sampler.compiled_sampler_for_task(task)
    assert compiled.config is cfg
    stats = compiled.sample(10)
    assert stats.shots == 10


def test_compiled_sampler_accepts_raw_circuit_and_sample_detection_events():
    circuit = stim.Circuit("""
        H 0
        CX 0 1
        M 0 1
        DETECTOR(0, 0) rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    sampler = MaestroCompiledSampler(circuit)
    dets, obs = sampler.sample_detection_events(shots=50)
    assert dets.shape == (50, 1)
    assert obs.shape == (50, 1)
    # Bell state parity check: detector rec[-1] ^ rec[-2] is always 0
    assert not np.any(dets)


def test_compiled_sampler_shots_kwarg():
    sampler = MaestroCompiledSampler(stim.Circuit("H 0\nM 0"))
    stats = sampler.sample(shots=25)
    assert stats.shots == 25


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
    shots = 400

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
    tol = max(3.0 * sigma, 0.06)
    assert abs(p_m - p_s) <= tol


def test_sinter_collect_with_pymatching_end_to_end():
    """Verify that sinter.collect() works with multiprocessing and PyMatching decoder."""
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        distance=3,
        rounds=2,
        after_clifford_depolarization=0.01,
    )
    task = sinter.Task(circuit=circuit)
    sampler = MaestroSinterSampler(chi=16, decoder="pymatching")
    stats = sinter.collect(
        num_workers=2,
        max_shots=100,
        tasks=[task],
        decoders=["maestro"],
        custom_decoders={"maestro": sampler},
    )[0]

    assert stats.shots == 100
    assert stats.decoder == "maestro"
    assert isinstance(stats, sinter.TaskStats)


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


def test_compiled_sampler_detector_postselection():
    """Verify detector postselection discards shots when a postselected detector triggers."""
    circuit = stim.Circuit("""
        I 0
        X_ERROR(1.0) 0
        M 0
        DETECTOR rec[-1]
    """)
    mask = np.array([1], dtype=np.uint8)
    task = sinter.Task(circuit=circuit, postselection_mask=mask)
    compiled = MaestroSinterSampler(chi=16).compiled_sampler_for_task(task)
    stats = compiled.sample(15)

    assert stats.shots == 15
    assert stats.discards == 15


def test_non_clifford_task_circuit_methods():
    circuit_str = "T 0\nH 0\nM 0"
    tc = NonCliffordTaskCircuit(circuit_str)
    with pytest.raises(ValueError, match="Circuit contains non-Clifford gates"):
        tc.compile_detector_sampler()
    instructions = tc.flattened()
    assert len(instructions) == 3
    assert str(tc) == circuit_str


def test_compiled_sampler_accepts_circuit_string():
    sampler = MaestroCompiledSampler("H 0\nM 0")
    assert sampler.num_measurements == 1
    stats = sampler.sample(5)
    assert stats.shots == 5


def test_sampler_use_gpu_flag():
    sampler = MaestroSinterSampler(use_gpu=True)
    assert sampler.use_gpu is True
    task = sinter.Task(circuit=stim.Circuit("H 0\nM 0"))
    compiled = sampler.compiled_sampler_for_task(task)
    assert compiled.config.simulator_type == maestro.SimulatorType.Gpu


# =============================================================================
# Prefix State Checkpointing Tests
# =============================================================================


def test_split_stim_prefix_suffix_boundaries():
    """Verify split_stim_prefix_suffix properly detects deterministic prefix boundary."""
    # Repetition code has coordinates and initial resets
    rep_circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    prefix, suffix, cut_idx = split_stim_prefix_suffix(rep_circuit)
    assert cut_idx > 0
    assert len(prefix) > 0
    assert len(suffix) > 0
    # Suffix must contain all measurements
    assert suffix.num_measurements == rep_circuit.num_measurements

    # Surface code has coordinates and resets
    surf_circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    prefix, suffix, cut_idx = split_stim_prefix_suffix(surf_circuit)
    assert cut_idx > 0
    assert len(prefix) > 0
    assert suffix.num_measurements == surf_circuit.num_measurements

    # Circuit with early measurement
    early_meas_circ = stim.Circuit("""
        R 0 1
        H 0
        M 0
        H 1
        M 1
    """)
    prefix, suffix, cut_idx = split_stim_prefix_suffix(early_meas_circ)
    assert cut_idx == 2  # R and H are prefix (index 0, 1), M 0 is at index 2
    assert suffix.num_measurements == 2

    # Circuit with external noise model:
    # 1. If circuit starts with gates (e.g. H 0), cut_idx must be 0
    gate_circ = stim.Circuit("H 0\nM 0")
    prefix, suffix, cut_idx = split_stim_prefix_suffix(gate_circ, has_external_noise=True)
    assert cut_idx == 0
    assert len(prefix) == 0

    # 2. If circuit starts with R/TICK, cut point is at the first noisy gate (CX at index 2)
    prefix, suffix, cut_idx = split_stim_prefix_suffix(rep_circuit, has_external_noise=True)
    assert cut_idx == 2
    assert len(prefix) > 0

    # String circuit with in-circuit noise cuts at the noise channel (DEPOLARIZE1 at index 2)
    str_circ_noisy = "R 0\nH 0\nDEPOLARIZE1(0.01) 0\nM 0"
    prefix, suffix, cut_idx = split_stim_prefix_suffix(str_circ_noisy)
    assert cut_idx == 2
    assert prefix == "R 0\nH 0"

    # String circuit with external noise cuts before the first gate (H 0 at index 1)
    prefix_ext, suffix_ext, cut_ext = split_stim_prefix_suffix(str_circ_noisy, has_external_noise=True)
    assert cut_ext == 1
    assert prefix_ext == "R 0"

    # String circuit without noise keeps deterministic gates in prefix (cuts at M 0 at index 2)
    str_circ_noiseless = "R 0\nH 0\nM 0"
    prefix, suffix, cut_idx = split_stim_prefix_suffix(str_circ_noiseless)
    assert cut_idx == 2
    assert prefix == "R 0\nH 0"


def test_checkpoint_zero_mismatch_repetition_code():
    """Verify 0 mismatch between baseline and checkpointed simulation on repetition code."""
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    seed = 42
    shots = 200

    sampler_base = MaestroCompiledSampler(circuit, seed=seed, enable_checkpoint=False)
    dets_base, obs_base = sampler_base.sample_detection_events(shots=shots, seed=seed)

    sampler_chk = MaestroCompiledSampler(circuit, seed=seed, enable_checkpoint=True)
    assert sampler_chk.checkpoint_sim is not None
    dets_chk, obs_chk = sampler_chk.sample_detection_events(shots=shots, seed=seed)

    assert np.array_equal(dets_base, dets_chk)
    assert np.array_equal(obs_base, obs_chk)


def test_checkpoint_zero_mismatch_surface_code():
    """Verify 0 mismatch between baseline and checkpointed simulation on rotated surface code."""
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3)
    seed = 42
    shots = 200

    sampler_base = MaestroCompiledSampler(circuit, seed=seed, enable_checkpoint=False)
    dets_base, obs_base = sampler_base.sample_detection_events(shots=shots, seed=seed)

    sampler_chk = MaestroCompiledSampler(circuit, seed=seed, enable_checkpoint=True)
    assert sampler_chk.checkpoint_sim is not None
    dets_chk, obs_chk = sampler_chk.sample_detection_events(shots=shots, seed=seed)

    assert np.array_equal(dets_base, dets_chk)
    assert np.array_equal(obs_base, obs_chk)


def test_checkpoint_with_external_noise_model():
    """Verify clean execution with external NoiseModel when checkpointing is enabled."""
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    nm = maestro.NoiseModel()
    nm.set_all_t1(3, 0.0001)
    nm.set_all_dephasing(3, 0.005)

    sampler = MaestroCompiledSampler(
        circuit, noise_model=nm, enable_checkpoint=True, seed=123
    )
    stats = sampler.sample(suggested_shots=50)
    assert stats.shots == 50
    assert stats.seconds > 0

    dets, obs = sampler.sample_detection_events(shots=50, seed=123)
    assert dets.shape[0] == 50
    assert obs.shape[0] == 50


def test_checkpoint_sinter_sample_task_stats():
    """Verify Sinter task sampling works with checkpointed simulator."""
    circuit = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=2)
    task = sinter.Task(circuit=circuit)
    sinter_sampler = MaestroSinterSampler(chi=16, enable_checkpoint=True, seed=99)
    compiled = sinter_sampler.compiled_sampler_for_task(task)
    assert compiled.checkpoint_sim is not None

    stats = compiled.sample(suggested_shots=100)
    assert stats.shots == 100
    assert stats.seconds > 0


def test_count_nontrivial_prefix_gates():
    """Verify counting non-trivial gates in prefix circuits."""
    # Pure resets and coords: (0, 0)
    c_trivial = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        R 0 1 2
        TICK
    """)
    assert count_nontrivial_prefix_gates(c_trivial) == (0, 0)

    # 1-qubit gates: H, S
    c_1q = stim.Circuit("""
        R 0 1
        H 0
        S 1
    """)
    assert count_nontrivial_prefix_gates(c_1q) == (0, 2)

    # 2-qubit gates: CX
    c_2q = stim.Circuit("""
        R 0 1
        CX 0 1
    """)
    assert count_nontrivial_prefix_gates(c_2q) == (1, 0)

    # String circuit
    s_trivial = "QUBIT_COORDS(1, 1) 0\nR 0\nTICK"
    assert count_nontrivial_prefix_gates(s_trivial) == (0, 0)
    s_nontrivial = "R 0 1\nH 0\nCX 0 1"
    assert count_nontrivial_prefix_gates(s_nontrivial) == (1, 1)


def test_trivial_prefix_falls_back_to_direct_execution():
    """Verify trivial prefix (only resets/coords) bypasses checkpoint simulator."""
    c = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        R 0 1
        TICK
        DEPOLARIZE1(0.01) 0
        M 0 1
        DETECTOR rec[-1]
    """)
    sampler = MaestroCompiledSampler(c, enable_checkpoint=True)
    # Trivial prefix -> checkpoint_sim is None, falls back to direct execution
    assert sampler.checkpoint_sim is None
    stats = sampler.sample(suggested_shots=20)
    assert stats.shots == 20


def test_nontrivial_prefix_enables_checkpoint_simulator():
    """Verify non-trivial prefix (with 2q or 1q gates) instantiates checkpoint simulator."""
    c = stim.Circuit("""
        R 0 1
        H 0
        CX 0 1
        TICK
        DEPOLARIZE1(0.01) 0
        M 0 1
        DETECTOR rec[-1]
    """)
    sampler = MaestroCompiledSampler(c, enable_checkpoint=True)
    assert sampler.checkpoint_sim is not None
    stats = sampler.sample(suggested_shots=20)
    assert stats.shots == 20


