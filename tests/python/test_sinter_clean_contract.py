"""Regression tests for clean translation and independent sampling calls."""
import copy
import numpy as np
import pytest
import stim
import maestro
import maestro.sinter as bridge


def test_initial_resets_allocate_qubits():
    qc, _, count = bridge.translate_stim_to_maestro(stim.Circuit('R 5 2'))
    assert qc.num_qubits == 6
    assert count == 0


def test_explicit_order_preserves_repeated_measurement_record():
    qc, _, count = bridge.translate_stim_to_maestro(
        [('R', [], [2, 0, 1]), ('X', [], [2]), ('MR', [], [2]),
         ('M', [], [0, 2, 1])], qubit_order=[0, 1, 2])
    cfg = maestro.SimulatorConfig()
    cfg.simulation_type = maestro.SimulationType.Stabilizer
    assert count == 4
    assert qc.execute(cfg, shots=16)['counts'] == {'1000': 16}


@pytest.mark.parametrize('source', [stim.Circuit('R 0\nX_ERROR(0.1) 0\nM 0'),
                                    stim.Circuit('M(0.1) 0'), stim.Circuit('M !0'),
                                    'M !0', 'CX rec[-1] 0'])
def test_unsupported_noise_and_targets_reject(source):
    with pytest.raises(ValueError):
        bridge.translate_stim_to_maestro(source)


def test_identity_is_supported_by_stabilizer():
    qc, _, _ = bridge.translate_stim_to_maestro(stim.Circuit('I 0\nM 0'))
    cfg = maestro.SimulatorConfig()
    cfg.simulation_type = maestro.SimulationType.Stabilizer
    assert qc.execute(cfg, shots=4)['counts'] == {'0': 4}


def test_config_preserved_and_seed_stream_advances():
    cfg = maestro.SimulatorConfig()
    cfg.simulation_type = maestro.SimulationType.Stabilizer
    cfg.seed = 123
    c = stim.Circuit('R 0\nH 0\nM 0\nDETECTOR rec[-1]')
    a = bridge.MaestroCompiledSampler(c, config=cfg, seed=7, enable_checkpoint=False)
    b = bridge.MaestroCompiledSampler(c, config=cfg, seed=7, enable_checkpoint=False)
    first = a.sample_detection_events(257)[0]
    seed1 = a.config.seed
    second = a.sample_detection_events(257)[0]
    assert a.config.seed != seed1
    assert cfg.seed == 123
    np.testing.assert_array_equal(first, b.sample_detection_events(257)[0])
    np.testing.assert_array_equal(second, b.sample_detection_events(257)[0])


@pytest.mark.parametrize('shots', [0, -1, True, 1.5])
def test_invalid_shots_reject(shots):
    sampler = bridge.MaestroCompiledSampler(stim.Circuit('R 0\nM 0'), enable_checkpoint=False)
    with pytest.raises(ValueError, match='positive integer'):
        sampler.sample_detection_events(shots)


def test_factory_rejects_replayed_seed():
    with pytest.raises(ValueError, match='seed'):
        bridge.MaestroSinterSampler(seed=1)


@pytest.mark.parametrize('counts,width,shots', [({}, 1, 2), ({'0': 1}, 1, 2),
    ({'00': 2}, 1, 2), ({'x': 2}, 1, 2), ({'0': -1}, 1, 2),
    ({'0': True}, 1, 1), ({'0': 1.5}, 1, 1)])
def test_histogram_validation(counts, width, shots):
    from maestro._sinter_validation import counts_to_measurements
    with pytest.raises(ValueError):
        counts_to_measurements(counts, width, shots)


def test_histogram_order_and_declared_zero_padding():
    from maestro._sinter_validation import counts_to_measurements
    rows = counts_to_measurements({'1000': 2, '0100': 1}, 2, 3, storage_width=4)
    np.testing.assert_array_equal(rows, [[1, 0], [1, 0], [0, 1]])
    with pytest.raises(ValueError, match='outside'):
        counts_to_measurements({'1010': 1}, 2, 1, storage_width=4)


@pytest.mark.parametrize('order', [[0, 0], [0], [0, -1], [0, True]])
def test_bad_translation_allocation_rejects(order):
    with pytest.raises(ValueError, match='qubit_order'):
        bridge.translate_stim_to_maestro(stim.Circuit('R 0 1\nM 0 1'), qubit_order=order)


@pytest.mark.parametrize('source', ['! 0', [('M', [], [-1])], [('CX', [], [0])]])
def test_malformed_instruction_rejects(source):
    with pytest.raises(ValueError):
        bridge.translate_stim_to_maestro(source)


def test_factory_rejects_seed_in_config():
    cfg = maestro.SimulatorConfig()
    cfg.seed = 123
    with pytest.raises(ValueError, match='seed'):
        bridge.MaestroSinterSampler(config=cfg)


def test_direct_config_seed_is_reproducible():
    cfg = maestro.SimulatorConfig()
    cfg.seed = 123
    c = stim.Circuit('R 0\nH 0\nM 0\nDETECTOR rec[-1]')
    a = bridge.MaestroCompiledSampler(c, config=cfg, enable_checkpoint=False)
    b = bridge.MaestroCompiledSampler(c, config=cfg, enable_checkpoint=False)
    np.testing.assert_array_equal(a.sample_detection_events(101)[0], b.sample_detection_events(101)[0])
    assert a.config.seed == b.config.seed


@pytest.mark.parametrize('wrapped', [False, True])
def test_external_noise_prefix_stops_before_parameterized_rz(wrapped):
    source = 'R 0\nRZ(0.2) 0\nM 0'
    if wrapped:
        source = bridge.NonCliffordTaskCircuit(source)
    prefix, suffix, cut = bridge.split_stim_prefix_suffix(source, has_external_noise=True)
    assert cut == 1
    assert str(prefix).strip() == 'R 0'
    assert 'RZ(0.2)' in str(suffix)


def test_external_noise_keeps_plain_rz_reset_in_prefix():
    prefix, _, cut = bridge.split_stim_prefix_suffix(stim.Circuit('RZ 0\nH 0\nM 0'),
                                                    has_external_noise=True)
    assert cut == 1
    assert str(prefix).strip() == 'R 0'


@pytest.mark.parametrize('source', [stim.Circuit('R 0\nM 0\nMPAD 1'),
                                    'R 0\nM 0\nMPAD 1',
                                    [('R', [], [0]), ('M', [], [0]), ('MPAD', [], [1])]])
def test_record_padding_is_explicitly_unsupported(source):
    with pytest.raises(ValueError, match='MPAD'):
        bridge.translate_stim_to_maestro(source)
