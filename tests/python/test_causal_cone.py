"""Regression coverage for expectation cone sizing and full-execution fallback."""
import math

import maestro
import pytest


def test_large_statevector_estimate_allocates_only_the_cone():
    circuit = maestro.QasmToCirc().parse_and_translate(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[118] q; '
        'ry(0.7) q[0]; cx q[0], q[1]; h q[117];'
    )
    result = circuit.estimate(
        "Z" + "I" * 117,
        config=maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector),
    )
    assert result["expectation_values"] == pytest.approx([math.cos(0.7)], rel=0, abs=1e-12)
    assert result["method"] == maestro.SimulationType.Statevector.value


@pytest.mark.parametrize("enabled", [False, True])
def test_classical_control_falls_back_without_losing_dependency(enabled):
    circuit = maestro.QasmToCirc().parse_and_translate(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[3] c; '
        'x q[1]; c[2] = measure q[1]; if (c[2]) { x q[0]; }'
    )
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector)
    config.enable_causal_cone_reduction = enabled
    result = circuit.estimate("ZII", config=config)
    assert result["expectation_values"] == pytest.approx([-1.0], rel=0, abs=1e-12)


@pytest.mark.parametrize("enabled", [False, True])
def test_full_width_fallback_resizes_initial_backend(enabled):
    circuit = maestro.QasmToCirc().parse_and_translate(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; '
        'ry(0.7) q[0]; cx q[0], q[1]; x q[2];'
    )
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector)
    config.enable_causal_cone_reduction = enabled
    result = circuit.estimate("ZZZ", config=config)
    assert result["expectation_values"] == pytest.approx([-1.0], rel=0, abs=1e-12)


def test_cone_options_survive_pickling():
    import pickle

    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector)
    assert config.enable_causal_cone_reduction is True
    assert config.causal_cone_statevector_threshold == 20
    config.enable_causal_cone_reduction = False
    config.causal_cone_statevector_threshold = 7
    restored = pickle.loads(pickle.dumps(config))
    assert restored.enable_causal_cone_reduction is False
    assert restored.causal_cone_statevector_threshold == 7
