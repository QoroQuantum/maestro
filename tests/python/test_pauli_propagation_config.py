"""Pauli propagation tuning options on SimulatorConfig."""
import pickle

import maestro
import pytest

TUNING_VALUES = {
    "pp_workers": 4,
    "pp_max_fused_gates": 32,
    "pp_memory_limit_bytes": 1 << 30,
    "pp_sampling_cache_nodes": 0,
}

INT64_MAX = 2**63 - 1

# (field, value, simulator_type) triples outside the native limits.
OUT_OF_RANGE = [
    ("pp_workers", 0, maestro.SimulatorType.Gpu),
    ("pp_workers", 65, maestro.SimulatorType.Gpu),
    ("pp_workers", 1025, maestro.SimulatorType.QCSim),
    ("pp_max_fused_gates", 0, maestro.SimulatorType.Gpu),
    ("pp_max_fused_gates", 257, maestro.SimulatorType.Gpu),
    ("pp_memory_limit_bytes", 0, maestro.SimulatorType.Gpu),
    ("pp_memory_limit_bytes", INT64_MAX + 1, maestro.SimulatorType.Gpu),
    ("pp_sampling_cache_nodes", INT64_MAX + 1, maestro.SimulatorType.QCSim),
]

# Values at the edges of the native limits.
IN_RANGE = [
    ("pp_workers", 1, maestro.SimulatorType.Gpu),
    ("pp_workers", 64, maestro.SimulatorType.Gpu),
    ("pp_workers", 0, maestro.SimulatorType.QCSim),
    ("pp_workers", 1024, maestro.SimulatorType.QCSim),
    ("pp_max_fused_gates", 1, maestro.SimulatorType.Gpu),
    ("pp_max_fused_gates", 256, maestro.SimulatorType.Gpu),
    ("pp_memory_limit_bytes", INT64_MAX, maestro.SimulatorType.Gpu),
    ("pp_sampling_cache_nodes", 0, maestro.SimulatorType.QCSim),
    ("pp_sampling_cache_nodes", INT64_MAX, maestro.SimulatorType.QCSim),
]


@pytest.mark.parametrize("field", list(TUNING_VALUES))
def test_field_is_listed_and_defaults_to_none(field):
    assert field in maestro.SimulatorConfig._fields
    config = maestro.SimulatorConfig()
    assert getattr(config, field) is None
    assert f"{field}=None" in repr(config)


@pytest.mark.parametrize("field,value", list(TUNING_VALUES.items()))
def test_field_round_trips(field, value):
    config = maestro.SimulatorConfig(**{field: value})
    assert getattr(config, field) == value
    assert f"{field}={value!r}" in repr(config)
    assert getattr(pickle.loads(pickle.dumps(config)), field) == value

    config = maestro.SimulatorConfig()
    setattr(config, field, value)
    assert getattr(config, field) == value
    setattr(config, field, None)
    assert getattr(config, field) is None


@pytest.mark.parametrize("field,value,backend", IN_RANGE)
def test_limits_are_inclusive(field, value, backend):
    config = maestro.SimulatorConfig(simulator_type=backend, **{field: value})
    assert getattr(config, field) == value


@pytest.mark.parametrize("field,value,backend", OUT_OF_RANGE)
def test_constructor_rejects_out_of_range(field, value, backend):
    with pytest.raises(ValueError, match=field):
        maestro.SimulatorConfig(simulator_type=backend, **{field: value})


@pytest.mark.parametrize("field,value,backend", OUT_OF_RANGE)
def test_failed_setter_leaves_config_unchanged(field, value, backend):
    config = maestro.SimulatorConfig(simulator_type=backend)
    before = repr(config)
    with pytest.raises(ValueError, match=field):
        setattr(config, field, value)
    assert repr(config) == before


@pytest.mark.parametrize("field", list(TUNING_VALUES))
@pytest.mark.parametrize("value", [-1, 2.0])
def test_rejects_negative_and_non_integer_values(field, value):
    with pytest.raises(TypeError):
        maestro.SimulatorConfig(**{field: value})
    config = maestro.SimulatorConfig()
    with pytest.raises(TypeError):
        setattr(config, field, value)
    assert getattr(config, field) is None


def test_gpu_worker_limit_follows_the_simulator_type():
    """CPU accepts 0 workers (automatic) and up to 1024; the GPU needs 1 to 64."""
    config = maestro.SimulatorConfig(pp_workers=0)
    with pytest.raises(ValueError, match="pp_workers"):
        config.simulator_type = maestro.SimulatorType.Gpu
    assert config.simulator_type == maestro.SimulatorType.QCSim
    config.pp_workers = 8
    config.simulator_type = maestro.SimulatorType.Gpu
    assert config.pp_workers == 8


QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.3) q[0];
ry(0.5) q[1];
ry(0.7) q[2];
ry(0.9) q[3];
ry(1.1) q[4];
ry(1.3) q[5];
ry(1.5) q[6];
ry(1.7) q[7];
rx(0.2) q[1];
rx(0.6) q[6];
cx q[0],q[1];
cx q[1],q[2];
rz(0.3) q[2];
cx q[3],q[4];
rx(0.7) q[4];
cx q[5],q[6];
cx q[6],q[7];
ry(1.1) q[7];
cx q[2],q[3];
rz(0.9) q[3];
cx q[4],q[5];
"""

OBSERVABLES = [
    "ZZIIIIII", "IZZIIIII", "IIZZIIII", "IIIZZIII", "IIIIZZII", "IIIIIZZI",
    "IIIIIIZZ", "XXIIIIII", "IIIXXIII", "IIIIIXXI", "YIYIIIII", "IIIIIYIY",
]


def _estimate(backend, **knobs):
    config = maestro.SimulatorConfig(
        simulator_type=backend,
        simulation_type=maestro.SimulationType.PauliPropagator,
        fixed_backend=True,
        **knobs,
    )
    result = maestro.simple_estimate(QASM, OBSERVABLES, config=config)
    assert result["simulator"] == backend.value
    return result["expectation_values"]


def test_cpu_estimates_do_not_depend_on_worker_count():
    serial = _estimate(maestro.SimulatorType.QCSim, pp_workers=1)
    reference = _estimate(maestro.SimulatorType.QCSim)
    assert serial == pytest.approx(reference, abs=1e-12)
    for workers in (0, 4):
        values = _estimate(maestro.SimulatorType.QCSim, pp_workers=workers,
                           pp_sampling_cache_nodes=0)
        assert values == pytest.approx(serial, abs=1e-12)


@pytest.mark.skipif(not maestro.is_gpu_available(),
                    reason="GPU plugin/device unavailable")
def test_gpu_estimates_do_not_depend_on_worker_count():
    gpu = maestro.SimulatorType.Gpu
    serial = _estimate(gpu, pp_workers=1)
    parallel = _estimate(gpu, pp_workers=4)
    assert parallel == pytest.approx(serial, abs=1e-9)
    tuned = _estimate(gpu, pp_workers=4, pp_max_fused_gates=8,
                      pp_memory_limit_bytes=256 << 20,
                      pp_sampling_cache_nodes=1024)
    assert tuned == pytest.approx(serial, abs=1e-9)
    cpu = _estimate(maestro.SimulatorType.QCSim)
    assert serial == pytest.approx(cpu, abs=1e-9)
