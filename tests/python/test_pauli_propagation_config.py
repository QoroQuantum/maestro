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


@pytest.mark.parametrize("order,strategy", [("auto", "terminal_v1"), ("written", "trajectory_v1")])
@pytest.mark.parametrize("method", [maestro.SimulationType.PauliPropagator, maestro.SimulationType.Statevector,
                                    maestro.SimulationType.MatrixProductState, maestro.SimulationType.DensityMatrix])
def test_measurement_order_preserves_entangled_readout(method, order, strategy):
    prefix = maestro.circuits.QuantumCircuit()
    suffix = maestro.circuits.QuantumCircuit()
    full = maestro.circuits.QuantumCircuit()
    for circuit in (prefix, full):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.swap(1, 2)
    for circuit in (suffix, full):
        circuit.measure([(0, 0)])
        circuit.x(2)
        circuit.measure([(2, 1)])

    options = {"pp_max_pauli_weight": 2, "pp_coefficient_threshold": 1e-8} if method == maestro.SimulationType.PauliPropagator else {}
    config = maestro.SimulatorConfig(simulation_type=method, measurement_order=order,
                                     fixed_backend=True, max_simulators=1, seed=23, **options)
    shots = 128
    counts = full.execute(config=config, shots=shots)["counts"]
    assert set(counts) == {"010", "100"} and sum(counts.values()) == shots
    raw = full.execute(config=config, shots=shots, return_samples=True)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 3, config)
    saved = checkpoint.execute_suffix(suffix, shots=shots, num_measurements=3, return_samples=True)
    for result in (raw, saved):
        assert result["sampling_strategy"] == strategy
        rows = result["samples"]
        assert rows.shape == (shots, 3)
        assert (rows[:, 0] != rows[:, 1]).all()
        assert not rows[:, 2].any()
        assert 0 < rows[:, 0].sum() < shots


@pytest.mark.parametrize("order", [None, "auto", "written"])
@pytest.mark.parametrize("candidate", [False, True], ids=["fixed", "candidate"])
def test_pauli_weight_cap_changes_even_an_uncoupled_readout(order, candidate):
    prefix = maestro.circuits.QuantumCircuit()
    suffix = maestro.circuits.QuantumCircuit()
    full = maestro.circuits.QuantumCircuit()
    for circuit in (full, suffix):
        circuit.measure([(0, 0)])
        circuit.cx(1, 2)
        circuit.rz(1, .8)
        circuit.cx(1, 2)
        circuit.measure([(2, 1)])
    # With automatic scheduling, backward propagation of Z0 Z2 produces
    # Z0 Z1 Z2 and the weight cap drops it. This gives P(c1=1) = 1/4.
    # In written order Z2 passes through both CXs before reaching q0's
    # projector. It never exceeds weight two, and P(c1=1) = 0 exactly.
    options = {"measurement_order": order} if order is not None else {}
    method = maestro.SimulationType.PauliPropagator
    if candidate:
        options["optimization_candidates"] = [(maestro.SimulatorType.QCSim, method)]
    else:
        options.update(fixed_backend=True, simulation_type=method)
    config = maestro.SimulatorConfig(pp_max_pauli_weight=2, pp_gates_between_trims=1,
                                     pp_gates_between_deduplications=1, max_simulators=1, seed=31, **options)
    shots = 512
    expected = 0 if order == "written" else .25
    result = full.execute(config=config, shots=shots)
    counts = result["counts"]
    raw = full.execute(config=config, shots=shots, return_samples=True)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 3, config)
    saved = checkpoint.execute_suffix(suffix, shots=shots, num_measurements=3, return_samples=True)
    assert all(entry["method"] == method.value for entry in (result, raw, saved))
    for probability in (counts.get("010", 0) / shots, raw["samples"][:, 1].mean(), saved["samples"][:, 1].mean()):
        assert probability == pytest.approx(expected, abs=.06 if expected else 0.)


@pytest.mark.parametrize("method", [maestro.SimulationType.PauliPropagator, maestro.SimulationType.Statevector])
def test_written_order_executes_every_terminal_measurement(method):
    prefix = maestro.circuits.QuantumCircuit()
    full = maestro.circuits.QuantumCircuit()
    suffix = maestro.circuits.QuantumCircuit()
    for circuit in (full, prefix):
        circuit.cx(1, 2)
        circuit.rz(1, .8)
        circuit.cx(1, 2)
    for circuit in (full, suffix):
        circuit.measure([(2, 1)])
        circuit.measure([(0, 1)])
        circuit.measure([(2, 0)])
    # The first read is overwritten. It still inserts q2's projector before
    # measuring q0, which gives P(c1=1)=1/4 at weight two. Removing that
    # read and sampling the final writes jointly can flip c0 instead.
    pauli = method == maestro.SimulationType.PauliPropagator
    limits = dict(pp_max_pauli_weight=2, pp_gates_between_trims=1, pp_gates_between_deduplications=1) if pauli else {}
    config = maestro.SimulatorConfig(simulation_type=method, fixed_backend=True, max_simulators=1,
                                     measurement_order="written", seed=31, **limits)
    shots = 512
    counts = full.execute(config=config, shots=shots)["counts"]
    assert set(counts) == ({"000", "010"} if pauli else {"000"})
    raw = full.execute(config=config, shots=shots, return_samples=True)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 3, config)
    saved = checkpoint.execute_suffix(suffix, shots=shots, num_measurements=3, return_samples=True)
    for result in (raw, saved):
        assert not result["samples"][:, [0, 2]].any()
        assert result["samples"][:, 1].mean() == pytest.approx(.25 if pauli else 0., abs=.06 if pauli else 0.)


def test_measurement_order_configuration_roundtrip():
    config = maestro.SimulatorConfig(measurement_order="written")
    assert pickle.loads(pickle.dumps(config)).measurement_order == "written"
    with pytest.raises(ValueError, match="measurement_order"):
        maestro.SimulatorConfig(measurement_order="invalid")
    with pytest.raises(ValueError, match="measurement_order"):
        config.measurement_order = "invalid"
    assert config.measurement_order == "written"
    config.measurement_order = "auto"
    assert config.measurement_order == "auto"


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
    # The GPU propagator runs in single precision unless asked otherwise
    # (7e-8 from QCSim here); in double it matches QCSim.
    cpu = _estimate(maestro.SimulatorType.QCSim)
    double = _estimate(gpu, pp_workers=1, precision="double")
    assert double == pytest.approx(cpu, abs=1e-9)
