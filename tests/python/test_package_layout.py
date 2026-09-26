"""Tests for package layout, exports, and import guards."""

import pytest
import sys


def test_maestro_package_exports():
    import maestro

    assert hasattr(maestro, "SimulatorConfig")
    assert hasattr(maestro, "SimulatorType")
    assert hasattr(maestro, "SimulationType")
    assert hasattr(maestro, "circuits")
    assert hasattr(maestro.circuits, "QuantumCircuit")
    assert hasattr(maestro, "NoiseModel")
    assert hasattr(maestro, "Maestro")


def test_sinter_exports():
    pytest.importorskip("sinter")
    pytest.importorskip("stim")
    import maestro.sinter
    from maestro.sinter import MaestroSinterSampler, MaestroCompiledSampler

    assert issubclass(MaestroSinterSampler, object)
    assert issubclass(MaestroCompiledSampler, object)


def test_sinter_import_guard(monkeypatch):
    """Verify that importing sinter raises the expected error message when sinter is missing."""
    import importlib

    # Hide sinter module from importlib
    monkeypatch.setitem(sys.modules, "sinter", None)

    # Reloading or importing without sinter should raise the informative error
    with pytest.raises(ImportError) as exc_info:
        # Simulate clean import attempt
        import maestro.sinter as ms
        importlib.reload(ms)

    assert "Install qoro-maestro[sinter] to use the Sinter sampler." in str(exc_info.value)


def test_simulator_config_pickle():
    """Verify that SimulatorConfig can be pickled and unpickled across multiprocessing boundaries."""
    import pickle
    import maestro

    cfg = maestro.SimulatorConfig()
    cfg.simulation_type = maestro.SimulationType.MatrixProductState
    cfg.max_bond_dimension = 64

    data = pickle.dumps(cfg)
    restored = pickle.loads(data)

    assert restored.simulation_type == maestro.SimulationType.MatrixProductState
    assert restored.max_bond_dimension == 64


def test_noise_model_pickle_keeps_appended_ou_bands():
    """Pickling replays the recorded set_* calls, so an appended band must be one."""
    import pickle
    import maestro
    from maestro.circuits import QuantumCircuit

    def x_expectation(nm):
        qc = QuantumCircuit()
        qc.h(0)
        for _ in range(10):
            qc.x(0)
        res = qc.full_noise_estimate("X", nm, noise_realizations=50, noise_seed=7)
        return res["expectation_values"][0]

    single = maestro.NoiseModel()
    single.set_correlated_ou(0, sigma=20.0, alpha=2.0, gate_time=100e-9)
    banded = maestro.NoiseModel()
    banded.set_correlated_ou(0, sigma=20.0, alpha=2.0, gate_time=100e-9)
    banded.set_correlated_ou_band(0, sigma=40.0, alpha=5.0, gate_time=100e-9)

    restored = pickle.loads(pickle.dumps(banded))

    assert x_expectation(restored) == x_expectation(banded)
    assert x_expectation(banded) != x_expectation(single)
