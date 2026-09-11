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
