"""Maestro quantum circuit simulation library."""

import copyreg
import sys
from .maestro import *
from . import maestro as _raw_maestro

# Re-export submodules
circuits = _raw_maestro.circuits
sys.modules[__name__ + ".circuits"] = circuits


def _rebuild_noise_model(call_log):
    nm = NoiseModel()
    for name, args, kwargs in call_log:
        getattr(super(NoiseModel, nm), name)(*args, **kwargs)
        nm._call_log.append((name, args, kwargs))
    return nm


class NoiseModel(_raw_maestro.NoiseModel):
    """Python wrapper around C++ NoiseModel enabling multiprocessing and pickle serialization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_call_log", [])

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if callable(attr) and name.startswith("set_"):
            def wrapper(*args, **kwargs):
                self._call_log.append((name, args, kwargs))
                return attr(*args, **kwargs)
            return wrapper
        return attr

    def __reduce__(self):
        return (_rebuild_noise_model, (self._call_log,))


def _rebuild_simulator_config(d):
    return _raw_maestro.SimulatorConfig(**d)


def _reduce_simulator_config(cfg):
    fields = _raw_maestro.SimulatorConfig._fields
    return (_rebuild_simulator_config, ({name: getattr(cfg, name) for name in fields},))


copyreg.pickle(_raw_maestro.SimulatorConfig, _reduce_simulator_config)


def _wrap_noise_seed(fn):
    def _wrapped(*args, **kwargs):
        if "seed" in kwargs and "noise_seed" not in kwargs:
            kwargs["noise_seed"] = kwargs.pop("seed")
        return fn(*args, **kwargs)
    _wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    _wrapped.__doc__ = getattr(fn, "__doc__", None)
    return _wrapped


for _fn_name in (
    "noisy_execute",
    "coherent_execute",
    "full_noise_execute",
    "noisy_estimate_monte_carlo",
    "coherent_estimate",
    "full_noise_estimate",
    "noisy_fidelity",
):
    if hasattr(_raw_maestro, _fn_name):
        globals()[_fn_name] = _wrap_noise_seed(getattr(_raw_maestro, _fn_name))

