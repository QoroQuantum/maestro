"""Maestro quantum circuit simulation library."""

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
