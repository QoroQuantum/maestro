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


_CONFIG_ATTRS = (
    "disable_optimized_swapping",
    "gpu_device",
    "distributed_options",
    "lookahead_depth",
    "max_bond_dimension",
    "mpo_hermitize_after_truncation",
    "mpo_kraus_completeness_check",
    "mpo_restore_trace_after_truncation",
    "mpo_use_gesvd",
    "mpo_use_gesvdj",
    "mpo_use_gesvdp",
    "mpo_use_gesvdr",
    "mps_measure_no_collapse",
    "mps_use_gesvd",
    "mps_use_gesvdj",
    "mps_use_gesvdp",
    "mps_use_gesvdr",
    "path_integral_threshold",
    "pp_coefficient_threshold",
    "pp_pauli_weight_threshold",
    "pp_steps_between_deduplications",
    "pp_steps_between_trims",
    "precision",
    "seed",
    "simulation_type",
    "simulator_type",
    "singular_value_threshold",
    "tensor_network_use_gesvd",
    "tensor_network_use_gesvdj",
    "tensor_network_use_gesvdp",
    "tensor_network_use_gesvdr",
    "truncation_mode",
    "use_double_precision",
)


def _rebuild_simulator_config(d):
    cfg = _raw_maestro.SimulatorConfig()
    for k, v in d.items():
        try:
            setattr(cfg, k, v)
        except Exception:
            pass
    return cfg


def _reduce_simulator_config(cfg):
    d = {}
    for attr in _CONFIG_ATTRS:
        try:
            d[attr] = getattr(cfg, attr)
        except Exception:
            pass
    return (_rebuild_simulator_config, (d,))


copyreg.pickle(_raw_maestro.SimulatorConfig, _reduce_simulator_config)
