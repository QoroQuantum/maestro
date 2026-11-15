"""Maestro quantum circuit simulation library."""

# Standard modules carry a leading underscore so that `from maestro import *`
# exports only Maestro's API.
import ast as _ast
import copyreg as _copyreg
from copy import deepcopy as _deepcopy
import functools as _functools
import math as _math
import numbers as _numbers
import re as _re
import sys as _sys
from .maestro import *
from . import maestro as _raw_maestro
if hasattr(_raw_maestro, "_compile_reference_sampler"):
    _compile_reference_sampler = _raw_maestro._compile_reference_sampler
if hasattr(_raw_maestro, "_resolve_noise_realizations"):
    _resolve_noise_realizations = _raw_maestro._resolve_noise_realizations


_BUILD_INFO = None


def build_info():
    """The identity of this build: ``version``, ``source_revision`` (Maestro's Git
    revision) and ``qcsim_revision`` (that of the QCSim sources it was compiled
    against), each ``"unknown"`` where the build could not tell. The revisions are
    those CMake saw when it configured the build."""
    global _BUILD_INFO
    if _BUILD_INFO is None:
        import json

        _BUILD_INFO = dict(json.loads(_raw_maestro._capabilities_json()).get("build", {}))
    return dict(_BUILD_INFO)


def __getattr__(name):
    # The version is asked of the build when first wanted, not at every import.
    if name == "__version__":
        return build_info().get("version", "unknown")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Re-export submodules
circuits = _raw_maestro.circuits
_sys.modules[__name__ + ".circuits"] = circuits


#: Setters whose native channel kind is not their name without ``set_``, a
#: leading ``all_`` or a trailing ``_from_time``.
_NATIVE_KIND_EXCEPTIONS = {
    "set_qubit_noise": "pauli",
    "set_idle_noise": "idle",
    "set_kraus_channel": "kraus",
    "set_readout_error": "readout",
    "set_readout_error_symmetric": "readout",
    "set_all_readout_error": "readout",
    "set_coherent_strength": "coherent_depolarizing",
    "set_1_over_f_noise": "one_over_f",
    "set_2q_depolarizing": "pair_depolarizing",
    "set_1q_gate_depolarizing": "gate_depolarizing_1q",
    "set_all_1q_gate_depolarizing": "gate_depolarizing_1q",
    "set_2q_gate_depolarizing": "gate_depolarizing_2q",
    "set_all_2q_gate_depolarizing": "gate_depolarizing_2q",
    "set_all_correlated_from_power": "correlated_ou",
}

#: Setter parameters whose native field is not their name without ``_s``.
_NATIVE_FIELD_EXCEPTIONS = {
    "p": "probability",
    "p_error": "probability",
    "kraus_operators": "operators",
    "gate_time_s": "duration",
    "t1_time_s": "t1",
}


def _native_kind(method):
    return _NATIVE_KIND_EXCEPTIONS.get(method) or _re.sub(r"^set_(all_)?|_from_time$", "", method)


def _native_field(name):
    return _NATIVE_FIELD_EXCEPTIONS.get(name) or _re.sub(r"_s$", "", name)

_TARGET_PARAMS = frozenset({"qubit", "num_qubits", "q1", "q2", "targets"})


@_functools.lru_cache(maxsize=None)
def _setter_parameters(method):
    """A setter's parameter names in call order, read from its binding's signature."""
    signature = getattr(_raw_maestro.NoiseModel, method).__nb_signature__[0][0]
    function = _ast.parse(_re.sub(r"\\=?\d+", "...", signature) + ": ...").body[0]
    return tuple(arg.arg for arg in function.args.args[1:])


def _native_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, _numbers.Integral):
        return int(value)
    if isinstance(value, _numbers.Real):
        return "infinity" if _math.isinf(value) else float(value)
    return [_native_value(item) for item in value]


def _native_operators(operators):
    """Kraus matrices as flat row-major ``[real, imaginary]`` entries."""
    return [[[complex(entry).real, complex(entry).imag] for row in operator for entry in row]
            for operator in operators]


def _native_targets(values):
    if "num_qubits" in values:
        return [[q] for q in range(values["num_qubits"])]
    if "q1" in values:
        return [[values["q1"], values["q2"]]]
    if "targets" in values:
        return [list(values["targets"])]
    return [[values["qubit"]]]


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
            @_functools.wraps(attr)
            def wrapper(*args, **kwargs):
                # Snapshot mutable inputs before the native model can change.
                saved_args, saved_kwargs = _deepcopy((args, kwargs))
                result = attr(*args, **kwargs)
                self._call_log.append((name, saved_args, saved_kwargs))
                return result
            return wrapper
        return attr

    def __reduce__(self):
        return (_rebuild_noise_model, (self._call_log,))

    def native_channels(self, num_qubits, include_readout=True):
        """The model as the ``channels`` of a native request's ``noise`` block,
        for a circuit of ``num_qubits`` qubits.

        Each recorded setter call becomes the channel that applies it, and a
        later call replaces an earlier one on the same channel and targets,
        except OU bands, which accumulate. Channels on qubits the circuit lacks
        are left out, and so are readout errors unless ``include_readout``.
        A model that re-sets T1 around thermal relaxation can depend on the
        order of those calls, which the channels do not keep, and is then
        rejected by the native request.
        """
        channels = []
        for method, args, kwargs in self._call_log:
            kind = _native_kind(method)
            if kind == "readout" and not include_readout:
                continue
            values = dict(zip(_setter_parameters(method), args)) | kwargs
            if method == "set_all_correlated_from_power":
                values["sigma"] = _math.sqrt(values.pop("power") / (
                    values["num_qubits"] * _math.pi * values["alpha"] * values["gate_time"]))
            fields = {
                _native_field(name):
                    _native_operators(value) if name == "kraus_operators" else _native_value(value)
                for name, value in values.items() if name not in _TARGET_PARAMS
            }
            for targets in _native_targets(values):
                targets = [int(q) for q in targets]
                if any(q >= num_qubits for q in targets):
                    continue
                if kind != "correlated_ou_band":
                    channels = [c for c in channels if (c["kind"], c["targets"]) != (kind, targets)]
                channels.append({"kind": kind, "targets": targets, **fields})
        return channels


def _rebuild_simulator_config(d):
    return _raw_maestro.SimulatorConfig(**d)


def _reduce_simulator_config(cfg):
    fields = _raw_maestro.SimulatorConfig._fields
    return (_rebuild_simulator_config, ({name: getattr(cfg, name) for name in fields},))


_copyreg.pickle(_raw_maestro.SimulatorConfig, _reduce_simulator_config)


def _wrap_noise_seed(fn):
    """Accept ``seed=`` as an alias of ``noise_seed=``, keeping fn's metadata."""
    @_functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        if "seed" in kwargs:
            if "noise_seed" in kwargs:
                raise TypeError("pass either seed or noise_seed, not both")
            kwargs["noise_seed"] = kwargs.pop("seed")
        return fn(*args, **kwargs)
    # nanobind publishes signatures here rather than to inspect.
    for _attribute in ("__nb_signature__", "__text_signature__"):
        if hasattr(fn, _attribute):
            try:
                setattr(_wrapped, _attribute, getattr(fn, _attribute))
            except (AttributeError, TypeError):
                pass
    return _wrapped


_NOISE_SEED_FUNCTIONS = (
    "noisy_execute",
    "coherent_execute",
    "full_noise_execute",
    "noisy_estimate_montecarlo",
    "coherent_estimate",
    "full_noise_estimate",
    "noisy_fidelity",
)

for _fn_name in _NOISE_SEED_FUNCTIONS:
    globals()[_fn_name] = _wrap_noise_seed(getattr(_raw_maestro, _fn_name))
    # The QuantumCircuit methods of the same name take seed= too.
    setattr(circuits.QuantumCircuit, _fn_name,
            _wrap_noise_seed(getattr(circuits.QuantumCircuit, _fn_name)))
# Checkpoint suffixes take seed= too.
PrefixCheckpointedSimulator.execute_suffix = _wrap_noise_seed(PrefixCheckpointedSimulator.execute_suffix)
