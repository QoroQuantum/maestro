---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

(py_sinter)=
# QEC sampling with Stim and Sinter

Sample [Stim](https://github.com/quantumlib/Stim) circuits through Maestro, and
plug Maestro into [Sinter](https://pypi.org/project/sinter/) as a sampler for
error-correction benchmarks.

```{code-cell} python
import stim
from maestro.sinter import MaestroCompiledSampler

circuit = stim.Circuit.generated(
    "repetition_code:memory", rounds=3, distance=5, after_clifford_depolarization=0.01
)

maestro_events, maestro_flips = MaestroCompiledSampler(circuit).sample_detection_events(shots=5000)
stim_events, stim_flips = circuit.compile_detector_sampler(seed=1).sample(5000, separate_observables=True)

print("detection events:", maestro_events.shape)
print(f"detection event rate: Maestro {maestro_events.mean():.4f}, Stim {stim_events.mean():.4f}")
```

Maestro carries Stim's own noise exactly, so its detection-event rate matches
Stim's sampler. For Clifford circuits with Pauli noise, Maestro compiles
noisy-Clifford reference frames, as Stim does, and samples them natively
(`sampling_strategy: "reference_frame_v1"`). It also covers what a stabilizer
simulator can't: non-Clifford gates, coherent or device-level noise from a
`NoiseModel`, and MPS or density-matrix simulation.

(py_sinter_install)=
## Install

```bash
pip install "qoro-maestro[sinter]" pymatching
```

The `sinter` extra installs `stim` and `sinter`. The decoding examples below also
need [PyMatching](https://pypi.org/project/PyMatching/).

(py_sinter_classes)=
## Classes and functions

| Class or function | Description |
|-------------------|-------------|
| `MaestroCompiledSampler` | A `sinter.CompiledSampler` for one circuit. Call it directly to sample detection events or measurement records. |
| `MaestroSinterSampler` | A `sinter.Sampler`. Pass it to `sinter.collect()` in `custom_decoders`. |
| `translate_stim_to_maestro(circuit, *, qubit_order=None)` | Converts a clean Stim circuit, extended gate string or instruction tuples into `(QuantumCircuit, NoiseModel, measurement_count)`. |

Both samplers accept a `SimulatorConfig` (`config=`) and a `NoiseModel`
(`noise_model=`). `MaestroCompiledSampler` also returns raw measurement
records:

```{code-cell} python
bell = stim.Circuit("""
    H 0
    CX 0 1
    M 0 1
    DETECTOR(0, 0) rec[-1] rec[-2]
""")
records = MaestroCompiledSampler(bell).sample_measurements(shots=1000)
print(records.shape, "bits always agree:", bool((records[:, 0] == records[:, 1]).all()))
```

:::{admonition} Requires GPU access
:class: note
`MaestroCompiledSampler(..., use_gpu=True)` and GPU configs need the GPU
plugin. See {doc}`/getting-started/gpu-access`.
:::

(py_sinter_example_collect)=
## Using Maestro in `sinter collect`

Pass `MaestroSinterSampler` to `sinter.collect` through `custom_decoders`. Here
the decoder prior carries Stim's noise, while Maestro executes the clean circuit
on an MPS with its own `NoiseModel`:

```{code-cell} python
import maestro
import sinter
from maestro.sinter import MaestroSinterSampler

prior = stim.Circuit.generated(
    "surface_code:rotated_memory_z", rounds=3, distance=3, after_clifford_depolarization=0.001
)
task = sinter.Task(
    circuit=prior.without_noise(),
    detector_error_model=prior.detector_error_model(decompose_errors=True),
    json_metadata={"d": 3, "r": 3},
)

mps_config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
noise = maestro.NoiseModel()
for q in range(prior.num_qubits):
    noise.set_depolarizing(q, 0.001)

stats = sinter.collect(
    num_workers=2,
    max_shots=2_000,
    max_errors=50,
    tasks=[task],
    decoders=["maestro"],
    custom_decoders={
        "maestro": MaestroSinterSampler(config=mps_config, noise_model=noise, decoder="pymatching"),
    },
)
for sample in stats:
    print(f"{sample.errors} logical errors in {sample.shots} shots")
```

`decoder=` is required. Under `sinter collect`, `task.decoder` is the
`custom_decoders` key (`"maestro"` above), not a decoder, and an unresolved key
raises `ValueError`. Use `decoder="vacuous"` for the undecoded logical error
rate. `noise_model` can also be a picklable `task -> NoiseModel` function
defined at module level, so one sampler can serve a sweep whose noise strength
is stored in `task.json_metadata`.

The noise here is a native physical model chosen independently of the decoder
prior, not an exact execution of the prior's noise. To run the prior's own
noise, pass the noisy circuit as the task circuit and omit `noise_model`; see
{ref}`py_sinter_contract`.

(py_sinter_contract)=
## How Stim circuits are translated

`CLEAN_CIRCUIT_API_VERSION = 2` identifies translation with explicit initial
resets, measurement and reset basis handling, classical record ordering,
optional qubit allocation order and exact Stim Pauli noise. Measurement
character *k* is classical bit *k*. Correlated (`E`) and heralded noise,
measurement inversion and record-controlled operations raise `ValueError`
instead of silently changing their meaning.

Stim's own noise is carried exactly, instruction by instruction:
`DEPOLARIZE1/2`, `X/Y/Z_ERROR` and `PAULI_CHANNEL_1/2` become
`QuantumCircuit.pauli_channel` operations, and measurement flips such as
`M(p)` become per-measurement readout errors. A noisy Stim circuit (SI1000,
for example) therefore runs with Stim's noise and needs no `NoiseModel`.
Statevector and MPS backends draw one Pauli per channel per shot; density
matrix and MPO backends apply the channels exactly.

A `NoiseModel`, if given, adds its hardware noise on top. It attaches noise
after gates only, and the basis changes that translation inserts for `RX`,
`RY`, `MX`, `MY`, `MRX` and `MRY` are marked noiseless, so the model adds gate
noise exactly where Stim has gates. Its readout error composes with any `M(p)`
flip.

Counts must contain exactly the requested positive number of shots.
Unexpected record widths and nonzero padding raise `ValueError`. A caller's
`SimulatorConfig` is copied before sampling. Direct compiled samplers accept a
`seed` (or `config.seed`); successive calls advance their random stream, and
an explicit per-call seed replays that call without consuming the stream.
`MaestroSinterSampler` rejects `seed` and `config.seed`, because `collect`
worker identities aren't available for reproducible stream partitioning.
Frames, direct trajectories and checkpoints keep each shot's record and shot
order; MPS, MPO and tensor-network histograms shuffle complete records within
each noise realization, so their rows are exchangeable rather than chronological.
