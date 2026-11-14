---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
mystnb:
  merge_streams: true
---

(py_deltakit)=
# Use Maestro from Deltakit

Run [Deltakit](https://deltakit.riverlane.com/) error-correction circuits on
Maestro. Pauli noise samples as fast as with Stim's own sampler, and Maestro also
simulates coherent over-rotations and T1 decay, which no Clifford simulator
can, while the decoder gets a detector error model that matches the noise
being run.

```bash
pip install deltakit-maestro pymatching
```

```{code-cell} python
# Import Deltakit before PyMatching; see Limitations below.
from deltakit.circuit.gates import PauliBasis
from deltakit.explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit.explorer.qpu import QPU, SI1000Noise

from deltakit_maestro import MaestroBackend

code = RotatedPlanarCode(width=3, height=3)
circuit = css_code_memory_circuit(code, num_rounds=3, logical_basis=PauliBasis.Z)
noisy = QPU(circuit.qubits, noise_model=SI1000Noise(p=0.01)).compile_and_add_noise_to_circuit(circuit)

backend = MaestroBackend()
sampler = backend.compile_sampler(noisy, seed=1)

dets, obs = sampler.sample_detection_events(100_000)
print("detection events:", dets.shape, "observable flips:", obs.shape)
print(f"detection event rate: {dets.mean():.4f}")

stats = sampler.sample(100_000)  # decoded with PyMatching
print(f"logical error rate: {stats.errors / stats.shots:.4f}")
```

The plugin is a separate package,
[deltakit-maestro](https://github.com/QoroQuantum/deltakit-maestro), which
installs Deltakit and `qoro-maestro[sinter]` for you. It needs Python 3.10 or
later, within the versions Deltakit supports. `MaestroBackend` accepts a
Deltakit `Circuit` or a `stim.Circuit`, and is built on Maestro's Sinter
samplers ({doc}`/user-guide/qec-sinter`):

| Method | Returns |
|---|---|
| `compile_sampler(circuit)` | A `MaestroCompiledSampler`: detection events, raw measurements, or decoded statistics |
| `simulate(circuit, shots)` | Measurements, detection events and observable flips in one call |
| `as_sinter_task(circuit)` and `sinter_decoders()` | A `sinter.Task` and the `custom_decoders` to run it with `sinter.collect` |
| `twirled_circuit(circuit)` | The Stim circuit whose detector error model the decoder uses |

(py_deltakit_noise)=
## How the noise runs

Deltakit converts the circuit to Stim with its noise. `noise_type` decides what
Maestro does with that noise, and the simulation type follows from it:

| `noise_type` | Gate noise | Simulation type |
|---|---|---|
| `"pauli"` (default) | The circuit's own Stim noise, run exactly | `Stabilizer`, sampled on Pauli frames |
| `"coherent"` | Gate depolarizing replaced by a coherent over-rotation | `MatrixProductState` |
| `"full"` | The circuit's Stim noise plus the coherent over-rotation | `MatrixProductState` |
| `"none"` | None: the circuit runs noiseless | `Stabilizer` |

The over-rotation with infidelity *p* turns each qubit by ε = 2·arcsin√*p*
about a body diagonal after every gate. Its Pauli twirl is `DEPOLARIZE1(p)`.
In every mode except `"none"`, the circuit's idle, reset and readout noise
stay. `config_for` returns the `SimulatorConfig` a run uses:

```{code-cell} python
for noise_type in ["pauli", "coherent", "full", "none"]:
    print(f"{noise_type:>8}: {backend.config_for(noise_type).simulation_type}")
```

To run everything with one simulation type, pass `simulation_type=` to
`MaestroBackend`. The type that runs is pinned (`fixed_backend=True`), so an
MPS run of a Clifford circuit really runs on the MPS; without that, Maestro
would sample it on Pauli frames. Other `SimulatorConfig` fields go to
`MaestroBackend` as keywords too, such as
`simulator_type=maestro.SimulatorType.Gpu`, which needs
{doc}`GPU access <gpu-access>`.

(py_deltakit_keywords)=
### Override or add noise

Keywords on `simulate`, `compile_sampler`, `as_sinter_task` and
`twirled_circuit` override the rates read from the circuit, or add noise to it:

| Keyword | Noise |
|---|---|
| `p_1q`, `p_2q` | Depolarizing after single- and two-qubit gates, replacing the circuit's gate noise |
| `readout_error` | Measurement flip probability, replacing the circuit's `M(p)` |
| `t1` | T1 decay with probability γ after every gate |
| `coherent` | Infidelity of the coherent over-rotation in `"coherent"` and `"full"`; the circuit's `p_2q` by default |

T1 decay is not a Pauli channel, so a run with `t1=` and no coherent noise
runs on the stabilizer simulator shot by shot instead of on Pauli frames:

```{code-cell} python
t1_run = backend.simulate(noisy, shots=2000, t1=1e-3, seed=1)
print(f"detection event rate with T1 decay: {t1_run.detection_events.mean():.4f}")
```

An unknown keyword raises `TypeError`, and a keyword the `noise_type` doesn't
use, such as `coherent=` with `"pauli"`, raises `ValueError`, before any shots
run.

(py_deltakit_coherent)=
## Coherent noise against its Pauli twirl

`twirled_circuit` returns the Stim circuit with the Pauli twirl of the noise
Maestro runs. Its detector error model is the one the decoder gets, so the
decoder always knows the strength of the noise. Sampling it gives a Pauli
baseline, and the gap between a coherent run and its twirl is the effect of
coherence alone:

```{code-cell} python
coherent = backend.simulate(noisy, shots=300, noise_type="coherent", seed=1)
twirl = backend.simulate(backend.twirled_circuit(noisy, noise_type="coherent"), shots=100_000, seed=1)

print(f"coherent, on an MPS:          {coherent.detection_events.mean():.4f}")
print(f"its Pauli twirl, on frames:   {twirl.detection_events.mean():.4f}")
```

Compare a coherent run with its own twirl, not with the original SI1000 run.
`"coherent"` sets one strength per qubit, applied after every gate, so at the
default `coherent = p_2q` it applies more gate noise than SI1000(*p*): an
infidelity of *p* after each single-qubit gate, where SI1000 has *p*/10, and
*p* on each qubit of a two-qubit gate.

An MPS shot costs far more than a Pauli-frame shot: on an Apple M4 Pro, about
57 ms a shot per core on this code at the default bond dimension χ = 32. One sampler spreads
its shots over every core.

(py_deltakit_sinter)=
## Collect statistics with Sinter

`as_sinter_task` packs the circuit, its decoder prior and the noise settings
into a `sinter.Task`, and `sinter_decoders` gives `sinter.collect` the sampler
that runs it:

```{code-cell} python
import sinter

tasks = []
for p in [0.002, 0.005, 0.01]:
    noisy_p = QPU(circuit.qubits, noise_model=SI1000Noise(p=p)).compile_and_add_noise_to_circuit(circuit)
    tasks.append(backend.as_sinter_task(noisy_p, json_metadata={"p": p}))

collected = sinter.collect(
    num_workers=2,
    tasks=tasks,
    custom_decoders=backend.sinter_decoders(decoder="pymatching"),
    max_shots=20_000,
    max_errors=500,
)
for result in sorted(collected, key=lambda r: r.json_metadata["p"]):
    rate = result.errors / result.shots
    print(f"p={result.json_metadata['p']}: logical error rate {rate:.4f} over {result.shots} shots")
```

Pass `noise_type=` and the noise keywords to `as_sinter_task` to collect
coherent or T1 statistics the same way. `decoder="vacuous"` gives undecoded
statistics. Codes whose detector error model can't be decomposed into
graphlike errors, such as Deltakit's bivariate bicycle codes, need a decoder
such as BP-OSD instead of PyMatching.

(py_deltakit_chi)=
## Check the bond dimension

`MaestroBackend(max_bond_dimension=...)` sets χ for coherent runs. Too small a
χ truncates the state, and truncation shows up as extra detection events, in
the same direction as the coherent effect being studied. Check χ in two steps.

First, run noiselessly on a pinned MPS. Every detector must stay silent, so
any detection event is truncation error:

```{code-cell} python
import maestro

for chi in [4, 8]:
    mps = MaestroBackend(simulation_type=maestro.SimulationType.MatrixProductState, max_bond_dimension=chi)
    check = mps.simulate(noisy, shots=50, noise_type="none", seed=1)
    print(f"χ={chi}: {check.detection_events.any(axis=1).mean():.0%} of shots fire a detector")
```

Here χ = 4 truncates and χ = 8 is exact. Second, coherent noise needs a larger χ than the noiseless check. Double χ
until the coherent result stops changing within its error bars. On this code
the detection event rate is 0.294 at χ = 8, 0.272 at χ = 16, and 0.263 at
χ = 32 and 64.

On an MPS, only the rotated distance-3 code is practical: the unrotated d = 3
and rotated d = 5 codes still truncate at χ = 32 to 64, at about a second a
shot. Pauli and T1 runs, on the stabilizer simulator, have no such limit.

(py_deltakit_limits)=
## Limitations

- **Import Deltakit before PyMatching.** With PyMatching 2.4 imported first,
  Deltakit's circuit builders fail with
  `RuntimeError: Caught an unknown exception!`.
- **Decode against the backend's detector error model**,
  `sampler.task.detector_error_model` or
  `as_sinter_task(...).detector_error_model`. Deltakit's `PyMatchingDecoder`
  builds its own Stim circuit, whose detectors are in a different order.
- **`sinter.collect` can't be seeded.** Pass `seed=` to `simulate` or
  `compile_sampler` instead. Deltakit orders a circuit differently in each
  Python process, so to reproduce a run in another process, also fix
  `PYTHONHASHSEED`.
- `MPP`, correlated (`E`) and heralded noise, classically controlled gates and
  inverted measurement targets are rejected, as in {ref}`py_sinter_contract`.

The [deltakit-maestro README](https://github.com/QoroQuantum/deltakit-maestro)
has the full API and examples for every Deltakit code family, including the
[[72,12,6]] bivariate bicycle code decoded with BP-OSD.
