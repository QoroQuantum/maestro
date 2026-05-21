#!/usr/bin/env python3
"""Benchmark <Z> accuracy on the last (readout) qubit.

Computes ``<Z>`` on the last qubit of every sasquatch_epoch circuit
through five pipelines (the same five as ``benchmark_simulation_time.py``)
and reports the **absolute error vs. the exact (analytical) value** of
the four noisy pipelines:

  1. ``exact``                   — reference, via ``maestro.simple_estimate``
                                   (analytical, zero shot noise).
  2. noisy maestro, 1 noise realization   — counts → <Z>
  3. noisy maestro, 64 noise realizations — counts → <Z>
  4. noisy maestro, shots noise realizations — counts → <Z>
  5. noisy aer (fake backend)             — counts → <Z>

For each noisy pipeline, the per-circuit error ``|<Z>_method − <Z>_exact|``
is collected; the bar chart shows the mean error with stdev error bars.

The "last qubit" means the last qubit of the original logical QASM
(``qc.num_qubits - 1``). After transpilation this can be mapped to any
physical qubit on the backend, but the **classical bit it measures into
remains c[n_logical - 1]**. We use that classical-bit index for the
counts-based pipelines, and the transpilation-resolved physical qubit
for the analytical observable.

Conventions (verified against maestro and qiskit-aer outputs):

* Maestro counts are 0-LEFT and indexed by **classical bit position**,
  so bit at left-position ``c_idx`` is the value measured into ``c[c_idx]``.
* Qiskit/aer counts are c[0]-RIGHT (standard qiskit), so ``c[c_idx]``
  sits at left-position ``creg_width − 1 − c_idx``.
* Maestro's ``simple_estimate`` observable string is 0-LEFT and indexed
  by **physical qubit**, so Pauli at left-position ``q`` acts on
  qubit ``q`` of the transpiled circuit.

Usage::

    python python/validation/benchmark_accuracy.py
    python python/validation/benchmark_accuracy.py --limit 20
    python python/validation/benchmark_accuracy.py --shots 4096
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import statistics
import sys
import time
import warnings

# Allow running directly from the repo without installing.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PYTHON_DIR)
_BUILD = os.path.join(_REPO, "build", "cp312-cp312-macosx_15_0_x86_64")
for _p in (_BUILD, _PYTHON_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import maestro  # noqa: E402
import qiskit  # noqa: E402
import qiskit.qasm2  # noqa: E402

import ibm_noise_bridge as br  # noqa: E402

DEFAULT_CIRCUITS_DIR = os.path.join(_HERE, "sasquatch_epoch")
DEFAULT_BACKEND = "FakeAlmadenV2"
DEFAULT_OUTPUT = os.path.join(_PYTHON_DIR, "_plots",
                              "benchmark_accuracy.png")


def _statevector_config() -> maestro.SimulatorConfig:
    return maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.Statevector,
    )


def _load_fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider
    if not hasattr(fake_provider, name):
        candidates = [n for n in dir(fake_provider) if n.startswith("Fake")]
        raise SystemExit(
            f"Backend '{name}' not found. Choices include: {candidates[:10]}..."
        )
    return getattr(fake_provider, name)()


def _live_backend_name_for(fake_name: str) -> str:
    """Map a fake-backend class name to its corresponding live IBM
    backend name. ``FakeAlmadenV2`` → ``ibm_almaden``, ``FakeFez`` →
    ``ibm_fez``, etc.
    """
    name = fake_name
    if name.startswith("Fake"):
        name = name[4:]
    for suffix in ("V2", "V1"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"ibm_{name.lower()}"


def _fake_class_name_for_live(live_name: str) -> str:
    """Map a live IBM backend name to its corresponding fake-backend
    class. ``ibm_fez`` → ``FakeFez``, ``ibm_almaden`` → ``FakeAlmadenV2``
    (prefers V2 when both exist). Returns the FIRST class that exists in
    ``qiskit_ibm_runtime.fake_provider``.
    """
    from qiskit_ibm_runtime import fake_provider
    stem = live_name.replace("ibm_", "")
    capped = stem.capitalize()
    for candidate in (f"Fake{capped}V2", f"Fake{capped}"):
        if hasattr(fake_provider, candidate):
            return candidate
    raise SystemExit(
        f"No fake backend found matching live backend '{live_name}'. "
        f"Tried Fake{capped}V2 and Fake{capped}."
    )


def _z_from_maestro_counts(counts: dict[str, int], c_idx: int,
                           shots: int) -> float:
    """<Z> on classical bit c[c_idx] from maestro counts.

    Maestro emits qubit-0-LEFT bitstrings indexed by **classical bit
    position**, so bit at left-position ``c_idx`` is c[c_idx]'s value.
    """
    s = 0
    for bits, n in counts.items():
        if c_idx < len(bits):
            s += n * (1 if bits[c_idx] == "0" else -1)
        else:
            s += n  # untouched bit stays |0> → Z eigenvalue +1
    return s / shots


def _z_from_aer_counts(counts: dict[str, int], c_idx: int, creg_w: int,
                       shots: int) -> float:
    """<Z> on classical bit c[c_idx] from qiskit/aer counts.

    Qiskit's convention is c[0]-RIGHT, so c[c_idx] is at left-position
    ``creg_w − 1 − c_idx``.
    """
    pos = creg_w - 1 - c_idx
    s = 0
    for bits, n in counts.items():
        s += n * (1 if bits[pos] == "0" else -1)
    return s / shots


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--circuits-dir", default=DEFAULT_CIRCUITS_DIR,
                        help="Directory with .qasm files to benchmark.")
    parser.add_argument("--limit", type=int, default=10,
                        help="Run only the first N circuits (default 10). "
                             "Each 17q circuit takes ~3 min total across "
                             "the four noisy pipelines at the default shot "
                             "count, dominated by qiskit-aer. Pass --limit "
                             "0 to use all circuits in the directory.")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Shots per simulation (default 1024).")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="Fake backend class name from "
                             "qiskit_ibm_runtime.fake_provider.")
    parser.add_argument("--skip-realizations-shots", action="store_true",
                        help="Skip the noisy-maestro 'realizations=shots' "
                             "pipeline. Useful as a workaround when "
                             "maestro's C++ noise injector runs into "
                             "heap-corruption issues at large realization "
                             "counts (the per-circuit allocation pressure "
                             "is highest in this variant).")
    parser.add_argument("--max-realizations", type=int, default=None,
                        help="Cap noise_realizations on the "
                             "'realizations=shots' pipeline. Default: no "
                             "cap (= --shots). Set this to a lower value "
                             "(e.g. 256) if you hit heap-corruption errors "
                             "at high shot counts.")
    parser.add_argument("--hardware", action="store_true",
                        help="Also run on the corresponding live IBM "
                             "backend (FakeAlmadenV2 → ibm_almaden, "
                             "FakeFez → ibm_fez, etc.). Requires "
                             "IBM_QUANTUM_TOKEN env var. Submits all "
                             "circuits as a single batch via SamplerV2; "
                             "the job will queue and may take minutes to "
                             "hours. When set, hardware becomes the "
                             "reference: errors are measured as "
                             "|<Z>_method - <Z>_hardware|, and an exact "
                             "(noiseless) bar is added showing how far "
                             "the noise-free simulation drifts from the "
                             "real hardware.")
    parser.add_argument("--hardware-backend", default=None,
                        help="Override the auto-derived live backend name "
                             "(e.g. --hardware-backend ibm_fez when the "
                             "Fake-named device is retired).")
    parser.add_argument("--hardware-shots", type=int, default=8192,
                        help="Shots per circuit on hardware (default "
                             "8192). Higher = tighter <Z>_hardware "
                             "estimate, at the cost of more queue time. "
                             "Independent from --shots, which only "
                             "controls the simulator pipelines.")
    parser.add_argument("--channel", default="ibm_quantum_platform",
                        choices=["ibm_quantum_platform", "ibm_cloud"],
                        help="QiskitRuntimeService channel for the "
                             "hardware path (default ibm_quantum_platform).")
    parser.add_argument("--instance", default=None,
                        help="Optional CRN/instance for the runtime "
                             "service.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output plot path.")
    args = parser.parse_args()

    # --- Inputs -----------------------------------------------------------
    files = sorted(glob.glob(os.path.join(args.circuits_dir, "*.qasm")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .qasm files in {args.circuits_dir}")
    print(f"Loaded {len(files)} circuits from {args.circuits_dir}",
          flush=True)

    cfg = _statevector_config()

    # When --hardware is set we authenticate up front and pair the live
    # backend with its matching fake backend (e.g., ibm_fez ↔ FakeFez).
    # The fake drives the simulator pipelines (transpile, noise model,
    # aer); the live drives the hardware reference. The fake's coupling
    # map and basis gates match the live device by construction, so a
    # single transpilation works for both.
    hardware_backend_name: str | None = None
    sampler = None
    live_backend = None
    if args.hardware:
        token = os.environ.get("IBM_QUANTUM_TOKEN")
        if not token:
            raise SystemExit(
                "--hardware requires IBM_QUANTUM_TOKEN to be set in the "
                "environment."
            )
        hardware_backend_name = (args.hardware_backend
                                 or _live_backend_name_for(args.backend))
        # Match the fake to the live device, overriding --backend.
        derived_fake = _fake_class_name_for_live(hardware_backend_name)
        if derived_fake != args.backend:
            print(f"--hardware: pairing live '{hardware_backend_name}' "
                  f"with fake '{derived_fake}' for the simulator "
                  f"pipelines (overrides --backend={args.backend}).",
                  flush=True)
            args.backend = derived_fake
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        print(f"Authenticating with IBM Quantum (channel={args.channel})...",
              flush=True)
        svc_kwargs: dict = {"channel": args.channel, "token": token}
        if args.instance:
            svc_kwargs["instance"] = args.instance
        service = QiskitRuntimeService(**svc_kwargs)
        live_backend = service.backend(hardware_backend_name)
        print(f"Live backend: {live_backend.name} "
              f"({live_backend.num_qubits} qubits)", flush=True)
        sampler = Sampler(mode=live_backend)

    backend = _load_fake_backend(args.backend)
    print(f"Simulator backend (fake): {args.backend} "
          f"({backend.num_qubits} qubits)", flush=True)

    # Aer simulator with the noise model from the fake backend.
    # AerSimulator works for both legacy fake backends (.run()) and
    # by passing noise_model explicitly, so we use it uniformly.
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    print("Building qiskit-aer NoiseModel from fake backend (one-time)...",
          flush=True)
    aer_noise = NoiseModel.from_backend(backend)
    aer_sim = AerSimulator(noise_model=aer_noise)

    print("Building maestro noise model from fake backend (one-time)...",
          flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_model = br.maestro_noise_from_backend(backend)
    parser_obj = maestro.QasmToCirc()

    qasm_kwargs = dict(
        custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
    )

    # --- Pre-prepare every circuit, plus its readout-qubit mapping --------
    print("Transpiling and parsing all circuits (one-time)...", flush=True)
    prepared = []  # (name, tr, mc, c_idx_last, phys_last, obs_str)
    for f in files:
        with open(f) as fp:
            qasm = fp.read()
        qc = qiskit.qasm2.loads(qasm, **qasm_kwargs)
        c_idx_last = qc.num_clbits - 1   # "last classical bit" = c[n-1]
        tr = qiskit.transpile(qc, backend=backend, optimization_level=1,
                              seed_transpiler=42)
        tr_qasm = qiskit.qasm2.dumps(tr)
        mc = parser_obj.parse_and_translate(tr_qasm)

        # Find the physical qubit that's measured into c[c_idx_last]
        phys_last = None
        for inst in tr.data:
            if inst.operation.name == "measure":
                if tr.find_bit(inst.clbits[0]).index == c_idx_last:
                    phys_last = tr.find_bit(inst.qubits[0]).index
                    break
        if phys_last is None:
            print(f"  WARN: no measure for c[{c_idx_last}] in {f}, skipping",
                  flush=True)
            continue

        # Observable for simple_estimate: Z at left-position phys_last
        obs = ["I"] * tr.num_qubits
        obs[phys_last] = "Z"
        obs_str = "".join(obs)

        prepared.append((os.path.basename(f), tr, mc, c_idx_last,
                         phys_last, obs_str))

    print(f"Prepared {len(prepared)} circuits", flush=True)

    # --- Optional: submit every circuit to the live hardware in one batch -
    hardware_counts_by_name: dict[str, dict[str, int]] = {}
    if args.hardware:
        # The circuits were transpiled for the matched fake backend; the
        # live backend has the same coupling map and basis gates, so the
        # transpiled circuits are valid. Submit them as a single batch
        # using --hardware-shots (independent from --shots, so the
        # hardware reference can have tight statistics).
        transpiled_list = [tr for (_, tr, _, _, _, _) in prepared]
        print(f"Submitting {len(transpiled_list)} circuits to hardware "
              f"as a single batch ({args.hardware_shots} shots each)...",
              flush=True)
        job = sampler.run(transpiled_list, shots=args.hardware_shots)
        print(f"Job ID: {job.job_id()}  — monitor at "
              f"https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)
        print("Waiting for queue + execution (this can take a while)...",
              flush=True)
        result = job.result()
        for (name, _, _, _, _, _), pub in zip(prepared, result):
            # The classical register is named "c" in our QASM source.
            creg_name = "c"
            if not hasattr(pub.data, creg_name):
                creg_name = next(iter(pub.data.keys()))
            hardware_counts_by_name[name] = getattr(
                pub.data, creg_name
            ).get_counts()
        print(f"Hardware results received for {len(hardware_counts_by_name)} "
              f"circuits.", flush=True)

    # --- Per-circuit <Z>, then absolute error vs. the chosen reference ----
    # Reference = hardware when --hardware is set; otherwise exact
    # (analytical, noiseless) <Z>. When hardware is the reference, an
    # additional "exact" bar appears in the chart showing how far the
    # noiseless simulator drifts from the real device.
    errors_exact_vs_ref: list[float] = []  # only populated if hardware is on
    errors_1real: list[float] = []
    errors_64real: list[float] = []
    errors_shotsreal: list[float] = []
    errors_aer: list[float] = []

    realizations_shots = args.max_realizations or args.shots
    print(f"Benchmarking <Z> accuracy on {len(prepared)} circuits "
          f"({args.shots} shots)...", flush=True)
    if args.skip_realizations_shots:
        print("  (skipping realizations=shots pipeline per --skip-realizations-shots)",
              flush=True)
    elif args.max_realizations is not None:
        print(f"  realizations on the 'realizations=shots' pipeline capped "
              f"at {realizations_shots}", flush=True)
    t_start = time.perf_counter()
    for i, (name, tr, mc, c_idx_last, phys_last, obs_str) in enumerate(
            prepared, 1):
        # Analytical noise-free <Z> (always computed)
        z_exact = maestro.simple_estimate(
            mc, [obs_str], config=cfg,
        )["expectation_values"][0]

        # Pick the reference for the error metric
        if args.hardware:
            hw_counts = hardware_counts_by_name[name]
            z_ref = _z_from_aer_counts(hw_counts, c_idx_last, tr.num_clbits,
                                       args.hardware_shots)
            errors_exact_vs_ref.append(abs(z_exact - z_ref))
        else:
            z_ref = z_exact

        # 1 noise realization
        res = maestro.noisy_execute(mc, noise_model, cfg,
                                    noise_realizations=1, shots=args.shots)
        z = _z_from_maestro_counts(res["counts"], c_idx_last, args.shots)
        errors_1real.append(abs(z - z_ref))
        del res

        # 64 noise realizations
        res = maestro.noisy_execute(mc, noise_model, cfg,
                                    noise_realizations=64, shots=args.shots)
        z = _z_from_maestro_counts(res["counts"], c_idx_last, args.shots)
        errors_64real.append(abs(z - z_ref))
        del res

        # shots noise realizations (or skipped via flag — the most
        # allocation-heavy maestro pipeline)
        if not args.skip_realizations_shots:
            res = maestro.noisy_execute(mc, noise_model, cfg,
                                        noise_realizations=realizations_shots,
                                        shots=args.shots)
            z = _z_from_maestro_counts(res["counts"], c_idx_last, args.shots)
            errors_shotsreal.append(abs(z - z_ref))
            del res

        # aer (slow — usually dominates wall-clock)
        aer_counts = aer_sim.run(tr, shots=args.shots).result().get_counts()
        z = _z_from_aer_counts(aer_counts, c_idx_last, tr.num_clbits,
                               args.shots)
        errors_aer.append(abs(z - z_ref))
        del aer_counts

        # Encourage prompt cleanup of Python-side references between
        # circuits. Won't fix C++ memory bugs, but reduces Python
        # accumulation that can hide / interact with them.
        gc.collect()

        ref_tag = "hw" if args.hardware else "exact"
        line = (f"  {i}/{len(prepared)} {name}  "
                f"{ref_tag}_ref={z_ref:+.3f}  "
                f"err1={errors_1real[-1]:.3f}  "
                f"err64={errors_64real[-1]:.3f}  ")
        if not args.skip_realizations_shots:
            line += f"errN={errors_shotsreal[-1]:.3f}  "
        line += f"errAer={errors_aer[-1]:.3f}"
        if args.hardware:
            line += f"  errExact={errors_exact_vs_ref[-1]:.3f}"
        print(line, flush=True)
    elapsed = time.perf_counter() - t_start
    print(f"Done in {elapsed:.1f}s", flush=True)

    # --- Summary ---------------------------------------------------------
    series: dict[str, list[float]] = {}
    if args.hardware:
        # Hardware is the reference; show the noise-free simulator as
        # one of the bars so we see how far ideal drifts from hardware.
        series["exact\n(noise-free)"] = errors_exact_vs_ref
    series["noisy maestro,\n1 realization"] = errors_1real
    series["noisy maestro,\n64 realizations"] = errors_64real
    if not args.skip_realizations_shots:
        cap_label = (f"realizations={realizations_shots}"
                     if args.max_realizations
                     else "realizations=shots")
        series[f"noisy maestro,\n{cap_label}"] = errors_shotsreal
    series["noisy aer\n(fake backend)"] = errors_aer

    print("\n=== Mean |<Z>_method − <Z>_exact| across circuits ===")
    for k, v in series.items():
        mean = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"  {k.replace(chr(10), ' '):<40}  {mean:.4f} ± {sd:.4f}")

    # --- Plot -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(series.keys())
    means = [statistics.mean(v) for v in series.values()]
    stdevs = [(statistics.stdev(v) if len(v) > 1 else 0.0)
              for v in series.values()]
    # Color slots — match benchmark_simulation_time.py exactly:
    # exact=blue, 1-realization=red, 64-realization=green,
    # shots-realization=purple, aer=darkkhaki.
    color_for = {
        "exact\n(noise-free)": "#1f77b4",
        "noisy maestro,\n1 realization": "#d62728",
        "noisy maestro,\n64 realizations": "#2ca02c",
        "noisy aer\n(fake backend)": "darkkhaki",
    }
    colors = [color_for.get(lbl, "purple") for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    x = list(range(len(labels)))
    ax.bar(x, means, color=colors, alpha=1.0,
           yerr=stdevs, capsize=8, ecolor="#333333",
           edgecolor="black", linewidth=0.5)

    for xi, m, s in zip(x, means, stdevs):
        ax.annotate(f"{m:.4f}\n± {s:.4f}",
                    xy=(xi, m + s), xytext=(0, 6),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    if args.hardware:
        ref_label = f"<Z>_{hardware_backend_name}"
        title_suffix = (f"reference = {hardware_backend_name} "
                        f"({args.hardware_shots} shots)")
    else:
        ref_label = "<Z>_exact"
        title_suffix = "reference = analytical exact"
    ax.set_ylabel(f"|<Z>_method − {ref_label}| (last qubit)", fontsize=11)
    ax.set_title(
        f"Accuracy of <Z> on readout qubit — {len(prepared)} "
        f"sasquatch_epoch circuit{'s' if len(prepared) != 1 else ''} (17q), "
        f"{args.shots} shots, simulator backend = {args.backend}\n"
        f"{title_suffix}",
        fontsize=12,
    )
    ax.set_ylim(0, max(m + s for m, s in zip(means, stdevs)) * 1.30)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
