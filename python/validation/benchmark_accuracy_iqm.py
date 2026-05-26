#!/usr/bin/env python3
"""Benchmark <Z> accuracy on the last (readout) qubit — IQM backend.

Mirrors ``benchmark_accuracy.py`` but targets IQM devices.  Five pipelines:

  1. ``exact``                          — ``maestro.simple_estimate`` (noiseless).
  2. noisy maestro, 1 noise realization
  3. noisy maestro, 64 noise realizations
  4. noisy maestro, realizations=shots   (skippable)
  5. noisy aer (IQM fake backend)        — ``AerSimulator(NoiseModel.from_backend(...))``

And an optional hardware reference (``--hardware``):

  6. real IQM hardware                  — via ``IQMProvider``.  When enabled,
     hardware counts become the reference and an extra ``exact (noise-free)``
     bar shows how far the noiseless simulator drifts from the device.

Backend note
------------
The fake backend is an ``IQMFakeBackend`` subclass (e.g. ``IQMFakeApollo``,
``IQMFakeGarnet``).  These implement the standard qiskit ``BackendV2``
interface so ``AerSimulator(NoiseModel.from_backend(...))`` works unchanged.
The maestro bridge reads the same backend via its IQM-specific extraction
path (``error_profile``), so both aer and maestro see identical calibration.

When ``--hardware`` is given the live ``IQMBackend`` (from ``IQMProvider``)
drives both the maestro noise model and the hardware shots — the fake backend
is still used for the aer pipeline because aer cannot target a live IQM device.

Usage::

    # Offline only (fake backend)
    python python/validation/benchmark_accuracy_iqm.py
    python python/validation/benchmark_accuracy_iqm.py --limit 5 --shots 2048
    python python/validation/benchmark_accuracy_iqm.py --backend IQMFakeGarnet

    # With live hardware (set DEFAULT_DEVICE in code, then just pass the token)
    export IQM_TOKEN=<your-token>
    python python/validation/benchmark_accuracy_iqm.py --hardware
    python python/validation/benchmark_accuracy_iqm.py --hardware --device emerald
    python python/validation/benchmark_accuracy_iqm.py --hardware --hardware-shots 4096
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PYTHON_DIR)

# Locate the in-tree maestro extension.  Try the fixed arch path first, then
# fall back to a glob over all build sub-directories.
_BUILD_FIXED = os.path.join(_REPO, "build", "cp312-cp312-macosx_15_0_x86_64")
_BUILD_GLOB = next(
    (p for p in glob.glob(os.path.join(_REPO, "build", "cp3*"))
     if os.path.isdir(p)),
    None,
)
for _p in (_BUILD_FIXED, _BUILD_GLOB, _PYTHON_DIR):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import maestro        # noqa: E402
import qiskit         # noqa: E402
import qiskit.qasm2   # noqa: E402
import noise_bridge as br  # noqa: E402

DEFAULT_CIRCUITS_DIR = os.path.join(_HERE, "sasquatch_epoch")
DEFAULT_OUTPUT = os.path.join(_PYTHON_DIR, "_plots", "benchmark_accuracy_iqm.png")

# ── Device registry ────────────────────────────────────────────────────────
# Maps a short device name to (live URL, matching IQMFakeBackend subclass).
# Edit DEFAULT_DEVICE to switch targets without touching any flags.
_IQM_DEVICES: dict[str, tuple[str, str]] = {
    "garnet": ("https://cocos.resonance.meetiqm.com/garnet", "IQMFakeGarnet"),
}
DEFAULT_DEVICE = "garnet"
# ──────────────────────────────────────────────────────────────────────────

# IQM's native gate set (prx / cz) needs the legacy QASM2 instructions.
_QASM_KWARGS = dict(
    custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _statevector_config() -> maestro.SimulatorConfig:
    return maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.Statevector,
    )


def _load_iqm_fake(name: str):
    """Instantiate an IQMFakeBackend by class/function name.

    Search order:
    1. ``iqm.qiskit_iqm.fake_backends`` package level
    2. ``iqm.qiskit_iqm`` package level
    3. ``iqm.qiskit_iqm.fake_backends.fake_{suffix}`` submodule, where
       suffix is the device name stripped from the class name
       (``IQMFakeGarnet`` → ``fake_garnet``).
    """
    import importlib

    # Package-level search first.
    for mod_path in ("iqm.qiskit_iqm.fake_backends", "iqm.qiskit_iqm"):
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, name):
                return getattr(mod, name)()
        except ImportError:
            continue

    # Fall back to the per-device submodule (e.g. IQMFakeGarnet → fake_garnet).
    suffix = name.removeprefix("IQMFake").lower()
    submod_path = f"iqm.qiskit_iqm.fake_backends.fake_{suffix}"
    try:
        submod = importlib.import_module(submod_path)
        if hasattr(submod, name):
            return getattr(submod, name)()
    except ImportError:
        pass

    raise SystemExit(
        f"IQM fake backend '{name}' not found.  "
        "Ensure iqm-client[qiskit] is installed: "
        "uv add 'iqm-client[qiskit]'"
    )


def _connect_iqm_live(url: str, token: str | None):
    """Return a live IQMBackend from IQMProvider."""
    try:
        from iqm.qiskit_iqm import IQMProvider
    except ImportError:
        raise SystemExit(
            "--hardware requires iqm-client and qiskit-iqm.  "
            "Run: pip install iqm-client qiskit-iqm"
        )
    kwargs: dict = {"url": url}
    if token:
        kwargs["token"] = token
    provider = IQMProvider(**kwargs)
    return provider.get_backend()


def _z_from_maestro_counts(counts: dict[str, int], c_idx: int,
                            shots: int) -> float:
    """<Z> on classical bit c[c_idx] from maestro counts (qubit-0-LEFT)."""
    s = 0
    for bits, n in counts.items():
        if c_idx < len(bits):
            s += n * (1 if bits[c_idx] == "0" else -1)
        else:
            s += n
    return s / shots


def _z_from_qiskit_counts(counts: dict[str, int], c_idx: int, creg_w: int,
                           shots: int) -> float:
    """<Z> on c[c_idx] from qiskit-style counts (c[0]-RIGHT)."""
    pos = creg_w - 1 - c_idx
    s = 0
    for bits, n in counts.items():
        s += n * (1 if bits[pos] == "0" else -1)
    return s / shots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--circuits-dir", default=DEFAULT_CIRCUITS_DIR,
                        help="Directory with .qasm files (default: sasquatch_epoch).")
    parser.add_argument("--limit", type=int, default=10,
                        help="Run only the first N circuits (default 10).  "
                             "Pass 0 for all.")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Shots per simulation (default 1024).")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        choices=list(_IQM_DEVICES),
                        help=f"IQM device name (default {DEFAULT_DEVICE!r}).  "
                             "Sets both the live URL and the matching fake "
                             "backend automatically.")
    parser.add_argument("--skip-realizations-shots", action="store_true",
                        help="Skip the realizations=shots maestro pipeline.")
    parser.add_argument("--max-realizations", type=int, default=None,
                        help="Cap noise_realizations on the shots pipeline.")
    parser.add_argument("--hardware", action="store_true",
                        help="Also run on the real IQM device.  Requires "
                             "--iqm-token or IQM_TOKEN env var.  When set, "
                             "hardware counts become the reference.")
    parser.add_argument("--iqm-token", default=None,
                        help="IQM API token (overrides IQM_TOKEN env var).")
    parser.add_argument("--hardware-shots", type=int, default=4096,
                        help="Shots per circuit on hardware (default 4096).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output plot path.")
    args = parser.parse_args()

    # --- Inputs ------------------------------------------------------------
    files = sorted(glob.glob(os.path.join(args.circuits_dir, "*.qasm")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .qasm files in {args.circuits_dir}")
    print(f"Loaded {len(files)} circuits from {args.circuits_dir}", flush=True)

    cfg = _statevector_config()

    # --- Resolve device → (URL, fake class) from registry -----------------
    device_url, fake_class_name = _IQM_DEVICES[args.device]

    # --- Fake backend (transpilation + aer) --------------------------------
    fake_backend = _load_iqm_fake(fake_class_name)
    print(f"Device: {args.device}  fake={type(fake_backend).__name__} "
          f"({fake_backend.num_qubits} qubits)", flush=True)

    # --- Live backend (noise model + hardware shots, optional) -------------
    live_backend = None
    if args.hardware:
        token = args.iqm_token or os.environ.get("IQM_TOKEN")
        if not token:
            raise SystemExit(
                "--hardware requires --iqm-token or IQM_TOKEN env var."
            )
        print(f"Connecting to {args.device} at {device_url}...", flush=True)
        live_backend = _connect_iqm_live(device_url, token)
        print(f"Live backend: {live_backend.name} "
              f"({live_backend.num_qubits} qubits)", flush=True)

    # Noise model: prefer live calibration when available.
    noise_source = live_backend if live_backend is not None else fake_backend
    print(f"Building maestro noise model from "
          f"{'live' if live_backend else 'fake'} backend...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_model = br.maestro_noise_from_backend(noise_source)

    # Aer always uses the fake backend (aer cannot target a live IQM device).
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel as AerNoiseModel
    print("Building qiskit-aer NoiseModel from fake backend...", flush=True)
    aer_noise = AerNoiseModel.from_backend(fake_backend)
    aer_sim = AerSimulator(noise_model=aer_noise)

    parser_obj = maestro.QasmToCirc()

    # --- Prepare circuits --------------------------------------------------
    # Transpile against the fake backend so the coupling map is known
    # statically.  When --hardware is set the live device has the same
    # topology (user selects a matching fake via --backend), so the
    # transpiled circuits are valid on both.
    print("Transpiling and parsing all circuits...", flush=True)
    prepared = []  # (name, tr, mc, c_idx_last, phys_last, obs_str)
    for f in files:
        with open(f) as fp:
            qasm = fp.read()
        qc = qiskit.qasm2.loads(qasm, **_QASM_KWARGS)
        c_idx_last = qc.num_clbits - 1
        tr = qiskit.transpile(qc, backend=fake_backend, optimization_level=1,
                              seed_transpiler=42)
        tr_qasm = qiskit.qasm2.dumps(tr)
        mc = parser_obj.parse_and_translate(tr_qasm)

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

        obs = ["I"] * tr.num_qubits
        obs[phys_last] = "Z"
        obs_str = "".join(obs)
        prepared.append((os.path.basename(f), tr, mc, c_idx_last,
                         phys_last, obs_str))

    print(f"Prepared {len(prepared)} circuits", flush=True)

    # --- Submit hardware circuits ------------------------------------------
    hardware_counts_by_name: dict[str, dict[str, int]] = {}
    if args.hardware:
        print(f"Submitting {len(prepared)} circuits to hardware "
              f"({args.hardware_shots} shots each)...", flush=True)
        for name, tr, _, _, _, _ in prepared:
            print(f"  {name}...", end=" ", flush=True)
            job = live_backend.run(tr, shots=args.hardware_shots)
            hardware_counts_by_name[name] = job.result().get_counts()
            print("done", flush=True)
        print(f"Hardware results received for "
              f"{len(hardware_counts_by_name)} circuits.", flush=True)

    # --- Per-circuit errors ------------------------------------------------
    errors_exact_vs_ref: list[float] = []
    errors_1real: list[float] = []
    errors_64real: list[float] = []
    errors_shotsreal: list[float] = []
    errors_aer: list[float] = []

    realizations_shots = args.max_realizations or args.shots
    print(f"\nBenchmarking <Z> on {len(prepared)} circuits "
          f"({args.shots} shots)...", flush=True)
    if args.skip_realizations_shots:
        print("  (skipping realizations=shots pipeline)", flush=True)
    elif args.max_realizations is not None:
        print(f"  realizations capped at {realizations_shots}", flush=True)

    t_start = time.perf_counter()
    for i, (name, tr, mc, c_idx_last, phys_last, obs_str) in enumerate(
            prepared, 1):
        z_exact = maestro.simple_estimate(
            mc, [obs_str], config=cfg,
        )["expectation_values"][0]

        if args.hardware:
            hw_counts = hardware_counts_by_name[name]
            z_ref = _z_from_qiskit_counts(
                hw_counts, c_idx_last, tr.num_clbits, args.hardware_shots
            )
            errors_exact_vs_ref.append(abs(z_exact - z_ref))
        else:
            z_ref = z_exact

        res = maestro.noisy_execute(mc, noise_model, cfg,
                                    noise_realizations=1, shots=args.shots)
        errors_1real.append(abs(
            _z_from_maestro_counts(res["counts"], c_idx_last, args.shots) - z_ref
        ))
        del res

        res = maestro.noisy_execute(mc, noise_model, cfg,
                                    noise_realizations=64, shots=args.shots)
        errors_64real.append(abs(
            _z_from_maestro_counts(res["counts"], c_idx_last, args.shots) - z_ref
        ))
        del res

        if not args.skip_realizations_shots:
            res = maestro.noisy_execute(mc, noise_model, cfg,
                                        noise_realizations=realizations_shots,
                                        shots=args.shots)
            errors_shotsreal.append(abs(
                _z_from_maestro_counts(res["counts"], c_idx_last, args.shots) - z_ref
            ))
            del res

        aer_counts = aer_sim.run(tr, shots=args.shots).result().get_counts()
        errors_aer.append(abs(
            _z_from_qiskit_counts(aer_counts, c_idx_last, tr.num_clbits,
                                  args.shots) - z_ref
        ))
        del aer_counts

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

    # --- Summary -----------------------------------------------------------
    series: dict[str, list[float]] = {}
    if args.hardware:
        series["exact\n(noise-free)"] = errors_exact_vs_ref
    series["noisy maestro,\n1 realization"] = errors_1real
    series["noisy maestro,\n64 realizations"] = errors_64real
    if not args.skip_realizations_shots:
        cap_label = (f"realizations={realizations_shots}"
                     if args.max_realizations else "realizations=shots")
        series[f"noisy maestro,\n{cap_label}"] = errors_shotsreal
    series["noisy aer\n(IQM fake backend)"] = errors_aer

    print("\n=== Mean |<Z>_method − <Z>_ref| across circuits ===")
    for k, v in series.items():
        mean = statistics.mean(v)
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        print(f"  {k.replace(chr(10), ' '):<40}  {mean:.4f} ± {sd:.4f}")

    # --- Plot --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(series.keys())
    means = [statistics.mean(v) for v in series.values()]
    stdevs = [(statistics.stdev(v) if len(v) > 1 else 0.0)
              for v in series.values()]
    color_for = {
        "exact\n(noise-free)": "#1f77b4",
        "noisy maestro,\n1 realization": "#d62728",
        "noisy maestro,\n64 realizations": "#2ca02c",
        "noisy aer\n(IQM fake backend)": "darkkhaki",
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
    backend_label = (live_backend.name if live_backend
                     else type(fake_backend).__name__)
    if args.hardware:
        ref_label = f"<Z>_{backend_label}"
        title_suffix = (f"reference = {backend_label} hardware "
                        f"({args.hardware_shots} shots/circuit)")
    else:
        ref_label = "<Z>_exact"
        title_suffix = "reference = analytical exact"
    ax.set_ylabel(f"|<Z>_method − {ref_label}| (last qubit)", fontsize=11)
    ax.set_title(
        f"Accuracy of <Z> on readout qubit — {len(prepared)} "
        f"sasquatch_epoch circuit{'s' if len(prepared) != 1 else ''}, "
        f"{args.shots} shots, backend = {backend_label}\n{title_suffix}",
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
