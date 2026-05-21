#!/usr/bin/env python3
"""Benchmark simulation runtime on the sasquatch_epoch circuit set.

Three pipelines per circuit:

  1. ``exact``           — ``maestro.simple_execute`` (statevector, no noise)
  2. ``noisy maestro``   — ``maestro.noisy_execute`` with the noise model
                           the bridge builds from the fake backend.
  3. ``noisy aer``       — ``fake_backend.run(circuit, ...)``, which under
                           the hood uses ``AerSimulator(noise_model=
                           NoiseModel.from_backend(fake_backend))``.

All three pipelines time *only the simulator call*. Setup (transpilation,
noise-model construction, QASM parsing) is done once per circuit before
the timers start, so we measure simulation speed, not pipeline overhead.

Renders a bar chart with the **noisy maestro** time as the 1.0× baseline
and the other two pipelines as multipliers; both the multipliers and the
absolute wall-clock times (in ms) are annotated on the bars.

Usage::

    python python/validation/benchmark_simulation_time.py
    python python/validation/benchmark_simulation_time.py --limit 50
    python python/validation/benchmark_simulation_time.py --backend FakeKolkataV2
    python python/validation/benchmark_simulation_time.py --shots 2048
"""

from __future__ import annotations

import argparse
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

import noise_bridge as br  # noqa: E402

DEFAULT_CIRCUITS_DIR = os.path.join(_HERE, "sasquatch_epoch")
DEFAULT_BACKEND = "FakeAlmadenV2"  # 20 qubits — the tightest legacy V2
                                   # fake backend that fits the 17-qubit
                                   # ansätze. No rzz/crosstalk data, so the
                                   # bridge's noise injection stays inside
                                   # the 20q register (2^20 amplitudes =
                                   # ~16 MB statevector, fast).
DEFAULT_OUTPUT = os.path.join(_PYTHON_DIR, "_plots",
                              "benchmark_simulation_time.png")


def _statevector_config() -> maestro.SimulatorConfig:
    """Force QCSim + Statevector to match the qiskit-aer baseline."""
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--circuits-dir", default=DEFAULT_CIRCUITS_DIR,
                        help="Directory with .qasm files to benchmark.")
    parser.add_argument("--limit", type=int, default=1,
                        help="Run only the first N circuits (default 1). "
                             "Each 17q circuit takes ~1 min total across "
                             "the three pipelines at the default shot "
                             "count, dominated by qiskit-aer "
                             "(~50s/circuit). Pass --limit 0 to use all "
                             "circuits in the directory.")
    parser.add_argument("--shots", type=int, default=1024,
                        help="Shots per simulation (default 1024).")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="Fake backend class name from "
                             "qiskit_ibm_runtime.fake_provider.")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup runs per pipeline before timing "
                             "(default 2). Lets JIT/caches settle.")
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

    backend = _load_fake_backend(args.backend)
    print(f"Backend: {args.backend} ({backend.num_qubits} qubits)",
          flush=True)

    cfg = _statevector_config()

    # --- One-time setup outside the timed regions -------------------------
    print("Building noise model from backend (one-time)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        noise_model = br.maestro_noise_from_backend(backend)
    parser_obj = maestro.QasmToCirc()

    # Pre-load, pre-transpile, and pre-parse all circuits so the timed
    # regions only contain the simulator call.
    print("Transpiling and parsing all circuits (one-time)...", flush=True)
    prepared: list[tuple[str, qiskit.QuantumCircuit, str, "maestro.circuits.QuantumCircuit"]] = []
    # The sasquatch_epoch circuits use gates (swap, cry, …) that qiskit's
    # strict qasm2 parser treats as undefined. The LEGACY_CUSTOM_INSTRUCTIONS
    # set is qiskit's shim for the full qelib1 family.
    qasm_kwargs = dict(
        custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
    )
    for f in files:
        with open(f) as fp:
            qasm = fp.read()
        qc = qiskit.qasm2.loads(qasm, **qasm_kwargs)
        tr = qiskit.transpile(qc, backend=backend, optimization_level=1,
                              seed_transpiler=42)
        tr_qasm = qiskit.qasm2.dumps(tr)
        mc = parser_obj.parse_and_translate(tr_qasm)
        prepared.append((os.path.basename(f), tr, tr_qasm, mc))

    # --- Warmup -----------------------------------------------------------
    if args.warmup > 0 and prepared:
        print(f"Warming up ({args.warmup} runs per pipeline)...", flush=True)
        warmup_tr, warmup_tr_qasm, warmup_mc = prepared[0][1], prepared[0][2], prepared[0][3]
        for _ in range(args.warmup):
            maestro.simple_execute(warmup_mc, config=cfg, shots=args.shots)
            maestro.noisy_execute(warmup_mc, noise_model, cfg, shots=args.shots)
            backend.run(warmup_tr, shots=args.shots).result()

    # --- Timed runs -------------------------------------------------------
    times_exact: list[float] = []
    times_bridge_once: list[float] = []
    times_bridge_64: list[float] = []
    times_bridge_shots: list[float] = []
    times_aer: list[float] = []

    print(f"Benchmarking {len(prepared)} circuits...", flush=True)
    for i, (name, tr, tr_qasm, mc) in enumerate(prepared, 1):
        t0 = time.perf_counter()
        maestro.simple_execute(mc, config=cfg, shots=args.shots)
        times_exact.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        maestro.noisy_execute(mc, noise_model, cfg, noise_realizations=1, shots=args.shots)
        times_bridge_once.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        maestro.noisy_execute(mc, noise_model, cfg, noise_realizations=64, shots=args.shots)
        times_bridge_64.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        maestro.noisy_execute(mc, noise_model, cfg, noise_realizations=args.shots, shots=args.shots)
        times_bridge_shots.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        backend.run(tr, shots=args.shots).result()
        times_aer.append(time.perf_counter() - t0)

        if i % max(1, len(prepared) // 20) == 0 or i == len(prepared):
            print(f"  {i}/{len(prepared)} circuits "
                  f"(exact={times_exact[-1]*1e3:.1f}ms, "
                  f"bridge, one realization={times_bridge_once[-1]*1e3:.1f}ms, "
                  f"bridge, 64 realizations={times_bridge_64[-1]*1e3:.1f}ms, "
                  f"bridge, {args.shots} realizations={times_bridge_shots[-1]*1e3:.1f}ms, "
                  f"aer={times_aer[-1]*1e3:.1f}ms)",
                  flush=True)

    # --- Summary statistics -----------------------------------------------
    series = {
        "exact": times_exact,
        "noisy maestro, 1 noise realization": times_bridge_once,
        "noisy maestro, 64 noise realizations": times_bridge_64,
        "noisy maestro, noise realizations=shots": times_bridge_shots,
        "noisy aer (fake backend)": times_aer,
    }
    stats: dict[str, dict[str, float]] = {}
    for k, v in series.items():
        stats[k] = {
            "mean_ms": statistics.mean(v) * 1e3,
            "median_ms": statistics.median(v) * 1e3,
            "stdev_ms": (statistics.stdev(v) * 1e3) if len(v) > 1 else 0.0,
            "min_ms": min(v) * 1e3,
            "max_ms": max(v) * 1e3,
            "total_s": sum(v),
        }

    print(f"\n=== Per-circuit runtime ({len(prepared)} circuits, "
          f"{args.shots} shots, backend={args.backend}) ===")
    print(f"{'pipeline':<28}  {'mean (ms)':>10}  {'median (ms)':>12}  "
          f"{'stdev (ms)':>11}  {'total (s)':>10}")
    for k in series:
        s = stats[k]
        print(f"{k:<28}  {s['mean_ms']:>10.2f}  {s['median_ms']:>12.2f}  "
              f"{s['stdev_ms']:>11.2f}  {s['total_s']:>10.2f}")

    base_mean = stats["noisy maestro, 1 noise realization"]["mean_ms"]
    print(f"\nNormalized to noisy maestro = 1.00× ({base_mean:.2f} ms):")
    for k in series:
        mult = stats[k]["mean_ms"] / base_mean
        print(f"  {k:<28}  {mult:>6.2f}×")

    # --- Bar chart --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(series.keys())
    means_ms = [stats[k]["mean_ms"] for k in labels]
    medians_ms = [stats[k]["median_ms"] for k in labels]
    stdev_ms = [stats[k]["stdev_ms"] for k in labels]
    multipliers = [m / base_mean for m in means_ms]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "purple", "darkkhaki"]

    def _fmt_time(ms: float) -> str:
        if ms >= 1000:
            return f"{ms / 1000:.2f} s"
        if ms >= 10:
            return f"{ms:.1f} ms"
        return f"{ms:.2f} ms"

    x = list(range(len(labels)))
    yerrs = [s / base_mean for s in stdev_ms]
    sorted_mults = sorted(multipliers)
    big = sorted_mults[-1]
    second = sorted_mults[-2] if len(sorted_mults) > 1 else big
    # Use a broken y-axis only if there's a clear order-of-magnitude gap
    # between the tallest bar and the next one. Otherwise a single axis
    # is fine.
    BREAK_RATIO = 5.0
    use_break = (second > 0 and big / second > BREAK_RATIO)

    if use_break:
        # Use the brokenaxes library — the canonical clean-looking
        # broken-axis idiom in matplotlib's own gallery example and the
        # widely-recommended approach for this pattern. It draws the
        # standard small diagonal break markers on both spines
        # automatically; we don't hand-roll any zigzag polygons.
        from brokenaxes import brokenaxes

        # Fixed break window — anything below 100× is shown 1:1 in the
        # lower panel; anything above 950× is shown in the upper panel.
        # Bars that fall inside the (100, 950) range would be invisible,
        # so the script intentionally clamps the break to known data.
        lower_top = 100.0
        upper_low = 950.0
        upper_high = max(big * 1.18, upper_low + 50)

        fig = plt.figure(figsize=(10, 6.5))
        bax = brokenaxes(
            ylims=((0, lower_top), (upper_low, upper_high)),
            hspace=0.06,
            d=0.006,              # diagonal break-marker length
            tilt=55,               # tilt angle of the slashes
            despine=False,
        )
        bax.bar(x, multipliers, color=colors, alpha=1.0,
                yerr=yerrs, capsize=6, ecolor="#333333",
                edgecolor="black", linewidth=0.5)
        bax.axhline(1.0, color="#888888", linestyle="--", linewidth=1,
                    zorder=0, label="noisy-maestro baseline (1.00×)")

        # Annotate every bar; brokenaxes routes ``annotate`` to whichever
        # underlying axis the y-coord falls in.
        for xi, mult, mean_ms, median_ms in zip(
                x, multipliers, means_ms, medians_ms):
            label = (f"{mult:.2f}×\n"
                     f"mean {_fmt_time(mean_ms)}\n"
                     f"median {_fmt_time(median_ms)}")
            bax.annotate(label, xy=(xi, mult), xytext=(0, 8),
                         textcoords="offset points",
                         ha="center", va="bottom",
                         fontsize=10, fontweight="bold")

        # x-ticks only on the bottom panel; brokenaxes stores its
        # underlying axes in bax.axs (top first, bottom last).
        bax.axs[-1].set_xticks(x)
        bax.axs[-1].set_xticklabels(labels, fontsize=10,
                                    rotation=22, ha="right")
        bax.axs[0].tick_params(labelbottom=False, bottom=False)

        bax.set_ylabel(
            f"runtime ratio (1.00× = noisy maestro = "
            f"{_fmt_time(base_mean)})",
            fontsize=11, labelpad=30,
        )
        bax.legend(loc="upper left", fontsize=9)
        fig.suptitle(
            f"Simulation time per circuit — {len(prepared)} sasquatch_epoch "
            f"circuit{'s' if len(prepared) != 1 else ''} (17q), "
            f"{args.shots} shots, backend = {args.backend}",
            fontsize=12, y=0.98,
        )
        fig.subplots_adjust(top=0.92, bottom=0.22, left=0.11, right=0.96)
    else:
        # Single axis when the multipliers all sit in the same order of
        # magnitude.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x, multipliers, color=colors, alpha=0.9,
               yerr=yerrs, capsize=6, ecolor="#333333")
        ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1,
                   zorder=0, label="noisy-maestro baseline (1.00×)")
        for xi, mult, mean_ms, median_ms in zip(
                x, multipliers, means_ms, medians_ms):
            label = (f"{mult:.2f}×\n"
                     f"mean {_fmt_time(mean_ms)}\n"
                     f"median {_fmt_time(median_ms)}")
            ax.annotate(label, xy=(xi, mult), xytext=(0, 8),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=20, ha="right")
        ax.set_ylabel(f"runtime ratio (1.00× = noisy maestro = "
                      f"{_fmt_time(base_mean)})")
        ax.set_title(
            f"Simulation time per circuit — {len(prepared)} sasquatch_epoch "
            f"circuit{'s' if len(prepared) != 1 else ''} (17q), "
            f"{args.shots} shots, backend = {args.backend}"
        )
        ax.set_ylim(0, max(multipliers) * 1.35)
        ax.legend(loc="upper left", fontsize=9)
        fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=120)
    plt.close(fig)
    print(f"\nPlot saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
