#!/usr/bin/env python3
"""Compare maestro bridge vs qiskit-aer vs real IQM hardware on
windowed mirror circuits.

Mirrors bridge_vs_hardware.py but targets IQM devices (garnet / emerald /
sirius).  Windowed subsystem-mirror circuits have a deterministic ``|00…0⟩``
ideal outcome, so Hellinger fidelity = P("00…0") in the offline case.
When ``--hardware`` is passed, hardware counts become the reference and each
simulator's distribution is compared against it via full Hellinger fidelity.

Four pathways per circuit:

  1. ``exact``          — noiseless ``maestro.simple_execute``
  2. ``qiskit-aer``     — ``AerSimulator(NoiseModel.from_backend(fake))``
  3. ``maestro bridge`` — ``br.execute_backend_noise_model``
  4. ``hardware``       — real IQM backend via ``IQMProvider`` (opt-in)

Reference distributions
-----------------------
* **Offline** (no ``--hardware``): ``δ_{00…0}`` ideal, so fidelity
  = P("00…0") for each pathway. Exact should be ≈ 1; noisy pathways reflect
  the noise model's predicted device error.
* **Hardware** (``--hardware``): the live device output.  Fidelity measures
  how closely each simulator reproduces actual hardware noise.

Usage
-----
    # Offline (fake backend, ideal reference)
    python python/validation/bridge_vs_hardware_iqm.py
    python python/validation/bridge_vs_hardware_iqm.py --limit 5 --shots 2048

    # With live hardware (hardware as reference)
    export IQM_TOKEN=<your-token>
    python python/validation/bridge_vs_hardware_iqm.py --hardware
    python python/validation/bridge_vs_hardware_iqm.py --hardware --device emerald
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PYTHON_DIR)

_BUILD_FIXED = os.path.join(_REPO, "build", "cp312-cp312-macosx_15_0_x86_64")
_BUILD_GLOB = next(
    (p for p in glob.glob(os.path.join(_REPO, "build", "cp3*"))
     if os.path.isdir(p)),
    None,
)
for _p in (_BUILD_FIXED, _BUILD_GLOB, _PYTHON_DIR):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import maestro       # noqa: E402
import qiskit        # noqa: E402
import qiskit.qasm2  # noqa: E402
import noise_bridge as br  # noqa: E402

DEFAULT_CIRCUITS_DIR = os.path.join(_HERE, "windowed_mirror_circuits_iqm")
DEFAULT_OUTPUT = os.path.join(_PYTHON_DIR, "_plots",
                              "bridge_vs_hardware_iqm.png")

# ── Device registry ────────────────────────────────────────────────────────
_IQM_DEVICES: dict[str, tuple[str, str]] = {
    "garnet": ("https://cocos.resonance.meetiqm.com/garnet", "IQMFakeGarnet"),
}
DEFAULT_DEVICE = "garnet"
# ──────────────────────────────────────────────────────────────────────────

# IQM native gate set (prx / cz) requires the legacy QASM2 instructions.
_QASM_KWARGS = dict(
    custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
)


# ---------------------------------------------------------------------------
# Simulator config
# ---------------------------------------------------------------------------


def _sv_config() -> maestro.SimulatorConfig:
    return maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.Statevector,
    )


# ---------------------------------------------------------------------------
# IQM backend helpers (mirrors benchmark_accuracy_iqm.py)
# ---------------------------------------------------------------------------


def _load_iqm_fake(name: str):
    """Instantiate an IQMFakeBackend by class name with submodule fallback."""
    import importlib

    for mod_path in ("iqm.qiskit_iqm.fake_backends", "iqm.qiskit_iqm"):
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, name):
                return getattr(mod, name)()
        except ImportError:
            continue

    suffix = name.removeprefix("IQMFake").lower()
    submod_path = f"iqm.qiskit_iqm.fake_backends.fake_{suffix}"
    try:
        submod = importlib.import_module(submod_path)
        if hasattr(submod, name):
            return getattr(submod, name)()
    except ImportError:
        pass

    raise SystemExit(
        f"IQM fake backend '{name}' not found. "
        "Ensure iqm-client[qiskit] is installed: uv add 'iqm-client[qiskit]'"
    )


def _connect_iqm_live(url: str, token: str | None):
    """Return a live IQMBackend from IQMProvider."""
    try:
        from iqm.qiskit_iqm import IQMProvider
    except ImportError:
        raise SystemExit(
            "--hardware requires iqm-client[qiskit].  "
            "Run: uv add 'iqm-client[qiskit]'"
        )
    kwargs: dict = {"url": url}
    if token:
        kwargs["token"] = token
    return IQMProvider(**kwargs).get_backend()


# ---------------------------------------------------------------------------
# Bitstring marginalisation (maestro full register → qiskit creg format)
# ---------------------------------------------------------------------------


def _measure_map(transpiled: qiskit.QuantumCircuit) -> tuple[dict[int, int], int]:
    """Build a qubit_index → classical_bit_index map from a transpiled circuit."""
    mapping: dict[int, int] = {}
    for inst in transpiled.data:
        if inst.operation.name == "measure":
            q = transpiled.find_bit(inst.qubits[0]).index
            c = transpiled.find_bit(inst.clbits[0]).index
            mapping[q] = c
    return mapping, transpiled.num_clbits


def _marginalize_counts(counts: dict[str, int], measure_map: dict[int, int],
                        creg_w: int) -> dict[str, int]:
    """Project maestro full-register bitstrings onto a qiskit creg.

    Maestro: qubit-0-LEFT (leftmost char = qubit 0).
    Qiskit:  c[0]-RIGHT  (rightmost char = c[0]).
    Qubit q → creg bit cb means out position = creg_w - 1 - cb.
    """
    out: dict[str, int] = {}
    for bits, n in counts.items():
        marg = ["0"] * creg_w
        for q, cb in measure_map.items():
            if q < len(bits):
                marg[creg_w - 1 - cb] = bits[q]
        k = "".join(marg)
        out[k] = out.get(k, 0) + n
    return out


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------


def _fidelity_to_ideal(counts: dict[str, int], shots: int,
                       creg_w: int) -> float:
    """For a mirror circuit, F_H(P, δ_{00…0}) = P("00…0")."""
    return counts.get("0" * creg_w, 0) / shots


def _hellinger_fidelity(p: dict[str, int], shots_p: int,
                        q: dict[str, int], shots_q: int) -> float:
    """``F_H = (Σ_x √(p_x · q_x))²`` between two count histograms."""
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        s += math.sqrt((p.get(k, 0) / shots_p) * (q.get(k, 0) / shots_q))
    return s * s


# ---------------------------------------------------------------------------
# Per-pathway runners
# ---------------------------------------------------------------------------


def _run_exact(tr_qasm: str, shots: int) -> dict[str, int]:
    return maestro.simple_execute(tr_qasm, _sv_config(), shots=shots)["counts"]


def _run_aer(transpiled: qiskit.QuantumCircuit, aer_sim, shots: int,
             seed: int = 42) -> dict[str, int]:
    return aer_sim.run(transpiled, shots=shots, seed_simulator=seed).result().get_counts()


def _run_bridge(qc: qiskit.QuantumCircuit, backend, shots: int) -> dict[str, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = br.execute_backend_noise_model(
            qc, backend, config=_sv_config(), shots=shots,
        )
    return result["counts"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _mean_nanaware(vals: list[float]) -> float:
    good = [v for v in vals if not math.isnan(v)]
    return sum(good) / len(good) if good else float("nan")


def _render_plot(window_indices: list[int],
                 series: dict[str, tuple[list[float], str]],
                 shots: int, backend_name: str, out_path: str,
                 hardware_shots: int | None, ref_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, len(window_indices) * 0.9 + 2), 5.5))
    x = list(range(len(window_indices)))
    n_src = len(series)
    w = 0.8 / max(1, n_src)

    for i, (label, (fids, color)) in enumerate(series.items()):
        offset = (i - (n_src - 1) / 2) * w
        ax.bar([xi + offset for xi in x], fids, width=w,
               label=label, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"q[{5*nn}..{5*nn+4}]" for nn in window_indices],
                       rotation=60, ha="right", fontsize=8,
                       fontfamily="monospace")
    ax.set_ylabel(f"Hellinger fidelity vs {ref_label}")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    shot_tag = f"{shots} shots/sim"
    if hardware_shots is not None and hardware_shots != shots:
        shot_tag += f", {hardware_shots} shots/hw"
    ax.set_title(
        f"Bridge vs hardware benchmark — IQM {backend_name}\n"
        f"Hellinger fidelity vs {ref_label} ({shot_tag})"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--circuits-dir", default=DEFAULT_CIRCUITS_DIR,
                        help="Directory with subsystem_mirror_NN.qasm files "
                             "(default: windowed_mirror_circuits_iqm).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Run only the first N circuits (default 0 = all).")
    parser.add_argument("--shots", type=int, default=4000,
                        help="Shots per simulator pathway (default 4000).")
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        choices=list(_IQM_DEVICES),
                        help=f"IQM device (default {DEFAULT_DEVICE!r}). "
                             "Sets both the live URL and the matching fake backend.")
    parser.add_argument("--skip-exact", action="store_true",
                        help="Skip the noiseless maestro run.")
    parser.add_argument("--skip-aer", action="store_true",
                        help="Skip the qiskit-aer pathway.")
    parser.add_argument("--hardware", action="store_true",
                        help="Run on the real IQM device and use those counts as "
                             "the Hellinger reference.  Requires --iqm-token or "
                             "IQM_TOKEN env var.")
    parser.add_argument("--iqm-token", default=None,
                        help="IQM API token (overrides IQM_TOKEN env var).")
    parser.add_argument("--hardware-shots", type=int, default=None,
                        help="Shots for hardware runs (default: same as --shots).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output plot path.")
    args = parser.parse_args()

    hardware_shots = args.hardware_shots or args.shots

    # --- Device registry ---------------------------------------------------
    device_url, fake_class_name = _IQM_DEVICES[args.device]

    # --- Fake backend (transpilation + aer) --------------------------------
    fake_backend = _load_iqm_fake(fake_class_name)
    print(f"Device: {args.device}  fake={type(fake_backend).__name__} "
          f"({fake_backend.num_qubits} qubits)", flush=True)

    # --- Live backend (optional) ------------------------------------------
    live_backend = None
    if args.hardware:
        token = args.iqm_token or os.environ.get("IQM_TOKEN")
        if not token:
            raise SystemExit("--hardware requires --iqm-token or IQM_TOKEN env var.")
        print(f"Connecting to {args.device} at {device_url}...", flush=True)
        live_backend = _connect_iqm_live(device_url, token)
        print(f"Live backend: {live_backend.name} "
              f"({live_backend.num_qubits} qubits)", flush=True)
    else:
        print("Offline mode — reference is ideal |00…0⟩.", flush=True)

    # Noise source: live calibration when available, fake otherwise.
    noise_source = live_backend if live_backend is not None else fake_backend

    # --- Load and transpile circuits --------------------------------------
    files = sorted(glob.glob(os.path.join(args.circuits_dir,
                                          "subsystem_mirror_*.qasm")))
    if not files:
        raise SystemExit(
            f"No subsystem_mirror_*.qasm files in {args.circuits_dir}\n"
            "Generate IQM windowed-mirror circuits and place them there."
        )
    if args.limit:
        files = files[: args.limit]
    print(f"\nLoaded {len(files)} windowed-mirror circuits from "
          f"{args.circuits_dir}", flush=True)

    # tuple: (window_index, stem, qc, tr, tr_qasm, measure_map, creg_w)
    prepared: list[tuple[int, str, qiskit.QuantumCircuit,
                         qiskit.QuantumCircuit, str,
                         dict[int, int], int]] = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            nn = int(stem.rsplit("_", 1)[-1])
        except ValueError:
            nn = len(prepared)
        with open(f) as fp:
            qasm = fp.read()
        qc = qiskit.qasm2.loads(qasm, **_QASM_KWARGS)
        tr = qiskit.transpile(qc, backend=fake_backend, optimization_level=1,
                              seed_transpiler=42)
        tr_qasm = qiskit.qasm2.dumps(tr)
        mm, creg_w = _measure_map(tr)
        prepared.append((nn, stem, qc, tr, tr_qasm, mm, creg_w))

    print(f"Transpiled {len(prepared)} circuits (target: "
          f"{type(fake_backend).__name__}).", flush=True)

    # --- Aer simulator ----------------------------------------------------
    aer_sim = None
    if not args.skip_aer:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel as AerNoiseModel
        print("Building qiskit-aer NoiseModel from fake backend...", flush=True)
        aer_noise = AerNoiseModel.from_backend(fake_backend)
        aer_sim = AerSimulator(noise_model=aer_noise)

    # --- Hardware: submit all circuits up front ---------------------------
    hardware_counts_list: list[dict[str, int]] | None = None
    if live_backend is not None:
        hardware_counts_list = []
        print(f"\nSubmitting {len(prepared)} circuits to hardware "
              f"({hardware_shots} shots each)...", flush=True)
        for nn, stem, _, tr, _, _, _ in prepared:
            print(f"  q[{5*nn}..{5*nn+4}]...", end=" ", flush=True)
            job = live_backend.run(tr, shots=hardware_shots)
            hardware_counts_list.append(job.result().get_counts())
            print("done", flush=True)
        print(f"Hardware results received for {len(hardware_counts_list)} "
              f"circuits.", flush=True)

    use_hw_ref = hardware_counts_list is not None

    # --- Per-circuit simulator pathways -----------------------------------
    fid_exact:  list[float] = []
    fid_aer:    list[float] = []
    fid_bridge: list[float] = []

    print("\nRunning simulator pathways per circuit...", flush=True)
    for i, (nn, stem, qc, tr, tr_qasm, mm, creg_w) in enumerate(prepared):
        tag = f"[{i + 1:>2}/{len(prepared)}] q[{5*nn}..{5*nn+4}]"
        parts = [tag]

        def _fid(counts_q: dict[str, int], shots: int) -> float:
            """Fidelity vs the chosen reference (hardware or ideal)."""
            if use_hw_ref:
                return _hellinger_fidelity(counts_q, shots,
                                           hardware_counts_list[i], hardware_shots)
            return _fidelity_to_ideal(counts_q, shots, creg_w)

        # 1. Exact (noiseless)
        if not args.skip_exact:
            exact_full = _run_exact(tr_qasm, args.shots)
            exact_q = _marginalize_counts(exact_full, mm, creg_w)
            f = _fid(exact_q, args.shots)
            fid_exact.append(f)
            parts.append(f"F_exact={f:.3f}")
        else:
            fid_exact.append(float("nan"))

        # 2. Qiskit-aer
        if aer_sim is not None:
            aer_counts = _run_aer(tr, aer_sim, args.shots)
            f = _fid(aer_counts, args.shots)
            fid_aer.append(f)
            parts.append(f"F_aer={f:.3f}")
        else:
            fid_aer.append(float("nan"))

        # 3. Maestro bridge (pass original qc; bridge handles transpilation)
        bridge_full = _run_bridge(qc, noise_source, args.shots)
        bridge_q = _marginalize_counts(bridge_full, mm, creg_w)
        f = _fid(bridge_q, args.shots)
        fid_bridge.append(f)
        parts.append(f"F_bridge={f:.3f}")

        print("  " + "  ".join(parts), flush=True)

    # --- Summary ----------------------------------------------------------
    ref_label = (f"{live_backend.name if live_backend else args.device} hardware"
                 if use_hw_ref else "ideal |00…0⟩")
    print(f"\n=== Hellinger fidelity vs {ref_label} (mean across circuits) ===")
    if not args.skip_exact:
        print(f"  exact          mean F_H = {_mean_nanaware(fid_exact):.4f}")
    if aer_sim is not None:
        print(f"  qiskit-aer     mean F_H = {_mean_nanaware(fid_aer):.4f}")
    print(    f"  maestro bridge mean F_H = {_mean_nanaware(fid_bridge):.4f}")

    # --- Plot -------------------------------------------------------------
    window_indices = [nn for (nn, *_) in prepared]
    series: dict[str, tuple[list[float], str]] = {}
    if not args.skip_exact:
        series["exact (noise-free)"] = (fid_exact, "#1f77b4")
    if aer_sim is not None:
        series["qiskit-aer"] = (fid_aer, "#2ca02c")
    series["maestro bridge"] = (fid_bridge, "#d62728")

    backend_label = (live_backend.name if live_backend
                     else type(fake_backend).__name__)
    _render_plot(window_indices, series, args.shots, backend_label,
                 args.output, hardware_shots if use_hw_ref else None,
                 ref_label)
    print(f"\nPlot saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
