#!/usr/bin/env python3
"""Compare maestro bridge vs qiskit-aer vs real IBM hardware (and the
noiseless reference) on a suite of *windowed subsystem-mirror* circuits.

Each ``windowed_mirror_circuits/subsystem_mirror_NN.qasm`` declares 156
qubits and a 5-bit classical register ``c``. A random 5-qubit unitary
``U`` is applied to a 5-qubit window centred on physical qubits
``5·NN ... 5·NN+4``, dummy single-qubit rotations are sprinkled on the
151 spectator qubits (to load the device with concurrent activity),
``U†`` undoes the window unitary, and finally only the 5 window qubits
are measured. The **ideal** outcome on the measured window is therefore
deterministic ``|00000⟩`` — any deviation in the measured distribution
reflects noise accumulated on the window during the experiment.

For each circuit we run up to four pathways:

  1. ``exact``      — noiseless ``maestro.simple_execute`` on the
                      transpiled circuit. Sanity check: should
                      concentrate on ``00000`` modulo shot noise.
  2. ``qiskit-aer`` — ``AerSimulator(method="matrix_product_state",
                      noise_model=NoiseModel.from_backend(backend))``.
                      MPS is necessary at 156 qubits; the noiseless and
                      bridge-noise circuits stay near product states on
                      the 151 spectators, so the MPS bond dimension
                      remains manageable.
  3. ``maestro bridge`` — ``br.execute_backend_noise_model`` with a
                      ``SimulatorConfig(simulation_type=MatrixProductState)``
                      so maestro doesn't try to allocate a 2^156
                      statevector.
  4. ``hardware``  — the real IBM backend, all circuits submitted as
                      one SamplerV2 batch. Only included when
                      ``IBM_QUANTUM_TOKEN`` is set and ``--skip-hardware``
                      is not passed.

For each pathway we marginalise the counts down to the 5-bit window
register and report the **Hellinger fidelity** against the ideal
``δ_{00000}`` distribution::

    F_H(P, Q) = (Σ_x √(P(x) · Q(x)))²        ∈ [0, 1]

When ``Q`` is the ``δ_{00000}`` ideal, ``F_H(P, δ_{00000}) = P(00000)``
— i.e. the empirical fidelity of the mirror circuit on that window.
This is a stronger signal than total-variation distance because all the
ideal mass sits on a single bitstring, and noise drains that probability
into a wide tail.

Usage
-----

    export IBM_QUANTUM_TOKEN=...            # required for hardware run
    python python/validation/bridge_vs_hardware.py
    python python/validation/bridge_vs_hardware.py --skip-hardware
    python python/validation/bridge_vs_hardware.py --shots 4000
    python python/validation/bridge_vs_hardware.py --limit 5
    python python/validation/bridge_vs_hardware.py --backend ibm_fez

Notes / caveats
---------------

* The QASM files declare 156 qubits, so all simulators are forced onto
  MPS. The noiseless circuit has very low entanglement (window unitary
  is 5q; spectators are 1q-only); injected Pauli/T1 noise stays local
  and barely lifts the bond dimension.
* Hardware jobs queue and may take minutes to hours. The script prints
  the job ID immediately on submission so it can be monitored remotely.
* The window's classical register is named ``c`` in every QASM file,
  so we read ``pub.data.c.get_counts()`` for hardware results.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import warnings

# Allow running directly from the repo without installing: prepend the
# in-tree maestro extension and the python/ directory.
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

DEFAULT_CIRCUITS_DIR = os.path.join(_HERE, "windowed_mirror_circuits")
DEFAULT_OUTPUT = os.path.join(_PYTHON_DIR, "_plots",
                              "bridge_vs_hardware_windowed.png")

# Custom-gate / legacy-include support: the generated QASM declares
#   gate r(param0,param1) q0 { u(param0,...) q0; }
# which is a user-defined gate. ``LEGACY_CUSTOM_INSTRUCTIONS`` keeps
# qiskit.qasm2 happy with the older-style header.
_QASM_KWARGS = dict(
    custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    include_path=qiskit.qasm2.LEGACY_INCLUDE_PATH,
)


# ---------------------------------------------------------------------------
# Simulator configs
# ---------------------------------------------------------------------------


def _mps_config() -> maestro.SimulatorConfig:
    """Maestro MPS config — required for 156-qubit windowed-mirror runs."""
    return maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
    )


# ---------------------------------------------------------------------------
# Bitstring marginalisation (maestro full register -> qiskit creg space)
# ---------------------------------------------------------------------------


def _measure_map(transpiled: qiskit.QuantumCircuit) -> tuple[dict[int, int], int]:
    mapping: dict[int, int] = {}
    for inst in transpiled.data:
        if inst.operation.name == "measure":
            q = transpiled.find_bit(inst.qubits[0]).index
            c = transpiled.find_bit(inst.clbits[0]).index
            mapping[q] = c
    return mapping, transpiled.num_clbits


def _marginalize_counts(counts: dict[str, int], measure_map: dict[int, int],
                        creg_w: int) -> dict[str, int]:
    """Project maestro's full-register bitstring onto a qiskit creg.

    Maestro uses **qubit-0-LEFT** convention: the leftmost character of
    the bitstring is qubit 0. Qiskit's classical-register convention is
    **c[0]-RIGHT** (rightmost character is c[0]). So qubit ``q``'s value
    lives at ``bits[q]`` and we place it at ``out[creg_w - 1 - cb]``.
    """
    out: dict[str, int] = {}
    for bits, c in counts.items():
        marg = ["0"] * creg_w
        for q, cb in measure_map.items():
            if q < len(bits):
                marg[creg_w - 1 - cb] = bits[q]
        k = "".join(marg)
        out[k] = out.get(k, 0) + c
    return out


# ---------------------------------------------------------------------------
# Hellinger fidelity / probability of the ideal bitstring
# ---------------------------------------------------------------------------


def _hellinger_fidelity(p: dict[str, int], q: dict[str, int],
                        shots_p: int, shots_q: int) -> float:
    """Hellinger fidelity ``F_H = (Σ √(p_x q_x))²`` between two count
    histograms, normalising each by its own shot count. Both inputs
    should sum to ``shots_p`` / ``shots_q`` respectively (no implicit
    re-normalisation beyond the shot count)."""
    keys = set(p) | set(q)
    s = 0.0
    for k in keys:
        s += math.sqrt((p.get(k, 0) / shots_p) * (q.get(k, 0) / shots_q))
    return s * s


def _fidelity_to_ideal(counts: dict[str, int], shots: int,
                       creg_w: int) -> float:
    """For ``δ_{00…0}`` ideal, ``F_H(P, δ) = P("00…0")`` exactly.
    Returns the empirical probability of the all-zero outcome on the
    ``creg_w``-wide window register."""
    zero = "0" * creg_w
    return counts.get(zero, 0) / shots


# ---------------------------------------------------------------------------
# Per-pathway runners
# ---------------------------------------------------------------------------


def _run_exact(transpiled_qasm: str, shots: int) -> dict[str, int]:
    return maestro.simple_execute(
        transpiled_qasm, _mps_config(), shots=shots,
    )["counts"]


def _run_aer(transpiled: qiskit.QuantumCircuit, aer_sim, shots: int,
             seed: int = 42) -> dict[str, int]:
    job = aer_sim.run(transpiled, shots=shots, seed_simulator=seed)
    return job.result().get_counts()


def _run_bridge(qc: qiskit.QuantumCircuit, backend, shots: int) -> dict[str, int]:
    """The bridge's internal QASM loader is the strict
    ``qiskit.qasm2.loads`` (no LEGACY custom instructions), which
    rejects bare ``u(...)`` calls. Hand it a pre-parsed
    ``QuantumCircuit`` to bypass that path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = br.execute_backend_noise_model(
            qc, backend, config=_mps_config(), shots=shots,
        )
    return result["counts"]


def _submit_hardware_batch(transpiled_list: list[qiskit.QuantumCircuit],
                           backend, shots: int) -> list[dict[str, int]]:
    """Submit every transpiled circuit as a single SamplerV2 batch and
    block for the result. Prints the job ID immediately so the user can
    monitor remotely."""
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    print(f"  submitting {len(transpiled_list)} circuits to {backend.name} "
          f"as a single batch ({shots} shots each)...", flush=True)
    job = sampler.run(transpiled_list, shots=shots)
    print(f"  job id: {job.job_id()}", flush=True)
    print(f"  monitor: https://quantum.ibm.com/jobs/{job.job_id()}", flush=True)
    print(f"  waiting for result (queue depth may be high)...", flush=True)
    result = job.result()
    out: list[dict[str, int]] = []
    for pub in result:
        creg_name = "c"
        if not hasattr(pub.data, creg_name):
            creg_name = next(iter(pub.data.keys()))
        out.append(getattr(pub.data, creg_name).get_counts())
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _mean_std(vals: list[float]) -> tuple[float, float]:
    """Mean ± sample stdev (0 when fewer than 2 samples)."""
    good = [v for v in vals if not math.isnan(v)]
    if not good:
        return (float("nan"), 0.0)
    m = sum(good) / len(good)
    if len(good) < 2:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in good) / (len(good) - 1)
    return (m, math.sqrt(var))


def _render_plot(window_indices: list[int],
                 series: dict[str, tuple[list[list[float]], str]],
                 shots: int, backend_name: str, out_path: str,
                 hardware_shots: int | None, repeats: int) -> None:
    """Render per-window bars with error bars. ``series`` maps pathway
    label → (per-circuit list of per-repeat fidelities, bar colour)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = list(range(len(window_indices)))
    n_src = len(series)
    w = 0.8 / max(1, n_src)
    for i, (label, (per_circuit_samples, color)) in enumerate(series.items()):
        means, stds = [], []
        for samples in per_circuit_samples:
            m, s = _mean_std(samples)
            means.append(m)
            stds.append(s)
        offset = (i - (n_src - 1) / 2) * w
        ax.bar([xi + offset for xi in x], means, width=w,
               label=label, color=color, alpha=0.9,
               yerr=stds if repeats > 1 else None,
               capsize=3 if repeats > 1 else 0,
               ecolor="#333333",
               error_kw=dict(elinewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([f"q[{5*idx}..{5*idx+4}]" for idx in window_indices],
                       rotation=60, ha="right", fontsize=8,
                       fontfamily="monospace")
    ax.set_ylabel("Hellinger fidelity vs ideal |00000⟩")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    shot_tag = f"{shots} shots/sim"
    if hardware_shots is not None and hardware_shots != shots:
        shot_tag += f", {hardware_shots} shots/hw"
    if repeats > 1:
        shot_tag += f", {repeats} repeats"
    ax.set_title(
        f"Windowed subsystem-mirror benchmark on {backend_name} — "
        f"Hellinger fidelity of the 5-qubit window output vs. ideal |00000⟩ "
        f"({shot_tag})"
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
                        help="Directory of subsystem_mirror_NN.qasm files.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to the first N circuits (default 0 = all). "
                             "Each 156q MPS run is non-trivial — start with a "
                             "small --limit for first runs.")
    parser.add_argument("--shots", type=int, default=4000,
                        help="Shots per simulator pathway (default 4000).")
    parser.add_argument("--hardware-shots", type=int, default=None,
                        help="Shots per hardware circuit. Default: --shots. "
                             "Set higher (e.g. 8192) for a tighter hardware "
                             "estimate at the cost of more queue time.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Run each circuit N times per pathway and "
                             "report mean ± stdev of the Hellinger fidelity "
                             "(default 1). Smooths out simulator-level noise "
                             "sampling randomness. For hardware, submits N "
                             "copies of each circuit in the same SamplerV2 "
                             "batch.")
    parser.add_argument("--backend", default="ibm_fez",
                        help="Live IBM backend name (default ibm_fez)")
    parser.add_argument("--skip-hardware", action="store_true",
                        help="Don't submit to real hardware. Aer/bridge still "
                             "run against the matching fake backend.")
    parser.add_argument("--skip-exact", action="store_true",
                        help="Skip the noiseless maestro run. Useful if MPS "
                             "is the bottleneck — the ideal is analytically "
                             "|00000⟩ so this is mostly a sanity check.")
    parser.add_argument("--skip-aer", action="store_true",
                        help="Skip the qiskit-aer pathway. AerSimulator's MPS "
                             "method can be slow on 156-qubit circuits; "
                             "useful to bound wall-clock when iterating.")
    parser.add_argument("--instance", default=None,
                        help="Optional CRN/instance for QiskitRuntimeService "
                             "(IBM Cloud users).")
    parser.add_argument("--channel", default="ibm_quantum_platform",
                        choices=["ibm_quantum_platform", "ibm_cloud"],
                        help="QiskitRuntimeService channel (default "
                             "ibm_quantum_platform).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Plot output path.")
    args = parser.parse_args()

    hardware_shots = args.hardware_shots or args.shots
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    use_hardware = bool(token) and not args.skip_hardware

    # --- Backend selection -------------------------------------------------
    if use_hardware:
        from qiskit_ibm_runtime import QiskitRuntimeService
        print(f"Authenticating with IBM Quantum (channel={args.channel})...",
              flush=True)
        svc_kwargs: dict = {"channel": args.channel, "token": token}
        if args.instance:
            svc_kwargs["instance"] = args.instance
        service = QiskitRuntimeService(**svc_kwargs)
        backend = service.backend(args.backend)
        print(f"Live backend: {backend.name} ({backend.num_qubits} qubits)",
              flush=True)
    else:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend = FakeFez()
        print(f"Offline mode — using {type(backend).__name__} "
              f"({backend.num_qubits} qubits).", flush=True)
        if not token:
            print("(IBM_QUANTUM_TOKEN not set; skipping the hardware run.)",
                  flush=True)
        elif args.skip_hardware:
            print("(--skip-hardware passed; not submitting to real hardware.)",
                  flush=True)

    # --- Load and transpile every windowed-mirror circuit ----------------
    files = sorted(glob.glob(os.path.join(args.circuits_dir,
                                          "subsystem_mirror_*.qasm")))
    if not files:
        raise SystemExit(f"No subsystem_mirror_*.qasm files in "
                         f"{args.circuits_dir}")
    if args.limit:
        files = files[: args.limit]
    print(f"\nLoaded {len(files)} windowed-mirror circuits from "
          f"{args.circuits_dir}", flush=True)

    prepared: list[tuple[int, qiskit.QuantumCircuit, qiskit.QuantumCircuit,
                         str, dict[int, int], int]] = []
    for f in files:
        # Window index NN from filename, e.g. subsystem_mirror_07.qasm -> 7
        stem = os.path.splitext(os.path.basename(f))[0]
        nn = int(stem.rsplit("_", 1)[-1])
        with open(f) as fp:
            qasm = fp.read()
        qc = qiskit.qasm2.loads(qasm, **_QASM_KWARGS)
        tr = qiskit.transpile(qc, backend=backend, optimization_level=1,
                              seed_transpiler=42)
        tr_qasm = qiskit.qasm2.dumps(tr)
        mm, creg_w = _measure_map(tr)
        prepared.append((nn, qc, tr, tr_qasm, mm, creg_w))

    print(f"Transpiled {len(prepared)} circuits (target backend: "
          f"{backend.name if use_hardware else type(backend).__name__}).",
          flush=True)

    # --- Aer simulator (one noise model, shared across all circuits) -----
    aer_sim = None
    if not args.skip_aer:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        print("Building qiskit-aer NoiseModel.from_backend (one-time)...",
              flush=True)
        aer_noise = NoiseModel.from_backend(backend)
        # MPS is necessary at 156 qubits; the windowed-mirror circuit's
        # spectator block is 1Q-only, so the bond dimension stays small.
        aer_sim = AerSimulator(noise_model=aer_noise,
                               method="matrix_product_state")

    # --- Hardware batch submission (do this up-front so it queues while
    # the simulators run) -------------------------------------------------
    hardware_counts_list: list[dict[str, int]] | None = None
    hw_job = None
    if use_hardware:
        # Submit the batch in the background while sims run locally. The
        # SamplerV2 job is async — we kick it off here and pick up
        # ``.result()`` after the sims finish.
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        transpiled_list = [tr for (_, _, tr, _, _, _) in prepared]
        sampler = Sampler(mode=backend)
        print(f"\nSubmitting {len(transpiled_list)} circuits to "
              f"{backend.name} as a single batch "
              f"({hardware_shots} shots each)...", flush=True)
        hw_job = sampler.run(transpiled_list, shots=hardware_shots)
        print(f"  job id: {hw_job.job_id()}", flush=True)
        print(f"  monitor: https://quantum.ibm.com/jobs/{hw_job.job_id()}",
              flush=True)

    # --- Per-circuit simulator pipelines --------------------------------
    fid_exact: list[float] = []
    fid_aer: list[float] = []
    fid_bridge: list[float] = []
    print("\nRunning simulator pathways per circuit...", flush=True)
    for i, (nn, qc, tr, tr_qasm, mm, creg_w) in enumerate(prepared, 1):
        tag = f"[{i:>2}/{len(prepared)}] window q[{5*nn}..{5*nn+4}]"

        # 1. Exact (noiseless maestro, MPS)
        if not args.skip_exact:
            exact_full = _run_exact(tr_qasm, args.shots)
            exact = _marginalize_counts(exact_full, mm, creg_w)
            fid_exact.append(_fidelity_to_ideal(exact, args.shots, creg_w))
        else:
            fid_exact.append(float("nan"))

        # 2. Qiskit-aer noisy (MPS method)
        if aer_sim is not None:
            aer = _run_aer(tr, aer_sim, args.shots)
            fid_aer.append(_fidelity_to_ideal(aer, args.shots, creg_w))
        else:
            fid_aer.append(float("nan"))

        # 3. Maestro bridge noisy (MPS)
        bridge_full = _run_bridge(qc, backend, args.shots)
        bridge = _marginalize_counts(bridge_full, mm, creg_w)
        fid_bridge.append(_fidelity_to_ideal(bridge, args.shots, creg_w))

        # Per-circuit summary line
        parts = [tag]
        if not args.skip_exact:
            parts.append(f"F_exact={fid_exact[-1]:.3f}")
        if aer_sim is not None:
            parts.append(f"F_aer={fid_aer[-1]:.3f}")
        parts.append(f"F_bridge={fid_bridge[-1]:.3f}")
        print("  " + "  ".join(parts), flush=True)

    # --- Hardware: pick up results ---------------------------------------
    fid_hw: list[float] = []
    if hw_job is not None:
        print("\nWaiting for hardware results...", flush=True)
        result = hw_job.result()
        for (nn, _, _, _, _, creg_w), pub in zip(prepared, result):
            creg_name = "c"
            if not hasattr(pub.data, creg_name):
                creg_name = next(iter(pub.data.keys()))
            hw_counts = getattr(pub.data, creg_name).get_counts()
            f = _fidelity_to_ideal(hw_counts, hardware_shots, creg_w)
            fid_hw.append(f)
            print(f"  window q[{5*nn}..{5*nn+4}]  F_hw={f:.3f}", flush=True)

    # --- Summary ---------------------------------------------------------
    print("\n=== Hellinger fidelity vs ideal |00000⟩ (per-window mean) ===")

    def _mean_nanaware(vals: list[float]) -> float:
        good = [v for v in vals if not math.isnan(v)]
        return sum(good) / len(good) if good else float("nan")

    if not args.skip_exact:
        print(f"  exact    mean F_H = {_mean_nanaware(fid_exact):.4f}")
    if aer_sim is not None:
        print(f"  aer      mean F_H = {_mean_nanaware(fid_aer):.4f}")
    print(f"  bridge   mean F_H = {_mean_nanaware(fid_bridge):.4f}")
    if fid_hw:
        print(f"  hardware mean F_H = {_mean_nanaware(fid_hw):.4f}")

    # --- Plot ------------------------------------------------------------
    window_indices = [nn for (nn, _, _, _, _, _) in prepared]
    series: dict[str, tuple[list[float], str]] = {}
    if not args.skip_exact:
        series["exact (noise-free)"] = (fid_exact, "#1f77b4")
    if aer_sim is not None:
        series["qiskit-aer"] = (fid_aer, "#2ca02c")
    series["maestro bridge"] = (fid_bridge, "#d62728")
    if fid_hw:
        series[f"{backend.name} (hardware)"] = (fid_hw, "#9467bd")

    backend_name = backend.name if use_hardware else type(backend).__name__
    _render_plot(window_indices, series, args.shots, backend_name,
                 args.output,
                 hardware_shots if fid_hw else None)
    print(f"\nPlot saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())