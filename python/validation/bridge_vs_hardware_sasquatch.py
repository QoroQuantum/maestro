#!/usr/bin/env python3
"""Compare maestro bridge vs qiskit-aer vs real IBM hardware (and the
noiseless reference) on a GHZ-N circuit.

Runs the same transpiled circuit through up to four pathways and renders a
side-by-side bar plot of the resulting marginal distributions:

  1. ``exact``       — noiseless ``maestro.simple_execute`` on the transpiled
                       circuit.
  2. ``qiskit-aer``  — ``AerSimulator(NoiseModel.from_backend(...))`` driven
                       by the same calibration data the bridge sees.
  3. ``maestro bridge`` — ``br.execute_backend_noise_model`` with the noise
                       model this repo builds (see ``noise_bridge.py``).
  4. ``hardware``    — the real IBM backend, via the Runtime SamplerV2
                       primitive. Only included when ``IBM_QUANTUM_TOKEN``
                       is set and ``--skip-hardware`` is not passed.

The exact / aer / bridge runs all use the *same* backend properties: when
a token is present we authenticate, fetch the live ``backend.properties()``
snapshot, and use it for both the aer and bridge noise models so that all
four pathways correspond to the same calibration. Without a token we fall
back to ``FakeFez`` (a snapshot bundled with qiskit-ibm-runtime).

Usage
-----

    export IBM_QUANTUM_TOKEN=...                # required for hardware run
    python python/validation/bridge_vs_hardware_sasquatch.py
    python python/validation/bridge_vs_hardware_sasquatch.py --skip-hardware
    python python/validation/bridge_vs_hardware_sasquatch.py --shots 4000 --ghz-n 7
    python python/validation/bridge_vs_hardware_sasquatch.py --qubits 0,1,2,3,4,5,6
    python python/validation/bridge_vs_hardware_sasquatch.py --backend ibm_fez

The script never writes the API token to disk and never echoes it. It uses
the modern ``QiskitRuntimeService`` + ``SamplerV2`` primitive path; the
deprecated ``backend.run()`` path is not used.

Notes / caveats
---------------

* IBM Fez is 156 qubits. Maestro's statevector backend allocates 2^N
  amplitudes where N is the highest physical qubit index the transpiled
  circuit touches, so the script prefers a low-index connected chain via
  ``initial_layout`` (auto-discovered from the coupling map, or specified
  via ``--qubits``). A warning is printed if the chosen layout pushes the
  active qubit window above 25 qubits.
* Hardware jobs queue and may take minutes to hours. The script prints
  the job ID immediately on submission so you can monitor it through the
  IBM Quantum dashboard.
* The "TVD vs. hardware" numbers are statistically noisy at the default
  ~4k shots (IBM's typical max per job is 100k; you can raise ``--shots``
  for a tighter estimate, queue permitting).
"""

from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------------------
# Circuit / layout helpers
# ---------------------------------------------------------------------------


def _ghz_qasm(n: int) -> str:
    """GHZ-n preparation with measurements on every qubit (creg name ``c``)."""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n}];",
        f"creg c[{n}];",
        "h q[0];",
    ]
    for i in range(n - 1):
        lines.append(f"cx q[{i}], q[{i+1}];")
    for i in range(n):
        lines.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(lines)


def _find_low_index_chain(coupling_map, length: int,
                          max_qubit: int = 30) -> list[int]:
    """DFS for a length-N linear chain in ``coupling_map`` restricted to
    physical qubit indices below ``max_qubit``. We pick low indices so
    that the maestro statevector (which allocates 2^max_used amplitudes)
    stays small.

    Falls back to relaxing ``max_qubit`` if no chain is found at the
    initial bound.
    """
    pairs = [(a, b) for a, b in coupling_map]
    while True:
        adj: dict[int, set[int]] = {}
        for a, b in pairs:
            if a < max_qubit and b < max_qubit:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

        def dfs(node: int, path: list[int], seen: set[int]) -> list[int] | None:
            if len(path) == length:
                return path
            for nbr in sorted(adj.get(node, ())):
                if nbr in seen:
                    continue
                result = dfs(nbr, path + [nbr], seen | {nbr})
                if result is not None:
                    return result
            return None

        for start in sorted(adj):
            chain = dfs(start, [start], {start})
            if chain is not None:
                return chain

        # Relax the bound and try again, up to the full device.
        if max_qubit >= 1000:
            raise RuntimeError(
                f"No {length}-qubit chain found in coupling map at any bound."
            )
        max_qubit = min(1000, max_qubit * 2)


# ---------------------------------------------------------------------------
# Bitstring marginalization (maestro full register -> qiskit creg space)
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
    **c[0]-RIGHT** (rightmost character is c[0]). So qubit q's value
    lives at ``bits[q]`` and we place it at ``out[creg_w - 1 - cb]``.
    """
    out: dict[str, int] = {}
    for bits, c in counts.items():
        marg = ["0"] * creg_w
        for q, cb in measure_map.items():
            marg[creg_w - 1 - cb] = bits[q]
        k = "".join(marg)
        out[k] = out.get(k, 0) + c
    return out


def _tvd(a: dict[str, int], b: dict[str, int], shots: int) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


# ---------------------------------------------------------------------------
# Per-pathway runners
# ---------------------------------------------------------------------------


def _run_exact(transpiled_qasm: str, shots: int) -> dict[str, int]:
    return maestro.simple_execute(transpiled_qasm, shots=shots)["counts"]


def _run_aer(transpiled: qiskit.QuantumCircuit, backend, shots: int,
             seed: int = 42) -> dict[str, int]:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    nm = NoiseModel.from_backend(backend)
    sim = AerSimulator(noise_model=nm)
    job = sim.run(transpiled, shots=shots, seed_simulator=seed)
    return job.result().get_counts()


def _run_bridge(qasm: str, backend, shots: int) -> dict[str, int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = br.execute_backend_noise_model(qasm, backend, shots=shots)
    return result["counts"]


def _run_hardware(transpiled: qiskit.QuantumCircuit, backend, shots: int) -> dict[str, int]:
    """Submit ``transpiled`` to ``backend`` via SamplerV2 and block for the
    result. Prints the job ID immediately so the user can monitor remotely.
    """
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    print(f"  submitting job to {backend.name}...", flush=True)
    job = sampler.run([transpiled], shots=shots)
    print(f"  job id: {job.job_id()}", flush=True)
    print(f"  waiting for result (queue depth may be high)...", flush=True)
    result = job.result()
    # SamplerV2 returns a PrimitiveResult; the first PUB's data carries the
    # classical register named in the circuit. Our circuit declares
    # ``creg c[...]`` so we read ``.data.c``.
    pub = result[0]
    creg_name = "c"
    if not hasattr(pub.data, creg_name):
        # Fall back to whatever single creg-like field exists.
        creg_name = next(iter(pub.data.keys()))
    return getattr(pub.data, creg_name).get_counts()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _render_plot(sources: list[tuple[str, dict[str, int], str]],
                 shots: int, backend_name: str, ghz_n: int,
                 hardware_tvd_text: str, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_keys: set[str] = set()
    for _, d, _ in sources:
        all_keys |= set(d)
    scored = sorted(all_keys,
                    key=lambda k: -sum(d.get(k, 0) for _, d, _ in sources))
    keys = scored[:20]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = list(range(len(keys)))
    n_src = len(sources)
    w = 0.8 / max(1, n_src)
    for i, (label, counts, color) in enumerate(sources):
        offset = (i - (n_src - 1) / 2) * w
        freqs = [counts.get(k, 0) / shots for k in keys]
        ax.bar([xi + offset for xi in x], freqs, width=w,
               label=label, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontfamily="monospace", fontsize=8,
                       rotation=60, ha="right")
    ax.set_ylabel("probability")
    title = (f"GHZ-{ghz_n} on {backend_name} — top 20 outcomes "
             f"({shots} shots)")
    if hardware_tvd_text:
        title += "\n" + hardware_tvd_text
    ax.set_title(title)
    ax.legend()
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
    parser.add_argument("--shots", type=int, default=10000,
                        help="Shots per pathway (default 10000)")
    parser.add_argument("--ghz-n", type=int, default=7,
                        help="GHZ state width (default 7)")
    parser.add_argument("--backend", default="ibm_fez",
                        help="Live IBM backend name (default ibm_fez)")
    parser.add_argument("--qubits", default=None,
                        help="Comma-separated initial_layout. Default: auto-find "
                             "a low-index connected chain on the device.")
    parser.add_argument("--skip-hardware", action="store_true",
                        help="Don't submit to real hardware. Falls back to "
                             "FakeFez for aer/bridge; exact and aer/bridge "
                             "still run.")
    parser.add_argument("--instance", default=None,
                        help="Optional CRN/instance for QiskitRuntimeService "
                             "(IBM Cloud users).")
    parser.add_argument("--channel", default="ibm_quantum_platform",
                        choices=["ibm_quantum_platform", "ibm_cloud"],
                        help="QiskitRuntimeService channel (default "
                             "ibm_quantum_platform).")
    parser.add_argument("--output", default=None,
                        help="Plot output path (default "
                             "python/_plots/bridge_vs_hardware.png).")
    args = parser.parse_args()

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

    # --- Pick qubits for the GHZ chain ------------------------------------
    if args.qubits:
        layout = [int(q) for q in args.qubits.split(",")]
        if len(layout) != args.ghz_n:
            print(f"ERROR: --qubits must list exactly {args.ghz_n} qubits.",
                  file=sys.stderr)
            return 2
    else:
        cmap = list(backend.coupling_map)
        layout = _find_low_index_chain(cmap, args.ghz_n, max_qubit=30)
    print(f"GHZ-{args.ghz_n} physical qubits: {layout}", flush=True)

    # --- Transpile once; all four pathways consume the same circuit ------
    qasm = _ghz_qasm(args.ghz_n)
    qc = qiskit.qasm2.loads(qasm)
    transpiled = qiskit.transpile(
        qc, backend=backend, initial_layout=layout,
        seed_transpiler=42, optimization_level=1,
    )
    transpiled_qasm = qiskit.qasm2.dumps(transpiled)
    measure_map, creg_w = _measure_map(transpiled)
    max_used_q = max(measure_map.keys())
    if max_used_q > 25:
        print(f"WARNING: highest used physical qubit is q[{max_used_q}]; "
              f"maestro will allocate ~2^{max_used_q + 1} amplitudes for the "
              f"statevector. Consider --qubits to pick a lower-index chain.",
              flush=True)

    # --- Run the pathways --------------------------------------------------
    print("\nRunning exact (no noise)...", flush=True)
    exact_full = _run_exact(transpiled_qasm, args.shots)
    exact = _marginalize_counts(exact_full, measure_map, creg_w)

    print("Running qiskit-aer NoiseModel.from_backend...", flush=True)
    aer = _run_aer(transpiled, backend, args.shots)

    print("Running maestro bridge noise model...", flush=True)
    bridge_full = _run_bridge(qasm, backend, args.shots)
    bridge = _marginalize_counts(bridge_full, measure_map, creg_w)

    hardware: dict[str, int] | None = None
    if use_hardware:
        print("Running on real hardware...", flush=True)
        hardware = _run_hardware(transpiled, backend, args.shots)
        top = sorted(hardware.items(), key=lambda kv: -kv[1])[:5]
        print(f"  hardware top-5: {top}", flush=True)

    # --- TVDs -------------------------------------------------------------
    print("\n=== TVDs (lower is closer agreement) ===")
    print(f"  TVD(exact,  aer)     = {_tvd(exact, aer, args.shots):.4f}")
    print(f"  TVD(exact,  bridge)  = {_tvd(exact, bridge, args.shots):.4f}")
    print(f"  TVD(aer,    bridge)  = {_tvd(aer, bridge, args.shots):.4f}")
    hardware_tvd_text = ""
    if hardware:
        tvd_hw_exact = _tvd(hardware, exact, args.shots)
        tvd_hw_aer = _tvd(hardware, aer, args.shots)
        tvd_hw_bridge = _tvd(hardware, bridge, args.shots)
        tvd_aer_bridge = _tvd(aer, bridge, args.shots)
        print(f"  TVD(hardware, exact)  = {tvd_hw_exact:.4f}")
        print(f"  TVD(hardware, aer)    = {tvd_hw_aer:.4f}")
        print(f"  TVD(hardware, bridge) = {tvd_hw_bridge:.4f}")
        hardware_tvd_text = (
            f"TVD(hw, aer) = {tvd_hw_aer:.3f}; "
            f"TVD(hw, bridge) = {tvd_hw_bridge:.3f}; "
            f"TVD(aer, bridge) = {tvd_aer_bridge:.3f}"
        )

    # --- Plot -------------------------------------------------------------
    sources = [
        ("exact (noise-free)", exact, "#1f77b4"),
        ("qiskit-aer", aer, "#2ca02c"),
        ("maestro bridge", bridge, "#d62728"),
    ]
    if hardware:
        sources.append(("ibm_fez (hardware)", hardware, "#9467bd"))
    backend_name = backend.name if use_hardware else type(backend).__name__
    out_path = args.output or os.path.join(
        _PYTHON_DIR, "_plots", "bridge_vs_hardware.png"
    )
    _render_plot(sources, args.shots, backend_name, args.ghz_n,
                 hardware_tvd_text, out_path)
    print(f"\nPlot saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
