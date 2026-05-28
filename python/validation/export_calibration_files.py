#!/usr/bin/env python3
"""Export calibration json files for IBM Fez and IQM Garnet.

Uses fake backends unless live tokens are present in the environment.

    IBM_QUANTUM_TOKEN   — optional; enables live IBM backend
    IQM_TOKEN           — optional; enables live IQM backend

Output files are written to the same directory as this script.
"""

from __future__ import annotations

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PYTHON_DIR)

_BUILD_GLOB = next(
    (p for p in glob.glob(os.path.join(_REPO, "build", "cp3*"))
     if os.path.isdir(p)),
    None,
)
for _p in (_BUILD_GLOB, _PYTHON_DIR):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import noise_bridge as br  # noqa: E402


def export_ibm_fez() -> None:
    ibm_token = os.environ.get("IBM_QUANTUM_TOKEN")
    if ibm_token:
        from qiskit_ibm_runtime import QiskitRuntimeService
        print("IBM: authenticating with live backend...", flush=True)
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
        backend = svc.backend("ibm_fez")
        tag = "live"
    else:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend = FakeFez()
        tag = "fake"

    print(f"IBM: reading calibration from {type(backend).__name__} "
          f"({backend.num_qubits} qubits)...", flush=True)
    out = os.path.join(_HERE, f"ibm_fez_{tag}.json")
    br.save_calibration_file(backend, out)
    print(f"IBM: saved → {out}")


def export_iqm_garnet() -> None:
    iqm_token = os.environ.get("IQM_TOKEN")
    if iqm_token:
        try:
            from iqm.qiskit_iqm import IQMProvider
        except ImportError:
            raise SystemExit("IQM_TOKEN set but iqm-client[qiskit] not installed. "
                             "Run: uv add 'iqm-client[qiskit]'")
        print("IQM: connecting to live garnet backend...", flush=True)
        backend = IQMProvider(
            url="https://cocos.resonance.meetiqm.com/garnet",
            token=iqm_token,
        ).get_backend()
        tag = "live"
    else:
        import importlib
        name = "IQMFakeGarnet"
        backend = None
        suffix = name.removeprefix("IQMFake").lower()
        candidates = [
            "iqm.qiskit_iqm.fake_backends",
            "iqm.qiskit_iqm",
            f"iqm.qiskit_iqm.fake_backends.fake_{suffix}",
        ]
        for mod_path in candidates:
            try:
                mod = importlib.import_module(mod_path)
                if hasattr(mod, name):
                    backend = getattr(mod, name)()
                    break
            except ImportError:
                continue
        if backend is None:
            raise SystemExit(f"Could not import {name}. "
                             "Run: uv add 'iqm-client[qiskit]'")
        tag = "fake"

    print(f"IQM: reading calibration from {type(backend).__name__} "
          f"({backend.num_qubits} qubits)...", flush=True)
    out = os.path.join(_HERE, f"iqm_garnet_{tag}.json")
    br.save_calibration_file(backend, out)
    print(f"IQM: saved → {out}")


if __name__ == "__main__":
    export_ibm_fez()
    export_iqm_garnet()
    print("\nDone. Reload either file with:")
    print("  backend = br.load_calibration_file('<path>.json')")