"""Check that Maestro on macOS loads one libomp and runs next to torch and scikit-learn.

Usage: python check_macos_libomp.py PACKAGE [PACKAGE ...]

Imports and runs maestro, torch or sklearn in the order given. Maestro must use
torch's libomp when torch is installed and its bundled copy otherwise, and no
other libomp may be loaded except scikit-learn's own, which runs alongside
Maestro's without aborting.

Run "maestro" alone, before torch is installed, under sandbox-exec with the
Homebrew prefixes unreadable to show the wheel does not need Homebrew's libomp.
"""

import ctypes
import importlib.util
import os
import re
import sys
from pathlib import Path

# scikit-learn sets KMP_DUPLICATE_LIB_OK=True on import, which would hide the
# abort from a second libomp that detects the first.
os.environ["KMP_DUPLICATE_LIB_OK"] = "FALSE"

QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
"""
for _layer in range(4):
    QASM += "".join(f"h q[{i}];\nrx({0.1 * (i + 1)}) q[{i}];\n" for i in range(16))
    QASM += "".join(f"cx q[{i}], q[{i + 1}];\n" for i in range(15))
OBSERVABLES = ";".join(["Z" * 16, "X" * 16, "ZZ" + "I" * 14])


def loaded_libomps():
    # Asks dyld directly: vmmap masks image paths under sandbox-exec.
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    paths = (dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count()))
    # LLVM's libomp and Intel's libiomp5 both abort on finding a second copy.
    return sorted({os.path.realpath(p) for p in paths if re.search(r"/lib(omp|iomp5)\.dylib$", p)})


def run_maestro():
    import maestro

    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=32,
    )
    counts = maestro.simple_execute(QASM + "measure q -> c;\n", shots=2000, config=config)["counts"]
    assert sum(counts.values()) == 2000, counts
    estimate = maestro.simple_estimate(QASM, OBSERVABLES, config=config)
    assert estimate is not None
    return Path(maestro.__file__).resolve().parent


def run_torch():
    import torch

    assert "OpenMP" in torch.__config__.parallel_info(), torch.__config__.parallel_info()
    x = torch.randn(4_000_000)
    assert torch.isfinite(x.exp().sum())
    return Path(torch.__file__).resolve().parent


def run_sklearn():
    import numpy as np
    import sklearn
    from sklearn.cluster import KMeans

    # Lloyd's iterations run in OpenMP parallel loops.
    X = np.random.default_rng(0).normal(size=(50_000, 8))
    KMeans(n_clusters=8, n_init=1, random_state=0).fit(X)
    return Path(sklearn.__file__).resolve().parent


RUNS = {"maestro": run_maestro, "torch": run_torch, "sklearn": run_sklearn}


def main():
    order = sys.argv[1:]
    if "maestro" not in order or any(package not in RUNS for package in order):
        sys.exit(__doc__)
    dirs = {}
    for package in order:
        dirs[package] = RUNS[package]()

    torch = importlib.util.find_spec("torch")
    if torch:
        expected = Path(torch.origin).resolve().parent / "lib" / "libomp.dylib"
    else:
        expected = dirs["maestro"] / ".dylibs" / "libomp.dylib"
    expected = str(expected.resolve())

    loaded = loaded_libomps()
    sklearn = dirs.get("sklearn")
    others = [p for p in loaded if p != expected and not (sklearn and Path(p).is_relative_to(sklearn))]
    print(f"{' -> '.join(order)}: OpenMP runtimes loaded: {loaded}")
    if expected not in loaded or others:
        sys.exit(f"{' -> '.join(order)}: expected Maestro to use {expected} and no other libomp but scikit-learn's")


if __name__ == "__main__":
    main()
