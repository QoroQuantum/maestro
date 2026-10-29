"""Check that Maestro on macOS runs with exactly one libomp loaded.

Usage: python check_macos_libomp.py MODE
  alone          torch must not be installed; Maestro must use its bundled libomp
  torch-first    import torch, then maestro; both must use torch's libomp
  maestro-first  import maestro, then torch; both must use torch's libomp

Run "alone" under sandbox-exec with the Homebrew prefixes unreadable to show
the wheel does not need Homebrew's libomp.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

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
    out = subprocess.run(["vmmap", "-w", str(os.getpid())], capture_output=True, text=True, check=True).stdout
    # LLVM's libomp and Intel's libiomp5 both abort on finding a second copy.
    paths = {line.split()[-1] for line in out.splitlines() if re.search(r"/lib(omp|iomp5)\.dylib$", line.rstrip())}
    return sorted({os.path.realpath(p) for p in paths})


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


def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else None
    if mode == "alone":
        assert importlib.util.find_spec("torch") is None, "torch is installed"
        expected = run_maestro() / ".dylibs" / "libomp.dylib"
    elif mode == "torch-first":
        expected = run_torch() / "lib" / "libomp.dylib"
        run_maestro()
    elif mode == "maestro-first":
        run_maestro()
        expected = run_torch() / "lib" / "libomp.dylib"
        # Maestro again, now that torch's OpenMP runtime is running too.
        run_maestro()
    else:
        sys.exit(__doc__)

    loaded = loaded_libomps()
    print(f"{mode}: OpenMP runtimes loaded: {loaded}")
    if loaded != [str(expected.resolve())]:
        sys.exit(f"{mode}: expected only {expected}")


if __name__ == "__main__":
    main()
