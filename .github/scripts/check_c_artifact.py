"""Check that a C/C++ release archive works without any Maestro dependencies installed.

Usage: python check_c_artifact.py ARCHIVE

Run it on a clean image with only Python (no compiler, Boost, OpenBLAS or
OpenMP) to show the bundled libraries are enough. It extracts the archive,
checks the layout, loads lib/libmaestro.so, reads the native capabilities and
runs a seeded Bell circuit through the native request API.
"""

import ctypes
import json
import pathlib
import sys
import tarfile
import tempfile

BELL = (
    "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; "
    "h q[0]; cx q[0],q[1]; measure q -> c;"
)


def call_json(function, *args):
    pointer = function(*args)
    if not pointer:
        raise RuntimeError(f"{function.__name__} returned null")
    try:
        return json.loads(ctypes.string_at(pointer).decode("utf-8"))
    finally:
        library.FreeResult(pointer)


archive = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as scratch:
    with tarfile.open(archive) as tar:
        tar.extractall(scratch, filter="data")
    (root,) = pathlib.Path(scratch).iterdir()

    for required in [
        "include/maestrolib/Interface.h",
        "include/maestrolib/InterfaceTypes.h",
        "lib/libmaestro.so",
        "LICENSE",
        "BUILD_INFO",
    ]:
        if not (root / required).is_file():
            raise SystemExit(f"{archive.name}: missing {required}")
    print((root / "BUILD_INFO").read_text())

    library = ctypes.CDLL(str(root / "lib/libmaestro.so"))
    library.FreeResult.argtypes = [ctypes.c_void_p]
    library.FreeResult.restype = None
    library.MaestroGetCapabilitiesJson.argtypes = []
    library.MaestroGetCapabilitiesJson.restype = ctypes.c_void_p
    library.MaestroRunRequestJson.argtypes = [ctypes.c_char_p]
    library.MaestroRunRequestJson.restype = ctypes.c_void_p

    capabilities = call_json(library.MaestroGetCapabilitiesJson)
    if capabilities.get("ok") is not True:
        raise SystemExit(f"capabilities failed: {capabilities}")
    print("build:", capabilities.get("build"))

    request = {
        "schema_version": 2,
        "operation": "execute",
        "circuit": {"num_qubits": 2, "source": BELL},
        "execution": {"shots": 200, "seed": 7},
    }
    result = call_json(library.MaestroRunRequestJson, json.dumps(request).encode())
    if result.get("ok") is not True:
        raise SystemExit(f"Bell circuit failed: {result}")
    counts = result["counts"]
    # Both outcomes must appear: a library that skipped the gates would give
    # only "00". With 200 shots a real Bell state misses one with p = 2**-199.
    if set(counts) != {"00", "11"} or any(count <= 0 for count in counts.values()) or sum(counts.values()) != 200:
        raise SystemExit(f"unexpected Bell counts: {counts}")
    print("Bell counts:", counts)
    print(f"{archive.name}: OK")
