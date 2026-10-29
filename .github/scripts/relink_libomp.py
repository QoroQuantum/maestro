"""Make a delocated macOS wheel share libomp with torch when both are installed.

delocate bundles libomp into maestro/.dylibs and links it by @loader_path, so
Maestro always loads its own copy. torch ships another, and LLVM's OpenMP
runtime aborts (OMP Error #15) when a second copy initializes in the same
process. This links libomp as @rpath/libomp.dylib from every binary that uses
it, with rpaths that try torch's copy first and the bundled one second, so
exactly one libomp is loaded with or without torch and in either import order.

Usage: python relink_libomp.py WHEEL [WHEEL ...]  (rewrites each wheel in place)
"""

import base64
import csv
import hashlib
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

LIBOMP = "libomp.dylib"
INSTALL_NAME = "@rpath/libomp.dylib"
# Searched before the bundled copy, relative to site-packages.
SHARED_DIRS = ["torch/lib"]
LOAD_COMMANDS = {
    "LC_LOAD_DYLIB",
    "LC_LOAD_WEAK_DYLIB",
    "LC_REEXPORT_DYLIB",
    "LC_LAZY_LOAD_DYLIB",
    "LC_LOAD_UPWARD_DYLIB",
}
MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
    b"\xca\xfe\xba\xbf",
}


def run(*cmd):
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise


def is_macho(path):
    with open(path, "rb") as f:
        return f.read(4) in MACHO_MAGICS


def load_commands(path):
    """(cmd, name or path) for each load command that carries one."""
    commands = []
    cmd = None
    for line in run("otool", "-l", str(path)).splitlines():
        parts = line.split()
        if parts[:1] == ["cmd"]:
            cmd = parts[1]
        elif cmd and parts[:1] in (["name"], ["path"]):
            commands.append((cmd, parts[1]))
            cmd = None
    return commands


def loader_path(target, binary):
    rel = os.path.relpath(target, binary.parent)
    return "@loader_path" if rel == "." else f"@loader_path/{rel}"


def relink(wheel):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel) as zf:
            infos = zf.infolist()
            zf.extractall(root)

        binaries = [p for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink() and is_macho(p)]
        bundled = [p for p in binaries if p.name == LIBOMP]
        if len(bundled) != 1:
            sys.exit(f"{wheel.name}: expected one bundled {LIBOMP}, found {len(bundled)}")
        bundled = bundled[0]

        changed = [bundled]
        run("install_name_tool", "-id", INSTALL_NAME, str(bundled))
        for binary in binaries:
            commands = load_commands(binary)
            refs = [name for cmd, name in commands if cmd in LOAD_COMMANDS and Path(name).name == LIBOMP]
            if not refs:
                continue
            rel = binary.relative_to(root)
            if rel.parts[0].endswith(".data"):
                sys.exit(f"{wheel.name}: {rel} uses libomp but installs outside site-packages")

            # Put ours ahead of any existing rpaths so nothing else is searched first.
            old = [path for cmd, path in commands if cmd == "LC_RPATH"]
            new = [loader_path(root / d, binary) for d in SHARED_DIRS] + [loader_path(bundled.parent, binary)]
            args = []
            for ref in refs:
                if ref != INSTALL_NAME:
                    args += ["-change", ref, INSTALL_NAME]
            for path in old:
                args += ["-delete_rpath", path]
            if args:
                run("install_name_tool", *args, str(binary))
            args = []
            for path in new + [p for p in old if p not in new]:
                args += ["-add_rpath", path]
            run("install_name_tool", *args, str(binary))
            changed.append(binary)
            print(f"{rel}: {', '.join(refs)} -> {INSTALL_NAME}, rpaths {new}")

        for binary in changed:
            run("codesign", "--force", "--sign", "-", str(binary))

        repack(wheel, root, infos)


def repack(wheel, root, infos):
    record = next(i.filename for i in infos if i.filename.endswith(".dist-info/RECORD"))
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    out = wheel.with_suffix(".tmp")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for info in infos:
            if info.is_dir() or info.filename == record:
                continue
            data = (root / info.filename).read_bytes()
            zf.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([info.filename, f"sha256={digest}", len(data)])
        writer.writerow([record, "", ""])
        zf.writestr(next(i for i in infos if i.filename == record), rows.getvalue())
    out.replace(wheel)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        relink(Path(arg))
