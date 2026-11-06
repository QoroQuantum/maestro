"""Generate or check seeded fixtures using an independently built old source.

The baseline source and native binaries must already exist. Use the Python
interpreter that built those bindings. This helper neither builds them nor
imports the working-tree bindings. It reads case descriptions
from the fixture, executes those descriptions through the baseline public API,
and obtains fresh counts. Existing expected counts are used only in check mode.

Example, after building the archived 0.3.13 source separately::

    python -B tests/generate_sampling_legacy_fixture.py \
        --source-root build/sampling-baseline/source \
        --build-root build/sampling-baseline/native \
        --archive build/sampling-baseline/source-head.tar --check-existing

To generate a candidate for review, replace --check-existing with --output and
a new JSON path. Output files are created exclusively, never overwritten.
Use --extension-dir and --dll-dir for differently staged native artifacts.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import tarfile


DEFAULT_CASES = Path(__file__).resolve().parent / "python/data/sampling_counts_windows_0_3_13.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(directory, extensions=None):
    directory = Path(directory)
    digest = hashlib.sha256()
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts
         and (extensions is None or path.suffix in extensions)),
        key=lambda path: path.relative_to(directory).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return {"sha256": digest.hexdigest(), "files": len(files)}


def read_cache(build_root):
    values = {}
    for line in (build_root / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def verify_archive(archive_path, source_root):
    """Check source bytes against the archive without extracting any files."""
    with tarfile.open(archive_path, "r:*") as archive:
        commit = archive.pax_headers.get("comment", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("The archive must contain git archive's embedded commit ID")
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError(f"Unsupported archive entry: {member.name}")
            source = source_root.joinpath(*name.parts).resolve()
            if not source.is_relative_to(source_root):
                raise ValueError(f"Archive entry resolves outside the source: {member.name}")
            with archive.extractfile(member) as stream:
                expected = hashlib.sha256(stream.read()).hexdigest()
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"Baseline source differs from the archive: {member.name}")
    return commit


def load_baseline(source_root, extension_dir, dll_dirs):
    if "maestro" in sys.modules:
        raise RuntimeError("Run the helper in a fresh Python process; maestro is already imported")
    extensions = [path for path in extension_dir.glob("maestro*")
                  if any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)]
    if len(extensions) != 1:
        raise ValueError(f"Expected one maestro extension in {extension_dir}, found {len(extensions)}")
    handles = []
    if hasattr(os, "add_dll_directory"):
        for directory in dict.fromkeys(dll_dirs):
            if not directory.is_dir():
                raise ValueError(f"DLL directory does not exist: {directory}")
            handles.append(os.add_dll_directory(str(directory)))
    spec = importlib.util.spec_from_file_location(
        "maestro", source_root / "maestro/__init__.py",
        submodule_search_locations=[str(source_root / "maestro"), str(extension_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["maestro"] = module
    spec.loader.exec_module(module)
    if Path(module._raw_maestro.__file__).resolve() != extensions[0].resolve():
        raise RuntimeError("The selected baseline extension was not loaded")
    if hasattr(module.SimulatorConfig(), "counts_policy") or hasattr(module, "_compile_reference_sampler"):
        raise RuntimeError("The selected native module contains the new sampling APIs, not the pre-change baseline")
    return module, handles, extensions[0]


def make_circuit(module, description):
    if "qasm" in description:
        return module.QasmToCirc().parse_and_translate(description["qasm"])
    circuit = module.circuits.QuantumCircuit()
    for name, arguments in description["operations"]:
        getattr(circuit, name)(*arguments)
    return circuit


def sample_case(module, case):
    options = dict(case["config"])
    options["simulation_type"] = getattr(module.SimulationType, options["simulation_type"])
    config = module.SimulatorConfig(**options)
    circuit = make_circuit(module, case["circuit"])
    noise = module.NoiseModel()
    for name, arguments in case.get("noise", []):
        getattr(noise, name)(*arguments)
    arguments = dict(case["arguments"])
    if case["api"] == "execute":
        result = circuit.execute(config=config, **arguments)
    elif case["api"] == "execute_suffix":
        checkpoint = module.PrefixCheckpointedSimulator(make_circuit(module, case["prefix"]), case["num_qubits"], config)
        result = checkpoint.execute_suffix(circuit, noise_model=noise, **arguments)
    else:
        result = getattr(circuit, case["api"])(noise, config=config, **arguments)
    return dict(sorted((str(bits), int(count)) for bits, count in result["counts"].items()))


def provenance(args, cache, commit, cases, extension):
    compiler = {}
    for path in sorted((args.build_root / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")):
        text = path.read_text(encoding="utf-8")
        for field in ["ID", "VERSION"]:
            match = re.search(rf'set\(CMAKE_CXX_COMPILER_{field} "([^\"]+)"\)', text)
            if match:
                compiler[field.lower()] = match.group(1)
    dependencies = {}
    qcsim = Path(cache["QCSIM_INCLUDE_DIR"])
    headers = tree_hash(qcsim, {".h", ".hpp"})
    dependencies["qcsim"] = {
        "path": str(qcsim), "headers_sha256": headers["sha256"], "header_count": headers["files"],
        "revision_note": "Local headers were hashed; no dependency git revision is asserted.",
    }
    for name, key, header in [("eigen", "EIGEN5_INCLUDE_DIR", "Eigen/Core"),
                              ("boost", "BOOST_ROOT", "boost/version.hpp")]:
        if key in cache:
            root = Path(cache[key])
            dependencies[name] = {"path": str(root), "version_header_sha256": sha256(root / header)}
    if "FETCHCONTENT_SOURCE_DIR_NANOBIND" in cache:
        directory = Path(cache["FETCHCONTENT_SOURCE_DIR_NANOBIND"])
        header_hash = tree_hash(directory / "include", {".h", ".hpp"})
        dependencies["nanobind"] = {"path": str(directory), "headers_sha256": header_hash["sha256"]}
    source = tree_hash(args.source_root)
    version = re.search(r'^version\s*=\s*"([^\"]+)"',
                        (args.source_root / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE).group(1)
    libraries = [args.build_root / "bin/maestro.dll", args.build_root / "bin/libmaestro.so",
                 args.build_root / "libmaestro.so", args.build_root / "bin/libmaestro.dylib",
                 args.build_root / "libmaestro.dylib"]
    artifacts = {"extension_sha256": sha256(extension)}
    library = next((path for path in libraries if path.is_file()), None)
    if library:
        artifacts["maestro_library_sha256"] = sha256(library)
    return {
        "source_kind": "git archive HEAD; fresh isolated native build",
        "scope": "Pre-change Maestro source compiled against recorded local dependencies; not a published wheel.",
        "commit": commit, "version": version, "archive_sha256": sha256(args.archive),
        "source_tree_sha256": source["sha256"], "source_file_count": source["files"],
        "case_specifications_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
        "compiler": compiler, "generator": cache.get("CMAKE_GENERATOR"),
        "configured_source_root": cache.get("CMAKE_HOME_DIRECTORY"),
        "build": {"openmp": cache.get("MAESTRO_ENABLE_OPENMP"), "omp_num_threads": 1,
                  "aer": cache.get("ENABLE_QISKIT_AER"), "tests": cache.get("COMPILE_TESTS"),
                  "python_bindings": cache.get("BUILD_PYTHON_BINDINGS")},
        "dependencies": dependencies, "artifacts": artifacts,
        "fixture_generator_sha256": sha256(__file__),
        "generator_arguments": {"source_root": str(args.source_root), "build_root": str(args.build_root),
                                "extension_dir": str(args.extension_dir), "archive": str(args.archive),
                                "case_file": str(args.cases), "dll_dirs": [str(path) for path in args.dll_dir]},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True, help="Unmodified extracted baseline source")
    parser.add_argument("--build-root", type=Path, required=True, help="Configured baseline native build containing CMakeCache.txt")
    parser.add_argument("--archive", type=Path, required=True, help="git archive tar containing that baseline source")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Fixture containing public-API case descriptions")
    parser.add_argument("--extension-dir", type=Path, help="Directory containing the baseline extension; defaults to BUILD_ROOT/Release")
    parser.add_argument("--dll-dir", type=Path, action="append", default=[], help="Additional dependency DLL directory; repeatable")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-existing", action="store_true", help="Compare baseline results to --cases without writing files")
    action.add_argument("--output", type=Path, help="New candidate JSON file; must not already exist")
    args = parser.parse_args(argv)
    for name in ["source_root", "build_root", "archive", "cases"]:
        setattr(args, name, getattr(args, name).resolve())
    args.extension_dir = (args.extension_dir or args.build_root / "Release").resolve()
    args.dll_dir = [path.resolve() for path in args.dll_dir]
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite an existing output: {args.output}")
    reference = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = [{key: value for key, value in case.items() if key != "counts"} for case in reference["cases"]]
    commit = verify_archive(args.archive, args.source_root)
    if commit != reference["provenance"]["commit"]:
        parser.error("Archive commit does not match the case fixture's baseline commit")
    cache = read_cache(args.build_root)
    dll_dirs = [args.build_root / "bin"]
    if "Boost_LIBRARY_DIR_RELEASE" in cache:
        dll_dirs.append(Path(cache["Boost_LIBRARY_DIR_RELEASE"]))
    dll_dirs.extend(args.dll_dir)
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.dont_write_bytecode = True
    module, dll_handles, extension = load_baseline(args.source_root, args.extension_dir, dll_dirs)
    outputs = []
    for case, expected in zip(cases, reference["cases"]):
        counts = sample_case(module, case)
        if counts != sample_case(module, case):
            raise AssertionError(f"Baseline seed did not replay: {case['id']}")
        if sum(counts.values()) != case["arguments"]["shots"]:
            raise AssertionError(f"Incomplete baseline histogram: {case['id']}")
        if args.check_existing and counts != expected["counts"]:
            raise AssertionError(f"Baseline differs from existing fixture: {case['id']}\n{counts}\n{expected['counts']}")
        outputs.append({**case, "counts": counts})
        print(case["id"], "matches" if args.check_existing else "sampled", flush=True)
    if args.output is not None:
        result = {"provenance": provenance(args, cache, commit, cases, extension), "cases": outputs}
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
        print("Wrote candidate", args.output, flush=True)
    else:
        print(f"Verified {len(outputs)} baseline cases; no files written.", flush=True)
    # Keep Windows DLL directory handles alive until all native work completes.
    assert dll_handles is not None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
