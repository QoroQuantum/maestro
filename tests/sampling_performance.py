"""Manual end-to-end noisy sampling benchmark; not part of pytest.

Run against the intended installed/staged Maestro package, for example:
    python tests/sampling_performance.py --threads 1 --workers 1 --repeats 3
    python tests/sampling_performance.py --distances 3 5 7 --shot-counts 1000 100000 --skip-mps

Optional dependencies: stim and sinter (qoro-maestro[sinter]). No package/path
rewriting is performed. Set PYTHONPATH and native-library search paths before
starting Python when benchmarking a local build.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--shot-counts", type=int, nargs="+", help="surface-code grid; overrides --shots")
    parser.add_argument("--distances", type=int, nargs="+", default=[3], help="odd code distances; rounds equal distance")
    parser.add_argument("--include-legacy-large", action="store_true", help="also run slow legacy/checkpoint cases above 1000 shots")
    parser.add_argument("--mps-shots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--probability", type=float, default=0.001)
    parser.add_argument("--coherent-probability", type=float, default=0.001)
    parser.add_argument("--skip-mps", action="store_true")
    args = parser.parse_args()
    for name in ("threads", "workers", "repeats", "shots", "mps_shots"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0 or not 0 <= args.seed < 2**32:
        parser.error("warmups must be nonnegative and seed must fit in 32 bits")
    if not 0 < args.probability <= 1 or not 0 < args.coherent_probability <= 1:
        parser.error("noise probabilities must be in (0, 1]")
    args.shot_counts = args.shot_counts or [args.shots]
    if any(shots < 1 for shots in args.shot_counts):
        parser.error("shot counts must be positive")
    if any(distance < 3 or distance % 2 != 1 for distance in args.distances):
        parser.error("distances must be odd integers of at least three")
    return args


def compact_qubits(stim, circuit):
    """Remove unused index slots without changing operations or record order."""
    instructions = list(circuit.flattened())
    used = sorted({target.value for instruction in instructions
                   for target in instruction.targets_copy() if target.is_qubit_target})
    mapping = {qubit: index for index, qubit in enumerate(used)}
    compact = stim.Circuit()
    for instruction in instructions:
        targets = [mapping[target.value] if target.is_qubit_target else target
                   for target in instruction.targets_copy()]
        compact.append(instruction.name, targets, instruction.gate_args_copy())
    return compact


def surface_code(stim, distance, probability=0):
    return compact_qubits(stim, stim.Circuit.generated("surface_code:rotated_memory_z",
        distance=distance, rounds=distance, after_clifford_depolarization=probability,
        before_round_data_depolarization=probability, after_reset_flip_probability=probability,
        before_measure_flip_probability=probability))


def describe_result(result, shots, expected_strategy=None):
    if isinstance(result, dict):
        summary = {}
        if "counts" in result:
            if sum(result["counts"].values()) != shots:
                raise RuntimeError("incomplete count histogram")
            summary["format"] = "counts"
        elif "samples" in result:
            samples = result["samples"]
            if samples.ndim != 2 or samples.shape[0] != shots:
                raise RuntimeError("sample matrix has the wrong shape")
            summary.update(format="samples", width=int(samples.shape[1]))
        else:
            raise RuntimeError("execution returned neither counts nor samples")
        if "sampling_strategy" in result:
            summary["strategy"] = result["sampling_strategy"]
        if expected_strategy and summary.get("strategy") != expected_strategy:
            raise RuntimeError(f"expected {expected_strategy}, got {summary.get('strategy')}")
        return summary
    if result.ndim != 2 or result.shape[0] != shots:
        raise RuntimeError("measurement array has the wrong shape")
    return {"format": "measurements", "width": int(result.shape[1])}


def measure(args, results, name, call, shots, *, expected_strategy=None, **details):
    try:
        for _ in range(args.warmups):
            describe_result(call(), shots, expected_strategy)
        durations = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = call()
            durations.append(time.perf_counter() - start)
            summary = describe_result(result, shots, expected_strategy)
        median = statistics.median(durations)
        results.append({"name": name, "status": "ok", "shots": shots,
                        "median_seconds": median, "microseconds_per_shot": median * 1e6 / shots,
                        "seconds": durations, **summary, **details})
    except Exception as error:
        results.append({"name": name, "status": "error", "error": f"{type(error).__name__}: {error}", **details})


def surface_benchmark(args, maestro, helpers, make_config, results, distance, shots):
    stim, MaestroCompiledSampler, split_prefix, translate = helpers
    noisy = surface_code(stim, distance, args.probability)
    qc, _, measurements = translate(noisy)
    model = maestro.NoiseModel()
    legacy = make_config(maestro.SimulationType.Stabilizer, counts_policy="legacy")
    frames = make_config(maestro.SimulationType.Stabilizer, counts_policy="frame_v1")
    automatic = make_config(maestro.SimulationType.Stabilizer, counts_policy="auto", seed=None)
    details = {"workload": f"surface_code_rotated_memory_z_d{distance}_r{distance}",
               "distance": distance, "rounds": distance, "qubits": noisy.num_qubits,
               "measurements": measurements, "probability": args.probability,
               "backend": "QCSim", "method": "Stabilizer", "noise_realizations": shots}
    if shots <= 1000 or args.include_legacy_large:
        measure(args, results, "maestro_counts_legacy", lambda: qc.full_noise_execute(model,
            config=legacy, shots=shots, noise_realizations=shots, noise_seed=args.seed), shots,
            counts_policy="legacy", seed=args.seed, noise_seed=args.seed, **details)
    else:
        results.append({"name": "maestro_counts_legacy", "status": "skipped", "shots": shots,
                        "reason": "pass --include-legacy-large to run slow trajectory cases", **details})
    for policy, cfg, seed in (("frame_v1", frames, args.seed), ("auto", automatic, None)):
        measure(args, results, f"maestro_counts_{policy}", lambda cfg=cfg, seed=seed:
            qc.full_noise_execute(model, config=cfg, shots=shots, noise_realizations=shots, noise_seed=seed), shots,
            expected_strategy="reference_frame_v1", counts_policy=policy, seed=seed, noise_seed=seed, **details)
    measure(args, results, "maestro_raw_frame", lambda: qc.full_noise_execute(model,
        config=frames, shots=shots, noise_realizations=shots, noise_seed=args.seed, return_samples=True), shots,
        expected_strategy="reference_frame_v1", seed=args.seed, noise_seed=args.seed, preparation="included", **details)

    plan = maestro._compile_reference_sampler(qc, model, frames)
    if plan is None:
        raise RuntimeError("generated Clifford circuit did not compile a frame plan")
    measure(args, results, "maestro_precompiled_frame_sample", lambda: plan.sample(shots, args.seed), shots,
        expected_strategy="reference_frame_v1", seed=args.seed, preparation="excluded", **details)
    compiled = MaestroCompiledSampler(noisy, config=frames, seed=args.seed, enable_checkpoint=False)
    measure(args, results, "maestro_sinter_raw", lambda: compiled.sample_measurements(shots, seed=args.seed), shots,
        seed=args.seed, preparation="excluded", **details)

    if shots <= 1000 or args.include_legacy_large:
        prefix, suffix, _ = split_prefix(noisy)
        prefix_qc, _, _ = translate(prefix)
        suffix_qc, _, _ = translate(suffix)
        checkpoint = maestro.PrefixCheckpointedSimulator(prefix_qc, noisy.num_qubits, legacy)
        for raw in (False, True):
            measure(args, results, "checkpoint_raw_trajectory" if raw else "checkpoint_counts",
                lambda raw=raw: checkpoint.execute_suffix(suffix_qc, shots=shots,
                    noise_model=model, noise_realizations=shots, noise_seed=args.seed,
                    num_measurements=measurements, return_samples=raw), shots,
                seed=args.seed, noise_seed=args.seed, **details)
    else:
        results.append({"name": "checkpoint_trajectory_cases", "status": "skipped", "shots": shots,
                        "reason": "pass --include-legacy-large to run slow trajectory cases", **details})

    stim_sampler = noisy.compile_sampler(seed=args.seed)
    stim_details = {key: details[key] for key in ("workload", "distance", "rounds", "qubits", "measurements", "probability")}
    measure(args, results, "stim_precompiled_sample", lambda: stim_sampler.sample(shots), shots,
        backend="Stim", seed=args.seed, preparation="excluded", **stim_details)
    measure(args, results, "stim_compile_and_sample", lambda: noisy.compile_sampler(seed=args.seed).sample(shots), shots,
        backend="Stim", seed=args.seed, preparation="included", **stim_details)


def main():
    args = arguments()
    # OpenMP reads these on runtime initialization; set them before NumPy/Maestro.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_THREAD_LIMIT"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    import maestro

    results = []
    report = {
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "maestro_path": maestro.__file__, "maestro_version": getattr(maestro, "__version__", "unknown"),
                        "maestro_native_path": maestro._raw_maestro.__file__,
                        "omp_num_threads": args.threads, "omp_thread_limit": args.threads,
                        "max_simulators": args.workers, "warmups": args.warmups, "repeats": args.repeats,
                        "seed": args.seed, "clock": "perf_counter", "statistic": "median"},
        "grid": {"distances": args.distances, "shot_counts": args.shot_counts,
                 "include_legacy_large": args.include_legacy_large},
        "scope": "Current count/raw paths, not a historical-library baseline. Translation, Sinter construction and checkpoint preparation excluded. Direct execution includes per-call preparation; compiled-frame and Sinter rows exclude compilation. Stim reports compiled and compile-plus-sample cases.",
        "results": results,
    }
    helpers = None
    try:
        import stim
        from maestro.sinter import MaestroCompiledSampler, split_stim_prefix_suffix, translate_stim_to_maestro
        helpers = (stim, MaestroCompiledSampler, split_stim_prefix_suffix, translate_stim_to_maestro)
        report["environment"]["stim_version"] = stim.__version__
    except ImportError as error:
        results.append({"name": "surface_code_suite", "status": "skipped", "reason": str(error)})

    def config(method, *, seed=args.seed, counts_policy="legacy", **kwargs):
        return maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=method, seed=seed, counts_policy=counts_policy, max_simulators=args.workers,
            trajectory_max_memory_mb=1024, auto_reduce=False, **kwargs)

    clean = None
    if helpers:
        clean = surface_code(helpers[0], 3)
        for distance in args.distances:
            for shots in args.shot_counts:
                try:
                    surface_benchmark(args, maestro, helpers, config, results, distance, shots)
                except Exception as error:
                    results.append({"name": "surface_setup", "status": "error", "distance": distance,
                                    "shots": shots, "error": f"{type(error).__name__}: {error}"})

    if not args.skip_mps:
        if clean is not None:
            qc, _, measurements = helpers[3](clean)
            qubits = clean.num_qubits
            workload = "surface_code_rotated_memory_z_d3_r3_coherent"
        else:
            # Still provide a useful native-only benchmark if optional QEC
            # dependencies are absent. This is explicitly a different circuit.
            qc = maestro.circuits.QuantumCircuit()
            qubits = 17
            for q in range(qubits):
                qc.h(q)
            for _ in range(3):
                for q in range(qubits - 1):
                    qc.cx(q, q + 1)
                for q in range(qubits):
                    qc.h(q)
            qc.measure_all()
            measurements = qubits
            workload = "native_17_qubit_chain_3_layers_coherent_optional_qec_unavailable"
        cfg = config(maestro.SimulationType.MatrixProductState, max_bond_dimension=16)
        model = maestro.NoiseModel()
        model.set_all_coherent_depolarizing(qubits, args.coherent_probability)
        for realizations, raw, name in ((1, False, "mps_coherent_shared_counts"),
                                       (args.mps_shots, False, "mps_coherent_independent_counts"),
                                       (args.mps_shots, True, "mps_coherent_independent_raw")):
            measure(args, results, name,
                lambda realizations=realizations, raw=raw: qc.full_noise_execute(model, config=cfg,
                    shots=args.mps_shots, noise_realizations=realizations,
                    noise_seed=args.seed, return_samples=raw), args.mps_shots,
                expected_strategy="trajectory_v1" if raw else None,
                workload=workload, qubits=qubits, measurements=measurements, backend="QCSim",
                method="MatrixProductState", max_bond_dimension=16,
                coherent_probability=args.coherent_probability, noise_realizations=realizations,
                semantics="shared_coherent_sign_pattern" if realizations == 1 else "independent_coherent_sign_pattern_per_shot")

    print(json.dumps(report, separators=(",", ":")))
    return int(any(result["status"] == "error" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
