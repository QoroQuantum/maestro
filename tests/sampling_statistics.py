"""Compare generated surface-code frame samples with independent Stim samples.

Examples (optional dependencies: qoro-maestro[sinter]):
    python tests/sampling_statistics.py --distances 3 5 --shots 400000
    python tests/sampling_statistics.py --distances 5 --shots 2000000 --metrics logical

Set PYTHONPATH/native-library paths before starting Python for a local build.
This script does not alter import paths or write artifacts. JSON includes the
counts needed to recompute every comparison; maximum z scores are diagnostics,
not a multiple-testing-adjusted proof of equal joint distributions.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--shots", type=int, default=400000)
    parser.add_argument("--batch-shots", type=int, default=20000)
    parser.add_argument("--probability", type=float, default=0.003)
    parser.add_argument("--maestro-seed", type=int, default=12345)
    parser.add_argument("--stim-seeds", type=int, nargs="+", default=[54321, 98765])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--metrics", choices=["detectors", "pairs", "logical"], nargs="+", default=["detectors", "pairs", "logical"])
    parser.add_argument("--pair-limit", type=int, default=0, help="deterministic subset of detector pairs; zero means all")
    args = parser.parse_args()
    if any(distance < 3 or distance % 2 != 1 for distance in args.distances):
        parser.error("distances must be odd integers of at least three")
    if min(args.shots, args.batch_shots, args.threads) < 1 or args.pair_limit < 0:
        parser.error("shots, batch-shots and threads must be positive; pair-limit must be nonnegative")
    if not 0 <= args.probability <= 1:
        parser.error("probability must be in [0, 1]")
    if any(not 0 <= seed < 2**64 for seed in [args.maestro_seed, *args.stim_seeds]):
        parser.error("seeds must fit in unsigned 64 bits")
    if len(set(args.stim_seeds)) != len(args.stim_seeds):
        parser.error("Stim seeds must be distinct for independent comparison runs")
    return args


def surface_code(stim, distance, probability):
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=distance,
        after_clifford_depolarization=probability, before_round_data_depolarization=probability,
        after_reset_flip_probability=probability, before_measure_flip_probability=probability)
    instructions = list(circuit.flattened())
    used = sorted({target.value for instruction in instructions
                   for target in instruction.targets_copy() if target.is_qubit_target})
    mapping = {qubit: index for index, qubit in enumerate(used)}
    compact = stim.Circuit()
    for instruction in instructions:
        compact.append(instruction.name,
            [mapping[target.value] if target.is_qubit_target else target for target in instruction.targets_copy()],
            instruction.gate_args_copy())
    return compact


def call_seed(seed, batch):
    """Mix outer call identities before the sampler derives internal batches."""
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15 * (batch + 1)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def collect(np, args, circuit, pairs, sample, name, seed):
    detector_counts = np.zeros(circuit.num_detectors, dtype=np.int64)
    pair_counts = np.zeros(len(pairs), dtype=np.int64)
    logical_counts = np.zeros(circuit.num_observables, dtype=np.int64)
    any_logical = 0
    start = time.perf_counter()
    for first in range(0, args.shots, args.batch_shots):
        count = min(args.batch_shots, args.shots - first)
        detectors, observables = sample(count, first // args.batch_shots)
        if detectors.shape != (count, circuit.num_detectors) or observables.shape != (count, circuit.num_observables):
            raise RuntimeError("sampler returned unexpected detector/observable dimensions")
        if "detectors" in args.metrics:
            detector_counts += np.count_nonzero(detectors, axis=0)
        if "pairs" in args.metrics and len(pairs):
            # BLAS sums products of zeros and ones exactly at these shot budgets.
            # Avoid an expensive integer matrix product or a shots x pairs array.
            matrix = np.asarray(detectors, dtype=np.float64)
            coincidences = matrix.T @ matrix
            pair_counts += coincidences[pairs[:, 0], pairs[:, 1]].astype(np.int64)
        if "logical" in args.metrics:
            logical_counts += np.count_nonzero(observables, axis=0)
            any_logical += int(np.count_nonzero(np.any(observables, axis=1)))
    result = {"sampler": name, "seed": seed, "shots": args.shots,
              "seconds": time.perf_counter() - start}
    if "detectors" in args.metrics:
        result["detector_counts"] = detector_counts.tolist()
    if "pairs" in args.metrics:
        result["pair_counts"] = pair_counts.tolist()
    if "logical" in args.metrics:
        result["observable_counts"] = logical_counts.tolist()
        result["any_observable_flip_count"] = any_logical
    return result


def difference(np, left, right, field, labels):
    a, b = np.asarray(left[field], dtype=np.float64), np.asarray(right[field], dtype=np.float64)
    if not a.size:
        return {"comparisons": 0, "max_absolute_z": 0, "max_absolute_rate_difference": 0}
    rate_a, rate_b = a / left["shots"], b / right["shots"]
    pooled = (a + b) / (left["shots"] + right["shots"])
    variance = pooled * (1 - pooled) * (1 / left["shots"] + 1 / right["shots"])
    z = np.divide(rate_a - rate_b, np.sqrt(variance), out=np.zeros_like(pooled), where=variance > 0)
    worst = int(np.argmax(np.abs(z)))
    return {"comparisons": len(a), "max_absolute_z": float(abs(z[worst])),
            "max_absolute_rate_difference": float(np.max(np.abs(rate_a - rate_b))),
            "worst": {"index": labels[worst], "left_count": int(a[worst]), "right_count": int(b[worst]),
                      "left_rate": float(rate_a[worst]), "right_rate": float(rate_b[worst]), "z": float(z[worst])}}


def main():
    args = arguments()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_THREAD_LIMIT"] = str(args.threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    # Keep BLAS pair-count accumulation within the same requested CPU budget.
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    report = {"configuration": vars(args), "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
              "method": "Independent binomial rate comparisons with pooled two-sample standard errors; maxima are not corrected for multiple comparisons. Logical rates are undecoded observable flips.",
              "maestro_call_seed_scheme": "splitmix64_finalizer_v1: x=(base_seed+0x9e3779b97f4a7c15*(call_index+1)) mod 2^64; x=(x xor (x>>30))*0xbf58476d1ce4e5b9 mod 2^64; x=(x xor (x>>27))*0x94d049bb133111eb mod 2^64; seed=x xor (x>>31)",
              "workloads": []}
    try:
        import numpy as np
        import stim
        import maestro
        from maestro.sinter import translate_stim_to_maestro
    except ImportError as error:
        report["status"] = "skipped"
        report["reason"] = str(error)
        print(json.dumps(report, separators=(",", ":")))
        return 0
    report["environment"].update(stim_version=stim.__version__, maestro_path=maestro.__file__,
                                 maestro_native_path=maestro._raw_maestro.__file__, numpy_version=np.__version__)
    for distance in args.distances:
        workload = {"distance": distance, "rounds": distance, "datasets": [], "comparisons": []}
        report["workloads"].append(workload)
        try:
            circuit = surface_code(stim, distance, args.probability)
            workload.update(qubits=circuit.num_qubits, measurements=circuit.num_measurements,
                            detectors=circuit.num_detectors, observables=circuit.num_observables)
            pairs = np.empty((0, 2), dtype=np.int64)
            if "pairs" in args.metrics:
                pairs = np.column_stack(np.triu_indices(circuit.num_detectors, k=1))
                if 0 < args.pair_limit < len(pairs):
                    indices = np.random.default_rng(args.maestro_seed).choice(len(pairs), args.pair_limit, replace=False)
                    pairs = pairs[np.sort(indices)]
                workload["detector_pairs"] = pairs.tolist()
            qc, _, _ = translate_stim_to_maestro(circuit)
            cfg = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim,
                simulation_type=maestro.SimulationType.Stabilizer, seed=args.maestro_seed, max_simulators=1)
            plan = maestro._compile_reference_sampler(qc, maestro.NoiseModel(), cfg)
            if plan is None:
                raise RuntimeError("generated circuit did not compile a reference frame plan")
            converter = circuit.compile_m2d_converter()

            def maestro_sample(count, batch):
                # An additive golden-ratio stride alone aliases the sampler's
                # internal batch derivation. Mix it before deriving those streams.
                # Reproduction includes this script's batch size and seed scheme.
                seed = call_seed(args.maestro_seed, batch)
                raw = plan.sample(count, seed)["samples"]
                measurements = raw[:, :circuit.num_measurements].view(np.bool_)
                return converter.convert(measurements=measurements, separate_observables=True, bit_packed=False)

            datasets = workload["datasets"]
            datasets.append(collect(np, args, circuit, pairs, maestro_sample, "maestro_frame", args.maestro_seed))
            for seed in args.stim_seeds:
                sampler = circuit.compile_detector_sampler(seed=seed)
                datasets.append(collect(np, args, circuit, pairs,
                    lambda count, _, sampler=sampler: sampler.sample(count, separate_observables=True), "stim", seed))
            # Also compare Stim seeds to one another as an independent noise baseline.
            for i, left in enumerate(datasets):
                for right in datasets[i + 1:]:
                    comparison = {"left": {"sampler": left["sampler"], "seed": left["seed"]},
                                  "right": {"sampler": right["sampler"], "seed": right["seed"]}}
                    for metric, field, labels in (("detectors", "detector_counts", list(range(circuit.num_detectors))),
                                                  ("pairs", "pair_counts", pairs.tolist()),
                                                  ("logical", "observable_counts", list(range(circuit.num_observables)))):
                        if metric in args.metrics:
                            comparison[metric] = difference(np, left, right, field, labels)
                    workload["comparisons"].append(comparison)
            workload["status"] = "ok"
        except Exception as error:
            workload["status"] = "error"
            workload["error"] = f"{type(error).__name__}: {error}"
    failed = any(workload["status"] == "error" for workload in report["workloads"])
    report["status"] = "error" if failed else "ok"
    print(json.dumps(report, separators=(",", ":")))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
