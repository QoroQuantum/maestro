# Benchmarks

(bench-mps-estimator)=
## MPS expectation values from Qiskit: Maestro against Aer

This is the comparison a Qiskit user makes when they swap Aer's `EstimatorV2`
for [qiskit-maestro](https://github.com/QoroQuantum/qiskit-maestro)'s
`MaestroEstimator`, both on matrix product states, for circuits of 120 and 200
qubits. Both estimators get the same circuits and observables.

| Workload | Qubits | Aer `EstimatorV2` | `MaestroEstimator` | Maestro speed-up | Largest difference |
|---|---:|---:|---:|---:|---:|
| VQE energy | 120 | 0.120 s | 0.019 s | 6.2× | 3 × 10⁻¹¹ |
| VQE energy | 200 | 0.315 s | 0.042 s | 7.5× | 9 × 10⁻¹² |
| Trotter dynamics, one time point | 120 | 0.145 s | 0.164 s | 0.9× | 1 × 10⁻⁸ |
| Trotter dynamics, one time point | 200 | 0.342 s | 0.328 s | 1.0× | 4 × 10⁻⁸ |
| Trotter dynamics, every step | 120 | 2.75 s | 0.56 s | 4.9× | 6 × 10⁻⁶ |
| Trotter dynamics, every step | 200 | 6.98 s | 1.00 s | 7.0× | 2 × 10⁻⁵ |

Times are the median of five runs after a warm-up run. The last column is the
largest difference between the two estimators' expectation values. The script
stops if any difference exceeds 10⁻⁴ of the value's size; with truncation,
both estimators are about 10⁻⁵ of that size from the exact values.

The workloads:

- **VQE energy**: one energy evaluation, as in each step of a VQE loop. The
  circuit is `efficient_su2(n, reps=3, entanglement="linear")` with random
  angles, and the observable is a transverse-field Ising Hamiltonian with
  2n − 1 terms, giving one expectation value.
- **Trotter dynamics, one time point**: 20 first-order Trotter steps of the
  same Ising model (`rzz` and `rx` gates), then ⟨Z⟩ on every qubit, giving n
  expectation values.
- **Trotter dynamics, every step**: the same evolution measured after 0, 1, ...,
  20 steps, as dynamics code does it: one circuit per time point, sent as 21
  PUBs in one call, each with the energy and ⟨Z⟩ on every qubit.

Both estimators cap the bond dimension at 64, use a truncation threshold of
10⁻¹⁰, and use all CPU cores, as they do by default.

### Why measuring every step is faster

`MaestroEstimator` recognises that the 21 circuits are one initial state
followed by 0 to 20 repetitions of the same step. It runs them as one
evolution on a persistent simulator, using Maestro's `incremental_evolve`,
and reads every time point on the way: 20 steps of simulation in total,
where Aer simulates each circuit from the start, 210 steps in all. With that
turned off (`incremental_evolution=False`), Maestro took 1.58 s at 120 qubits
and 3.25 s at 200, so incremental evolution accounts for about 3× of the
speed-up.

### Why one time point is even

Aer applies `rzz` gates natively. Maestro's Python API has no `rzz` yet, so
qiskit-maestro expands each one into `cx`, `rz`, `cx`, which doubles the
two-qubit gates the MPS has to apply. With every circuit transpiled to `rx`,
`ry`, `rz` and `cx` first, so both simulators apply the same gates, Maestro is
1.3× faster at 120 qubits and 1.5× faster at 200. The VQE energy results don't
change (5.9× and 7.4×), because that circuit has no `rzz`.

### A VQE loop

The [example notebook](https://github.com/QoroQuantum/qiskit-maestro/blob/main/examples/vqe_aer_vs_maestro.ipynb)
runs 40 SPSA iterations on a 100-qubit Ising chain, with three energy
evaluations per iteration, on each estimator. On the machine below, the whole
loop took 3.1 s on Aer and 0.38 s on Maestro, and the two energy traces agreed
to 3 × 10⁻⁹.

### Setup and reproducing

Apple M4 Pro (12 cores, 48 GB), macOS 26.6.2, Python 3.12.13, Qiskit 2.6.1,
Qiskit Aer 0.18.2, qoro-maestro 0.3.10, qiskit-maestro 0.1.0. One machine
and the CPU only: expect different ratios elsewhere.

The script is
[`benchmarks/estimator_mps.py`](https://github.com/QoroQuantum/qiskit-maestro/blob/main/benchmarks/estimator_mps.py)
in the qiskit-maestro repository:

```bash
pip install qiskit-maestro qiskit-aer
python benchmarks/estimator_mps.py --qubits 120 200
python benchmarks/estimator_mps.py --qubits 120 200 --common-basis
```

:::{admonition} To do
:class: warning
Add the remaining benchmarks: Linux and GPU hardware, memory use, the other
simulation methods against Aer, and sampling as well as expectation values.
:::
