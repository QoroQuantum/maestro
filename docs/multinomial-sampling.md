# Multinomial count sampling in Maestro

## Finding

Sampling the histogram directly pays when many independent shots use the same output distribution and often repeat the same outcomes. The method applies to any simulator that can supply the joint probabilities of the measured bits at reasonable cost. A small marginal of a large state can be an excellent case. A broad distribution on a compressed backend can be expensive to enumerate even when individual samples are cheap.

For example, the computational-basis histogram of a 100-qubit GHZ state has two possible entries. If their probabilities are known, one fair binomial draw gives the first count and subtraction gives the second. A 100-qubit product of plus states has the same register width and $2^{100}$ possible outputs. A stabilizer sampler can generate the requested strings directly without building that table.

## What the paper measures

Given probabilities $p_i$ and $S$ shots, the counts follow a multinomial distribution. With $R$ shots and probability mass $P$ left, draw the next count from `Binomial(R, p_i / P)`. The last category receives the remainder. The basic sweep uses at most $D-1$ binomial variates for $D$ categories. With an expected constant-time binomial generator its expected arithmetic cost is $O(D)$. The ordinary finite-shot fluctuations and joint correlations are preserved. The authors' SeqBin implementation groups rare bins and redistributes their realized counts individually. A dense input still has to be read, and that redistribution has a per-hit cost. [Algorithm and optimizations](https://arxiv.org/html/2610.35119v1#S3)

The paper compared up to 20 qubits and one million shots. Its count samplers started with a probability vector. Its Qiskit benchmarks started with a circuit and extracted counts from shot records. The authors report speedups above 1,000× in most million-shot comparisons. In one profiled 15-qubit StatevectorSampler run, Python conversion of individual bitstrings accounted for 85.9% of runtime. With precomputed dense 20-qubit probabilities at one million shots, compiled NumPy multinomial and SeqBin took about 50 and 52 ms, respectively. On GHZ they took 7.1 and 2.2 ms. Maestro already accumulates native histograms and uses constant-time alias draws when the table fits. Its baseline avoids the Python conversion. [Experimental setup](https://arxiv.org/html/2610.35119v1#S4) · [Pipeline profile](https://arxiv.org/html/2610.35119v1#A1)

The decomposition is mathematically exact. The authors' BTPE generator (`btpe_draw_numba`) adds all four Stirling corrections in its final acceptance test, as the original BTPE publication and ranlib do. GSL's implementation subtracts the corrections for $x+1$ and $n-x+1$, calls the all-plus form a mistake confirmed with Kachitvichyanukul, and reports that it could not observe any difference, since the corrections are about $10^{-5}$ of the compared logarithms. Maestro does not use BTPE: it uses Boost.Random's binomial inversion and BTRD rejection implementation, with geometric waiting times for very small probabilities. [Released generator](https://github.com/ijaz-3/seqbin/blob/6d3d4999ee4cc09a276c9153911a711d90436ab4/seqbin.py) · [GSL correction](https://github.com/ampl/gsl/blob/master/randist/binomial_tpe.c) · [Maestro binomial generator](../Utils/Sampling/Multinomial.h)

## The Maestro sampler

The prepared count sampler first validates the probabilities in fixed blocks and counts the positive rows. A count plan then splits the shots across populated blocks and, inside each block, across rows. Both splits read prefix masses in reverse order, so each conditional probability is `member / (member + preceding)` computed from sums rather than by subtracting nearly equal masses (the paper's App. C.1). Each split draws the less probable side directly, which preserves very small tails when the larger side's normalized probability rounds to one. [Prepared sampler](../Utils/Sampling/Sampling.h)

**Rare-outcome grouping** (the paper's Sec. 3.3.1). Rows expected to receive at most two shots, $p_i \le 2\,\mathrm{mass}/S$ (computed as $\mathrm{mass}/S$ times two, so that a mass near the largest double cannot overflow), form one group per block. The group is split from the block's other rows with one binomial draw, and its hits are placed by independent categorical draws over the group's rows. Normalized exponential spacings give those draws as sorted uniforms in linear time, so one sweep of the block places a batch without sorting. Batches are bounded by the scratch budget, and one reservation of that size holds all of them, so the buffer never regrows. Targets are compared with cumulative masses as fractions of the group, so a group whose mass is subnormal keeps equal rows equally likely. The decomposition stays exact for any threshold, which therefore only affects speed. A total weight outside $[2^{-900}, 2^{900}]$ is first shifted by an exact power of two, a tiny one up into $[0.5, 1)$ and a huge one down to just below $2^{900}$, as for every sampler family (see [Sampling](sampling.md)). The threshold then cannot underflow, and the separately accumulated common and rare subtotals of a block cannot overflow when they are added.

**Crossover.** The plan uses multinomial counts when there are at least 256 shots and either

- at most one positive row per 16 shots, or per 24 shots for tables of at most 256 rows, whose per-shot draws stay in cache, or
- a grouped tail: at most one binomial draw (populated block or common row) per 16 shots, with the same small-table rule and at most a quarter of the mass in grouped rows.

The second gate first bounds both costs from each block's largest row, recorded during validation: a block whose largest row is at most the threshold holds only grouped rows, and any other block holds at least one common row. That rejects a spread distribution without reading its rows again. Otherwise one serial pass over the blocks that mix both kinds of rows stops as soon as either limit is exceeded. A concentrated distribution with a long tail of tiny but nonzero amplitudes, typical of variational workloads, therefore qualifies even though every row is positive. A distribution spread over many rows does not, and keeps the alias or CDF sampler.

**Binomial variates.** `BinomialCount` uses Boost.Random's `binomial_distribution`: inversion for means below 11 and Hormann's BTRD otherwise. For a rare-side probability below $10^{-8}$ with a small expected count, it skips failures geometrically with `log1p`, because $(1-p)^n$ can round to one there even when the expected count is not small. Counts above $2^{48}$ are split into independent chunks, using $\mathrm{Bin}(n_1,p) + \mathrm{Bin}(n_2,p) = \mathrm{Bin}(n_1+n_2,p)$. Seeded `multinomial_v1` sequences therefore depend on Boost.Random's binomial algorithm, which has been unchanged across recent Boost releases. [Maestro binomial generator](../Utils/Sampling/Multinomial.h)

**Validation.** The adaptive-sampling tests compare every binomial branch with the exact probability mass function, and compare complete joint histograms (every composition of a few shots over rows inside one block and across blocks, including grouped rows and one-target batches) with the exact multinomial distribution, using fixed seeds and a 4.5-sigma Pearson bound. They also check scale invariance: weights scaled by the smallest double or by $2^{1000}$ give bitwise the same seeded counts on the count path, the alias table, the CDF and the bounded fallback, including rows of $2^{-1070}$ beside common rows. A block whose common subtotal rounds to `DBL_MAX` keeps its rare rows, and finite weights whose sum overflows in row order are sampled. They pin the policy's constants, its crossover decisions (including an odd row count at the small-table rule), the tail-grouping gate's rare-mass cutoff and draw limit on both sides, seeded histograms accepted by each gate, a histogram whose rows expect 2.5 shots each and so change sampler with the rare-row definition, and binomial draws on both sides of the geometric cutoff; the values are the same with MSVC and GCC. An independent check on Linux (GCC 13.3) also matched the mean and exact variance of the Pearson statistic over 2,852 and 48,054 categories, and the pooled frequencies of a 70,000-row grouped tail ($\chi^2/\mathrm{dof} = 1.005$). [Tests](../tests/AdaptiveSamplingTests.cpp)

## Where it applies

| Method | Evaluation and implementation |
| --- | --- |
| QCSim statevector | Uses squared amplitudes over the existing support bound. The count plan skips zero-mass blocks. Up to 12 distinct measured bits inside the support are projected into a joint marginal in one pass, by one shift and mask for a contiguous selection or one table lookup per touched byte of the basis index otherwise. Fixed chunks run that pass in parallel with reproducible sums. Their count depends only on the sizes: at most 256 chunks of at least 8,192 rows, and at most one partial entry per 64 rows, so zeroing and reducing the partial marginals stays small beside the pass. The reduction is split by outcome. |
| QCSim density matrix | Uses diagonal populations, including the existing joint marginal for eligible selections of up to 12 distinct bits. Once a four-bit marginal is available, at most 15 binomial draws suffice. |
| QCSim path integral | Uses the normalized retained basis rows, preserving the backend's pruning semantics and wide labels. If a probability and label snapshot exceeds the scratch budget, the existing bounded CDF sampler handles the request. |
| QCSim matrix product state and operator | For one distinct measured qubit, obtains the marginal once and splits all shots with one binomial draw. Repeated selections reproduce the same bit. An MPO with a bond cap or truncation threshold keeps the native sampler. Wider measurements keep the native chain samplers (see below). |
| QCSim extended stabilizer | For small terminal readouts, obtains joint probabilities from the frame's Z parity expectations. This includes interference between coherent Clifford-basis components. The implementation and measurements below distinguish the prepared-state backend from noisy batched frames. |
| Stabilizer, tensor networks and GPU backends | Native reference frames process 64 Clifford shots per word. Tensor-network methods sample conditionally without constructing a full probability vector. GPU crossover depends on whether the probabilities and histogram stay on device. These paths retain their native sampling algorithms. |

A noise trajectory has its own conditional distribution. Histograms from separate realizations must be sampled with their assigned shot budgets and then added. Pooling shots from one trajectory would change the correlations. Mid-circuit control can also change the state from one branch to the next. Finally, emitting ordered shot records requires work proportional to the number of rows. Histogram speedups concern count requests. [Maestro sampling and noise semantics](sampling.md)

## Matrix product states and operators

Let $n$ be the chain length, $\chi$ the largest bond dimension and $q$ the chain position of the measured qubit.

**MPS.** QCSim keeps sites to the right in canonical form, so `GetProbability(q)` contracts only the site and its left bond: $O(\chi_{q-1}\chi_q)$ time and no scratch. The native one-qubit sampler already computed this marginal once and then drew every shot as a separate Bernoulli trial. The new path draws one binomial from the same marginal, so the distribution is identical and the cost no longer grows with $S$. With 1,024-dimensional bonds that is about a million complex products, once per call. [MPS adapter](../Simulators/QCSim/QCSimMPSSimulator.h)

**MPO.** The marginal $\mathrm{Tr}(\rho\,\Pi^0_q)/\mathrm{Tr}\,\rho$ multiplies a row vector through the chain twice, once with the selected site and once for the trace. That is $O(n\chi^2)$ time with one $\chi\times\chi$ site matrix of scratch at a time, independent of $S$. The native sampler builds and caches traced suffixes, also $O(n\chi^2)$, then conditions every shot along the chain up to the measured site, $O(q\chi^2)$ per shot. Right after evolution, when that cache is cold, the new path is therefore within a small constant of the native one even for $q = 0$, and about $Sq/n$ times cheaper otherwise. Repeated calls on an unchanged MPO whose measured qubit sits near the start of the chain can still favor the native sampler's warm cache. [MPO adapter](../Simulators/QCSim/QCSimMPOSimulator.h)

For an exact operator both samplers follow the Born law. Truncation can leave negative populations. The native sampler then clamps each conditional probability along the chain, while the marginal would clamp only $\mathrm{Tr}(\rho\,\Pi^0_q)/\mathrm{Tr}\,\rho$, and the two laws can differ substantially. With a bond cap of 2, one three-qubit circuit gave 35.5% ones natively and 8.4% from the marginal; the exact answer is 50%. The adapter therefore uses the marginal only for an operator to which no bond cap and no positive singular-value threshold has been applied since it was created. The decision cannot rest on observed bond dimensions. QCSim installs a basis mixture, and truncates in `Trim()` and `Hermitize()` (which doubles the bonds and trims them back), without reporting bond dimensions. A saved, truncated state can also be restored after `Reset()`. The flag therefore survives `Reset`, `SaveState`/`RestoreState` and clones, and only creating or clearing the operator resets it. An MPO whose cap is never reached also keeps the native sampler. A trace too small to normalize ($\mathrm{Tr}\,\rho \le \varepsilon$), which the native sampler rescales along the chain, also keeps the native sampler. Both checks run before any RNG is consumed, so these requests reproduce the `legacy` counts.

**Scheduling.** For one qualifying measured qubit, the network keeps an MPS or MPO batch on one worker. Splitting the shots would repeat the contraction on every worker and could put each one below the crossover. An MPO configured with a bond cap or a singular-value threshold keeps its shot workers, since it uses the native sampler. [Network scheduling](../Network/SimpleDisconnectedNetwork.h)

**Wider measurements.** These keep the native samplers and their compressed representations. A joint marginal over $k$ measured qubits would cost $2^k$ prefix contractions without the paper's tree splitting. The natural extension splits counts at occupied measurement prefixes, depth first: one binomial per occupied prefix, one left vector of size $\chi$ per level, and contractions shared by every shot below a prefix. The number of contractions is bounded by both $2^k$ and $kS$, and memory stays at $O(k\chi)$. It needs QCSim access to the site tensors (MPS) or traced suffixes (MPO) and is left for a later change. The MPS sampler for several qubits already moves the measured qubits to the start of the chain with swaps before sampling, which can truncate the state.

## Dispatch and reproducibility

`counts_policy="auto"` allows eligible unseeded terminal counts to use the new path. Existing reference-frame and noisy extended-frame dispatch retain priority. A supplied simulator or noise seed, including zero, preserves the earlier default sequence. The exception is an input whose total weight lies outside $[2^{-900}, 2^{900}]$; for a path integral that is the total of the squared amplitudes, including one that overflows. Earlier builds sampled such an input with a bias, could drop its rare rows or rejected it, and its seeded counts now differ. On a path-integral state that earlier builds rejected, a one-shot or measurement now draws from QCSim's stream and a batch from Maestro's batch stream, which shifts the later draws from that stream on the same simulator until it is reseeded (see [Sampling](sampling.md)). Select `counts_policy="multinomial_v1"` to opt seeded requests into the adaptive sampler and reproduce its own sequence. Replay a native request with its returned seed and `execution_metadata.configured_options.counts_policy`, as for frame counts: a stored seed alone selects the earlier sampler. The policy names the permitted algorithm family. A request below its crossover still uses the existing sampler. The `multinomial_v1` decisions and stream are fixed from its first release: a later change to the crossover, the grouping or the variate generators belongs in a new policy version, so that recorded results keep replaying. A pinned test enforces this.

On a bare QCSim simulator, opt in with `Configure("counts_policy", "multinomial_v1")`. Its default preserves replay of the automatically generated `sampling_seed` through `SetSeed`. An explicitly configured low-level `sampling_policy` pins the existing sampler. These distinctions preserve old seeds, including the RNG position after a batch. [Policy selection](../Execution/SimulatorConfig.h) · [Backend configuration](../Simulators/QCSim/ImmediateQCSimSimulator.h)

At terminal checkpoints without readout errors, the count path transfers each distinct outcome's multiplicity to the classical register once. Independent readout errors still require sampling each copy unless their joint effect is also aggregated. Circuit evolution, measurement-to-classical mapping and output serialization all contribute to end-to-end runtime. [Checkpoint sampling](../Execution/TerminalSampling.h)

## CPU measurements

The initial feasibility comparison used normalized uniform probabilities, one CPU thread, packed integer count-map outputs and nine timed runs after warm-up. It compared Maestro at `e17aac3c` to NumPy's compiled conditional-binomial generator. At 1,024 outcomes and one million shots, Maestro took 3.95 ms and NumPy took 0.129 ms. At 262,144 outcomes and 1,000 shots, Maestro took 0.583 ms and NumPy took 4.32 ms. At 262,144 outcomes and one million shots both took about 43 ms. These figures characterize the method family.

The implementation measurements below compare the **actual Maestro adapters** on an Intel Core i9-13900KS (24 cores), Windows Release, MSVC 19.44 with OpenMP. Both policies start from the same prepared quantum state and return the same native count-map format. The 18-qubit uniform control uses packed integer keys. The other adapter rows use bit-vector keys. Timings include probability preparation, RNG, output projection and histogram construction. Initialization, gates, compilation and correctness checks are outside the timer. Each figure is a median after warm-up, alternating policy order. Thread counts are permitted maxima. Very short timings have limited precision. [Benchmark source](../tests/SamplingBenchmarks.cpp)

| Workload | Shots | Threads | Existing sampler, ms | Adaptive policy, ms |
| --- | ---: | ---: | ---: | ---: |
| Statevector, 10 qubits, random, all bits | 1,000,000 | 1 | 11.064 | 0.146 |
| Statevector, 18 qubits, GHZ, all bits | 1,000,000 | 8 | 3.705 | 0.108 |
| Statevector, 18 qubits, GHZ, four-bit marginal | 256 | 8 | 0.062 | 0.019 |
| Statevector, 18 qubits, random, four-bit marginal | 1,000,000 | 1 | 20.954 | 0.202 |
| Statevector, 18 qubits, uniform, all bits | 1,000 | 1 | 0.656 | 0.612 |
| Statevector, 20 qubits, 99% on two outcomes and a 1% tail on all others, all bits | 1,000,000 | 1 | 19.992 | 5.551 |
| Statevector, 20 qubits, same tail, all bits | 1,000,000 | 16 | 18.273 | 3.900 |
| Statevector, 24 qubits, random, four-bit marginal | 4,096 | 16 | 7.625 | 4.698 |
| Statevector, 22 qubits, GHZ, 12-bit marginal | 65,536 | 16 | 2.111 | 1.218 |
| Statevector, 22 qubits, random, 12-bit marginal | 65,536 | 16 | 11.661 | 2.485 |
| Statevector, 22 qubits, uniform, all bits (rejected) | 100,000 | 1 | 38.766 | 39.213 |
| Density matrix, 10 qubits, random diagonal, four-bit marginal | 1,000,000 | 1 | 5.681 | 0.0107 |
| Path integral, 128 qubits, 4,096 retained rows, 12 measured bits | 1,000,000 | 1 | 11.821 | 0.554 |
| MPS, eight qubits, uniform, one measured bit | 1,000,000 | 1 | 38.879 | 0.0020 |
| MPO, 16 qubits, uniform (bond dimension 1), one measured bit | 1,000,000 | 1 | 526.58 | 0.0053 |
| MPO, 16 qubits, three random layers (bond dimension 64), last site | 10,000 | 1 | 266.84 | 0.127 |
| MPO, 24 qubits, four random layers (bond dimension 256), last site | 2,048 | 1 | 1,922.8 | 7.713 |

Statevector and density-matrix figures are medians of nine repetitions, path integral of seven, MPS of 15, and MPO and extended stabilizer of five. The uniform 1,000-shot control stays below the crossover, so both policies run the same sampler. In the tail rows every row is positive, which the per-row rule rejects; the grouped plan instead draws one binomial per populated 64-row block and per common row and places the tail's hits categorically. The 24-qubit marginal uses a thousandth as many shots as basis states. With the parallel projection the marginal is cheaper than the full-row sampler even there, so it no longer requires a quarter as many shots as basis states. Fixed 8,192-row chunks keep that projection parallel for small registers too, as the 256-shot GHZ row shows. Bounding the partial marginals by the pass size keeps a wide marginal of a sparse state cheap: the 12-bit GHZ row was 2.2 times slower than the existing sampler with 256 chunks and is now faster with 16. The uniform 22-qubit row is rejected by the grouping gate from the per-block maxima, without a second pass over the rows, so it costs the same as the existing sampler: three runs of 15 repetitions gave ratios of 0.98 to 1.03. The MPO rows measure the last site of the chain, the most expensive position for the native sampler; each bond dimension is the largest in its fixture.

Forced kernel comparisons located the crossover separately from the dispatch measurements. On 16,384 uniform outcomes, the ratio of existing time to multinomial time was:

| Shots per outcome | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| One thread | 0.87 | 0.95 | 1.12 | 1.63 |
| 16 threads | 0.92 | 1.10 | 1.45 | 1.93 |

The existing per-shot draw is bound by its serial random-number generation and gains little from threads, so the crossover holds with and without them. Between 8 and 16 shots per draw the two samplers break even, which sets the rule of 16. Small tables stay in cache and move the break-even up. At 16 shots per outcome, 256 outcomes broke even (0.98–1.02×) and 512 gave 1.10×. At 24 shots, 256 outcomes gave 1.30×, which sets the rule of 24 for tables of at most 256 rows. The sparse GHZ case illustrates the value of counting positive rows. Its support stays at two outcomes as register width grows.

## Extended-stabilizer frames

For a coherent frame superposition, QCSim's ordinary terminal sampler restores a private state and measures it for each shot. With a single Clifford-basis component it already samples an affine support efficiently. The new sampler uses QCSim to enumerate the measured part of that support in Gray-code order. The new path queries the joint marginal of the requested bits without expanding a full-register statevector. For a coherent superposition and $m$ distinct bits, each subset's Z-product expectation is a parity moment. An inverse Walsh–Hadamard transform of the $2^m$ moments gives the $2^m$ joint probabilities. The empty subset holds the frame's squared norm, which the native measurement also divides by. Computing each bit's marginal independently would lose entanglement correlations. [Frame adapter](../Simulators/QCSim/QCSimExtendedStabilizerSimulator.h)

An outcome of zero probability can come out of the transform a few ulps below zero. Values within $10^{-12}$ in probability units are set to zero. A materially negative or nonfinite reconstruction falls back before the sampler consumes RNG. When no measured qubit's Z image has an X part in the frame's Clifford basis, every component has a definite outcome. The joint weights are then sums of squared amplitudes by outcome, computed in one pass over the components without parity moments. These sums of positive terms keep small probabilities accurate. Otherwise each outcome's probability is a difference of parity moments near the norm, whose rounding grows with the number of components $k$: a probability below roughly $k \cdot 10^{-16}$ of the norm can be lost. In a 20-qubit product of rotations (about $10^6$ components, diagonal for the measured qubit), QCSim's $1 - \langle Z_0\rangle$ came out as $1.43\cdot10^{-11}$ against an exact $3.20\cdot10^{-11}$. The parity method then drew no ones in $2^{40}$ shots, where 17.6 were expected; the diagonal sums give the exact probability. For off-diagonal images the loss matters only when the shot count times $k \cdot 10^{-16}$ approaches one.

Enumeration is bounded to eight distinct measured bits. The frame-specific crossover is at least 256 shots and at least eight shots per possible measured outcome. At eight bits, 2,048 shots clear that gate. Native frame measurement and collapse cost more per shot than a prepared alias-table lookup. Exact-mode frames can use the joint marginal, and so can a single stabilizer component in approximate mode, since measuring it never prunes. Otherwise, with approximate pruning, the optimization is limited to one distinct bit because QCSim can prune components after a measurement and thereby change the conditional state for the next bit. The separate noisy `extended_frame_v1` executor propagates shot-dependent frames and amplitudes. Those lanes cannot in general use one joint distribution. Its seeded frame and raw-record behavior is preserved. [Batched frame measurement](../Execution/ExtendedFrameBatch.h)

The non-Clifford fixture is a 10-qubit GHZ circuit with $R_y(\pi/7)$ rotations on qubits 0, 2 and 4. The Clifford controls cover an affine distribution with two outputs and one with full support. All measurements in this table use one thread.

| State | Measured bits | Shots | Existing sampler, ms | Adaptive policy, ms |
| --- | ---: | ---: | ---: | ---: |
| Rotated GHZ | 1 | 256 | 0.0718 | 0.0014 |
| Rotated GHZ | 4 | 512 | 0.178 | 0.0080 |
| Rotated GHZ | 8 | 2,048 | 0.737 | 0.0476 |
| Rotated GHZ | 8 | 8,192 | 2.912 | 0.0473 |
| GHZ | 8 | 8,192 | 0.184 | 0.0030 |
| Uniform | 8 | 8,192 | 0.706 | 0.0382 |

A wider control kept ten active qubits in a 1,024-qubit register and included qubit 1,023 among its eight measured bits. At 8,192 shots, the rotated GHZ fixture took 92.80 ms with the existing sampler and 0.339 ms with the adaptive policy (274×). For the Clifford GHZ control the times were 0.309 and 0.0042 ms. This checks the cost of querying a high-index measurement without allocating a full-register probability vector.

The number of frame components also affects the cost of each parity query. A larger coherent superposition kept the advantage: with 4,096 components, eight measured bits, off-diagonal measurements and 2,048 shots, a public request took 39.06 ms with the existing sampler and 11.36 ms with the adaptive one at a 16-worker limit. The optimal crossover for larger superpositions and on other CPUs remains unmeasured.

## End-to-end results

The public `MaestroRunRequestJson` benchmark timed fixed-backend OpenQASM requests from parsing through result serialization. Each figure is the median of ten alternating, paired runs after warm-up, with a four-worker limit. For qualifying MPS, MPO and extended-stabilizer counts, the scheduler keeps the batch together so that dividing the shots among workers cannot push each worker below the crossover. These figures were measured with the first implementation, before rare-row grouping and the MPO path. The follow-up changes leave these two request paths' sampling work the same. [Network scheduling](../Network/SimpleDisconnectedNetwork.h) · [Worker and replay tests](../tests/NetworkJobTests.cpp)

| Workload | Shots | Existing policy, ms | Adaptive policy, ms | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Extended stabilizer, rotated GHZ, eight measured bits | 2,048 | 0.700 | 0.368 | 1.90× |
| Extended stabilizer, rotated GHZ, eight measured bits | 8,192 | 1.279 | 0.391 | 3.27× |
| MPS, eight qubits, one measured bit | 256 | 0.463 | 0.307 | 1.51× |
| MPS, eight qubits, one measured bit | 4,096 | 0.492 | 0.299 | 1.64× |

With a one-worker limit, the 8,192-shot extended-stabilizer request improved from 3.539 to 0.395 ms (8.97×). All four fixtures conserved shots and returned identical seeded adaptive histograms with one and four workers.

## Compatibility checks

Against a pre-change Maestro DLL built at `7b1a5cf8`, the default seeded policy of the first implementation reproduced 40 complete histogram maps and 40 ordered shot buffers byte for byte (409,600 bytes of packed records). The comparison ran the two DLLs in separate processes. It covered all five affected CPU methods, seeds 0 and 97, one and four workers, repeated terminal readouts, dynamic measurements and a 70-bit classical register. The follow-up changes touch only the opted-in paths. Their tests check again that seeded `auto` and `legacy` requests keep the historical counts. On Windows Release, all 23 configured native CTest targets, the CI-filtered C++ unit suite and the 1,686 Python tests that run without a GPU pass.

## Reproducing the comparison

Build the optional target, then run the resulting executable from its output directory. The final two arguments select one outer worker and the existing-versus-multinomial comparison. Setting the internal thread argument to zero selects serial execution. The `tail` state puts 99% of the mass on two outcomes and spreads 1% over all others. For `mpo`, `random` applies `active_bits` layers of random $R_y$ rotations and a CNOT chain, and with `active_bits` below the register width the measured qubit is the last site of the chain.

```text
cmake --build build --config Release --target sampling_benchmarks
sampling_benchmarks sv 10 10 1000000 0 random 10 many 9 1 multinomial
sampling_benchmarks sv 20 20 1000000 0 tail 20 many 9 1 multinomial
sampling_benchmarks sv 24 24 4096 16 random 4 many 9 1 multinomial
sampling_benchmarks dm 10 10 1000000 0 random 4 many 9 1 multinomial
sampling_benchmarks path 128 12 1000000 0 random 12 many 7 1 multinomial
sampling_benchmarks mps 8 8 1000000 0 uniform 1 many 15 1 multinomial
sampling_benchmarks mpo 24 4 2048 0 random 1 many 5 1 multinomial
sampling_benchmarks extended 10 10 8192 0 rotated 8 many 5 1 multinomial
sampling_benchmarks extended 1024 10 8192 0 rotated 8 many 5 1 multinomial
sampling_benchmarks kernel 14 14 262144 0 uniform 14 packed 9 1 multinomial
```

To repeat the public API measurements, use schema version 2, operation `execute`, QCSim with fixed selection, and `max_simulators` set to 1 or 4. Build the rotated GHZ circuit using H on qubit 0, a chain of CNOTs from qubit 0 through 9, and the three rotations described above. For MPS, apply H to each of eight qubits. Map the lowest measured qubits to the corresponding classical bits. Use seed 0 for warm-up, then time one call to `MaestroRunRequestJson` for each policy at each seed from 1 through 10, alternating which policy runs first. Take the median per policy. The native call includes JSON serialization. The timer stops before decoding the returned JSON in the caller.
