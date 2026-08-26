Yes. The cleanest conceptual distinction is that a density-matrix simulator and an MPO representation of a density matrix evolve the **statistical state of the system**, whereas a statevector/MPS normally evolves a **single pure realization**.

Mathematically, your new simulators promote the evolution from

\[
|\psi\rangle \rightarrow U|\psi\rangle
\]

to the much more general quantum-channel evolution

\[
\rho \rightarrow \mathcal E(\rho)
       =\sum_k K_k\rho K_k^\dagger ,
\qquad
\sum_k K_k^\dagger K_k=I.
\]

Unitary evolution is just the special case with one Kraus operator \(K_0=U\).

That difference opens the door to relaxation, decoherence, thermalization, classical uncertainty, coupling to an environment, etc. It is essentially the difference between simulating a **closed quantum system** and an **open quantum system**.

One qualification: statevector/MPS simulators can also simulate many kinds of noise through **quantum trajectories**—randomly choosing Kraus outcomes and averaging many pure-state runs. And modern Pauli-propagation methods can incorporate quite general noise channels efficiently for certain observable-estimation problems [3].  So DM/MPO aren't uniquely *capable* of noise; their big advantage is that they represent the mixed state directly.

## 1. What noise actually means

For an ideal gate,

\[
\rho' = U\rho U^\dagger.
\]

For a noisy implementation you instead have something like

\[
\rho' =
\mathcal E_U(\rho)
=
\mathcal N_U(U\rho U^\dagger),
\]

where \(\mathcal N_U\) is a CPTP map.

In practice you might associate noise with:

\[
G_{\rm physical}
=
\mathcal N_{\rm after}\circ
\mathcal U_G\circ
\mathcal N_{\rm before}.
\]

Or, for continuously acting noise, evolve the noise during the gate rather than inserting it afterward.

A particularly useful distinction is:

- **coherent errors**: still unitary, e.g. over-rotation;
- **stochastic/unitary-mixture errors**: random unitary operations;
- **incoherent/non-unitary errors**: relaxation, decoherence, thermalization;
- **correlated noise**: channels acting jointly on several qubits;
- **non-Markovian noise**: environment retains memory.

The last three are where density operators become especially natural.

---

# 2. Standard noise models

## Bit-flip noise

With probability \(p\), apply \(X\):

\[
\mathcal E(\rho)
=
(1-p)\rho+pX\rho X.
\]

Kraus operators:

\[
K_0=\sqrt{1-p}\,I,
\qquad
K_1=\sqrt p\,X.
\]

This is actually easy for a statevector simulator: randomly decide whether to apply \(X\).

But the density matrix gives you the ensemble in one evolution rather than requiring sampling.

---

## Phase-flip / dephasing

One convention is

\[
\mathcal E(\rho)
=
(1-p)\rho+pZ\rho Z.
\]

If

\[
\rho=
\begin{pmatrix}
\rho_{00}&\rho_{01}\\
\rho_{10}&\rho_{11}
\end{pmatrix},
\]

then

\[
\rho_{01}\rightarrow (1-2p)\rho_{01}.
\]

Thus the populations stay unchanged while coherences decay.

A continuous physical dephasing model usually looks like

\[
\rho_{01}(t)=e^{-t/T_\phi}\rho_{01}(0).
\]

This is exactly the phenomenon that a pure-state representation doesn't naturally express: you're progressively destroying coherence without necessarily changing the populations.

---

# 3. Depolarizing noise

A common single-qubit convention is

\[
\mathcal E(\rho)
=
(1-p)\rho+
\frac p3
(X\rho X+Y\rho Y+Z\rho Z).
\]

Equivalent conventions parameterize \(p\) differently, so that

\[
\mathcal E(\rho)=(1-p)\rho+p\frac{I}{2}.
\]

Conceptually,

\[
\rho\rightarrow I/2
\]

as noise becomes large.

For \(n\)-qubit gates you can similarly use all nonidentity Pauli strings:

\[
\mathcal E(\rho)
=
(1-p)\rho+
\frac{p}{4^n-1}
\sum_{P\ne I}P\rho P.
\]

This is widely used because it is simple rather than because real hardware is literally depolarizing. Hardware-calibrated models combining several different channels generally reproduce devices better than a single generic model [1]. 

---

# 4. Amplitude damping — where density matrices really become important

This models energy relaxation

\[
|1\rangle\rightarrow |0\rangle
\]

with probability \(\gamma\).

The Kraus operators are

\[
K_0=
\begin{pmatrix}
1&0\\
0&\sqrt{1-\gamma}
\end{pmatrix},
\]

\[
K_1=
\begin{pmatrix}
0&\sqrt{\gamma}\\
0&0
\end{pmatrix}.
\]

Now start from

\[
\rho=|1\rangle\langle1|.
\]

After the channel,

\[
\rho'
=
(1-\gamma)|1\rangle\langle1|
+
\gamma|0\rangle\langle0|.
\]

This is a genuinely mixed state:

\[
\operatorname{Tr}(\rho'^2)<1
\]

for \(0<\gamma<1\).

There is **no statevector of the qubit alone** corresponding to it.

A statevector simulator has to do one of two things:

1. introduce an environment ancilla and entangle with it, or
2. randomly choose a quantum trajectory.

Your density-matrix simulator simply performs

\[
\rho' =K_0\rho K_0^\dagger+
       K_1\rho K_1^\dagger.
\]

No stochastic sampling is needed.

---

# 5. \(T_1\) relaxation and \(T_2\) decoherence

This is probably one of the most useful realistic models to add to your simulator.

For a gate/idle lasting \(t\),

\[
\gamma_{T_1}=1-e^{-t/T_1}.
\]

That gives amplitude damping.

For decoherence,

\[
\rho_{01}(t)
=
e^{-t/T_2}\rho_{01}(0).
\]

Usually one separates relaxation and pure dephasing:

\[
\frac1{T_2}
=
\frac1{2T_1}
+
\frac1{T_\phi}.
\]

Thus

\[
\frac1{T_\phi}
=
\frac1{T_2}-\frac1{2T_1}.
\]

This has an important simulator implication: **idle time matters**.

Suppose:

- \(X\) takes 30 ns,
- CX takes 300 ns,
- measurement takes 1 μs.

The other qubits should undergo their corresponding relaxation/dephasing while waiting.

So a realistic noise simulator eventually needs some notion of

\[
\text{gate duration}.
\]

That's a substantial improvement over simply doing

```text
apply_gate()
apply_noise(p)
```

with the same \(p\) for every gate.

---

# 6. Generalized amplitude damping / thermal relaxation

At nonzero temperature the environment doesn't only cause

\[
|1\rangle\rightarrow|0\rangle;
\]

it can also thermally excite

\[
|0\rangle\rightarrow|1\rangle.
\]

The equilibrium state is approximately

\[
\rho_{\rm thermal}
=
\begin{pmatrix}
1-p_e&0\\
0&p_e
\end{pmatrix}.
\]

Generalized amplitude damping has four Kraus operators rather than two.

This is a nice example where direct mixed-state simulation is substantially more elegant than pure-state simulation.

---

# 7. Coherent gate errors

Not all noise actually requires your new simulators.

For example, suppose an intended rotation is

\[
R_x(\theta)
\]

but the hardware actually performs

\[
R_x(\theta+\epsilon).
\]

Then

\[
U_{\rm real}=e^{-i(\theta+\epsilon)X/2}.
\]

That is still unitary.

Your statevector, MPS, path-integral, etc. simulator can handle it perfectly well.

Similarly,

\[
U_{\rm CX}^{\rm real}
=
e^{-i\epsilon H_{\rm error}}
U_{\rm CX}
\]

is coherent error.

An important realistic model therefore often combines

\[
\boxed{
\text{coherent error}
+
\text{incoherent noise}
}.
\]

For example:

\[
\mathcal E(\rho)
=
\mathcal D_p
\left(
U_{\rm erroneous}
\rho
U_{\rm erroneous}^\dagger
\right).
\]

---

# 8. Readout error

Measurement noise is somewhat different.

A simple model is the classical confusion matrix

\[
M=
\begin{pmatrix}
P(0|0)&P(0|1)\\
P(1|0)&P(1|1)
\end{pmatrix}.
\]

For example,

\[
P(1|0)=0.02,\qquad
P(0|1)=0.04.
\]

You don't actually need density matrices for this because you can corrupt the measurement results classically.

A more general treatment uses a POVM,

\[
p_m=\operatorname{Tr}(E_m\rho),
\]

which fits naturally with the density-matrix simulator.

---

# 9. Leakage

This is another interesting capability.

A physical transmon isn't a qubit:

\[
|0\rangle,\quad |1\rangle,\quad |2\rangle,\ldots
\]

and a gate may cause leakage

\[
|1\rangle\rightarrow|2\rangle.
\]

To model this properly you'd increase your local dimension from

\[
d=2
\]

to perhaps

\[
d=3.
\]

Then the density matrix goes from a local \(2\times2\) object to a \(3\times3\) object.

For the MPO, your local physical dimension correspondingly changes from

\[
d^2=4
\]

to

\[
d^2=9.
\]

That's architecturally quite clean if your tensor implementation already supports arbitrary physical dimension.

---

# 10. Correlated noise and crosstalk

An independent-error model assumes

\[
\mathcal E
=
\mathcal E_1\otimes\mathcal E_2\otimes\cdots.
\]

Real hardware doesn't necessarily behave this way.

For example, correlated phase noise might contain

\[
K_1=\sqrt p\,Z_iZ_j.
\]

Or correlated coherent noise might be

\[
U=e^{-i\epsilon Z_iZ_j}.
\]

You might also have a Pauli channel

\[
\mathcal E(\rho)
=
\sum_P p_P P\rho P
\]

where \(P\) includes strings such as

\[
XI,\quad IX,\quad XX,\quad ZZ,\ldots
\]

rather than factorizing into independent errors.

This is an area where MPOs are particularly appealing: a spatially local or structured correlated channel can itself have a compact MPO representation.

Tomography-assisted MPDO work has, for example, incorporated experimentally determined channels including crosstalk rather than assuming simple textbook noise [2]. 

---

# 11. Lindblad evolution

For continuous-time noisy dynamics, I would definitely support a Lindblad representation eventually.

The master equation is

\[
\frac{d\rho}{dt}
=
-i[H,\rho]
+
\sum_k\gamma_k
\left[
L_k\rho L_k^\dagger
-\frac12
\{L_k^\dagger L_k,\rho\}
\right].
\]

For relaxation,

\[
L=\sigma_-.
\]

For dephasing,

\[
L=Z
\]

up to convention-dependent factors.

You can write this as

\[
\frac{d}{dt}|\rho\rangle\rangle
=
\mathcal L|\rho\rangle\rangle
\]

after vectorizing the density matrix, and therefore

\[
|\rho(t+\Delta t)\rangle\rangle
=
e^{\mathcal L\Delta t}
|\rho(t)\rangle\rangle.
\]

This viewpoint fits your MPO simulator particularly nicely.

---

# 12. Why MPO is such a natural density-matrix representation

For \(n\) qubits, a dense statevector requires

\[
2^n
\]

complex numbers.

A dense density matrix requires

\[
2^n\times2^n=4^n.
\]

That becomes brutal very quickly.

Your MPO replaces

\[
\rho_{i_1\cdots i_n,j_1\cdots j_n}
\]

by

\[
\rho=
\sum
W^{[1]}W^{[2]}\cdots W^{[n]}.
\]

If you combine the ket/bra indices,

\[
(i_k,j_k)\rightarrow s_k,
\]

then every qubit has local physical dimension

\[
d_{\rm MPO}=2^2=4.
\]

So mathematically it's almost exactly an MPS with physical dimension 4.

The crucial quantity becomes the **operator bond dimension**

\[
\chi.
\]

Instead of \(4^n\), storage becomes roughly

\[
O(n\,4\,\chi^2).
\]

---

# 13. Applying noise to the MPO

This is probably the nicest implementation detail.

Vectorize

\[
\rho\rightarrow|\rho\rangle\rangle.
\]

Then

\[
K\rho K^\dagger
\]

becomes, depending on your vectorization convention,

\[
(K^*\otimes K)|\rho\rangle\rangle.
\]

Consequently a complete channel becomes

\[
|\rho'\rangle\rangle
=
\left[
\sum_k K_k^*\otimes K_k
\right]
|\rho\rangle\rangle.
\]

Define

\[
S_{\mathcal E}
=
\sum_kK_k^*\otimes K_k.
\]

A single-qubit channel is therefore just a

\[
4\times4
\]

matrix acting on an MPO physical index.

For two-qubit noise it is

\[
16\times16.
\]

This is extremely convenient.

Your simulator architecture could essentially have

```cpp
ApplyChannel(qubits, SuperOperator S);
```

where

\[
S=\sum_k K_k^*\otimes K_k.
\]

Then the distinction between gates and noise almost disappears.

For a unitary,

\[
S_U=U^*\otimes U.
\]

For a general channel,

\[
S_{\mathcal E}
=\sum_k K_k^*\otimes K_k.
\]

The same MPO machinery handles both.

---

# 14. Pauli-transfer representation may be even nicer

Since you already have Pauli-propagation machinery, I'd strongly consider allowing channels to be represented in the Pauli basis.

For

\[
P=(I,X,Y,Z),
\]

define

\[
R_{ij}
=
\frac12
\operatorname{Tr}
\left[
P_i\,\mathcal E(P_j)
\right].
\]

Then

\[
\begin{pmatrix}
I\\X\\Y\\Z
\end{pmatrix}
\rightarrow
R
\begin{pmatrix}
I\\X\\Y\\Z
\end{pmatrix}.
\]

For dephasing this is basically diagonal:

\[
R_{\rm deph}
=
\begin{pmatrix}
1&0&0&0\\
0&\lambda&0&0\\
0&0&\lambda&0\\
0&0&0&1
\end{pmatrix}.
\]

Depolarizing gives

\[
R_{\rm depol}
=
\operatorname{diag}(1,\lambda,\lambda,\lambda).
\]

Amplitude damping is more interesting because it is **non-unital**:

\[
I\rightarrow I+\gamma Z,
\]

up to convention/direction.

This is actually a useful conceptual connection between your **Pauli-propagation** and **MPO** simulators: they can share essentially the same channel representation.

---

# 15. Density matrix vs stochastic statevector

Suppose

\[
\mathcal E(\rho)
=
\sum_kK_k\rho K_k^\dagger.
\]

A statevector trajectory simulator chooses \(k\) randomly according to

\[
p_k=
\langle\psi|
K_k^\dagger K_k
|\psi\rangle
\]

and evolves

\[
|\psi\rangle
\rightarrow
\frac{K_k|\psi\rangle}
{\sqrt{p_k}}.
\]

Repeat this \(N\) times:

\[
\rho\approx
\frac1N
\sum_{r=1}^N
|\psi_r\rangle\langle\psi_r|.
\]

So:

### Statevector/MPS trajectories

Memory per trajectory:

\[
O(2^n)
\]

or MPS-sized.

But statistical error scales approximately as

\[
O(N^{-1/2}).
\]

### Density matrix/MPO

You get the ensemble result directly:

\[
\rho=\mathbb E[
|\psi\rangle\langle\psi|
].
\]

No trajectory sampling variance.

That's probably the most important practical advantage.

If you're calculating many expectation values,

\[
\langle O\rangle=\operatorname{Tr}(\rho O),
\]

you've already captured the entire ensemble.

---

# 16. An important—and slightly surprising—MPO advantage: noise can make simulation easier

For a noiseless highly entangling circuit, MPS bond dimensions can explode.

For an MPO describing a pure state

\[
\rho=|\psi\rangle\langle\psi|,
\]

if the MPS has Schmidt rank \(D\), then the corresponding MPO can require roughly

\[
\chi\sim D^2.
\]

So **at zero noise an MPO can actually be substantially worse than your MPS simulator**.

But noise destroys quantum correlations.

Strong enough decoherence suppresses long-range operator entanglement, meaning that the MPO bond dimension may saturate or even shrink.

This leads to an interesting phenomenon:

\[
\boxed{\text{a sufficiently noisy quantum circuit can be easier to simulate classically than the ideal circuit}}
\]

Cheng et al. explicitly observed this for dephasing, depolarizing, and amplitude-damping noise: at sufficiently strong noise, relatively small tensor-network bond dimensions can describe the circuit accurately [2]. 

This is one of the strongest motivations for your MPO simulator.

---

# 17. But MPO truncation has a subtle problem: positivity

This is worth paying attention to in your implementation.

For an exact density matrix,

\[
\rho\succeq0.
\]

If you perform ordinary SVD truncation on a generic MPO,

\[
\rho\rightarrow\tilde\rho,
\]

you preserve neither automatically:

\[
\tilde\rho\succeq0
\]

nor necessarily

\[
\operatorname{Tr}\tilde\rho=1.
\]

Trace is easy to renormalize.

Positivity is harder.

Guo and Yang found that ordinary MPO representations can capture mixed-state correlations well, but MPO truncation can break positivity [4]. 

This is why you see names such as:

- MPDO — matrix product density operator;
- LPDO — locally purified density operator;
- purified MPO.

Instead of representing

\[
\rho
\]

directly, they represent

\[
\rho=XX^\dagger.
\]

Then positivity is automatic.

There is a price: you introduce an additional **Kraus/environment dimension**.

For a first implementation, though, I would absolutely keep your ordinary MPO. Just monitor things such as

\[
\operatorname{Tr}\rho,
\]

Hermiticity,

\[
\|\rho-\rho^\dagger\|,
\]

and perhaps the smallest eigenvalue for small systems where you can reconstruct the dense matrix.

---

# 18. How your simulators compare

| Simulator | State represented | Noise | Main advantage | Main problem |
|---|---|---|---|---|
| Statevector | \(|\psi\rangle\) | trajectories | exact pure-state evolution | \(2^n\), sampling for noise |
| MPS | \(|\psi\rangle\) | trajectories | large \(n\) if low entanglement | trajectory sampling + bond growth |
| Density matrix | \(\rho\) | native | exact mixed state, no noise sampling | \(4^n\) |
| **MPO** | \(\rho\) | **native** | compressed mixed state | operator bond growth, positivity after truncation |
| Pauli propagation | observables/operators | often native | excellent for selected observables | not usually full-state access |
| Path integral | amplitudes/paths | possible via environment/influence functional | alternative structure/exploitation | open-system treatment is considerably more involved |

And there is an interesting correction concerning your Pauli-propagation simulator specifically. Recent work shows that Pauli propagation with suitable truncation can simulate arbitrary **local incoherent noise**, including non-unital amplitude damping, efficiently for broad classes of typical circuits [3]. 

So I wouldn't classify your Pauli-propagation simulator as intrinsically unitary-only. Its channel support can actually be quite elegant.

---

# 19. Where DM/MPO really win

I would summarize their unique practical appeal this way.

Suppose the computation is

\[
\rho_0
\xrightarrow{\mathcal E_1}
\rho_1
\xrightarrow{\mathcal E_2}
\cdots
\xrightarrow{\mathcal E_D}
\rho_D.
\]

Your DM/MPO simulator gives you exactly this open-system evolution.

You can then directly calculate:

\[
\operatorname{Tr}(\rho O),
\]

purity

\[
\operatorname{Tr}(\rho^2),
\]

fidelity,

\[
F(\rho,\sigma),
\]

entropy,

\[
S(\rho)=-\operatorname{Tr}(\rho\log\rho),
\]

reduced density matrices,

\[
\rho_A=\operatorname{Tr}_B\rho,
\]

mutual information, noise-induced correlations, channel fidelity, etc.

A trajectory simulator can ultimately obtain most of these, but often only after accumulating a large ensemble.

---

# 20. Markovian versus non-Markovian noise

Everything we've discussed so far implicitly assumes

\[
\rho_{t+1}=\mathcal E_t(\rho_t).
\]

That is essentially Markovian: the present density matrix contains everything needed to predict the next one.

A real environment can instead retain information:

\[
S\leftrightarrow E
\]

and feed it back later.

Then there may be **no CPTP map acting only on the system from each step to the next**.

One way to simulate this is simply to enlarge the state:

\[
\rho_S
\rightarrow
\rho_{SE}.
\]

Another more sophisticated approach is a **process tensor**, which can itself be represented as an MPO. MPO-based process-tensor approaches have been used for long-time non-Markovian open-system evolution [5]. 

That's probably a later project, but it fits extremely naturally with the infrastructure you're building.

---

# 21. What I would add to your implementation

Given the simulators you now have, I'd probably make the central abstraction something like

\[
\boxed{\texttt{QuantumChannel}}
\]

with alternative representations:

\[
\begin{aligned}
&\text{Kraus:} &&\{K_i\},\\
&\text{Superoperator:} &&S=\sum_iK_i^*\otimes K_i,\\
&\text{Pauli transfer:} &&R_{ij},\\
&\text{Lindbladian:} &&\mathcal L.
\end{aligned}
\]

Then initially implement:

1. bit flip;
2. phase flip;
3. depolarizing;
4. amplitude damping;
5. phase damping;
6. \(T_1/T_2\) thermal relaxation;
7. generalized amplitude damping;
8. arbitrary Kraus channel;
9. arbitrary Pauli channel;
10. measurement/readout error;
11. two-qubit correlated Pauli noise.

That would already make the simulator quite serious.

For the dense DM simulator,

\[
\rho\leftarrow
\sum_iK_i\rho K_i^\dagger.
\]

For the MPO,

\[
W\leftarrow
\left(
\sum_iK_i^*\otimes K_i
\right)W
\]

on the appropriate physical leg, followed by SVD compression where required.

And for your Pauli-propagation simulator, convert the same channel to its PTM and propagate the Pauli coefficients. That would let **all three simulators share the same noise-model library**, which I think would be a particularly clean architecture.

### Research references

[1] [Modeling and simulating the noisy behavior of near-term quantum computers](https://consensus.app/papers/modeling-and-simulating-the-noisy-behavior-of-nearterm-georgopoulos-emary/a00b8bcaddcb5773bf571c2db3d28553/?utm_source=chatgpt) — K. Georgopoulos, C. Emary & P. Zuliani, 2021, *Physical Review A*, 139 citations. 

[2] [Simulating Noisy Quantum Circuits with Matrix Product Density Operators](https://consensus.app/papers/simulating-noisy-quantum-circuits-with-matrix-product-cheng-cao/095e8b268a3053538c29bac368bcd64b/?utm_source=chatgpt) — Song Cheng, Chenfeng Cao, Chao Zhang, Yongxiang Liu, Shi-Yao Hou, Pengxiang Xu & B. Zeng, 2020, *arXiv* abs/2004.02388, 52 citations. 

[3] [Simulating quantum circuits with arbitrary local noise using Pauli Propagation](https://consensus.app/papers/simulating-quantum-circuits-with-arbitrary-local-noise-angrisani-mele/4ad379450ead5a1ba69858c8db2ea926/?utm_source=chatgpt) — Armando Angrisani, A. A. Mele, M. S. Rudolph, M. Cerezo & Zoe Holmes, 2025, *PRX Quantum*, 56 citations. 

[4] [Locally purified density operators for noisy quantum circuits](https://consensus.app/papers/locally-purified-density-operators-for-noisy-quantum-guo-yang/1132e3efb19a5f4480d839e1789971cf/?utm_source=chatgpt) — Yuchen Guo & Shuo Yang, 2023, *Chinese Physics Letters* 41, 12 citations. 

[5] [Open Quantum System Dynamics from Infinite Tensor Network Contraction.](https://consensus.app/papers/open-quantum-system-dynamics-from-infinite-tensor-network-link-tu/f3740f1dc82c574db9d900a306cfb705/?utm_source=chatgpt) — Valentin Link, Hong-Hao Tu & W. Strunz, 2023, *Physical Review Letters* 132, 200403, 70 citations. 

The part I think would be most useful to dig into next is the **actual MPO update for a Kraus channel**—i.e. starting from your \(W^{[i]}_{\alpha,(s,s'),\beta}\), deriving explicitly how a one- or two-qubit CPTP map changes the physical tensor and how the bond dimension grows. It connects almost directly to the MPS/MPO contraction code you've already been working with. 

Upgrade to Consensus Pro to return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm_source=chatgpt

