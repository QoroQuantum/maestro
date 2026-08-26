Yes. The density-matrix formalism is essentially the natural language of a **quantum computer viewed as an open quantum system**. Statevectors are enough for isolated systems in pure states; density matrices let you describe mixtures, discarded subsystems, decoherence, relaxation, stochastic errors, measurement without conditioning on an outcome, and general coupling to an environment.

## 1. From statevectors to density matrices

For a pure state

\[
|\psi\rangle = \sum_i c_i |i\rangle,
\]

define the density operator

\[
\rho = |\psi\rangle\langle\psi|.
\]

For example, for one qubit

\[
|\psi\rangle = \alpha |0\rangle+\beta |1\rangle,
\]

we get

\[
\rho=
\begin{pmatrix}
|\alpha|^2 & \alpha\beta^*\\
\alpha^*\beta & |\beta|^2
\end{pmatrix}.
\]

The diagonal elements are populations; the off-diagonal elements are coherences.

But density matrices can represent something a single statevector cannot: a **statistical mixture**. If

\[
|\psi_i\rangle
\]

occurs with classical probability \(p_i\), then

\[
\boxed{
\rho=\sum_i p_i|\psi_i\rangle\langle\psi_i|
}
\]

with

\[
p_i\ge 0,\qquad \sum_i p_i=1.
\]

A valid density matrix therefore satisfies

\[
\rho^\dagger=\rho,\qquad
\rho\ge0,\qquad
\operatorname{Tr}\rho=1.
\]

For a pure state,

\[
\rho^2=\rho,
\]

and consequently

\[
\operatorname{Tr}(\rho^2)=1.
\]

For a mixed state,

\[
\operatorname{Tr}(\rho^2)<1.
\]

This quantity,

\[
\mathcal P=\operatorname{Tr}(\rho^2),
\]

is the **purity**.

For \(n\) qubits:

- statevector: \(2^n\) complex numbers
- density matrix: \(2^n\times2^n=4^n\) complex numbers.

That factor is the fundamental price of exact mixed-state simulation.

---

# 2. Ordinary gates are still simple

A unitary gate acts on a statevector as

\[
|\psi'\rangle=U|\psi\rangle.
\]

Therefore

\[
\rho'
=
|\psi'\rangle\langle\psi'|
=
U|\psi\rangle\langle\psi|U^\dagger,
\]

so

\[
\boxed{\rho' = U\rho U^\dagger}.
\]

Thus density-matrix simulation contains statevector simulation as a special case.

If initially

\[
\rho=|\psi\rangle\langle\psi|,
\]

unitary evolution keeps it pure:

\[
\operatorname{Tr}[(U\rho U^\dagger)^2]
=
\operatorname{Tr}(\rho^2)=1.
\]

The interesting difference appears once the system is no longer closed.

---

# 3. Why coupling to an environment produces non-unitary evolution

Suppose your quantum computer is a system \(S\), while the environment is \(E\).

The complete system still follows ordinary unitary quantum mechanics:

\[
\rho_{SE}' =
U_{SE}\rho_{SE}U_{SE}^\dagger.
\]

But you do not observe the environment.

The state of the quantum computer alone is therefore

\[
\boxed{
\rho_S' =
\operatorname{Tr}_E(\rho_{SE}')
}.
\]

This partial trace is the fundamental origin of open-system, apparently non-unitary evolution.

Assume initially

\[
\rho_{SE}=\rho_S\otimes\rho_E.
\]

Then

\[
\boxed{
\rho'_S=
\operatorname{Tr}_E
\left[
U_{SE}
(\rho_S\otimes\rho_E)
U_{SE}^\dagger
\right].
}
\]

Even though \(U_{SE}\) is perfectly unitary, there generally exists **no unitary \(U_S\)** such that

\[
\rho'_S=U_S\rho_SU_S^\dagger.
\]

Information has escaped into correlations with \(E\).

---

# 4. Entanglement itself makes subsystems mixed

This is worth emphasizing because it is perhaps the cleanest conceptual example.

Take the Bell state

\[
|\Psi\rangle
=
\frac{|00\rangle+|11\rangle}{\sqrt2}.
\]

The complete density matrix is pure:

\[
\rho_{AB}=|\Psi\rangle\langle\Psi|.
\]

But if qubit \(B\) is inaccessible,

\[
\rho_A=\operatorname{Tr}_B\rho_{AB}
=
\frac12
\begin{pmatrix}
1&0\\
0&1
\end{pmatrix}
=
\frac{I}{2}.
\]

Thus

\[
\operatorname{Tr}\rho_A^2=\frac12.
\]

The subsystem is maximally mixed even though the complete universe is in a pure state.

This is exactly the sense in which an environment produces decoherence: information formerly present as local phase coherence becomes encoded in correlations between the system and environment.

---

# 5. From system+environment evolution to Kraus operators

This leads directly to the formalism most useful for a density-matrix simulator.

Suppose initially the environment is

\[
|e_0\rangle.
\]

Then

\[
\rho'_S =
\operatorname{Tr}_E
\left[
U(\rho_S\otimes |e_0\rangle\langle e_0|)U^\dagger
\right].
\]

Insert an orthonormal basis \(\{|e_k\rangle\}\) for the environment:

\[
\rho'_S
=
\sum_k
\langle e_k|
U
(\rho_S\otimes|e_0\rangle\langle e_0|)
U^\dagger
|e_k\rangle.
\]

Define

\[
K_k=\langle e_k|U|e_0\rangle.
\]

Then

\[
\boxed{
\rho'=\sum_k K_k\rho K_k^\dagger.
}
\]

These \(K_k\) are **Kraus operators**.

For a trace-preserving process,

\[
\boxed{
\sum_k K_k^\dagger K_k=I.
}
\]

This is probably the single most important equation for implementing noise in a density-matrix simulator.

Instead of

```text
rho <- U rho U†
```

you have

```text
rho_new = 0

for K in KrausOperators:
    rho_new += K rho K†
```

The first operation is simply the special case with one Kraus operator

\[
K_0=U.
\]

---

# 6. Quantum channels: CPTP maps

The general deterministic evolution of an open quantum system is written

\[
\rho'=\mathcal E(\rho),
\]

where \(\mathcal E\) is a **quantum channel**.

Physical deterministic channels must normally be:

- linear;
- trace preserving;
- completely positive.

Hence the abbreviation **CPTP**.

The Kraus form

\[
\mathcal E(\rho)=
\sum_k K_k\rho K_k^\dagger
\]

automatically guarantees complete positivity.

The condition

\[
\sum_k K_k^\dagger K_k=I
\]

guarantees trace preservation.

### Why "complete" positivity?

Positivity alone means

\[
\rho\ge0 \quad\Longrightarrow\quad \mathcal E(\rho)\ge0.
\]

But the qubit being transformed may be entangled with another untouched system.

Therefore we require

\[
(\mathcal E\otimes I)(\rho_{SA})\ge0
\]

for arbitrary ancillary system \(A\).

That is **complete positivity**.

This distinction matters in open-system theory because some apparently reasonable linear maps, such as matrix transposition, are positive but not completely positive.

---

# 7. Noise as quantum channels

Now many familiar hardware errors become very straightforward.

## Bit-flip noise

With probability \(p\), apply \(X\):

\[
\boxed{
\mathcal E(\rho)
=
(1-p)\rho+pX\rho X.
}
\]

Kraus operators:

\[
K_0=\sqrt{1-p}\,I,
\qquad
K_1=\sqrt p\,X.
\]

This is a stochastic unitary channel.

---

## Phase-flip noise

\[
\mathcal E(\rho)
=
(1-p)\rho+pZ\rho Z.
\]

For

\[
\rho=
\begin{pmatrix}
a&c\\
c^*&b
\end{pmatrix},
\]

we obtain

\[
\rho'=
\begin{pmatrix}
a&(1-2p)c\\
(1-2p)c^*&b
\end{pmatrix}.
\]

Notice that populations are unchanged while coherence disappears.

This is the essence of dephasing.

---

# 8. General Pauli channel

A very convenient quantum-computing noise model is

\[
\boxed{
\mathcal E(\rho)=
p_I\rho+
p_X X\rho X+
p_Y Y\rho Y+
p_Z Z\rho Z
}
\]

where

\[
p_I+p_X+p_Y+p_Z=1.
\]

Its Kraus operators are simply

\[
\sqrt{p_I}I,\quad
\sqrt{p_X}X,\quad
\sqrt{p_Y}Y,\quad
\sqrt{p_Z}Z.
\]

For multiple qubits you can similarly have

\[
P\in\{I,X,Y,Z\}^{\otimes n}
\]

and

\[
\mathcal E(\rho)
=
\sum_P p_P P\rho P.
\]

This makes Pauli noise particularly simple to simulate.

It is also why Pauli twirling is attractive: a complicated noise process can sometimes be approximated by a stochastic Pauli channel.

---

# 9. Depolarizing noise

One common convention is

\[
\boxed{
\mathcal E(\rho)
=
(1-p)\rho
+
p\,\frac{I}{2}.
}
\]

So with increasing \(p\), every input state moves towards the center of the Bloch sphere.

Another convention defines depolarization as equal probabilities for \(X,Y,Z\):

\[
\mathcal E(\rho)
=
(1-p)\rho
+
\frac p3
(X\rho X+Y\rho Y+Z\rho Z).
\]

These conventions use different definitions of \(p\), so one has to be careful when implementing APIs.

---

# 10. Amplitude damping: an important genuinely non-unital channel

Amplitude damping models energy relaxation,

\[
|1\rangle\rightarrow|0\rangle.
\]

Its Kraus operators are

\[
K_0=
\begin{pmatrix}
1&0\\
0&\sqrt{1-\gamma}
\end{pmatrix},
\qquad
K_1=
\begin{pmatrix}
0&\sqrt{\gamma}\\
0&0
\end{pmatrix}.
\]

Thus

\[
\mathcal E(\rho)=
K_0\rho K_0^\dagger+
K_1\rho K_1^\dagger.
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
\rho'=
\begin{pmatrix}
\rho_{00}+\gamma\rho_{11}
&
\sqrt{1-\gamma}\rho_{01}
\\
\sqrt{1-\gamma}\rho_{10}
&
(1-\gamma)\rho_{11}
\end{pmatrix}.
\]

So

\[
\rho_{11}\rightarrow(1-\gamma)\rho_{11},
\]

and the lost excited-state population appears in \(|0\rangle\).

For a relaxation time \(T_1\),

\[
\gamma(t)=1-e^{-t/T_1}.
\]

---

# 11. Deriving amplitude damping from an environment

This gives a particularly nice demonstration of how apparently non-unitary dynamics emerges.

Introduce an environment initially in

\[
|0_E\rangle.
\]

Let the joint unitary behave as

\[
|0\rangle|0_E\rangle
\rightarrow
|0\rangle|0_E\rangle,
\]

and

\[
|1\rangle|0_E\rangle
\rightarrow
\sqrt{1-\gamma}|1\rangle|0_E\rangle
+
\sqrt{\gamma}|0\rangle|1_E\rangle.
\]

An excitation has either remained in the qubit or escaped into the environment.

If

\[
|\psi\rangle=\alpha|0\rangle+\beta|1\rangle,
\]

the combined state becomes

\[
\alpha|0,0_E\rangle+
\beta\sqrt{1-\gamma}|1,0_E\rangle+
\beta\sqrt{\gamma}|0,1_E\rangle.
\]

If you trace out the environment, the two environmental alternatives cannot interfere, and you obtain precisely the amplitude-damping density matrix.

The corresponding Kraus operators are

\[
K_0=\langle0_E|U|0_E\rangle,
\qquad
K_1=\langle1_E|U|0_E\rangle.
\]

This is a good microscopic interpretation of Kraus operators: **each Kraus operator corresponds roughly to an alternative thing that happened in the inaccessible environment**.

---

# 12. \(T_1\) versus \(T_2\)

For actual qubits, two important quantities are

\[
T_1 = \text{energy relaxation time}
\]

and

\[
T_2 = \text{coherence time}.
\]

Amplitude damping produces \(T_1\) relaxation.

Independent pure dephasing gives a timescale \(T_\phi\).

For the usual Markovian model,

\[
\boxed{
\frac1{T_2}
=
\frac1{2T_1}
+
\frac1{T_\phi}.
}
\]

Thus even with no additional phase noise,

\[
T_2\le2T_1.
\]

In a gate simulator, for a gate taking time \(t_g\), one often obtains parameters such as

\[
\gamma_1
=
1-e^{-t_g/T_1},
\]

and coherence decay factors involving

\[
e^{-t_g/T_2}.
\]

So gate duration can directly determine the channel strength.

---

# 13. Thermal relaxation

Ordinary amplitude damping assumes the environment is effectively at zero temperature:

\[
|1\rangle\rightarrow |0\rangle
\]

but not vice versa.

At finite temperature you want **generalized amplitude damping**, where both

\[
|1\rangle\rightarrow|0\rangle
\]

and

\[
|0\rangle\rightarrow|1\rangle
\]

are possible.

The stationary density matrix is then thermal rather than

\[
|0\rangle\langle0|.
\]

This is relevant when modeling hardware where the equilibrium excited-state population is nonzero.

---

# 14. Reset is itself a non-unitary quantum channel

A perfect reset

\[
\rho\rightarrow|0\rangle\langle0|
\]

cannot be implemented by a unitary on the qubit alone because different initial states all map onto the same state.

A Kraus representation is

\[
K_0=|0\rangle\langle0|,
\qquad
K_1=|0\rangle\langle1|.
\]

Then

\[
K_0\rho K_0^\dagger+
K_1\rho K_1^\dagger
=
|0\rangle\langle0|\operatorname{Tr}\rho
=
|0\rangle\langle0|.
\]

This is a good example of why reset is naturally described with density matrices/open-system theory.

Physically, reset has dumped information into an environment.

---

# 15. Measurement fits naturally into the same framework

For a projective measurement with projectors \(P_m\),

\[
p_m=\operatorname{Tr}(P_m\rho).
\]

If you know that outcome \(m\) occurred,

\[
\boxed{
\rho_m=
\frac{P_m\rho P_m}{p_m}.
}
\]

But suppose you perform the measurement and throw away its result.

Then

\[
\boxed{
\rho'
=
\sum_mP_m\rho P_m.
}
\]

For measurement in the computational basis,

\[
\rho=
\begin{pmatrix}
a&c\\
c^*&b
\end{pmatrix}
\]

becomes

\[
\rho'=
\begin{pmatrix}
a&0\\
0&b
\end{pmatrix}.
\]

So an unread measurement is effectively complete dephasing.

This illustrates an important distinction.

The map

\[
\rho\rightarrow P_m\rho P_m
\]

for one particular outcome is **trace decreasing**:

\[
\operatorname{Tr}(P_m\rho P_m)=p_m.
\]

The normalized map

\[
\rho\rightarrow
\frac{P_m\rho P_m}{p_m}
\]

is nonlinear because you are conditioning on acquired classical information.

The sum over all branches is again linear and trace preserving.

---

# 16. Generalized measurements

More generally, measurement operators \(M_m\) obey

\[
\sum_mM_m^\dagger M_m=I.
\]

Then

\[
p_m=
\operatorname{Tr}
(M_m\rho M_m^\dagger)
=
\operatorname{Tr}(E_m\rho),
\]

where

\[
E_m=M_m^\dagger M_m
\]

are POVM elements.

The post-measurement state is

\[
\rho_m=
\frac{
M_m\rho M_m^\dagger
}{
p_m
}.
\]

So Kraus operators, noise and generalized measurement are really parts of the same mathematical structure.

---

# 17. Lindblad evolution: continuous-time open systems

Kraus channels are particularly convenient for gate-based simulation:

\[
\rho_{t+\Delta t}=\mathcal E_{\Delta t}(\rho_t).
\]

For continuous-time evolution, the standard Markovian model is the **Lindblad master equation**:

\[
\boxed{
\frac{d\rho}{dt}
=
-i[H,\rho]
+
\sum_k\gamma_k
\left[
L_k\rho L_k^\dagger
-
\frac12
\{L_k^\dagger L_k,\rho\}
\right].
}
\]

The first term,

\[
-i[H,\rho],
\]

is ordinary Hamiltonian evolution.

The remaining terms describe irreversible interaction with the environment.

The \(L_k\) are often called:

- Lindblad operators,
- jump operators,
- collapse operators.

---

# 18. Example: relaxation from Lindblad theory

For energy relaxation use

\[
L=\sigma_- = |0\rangle\langle1|.
\]

Then

\[
\frac{d\rho}{dt}
=
\gamma
\left[
\sigma_-\rho\sigma_+
-
\frac12
\{\sigma_+\sigma_-,\rho\}
\right].
\]

This gives

\[
\rho_{11}(t)
=
e^{-\gamma t}\rho_{11}(0),
\]

so

\[
T_1=\frac1\gamma.
\]

The off-diagonal component decays as

\[
\rho_{01}(t)
=
e^{-\gamma t/2}\rho_{01}(0)
\]

if relaxation is the only source of decoherence.

This is where the \(2T_1\) limit on \(T_2\) originates.

---

# 19. Pure dephasing in Lindblad form

For example, choosing a suitable rate with

\[
L=Z
\]

produces

\[
\dot\rho_{01}
=
-\Gamma_\phi\rho_{01}
\]

while leaving the populations unchanged.

Therefore

\[
\rho_{01}(t)
=
e^{-t/T_\phi}\rho_{01}(0).
\]

This gives you a very natural way to specify physically motivated continuous noise rather than arbitrary per-gate error probabilities.

---

# 20. Liouville-space representation

There is another representation that becomes useful when writing simulators.

Vectorize the density matrix:

\[
\rho
\quad\longrightarrow\quad
|\rho\rangle\rangle.
\]

For an \(N\times N\) density matrix, this vector has dimension

\[
N^2.
\]

For \(n\) qubits,

\[
N=2^n
\]

and therefore

\[
|\rho\rangle\rangle
\in\mathbb C^{4^n}.
\]

Using column vectorization,

\[
\operatorname{vec}(A\rho B^\dagger)
=
(B^*\otimes A)\operatorname{vec}(\rho).
\]

Thus a Kraus channel becomes

\[
|\rho'\rangle\rangle
=
\left(
\sum_k K_k^*\otimes K_k
\right)
|\rho\rangle\rangle.
\]

Define the **superoperator**

\[
\boxed{
S_{\mathcal E}
=
\sum_kK_k^*\otimes K_k.
}
\]

Then

\[
|\rho'\rangle\rangle
=
S_{\mathcal E}|\rho\rangle\rangle.
\]

For a unitary,

\[
S_U=U^*\otimes U.
\]

This reveals an interesting correspondence:

> Density-matrix evolution is formally statevector evolution in a \(4^n\)-dimensional Liouville space.

This viewpoint connects especially naturally with an MPO representation.

---

# 21. But don't normally construct the full superoperator

There is an important implementation detail here.

\[
|\rho\rangle\rangle
\]

contains \(4^n\) numbers.

But the complete matrix \(S\) would contain

\[
(4^n)^2=16^n
\]

numbers.

So for simulation you generally do **not** form a global superoperator.

Instead you apply:

- local gates;
- local Kraus operators;
- local Lindblad terms;

directly to the corresponding tensor indices.

Exactly the same principle as avoiding construction of a \(2^n\times2^n\) full gate matrix in a statevector simulator.

---

# 22. Bloch-sphere interpretation

For a single qubit,

\[
\rho
=
\frac12
\left(
I+xX+yY+zZ
\right).
\]

Equivalently,

\[
\rho=\frac12(I+\mathbf r\cdot\boldsymbol\sigma).
\]

Pure states satisfy

\[
|\mathbf r|=1.
\]

Mixed states satisfy

\[
|\mathbf r|<1.
\]

The maximally mixed state has

\[
\mathbf r=0.
\]

A general qubit channel acts affinely:

\[
\boxed{
\mathbf r'
=
A\mathbf r+\mathbf c.
}
\]

Unitary evolution corresponds to a rotation:

\[
\mathbf r'=R\mathbf r.
\]

Dephasing squashes the Bloch sphere in the \(x,y\) directions.

Depolarization shrinks it isotropically towards the origin.

Amplitude damping both shrinks it **and moves its center towards \(|0\rangle\)**.

That latter distinction leads to another useful concept.

---

# 23. Unital versus non-unital noise

A channel is unital if

\[
\mathcal E(I)=I.
\]

Equivalently,

\[
\mathcal E\left(\frac I2\right)=\frac I2.
\]

Examples:

- bit flip: unital
- phase flip: unital
- Pauli channel: unital
- depolarizing: unital
- amplitude damping: **not unital**

Amplitude damping drives everything towards

\[
|0\rangle\langle0|,
\]

not toward the maximally mixed state.

This makes relaxation qualitatively different from stochastic Pauli errors.

---

# 24. Coherent errors do not actually require density matrices

This distinction is useful in simulator design.

Suppose a requested gate is

\[
R_x(\theta)
\]

but the hardware implements

\[
R_x(\theta+\epsilon).
\]

That is a coherent calibration error.

It is still unitary, and therefore can be represented perfectly by a statevector simulator.

Likewise coherent crosstalk such as

\[
e^{-i\epsilon Z_iZ_j}
\]

is still unitary.

Density matrices become necessary when you want the **ensemble-averaged effect of processes that cannot be represented by one system-only unitary**:

- relaxation;
- dephasing;
- stochastic noise;
- discarded measurements;
- reset;
- thermalization;
- coupling to inaccessible degrees of freedom.

---

# 25. Strictly speaking, statevector simulators can simulate noise too

There is a subtle but important qualification.

Suppose

\[
\mathcal E(\rho)
=
\sum_kK_k\rho K_k^\dagger.
\]

Starting with a pure state \(|\psi\rangle\), define

\[
|\tilde\psi_k\rangle=K_k|\psi\rangle.
\]

The probability of branch \(k\) is

\[
p_k
=
\langle\tilde\psi_k|\tilde\psi_k\rangle
=
\langle\psi|K_k^\dagger K_k|\psi\rangle.
\]

Choose \(k\) randomly and update

\[
|\psi\rangle
\rightarrow
\frac{K_k|\psi\rangle}{\sqrt{p_k}}.
\]

Repeating many trajectories gives

\[
\rho
\simeq
\frac1M
\sum_{j=1}^M
|\psi_j\rangle\langle\psi_j|.
\]

As

\[
M\rightarrow\infty,
\]

this converges to the density-matrix result.

This is the **quantum trajectory / Monte Carlo wavefunction** approach.

So the distinction is not:

> statevector = impossible to simulate noise.

It is rather:

### Density matrix

Computes the ensemble average exactly:

\[
\rho'=\sum_kK_k\rho K_k^\dagger.
\]

### Trajectory statevector

Samples one environmental history:

\[
|\psi\rangle\rightarrow
K_k|\psi\rangle/\sqrt{p_k}.
\]

Many trajectories are necessary for statistics.

---

# 26. Density matrix versus trajectories

There is therefore an important computational tradeoff.

For \(n\) qubits:

\[
\text{statevector memory}\sim O(2^n)
\]

whereas

\[
\text{density matrix memory}\sim O(4^n).
\]

With \(M\) trajectories,

\[
\text{cost}\sim M\,O(2^n).
\]

Roughly speaking:

- small/medium \(n\), many observables or high precision → density matrix is attractive;
- larger \(n\), modest statistical precision → trajectories may win;
- highly structured systems → MPS/MPO methods may win.

Density matrices also avoid Monte Carlo statistical noise.

---

# 27. Lindblad evolution and trajectories are connected

The Lindblad equation can itself be unraveled into quantum trajectories.

Define the effective non-Hermitian Hamiltonian

\[
H_{\rm eff}
=
H-
\frac{i}{2}
\sum_k\gamma_kL_k^\dagger L_k.
\]

Between jumps,

\[
|\psi(t+\Delta t)\rangle
\approx
e^{-iH_{\rm eff}\Delta t}|\psi(t)\rangle.
\]

Occasionally a quantum jump occurs,

\[
|\psi\rangle
\rightarrow
L_k|\psi\rangle.
\]

Averaging all such trajectories produces the Lindblad density matrix.

The apparent non-unitarity of

\[
H_{\rm eff}
\]

represents the decreasing probability that no environmental jump has occurred.

This is another nice theoretical connection between pure-state and density-matrix simulation.

---

# 28. Explicit environment simulation versus density-matrix noise

You actually have two conceptually different ways of modeling the environment.

### Explicit environment

Simulate

\[
S+E
\]

as one large statevector:

\[
|\Psi_{SE}\rangle.
\]

Apply joint unitary evolution.

Then compute

\[
\rho_S=\operatorname{Tr}_E|\Psi_{SE}\rangle\langle\Psi_{SE}|.
\]

Advantages:

- microscopic model;
- keeps system-environment correlations;
- naturally handles memory effects;
- can model non-Markovian behavior.

Disadvantage:

\[
\dim(\mathcal H_S\otimes\mathcal H_E)
=
\dim\mathcal H_S\,\dim\mathcal H_E.
\]

The environment can become enormous.

### Reduced density-matrix simulation

Replace the environment by an effective channel

\[
\rho_S\rightarrow\mathcal E(\rho_S).
\]

Much cheaper and usually what circuit noise simulators do.

But you have discarded environmental state and memory.

---

# 29. Markovian versus non-Markovian noise

Most gate-level simulators implicitly assume something like

\[
\rho_{k+1}
=
\mathcal E_k(\rho_k).
\]

The environment is effectively reset or forgotten after every step.

This is approximately **Markovian**:

\[
\text{future evolution}
\approx
\text{function of current }\rho,
\]

rather than its full history.

But suppose the same environmental degree of freedom interacts with the qubit repeatedly.

Then information can flow

\[
S\rightarrow E\rightarrow S.
\]

Now knowing \(\rho_S(t)\) may not be enough to predict \(\rho_S(t+\Delta t)\).

You may have to preserve:

- explicit environmental modes;
- system-environment correlations;
- a memory kernel;
- an enlarged effective state.

A simple sequence of independent Kraus channels then does not capture the physics.

---

# 30. A subtle assumption behind ordinary quantum channels

When deriving

\[
\rho'_S=\mathcal E(\rho_S),
\]

one generally assumes initially

\[
\rho_{SE}
=
\rho_S\otimes\rho_E.
\]

If \(S\) and \(E\) are already strongly correlated,

\[
\rho_{SE}\ne\rho_S\otimes\rho_E,
\]

then in general there need not exist one CPTP map

\[
\mathcal E
\]

that correctly acts on every possible \(\rho_S\).

This doesn't matter much for ordinary circuit noise simulation, but it becomes important if you want to model genuinely non-Markovian environments.

---

# 31. Density matrices distinguish classical and quantum loss of information—but not by themselves

Consider

\[
\rho=\frac12|0\rangle\langle0|
+
\frac12|1\rangle\langle1|
=
\frac I2.
\]

This could mean:

1. someone prepared \(|0\rangle\) or \(|1\rangle\) randomly;
2. your qubit is half of a Bell pair;
3. some noisy environment completely depolarized it.

Locally, all three give exactly the same density matrix.

There is no measurement on that qubit alone that distinguishes them.

This illustrates an important fact:

> The density matrix contains all physically observable information about the subsystem, but deliberately forgets how that mixed state was prepared.

---

# 32. Connection to an MPO simulator

There is a very natural tensor-network interpretation.

For a qubit site, a statevector/MPS physical index has dimension

\[
d=2.
\]

A density operator has a ket and a bra index:

\[
\rho_{s,s'}.
\]

Thus locally you have

\[
d^2=4
\]

basis operators, e.g.

\[
|0\rangle\langle0|,
\quad
|0\rangle\langle1|,
\quad
|1\rangle\langle0|,
\quad
|1\rangle\langle1|.
\]

You can therefore vectorize the pair

\[
(s,s')
\]

into one physical index

\[
\mu=0,\ldots,3.
\]

The density matrix becomes an MPS-like object with physical dimension \(4\), i.e. equivalently an **MPO**:

\[
\rho_{s_1\ldots s_n,s'_1\ldots s'_n}
=
\sum_{\{\alpha\}}
A^{[1]}_{\alpha_0\alpha_1}(s_1,s'_1)
\cdots
A^{[n]}_{\alpha_{n-1}\alpha_n}(s_n,s'_n).
\]

Then the channel

\[
\rho\rightarrow
\sum_kK_k\rho K_k^\dagger
\]

becomes a local transformation on these physical indices.

This is essentially tensor-network simulation in **Liouville space**.

---

# 33. What MPO compression is compressing

For an MPS, bond dimension reflects roughly the amount of state entanglement across a cut.

For an MPO density matrix, the analogous quantity is often called **operator entanglement**.

A completely arbitrary density matrix may require exponentially large bond dimension.

But physically relevant noisy states may admit

\[
\chi\ll 2^n.
\]

Then the MPO memory becomes something like

\[
O(n\,d^2\chi^2)
\]

rather than

\[
O(4^n).
\]

One interesting effect is that noise can sometimes **help compression**.

Unitary chaotic evolution may build strong operator entanglement, while decoherence destroys long-range coherence and can reduce the required MPO bond dimension.

It is not guaranteed, but this is one reason MPO simulation of noisy circuits can be surprisingly effective.

---

# 34. There is one difficulty with truncated MPO density matrices

An exact density matrix must obey

\[
\rho^\dagger=\rho,
\qquad
\rho\ge0,
\qquad
\operatorname{Tr}\rho=1.
\]

Ordinary SVD truncation of an MPO does not automatically preserve all three.

In particular, small truncation errors can produce tiny negative eigenvalues:

\[
\lambda_{\min}(\rho)<0.
\]

So an MPO noise simulator should ideally monitor at least:

\[
|\operatorname{Tr}\rho-1|,
\]

Hermiticity error, and possibly positivity.

Positivity is much harder to test globally for large MPOs.

There are tensor-network representations based on purification,

\[
\rho=XX^\dagger,
\]

that preserve positivity by construction, but they're somewhat more expensive.

---

# 35. Another useful representation: purification

Every mixed state can be regarded as part of a larger pure state.

Given

\[
\rho
=
\sum_i\lambda_i|i\rangle\langle i|,
\]

introduce an ancilla and define

\[
|\Psi\rangle
=
\sum_i\sqrt{\lambda_i}|i\rangle_S|i\rangle_A.
\]

Then

\[
\boxed{
\rho_S=
\operatorname{Tr}_A
|\Psi\rangle\langle\Psi|.
}
\]

So fundamentally, density matrices do not require adding a new postulate to quantum mechanics.

They can always be interpreted as:

\[
\boxed{
\text{pure state on a bigger Hilbert space}
+
\text{discarded degrees of freedom}.
}
\]

This picture is extremely useful for understanding noise.

---

# 36. A circuit-level noisy simulation

Suppose an ideal circuit is

\[
U_3U_2U_1.
\]

A noiseless density-matrix simulator computes

\[
\rho_1=U_1\rho_0U_1^\dagger,
\]

\[
\rho_2=U_2\rho_1U_2^\dagger,
\]

etc.

A noisy circuit might instead be

\[
\rho_1=
\mathcal N_1
\left(
U_1\rho_0U_1^\dagger
\right),
\]

\[
\rho_2=
\mathcal N_2
\left(
U_2\rho_1U_2^\dagger
\right),
\]

etc.

Or a more realistic model could have noise before and after the gate:

\[
\mathcal E_i
=
\mathcal N_i^{(\mathrm{post})}
\circ
\mathcal U_i
\circ
\mathcal N_i^{(\mathrm{pre})}.
\]

There can also be different noise attached to:

- idle periods;
- single-qubit gates;
- two-qubit gates;
- reset;
- measurement.

Two-qubit gates frequently deserve correlated error channels rather than two independent single-qubit channels.

---

# 37. A useful hierarchy for a simulator

You can think of the mathematical API in roughly this hierarchy:

\[
\boxed{
U
\subset
\{K_i\}
\subset
\text{CPTP channels}
}
\]

with continuous-time evolution

\[
\boxed{
(H,\{L_i,\gamma_i\})
\rightarrow
\text{Lindblad generator}
\rightarrow
\mathcal E_t.
}
\]

So internally you could regard

```text
Unitary gate
```

as a channel with one Kraus operator,

```text
PauliNoise
AmplitudeDamping
Depolarizing
...
```

as channels with several Kraus operators, and perhaps

```text
LindbladNoise
```

as something that generates a channel for a specified time interval.

That provides a fairly unified architecture.

---

# 38. One more distinction: noise versus readout error

Not all "noise" belongs in \(\rho\).

Suppose the actual qubit measurement is perfect, but the electronics report

\[
0\rightarrow1
\]

with probability \(p_{01}\), and

\[
1\rightarrow0
\]

with probability \(p_{10}\).

That can simply be represented by a classical confusion matrix

\[
M=
\begin{pmatrix}
1-p_{01} & p_{10}\\
p_{01} & 1-p_{10}
\end{pmatrix}.
\]

First calculate the quantum probabilities

\[
\mathbf p=
\begin{pmatrix}
p(0)\\p(1)
\end{pmatrix},
\]

then

\[
\mathbf p_{\rm observed}
=
M\mathbf p.
\]

There is no reason to turn every hardware imperfection into a quantum channel if its effect occurs after measurement.

---

# 39. Leakage needs a larger Hilbert space

A qubit density matrix is

\[
2\times2.
\]

But a superconducting device, for example, has states such as

\[
|0\rangle,\quad|1\rangle,\quad|2\rangle,\ldots
\]

Leakage

\[
|1\rangle\rightarrow|2\rangle
\]

cannot be represented exactly by a trace-preserving channel entirely inside the 2-dimensional qubit space while retaining the leakage state.

You either:

- extend the simulator to qutrits/qudits;
- use an effective leakage flag/model;
- use a trace-decreasing qubit subspace description.

This is one point where "density matrix" does not automatically mean "all noise can be represented": the Hilbert space must also contain the relevant physical states.

---

# 40. The overall theoretical picture

The relationships can be summarized as

\[
\boxed{
|\psi\rangle
\longrightarrow
\rho=|\psi\rangle\langle\psi|
}
\]

followed by

\[
\boxed{
\text{closed system}:
\quad
\rho\rightarrow U\rho U^\dagger
}
\]

whereas an open system has

\[
\boxed{
\rho_S'
=
\operatorname{Tr}_E
[
U_{SE}
(\rho_S\otimes\rho_E)
U_{SE}^\dagger
]
}
\]

which gives

\[
\boxed{
\rho'
=
\sum_kK_k\rho K_k^\dagger
}
\]

and, in continuous time,

\[
\boxed{
\dot\rho
=
-i[H,\rho]
+
\sum_k
\gamma_k
\left(
L_k\rho L_k^\dagger
-\frac12\{L_k^\dagger L_k,\rho\}
\right).
}
\]

Those equations correspond roughly to:

\[
\begin{array}{c}
\text{Schrödinger equation}\\
\downarrow\\
\text{unitary circuit simulator}\\[3mm]
\text{system+environment}\\
\downarrow\text{ partial trace}\\
\text{Kraus/CPTP channel}\\
\downarrow\\
\text{density-matrix circuit simulator}\\[3mm]
\text{continuous-time limit}\\
\downarrow\\
\text{Lindblad simulator}.
\end{array}
\]

And vectorizing

\[
\rho\rightarrow|\rho\rangle\rangle
\]

turns that into Liouville-space evolution of dimension \(4^n\), which is precisely the viewpoint from which an **MPO is a compressed density matrix**.

The next particularly useful level would be to derive the major channels one by one—**amplitude damping, phase damping, depolarization, thermal relaxation and correlated two-qubit noise—from explicit system+environment unitary interactions**, and then translate each derivation directly into the operations a density-matrix/MPO simulator needs.


Yes. For the theory we were discussing—density matrices, CPTP/Kraus maps, system–environment coupling, Lindblad evolution, trajectories, and MPO/MPDO simulation—I’d recommend the following.

1. **M. Paris, “[The modern tools of quantum mechanics](https://consensus.app/papers/the-modern-tools-of-quantum-mechanics-paris/c534f40b6b9d551fbb9df106f1de9708/?utm_source=chatgpt)”** — *European Physical Journal Special Topics* 203, 61–86 (2011), **66 citations**. [1]  
   This is probably the **best place to start for exactly your current question**. It develops the density operator, purification, generalized measurements, completely positive maps, Kraus/operator-sum representations, and Stinespring dilation. In particular, it makes explicit the connection
   \[
   S+E\text{ unitary evolution}
   \rightarrow \operatorname{Tr}_E
   \rightarrow \text{CPTP/Kraus map}.
   \]
   That is almost exactly the theoretical foundation behind a density-matrix quantum-circuit simulator. 

2. **D. Manzano, “[A short introduction to the Lindblad master equation](https://consensus.app/papers/a-short-introduction-to-the-lindblad-master-equation-manzano/5c30c64f1a5b5462a7205c4b3127efb0/?utm_source=chatgpt)”** — *AIP Advances* (2019), **733 citations**. [2]  
   Very readable and quite self-contained. It derives and explains
   \[
   \dot\rho=-i[H,\rho]+\sum_k\gamma_k
   \left(L_k\rho L_k^\dagger-\frac12\{L_k^\dagger L_k,\rho\}\right),
   \]
   and discusses solution methods. If you intend to add continuous-time noise/Lindblad evolution to your simulator rather than just discrete Kraus channels, I would definitely read this one. 

3. **C. J. Wood, J. Biamonte, D. Cory, “[Tensor networks and graphical calculus for open quantum systems](https://consensus.app/papers/tensor-networks-and-graphical-calculus-for-open-quantum-wood-biamonte/665f4c60277d51199a096fbe6c004872/?utm_source=chatgpt)”** — *Quantum Information & Computation* 15, 759–811 (2011), **154 citations**. [3]  
   This one is particularly relevant to your implementation because it puts **Kraus maps, Choi matrices, Liouville superoperators, process matrices, and system-environment representations into one framework**, explicitly using tensor-network notation. It shows the relations
   \[
   \{K_i\}
   \leftrightarrow
   \mathcal S
   \leftrightarrow
   J(\mathcal E)
   \leftrightarrow
   U_{SE},
   \]
   which is extremely helpful once you start thinking of a density matrix as an MPO or vectorized Liouville-space state. 

4. **D. Jaschke, S. Montangero, L. Carr, “[One-dimensional many-body entangled open quantum systems with tensor network methods](https://consensus.app/papers/onedimensional-manybody-entangled-open-quantum-systems-jaschke-montangero/17bf7905f4fe52d8ae582f2fd00c29a3/?utm_source=chatgpt)”** — *Quantum Science and Technology* 4 (2018), **78 citations**. [4]  
   For your **MPO simulator**, this is probably the most directly useful paper in the list. It compares three approaches to Lindblad dynamics:
   \[
   \boxed{\text{quantum trajectories}}
   \qquad
   \boxed{\text{MPDO/MPO density matrices}}
   \qquad
   \boxed{\text{locally purified tensor networks}}.
   \]
   It also discusses spontaneous emission and dephasing examples and compares numerical behavior. The locally purified approach is especially interesting because it addresses the positivity problem I mentioned with truncated MPO density matrices. 

5. **G. Lindblad, “[On the generators of quantum dynamical semigroups](https://consensus.app/papers/on-the-generators-of-quantum-dynamical-semigroups-lindblad/15018acbac3a5706a0c247fa001f8b31/?utm_source=chatgpt)”** — *Communications in Mathematical Physics* 48, 119–130 (1976), **6592 citations**. [5]  
   This is the classic original paper. It derives the structure of generators of completely positive dynamical semigroups—the mathematical origin of the Lindblad/GKSL equation. It is more mathematical than [2], so I'd read Manzano first and then this if you want to understand exactly **why** the generator must have that form. 

6. **J. Dalibard, Y. Castin, K. Mølmer, “[Wave-function approach to dissipative processes in quantum optics](https://consensus.app/papers/wavefunction-approach-to-dissipative-processes-in-dalibard-castin/69d5e0561bb059098c588bdde754d0df/?utm_source=chatgpt)”** — *Physical Review Letters* 68, 580–583 (1992), **1600 citations**. [6]  
   This is one of the foundational **quantum trajectory / Monte Carlo wave-function** papers. It is useful precisely for understanding the alternative to your density-matrix simulator:
   \[
   \rho(t)
   \quad\leftrightarrow\quad
   \text{ensemble of stochastic }|\psi(t)\rangle.
   \]
   The paper shows how reservoir coupling and dissipation can be replaced by non-Hermitian evolution interrupted by stochastic quantum jumps. Computationally, that gives the \(N\) versus \(N^2\) state-size distinction we discussed. 

7. **D. Chruściński, “[Dynamical maps beyond Markovian regime](https://consensus.app/papers/dynamical-maps-beyond-markovian-regime-chruscinski/42e76d2e58f656c5a2c1389475173ed6/?utm_source=chatgpt)”** — *Physics Reports* (2022), **131 citations**. [7]  
   This is a much deeper review for when you want to go beyond
   \[
   \rho(t+\Delta t)=\mathcal E_{\Delta t}[\rho(t)]
   \]
   with independent Markovian channels. It covers dynamical maps, CP-divisibility, time-local generators, memory kernels, and non-Markovian dynamics. It is particularly relevant if you eventually want your simulator to represent an environment that retains memory rather than just applying independent \(T_1/T_2\) channels. 

8. **Z. Hu, R. Xia, S. Kais, “[A quantum algorithm for evolving open quantum dynamics on quantum computing devices](https://consensus.app/papers/a-quantum-algorithm-for-evolving-open-quantum-dynamics-on-hu-xia/6eb2a97b7d875528ab0ad6366c631f91/?utm_source=chatgpt)”** — *Scientific Reports* 10 (2019), **184 citations**. [8]  
   This is somewhat different because it asks how to realize open-system dynamics **on a quantum computer**, but it is a nice concrete application of Kraus operators and dilation. They explicitly treat amplitude damping and convert non-unitary Kraus evolution into enlarged-space unitary evolution. 

### What I would actually read

For your purpose, I would go in this order:

\[
\boxed{[1]\rightarrow[2]\rightarrow[3]\rightarrow[4]}
\]

Those four essentially give you the entire chain

\[
\begin{aligned}
\text{density matrix}
&\rightarrow \text{partial trace}\\
&\rightarrow \text{CP/CPTP maps}\\
&\rightarrow \text{Kraus representation}\\
&\rightarrow \text{Liouville/superoperator representation}\\
&\rightarrow \text{Lindblad equation}\\
&\rightarrow \text{MPO/MPDO implementation}.
\end{aligned}
\]

Then I would read [6] because it gives the main competing simulation strategy—**stochastic statevector trajectories instead of explicit \(4^n\) density matrices**. [5] is worthwhile for the rigorous origin of Lindblad, while [7] is the natural next step if you become interested in genuinely non-Markovian environment models.

For the specific simulators you've implemented, **[3] and [4] are probably the papers I'd expect to be most immediately useful at the code/design level**, while [1] and [2] give the theoretical foundation.

Upgrade to Consensus Pro to return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm_source=chatgpt

Yes. The most useful way to derive these for a simulator is to start from the same general construction and then specialize it. The important practical point is that the **system+environment unitary is the physics**, while the **Kraus map or local Liouville superoperator is usually what you actually implement**.

A microscopic derivation of amplitude- and phase-damping Kraus operators from system–environment dynamics is discussed explicitly by Arsenijević & Banković [1]; generalized amplitude damping corresponds to interaction with a thermal bath [2], and analogous microscopic derivations exist for depolarizing dynamics [3]. Correlated environments require genuinely multi-qubit channels rather than products of single-qubit ones [4,5].   

---

# 1. The general recipe: \(S+E\rightarrow\) Kraus operators

Suppose the system is \(S\), the environment is \(E\), and initially

\[
\rho_{SE}=\rho_S\otimes |0_E\rangle\langle0_E|.
\]

The combined system evolves unitarily,

\[
\rho_{SE}'=
U_{SE}
(\rho_S\otimes |0_E\rangle\langle0_E|)
U_{SE}^\dagger.
\]

We throw away the environment:

\[
\rho_S'=\operatorname{Tr}_E\rho_{SE}'.
\]

Insert an orthonormal environment basis \(\{|e_k\rangle\}\):

\[
\rho_S'
=
\sum_k
\langle e_k|
U_{SE}
(\rho_S\otimes|0_E\rangle\langle0_E|)
U_{SE}^\dagger
|e_k\rangle .
\]

Define

\[
\boxed{
K_k=\langle e_k|U_{SE}|0_E\rangle
}
\]

and therefore

\[
\boxed{
\rho'=\sum_k K_k\rho K_k^\dagger.
}
\]

The trace-preserving condition follows from unitarity:

\[
\sum_kK_k^\dagger K_k=I.
\]

For a **mixed environment**

\[
\rho_E=\sum_\mu q_\mu|\mu\rangle\langle\mu|,
\]

the Kraus operators become

\[
\boxed{
K_{k\mu}=
\sqrt{q_\mu}\,
\langle e_k|U|\mu\rangle .
}
\]

That formula is particularly important for thermal relaxation.

There is also a useful reverse statement. Given any Kraus channel,

\[
\mathcal E(\rho)=\sum_kK_k\rho K_k^\dagger,
\]

define

\[
V|\psi\rangle
=
\sum_k K_k|\psi\rangle\otimes|k\rangle.
\]

Because

\[
V^\dagger V
=
\sum_kK_k^\dagger K_k=I,
\]

\(V\) is an isometry and can always be extended to some unitary \(U_{SE}\).

So:

\[
\boxed{
\text{Kraus channel}
\iff
\text{unitary on a larger system + discarded environment}.
}
\]

The dilation is not unique.

---

# 2. Amplitude damping

This is probably the cleanest physically motivated noise channel.

It describes

\[
|1\rangle\rightarrow |0\rangle
\]

through loss of an excitation into the environment.

Examples include spontaneous emission and \(T_1\) relaxation.

## 2.1 Microscopic picture

Take an environment qubit initially in

\[
|0_E\rangle.
\]

Think of

\[
|0_E\rangle = \text{no emitted excitation},
\]

\[
|1_E\rangle = \text{one emitted excitation}.
\]

Define the system-environment evolution by

\[
|0\rangle|0_E\rangle
\longrightarrow
|0\rangle|0_E\rangle
\]

and

\[
|1\rangle|0_E\rangle
\longrightarrow
\sqrt{1-\gamma}
|1\rangle|0_E\rangle
+
\sqrt{\gamma}
|0\rangle|1_E\rangle.
\]

If

\[
\gamma=\sin^2\theta,
\]

this can come from an excitation-exchange unitary acting as a rotation in the

\[
\{|10\rangle,|01\rangle\}
\]

subspace.

Physically this can arise from an interaction such as

\[
H_I
\sim
g
\left(
\sigma_+\otimes b+
\sigma_-\otimes b^\dagger
\right).
\]

An excitation can move

\[
S\rightarrow E.
\]

---

# 3. Deriving the Kraus operators

We calculate

\[
K_0=\langle0_E|U|0_E\rangle
\]

and

\[
K_1=\langle1_E|U|0_E\rangle.
\]

From the transformations above,

\[
\boxed{
K_0=
\begin{pmatrix}
1&0\\
0&\sqrt{1-\gamma}
\end{pmatrix}
}
\]

and

\[
\boxed{
K_1=
\begin{pmatrix}
0&\sqrt{\gamma}\\
0&0
\end{pmatrix}.
}
\]

Check:

\[
K_0^\dagger K_0+
K_1^\dagger K_1
=
I.
\]

Now take

\[
\rho=
\begin{pmatrix}
a&c\\
c^*&b
\end{pmatrix}.
\]

Then

\[
\rho'
=
K_0\rho K_0^\dagger+
K_1\rho K_1^\dagger
\]

gives

\[
\boxed{
\rho'=
\begin{pmatrix}
a+\gamma b&
\sqrt{1-\gamma}\,c\\
\sqrt{1-\gamma}\,c^*&
(1-\gamma)b
\end{pmatrix}.
}
\]

Three things happen simultaneously:

\[
\rho_{11}
\rightarrow
(1-\gamma)\rho_{11},
\]

\[
\rho_{00}
\rightarrow
\rho_{00}+\gamma\rho_{11},
\]

and

\[
\rho_{01}
\rightarrow
\sqrt{1-\gamma}\rho_{01}.
\]

The reduction of coherence is not an additional assumption. It follows automatically from entanglement with the environment.

---

# 4. Why the coherence disappears

Start from

\[
|\psi\rangle
=
\alpha|0\rangle+\beta|1\rangle.
\]

After interaction,

\[
|\Psi_{SE}\rangle
=
\alpha|0,0_E\rangle
+
\beta\sqrt{1-\gamma}|1,0_E\rangle
+
\beta\sqrt{\gamma}|0,1_E\rangle.
\]

The environment now carries information about whether the excitation decayed.

The \(|1_E\rangle\) component is orthogonal to \(|0_E\rangle\).

When we trace \(E\) out, interference between these alternatives disappears.

That is the microscopic origin of the non-unitarity.

---

# 5. \(T_1\) connection

For Markovian relaxation,

\[
P_1(t)=e^{-t/T_1}P_1(0).
\]

Consequently

\[
\boxed{
1-\gamma=e^{-t/T_1}
}
\]

or

\[
\boxed{
\gamma=1-e^{-t/T_1}.
}
\]

Therefore

\[
\sqrt{1-\gamma}
=
e^{-t/(2T_1)}.
\]

Relaxation alone already causes coherence decay with time constant

\[
2T_1.
\]

---

# 6. Liouville representation of amplitude damping

For an MPO implementation this form becomes convenient.

Fuse

\[
(s,s')
\]

into a local physical index of dimension \(4\).

Using ordering

\[
(\rho_{00},\rho_{01},\rho_{10},\rho_{11}),
\]

write

\[
|\rho\rangle\rangle
=
\begin{pmatrix}
\rho_{00}\\
\rho_{01}\\
\rho_{10}\\
\rho_{11}
\end{pmatrix}.
\]

Then

\[
|\rho'\rangle\rangle
=
S_{\rm AD}
|\rho\rangle\rangle
\]

where

\[
\boxed{
S_{\rm AD}=
\begin{pmatrix}
1&0&0&\gamma\\
0&\sqrt{1-\gamma}&0&0\\
0&0&\sqrt{1-\gamma}&0\\
0&0&0&1-\gamma
\end{pmatrix}.
}
\]

This \(4\times4\) matrix is exactly what you can apply to one physical leg of a density-MPO.

---

# 7. Phase damping / pure dephasing

Now consider a system that exchanges **no energy** with its environment.

Instead, the environment learns something about whether the qubit is in \(|0\rangle\) or \(|1\rangle\).

Let

\[
|0\rangle|e\rangle
\rightarrow
|0\rangle|e_0\rangle
\]

and

\[
|1\rangle|e\rangle
\rightarrow
|1\rangle|e_1\rangle.
\]

Take

\[
|\psi\rangle=\alpha|0\rangle+\beta|1\rangle.
\]

After interaction,

\[
|\Psi\rangle
=
\alpha|0\rangle|e_0\rangle
+
\beta|1\rangle|e_1\rangle.
\]

Tracing out \(E\),

\[
\rho_S=
\begin{pmatrix}
|\alpha|^2&
\alpha\beta^*\langle e_1|e_0\rangle\\
\alpha^*\beta\langle e_0|e_1\rangle&
|\beta|^2
\end{pmatrix}.
\]

Define

\[
\eta=\langle e_1|e_0\rangle.
\]

Then

\[
\boxed{
\rho_{01}\rightarrow\eta\rho_{01}
}
\]

while

\[
\rho_{00},\rho_{11}
\]

remain unchanged.

This gives a very intuitive interpretation of decoherence:

\[
\boxed{
\text{coherence remaining}
=
\text{overlap of environmental states}.
}
\]

If

\[
|e_0\rangle=|e_1\rangle,
\]

then

\[
\eta=1
\]

and no decoherence occurs.

If they become orthogonal,

\[
\eta=0,
\]

the environment can perfectly distinguish the two system alternatives and coherence disappears completely.

---

# 8. Explicit environment qubit for phase damping

Take

\[
|0,0_E\rangle
\rightarrow
|0,0_E\rangle,
\]

\[
|1,0_E\rangle
\rightarrow
|1\rangle
\left(
\sqrt{1-\lambda}|0_E\rangle
+
\sqrt{\lambda}|1_E\rangle
\right).
\]

Then

\[
\eta=\sqrt{1-\lambda}.
\]

The Kraus operators are

\[
\boxed{
K_0=
\begin{pmatrix}
1&0\\
0&\sqrt{1-\lambda}
\end{pmatrix}
}
\]

and

\[
\boxed{
K_1=
\begin{pmatrix}
0&0\\
0&\sqrt{\lambda}
\end{pmatrix}.
}
\]

Thus

\[
\rho'
=
\begin{pmatrix}
\rho_{00}&
\sqrt{1-\lambda}\rho_{01}\\
\sqrt{1-\lambda}\rho_{10}&
\rho_{11}
\end{pmatrix}.
\]

Microscopic derivations of both amplitude- and phase-damping Kraus operators are treated in [1]. 

---

# 9. Phase damping versus phase-flip

These often get confused because they generate essentially the same type of reduced dynamics.

Another Kraus representation is

\[
K_0=\sqrt{1-p}\,I,
\qquad
K_1=\sqrt p\,Z.
\]

Then

\[
\mathcal E(\rho)
=
(1-p)\rho+pZ\rho Z
\]

and therefore

\[
\rho_{01}'
=
(1-2p)\rho_{01}.
\]

So if

\[
\eta=1-2p,
\]

this produces the same dephasing map.

This is an important example of the non-uniqueness of Kraus representations.

---

# 10. \(T_\phi\)

For pure Markovian dephasing,

\[
\eta(t)=e^{-t/T_\phi}.
\]

So in the first parameterization,

\[
\sqrt{1-\lambda}=e^{-t/T_\phi}
\]

and therefore

\[
\boxed{
\lambda=1-e^{-2t/T_\phi}.
}
\]

The local Liouville matrix is particularly simple:

\[
\boxed{
S_\phi=
\begin{pmatrix}
1&0&0&0\\
0&\eta&0&0\\
0&0&\eta&0\\
0&0&0&1
\end{pmatrix}.
}
\]

For an MPO this is just a local \(4\times4\) transformation.

No bond dimension increase whatsoever.

---

# 11. Depolarization from an explicit environment

Depolarization is somewhat different.

A useful physical interpretation is that the environment records which Pauli error occurred.

Introduce four orthogonal environment states

\[
|I\rangle_E,\quad
|X\rangle_E,\quad
|Y\rangle_E,\quad
|Z\rangle_E.
\]

Construct the isometry

\[
\boxed{
V|\psi\rangle
=
\sqrt{p_I}\,|\psi\rangle|I\rangle
+
\sqrt{p_X}\,X|\psi\rangle|X\rangle
+
\sqrt{p_Y}\,Y|\psi\rangle|Y\rangle
+
\sqrt{p_Z}\,Z|\psi\rangle|Z\rangle.
}
\]

Tracing out \(E\),

\[
\boxed{
\mathcal E(\rho)
=
p_I\rho
+
p_X X\rho X
+
p_Y Y\rho Y
+
p_Z Z\rho Z.
}
\]

This is the general Pauli channel.

The environment dimension required by this simple pure-environment dilation is four, corresponding to Kraus rank four.

A microscopic treatment of depolarizing Kraus maps is discussed by Arsenijević et al. [3]. 

---

# 12. Isotropic depolarizing channel

I prefer defining the parameter \(p\) by

\[
\boxed{
\mathcal E_p(\rho)
=
(1-p)\rho+
p\frac{I}{2}.
}
\]

Then the Bloch vector transforms as

\[
\mathbf r\rightarrow(1-p)\mathbf r.
\]

The corresponding Pauli probabilities are

\[
p_X=p_Y=p_Z=\frac p4,
\]

and

\[
p_I=1-\frac{3p}{4}.
\]

Indeed,

\[
\boxed{
\mathcal E_p(\rho)
=
\left(1-\frac{3p}{4}\right)\rho
+
\frac p4
(X\rho X+Y\rho Y+Z\rho Z).
}
\]

This is why depolarization is often implemented simply as a random Pauli channel.

---

# 13. Depolarizing superoperator

For arbitrary matrices it is better to write

\[
\mathcal E(\rho)
=
(1-p)\rho+
\frac p2\operatorname{Tr}(\rho)I,
\]

which keeps the map explicitly linear.

Then

\[
S_{\rm dep}
=
\boxed{
\begin{pmatrix}
1-\frac p2&0&0&\frac p2\\
0&1-p&0&0\\
0&0&1-p&0\\
\frac p2&0&0&1-\frac p2
\end{pmatrix}.
}
\]

Again: one \(4\times4\) matrix applied to one MPO physical index.

---

# 14. Thermal relaxation: finite-temperature amplitude damping

Ordinary amplitude damping assumes that the environment starts in its ground state:

\[
|0_E\rangle.
\]

But at finite temperature the environment occasionally contains an excitation.

Then both

\[
|1\rangle\rightarrow|0\rangle
\]

and

\[
|0\rangle\rightarrow|1\rangle
\]

can happen.

This gives **generalized amplitude damping**.

Generalized amplitude damping is the standard dissipative channel associated with interaction with a thermal bath [2]. 

---

# 15. Thermal environment

For an environment qubit with energy gap

\[
\Delta E=\hbar\omega,
\]

the thermal state is

\[
\rho_E
=
p_g|0\rangle\langle0|
+
p_e|1\rangle\langle1|,
\]

where

\[
p_e
=
\frac{e^{-\beta\hbar\omega}}
{1+e^{-\beta\hbar\omega}}
=
\frac1{1+e^{\beta\hbar\omega}},
\]

and

\[
p_g=1-p_e.
\]

Use the same excitation-exchange unitary as before.

If the environment initially happens to be in \(|0\rangle\), the qubit can decay.

If it is in \(|1\rangle\), the environment can excite the qubit.

---

# 16. Four Kraus operators

The resulting generalized amplitude damping channel can be written

\[
K_0
=
\sqrt{p_g}
\begin{pmatrix}
1&0\\
0&\sqrt{1-\gamma}
\end{pmatrix},
\]

\[
K_1
=
\sqrt{p_g}
\begin{pmatrix}
0&\sqrt{\gamma}\\
0&0
\end{pmatrix},
\]

\[
K_2
=
\sqrt{p_e}
\begin{pmatrix}
\sqrt{1-\gamma}&0\\
0&1
\end{pmatrix},
\]

\[
K_3
=
\sqrt{p_e}
\begin{pmatrix}
0&0\\
\sqrt{\gamma}&0
\end{pmatrix}.
\]

Then

\[
\boxed{
\rho'=\sum_{k=0}^3K_k\rho K_k^\dagger.
}
\]

---

# 17. Population evolution

A very nice result is

\[
\boxed{
\rho_{11}'
=
(1-\gamma)\rho_{11}
+
\gamma p_e.
}
\]

Equivalently,

\[
\rho_{11}'-p_e
=
(1-\gamma)
(\rho_{11}-p_e).
\]

Thus \(p_e\) is the fixed-point excited-state population.

For continuous Markovian relaxation,

\[
1-\gamma=e^{-t/T_1},
\]

so

\[
\boxed{
\rho_{11}(t)
=
p_e+
[\rho_{11}(0)-p_e]e^{-t/T_1}.
}
\]

Likewise

\[
\rho_{00}(t)
=
p_g+
[\rho_{00}(0)-p_g]e^{-t/T_1}.
\]

The state approaches

\[
\boxed{
\rho_{\rm thermal}
=
\begin{pmatrix}
p_g&0\\
0&p_e
\end{pmatrix}.
}
\]

---

# 18. Coherence under thermal relaxation

Generalized amplitude damping gives

\[
\rho_{01}(t)
=
e^{-t/(2T_1)}\rho_{01}(0)
\]

if there is no additional phase noise.

Real hardware generally has additional pure dephasing.

Introduce

\[
T_\phi.
\]

Then

\[
\boxed{
\frac1{T_2}
=
\frac1{2T_1}
+
\frac1{T_\phi}.
}
\]

Consequently

\[
\boxed{
\rho_{01}(t)
=
e^{-t/T_2}\rho_{01}(0).
}
\]

This is probably the most useful **hardware-style thermal relaxation channel** for your simulator.

For a normalized input,

\[
\boxed{
\rho(t)=
\begin{pmatrix}
p_g+e^{-t/T_1}(\rho_{00}-p_g)
&
e^{-t/T_2}\rho_{01}
\\
e^{-t/T_2}\rho_{10}
&
p_e+e^{-t/T_1}(\rho_{11}-p_e)
\end{pmatrix}.
}
\]

That is an extremely convenient direct implementation.

---

# 19. Lindblad derivation of the same channel

The corresponding master equation is

\[
\dot\rho
=
\Gamma_\downarrow
\mathcal D[\sigma_-]\rho
+
\Gamma_\uparrow
\mathcal D[\sigma_+]\rho
+
\frac{\Gamma_\phi}{2}
(Z\rho Z-\rho),
\]

where

\[
\mathcal D[L]\rho
=
L\rho L^\dagger
-
\frac12\{L^\dagger L,\rho\}.
\]

Then

\[
\boxed{
\frac1{T_1}
=
\Gamma_\downarrow+\Gamma_\uparrow
}
\]

and

\[
\boxed{
p_e
=
\frac{\Gamma_\uparrow}
{\Gamma_\downarrow+\Gamma_\uparrow}.
}
\]

Detailed balance gives

\[
\frac{\Gamma_\uparrow}{\Gamma_\downarrow}
=
e^{-\beta\hbar\omega}.
\]

Meanwhile

\[
\frac1{T_2}
=
\frac{\Gamma_\downarrow+\Gamma_\uparrow}{2}
+
\Gamma_\phi.
\]

This gives you a clean conversion between physical device parameters and simulator parameters.

---

# 20. Implementation trick: don't make an eight-Kraus channel

Suppose you model thermal relaxation as

\[
\mathcal E_{T_1}
\]

followed by pure dephasing

\[
\mathcal E_\phi.
\]

Naively composing their Kraus representations creates

\[
4\times2=8
\]

Kraus operators.

You don't need to do that.

For a dense density matrix, directly use the closed formula

\[
\rho_{11}'
=
p_e+q(\rho_{11}-p_e)
\]

and

\[
\rho_{01}'=\eta\rho_{01},
\]

where

\[
q=e^{-t/T_1},
\qquad
\eta=e^{-t/T_2}.
\]

For an MPO, simply build the corresponding \(4\times4\) local superoperator.

---

# 21. Thermal relaxation superoperator

Let

\[
q=e^{-t/T_1},
\qquad
\eta=e^{-t/T_2}.
\]

For arbitrary trace, the population map is

\[
\rho_{00}'
=
q\rho_{00}
+
(1-q)p_g(\rho_{00}+\rho_{11}),
\]

\[
\rho_{11}'
=
q\rho_{11}
+
(1-q)p_e(\rho_{00}+\rho_{11}).
\]

Thus

\[
\boxed{
S_{\rm therm}=
\begin{pmatrix}
q+(1-q)p_g&0&0&(1-q)p_g\\
0&\eta&0&0\\
0&0&\eta&0\\
(1-q)p_e&0&0&q+(1-q)p_e
\end{pmatrix}.
}
\]

This is probably what I would implement internally for a gate-duration-based \(T_1/T_2\) noise model.

---

# 22. Correlated two-qubit noise is fundamentally different

There isn't one unique "correlated noise channel."

The important distinction is

\[
\boxed{
\mathcal E_{12}
\ne
\mathcal E_1\otimes\mathcal E_2.
}
\]

If the noise factorizes, you can treat both qubits independently.

Correlated noise means that the same environmental degree of freedom couples to both qubits.

Spatially correlated noise is relevant in realistic processors and can have both Markovian and non-Markovian contributions [5]. 

Let's look at several particularly useful examples.

---

# 23. Correlated Pauli noise

Suppose a common environment determines which *joint* error occurs.

Define

\[
V|\psi\rangle
=
\sum_k\sqrt{p_k}
P_k|\psi\rangle|k_E\rangle,
\]

where now

\[
P_k
\]

are **two-qubit Pauli strings**, e.g.

\[
II,\quad ZI,\quad IZ,\quad ZZ,\ldots
\]

Tracing out the environment gives

\[
\boxed{
\mathcal E(\rho)
=
\sum_kp_kP_k\rho P_k.
}
\]

For example,

\[
\boxed{
\mathcal E_{\rm corrZ}(\rho)
=
(1-p)\rho+p\,ZZ\rho ZZ.
}
\]

This means:

> either nothing happens to either qubit, or both suffer a phase flip together.

That is very different from independent phase flips.

---

# 24. Independent versus correlated phase errors

For independent noise,

\[
\mathcal E_{\rm independent}
=
\mathcal E_Z^{(1)}
\otimes
\mathcal E_Z^{(2)}.
\]

This produces four possibilities:

\[
II,\quad ZI,\quad IZ,\quad ZZ.
\]

For fully correlated noise,

\[
\mathcal E_{\rm correlated}
\]

might contain only

\[
II,\quad ZZ.
\]

A simple interpolation is

\[
\boxed{
\mathcal E_\mu
=
(1-\mu)\mathcal E_{\rm independent}
+
\mu\mathcal E_{\rm correlated}.
}
\]

Here

\[
0\le\mu\le1
\]

acts as a memory/correlation strength.

---

# 25. More physical correlated dephasing: common bath

A more interesting microscopic interaction is

\[
\boxed{
H_I
=
(Z_1+Z_2)\otimes B_E.
}
\]

The unitary is

\[
U(t)
=
e^{-it(Z_1+Z_2)\otimes B_E}.
\]

Let

\[
A=Z_1+Z_2.
\]

Its eigenvalues are

\[
|00\rangle:\quad +2,
\]

\[
|01\rangle:\quad0,
\]

\[
|10\rangle:\quad0,
\]

\[
|11\rangle:\quad-2.
\]

If

\[
A|m\rangle=a_m|m\rangle,
\]

the reduced density-matrix element evolves roughly as

\[
\boxed{
\rho_{mn}(t)
=
\chi[(a_m-a_n)t]\,
\rho_{mn}(0),
}
\]

where

\[
\chi(s)
=
\operatorname{Tr}
\left(
\rho_Ee^{-isB_E}
\right)
\]

is an environment correlation/characteristic function.

Now notice something remarkable:

\[
a_{01}=a_{10}=0.
\]

Therefore

\[
\rho_{01,10}
\]

does **not** decohere under perfectly collective dephasing.

The subspace

\[
\operatorname{span}\{|01\rangle,|10\rangle\}
\]

is a decoherence-free subspace.

That is something you can never obtain by treating the two qubits as independently dephasing.

---

# 26. Correlated amplitude damping

This is even more interesting.

Instead of two independent baths,

\[
H_I
=
\sigma_-^{(1)}\otimes B_1^\dagger
+
\sigma_-^{(2)}\otimes B_2^\dagger
+\text{h.c.},
\]

suppose they couple to the **same** bath:

\[
\boxed{
H_I
=
g
\left[
(\sigma_-^{(1)}+\sigma_-^{(2)})b^\dagger
+
(\sigma_+^{(1)}+\sigma_+^{(2)})b
\right].
}
\]

Define

\[
J_-=
\sigma_-^{(1)}+\sigma_-^{(2)}.
\]

Then the Markovian master equation contains

\[
\boxed{
\dot\rho
=
\Gamma
\mathcal D[J_-]\rho.
}
\]

This is **not**

\[
\Gamma\mathcal D[\sigma_-^{(1)}]\rho
+
\Gamma\mathcal D[\sigma_-^{(2)}]\rho.
\]

The bath cannot determine which qubit emitted the excitation, so amplitudes interfere.

Collective amplitude damping in two-qubit systems has been studied explicitly in circuit and master-equation form [4]. 

---

# 27. Superradiant and dark states appear automatically

Define

\[
|\psi_+\rangle
=
\frac{|01\rangle+|10\rangle}{\sqrt2},
\]

\[
|\psi_-\rangle
=
\frac{|01\rangle-|10\rangle}{\sqrt2}.
\]

Apply \(J_-\):

\[
J_-|\psi_+\rangle
=
\sqrt2|00\rangle,
\]

while

\[
\boxed{
J_-|\psi_-\rangle=0.
}
\]

Therefore the symmetric state decays faster:

\[
|\psi_+\rangle
\rightarrow |00\rangle,
\]

while the antisymmetric state is dark.

Again, this behavior is impossible to reproduce with two independent amplitude-damping channels.

---

# 28. Short-time Kraus form of collective damping

For a small time step \(dt\),

\[
K_1
=
\sqrt{\Gamma\,dt}\,J_-,
\]

and

\[
K_0
=
I-
\frac{\Gamma\,dt}{2}
J_+J_-
\]

to first order.

Then

\[
K_0^\dagger K_0+
K_1^\dagger K_1
=
I+O(dt^2).
\]

Apply

\[
\rho(t+dt)
=
K_0\rho K_0^\dagger+
K_1\rho K_1^\dagger.
\]

Expanding,

\[
\rho(t+dt)-\rho(t)
=
\Gamma dt
\left[
J_-\rho J_+
-\frac12
\{J_+J_-,\rho\}
\right],
\]

hence

\[
\boxed{
\dot\rho
=
\Gamma
\left[
J_-\rho J_+
-\frac12\{J_+J_-,\rho\}
\right].
}
\]

This is a very nice direct connection

\[
\boxed{
\text{short-time Kraus channel}
\longleftrightarrow
\text{Lindblad equation}.
}
\]

---

# 29. Most general correlated relaxation model

For two qubits, write

\[
\boxed{
\dot\rho
=
\sum_{i,j=1}^{2}
\Gamma_{ij}
\left[
\sigma_-^{(i)}
\rho
\sigma_+^{(j)}
-
\frac12
\left\{
\sigma_+^{(j)}
\sigma_-^{(i)},\rho
\right\}
\right].
}
\]

If

\[
\Gamma_{12}=0,
\]

the noise is independent.

If

\[
\Gamma_{12}\ne0,
\]

there are bath-mediated correlations.

For identical qubits you might have

\[
\Gamma=
\begin{pmatrix}
\gamma&c\\
c^*&\gamma
\end{pmatrix}.
\]

Diagonalizing this matrix produces collective jump operators roughly of the form

\[
L_+
\propto
\sigma_-^{(1)}
+
e^{i\phi}\sigma_-^{(2)}
\]

and

\[
L_-
\propto
\sigma_-^{(1)}
-
e^{i\phi}\sigma_-^{(2)}.
\]

So numerically it is often easiest to diagonalize the correlation matrix first and simulate the resulting collective Lindblad operators.

---

# 30. Correlated dephasing has the same structure

You can similarly write

\[
\dot\rho
=
\sum_{i,j}
C_{ij}
\left[
Z_i\rho Z_j
-
\frac12
\{Z_jZ_i,\rho\}
\right],
\]

with \(C\) positive semidefinite.

Independent noise:

\[
C_{12}=0.
\]

Common-mode noise:

\[
C_{12}\ne0.
\]

Diagonalizing \(C\) gives jump operators such as

\[
L_+
\propto Z_1+Z_2,
\]

\[
L_-
\propto Z_1-Z_2.
\]

This is a very general and useful way to parameterize spatially correlated Markovian noise.

---

# 31. Now translate all this to your dense density-matrix simulator

Suppose the channel acts on one qubit \(q\) and has Kraus operators

\[
K_1,\ldots,K_r.
\]

Conceptually,

\[
\rho'
=
\sum_k
K_k^{(q)}
\rho
K_k^{(q)\dagger}.
\]

You should **not** construct the global

\[
2^n\times2^n
\]

\(K_k^{(q)}\).

Instead reshape

\[
\rho_{i_1\ldots i_n,
      j_1\ldots j_n}
\]

as a rank-\(2n\) tensor.

For each \(K_k\):

- contract \(K_k\) into ket index \(i_q\);
- contract \(K_k^*\) into bra index \(j_q\);
- sum the \(k\) branches.

Schematically,

\[
\boxed{
\rho'_{\ldots a\ldots,\ldots b\ldots}
=
\sum_{k,s,s'}
(K_k)_{as}
(K_k)^*_{bs'}
\rho_{\ldots s\ldots,\ldots s'\ldots}.
}
\]

This is exactly a local tensor operation.

---

# 32. Or construct the local superoperator once

Define

\[
\boxed{
\mathcal S_{ab,ss'}
=
\sum_k
(K_k)_{as}(K_k)^*_{bs'}.
}
\]

Fuse

\[
(a,b)\rightarrow\mu',
\qquad
(s,s')\rightarrow\mu.
\]

For a one-qubit channel,

\[
\mathcal S
\]

is only

\[
4\times4.
\]

Then simply transform the appropriate density-matrix physical index.

This is often cleaner than explicitly looping over Kraus branches.

---

# 33. MPO implementation: one-qubit channel

Suppose your MPO tensor is

\[
A^{[i]}_{\alpha\beta}(s,s').
\]

Fuse \(s,s'\) into a physical index

\[
\mu=0,\ldots,3.
\]

A one-qubit channel is then simply

\[
\boxed{
A'^{[i]}_{\alpha\beta}(\mu')
=
\sum_\mu
S_{\mu'\mu}
A^{[i]}_{\alpha\beta}(\mu).
}
\]

That's all.

Most importantly:

\[
\boxed{
\chi_{\rm new}=\chi_{\rm old}
}
\]

for an exactly local single-qubit channel.

There is **no MPO bond growth** merely from amplitude damping, dephasing, thermal relaxation, or single-qubit depolarization.

This is one of the major practical advantages of applying the channel as a local superoperator rather than forming separate MPOs for every Kraus branch.

---

# 34. Why applying Kraus branches separately is less attractive for an MPO

If you calculate

\[
K_0\rho K_0^\dagger,
\]

\[
K_1\rho K_1^\dagger,
\]

etc. as separate MPOs and then add them, the direct sum involved in MPO addition can increase the bond dimension roughly as

\[
\chi\rightarrow
\sum_k\chi_k.
\]

You then have to compress it.

For a one-site channel that is unnecessary.

Instead calculate

\[
S=\sum_kK_k\otimes K_k^*
\]

locally and apply \(S\) to the physical leg.

The bond dimension stays unchanged.

---

# 35. Two-qubit correlated channel in an MPO

Now suppose your channel acts on neighboring sites \(i,i+1\).

Each density-MPO site has physical dimension

\[
4.
\]

Therefore a two-site physical space has dimension

\[
4^2=16.
\]

The channel is a

\[
\boxed{
16\times16
}
\]

local superoperator.

Construct the two-site tensor

\[
\Theta_{\alpha\beta}^{\mu\nu}
=
\sum_\gamma
A^{[i]\mu}_{\alpha\gamma}
A^{[i+1]\nu}_{\gamma\beta}.
\]

Apply the correlated channel:

\[
\Theta'^{\mu'\nu'}
=
\sum_{\mu,\nu}
S^{(2)}_{\mu'\nu',\mu\nu}
\Theta^{\mu\nu}.
\]

Then reshape as

\[
(\alpha,\mu')
\quad\text{versus}\quad
(\nu',\beta)
\]

and SVD:

\[
\Theta'=USV^\dagger.
\]

Split back into two MPO tensors and truncate as required.

This is almost exactly analogous to applying a two-qubit unitary to an MPS.

The main difference is

\[
d_{\rm physical}=4
\]

instead of

\[
d_{\rm physical}=2.
\]

---

# 36. When correlated noise increases MPO bond dimension

Consider

\[
\mathcal E(\rho)
=
(1-p)\rho+
p\,ZZ\rho ZZ.
\]

As a superoperator,

\[
S
=
(1-p)
(I\otimes I)_{\rm ket/bra}
+
p
(ZZ)\otimes(ZZ)^*.
\]

This channel contains correlations across the two sites.

Its operator-Schmidt rank across that bond is larger than one, so applying it can increase MPO bond dimension.

By contrast,

\[
\mathcal E_1\otimes\mathcal E_2
\]

can be applied as two independent one-site operations without bond growth.

This gives a useful interpretation:

\[
\boxed{
\text{MPO bond growth from noise}
\leftrightarrow
\text{spatial correlations created by the channel}.
}
\]

Not every noisy channel increases the MPO complexity.

---

# 37. The channel representation I would use internally

For your simulator I would probably have something conceptually like

\[
\texttt{QuantumChannel}
\]

with either:

\[
\{K_i\}
\]

or directly a local superoperator

\[
S.
\]

For example:

\[
\texttt{AmplitudeDamping}(\gamma)
\]

generates

\[
S_{\rm AD},
\]

\[
\texttt{PureDephasing}(\eta)
\]

generates

\[
S_\phi,
\]

\[
\texttt{Depolarizing}(p)
\]

generates

\[
S_{\rm dep},
\]

and

\[
\texttt{ThermalRelaxation}
(T_1,T_2,t,p_e)
\]

generates

\[
S_{\rm therm}.
\]

Then

\[
\texttt{CorrelatedChannel}(q_1,q_2,S_{16\times16})
\]

handles two-qubit noise.

---

# 38. Gate-duration-based noise becomes very natural

Suppose you have a CNOT with duration

\[
t_g=300\ {\rm ns}.
\]

For qubit \(i\),

\[
q_i=e^{-t_g/T_{1,i}},
\]

and

\[
\eta_i=e^{-t_g/T_{2,i}}.
\]

You can construct the thermal relaxation channel for each qubit during the gate.

Then perhaps apply an additional two-qubit correlated channel representing:

- correlated \(ZZ\) phase noise;
- crosstalk;
- common-bath relaxation;
- two-qubit depolarization.

Your effective gate map becomes something like

\[
\boxed{
\mathcal E_{\rm gate}
=
\mathcal N_{\rm correlated}
\circ
(\mathcal N_1\otimes\mathcal N_2)
\circ
\mathcal U_{\rm ideal}.
}
\]

This is substantially more realistic than attaching a single arbitrary "error probability" to the CNOT.

---

# 39. One important distinction: coherent correlated error

Suppose instead the unwanted interaction is

\[
H_{\rm xtalk}
=
\epsilon Z_1Z_2.
\]

Then

\[
U_{\rm err}
=
e^{-i\epsilon t Z_1Z_2}.
\]

That's a **coherent correlated error**.

It is still unitary.

Your statevector/MPS simulator can simulate it perfectly.

If instead \(\epsilon\) fluctuates randomly or is coupled to an inaccessible environment,

\[
\epsilon\rightarrow\epsilon(E),
\]

and you average over the environment, it becomes a genuine decohering correlated channel.

This is the useful dividing line:

\[
\boxed{
\text{unknown but fixed unitary}
\neq
\text{open-system noise}.
}
\]

---

# 40. A compact comparison

| Noise | Microscopic idea | Kraus rank | Fixed point | Energy exchange | 1-site MPO bond growth |
|---|---|---:|---|---|---|
| Amplitude damping | excitation leaves into vacuum | 2 | \(|0\rangle\) | yes | none |
| Pure dephasing | environment learns \(Z\) state | 2 | many | no | none |
| Depolarizing | environment records random Pauli | up to 4 | \(I/2\) | not specifically | none |
| Thermal relaxation | exchange with thermal bath | up to 4 | Gibbs state | yes, both directions | none |
| Correlated dephasing | common \(Z\)-coupled bath | multi-qubit | model dependent | no | potentially |
| Collective damping | common excitation bath | multi-qubit | model dependent | yes | potentially |

---

# 41. The conceptual chain is therefore very clean

For amplitude damping:

\[
\boxed{
|1,0_E\rangle
\rightarrow
\sqrt{1-\gamma}|1,0_E\rangle+
\sqrt{\gamma}|0,1_E\rangle
}
\]

\[
\Downarrow
\]

\[
K_0,K_1
\]

\[
\Downarrow
\]

\[
\rho'
=
K_0\rho K_0^\dagger+
K_1\rho K_1^\dagger
\]

\[
\Downarrow
\]

\[
S_{\rm AD}^{4\times4}
\]

\[
\Downarrow
\]

local MPO physical-index transformation.

For two-qubit correlated noise,

\[
S+E
\]

has a common environmental degree of freedom,

\[
\Downarrow
\]

the Kraus operators act jointly on two qubits,

\[
K_k\in\mathbb C^{4\times4},
\]

\[
\Downarrow
\]

the Liouville operator is

\[
S\in\mathbb C^{16\times16},
\]

\[
\Downarrow
\]

you apply it exactly like a two-site gate to your MPO and then SVD/compress.

That, in my view, is the cleanest bridge between the **open-system physics** and the architecture of the two simulators.

### References from the literature search

[1] M. Arsenijević & N. Banković, [“Microscopic derivation of the one qubit Kraus operators for amplitude and phase damping”](https://consensus.app/papers/microscopic-derivation-of-the-one-qubit-kraus-operators-arsenijević-banković/d82369ef70f35d02b0b8455df163dfa7/?utm_source=chatgpt), *arXiv: Quantum Physics* (2016), **2 citations**. The paper explicitly derives amplitude- and phase-damping Kraus operators from microscopic dynamics. 

[2] R. Srikanth & S. Banerjee, [“Squeezed generalized amplitude damping channel”](https://consensus.app/papers/squeezed-generalized-amplitude-damping-channel-srikanth-banerjee/4a43612a39375e95bd56706d71381681/?utm_source=chatgpt), *Physical Review A* 77, 012318 (2007), **111 citations**. It treats generalized amplitude damping as dissipative interaction with a thermal bath and extends it to squeezed thermal environments. 

[3] M. Arsenijević, J. Jeknić-Dugić & M. Dugić, [“Generalized Kraus Operators for the One-Qubit Depolarizing Quantum Channel”](https://consensus.app/papers/generalized-kraus-operators-for-the-onequbit-arsenijević-jeknić-dugić/c1eaa89b7e995b7d80b7d9587859257a/?utm_source=chatgpt), *Brazilian Journal of Physics* 47, 339–349 (2015), **15 citations**. It derives depolarizing-channel Kraus operators starting from a microscopic open-system model. 

[4] Y. Hama, [“Quantum Circuits for Collective Amplitude Damping in Two-Qubit Systems”](https://consensus.app/papers/quantum-circuits-for-collective-amplitude-damping-in-hama/89cd1ae9317b5388bec010c2a9371d12/?utm_source=chatgpt), *arXiv: Quantum Physics* (2020), **2 citations**. It treats collective two-qubit amplitude damping and compares circuit simulation with master-equation evolution. 

[5] J. Zou, S. Bosco & D. Loss, [“Spatially correlated classical and quantum noise in driven qubits”](https://consensus.app/papers/spatially-correlated-classical-and-quantum-noise-in-zou-bosco/4ee5551c079b5564b93de02e25f48a7f/?utm_source=chatgpt), *npj Quantum Information* 10 (2023), **47 citations**. It analyzes spatially correlated quantum and classical noise, including Markovian and non-Markovian effects. 

A useful next step would be to take the **exact internal representation you use for your density-matrix and MPO tensors** and work through how I would implement `ApplyKraus1Q`, `ApplyChannel1Q`, `ApplyChannel2Q`, and a Lindblad time step—including index ordering, contractions, SVD placement, and the difference between applying a Kraus sum versus the fused Liouville operator. That would get very close to code.

Upgrade to Consensus Pro to return 20 results per search instead of 10, and include more data like study design and key takeaways for every result.: https://consensus.app/pricing/?utm_source=chatgpt