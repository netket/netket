# Representation theory of Permutations

When encoding and manipulating permutations and their associated operators on the Hilbert space it is easy to make mistakes. For optimal clarity, the following fully details and motivates the definitions taken in NetKet.

In this document, you will learn

* What a permutation is and how to encode them. 

* How to construct *representations* of permutation groups on many-body hilbert spaces for spins and fermions. 

* How to construct quantum states that transform as *irreps* (irreducible representations) of a symmetry group. 

## Encoding permutations

A permutation $\sigma$ is a function
```{math}
\sigma : \left\{ 0, \ldots, n-1 \right\} \to \left\{ 0, \ldots, n-1 \right\}
```

If an array $a$ is thought of as a function 
```{math}
a : \left\{ 0, \ldots, n-1 \right\} \to X
```
for some set $X$, then it is natural to encode a permutation $\sigma$ through an array $a_\sigma$ defined by $a_\sigma[k] = \sigma(k)$. The product of permutation, that is, the composition, then becomes composition of arrays, $a_\sigma \circ a_{\sigma'} = a_{\sigma \circ \sigma'}$, which is performed using array indexing: $a \circ b$ is equivalent to `a[b]`.

We can then easily deduce how to obtain the inverse of a permutation. By definition of the `argsort`, we have `a_sigma[argsort[a_sigma]] = [0, 1, ..., n-1]` such that $\sigma^{-1}$ is given by `argsort(a_sigma)`.

Permutations are implemented with the class `Permutation`.

## Permutation Operators

Lattice symmetries correspond to permutation operators. It is frequent to hear that a system is "translationally invariant", or "invariant under reflection". However, depending on the context, the actual symmetry operator corresponding to such statements may be very different. In particular, we will see that permutation operators on fermionic systems are different from their spin counterpart, despite the Hilbert space indices being the same.

They must be representations.


### Symmetry operator on a spin system

Let $\mathcal{H} = V^{\otimes n}$ be the Hilbert space of a system of $n$ spins, where $V$ denotes the single spin Hilbert space. For $A$ be an operator of $V$, we define as usual $A_k$ the operator of $\mathcal{H}$ that acts as $A$ on the $k$-th copy of $V$ and does nothing on the others,
```{math}
A_k = 1 \otimes \ldots \otimes \underbrace{A}_{\text{k-th copy}} \otimes \ldots \otimes 1.
```

The permutation operator associated to the permutation $\sigma \in S_n$ is the unitary operator such that
```{math}
:label: eq-perm-op
\forall k \in \{0, 1, \ldots, n-1\}, \ \forall A \in \mathcal{L}(V), \ P_\sigma A_k P_\sigma^{\dagger} = A_{\sigma(k)}.
```
At this point, we need to address three questions:
1. Does there indeed exist a unique operator $P_\sigma$ satisfying this definition?
2. What is the action of $P_\sigma$ on a basis element?
3. Is $\sigma \mapsto P_\sigma$ a representation of the symmetric group?


We will answer question 1 and 2 at once, by performing what is known in french as _analyse synthèse_. We will show that if an operator satisfies eq. (4), then we can deduce its action on the basis, and it would thus be unique. Then, we will show that, conversely, the operator defined by that specific action on the basis does satisfy eq. (4), showing that $P_\sigma$ does exist and is unique.


#### _Analyse_

Let $P_\sigma$ be an operator that satisfies eq. (4). Let us fix one element of the local basis and denote it by $| 0 \rangle$ (for example we can denote $| 0 \rangle = |s = 1/2 \rangle$ for a spin-1/2 system). For any basis element $| x_0, x_1, \ldots, x_{n-1} \rangle$, there exists $n$ local operators $A^{(k)} \in \mathcal{L}(V)$ such that $| x_k \rangle = A^{(k)} | 0 \rangle$. We therefore have
```{math}
\begin{align}
| x_0, x_1, \ldots, x_{n-1} \rangle &= A^{(0)} \otimes A^{(1)} \otimes \ldots \otimes A^{(n-1)} | 0, 0, \ldots, 0 \rangle \\
&= A_0^{(0)} A_1^{(1)} \ldots A_{n-1}^{(n-1)} | 0, 0, \ldots, 0 \rangle.
\end{align}
```

Then, applying eq. (4), we see that 
```{math}
\begin{align}
P_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle &= P_\sigma A_0^{(0)} P_\sigma^{\dagger} P_\sigma A_1^{(1)} P_\sigma^{\dagger} \ldots P_\sigma A_{n-1}^{(n-1)} P_\sigma^{\dagger} P_\sigma | 0, 0, \ldots, 0 \rangle \\
&= A_{\sigma(0)}^{(0)} A_{\sigma(1)}^{(1)} \ldots A_{\sigma(n-1)}^{(n-1)} P_\sigma | 0, 0, \ldots, 0 \rangle \\
&= A^{(\sigma^{-1}(0))} \otimes A^{(\sigma^{-1}(1))} \otimes \ldots \otimes A^{(\sigma^{-1}(n-1))} P_\sigma | 0, 0, \ldots, 0 \rangle
\end{align}
```

We therefore need to identify the state $| \psi \rangle = P_\sigma | 0, 0, \ldots, 0 \rangle$. In order to do so, we have to return to eq. (4), which can be rearranged as
```{math}
\begin{equation}
\forall k \in \{0, 1, \ldots, n-1\}, \ \forall A \in \mathcal{L}(V), \ P_\sigma^{\dagger} A_k P_\sigma = A_{\sigma^{-1}(k)}.
\end{equation}
```
Taking the braket with $| 0, 0, \ldots, 0 \rangle$, we obtain
```{math}
\begin{equation}
\forall k \in \{0, 1, \ldots, n-1\}, \ \forall A \in \mathcal{L}(V), \ \langle \psi | A_k | \psi \rangle = \langle 0, 0, \ldots, 0 | A_{\sigma^{-1}(k)} | 0, 0, \ldots, 0 \rangle.
\end{equation}
```
Since $| 0, 0, \ldots, 0 \rangle$ is a product state with the same local state in each subsystem, the average value is independent of the subsystem on which $A$ applies, such that 
```{math}
\begin{equation}
\forall k \in \{0, 1, \ldots, n-1\}, \ \forall A \in \mathcal{L}(V), \ \langle \psi | A_k | \psi \rangle = \langle 0, 0, \ldots, 0 | A_k | 0, 0, \ldots, 0 \rangle.
\end{equation}
```

The states $| \psi \rangle$ and $| 0, 0, \ldots, 0 \rangle$ have the same average values on all local operators, and therefore on all operators. We deduce that
```{math}
\begin{equation}
P_\sigma | 0, 0, \ldots, 0 \rangle = e^{i \phi} | 0, 0, \ldots, 0 \rangle
\end{equation}
```
for some phase $\phi$.

We can now substitute $P_\sigma | 0, 0, \ldots, 0 \rangle$ in eq. (9) to obtain
```{math}
\begin{align}
P_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle &= e^{i \phi} A^{(\sigma^{-1}(0))} \otimes A^{(\sigma^{-1}(1))} \otimes \ldots \otimes A^{(\sigma^{-1}(n-1))} | 0, 0, \ldots, 0 \rangle \\
&= e^{i \phi} | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle
\end{align}
```
The global phase $e^{i \phi}$ is irrelevant for a unitary operator, and can be dropped without changing the action of the operator on the Hilbert space.

We have thus shown that if an operator $P_\sigma$ satisfies eq. (4), then it must be the operator defined by
```{math}
\begin{equation}
P_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle = | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle.
\end{equation}
```

#### _Synthèse_

Let us now show that the operator $P_\sigma$ defined by eq. (16) does indeed satisfy eq. (4).





#### $\sigma \mapsto P_\sigma$ is a representation


### (Left) action of a permutations

An action of a group $G$ over a set $X$ is a function that maps each element of $G$ onto a bijection of $X$,
```{math}
\begin{equation}
\pi : g \mapsto \left( \pi_g : X \to X \right)
\end{equation}
```
which satisfies the following axioms:
```{math}
\begin{equation}
\begin{cases}
\forall x \in X, \ \pi_e(x) = x &\text{(identity)}\\
\forall g, g' \in G, \ \forall x \in X, \ \pi_{g g'}(x) = \pi_{g}(\pi_{g'}(x)) &\text{(compatibility)}
\end{cases}
\end{equation}
```

We note that if $X$ is a vector space, then $\pi$ is a representation. And indeed, we will first define the action of a permutation on the elements of the computational basis, and then extend it to a representation.

Let $X$ be the set of elements of the local basis. For example, for a qubit system, we would have $X = \left\{ 0, 1 \right\}$. The set of elements of the basis of the $n$-sites Hilbert space is then $X^n$. We want to define the action of a permutation $\sigma$ on an element of $X^n$. Since an element $x \in X^n$ is effectively a function
```{math}
\begin{equation}
x :
\begin{cases}
\left\{ 0, \ldots, n-1 \right\} &\to X \\
k &\mapsto x_k
\end{cases}
\end{equation}
```
It is therefore very natural to use composition with $\sigma$ on the right to define the left action of $\sigma$. However, in order to satisfy the compatibility axiom of a left action, it is necessary to instead use composition with $\sigma^{-1}$. If we define $\pi_\sigma(x) = x \circ \sigma^{-1}$, then we have
```{math}
\begin{align}
\pi_{\sigma \circ \sigma'}(x) &= x \circ \left( \sigma \circ \sigma' \right)^{-1} \\
&= x \circ \sigma'^{-1} \circ \sigma^{-1} \\
&= \pi_{\sigma'}(x) \circ \sigma^{-1} \\
&= \pi_\sigma(\pi_{\sigma'}(x))
\end{align}
```
and the identity axiom is trivially satisfied, such that $\pi$ indeed defines an action of permutations on $X^n$.

### Representation of permutations

We have now constructed a left action $\pi$ of permutations on $X^n$, the set of basis elements. It is then natural to generalize $\pi$ to an action on the Hilbert space $\mathcal{H}$ spanned by a basis indexed by the elements of $X^n$. We can simply define the action $U$ of permutations on $\mathcal{H}$ by defining its action on a basis according to $\pi$, $U_g | x \rangle = | \pi_g(x) \rangle$. More explicitly, it acts as
```{math}
\begin{equation}
U_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle = | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle.
\end{equation}
```
As we mentioned before, a left action on a Hilbert space is a representation. $U$ therefore defines a representation of $S_n$ on $\mathcal{H}$.

In NetKet, an operator is defined by the method `get_conn_padded`. Given a configuration `x`, it should return the configurations `x_primes` such that $\langle x | A | x' \rangle \neq 0$, and the values themselves `matrix_elements`. In the case of a permutation operator $U_\sigma$, for a given configuration $x$, we have
```{math}
\begin{align}
\langle x | U_\sigma | x' \rangle &= \left( U_\sigma^{\dagger} | x \rangle \right)^\dagger | x' \rangle \\
&= \left( U_\sigma^{-1} | x \rangle \right)^\dagger | x' \rangle \\
&= \left( U_{\sigma^{-1}} | x \rangle \right)^\dagger | x' \rangle \\
&= \langle \pi_{\sigma^{-1}}(x) | x' \rangle \\
&= \delta_{x \circ \sigma, x'} \\
\end{align}
```
Therefore, the configuration $x$ is connected to a single element $x' = x \circ \sigma$, and the matrix element is always 1.

### Representation of permutations on Fermionic Fock spaces
Let $\mathcal{F}$ be a fermionic Fock space with $m$ single-particle states and $G$ a subgroup of $\mathcal{S}_m$. Elements $|n\rangle$ of $\mathcal{F}$ are expressed of products of fermionic creation operators acting on the Fock vacuum $|0 \rangle$: 

```{math}
\begin{equation}
 |n\rangle = \hat{c}^{\dagger n_{1}}_{1} \hat{c}^{\dagger n_{2}}_{2} \ldots \hat{c}^{\dagger n_{m}}_{m} |0 \rangle
\end{equation}
```
where $n_{i} \in \{0,1\}$ due to the Pauli exclusion principle and a canonical ordering has been defined for the single particle states. The correct way to define a representation $\hat{U}: G \to \mathcal{F}$ is via the following rules, which are analogous to the case of spins.

* For all $g \in G$, $\hat{U}_g |0\rangle= |0\rangle$
* For all $g \in G$ and all single particle states $i$, $\hat{U}_g \hat{c}^\dagger_{i} \hat{U}_g^{\dagger} = \hat{c}^\dagger_{g(i)}$.

First, we will use this definition to determine a compact expression for the action of the representation on basis states. Then, we will verify that the rules given above are consistent with the definition of a representation. 

The action of the representation of $|n\rangle$ is given by inserting a factor $\hat{U}_g^{\dagger} \hat{U}_g$ in between each creation operator. Therefore, for all $g \in G$

```{math}
\begin{align}
 \hat{U}_g |n\rangle &= \hat{U}_g \hat{c}^{\dagger n_{1}}_{1} \hat{U}_g^{\dagger} \hat{U}_g \ldots \hat{U}_g \hat{c}^{\dagger n_{m}}_{m} \hat{U}_g^{\dagger} \hat{U}_g|0\rangle  \nonumber \\
 &= \hat{c}^{\dagger n_{1}}_{g(1)} \hat{c}^{\dagger n_{2}}_{g(2)} \ldots \hat{c}^{\dagger n_{m}}_{g(m)} |0 \rangle \nonumber\\
 &= \xi_g(n) \hat{c}^{\dagger n'_{1}}_{1} \hat{c}^{\dagger n'_{2}}_{2} \ldots \hat{c}^{\dagger n'_{m}}_{m} |0\rangle.
\end{align}
```
Immediately after applying $\hat{U}_g$ to the state, the creation operators may no longer appear in the canonical order. Consequently, a sign $\xi_g(n)$ is induced when ordering them. 

In the transformed state, mode $i$ is occupied if there was $j$ such that $n_{j}=1$ and $g(j) = i$. In other words, mode $i$ is occupied in the transformed state if and only if mode $g^{-1}(i)$ was occupied in the original state. Consequently, the occupation number $n'_{i}$ in the transformed state is $n_{g^{-1}(i)}.$ This leads to the following expression for the action of any element of the representation $\hat{U}_g$ on any state of the canonical basis 

```{math}
\begin{equation}
\hat{U}_g |n\rangle = \xi_g(n) | n \circ g^{-1} \rangle,
\end{equation}
```
where $n \circ g^{-1}$ corresponds to the left action of $g$ on the function $n: \{1,2, \ldots m\} \to \{0,1\}$.
The above equation satisfies the condition for $\hat{U}$ to be a representation, since for all $g,g' \in G$

```{math}
\begin{equation}
 \hat{U}_g \hat{U}_{g'} |n \rangle  = \xi_g(n)\xi_{g'}(n) |n \circ g^{\prime -1} \circ g^{-1} \rangle = \xi_{gg'}(n) | n \circ (gg')^{-1} \rangle = \hat{U}_{gg'} |n\rangle.
\end{equation}
```

**Example**


For illustrative purpose, let's look at the following example. Consider a system with 4 fermion modes and the following state with 3 fermions: $|n\rangle= |1110\rangle= \hat{c}^{\dagger n_1}_1 \hat{c}^{\dagger n_2}_2 \hat{c}^{\dagger n_3}_3 \hat{c}^{\dagger n_4}_4 |0\rangle$ with $n_1=n_2=n_3=1$ and $n_4=0$. Let $g$ be the following permutation expressed in two-row notation (the expression of the inverse is also provided)
```{math}
\begin{equation}
    g = \begin{pmatrix}
        1 & 2 & 3 & 4 \\
        4 & 3 & 1 & 2    \end{pmatrix}, \quad g^{-1} = \begin{pmatrix}
        1 & 2 & 3 & 4 \\
        3 & 4 & 2 & 1
        
    \end{pmatrix}
\end{equation}
```
Then, $\hat{U}_g$ acts on $|n\rangle$ using the rules above: $\hat{U}_g |n\rangle = \hat{c}^{\dagger n_1}_4 \hat{c}^{\dagger n_2}_3 \hat{c}^{\dagger n_3}_1 \hat{c}^{\dagger n_4}_2 |0\rangle = -\hat{c}^{\dagger n_3}_1 \hat{c}^{\dagger n_2}_3 \hat{c}^{\dagger n_1}_4 |0\rangle$ (since $n_4=0$). It is clear to see now that $n'_1 = n_3 = n_{g^{-1}(1)}$, $n_3' = n_2 = n_{g^{-1}(2)}$ and $n_4 = n_1 = n_{g^{-1}(4)}$.

## Symmetrizing Quantum States

### Why are symmetries important? 
An important lemma of representation theory, namely **Schur's lemma** tells us that when a quantum Hamiltonian $\hat H$ commutes with a set of operators that correspond to a representation of a finite group $G$, then 

* When $\hat H$ is written in a basis of states that transform according to irreps of $G$, it takes on a block diagonal form. 

* The irreps of $G$ can be used to label the eigenstates of $\hat H$. Each eigenstate belongs to a specific irrep, which can serve as a *quantum* number for classifying the state. 

In the context of Variational Monte Carlo (VMC) it can be advantageous to exploit the symmetries of the Hamiltonian to obtain better energies and physically consistent observables.

### The Symmetrizer Projector
During optimization, a variational state may not exhibit these symmetries so they may be enforced using quantum number projection. 

To set things up correctly, let's consider a permutation group $G$ and a representation $\hat{U}: G \to \mathcal{H}$. We label the irreps of $G$ using greek letters, e.g $\mu$. 

Now we can define the symmetrizer for a particular irrep $\mu$ as

```{math}
\begin{equation}
\hat{\mathcal{P}}_\mu = \frac{1}{|G|} \sum_{g \in G} \chi_\mu^*(g) \hat{U}_g
\end{equation}
```
where $|G|$ is the order of a group and $\chi_\mu(g)$ is the character of the irrep evaluated on element $g$. We can show that $\hat{\mathcal{P}}_\mu$ is a projection operator. Starting from

```{math}
\begin{equation}
 \hat{\mathcal{P}}_\mu^2 = \frac{1}{|G|^2} \sum_{g,g'} \chi_\mu^*(g) \chi_\mu^*(g') \hat{U}_{gg'}
\end{equation}
```
we can use the group rearrangement theorem to write $gg' = h$ where $h$ is another element of $G$. This means that $g' = g^{-1} h$   and therefore

```{math}
\begin{equation}
 \hat{\mathcal{P}}_\mu^2 = \frac{1}{|G|^2} \sum_g \chi_\mu^*(g) \chi_\mu(g) \sum_h \chi_{\mu}^*(h) \hat{U}_h = \hat{\mathcal{P}}_\mu.
\end{equation}
```
We used the orthogonality of the characters as well as the property $\chi_{\mu}(g) \chi_\mu(g') = \chi_\mu(gg')$. 

The symmetrizer also has the following property: for all $g \in G$, $[ \hat{\mathcal{P}}_\mu, \hat{U}_g]= 0$. For example,

```{math}
\begin{align}
 \hat{\mathcal{P}}_\mu \hat{U}_{g'} &= \frac{1}{|G|} \sum_{g \in G} \chi_\mu^*(g) \hat{U}_{gg'} = \frac{1}{|G|} \sum_{h \in G} \chi_\mu^*(h g^{\prime -1}) \hat{U}_h \nonumber \\
 &= \chi_\mu(g') \hat{\mathcal{P}}_\mu.
\end{align}
```
Consequently, the symmetrizer can be used to construct wavefunction amplitudes that transform like an irrep of $G$. For spins, we have

```{math}
\begin{align}
 \psi_\mu(x) &= \langle x | \hat{\mathcal{P}}_\mu |\psi\rangle = \frac{1}{|G|} \sum_{g \in G} \chi_\mu^*(g) \langle x| \hat{U}_g |\psi \rangle \nonumber \\
 &=\frac{1}{|G|} \sum_{g \in G} \chi_\mu^*(g) \psi(x \circ g)
\end{align}
```
For fermions, the extra sign needs to be included: 

```{math}
\begin{equation}
\psi_\mu(n) = \frac{1}{|G|} \sum_{g \in G} \chi_\mu^*(g) \xi_{g^{-1}}(n) \psi(n \circ g).
\end{equation}
```
This followed from $\langle n | \hat{U}_g = \left( \hat{U}_g | n \rangle \right)^\dagger$. 