When encoding and manipulating permutations and their associated operators on the Hilbert space it is easy to make mistakes. For optimal clarity, the following fully details and motivates the definitions taken in NetKet.

### Encoding permutations

A permutation $\sigma$ is a function
$$
\begin{equation}
\sigma : \left\{ 0, \ldots, n-1 \right\} \to \left\{ 0, \ldots, n-1 \right\}
\end{equation}
$$

If an array $a$ is thought of as a function 
$$
\begin{equation}
a : \left\{ 0, \ldots, n-1 \right\} \to X
\end{equation}
$$
for some set $X$, then it is natural to encode a permutation $\sigma$ through an array $a_\sigma$ defined by $a_\sigma[k] = \sigma(k)$. The product of permutation, that is, the composition, then becomes composition of arrays, $a_\sigma \circ a_{\sigma'} = a_{\sigma \circ \sigma'}$, which is performed using array indexing: $a \circ b$ is equivalent to `a[b]`.

We can then easily deduce how to obtain the inverse of a permutation. By definition of the `argsort`, we have `a_sigma[argsort[a_sigma]] = [0, 1, ..., n-1]` such that $\sigma^{-1}$ is given by `argsort(a_sigma)`.

Permutations are implemented with the class `Permutation`.

## Permutation Operators

Lattice symmetries correspond to permutation operators. It is frequent to hear that a system is "translationnally invariant", or "invariant under reflection". However, depending on the context, the actual symmetry operator corresponding to such statements may be very different. In particular, we will see that permutation operators on fermionic systems are different from their spin counterpart, despite the Hilbert space indices being the same.

They must be representations.


### Symmetry operator on a spin system

Let $\mathcal H = V^{\otimes n}$ be the Hilbert space of a system of $n$ spins, where $V$ denotes the single spin Hilbert space. For $A$ be an operator of $V$, we define as usual $A_k$ the operator of $\mathcal H$ that acts as $A$ on the $k$-th copy of $V$ and does nothing on the others,
$$
\begin{equation}
A_k = 1 \otimes \ldots \otimes \underbrace{A}_{\text{$k$-th copy}} \otimes \ldots \otimes 1.
\end{equation}
$$

The permutation operator associated to the permutation $\sigma \in S_n$ is the unitary operator such that
$$
\begin{equation}
\forall k \in \llbracket 0, n-1 \rrbracket, \ \forall A \in \mathcal L(V), \ P_\sigma A_k P_\sigma^\dagger = A_{\sigma(k)}.
\end{equation}
$$
At this point, we need to address three questions:
1. Does there indeed exist a unique operator $P_\sigma$ satisfying this definition?
2. What is the action of $P_\sigma$ on a basis element?
3. Is $\sigma \mapsto P_\sigma$ a representation of the symmetric group?


We will answer question 1 and 2 at once, by performing what is known in french as _analyse synthèse_. We will show that if an operator satisfies eq. (4), then we can deduce its action on the basis, and it would thus be unique. Then, we will show that, conversely, the operator defined by that specific action on the basis does satisfy eq. (4), showing that $P_\sigma$ does exist and is unique.


#### _Analyse_

Let $P_\sigma$ be an operator that satisfies eq. (4). Let us fix one element of the local basis and denote it by $| 0 \rangle$ (for example we can denote $| 0 \rangle = |s = 1/2 \rangle$ for a spin-1/2 system). For any basis element $| x_0, x_1, \ldots, x_{n-1} \rangle$, there exists $n$ local operators $A^{(k)} \in \mathcal L(V)$ such that $| x_k \rangle = A^{(k)} | 0 \rangle$. We therefore have
$$
\begin{align}
| x_0, x_1, \ldots, x_{n-1} \rangle &= A^{(0)} \otimes A^{(1)} \otimes \ldots \otimes A^{(n-1)} | 0, 0, \ldots, 0 \rangle \\
&= A_0^{(0)} A_1^{(1)} \ldots A_{n-1}^{(n-1)} | 0, 0, \ldots, 0 \rangle.
\end{align}
$$

Then, applying eq. (4), we see that 
$$
\begin{align}
P_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle &= P_\sigma A_0^{(0)} P_\sigma^\dagger P_\sigma A_1^{(1)} P_\sigma^\dagger \ldots P_\sigma A_{n-1}^{(n-1)} P_\sigma^\dagger P_\sigma | 0, 0, \ldots, 0 \rangle \\
&= A_{\sigma(0)}^{(0)} A_{\sigma(1)}^{(1)} \ldots A_{\sigma(n-1)}^{(n-1)} P_\sigma | 0, 0, \ldots, 0 \rangle \\
&= A^{(\sigma^{-1}(0))} \otimes A^{(\sigma^{-1}(1))} \otimes \ldots \otimes A^{(\sigma^{-1}(n-1))} P_\sigma | 0, 0, \ldots, 0 \rangle
\end{align}
$$

We therefore need to identify the state $| \psi \rangle = P_\sigma | 0, 0, \ldots, 0 \rangle$. In order to do so, we have to return to eq. (4), which can be rearranged as
$$
\begin{equation}
\forall k \in \llbracket 0, n-1 \rrbracket, \ \forall A \in \mathcal L(V), \ P_\sigma^\dagger A_k P_\sigma = A_{\sigma^{-1}(k)}.
\end{equation}
$$
Taking the braket with $| 0, 0, \ldots, 0 \rangle$, we obtain
$$
\begin{equation}
\forall k \in \llbracket 0, n-1 \rrbracket, \ \forall A \in \mathcal L(V), \ \langle \psi | A_k | \psi \rangle = \langle 0, 0, \ldots, 0 | A_{\sigma^{-1}(k)} | 0, 0, \ldots, 0 \rangle.
\end{equation}
$$
Since $| 0, 0, \ldots, 0 \rangle$ is a product state with the same local state in each subsystem, the average value is independent of the subsystem on which $A$ applies, such that 
$$
\begin{equation}
\forall k \in \llbracket 0, n-1 \rrbracket, \ \forall A \in \mathcal L(V), \ \langle \psi | A_k | \psi \rangle = \langle 0, 0, \ldots, 0 | A_k | 0, 0, \ldots, 0 \rangle.
\end{equation}
$$

The states $| \psi \rangle$ and $| 0, 0, \ldots, 0 \rangle$ have the same average values on all local operators, and therefore on all operators. We deduce that
$$
\begin{equation}
P_\sigma | 0, 0, \ldots, 0 \rangle = e^{i \phi} | 0, 0, \ldots, 0 \rangle
\end{equation}
$$
for some phase $\phi$.

We can now substitute $P_\sigma | 0, 0, \ldots, 0 \rangle$ in eq. (9) to obtain
$$
\begin{align}
P_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle &= e^{i \phi} A^{(\sigma^{-1}(0))} \otimes A^{(\sigma^{-1}(1))} \otimes \ldots \otimes A^{(\sigma^{-1}(n-1))} | 0, 0, \ldots, 0 \rangle \\
&= e^{i \phi} | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle
\end{align}
$$
The global phase $e^{i \phi}$ is irrelevant for a unitary operator, and can be dropped without changing the action of the operator on the Hilbert space.

We have thus shown that if an operator $P_\sigma$ satisfies eq. (4), then it must be the operator defined by
$$
\begin{equation}
P_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle = | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle.
\end{equation}
$$

#### _Synthèse_

Let us now show that the operator $P_\sigma$ defined by eq. (16) does indeed satisfy eq. (4).





#### $\sigma \mapsto P_\sigma$ is a representation


### (Left) action of a permutations

An action of a group $G$ over a set $X$ is a function that maps each element of $G$ onto a bijection of $X$,
$$
\begin{equation}
\pi : g \mapsto \left( \pi_g : X \to X \right)
\end{equation}
$$
which satisfies the following axioms:
$$
\begin{equation}
\begin{cases}
\forall x \in X, \ \pi_e(x) = x &\text{(identity)}\\
\forall g, g' \in G, \ \forall x \in X, \ \pi_{g g'}(x) = \pi_{g}(\pi_{g'}(x)) &\text{(compatibility)}
\end{cases}
\end{equation}
$$

We note that if $X$ is a vector space, then $\pi$ is a representation. And indeed, we will first define the action of a permutation on the elements of the computational basis, and then extend it to a representation.

Let $X$ be the set of elements of the local basis. For example, for a qubit system, we would have $X = \left\{ 0, 1 \right\}$. The set of elements of the basis of the $n$-sites Hilbert space is then $X^n$. We want to define the action of a permutation $\sigma$ on an element of $X^n$. Since an element $x \in X^n$ is effectively a function
$$
\begin{equation}
x :
\begin{cases}
\left\{ 0, \ldots, n-1 \right\} &\to X \\
k &\mapsto x_k
\end{cases}
\end{equation}
$$
It is therefore very natural to use composition with $\sigma$ on the right to define the left action of $\sigma$. However, in order to satisfy the compatibility axiom of a left action, it is necessary to instead use composition with $\sigma^{-1}$. If we define $\pi_\sigma(x) = x \circ \sigma^{-1}$, then we have
$$
\begin{align}
\pi_{\sigma \circ \sigma'}(x) &= x \circ \left( \sigma \circ \sigma' \right)^{-1} \\
&= x \circ \sigma'^{-1} \circ \sigma^{-1} \\
&= \pi_{\sigma'}(x) \circ \sigma^{-1} \\
&= \pi_\sigma(\pi_{\sigma'}(x))
\end{align}
$$
and the identity axiom is trivially satisfied, such that $\pi$ indeed defines an action of permutations on $X^n$.

### Representation of permutations

We have now constructed a left action $\pi$ of permutations on $X^n$, the set of basis elements. It is then natural to generalize $\pi$ to an action on the Hilbert space $\mathcal H$ spanned by a basis indexed by the elements of $X^n$. We can simply define the action $U$ of permutations on $\mathcal H$ by defining its action on a basis according to $\pi$, $U_g | x \rangle = | \pi_g(x) \rangle$. More explicitly, it acts as
$$
\begin{equation}
U_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle = | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle.
\end{equation}
$$
As we mentionned before, a left action on a Hilbert space is a representation. $U$ therefore defines a representation of $S_n$ on $\mathcal H$.

In NetKet, an operator is defined by the method `get_conn_padded`. Given a configuration `x`, it should return the configurations `x_primes` such that $\langle x | A | x' \rangle \neq 0$, and the values themselves `matrix_elements`. In the case of a permutation operator $U_\sigma$, for a given configuration $x$, we have
$$
\begin{align}
\langle x | U_\sigma | x' \rangle &= \left( U_\sigma^\dagger | x \rangle \right)^\dagger | x' \rangle \\
&= \left( U_\sigma^{-1} | x \rangle \right)^\dagger | x' \rangle \\
&= \left( U_{\sigma^{-1}} | x \rangle \right)^\dagger | x' \rangle \\
&= \langle \pi_{\sigma^{-1}}(x) | x' \rangle \\
&= \delta_{x \circ \sigma, x'} \\
\end{align}
$$
Therefore, the configuration $x$ is connected to a single element $x' = x \circ \sigma$, and the matrix element is always 1.


NORD VPN! 