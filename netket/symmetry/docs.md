### Encoding permutations

A permutation $\sigma$ is a function
$$\sigma : \left\{ 0, \ldots, n-1 \right\} \to \left\{ 0, \ldots, n-1 \right\} $$

If an array $a$ is thought of as a function 
$$a : \left\{ 0, \ldots, n-1 \right\} \to X $$
for some set $X$, then it is natural to encode a permutation $\sigma$ through an array $a_\sigma$ defined by $a_\sigma[k] = \sigma(k)$. The product of permutation, that is, the composition, then becomes composition of arrays, $a_\sigma \circ a_{\sigma'} = a_{\sigma \circ \sigma'}$, which is performed using array indexing: $a \circ b$ is equivalent to `a[b]`.

We can then easily deduce how to obtain the inverse of a permutation. By definition of the `argsort`, we have `a_sigma[argsort[a_sigma]] = [0, 1, ..., n-1]` such that $\sigma^{-1}$ is given by `argsort(a_sigma)`.

### (Left) action of a permutations

An action of a group $G$ over a set $X$ is a function that maps each element of $G$ onto a bijection of $X$,
$$ \pi : g \mapsto \left( \pi_g : X \to X \right) $$
which satisfies the following axioms:
$$\begin{cases}
\forall x \in X, \ \pi_e(x) = x &\text{(identity)}\\
\forall g, g' \in G, \ \forall x \in X, \ \pi_{g g'}(x) = \pi_{g}(\pi_{g'}(x)) &\text{(compatibility)}
\end{cases}$$

We note that if $X$ is a vector space, then $\pi$ is a representation. And indeed, we will first define the action of a permutation on the elements of the computational basis, and then extend it to a representation.

Let $X$ be the set of elements of the local basis. For example, for a qubit system, we would have $X = \left\{ 0, 1 \right\}$. The set of elements of the basis of the $n$-sites Hilbert space is then $X^n$. We want to define the action of a permutation $\sigma$ on an element of $X^n$. Since an element $x \in X^n$ is effectively a function
$$x :
\begin{cases}
\left\{ 0, \ldots, n-1 \right\} &\to X \\
k &\mapsto x_k
\end{cases}
$$
It is therefore very natural to use composition with $\sigma$ on the right to define the left action of $\sigma$. However, in order to satisfy the compatibility axiom of a left action, it is necessary to instead use composition with $\sigma^{-1}$. If we define $\pi_\sigma(x) = x \circ \sigma^{-1}$, then we have
\begin{align}
\pi_{\sigma \circ \sigma'}(x) &= x \circ \left( \sigma \circ \sigma' \right)^{-1} \\
&= x \circ \sigma'^{-1} \circ \sigma^{-1} \\
&= \pi_{\sigma'}(x) \circ \sigma^{-1} \\
&= \pi_\sigma(\pi_{\sigma'}(x))
\end{align}
and the identity axiom is trivially satisfied, such that $\pi$ indeed defines an action of permutations on $X^n$.

### Representation of permutations

We have now constructed a left action $\pi$ of permutations on $X^n$, the set of basis elements. It is then natural to generalize $\pi$ to an action on the Hilbert space $\mathcal H$ spanned by a basis indexed by the elements of $X^n$. We can simply define the action $U$ of permutations on $\mathcal H$ by defining its action on a basis according to $\pi$, $U_g | x \rangle = | \pi_g(x) \rangle$. More explicitly, it acts as
$$U_\sigma | x_0, x_1, \ldots, x_{n-1} \rangle = | x_{\sigma^{-1}(0)}, x_{\sigma^{-1}(1)}, \ldots, x_{\sigma^{-1}(n-1)} \rangle.$$
As we mentionned before, a left action on a Hilbert space is a representation. $U$ therefore defines a representation of $S_n$ on $\mathcal H$.

In NetKet, an operator is defined by the method `get_conn_padded`. Given a configuration `x`, it should return the configurations `x_primes` such that $\langle x | A | x' \rangle \neq 0$, and the values themselves `matrix_elements`. In the case of a permutation operator $U_\sigma$, for a given configuration $x$, we have
\begin{align}
\langle x | U_\sigma | x' \rangle &= \left( U_\sigma^\dagger | x \rangle \right)^\dagger | x' \rangle \\
&= \left( U_\sigma^{-1} | x \rangle \right)^\dagger | x' \rangle \\
&= \left( U_{\sigma^{-1}} | x \rangle \right)^\dagger | x' \rangle \\
&= \langle \pi_{\sigma^{-1}}(x) | x' \rangle \\
&= \delta_{x \circ \sigma, x'} \\
\end{align}
Therefore, the configuration $x$ is connected to a single element $x' = x \circ \sigma$, and the matrix element is always 1.