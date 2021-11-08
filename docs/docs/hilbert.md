(Hilbert)=
# The Hilbert module

```{eval-rst}
.. currentmodule:: netket.hilbert
```

The [Hilbert](`netket.hilbert`) module defines the Abstract Hilbert Space API and some concrete implementations, such as {ref}`netket.hilbert.Spin`, {ref}`netket.hilbert.Fock` or others.
An `Hilbert` object represents an Hilbert Space and a choice of computational basis on this space.
They are needed to construct most other objects in NetKet, but they can also be useful to experiment and validate Variational Ansatzes.

Hilbert space objects are all sub-classes of the abstract class {ref}`netket.hilbert.AbstractHilbert`, which defines the general API respected by all implementations. 
You can see a birds-eye of the inheritance diagram among the various kinds of Hilbert spaces included with NetKet below (you can click on the nodes in the graph to go to their API documentation page).

```{eval-rst}
.. inheritance-diagram:: netket.hilbert
	:top-classes: netket.hilbert.AbstractHilbert
	:parts: 1

```

{ref}`netket.hilbert.AbstractHilbert` makes very few assumptions on the structure of the resulting space and you will generally very rarely interact with it directly.

There are then two more abstract Hilbert space types: {ref}`netket.hilbert.DiscreteHilbert`, representing Hilbert spaces where the local degrees of freedom are countable, and {ref}`netket.hilbert.ContinuousHilbert`, representing the Hilbert spaces with continuous bases, such as particles in a box. 

Those two abstract types are very different: `ContinuousHilbert` spaces are still experimental and we don't support yet many ways to manipulate them, while `DiscreteHilbert` spaces are much more developed and offer many utilities and handy functionalities.

Among the discrete Hilbert spaces the most important ones are {ref}`netket.hilbert.HomogeneousHilbert` spaces, where the local degrees of freedom are identical among different sites, of which there exist the {ref}`netket.hilbert.Fock`, {ref}`netket.hilbert.Spin` and {ref}`netket.hilbert.Qubit` implementation.

{ref}`netket.hilbert.TensorHilbert` represents tensor products of different homogeneous hilbert spaces, therefore it is not homogeneous. You can use it to represent composite systems such as spin-boson setups.

{ref}`netket.hilbert.DoubledHilbert` represents a space doubled through Choi's Isomorphism, which is where the density matrix of a physical system lives. It is used to work with dissipative/open systems.

## The AbstractHilbert interface

As we mentioned before, an Hilbert object represents at the same time a choice of Hilbert space and computational basis. 
The reason why we need to specify an Hilbert space is evident, I hope.
The reason why we need to specify a computational basis is the following: with Variational methods we often have to perform summations (or sample) the hilbert space. For example, we often write the wavefunction as 

$$
|\psi\rangle = \sum_{\bf{\sigma}\in\mathcal{H}} \psi(\sigma) |\bf{\sigma}\rangle
$$

The choice of computational basis affects the values that those $\bf{\sigma} = |\sigma_0, \sigma_1, \sigma_2, \dots, \sigma_N\rangle $ will take
To give an example: when working with Qubits we often take as the basis the $\hat{Z}$ basis, where $\sigma_i=\{0,1\}$, but we could have also chosen the $\hat{Y}$ or $\hat{X}$ basis, where operators would have different basis elements.

Unfortunately, all the operators shipping with NetKet hardcode the choice of $\hat{Z}$ (or number-basis in Fock space) as the computational basis, but eventually we might relax this constraint.

### Attributes

All Hilbert spaces expose one attribute: {attr}`~netket.hilbert.AbstractHilbert.size` 
This is an integer that exposes how many degrees of freedom has the basis of the Hilbert space.
For Discrete hilbert spaces this corresponds exactly to the number of lattice sites or spins in the system.
Therefore, elements of the basis of an $N$ spin-$1/2$ system are vectors in $\mathbb{R}^N$, an $N-$ dimensional space.

As NetKet is a package focused on monte-carlo calculations, we also need a way to generate random elements from the basis of an hilbert space.
This can be achieved through the method {meth}`~netket.hilbert.AbstractHilbert.random_state`. 

```{eval-rst}
.. autofunction:: netket.hilbert.AbstractHilbert.random_state

```

`random_state` behaves similarly to {fun}`jax.random.uniform`: the first argument is a Jax PRNGKey, the second is the shape or number of resulting elements and the third is the dtype of the output (which defaults to `jnp.float32`, or single precision.
The resulting basis elements will be distributed uniformly.

```note
If you are not familiar with Jax random number generators: Jax PRNGKey is the state of the Pseudo-random number generator, that determines what will be the next random numbers generated. To learn more about it, refer to [this documentation](https://jax.readthedocs.io/en/latest/jax.random.html)).
```

### Composing Hilbert spaces

Hilbert spaces can be composed together.
The syntax to do that is Python's multiplication operator, `*`, which will be interpreted as a kronecker product, or tensor product, of those hilbert spaces, in the specified order.

It is also possible to take kronecker powers of an Hilbert space with the exponent operator `**` using an integer exponent. This will be interpreted as repeating the kronecker product N times.

At times when trying to compose Hilbert spaces, you might hit a `NotImplementedError`. 
This means that the composition of those 2 spaces is not supported, probably because nobody needed it before.
Please do open an issue or a feature request on the GitHub repository if you hit this case.


## The DiscreteHilbert interface

Discrete Hilbert spaces have a much more full fledged API. 
{ref}`netket.hilbert.DiscreteHilbert` is also an abstract class from which any hilbert space with countable (or discrete) local degrees of freedom must inherit.
Examples of such spaces are spins or bosons on a lattice.

You can always probe their {attr}`~netket.hilbert.DiscreteHilbert.shape`, which returns a tuple 
with the size of the hilbert space on every site/degree of freedom.
For example, for 4 spins-$1/2$ coupled to a bosonic mode with a cutoff of 5 bosons, the shape will be
`[2,2,2,2,6]`.

```python
>>> from netket.hilbert import Spin, Fock
>>> hi = Spin(1/2, 4)*Fock(5)
>>> hi.shape
array([2, 2, 2, 2, 6])
```
The `shape` is also linked to the local Hilbert basis, which lists all possible values that a basis elements can take on this particular lattice site/subsystem.
For example, on the first four sites of the example above, the basis elements are only 2: `[-1, 1]`, while on the last site they are 6: `[0,1,2,3,4,5]`.

This information can be extracted with the {meth}`~netket.hilbert.DiscreteHilbert.states_at_index` method, as shown below:

```python
>>> hi.states_at_index(0)
[-1.0, 1.0]
>>> hi.states_at_index(1)
[-1.0, 1.0]
>>> hi.states_at_index(4)
[0, 1, 2, 3, 4, 5]
```

It should be now evident why NetKet distinguishes locally discrete/countable spaces from arbitrary (e.g: continuous) spaces: if we can index the local basis, we can perform many optimisations and write efficient kernels to compute matrix elements of operators, but also Monte-Carlo samplers will propose transitions in a very different way than in continuous spaces.

You can also obtain the total size of the hilbert space by invoking {attr}`~netket.hilbert.DiscreteHilbert.n_states`, which in general is equivalent to calling `np.prod(hi.shape)`.

```python
>>> hi.n_states
96
```

Do bear in mind that this attribute only works if the hilbert space is indexable ({attr}`~netket.hilbert.DiscreteHilbert.is_indexable`), which is True when it has a size smaller than $2^{64}$. 

NetKet also supports discrete-but-infinite hilbert spaces, such as Fock spaces with no cutoff. 
Those hilbert spaces are of course not indexable ({attr}`~netket.hilbert.DiscreteHilbert.is_indexable` will return `False`) and they are further signaled by the attribute ({attr}`~netket.hilbert.DiscreteHilbert.is_finite`, which will be set to `False`.

The only non-finite (discrete) hilbert space implemented in NetKet is the Fock space, and it can be constructed by not specifying the cutoff, as shown below:

```python
>>> Fock() # 1 mode with no cutoff
Fock(n_max=INT_MAX, N=1)
>>> Fock(None, N=3)  # 3 modes with no cutoff
Fock(n_max=INT_MAX, N=3)
>>> Fock()**3  # 3 modes with no cutoff, alternative syntax
Fock(n_max=INT_MAX, N=3)
```

Do bear in mind that before of computational limitations, _infinite_ hilbert spaces are not technically infinite, but simply have their cutoff set to `2^{63}`, the largest signed integer.

### Indexable spaces

If a space is indexable it is possible to perform several handy operations on it, especially useful when you are checking the correctness of your calculations.
In practice all those operations rely on converting elements of the basis such as `[0,1,1,0]` to an integer index labelling all basis elements.

For the following examples, we will be using the {ref}`netket.hilbert.Qubit` hilbert space, whose local basis is `[0,1]`.

```python
>>> import netket as nk
>>> hi = nk.hilbert.Qubit(3)
Qubit(N=3)
```

Converting indices to basis elements can be performed through the {meth}`~netket.hilbert.DiscreteHilbert.numbers_to_states` method. 
When converting indices to a basis-element, NetKet relies on a sort of big-endian (or Most-Significant-Bit first) N-ary-encoding: for qubits, index $0$ will correspond to $|0,0,0\rangle$, index $1$ to $|0,0,1\rangle$, index $2$ to $|0,1,0\rangle$ and so on.
For hilbert spaces with larger local dimensions, all the local states are iterated continuously.

```python
>>> hi.numbers_to_states(0)
array([0., 0., 0.])
>>> hi.numbers_to_states(1)
array([0., 0., 1.])
>>> hi.numbers_to_states(2)
array([0., 1., 0.])
>>> hi.numbers_to_states(3)
array([0., 1., 1.])
>>> hi.numbers_to_states(7)
array([1., 1., 1.])
```

It is also possible to perform the opposite transformation, and go from a basis element to an integer index using the {meth}`~netket.hilbert.DiscreteHilbert.states_to_numbers` method.

```python
>>> hi.states_to_numbers(np.array([0,0,0]))
0
>>> hi.states_to_numbers(np.array([0,0,1]))
1
>>> hi.states_to_numbers(np.array([1,0,1]))
5
```

Do notice that all those methods work with arrays too, and will convert an array of $M$ indices to a batch of states, that is, a matrix of size $M \times N$.

Lastly, it is also possible to obtain the batch of all basis states with the {meth}`~netket.hilbert.DiscreteHilbert.all_states` method. 


## Generating uniform samples

It is always possible to sample the basis set of an Hilbert space according to the uniform distribution.
This can be done using the method {meth}`~netket.hilbert.DiscreteHilbert.random_state`. 
This method behaves similarly to {ref}`jax.random` generators: the first argument must be a valid {ref}`jax.random.PRNGKey`, 
the second argument is an optional shape argument and the last one is the data type (dtype) of the output.

````python
>>> key = jax.random.PRNGKey(1)
>>> hi.random_state(key)
DeviceArray([0., 1., 0.], dtype=float32)
>>> hi.random_state(key, 3)
DeviceArray([[0., 0., 0.],
             [1., 1., 1.],
             [0., 0., 1.]], dtype=float32)

```

The distribution respects


## Using Hilbert spaces with {ref}`jax.jit`ted functions

Hilbert spaces are immutable, hashable objects. 
Their hash is computed by hashing their inner fields, determined by the internal {meth}`~netket.hilbert.DiscreteHilbert._attrs` method.
You can freely use `AbstractHilbert` objects inside of `jax.jit`ted functions as long as you specify that they are `static`.

All attributes and methods of Hilbert spaces can be freely used inside of a `jax.jit` block except for 
{meth}`~netket.hilbert.DiscreteHilbert.states_to_numbers` and {meth}`~netket.hilbert.DiscreteHilbert.numbers_to_states`, because
they are written using {ref}`numpy` instead of jax.

In particular, `random_state` method can be used inside of jitted blocks, as it is written in jax, as long as you pass a valid jax 
{ref}`jax.random.PRNGKey` object as the first argument.

