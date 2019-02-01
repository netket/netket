# MetropolisHamiltonian
Sampling based on the off-diagonal elements of a Hamiltonian (or a generic Operator). In this case, the transition matrix is taken to be: $$ T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|), $$ where $$ \theta(x) $$ is the Heaviside step function, and $$ \mathcal{N}(\mathbf{s}) $$ is a state-dependent normalization. The effect of this transition probability is then to connect (with uniform probability) a given state $$ \mathbf{s} $$ to all those states $$ \mathbf{s}^\prime $$ for which the Hamiltonian has finite matrix elements. Notice that this sampler preserves by construction all the symmetries of the Hamiltonian. This is in generally not true for the local samplers instead.

## Class Constructor
Constructs a new ``MetropolisHamiltonian`` sampler given a machine
and a Hamiltonian operator (or in general an arbitrary Operator).

| Argument  |         Type         |                                            Description                                             |
|-----------|----------------------|----------------------------------------------------------------------------------------------------|
|machine    |netket.machine.Machine|A machine used for the sampling. The probability distribution being sampled from is $$\|\Psi(s)\|^2$$.|
|hamiltonian|netket.Operator       |The operator used to perform off-diagonal transition.                                               |


### Examples
Sampling from a RBM machine in a 1D lattice of spin 1/2

```python
>>> import netket as nk
>>> from mpi4py import MPI
>>>
>>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
>>> hi=nk.hilbert.Spin(s=0.5,graph=g)
>>>
>>> # RBM Spin Machine
>>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
>>>
>>> # Transverse-field Ising Hamiltonian
>>> ha = nk.operator.Ising(hilbert=hi, h=1.0)
>>>
>>> # Construct a MetropolisHamiltonian Sampler
>>> sa = nk.sampler.MetropolisHamiltonian(machine=ma,hamiltonian=ha)

```



## Class Methods 
### reset
Resets the state of the sampler, including the acceptance rate statistics
and optionally initializing at random the visible units being sampled.

| Argument  |   Type   |                  Description                  |
|-----------|----------|-----------------------------------------------|
|init_random|bool=False|If ``True`` the quantum numbers (visible units)|


### seed
Seeds the random number generator used by the ``Sampler``.

|Argument |Type|                 Description                 |
|---------|----|---------------------------------------------|
|base_seed|int |The base seed for the random number generator|


### sweep
Performs a sampling sweep. Typically a single sweep
consists of an extensive number of local moves.



## Properties

| Property |               Type               |                                                        Description                                                        |
|----------|----------------------------------|---------------------------------------------------------------------------------------------------------------------------|
|acceptance|         numpy.array              | The measured acceptance rate for the sampling.         In the case of rejection-free sampling this is always equal to 1.  |
|hilbert   |         netket.hilbert           | The Hilbert space used for the sampling.                                                                                  |
|machine   |         netket.machine           | The machine used for the sampling.                                                                                        |
|visible   |                       numpy.array| The quantum numbers being sampled,                        and distributed according to $$\|\Psi(v)\|^2$$                    |

