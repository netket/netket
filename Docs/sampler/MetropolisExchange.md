# MetropolisExchange
This sampler acts locally only on two local degree of freedom $$ s_i $$ and $$ s_j $$,
 and proposes a new state: $$ s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N $$,
 where in general $$ s^\prime_i \neq s_i $$ and $$ s^\prime_j \neq s_j $$ .
 The sites $$ i $$ and $$ j $$ are also chosen to be within a maximum graph
 distance of $$ d_{\mathrm{max}} $$.

 The transition probability associated to this sampler can
 be decomposed into two steps:

 1. A pair of indices $$ i,j = 1\dots N $$, and such
 that $$ \mathrm{dist}(i,j) \leq d_{\mathrm{max}} $$,
 is chosen with uniform probability.
 2. The sites are exchanged, i.e. $$ s^\prime_i = s_j $$ and $$ s^\prime_j = s_i $$.

 Notice that this sampling method generates random permutations of the quantum
 numbers, thus global quantities such as the sum of the local quantum n
 umbers are conserved during the sampling.
 This scheme should be used then only when sampling in a
 region where $$ \sum_i s_i = \mathrm{constant} $$ is needed,
 otherwise the sampling would be strongly not ergodic.

## Class Constructor
Constructs a new ``MetropolisExchange`` sampler given a machine and a
graph.

|Argument|         Type         |                                            Description                                             |
|--------|----------------------|----------------------------------------------------------------------------------------------------|
|machine |netket.machine.Machine|A machine used for the sampling. The probability distribution being sampled from is $$\|\Psi(s)\|^2$$.|
|graph   |netket.graph.Graph    |A graph used to define the distances among the degrees of freedom being sampled.                    |
|d_max   |int=1                 |The maximum graph distance allowed for exchanges.                                                   |


### Examples
Sampling from a RBM machine in a 1D lattice of spin 1/2, using
nearest-neighbours exchanges.

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
>>> # Construct a MetropolisExchange Sampler
>>> sa = nk.sampler.MetropolisExchange(machine=ma,graph=g,d_max=1)
>>> print(sa.hilbert.size)
100

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

