# MetropolisExchangePt
This sampler performs parallel-tempering moves in addition to
    the local exchange moves implemented in `MetropolisExchange`.
    The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.

## Class Constructor
Constructs a new ``MetropolisExchangePt`` sampler given a machine, a
graph, and a number of replicas.

| Argument |         Type         |                                            Description                                             |
|----------|----------------------|----------------------------------------------------------------------------------------------------|
|machine   |netket.machine.Machine|A machine used for the sampling. The probability distribution being sampled from is $$\|\Psi(s)\|^2$$.|
|graph     |netket.graph.Graph    |A graph used to define the distances among the degrees of freedom being sampled.                    |
|d_max     |int=1                 |The maximum graph distance allowed for exchanges.                                                   |
|n_replicas|int=1                 |The number of replicas used for parallel tempering.                                                 |

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
>>> # Construct a MetropolisExchange Sampler with parallel tempering
>>> sa = nk.sampler.MetropolisExchangePt(machine=ma,graph=g,d_max=1,n_replicas=16)

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
