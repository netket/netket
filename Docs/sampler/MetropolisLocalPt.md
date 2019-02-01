# MetropolisLocalPt
This sampler performs parallel-tempering moves in addition to the local moves implemented in `MetropolisLocal`. The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.

## Class Constructor
Constructs a new ``MetropolisLocalPt`` sampler given a machine
and the number of replicas.

| Argument |         Type         |                                            Description                                             |
|----------|----------------------|----------------------------------------------------------------------------------------------------|
|machine   |netket.machine.Machine|A machine used for the sampling. The probability distribution being sampled from is $$\|\Psi(s)\|^2$$.|
|n_replicas|int=1                 |The number of replicas used for parallel tempering.                                                 |


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
>>> # Construct a MetropolisLocalPt Sampler
>>> sa = nk.sampler.MetropolisLocalPt(machine=ma,n_replicas=16)
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

