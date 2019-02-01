# ExactSampler
This sampler generates i.i.d. samples from $$|\Psi(s)|^2$$. In order to perform exact sampling, $$|\Psi(s)|^2$$ is precomputed an all the possible values of the quantum numbers $$s$$. This sampler has thus an exponential cost with the number of degrees of freedom, and cannot be used for large systems, where Metropolis-based sampling are instead a viable option.

## Class Constructor
Constructs a new ``ExactSampler`` given a machine.

|Argument|         Type         |                                            Description                                             |
|--------|----------------------|----------------------------------------------------------------------------------------------------|
|machine |netket.machine.Machine|A machine used for the sampling. The probability distribution being sampled from is $$\|\Psi(s)\|^2$$.|


### Examples
Exact sampling from a RBM machine in a 1D lattice of spin 1/2

```python
>>> import netket as nk
>>> from mpi4py import MPI
>>>
>>> g=nk.graph.Hypercube(length=8,n_dim=1,pbc=True)
>>> hi=nk.hilbert.Spin(s=0.5,graph=g)
>>>
>>> # RBM Spin Machine
>>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
>>>
>>> sa = nk.sampler.ExactSampler(machine=ma)

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

