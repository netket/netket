# CustomSampler
Custom Sampler, where transition operators are specified by the user. For the moment, this functionality is limited to transition operators which are sums of $$k$$-local operators: $$ \mathcal{M}= \sum_i M_i $$ where the move operators $$ M_i $$ act on an (arbitrary) subset of sites. The operators $$ M_i $$ are specified giving their matrix elements, and a list of sites on which they act. Each operator $$ M_i $$ must be real, symmetric, positive definite and stochastic (i.e. sum of each column and line is 1). The transition probability associated to a custom sampler can be decomposed into two steps: 1. One of the move operators $$ M_i $$ is chosen with a weight given by the user (or uniform probability by default). If the weights are provided, they do not need to sum to unity. 2. Starting from state $$ |n \rangle $$, the probability to transition to state $$ |m\rangle $$ is given by $$ \langle n| M_i | m \rangle $$.

## Class Constructor
Constructs a new ``CustomSampler`` given a machine and a list of local
stochastic move (transition) operators.

|   Argument   |            Type             |                                            Description                                             |
|--------------|-----------------------------|----------------------------------------------------------------------------------------------------|
|machine       |netket.machine.Machine       |A machine used for the sampling. The probability distribution being sampled from is $$\|\Psi(s)\|^2$$.|
|move_operators|netket.operator.LocalOperator|The stochastic `LocalOperator` $$\mathcal{M}= \sum_i M_i$$ used for transitions.                    |
|move_weights  |List[float]=[]               |For each $$ i $$, the probability to pick one of the move operators (must sum to one).              |


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
>>> # Construct a Custom Sampler
>>> # Using random local spin flips (Pauli X operator)
>>> X = [[0, 1],[1, 0]]
>>> move_op = nk.operator.LocalOperator(hilbert=hi,operators=[X] * g.size,acting_on=[[i] for i in range(g.size)])
>>> sa = nk.sampler.CustomSampler(machine=ma, move_operators=move_op)

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

