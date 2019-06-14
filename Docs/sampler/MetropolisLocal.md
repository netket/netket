# MetropolisLocal
This sampler acts locally only on one local degree of freedom $$s_i$$,
 and proposes a new state: $$ s_1 \dots s^\prime_i \dots s_N $$,
 where $$ s^\prime_i \neq s_i $$.

 The transition probability associated to this
 sampler can be decomposed into two steps:

 1. One of the site indices $$ i = 1\dots N $$ is chosen
 with uniform probability.
 2. Among all the possible ($$m$$) values that $$s_i$$ can take,
 one of them is chosen with uniform probability.

 For example, in the case of spin $$1/2$$ particles, $$m=2$$
 and the possible local values are $$s_i = -1,+1$$.
 In this case then `MetropolisLocal` is equivalent to flipping a random spin.

 In the case of bosons, with occupation numbers
 $$s_i = 0, 1, \dots n_{\mathrm{max}}$$, `MetropolisLocal`
 would pick a random local occupation number uniformly between $$0$$
 and $$n_{\mathrm{max}}$$.

## Class Constructor
Constructs a new ``MetropolisLocal`` sampler given a machine.

|Argument|              Type              |                                                                                     Description                                                                                     |
|--------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|machine |netket._C_netket.machine.Machine|A machine $$\Psi(s)$$ used for the sampling. The probability distribution being sampled from is $$F(\Psi(s))$$, where the function $$F(X)$$, is arbitrary, by default $$F(X)=\|X\|^2$$.|

### Examples
Sampling from a RBM machine in a 1D lattice of spin 1/2

```python
>>> import netket as nk
>>>
>>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
>>> hi=nk.hilbert.Spin(s=0.5,graph=g)
>>>
>>> # RBM Spin Machine
>>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
>>>
>>> # Construct a MetropolisLocal Sampler
>>> sa = nk.sampler.MetropolisLocal(machine=ma)
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

|  Property  |                    Type                    |                                                                                          Description                                                                                          |
|------------|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|acceptance  |         numpy.array                        | The measured acceptance rate for the sampling.         In the case of rejection-free sampling this is always equal to 1.                                                                      |
|hilbert     |         netket.hilbert                     | The Hilbert space used for the sampling.                                                                                                                                                      |
|machine     |         netket.machine                     | The machine used for the sampling.                                                                                                                                                            |
|machine_func|                           function(complex)| The function to be used for sampling.                                        by default $$\|\Psi(x)\|^2$$ is sampled,                                        however in general $$F(\Psi(v))$$  |
|visible     |                       numpy.array          | The quantum numbers being sampled,                        and distributed according to $$F(\Psi(v))$$                                                                                         |
