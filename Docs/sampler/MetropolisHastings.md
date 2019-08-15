# MetropolisHastings
``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
 a local transition kernel to perform moves in the Markov Chain.
 The transition kernel is used to generate
 a proposed state $$ s^\prime $$, starting from the current state $$ s $$.
 The move is accepted with probability

 $$
 A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),
 $$

 where the probability being sampled is $$ F(\Psi(s)) $$ (by default $$ F(x)=|x|^2 $$)
 and $L(s,s^\prime)$ is a correcting factor computed by the transition kernel.

## Class Constructor
Constructs a new ``MetropolisHastings`` sampler given a machine and
a transition kernel.

|    Argument     |                                                                                        Type                                                                                        |                                                                                                                  Description                                                                                                                   |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|machine          |netket._C_netket.machine.Machine                                                                                                                                                    |A machine $$\Psi(s)$$ used for the sampling. The probability distribution being sampled from is $$F(\Psi(s))$$, where the function $$F(X)$$, is arbitrary, by default $$F(X)=\|X\|^2$$.                                                           |
|transition_kernel|Callable[[numpy.ndarray[float64[m, n], flags.c_contiguous], numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous], numpy.ndarray[float64[m, 1], flags.writeable]], None]|A function to generate a transition. This should take as an input the current state (in batches) and return a modified state (also in batches). This function must also return an array containing the `log_prob_corrections` $$L(s,s^\prime)$$.|
|batch_size       |int = 16                                                                                                                                                                            |The number of Markov Chain to be run in parallel on a single process.                                                                                                                                                                           |
|sweep_size       |Optional[int] = None                                                                                                                                                                |The number of exchanges that compose a single sweep. If None, sweep_size is equal to the number of degrees of freedom (n_visible).                                                                                                              |

### Examples
Sampling from a RBM machine in a 1D lattice of spin 1/2, using
nearest-neighbours exchanges with a custom kernel.

```python
import netket as nk
import numpy as np

# 1D Lattice
g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

# Hilbert space of spins on the graph
# with total Sz equal to 0
hi = nk.hilbert.Spin(s=0.5, graph=g, total_sz=0)

# Heisenberg hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi)

# Symmetric RBM Spin Machine
ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
ma.init_random_parameters(seed=1234, sigma=0.01)

# Defining a custom kernel for MetropolisHastings
# Notice that this sampler exchanges two random sites
# thus preserving the total magnetization
# Also notice that it is not recommended to define custom kernels in python
# For speed reasons it is better to define exchange kernels using CustomSampler
def exchange_kernel(v, vnew, logprobcorr):

    vnew[:, :] = v[:, :]
    logprobcorr[:] = 0.0

    rands = np.random.randint(v.shape[1], size=(v.shape[0], 2))

    for i in range(v.shape[0]):
        iss = rands[i, 0]
        jss = rands[i, 1]

        vnew[i, iss], vnew[i, jss] = vnew[i, jss], vnew[i, iss]


sa = nk.sampler.MetropolisHastings(ma, exchange_kernel, batch_size=16, sweep_size=20)


```



## Class Methods 
### reset
Resets the state of the sampler, including the acceptance rate statistics
and optionally initializing at random the visible units being sampled.

| Argument  |    Type    |                  Description                  |
|-----------|------------|-----------------------------------------------|
|init_random|bool = False|If ``True`` the quantum numbers (visible units)|

### seed
Seeds the random number generator used by the ``Sampler``.

|Argument |Type|                 Description                 |
|---------|----|---------------------------------------------|
|base_seed|int |The base seed for the random number generator|

### sweep
Performs a sampling sweep. Typically a single sweep
consists of an extensive number of local moves.



## Properties

|  Property  |         Type          |                                                                                     Description                                                                                     |
|------------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|batch_size  |         int           | Number of samples in a batch.                                                                                                                                                       |
|machine     |         netket.machine| The machine used for the sampling.                                                                                                                                                  |
|machine_func|function(complex)      | The function to be used for sampling.                                    by default $$\|\Psi(x)\|^2$$ is sampled,                                    however in general $$F(\Psi(v))$$|
|visible     |                       |A matrix of current visible configurations. Every row                 corresponds to a visible configuration                                                                         |
