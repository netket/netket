# MetropolisHastingsPt
This sampler performs parallel-tempering
 moves in addition to the moves implemented in `MetropolisHastings`.
 The number of replicas can be $$ N_{\mathrm{rep}} $$ chosen by the user.

## Class Constructor
Constructs a new ``MetropolisHastingsPt`` sampler.

|    Argument     |                                                                                        Type                                                                                        |                                                                                                                  Description                                                                                                                   |
|-----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|machine          |netket._C_netket.machine.Machine                                                                                                                                                    |A machine $$\Psi(s)$$ used for the sampling. The probability distribution being sampled from is $$F(\Psi(s))$$, where the function $$F(X)$$, is arbitrary, by default $$F(X)=\|X\|^2$$.                                                           |
|transition_kernel|Callable[[numpy.ndarray[float64[m, n], flags.c_contiguous], numpy.ndarray[float64[m, n], flags.writeable, flags.c_contiguous], numpy.ndarray[float64[m, 1], flags.writeable]], None]|A function to generate a transition. This should take as an input the current state (in batches) and return a modified state (also in batches). This function must also return an array containing the `log_prob_corrections` $$L(s,s^\prime)$$.|
|n_replicas       |int = 16                                                                                                                                                                            |The number of replicas used for parallel tempering.                                                                                                                                                                                             |
|sweep_size       |Optional[int] = None                                                                                                                                                                |The number of exchanges that compose a single sweep. If None, sweep_size is equal to the number of degrees of freedom (n_visible).                                                                                                              |

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
