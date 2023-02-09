# The Drivers API

In this section we will briefly describe the capabilities of the drivers API.
This page assumes that you have already read and are familiar with the [`VariationalState` interface](varstate).

In Netket there are three drivers, even though you can define your own; those are:

1. {class}`~netket.driver.VMC`, to find the ground state of an Hamiltonian
2. {class}`~netket.driver.SteadyState`, to find the steady-state of a liouvillian

A driver, will run your optimisation loop, computing the loss function and the gradient,
using the gradient to update the parameters, and logging to yours sinks any data that you
may wish.

## Constructing a driver

There are two objects both drivers above need in order to be constructed:

- The {class}`~netket.operator.AbstractOperator` defining the problem we wish to solve, such as the Hamiltonian for which we want to find the ground state or the Lindbladian for which we want to find the steady state.
- The [Optimizer](netket_optimizer_api) to use in order to update the weights among iterations.

Those are respectively the first and second argument of the constructor.

Of course, you then need to communicate the optimization driver what is the state you wish to optimize.
For that reason, assuming you have constructed a variational state {code}`vstate`, you should pass it as
a keyword argument {code}`variational_state=vstate` to the constructor.
The resulting code looks a bit like this:

```python
hamiltonian = nk.operator.Ising(hilbert, ...)

optimizer = nk.optimizer.SGD(learning_rate=0.1)

vstate = nk.vqs.MCState(sampler, model, n_samples=1000)

gs = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate)
```

There also exist an alternative syntax, where instead of passing the variational state you pass the arguments needed to construct the variational state to the driver itself.

```python
hamiltonian = nk.operator.Ising(hilbert, ...)

optimizer = nk.optimizer.SGD(learning_rate=0.1)

gs = nk.driver.VMC(hamiltonian, optimizer, sampler, model, n_samples=1000)
```

And you can then access the variational state constructed like that through the attribute `gs.state`.
The latter is there to guarantee better compatibility with legacy codebases, therefore we suggest to
use the more first API, where the variational state is built explicitly.

## Running the optimisation

The simplest way to use optimization drivers to perform the optimisation is to use their {attr}`~netket.driver.AbstractVariationalDriver.run` method.

This method will run the optimisation for the desired number of steps, while logging data to the
desired output.

The most important arguments are the following:

```python
run(n_iter, out=None, obs=None, callback=None, step_size=None)
```

- The first argument must be the number of iterations to perform.

- The {code}`out` argument is used to pass the output loggers to the optimiser. It can take several values:

  - {code}`None`: No output will be logged (default).

  - {code}`string`: A default Json logger will be created, serializing data to the specified filename.

  - {code}`Logger`: a logger, or iterable of loggers, respecting the standard logging interface. The available loggers are listed [here](netket_logging_api).

  - The {code}`callbacks` can be used to pass callbacks to the optimisation driver. Callbacks must be callables with the signature
    .. code:: python

    > (step:int, logdata:dict, driver:AbstractVariationalDriver) -> bool

  The first argument is the step number, the second argument is the dictionary holding data that will be logged, and it can be modified by the callback, and the third is the driver itself, which can be used to access the current state or any other quantity.
  The output of the callback must be a boolean, which signals whether to continue the optimisation or not. When any one of the callbacks return {code}`False`, the optimisation will be stopped.
  NetKet comes with a few built-in callbacks, listed [in the API docs](netket_callbacks_api)`, but you can also implement your own.

- {code}`step_size`: Data will be logged and callbacks will be called every {code}`step_size` optimisation steps. Useful if your callbacks have a high computational cost. If unspecified, logs at every step.
