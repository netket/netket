(api)=

# Public API

```{eval-rst}
.. currentmodule:: netket

```

(graph-api)=

## Graph

```{eval-rst}
.. autosummary::
   :toctree: _generated/graph
   :template: class
   :nosignatures:

   netket.graph.AbstractGraph
   netket.graph.Graph
   netket.graph.Lattice
   netket.graph.Edgeless
   netket.graph.Hypercube
   netket.graph.lattice.LatticeSite
   netket.graph.Chain
   netket.graph.Grid
```

(hilbert-api)=

## Hilbert

```{eval-rst}
.. autosummary::
   :toctree: _generated/hilbert
   :template: class
   :nosignatures:

   netket.hilbert.AbstractHilbert
   netket.hilbert.ContinuousHilbert
   netket.hilbert.DiscreteHilbert
   netket.hilbert.HomogeneousHilbert
   netket.hilbert.CustomHilbert
   netket.hilbert.TensorHilbert
   netket.hilbert.DoubledHilbert
   netket.hilbert.Spin
   netket.hilbert.Qubit
   netket.hilbert.Fock
   netket.hilbert.Particle
```

(operators-api)=

## Operators

```{eval-rst}
.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   netket.operator.AbstractOperator
   netket.operator.DiscreteOperator
   netket.operator.BoseHubbard
   netket.operator.GraphOperator
   netket.operator.LocalOperator
   netket.operator.Ising
   netket.operator.Heisenberg
   netket.operator.PauliStrings
   netket.operator.LocalLiouvillian

```

### Pre-defined operators

```{eval-rst}
.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   netket.operator.boson.create
   netket.operator.boson.destroy
   netket.operator.boson.number
   netket.operator.boson.proj
   netket.operator.spin.sigmax
   netket.operator.spin.sigmay
   netket.operator.spin.sigmaz
   netket.operator.spin.sigmap
   netket.operator.spin.sigmam

```

### Continuous space operators

```{eval-rst}
.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   netket.operator.ContinuousOperator
   netket.operator.KineticEnergy
   netket.operator.PotentialEnergy
   netket.operator.SumOperator
```

(exact-api)=

## Exact solvers

```{eval-rst}
.. autosummary::
   :toctree: _generated/exact
   :nosignatures:

   netket.exact.full_ed
   netket.exact.lanczos_ed
   netket.exact.steady_state
```

(sampler-api)=

## Sampler

### Generic API

Those functions can be used to interact with samplers

```{eval-rst}
.. autosummary::
   :toctree: _generated/samplers

   netket.sampler.sampler_state
   netket.sampler.reset
   netket.sampler.sample_next
   netket.sampler.sample
   netket.sampler.samples
```

### List of Samplers

This is a list of all available samplers.
Please note that samplers with `Numpy` in their name are implemented in
Numpy and not in pure jax, and they will convert from numpy\<->jax at every
sampling step the state.
If you are using GPUs, this conversion can be very costly. On CPUs, while the
conversion is cheap, the dispatch cost of jax is considerate for small systems.

In general those samplers, while they have the same asyntotic cost of Jax samplers,
have a much higher overhead for small to moderate (for GPUs) system sizes.

This is because it is not possible to implement all transition rules in Jax.

```{eval-rst}
.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   netket.sampler.Sampler
   netket.sampler.ExactSampler
   netket.sampler.MetropolisSampler
   netket.sampler.MetropolisSamplerNumpy
   netket.experimental.sampler.MetropolisPtSampler

   netket.sampler.MetropolisLocal
   netket.sampler.MetropolisExchange
   netket.sampler.MetropolisHamiltonian
   netket.experimental.sampler.MetropolisLocalPt
   netket.experimental.sampler.MetropolisExchangePt

   netket.sampler.ARDirectSampler
```

### Transition Rules

Those are the transition rules that can be used with the Metropolis
Sampler. Rules with `Numpy` in their name can only be used with
{class}`netket.sampler.MetropolisSamplerNumpy`.

```{eval-rst}
.. autosummary::
  :toctree: _generated/samplers

  netket.sampler.MetropolisRule
  netket.sampler.rules.LocalRule
  netket.sampler.rules.ExchangeRule
  netket.sampler.rules.HamiltonianRule
  netket.sampler.rules.HamiltonianRuleNumpy
  netket.sampler.rules.CustomRuleNumpy

```

### Internal State

Those structure hold the state of the sampler.

```{eval-rst}
.. autosummary::
  :toctree: _generated/samplers

  netket.sampler.SamplerState
  netket.sampler.MetropolisSamplerState
```

(models)=

## Pre-built models

This sub-module contains several pre-built models to be used as
neural quantum states.

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_model
   :nosignatures:

   netket.models.RBM
   netket.models.RBMModPhase
   netket.models.RBMMultiVal
   netket.models.RBMSymm
   netket.models.Jastrow
   netket.models.MPSPeriodic
   netket.models.NDM
   netket.models.GCNN
   netket.models.AbstractARNN
   netket.models.ARNNDense
   netket.models.ARNNConv1D
   netket.models.ARNNConv2D
   netket.models.FastARNNConv1D
   netket.models.FastARNNConv2D

```

## Model tools

This sub-module wraps and re-exports [flax.nn](https://flax.readthedocs.io/en/latest/flax.linen.html).
Read more about the design goal of this module in their [README](https://github.com/google/flax/blob/master/flax/linen/README.md)

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

   netket.nn.Module
```

### Linear Modules

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

   netket.nn.Dense
   netket.nn.DenseGeneral
   netket.nn.DenseSymm
   netket.nn.DenseEquivariant
   netket.nn.Conv
   netket.nn.Embed

   netket.nn.MaskedDense1D
   netket.nn.MaskedConv1D
   netket.nn.MaskedConv2D

```

### Activation functions

```{eval-rst}
.. currentmodule:: netket.nn
```

```{eval-rst}
.. autosummary::
  :toctree: _generated/nn

    celu
    elu
    gelu
    glu
    log_sigmoid
    log_softmax
    relu
    sigmoid
    soft_sign
    softmax
    softplus
    swish
    log_cosh
    reim_relu
    reim_selu

```

(variational-api)=

## Variational State Interface

```{eval-rst}
.. currentmodule:: netket
```

```{eval-rst}
.. autosummary::
  :toctree: _generated/variational
  :nosignatures:

  netket.vqs.VariationalState
  netket.vqs.ExactState
  netket.vqs.MCState
  netket.vqs.MCMixedState
  netket.vqs.get_local_kernel
  netket.vqs.get_local_kernel_arguments

```

(optimizer-api)=

## Optimizer Module

```{eval-rst}
.. currentmodule:: netket
```

This module provides some optimisers,
implementations of the \{ref}\`Quantum Geometric Tensor \<QGT_and_SR>\` and preconditioners such
as SR.

### Optimizers

Optimizers in NetKet are simple wrappers of [optax](https://github.com/deepmind/optax)
optimizers. If you want to write a custom optimizer or use more advanced ones, we suggest
you have a look at optax documentation.

Check it out for up-to-date informations on available optimisers.

:::{warning}
Even if optimisers in `netket.optimizer` are optax optimisers, they have slightly different
names (they are capitalised) and the argument names have been rearranged and renamed.
This was chosen in order not to break our API from previous versions

In general, we advise you to directly use optax, as it is much more powerful, provides more
optimisers, and it's extremely easy to use step-dependent schedulers.
:::

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   netket.optimizer.Adam
   netket.optimizer.AdaGrad
   netket.optimizer.Sgd
   netket.optimizer.Momentum
   netket.optimizer.RmsProp

```

### Preconditioners

This module also provides an implemnetation of the Stochastic Reconfiguration/Natural
gradient preconditioner.

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   netket.optimizer.SR
```

### Quantum Geometric Tensor

It also provides the following implementation of the quantum geometric tensor:

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   netket.optimizer.qgt.QGTAuto
   netket.optimizer.qgt.QGTOnTheFly
   netket.optimizer.qgt.QGTJacobianPyTree
   netket.optimizer.qgt.QGTJacobianDense
```

### Dense solvers

And the following dense solvers for Stochastic Reconfiguration:

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   netket.optimizer.solver.svd
   netket.optimizer.solver.cholesky
   netket.optimizer.solver.LU
   netket.optimizer.solver.solve
```

(drivers-api)=

## Optimization drivers

Those are the optimization drivers already implmented in Netket:

```{eval-rst}
.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   netket.driver.AbstractVariationalDriver
   netket.driver.VMC
   netket.driver.SteadyState

```

(logging-api)=

## Logging output

Those are the loggers that can be used with the optimization drivers.

```{eval-rst}
.. autosummary::
   :toctree: _generated/driver
   :nosignatures:

   netket.logging.RuntimeLog
   netket.logging.JsonLog
   netket.logging.StateLog
   netket.logging.TensorBoardLog

```

(utils-api)=

## Utils

Utility functions and classes.

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   netket.utils.HashableArray
```

(callbacks-api)=

## Callbacks

Those callbacks can be used with the optimisation drivers.

```{eval-rst}
.. autosummary::
   :toctree: _generated/callbacks
   :nosignatures:

   netket.callbacks.EarlyStopping
   netket.callbacks.Timeout
```
