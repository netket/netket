
(netket_optimizer_api)=
# netket.optimizer

```{eval-rst}
.. currentmodule:: netket.optimizer
```

This module provides some optimisers,
implementations of the [Quantum Geometric Tensor](QGT_and_SR) and preconditioners such
as SR.

## Optimizers

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

   Adam
   AdaGrad
   Sgd
   Momentum
   RmsProp

```

## Preconditioners

This module also provides an implemnetation of the Stochastic Reconfiguration/Natural
gradient preconditioner.

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   SR
```

## Quantum Geometric Tensor

It also provides the following implementation of the quantum geometric tensor:

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   qgt.QGTAuto
   qgt.QGTOnTheFly
   qgt.QGTJacobianPyTree
   qgt.QGTJacobianDense
```

## Dense solvers

And the following dense solvers for Stochastic Reconfiguration:

```{eval-rst}
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   solver.svd
   solver.cholesky
   solver.LU
   solver.solve
```
