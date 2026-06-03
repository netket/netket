(netket_vqs_api)=
# netket.vqs

```{eval-rst}
.. currentmodule:: netket.vqs

```

This module defines the variational states, the heart of NetKet itself.


```{eval-rst}
.. inheritance-diagram:: netket.vqs
   :top-classes: netket.vqs.VariationalState
   :parts: 1

```

## Abstract Interface

```{eval-rst}
.. autosummary::
  :toctree: _generated/vqs
  :nosignatures:

  VariationalState
  VariationalMixedState
```

## Concrete Variational States

```{eval-rst}
.. autosummary::
  :toctree: _generated/vqs
  :nosignatures:

  FullSumState
  MCState
  MCMixedState
```
and the experimental Variational state for a single slater determinant state (which does not use Monte-Carlo sampling)

```{eval-rst}
.. currentmodule:: netket.experimental.vqs

.. autosummary::
  :toctree: _generated/vqs
  :nosignatures:

  DeterminantVariationalState
```

## Functions

```{eval-rst}
.. currentmodule:: netket.vqs

.. autosummary::
  :toctree: _generated/vqs
  :nosignatures:

  apply_operator
  local_estimators
  get_local_kernel
  get_local_kernel_arguments
```

### Freezing parameters

The following functions return a *new* variational state in which a subset of the
parameters has been frozen (moved from {attr}`~netket.vqs.VariationalState.parameters`
into {attr}`~netket.vqs.VariationalState.model_state`), so they are automatically
excluded from gradient computation and optimizer updates. See the
[freezing parameters example](https://github.com/netket/netket/blob/master/Examples/freeze_example.py) for a worked example.

```{eval-rst}
.. autosummary::
  :toctree: _generated/vqs
  :nosignatures:

  freeze_parameters
  unfreeze_parameters
```

