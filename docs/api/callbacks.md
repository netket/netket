(netket_callbacks_api)=
# netket.callbacks

```{eval-rst}
.. currentmodule:: netket.callbacks

```

Those callbacks can be used with the optimisation drivers.

For a detailed description of the run loop and all available callback hooks, see
{ref}`advanced_custom_callbacks`.

## Abstract base class

```{eval-rst}
.. autosummary::
   :toctree: _generated/callbacks
   :nosignatures:

   AbstractCallback
   StopRun
```

## Built-in callbacks

```{eval-rst}
.. autosummary::
   :toctree: _generated/callbacks
   :nosignatures:

   EarlyStopping
   ConvergenceStopping
   InvalidLossStopping
   Timeout
   AutoChunkSize
```
