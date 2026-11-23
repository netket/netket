(netket_checkpoint_api)=
# netket.checkpoint

```{eval-rst}
.. currentmodule:: netket.checkpoint

```

The state of a driver that a checkpointer saves, and loading one into a driver, e.g. to
start a new run from part of an old checkpoint. To save and resume a run, pass a
checkpointer to {meth}`~netket.driver.AbstractDriver.run`: see
{ref}`experimental-checkpoint-api`.

```{eval-rst}
.. autosummary::
   :toctree: _generated/checkpoint
   :nosignatures:

   checkpoint_state
   load_checkpoint_state
```
