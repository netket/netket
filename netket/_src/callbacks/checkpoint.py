from typing import Any

from netket.utils import struct

from advanced_drivers._src.callbacks.base import AbstractCallback

from netket_checkpoint._src.driver1.abstract_variational_driver import (
    _collect_extra_checkpoint_metadata,
    _restore_checkpoint,
    _save_checkpoint,
)


class CheckpointCallback(AbstractCallback, mutable=True):
    """
    Checkpointing callback for advanced variational drivers.
    """

    _checkpointer: Any = struct.field(pytree_node=False)
    _chkptr: Any = struct.field(pytree_node=False)
    _callbacks: Any = struct.field(pytree_node=False)

    def __init__(self, checkpointer):
        """
        Initialize the checkpointing callback.

        .. code-block:: python

            import netket_checkpoint as nkc
            import advanced_drivers as advd

            options = nkc.checkpoint.CheckpointManagerOptions(save_interval_steps=5, keep_period=20)
            ckpt = nkc.checkpoint.CheckpointManager(directory="/tmp/ckp4", options=options)
            ckpt_cb = advd.callbacks.CheckpointCallback(ckpt)

        Args:
            checkpointer: The checkpointer to use. Remember that a checkpointer can only
                be used once for a single run. This should be a chekckpointer from
                :class:`netket_checkpoint.checkpoint.CheckpointManager`.
        """
        self._checkpointer = checkpointer
        self._chkptr = None
        self._callbacks = None

    def on_run_start(self, step, driver, callbacks):
        n_iter = driver._step_end - driver.step_count
        extra_metadata = _collect_extra_checkpoint_metadata(driver, n_iter)

        self._chkptr = self._checkpointer.orbax_checkpointer(extra_metadata)

        # try
        if self._chkptr.latest_step() is not None:
            _, callbacks, starting_iter = _restore_checkpoint(
                driver,
                self._chkptr,
                self._chkptr.latest_step(),
                callbacks=callbacks,
                extra_metadata=extra_metadata,
            )
        self._callbacks = callbacks

    def on_reset_step_end(self, step, log_data, driver):
        _save_checkpoint(driver, self._chkptr, (), self._callbacks)

    def on_run_end(self, step, driver):
        _save_checkpoint(driver, self._chkptr, (), self._callbacks)
        self._chkptr.wait_until_finished()
        self._chkptr.close()

        self._callbacks = None

    def on_run_error(self, step, error, driver):
        self._callbacks = None
        if self._chkptr is not None:
            self._chkptr.close()
