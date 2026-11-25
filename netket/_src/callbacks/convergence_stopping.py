# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from flax import serialization

from netket.utils import struct

from netket._src.callbacks.base import (
    AbstractCallback,
    StopRun,
    STOPPING_CALLBACK_ORDER,
)


class ConvergenceStopping(AbstractCallback, mutable=True):
    """A simple callback to stop the optimisation when the monitored quantity gets
    below a certain threshold for at least `patience` steps.
    """

    target: float = struct.field(pytree_node=False)
    """Target value for the monitored quantity. Training will stop if the driver drops below this value."""
    monitor: str = struct.field(pytree_node=False)
    """Loss statistic to monitor. Should be one of 'mean', 'variance', 'error_of_mean'."""
    smoothing_window: int = struct.field(pytree_node=False)
    """The loss is smoothed over the last `smoothing_window` iterations to
    reduce statistical fluctuations."""
    patience: int = struct.field(pytree_node=False)
    """The loss must be consistently below this value for this number of
    iterations in order to stop the optimisation."""

    # Saved, so that a run resumed from a checkpoint keeps its history. The window
    # has a fixed length: only its last `_n_losses` entries are filled.
    _loss_window: tuple[float, ...] = struct.field(pytree_node=False, serialize=True)
    _n_losses: int = struct.field(pytree_node=False, serialize=True, default=0)
    _patience_counter: int = struct.field(pytree_node=False, serialize=True, default=0)

    def __init__(
        self,
        target: float,
        monitor: str = "mean",
        *,
        smoothing_window: int = 10,
        patience: int = 10,
    ):
        """
        Construct a callback stopping the optimisation when the monitored quantity
        gets below a certain threshold for at least `patience` steps.

        Args:
            target: the threshold value for the monitored quantity. Training will stop if the driver drops below this value.
            monitor: a string with the name of the quantity to be monitored. This
                is applied to the standard loss optimised by a driver, such as the
                Energy for the VMC driver. Should be one of
                'mean', 'variance', 'error_of_mean' (default: 'mean').
            smoothing_window: an integer number of steps over which the monitored value
                is averaged before comparing to target.
            patience: Number of steps to wait before stopping the execution after
                the tracked quantity drops below the target value (default 0, meaning
                that it stops immediately).
        """
        self.target = target
        self.monitor = monitor
        self.smoothing_window = smoothing_window
        self.patience = patience

        self._loss_window = (0.0,) * smoothing_window
        self._n_losses = 0
        self._patience_counter = 0

    def __process_deserialization_state__(self, state):
        # If the file has no window (older versions) or one of a different size,
        # start the window over.
        if len(state.get("_loss_window", ())) != self.smoothing_window:
            state = {
                k: v for k, v in state.items() if k not in ("_loss_window", "_n_losses")
            }
        names = ("_loss_window", "_n_losses", "_patience_counter")
        defaults = {n: serialization.to_state_dict(getattr(self, n)) for n in names}
        return {**defaults, **state}

    @property
    def callback_order(self) -> int:
        # Run last, so raising StopRun never skips a later callback's collective.
        return STOPPING_CALLBACK_ORDER

    def on_step_end(self, step, log_data, driver):
        loss = np.real(getattr(log_data[driver._loss_name], self.monitor))

        self._loss_window = (*self._loss_window[1:], float(np.mean(loss)))
        self._n_losses = min(self._n_losses + 1, self.smoothing_window)
        loss_smooth = np.mean(self._loss_window[-self._n_losses :])

        if loss_smooth <= self.target:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        if self._patience_counter > self.patience:
            raise StopRun(
                f"ConvergenceStopping: smoothed loss {loss_smooth:.6g} has been below "
                f"target {self.target} for {self._patience_counter} steps."
            )
