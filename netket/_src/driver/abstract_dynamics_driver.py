# Copyright 2021 The NetKet Authors - All rights reserved.
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

from typing import Union
from collections.abc import Iterable

import optax

from netket.logging import AbstractLog
from netket.utils import struct, KahanSum
from netket.utils.iterators import to_iterable
from netket.vqs import VariationalState

from netket._src.callbacks.base import AbstractCallback
from netket._src.callbacks.legacy_wrappers import LegacyCallbackT
from netket._src.callbacks.stop_at_time import _StopAtTime
from netket._src.callbacks.progressbar import (
    TimeProgressBarCallback,
)
from netket._src.driver.abstract_variational_driver import AbstractDriver
from netket._src.callbacks.callback_list import CallbackList


class AbstractDynamicsDriver(AbstractDriver):
    """
    Abstract base class for time-evolution (dynamics) drivers.

    Unlike optimization drivers there is no optimizer: ``update_parameters``
    applies the parameter delta directly and advances the simulation clock.

    .. note::

        How to implement a new dynamics driver

        Subclass this class and implement:

        - :meth:`compute_loss_and_update`: compute one integration step.
          Return ``(loss_stats, Δθ)`` where ``Δθ`` is the full parameter
          delta for this time step (i.e. it already includes the factor of ``dt``).

        - :attr:`dt` property: the current time step size (may vary for adaptive
          integrators).

        Optionally override :attr:`t` and the ``t`` setter if your integrator
        owns the time state rather than using the built-in ``_t`` field
        (e.g. when wrapping an external ODE integrator).

    The :meth:`run` method accepts either a total evolution time ``T`` (float)
    or a step count (int), and uses the :class:`~netket.callbacks.StopRun`
    mechanism internally so the full callback system works unchanged.
    """

    _t: KahanSum = struct.field(pytree_node=True, serialize=True, default=None)

    def __init__(
        self,
        variational_state: VariationalState,
        *,
        t0: float = 0.0,
        minimized_quantity_name: str = "loss",
    ):
        """
        Initializes a dynamics driver.

        Args:
            variational_state: The variational state.
            t0: Initial simulation time (default 0.0).
            minimized_quantity_name: Name of the monitored quantity in logged data.
        """
        super().__init__(
            variational_state, minimized_quantity_name=minimized_quantity_name
        )
        self._t = KahanSum(t0)

    @property
    def dt(self) -> float:
        """
        Current time step size.

        For fixed-step drivers this is a constant. For adaptive drivers this
        is the accepted step size of the last completed step.

        Subclasses must override this property.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement the dt property."
        )

    @property
    def t(self) -> float:
        """Current simulation time."""
        return self._t.value

    @t.setter
    def t(self, value: float):
        self._t = KahanSum(float(value))

    def update_parameters(self, dp):
        """
        Apply the parameter delta and advance the simulation clock.

        No optimizer is involved. ``dp`` must already be the full ``Δθ``
        (i.e. it includes the factor of ``dt`` from the integration scheme).
        """
        self.state.parameters = optax.apply_updates(self.state.parameters, dp)
        self._t = self._t + self.dt

    def reset_step(self, hard: bool = False):
        """
        Reset the sampler at the beginning of a step attempt.

        On the first attempt (``_step_attempt == 0``), resets normally.
        On subsequent attempts after a step rejection (e.g. from adaptive
        integrators), skips the reset — existing samples remain valid for
        the revised candidate ``dt``.
        """
        if hard or self._step_attempt == 0:
            super().reset_step(hard=hard)

    def _log_additional_data(self, log_dict: dict):
        super()._log_additional_data(log_dict)
        log_dict["t"] = self.t

    def _default_callbacks(self, callbacks, *, n_iter, show_progress, **kwargs):
        # Suppress the step-count progress bar from the base class and replace
        # it with a time-aware one when T_final is known.
        callback_list = list(
            super()
            ._default_callbacks(callbacks, n_iter=n_iter, show_progress=False, **kwargs)
            .callbacks
        )
        callbacks = CallbackList(callback_list)
        return callbacks

    def run(
        self,
        T=None,
        out: Iterable[AbstractLog] | None = (),
        obs=None,
        *,
        n_iter: int | None = None,
        show_progress: bool = True,
        callback: Union[LegacyCallbackT, AbstractCallback, None] = None,
        timeit: bool = False,
        step_size: float = 0.0,
    ):
        """
        Run the time evolution.

        Args:
            T: Total evolution time as a float (e.g. ``run(1.0)``), or a
                fixed number of steps as an int (e.g. ``run(100)``).
                When a float is given the loop runs until ``driver.t >= T_or_n``,
                using :class:`~netket.callbacks.StopRun` internally so all
                callbacks fire normally.
            out: Logger or iterable of loggers for output.
            obs: Observables to compute at each logging step.
            max_steps: Safety cap on iterations when ``T_or_n`` is a float,
                to prevent infinite loops if ``dt`` is zero.
            show_progress: Show a progress bar (default True).
            callback: User callbacks.
            timeit: If True, print timing information after the run.
        """
        if n_iter is None and T is None:
            raise ValueError(
                "Must specify either a total time T or a step count n_iter."
            )
        elif n_iter is not None and T is not None:
            raise ValueError(
                "Cannot specify both a total time T and a step count n_iter."
            )

        if T is not None:
            stop_cb = _StopAtTime(T)
            callback = [stop_cb, *to_iterable(callback, none_is_empty=True)]
            n_iter = 100_000_000_000
            if show_progress:
                callback.append(TimeProgressBarCallback(T))
                show_progress = False

        return super().run(
            n_iter,
            out=out,
            obs=obs,
            show_progress=False,
            callback=callback,
            timeit=timeit,
            step_size=step_size,
        )
