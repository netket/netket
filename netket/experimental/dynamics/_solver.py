# Copyright 2021 The NetKet Authors - All Rights Reserved.
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

import jax.numpy as jnp

from typing import Callable
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Array
from netket.utils.numbers import dtype as _dtype
from abc import abstractmethod


# from ._integrator import Integrator
from ._state import IntegratorState, SolverFlags


@struct.dataclass(_frozen=True)
class AbstractSolver:
    r"""
    The ODE solver, which also works as a constructor for the `Integrator` and `AbstractSolver` instances.
    """
    def __init__(self, dt, *, adaptive=False, **kwargs):
        r"""
        Args:
            dt: The initial time-step of the solver.
            adaptive: A boolean indicator whether to use an adaptive scheme.
            error_order: The error order of the solver-scheme
        """
        self.initial_dt = dt
        self.adaptive = adaptive
        self.kwargs = kwargs

    def _init_state(self, t0, y0):
        r"""
        Initializes the `Integrator` structure containing the solver and state, 
        given the necessary information.
        Args:
            f: The ODE function
            t0: The initial time of evolution
            y0: The solution at initial time `t0`
            norm: The norm used to estimate the error

        Returns:
            An `Integrator` instance intialized with the passed arguments 
        """
        t_dtype = jnp.result_type(_dtype(t0), _dtype(self.initial_dt))

        return IntegratorState(
            y=y0,
            t=jnp.array(t0, dtype=t_dtype),
            dt=jnp.array(self.initial_dt, dtype=t_dtype),
            last_norm=0.0 if self.adaptive else None,
            last_scaled_error=0.0 if self.adaptive else None,
            flags=SolverFlags(0),
        )
    
    def _update_state(self, state, *args):
        return state

    @abstractmethod
    def step(
        self, f: Callable, t: float, dt: float, y_t: Array, state: IntegratorState
    ):
        """Perform one fixed-size step from `t` to `t + dt`."""

        raise NotImplementedError("You need to define the method `step` in your `AbstractSolver`.")
    
    @abstractmethod
    def step_with_error(
        self, f: Callable, t: float, dt: float, y_t: Array, state: IntegratorState
    ):
        """
        Perform one fixed-size step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        raise NotImplementedError("You need to define the method `step_with_error` in your `AbstractSolver`.")

    def __repr__(self):
        return "{}(dt={}, adaptive={}{})".format(
            self.__class__.__name__,
            self.initial_dt,
            self.adaptive,
            f", **kwargs={self.kwargs}" if self.kwargs else "",
        )

    @property
    @abstractmethod
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        pass

    @property
    @abstractmethod
    def is_adaptive(self):
        """Boolean indication whether the integrator can be adaptive."""
        pass

    @property
    @abstractmethod
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function) of the scheme.
        """
        pass
