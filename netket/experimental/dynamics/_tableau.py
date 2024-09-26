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

from typing import Callable
from netket.utils.struct import dataclass, field
from netket.utils.types import Array
from abc import abstractmethod

from ._state import IntegratorState


@dataclass(_frozen=True)
class Tableau:
    r"""
    Class representing the general tableaus for various methods for a given the ODE :math:`dy/dt = F(t, y)`

    If :code:`self.is_adaptive`, the tableau also contains the coefficients
    which can be used to estimate the local truncation error (if necessary).
    """

    name: str = field(pytree_node=False)
    """The name of the tableau."""

    order: tuple[int, int]
    """The order of the tableau"""

    def __repr__(self) -> str:
        return self.name

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

    @property
    @abstractmethod
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.is_adaptive:
            return None
        else:
            pass

    @abstractmethod
    def step(
        self, f: Callable, t: float, dt: float, y_t: Array, state: IntegratorState
    ):
        """Perform one fixed-size step from `t` to `t + dt`."""
        pass

    @abstractmethod
    def step_with_error(
        self, f: Callable, t: float, dt: float, y_t: Array, state: IntegratorState
    ):
        """
        Perform one fixed-size step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        pass
