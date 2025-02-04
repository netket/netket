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

from collections.abc import Callable

import jax
import jax.numpy as jnp

from netket.utils.types import Array
from netket.utils.struct import field
from .._utils import expand_dim
from ._tableau import TableauRKExplicit
from .._solver import (
    AbstractSolver,
    AbstractSolverState,
    append_docstring,
    args_adaptive_docstring,
    args_fixed_dt_docstring,
)


class RKExplicitSolver(AbstractSolver):
    r"""
    Class representing the Butcher tableau of an explicit Runge-Kutta method [1,2],
    which, given the ODE :math:`dy/dt = F(t, y)`, updates the solution as

    .. math::
        y_{t+dt} = y_t + \sum_l b_l k_l

    with the intermediate slopes

    .. math::
        k_l = F(t + c_l dt, y_t + \sum_{m < l} a_{lm} k_m).

    If :code:`self.is_adaptive`, the tableau also contains the coefficients :math:`b'_l`
    which can be used to estimate the local truncation error by the formula

    .. math::
        y_{\mathrm{err}} = \sum_l (b_l - b'_l) k_l.

    [1] https://en.wikipedia.org/w/index.php?title=Runge%E2%80%93Kutta_methods&oldid=1055669759
    [2] J. Stoer and R. Bulirsch, Introduction to Numerical Analysis, Springer NY (2002).
    """

    tableau: TableauRKExplicit = field(pytree_node=False)
    """The Butcher tableau containing all coefficients for solving the ODE."""

    def __init__(self, dt, tableau, adaptive=False, **kwargs):
        self.tableau = tableau
        if adaptive and not tableau.is_adaptive:
            raise AttributeError(f"Tableau of type {tableau} cannot be adaptve.")
        super().__init__(dt=dt, adaptive=adaptive, **kwargs)

    def __repr__(self) -> str:
        return "{}(tableau={}, dt={}, adaptive={}, integrator_parameters={})".format(
            "RKExplicitSolver",
            self.tableau,
            self.dt,
            self.adaptive,
            self.integrator_params,
        )

    @property
    def is_explicit(self):
        """Boolean indication whether the integrator is explicit."""
        return jnp.allclose(
            self.tableau.a, jnp.tril(self.tableau.a)
        )  # check if lower triangular

    @property
    def is_adaptive(self):
        """Boolean indication whether the integrator can be adaptive."""
        return self.tableau.is_adaptive

    @property
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function)
        of the scheme.
        """
        return len(self.tableau.c)

    @property
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.is_adaptive:
            return None
        else:
            return self.tableau.order[1]

    def _compute_slopes(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Computes the intermediate slopes k_l."""
        times = t + self.tableau.c * dt

        # TODO: Use FSAL

        k = expand_dim(y_t, self.stages)
        for l in range(self.stages):
            dy_l = jax.tree_util.tree_map(
                lambda k: jnp.tensordot(
                    jnp.asarray(self.tableau.a[l], dtype=k.dtype), k, axes=1
                ),
                k,
            )
            y_l = jax.tree_util.tree_map(
                lambda y_t, dy_l: jnp.asarray(y_t + dt * dy_l, dtype=dy_l.dtype),
                y_t,
                dy_l,
            )
            k_l = f(times[l], y_l, stage=l)
            k = jax.tree_util.tree_map(lambda k, k_l: k.at[l].set(k_l), k, k_l)

        return k

    def step(
        self, f: Callable, dt: float, t: float, y_t: Array, state: AbstractSolverState
    ):
        """Perform one fixed-size RK step from `t` to `t + dt`."""
        k = self._compute_slopes(f, t, dt, y_t)

        b = self.tableau.b[0] if self.tableau.b.ndim == 2 else self.tableau.b
        y_tp1 = jax.tree_util.tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(b, dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )

        return y_tp1, state

    def step_with_error(
        self, f: Callable, dt: float, t: float, y_t: Array, state: AbstractSolverState
    ):
        """
        Perform one fixed-size RK step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        if not self.is_adaptive:
            raise RuntimeError(f"{self} is not adaptive")

        k = self._compute_slopes(f, t, dt, y_t)

        y_tp1 = jax.tree_util.tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(self.tableau.b[0], dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )
        db = self.tableau.b[0] - self.tableau.b[1]
        y_err = jax.tree_util.tree_map(
            lambda k: jnp.asarray(dt, dtype=k.dtype)
            * jnp.tensordot(jnp.asarray(db, dtype=k.dtype), k, axes=1),
            k,
        )

        return y_tp1, y_err, state


@append_docstring(args_fixed_dt_docstring)
def Euler(dt):
    r"""
    The canonical first-order forward Euler method. Fixed timestep only.

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_feuler)


@append_docstring(args_fixed_dt_docstring)
def Midpoint(dt):
    r"""
    The second order midpoint method. Fixed timestep only.

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_midpoint)


@append_docstring(args_fixed_dt_docstring)
def Heun(dt):
    r"""
    The second order Heun's method. Fixed timestep only.

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_heun)


@append_docstring(args_fixed_dt_docstring)
def RK4(dt):
    r"""
    The canonical Runge-Kutta Order 4 method. Fixed timestep only.

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_rk4)


@append_docstring(args_adaptive_docstring)
def RK12(dt, **kwargs):
    r"""
    The second order Heun's method. Uses embedded Euler method for adaptivity.
    Also known as Heun-Euler method.

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_rk12, **kwargs)


@append_docstring(args_adaptive_docstring)
def RK23(dt, **kwargs):
    r"""
    2nd order adaptive solver with 3rd order error control,
    using the Bogacki–Shampine coefficients

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_rk23, **kwargs)


@append_docstring(args_adaptive_docstring)
def RK45(dt, **kwargs):
    r"""
    Dormand-Prince's 5/4 Runge-Kutta method. (free 4th order interpolant).

    """
    from . import _tableau as rkt

    return RKExplicitSolver(dt, tableau=rkt.bt_rk4_dopri, **kwargs)
