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

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from netket.utils.struct import dataclass
from netket.utils.types import Array, PyTree

default_dtype = jnp.float64


def expand_dim(tree: PyTree, sz: int):
    """
    creates a new pytree with same structure as input `tree`, but where very leaf
    has an extra dimension at 0 with size `sz`.
    """

    def _expand(x):
        return jnp.zeros((sz,) + x.shape, dtype=x.dtype)

    return jax.tree_map(_expand, tree)


@dataclass
class TableauRKExplicit:
    r"""
    Class representing the Butcher tableau of an explicit Runge-Kutta method [1,2],
    which, given the ODE dy/dt = F(t, y), updates the solution as

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

    order: Tuple[int, int]
    """The order of the tableau"""
    a: jax.numpy.ndarray
    b: jax.numpy.ndarray
    c: jax.numpy.ndarray
    c_error: Optional[jax.numpy.ndarray]
    """Coefficients for error estimation."""

    @property
    def is_explicit(self):
        jnp.allclose(self.a, jnp.tril(self.a))  # check if lower triangular

    @property
    def is_adaptive(self):
        return self.b.ndim == 2

    @property
    def is_fsal(self):
        """Returns True if the first iteration is the same as last."""
        # TODO: this is not yet supported
        return False

    @property
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function)
        of the RK scheme.
        """
        return len(self.c)

    @property
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.is_adaptive:
            return None
        else:
            return self.order[1]

    def _compute_slopes(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Computes the intermediate slopes k_l."""
        times = t + self.c * dt

        # TODO: Use FSAL

        k = expand_dim(y_t, self.stages)
        for l in range(self.stages):
            dy_l = jax.tree_map(lambda k: jnp.tensordot(self.a[l], k, axes=1), k)
            y_l = jax.tree_multimap(lambda y_t, dy_l: y_t + dt * dy_l, y_t, dy_l)
            k_l = f(times[l], y_l, stage=l)
            k = jax.tree_multimap(lambda k, k_l: k.at[l].set(k_l), k, k_l)

        return k

    def step(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Perform one fixed-size RK step from `t` to `t + dt`."""
        k = self._compute_slopes(f, t, dt, y_t)

        b = self.b[0] if self.b.ndim == 2 else self.b
        y_tp1 = jax.tree_multimap(
            lambda y_t, k: y_t + dt * jnp.tensordot(b, k, axes=1), y_t, k
        )

        return y_tp1

    def step_with_error(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """
        Perform one fixed-size RK step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        if not self.is_adaptive:
            raise RuntimeError(f"{self} is not adaptive")

        k = self._compute_slopes(f, t, dt, y_t)

        y_tp1 = jax.tree_multimap(
            lambda y_t, k: y_t + dt * jnp.tensordot(self.b[0], k, axes=1), y_t, k
        )
        db = self.b[0] - self.b[1]
        y_err = jax.tree_map(lambda k: dt * jnp.tensordot(db, k, axes=1), k)

        return y_tp1, y_err


@dataclass
class NamedTableau:
    name: str
    data: TableauRKExplicit

    def __repr__(self) -> str:
        return self.name


# fmt: off
# flake8: noqa: E123, E126, E201, E202, E221, E226, E231, E241, E251

# Fixed Step methods
bt_feuler = TableauRKExplicit(
                order = (1,),
                a = jnp.zeros((1,1), dtype=default_dtype),
                b = jnp.ones((1,), dtype=default_dtype),
                c = jnp.zeros((1), dtype=default_dtype),
                c_error = None,
                )
bt_feuler = NamedTableau("Euler", bt_feuler)


bt_midpoint = TableauRKExplicit(
                order = (2,),
                a = jnp.array([[0,   0],
                               [1/2, 0]], dtype=default_dtype),
                b = jnp.array( [0,   1], dtype=default_dtype),
                c = jnp.array( [0, 1/2], dtype=default_dtype),
                c_error = None,
                )
bt_midpoint = NamedTableau("Midpoint", bt_midpoint)


bt_heun = TableauRKExplicit(
                order = (2,),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=default_dtype),
                b = jnp.array( [1/2, 1/2], dtype=default_dtype),
                c = jnp.array( [0, 1], dtype=default_dtype),
                c_error = None,
                )
bt_heun = NamedTableau("Heun", bt_heun)


bt_rk4  = TableauRKExplicit(
                order = (4,),
                a = jnp.array([[0,   0,   0,   0],
                               [1/2, 0,   0,   0],
                               [0,   1/2, 0,   0],
                               [0,   0,   1,   0]], dtype=default_dtype),
                b = jnp.array( [1/6,  1/3,  1/3,  1/6], dtype=default_dtype),
                c = jnp.array( [0, 1/2, 1/2, 1], dtype=default_dtype),
                c_error = None,
                )
bt_rk4 = NamedTableau("RK4", bt_rk4)


# Adaptive step:
# Heun Euler https://en.wikipedia.org/wiki/Runge–Kutta_methods
bt_rk12  = TableauRKExplicit(
                order = (2,1),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=default_dtype),
                b = jnp.array([[1/2, 1/2],
                               [1,   0]], dtype=default_dtype),
                c = jnp.array( [0, 1], dtype=default_dtype),
                c_error = None,
                )
bt_rk12 = NamedTableau("RK12", bt_rk12)


# Bogacki–Shampine coefficients
bt_rk23  = TableauRKExplicit(
                order = (2,3),
                a = jnp.array([[0,   0,   0,   0],
                               [1/2, 0,   0,   0],
                               [0,   3/4, 0,   0],
                               [2/9, 1/3, 4/9, 0]], dtype=default_dtype),
                b = jnp.array([[7/24,1/4, 1/3, 1/8],
                               [2/9, 1/3, 4/9, 0]], dtype=default_dtype),
                c = jnp.array( [0, 1/2, 3/4, 1], dtype=default_dtype),
                c_error = None,
                )
bt_rk23 = NamedTableau("RK23", bt_rk23)


bt_rk4_fehlberg = TableauRKExplicit(
                order = (4,5),
                a = jnp.array([[ 0,          0,          0,           0,            0,      0 ],
                              [  1/4,        0,          0,           0,            0,      0 ],
                              [  3/32,       9/32,       0,           0,            0,      0 ],
                              [  1932/2197,  -7200/2197, 7296/2197,   0,            0,      0 ],
                              [  439/216,    -8,         3680/513,    -845/4104,    0,      0 ],
                              [  -8/27,      2,          -3544/2565,  1859/4104,    11/40,  0 ]], dtype=default_dtype),
                b = jnp.array([[ 25/216,     0,          1408/2565,   2197/4104,    -1/5,   0 ],
                               [ 16/135,     0,          6656/12825,  28561/56430,  -9/50,  2/55]], dtype=default_dtype),
                c = jnp.array( [  0,         1/4,        3/8,         12/13,        1,      1/2], dtype=default_dtype),
                c_error = None,
                )
bt_rk4_fehlberg = NamedTableau("RK45Fehlberg", bt_rk4_fehlberg)


bt_rk4_dopri  = TableauRKExplicit(
                order = (5,4),
                a = jnp.array([[ 0,           0,           0,           0,        0,             0,         0 ],
                              [  1/5,         0,           0,           0,        0,             0,         0 ],
                              [  3/40,        9/40,        0,           0,        0,             0,         0 ],
                              [  44/45,       -56/15,      32/9,        0,        0,             0,         0 ],
                              [  19372/6561,  -25360/2187, 64448/6561,  -212/729, 0,             0,         0 ],
                              [  9017/3168,   -355/33,     46732/5247,  49/176,   -5103/18656,   0,         0 ],
                              [  35/384,      0,           500/1113,    125/192,  -2187/6784,    11/84,     0 ]], dtype=default_dtype),
                b = jnp.array([[ 35/384,      0,           500/1113,    125/192,  -2187/6784,    11/84,     0 ],
                               [ 5179/57600,  0,           7571/16695,  393/640,  -92097/339200, 187/2100,  1/40 ]], dtype=default_dtype),
                c = jnp.array( [ 0,           1/5,         3/10,        4/5,      8/9,           1,         1], dtype=default_dtype),
                c_error = None,
                )
bt_rk4_dopri = NamedTableau("RK45", bt_rk4_dopri)

# fmt: on
