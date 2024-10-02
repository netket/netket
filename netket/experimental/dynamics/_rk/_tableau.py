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

import jax
import jax.numpy as jnp

from netket import config
from netket.utils.struct import dataclass, field

default_dtype = jnp.float64 if config.netket_enable_x64 else jnp.float32


@dataclass
class TableauRKExplicit:
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

    order: tuple[int, int]
    """The order of the tableau"""

    a: jax.numpy.ndarray = field(repr=False)
    """Coefficients of th intermediate states."""
    b: jax.numpy.ndarray = field(repr=False)
    """Coefficients of the intermediate slopes."""
    c: jax.numpy.ndarray = field(repr=False)
    """Coefficients of the intermediate times."""

    name: str = field(pytree_node=False, default="RKTableau")
    """The name of the tableau."""

    def __repr__(self):
        return self.name

    @property
    def is_adaptive(self):
        """Boolean indication whether the integrator can beå adaptive."""
        return self.b.ndim == 2


# fmt: off
# flake8: noqa: E123, E126, E201, E202, E221, E226, E231, E241, E251

# Fixed Step methods
bt_feuler = TableauRKExplicit(
                order = (1,),
                a = jnp.zeros((1,1), dtype=default_dtype),
                b = jnp.ones((1,), dtype=default_dtype),
                c = jnp.zeros((1), dtype=default_dtype),
                name = "Euler"
                )


bt_midpoint = TableauRKExplicit(
                order = (2,),
                a = jnp.array([[0,   0],
                               [1/2, 0]], dtype=default_dtype),
                b = jnp.array( [0,   1], dtype=default_dtype),
                c = jnp.array( [0, 1/2], dtype=default_dtype),
                name = "Midpoint"
                )


bt_heun = TableauRKExplicit(
                order = (2,),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=default_dtype),
                b = jnp.array( [1/2, 1/2], dtype=default_dtype),
                c = jnp.array( [0, 1], dtype=default_dtype),
                name = "Heun"
                )


bt_rk4  = TableauRKExplicit(
                order = (4,),
                a = jnp.array([[0,   0,   0,   0],
                               [1/2, 0,   0,   0],
                               [0,   1/2, 0,   0],
                               [0,   0,   1,   0]], dtype=default_dtype),
                b = jnp.array( [1/6,  1/3,  1/3,  1/6], dtype=default_dtype),
                c = jnp.array( [0, 1/2, 1/2, 1], dtype=default_dtype),
                name = "RK4"
                )


# Adaptive step:
# Heun Euler https://en.wikipedia.org/wiki/Runge–Kutta_methods
bt_rk12  = TableauRKExplicit(
                order = (2,1),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=default_dtype),
                b = jnp.array([[1/2, 1/2],
                               [1,   0]], dtype=default_dtype),
                c = jnp.array( [0, 1], dtype=default_dtype),
                name = "RK12"
                )


# Bogacki–Shampine coefficients
bt_rk23  = TableauRKExplicit(
                order = (3,2),
                a = jnp.array([[0,   0,   0,   0],
                               [1/2, 0,   0,   0],
                               [0,   3/4, 0,   0],
                               [2/9, 1/3, 4/9, 0]], dtype=default_dtype),
                b = jnp.array([[7/24,1/4, 1/3, 1/8],
                               [2/9, 1/3, 4/9, 0]], dtype=default_dtype),
                c = jnp.array( [0, 1/2, 3/4, 1], dtype=default_dtype),
                name = "RK23"
                )


bt_rk4_fehlberg = TableauRKExplicit(
                order = (5,4),
                a = jnp.array([[ 0,          0,          0,           0,            0,      0 ],
                              [  1/4,        0,          0,           0,            0,      0 ],
                              [  3/32,       9/32,       0,           0,            0,      0 ],
                              [  1932/2197,  -7200/2197, 7296/2197,   0,            0,      0 ],
                              [  439/216,    -8,         3680/513,    -845/4104,    0,      0 ],
                              [  -8/27,      2,          -3544/2565,  1859/4104,    11/40,  0 ]], dtype=default_dtype),
                b = jnp.array([[ 25/216,     0,          1408/2565,   2197/4104,    -1/5,   0 ],
                               [ 16/135,     0,          6656/12825,  28561/56430,  -9/50,  2/55]], dtype=default_dtype),
                c = jnp.array( [  0,         1/4,        3/8,         12/13,        1,      1/2], dtype=default_dtype),
                name = "RK45Fehlberg"
                )


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
                name = "RK45"
                )
