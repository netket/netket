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

from netket import config

from ._rk_tableau import TableauRKExplicit, NamedTableau

default_dtype = jnp.float64 if config.netket_enable_x64 else jnp.float32


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
