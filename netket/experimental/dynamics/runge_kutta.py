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

from builtins import RuntimeError, next
import dataclasses
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from netket.utils.struct import dataclass
from netket.utils.types import Array

dtype = jnp.float64


@dataclass
class TableauRKExplicit:
    name: str
    """The name of the RK Tableau."""
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

    def _get_ks(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Computes the intermediate slopes k_l."""
        times = t + self.c * dt

        k = jnp.zeros((y_t.shape[0], self.stages), dtype=y_t.dtype)
        for l in range(self.stages):
            dy_l = jnp.sum(self.a[l, :] * k, axis=1)
            k_l = f(times[l], y_t + dt * dy_l, stage=l)
            k = k.at[:, l].set(k_l)

        return k

    def step(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Perform one fixed-size RK step from `t` to `t + dt`."""
        k = self._get_ks(f, t, dt, y_t)

        b = self.b[0] if self.b.ndim == 2 else self.b
        y_tp1 = y_t + dt * jnp.sum(b * k, axis=1)
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

        k = self._get_ks(f, t, dt, y_t)

        y_tp1 = y_t + dt * jnp.sum(self.b[0] * k, axis=1)
        y_err = dt * jnp.sum((self.b[0] - self.b[1]) * k, axis=1)
        return y_tp1, y_err


@dataclass
class RungeKuttaState:
    step_no: int
    t: float
    y: Array
    dt: float
    last_norm: Optional[float]


def scaled_error(y, y_err, atol, rtol, *, norm=None):
    if norm is None:
        norm = jnp.linalg.norm
    scale = (atol + norm(y) * rtol) / y_err.shape[0]
    return norm(y_err) / scale


def adapt_time_step(dt, scaled_error):
    safety_factor = 0.95
    err_exponent = -1.0 / 5.0
    return dt * jnp.clip(
        safety_factor * scaled_error ** err_exponent,
        1e-1,
        1e1,
    )


@partial(jax.jit, static_argnames=["step_fn", "norm_fn"])
def general_time_step_adaptive(
    step_fn: Callable,
    rkstate: RungeKuttaState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
):
    next_y, y_err = step_fn(rkstate.t, rkstate.dt, rkstate.y)
    scaled_err = scaled_error(rkstate.y, y_err, atol, rtol, norm=norm_fn)
    next_dt = adapt_time_step(rkstate.dt, scaled_err)
    return jax.lax.cond(
        scaled_err < 1.0,
        lambda _: rkstate.replace(
            step_no=rkstate.step_no + 1,
            y=next_y,
            t=rkstate.t + rkstate.dt,
            dt=next_dt,
        ),
        lambda _: rkstate.replace(dt=next_dt),
        None,
    )


@partial(jax.jit, static_argnames=["step_fn", "norm_fn"])
def general_time_step_fixed(step_fn: Callable, rkstate: RungeKuttaState):
    next_y = step_fn(rkstate.t, rkstate.dt, rkstate.y)
    r = rkstate.replace(t=rkstate.t + rkstate.dt, y=next_y, step_no=rkstate.step_no + 1)
    return r


@dataclasses.dataclass
class RungeKuttaSolver:
    tableau: TableauRKExplicit

    f: Callable
    t0: float
    y0: Array

    dt: float

    tend: float

    use_adaptive: bool
    norm: Callable

    atol: float = 0.0
    rtol: float = 1e-7

    def __post_init__(self):
        if self.use_adaptive and not self.tableau.is_adaptive:
            raise RuntimeError(
                f"Solver {self.tablaeu} does not support adaptive step size"
            )
        if self.use_adaptive:
            self._do_step = lambda state: general_time_step_adaptive(
                lambda t, dt, y, **kw: self.tableau.step_with_error(
                    self.f, t, dt, y, **kw
                ),
                state,
                atol=self.atol,
                rtol=self.rtol,
                norm_fn=self.norm,
            )
        else:
            self._do_step = lambda state: general_time_step_fixed(
                lambda t, dt, y, **kw: self.tableau.step(self.f, t, dt, y, **kw),
                state,
            )

        self._rkstate = RungeKuttaState(
            step_no=0,
            t=self.t0,
            y=self.y0,
            dt=self.dt,
            last_norm=None,
        )

    def step(self):
        self._rkstate = self._do_step(self._rkstate)

    @property
    def status(self):
        if self.t < self.tend:
            return "running"
        else:
            return "done"

    @property
    def t(self):
        return self._rkstate.t

    @property
    def y(self):
        return self._rkstate.y

    @property
    def current_dt(self):
        return self._rkstate.dt


def RKSolver(dt, tableau, adaptive=False, norm=None, **kwargs):
    def make(f, tspan, y0):
        y0 = jnp.asarray(y0)
        t0, tend = tspan
        return RungeKuttaSolver(
            tableau, f, t0, y0, dt, tend, use_adaptive=adaptive, norm=norm, **kwargs
        )

    return make


# fmt: off
# Fixed Step methods
bt_feuler = TableauRKExplicit(
                name = "feuler", 
                order = (1,),
                a = jnp.zeros((1,1), dtype=dtype),
                b = jnp.ones((1,1), dtype=dtype),
                c = jnp.zeros((1), dtype=dtype),
                c_error = None,
                )
Euler = partial(RKSolver, tableau=bt_feuler)


bt_midpoint = TableauRKExplicit(
                name = "midpoint", 
                order = (2,),
                a = jnp.array([[0,   0],
                               [1/2, 0]], dtype=dtype),
                b = jnp.array( [0,   1], dtype=dtype),
                c = jnp.array( [0, 1/2], dtype=dtype),
                c_error = None,
                )
Midpoint = partial(RKSolver, tableau=bt_midpoint)


bt_heun = TableauRKExplicit(
                name = "heun", 
                order = (2,),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=dtype),
                b = jnp.array( [1/2, 1/2], dtype=dtype),
                c = jnp.array( [0, 1], dtype=dtype),
                c_error = None,
                )
Heun = partial(RKSolver, tableau=bt_heun)

bt_rk4  = TableauRKExplicit(
                name = "rk4", 
                order = (4,),
                a = jnp.array([[0,   0,   0,   0],
                               [1/2, 0,   0,   0],
                               [0,   1/2, 0,   0],
                               [0,   0,   1,   1]], dtype=dtype),
                b = jnp.array( [1/6,  1/3,  1/3,  1/6], dtype=dtype),
                c = jnp.array( [0, 1/2, 1/2, 1], dtype=dtype),
                c_error = None,
                )
RK4 = partial(RKSolver, tableau=bt_rk4)


# Adaptive step:
# Heun Euler https://en.wikipedia.org/wiki/Runge–Kutta_methods
bt_rk12  = TableauRKExplicit(
                name = "rk21", 
                order = (2,1),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=dtype),
                b = jnp.array([[1/2, 1/2],
                               [1,   0]], dtype=dtype),
                c = jnp.array( [0, 1], dtype=dtype),
                c_error = None,
                )
RK12 = partial(RKSolver, tableau=bt_rk12)

# Bogacki–Shampine coefficients
bt_rk23  = TableauRKExplicit(
                name = "rk23", 
                order = (2,3),
                a = jnp.array([[0,   0,   0,   0],
                               [1/2, 0,   0,   0],
                               [0,   3/4, 0,   0],
                               [2/9, 1/3, 4/9, 0]], dtype=dtype),
                b = jnp.array([[7/24,1/4, 1/3, 1/8],
                               [2/9, 1/3, 4/9, 0]], dtype=dtype),
                c = jnp.array( [0, 1/2, 3/4, 1], dtype=dtype),
                c_error = None,
                )
RK23 = partial(RKSolver, tableau=bt_rk23)

bt_rk4_fehlberg  = TableauRKExplicit(
                name = "fehlberg", 
                order = (4,5),
                a = jnp.array([[ 0,          0,          0,           0,            0,      0 ],
                              [  1/4,        0,          0,           0,            0,      0 ],
                              [  3/32,       9/32,       0,           0,            0,      0 ],
                              [  1932/2197,  -7200/2197, 7296/2197,   0,            0,      0 ],
                              [  439/216,    -8,         3680/513,    -845/4104,    0,      0 ],
                              [  -8/27,      2,          -3544/2565,  1859/4104,    11/40,  0 ]], dtype=dtype),
                b = jnp.array([[ 25/216,     0,          1408/2565,   2197/4104,    -1/5,   0 ],
                               [ 16/135,     0,          6656/12825,  28561/56430,  -9/50,  2/55]], dtype=dtype),
                c = jnp.array( [  0,         1/4,        3/8,         12/13,        1,      1/2], dtype=dtype),
                c_error = None,
                )

bt_rk4_dopri  = TableauRKExplicit(
                name = "dopri", 
                order = (5,4),
                a = jnp.array([[ 0,           0,           0,           0,        0,             0,         0 ],
                              [  1/5,         0,           0,           0,        0,             0,         0 ],
                              [  3/40,        9/40,        0,           0,        0,             0,         0 ],
                              [  44/45,       -56/15,      32/9,        0,        0,             0,         0 ],
                              [  19372/6561,  -25360/2187, 64448/6561,  -212/729, 0,             0,         0 ],
                              [  9017/3168,   -355/33,     46732/5247,  49/176,   -5103/18656,   0,         0 ],
                              [  35/384,      0,           500/1113,    125/192,  -2187/6784,    11/84,     0 ]], dtype=dtype),
                b = jnp.array([[ 35/384,      0,           500/1113,    125/192,  -2187/6784,    11/84,     0 ],
                               [ 5179/57600,  0,           7571/16695,  393/640,  -92097/339200, 187/2100,  1/40 ]], dtype=dtype),
                c = jnp.array( [ 0,           1/5,         3/10,        4/5,      8/9,           1,         1], dtype=dtype),
                c_error = None,
                )
RK45 = partial(RKSolver, tableau=bt_rk4_dopri)
# fmt: on
