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
from ctypes import c_buffer
import dataclasses
from functools import partial
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import KahanSum
from netket.utils.struct import dataclass
from netket.utils.types import Array, PyTree
from netket import jax as nkjax

dtype = jnp.float64


def expand_dim(tree: PyTree, sz: int):
    """
    creates a new pytree with same structure as input `tree`, but where very leaf
    has an extra dimension at 0 with size `sz`.
    """

    def _expand(x):
        return jnp.zeros((sz,) + x.shape, dtype=x.dtype)

    return jax.tree_map(_expand, tree)


def euclidean_norm(x: Union[PyTree, Array]):
    """
    Computes the Euclidean L2 norm of the Array or PyTree intended as a flattened array
    """
    if isinstance(x, jnp.ndarray):
        return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 2), x),
            )
        )


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

    def _compute_slopes(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Computes the intermediate slopes k_l."""
        times = t + self.c * dt

        # TODO: Use FSALLast

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
class RungeKuttaState:
    step_no: int
    """Number of successful steps since the start of the iteration."""
    step_no_total: int
    """Number of steps since the start of the iteration, including rejected steps."""
    t: KahanSum
    """Current time."""
    y: Array
    """Solution at current time."""
    dt: float
    """Current step size."""
    last_norm: Optional[float] = None
    """Solution norm at previous time step."""
    accepted: bool = True
    """Whether the last RK step was accepted or should be repeated."""

    def __repr__(self):
        return "RKState(step_no(total)={}({}), t={}, dt={:.2e}{}{})".format(
            self.step_no,
            self.step_no_total,
            self.t.value,
            self.dt,
            f", {self.last_norm:.2e}" if self.last_norm is not None else "",
            f", {'A' if self.accepted else 'R'}",
        )


def scaled_error(y, y_err, atol, rtol, *, last_norm_y=None, norm_fn):
    norm_y = norm_fn(y)
    scale = (atol + jnp.maximum(norm_y, last_norm_y) * rtol) / nkjax.tree_size(y_err)
    return norm_fn(y_err) / scale, norm_y


LimitsType = Tuple[Optional[float], Optional[float]]


def propose_time_step(dt, scaled_error, limits: LimitsType):
    safety_factor = 0.95
    err_exponent = -1.0 / 5.0
    return jnp.clip(
        dt * safety_factor * scaled_error ** err_exponent,
        limits[0],
        limits[1],
    )


# TODO: Allow JITing
# @partial(jax.jit, static_argnames=["step_fn", "norm_fn"])
def general_time_step_adaptive(
    step_fn: Callable,
    rk_state: RungeKuttaState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
    max_dt: Optional[float],
    dt_limits: LimitsType,
):
    if max_dt is None:
        actual_dt = rk_state.dt
    else:
        actual_dt = jnp.minimum(rk_state.dt, max_dt)

    next_y, y_err = step_fn(rk_state.t.value, actual_dt, rk_state.y)

    scaled_err, norm_y = scaled_error(
        rk_state.y,
        y_err,
        atol,
        rtol,
        last_norm_y=rk_state.last_norm,
        norm_fn=norm_fn,
    )

    # Propose the next time step, but limited within [0.1 dt, 10 dt] and potential
    # global limits in dt_limits. Not used when actual_dt < rk_state.dt (i.e., the
    # integrator is doing a smaller step to hit a specific stop).
    def next_dt():
        return propose_time_step(
            actual_dt,
            scaled_err,
            limits=(
                jnp.maximum(0.1 * rk_state.dt, dt_limits[0])
                if dt_limits[0]
                else 0.1 * rk_state.dt,
                jnp.minimum(10.0 * rk_state.dt, dt_limits[1])
                if dt_limits[1]
                else 10.0 * rk_state.dt,
            ),
        )

    return jax.lax.cond(
        scaled_err < 1.0,
        # step accepted
        lambda _: rk_state.replace(
            step_no=rk_state.step_no + 1,
            step_no_total=rk_state.step_no_total + 1,
            y=next_y,
            t=rk_state.t + actual_dt,
            dt=jax.lax.cond(
                actual_dt == rk_state.dt,
                lambda _: next_dt(),
                lambda _: rk_state.dt,
                None,
            ),
            last_norm=norm_y,
            accepted=True,
        ),
        # step rejected, repeat with lower dt
        lambda _: rk_state.replace(
            step_no_total=rk_state.step_no_total + 1,
            dt=next_dt(),
            accepted=False,
        ),
        None,
    )


# TODO: Allow JITing
# @partial(jax.jit, static_argnames=["step_fn", "norm_fn"])
def general_time_step_fixed(
    step_fn: Callable, rk_state: RungeKuttaState, max_dt: Optional[float]
):
    if max_dt is None:
        actual_dt = rk_state.dt
    else:
        actual_dt = jnp.minimum(rk_state.dt, max_dt)
    next_y = step_fn(rk_state.t.value, actual_dt, rk_state.y)
    return rk_state.replace(
        step_no=rk_state.step_no + 1,
        step_no_total=rk_state.step_no_total + 1,
        t=rk_state.t + actual_dt,
        y=next_y,
        accepted=True,
    )


@dataclasses.dataclass
class RungeKuttaIntegrator:
    tableau: TableauRKExplicit

    f: Callable
    t0: float
    y0: Array

    initial_dt: float

    use_adaptive: bool
    norm: Callable

    atol: float = 0.0
    rtol: float = 1e-7
    dt_limits: Optional[LimitsType] = None

    def __post_init__(self):
        if self.use_adaptive and not self.tableau.is_adaptive:
            raise RuntimeError(
                f"Solver {self.tableau} does not support adaptive step size"
            )
        if self.use_adaptive:
            self._do_step = self._do_step_adaptive
        else:
            self._do_step = self._do_step_fixed

        if self.norm is None:
            self.norm = euclidean_norm

        if self.dt_limits is None:
            self.dt_limits = (None, 10 * self.initial_dt)

        self._rkstate = RungeKuttaState(
            step_no=0,
            step_no_total=0,
            t=KahanSum(self.t0),
            y=self.y0,
            dt=self.initial_dt,
            last_norm=0.0 if self.use_adaptive else None,
        )

    def step(self, max_dt=None):
        """
        Perform one full Runge-Kutta step by min(self.dt, max_dt).


        Returns:
            A boolean indicating whether the step was sucessful or
            was rejected by the step controller and should be retried.

            Note that the step size can be adjusted by the step controller
            in both cases, so the integrator state will have changed
            even after a rejected step.
        """
        self._rkstate = self._do_step(self._rkstate, max_dt)
        return self._rkstate.accepted

    def _do_step_fixed(self, rk_state, max_dt=None):
        return general_time_step_fixed(
            lambda t, dt, y, **kw: self.tableau.step(self.f, t, dt, y, **kw),
            rk_state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, rk_state, max_dt=None):
        return general_time_step_adaptive(
            lambda t, dt, y, **kw: self.tableau.step_with_error(self.f, t, dt, y, **kw),
            rk_state,
            atol=self.atol,
            rtol=self.rtol,
            norm_fn=self.norm,
            max_dt=max_dt,
            dt_limits=self.dt_limits,
        )

    @property
    def t(self):
        return self._rkstate.t.value

    @property
    def y(self):
        return self._rkstate.y

    @property
    def dt(self):
        return self._rkstate.dt


class RKIntegratorConfig:
    def __init__(self, dt, tableau, *, adaptive=False, **kwargs):
        self.dt = dt
        self.tableau = tableau
        self.adaptive = adaptive
        self.kwargs = kwargs

    def __call__(self, f, t0, y0, *, norm=None):
        return RungeKuttaIntegrator(
            self.tableau,
            f,
            t0,
            y0,
            initial_dt=self.dt,
            use_adaptive=self.adaptive,
            norm=norm,
            **self.kwargs,
        )

    def __repr__(self):
        return "{}(tableau={}, dt={}, adaptive={}{})".format(
            "RKIntegratorConfig",
            self.tableau.name,
            self.dt,
            self.adaptive,
            f", **kwargs={self.kwargs}" if self.kwargs else "",
        )


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
Euler = partial(RKIntegratorConfig, tableau=bt_feuler)


bt_midpoint = TableauRKExplicit(
                name = "midpoint", 
                order = (2,),
                a = jnp.array([[0,   0],
                               [1/2, 0]], dtype=dtype),
                b = jnp.array( [0,   1], dtype=dtype),
                c = jnp.array( [0, 1/2], dtype=dtype),
                c_error = None,
                )
Midpoint = partial(RKIntegratorConfig, tableau=bt_midpoint)


bt_heun = TableauRKExplicit(
                name = "heun", 
                order = (2,),
                a = jnp.array([[0,   0],
                               [1,   0]], dtype=dtype),
                b = jnp.array( [1/2, 1/2], dtype=dtype),
                c = jnp.array( [0, 1], dtype=dtype),
                c_error = None,
                )
Heun = partial(RKIntegratorConfig, tableau=bt_heun)

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
RK4 = partial(RKIntegratorConfig, tableau=bt_rk4)


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
RK12 = partial(RKIntegratorConfig, tableau=bt_rk12)

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
RK23 = partial(RKIntegratorConfig, tableau=bt_rk23)

bt_rk4_fehlberg = TableauRKExplicit(
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
RK45 = partial(RKIntegratorConfig, tableau=bt_rk4_dopri)
# fmt: on
