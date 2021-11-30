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
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import KahanSum
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

    def _compute_slopes(
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
        k = self._compute_slopes(f, t, dt, y_t)

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

        k = self._compute_slopes(f, t, dt, y_t)

        y_tp1 = y_t + dt * jnp.sum(self.b[0] * k, axis=1)
        y_err = dt * jnp.sum((self.b[0] - self.b[1]) * k, axis=1)

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
            f", {self.last_norm:.2e}" if self.last_norm else "",
            f", {'A' if self.accepted else 'R'}",
        )


def scaled_error(y, y_err, atol, rtol, *, norm=jnp.linalg.norm):
    scale = (atol + norm(y) * rtol) / y_err.shape[0]
    return norm(y_err) / scale


LimitsType = Tuple[Optional[float], Optional[float]]


def propose_time_step(dt, scaled_error, limits: LimitsType):
    safety_factor = 0.95
    err_exponent = -1.0 / 5.0
    return jnp.clip(
        dt * safety_factor * scaled_error ** err_exponent,
        limits[0],
        limits[1],
    )


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
    scaled_err = scaled_error(rk_state.y, y_err, atol, rtol, norm=norm_fn)

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
            accepted=True,
        ),
        lambda _: rk_state.replace(
            step_no_total=rk_state.step_no_total + 1,
            dt=next_dt(),
            accepted=False,
        ),
        None,
    )


# @partial(jax.jit, static_argnames=["step_fn", "norm_fn"])
def general_time_step_fixed(step_fn: Callable, rk_state: RungeKuttaState):
    next_y = step_fn(rk_state.t.value, rk_state.dt, rk_state.y)
    return rk_state.replace(
        step_no=rk_state.step_no + 1,
        step_no_total=rk_state.step_no_total + 1,
        t=rk_state.t + rk_state.dt,
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

        if self.dt_limits is None:
            self.dt_limits = (None, 10 * self.initial_dt)

        self._rkstate = RungeKuttaState(
            step_no=0,
            step_no_total=0,
            t=KahanSum(self.t0),
            y=self.y0,
            dt=self.initial_dt,
        )

    def step(self, max_dt=None):
        """
        Perform one full Runge-Kutta step.
        """
        # print(f"RK.step (t={self.t}, {max_dt=})")
        # print(f"      => {self._rkstate}")
        self._rkstate = self._do_step(self._rkstate, max_dt)
        # print(f"      <= {self._rkstate}")
        return self._rkstate.accepted

    def _do_step_fixed(self, rk_state, max_dt=None):
        return general_time_step_fixed(
            lambda t, dt, y, **kw: self.tableau.step(self.f, t, dt, y, **kw),
            rk_state,
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


def RKSolver(dt, tableau, adaptive=False, norm=jnp.linalg.norm, **kwargs):
    def make(f, t0, y0):
        y0 = jnp.asarray(y0)
        return RungeKuttaIntegrator(
            tableau,
            f,
            t0,
            y0,
            initial_dt=dt,
            use_adaptive=adaptive,
            norm=norm,
            **kwargs,
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
RK45 = partial(RKSolver, tableau=bt_rk4_dopri)

# Tsit5 method. Tableau entries taken from DiffEqDevTools.jl:
# https://github.com/SciML/DiffEqDevTools.jl/blob/37fde034a03b8dcf440613a81df1483a06ea25f9/src/ode_tableaus.jl#L988-L1042
def _make_tsit5(dtype=None):
    a = np.zeros((7, 7), dtype)
    b = np.zeros((2, 7), dtype)
    c = np.zeros((7,), dtype)

    a[1, 0] = 161 / 1000
    a[2, 0] = -.8480655492356988544426874250230774675121177393430391537369234245294192976164141156943e-2
    a[2, 1] = .3354806554923569885444268742502307746751211773934303915373692342452941929761641411569
    a[3, 0] = 2.897153057105493432130432594192938764924887287701866490314866693455023795137503079289
    a[3, 1] = -6.359448489975074843148159912383825625952700647415626703305928850207288721235210244366
    a[3, 2] = 4.362295432869581411017727318190886861027813359713760212991062156752264926097707165077
    a[4, 0] = 5.325864828439256604428877920840511317836476253097040101202360397727981648835607691791
    a[4, 1] = -11.74888356406282787774717033978577296188744178259862899288666928009020615663593781589
    a[4, 2] = 7.495539342889836208304604784564358155658679161518186721010132816213648793440552049753
    a[4, 3] = -.9249506636175524925650207933207191611349983406029535244034750452930469056411389539635e-1
    a[5, 0] = 5.861455442946420028659251486982647890394337666164814434818157239052507339770711679748
    a[5, 1] = -12.92096931784710929170611868178335939541780751955743459166312250439928519268343184452
    a[5, 2] = 8.159367898576158643180400794539253485181918321135053305748355423955009222648673734986
    a[5, 3] = -.7158497328140099722453054252582973869127213147363544882721139659546372402303777878835e-1
    a[5, 4] = -.2826905039406838290900305721271224146717633626879770007617876201276764571291579142206e-1
    a[6, 0] = .9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1
    a[6, 1] = 1 / 100
    a[6, 2] = .4798896504144995747752495322905965199130404621990332488332634944254542060153074523509
    a[6, 3] = 1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331
    a[6, 4] = -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677
    a[6, 5] = 2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841
    
    b[0, 0] = .9646076681806522951816731316512876333711995238157997181903319145764851595234062815396e-1
    b[0, 1] = 1 / 100
    b[0, 2] = .4798896504144995747752495322905965199130404621990332488332634944254542060153074523509
    b[0, 3] = 1.379008574103741893192274821856872770756462643091360525934940067397245698027561293331
    b[0, 4] = -3.290069515436080679901047585711363850115683290894936158531296799594813811049925401677
    b[0, 5] = 2.324710524099773982415355918398765796109060233222962411944060046314465391054716027841
    b[1, 0] = .9468075576583945807478876255758922856117527357724631226139574065785592789071067303271e-1
    b[1, 1] = .9183565540343253096776363936645313759813746240984095238905939532922955247253608687270e-2
    b[1, 2] = .4877705284247615707855642599631228241516691959761363774365216240304071651579571959813
    b[1, 3] = 1.234297566930478985655109673884237654035539930748192848315425833500484878378061439761
    b[1, 4] = -2.707712349983525454881109975059321670689605166938197378763992255714444407154902012702
    b[1, 5] = 1.866628418170587035753719399566211498666255505244122593996591602841258328965767580089
    b[1, 6] = 1 / 66

    c[1] = 161 / 1000
    c[2] = 327 / 1000
    c[3] =   9 / 10
    c[4] = 0.9800255409045096857298102862870245954942137979563024768854764293221195950761080302604
    c[5] = 1.0
    c[6] = 1.0

    return TableauRKExplicit(
        name="Tsit5",
        order=(5, 4),
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        c=jnp.asarray(c),
        c_error=None,
    )

Tsit5 = partial(RKSolver, tableau=_make_tsit5())
# fmt: on
