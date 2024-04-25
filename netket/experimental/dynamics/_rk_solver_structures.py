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

from enum import IntFlag, auto
from functools import partial, wraps
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

import netket as nk
from netket import config
from netket.utils.mpi.primitives import mpi_all_jax
from netket.utils.struct import dataclass, field
from netket.utils.types import Array, PyTree
from netket.utils.numbers import dtype as _dtype

from . import _rk_tableau as rkt


def maybe_jax_jit(fun, *jit_args, **jit_kwargs):
    """
    Only jit if `config.netket_experimental_disable_ode_jit` is False.

    This is used to disable jitting when this config is set. The switch is
    performed at runtime so that the flag can be changed as desired.
    """

    # jit the function only once:
    jitted_fun = jax.jit(fun, *jit_args, **jit_kwargs)

    @wraps(fun)
    def _maybe_jitted_fun(*args, **kwargs):
        if config.netket_experimental_disable_ode_jit:
            with jax.spmd_mode("allow_all"):
                res = fun(*args, **kwargs)
            return res
        else:
            return jitted_fun(*args, **kwargs)

    return _maybe_jitted_fun


class SolverFlags(IntFlag):
    """
    Enum class containing flags for signaling solver information from within `jax.jit`ed code.
    """

    NONE = 0
    INFO_STEP_ACCEPTED = auto()
    WARN_MIN_DT = auto()
    WARN_MAX_DT = auto()
    ERROR_INVALID_DT = auto()

    WARNINGS_FLAGS = WARN_MIN_DT | WARN_MAX_DT
    ERROR_FLAGS = ERROR_INVALID_DT

    __MESSAGES__ = {
        INFO_STEP_ACCEPTED: "Step accepted",
        WARN_MIN_DT: "dt reached lower bound",
        WARN_MAX_DT: "dt reached upper bound",
        ERROR_INVALID_DT: "Invalid value of dt",
    }

    def message(self) -> str:
        """Returns a string with a description of the currently set flags."""
        msg = self.__MESSAGES__
        return ", ".join(msg[flag] for flag in msg.keys() if flag & self != 0)


def set_flag_jax(condition, flags, flag):
    """
    If `condition` is true, `flags` is updated by setting `flag` to 1.
    This is equivalent to the following code, but compatible with jax.jit:
        if condition:
            flags |= flag
    """
    return jax.lax.cond(
        condition,
        lambda x: x | flag,
        lambda x: x,
        flags,
    )


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
                jax.tree_util.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 2), x),
            )
        )


def maximum_norm(x: Union[PyTree, Array]):
    """
    Computes the maximum norm of the Array or PyTree intended as a flattened array
    """
    if isinstance(x, jnp.ndarray):
        return jnp.max(jnp.abs(x))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                jnp.maximum,
                jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), x),
            )
        )


@dataclass
class RungeKuttaState:
    step_no: int
    """Number of successful steps since the start of the iteration."""
    step_no_total: int
    """Number of steps since the start of the iteration, including rejected steps."""
    t: nk.utils.KahanSum
    """Current time."""
    y: Array
    """Solution at current time."""
    dt: float
    """Current step size."""
    last_norm: Optional[float] = None
    """Solution norm at previous time step."""
    last_scaled_error: Optional[float] = None
    """Error of the TDVP integrator at the last time step."""
    flags: SolverFlags = SolverFlags.INFO_STEP_ACCEPTED
    """Flags containing information on the solver state."""

    def __repr__(self):
        try:
            dt = "{self.dt:.2e}"
            last_norm = f", {self.last_norm:.2e}" if self.last_norm is not None else ""
            accepted = (f", {'A' if self.accepted else 'R'}",)
        except (ValueError, TypeError):
            dt = f"{self.dt}"
            last_norm = f"{self.last_norm}"
            accepted = f"{SolverFlags.INFO_STEP_ACCEPTED}"

        return "RKState(step_no(total)={}({}), t={}, dt={}{}{})".format(
            self.step_no,
            self.step_no_total,
            self.t.value,
            dt,
            last_norm,
            accepted,
        )

    @property
    def accepted(self):
        return SolverFlags.INFO_STEP_ACCEPTED & self.flags != 0


def scaled_error(y, y_err, atol, rtol, *, last_norm_y=None, norm_fn):
    norm_y = norm_fn(y)
    scale = (atol + jnp.maximum(norm_y, last_norm_y) * rtol) / nk.jax.tree_size(y_err)
    return norm_fn(y_err) / scale, norm_y


LimitsType = tuple[Optional[float], Optional[float]]
"""Type of the dt limits field, having independently optional upper and lower bounds."""


def propose_time_step(
    dt: float, scaled_error: float, error_order: int, limits: LimitsType
):
    """
    Propose an updated dt based on the scheme suggested in Numerical Recipes, 3rd ed.
    """
    SAFETY_FACTOR = 0.95
    err_exponent = -1.0 / (1 + error_order)
    return jnp.clip(
        dt * SAFETY_FACTOR * scaled_error**err_exponent,
        limits[0],
        limits[1],
    )


@partial(maybe_jax_jit, static_argnames=["f", "norm_fn", "dt_limits"])
def general_time_step_adaptive(
    tableau: rkt.TableauRKExplicit,
    f: Callable,
    rk_state: RungeKuttaState,
    atol: float,
    rtol: float,
    norm_fn: Callable,
    max_dt: Optional[float],
    dt_limits: LimitsType,
):
    flags = SolverFlags(0)

    if max_dt is None:
        actual_dt = rk_state.dt
    else:
        actual_dt = jnp.minimum(rk_state.dt, max_dt)

    y_tp1, y_err = tableau.step_with_error(f, rk_state.t.value, actual_dt, rk_state.y)

    scaled_err, norm_y = scaled_error(
        y_tp1,
        y_err,
        atol,
        rtol,
        last_norm_y=rk_state.last_norm,
        norm_fn=norm_fn,
    )

    # Propose the next time step, but limited within [0.1 dt, 5 dt] and potential
    # global limits in dt_limits. Not used when actual_dt < rk_state.dt (i.e., the
    # integrator is doing a smaller step to hit a specific stop).
    next_dt = propose_time_step(
        actual_dt,
        scaled_err,
        tableau.error_order,
        limits=(
            jnp.maximum(0.1 * rk_state.dt, dt_limits[0])
            if dt_limits[0]
            else 0.1 * rk_state.dt,
            jnp.minimum(5.0 * rk_state.dt, dt_limits[1])
            if dt_limits[1]
            else 5.0 * rk_state.dt,
        ),
    )

    # check if next dt is NaN
    flags = set_flag_jax(~jnp.isfinite(next_dt), flags, SolverFlags.ERROR_INVALID_DT)

    # check if we are at lower bound for dt
    if dt_limits[0] is not None:
        is_at_min_dt = jnp.isclose(next_dt, dt_limits[0])
        flags = set_flag_jax(is_at_min_dt, flags, SolverFlags.WARN_MIN_DT)
    else:
        is_at_min_dt = False
    if dt_limits[1] is not None:
        is_at_max_dt = jnp.isclose(next_dt, dt_limits[1])
        flags = set_flag_jax(is_at_max_dt, flags, SolverFlags.WARN_MAX_DT)

    # accept if error is within tolerances or we are already at the minimal step
    accept_step = jnp.logical_or(scaled_err < 1.0, is_at_min_dt)
    # accept the time step iff it is accepted by all MPI processes
    accept_step, _ = mpi_all_jax(accept_step)

    return jax.lax.cond(
        accept_step,
        # step accepted
        lambda _: rk_state.replace(
            step_no=rk_state.step_no + 1,
            step_no_total=rk_state.step_no_total + 1,
            y=y_tp1,
            t=rk_state.t + actual_dt,
            dt=jax.lax.cond(
                actual_dt == rk_state.dt,
                lambda _: next_dt,
                lambda _: rk_state.dt,
                None,
            ),
            last_norm=norm_y,
            last_scaled_error=scaled_err,
            flags=flags | SolverFlags.INFO_STEP_ACCEPTED,
        ),
        # step rejected, repeat with lower dt
        lambda _: rk_state.replace(
            step_no_total=rk_state.step_no_total + 1,
            dt=next_dt,
            flags=flags,
        ),
        None,
    )


@partial(maybe_jax_jit, static_argnames=["f"])
def general_time_step_fixed(
    tableau: rkt.TableauRKExplicit,
    f: Callable,
    rk_state: RungeKuttaState,
    max_dt: Optional[float],
):
    if max_dt is None:
        actual_dt = rk_state.dt
    else:
        actual_dt = jnp.minimum(rk_state.dt, max_dt)

    y_tp1 = tableau.step(f, rk_state.t.value, actual_dt, rk_state.y)

    return rk_state.replace(
        step_no=rk_state.step_no + 1,
        step_no_total=rk_state.step_no_total + 1,
        t=rk_state.t + actual_dt,
        y=y_tp1,
        flags=SolverFlags.INFO_STEP_ACCEPTED,
    )


@dataclass(_frozen=False)
class RungeKuttaIntegrator:
    tableau: rkt.NamedTableau

    f: Callable = field(repr=False)
    t0: float
    y0: Array = field(repr=False)

    initial_dt: float

    use_adaptive: bool
    norm: Callable

    atol: float = 0.0
    rtol: float = 1e-7
    dt_limits: Optional[LimitsType] = None

    def __post_init__(self):
        if self.use_adaptive and not self.tableau.data.is_adaptive:
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

        t_dtype = jnp.result_type(_dtype(self.t0), _dtype(self.initial_dt))
        setattr(self, "t0", jnp.array(self.t0, dtype=t_dtype))
        setattr(self, "initial_dt", jnp.array(self.initial_dt, dtype=t_dtype))

        self._rkstate = RungeKuttaState(
            step_no=0,
            step_no_total=0,
            t=nk.utils.KahanSum(self.t0),
            y=self.y0,
            dt=self.initial_dt,
            last_norm=0.0 if self.use_adaptive else None,
            last_scaled_error=0.0 if self.use_adaptive else None,
            flags=SolverFlags(0),
        )

    def step(self, max_dt=None):
        """
        Perform one full Runge-Kutta step by min(self.dt, max_dt).


        Returns:
            A boolean indicating whether the step was successful or
            was rejected by the step controller and should be retried.

            Note that the step size can be adjusted by the step controller
            in both cases, so the integrator state will have changed
            even after a rejected step.
        """
        self._rkstate = self._do_step(self._rkstate, max_dt)
        return self._rkstate.accepted

    def _do_step_fixed(self, rk_state, max_dt=None):
        return general_time_step_fixed(
            self.tableau.data,
            self.f,
            rk_state,
            max_dt=max_dt,
        )

    def _do_step_adaptive(self, rk_state, max_dt=None):
        return general_time_step_adaptive(
            self.tableau.data,
            self.f,
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

    def _get_solver_flags(self, intersect=SolverFlags.NONE) -> SolverFlags:
        """Returns the currently set flags of the solver, intersected with `intersect`."""
        # _rkstate.flags is turned into an int-valued DeviceArray by JAX,
        # so we convert it back.
        return SolverFlags(int(self._rkstate.flags) & intersect)

    @property
    def errors(self) -> SolverFlags:
        """Returns the currently set error flags of the solver."""
        return self._get_solver_flags(SolverFlags.ERROR_FLAGS)

    @property
    def warnings(self) -> SolverFlags:
        """Returns the currently set warning flags of the solver."""
        return self._get_solver_flags(SolverFlags.WARNINGS_FLAGS)


class RKIntegratorConfig:
    def __init__(self, dt, tableau, *, adaptive=False, **kwargs):
        if not tableau.data.is_adaptive and adaptive:
            raise ValueError(
                "Cannot set `adaptive=True` for a non-adaptive integrator."
            )

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
            self.tableau,
            self.dt,
            self.adaptive,
            f", **kwargs={self.kwargs}" if self.kwargs else "",
        )
