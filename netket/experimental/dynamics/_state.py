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

from netket.utils import struct, KahanSum
from netket.utils.types import Array

import jax
import jax.numpy as jnp


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


@struct.dataclass
class IntegratorState(struct.Pytree):
    r"""
    Dataclass containing the state of an ODE solver.
    In particular, it stores the current state of the system, former usefull values
    and information about integration (number of step, errors, etc)
    """

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
    last_norm: float | None = None
    """Solution norm at previous time step."""
    last_scaled_error: float | None = None
    """Error of the TDVP integrator at the last time step."""
    flags: SolverFlags = SolverFlags.INFO_STEP_ACCEPTED
    """Flags containing information on the solver state."""

    def __init__(
        self,
        dt: float,
        y,
        t,
        *,
        step_no=0,
        step_no_total=0,
        last_norm=None,
        last_scaled_error=None,
        flags=SolverFlags.INFO_STEP_ACCEPTED,
    ):
        step_dtype = jnp.int64 if jax.config.x64_enabled else jnp.int32
        err_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32

        self.step_no = jnp.asarray(step_no, dtype=step_dtype)
        self.step_no_total = jnp.asarray(step_no_total, dtype=step_dtype)

        if not isinstance(t, KahanSum):
            t = KahanSum(t)
        self.t = t
        if not isinstance(dt, jax.Array):
            dt = jnp.asarray(dt)
        self.dt = dt

        # To avoid creating problems, just convert to array if the leaves
        # in the parameters are not jax arrays already.
        def _maybe_asarray(x):
            if isinstance(x, jax.Array):
                return x
            else:
                return jnp.asarray(x)

        self.y = jax.tree_util.tree_map(_maybe_asarray, y)

        if last_norm is not None:
            last_norm = jnp.asarray(last_norm, dtype=err_dtype)
        self.last_norm = last_norm
        if last_scaled_error is not None:
            last_scaled_error = jnp.asarray(last_scaled_error, dtype=err_dtype)
        self.last_scaled_error = last_scaled_error

        # Todo: if making SolverFlag a proper pytree this can be restored.
        # if not isinstance(flags, SolverFlags):
        #    raise TypeError(f"flags must be SolverFlags but got {type(flags)} : {flag}")
        self.flags = flags

    def __repr__(self):
        try:
            dt = f"{self.dt:.2e}"
            last_norm = f", {self.last_norm:.2e}" if self.last_norm is not None else ""
            accepted = f", {'A' if self.accepted else 'R'}"
        except (ValueError, TypeError):
            dt = f"{self.dt}"
            last_norm = f"{self.last_norm}"
            accepted = f"{SolverFlags.INFO_STEP_ACCEPTED}"

        return f"IntegratorState(step_no(total)={self.step_no}({self.step_no_total}), t={self.t.value}, dt={dt}{last_norm}{accepted})"

    @property
    def accepted(self):
        """Boolean indicating whether the last step was accepted."""
        return SolverFlags.INFO_STEP_ACCEPTED & self.flags != 0
