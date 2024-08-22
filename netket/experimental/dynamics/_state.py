from enum import IntFlag, auto
from typing import Optional

import netket as nk
from netket.utils.struct import dataclass
from netket.utils.types import Array


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


@dataclass(_frozen=False)
class IntegratorState:
    r"""
    Dataclass containing the state of an ODE solver.
    In particular, it stores the current state of the system, former usefull values
    and information about integration (number of step, errors, etc)
    """

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

        return f"IntegratorState(step_no(total)={self.step_no}({self.step_no_total}), t={self.t.value}, dt={dt}{last_norm}{accepted})"

    @property
    def accepted(self):
        """Boolean indicating whether the last step was accepted."""
        return SolverFlags.INFO_STEP_ACCEPTED & self.flags != 0
