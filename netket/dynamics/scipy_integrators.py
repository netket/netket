import scipy
import numpy as np
from scipy.integrate import OdeSolver, DenseOutput
from warnings import warn


def validate_first_step(first_step, t0, t_bound):
    """Assert that first_step is valid and return it."""
    if first_step <= 0:
        raise ValueError("`first_step` must be positive.")
    if first_step > np.abs(t_bound - t0):
        raise ValueError("`first_step` exceeds bounds.")
    return first_step


def _warn_extraneous(extraneous):
    """Display a warning for extraneous keyword arguments.

    The initializer of each solver class is expected to collect keyword
    arguments that it doesn't understand and warn about them. This function
    prints a warning for each key in the supplied dictionary.

    Parameters
    ----------
    extraneous : dict
        Extraneous keyword arguments
    """
    if extraneous:
        warn(
            "The following arguments have no effect for a chosen solver: {}.".format(
                ", ".join("`{}`".format(x) for x in extraneous)
            )
        )


class EulerSolver(OdeSolver):
    """Base class for explicit Runge-Kutta methods."""

    B = np.array(
        [
            1,
        ]
    )
    order = NotImplemented
    error_estimator_order = NotImplemented
    n_stages = 1

    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        step=None,
        vectorized=False,
        first_step=None,
        **extraneous,
    ):
        _warn_extraneous(extraneous)
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)
        self.y_old = None
        self.f = self.fun(self.t, self.y)

        self.K = np.empty((self.n_stages + 1, self.n), dtype=self.y.dtype)

        if step is None and first_step is None:
            raise ValueError("Must pass step or first_step as dt")
        elif step is None:
            step = first_step
        elif step is not None and first_step is not None:
            raise ValueError("Only one among step or first step must be passed")

        self.h_abs = validate_first_step(step, t0, t_bound)

    def _step_impl(self):
        t = self.t
        y = self.y

        h_abs = self.h_abs

        h = h_abs * self.direction
        t_new = t + h

        if self.direction * (t_new - self.t_bound) > 0:
            t_new = self.t_bound

        h = t_new - t
        h_abs = np.abs(h)

        y_new, f_new = self._step_fun(self.fun, t, y, self.f, h, self.B, self.K)

        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _step_fun(self, fun, t, y, f, h, B, K):
        K[0] = f

        y_new = y + h * np.dot(K[:1].T, B)
        f_new = fun(t + h, y_new)

        K[-1] = f_new

        return y_new, f_new

    def _dense_output_impl(self):
        return NADenseOutput(self.t_old, self.t, self.y_old, self.y)


class NADenseOutput(DenseOutput):
    def __init__(self, t_old, t, y_old, y):
        super().__init__(t_old, t)
        self.h = t - t_old
        self.y = y
        self.order = 1
        self.y_old = y_old

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h
        dy = (self.y - self.y_old) / self.h
        y = self.y_old + dy * x

        return y
