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

from . import _rk_tableau as rkt
from ._rk_solver_structures import RKIntegratorConfig

args_fixed_dt_docstring = """
    Args:
        dt: Timestep (floating-point number).
"""

args_adaptive_docstring = """
    Args:
        dt: Timestep (floating-point number). When :code:`adaptive==False` this value
            is never changed, when :code:`adaptive == True` this is the initial timestep.
        adaptive: Whether to use adaptive timestepping (Defaults to False).
            Not all integrators support adaptive timestepping.
        atol: Maximum absolute error at every time-step during adaptive timestepping.
            A larger value will lead to larger timestep. This option is ignored if
            `adaptive=False`. A value of 0 means it is ignored. Note that the `norm` used
            to compute the error can be  changed in the :ref:`netket.experimental.TDVP`
            driver. (Defaults to 0).
        rtol: Maximum relative error at every time-step during adaptive timestepping.
            A larger value will lead to larger timestep. This option is ignored if
            `adaptive=False`. Note that the `norm` used to compute the error can be
            changed in the :ref:`netket.experimental.TDVP` driver. (Defaults to 1e-7)
        dt_limits: A length-2 tuple of minimum and maximum timesteps considered by
            adaptive time-stepping. A value of None signals that there is no bound.
            Defaults to :code:`(None, 10*dt)`.
"""


def append_docstring(doc):
    """
    Decorator that appends the string `doc` to the decorated function.

    This is needed here because docstrings cannot be f-strings or manipulated strings.
    """

    def _append_docstring(fun):
        fun.__doc__ = fun.__doc__ + doc
        return fun

    return _append_docstring


@append_docstring(args_fixed_dt_docstring)
def Euler(dt):
    r"""
    The canonical first-order forward Euler method. Fixed timestep only.

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_feuler)


@append_docstring(args_fixed_dt_docstring)
def Midpoint(dt):
    r"""
    The second order midpoint method. Fixed timestep only.

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_midpoint)


@append_docstring(args_fixed_dt_docstring)
def Heun(dt):
    r"""
    The second order Heun's method. Fixed timestep only.

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_heun)


@append_docstring(args_fixed_dt_docstring)
def RK4(dt):
    r"""
    The canonical Runge-Kutta Order 4 method. Fixed timestep only.

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_rk4)


@append_docstring(args_adaptive_docstring)
def RK12(dt, **kwargs):
    r"""
    The second order Heun's method. Uses embedded Euler method for adaptivity.
    Also known as Heun-Euler method.

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_rk12, **kwargs)


@append_docstring(args_adaptive_docstring)
def RK23(dt, **kwargs):
    r"""
    2nd order adaptive solver with 3rd order error control,
    using the Bogacki–Shampine coefficients

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_rk23, **kwargs)


@append_docstring(args_adaptive_docstring)
def RK45(dt, **kwargs):
    r"""
    Dormand-Prince's 5/4 Runge-Kutta method. (free 4th order interpolant).

    """
    return RKIntegratorConfig(dt, tableau=rkt.bt_rk4_dopri, **kwargs)
