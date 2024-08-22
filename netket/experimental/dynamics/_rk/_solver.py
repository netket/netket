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

from . import _tableau as rkt
from .._integrator import IntegratorConfig
from .._structures import (
    append_docstring,
    args_adaptive_docstring,
    args_fixed_dt_docstring,
)


@append_docstring(args_fixed_dt_docstring)
def Euler(dt):
    r"""
    The canonical first-order forward Euler method. Fixed timestep only.

    """
    return IntegratorConfig(dt, tableau=rkt.bt_feuler)


@append_docstring(args_fixed_dt_docstring)
def Midpoint(dt):
    r"""
    The second order midpoint method. Fixed timestep only.

    """
    return IntegratorConfig(dt, tableau=rkt.bt_midpoint)


@append_docstring(args_fixed_dt_docstring)
def Heun(dt):
    r"""
    The second order Heun's method. Fixed timestep only.

    """
    return IntegratorConfig(dt, tableau=rkt.bt_heun)


@append_docstring(args_fixed_dt_docstring)
def RK4(dt):
    r"""
    The canonical Runge-Kutta Order 4 method. Fixed timestep only.

    """
    return IntegratorConfig(dt, tableau=rkt.bt_rk4)


@append_docstring(args_adaptive_docstring)
def RK12(dt, **kwargs):
    r"""
    The second order Heun's method. Uses embedded Euler method for adaptivity.
    Also known as Heun-Euler method.

    """
    return IntegratorConfig(dt, tableau=rkt.bt_rk12, **kwargs)


@append_docstring(args_adaptive_docstring)
def RK23(dt, **kwargs):
    r"""
    2nd order adaptive solver with 3rd order error control,
    using the Bogacki–Shampine coefficients

    """
    return IntegratorConfig(dt, tableau=rkt.bt_rk23, **kwargs)


@append_docstring(args_adaptive_docstring)
def RK45(dt, **kwargs):
    r"""
    Dormand-Prince's 5/4 Runge-Kutta method. (free 4th order interpolant).

    """
    return IntegratorConfig(dt, tableau=rkt.bt_rk4_dopri, **kwargs)
