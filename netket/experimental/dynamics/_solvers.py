# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.integrate as integrate

from ._scipy_integrators import EulerSolver


# List of integrated methods
METHODS = {
    "RK23": integrate.RK23,
    "RK45": integrate.RK45,
    "DOP853": integrate.DOP853,
    "RADAU": integrate.Radau,
    "BDF": integrate.BDF,
    "LSODA": integrate.LSODA,
    "EULER": EulerSolver,
}


def build_solver(METHOD, adaptive: int = False, *args, dt=None, **kwargs):

    if isinstance(METHOD, str):
        METHOD = METHOD.upper()
        if METHOD not in METHODS:
            raise ValueError("Unknown method")
        else:
            solver = METHODS[METHOD]

    if adaptive is False and dt is None:
        raise ValueError("If adaptivity is turned off, must specify dt.")

    def _solver(fun, tspan, y0):
        if isinstance(tspan, tuple):
            t0 = tspan[0]
            tend = tspan[-1]
        else:
            t0 = tspan
            tend = np.inf

        if adaptive is False:
            return solver(
                fun,
                t0=t0,
                y0=y0,
                t_bound=tend,
                max_step=dt,
                first_step=dt,
                rtol=np.inf,
                atol=np.inf,
            )
        else:
            return solver(fun, t0=t0, y0=y0, t_bound=tend, **kwargs)

    return _solver


def Euler(*args, **kwargs):
    """
    Euler solver

    Args:
        dt: the timestep
    """
    return build_solver("EULER", False, *args, **kwargs)


def RK23(*args, **kwargs):
    """
    RK23 solver

    Args:
        adaptive: Whever to be adaptive or not
        dt: the timestep (if non adaptive) or the initial timestep (if adaptive).
    """
    return build_solver("RK23", *args, **kwargs)


def RK45(*args, **kwargs):
    """
    RK45 solver

    Args:
        adaptive: Whever to be adaptive or not
        dt: the timestep (if non adaptive) or the initial timestep (if adaptive).
    """
    return build_solver("RK45", *args, **kwargs)


def DOP853(*args, **kwargs):
    """
    DOP853 solver

    Args:
        adaptive: Whever to be adaptive or not
        dt: the timestep (if non adaptive) or the initial timestep (if adaptive).
    """
    return build_solver("DOP853", *args, **kwargs)


def RADAU(*args, **kwargs):
    """
    RADAU solver

    Args:
        adaptive: Whever to be adaptive or not
        dt: the timestep (if non adaptive) or the initial timestep (if adaptive).
    """
    return build_solver("RADAU", True, *args, **kwargs)


def BDF(*args, **kwargs):
    """
    BDF solver

    Args:
        adaptive: Whever to be adaptive or not
        dt: the timestep (if non adaptive) or the initial timestep (if adaptive).
    """
    return build_solver("BDF", True, *args, **kwargs)


def LSODA(*args, **kwargs):
    """
    LSODA solver

    Args:
        adaptive: Whever to be adaptive or not
        dt: the timestep (if non adaptive) or the initial timestep (if adaptive).
    """
    return build_solver("LSODA", True, *args, **kwargs)
