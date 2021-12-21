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

import os
import pytest
import numpy as np

import scipy.integrate as sci

from netket.experimental.dynamics import Euler, Heun, Midpoint, RK4, RK12, RK23, RK45

explicit_fixed_step_solvers = {
    "Euler": Euler,
    "Heun": Heun,
    "Midpoint": Midpoint,
    "RK4": RK4,
}

explicit_adaptive_solvers = {
    "RK12": RK12,
    "RK23": RK23,
    "RK45": RK45,
}


@pytest.mark.parametrize("solver", explicit_fixed_step_solvers)
def test_ode_solver(solver):
    if solver == "Euler":  # first order

        def ode(_t, _x, **_):
            return 1.0

    else:  # quadratic function for higher-order solvers

        def ode(t, _x, **_):
            return t

    solver = explicit_fixed_step_solvers[solver]

    y0 = np.array([1.0])
    times = np.linspace(0, 2, 10, endpoint=False)

    sol = sci.solve_ivp(ode, (0.0, 2.0), y0, t_eval=times)
    y_ref = sol.y[0]

    solv = solver(dt=0.2)(ode, 0.0, y0)

    t = []
    y_t = []
    for _ in range(10):
        t.append(solv.t)
        y_t.append(solv.y)
        solv.step()
    y_t = np.asarray(y_t)

    np.testing.assert_allclose(t, times)
    np.testing.assert_allclose(y_t[:, 0], y_ref)


@pytest.mark.parametrize("solver", explicit_adaptive_solvers)
def test_adaptive_solver(solver):
    solver = explicit_adaptive_solvers[solver]

    tol = 1e-7

    def ode(t, x, **_):
        return -t * x

    y0 = np.array([1.0])
    solv = solver(dt=0.2, adaptive=True, atol=0.0, rtol=tol)(ode, 0.0, y0)

    t = []
    y_t = []
    last_step = -1
    while solv.t <= 2.0:
        print(solv._rkstate)
        if solv._rkstate.step_no != last_step:
            last_step = solv._rkstate.step_no
            t.append(solv.t)
            y_t.append(solv.y)
        solv.step()
    y_t = np.asarray(y_t)

    print(t)
    sol = sci.solve_ivp(
        ode, (0.0, 2.0), y0, t_eval=t, atol=0.0, rtol=tol, method="RK45"
    )
    y_ref = sol.y[0]

    np.testing.assert_allclose(y_t[:, 0], y_ref, rtol=1e-5)
