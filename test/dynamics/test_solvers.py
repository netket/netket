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

import pytest
import numpy as np

from netket.dynamics import Euler, Heun, Midpoint, RK4

explicit_fixed_step_solvers = {
    "Euler": Euler,
    "Heun": Heun,
    "Midpoint": Midpoint,
    "RK4": RK4,
}


@pytest.mark.parametrize("solver", explicit_fixed_step_solvers.keys())
def test_ode_solver(solver):
    solver = explicit_fixed_step_solvers[solver]

    def ode(t, x):
        return t * x

    def solution(t):
        return 2.0 * np.exp(t**2 / 2.0)

    times = np.linspace(0, 2, 100)
    y_ref = solution(times)
    
    y0 = [2.0]
    solv = solver(dt=0.02)(ode, (0.0, 2.0), y0)

    y_t = []
    for i in range(200):
        y_t.append(solv.y)
        solv.step()
    
    np.testing.assert_allclose(y_t, y_ref)
