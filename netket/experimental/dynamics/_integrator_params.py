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


from netket.utils import struct

from ._utils import LimitsDType


@struct.dataclass
class IntegratorParameters(struct.Pytree):
    dt: float
    """The initial time-step size of the integrator."""

    atol: float
    """The tolerance for the absolute error on the solution."""
    rtol: float
    """The tolerance for the relative error on the solution."""

    dt_limits: LimitsDType | None = struct.field(pytree_node=False)
    """The extremal accepted values for the time-step size `dt`."""

    def __init__(
        self,
        dt: float,
        atol: float = 0.0,
        rtol: float = 1e-7,
        dt_limits: LimitsDType | None = None,
    ):
        r"""
        Args:
            dt: The initial time-step size of the integrator.
            atol: The tolerance for the absolute error on the solution.
                defaults to :code:`0.0`.
            rtol: The tolerance for the relative error on the solution.
                defaults to :code:`1e-7`.
            dt_limits: The extremal accepted values for the time-step size `dt`.
                defaults to :code:`(None, 10 * dt)`.
        """
        self.dt = dt

        self.atol = atol
        self.rtol = rtol
        if dt_limits is None:
            dt_limits = (None, 10 * dt)
        self.dt_limits = dt_limits
