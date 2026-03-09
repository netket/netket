# Copyright 2021 The NetKet Authors - All rights reserved.
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

from netket._src.callbacks.base import AbstractCallback, StopRun
from netket.utils import struct


class _StopAtTime(AbstractCallback):
    """
    Internal callback: stops the run loop once ``driver.t >= T_final``.

    Used by :class:`~netket.driver.AbstractDynamicsDriver` to implement
    ``run(T_final: float)`` on top of the standard integer-step loop.
    Not part of the public API.
    """

    T_final: float = struct.field(pytree_node=False, serialize=False)

    def __init__(self, T_final: float):
        self.T_final = T_final

    def on_step_end(self, step, log_data, driver):
        if driver.t >= self.T_final:
            raise StopRun(f"Reached t={driver.t:.6g} >= T_final={self.T_final:.6g}")
