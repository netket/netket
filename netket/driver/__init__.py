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

from netket._src.driver.abstract_variational_driver import (
    AbstractDriver as AbstractDriver,
)
from netket._src.driver.abstract_optimization_driver import (
    AbstractOptimizationDriver as AbstractOptimizationDriver,
)
from netket._src.driver.abstract_dynamics_driver import (
    AbstractDynamicsDriver as AbstractDynamicsDriver,
)
from netket.driver.vmc import VMC as VMC
from netket.driver.steady_state import SteadyState as SteadyState

from netket._src.driver.infidelity_sr import Infidelity_SR as Infidelity_SR
from netket._src.driver.vmc_sr import VMC_SR as VMC_SR

from .auto_chunk import find_chunk_size

from netket.utils import _hide_submodules


# TODO: At some point deprecate the old AbstractVariationalDriver name.
from netket._src.driver.abstract_optimization_driver import (
    AbstractOptimizationDriver as AbstractVariationalDriver,
)

# from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

# _deprecations = {
#     # March 2026 — AbstractVariationalDriver renamed to AbstractDriver;
#     # the old name is now an alias for AbstractOptimizationDriver for backward compat.
#     "AbstractVariationalDriver": (
#         "netket.driver.AbstractVariationalDriver has been renamed. "
#         "Use netket.driver.AbstractOptimizationDriver to write optimization drivers, "
#         "netket.driver.AbstractDynamicsDriver for dynamics drivers, or "
#         "netket.driver.AbstractDriver for the minimal base class.",
#         AbstractOptimizationDriver,
#     ),
# }

# __getattr__ = _deprecation_getattr(__name__, _deprecations)
# del _deprecation_getattr

_hide_submodules(__name__)
