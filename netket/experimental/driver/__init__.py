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

from netket.experimental.driver.tdvp import TDVP
from netket.experimental.driver.tdvp_schmitt import TDVPSchmitt

from netket._src.driver.infidelity_sr import Infidelity_SR as _InfidelitySR_deprecated

_deprecations = {
    # March 2026 NetKet 3.22
    "Infidelity_SR": (
        "netket.experimental.driver.Infidelity_SR is now stable: use it from "
        "netket.driver.Infidelity_SR (netket >= 3.22)",
        _InfidelitySR_deprecated,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _hide_submodules

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_hide_submodules(__name__)

del _deprecation_getattr
