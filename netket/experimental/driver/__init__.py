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

from .tdvp import TDVP
from .tdvp_schmitt import TDVPSchmitt
from .vmc_sr import VMC_SR, VMC_SRt as _VMC_SRt_deprecated
from .infidelity_sr import Infidelity_SR

_deprecations = {
    # May 2024, NetKet 3.12
    "VMC_SRt": (
        "netket.experimental.driver.VMC_SRt is deprecated: use the new SR driver "
        "netket.experimental.driver.VMC_SR(..., use_ntk=True) (netket >= 3.17)",
        _VMC_SRt_deprecated,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _hide_submodules

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_hide_submodules(__name__)

del _deprecation_getattr
