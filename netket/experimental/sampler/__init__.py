# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

from . import rules


# Deprecated bindings from stabilisaation
from netket.sampler import (
    MetropolisFermionHop as _deprecated_MetropolisParticleExchange,
)

_deprecations = {
    # June 2024, NetKet 3.13
    "MetropolisParticleExchange": (
        "netket.experimental.sampler.MetropolisParticleExchange is deprecated: use "
        "netket.sampler.MetropolisFermionHop (netket >= 3.13)",
        _deprecated_MetropolisParticleExchange,
    ),
}


from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _hide_submodules

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_hide_submodules(__name__)

del _deprecation_getattr
