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

from .metropolis_pmap import MetropolisSamplerPmap


# Deprecated bindings from stabilisaation
from netket.sampler import (
    ParallelTemperingSampler as _deprecated_MetropolisPtSampler,
    ParallelTemperingLocal as _deprecated_MetropolisLocalPtSampler,
    ParallelTemperingExchange as _deprecated_MetropolisExchangePtSampler,
    MetropolisFermionHop as _deprecated_MetropolisParticleExchange,
)

_deprecations = {
    # May 2024, NetKet 3.12
    "MetropolisPtSampler": (
        "netket.experimental.sampler.MetropolisPtSampler is deprecated: use "
        "netket.sampler.ParallelTemperingSampler (netket >= 3.12)",
        _deprecated_MetropolisPtSampler,
    ),
    "MetropolisLocalPt": (
        "netket.experimental.sampler.MetropolisLocalPt is deprecated: use "
        "netket.sampler.ParallelTemperingLocal (netket >= 3.12)",
        _deprecated_MetropolisLocalPtSampler,
    ),
    "MetropolisExchangePt": (
        "netket.experimental.sampler.MetropolisExchangePt is deprecated: use "
        "netket.sampler.ParallelTemperingExchange (netket >= 3.12)",
        _deprecated_MetropolisExchangePtSampler,
    ),
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
