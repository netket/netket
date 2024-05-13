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

from .metropolis import MetropolisParticleExchange


# Deprecated bindings from stabilisaation
from netket.sampler import (
    MetropolisPtSampler as _deprecated_MetropolisPtSampler,
    MetropolisLocalPt as _deprecated_MetropolisLocalPtSampler,
    MetropolisExchangePt as _deprecated_MetropolisExchangePtSampler,
)

_deprecations = {
    # May 2024, NetKet 3.12
    "MetropolisPtSampler": (
        "netket.experimental.sampler.MetropolisPtSampler is deprecated: use "
        "netket.sampler.MetropolisPtSampler (netket >= 3.12)",
        _deprecated_MetropolisPtSampler,
    ),
    "MetropolisLocalPt": (
        "netket.experimental.sampler.MetropolisLocalPt is deprecated: use "
        "netket.sampler.MetropolisLocalPt (netket >= 3.12)",
        _deprecated_MetropolisLocalPtSampler,
    ),
    "MetropolisExchangePt": (
        "netket.experimental.sampler.MetropolisExchangePt is deprecated: use "
        "netket.sampler.MetropolisExchangePt (netket >= 3.12)",
        _deprecated_MetropolisExchangePtSampler,
    ),
}

from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr
from netket.utils import _hide_submodules

__getattr__ = _deprecation_getattr(__name__, _deprecations)
_hide_submodules(__name__)

del _deprecation_getattr
