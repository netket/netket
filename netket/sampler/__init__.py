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

from netket import config as _config

from .base import (
    Sampler,
    SamplerState,
    sampler_state,
    reset,
    sample_next,
    sample,
    samples,
)

from .exact import ExactSampler
from .metropolis import (
    MetropolisSampler,
    MetropolisLocal,
    MetropolisExchange,
    MetropolisRule,
    MetropolisSamplerState,
    MetropolisHamiltonian,
)

from .metropolis_numpy import (
    MetropolisSamplerNumpy,
    MetropolisLocalNumpy,
    MetropolisHamiltonianNumpy,
    MetropolisCustomNumpy,
)

from .metropolis_pt import (
    MetropolisPtSampler,
    MetropolisLocalPt,
    MetropolisExchangePt,
)

from . import rules

# Shorthand
Metropolis = MetropolisSampler
MetropolisPt = MetropolisPtSampler
MetropolisNumpy = MetropolisSamplerNumpy

# Replacements for effficiency
# MetropolisHamiltonian = MetropolisHamiltonianNumpy
MetropolisCustom = MetropolisCustomNumpy

from netket.utils import _hide_submodules

_hide_submodules(__name__)
