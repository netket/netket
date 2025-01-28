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

from .base import (
    Sampler,
    SamplerState,
)

from .exact import ExactSampler

from .metropolis import (
    MetropolisSampler,
    MetropolisLocal,
    MetropolisExchange,
    MetropolisRule,
    MetropolisSamplerState,
    MetropolisHamiltonian,
    MetropolisGaussian,
    MetropolisAdjustedLangevin,
    MetropolisFermionHop,
)

from .parallel_tempering import (
    ParallelTemperingSampler,
    ParallelTemperingLocal,
    ParallelTemperingExchange,
    ParallelTemperingHamiltonian,
)

from .metropolis_numpy import (
    MetropolisSamplerNumpy,
    MetropolisLocalNumpy,
    MetropolisHamiltonianNumpy,
    MetropolisCustomNumpy,
)

from .autoreg import ARDirectSampler

from . import rules

# Shorthand
Metropolis = MetropolisSampler
MetropolisNumpy = MetropolisSamplerNumpy

# Replacements for efficiency
# MetropolisHamiltonian = MetropolisHamiltonianNumpy
MetropolisCustom = MetropolisCustomNumpy

from netket.utils import _hide_submodules

_hide_submodules(__name__)
