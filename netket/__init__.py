# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

# enable x64 on jax
# must be done at 0 startup.
from jax.config import config

config.update("jax_enable_x64", True)
del config

__all__ = [
    "exact",
    "graph",
    "callbacks",
    "hilbert",
    "operator",
    "optimizer",
    "sampler",
    "stats",
    "utils",
    "variational",
    "nn",
]

from . import legacy

from . import (
    hilbert,
    exact,
    callbacks,
    graph,
    logging,
    operator,
    optimizer,
    models,
    sampler,
    jax,
    nn,
    stats,
    utils,
    variational,
)

# Main applications
from .drivers import Vmc
from .drivers import SteadyState

# from .drivers import Qsr

# deprecations
optim = optimizer

from .version import __version__
