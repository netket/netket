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

from . import jax

from . import (
    exact,
    callbacks,
    graph,
    hilbert,
    logging,
    operator,
    optim,
    sampler,
    nn,
    stats,
    utils,
    _vmc,
)

# Main applications
from ._vmc import Vmc

# from ._qsr import Qsr

from .vmc_common import (
    tree_map as _tree_map,
    trees2_map as _trees2_map,
)

# deprecations
optimizer = optim
