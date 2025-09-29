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

__all__ = [
    "callbacks",
    "errors",
    "exact",
    "experimental",
    "graph",
    "hilbert",
    "logging",
    "models",
    "nn",
    "operator",
    "optimizer",
    "sampler",
    "stats",
    "utils",
    "vqs",
    "nn",
    "symmetry",
    "cite",
    "config",
]

from ._version import version as __version__  # noqa: F401

from .utils import config

from . import utils
from . import errors

from . import jax
from . import stats

from . import graph
from . import hilbert

from . import nn

from . import (
    exact,
    callbacks,
    logging,
    operator,
    models,
    sampler,
    vqs,
    optimizer,
    symmetry,
)

from . import experimental


# Main applications
from .driver import VMC
from .driver import SteadyState

# Citation system
from netket.utils.citations import cite

# Show tips if in interactive mode
from netket.utils._tips import show_random_tip

show_random_tip()
del show_random_tip
