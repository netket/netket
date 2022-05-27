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

# enable x64 on jax
# must be done at startup.
from jax.config import config

config.update("jax_enable_x64", True)
del config

from ._version import version as __version__  # noqa: F401

from . import utils
from .utils import config, deprecated_new_name as _deprecated

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
    "vqs",
    "nn",
]

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
)


# Main applications
from .driver import VMC
from .driver import SteadyState


# Deprecations
@_deprecated("VMC")
def Vmc(*args, **kwargs):
    return VMC(*args, **kwargs)


from . import variational
from . import optim
