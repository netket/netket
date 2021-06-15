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

from .config_flags import config

from . import dispatch
from . import struct
from . import numbers
from . import types
from . import float

from .array import HashableArray
from .jax import get_afun_if_module, wrap_afun
from . import mpi
from .optional_deps import torch_available, tensorboard_available, backpack_available
from .seed import random_seed

from .deprecation import warn_deprecation, deprecated, deprecated_new_name, wraps_legacy
from .moduletools import _hide_submodules, rename_class

from .model_frameworks import maybe_wrap_module

from .history import History, accum_in_tree, accum_histories_in_tree

# TODO: legacy -> to be removed
jax_available = True
flax_available = True
mpi4jax_available = mpi.mpi_available

_hide_submodules(
    __name__, remove_self=False, ignore=["numbers", "types", "float", "dispatch"]
)
