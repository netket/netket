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

from netket.utils.config_flags import config

from netket.utils.moduletools import (
    _hide_submodules,
    rename_class,
    auto_export as _auto_export,
)
from netket.utils.version_check import module_version

# error if old dependencies are detected
from netket.utils import _dependencies_check

from netket.utils import dispatch
from netket.utils import struct
from netket.utils import numbers
from netket.utils import types
from netket.utils import float
from netket.utils import optional_deps
from netket.utils import timing
from netket.utils import display

from netket.utils.array import HashableArray, array_in
from netket.utils.partial import HashablePartial
from netket.utils.jax import get_afun_if_module, wrap_afun, wrap_to_support_scalar
from netket.utils.seed import random_seed
from netket.utils.summation import KahanSum

from netket.utils.holomorphic import is_probably_holomorphic

from netket.utils.deprecation import (
    warn_deprecation,
    deprecated,
    deprecated_new_name,
)

from netket.utils.model_frameworks import maybe_wrap_module

from netket.utils.history import History, accum_in_tree, accum_histories_in_tree

from netket.utils.static_range import StaticRange

# TODO: remove this import as it is deprecated
from netket.utils import mpi

_hide_submodules(
    __name__,
    remove_self=False,
    ignore=[
        "numbers",
        "types",
        "float",
        "dispatch",
        "errors",
        "timing",
        "display",
    ],
)
