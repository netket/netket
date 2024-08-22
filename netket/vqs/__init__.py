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
    VariationalState,
    VariationalMixedState,
    expect,
    expect_and_grad,
    expect_and_forces,
)

from .mc import MCState, MCMixedState, get_local_kernel_arguments, get_local_kernel
from .full_summ import FullSumState

_deprecations = {
    # May 2023
    "ExactState": (
        "netket.vqs.ExactState is deprecated: use netket.vqs.FullSumState (netket >= 3.12)",
        FullSumState,
    ),
}

from netket.utils import _hide_submodules
from netket.utils.deprecation import deprecation_getattr as _deprecation_getattr

_hide_submodules(__name__, ignore=["experimental"], hide_folder=["mc"])
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
