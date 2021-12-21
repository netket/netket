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

from .base import VariationalState, VariationalMixedState, expect, expect_and_grad

from .mc import MCState, MCMixedState, get_local_kernel_arguments, get_local_kernel
from .exact import ExactState

# TODO: this is deprecated in favour of netket.experimental.vqs
# eventually remove this file and import
from . import experimental

from netket.utils import _hide_submodules

_hide_submodules(__name__, ignore=["experimental"], hide_folder=["mc"])
