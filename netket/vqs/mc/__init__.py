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

"""
The mc module contains everything that is needed to define the variational
MCState and MCMixedState (two particular kind of VariationalState). This
module also contains all the relevant definitions needed to compute expectation
values and gradients using NetKet-default operators.

In `common.py` some common methods are defined. Those are used both by MCState
and MCMixedState to more easily define expect/expect_and_grad methods.

In `kernels.py` the local-expect kernels are defined that are then passed to the
expect/expect_and_grad methods.
"""

from .common import check_hilbert, get_local_kernel_arguments, get_local_kernel

from .mc_state import MCState
from .mc_mixed_state import MCMixedState
