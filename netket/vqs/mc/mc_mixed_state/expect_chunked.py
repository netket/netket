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

from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractSuperOperator,
    DiscreteOperator,
    Squared,
)

from netket.vqs.mc import kernels, get_local_kernel

from .state import MCMixedState


# Dispatches to select what expect-kernel to use
@dispatch
def get_local_kernel(  # noqa: F811
    vstate: MCMixedState, Ô: Squared[AbstractSuperOperator], chunk_size: int
):
    return kernels.local_value_squared_kernel_chunked


@dispatch
def get_local_kernel(  # noqa: F811
    vstate: MCMixedState, Ô: DiscreteOperator, chunk_size: int
):
    return kernels.local_value_op_op_cost_chunked
