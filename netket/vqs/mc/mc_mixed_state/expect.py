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

from functools import partial
from typing import Callable

import numpy as np

from netket.utils.dispatch import dispatch

from netket.operator import (
    DiscreteOperator,
    AbstractSuperOperator,
    Squared,
)

from netket.vqs.mc import kernels, check_hilbert, get_configs, get_fun

from .state import MCMixedState


@dispatch
def get_configs(vstate: MCMixedState, Ô: DiscreteOperator):
    check_hilbert(vstate.diagonal.hilbert, Ô.hilbert)

    σ = vstate.diagonal.samples
    σp, mels = Ô.get_conn_padded(σ)
    return σ, σp, mels


@dispatch
def get_fun(vstate: MCMixedState, Ô: DiscreteOperator):
    return kernels.local_value_op_op_cost


@dispatch
def get_configs(vstate: MCMixedState, Ô: Squared[AbstractSuperOperator]):
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.parent.get_conn_padded(σ)
    return σ, σp, mels


@dispatch
def get_fun(vstate: MCMixedState, Ô: Squared[AbstractSuperOperator]):
    return kernels.local_value_squared_kernel
