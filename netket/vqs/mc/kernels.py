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
This module implements some common kernels used by MCState and MCMixedState.
"""

from typing import Any

import jax
from jax import numpy as jnp


def local_value_kernel(logpsi, pars, σ, σp, mel):
    """
    local_value kernel for MCState and generic operators
    """
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_squared_kernel(logpsi, pars, σ, σp, mel):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return jnp.abs(local_value_kernel(logpsi, pars, σ, σp, mel)) ** 2


def local_value_op_op_cost(logpsi, pars, σ, σp, mel):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σ_σp = jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(σp, σ)
    σ_σ = jnp.hstack((σ, σ))
    return jnp.sum(mel * jnp.exp(logpsi(pars, σ_σp) - logpsi(pars, σ_σ)))
