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

import flax
from flax import linen as nn
from flax import struct
import jax
from jax import numpy as jnp

from netket.hilbert import AbstractHilbert
from typing import Union, Tuple, Optional, Any, Callable

from jax.random import PRNGKey

from flax.linen import Module


class JaxWrapModule(nn.Module):
    """
    Wrapper for Jax bare modules made by a init_fun and apply_fun
    """

    init_fun: Callable
    apply_fun: Callable

    @nn.compact
    def __call__(self, x):
        if jnp.ndim(x) == 1:
            x = jnp.atleast_1d(x)
        pars = self.param(
            "jax", lambda rng, shape: self.init_fun(rng, shape)[1], x.shape
        )

        return self.apply_fun(pars, x)


def wrap_jax(mod):
    """
    Wrap a Jax module into a flax module
    """
    return JaxWrapModule(*mod)
