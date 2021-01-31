# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Optional, Tuple, Any, Callable, Iterable

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn

from netket.hilbert import AbstractHilbert
from netket.graph import AbstractGraph

from netket import nn as nknn
from netket.nn.initializers import lecun_normal, variance_scaling, zeros


PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any


class RBM(nn.Module):
    dtype: Any = np.float64
    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    use_bias: bool = True
    use_visible: bool = True

    visible_bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, x):
        if self.use_visible:
            v_bias = self.param(
                "visible_bias", self.visible_bias_init, (x.shape[-1]), self.dtype
            )
            out_bias = jnp.dot(x, v_bias)

        x = nknn.Dense(
            name="Dense",
            features=self.alpha * x.shape[-1],
            dtype=self.dtype,
            use_bias=self.use_bias,
        )(x)
        x = self.activation(x)
        return jnp.sum(x, axis=-1) + out_bias


class RBMModPhase(nn.Module):
    dtype: Any = np.float64
    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        re = nknn.Dense(
            features=self.alpha * x.shape[-1], dtype=self.dtype, use_bias=self.use_bias
        )(x)
        re = self.activation(re)
        re = jnp.sum(re, axis=-1)

        im = nknn.Dense(
            features=self.alpha * x.shape[-1], dtype=self.dtype, use_bias=self.use_bias
        )(x)
        im = self.activation(im)
        im = jnp.sum(im, axis=-1)

        return mod + 1j * im
