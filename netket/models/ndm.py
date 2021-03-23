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
from netket.nn.initializers import lecun_normal, variance_scaling, zeros, normal

from .rbm import RBM

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?
Array = Any

default_kernel_init = normal(stddev=0.001)


class PureRBM(nn.Module):
    """
    Encodes the pure state |ψ><ψ|, because it acts on row and column
    indices with the same RBM.
    """

    dtype: Any = np.float64
    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    use_hidden_bias: bool = True
    use_visible_bias: bool = True

    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
    visible_bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, σr, σc, symmetric=True):
        W = nknn.Dense(
            name="Dense",
            features=int(self.alpha * σr.shape[-1]),
            dtype=self.dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        xr = self.activation(W(σr)).sum(axis=-1)
        xc = self.activation(W(σc)).sum(axis=-1)

        if symmetric:
            y = xr + xc
        else:
            y = xr - xc

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias", self.visible_bias_init, (σr.shape[-1],), self.dtype
            )
            if symmetric:
                out_bias = jnp.dot(σr + σc, v_bias)
            else:
                out_bias = jnp.dot(σr - σc, v_bias)

            y = y + out_bias

        return 0.5 * y


class MixedRBM(nn.Module):
    """
    Encodes the pure state |ψ><ψ|, because it acts on row and column
    indices with the same RBM.
    """

    dtype: Any = np.float64
    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    use_hidden_bias: bool = True

    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, σr, σc, symmetric=True):
        U_S = nknn.Dense(
            name="Symm",
            features=int(self.alpha * σr.shape[-1]),
            dtype=self.dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
        )
        U_A = nknn.Dense(
            name="ASymm",
            features=int(self.alpha * σr.shape[-1]),
            dtype=self.dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
        )
        y = U_S(0.5 * (σr + σc)) + 1j * U_A(0.5 * (σr - σc))

        if self.use_hidden_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (int(self.alpha * σr.shape[-1]),),
                jax.dtypes.dtype_real(self.dtype),
            )
            y = y + bias

        y = self.activation(y)
        return y.sum(axis=-1)


class NDM(nn.Module):
    """
    Encodes a Positive-Definite Neural Density Matrix using the ansatz from Torlai and
    Melko, PRL 120, 240503 (2018).

    Assumes real dtype.
    A discussion on the effect of the feature density for the pure and mixed part is
    given in Vicentini et Al, PRL 122, 250503 (2019).

    Attributes:
        activation: The nonlinear activation function.
        alpha: The feature density for the pure-part of the ansatz.
        beta: The feature density for the mixed-part of the ansatz.
        use_bias: whever to use the hidden bias in the dense layers.
        use_visible_bias: whever to use a visible bias.
        dtype: The dtype of the parameters.
        kernel_init: the initializer for the dense kernels.
        bias_init: the initializer for the biases.
        visible_bias_init: the initialzier for the visible bias.
    """

    activation: Any = nknn.logcosh
    alpha: Union[float, int] = 1
    beta: Union[float, int] = 1
    use_hidden_bias: bool = True
    use_visible_bias: bool = True
    dtype: Any = np.float64

    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    visible_bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init

    @nn.compact
    def __call__(self, σ):
        σr, σc = jax.numpy.split(σ, 2, axis=-1)

        ψ_S = PureRBM(
            name="PureSymm",
            alpha=self.alpha,
            activation=self.activation,
            dtype=self.dtype,
            use_hidden_bias=self.use_hidden_bias,
            use_visible_bias=self.use_visible_bias,
            visible_bias_init=self.visible_bias_init,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        ψ_A = PureRBM(
            name="PureASymm",
            alpha=self.alpha,
            activation=self.activation,
            dtype=self.dtype,
            use_hidden_bias=self.use_hidden_bias,
            use_visible_bias=self.use_visible_bias,
            visible_bias_init=self.visible_bias_init,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )

        Π = MixedRBM(
            name="Mixed",
            alpha=self.beta,
            dtype=self.dtype,
            use_hidden_bias=self.use_hidden_bias,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )
        return (
            ψ_S(σr, σc, symmetric=True) + 1j * ψ_A(σr, σc, symmetric=False) + Π(σr, σc)
        )

        x = self.activation(x)

        return jnp.sum(x, axis=-1) + out_bias
