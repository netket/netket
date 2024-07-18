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

from typing import Any

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, normal

from netket.utils.types import NNInitFunc
from netket import jax as nkjax
from netket import nn as nknn

default_kernel_init = normal(stddev=0.001)


class PureRBM(nn.Module):
    """
    Encodes the pure state |ψ><ψ|, because it acts on row and column
    indices with the same RBM.
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """Numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = zeros
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, σr, σc, symmetric=True):
        W = nn.Dense(
            name="Dense",
            features=int(self.alpha * σr.shape[-1]),
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
            precision=self.precision,
        )
        xr = self.activation(W(σr)).sum(axis=-1)
        xc = self.activation(W(σc)).sum(axis=-1)

        if symmetric:
            y = xr + xc
        else:
            y = xr - xc

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (σr.shape[-1],),
                self.param_dtype,
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

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    @nn.compact
    def __call__(self, σr, σc):
        U_S = nn.Dense(
            name="Symm",
            features=int(self.alpha * σr.shape[-1]),
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
            precision=self.precision,
        )
        U_A = nn.Dense(
            name="ASymm",
            features=int(self.alpha * σr.shape[-1]),
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=self.kernel_init,
            precision=self.precision,
        )
        y = U_S(0.5 * (σr + σc)) + 1j * U_A(0.5 * (σr - σc))

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (int(self.alpha * σr.shape[-1]),),
                nkjax.dtype_real(self.param_dtype),
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
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """The feature density for the pure-part of the ansatz.
    Number of features equal to alpha * input.shape[-1]
    """
    beta: float | int = 1
    """The feature density for the mixed-part of the ansatz.
    Number of features equal to beta * input.shape[-1]
    """
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_ancilla_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, σ):
        σr, σc = jax.numpy.split(σ, 2, axis=-1)

        ψ_S = PureRBM(
            name="PureSymm",
            alpha=self.alpha,
            activation=self.activation,
            param_dtype=self.param_dtype,
            use_hidden_bias=self.use_hidden_bias,
            use_visible_bias=self.use_visible_bias,
            visible_bias_init=self.visible_bias_init,
            kernel_init=self.kernel_init,
            hidden_bias_init=self.bias_init,
            precision=self.precision,
        )

        ψ_A = PureRBM(
            name="PureASymm",
            alpha=self.alpha,
            activation=self.activation,
            param_dtype=self.param_dtype,
            use_hidden_bias=self.use_hidden_bias,
            use_visible_bias=self.use_visible_bias,
            visible_bias_init=self.visible_bias_init,
            kernel_init=self.kernel_init,
            hidden_bias_init=self.bias_init,
            precision=self.precision,
        )

        Π = MixedRBM(
            name="Mixed",
            alpha=self.beta,
            param_dtype=self.param_dtype,
            use_bias=self.use_ancilla_bias,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )
        return (
            ψ_S(σr, σc, symmetric=True) + 1j * ψ_A(σr, σc, symmetric=False) + Π(σr, σc)
        )
