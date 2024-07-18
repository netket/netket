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
from jax.nn.initializers import normal

from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket.utils.group import PermutationGroup
from netket import nn as nknn

default_kernel_init = normal(stddev=0.01)


class RBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
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
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias",
                self.visible_bias_init,
                (input.shape[-1],),
                self.param_dtype,
            )
            out_bias = jnp.dot(input, v_bias)
            return x + out_bias
        else:
            return x


class RBMModPhase(nn.Module):
    r"""
    A fully connected Restricted Boltzmann Machine (RBM) with real-valued parameters.

    In this case, two RBMs are taken to parametrize, respectively, the real
    and imaginary part of the log-wave-function, as introduced in Torlai et al.,
    Nature Physics 14, 447–450(2018).

    This type of RBM has spin 1/2 hidden units and is defined by:

    .. math::

        \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M
            \cosh \left(\sum_i^N W_{ij} s_i + b_j \right)

    for arbitrary local quantum numbers :math:`s_i`.
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    precision: Any = None
    """Numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""

    @nn.compact
    def __call__(self, x):
        re = nn.Dense(
            features=int(self.alpha * x.shape[-1]),
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(x)
        re = self.activation(re)
        re = jnp.sum(re, axis=-1)

        im = nn.Dense(
            features=int(self.alpha * x.shape[-1]),
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(x)
        im = self.activation(im)
        im = jnp.sum(im, axis=-1)

        return re + 1j * im


class RBMMultiVal(nn.Module):
    """
    A fully connected Restricted Boltzmann Machine (see :class:`netket.models.RBM`) suitable for large local hilbert spaces.
    Local quantum numbers are passed through a one hot encoding that maps them onto
    an enlarged space of +/- 1 spins. In turn, these quantum numbers are used with a
    standard :class:`~netket.models.RBM` wave function.
    """

    n_classes: int
    """The number of classes in the one-hot encoding"""
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
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    def setup(self):
        self.RBM = RBM(
            param_dtype=self.param_dtype,
            activation=self.activation,
            alpha=self.alpha,
            use_hidden_bias=self.use_hidden_bias,
            use_visible_bias=self.use_visible_bias,
            kernel_init=self.kernel_init,
            hidden_bias_init=self.hidden_bias_init,
            visible_bias_init=self.visible_bias_init,
        )

    def __call__(self, x):
        batches = x.shape[:-1]
        N = x.shape[-1]

        # do the one hot encoding: output x.shape +(n_classes,)
        x_oh = jax.nn.one_hot(x, self.n_classes)
        # vectorize the last two dimensions
        x_oh = jnp.reshape(
            x_oh,
            (
                *batches,
                self.n_classes * N,
            ),
        )
        # apply the rbm to this output
        return self.RBM(x_oh)


class RBMSymm(nn.Module):
    """A symmetrized RBM using the :func:`netket.nn.DenseSymm` layer internally."""

    symmetries: HashableArray | PermutationGroup
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
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

    kernel_init: NNInitFunc = normal(stddev=0.1)
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = normal(stddev=0.1)
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = normal(stddev=0.1)
    """Initializer for the visible bias."""

    def setup(self):
        self.n_symm, self.n_sites = np.asarray(self.symmetries).shape
        self.features = int(self.alpha * self.n_sites / self.n_symm)
        if self.alpha > 0 and self.features == 0:
            raise ValueError(
                f"RBMSymm: alpha={self.alpha} is too small "
                f"for {self.n_symm} permutations, alpha ≥ {self.n_symm / self.n_sites} is needed."
            )

    @nn.compact
    def __call__(self, x_in):
        x = x_in
        if x.ndim < 3:
            x = jnp.expand_dims(x, -2)
        x = nknn.DenseSymm(
            name="Dense",
            mode="matrix",
            symmetries=self.symmetries,
            features=self.features,
            param_dtype=self.param_dtype,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
            precision=self.precision,
        )(x)
        x = self.activation(x)

        x = x.reshape(-1, self.features * self.n_symm)
        x = jnp.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias", self.visible_bias_init, (1,), self.param_dtype
            )
            out_bias = v_bias[0] * jnp.sum(x_in, axis=-1)
            return x + out_bias
        else:
            return x
