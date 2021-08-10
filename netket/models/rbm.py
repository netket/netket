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

from typing import Union, Any

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket.utils.group import PermutationGroup

from netket import nn as nknn
from netket.nn.initializers import normal
from netket.models.equivariant import GCNN
from netket.graph import Lattice, Graph


default_kernel_init = normal(stddev=0.01)


class RBM(nn.Module):
    """A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
    """

    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    @nn.compact
    def __call__(self, input):
        x = nknn.Dense(
            name="Dense",
            features=int(self.alpha * input.shape[-1]),
            dtype=self.dtype,
            precision=self.precision,
            use_bias=self.use_hidden_bias,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(input)
        x = self.activation(x)
        x = jnp.sum(x, axis=-1)

        if self.use_visible_bias:
            v_bias = self.param(
                "visible_bias", self.visible_bias_init, (input.shape[-1],), self.dtype
            )
            out_bias = jnp.dot(input, v_bias)
            return x + out_bias
        else:
            return x


class RBMModPhase(nn.Module):
    """
    A fully connected Restricted Boltzmann Machine (RBM) with real-valued parameters.

    In this case, two RBMs are taken to parameterize, respectively, the real
    and imaginary part of the log-wave-function, as introduced in Torlai et al.,
    Nature Physics 14, 447–450(2018).

    This type of RBM has spin 1/2 hidden units and is defined by:

    .. math:: \\Psi(s_1,\\dots s_N) = e^{\\sum_i^N a_i s_i} \\times \\Pi_{j=1}^M
            \\cosh \\left(\\sum_i^N W_{ij} s_i + b_j \\right)

    for arbitrary local quantum numbers :math:`s_i`.
    """

    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""

    @nn.compact
    def __call__(self, x):
        re = nknn.Dense(
            features=int(self.alpha * x.shape[-1]),
            dtype=self.dtype,
            use_bias=self.use_hidden_bias,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.hidden_bias_init,
        )(x)
        re = self.activation(re)
        re = jnp.sum(re, axis=-1)

        im = nknn.Dense(
            features=int(self.alpha * x.shape[-1]),
            dtype=self.dtype,
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
    A fully connected Restricted Boltzmann Machine (see :ref:`netket.models.RBM`) suitable for large local hilbert spaces.
    Local quantum numbers are passed through a one hot encoding that maps them onto
    an enlarged space of +/- 1 spins. In turn, these quantum numbers are used with a
    standard :class:`~netket.models.RBM` wave function.
    """

    n_classes: int
    """The number of classes in the one-hot encoding"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    def setup(self):
        self.RBM = RBM(
            dtype=self.dtype,
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
        # vectorizee the last two dimensions
        x_oh = jnp.reshape(x_oh, batches + (self.n_classes * N,))
        # apply the rbm to this output
        return self.RBM(x_oh)


def RBMSymm(
    symmetries=None,
    alpha=None,
    features=1,
    activation=nknn.log_cosh,
    use_hidden_bias=True,
    hidden_bias_init=None,
    **kwargs,
):
    """
    A symmetrized RBM using the :ref:`netket.nn.DenseSymm` layer internally.

    Args:
        symmetries: A specification of the symmetry group. Can be given by a
            nk.graph.Graph, a nk.utils.PermuationGroup, or an array [n_symm, n_sites]
            specifying the permutations corresponding to symmetry transformations
            of the lattice.
        dtype: The dtype of the weights.
        activation: The nonlinear activation function.
        alpha: feature density. Number of features equal to alpha * input.shape[-1]
        use_hidden_bias: if True uses a bias in the dense layer (hidden layer bias).
        precision: numerical precision of the computation see `jax.lax.Precision`for details.
        kernel_init: Initializer for the Dense layer matrix.
        hidden_bias_init: Initializer for the hidden bias.
        characters: Array specifying the characters of the desired symmetry representation
        parity: Optional argument with value +/-1 that specifies the eigenvalue
            with respect to parity (only use on two level systems).
        equal_amplitudes: If True forces all basis states to have equal amplitude
            by setting Re[psi] = 0.
        mode: string "fft, matrix, auto" specifying whether to use a fast
            fourier transform over the translation group, or by constructing
            the full kernel matrix.
        point_group: The point group, from which the space group is built.
            If symmetries is a graph the default point group is overwritten.
    """

    if alpha is not None:
        if isinstance(symmetries, Graph) and (
            point_group is not None or symmetries._point_group is not None
        ):
            n_sites = symmetries.n_nodes
            n_symm = symmetries.space_group(point_group)
        elif isinstance(symmetries, Graph):
            n_sites = symmetries.n_nodes
            n_symm = len(symmetries.automorphisms())
        else:
            n_symm, n_sites = np.asarray(symmetries).shape

        features = int(alpha * n_sites / n_symm)
        if alpha > 0 and features == 0:
            raise ValueError(
                f"RBMSymm: alpha={alpha} is too small "
                f"for {n_symm} permutations, alpha ≥ {n_symm / n_sites} is needed."
            )
    if hidden_bias_init is None:
        return GCNN(
            symmetries=symmetries,
            layers=1,
            features=features,
            output_activation=activation,
            use_bias=use_hidden_bias,
            **kwargs,
        )
    else:
        return GCNN(
            symmetries=symmetries,
            layers=1,
            features=features,
            output_activation=activation,
            use_bias=use_hidden_bias,
            bias_init=hidden_bias_init,
            **kwargs,
        )
