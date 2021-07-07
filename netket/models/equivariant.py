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

from typing import Union, Optional, Tuple, Any

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn

from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket.utils.group import PermutationGroup
from netket.graph import Graph

from netket import nn as nknn
from netket.nn.initializers import zeros, unit_normal_scaling
from netket.nn.symmetric_linear import (
    DenseSymmMatrix,
    DenseSymmFFT,
    DenseEquivariantFFT,
    DenseEquivariantMatrix,
    DenseEquivariantIrrep,
)


class GCNN_FFT(nn.Module):

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the network should be equivariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    product_table: HashableArray
    """Product table describing the algebra of the symmetry group
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    shape: Tuple
    """Shape of the translation group"""
    layers: int
    """Number of layers (not including sum layer over output)."""
    features: Tuple
    """Number of features in each layer starting from the input. If a single number is given,
    all layers will have the same number of features."""
    characters: HashableArray
    """Array specifying the characters of the desired symmetry representation"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = jax.nn.selu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = identity
    """The nonlinear activation before the output. Defaults to the identity."""
    imag_part: bool = False
    """If true return only the imaginary part of the output"""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = unit_normal_scaling
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):

        self.n_symm = np.asarray(self.symmetries).shape[0]

        self.dense_symm = DenseSymmFFT(
            space_group=self.symmetries,
            shape=self.shape,
            features=self.features[0],
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            DenseEquivariantFFT(
                product_table=self.product_table,
                shape=self.shape,
                in_features=self.features[layer],
                out_features=self.features[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x):
        x = self.dense_symm(x)

        for layer in range(self.layers - 1):
            x = self.activation(x)
            x = self.equivariant_layers[layer](x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        x = x.reshape(-1, self.features[-1] * self.n_symm)
        x_max = jnp.max(x, axis=-1, keepdims=True)
        x = jnp.exp(x - x_max)
        x = x.reshape(-1, self.features[-1], self.n_symm)
        x = jnp.sum(x, 1)

        x = jnp.sum(x * jnp.array(self.characters), -1)

        x = jnp.log(x) + jnp.squeeze(x_max)

        if self.imag_part:
            return 1j * jnp.imag(x)
        else:
            return x


class GCNN_Irrep(nn.Module):

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the network should be equivariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    irreps: Tuple[HashableArray]
    """List of irreducible represenation matrices"""
    layers: int
    """Number of layers (not including sum layer over output)."""
    features: Tuple
    """Number of features in each layer starting from the input. If a single number is given,
    all layers will have the same number of features."""
    characters: HashableArray
    """Array specifying the characters of the desired symmetry representation"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = jax.nn.selu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = None
    """The nonlinear activation before the output."""
    imag_part: bool = False
    """If true return only the imaginary part of the output"""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = unit_normal_scaling
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):

        self.n_symm = np.asarray(self.symmetries).shape[0]

        self.dense_symm = DenseSymmMatrix(
            symmetries=self.symmetries,
            features=self.features[0],
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            DenseEquivariantIrrep(
                irreps=self.irreps,
                in_features=self.features[layer],
                out_features=self.features[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x):
        x = self.dense_symm(x)

        for layer in range(self.layers - 1):

            x = self.activation(x)
            x = self.equivariant_layers[layer](x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        x = x.reshape(-1, self.features[-1] * self.n_symm)
        x_max = jnp.max(x, axis=-1, keepdims=True)
        x = jnp.exp(x - x_max)
        x = x.reshape(-1, self.features[-1], self.n_symm)
        x = jnp.sum(x, 1)

        x = jnp.sum(x * jnp.array(self.characters), -1)

        x = jnp.log(x) + jnp.squeeze(x_max)

        if self.imag_part:
            return 1j * jnp.imag(x)
        else:
            return x


class GCNN_Parity_FFT(nn.Module):

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the network should be equivariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    product_table: HashableArray
    """Product table describing the algebra of the symmetry group
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    shape: Tuple
    """Shape of the translation group"""
    layers: int
    """Number of layers (not including sum layer over output)."""
    features: Tuple
    """Number of features in each layer starting from the input. If a single number is given,
    all layers will have the same number of features."""
    characters: HashableArray
    """Array specifying the characters of the desired symmetry representation"""
    parity: int
    """Integer specifying the eigenvalue with respect to parity"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = jax.nn.selu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = None
    """The nonlinear activation before the output."""
    imag_part: bool = False
    """If true return only the imaginary part of the output"""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = unit_normal_scaling
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):

        self.n_symm = np.asarray(self.symmetries).shape[0]

        self.dense_symm = DenseSymmFFT(
            space_group=self.symmetries,
            shape=self.shape,
            features=self.features[0],
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            DenseEquivariantFFT(
                product_table=self.product_table,
                shape=self.shape,
                in_features=self.features[layer],
                out_features=self.features[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

        self.equivariant_layers_flip = [
            DenseEquivariantFFT(
                product_table=self.product_table,
                shape=self.shape,
                in_features=self.features[layer],
                out_features=self.features[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x):

        x_flip = self.dense_symm(-1 * x)
        x = self.dense_symm(x)

        for layer in range(self.layers - 1):
            x = self.activation(x)
            x_flip = self.activation(x_flip)

            x_new = (self.equivariant_layers[layer](x) + self.equivariant_layers_flip[
                layer
            ](x_flip))/np.sqrt(2)
            x_flip = (self.equivariant_layers[layer](
                x_flip
            ) + self.equivariant_layers_flip[layer](x))/np.sqrt(2)
            x = jnp.array(x_new, copy=True)
            
        x = jnp.concatenate((x, x_flip), -2)

        if self.output_activation is not None:
            x = self.output_activation(x)

        x = x.reshape(-1, 2 * self.features[-1] * self.n_symm)
        x_max = jnp.max(x, axis=-1, keepdims=True)
        x = jnp.exp(x - x_max)
        x = x.reshape(-1, self.features[-1], 2 * self.n_symm)
        x = jnp.sum(x, 1)

        if self.parity == 1:
            par_chars = jnp.expand_dims(
                jnp.concatenate(
                    (jnp.array(self.characters), jnp.array(self.characters)), 0
                ),
                0,
            )
        else:
            par_chars = jnp.expand_dims(
                jnp.concatenate(
                    (jnp.array(self.characters), -1 * jnp.array(self.characters)), 0
                ),
                0,
            )

        x = jnp.sum(x * par_chars, -1)

        x = jnp.log(x) + jnp.squeeze(x_max)

        if self.imag_part:
            return 1j * jnp.imag(x)
        else:
            return x


class GCNN_Parity_Irrep(nn.Module):

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the network should be equivariant.
    Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    irreps: Tuple[HashableArray]
    """List of irreducible represenation matrices"""
    layers: int
    """Number of layers (not including sum layer over output)."""
    features: Tuple
    """Number of features in each layer starting from the input. If a single number is given,
    all layers will have the same number of features."""
    characters: HashableArray
    """Array specifying the characters of the desired symmetry representation"""
    parity: int
    """Integer specifying the eigenvalue with respect to parity"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = jax.nn.selu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = None
    """The nonlinear activation before the output."""
    imag_part: bool = False
    """If true return only the imaginary part of the output"""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = unit_normal_scaling
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):

        self.n_symm = np.asarray(self.symmetries).shape[0]

        self.dense_symm = DenseSymmMatrix(
            symmetries=self.symmetries,
            features=self.features[0],
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            DenseEquivariantIrrep(
                irreps=self.irreps,
                in_features=self.features[layer],
                out_features=self.features[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

        self.equivariant_layers_flip = [
            DenseEquivariantIrrep(
                irreps=self.irreps,
                in_features=self.features[layer],
                out_features=self.features[layer + 1],
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x):

        x_flip = self.dense_symm(-1 * x)
        x = self.dense_symm(x)

        for layer in range(self.layers - 1):
            x = self.activation(x)
            x_flip = self.activation(x_flip)

            x_new = (self.equivariant_layers[layer](x) + self.equivariant_layers_flip[
                layer
            ](x_flip))/np.sqrt(2)
            x_flip = (self.equivariant_layers[layer](
                x_flip
            ) + self.equivariant_layers_flip[layer](x))/np.sqrt(2)
            x = jnp.array(x_new, copy=True)

        x = jnp.concatenate((x, x_flip), -2)

        if self.output_activation is not None:
            x = self.output_activation(x)

        x = x.reshape(-1, 2 * self.features[-1] * self.n_symm)
        x_max = jnp.max(x, axis=-1, keepdims=True)
        x = jnp.exp(x - x_max)
        x = x.reshape(-1, self.features[-1], 2 * self.n_symm)
        x = jnp.sum(x, 1)

        if self.parity == 1:
            par_chars = jnp.expand_dims(
                jnp.concatenate(
                    (jnp.array(self.characters), jnp.array(self.characters)), 0
                ),
                0,
            )
        else:
            par_chars = jnp.expand_dims(
                jnp.concatenate(
                    (jnp.array(self.characters), -1 * jnp.array(self.characters)), 0
                ),
                0,
            )

        x = jnp.sum(x * par_chars, -1)

        x = jnp.log(x) + jnp.squeeze(x_max)

        if self.imag_part:
            return 1j * jnp.imag(x)
        else:
            return x


def GCNN(symmetries=None, mode="auto", **kwargs):

    r"""Implements a Group Convolutional Neural Network (G-CNN) that outputs a wavefunction
    that is invariant over a specified symmetry group.

    The G-CNN is described in ` Cohen et. *al* <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    and applied to quantum many-body problems in ` Roth et. *al* <https://arxiv.org/pdf/2104.05085.pdf>`_.

    The G-CNN alternates convolution operations with pointwise non-linearities. The first
    layer is symmetrized linear transform given by DenseSymm, while the other layers are
    G-convolutions given by DenseEquivariant. The hidden layers of the G-CNN are related by
    the following equation:

    .. math ::

        {\bf f}^{i+1}_h = \Gamma( \sum_h W_{g^{-1} h} {\bf f}^i_h).

    Args:
        symmetries: A group of symmetry operations (or array of permutation indices)
            over which the network should be equivariant. Numpy/Jax arrays must be
            wrapped into an :class:`netket.utils.HashableArray`.
        product_table: Product table describing the algebra of the symmetry group
            Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
        layers: Number of layers (not including sum layer over output).
        features: Number of features in each layer starting from the input. If a single
            number is given, all layers will have the same number of features.
        characters: Array specifying the characters of the desired symmetry representation
        dtype: The dtype of the weights
        parity: Optional argument with value +/-1 that specifies the eigenvalue
            with respect to parity (only use on two level systems)
        activation: The nonlinear activation function between hidden layers.
        output_activation: The nonlinear activation before the output.
        imag_part: If true return only the imaginary part of the output
        use_bias: if True uses a bias in all layers.
        precision: numerical precision of the computation see `jax.lax.Precision`for details.
        kernel_init: Initializer for the Dense layer matrix.
        bias_init: Initializer for the hidden bias.
    """

    if isinstance(symmetries, Graph):
        # With graph try to find point group, otherwise default to automorphisms
        if "point_group" in kwargs:
            sg = symmetries.space_group(kwargs["point_group"])
            if mode == "auto":
                mode = "fft"
            del kwargs["point_group"]
        elif symmetries._point_group:
            sg = symmetries.space_group()
            if mode == "auto":
                mode = "fft"
        else:
            sg = symmetry_info.automorphisms()
            if mode == "auto" or mode == "fft":
                mode = "irreps"
            if mode == "fft":
                warnings.warn(
                    "Graph without a space group specified. Switching to irrep implementation",
                )
        if mode == "fft":
            kwargs["shape"] = tuple(symmetries.extent)
    elif isinstance(symmetries, PermutationGroup):
        # If we get a group and default to irrep projection
        if mode == "auto":
            mode = "irreps"
        sg = symmetries
    else:
        if "irreps" in kwargs and (mode == "irreps" or mode == "auto"):
            mode = "irreps"
            sg = symmetries
            irreps = tuple(HashableArray(irrep) for irrep in kwargs["irreps"])
            del kwargs["irreps"]
        elif "product_table" in kwargs and (mode == "fft" or mode == "auto"):
            mode = "fft"
            sg = symmetries
            product_table = HashableArray(kwargs["product_table"])
            del kwargs["product_table"]
        else:
            raise ValueError(
                "Specification of symmetries is wrong or incompatible with selected mode"
            )

    if mode == "fft":
        if not "shape" in kwargs:
            raise KeyError(
                "Must pass keyword argument shape which specifies the shape of the translation group"
            )
        else:
            kwargs["shape"] = tuple(kwargs["shape"])
    else:
        if "shape" in kwargs:
            del kwargs["shape"]

    if isinstance(kwargs["features"], int):
        kwargs["features"] = (kwargs["features"],) * kwargs["layers"]

    if "characters" not in kwargs:
        kwargs["characters"] = HashableArray(np.ones(len(np.asarray(sg))))

    if mode == "fft":
        sym = HashableArray(np.asarray(sg))
        if not "product_table" in locals():
            product_table = HashableArray(sg.product_table)

        if "parity" in kwargs:
            return GCNN_Parity_FFT(sym, product_table, **kwargs)
        else:
            return GCNN_FFT(sym, product_table, **kwargs)
    else:
        sym = HashableArray(np.asarray(sg))

        if not "irreps" in locals():
            irreps = tuple(HashableArray(irrep) for irrep in sg.irrep_matrices())

        if "parity" in kwargs:
            return GCNN_Parity_Irrep(sym, irreps, **kwargs)
        else:
            return GCNN_Irrep(sym, irreps, **kwargs)
