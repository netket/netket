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

from typing import Union, Optional, Tuple, Any, Callable, Iterable

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn

from netket.hilbert import AbstractHilbert
from netket.graph import AbstractGraph, SymmGroup
from netket.utils.types import PRNGKeyT, Shape, DType, Array, NNInitFunc

from netket import nn as nknn
from netket.nn.initializers import lecun_complex, zeros


class GCNN(nn.Module):
    """Implements a group convolutional neural network with symmetry
    averaged eigenvalues."""

    permutations: Callable[[],Array]
    """permutations specifying symmetry group"""
    group_algebra: Tuple
    """Matrix specifying algebra of symmetry group given by SymmGroup"""
    layers: int
    """Number of layers (not including sum layer over output)"""
    features: int
    """Number of features. The dimension of the hidden state is features*n_symm"""
    dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.swish
    """The nonlinear activation function."""
    alpha: Union[float, int] = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = lecun_complex()
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""

    def setup(self):
        self.n_symm = int(np.sqrt(len(self.group_algebra)))
        
        self.dense_symm = nknn.DenseSymm(
            permutations=self.permutations,
            features=self.features,
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            precision=self.precision,
        )

        self.equivariant_layers = [
            nknn.DenseEquivariant(
                group_algebra=self.group_algebra,
                features=self.features,
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.layers - 1)
        ]

    @nn.compact
    def __call__(self, x_in):
        x = self.dense_symm(x_in)
        for layer in range(self.layers - 1):
            x = self.equivariant_layers[layer](x)
            x = self.activation(x)

        x = jnp.mean(x, axis=-1)

        return x
