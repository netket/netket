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

import abc
from typing import Any, Callable, Iterable, Tuple, Union

from flax import linen as nn
from jax import numpy as jnp
from netket.nn import MaskedDense1D
from netket.nn.initializers import lecun_normal, zeros
from netket.utils.types import Array, DType, NNInitFunc, PyTree


class ARNN(nn.Module):
    """Base class for autoregressive neural networks."""

    @abc.abstractmethod
    def conditionals(self, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
        """
        Computes the probabilities for each spin to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          cache: auxiliary states, e.g., used to implement fast autoregressive sampling.

        Returns:
          p: the probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).
          cache: the updated cache.
        """
        raise NotImplementedError


class ARNNDense(ARNN):
    """Autoregressive neural network with dense layers."""

    hilbert_local_size: int
    """local size of the Hilbert space."""
    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """number of features in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    activation: Callable[[Array], Array] = nn.silu
    """the nonlinear activation function between hidden layers (default: silu)."""
    use_bias: bool = True
    """whether to add a bias to the output (default: True)."""
    dtype: DType = jnp.float64
    """the dtype of the weights (default: float64)."""
    precision: Any = None
    """numerical precision of the computation, see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = lecun_normal()
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    eps: float = 1e-7
    """a small number to avoid numerical instability."""

    def setup(self):
        if isinstance(self.features, int):
            features = [self.features] * (self.layers - 1) + [self.hilbert_local_size]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == self.hilbert_local_size

        self.dense_layers = [
            MaskedDense1D(
                features=features[i],
                exclusive=(i == 0),
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]

    def conditionals(self, inputs: Array, cache: PyTree) -> Tuple[Array, PyTree]:
        x = jnp.expand_dims(inputs, axis=2)
        for i in range(self.layers):
            if i > 0:
                x = self.activation(x)
            x = self.dense_layers[i](x)
        p = nn.softmax(x, axis=2)
        return p, cache

    def __call__(self, inputs: Array) -> Array:
        """Returns log_psi, where psi is real."""

        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)

        idx = (inputs + self.hilbert_local_size - 1) / 2
        idx = jnp.asarray(idx, jnp.int64)
        idx = jnp.expand_dims(idx, axis=2)

        p, _ = self.conditionals(inputs, None)
        p = jnp.take_along_axis(p, idx, axis=1)

        log_psi = 1 / 2 * jnp.log(p + self.eps).sum(axis=(1, 2))
        return log_psi
