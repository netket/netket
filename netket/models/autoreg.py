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
    def init_state(self, inputs: Array) -> PyTree:
        """
        Initializes the model state.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).

        Returns:
          state: auxiliary model state, e.g., used to implement fast autoregressive sampling.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def conditionals(self, inputs: Array, state: PyTree) -> Tuple[Array, PyTree]:
        """
        Computes the probabilities for each spin to take each value.

        Args:
          inputs: configurations with dimensions (batch, Hilbert.size).
          state: auxiliary model state, e.g., used to implement fast autoregressive sampling.

        Returns:
          p: the probabilities with dimensions (batch, Hilbert.size, Hilbert.local_size).
          state: the updated model state.
        """
        raise NotImplementedError


class ARNNDense(ARNN):
    """Autoregressive neural network with dense layers, assuming spin 1/2."""

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
            features = [self.features] * (self.layers - 1) + [1]
        else:
            features = self.features
        assert len(features) == self.layers
        assert features[-1] == 1

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

    def init_state(self, inputs: Array) -> PyTree:
        return None

    def conditionals(self, inputs: Array, state: PyTree) -> Tuple[Array, PyTree]:
        x = jnp.expand_dims(inputs, axis=2)
        for i in range(self.layers):
            if i > 0:
                x = self.activation(x)
            x = self.dense_layers[i](x)
        x = x.squeeze(axis=2)

        p = nn.sigmoid(x)
        p = jnp.stack([1 - p, p], axis=2)

        return p, state

    def __call__(self, inputs: Array) -> Array:
        """Returns log_psi, where psi is real."""
        p, _ = self.conditionals(inputs, None)
        mask = (inputs + 1) / 2
        p = (1 - mask) * p[:, :, 0] + mask * p[:, :, 1]
        log_psi = 1 / 2 * jnp.log(p + self.eps).sum(axis=1)
        return log_psi
