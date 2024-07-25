# Copyright 2022 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import jax
import jax.numpy as jnp

from flax import linen as nn
from jax.nn.initializers import lecun_normal, zeros

from netket.utils.types import NNInitFunc, DType
from netket import nn as nknn

default_kernel_init = lecun_normal()
default_bias_init = zeros


class MLP(nn.Module):
    r"""A Multi-Layer Perceptron with hidden layers.

    This model uses the MLP block with output dimension 1, which is squeezed.

    This combines multiple dense layers and activations functions into a single object.
    It separates the output layer from the hidden layers,
    since it typically has a different form.
    One can specify the specific activation functions per layer.
    The size of the hidden dimensions can be provided as a number,
    or as a factor relative to the input size (similar as for RBM).
    The default model is a single linear layer without activations.

    Forms a common building block for models such as
    `PauliNet (continuous) <https://www.nature.com/articles/s41557-020-0544-y>`_
    """

    hidden_dims: int | tuple[int, ...] | None = None
    """The size of the hidden layers, excluding the output layer."""
    hidden_dims_alpha: int | tuple[int, ...] | None = None
    """The size of the hidden layers provided as number of times the input size.
    One must choose to either specify this or the hidden_dims keyword argument"""
    param_dtype: DType = jnp.float64
    """The dtype of the weights."""
    hidden_activations: Callable | tuple[Callable, ...] | None = nknn.gelu
    """The nonlinear activation function after each hidden layer.
    Can be provided as a single activation,
    where the same activation will be used for every layer."""
    output_activation: Callable | None = None
    """The nonlinear activation at the output layer.
    If None is provided, the output layer will be essentially linear."""
    use_hidden_bias: bool = True
    """if True uses a bias in the hidden layer."""
    use_output_bias: bool = False
    """if True adds a bias to the output layer."""
    precision: jax.lax.Precision | None = None
    """Numerical precision of the computation see :class:`jax.lax.Precision` for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = default_bias_init
    """Initializer for the biases."""

    @nn.compact
    def __call__(self, input):
        x = nknn.blocks.MLP(
            output_dim=1,  # a netket model has a single output
            hidden_dims=self.hidden_dims,
            hidden_dims_alpha=self.hidden_dims_alpha,
            param_dtype=self.param_dtype,
            hidden_activations=self.hidden_activations,
            output_activation=self.output_activation,
            use_hidden_bias=self.use_hidden_bias,
            use_output_bias=self.use_output_bias,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(input)
        x = x.squeeze(-1)
        return x
