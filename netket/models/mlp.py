from typing import Any, Tuple

import numpy as np

from flax import linen as nn
from jax.nn.initializers import lecun_normal, zeros

from netket.utils import deprecate_dtype
from netket.utils.types import NNInitFunc
from netket import nn as nknn

default_kernel_init = lecun_normal()
default_bias_init = zeros


@deprecate_dtype
class MLP(nn.Module):
    r"""A Multi-Layer Perceptron"""
    output_dim: int = 1
    """The output dimension"""
    hidden_dims: Tuple[int] = None
    """The size of the hidden layers"""
    alpha_hidden_dims: Tuple[int] = None
    """The size of the hidden layers provided as number of times the input size"""
    param_dtype: Any = np.float64
    """The dtype of the weights."""
    hidden_activation: Any = nknn.gelu
    """The nonlinear activation function after each hidden layer."""
    output_activation: Any = None
    """The nonlinear activation at the output layer."""
    use_hidden_bias: bool = True
    """if True uses a bias in the hidden layer."""
    use_output_bias: bool = False
    """if True adds a bias to the output layer."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = default_bias_init
    """Initializer for the bias."""
    squeeze_output: bool = False
    """Whether to remove output dimension 1 if it is present"""

    @nn.compact
    def __call__(self, input):
        hidden_dims = self.hidden_dims
        if hidden_dims is None:
            if self.alpha_hidden_dims is not None:
                hidden_dims = [
                    int(nh * input.shape[-1]) for nh in self.alpha_hidden_dims
                ]
        elif self.alpha_hidden_dims is not None:
            raise ValueError(
                "Cannot specify both hidden_dims and alpha_hidden_dims, choose one way to provide the hidden dimensions"
            )

        if hidden_dims is None:
            hidden_dims = []

        x = input

        # hidden layers
        for nh in hidden_dims:
            x = nn.Dense(
                features=nh,
                param_dtype=self.param_dtype,
                precision=self.precision,
                use_bias=self.use_hidden_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            if self.hidden_activation:
                x = self.hidden_activation(x)

        # output layer
        x = nn.Dense(
            features=self.output_dim,
            param_dtype=self.param_dtype,
            precision=self.precision,
            use_bias=self.use_output_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.output_activation:
            x = self.output_activation(x)

        if self.squeeze_output:
            if not self.output_dim == 1:
                raise ValueError(
                    "can only squeeze the output if the output dimension is 1"
                )
            x = x.squeeze(-1)

        return x
