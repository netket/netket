from typing import Any, Tuple, Callable, Union

import numpy as np

from flax import linen as nn
from jax.nn.initializers import lecun_normal, zeros

from netket.utils.types import NNInitFunc
from netket import nn as nknn

default_kernel_init = lecun_normal()
default_bias_init = zeros


class MLP(nn.Module):
    r"""A Multi-Layer Perceptron with hidden layers.

    This combines multiple dense layers and activations functions into a single object.
    It separates the output layer from the hidden layers, since it typically has a different form.
    One can specify the specific activation functions per layer.
    The size of the hidden dimensions can be provided as a number, or as a factor relative to the input size (similar as for RBM)
    """
    output_dim: int = 1
    """The output dimension"""
    hidden_dims: Tuple[int] = None
    """The size of the hidden layers, excluding the output layer."""
    alpha_hidden_dims: Tuple[int] = None
    """The size of the hidden layers provided as number of times the input size. One must choose to either specify this or the hidden_dims keyword argument"""
    param_dtype: Any = np.float64
    """The dtype of the weights."""
    hidden_activation: Union[Callable, Tuple[Callable]] = nknn.gelu
    """The nonlinear activation function after each hidden layer. Can be provided as a single activation, where the same activation will be used for every layer."""
    output_activation: Callable = None
    """The nonlinear activation at the output layer. If None is provided, the output layer will be essentially linear."""
    use_hidden_bias: bool = True
    """if True uses a bias in the hidden layer."""
    use_output_bias: bool = False
    """if True adds a bias to the output layer."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = default_bias_init
    """Initializer for the biases."""
    squeeze_output: bool = False
    """Whether to remove output dimension 1 if it is present. This is typically useful if we want to use the MLP as an NQS directly, where we do not need the final dimension 1."""

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

        if self.hidden_activation is None:
            hidden_activation = [None] * len(hidden_dims)
        elif hasattr(self.hidden_activation, "__len__"):
            if len(self.hidden_activation) != len(hidden_dims):
                raise ValueError(
                    "number of hidden activations must be the same as the length of the hidden dimensions list"
                )
            hidden_activation = self.hidden_activation
        else:
            hidden_activation = [self.hidden_activation] * len(hidden_dims)

        x = input

        # hidden layers
        for nh, act_h in zip(hidden_dims, hidden_activation):
            x = nn.Dense(
                features=nh,
                param_dtype=self.param_dtype,
                precision=self.precision,
                use_bias=self.use_hidden_bias,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            if act_h:
                x = act_h(x)

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
