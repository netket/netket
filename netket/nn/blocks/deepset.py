from typing import Tuple, Any, Callable

import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.utils.types import NNInitFunc
from jax.nn.initializers import (
    zeros,
    lecun_normal,
)

from .mlp import MLP


def check_features_length(features, n_layers, name):
    if len(features) != n_layers:
        raise ValueError(
            f"The number of {name} layers ({n_layers}) does not match "
            f"the length of the features list ({len(features)})."
        )


def _process_features(features):
    """
    Convert some inputs to a consistent format of features.
    Returns output dimension and hidden features of the MLP
    """
    if features is None:
        feat, out = None, None
    elif isinstance(features, int):
        feat, out = None, features
    elif len(features) == 0:
        feat, out = None, None
    elif len(features) == 1:
        feat, out = None, features[0]
    else:
        feat, out = features[:-1], features[-1]
    return feat, out


class DeepSetMLP(nn.Module):
    r"""Implements the DeepSets architecture, which is permutation invariant.

    .. math ::

        f(x_1,...,x_N) = \rho\left(\sum_i \phi(x_i)\right)

    that is suitable for the simulation of bosonic.

    The input shape must have an axis that is reshaped to (..., N, D), where we pool over N.

    """

    features_phi: Tuple[int] = None
    """Number of features in each layer for phi network."""
    features_rho: Tuple[int] = None
    """
    Number of features in each layer for rho network.
    Should include final dimension of size 1.
    """

    param_dtype: Any = jnp.float64
    """The dtype of the weights."""

    hidden_activation: Callable = jax.nn.gelu
    """The nonlinear activation function between hidden layers."""
    output_activation: Callable = None
    """The nonlinear activation function at the output layer."""

    pooling: Callable = jnp.sum
    """The pooling operation to be used after the phi-transformation"""

    use_bias: bool = True
    """if True uses a bias in all layers."""

    kernel_init: NNInitFunc = lecun_normal()
    """Initializer for the Dense layer matrix"""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias"""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    def setup(self):
        def _create_mlp(features, output_activation, name):
            hidden_dims, out_dim = _process_features(features)
            if out_dim is None:
                return None
            else:
                return MLP(
                    output_dim=out_dim,
                    hidden_dims=hidden_dims,
                    param_dtype=self.param_dtype,
                    hidden_activations=self.hidden_activation,
                    output_activation=output_activation,
                    use_hidden_bias=self.use_bias,
                    use_output_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    name=name,
                )

        self.phi = _create_mlp(self.features_phi, self.hidden_activation, "ds_phi")
        self.rho = _create_mlp(self.features_rho, self.output_activation, "ds_rho")

    @nn.compact
    def __call__(self, x):
        """The input shape must have an axis that is reshaped to (..., N, D), where we pool over N."""
        if x.ndim < 2:
            raise ValueError(
                f"input of deepset should have shape (..., N, D), but got {x.shape}"
            )

        if self.phi:
            x = self.phi(x)
        if self.pooling:
            x = self.pooling(x, axis=-2)
        if self.rho:
            x = self.rho(x)

        return x
