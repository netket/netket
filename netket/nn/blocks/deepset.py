from collections.abc import Callable

import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.utils.types import NNInitFunc, DType
from jax.nn.initializers import (
    zeros,
    lecun_normal,
)

from .mlp import MLP


def _process_features(features) -> tuple[tuple[int, ...] | None, int | None]:
    """
    Convert some inputs to a consistent format of features.
    Returns hidden dimensions and output dimensions of the MLP separately.
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
        feat, out = tuple(features[:-1]), features[-1]
    return feat, out


class DeepSetMLP(nn.Module):
    r"""Implements the DeepSets architecture, which is permutation invariant
    and is suitable for the encoding of bosonic systems.

    .. math::

        f(x_1,...,x_N) = \rho\left(\sum_i \phi(x_i)\right)

    The input shape must have an axis that is reshaped to `(..., N, D)`, where we pool over N.

    """

    features_phi: int | tuple[int, ...] | None = None
    """
    Number of features in each layer for phi network.
    When features_phi is None, no phi network is created.
    """
    features_rho: int | tuple[int, ...] | None = None
    """
    Number of features in each layer for rho network.
    Should include final dimension of the network.
    When features_rho is None, no rho network is created.
    """

    param_dtype: DType = jnp.float64
    """The dtype of the weights."""

    hidden_activation: Callable | None = jax.nn.gelu
    """The nonlinear activation function between hidden layers."""
    output_activation: Callable | None = None
    """The nonlinear activation function at the output layer."""

    pooling: Callable = jnp.sum
    """The pooling operation to be used after the phi-transformation"""

    use_bias: bool = True
    """if True uses a bias in all layers."""

    kernel_init: NNInitFunc = lecun_normal()
    """Initializer for the Dense layer matrix"""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias"""
    precision: jax.lax.Precision | None = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

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

        if self.pooling is None:
            raise ValueError("Must specifyc pooling function for a DeepSet")

    @nn.compact
    def __call__(self, x):
        """The input shape must have an axis that is reshaped to (..., N, D), where we pool over N."""
        if x.ndim < 2:
            raise ValueError(
                f"input of deepset should have shape (..., N, D), but got {x.shape}"
            )

        if self.phi:
            x = self.phi(x)
        x = self.pooling(x, axis=-2)
        if self.rho:
            x = self.rho(x)

        return x
