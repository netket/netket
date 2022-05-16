from typing import Union, Tuple, Any, Optional

import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.utils.types import NNInitFunc
from jax.nn.initializers import (
    zeros,
    ones,
    lecun_normal,
)


class DeepSet(nn.Module):
    r"""Implements an equivariant version of the DeepSets architecture
    given by (https://arxiv.org/abs/1703.06114)

    .. math ::

        f(x_1,...,x_N) = \rho\left(\sum_i \phi(x_i)\right)

    that is suitable for the simulation of periodic systems.
    Additionally one can add a cusp condition by specifying the
    asymptotic exponent.
    For helium the Ansatz reads:

    .. math ::

        \psi(x_1,...,x_N) = \rho\left(\sum_i \phi(d_{\sin}(x_i,x_j))\right) \cdot \exp\left[-\frac{1}{2}\left(b/d_{\sin}(x_i,x_j)\right)^5\right]

    """
    L: jnp.float64
    """boxsize"""
    sdim: int
    """number of spatial dimensions"""

    layers_phi: int
    """Number of layers in phi network"""
    layers_rho: int
    """Number of layers in rho network"""

    features_phi: Union[Tuple, int]
    """Number of features in each layer for phi network."""
    features_rho: Union[Tuple, int]
    """Number of features in each layer for rho network."""

    cusp_exponent: Optional[int] = None
    """exponent of Katos cusp condition"""

    dtype: Any = jnp.float64
    """The dtype of the weights."""

    activation: Any = jax.nn.gelu
    """The nonlinear activation function between hidden layers."""

    use_bias: bool = True
    """if True uses a bias in all layers."""

    kernel_init: NNInitFunc = lecun_normal()
    """Initializer for the Dense layer matrix"""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias"""
    params_init: NNInitFunc = ones
    """Initializer for the parameter in the cusp"""

    def setup(self):
        if isinstance(self.features_phi, int):
            self.features_phi = [self.features_phi] * (self.layers_phi - 1)

        if isinstance(self.features_rho, int):
            self.features_rho = [self.features_rho] * (self.layers_rho - 1)

        self.phi = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                param_dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in self.features_phi
        ]

        self.rho = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                param_dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in self.features_rho
        ]

    def distance(self, x, sdim, L):
        n_particles = x.shape[0] // sdim
        x = x.reshape(-1, sdim)

        dis = -x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :]
        dis = dis[jnp.triu_indices(n_particles, 1)]

        return dis

    @nn.compact
    def __call__(self, x):
        sha = x.shape
        param = self.param("cusp", self.params_init, (1,), self.dtype)

        d = jax.vmap(self.distance, in_axes=(0, None, None))(x, self.sdim, self.L)

        d = (
            self.L
            / 2.0
            * jnp.sin(jnp.pi / self.L * jnp.linalg.norm(d, axis=-1, keepdims=True))
        )
        cusp = 0.0
        if self.cusp_exponent is not None:
            cusp = -0.5 * jnp.sum(param / d**self.cusp_exponent, axis=-2)

        y = d**2
        """ The phi transformation """
        for layer in self.phi:
            y = self.activation(layer(y))

        """ Pooling operation """
        y = jnp.sum(y, axis=-2)

        """ The rho transformation """
        for i, layer in enumerate(self.rho):
            y = layer(y)
            if i == len(self.rho) - 1:
                break
            y = self.activation(y)

        return (y + cusp).reshape(sha[0])
