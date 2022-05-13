from typing import Union, Tuple, Any

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
    L: jnp.float64  # boxsize
    sdim: int  # nbr of spatial dimensions

    """Number of layers in phi/rho network"""
    layers_phi: int
    layers_rho: int

    """Number of features in each layer for phi/rho network."""
    features_phi: Union[Tuple, int]
    features_rho: Union[Tuple, int]

    """The dtype of the weights."""
    dtype: Any = jnp.float64

    """The nonlinear activation function between hidden layers."""
    activation: Any = jax.nn.gelu

    """if True uses a bias in all layers."""
    use_bias: bool = True

    """Initializer for the Dense layer matrix, hidden bias
    and parameter in the cusp"""
    kernel_init: NNInitFunc = lecun_normal()
    bias_init: NNInitFunc = zeros
    params_init: NNInitFunc = ones

    def setup(self):
        self.phi = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in self.features_phi
        ]

        self.rho = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                dtype=self.dtype,
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

        cusp = -0.5 * jnp.sum(param / d**5, axis=-2)

        y = d**2
        """ The phi transformation """
        for layer in range(self.layers_phi):
            y = self.phi[layer](y)
            y = self.activation(y)

        """ Pooling operation """
        y = jnp.sum(y, axis=-2)

        """ The rho transformation """
        for layer in range(self.layers_rho):
            y = self.rho[layer](y)
            if layer == self.layers_rho - 1:
                break
            y = self.activation(y)

        return (y + cusp).reshape(sha[0])
