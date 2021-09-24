from typing import Union, Tuple, Any

import numpy as np

import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from flax import linen as nn

from netket.utils.types import NNInitFunc

from netket.nn.initializers import (
    zeros,
    variance_scaling,
    ones,
    normal,
    uniform,
    lecun_normal,
    kaiming_normal,
    kaiming_uniform,
)


class DS(nn.Module):
    k: int
    """Number of moments for the periodic transformation."""
    L: jnp.float64
    sdim: int

    layers_phi: int
    layers_rho: int
    """Number of layers in phi/rho network"""

    features_phi: Union[Tuple, int]
    features_rho: Union[Tuple, int]
    """Number of features in each layer for phi/rho network.
    If int is given all layers have the same number of features."""

    dtype: Any = jnp.float64
    """The dtype of the weights."""
    activation: Any = jax.nn.gelu
    """The nonlinear activation function between hidden layers."""
    output_activation: Any = None
    """The nonlinear activation before the output."""
    use_bias: bool = True
    """if True uses a bias in all layers."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""
    # kernel_init: NNInitFunc = normal(stddev=0.1)
    kernel_init: NNInitFunc = lecun_normal()
    """Initializer for the Dense layer matrix."""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias."""
    params_init: NNInitFunc = ones

    special: bool = False

    def setup(self):

        if isinstance(self.features_phi, int):
            feature_dim_phi = [self.features_phi for layer in range(self.layers_phi)]

        else:
            if not len(self.features_phi) == self.layers_phi:
                raise ValueError(
                    """Length of vector specifying feature dimensions must be the same as the number of layers"""
                )
            feature_dim_phi = self.features_phi

        if isinstance(self.features_phi, int):
            feature_dim_rho = [self.features_rho for layer in range(self.layers_rho)]

        else:
            if not len(self.features_rho) == self.layers_rho:
                raise ValueError(
                    """Length of vector specifying feature dimensions must be the same as the number of layers"""
                )
            feature_dim_rho = self.features_rho

        self.phi = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in feature_dim_phi
        ]

        self.rho = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                dtype=self.dtype,
                precision=self.precision,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in feature_dim_rho
        ]

    @staticmethod
    def minimum_distance(x, sdim, L):
        n_particles = x.shape[0] // sdim
        x = x.reshape(-1, sdim)

        dist = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
            jnp.triu_indices(n_particles, 1)
        ]
        return jnp.linalg.norm(jnp.sin(jnp.pi / L * dist), axis=1)

    def single_sample(self, x_in, params):
        # compute two-body Jastrow cusp condition
        dist = self.minimum_distance(x_in, self.sdim, self.L)

        cusp = -0.5 * ((0.67112756 / dist) ** (5 * 0.42126763))  # *params[1]))

        def phi(x):
            # residual
            # temp = jnp.dot(params[1], x)
            for layer in range(self.layers_phi):
                x = self.phi[layer](x)
                if layer == self.layers_rho - 1:
                    break
                x = self.activation(x)
            return x  # + temp

        """
        # periodic transformation
        x = jnp.dstack((jnp.sin(2 * jnp.pi / self.L * x_in), jnp.cos(2 * jnp.pi / self.L * x_in)))
        for i in range(self.k - 1):
            x = jnp.dstack((x, jnp.sin(2 * (i + 2) * jnp.pi / self.L * x_in),
                            jnp.cos(2 * (i + 2) * jnp.pi / self.L * x_in)))
        x = x.reshape(-1, 2*self.sdim*self.k)
        """
        # x = dist.reshape(-1,1) ** 2
        encoding = jax.vmap(phi, in_axes=0, out_axes=0)
        y = encoding((dist ** 2).reshape(-1, 1)).reshape(-1)
        # y = logsumexp(y,axis=0)
        """
        y = (dist**2).reshape(-1,1)
        # rho network
        for layer in range(self.layers_rho):
            y = self.rho[layer](y)
            if layer == self.layers_rho - 1:
                break
            y = self.activation(y)
        """
        return jnp.sum(cusp) + jnp.sum(y)

    @nn.compact
    def __call__(self, x_in):
        special = False
        if len(x_in.shape) < 2:
            special = True
            x_in = x_in.reshape(1, x_in.shape[0])

        kernel = self.param("cusp", self.params_init, (2,), self.dtype)
        # res1 = self.param("res1", self.kernel_init, (self.features_phi[-1], 2*self.sdim*self.k), self.dtype)

        multiple = jax.vmap(self.single_sample, in_axes=(0, None), out_axes=0)

        logpsi = multiple(x_in, (kernel))  # , res1))

        if special:
            return jnp.sum(logpsi)

        return logpsi
