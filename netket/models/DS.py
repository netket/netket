from typing import Union, Tuple, Any, Callable

from netket.utils.types import PyTree

import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from flax import linen as nn

from netket.utils.types import NNInitFunc

from jax.nn.initializers import (
    zeros,
    ones,
    lecun_normal,
)


class DeepSet(nn.Module):
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

        if isinstance(self.features_rho, int):
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
    def minimum_distance(x, sdim):
        n_particles = x.shape[0] // sdim
        x = x.reshape(-1, sdim)

        dis = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])
        dis = dis[jnp.triu_indices(n_particles, 1)]

        return dis

    def single_sample(self, x_in, params):

        # compute two-body Jastrow cusp condition
        dis = self.minimum_distance(x_in, self.sdim)
        dist = self.L/2.*jnp.sin(jnp.pi/self.L * jnp.linalg.norm(dis, axis=-1))
        cusp = jnp.log(dist) * params[0]

        def phi(x):
            for layer in range(self.layers_phi):
                x = self.phi[layer](x)
                x = self.activation(x)
            return x

        encoding = jax.vmap(phi, in_axes=0, out_axes=0)

        # periodic transformation
        x_in = x_in.reshape(-1, self.sdim)
        x = jnp.hstack((jnp.sin(2 * jnp.pi / self.L * x_in), jnp.cos(2 * jnp.pi / self.L * x_in)))
        for i in range(self.k - 1):
            x = jnp.hstack((x, jnp.sin(2 * (i + 2) * jnp.pi / self.L * x_in),
                            jnp.cos(2 * (i + 2) * jnp.pi / self.L * x_in)))
        x = x.reshape(-1, 2*self.sdim*self.k)

        y = encoding(x)
        y = logsumexp(y, axis=0)

        for layer in range(self.layers_rho):
            y = self.rho[layer](y)
            if layer == self.layers_rho - 1:
                break
            y = self.activation(y)

        return jnp.sum(cusp) + jnp.sum(y)

    @nn.compact
    def __call__(self, x_in):
        special = False
        if len(x_in.shape) < 2:
            special = True
            x_in = x_in.reshape(1, x_in.shape[0])

        kernel = self.param("cusp", self.params_init, (1,), self.dtype)

        multiple = jax.vmap(self.single_sample, in_axes=(0, None), out_axes=0)

        logpsi = multiple(x_in, kernel)

        if special:
            return jnp.sum(logpsi)

        return logpsi
