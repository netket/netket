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

from netket.hilbert import ContinuousHilbert


def check_features_length(features, n_layers, name):
    if len(features) != n_layers:
        raise ValueError(
            f"The number of {name} layers ({n_layers}) does not match "
            f"the length of the features list ({len(features)})."
        )


class DeepSetRelDistance(nn.Module):
    r"""Implements an equivariant version of the DeepSets architecture
    given by (https://arxiv.org/abs/1703.06114)

    .. math ::

        f(x_1,...,x_N) = \rho\left(\sum_i \phi(x_i)\right)

    that is suitable for the simulation of periodic systems.
    Additionally one can add a cusp condition by specifying the
    asymptotic exponent.
    For helium the Ansatz reads (https://arxiv.org/abs/2112.11957):

    .. math ::

        \psi(x_1,...,x_N) = \rho\left(\sum_i \phi(d_{\sin}(x_i,x_j))\right) \cdot \exp\left[-\frac{1}{2}\left(b/d_{\sin}(x_i,x_j)\right)^5\right]

    """

    hilbert: ContinuousHilbert
    """The hilbert space defining the periodic box where this ansatz is defined."""

    layers_phi: int
    """Number of layers in phi network."""
    layers_rho: int
    """Number of layers in rho network."""

    features_phi: Union[Tuple, int]
    """Number of features in each layer for phi network."""
    features_rho: Union[Tuple, int]
    """
    Number of features in each layer for rho network.
    If specified as a list, the last layer must have 1 feature. 
    """

    cusp_exponent: Optional[int] = None
    """exponent of Katos cusp condition"""

    dtype: Any = jnp.float64
    """The dtype of the weights."""

    activation: Any = jax.nn.gelu
    """The nonlinear activation function between hidden layers."""

    pooling: Any = jnp.sum
    """The pooling operation to be used after the phi-transformation"""

    use_bias: bool = True
    """if True uses a bias in all layers."""

    kernel_init: NNInitFunc = lecun_normal()
    """Initializer for the Dense layer matrix"""
    bias_init: NNInitFunc = zeros
    """Initializer for the hidden bias"""
    params_init: NNInitFunc = ones
    """Initializer for the parameter in the cusp"""

    def setup(self):

        if not all(self.hilbert.pbc):
            raise ValueError(
                "The DeepSetRelDistance model only works with "
                "hilbert spaces with periodic boundary conditions "
                "among all directions."
            )

        features_phi = self.features_phi
        if isinstance(features_phi, int):
            features_phi = [features_phi] * self.layers_phi

        check_features_length(features_phi, self.layers_phi, "phi")

        features_rho = self.features_rho
        if isinstance(features_rho, int):
            features_rho = [features_rho] * (self.layers_rho - 1) + [1]

        check_features_length(features_rho, self.layers_rho, "rho")
        assert features_rho[-1] == 1

        self.phi = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                param_dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in features_phi
        ]

        self.rho = [
            nn.Dense(
                feat,
                use_bias=self.use_bias,
                param_dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for feat in features_rho
        ]

    def distance(self, x, sdim, L):
        n_particles = x.shape[0] // sdim
        x = x.reshape(-1, sdim)

        dis = -x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :]

        dis = dis[jnp.triu_indices(n_particles, 1)]
        dis = L[jnp.newaxis, :] / 2.0 * jnp.sin(jnp.pi * dis / L[jnp.newaxis, :])
        return dis

    @nn.compact
    def __call__(self, x):
        sha = x.shape
        param = self.param("cusp", self.params_init, (1,), self.dtype)

        L = jnp.array(self.hilbert.extent)
        sdim = L.size

        d = jax.vmap(self.distance, in_axes=(0, None, None))(x, sdim, L)
        dis = jnp.linalg.norm(d, axis=-1)

        cusp = 0.0
        if self.cusp_exponent is not None:
            cusp = -0.5 * jnp.sum(param / dis**self.cusp_exponent, axis=-1)

        y = (d / L[jnp.newaxis, :]) ** 2

        """ The phi transformation """
        for layer in self.phi:
            y = self.activation(layer(y))

        """ Pooling operation """
        y = self.pooling(y, axis=-2)

        """ The rho transformation """
        for i, layer in enumerate(self.rho):
            y = layer(y)
            if i == len(self.rho) - 1:
                break
            y = self.activation(y)

        return (y.reshape(-1) + cusp).reshape(sha[0])
