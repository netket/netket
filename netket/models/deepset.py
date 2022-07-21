from typing import Union, Tuple, Any, Optional, Callable

import jax
from jax import numpy as jnp
from flax import linen as nn
from netket.utils.types import NNInitFunc
from jax.nn.initializers import (
    zeros,
    ones,
    lecun_normal,
)

from netket.utils import deprecate_dtype
from netket.hilbert import ContinuousHilbert
import netket as nk


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


class DeepSet(nn.Module):
    r"""Implements the DeepSets architecture, which is permutation invariant.

    .. math ::

        f(x_1,...,x_N) = \rho\left(\sum_i \phi(x_i)\right)

    that is suitable for the simulation of bosonic.

    The input shape must have an axis that is reshaped to (..., N, D), where we pool over N.

    """

    features_phi: Tuple[int]
    """Number of features in each layer for phi network."""
    features_rho: Tuple[int]
    """
    Number of features in each layer for rho network.
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

    squeeze_output: bool = False
    """ Whether to remove the 1 dimension in the output, if present """

    def setup(self):
        def _create_mlp(features, output_activation, name):
            hidden_dims, out_dim = _process_features(features)
            if out_dim is None:
                return None
            else:
                return nk.models.MLP(
                    output_dim=out_dim,
                    hidden_dims=hidden_dims,
                    param_dtype=self.param_dtype,
                    hidden_activations=self.hidden_activation,
                    output_activation=output_activation,
                    use_hidden_bias=self.use_bias,
                    use_output_bias=self.use_bias,
                    kernel_init=self.kernel_init,
                    squeeze_output=False,
                    name=name,
                )

        self.phi = _create_mlp(self.features_phi, self.hidden_activation, "ds_phi")
        self.rho = _create_mlp(self.features_rho, self.output_activation, "ds_rho")

    @nn.compact
    def __call__(self, x):
        """The input shape must have an axis that is reshaped to (..., N, D), where we pool over N."""

        if self.phi:
            x = self.phi(x)
        if self.pooling:
            x = self.pooling(x, axis=-2)
        if self.rho:
            x = self.rho(x)

        if self.squeeze_output:
            if x.shape[-1] != 1:
                raise ValueError(
                    f"cannot squeeze the output of deepset with output dimension {x.shape[-1]}"
                )
            x = x.squeeze(-1)

        return x


@deprecate_dtype
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

    param_dtype: Any = jnp.float64
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
        features_phi = tuple(features_phi)

        check_features_length(features_phi, self.layers_phi, "phi")

        features_rho = self.features_rho
        if isinstance(features_rho, int):
            features_rho = [features_rho] * (self.layers_rho - 1) + [1]
        features_rho = tuple(features_rho)

        check_features_length(features_rho, self.layers_rho, "rho")
        assert features_rho[-1] == 1

        self.deepset = DeepSet(
            features_phi,
            features_rho,
            param_dtype=self.param_dtype,
            hidden_activation=self.activation,
            output_activation=None,
            pooling=self.pooling,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            squeeze_output=True,
        )

    def distance(self, x, sdim, L):
        n_particles = x.shape[0] // sdim
        x = x.reshape(-1, sdim)

        dis = -x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :]

        dis = dis[jnp.triu_indices(n_particles, 1)]
        dis = L[jnp.newaxis, :] / 2.0 * jnp.sin(jnp.pi * dis / L[jnp.newaxis, :])
        return dis

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-1]
        param = self.param("cusp", self.params_init, (1,), self.param_dtype)

        L = jnp.array(self.hilbert.extent)
        sdim = L.size

        d = jax.vmap(self.distance, in_axes=(0, None, None))(x, sdim, L)
        dis = jnp.linalg.norm(d, axis=-1)

        cusp = 0.0
        if self.cusp_exponent is not None:
            cusp = -0.5 * jnp.sum(param / dis**self.cusp_exponent, axis=-1)

        y = (d / L[jnp.newaxis, :]) ** 2

        y = self.deepset(y)

        return (y + cusp).reshape(*batch_shape)
