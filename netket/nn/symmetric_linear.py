# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Union

from flax.linen.module import Module, compact
from jax import lax
import jax.numpy as jnp
import numpy as np

from netket.nn.initializers import normal, zeros
from netket.utils import HashableArray
from netket.utils.semigroup import PermutationGroup
from netket.utils.types import Array, DType, PRNGKeyT, Shape

default_kernel_init = normal(stddev=0.01)


def _symmetrizer_col(perms, features):
    """
    Creates the mapping from symmetry-reduced kernel w to full kernel W, s.t.
        W[ij] = S[ij][kl] w[kl]
    where [ij] ∈ [0,...,n_sites×n_hidden) and [kl] ∈ [0,...,n_sites×features).
    For each [ij] there is only one [kl] such that S[ij][kl] is non-zero, in which
    case S[ij][kl] == 1. Thus, this method only returns the array of indices `col`
    of shape (n_sites×n_hidden,) satisfying
        W[ij] = w[col[ij]]  <=>  W = w[col].

    See Test/Models/test_nn.py:test_symmetrizer for how this relates to the
    matrix form of the symmetrizer.
    """
    n_symm, n_sites = perms.shape
    n_hidden = features * n_symm

    ij = np.arange(n_sites * n_hidden)
    i, j = np.unravel_index(ij, (n_sites, n_hidden))

    k = perms[j % n_symm, i]
    l = np.floor_divide(j, n_symm)
    kl = np.ravel_multi_index((k, l), (n_sites, features))

    return kl


class DenseSymm(Module):
    """A symmetrized linear transformation applied over the last dimension of the input.

    This layer uses a reduced number of parameters, which are arranged so that the full
    affine transformation is invariant under all of the given permutations when applied to s.
    """

    symmetries: Union[HashableArray, PermutationGroup]
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.

        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    features: int
    """The number of symmetry-reduced features. The full output size is len(symmetries) * features."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: Callable[[PRNGKeyT, Shape, DType], Array] = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: Callable[[PRNGKeyT, Shape, DType], Array] = zeros
    """Initializer for the bias."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        perms = np.asarray(self.symmetries)
        self.n_symm, self.n_sites = perms.shape
        self.n_hidden = self.features * self.n_symm

        self.symm_cols = jnp.asarray(_symmetrizer_col(perms, self.features))

    def full_kernel(self, kernel):
        """
        Converts the symmetry-reduced kernel of shape (n_sites, features) to
        the full Dense kernel of shape (n_sites, features * n_symm).
        """
        kernel = kernel.reshape(-1)
        result = kernel[self.symm_cols]
        return result.reshape(self.n_sites, -1)

    def full_bias(self, bias):
        """
        Convert symmetry-reduced bias of shape (features,) to the full bias of
        shape (n_symm * features,).
        """
        return jnp.repeat(bias, self.n_symm)

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies the symmetrized linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)
        inputs = jnp.asarray(inputs, dtype)

        kernel = self.param(
            "kernel", self.kernel_init, (inputs.shape[-1], self.features), self.dtype
        )
        kernel = self.full_kernel(kernel)
        kernel = jnp.asarray(kernel, dtype)

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(self.full_bias(bias), dtype)
            y += bias

        return y


class DenseEquivariant(Module):
    r"""A group convolution operation that is equivariant over a symmetry group

    Acts on a feature map of symmetry poses of shape [batch_size,n_symm*in_features]
    and returns a feature  map of poses of shape [batch_size,n_symm*out_features]

    G-convolutions are described in ` Cohen et. {\it al} <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    and applied to quantum many-body problems in ` Roth et. {\it al} <https://arxiv.org/pdf/2104.05085.pdf>`_

    The G-convolution generalizes the convolution to non-commuting groups:

    .. math ::

        C^i_g = \sum_h {\bf W}_{g^{-1} h} \cdot {\bf f}_h

    Symmetry poses that are linked by the same symmetry element are connected
    by the same filter. The output symmetry group is an involution over the
    input symmetry group, i.e. the symmetry group is inverted by G-convolution

    .. math ::

        {\bf C}*(g) = C(g^{-1})

    """

    symmetry_info: Union[HashableArray, PermutationGroup]
    """Flattened product table generated by PermutationGroup.produt_table().ravel()
    that specifies the product of the group with its involution, or the
    PermutationGroup object itself"""
    in_features: int
    """The number of symmetry-reduced input features. The full input size
    is n_symm*in_features."""
    out_features: int
    """The number of symmetry-reduced output features. The full output size
    is n_symm*out_features."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: Callable[[PRNGKeyT, Shape, DType], Array] = default_kernel_init
    """Initializer for the Dense layer matrix."""
    bias_init: Callable[[PRNGKeyT, Shape, DType], Array] = zeros
    """Initializer for the bias."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        if isinstance(self.symmetry_info, PermutationGroup):
            self.symmetry_info = HashableArray(self.symmetry_info.product_table.ravel())
        if not np.asarray(self.symmetry_info).ndim == 1:
            raise ValueError("Product table should be flattened")

        self.n_symm = int(np.sqrt(np.asarray(self.symmetry_info).shape[0]))

    def full_kernel(self, kernel):
        """
        Converts the symmetry-reduced kernel of shape (n_sites, features) to
        the full Dense kernel of shape (n_sites, features * n_symm).
        """

        result = jnp.take(kernel, self.symmetry_info, 0)
        result = result.reshape(
            self.n_symm, self.n_symm, self.in_features, self.out_features
        )
        result = result.transpose(2, 0, 3, 1).reshape(
            self.n_symm * self.in_features, -1
        )

        return result

    def full_bias(self, bias):
        """
        Convert symmetry-reduced bias of shape (features,) to the full bias of
        shape (n_symm * features,).
        """
        return jnp.repeat(bias, self.n_symm)

    @compact
    def __call__(self, inputs: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last dimension.
        Args:
          inputs: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        dtype = jnp.promote_types(inputs.dtype, self.dtype)
        inputs = jnp.asarray(inputs, dtype)

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (inputs.shape[-1] // self.in_features, self.in_features, self.out_features),
            self.dtype,
        )
        kernel = self.full_kernel(kernel)
        kernel = jnp.asarray(kernel, dtype)

        y = lax.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.out_features,), self.dtype)
            bias = jnp.asarray(self.full_bias(bias), dtype)
            y += bias

        return y
