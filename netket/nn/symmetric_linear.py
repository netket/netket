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

from typing import Any, Optional, Tuple

import numpy as np
import jax.numpy as jnp

from jax import lax
from jax.nn.initializers import zeros, lecun_normal
from flax.linen.module import Module, compact

from netket.utils import warn_deprecation
from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc
from netket.utils.group import PermutationGroup
from typing import Sequence
from netket.graph import Graph, Lattice

# All layers defined here have kernels of shape [out_features, in_features, n_symm]
default_equivariant_initializer = lecun_normal(in_axis=1, out_axis=0)


def _normalise_mask(mask, new_norm):
    mask = jnp.asarray(mask)
    return mask / jnp.linalg.norm(mask) * new_norm**0.5


def symm_input_warning(x_shape, new_x_shape, name):
    warn_deprecation(
        (
            f"{len(x_shape)}-dimensional input to {name} layer is deprecated.\n"
            f"Input shape {x_shape} has been reshaped to {new_x_shape}, where "
            "the middle dimension encodes different input channels.\n"
            "Please provide a 3-dimensional input.\nThis warning will become an "
            "error in the future."
        )
    )


class DenseSymmMatrix(Module):
    r"""Implements a symmetrized linear transformation over a permutation group
    using matrix multiplication."""

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    features: int
    """The number of output features. Will be the second dimension of the output."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""
    dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.n_symm, self.n_sites = np.asarray(self.symmetries).shape
        if self.mask is not None:
            self.scaled_mask = _normalise_mask(self.mask, self.n_sites)

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the symmetrized linear transformation to the inputs along the last dimension.

        Args:
          x: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)
        # infer in_features and ensure input dimensions (batch, in_features,n_sites)

        # TODO: Deprecated: Eventually remove and error if less than 3 dimensions
        if x.ndim < 3:
            old_shape = x.shape
            if x.ndim == 1:
                x = jnp.expand_dims(x, (0, 1))
            elif x.ndim == 2:
                x = jnp.expand_dims(x, 1)
            symm_input_warning(old_shape, x.shape, "DenseSymm")

        in_features = x.shape[1]

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.n_sites),
            self.dtype,
        )

        if self.mask is not None:
            kernel = kernel * jnp.expand_dims(self.scaled_mask, (0, 1))

        # Converts the convolutional kernel of shape (self.features, in_features, n_sites)
        # to a full dense kernel of shape (self.features, in_features, n_symm, n_sites).
        # result[out, in, g, r] == kernel[out, in, g^{-1}r]
        kernel = jnp.take(kernel, jnp.asarray(self.symmetries), 2)
        kernel = jnp.asarray(kernel, dtype)

        # x is      (batches,       in_featuers,         n_sites)
        # kernel is (self.features, in_features, n_symm, n_sites)
        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 2, x.ndim - 1), (1, 3)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)

            # Convert symmetry-reduced bias of shape (features,) to the full bias of
            # shape (..., features, 1).
            bias = jnp.expand_dims(bias, 1)
            bias = jnp.asarray(bias, dtype)

            x += bias

        return x


class DenseSymmFFT(Module):
    r"""Implements a symmetrized projection onto a space group using a Fast Fourier Transform"""

    space_group: HashableArray
    """Array that lists the space group as permutations"""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: Tuple
    """Tuple that corresponds to shape of lattice"""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""
    dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        sg = np.asarray(self.space_group)

        self.n_cells = np.product(np.asarray(self.shape))
        self.n_point = len(sg) // self.n_cells
        self.sites_per_cell = sg.shape[1] // self.n_cells

        if self.mask is not None:
            self.scaled_mask = _normalise_mask(self.mask, sg.shape[1])

        # maps (n_sites) dimension of kernels to (sites_per_cell, n_point, *shape)
        # as used in FFT-based group convolution
        self.mapping = (
            sg[:, : self.sites_per_cell]
            .reshape(self.n_cells, self.n_point, self.sites_per_cell)
            .transpose(2, 1, 0)
            .reshape(self.sites_per_cell, self.n_point, *self.shape)
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        # TODO: Deprecated: Eventually remove and error if less than 3 dimensions
        # infer in_features and ensure input dimensions (batch, in_features,n_sites)
        if x.ndim < 3:
            old_shape = x.shape
            if x.ndim == 1:
                x = jnp.expand_dims(x, (0, 1))
            elif x.ndim == 2:
                x = jnp.expand_dims(x, 1)
            symm_input_warning(old_shape, x.shape, "DenseSymm")

        in_features = x.shape[1]

        x = x.reshape(*x.shape[:-1], self.n_cells, self.sites_per_cell)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:-1], *self.shape)

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.n_cells * self.sites_per_cell),
            self.dtype,
        )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask is not None:
            kernel = kernel * jnp.expand_dims(self.scaled_mask, (0, 1))

        # Converts the convolutional kernel of shape (features, in_features, n_sites)
        # to the expanded kernel of shape (features, in_features, sites_per_cell,
        # n_point, *shape) used in FFT-based group convolutions.
        kernel = kernel[..., self.mapping]

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        # TODO: the batch ordering should be revised: batch dimensions should
        # be leading
        x = lax.dot_general(
            x, kernel, (((1, 2), (1, 2)), ((3,), (4,))), precision=self.precision
        )
        x = x.transpose(1, 2, 3, 0)
        x = x.reshape(*x.shape[:3], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        x = x.transpose(0, 1, 3, 2).reshape(*x.shape[:2], -1)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)
            x += jnp.expand_dims(bias, (0, 2))

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real


class DenseEquivariantFFT(Module):
    r"""Implements a group convolution using a fast fourier transform over the translation group.
    The group convolution can be written in terms of translational convolutions with
    symmetry transformed filters as desribed in ` Cohen et. *al* <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    The translational convolutions are then implemented with Fast Fourier Transforms.
    """

    product_table: HashableArray
    """ product table for space group"""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    shape: Tuple
    """Tuple that corresponds to shape of lattice"""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""
    dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):

        pt = np.asarray(self.product_table)

        self.n_cells = np.product(np.asarray(self.shape))
        self.n_point = len(pt) // self.n_cells
        if self.mask is not None:
            self.scaled_mask = _normalise_mask(self.mask, len(pt))

        # maps (n_sites) dimension of kernels to (n_point, n_point, *shape)
        # as used in FFT-based group convolution
        self.mapping = (
            pt[: self.n_point]
            .reshape(self.n_point, self.n_cells, self.n_point)
            .transpose(0, 2, 1)
            .reshape(self.n_point, self.n_point, *self.shape)
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        in_features = x.shape[-2]

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        x = x.reshape(*x.shape[:-1], self.n_cells, self.n_point)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:-1], *self.shape)

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.n_point * self.n_cells),
            self.dtype,
        )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask is not None:
            kernel = kernel * jnp.expand_dims(self.scaled_mask, (0, 1))

        # Convert the convolutional kernel of shape (features, in_features, n_symm)
        # to the expanded kernel of shape (features, in_features, n_point(in),
        # n_point(out), *shape) used in FFT-based group convolutions
        kernel = kernel[..., self.mapping]

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        x = lax.dot_general(
            x, kernel, (((1, 2), (1, 2)), ((3,), (4,))), precision=self.precision
        )
        x = x.transpose(1, 2, 3, 0)
        x = x.reshape(*x.shape[:3], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:2], -1)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)
            x += jnp.expand_dims(bias, (0, 2))

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real


class DenseEquivariantIrrep(Module):
    r"""Implements a group convolutional layer by projecting onto irreducible
    representations of the group.

    Acts on a feature map of shape [batch_size, in_features, n_symm] and
    eeturns a feature map of shape [batch_size, features, n_symm].
    The input and the output are related by

    .. math ::

        y^{(i)}_g = \sum_{h,j} f^{(j)}_h W^{(ij)}_{h^{-1}g}.

    Note that this switches the convention of Cohen et al. to use an actual group
    convolution, but this doesn't affect equivariance.
    The convolution is implemented in terms of a group Fourier transform.
    Therefore, the group structure is represented internally as the set of its
    irrep matrices. After Fourier transforming, the convolution translates to

    .. math ::

        y^{(i)}_\rho = \sum_j f^{(j)}_\rho W^{(ij)}_\rho,

    where all terms are d x d matrices rather than numbers, and the juxtaposition
    stands for matrix multiplication.
    """

    irreps: Tuple[HashableArray]
    """Irrep matrices of the symmetry group. Each element of the list is an
    array of shape [n_symm, d, d]; irreps[i][j] is the representation of the
    jth group element in irrep #i."""
    features: int
    """The number of output features. Will be the second dimension of the output."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""

    dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        self.n_symm = self.irreps[0].shape[0]
        if self.mask is not None:
            self.scaled_mask = _normalise_mask(self.mask, self.n_symm)

        self.forward = jnp.concatenate(
            [jnp.asarray(irrep).reshape(self.n_symm, -1) for irrep in self.irreps],
            axis=1,
        )
        self.inverse = jnp.concatenate(
            [
                jnp.asarray(irrep).conj().reshape(self.n_symm, -1)
                * (irrep.shape[-1] / self.n_symm)
                for irrep in self.irreps
            ],
            axis=1,
        ).transpose()

        # Convert between vectors of length n_symm and tuples of arrays of shape
        # n_irrep × irrep_size^2
        self.assemble = lambda arrays: jnp.concatenate(
            [array.reshape(array.shape[:-3] + (-1,)) for array in arrays], axis=-1
        )

        irrep_size = 1
        n_same_size = 0
        shapes = []
        for irrep in self.irreps:
            if irrep_size == irrep.shape[-1]:
                n_same_size += 1
            else:
                shapes.append((n_same_size, irrep_size, irrep_size))
                irrep_size = irrep.shape[-1]
                n_same_size = 1
        shapes.append((n_same_size, irrep_size, irrep_size))
        limits = np.cumsum([0] + [np.prod(shape) for shape in shapes])

        self.disassemble = lambda vecs: tuple(
            vecs[..., limits[i] : limits[i + 1]].reshape(vecs.shape[:-1] + shape)
            for i, shape in enumerate(shapes)
        )

    def forward_ft(self, inputs: Array) -> Tuple[Array]:
        r"""Performs a forward group Fourier transform on the input.
        This is defined by

        .. math ::

            \hat{f}_\rho = \sum_g f(g) \rho(g),

        where :math:`\rho` is an irrep of the group.
        The Fourier transform is performed over the last index, and is returned
        as a tuple of arrays, each entry corresponding to the entry of `irreps`
        in the same position, and the last dimension of length `n_symm` replaced
        by two dimensions of length `d` each.
        """
        return self.disassemble(jnp.tensordot(inputs, self.forward, axes=1))

    def inverse_ft(self, inputs: Tuple[Array]) -> Array:
        r"""Performs an inverse group Fourier transform on the input.
        This is defined by

        .. math ::

            f(g) = \frac{1}{|G|} \sum_\rho d_\rho {\rm Tr}(\rho(g^{-1}) \hat{f}_\rho)

        where the sum runs over all irreps of the group.
        The input is a tuple of arrays whose the last two dimensions match the
        dimensions of each irrep. The inverse Fourier transform is performed
        over these indices and is returned as an array where those dimensions
        are replaced by a single dimension of length `n_symm`
        """
        return jnp.asarray(
            jnp.tensordot(self.assemble(inputs), self.inverse, axes=1),
            # Irrep matrices might be complex, so `result` might be complex
            # even if the inputs are real
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """
        in_features = x.shape[-2]

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        x = self.forward_ft(x)

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.n_symm),
            self.dtype,
        )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask is not None:
            kernel = kernel * jnp.expand_dims(self.scaled_mask, (0, 1))

        kernel = self.forward_ft(kernel)

        x = tuple(
            lax.dot_general(
                x[i], kernel[i], (((1, 4), (1, 3)), ((2,), (2,)))
            ).transpose(1, 3, 0, 2, 4)
            for i in range(len(x))
        )

        x = self.inverse_ft(x)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(bias, dtype)

            x += jnp.expand_dims(bias, (0, 2))

        if jnp.can_cast(x, dtype):
            return x
        else:
            return x.real


class DenseEquivariantMatrix(Module):
    r"""Implements a group convolution operation that is equivariant over a symmetry group
    by multiplying by the full kernel matrix"""

    product_table: HashableArray
    """Flattened product table generated by PermutationGroup.produt_table().ravel()
    that specifies the product of the group with its involution, or the
    PermutationGroup object itself"""
    features: int
    """The number of symmetry-reduced output features. The full output size
    is n_symm*features."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Whether to mask the filters to restrict the connectivity"""
    dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = default_equivariant_initializer
    """Initializer for the kernel. Defaults to Lecun normal."""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias. Defaults to zero initialization."""

    def setup(self):
        self.n_symm = np.asarray(self.product_table).shape[0]
        if self.mask is not None:
            self.scaled_mask = _normalise_mask(self.mask, self.n_symm)

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last dimension.
        Args:
          x: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        in_features = x.shape[-2]

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, in_features, self.n_symm),
            self.dtype,
        )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask is not None:
            kernel = kernel * jnp.expand_dims(self.scaled_mask, (0, 1))

        # Converts the convolutional kernel of shape (features, in_features, n_symm)
        # to a full dense kernel of shape (features, in_features, n_symm, n_symm)
        # result[out, in, g, h] == kernel[out, in, g^{-1}h]
        # input dimensions are [in, g], output dimensions are [out, h]
        kernel = jnp.take(kernel, jnp.asarray(self.product_table), 2)
        kernel = jnp.asarray(kernel, dtype)

        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 2, x.ndim - 1), (1, 2)), ((), ())),
            precision=self.precision,
        )

        x = x.reshape(-1, self.features, self.n_symm)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            x += jnp.expand_dims(bias, 1)

        return x


def DenseSymm(symmetries, point_group=None, mode="auto", shape=None, **kwargs):
    r"""
    Implements a projection onto a symmetry group. The output will be
    equivariant with respect to the symmetry operations in the group and can
    be averaged to produce an invariant model.

    This layer maps an input of shape `(..., in_features, n_sites)` to an
    output of shape `(..., features, num_symm)`.

    Note: The output shape has changed to seperate the feature and symmetry
    dimensions. The previous shape was [num_samples, num_symm*features] and
    the new shape is [num_samples, features, num_symm]

    Args:
        symmetries: A specification of the symmetry group. Can be given by a
            :ref:`netket.graph.Graph`, a :ref:`netket.utils.group.PermutationGroup`, or an array
            of shape :code:`(n_symm, n_sites)`. A :ref:`netket.utils.HashableArray` may also
            be passed.
            specifying the permutations corresponding to symmetry transformations
            of the lattice.
        point_group: The point group, from which the space group is built.
            If symmetries is a graph the default point group is overwritten.
        mode: string "fft, matrix, auto" specifying whether to use a fast Fourier
            transform, matrix multiplication, or to choose a sensible default
            based on the symmetry group.
        shape: A tuple specifying the dimensions of the translation group.
        features: The number of output features. The full output shape
            is [n_batch,features,n_symm].
        use_bias: A bool specifying whether to add a bias to the output (default: True).
        mask: An optional array of shape [n_sites] consisting of ones and zeros
            that can be used to give the kernel a particular shape.
        dtype: The datatype of the weights. Defaults to a 64bit float.
        precision: Optional argument specifying numerical precision of the computation.
            see `jax.lax.Precision`for details.
        kernel_init: Optional kernel initialization function. Defaults to variance scaling.
        bias_init: Optional bias initialization function. Defaults to zero initialization.
    """
    if isinstance(symmetries, Lattice) and (
        point_group is not None or symmetries._point_group is not None
    ):
        shape = tuple(symmetries.extent)
        sym = HashableArray(np.asarray(symmetries.space_group(point_group)))
        if mode == "auto":
            mode = "fft"
    elif isinstance(symmetries, Graph):
        if mode == "fft":
            raise ValueError(
                "When requesting 'mode=fft' a valid point group must be specified"
                "in order to construct the space group"
            )
        sym = HashableArray(np.asarray(symmetries.automorphisms()))
    elif isinstance(symmetries, HashableArray):
        sym = symmetries
    else:
        sym = HashableArray(np.asarray(symmetries))

    if mode == "fft":
        if shape is None:
            raise TypeError(
                "When requesting `mode=fft`, the shape of the translation group must be specified. "
                "Either supply the `shape` keyword argument or pass a `netket.graph.Graph` object to "
                "the symmetries keyword argument."
            )
        else:
            return DenseSymmFFT(sym, shape=shape, **kwargs)
    elif mode in ["matrix", "auto"]:
        return DenseSymmMatrix(sym, **kwargs)
    else:
        raise ValueError(
            f"Unknown mode={mode}. Valid modes are 'fft', 'matrix', or 'auto'."
        )


def DenseEquivariant(
    symmetries,
    features: int = None,
    mode="auto",
    shape=None,
    point_group=None,
    in_features=None,
    **kwargs,
):
    r"""A group convolution operation that is equivariant over a symmetry group.

    Acts on a feature map of symmetry poses of shape [num_samples, in_features, num_symm]
    and returns a feature  map of poses of shape [num_samples, features, num_symm]

    G-convolutions are described in ` Cohen et. {\it al} <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    and applied to quantum many-body problems in ` Roth et. {\it al} <https://arxiv.org/pdf/2104.05085.pdf>`_

    The G-convolution generalizes the convolution to non-commuting groups:

    .. math ::

        C^i_g = \sum_h {\bf W}_{g^{-1} h} \cdot {\bf f}_h

    Group elements that differ by the same symmetry operation (i.e. :math:`g = xh`
    and :math:`g' = xh'`) are connected by the same filter.

    This layer maps an input of shape `(..., in_features, n_sites)` to an
    output of shape `(..., features, num_symm)`.

    Args:
        symmetries: A specification of the symmetry group. Can be given by a
            nk.graph.Graph, an nk.utils.PermuationGroup, a list of irreducible
            representations or a product table.
        point_group: The point group, from which the space group is built.
            If symmetries is a graph the default point group is overwritten.
        mode: string "fft, irreps, matrix, auto" specifying whether to use a fast
            fourier transform over the translation group, a fourier transform using
            the irreducible representations or by constructing the full kernel matrix.
        shape: A tuple specifying the dimensions of the translation group.
        features: The number of output features. The full output shape
            is [n_batch,features,n_symm].
        use_bias: A bool specifying whether to add a bias to the output (default: True).
        mask: An optional array of shape [n_sites] consisting of ones and zeros
            that can be used to give the kernel a particular shape.
        dtype: The datatype of the weights. Defaults to a 64bit float.
        precision: Optional argument specifying numerical precision of the computation.
            see `jax.lax.Precision`for details.
        kernel_init: Optional kernel initialization function. Defaults to variance scaling.
        bias_init: Optional bias initialization function. Defaults to zero initialization.
    """
    # deprecate in_features
    if in_features is not None:
        warn_deprecation(
            (
                "`in_features` is now automatically detected from the input and deprecated."
                "Please remove it when calling `DenseEquivariant`."
            )
        )
    if "out_features" in kwargs:
        warn_deprecation(
            "`out_features` has been renamed to `features` and the old name is "
            "now deprecated. Please update your code."
        )
        if features is not None:
            raise ValueError(
                "You must only specify `features`. `out_features` is deprecated."
            )
        features = kwargs.pop("out_features")

    if features is None:
        raise ValueError("`features` not specified (the number of output features).")

    kwargs["features"] = features

    if isinstance(symmetries, Lattice) and (
        point_group is not None or symmetries._point_group is not None
    ):
        shape = tuple(symmetries.extent)
        # With graph try to find point group, otherwise default to automorphisms
        sg = symmetries.space_group(point_group)
        if mode == "auto":
            mode = "fft"
    elif isinstance(symmetries, Graph):
        sg = symmetries.automorphisms()
        if mode == "auto":
            mode = "irreps"
        elif mode == "fft":
            raise ValueError(
                "When requesting 'mode=fft' a valid point group must be specified"
                "in order to construct the space group"
            )
    elif isinstance(symmetries, PermutationGroup):
        # If we get a group and default to irrep projection
        if mode == "auto":
            mode = "irreps"
        sg = symmetries

    elif isinstance(symmetries, Sequence):
        if mode not in ["irreps", "auto"]:
            raise ValueError("Specification of symmetries incompatible with mode")
        return DenseEquivariantIrrep(symmetries, **kwargs)
    else:
        if symmetries.ndim == 2 and symmetries.shape[0] == symmetries.shape[1]:
            if mode == "irreps":
                raise ValueError("Specification of symmetries incompatible with mode")
            elif mode == "matrix":
                return DenseEquivariantMatrix(symmetries, **kwargs)
            else:
                if shape is None:
                    raise TypeError(
                        "When requesting `mode=fft`, the shape of the translation group must be specified. "
                        "Either supply the `shape` keyword argument or pass a `netket.graph.Graph` object to "
                        "the symmetries keyword argument."
                    )
                else:
                    return DenseEquivariantFFT(symmetries, shape=shape, **kwargs)
        return ValueError("Invalid Specification of Symmetries")

    if mode == "fft":
        if shape is None:
            raise TypeError(
                "When requesting `mode=fft`, the shape of the translation group must be specified. "
                "Either supply the `shape` keyword argument or pass a `netket.graph.Graph` object to "
                "the symmetries keyword argument."
            )
        else:
            return DenseEquivariantFFT(
                HashableArray(sg.product_table), shape=shape, **kwargs
            )
    elif mode in ["irreps", "auto"]:
        irreps = tuple(HashableArray(irrep) for irrep in sg.irrep_matrices())
        return DenseEquivariantIrrep(irreps, **kwargs)
    elif mode == "matrix":
        return DenseEquivariantMatrix(HashableArray(sg.product_table), **kwargs)
    else:
        raise ValueError(
            f"Unknown mode={mode}. Valid modes are 'fft', 'matrix', 'irreps' or 'auto'."
        )
