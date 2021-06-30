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

from typing import Any, Callable, Union, Optional, Tuple

from flax.linen.module import Module, compact
from jax import lax
import jax.numpy as jnp
import numpy as np

from netket.nn.initializers import normal, zeros, unit_normal_scaling
from netket.utils import HashableArray
from netket.utils.types import Array, DType, PRNGKeyT, Shape, NNInitFunc
from netket.utils.group import PermutationGroup
from netket.graph import Graph
import warnings


def _symmetrizer_col(perms, features):
    """
    Creates the mapping from symmetry-reduced kernel w to full kernel W, s.t.
        W[ij] = S[ij][kl] w[kl]
    where [ij] ∈ [0,...,n_sites×n_hidden) and [kl] ∈ [0,...,n_sites×features).
    For each [ij] there is only one [kl] such that S[ij][kl] is non-zero, in which
    case S[ij][kl] == 1. Thus, this method only returns the array of indices `col`
    of shape (n_sites×n_hidden,) satisfying
        W[ij] = w[col[ij]]  <=>  W = w[col].

    See test/models/test_nn.py:test_symmetrizer for how this relates to the
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


class DenseSymmMatrix(Module):
    """Implements a symmetrized linear transformation over a permutation group
    using matrix multiplication."""

    symmetries: HashableArray
    """A group of symmetry operations (or array of permutation indices) over which the layer should be invariant.
        Numpy/Jax arrays must be wrapped into an :class:`netket.utils.HashableArray`.
    """
    features: int
    """The number of symmetry-reduced features. The full output size is len(symmetries) * features."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""
    dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: NNInitFunc = unit_normal_scaling()
    """Initializer for the Dense layer matrix. Defaults to variance scaling"""
    bias_init: NNInitFunc = zeros()
    """Initializer for the bias. Defaults to zero initialization"""

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
        return jnp.expand_dims(bias, (0, 2))

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

        kernel = self.param(
            "kernel", self.kernel_init, (x.shape[-1], self.features), self.dtype
        )

        if self.mask:
            kernel = kernel * jnp.expand_dims(self.mask, 1)

        kernel = self.full_kernel(kernel).reshape(-1, self.features, self.n_symm)
        kernel = jnp.asarray(kernel, dtype)

        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,), self.dtype)
            bias = jnp.asarray(self.full_bias(bias), dtype)
            x += bias

        return x


class DenseSymmFFT(Module):
    """Implements a symmetrized projection onto a space group using a Fast Fourier Transform"""

    space_group: HashableArray
    """Array that lists the space group as permutations"""
    features: int
    """The number of input features; must be the second dimension of the input."""
    shape: Tuple
    """Tuple that corresponds to shape of lattice"""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""
    dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None

    kernel_init: NNInitFunc = unit_normal_scaling()
    """Initializer for the Dense layer matrix. Defaults to variance scaling"""
    bias_init: NNInitFunc = zeros()
    """Initializer for the bias. Defaults to zero initialization"""

    def setup(self):
        self.n_cells = np.product(np.asarray(self.shape))
        self.n_symm = len(np.asarray(self.space_group)) // self.n_cells
        self.sites_per_cell = np.asarray(self.space_group).shape[1] // self.n_cells

        self.mapping = (
            np.asarray(self.space_group)[:, : self.sites_per_cell]
            .reshape(self.n_cells, self.n_symm, self.sites_per_cell)
            .transpose(1, 2, 0)
            .reshape(self.n_symm, self.sites_per_cell, *self.shape)
        )

    def make_kernel(self, kernel):
        kernel = kernel[..., self.mapping]

        return kernel

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        x = (
            x.reshape(-1, self.n_cells, self.sites_per_cell)
            .transpose(0, 2, 1)
            .reshape(-1, self.sites_per_cell, *self.shape)
        )

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.features, self.n_cells * self.sites_per_cell),
            self.dtype,
        )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask:
            kernel = kernel * jnp.expand_dims(self.mask, 0)

        kernel = self.make_kernel(kernel)

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:2], self.n_cells)

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:3], self.n_cells
        )

        x = lax.dot_general(
            x, kernel, (((1,), (2,)), ((2,), (3,))), precision=self.precision
        )
        x = x.transpose(1, 2, 3, 0)
        x = x.reshape(*x.shape[:3], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:3], self.n_cells)
        x = x.transpose(0, 1, 3, 2).reshape(*x.shape[:2], -1)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features, 1), self.dtype)
            bias = jnp.asarray(bias, dtype)
            x += bias

        return x


class DenseEquivariantFFT(Module):
    """Implements a group convolution over a space group using a Fast Fourier Transform
    over the translation group"""

    product_table: HashableArray
    """ product table for space group"""
    in_features: int
    """The number of input features; must be the second dimension of the input."""
    out_features: int
    """The number of input features; must be the second dimension of the input."""
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
    
    kernel_init: NNInitFunc = unit_normal_scaling()
    """Initializer for the Dense layer matrix. Defaults to variance scaling"""
    bias_init: NNInitFunc = zeros()
    """Initializer for the bias. Defaults to zero initialization"""

    def setup(self):

        self.n_cells = np.product(np.asarray(self.shape))
        self.n_symm = len(np.asarray(self.product_table)) // self.n_cells

        self.mapping = (
            np.asarray(self.product_table)[: self.n_symm]
            .reshape(self.n_symm, self.n_cells, self.n_symm)
            .transpose(0, 2, 1)
            .reshape(self.n_symm, self.n_symm, *self.shape)
        )

    def make_kernel(self, kernel):
        kernel = kernel[..., self.mapping]

        return kernel

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        x = x.reshape(*x.shape[:-1], self.n_cells, self.n_symm).transpose(0, 1, 3, 2)
        x = x.reshape(*x.shape[:-1], *self.shape)


        kernel = self.param(
            "kernel",
            self.kernel_init,
            (
                self.out_features,
                self.in_features,
                self.n_symm * self.n_cells,
            ),
            self.dtype,
        )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask:
            kernel = kernel * jnp.expand_dims(self.mask, (0, 1))

        kernel = self.make_kernel(kernel)

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
        x = x.transpose(0, 1, 3, 2).reshape(*x.shape[:2], -1)

        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.out_features, 1), self.dtype)
            bias = jnp.asarray(bias, dtype)
            x += bias

        return x


class DenseEquivariantIrrep(Module):
    """Implements a group convolutional layer by projecting onto irreducible
    represesntations of the space group."""

    """Acts on a feature map of shape [batch_size, in_features, n_symm] and 
    eeturns a feature map of shape [batch_size, out_features, n_symm]. 
    The input and the output are related by
    :: math ::
        y^{(i)}_g = \sum_{h,j} f^{(j)}_h W^{(ij)}_{h^{-1}g}.
    Note that this switches the convention of Cohen et al. to use an actual group
    convolution, but this doesn't affect equivariance.
    The convolution is implemented in terms of a group Fourier transform.
    Therefore, the group structure is represented internally as the set of its
    irrep matrices. After Fourier transforming, the convolution translates to
    :: math ::
        y^{(i)}_\rho = \sum_j f^{(j)}_\rho W^{(ij)}_\rho,
    where all terms are d x d matrices rather than numbers, and the juxtaposition
    stands for matrix multiplication.
    """

    irreps: Tuple[HashableArray]
    """Irrep matrices of the symmetry group. Each element of the list is an
    array of shape [n_symm, d, d]; irreps[i][j] is the representation of the
    jth group element in irrep #i."""
    in_features: int
    """The number of input features; must be the second dimension of the input."""
    out_features: int
    """The number of output features; returned as the second dimension of the output."""
    use_bias: bool = True
    """Whether to add a bias to the output (default: True)."""
    mask: Optional[HashableArray] = None
    """Mask that zeros out elements of the filter. Should be of shape [inp.shape[-1]]"""

    dtype: DType = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: Optional[NNInitFunc] = None
    """Initializer for the Dense layer matrix."""
    bias_init: Optional[NNInitFunc] = None
    """Initializer for the bias."""

    def setup(self):
        self.n_symm = self.irreps[0].shape[0]
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

        self.stdev = 1.0 / np.sqrt(self.in_features * self.n_symm)

    def forward_ft(self, inputs: Array, dtype: DType) -> Tuple[Array]:
        """Performs a forward group Fourier transform on the input.
        This is defined by
        :: math ::
            \hat{f}_\rho = \sum_g f(g) \rho(g),
        where :math:`\rho` is an irrep of the group.
        The Fourier transform is performed over the last index, and is returned
        as a tuple of arrays, each entry corresponding to the entry of `irreps`
        in the same position, and the last dimension of length `n_symm` replaced
        by two dimensions of length `d` each.
        """
        return self.disassemble(jnp.tensordot(inputs, self.forward, axes=1))

    def inverse_ft(self, inputs: Tuple[Array], dtype: DType) -> Array:
        """Performs an inverse group Fourier transform on the input.
        This is defined by
        :: math ::
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
            dtype=dtype,
        )

    @compact
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last two
        dimensions (-2: features, -1: group elements)
        """

        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        x = self.forward_ft(x, dtype=dtype)

        if self.kernel_init:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (self.in_features, self.out_features, self.n_symm),
                self.dtype,
            )

        else:
            kernel = self.param(
                "kernel",
                normal(self.stdev),
                (self.in_features, self.out_features, self.n_symm),
                self.dtype,
            )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask:
            kernel = kernel * jnp.expand_dims(self.mask, (0, 1))

        kernel = self.forward_ft(kernel, dtype=dtype)

        x = tuple(
            lax.dot_general(
                x[i], kernel[i], (((1, 4), (0, 3)), ((2,), (2,)))
            ).transpose(1, 3, 0, 2, 4)
            for i in range(len(x))
        )

        x = self.inverse_ft(x, dtype)

        if self.use_bias:
            if self.bias_init:
                bias = self.param("bias", self.bias_init, (self.out_features, 1), dtype)
            else:
                bias = self.param("bias", zeros, (self.out_features, 1), dtype)
            bias = jnp.asarray(bias, dtype)

            x += bias

        return x


class DenseEquivariantMatrix(Module):
    r"""Implements a group convolution operation that is equivariant over a symmetry group
    by multiplying by the full kernel matrix"""

    product_table: HashableArray
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
    mask: Optional[HashableArray] = None
    """Whether to mask the filters to restrict the connectivity"""
    dtype: Any = jnp.float64
    """The dtype of the weights."""
    precision: Any = None
    """numerical precision of the computation see `jax.lax.Precision`for details."""

    kernel_init: Optional[NNInitFunc] = None
    """Initializer for the Dense layer matrix."""
    bias_init: Optional[NNInitFunc] = None
    """Initializer for the bias."""

    def setup(self):

        self.n_symm = np.asarray(self.product_table).shape[0]

        self.stdev = 1.0 / np.sqrt(self.in_features * self.n_symm)

    def full_kernel(self, kernel):
        """
        Converts the symmetry-reduced kernel of shape (n_sites, features) to
        the full Dense kernel of shape (n_sites, features * n_symm).
        """

        result = jnp.take(kernel, jnp.asarray(self.product_table).ravel(), 0)
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
    def __call__(self, x: Array) -> Array:
        """Applies the equivariant transform to the inputs along the last dimension.
        Args:
          x: The nd-array to be transformed.
        Returns:
          The transformed input.
        """
        dtype = jnp.promote_types(x.dtype, self.dtype)
        x = jnp.asarray(x, dtype)

        x = x.reshape(-1, x.shape[1] * x.shape[2])

        if self.kernel_init:
            kernel = self.param(
                "kernel",
                self.kernel_init,
                (x.shape[-1], self.in_features, self.out_features),
                self.dtype,
            )
        else:
            kernel = self.param(
                "kernel",
                normal(self.stdev),
                (x.shape[-1], self.in_features, self.out_features),
                self.dtype,
            )

        kernel = jnp.asarray(kernel, dtype)

        if self.mask:
            kernel = kernel * jnp.expand_dims(self.mask, (1, 2))

        kernel = self.full_kernel(kernel)
        kernel = jnp.asarray(kernel, dtype)

        x = lax.dot_general(
            x,
            kernel,
            (((x.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )

        if self.use_bias:
            if self.bias_init:
                bias = self.param(
                    "bias", self.bias_init, (self.out_features,), self.dtype
                )
            else:
                bias = self.param("bias", zeros, (self.out_features,), self.dtype)
            bias = jnp.asarray(self.full_bias(bias), dtype)
            x += bias

        return x


def DenseSymm(symmetry_info, mode="auto", **kwargs):
    """
    Implements a projection onto a symmetry group. The output will be
    equivariant with respect to the symmetry operations in the group and can
    be averaged to produce an invariant model.

    Args:
        symmetry_info: A specification of the symmetry group. Can be given by a nk.graph.Graph,
        a nk.utils.PermuationGroup, or an array [n_symm, n_sites] specifying the permutations
        corresponding to symmetry transformations of the lattice.
        mode: string "fft, matrix, auto" specifying whether to use a fast fourier transform,
        matrix multiplication, or to make the choice based on the symmetry group
        multiplication to do the computation
        features: The number of symmetry-reduced features. The full output size is n_symm*features.
        use_bias: A bool specifying whether to add a bias to the output (default: True).
        mask: An optional array of shape [n_sites] consisting of ones and zeros that can be used
        to give the kernel a particular shape
        dtype: The datatype of the weights. Defaults to a 64bit float
        precision: Optional argument specifying numerical precision of the computation
        see `jax.lax.Precision`for details.
        kernel_init: Optional kernel initialization function. Defaults to variance scaling
        bias_init: Optional bias initialization function. Defaults to zero initialization
    """

    if isinstance(symmetry_info, Graph):
        try:
            symmetries = HashableArray(np.asarray(symmetry_info.space_group()))
            if mode == "auto" or mode == "fft":
                mode = "fft"
                kwargs["shape"] = symmetry_info.extent
        except:
            if mode == "fft":
                warnings.warn(
                    "Graph without a space group specified. Switching to matrix implementation",
                )
                mode = "matrix"
            symmetries = HashableArray(np.asarray(symmetry_info.automorphisms()))
    else:
        try:
            symmetries = HashableArray(np.asarray(symmetry_info))
        except:
            raise ValueError("Invalid specification of symmetries")

    if mode == "fft":
        if "shape" in kwargs:
            return DenseSymmFFT(symmetries, **kwargs)
        else:
            raise KeyError(
                "Must pass keyword argument shape which specifies the shape of the translation group"
            )
    else:
        if "shape" in kwargs:
            del kwargs["shape"]
        return DenseSymmMatrix(symmetries, **kwargs)


def DenseEquivariant(symmetry_info, mode="auto", **kwargs):
    r"""A group convolution operation that is equivariant over a symmetry group

    Acts on a feature map of symmetry poses of shape [batch_size,n_symm*in_features]
    and returns a feature  map of poses of shape [batch_size,n_symm*out_features]

    G-convolutions are described in ` Cohen et. {\it al} <http://proceedings.mlr.press/v48/cohenc16.pdf>`_
    and applied to quantum many-body problems in ` Roth et. {\it al} <https://arxiv.org/pdf/2104.05085.pdf>`_

    The G-convolution generalizes the convolution to non-commuting groups:

    .. math ::

        C^i_g = \sum_h {\bf W}_{g^{-1} h} \cdot {\bf f}_h

    Group elements that differ by the same symmetry operation (i.e. :math:`g = xh`
    and :math:`g' = xh'`) are connected by the same filter.

    Args:
        symmetry_info: A specification of the symmetry group. Can be given by a nk.graph.Graph,
        or the product table of a nk.utils.PermuationGroup.
        mode: string "fft, irreps, matrix, auto" specifying whether to use a fast fourier transform
        over the translation group, a fourier transform using the irreducible representations or
        by constructing the full kernel matrix. Auto will choose the fastest model based on the
        information specified
        features: The number of symmetry-reduced features. The full output size is n_symm*features.
        use_bias: A bool specifying whether to add a bias to the output (default: True).
        mask: An optional array of shape [n_sites] consisting of ones and zeros that can be used
        to give the kernel a particular shape
        dtype: The datatype of the weights. Defaults to a 64bit float
        precision: Optional argument specifying numerical precision of the computation
        see `jax.lax.Precision`for details.
        kernel_init: Optional kernel initialization function. Defaults to variance scaling
        bias_init: Optional bias initialization function. Defaults to zero initialization
    """

    if isinstance(symmetry_info, Graph):
        try:
            # If we can find a space group default to fast fourier transforms
            sg = symmetry_info.space_group()
            if mode == "irreps":
                symmetries = tuple(
                    HashableArray(irrep) for irrep in sg.irrep_matrices()
                )
            elif mode == "matrix":
                symmetries = HashableArray(sg.product_table)
            else:
                mode = "fft"
                symmetries = HashableArray(sg.product_table)
                if not "shape" in kwargs:
                    kwargs["shape"] = symmetry_info.extent
        except:
            # If we can't find a space group use the irrep projection
            sg = symmetry_info.automorphisms()
            if mode == "fft":
                warnings.warn(
                    "Graph without a space group specified. Switching to matrix implementation",
                )
                mode = "irreps"
                symmetries = tuple(
                    HashableArray(irrep) for irrep in sg.irrep_matrices()
                )
            elif mode == "matrix":
                symmetries = HashableArray(sg.product_table)
            else:
                mode = "irreps"
                symmetries = tuple(
                    HashableArray(irrep) for irrep in sg.irrep_matrices()
                )

    elif isinstance(symmetry_info, PermutationGroup):
        # If we get a group (and no other info) default to irrep projection
        if mode == "fft":
            symmetries = HashableArray(symmetry_info.product_table)
        elif mode == "matrix":
            symmetries = HashableArray(sg.product_table)
        else:
            mode = "irreps"
            symmetries = tuple(
                HashableArray(irrep) for irrep in symmetry_info.irrep_matrices()
            )
    else:
        try:
            # If we get an array, make sure it's a valid product table
            arr = np.asarray(symmetry_info)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                if mode == "fft":
                    symmetries = HashableArray(arr)
                if mode == "matrix" or mode == "auto":
                    mode == "matrix"
                    symmetries = HashableArray(arr)
                if mode == "irreps":
                    raise ValueError(
                        "For irreps mode, please pass the irreducible representations as a list generated by calling group.irrep_matrices()"
                    )
            else:
                raise ValueError("Invalid product table shape")
        except:
            # If we get something other than a Graph, PermutationGroup or Array it should be a list of irreps
            if isinstance(symmetry_info, list) or isinstance(symmetry_info, tuple):
                if mode == "auto" or mode == "irreps":
                    mode = "irreps"
                    symmetries = tuple(HashableArray(irrep) for irrep in symmetry_info)
                else:
                    raise ValueError("Specify mode irreps to use irrep matrices")
            else:
                # Error message if we can't be more specific
                raise ValueError("Invalid specification of symmetries")

    if mode == "fft":
        # Need to specify shape of translation group to do FFT
        if "shape" in kwargs:
            return DenseEquivariantFFT(symmetries, **kwargs)
        else:
            raise KeyError(
                "Must pass keyword argument shape which specifies the shape of the translation group"
            )
    else:
        if "shape" in kwargs:
            del kwargs["shape"]
        if mode == "irreps":
            return DenseEquivariantIrrep(symmetries, **kwargs)
        else:
            return DenseEquivariantMatrix(symmetries, **kwargs)
