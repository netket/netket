# Copyright 2024 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.nn.initializers import normal

from netket.hilbert import HomogeneousHilbert
from netket.utils.types import NNInitFunc
from netket.jax import dtype_complex
from netket.utils.types import DType

default_kernel_init = normal(stddev=0.01)


class MPSPeriodic(nn.Module):
    r"""
    A periodic Matrix Product State (MPS) for a quantum state of discrete
    degrees of freedom.
    The MPS is defined as

    .. math:: \Psi(s_1,\dots s_N) = \mathrm{Tr} \left[ A[s_1] \dots A[s_N] \right] ,

    for arbitrary local quantum numbers :math:`s_i`, where :math:`A[s_i]` are matrices of shape :math:`(\text{bond_dim}, \text{bond_dim})` for  :math:`i=1, \dots N` , depending on the value of the local quantum number :math:`s_i`.
    """

    hilbert: HomogeneousHilbert
    """Hilbert space on which the state is defined."""
    bond_dim: int
    """Bond dimension of the MPS tensors. See formula above."""
    symperiod: Optional[int] = None
    """
    Periodicity in the chain of MPS tensors.
    The chain of MPS tensors is constructed as a sequence of identical
    unit cells consisting of symperiod tensors. if None, symperiod equals the
    number of physical degrees of freedom.
    """
    unroll: int = 1
    """the number of scan iterations to unroll within a single iteration of a loop."""
    checkpoint: bool = True
    """Whether to use jax.checkpoint on the scan function for memory efficiency."""
    kernel_init: NNInitFunc = default_kernel_init
    """the initializer for the MPS weights. This is added to an identity tensor."""
    param_dtype: DType = float
    """complex or float, whether the variational parameters of the MPS are real or complex."""

    def setup(self):
        L, d, D = self.hilbert.size, self.hilbert.local_size, self.bond_dim
        self._L, self._d, self._D = L, d, D

        # determine the shape of the unit cell
        if self.symperiod is None:
            self._symperiod = L
        else:
            self._symperiod = self.symperiod

        if L % self._symperiod == 0 and self._symperiod > 0:
            unit_cell_shape = (self._symperiod, d, D, D)
        else:
            raise AssertionError(
                "The number of degrees of freedom of the Hilbert space needs to be a multiple of the period of the MPS"
            )

        # define diagonal tensors with correct unit cell shape
        iden_tensors = jnp.repeat(
            jnp.eye(D, dtype=self.param_dtype)[jnp.newaxis, :, :],
            self._symperiod * d,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(self._symperiod, d, D, D)

        self.tensors = (
            self.param("tensors", self.kernel_init, unit_cell_shape, self.param_dtype)
            + iden_tensors
        )

    def __call__(self, x):
        """
        Queries the tensor network for the input configurations x.

        Args:
            x: the input configuration
        """
        x = jnp.atleast_1d(x)
        assert x.shape[-1] == self.hilbert.size

        x_shape = x.shape
        if jnp.ndim(x) != 2:
            x = jax.lax.collapse(x, 0, -1)

        qn_x = self.hilbert.states_to_local_indices(x)

        ψ = jax.vmap(self.contract_mps, in_axes=0)(qn_x)
        ψ = ψ.reshape(x_shape[:-1])

        return jnp.log(ψ.astype(dtype_complex(self.param_dtype)))

    def contract_mps(self, qn):
        """
        Internal function, used to contract the tensor network with some input
        tensor.

        Args:
            qn: The input tensor to be contracted with this MPS
        """
        # create all tensors in mps from unit cell
        all_tensors = jnp.tile(self.tensors, (self._L // self._symperiod, 1, 1, 1))

        edge = jnp.eye(self._D, dtype=self.param_dtype)

        def base_scan_func(edge, pair):
            tensor, qn = pair
            edge = edge @ tensor[qn, :]
            return edge, None

        # Apply jax.checkpoint conditionally to the base_scan_func
        scan_func = (
            jax.checkpoint(base_scan_func) if self.checkpoint else base_scan_func
        )

        edge, _ = jax.lax.scan(scan_func, edge, (all_tensors, qn), unroll=self.unroll)

        return jnp.trace(edge)


class MPSOpen(nn.Module):
    r"""
    An open Matrix Product State (MPS) for a quantum state of discrete
    degrees of freedom.
    The MPS is defined as

    .. math:: \Psi(s_1,\dots s_N) = \mathrm{Tr} \left[ A[s_1]\dots A[s_N] \right] ,

    for arbitrary local quantum numbers :math:`s_i`, where :math:`A[s_{i}]` are vectors of shape :math:`(\text{bond_dim},)` for :math:`i=1,N` and matrices
    of shape :math:`(\text{bond_dim}, \text{bond_dim})` for  :math:`i=2, \dots N-1` , depending on the value of the local quantum number :math:`s_i`.
    """

    hilbert: HomogeneousHilbert
    """Hilbert space on which the state is defined."""
    bond_dim: int
    """Bond dimension of the MPS tensors. See formula above."""
    unroll: int = 1
    """the number of scan iterations to unroll within a single iteration of a loop."""
    checkpoint: bool = True
    """Whether to use jax.checkpoint on the scan function for memory efficiency."""
    kernel_init: NNInitFunc = default_kernel_init
    """the initializer for the MPS weights. This is added to an identity tensor."""
    param_dtype: DType = float
    """complex or float, whether the variational parameters of the MPS are real or complex."""

    def setup(self):
        L, d, D = self.hilbert.size, self.hilbert.local_size, self.bond_dim
        self._L, self._d, self._D = L, d, D
        self.param_dtype_cplx = dtype_complex(self.param_dtype)

        iden_boundary_tensor = jnp.ones((d, D), dtype=self.param_dtype)
        self.left_tensors = (
            self.param("left_tensors", self.kernel_init, (d, D), self.param_dtype)
            + iden_boundary_tensor
        )
        self.right_tensors = (
            self.param("right_tensors", self.kernel_init, (d, D), self.param_dtype)
            + iden_boundary_tensor
        )

        # determine the shape of the unit cell
        unit_cell_shape = (L - 2, d, D, D)

        iden_tensors = jnp.repeat(
            jnp.eye(D, dtype=self.param_dtype)[jnp.newaxis, :, :],
            (L - 2) * d,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(L - 2, d, D, D)
        self.middle_tensors = (
            self.param(
                "middle_tensors", self.kernel_init, unit_cell_shape, self.param_dtype
            )
            + iden_tensors
        )

    def __call__(self, x):
        """
        Queries the tensor network for the input configurations x.

        Args:
            x: the input configuration
        """
        x = jnp.atleast_1d(x)
        assert x.shape[-1] == self.hilbert.size
        x_shape = x.shape
        if x.ndim != 2:
            x = jax.lax.collapse(x, 0, -1)
        qn_x = self.hilbert.states_to_local_indices(x)

        ψ = jax.vmap(self.contract_mps)(qn_x)
        ψ = ψ.reshape(x_shape[:-1])

        return jnp.log(ψ.astype(dtype_complex(self.param_dtype)))

    def contract_mps(self, qn):
        """
        Internal function, used to contract the tensor network with some input
        tensor.

        Args:
            qn: The input tensor to be contracted with this MPS
        """

        edge = self.left_tensors[qn[0], :]

        def base_scan_func(edge, pair):
            tensor, qn = pair
            edge = edge @ tensor[qn, :]
            return edge, None

        # Apply jax.checkpoint conditionally to the base_scan_func
        scan_func = (
            jax.checkpoint(base_scan_func) if self.checkpoint else base_scan_func
        )

        edge, _ = jax.lax.scan(
            scan_func, edge, (self.middle_tensors, qn[1:-1]), unroll=self.unroll
        )
        psi = self.right_tensors[qn[-1], :] @ edge
        return psi
