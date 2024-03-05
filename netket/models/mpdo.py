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
# limitations under the License.Mon

from typing import Any, Optional

from flax import linen as nn
import jax
from jax import numpy as jnp

from jax.nn.initializers import normal

from netket.hilbert import HomogeneousHilbert
from netket.utils.types import NNInitFunc
from netket.jax import dtype_complex

default_kernel_init = normal(stddev=0.01)


class MPDOPeriodic(nn.Module):
    r"""
    A Matrix Product Density Operator (MPDO) with periodic boundary conditions for a quantum mixed state of discrete
    degrees of freedom. The purification is used.

    The MPDO is defined as

    .. math::
        \rho(s_1,\dots s_N, s_1',\dots s_N') = \sum_{\alpha_1, \dots, \alpha_{N-1}} A^{\alpha_1}[s_1, s_1'] \dots A^{\alpha_{N-1}}[s_N, s_N'] \dots A^{\alpha_N}[s_1, s_1'],

    for arbitrary local quantum numbers :math:`s_i`, where :math:`A^{\alpha_i}[s_i, s_i']` is a tensor
    of dimensions (bdim, bdim), depending on the value of the local quantum number :math:`s_i` and
    the bond index :math:`\alpha_{i}` connecting the adjacent tensors (the purification).

    The periodic boundary conditions imply that there are connections between the first and the last tensors.
    """

    hilbert: HomogeneousHilbert
    """Hilbert space on which the state is defined."""
    bond_dim: int
    """Bond dimension of the MPDO tensors."""
    kraus_dim: int = 2
    """Kraus dimension of the MPDO tensors."""
    symperiod: Optional[bool] = None
    """
    Periodicity in the chain of MPDO tensors.
    The chain of MPDO tensors is constructed as a sequence of identical
    unit cells consisting of symperiod tensors. if None, symperiod equals the
    number of physical degrees of freedom.
    """
    param_dtype: Any = jnp.float64
    """complex or float, whether the variational parameters of the MPDO are real or complex."""
    unroll: int = 1
    """the number of scan iterations to unroll within a single iteration of a loop."""
    kernel_init: NNInitFunc = default_kernel_init
    """the initializer for the MPS weights."""

    def setup(self):
        L, d, D, Χ = (
            self.hilbert.size,
            self.hilbert.local_size,
            self.bond_dim,
            self.kraus_dim,
        )
        self._L, self._d, self._D, self._Χ = L, d, D, Χ

        self.param_dtype_cplx = dtype_complex(self.param_dtype)

        # determine the shape of the unit cell
        if self.symperiod is None:
            self._symperiod = L
        else:
            self._symperiod = self.symperiod

        if L % self._symperiod == 0 and self._symperiod > 0:
            unit_cell_shape = (self._symperiod, d, D, D, Χ)
        else:
            raise AssertionError(
                "The number of degrees of freedom of the Hilbert space needs to be a multiple of the period of the MPS"
            )

        iden_tensors = jnp.repeat(
            jnp.eye(self.bond_dim, dtype=self.param_dtype)[jnp.newaxis, :, :],
            self._symperiod * d * Χ,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(self._symperiod, d, D, D, Χ)
        self.tensors = (
            self.param("tensors", self.kernel_init, unit_cell_shape, self.param_dtype)
            + iden_tensors
        )

    def __call__(self, x):
        # create all tensors in mps from unit cell
        all_tensors = jnp.tile(self.tensors, (self._L // self._symperiod, 1, 1, 1, 1))

        x = jnp.atleast_2d(x)
        qn_x = self.hilbert.states_to_local_indices(x)

        ρ = jax.vmap(self.contract_mpdo, in_axes=(0, None))(qn_x, all_tensors)
        return jnp.log(ρ.astype(self.param_dtype_cplx))

    def contract_mpdo(self, qn, all_tensors):
        edge = jnp.eye(self._D**2, dtype=self.param_dtype)

        def scan_func(edge, pair):
            tensor, qn_r, qn_c = pair
            tensor_contracted = jnp.einsum(
                "ijk,lmk->iljm", tensor[qn_r, :], jnp.conj(tensor[qn_c, :])
            )
            matrix = tensor_contracted.reshape(self._D**2, self._D**2)
            edge = edge @ matrix
            return edge, None

        qn_r, qn_c = jnp.split(qn, 2, axis=-1)
        edge, _ = jax.lax.scan(
            scan_func, edge, (all_tensors, qn_r, qn_c), unroll=self.unroll
        )
        rho = jnp.trace(edge)
        return rho


class MPDOOpen(nn.Module):
    r"""
    A Matrix Product Density Operator (MPDO) with open boundary conditions for a quantum mixed state of discrete
    degrees of freedom. The purification is used.

    The MPDO is defined as

    .. math::
        \rho(s_1,\dots, s_N, s_1',\dots, s_N') = \sum_{\alpha_1, \dots, \alpha_{N-1}} A[s_1, s_1']^{\alpha_1} \dots A[s_N, s_N']^{\alpha_{N-1}},

    for arbitrary local quantum numbers :math:`s_i`, where :math:`A^{\alpha_i}[s_i, s_i']` is a tensor
    of dimensions (bdim, bdim), depending on the value of the local quantum number :math:`s_i`
    and the bond index :math:`\alpha_{i}` connecting the adjacent tensors (the purification).

    The open boundary conditions imply that there are no connections between the first and the last tensors:
    :math:`A[s_1, s_1']^{\alpha_0} = A[s_N, s_N']^{\alpha_N} = \delta_{s_1, s_1'} \delta_{s_N, s_N'}`
    """

    hilbert: HomogeneousHilbert
    """Hilbert space on which the state is defined."""
    bond_dim: int
    """Bond dimension of the MPDO tensors."""
    kraus_dim: int = 2
    """Kraus dimension of the MPDO tensors."""
    unroll: int = 1
    """the number of scan iterations to unroll within a single iteration of a loop."""
    kernel_init: NNInitFunc = default_kernel_init
    """the initializer for the MPS weights."""
    param_dtype: Any = jnp.float64
    """complex or float, whether the variational parameters of the MPDO are real or complex."""

    def setup(self):
        L, d, D, Χ = (
            self.hilbert.size,
            self.hilbert.local_size,
            self.bond_dim,
            self.kraus_dim,
        )
        self._L, self._d, self._D, self._Χ = L, d, D, Χ
        self.param_dtype_cplx = dtype_complex(self.param_dtype)

        iden_boundary_tensor = jnp.ones((d, D, Χ), dtype=self.param_dtype)
        self.left_tensors = (
            self.param("left_tensors", self.kernel_init, (d, D, Χ), self.param_dtype)
            + iden_boundary_tensor
        )
        self.right_tensors = (
            self.param("right_tensors", self.kernel_init, (d, D, Χ), self.param_dtype)
            + iden_boundary_tensor
        )

        # determine the shape of the unit cell
        unit_cell_shape = (L - 2, d, D, D, Χ)

        # define diagonal tensors with correct unit cell shape
        iden_tensors = jnp.repeat(
            jnp.eye(D, dtype=self.param_dtype)[jnp.newaxis, :, :],
            (L - 2) * d * Χ,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(L - 2, d, D, D, Χ)
        self.middle_tensors = (
            self.param(
                "middle_tensors", self.kernel_init, unit_cell_shape, self.param_dtype
            )
            + iden_tensors
        )

    def __call__(self, x):
        x = jnp.atleast_2d(x)
        assert x.shape[-1] == self.hilbert.size
        qn_x = self.hilbert.states_to_local_indices(x)

        ρ = jax.vmap(self.contract_mpdo)(qn_x)

        return jnp.log(ρ.astype(self.param_dtype_cplx))

    def contract_mpdo(self, qn):
        qn_r, qn_c = jnp.split(qn, 2, axis=-1)
        left_edge = (
            self.left_tensors[qn_r[0], :] @ jnp.conj(self.left_tensors[qn_c[0], :]).T
        )
        right_edge = (
            self.right_tensors[qn_r[-1], :]
            @ jnp.conj(self.right_tensors[qn_c[-1], :]).T
        )

        @jax.checkpoint
        def scan_func(edge, pair):
            tensor, qn_r, qn_c = pair
            triangle_tensor = jnp.einsum("ijk,il->jkl", tensor[qn_r, :], edge)
            edge = jnp.einsum("jkl,lmk->jm", triangle_tensor, jnp.conj(tensor[qn_c, :]))
            return edge, None

        edge, _ = jax.lax.scan(
            scan_func,
            left_edge,
            (self.middle_tensors, qn_r[1:-1], qn_c[1:-1]),
            unroll=self.unroll,
        )

        rho = jnp.einsum("ij,ij->", edge, right_edge)
        return rho
