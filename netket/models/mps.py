from typing import Any, Optional

from flax import linen as nn
import jax
from jax import numpy as jnp
from jax.nn.initializers import normal

from netket.hilbert import HomogeneousHilbert
from netket.utils.types import NNInitFunc
from netket.jax import dtype_complex


class MPSPeriodic(nn.Module):
    r"""
    A periodic Matrix Product State (MPS) for a quantum state of discrete
    degrees of freedom, wrapped as Jax machine.
    The MPS is defined as
    .. math:: \Psi(s_1,\dots s_N) = \mathrm{Tr} \left[ A[s_1]\dots A[s_N] \right] ,
    for arbitrary local quantum numbers :math:`s_i`, where :math:`A[s_1]` is a matrix
    of dimension (bdim,bdim), depending on the value of the local quantum number :math:`s_i`.
    """
    hilbert: HomogeneousHilbert
    """Hilbert space on which the state is defined."""
    bond_dim: int
    """Bond dimension of the MPS tensors."""
    local_size: int = 2
    """Size of the local degrees of freedom"""
    L: Optional[int] = None
    """Number of sites in the MPS chain (which can be used to re-define the length)."""
    symperiod: Optional[bool] = None
    """
    Periodicity in the chain of MPS tensors.
    The chain of MPS tensors is constructed as a sequence of identical
    unit cells consisting of symperiod tensors. if None, symperiod equals the
    number of physical degrees of freedom.
    """
    unroll: int = 1
    """the number of scan iterations to unroll within a single iteration of a loop."""
    kernel_init: NNInitFunc = normal(stddev=0.01)
    """the initializer for the MPS weights."""
    param_dtype: Any = jnp.float64
    """complex or float, whether the variational parameters of the MPS are real or complex."""

    def setup(self):
        L, d, D = self.hilbert.size, self.hilbert.local_size, self.bond_dim
        if self.L is not None:
            L = self.L
        self._L, self._d, self._D = L, d, D

        self.param_dtype_cplx = dtype_complex(self.param_dtype)

        # determine shape of unit cell
        if self.symperiod is None:
            self._symperiod = L
        else:
            self._symperiod = self.symperiod

        if L % self._symperiod == 0 and self._symperiod > 0:
            unit_cell_shape = (self._symperiod, d, D, D)
        else:
            raise AssertionError("The number of degrees of freedom of the Hilbert space needs to be a multiple of the period of the MPS")

        # define diagonal tensors with correct unit cell shape
        iden_tensors = jnp.repeat(
            jnp.eye(D, dtype=self.param_dtype)[jnp.newaxis, :, :],
            self._symperiod * d,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(self._symperiod, d, D, D)

        self.tensors = self.param("tensors", self.kernel_init, unit_cell_shape, self.param_dtype) + iden_tensors

    def __call__(self, x):
        x = jnp.atleast_2d(x)

        x_shape = x.shape
        if jnp.ndim(x) != 2:
            x = x.reshape((-1, x_shape[-1]))

        qn_x = self.hilbert.states_to_local_indices(x)
        # create all tensors in mps from unit cell
        all_tensors = jnp.tile(self.tensors, (self._L // self._symperiod, 1, 1, 1))

        ψ = jax.vmap(self.contract_mps, in_axes=(0, None))(qn_x, all_tensors)
        ψ = ψ.reshape(x_shape[:-1])

        return jnp.log(ψ.astype(self.param_dtype_cplx))

    def contract_mps(self, qn, all_tensors):
        edge = jnp.eye(self._D, dtype=self.param_dtype)

        @jax.checkpoint
        def scan_func(edge, pair):
            tensor, qn = pair
            edge = edge @ tensor[qn, :]
            return edge, None

        edge, _ = jax.lax.scan(scan_func, edge, (all_tensors, qn), unroll=self.unroll)

        return jnp.trace(edge)


class MPSOpen(nn.Module):
    r"""
    An open Matrix Product State (MPS) for a quantum state of discrete
    degrees of freedom, wrapped as Jax machine.
    The MPS is defined as
    .. math:: \Psi(s_1,\dots s_N) = \mathrm{Tr} \left[ A[s_1]\dots A[s_N] \right] ,
    for arbitrary local quantum numbers :math:`s_i`, where :math:`A[s_1]` is a matrix
    of dimension (bdim,bdim), depending on the value of the local quantum number :math:`s_i`.
    """
    hilbert: HomogeneousHilbert
    """Hilbert space on which the state is defined."""
    bond_dim: int
    """Bond dimension of the MPS tensors."""
    L: Optional[int] = None
    """Number of sites in the MPS chain (which can be used to re-define the length)."""
    unroll: int = 1
    """the number of scan iterations to unroll within a single iteration of a loop."""
    kernel_init: NNInitFunc = normal(stddev=0.01)
    """the initializer for the MPS weights."""
    param_dtype: Any = jnp.float64
    """complex or float, whether the variational parameters of the MPS are real or complex."""

    def setup(self):
        L, d, D = self.hilbert.size, self.hilbert.local_size, self.bond_dim
        if self.L is not None:
            L = self.L
        self._L, self._d, self._D = L, d, D
        self.param_dtype_cplx = dtype_complex(self.param_dtype)

        iden_boundary_tensor = jnp.ones((d, D), dtype=self.param_dtype)
        self.left_tensors = self.param("left_tensors", self.kernel_init, (d, D), self.param_dtype) + iden_boundary_tensor
        self.right_tensors = self.param("right_tensors", self.kernel_init, (d, D), self.param_dtype) + iden_boundary_tensor

        # determine shape of unit cell
        unit_cell_shape = (L - 2, d, D, D)

        iden_tensors = jnp.repeat(
            jnp.eye(D, dtype=self.param_dtype)[jnp.newaxis, :, :],
            (L - 2) * d,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(L - 2, d, D, D)
        self.middle_tensors = self.param("middle_tensors", self.kernel_init, unit_cell_shape, self.param_dtype) + iden_tensors

    def __call__(self, x):
        x = jnp.atleast_2d(x)
        x_shape = x.shape
        if x.ndim > 2:
            x = jax.lax.collapse(x, 0, 2)
        qn_x = self.hilbert.states_to_local_indices(x)

        ψ = jax.vmap(self.contract_mps)(qn_x)
        ψ = ψ.reshape(x_shape[:-1])

        return jnp.log(ψ.astype(self.param_dtype_cplx))

    def contract_mps(self, qn):
        edge = self.left_tensors[qn[0], :]

        @jax.checkpoint
        def scan_func(edge, pair):
            tensor, qn = pair
            edge = edge @ tensor[qn, :]
            return edge, None

        edge, _ = jax.lax.scan(scan_func, edge, (self.middle_tensors, qn[1:-1]), unroll=self.unroll)
        psi = self.right_tensors[qn[-1], :] @ edge
        return psi
