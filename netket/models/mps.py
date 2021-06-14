# Copyright 2021 The NetKet Authors - All rights reserved.
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

from typing import Any
import warnings

import igraph as ig
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np

from netket.graph import AbstractGraph, Chain
from netket.hilbert import AbstractHilbert
from netket.nn.initializers import lecun_normal
from netket.utils.types import NNInitFunc

default_kernel_init = lecun_normal()


class MPSPeriodic(nn.Module):
    r"""
    A periodic Matrix Product State (MPS) for a quantum state of discrete
    degrees of freedom, wrapped as Jax machine.

    The MPS is defined as

    .. math:: \Psi(s_1,\dots s_N) = \mathrm{Tr} \left[ A[s_1]\dots A[s_N] \right] ,

    for arbitrary local quantum numbers :math:`s_i`, where :math:`A[s_1]` is a matrix
    of dimension (bdim,bdim), depending on the value of the local quantum number :math:`s_i`.
    """

    hilbert: AbstractHilbert
    """Hilbert space on which the state is defined."""
    graph: AbstractGraph
    """The graph on which the system is defined."""
    bond_dim: int
    """Virtual dimension of the MPS tensors."""
    diag: bool = False
    """Whether or not to use diagonal matrices in the MPS tensors."""
    symperiod: bool = None
    """
    Periodicity in the chain of MPS tensors.

    The chain of MPS tensors is constructed as a sequence of identical
    unit cells consisting of symperiod tensors. if None, symperiod equals the
    number of physical degrees of freedom.
    """
    kernel_init: NNInitFunc = jax.nn.initializers.normal(
        stddev=0.01
    )  # default standard deviation equals 1e-2
    """the initializer for the MPS weights."""
    dtype: Any = np.complex64
    """complex or float, whether the variational parameters of the MPS are real or complex."""

    def setup(self):
        L = self.hilbert.size
        phys_dim = self.hilbert.local_size

        self._L = L
        self._phys_dim = phys_dim

        # determine transformation from local states to indices
        local_states = np.array(self.hilbert.local_states)
        loc_vals_spacing = np.roll(local_states, -1)[0:-1] - local_states[0:-1]
        if np.max(loc_vals_spacing) == np.min(loc_vals_spacing):
            self._loc_vals_spacing = jnp.array(loc_vals_spacing[0])
        else:
            raise AssertionError(
                "JaxMpsPeriodic can only be used with evenly spaced hilbert local values"
            )
        self._loc_vals_bias = jnp.min(local_states)

        # check whether graph is periodic chain
        chain_graph = Chain(self.graph.n_edges).to_igraph()
        if not ig.Graph(edges=self.graph.edges()).isomorphic(chain_graph):
            warnings.warn(
                "Warning: graph is not isomorphic to chain with periodic boundary conditions",
                UserWarning,
            )

        # determine shape of unit cell
        if self.symperiod is None:
            self._symperiod = L
        else:
            self._symperiod = self.symperiod

        if L % self._symperiod == 0 and self._symperiod > 0:
            if self.diag:
                unit_cell_shape = (self._symperiod, phys_dim, self.bond_dim)
            else:
                unit_cell_shape = (
                    self._symperiod,
                    phys_dim,
                    self.bond_dim,
                    self.bond_dim,
                )
        else:
            raise AssertionError(
                "The number of degrees of freedom of the Hilbert space needs to be a multiple of the period of the MPS"
            )

        # define diagonal tensors with correct unit cell shape
        if self.diag:
            iden_tensors = jnp.ones(
                (self._symperiod, phys_dim, self.bond_dim), dtype=self.dtype
            )
        else:
            iden_tensors = jnp.repeat(
                jnp.eye(self.bond_dim, dtype=self.dtype)[jnp.newaxis, :, :],
                self._symperiod * phys_dim,
                axis=0,
            )
            iden_tensors = iden_tensors.reshape(
                self._symperiod, phys_dim, self.bond_dim, self.bond_dim
            )

        self.kernel = (
            self.param("kernel", self.kernel_init, unit_cell_shape, self.dtype)
            + iden_tensors
        )

    @nn.compact
    def __call__(self, x):
        # expand diagonal to square matrices if diagonal mps
        if self.diag:
            params = jnp.einsum(
                "ijk,kl->ijkl", self.kernel, jnp.eye(self.kernel.shape[-1])
            )
        else:
            params = self.kernel

        # create all tensors in mps from unit cell
        all_tensors = jnp.tile(params, (self._L // self._symperiod, 1, 1, 1))

        # transform input to indices
        x = (x - self._loc_vals_bias) / self._loc_vals_spacing
        if len(x.shape) == 1:  # batch size is one
            x = jnp.expand_dims(x, 0)

        def select_tensor(tensor, index):
            return tensor[index.astype(int)]

        def select_all_tensors(all_tensors, indices):
            return jax.vmap(select_tensor)(all_tensors, indices)

        # select right tensors using input for matrix multiplication
        selected_tensors = jax.vmap(select_all_tensors, (None, 0))(all_tensors, x)

        # create loop carry, in this case a unit matrix
        edges = jnp.repeat(
            jnp.eye(self.bond_dim, dtype=selected_tensors.dtype)[jnp.newaxis, :, :],
            selected_tensors.shape[0],
            axis=0,
        )

        def trace_mps(tensors, edge):
            def multiply_tensors(left_tensor, right_tensor):
                return jnp.einsum("ij,jk->ik", left_tensor, right_tensor), None

            edge, _ = jax.lax.scan(multiply_tensors, edge, tensors)

            return jnp.trace(edge)

        # trace the matrix multiplication
        return jnp.log(jax.vmap(trace_mps)(selected_tensors, edges))
