# Copyright 2023 The NetKet Authors - All rights reserved.
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

from functools import partial, wraps
from typing import List, Union

import numpy as np

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.hilbert import AbstractHilbert, HomogeneousHilbert
from netket.utils.types import DType

from .._discrete_operator_jax import DiscreteJaxOperator

from .base import PauliStringsBase
from .numba import pack_internals


@partial(jax.vmap, in_axes=(0, None, None))
def _ising_conn_states_jax(x, cond, local_states):
    was_state_0 = x == local_states[0]
    state_0 = jnp.asarray(local_states[0], dtype=x.dtype)
    state_1 = jnp.asarray(local_states[1], dtype=x.dtype)
    return jnp.where(cond ^ was_state_0, state_0, state_1)


def _broadcast_arange(a):
    return np.broadcast_to(np.arange(a.shape[-1]), a.shape)


def _get_mask(z_check, nz_check, n_op):
    mask1 = _broadcast_arange(z_check) < np.expand_dims(nz_check, 2)
    mask2_ = _broadcast_arange(nz_check) < np.expand_dims(n_op, 1)
    mask2 = np.expand_dims(mask2_, 2)
    mask = mask1 & mask2
    return mask, mask2_


def _get_sindmask(sites, ns):
    smask = _broadcast_arange(sites) < np.expand_dims(ns, 1)
    a = np.arange(sites.shape[1]) + 1
    sindmask = (
        (np.expand_dims(a, (1, 2)) == np.expand_dims((sites + 1) * smask, 0))
        .any(axis=2)
        .T
    )
    return sindmask


@partial(jax.vmap, in_axes=(0,) + (None,) * 5)
def _pauli_strings_mels_jax(x, mask, mask2, z_check, weights, local_states):
    n_z = (mask * (x[z_check] == local_states[1])).sum(axis=-1)
    mels = (mask2 * weights * (-1) ** n_z).sum(axis=1)
    return mels


@partial(jax.jit, static_argnames="local_states")
def _pauli_strings_kernel_jax(
    x, mask, mask2, sindmask, z_check, weights, cutoff, local_states
):
    batch_shape = x.shape[:-1]
    x = x.reshape((-1, x.shape[-1]))

    mels = _pauli_strings_mels_jax(x, mask, mask2, z_check, weights, local_states)
    mels = mels.reshape(batch_shape + mels.shape[1:])

    # Same function as Ising
    x_prime = _ising_conn_states_jax(x, sindmask, local_states)
    x_prime = x_prime.reshape(batch_shape + x_prime.shape[1:])

    cutoff_mask = jnp.abs(mels) > cutoff
    mels *= cutoff_mask
    x_prime *= jnp.expand_dims(cutoff_mask, -1)

    return x_prime, mels


@partial(jax.jit, static_argnames="local_states")
def _pauli_strings_n_conn_jax(x, mask, mask2, z_check, weights, cutoff, local_states):
    # TODO avoid computing mels twice
    mels = _pauli_strings_mels_jax(x, mask, mask2, z_check, weights, local_states)
    return (jnp.abs(mels) > cutoff).sum(axis=-1, dtype=jnp.int32)


@register_pytree_node_class
class PauliStringsJax(PauliStringsBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.PauliStrings`.
    """

    @wraps(PauliStringsBase.__init__)
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[str, List[str]] = None,
        weights: Union[float, complex, List[Union[float, complex]]] = None,
        *,
        cutoff: float = 1.0e-10,
        dtype: DType = complex,
    ):
        super().__init__(hilbert, operators, weights, cutoff=cutoff, dtype=dtype)

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "PauliStringsJax only supports Hamiltonians with two local states"
            )

        # check that it is homogeneous, throw error if it's not
        if not isinstance(self.hilbert, HomogeneousHilbert):
            local_states = self.hilbert.states_at_index(0)
            if not all(
                np.allclose(local_states, self.hilbert.states_at_index(i))
                for i in range(self.hilbert.size)
            ):
                raise ValueError(
                    "Hilbert spaces with non homogeneous local_states are not "
                    "yet supported by PauliStrings."
                )

        self._hi_local_states = tuple(self.hilbert.local_states)
        self._initialized = False

    def _setup(self, force=False):
        if force or not self._initialized:
            # use the numba packer internally, much easier...
            data = pack_internals(
                self.hilbert, self.operators, self.weights, self.dtype, self._cutoff
            )

            mask, mask2 = _get_mask(data["z_check"], data["nz_check"], data["n_op"])
            sindmask = _get_sindmask(data["sites"], data["ns"])[:, : self.hilbert.size]

            self._mask = jnp.asarray(mask)
            self._mask2 = jnp.asarray(mask2)
            self._sindmask = jnp.asarray(sindmask)
            self._masks = jax.tree_map(jnp.asarray, (mask, mask2, sindmask))
            self._args_jax = jax.tree_map(
                jnp.asarray, (data["z_check"], data["weights_numba"])
            )

            self._initialized = True

    def n_conn(self, x):
        self._setup()

        local_states = tuple(self.hilbert.local_states)
        return _pauli_strings_n_conn_jax(
            x, self._mask, self._mask2, *self._args_jax, self._cutoff, local_states
        )

    def get_conn_padded(self, x):
        self._setup()

        local_states = tuple(self.hilbert.local_states)
        return _pauli_strings_kernel_jax(
            x,
            self._mask,
            self._mask2,
            self._sindmask,
            *self._args_jax,
            self._cutoff,
            local_states,
        )

    def tree_flatten(self):
        self._setup()

        data = (self.weights, self._mask, self._mask2, self._sindmask, self._args_jax)
        metadata = {
            "hilbert": self.hilbert,
            "operators": self.operators,
            "dtype": self.dtype,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (weights, m, m2, sm, args) = data
        hi = metadata["hilbert"]
        operators = metadata["operators"]
        dtype = metadata["dtype"]

        op = cls(hi, operators, weights, dtype=dtype)
        op._mask = m
        op._mask2 = m2
        op._sindmask = sm
        op._args_jax = args
        op._initialized = True
        return op

    def to_numba_operator(self) -> "PauliStrings":  # noqa: F821
        """
        Returns the standard numba version of this operator, which is an
        instance of :class:`netket.operator.PauliStrings`.
        """
        from .numba import PauliStrings

        return PauliStrings(
            self.hilbert,
            self.operators,
            self.weights,
            dtype=self.dtype,
            cutoff=self.cutoff,
        )
