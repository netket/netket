# Copyright 2023-2024 The NetKet Authors - All rights reserved.
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


# this file contains a more-or-less 1:1 port of the numba localoperator to jax, using padding where necessary

from functools import partial

import jax
import jax.numpy as jnp
from jax.util import safe_map
from jax.tree_util import register_pytree_node_class

from netket.errors import JaxOperatorNotConvertibleToNumba

from .base import LocalOperatorBase
from .compile_helpers import pack_internals_jax

from .._discrete_operator_jax import DiscreteJaxOperator


@partial(jax.vmap, in_axes=(0, None, None))  # samples
@partial(jax.vmap, in_axes=(0, 0, 0))  # operators
def _state_to_number(x_i, local_states_i, basis_i):
    # convert array of local states to number
    # in the hilbert space of all the sites the operator is acting on

    # number of sites these operators are acting on
    acting_size_i = local_states_i.shape[-2]

    @partial(jax.vmap, in_axes=(None, None, None, None, 0))  # vmap over local sites (k)
    def _f(acting_size_i, x_i, local_states_i, basis_i, k):
        return basis_i[k] * jnp.searchsorted(
            local_states_i[acting_size_i - k - 1], x_i[acting_size_i - k - 1]
        )

    # vmap over sites we are acting on, computing the contribution to the total int,
    # then taking the sum
    return _f(
        acting_size_i, x_i, local_states_i, basis_i, jnp.arange(acting_size_i)
    ).sum()


@partial(jax.vmap, in_axes=(0, 0, None))  # Ns
@partial(jax.vmap, in_axes=(None, 0, 0))  # rows
@partial(jax.vmap, in_axes=(None, 0, None))  # ncmax
def _set_at(x, new_x_ao, acting_on):
    return x.at[acting_on].set(new_x_ao)


@partial(jax.vmap, in_axes=(None, 0))  # Ns
@partial(jax.vmap, in_axes=(0, 0))  # rows
def _index_at(diag_mels, i):
    return diag_mels[i]


# @partial(jax.vmap, in_axes=(0, 0, None, None))
# def _extr(xp, mels, max_conn_size, mel_cutoff):
#     index_nonzero = jnp.where(
#         jnp.abs(mels) > mel_cutoff, size=max_conn_size, fill_value=-1
#     )
#     return xp[index_nonzero], mels[index_nonzero]


@partial(jax.jit, static_argnums=(0, 1))
def _local_operator_kernel_jax(nonzero_diagonal, max_conn_size, mel_cutoff, op_args, x):
    assert x.ndim == 2

    # in the foollowing variables with a trailing underscore are per group
    # of operators with the same number of sites they act on

    (
        local_states_,
        acting_on_,
        n_conns_,
        diag_mels_,
        x_prime_,
        mels_,
        basis_,
        constant,
    ) = op_args

    n_groups = len(local_states_)
    ncmax_ = [m.shape[2] for m in mels_]

    ###
    # determine the row x corresponds to for each of the operators
    #
    i_row_ = []
    # iterate over all groups of operators acting on a certain number of sites
    for k in range(n_groups):
        # extract the local states of all sites the operators are acting on
        # and compute the corresponding number / row index
        i_row_.append(
            _state_to_number(x[:, acting_on_[k]], local_states_[k], basis_[k])
        )
    ###
    # compute the number of connected elements
    #
    # extract the number of (nonzero) off-diagonal elements of the rows
    n_conn_offdiag_ = safe_map(_index_at, n_conns_, i_row_)
    # sum for each group of operators acting on a certain number of sites
    # and sum over all gropus to get the total
    n_conn_offdiag = sum([n.sum(axis=-1) for n in n_conn_offdiag_])
    n_conn_diag = 1 if nonzero_diagonal else 0
    n_conn_total = n_conn_diag + n_conn_offdiag

    mels_offdiag_ = []
    xp_offdiag_ = []

    if nonzero_diagonal:
        xp_diag = x
        # extract diagonal mels of the rows
        mels_diag_ = safe_map(_index_at, diag_mels_, i_row_)
        # sum over operators
        mels_diag = constant + sum([m.sum(axis=-1) for m in mels_diag_])

    elif max_conn_size is None:
        # if we don't have non-zero and there are no connected elements
        # therefore if the operator is empty, still add something here
        # TODO why can't we just return empty xp and mels?
        xp_diag = x[:, None][:, :0]
        mels_diag = jnp.zeros(xp_diag.shape[:-1])
    else:
        # zero diagonal, but some other connected elements
        mels_diag = None
        xp_diag = None

    if mels_diag is not None:
        mels_offdiag_.append(mels_diag)
        xp_offdiag_.append(xp_diag)

    # iterate over all groups of operators acting on a certain number of sites
    for k in range(n_groups):
        # indices used to select rows
        i = i_row_[k]
        a = jnp.arange(i.shape[1])

        # mask for the connected elements
        # (not all operators have the same number nonzeros per row, and the
        # arrays we use here are padded to the max within each group)
        # this mask is False whenever it's just padding
        conn_maskall = jnp.arange(ncmax_[k])[None, None] < n_conns_[k][a, i][:, :, None]

        # set mels in the padding to 0
        # TODO check this is necessary (shouldn't it already be 0 ???)
        mels_offdiag = mels_[k][a, i] * conn_maskall
        #
        # compute xp
        #
        # we start from the xp for the sites the operator is acting on
        new = x_prime_[k][a, i].astype(x.dtype)  # (Ns, terms, ncmax, nsitesactiongon)
        acting_on = acting_on_[k]
        old = x[:, acting_on]  # (Ns, terms, nsitesactiongon)
        old = jnp.broadcast_to(old[:, :, None, :], new.shape)
        mask = jnp.broadcast_to(conn_maskall[:, :, :, None], new.shape)
        # select it only if we are not padding, otherwise keep old
        new_x_ao = jax.lax.select(mask, new, old)
        # now insert the local states into the full x
        xp_offdiag = _set_at(x, new_x_ao, acting_on)

        mels_offdiag_.append(mels_offdiag)
        xp_offdiag_.append(xp_offdiag)

    if max_conn_size is not None:
        # compute a mask which tells us where the actual mels are and
        # where there is just padding
        # (we could also just be lazy and check the mels for being 0,
        # but allows us to make the order of mels consistent with the numba op)
        mask_ = []
        if mels_diag is not None:
            mask_.append(jnp.full(mels_diag.shape, fill_value=True))
        # iterate over all groups of operators acting on a certain number of sites
        for k in range(n_groups):
            mask = jnp.arange(ncmax_[k])[None, None, :] < n_conn_offdiag_[k][:, :, None]
            mask_.append(mask)

        # pad with old state and mel 0
        xp_offdiag_.append(x[:, None][:, :1])
        mels_offdiag_.append(jnp.zeros(x.shape[:-1]))
        mask_.append(jnp.full((x.shape[0], 1), fill_value=False))

    mels = jnp.hstack([m.reshape(m.shape[0], -1) for m in mels_offdiag_])
    xp = jnp.hstack([x.reshape(x.shape[0], -1, x.shape[-1]) for x in xp_offdiag_])

    # TODO run unique on it, there might be repeated xps

    if max_conn_size is None:
        return xp, mels, n_conn_total
    else:
        if mel_cutoff is not None:
            raise NotImplementedError
        # this is the lazy one with checking mels
        # return *_extr(xp, mels, max_conn_size, mel_cutoff), n_conn_total, mels_diag

        # move nonzero mels to the front and keep exactly max_conn_size
        mask = jnp.hstack([m.reshape(m.shape[0], -1) for m in mask_])
        (ind,) = jax.vmap(partial(jnp.where, size=max_conn_size, fill_value=-1))(mask)
        return (
            xp[jnp.arange(len(ind))[:, None], ind],
            mels[jnp.arange(len(ind))[:, None], ind],
            n_conn_total,
        )


@register_pytree_node_class
class LocalOperatorJax(LocalOperatorBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.LocalOperator`.
    """

    _convertible: bool = True
    """Internal flag. True if it can be converted to numba, false
    otherwise."""

    def _setup(self, force=False):
        if force or not self._initialized:
            data = pack_internals_jax(
                self.hilbert,
                self._operators_dict,
                self.constant,
                self.dtype,
                self.mel_cutoff,
            )

            self._local_states = data["local_states"]
            self._acting_on = data["acting_on"]
            self._n_conns = data["n_conns"]
            self._diag_mels = data["diag_mels"]
            self._x_prime = data["x_prime"]
            self._mels = data["mels"]
            self._basis = data["basis"]
            self._nonzero_diagonal = data["nonzero_diagonal"]
            self._max_conn_size = data["max_conn_size"]

            self._initialized = True

    def _get_conn_padded(self, x):
        self._setup()

        shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        xp, mels, n_conn = _local_operator_kernel_jax(
            self._nonzero_diagonal,
            self._max_conn_size,
            None,
            (
                self._local_states,
                self._acting_on,
                self._n_conns,
                self._diag_mels,
                self._x_prime,
                self._mels,
                self._basis,
                self._constant,
            ),
            x,
        )
        xp = xp.reshape(shape[:-1] + xp.shape[-2:])
        mels = mels.reshape(shape[:-1] + mels.shape[-1:])
        n_conn = n_conn.reshape(shape[:-1])

        return xp, mels, n_conn

    def get_conn_padded(self, x):
        xp, mels, _ = self._get_conn_padded(x)
        return xp, mels

    def n_conn(self, x):
        _, _, n_conn = self._get_conn_padded(x)
        return n_conn

    def tree_flatten(self):
        self._setup()
        data = (
            self._local_states,
            self._acting_on,
            self._n_conns,
            self._diag_mels,
            self._x_prime,
            self._mels,
            self._basis,
            self._constant,
        )
        metadata = {
            "hilbert": self.hilbert,
            "dtype": self.dtype,
            "nonzero_diagonal": self._nonzero_diagonal,
            "max_conn_size": self._max_conn_size,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        hi = metadata["hilbert"]
        dtype = metadata["dtype"]

        op = cls(hi, dtype=dtype)

        op._nonzero_diagonal = metadata["nonzero_diagonal"]
        op._max_conn_size = metadata["max_conn_size"]
        (
            op._local_states,
            op._acting_on,
            op._n_conns,
            op._diag_mels,
            op._x_prime,
            op._mels,
            op._basis,
            op._constant,
        ) = data

        op._initialized = True
        op._convertible = False
        return op

    def to_numba_operator(self) -> "LocalOperator":  # noqa: F821
        """
        Returns the standard numba version of this operator, which is an
        instance of :class:`netket.operator.LocalOperator`.
        """
        from .numba import LocalOperator

        if self._convertible is False:
            raise JaxOperatorNotConvertibleToNumba(self)

        return LocalOperator(
            self.hilbert,
            self.operators,
            self.acting_on,
            self.constant,
            dtype=self.dtype,
        )
