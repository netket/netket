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
# TODO consider a complete rewrite in the future

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.util import safe_map
from jax.tree_util import register_pytree_node_class

from .base import LocalOperatorBase
from .compile_helpers import pack_internals, max_nonzero_per_row

from .._discrete_operator_jax import DiscreteJaxOperator


@partial(jax.vmap, in_axes=(None, None, None, None, 0))
def _inner_inner(acting_size_i, x_i, local_states_i, basis_i, k):
    tmp1 = jnp.searchsorted(
        local_states_i[acting_size_i - k - 1], x_i[acting_size_i - k - 1]
    )
    return tmp1 * basis_i[k]


@partial(jax.vmap, in_axes=(0, None, None))  # samples
@partial(jax.vmap, in_axes=(0, 0, 0))  # operators
def state_to_number(x_i, local_states_i, basis_i):
    # convert array of local states to number
    # in the hilbert space of all the sites the operator is acting on

    # number of sites these operators are acting on
    acting_size_i = local_states_i.shape[-2]
    return _inner_inner(
        acting_size_i, x_i, local_states_i, basis_i, jnp.arange(acting_size_i)
    ).sum()


@partial(jax.vmap, in_axes=(0, 0, None))  # Ns
@partial(jax.vmap, in_axes=(None, 0, 0))  # rows
@partial(jax.vmap, in_axes=(None, 0, None))  # ncmax
def _s(x, new_x_ao, acting_on):
    return x.at[acting_on].set(new_x_ao)


@partial(jax.vmap, in_axes=(0, 0, None, None))
def _extr(xp, mels, max_conn_size, mel_cutoff):
    index_nonzero = jnp.where(
        jnp.abs(mels) > mel_cutoff, size=max_conn_size, fill_value=-1
    )
    return xp[index_nonzero], mels[index_nonzero]


@partial(jax.vmap, in_axes=(None, 0))  # Ns
@partial(jax.vmap, in_axes=(0, 0))  # rows
def _sele(diag_mels, xs_n):
    return diag_mels[xs_n]


@partial(jax.jit, static_argnums=(0, 1, 2))
def _local_operator_kernel_jax(
    nonzero_diagonal, ncmax_jax, max_conn_size, mel_cutoff, op_args, x
):
    assert x.ndim == 2
    # batch_size = x.shape[0]
    (
        local_states_jax,
        acting_on_jax,
        n_conns_jax,
        diag_mels_jax,
        all_x_prime_jax,
        all_mels_jax,
        basis_jax,
        constant,
    ) = op_args

    ###
    # determine the row x corresponds to for each of the operators
    #
    xs_n2 = []
    # iterate over all groups of operators acting on a certain number of sites
    for ls, ao, nc, bsi in zip(local_states_jax, acting_on_jax, n_conns_jax, basis_jax):
        # extract the local states of all sites the operators are acting on
        xx = x[:, ao]
        # compute the corresponding number / row index
        xs_n2.append(state_to_number(xx, ls, bsi))
    ###
    # compute the number of connected elements
    #
    # extract the number of (nonzero) off-diagonal elements of the rows
    n_conn_i2_jax = safe_map(_sele, n_conns_jax, xs_n2)
    # sum for each group of operators acting on a certain number of sites
    # and sum over all gropus to get the total
    n_conn_offdiag = sum([n.sum(axis=-1) for n in n_conn_i2_jax])

    n_conn_total = n_conn_offdiag + (1 if nonzero_diagonal else 0)

    mels_offdiag_jax = []
    x_prime_jax = []

    if nonzero_diagonal:
        xp_diag = x
        # extract diagonal mels of the rows
        mels_diag_ = safe_map(_sele, diag_mels_jax, xs_n2)
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
        mels_offdiag_jax.append(mels_diag)
        x_prime_jax.append(xp_diag)

    # iterate over all groups of operators acting on a certain number of sites
    for kk in range(len(all_mels_jax)):
        xs_n = xs_n2[kk]
        all_mels = all_mels_jax[kk]
        acting_on = acting_on_jax[kk]
        all_x_prime = all_x_prime_jax[kk].astype(x.dtype)
        n_conns = n_conns_jax[kk]

        # TODO remove ncmax_jax and just extract it from the shapes
        ncmax = ncmax_jax[kk]
        assert all_mels.shape[2] == ncmax
        assert all_x_prime.shape[2] == ncmax

        a = np.arange(xs_n.shape[1])
        # select rows
        amixsall = all_mels[a, xs_n]
        nconnsall = n_conns[a, xs_n]
        axxsall = all_x_prime[a, xs_n]
        # mask for the connected elements
        # (not all operators have the same number nonzeros per row, and the
        # arrays we use here are padded to the max within each group)
        # this mask is False whenever it's just padding
        conn_maskall = np.arange(ncmax)[None, None] < nconnsall[:, :, None]

        # set mels in the padding to 0
        # TODO check this is necessary (shouldn't it already be 0 ???)
        mels_offdiag = amixsall * conn_maskall
        #
        # compute xp
        #
        # we start from the xp for the sites the operator is acting on
        new = axxsall  # (Ns, terms, ncmax, nsitesactiongon)
        old = x[:, acting_on]  # (Ns, terms, nsitesactiongon)
        old = jnp.broadcast_to(old[:, :, None, :], new.shape)
        mask = jnp.broadcast_to(conn_maskall[:, :, :, None], new.shape)
        # select it only if we are not padding, otherwise keep old
        new_x_ao = jax.lax.select(mask, new, old)
        # now insert the local states into the full x
        xp_offdiag = _s(x, new_x_ao, acting_on)

        mels_offdiag_jax.append(mels_offdiag)
        x_prime_jax.append(xp_offdiag)

    if max_conn_size is not None:
        # compute a mask which tells us where the actual mels are and
        # where there is just padding
        # (we could also just be lazy and check the mels for being 0,
        # but allows us to make the order of mels consistent with the numba op)
        mask_jax = []
        if mels_diag is not None:
            mask_jax.append(jnp.full(mels_diag.shape, fill_value=True))
        # iterate over all groups of operators acting on a certain number of sites
        for kk in range(len(ncmax_jax)):
            nc = n_conn_i2_jax[kk]
            ncm = ncmax_jax[kk]
            mask = jnp.arange(ncm)[None, None, :] < nc[:, :, None]
            mask_jax.append(mask)

        # pad with old state and mel 0
        x_prime_jax.append(x[:, None][:, :1])
        mels_offdiag_jax.append(jnp.zeros(x.shape[:-1]))
        mask_jax.append(jnp.full((x.shape[0], 1), fill_value=False))

    mels_jc = jnp.hstack([x.reshape(x.shape[0], -1) for x in mels_offdiag_jax])
    xp_jc = jnp.hstack([x.reshape(x.shape[0], -1, x.shape[-1]) for x in x_prime_jax])

    # TODO run unique on it, there might be repeated xps

    if max_conn_size is None:
        return xp_jc, mels_jc, n_conn_total
    else:
        if mel_cutoff is not None:
            raise NotImplementedError
        # this is the lazy one with checking mels
        # return *_extr(xp_jc, mels_jc, max_conn_size, mel_cutoff), n_conn_total, mels_diag

        # move nonzero mels to the front and keep exactly max_conn_size
        mask_jc = jnp.hstack([x.reshape(x.shape[0], -1) for x in mask_jax])
        (ind,) = jax.vmap(partial(jnp.where, size=max_conn_size, fill_value=-1))(
            mask_jc
        )
        return (
            xp_jc[jnp.arange(len(ind))[:, None], ind],
            mels_jc[jnp.arange(len(ind))[:, None], ind],
            n_conn_total,
        )


@register_pytree_node_class
class LocalOperatorJax(LocalOperatorBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.LocalOperator`.
    """

    def _setup(self, force=False):
        if force or not self._initialized:
            # TODO !! rewrite a version of pack_internals which directly
            # assembles the jax represenation below, avoiding padding
            # in there first and then removing the padding here again
            data = pack_internals(
                self.hilbert,
                self._operators_dict,
                self.constant,
                self.dtype,
                self.mel_cutoff,
            )

            acting_on = jnp.asarray(data["acting_on"])
            acting_size = data["acting_size"]

            diag_mels = jnp.array(data["diag_mels"])
            all_mels = jnp.array(data["mels"])
            all_x_prime = jnp.array(data["x_prime"])
            n_conns = jnp.asarray(data["n_conns"])
            local_states = jnp.asarray(data["local_states"])
            basis = jnp.asarray(data["basis"])

            self._nonzero_diagonal = bool(data["nonzero_diagonal"])
            self._max_conn_size = int(data["max_conn_size"])

            self._local_states_jax = []
            self._acting_on_jax = []
            self._n_conns_jax = []
            self._diag_mels_jax = []
            self._all_x_prime_jax = []
            self._all_mels_jax = []
            self._basis_jax = []

            operators = list(self._operators_dict.values())
            op_size = np.array(list(map(lambda x: x.shape[0], operators)))
            op_n_conns_offdiag = max_nonzero_per_row(operators, self.mel_cutoff)

            for s in np.unique(acting_size):
                (indices,) = np.where(acting_size == s)
                self._local_states_jax.append(local_states[indices, :s])
                self._acting_on_jax.append(acting_on[indices, :s])
                # compute the maximum size of any operator acting on s sites
                # (maximum size of the matrix / prod of local hilbert spaces)
                max_op_size_s = max(op_size[indices])
                # compute the maximum number of offdiag nonzeros in any row of any operator acting on s sites
                max_op_size_offdiag_s = max(op_n_conns_offdiag[indices])
                self._n_conns_jax.append(n_conns[indices, :max_op_size_s])
                self._diag_mels_jax.append(diag_mels[indices, :max_op_size_s])

                self._all_x_prime_jax.append(
                    all_x_prime[indices, :max_op_size_s, :max_op_size_offdiag_s, :s]
                )
                self._all_mels_jax.append(
                    all_mels[indices, :max_op_size_s, :max_op_size_offdiag_s]
                )
                self._basis_jax.append(basis[indices, :s])

            self._nconn_max_jax = tuple(
                map(int, jax.tree_map(jnp.max, self._n_conns_jax))
            )

            self._initialized = True

    def _get_conn_padded(self, x):
        self._setup()

        shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        xp, mels, n_conn = _local_operator_kernel_jax(
            self._nonzero_diagonal,
            self._nconn_max_jax,
            self._max_conn_size,
            None,
            (
                self._local_states_jax,
                self._acting_on_jax,
                self._n_conns_jax,
                self._diag_mels_jax,
                self._all_x_prime_jax,
                self._all_mels_jax,
                self._basis_jax,
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
            self._local_states_jax,
            self._acting_on_jax,
            self._n_conns_jax,
            self._diag_mels_jax,
            self._all_x_prime_jax,
            self._all_mels_jax,
            self._basis_jax,
            self._constant,
        )
        metadata = {
            "hilbert": self.hilbert,
            "dtype": self.dtype,
            "nonzero_diagonal": self._nonzero_diagonal,
            "max_conn_size": self._max_conn_size,
            "nconn_max_jax": self._nconn_max_jax,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        hi = metadata["hilbert"]
        dtype = metadata["dtype"]

        op = cls(hi, dtype=dtype)

        op._nonzero_diagonal = metadata["nonzero_diagonal"]
        op._max_conn_size = metadata["max_conn_size"]
        op._nconn_max_jax = metadata["nconn_max_jax"]

        (
            op._local_states_jax,
            op._acting_on_jax,
            op._n_conns_jax,
            op._diag_mels_jax,
            op._all_x_prime_jax,
            op._all_mels_jax,
            op._basis_jax,
            op._constant,
        ) = data

        op._initialized = True
        return op

    def to_numba_operator(self) -> "LocalOperator":  # noqa: F821
        """
        Returns the standard numba version of this operator, which is an
        instance of :class:`netket.operator.LocalOperator`.
        """
        from .numba import LocalOperator

        return LocalOperator(
            self.hilbert,
            self.operators,
            self.acting_on,
            self.constant,
            dtype=self.dtype,
        )
