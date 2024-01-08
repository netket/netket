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
from jax.tree_util import register_pytree_node_class

from .base import LocalOperatorBase
from .compile_helpers import pack_internals

from .._discrete_operator_jax import DiscreteJaxOperator


@partial(jax.vmap, in_axes=(None, None, None, None, 0))
def _inner_inner(acting_size_i, x_i, local_states_i, basis_i, k):
    tmp1 = jnp.searchsorted(
        local_states_i[acting_size_i - k - 1], x_i[acting_size_i - k - 1]
    )
    return tmp1 * basis_i[k]


@partial(jax.jit, static_argnums=0)
def _inner(acting_size_i, x_i, local_states_i, basis_i):
    return _inner_inner(
        acting_size_i, x_i, local_states_i, basis_i, jnp.arange(acting_size_i)
    ).sum()


# @partial(jax.jit, inline=True)
@partial(jax.vmap, in_axes=(0, None, None))  # samples
@partial(jax.vmap, in_axes=(0, 0, 0))  # operators
def inner(x_i, local_states_i, basis_i):
    acting_size_i = local_states_i.shape[-2]
    return _inner_inner(
        acting_size_i, x_i, local_states_i, basis_i, jnp.arange(acting_size_i)
    ).sum()


@partial(jax.vmap, in_axes=(0, 0, None))  # Ns
@partial(jax.vmap, in_axes=(None, 0, 0))  # terms
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
@partial(jax.vmap, in_axes=(0, 0))  # term
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

    xs_n2 = []
    conn_b2 = []
    for ls, ao, nc, bsi in zip(local_states_jax, acting_on_jax, n_conns_jax, basis_jax):
        xx = x[:, ao]
        xs_n2.append(inner(xx, ls, bsi))

    conn_b2_jax = jax.tree_map(
        lambda nc, xx: nc[np.arange(xx.shape[1]), xx].sum(axis=-1), n_conns_jax, xs_n2
    )
    conn_b2 = sum(conn_b2_jax)
    if nonzero_diagonal:
        conn_b2 = conn_b2 + 1

    # max_conn2 = conn_b2.max()
    # tot_conn2 = max_conn2 * batch_size

    # below we pad for every operator length to ncmax
    # TODO use this to deterministically find 0s
    n_conn_i2_jax = jax.tree_map(_sele, n_conns_jax, xs_n2)
    # start_jax = jax.tree_map(jax.vmap(lambda x: jnp.cumsum(jnp.pad(x, pad_width=(1,0)))[:-1]), n_conn_i2_jax)

    melsop_jax = []
    x_prime_jax = []

    if nonzero_diagonal:
        # If we have some non-zero in the diagonal, prepare mels0 accordingly
        mels0 = constant + sum(
            jax.tree_map(
                lambda nc, xx: _sele(nc, xx).sum(axis=-1), diag_mels_jax, xs_n2
            )
        )
        xp0 = x
    elif max_conn_size is None:
        # if we don't have non-zero and there are no connected elements (
        # therefore if the operator is empty, still add something here)
        xp0 = x[:, None][:, :0]
        mels0 = jnp.zeros(xp0.shape[:-1])
    else:
        # zero diagonal, but some other connected elements
        mels0 = None
        xp0 = None

    if mels0 is not None:
        melsop_jax.append(mels0)
        x_prime_jax.append(xp0)

    for kk in range(len(ncmax_jax)):
        ncmax = ncmax_jax[kk]

        # n_operators = acting_on_jax[kk].shape[0]
        xs_n = xs_n2[kk]
        all_mels = all_mels_jax[kk]
        acting_on = acting_on_jax[kk]
        all_x_prime = all_x_prime_jax[kk].astype(x.dtype)
        n_conns = n_conns_jax[kk]

        amixsall = all_mels[np.arange(xs_n.shape[1]), xs_n, :ncmax]
        nconnsall = n_conns[np.arange(xs_n.shape[1]), xs_n]
        axxsall = all_x_prime[np.arange(xs_n.shape[1]), xs_n, :ncmax]
        conn_maskall = np.arange(ncmax)[None, None] < nconnsall[:, :, None]

        melsop = amixsall * conn_maskall
        new = axxsall  # (Ns, terms, ncmax, nsitesactiongon)
        old = x[:, acting_on]  # (Ns, terms, nsitesactiongon)
        old = jnp.broadcast_to(old[:, :, None, :], new.shape)
        mask = jnp.broadcast_to(conn_maskall[:, :, :, None], new.shape)
        new_x_ao = jax.lax.select(mask, new, old)
        xpnew = _s(x, new_x_ao, acting_on)

        melsop_jax.append(melsop)
        x_prime_jax.append(xpnew)

    if max_conn_size is not None:
        mask_jax = []

        if mels0 is not None:
            mask_jax.append(jnp.full(mels0.shape, fill_value=True))
        for kk in range(len(ncmax_jax)):
            nc = n_conn_i2_jax[kk]
            ncm = ncmax_jax[kk]
            mask = jnp.arange(ncm)[None, None, :] < nc[:, :, None]
            mask_jax.append(mask)

        if xp0 is not None:
            melsop_jax.append(jnp.zeros(xp0.shape[:-1]))

        # pad with old state and mel 0
        # pad_value = -1
        xpm1 = x[:, None][:, :1]
        x_prime_jax.append(xpm1)
        melsop_jax.append(jnp.zeros(x.shape[:-1]))
        mask_jax.append(jnp.full((x.shape[0], 1), fill_value=False))

    mels_jc = jnp.hstack([x.reshape(x.shape[0], -1) for x in melsop_jax])
    xp_jc = jnp.hstack([x.reshape(x.shape[0], -1, x.shape[-1]) for x in x_prime_jax])

    # TODO run unique on it, there might be repeated xps

    if max_conn_size is None:
        return xp_jc, mels_jc, conn_b2
    else:
        if mel_cutoff is not None:
            raise NotImplementedError
        # this is the lazy one with checking mels
        # return *_extr(xp_jc, mels_jc, max_conn_size, mel_cutoff), conn_b2, mels0

        # here we compute it from where we padded:
        mask_jc = jnp.hstack([x.reshape(x.shape[0], -1) for x in mask_jax])
        (ind,) = jax.vmap(partial(jnp.where, size=max_conn_size, fill_value=-1))(
            mask_jc
        )
        return (
            xp_jc[jnp.arange(len(ind))[:, None], ind],
            mels_jc[jnp.arange(len(ind))[:, None], ind],
            conn_b2,
        )


@register_pytree_node_class
class LocalOperatorJax(LocalOperatorBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.LocalOperator`.
    """

    def _setup(self, force=False):
        if force or not self._initialized:
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
            for s in np.unique(acting_size):
                (indices,) = np.where(acting_size == s)
                self._local_states_jax.append(local_states[indices, :s])
                self._acting_on_jax.append(acting_on[indices, :s])
                self._n_conns_jax.append(
                    n_conns[indices, :]
                )  # TODO !! can we remove some of them, as a function of s???
                self._diag_mels_jax.append(diag_mels[indices])  # TODO how long
                self._all_x_prime_jax.append(all_x_prime[indices, :, :, :s])
                self._all_mels_jax.append(all_mels[indices])
                self._basis_jax.append(basis[indices])

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
