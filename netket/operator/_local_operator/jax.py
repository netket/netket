# this file contains a more-or-less 1:1 port of the numba localoperator to jax, using padding where necessary
# TODO consider a complete rewrite in the future


import numpy as np

import jax
import jax.numpy as jnp

from functools import partial
from jax.tree_util import Partial
from netket.jax import HashablePartial

from netket.operator import DiscreteJaxOperator, LocalOperator


from flax import struct


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
        all_x_prime = all_x_prime_jax[kk]
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


# same as _local_operator_kernel_jax but does flattening and unflattening of extra axes
@partial(jax.jit, static_argnums=(0, 1, 2))
def _local_operator_kernel_jax2(
    nonzero_diagonal, ncmax_jax, max_conn_size, mel_cutoff, op_args, x
):
    shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    xp, mels, n_conn = _local_operator_kernel_jax(
        nonzero_diagonal, ncmax_jax, max_conn_size, mel_cutoff, op_args, x
    )
    xp = xp.reshape(shape[:-1] + xp.shape[-2:])
    mels = mels.reshape(shape[:-1] + mels.shape[-1:])
    n_conn = n_conn.reshape(shape[:-1])
    return xp, mels, n_conn


def get_get_conn_padded_closure(self):
    # Args:
    #   self: the numba operator
    # TODO run setup before this

    local_states = jnp.asarray(self._local_states)
    basis = jnp.asarray(self._basis)
    constant = jnp.asarray(self._constant)
    diag_mels = jnp.asarray(self._diag_mels)
    n_conns = jnp.asarray(self._n_conns)
    all_mels = jnp.asarray(self._mels)
    all_x_prime = jnp.asarray(self._x_prime)
    acting_on = jnp.asarray(self._acting_on)

    mel_cutoff = None  # not implemented
    max_conn_size = self._max_conn_size
    acting_size = self._acting_size
    nonzero_diagonal = bool(self._nonzero_diagonal)

    local_states_jax = []
    acting_on_jax = []
    n_conns_jax = []
    diag_mels_jax = []
    all_x_prime_jax = []
    all_mels_jax = []
    basis_jax = []
    for s in np.unique(acting_size):
        (indices,) = np.where(acting_size == s)
        local_states_jax.append(local_states[indices, :s])
        acting_on_jax.append(acting_on[indices, :s])
        n_conns_jax.append(n_conns[indices, : 2 * s])  # TODO 2x is ok?
        diag_mels_jax.append(diag_mels[indices])  # TODO how long
        all_x_prime_jax.append(all_x_prime[indices, :, :, :s])
        all_mels_jax.append(all_mels[indices])
        basis_jax.append(basis[indices])
    op_args = (
        local_states_jax,
        acting_on_jax,
        n_conns_jax,
        diag_mels_jax,
        all_x_prime_jax,
        all_mels_jax,
        basis_jax,
        constant,
    )
    ncmax_jax = tuple(map(int, jax.tree_map(jnp.max, n_conns_jax)))

    return Partial(
        HashablePartial(
            _local_operator_kernel_jax2.__wrapped__,
            nonzero_diagonal,
            ncmax_jax,
            max_conn_size,
        ),
        mel_cutoff,
        op_args,
    )


# For now the only way to construct it is to first create the numba operator and then convert


@struct.dataclass
class LocalOperatorJax(DiscreteJaxOperator):
    get_conn_padded_fun: Partial
    operator: LocalOperator = struct.field(pytree_node=False)

    @jax.jit
    def get_conn_padded(self, x):
        xp, mels, _ = self.get_conn_padded_fun(x)
        return xp, mels

    @property
    def dtype(self):
        return self.operator.dtype

    @property
    def hilbert(self):
        return self.operator.hilbert

    @property
    def is_hermitian(self):
        return self.operator.is_hermitian

    @property
    def max_conn_size(self):
        return self.operator.max_conn_size

    @jax.jit
    def n_conn(self, x):
        _, _, n_conn = self.get_conn_padded_fun(x)
        return n_conn

    @classmethod
    def from_numba_operator(cls, local_operator_numba):
        local_operator_numba = local_operator_numba.copy()
        local_operator_numba._setup()
        gcp_fun = get_get_conn_padded_closure(local_operator_numba)
        return cls(gcp_fun, local_operator_numba)
