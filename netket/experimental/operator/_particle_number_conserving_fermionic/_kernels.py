# this file contains the logic of the operators, essentially _jw_kernel

from functools import partial

import numpy as np
import itertools

import jax
import jax.numpy as jnp

from netket.jax import reduce_xor


def _comb(kl, n):
    """
    compute all combinations of n elements from a set kl
    """
    if len(kl) < n:
        return jnp.zeros((n, 0), dtype=kl.dtype)
    c = list(itertools.combinations(np.arange(len(kl)), n))
    return kl[np.array(c, dtype=kl.dtype).T[::-1]]


def _jw_kernel(k_destroy, l_create, x):
    # destroy
    xd = jax.vmap(lambda i: x.at[i].set(0))(k_destroy.T)
    # create
    xp = jax.vmap(jax.vmap(lambda x, i: x.at[i].set(1), in_axes=(None, 0)))(
        xd, l_create
    )

    m = jnp.arange(x.shape[-1], dtype=k_destroy.dtype)

    # we apply the destruction operators in descending order,
    # the jordan-wigner sign of an operator does not depend on sites larger than it, therefore,
    # given it is in normal order, we can compute it all in terms of the initial state.
    # (sum the axis which is the one of the indices we destroy/create (size number of operators//2))
    jw_mask_destroy = reduce_xor(k_destroy[..., None] > m, axes=0)

    # same for when we create again, except then have to apply it to the state where we already destroyed
    jw_mask_create = reduce_xor(l_create[..., None] > m, axes=2)

    create_was_empty = jax.vmap(jax.vmap(lambda x, i: ~x[i].any(), in_axes=(None, 0)))(
        xd, l_create
    )

    sgn_destroy = reduce_xor(jw_mask_destroy * x[None], axes=-1)
    sgn_create = reduce_xor(jw_mask_create * xd[:, None], axes=-1)
    sgn = sgn_create + sgn_destroy[:, None]
    sgn = jax.lax.bitwise_and(sgn, jnp.ones_like(sgn)).astype(bool)
    sign = 1 - 2 * sgn.astype(np.int8)

    return xp, sign, create_was_empty


@partial(jax.jit, static_argnums=0)
@partial(jnp.vectorize, signature="(n)->(m,n),(m)", excluded=(0, 2, 3, 4))
def _get_conn_padded(n_fermions, x, index_array, create_array, weight_array):
    assert x.ndim == 1
    if index_array is not None:
        half_n_ops = index_array.ndim
    else:  # diagonal
        half_n_ops = weight_array.ndim

    if half_n_ops == 0:  # constant
        xp = x[None, :]
        mels = weight_array.reshape(xp.shape[:-1])
    else:
        dtype = x.dtype

        (l_occupied,) = jnp.where(x, size=n_fermions)
        k_destroy = _comb(l_occupied, half_n_ops)

        if index_array is None:  # diagonal
            weight = weight_array[tuple(k_destroy)]
            xp = x[None, :]
            # we first destroy in desc order, then create
            # sites not acted on cancel by the create/destroy pair of the same site,
            # so we can assume they are not there.
            # When we create all smaller sites acted on are 0,
            # therefore the jw sign is determined just from the signs from destroy.
            # Then it' is easy to see that only every other site counts (the rest cancel),
            # and the sign is given by (+1 if there is an even number of other sites, -1 if odd)
            # sign = [+,+,-,-,+,+,-,-,+,+,-,-,...][half_n_ops]
            sgn = (half_n_ops // 2) % 2
            sign = 1 - 2 * sgn
            mels = sign * weight.sum()[None]
        else:
            ind = index_array[tuple(k_destroy)]
            weight = weight_array[ind]
            l_create = create_array[ind]

            xp, sign, create_was_empty = _jw_kernel(k_destroy, l_create, x)
            mels = weight * sign * create_was_empty

            # make sure we don't return states w/ wrong number of electrons
            # because of the padding we check if the mel is 0
            # xp = jnp.where(create_was_empty[..., None], xp, x[..., None, None, :])
            xp = jnp.where((mels == 0)[:, :, None], x[None, None, :], xp)

            xp = jax.lax.collapse(xp, 0, xp.ndim - 1).astype(dtype)
            mels = jax.lax.collapse(mels, 0, mels.ndim)
    return xp, mels


# TODO do this in hilbert ?
@partial(jax.jit, static_argnames="n_spin_subsectors")
def unpack_spin_sectors(x, n_spin_subsectors=2):
    """
    split spin sectors of x
    """
    assert x.shape[-1] % n_spin_subsectors == 0
    x_ = x.reshape(x.shape[:-1] + (n_spin_subsectors, x.shape[-1] // n_spin_subsectors))
    return tuple(x_[..., i, :] for i in range(n_spin_subsectors))


@jax.jit
def pack_spin_sectors(*xs):
    """
    flatten spin sectors of xs
    """
    xs = jnp.broadcast_arrays(*xs)
    xd = xs[0]
    n_spin_subsectors = len(xs)
    res = jnp.zeros(
        xd.shape[:-1]
        + (
            n_spin_subsectors,
            xd.shape[-1],
        ),
        dtype=xd.dtype,
    )
    for i, xi in enumerate(xs):
        res = res.at[..., i, :].set(xi)
    return jax.lax.collapse(res, res.ndim - 2, res.ndim)


@partial(jax.jit, static_argnums=(0, 1))
@partial(jnp.vectorize, signature="(n),(n)->(m,n),(m,n),(m)", excluded=(0, 1, 4, 5, 6))
def _get_conn_padded_interaction_up_down(
    nelectron_down, nelectron_up, x_down, x_up, index_array, create_array, weight_array
):
    dtype = x_down.dtype

    assert x_down.ndim == 1
    if index_array is not None:
        assert index_array.ndim == 2
    else:  # diagonal
        assert weight_array.ndim == 2

    (down_occupied,) = jnp.where(x_down, size=nelectron_down)
    (up_occupied,) = jnp.where(x_up, size=nelectron_up)

    k_destroy_down, k_destroy_up = jnp.meshgrid(down_occupied, up_occupied)

    if index_array is None:  # diagonal
        weight = weight_array[k_destroy_down, k_destroy_up]
        xp_down = x_down[None, :]
        xp_up = x_up[None, :]
        sign = 1
        mels = sign * weight.sum()[None]
    else:
        ind = index_array[k_destroy_down, k_destroy_up].ravel()
        weight = weight_array[ind]
        l_create = create_array[ind]

        k_destroy_down = k_destroy_down.reshape(1, -1)
        k_destroy_up = k_destroy_up.reshape(1, -1)
        l_create_down = l_create[..., :1]
        l_create_up = l_create[..., 1:]

        xp_down, sign_down, down_create_is_not_occupied = _jw_kernel(
            k_destroy_down, l_create_down, x_down
        )
        xp_up, sign_up, up_create_is_not_occupied = _jw_kernel(
            k_destroy_up, l_create_up, x_up
        )

        up_is_diagonal = k_destroy_up[0][:, None] == l_create_up[..., 0]
        down_is_diagonal = k_destroy_down[0][:, None] == l_create_down[..., 0]
        both_not_occupied = (down_create_is_not_occupied | down_is_diagonal) & (
            up_create_is_not_occupied | up_is_diagonal
        )

        sign = sign_up * sign_down
        mels = weight * both_not_occupied * sign

        xp_down = jnp.where((mels == 0)[:, :, None], x_down[None, None, :], xp_down)
        xp_up = jnp.where((mels == 0)[:, :, None], x_up[None, None, :], xp_up)

        xp_down = jax.lax.collapse(xp_down, 0, xp_down.ndim - 1).astype(dtype)
        xp_up = jax.lax.collapse(xp_up, 0, xp_up.ndim - 1).astype(dtype)
        mels = jax.lax.collapse(mels, 0, mels.ndim)
    return xp_down, xp_up, mels


@partial(jax.jit, static_argnames=("n_fermions",))
def get_conn_padded_pnc(_operator_data, x, n_fermions):
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.integer):
        x = x.astype(jnp.int8)

    xp_list = []
    mels_list = []
    xp_diag = None
    mels_diag = 0
    for k, v in _operator_data["diag"].items():
        xp, mels = _get_conn_padded(n_fermions, x, *v)
        xp_diag = xp
        mels_diag = mels_diag + mels
        xp_list = [xp_diag]
        mels_list = [mels_diag]
    for k, v in _operator_data["offdiag"].items():
        xp, mels = _get_conn_padded(n_fermions, x, *v)
        xp_list.append(xp)
        mels_list.append(mels)
    xp = jnp.concatenate(xp_list, axis=-2)
    mels = jnp.concatenate(mels_list, axis=-1)
    return xp.astype(dtype), mels


@partial(jax.jit, static_argnames=("n_fermions_per_spin",))
def get_conn_padded_pnc_spin(_operator_data, x, n_fermions_per_spin):
    n_spin_subsectors = len(n_fermions_per_spin)
    xs = unpack_spin_sectors(x, n_spin_subsectors)
    xs_diag = tuple(a[..., None, :] for a in xs)
    dtype = xs[0].dtype

    xp_list = []
    mels_list = []
    xp_diag = None
    mels_diag = 0

    # TODO make sectors a jax array and use jax loop here to compile only once ?
    # (requires n_fermions_per_spin to be the same for all sectors)

    for (k, sectors), v in _operator_data["diag"].items():
        if k == 0:
            assert sectors == ()
            sectors = (0,)  # dummy sector

        for i in sectors:
            _, melsi = _get_conn_padded(n_fermions_per_spin[i], xs[i], *v)
            mels_diag = mels_diag + melsi
            xp_diag = x[..., None, :]

    for (k, sectors), v in _operator_data["mixed_diag"].items():
        if k != 4:
            raise NotImplementedError
        # TODO make sectors a jax array and use jax loop here to compile only once
        for i, j in sectors:
            assert i > j  # here i>j
            # e.g. take operator data to be c_ijkl + c_jilk so that here we only need to sum  ρ > σ (i.e. σ=d, ρ=u)
            *_, melsij = _get_conn_padded_interaction_up_down(
                n_fermions_per_spin[j], n_fermions_per_spin[i], xs[j], xs[i], *v
            )
            mels_diag = mels_diag + melsij
            xp_diag = x[..., None, :]

    # TODO optionally always add zero diagonal?
    if xp_diag is not None:
        xp_list = [xp_diag]
        mels_list = [mels_diag]

    for (k, sectors), v in _operator_data["offdiag"].items():
        # TODO make sectors a jax array and use jax loop here to compile only once ?
        # (requires n_fermions_per_spin to be the same for all sectors)
        for i in sectors:
            xpi, melsi = _get_conn_padded(n_fermions_per_spin[i], xs[i], *v)
            xpi = pack_spin_sectors(*xs_diag[:i], xpi, *xs_diag[i + 1 :])
            xp_list.append(xpi)
            mels_list.append(melsi)

    for (k, sectors), v in _operator_data["mixed_offdiag"].items():
        if k != 4:
            raise NotImplementedError
        for i, j in sectors:
            assert i > j  # here i>j
            # e.g. take operator data to be c_ijkl + c_jilk so that here we only need to sum  ρ > σ (i.e. σ=d, ρ=u)
            xpj, xpi, melsij = _get_conn_padded_interaction_up_down(
                n_fermions_per_spin[j], n_fermions_per_spin[i], xs[j], xs[i], *v
            )
            xpij = pack_spin_sectors(
                *xs_diag[:j], xpj, *xs_diag[j + 1 : i], xpi, *xs_diag[i + 1 :]
            )
            xp_list.append(xpij)
            mels_list.append(melsij)
    if len(xp_list) > 0:
        xp = jnp.concatenate(xp_list, axis=-2).astype(dtype)
        mels = jnp.concatenate(mels_list, axis=-1)
    else:
        xp = jnp.zeros((*x.shape[:-1], 0, x.shape[-1]), dtype=dtype)
        mels = jnp.zeros(xp.shape[:-1])  # TODO dtype?
    return xp, mels
