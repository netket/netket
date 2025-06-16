# this file contains the logic of the operators, essentially _jw_kernel

from typing import Union

from functools import partial

import numpy as np
import itertools

import jax
import jax.numpy as jnp

from netket.jax import COOArray
from netket.utils.types import Array

from ._operator_data import PNCOperatorDataType


def _comb(kl: Array, n: int) -> Array:
    r"""
    compute all combinations of n elements from a set kl
    Args:
        kl: 1d array of elements
        n: size n
    Returns:
        an array containg all combinations of size n of elemenets in kl
    """
    if len(kl) < n:
        return jnp.zeros((n, 0), dtype=kl.dtype)
    c = list(itertools.combinations(np.arange(len(kl)), n))
    return kl[np.array(c, dtype=kl.dtype).T[::-1]]


def _jw_kernel(
    k_destroy: Array, l_create: Array, x: Array
) -> tuple[Array, Array, Array]:
    r"""
    compute all matrix elements :math:`x^\prime` such that :math:`\langle x^\prime | \hat c^\dagger_{l_m} \cdots \hat c^\dagger_{l_1} \hat c_{k_n} \cdots \hat c_{k_1} | x\rangle \neq 0`
    of a batch of states x.

    Args:
        k_destroy: an array of indices of the :math:`\hat c`
        l_create: an array of indices of the :math:`\hat c^\dagger`
        x: an array of states
    Returns:
        xp: matrix elements :math:`x^\prime`
        sign: sign (value) of the matrix element if it is nonzero, arbitrary value otherwise
        create_was_empty: if the matrix element is zero
    """
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
    jw_mask_destroy = jax.lax.reduce_xor(k_destroy[..., None] > m, axes=(0,))

    # same for when we create again, except then have to apply it to the state where we already destroyed
    jw_mask_create = jax.lax.reduce_xor(l_create[..., None] > m, axes=(2,))

    create_was_empty = jax.vmap(jax.vmap(lambda x, i: ~x[i].any(), in_axes=(None, 0)))(
        xd, l_create
    )

    sgn_destroy = jax.lax.reduce_xor(
        jw_mask_destroy * x[None], axes=((jw_mask_destroy * x[None]).ndim - 1,)
    )
    sgn_create = jax.lax.reduce_xor(
        jw_mask_create * xd[:, None], axes=((jw_mask_create * xd[:, None]).ndim - 1,)
    )
    sgn = sgn_create + sgn_destroy[:, None]
    sgn = jax.lax.bitwise_and(sgn, jnp.ones_like(sgn)).astype(bool)
    sign = 1 - 2 * sgn.astype(np.int8)

    return xp, sign, create_was_empty


@partial(jax.jit, static_argnums=0)
@partial(jnp.vectorize, signature="(n)->(m,n),(m)", excluded=(0, 2, 3, 4))
def _get_conn_padded(
    n_fermions: int,
    x: Array,
    index_array: Union[Array, COOArray, None],
    create_array: Union[Array, None],
    weight_array: Array,
) -> tuple[Array, Array]:
    r"""
    helper function for the matrix elements functions defined below

    does not know about spin sectors

    Args:
        n_fermionsnumber of electrons
        x: occupation vectors
        index_array, create_array, weight_array: internal (sparse) operator data representation
    Returns:
        connected states and corresponding matrix elements
    """
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


@partial(jax.jit, static_argnames="n_spin_subsectors")
def unpack_spin_sectors(x: Array, n_spin_subsectors: int = 2):
    r"""
    split spin sectors of x

    Args:
        a single stacked array of occupations
    Returns:
        one array for each spin sector
    """
    assert x.shape[-1] % n_spin_subsectors == 0
    x_ = x.reshape(x.shape[:-1] + (n_spin_subsectors, x.shape[-1] // n_spin_subsectors))
    return tuple(x_[..., i, :] for i in range(n_spin_subsectors))


@jax.jit
def pack_spin_sectors(*xs: tuple[Array]) -> Array:
    r"""
    flatten spin sectors of xs

    Args:
        xs: one array of occupations for each spin sector
    Returns:
        a single stacked array
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
    nelectron_down: int,
    nelectron_up: int,
    x_down: Array,
    x_up: Array,
    index_array: Union[Array, COOArray, None],
    create_array: Union[Array, None],
    weight_array: Array,
) -> tuple[Array, Array, Array]:
    r"""
    helper function for the matrix elements for a 2-body interaction term in two different spin sectors
    i.e. of the operator :math:`\sum_{ijkl} w_{ijkl} \hat c^\dagger_{i\downarrow}  \hat c^\dagger_{j_downarrow} \hat c^\dagger_{k_uparrow} \hat c^\dagger_{l_uparrow}`

    Args:
        nelectron_down, nelectron_up: number of electrons in the down and up sector
        x_down, x_up: occupation vectors in both sectors
        index_array, create_array, weight_array: internal (sparse) operator data representation
    Returns:
        connected states and corresponding matrix elements
    """
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
def get_conn_padded_pnc(
    _operator_data: PNCOperatorDataType, x: Array, n_fermions: int
) -> tuple[Array, Array]:
    r"""
    compute the connected elements for ParticleNumberConservingFermioperator2nd

    Args:
        _operator_data: internal sparse operator representation
        x: occupation vectors
        n_fermions: number of electrons
    Returns:
        connected states and corresponding matrix elements
    """
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
def get_conn_padded_pnc_spin(
    _operator_data: PNCOperatorDataType, x: Array, n_fermions_per_spin: tuple[int]
) -> tuple[Array, Array]:
    r"""
    compute the connected elements for ParticleNumberAndSpinConservingFermioperator2nd

    Args:
        _operator_data: internal sparse operator representation
        x: occupation vectors (with concatenated spin sectors)
        n_fermions_per_spin: number of electrons in each spin sector
    Returns:
        connected states and corresponding matrix elements
    """
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
            # act on a single sector: we use the non-spin mel code here for the mels
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
            # act on a single sector: we use the non-spin mel code here, and just apply an identity in the other sectors
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
