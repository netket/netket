# Copyright 2018 The JAX Authors.
# Copyright 2023 The NetKet Authors.
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


# adapted version of jax._src.ops.special.logsumexp with mpi support

import jax
import operator
import numpy as np
import jax.numpy as jnp
from jax import lax
from netket.utils.mpi import mpi_max_jax, mpi_sum_jax, mpi_allgather_jax, n_nodes


def _promote_args_inexact(_, *args):
    return args


def _canonicalize_axis(axis, num_dims):
    axis = operator.index(axis)
    if not -num_dims <= axis < num_dims:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {num_dims}"
        )
    if axis < 0:
        axis = axis + num_dims
    return axis


def _reduction_dims(a, axis):
    if axis is None:
        return (tuple(range(np.ndim(a))),) * 2
    elif not isinstance(axis, (np.ndarray, tuple, list)):
        axis = (axis,)
    canon_axis = tuple(_canonicalize_axis(x, np.ndim(a)) for x in axis)
    if len(canon_axis) != len(set(canon_axis)):
        raise ValueError(f"duplicate value in 'axis': {axis}")
    canon_pos_axis = tuple(x for x in canon_axis if isinstance(x, int))
    if len(canon_pos_axis) != len(canon_axis):
        return canon_pos_axis, canon_axis
    else:
        return canon_axis, canon_axis


def mpi_logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False, token=None):
    r"""Log-sum-exp reduction.

    Assumes axis 0 is distributed over MPI
    Args:
        ...
        token: an optional token for mpi
    returns:
        ...
        token: mpi token
    Refer to the documentation of jax.scipy.special.logsumexp for the remaining arguments.

    """
    if b is not None:
        a_arr, b_arr = _promote_args_inexact("logsumexp_mpi", a, b)
        a_arr = jnp.where(b_arr != 0, a_arr, -jnp.inf)
    else:
        (a_arr,) = _promote_args_inexact("logsumexp_mpi", a)
        b_arr = a_arr  # for type checking
    pos_dims, dims = _reduction_dims(a_arr, axis)
    amax = jnp.max(a_arr, axis=dims, keepdims=keepdims)
    print("dims", dims)
    if (
        0 in dims or dims == ()
    ) and n_nodes > 1:  # skip if not on mpi as jnp.max(..., axis=0) fails on scalar
        if np.issubdtype(amax.dtype, np.complexfloating):
            # TODO mpi_max_jax does not work with complex numbers
            # We would need lexicographic ordering just like jax.lax.max
            # (consider first real part then imag part if equal)
            # TODO figure out if we can use MPI.MAXLOC
            # as a workaround we use mpi_allgather, and do the reduction in jax
            all_amax, token = mpi_allgather_jax(amax, token=token)
            amax = jnp.max(all_amax, axis=0)
        else:
            amax, token = mpi_max_jax(amax, token=token)
    amax = lax.stop_gradient(
        lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0))
    )
    amax_with_dims = amax if keepdims else lax.expand_dims(amax, pos_dims)
    # fast path if the result cannot be negative.
    if b is None and not np.issubdtype(a_arr.dtype, np.complexfloating):
        tmp1 = lax.exp(lax.sub(a_arr, amax_with_dims))  # TODO MPI
        tmp2 = jnp.sum(tmp1, axis=dims, keepdims=keepdims)
        if 0 in dims or (
            dims == () and n_nodes > 1
        ):  # if scalar but on mpi we still need to reduce
            tmp2, token = mpi_sum_jax(tmp2, token=token)
        out = lax.add(lax.log(tmp2), amax)

        sign = jnp.where(jnp.isnan(out), out, 1.0)
        sign = jnp.where(jnp.isneginf(out), 0.0, sign).astype(out.dtype)
    else:
        expsub = lax.exp(lax.sub(a_arr, amax_with_dims))
        if b is not None:
            expsub = lax.mul(expsub, b_arr)
        sumexp = jnp.sum(expsub, axis=dims, keepdims=keepdims)
        if 0 in dims or (
            dims == () and n_nodes > 1
        ):  # if scalar but on mpi we still need to reduce
            sumexp, token = mpi_sum_jax(sumexp, token=token)

        sign = lax.stop_gradient(jnp.sign(sumexp))
        if np.issubdtype(sumexp.dtype, np.complexfloating):
            if return_sign:
                sumexp = sign * sumexp
            out = lax.add(lax.log(sumexp), amax)
        else:
            out = lax.add(lax.log(lax.abs(sumexp)), amax)
    if return_sign:
        return (out, sign)
    if b is not None:
        if not np.issubdtype(out.dtype, np.complexfloating):
            with jax.debug_nans(False):
                out = jnp.where(sign < 0, jnp.array(np.nan, dtype=out.dtype), out)
    return out, token
