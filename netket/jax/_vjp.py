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

from typing import Any, Literal, overload, TypeVar
from collections.abc import Callable
from functools import partial

import jax

from jax import numpy as jnp
from jax.tree_util import Partial, tree_map

from netket.utils import HashablePartial

from ._utils_tree import tree_leaf_iscomplex, eval_shape

# These TypeVars are used below to express the fact that function types
# (i.e. call signatures) are invariant under the vmap transformation.
T = TypeVar("T")
U = TypeVar("U")


# _grad_CC, _RR and _RC are the chunked gradient functions for machines going
# from R -> C, R->R and R->C. Ditto for vjp
# Thee reason why R->C is more complicated is that it splits the calculation
# into the real and complex part in order to be more efficient.


def _cmplx(re, im, conj=False):
    """
    Safely convert real and imaginary part to a complex number, considering
    `float0` dtypes which cannot be summed upon.

    Those types appear when computing the `vjp` of functions with integer
    inputs.
    """
    # detect tangent-0 dtypes
    is_re_0 = jax.dtypes.issubdtype(re.dtype, jax.dtypes.float0)
    is_im_0 = jax.dtypes.issubdtype(re.dtype, jax.dtypes.float0)
    if is_re_0 or is_im_0:
        return re
    else:
        if conj:
            return re - 1j * im
        else:
            return re + 1j * im


def vjp_fun_cc(out_dtype, conjugate, _vjp_fun, ȳ):
    ȳ = jnp.asarray(ȳ, dtype=out_dtype)

    dȳ = _vjp_fun(ȳ)

    if conjugate:
        dȳ = tree_map(jnp.conjugate, dȳ)

    return dȳ


def vjp_cc(
    fun: Callable, *primals, has_aux: bool = False, conjugate: bool = False
) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
    if has_aux:
        out, _vjp_fun, aux = jax.vjp(fun, *primals, has_aux=True)
    else:
        out, _vjp_fun = jax.vjp(fun, *primals, has_aux=False)

    vjp_fun = Partial(HashablePartial(vjp_fun_cc, out.dtype, conjugate), _vjp_fun)

    if has_aux:
        return out, vjp_fun, aux
    else:
        return out, vjp_fun


def vjp_fun_rr(primals_out_dtype, conjugate, _vjp_fun, ȳ):
    """
    function computing the vjp product for a R->R function.
    """
    if not jnp.iscomplexobj(ȳ):
        out = _vjp_fun(jnp.asarray(ȳ, dtype=primals_out_dtype))
    else:
        out_r = _vjp_fun(jnp.asarray(ȳ.real, dtype=primals_out_dtype))
        out_i = _vjp_fun(jnp.asarray(ȳ.imag, dtype=primals_out_dtype))
        out = tree_map(partial(_cmplx, conj=conjugate), out_r, out_i)

    return out


def vjp_rr(
    fun: Callable, *primals, has_aux: bool = False, conjugate: bool = False
) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
    if has_aux:
        primals_out, _vjp_fun, aux = jax.vjp(fun, *primals, has_aux=True)
    else:
        primals_out, _vjp_fun = jax.vjp(fun, *primals, has_aux=False)

    vjp_fun = Partial(
        HashablePartial(vjp_fun_rr, primals_out.dtype, conjugate), _vjp_fun
    )

    if has_aux:
        return primals_out, vjp_fun, aux
    else:
        return primals_out, vjp_fun


def vjp_fun_rc(vals_dtype, conjugate, vjp_fun, ȳ):
    """
    function computing the vjp product for a R->C function.
    """
    # val = vals_r + vals_j
    v_j = vjp_fun(jnp.asarray(ȳ, dtype=vals_dtype))
    vs_j = vjp_fun(jnp.asarray(-1j * ȳ, dtype=vals_dtype))

    out = tree_map(partial(_cmplx, conj=False), v_j, vs_j)

    if conjugate:
        out = tree_map(jnp.conjugate, out)

    return out


def vjp_rc(
    fun: Callable, *primals, has_aux: bool = False, conjugate: bool = False
) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
    if has_aux:
        primals_out, vjp_fun, aux = jax.vjp(fun, *primals, has_aux=True)

    else:
        primals_out, vjp_fun = jax.vjp(fun, *primals, has_aux=False)

    vjp_fun = Partial(
        HashablePartial(vjp_fun_rc, primals_out.dtype, conjugate),
        vjp_fun,
    )

    if has_aux:
        return primals_out, vjp_fun, aux
    else:
        return primals_out, vjp_fun


# This function dispatches to the right
@overload
def vjp(
    fun: Callable[..., T],
    *primals: Any,
    has_aux: Literal[False] = False,
    conjugate: bool = False,
) -> tuple[T, Callable]: ...


@overload
def vjp(
    fun: Callable[..., tuple[T, U]],
    *primals: Any,
    has_aux: Literal[True] = True,  # Fix the default value
    conjugate: bool = False,
) -> tuple[T, Callable, U]: ...


def vjp(
    fun: Callable, *primals, has_aux: bool = False, conjugate: bool = False
) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
    # output dtype
    out_shape = eval_shape(fun, *primals, has_aux=has_aux)

    if tree_leaf_iscomplex(primals):
        if jnp.iscomplexobj(out_shape):  # C -> C
            return vjp_cc(fun, *primals, has_aux=has_aux, conjugate=conjugate)
        else:  # C -> R
            return vjp_cc(fun, *primals, has_aux=has_aux, conjugate=conjugate)
    else:
        if jnp.iscomplexobj(out_shape):  # R -> C
            return vjp_rc(fun, *primals, has_aux=has_aux, conjugate=conjugate)
        else:  # R -> R
            return vjp_rr(fun, *primals, has_aux=has_aux, conjugate=conjugate)
