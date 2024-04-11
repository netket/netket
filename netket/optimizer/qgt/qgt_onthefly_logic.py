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

import jax
from jax.tree_util import Partial
from functools import partial
from netket.stats import subtract_mean
from netket.utils import mpi
from netket.jax import tree_conj, tree_axpy
from netket.jax import (
    scanmap,
    scan_reduce,
    scan_append,
    chunk,
)
from netket.jax.sharding import sharding_decorator

# Stochastic Reconfiguration with jvp and vjp

# This file implements a factory function that returns a function that multiplies its
# input with the S-matrix, defined as Sₖₗ = ⟨ΔOₖ* ΔOₗ⟩, where ΔOₖ = Oₖ-⟨Oₖ⟩,
# and Oₖ is the derivative of log ψ w.r.t. parameter #k.

# Given the Jacobian J of the neural network, S = JᴴMᴴMJ, where M is an
# n_sample × n_sample matrix which subtracts the mean. As M is a projector, S = JᴴMJ.

# The factory function is used so that the gradient calculations in jax.linearize can be
# jitted; the arguments of mat_vec are outputs of jax.linearize, which are pytrees


def _mat_vec(jvp_fn, v, diag_shift, pdf=None):
    # Save linearisation work
    # TODO move to mat_vec_factory after jax v0.2.19
    vjp_fn = jax.linear_transpose(jvp_fn, v)

    w = jvp_fn(v)
    if pdf is None:
        w = w * (1.0 / (w.size * mpi.n_nodes))
        w = subtract_mean(w)  # w/ MPI
    else:
        w = pdf * (w - mpi.mpi_sum_jax(pdf @ w)[0])
    # Oᴴw = (wᴴO)ᴴ = (w* O)* since 1D arrays are not transposed
    # vjp_fn packages output into a length-1 tuple
    (res,) = tree_conj(vjp_fn(w.conjugate()))
    res = jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)

    return tree_axpy(diag_shift, v, res)  # res + diag_shift * v


@partial(jax.jit, static_argnums=0)
def mat_vec_factory(forward_fn, params, model_state, samples, pdf=None):
    """
    Prepare a function which computes the regularized SR matrix-vector product
    S v = ⟨ΔO† ΔO⟩v + δ v = ∑ₗ ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ
    using jvp and vjp.
    If pdf is provided uses the weighted expectation
    ⟨A⟩ = ∑ₓ pdf(x) A(x)
    otherwise uses the empirical estimate
    ⟨A⟩ = 1/n ∑ₓ A(x)
    Args:
        forward_fn: The forward pass of the Ansatz
        params : a pytree of parameters p
        model_state: untrained state parameters of the model
        samples : an array of (n in total) samples σ
        pdf: a vector of weights/probabiltiy density function to weight each
             sample in the expectation values;
             Pass |ψ(x)|^2 if exact optimization is being used, else None
    Returns:
        a function which does the SR matrix-vector product equal to
        lambda v,δ : (S + δ I) v
    """

    # "forward function" that maps params to outputs
    def fun(W):
        return forward_fn({"params": W, **model_state}, samples)

    _, jvp_fn = jax.linearize(fun, params)
    return Partial(_mat_vec, jvp_fn, pdf=pdf)


# -------------------------------------------------------------------------------
# Methods below are needed for the chunked version of QGTOnTheFly


@partial(sharding_decorator, sharded_args_tree=(False, False, True, False, False))
def _O_jvp(forward_fn, params, samples, v, chunk_size):
    @partial(scanmap, scan_fun=scan_append, argnums=2)
    def __O_jvp(forward_fn, params, samples, v):
        # TODO apply the transpose of sum_inplace (allreduce) to the arg v here
        # in order to get correct transposition with MPI
        _, res = jax.jvp(lambda p: forward_fn(p, samples), (params,), (v,))
        return res

    samples, unchunk_fn = chunk(samples, chunk_size)
    res = __O_jvp(forward_fn, params, samples, v)
    return unchunk_fn(res)


@partial(
    sharding_decorator,
    sharded_args_tree=(False, False, True, True, False),
    reduction_op_tree=jax.lax.psum,
)
def _O_vjp(forward_fn, params, samples, w, chunk_size):
    @partial(scanmap, scan_fun=scan_reduce, argnums=(2, 3))
    def __O_vjp(forward_fn, params, samples, w):
        _, vjp_fun = jax.vjp(forward_fn, params, samples)
        res, _ = vjp_fun(w)
        return res

    samples, _ = chunk(samples, chunk_size)
    w, _ = chunk(w, chunk_size)
    res = __O_vjp(forward_fn, params, samples, w)
    return res


def _OH_w(forward_fn, params, samples, w, chunk_size):
    return tree_conj(_O_vjp(forward_fn, params, samples, w.conjugate(), chunk_size))


def _Odagger_DeltaO_v(forward_fn, params, samples, v, chunk_size, pdf=None):
    w = _O_jvp(forward_fn, params, samples, v, chunk_size)
    if pdf is None:
        w = w * (1.0 / (samples.shape[0] * mpi.n_nodes))
        w = subtract_mean(w)  # w/ MPI
    else:
        w = pdf * (w - mpi.mpi_sum_jax(pdf @ w)[0])
    res = _OH_w(forward_fn, params, samples, w, chunk_size)
    return jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # MPI


# @partial(jax.jit, static_argnums=1)
def _mat_vec_chunked(forward_fn, params, samples, v, diag_shift, chunk_size, pdf=None):
    assert samples.ndim == 2  # require flat samples, axis 0 can be sharded
    res = _Odagger_DeltaO_v(forward_fn, params, samples, v, chunk_size, pdf)
    return tree_axpy(diag_shift, v, res)


def _mat_vec_chunked_transposable(
    forward_fn, params, samples, v, diag_shift, chunk_size, pdf=None
):
    extra_args = (params, samples, diag_shift, pdf)

    def _mv(extra_args, x):
        params, samples, diag_shift, pdf = extra_args
        return _mat_vec_chunked(
            forward_fn, params, samples, x, diag_shift, chunk_size, pdf
        )

    def _mv_trans(extra_args, y):
        # the linear operator is hermitian
        params, samples, diag_shift, pdf = extra_args
        return tree_conj(
            _mat_vec_chunked(
                forward_fn, params, samples, tree_conj(y), diag_shift, chunk_size, pdf
            )
        )

    return jax.custom_derivatives.linear_call(_mv, _mv_trans, extra_args, v)


@partial(jax.jit, static_argnums=(0, 5))
def mat_vec_chunked_factory(
    forward_fn, params, model_state, samples, pdf=None, chunk_size=None
):
    """

    Prepare a function which computes the regularized SR matrix-vector product
    S v = ⟨ΔO† ΔO⟩v + δ v = ∑ₗ ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ
    using jvp and vjp.
    If pdf is provided uses the weighted expectation
    ⟨A⟩ = ∑ₓ pdf(x) A(x)
    otherwise uses the empirical estimate
    ⟨A⟩ = 1/n ∑ₓ A(x)

    Same as mat_vec_factory but assumes samples are chunked,
    computations are performed in chunks.

    Args:
        forward_fn: The forward pass of the Ansatz
        params : a pytree of parameters p
        model_state: untrained state parameters of the model
        samples : an array of (n in total) chunked samples σ
        pdf: a vector of weights/probabiltiy density function to weight each
             sample in the expectation values;
             Pass |ψ(x)|^2 if exact optimization is being used, else None
             pdf is assumed to be chunked in correspondence with samples
    Returns:
        a function which does the SR matrix-vector product equal to
        lambda v,δ : (S + δ I) v
    """

    def fun(W, samples):
        return forward_fn({"params": W, **model_state}, samples)

    return Partial(
        partial(_mat_vec_chunked_transposable, fun, chunk_size=chunk_size),
        params,
        samples,
        pdf=pdf,
    )
    # return Partial(lambda f, *args: jax.jit(f)(*args), Partial(partial(_mat_vec_chunked, fun), params, samples))
