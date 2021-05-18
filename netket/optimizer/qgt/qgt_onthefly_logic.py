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
import jax.numpy as jnp
from functools import partial
from netket.stats import subtract_mean
from netket.utils import mpi
import netket.jax as nkjax
from netket.jax import tree_conj, tree_dot, tree_cast, tree_axpy

# Stochastic Reconfiguration with jvp and vjp

# Here O (Oks) is the jacobian (derivatives w.r.t. params) of the vectorised (in x) log wavefunction (forward_fn) evaluated at all samples.
# instead of computing (and storing) the full jacobian matrix jvp and vjp are used to implement the matrix vector multiplications with it.
# Expectation values are then just the mean over the leading dimension.


def O_jvp(forward_fn, params, samples, v):
    # TODO apply the transpose of mpi_sum_jax(x)[0] (allreduce) to v here
    # in order to get correct transposition with MPI
    _, res = jax.jvp(lambda p: forward_fn(p, samples), (params,), (v,))
    return res


def O_vjp(forward_fn, params, samples, w):
    _, vjp_fun = jax.vjp(forward_fn, params, samples)
    res, _ = vjp_fun(w)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # allreduce w/ MPI.SUM


def O_vjp_rc(forward_fn, params, samples, w):
    _, vjp_fun = jax.vjp(forward_fn, params, samples)
    res_r, _ = vjp_fun(w)
    res_i, _ = vjp_fun(-1.0j * w)
    res = jax.tree_multimap(jax.lax.complex, res_r, res_i)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # allreduce w/ MPI.SUM


def O_mean(forward_fn, params, samples, holomorphic=True):
    r"""
    compute \langle O \rangle
    i.e. the mean of the rows of the jacobian of forward_fn
    """

    # determine the output type of the forward pass
    dtype = jax.eval_shape(forward_fn, params, samples).dtype
    w = jnp.ones(samples.shape[0], dtype=dtype) * (
        1.0 / (samples.shape[0] * mpi.n_nodes)
    )

    homogeneous = nkjax.tree_ishomogeneous(params)
    real_params = not nkjax.tree_leaf_iscomplex(params)
    real_out = not nkjax.is_complex(jax.eval_shape(forward_fn, params, samples))

    if homogeneous and (real_params or holomorphic):
        if real_params and not real_out:
            # R->C
            return O_vjp_rc(forward_fn, params, samples, w)
        else:
            # R->R and holomorphic C->C
            return O_vjp(forward_fn, params, samples, w)
    else:
        # R&C -> C
        # non-holomorphic
        # C->R
        assert False


def OH_w(forward_fn, params, samples, w):
    r"""
    compute  O^H w
    (where ^H is the hermitian transpose)
    """

    # O^H w = (w^H O)^H
    # The transposition of the 1D arrays is omitted in the implementation:
    # (w^H O)^H -> (w* O)*

    # TODO The allreduce in O_vjp could be deferred until after the tree_cast
    # where the amount of data to be transferred would potentially be smaller
    res = tree_conj(O_vjp(forward_fn, params, samples, w.conjugate()))

    return tree_cast(res, params)


def Odagger_O_v(forward_fn, params, samples, v, *, center=False):
    r"""
    if center=False (default):
        compute \langle O^\dagger O \rangle v

    else (center=True):
        compute \langle O^\dagger \Delta O \rangle v
        where \Delta O = O - \langle O \rangle
    """

    # w is an array of size n_samples; each MPI rank has its own slice
    w = O_jvp(forward_fn, params, samples, v)
    # w /= n_samples (elementwise):
    w = w * (1.0 / (samples.shape[0] * mpi.n_nodes))

    if center:
        w = subtract_mean(w)  # w/ MPI

    return OH_w(forward_fn, params, samples, w)


Odagger_DeltaO_v = partial(Odagger_O_v, center=True)


def DeltaOdagger_DeltaO_v(forward_fn, params, samples, v, holomorphic=True):

    r"""
    compute \langle \Delta O^\dagger \Delta O \rangle v

    where \Delta O = O - \langle O \rangle
    """

    homogeneous = nkjax.tree_ishomogeneous(params)
    real_params = not nkjax.tree_leaf_iscomplex(params)
    #  real_out = not nkjax.is_complex(jax.eval_shape(forward_fn, params, samples))

    if not (homogeneous and (real_params or holomorphic)):
        # everything except R->R, holomorphic C->C and R->C
        params, reassemble = nkjax.tree_to_real(params)
        v, _ = nkjax.tree_to_real(v)
        _forward_fn = forward_fn

        def forward_fn(p, x):
            return _forward_fn(reassemble(p), x)

    omean = O_mean(forward_fn, params, samples, holomorphic=holomorphic)

    def forward_fn_centered(p, x):
        return forward_fn(p, x) - tree_dot(p, omean)

    res = Odagger_O_v(forward_fn_centered, params, samples, v)

    if not (homogeneous and (real_params or holomorphic)):
        res = reassemble(res)
    return res


# TODO block the computations (in the same way as done with MPI) if memory consumtion becomes an issue
def mat_vec(
    v, forward_fn, params, samples, diag_shift, centered=True, holomorphic=True
):
    r"""
    compute (S + diag_shift) v

    where the elements of S are given by one of the following equivalent formulations:

    if centered=True (default): S_kl = \langle \Delta O_k^\dagger \Delta O_l \rangle
    if centered=False : S_kl = \langle O_k^\dagger \Delta O_l \rangle

    where \Delta O_k = O_k - \langle O_k \rangle
    and O_k (operator) is derivative of the log wavefunction w.r.t parameter k
    The expectation values are calculated as mean over the samples

    v: a pytree with the same structure as params
    forward_fn(params, x): a vectorised function returning the logarithm of the wavefunction for each configuration in x
    params: pytree of parameters with arrays as leaves
    samples: an array of samples (when using MPI each rank has its own slice of samples)
    diag_shift: a scalar diagonal shift
    holomorphic: whether forward_fn is holomorphic (only needed if centered=True and forward_fn has complex params and output)
    """

    if centered:
        f = partial(DeltaOdagger_DeltaO_v, holomorphic=holomorphic)
    else:
        f = Odagger_DeltaO_v
    res = f(forward_fn, params, samples, v)
    # add diagonal shift:
    res = tree_axpy(diag_shift, v, res)  # res += diag_shift * v
    return res
