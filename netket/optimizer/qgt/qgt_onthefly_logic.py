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
from netket.jax import tree_conj, tree_dot, tree_axpy

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
    y, vjp_fun = jax.vjp(forward_fn, params, samples)
    res = jax.tree_map(lambda x: vjp_fun(x)[0], w)
    return jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # allreduce w/ MPI.SUM


def O_mean(forward_fn, params, samples):

    r"""
    compute ⟨O⟩
    i.e. the mean of the rows of the jacobian of forward_fn
    """
    y = jax.eval_shape(forward_fn, params, samples)
    w = jnp.array(1.0 / (samples.shape[0] * mpi.n_nodes), dtype=y.dtype)
    w = jax.lax.broadcast(w, y.shape)
    return O_vjp(forward_fn, params, samples, w)


def O_mean_complex(forward_fn, params, samples):
    r"""
    compute ⟨O⟩
    for a ℂ→ℂ function f(x+iy) = u(x,y) + i v(x,y)
    returns the mean along axis 0 of (∂x + i ∂y) u, (∂x + i ∂y) v
    """
    y = jax.eval_shape(forward_fn, params, samples)
    w = jnp.array(1.0 / (samples.shape[0] * mpi.n_nodes), dtype=y.dtype)
    w = jax.lax.broadcast(w, y.shape)
    return O_vjp(forward_fn, params, samples, (w, -1.0j * w))


def OH_w(forward_fn, params, samples, w):
    r"""
    compute  Oᴴw
    (where ᴴ denotes the hermitian transpose)
    """

    # Oᴴw = (wᴴO)ᴴ
    # The transposition of the 1D arrays is omitted in the implementation:
    # (wᴴO)ᴴ -> (w* O)*

    return tree_conj(O_vjp(forward_fn, params, samples, w.conjugate()))


def Odagger_O_v(forward_fn, params, samples, v, *, center=False):
    r"""
    if center=False (default):
        compute ⟨O†O⟩v

    else (center=True):
        compute ⟨O†ΔO⟩v
        where ΔO = O-⟨O⟩
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
    compute ⟨ΔO†ΔO⟩ v

    where ΔO = O-⟨O⟩
    """

    complex_params = not nkjax.tree_leaf_isreal(params)
    complex_out = nkjax.is_complex(jax.eval_shape(forward_fn, params, samples))

    if holomorphic and complex_params and complex_out:
        # holomorphic ℂ→ℂ
        omean = O_mean(forward_fn, params, samples)

        def forward_fn_centered(p, x):
            return forward_fn(p, x) - tree_dot(p, omean)

    elif complex_out:
        # non-holomorphic ℂ→ℂ
        # ℝ&ℂ→ℂ
        # ℝ→ℂ

        omean_r, omean_i = O_mean_complex(forward_fn, params, samples)

        def forward_fn_centered(p, x):
            return forward_fn(p, x) - jax.lax.complex(
                tree_dot(p, omean_r).real, tree_dot(p, omean_i).real
            )

    else:
        # ℝ→ℝ
        # ℂ→ℝ
        # ℝ&ℂ→ℝ

        omean = O_mean(forward_fn, params, samples)

        def forward_fn_centered(p, x):
            return forward_fn(p, x) - tree_dot(p, omean).real

    return Odagger_O_v(forward_fn_centered, params, samples, v)


# TODO block the computations (in the same way as done with MPI) if memory consumtion becomes an issue
def mat_vec(
    v, forward_fn, params, samples, diag_shift, centered=True, holomorphic=True
):
    r"""
    compute (S + diag_shift) v

    where the elements of S are given by one of the following equivalent formulations:

    if centered=True (default): Sₖₗ = ⟨ΔOₖ†ΔOₗ⟩
    if centered=False : Sₖₗ = ⟨Oₖ†ΔOₗ⟩

    where ΔOₖ = Oₖ-⟨Oₖ⟩
    and Oₖ (operator) is derivative of the log wavefunction w.r.t parameter k
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
