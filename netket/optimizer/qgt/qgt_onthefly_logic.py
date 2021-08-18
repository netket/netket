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

# Stochastic Reconfiguration with jvp and vjp

# This file implements a factory function that returns a function that multiplies its
# input with the S-matrix, defined as Sₖₗ = ⟨ΔOₖ* ΔOₗ⟩, where ΔOₖ = Oₖ-⟨Oₖ⟩,
# and Oₖ is the derivative of log ψ w.r.t. parameter #k.

# Given the Jacobian J of the neural network, S = JᴴMᴴMJ, where M is an
# n_sample × n_sample matrix which subtracts the mean. As M is a projector, S = JᴴMJ.

# The factory function is used so that the gradient calculations in jax.linearize can be
# jitted; the arguments of mat_vec are outputs of jax.linearize, which are pytrees


def mat_vec(jvp_fn, v, diag_shift):
    # Save linearisation work
    # TODO move to mat_vec_factory after jax v0.2.19
    vjp_fn = jax.linear_transpose(jvp_fn, v)

    w = jvp_fn(v)
    w = w * (1.0 / (w.size * mpi.n_nodes))
    w = subtract_mean(w)  # w/ MPI
    # Oᴴw = (wᴴO)ᴴ = (w* O)* since 1D arrays are not transposed
    # vjp_fn packages output into a length-1 tuple
    (res,) = tree_conj(vjp_fn(w.conjugate()))
    res = jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)

    return tree_axpy(diag_shift, v, res)  # res + diag_shift * v


@partial(jax.jit, static_argnums=0)
def mat_vec_factory(forward_fn, params, model_state, samples):
    # "forward function" that maps params to outputs
    def fun(W):
        return forward_fn({"params": W, **model_state}, samples)

    _, jvp_fn = jax.linearize(fun, params)
    return Partial(mat_vec, jvp_fn)
