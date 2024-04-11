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

from functools import partial

import jax
import jax.flatten_util
import jax.numpy as jnp

from netket.utils import mpi
from netket.utils.types import Array, PyTree, Scalar
from netket.jax import (
    tree_cast,
    tree_conj,
    tree_axpy,
)


def _jvp(oks: PyTree, v: PyTree) -> Array:
    """
    Compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    """
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    t = jax.tree_util.tree_map(td, oks, v)
    return jax.tree_util.tree_reduce(jnp.add, t)


def _vjp(oks: PyTree, w: Array) -> PyTree:
    """
    Compute the vector-matrix product between the vector w and the pytree jacobian oks
    """
    res = jax.tree_util.tree_map(partial(jnp.tensordot, w, axes=w.ndim), oks)
    return jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # MPI


def _mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    """
    Compute ⟨O† O⟩v = ∑ₗ ⟨Oₖᴴ Oₗ⟩ vₗ
    """
    res = tree_conj(_vjp(oks, _jvp(oks, v).conjugate()))
    return tree_cast(res, v)


def mat_vec(v: PyTree, centered_oks: PyTree, diag_shift: Scalar) -> PyTree:
    """
    Compute (S + δ) v = 1/n ⟨ΔO† ΔO⟩v + δ v = ∑ₗ 1/n ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ

    Only compatible with R→R, R→C, and holomorphic C→C
    for C→R, R&C→R, R&C→C and general C→C the parameters for generating ΔOⱼₖ should be converted to R,
    and thus also the v passed to this function as well as the output are expected to be of this form

    Args:
        v: pytree representing the vector v compatible with centered_oks
        centered_oks: pytree of gradients 1/√n ΔOⱼₖ
        diag_shift: a scalar diagonal shift δ
    Returns:
        a pytree corresponding to the sr matrix-vector product (S + δ) v
    """
    return tree_axpy(diag_shift, v, _mat_vec(v, centered_oks))
