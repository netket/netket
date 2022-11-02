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

from typing import Callable, Tuple

from jax import numpy as jnp

from netket.utils.types import Array, PyTree, Scalar
from netket.utils import mpi
import netket.jax as nkjax

from netket.jax.utils import RealImagTuple


def vec_to_real(vec: Array) -> Tuple[Array, Callable]:
    """
    If the input vector is real, splits the vector into real
    and imaginary parts and concatenates them along the 0-th
    axis.

    It is equivalent to changing the complex storage from AOS
    to SOA.

    Args:
        vec: a dense vector
    """
    if jnp.iscomplexobj(vec):
        out, reassemble = nkjax.tree_to_real(vec)
        out = jnp.concatenate([out.real, out.imag], axis=0)

        def reassemble_concat(x):
            x = RealImagTuple(jnp.split(x, 2, axis=0))
            return reassemble(x)

        return out, reassemble_concat

    else:
        return vec, lambda x: x


def mat_vec(v: PyTree, O: PyTree, diag_shift: Scalar) -> PyTree:
    w = O @ v
    res = jnp.tensordot(w.conj(), O, axes=w.ndim).conj()
    return mpi.mpi_sum_jax(res)[0] + diag_shift * v
