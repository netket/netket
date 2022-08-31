# Copyright 2022 The NetKet Authors - All rights reserved.
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

from textwrap import dedent

import jax
from jax import numpy as jnp

from netket.utils.types import PyTree
from netket.utils.errors import ComplexDomainError
from netket.jax.utils import RealImagTuple


def check_valid_vector_type(x: PyTree, target: PyTree):
    """
    Raises a TypeError if x is complex where target is real, because it is not
    supported by QGTOnTheFly and the imaginary part would be discarded after
    anyhow.
    """

    def check(x, target, target_im=None):
        par_iscomplex = jnp.iscomplexobj(x)

        # Account for split real-imaginary part in Jacobian*** methods
        vec_iscomplex = target_im is not None or jnp.iscomplexobj(target)

        if not par_iscomplex and vec_iscomplex:
            raise ComplexDomainError(
                dedent(
                    """
                    Cannot multiply the (real part of the) QGT by a complex vector.
                    You should either take the real part of the vector, or perform
                    the multiplication against the real and imaginary part of the
                    vector separately and then recomposing the two.

                    This is happening because you have real parameters or a non-holomorphic
                    complex wave function. In this case, the Quantum Geometric Tensor object
                    only stores the real part of the QGT.

                    If you were executing a matmul `G@vec`, try using:

                       >>> vec_real = jax.tree_map(lambda x: x.real, vec)
                       >>> G@vec_real

                    If you used the QGT in a linear solver, try using:

                       >>> vec_real = jax.tree_map(lambda x: x.real, vec)
                       >>> G.solve(linear_solver, vec_real)

                    to fix this error.

                    Be careful whether you need the real or imaginary part
                    of the vector in your equations!
                    """
                )
            )

    try:
        if isinstance(target, RealImagTuple):
            jax.tree_map(check, x, target.real, target.imag)
        else:
            jax.tree_map(check, x, target)
    except ValueError:
        # catches jax tree map errors
        pars_struct = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), x)
        vec_struct = jax.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), target
        )

        raise ValueError(
            "PyTree mismatch: Parameters have shape \n\n"
            f"{pars_struct}\n\n"
            "but the vector has shape \n\n"
            f"{vec_struct}\n\n"
        )
