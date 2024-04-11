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


import jax
from jax import numpy as jnp

from netket.utils.types import PyTree
from netket.errors import RealQGTComplexDomainError
from netket.jax._utils_tree import RealImagTuple


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
            raise RealQGTComplexDomainError()

    try:
        if isinstance(target, RealImagTuple):
            jax.tree_util.tree_map(check, x, target.real, target.imag)
        else:
            jax.tree_util.tree_map(check, x, target)
    except ValueError:
        # catches jax tree map errors
        pars_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), x
        )
        vec_struct = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), target
        )

        raise ValueError(
            "PyTree mismatch: Parameters have shape \n\n"
            f"{pars_struct}\n\n"
            "but the vector has shape \n\n"
            f"{vec_struct}\n\n"
        )
