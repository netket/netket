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


def check_valid_vector_type(x: PyTree, target: PyTree):
    """
    Raises a TypeError if x is complex where target is real, because it is not
    supported by QGTOnTheFly and the imaginary part would be dicscarded after
    anyhow.
    """

    def check(x, target):
        par_iscomplex = jnp.iscomplexobj(x)

        # Account for split real-imaginary part in Jacobian*** methods
        if isinstance(target, tuple):
            vec_iscomplex = True if len(target) == 2 else False
        else:
            vec_iscomplex = jnp.iscomplexobj(target)

        if not par_iscomplex and vec_iscomplex:
            raise TypeError(
                dedent(
                    """
                    Cannot multiply the (real part of the) QGT by a complex vector.
                    You should either take the real part of the vector, or perform
                    the multiplication against the real and imaginary part of the
                    vector separately and then recomposing the two.
                    """
                )
            )

    jax.tree_multimap(check, x, target)
