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

from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
import flax
from jax import numpy as jnp
from flax import struct

from plum import dispatch

from netket.utils.types import PyTree, Array

from .base import AbstractSMatrix


@struct.dataclass
class AbstractLazySMatrix(AbstractSMatrix):
    """
    Lazy representation of an S Matrix behving like a linear operator.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.

    This is an abstract type that can be subclassed to define new lazy
    S matrix representations.
    To subclass this class, simply implement :code:`__matmul__(self, other)`
    and :code:`solve(self, y, *kwargs)`
    """

    apply_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray] = struct.field(
        pytree_node=False
    )
    """The forward pass of the Ansatz."""

    params: PyTree
    """The first input to apply_fun (parameters of the ansatz)."""

    samples: jnp.ndarray
    """The second input to apply_fun (points where the ansatz is evaluated)."""

    model_state: Optional[PyTree] = None
    """Optional state of the ansataz."""

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.s

        Returns:
            A dense matrix representation of this S matrix.
        """
        Npars = nkjax.tree_size(self.params)
        I = jax.numpy.eye(Npars)
        return jax.vmap(lambda S, x: self @ x, in_axes=(None, 0))(self, I)
