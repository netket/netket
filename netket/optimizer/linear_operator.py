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

from typing import Callable, Optional, Tuple, Any

import jax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree


class _Uninitialized:
    pass


Uninitialized = _Uninitialized()

SolverT = Callable[["LinearOperator", PyTree], Tuple[PyTree, Any]]
"""Signature for the solver used by a linear operator."""


@struct.dataclass
class LinearOperator:
    """
    S matrix base class.
    This can either be a jnp matrix, a lazy wrapper, or anything, as long as
    it satisfies this basic API.

    An AbstractSMatrix must support the following API:

    - :code:`__matmul__(y)`, meaning you must be able to do S@vec, where vec is
        either a PyTree of parameters or it's dense ravelling.
    - :code:`solve(y, **kwargs)`, which must solve the linear system Sx=y with
        any method available, usually but not necessarily defined inside the
        sr object stored inside the AbstractSMatrix.
        This function must accept arbitrary additional arguments.


    Additionally, you get for free:
    - :code:`__call__(y)`, meaning you must be able to do S(vec) = S@vec.
        This is defined by the base class so you don't need to define it.
        This guarantees that you can pass this matrix to sparse solvers.
    - :code:`to_dense(self)`, that will concretize the dense representation.
        You get a slow default fallback by default, you might specify a faster
        override.

    """

    diag_shift: float = 0.00
    """Diagonal shift added to the S matrix."""

    # PUBLIC API: METHOD TO EXTEND IF YOU WANT TO DEFINE A NEW S object
    def __matmul__(self, vec):
        raise NotImplementedError()

    def __add__(self, eps):
        return self.replace(diag_shift=self.diag_shift + eps)

    # PUBLIC API: METHOD TO EXTEND Optionally IF YOU WANT TO DEFINE A NEW S object with
    # custom logic
    def _solve(
        self, solve_fun: SolverT, y: PyTree, *, x0: Optional[PyTree] = None, **kwargs
    ) -> PyTree:
        # dont pass x0 if it's unset.
        # some solvers might not need/require it.
        if x0 is not None:
            kwargs["x0"] = x0

        return solve_fun(self, y, **kwargs)

    # PUBLIC API: Extend _solve
    def solve(
        self, solve_fun: SolverT, y: PyTree, *, x0: Optional[PyTree] = None, **kwargs
    ) -> PyTree:
        """
        Solve the linear system x=⟨S⟩⁻¹⟨y⟩ with the chosen iterataive solver.

        Args:
            y: the vector y in the system above.
            kwargs: Any additional kwargs, which might or might not be used depending
                on the specific implementation.

        Returns:
            x: the PyTree solving the system.
            info: optional additional informations provided by the solver. Might be
                None if there are no additional informations provided.
        """
        return self._solve(jax.tree_util.Partial(solve_fun), y, x0=x0, **kwargs)

    # PUBLIC API: METHOD TO EXTEND IF YOU WANT TO DEFINE A NEW S object
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.s

        Returns:
            A dense matrix representation of this S matrix.
        """
        raise NotImplementedError()

    # PUBLIC API: Only override if you want to, but there should be no need.
    def __call__(self, vec):
        return self @ vec

    # PUBLIC API: Only override if you want to, but there should be no need.
    def __array__(self) -> jnp.ndarray:
        return self.to_dense()

    def __post_init__(self):
        pass
