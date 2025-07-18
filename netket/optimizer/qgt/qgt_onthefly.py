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

from collections.abc import Callable
import warnings

import jax
from jax import numpy as jnp
from flax import struct

import netket.jax as nkjax
from netket.utils import timing, HashablePartial
from netket.utils.types import PyTree
from netket.utils.api_utils import partial_from_kwargs
from netket.errors import (
    IllegalHolomorphicDeclarationForRealParametersError,
    NonHolomorphicQGTOnTheFlyDenseRepresentationError,
    HolomorphicUndeclaredWarning,
)

from .common import check_valid_vector_type
from .qgt_onthefly_logic import mat_vec_factory, mat_vec_chunked_factory

from ..linear_operator import LinearOperator, SolverT, Uninitialized


@partial_from_kwargs
def QGTOnTheFly(
    vstate, *, chunk_size=None, holomorphic: bool | None = None, **kwargs
) -> "QGTOnTheFlyT":
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.

    Args:
        vstate: The variational State.
        chunk_size: If supplied, overrides the chunk size of the variational state
                    (useful for models where the backward pass requires more
                    memory than the forward pass).
    """
    if kwargs.pop("diag_scale", None) is not None:
        raise NotImplementedError(
            "\n`diag_scale` argument is not yet supported by QGTOnTheFly."
            "Please use `QGTJacobianPyTree` or `QGTJacobianDense`.\n\n"
            "You are also encouraged to nag the developers to support "
            "this feature.\n\n"
        )

    # TODO: Find a better way to handle this case
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        samples = vstate._all_states
        pdf = vstate.probability_distribution()
    else:
        samples = vstate.samples
        pdf = None

    if chunk_size is None:
        chunk_size = getattr(vstate, "chunk_size", None)

    return QGTOnTheFly_DefaultConstructor(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        samples,
        pdf=pdf,
        chunk_size=chunk_size,
        holomorphic=holomorphic,
        **kwargs,
    )


@timing.timed
def QGTOnTheFly_DefaultConstructor(
    apply_fun,
    parameters,
    model_state,
    samples,
    pdf=None,
    *,
    chunk_size: int | None = None,
    holomorphic: bool | None = None,
    **kwargs,
) -> "QGTOnTheFlyT":
    """ """
    if pdf is not None:
        if not pdf.shape == samples.shape[:-1]:
            raise ValueError(
                "The shape of pdf must match the shape of the samples, "
                f"instead you provided (pdf.shape={pdf.shape}) != "
                f"(samples.shape={samples.shape[:-1]})"
            )
        if pdf.ndim >= 2:
            pdf = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(pdf, 0, 2)

    # The code does not support an extra batch dimension
    if samples.ndim >= 3:
        # use jit so that we can do it on global shared array
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)

    n_samples_per_rank = samples.shape[0] // jax.device_count()
    if chunk_size is None or chunk_size >= n_samples_per_rank:
        mv_factory = mat_vec_factory
        chunking = False
    else:
        mv_factory = HashablePartial(mat_vec_chunked_factory, chunk_size=chunk_size)
        chunking = True

    # check if holomorphic or not
    if holomorphic:
        if nkjax.tree_leaf_isreal(parameters):
            raise IllegalHolomorphicDeclarationForRealParametersError()
        else:
            mode = "holomorphic"
    else:
        if not nkjax.tree_leaf_iscomplex(parameters):
            mode = "real"
        else:
            if holomorphic is None:
                warnings.warn(HolomorphicUndeclaredWarning(), UserWarning)
            mode = "complex"

    nkjax.jacobian_default_mode(
        apply_fun,
        parameters,
        model_state,
        samples,
        holomorphic=holomorphic,
    )

    mat_vec = mv_factory(
        forward_fn=apply_fun,
        params=parameters,
        model_state=model_state,
        samples=samples,
        pdf=pdf,
    )
    return QGTOnTheFlyT(
        _mat_vec=mat_vec,
        _params=parameters,
        _chunking=chunking,
        _mode=mode,
        **kwargs,
    )


@struct.dataclass
class QGTOnTheFlyT(LinearOperator):
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.
    """

    _mat_vec: Callable[[PyTree, float], PyTree] = Uninitialized  # type: ignore
    """The S matrix-vector product as generated by mat_vec_factory.
    It's a jax.Partial, so can be used as pytree_node."""

    _params: PyTree = Uninitialized  # type: ignore
    """The first input to apply_fun (parameters of the ansatz).
    Only used as a shape placeholder."""

    _chunking: bool = struct.field(pytree_node=False, default=False)
    """Whether the implementation with chunks is used which currently does not support vmapping over it"""

    _mode: str = struct.field(pytree_node=False, default=None)
    """Differentiation mode:
        - "real": for real-valued R->R and C->R Ansätze, splits the complex inputs
                  into real and imaginary part.
        - "complex": for complex-valued R->C and C->C Ansätze, splits the complex
                  inputs and outputs into real and imaginary part
        - "holomorphic": for any Ansätze. Does not split complex values.
        - "auto": autoselect real or complex.
    """

    def __matmul__(self, y):
        return onthefly_mat_treevec(self, y)

    def _solve(
        self, solve_fun: SolverT, y: PyTree, *, x0: PyTree | None, **kwargs
    ) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        # This condition will be true if the user specified `holomorphic=False` and
        # if the parameters are complex. If the parameters are real and the user
        # did not specify holomorphic we will have mode==real and if holomorphic is
        # True mode==holomorphic.
        #
        # We must check this because the AD implementation will compute the wrong
        # QGT in that case
        if self._mode == "complex":
            raise NonHolomorphicQGTOnTheFlyDenseRepresentationError()

        return _to_dense(self)

    def __repr__(self):
        return f"QGTOnTheFly(diag_shift={self.diag_shift})"


@jax.jit
def onthefly_mat_treevec(
    S: QGTOnTheFly, vec: PyTree | jnp.ndarray
) -> PyTree | jnp.ndarray:
    """
    Perform the lazy mat-vec product, where vec is either a tree with the same structure as
    params or a ravelled vector
    """

    # if has a ndim it's an array and not a pytree
    if hasattr(vec, "ndim"):
        if not vec.ndim == 1:
            raise ValueError("Unsupported mat-vec for chunks of vectors")
        # If the input is a vector
        if not nkjax.tree_size(S._params) == vec.size:
            raise ValueError(
                """Size mismatch between number of parameters ({nkjax.tree_size(S.params)})
                                and vector size {vec.size}.
                             """
            )

        _, unravel = nkjax.tree_ravel(S._params)
        vec = unravel(vec)
        ravel_result = True
    else:
        ravel_result = False

    check_valid_vector_type(S._params, vec)

    vec = nkjax.tree_cast(vec, S._params)

    res = S._mat_vec(vec, S.diag_shift)

    if ravel_result:
        res, _ = nkjax.tree_ravel(res)

    return res


@jax.jit
def _solve(
    self: QGTOnTheFlyT, solve_fun, y: PyTree, *, x0: PyTree | None, **kwargs
) -> PyTree:
    check_valid_vector_type(self._params, y)

    y = nkjax.tree_cast(y, self._params)

    # we could cache this...
    if x0 is None:
        x0 = jax.tree_util.tree_map(jnp.zeros_like, y)

    out, info = solve_fun(self, y, x0=x0)
    return out, info


@jax.jit
def _to_dense(self: QGTOnTheFlyT) -> jnp.ndarray:
    """
    Convert the lazy matrix representation to a dense matrix representation

    Returns:
        A dense matrix representation of this S matrix.
    """
    Npars = nkjax.tree_size(self._params)
    I = jax.numpy.eye(Npars)

    if self._chunking:
        # the linear_call in mat_vec_chunked does currently not have a jax batching rule,
        # so it cannot be vmapped but we can use scan
        # which is better for reducing the memory consumption anyway
        _, out = jax.lax.scan(lambda _, x: (None, self @ x), None, I)
    else:
        out = jax.vmap(lambda x: self @ x, in_axes=0)(I)

    if jnp.iscomplexobj(out):
        out = out.T

    return out
