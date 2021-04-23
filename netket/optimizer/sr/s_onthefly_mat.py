from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
import flax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree, Array
from netket.utils import rename_class
import netket.jax as nkjax

from plum import dispatch

from .sr_onthefly_logic import mat_vec as mat_vec_onthefly, tree_cast

from .base import SR, AbstractSMatrix


@struct.dataclass
class SRLazy(SR):
    """
    Base class holding the parameters for the iterative solution of the
    SR system x = ⟨S⟩⁻¹⟨F⟩, where S is a lazy linear operator

    Tolerances are applied according to the formula
    :code:`norm(residual) <= max(tol*norm(b), atol)`
    """

    centered: bool = struct.field(pytree_node=False, default=True)
    """Uses S=⟨ΔÔᶜΔÔ⟩ if True (default), S=⟨ÔᶜΔÔ⟩ otherwise. The two forms are 
    mathematically equivalent, but might lead to different results due to numerical
    precision. The non-centered variaant should bee approximately 33% faster.
    """


@struct.dataclass
class LazySMatrix(AbstractSMatrix):
    """
    Lazy representation of an S Matrix behving like a linear operator.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    apply_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray] = struct.field(
        pytree_node=False
    )
    """The forward pass of the Ansatz."""

    params: PyTree
    """The first input to apply_fun (parameters of the ansatz)."""

    samples: jnp.ndarray
    """The second input to apply_fun (points where the ansatz is evaluated)."""

    sr: SR
    """Parameters for the solution of the system."""

    model_state: Optional[PyTree] = None
    """Optional state of the ansataz."""

    def __matmul__(self, vec):
        return lazysmatrix_mat_treevec(self, vec)

    def __rtruediv__(self, y):
        return self.solve(y)

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

    def __array__(self) -> jnp.ndarray:
        return self.to_dense()


@jax.jit
def lazysmatrix_mat_treevec(
    S: LazySMatrix, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:
    """
    Perform the lazy mat-vec product, where vec is either a tree with the same structure as
    params or a ravelled vector
    """

    def fun(W, σ):
        return S.apply_fun({"params": W, **S.model_state}, σ)

    # if hasa ndim it's an array and not a pytree
    if hasattr(vec, "ndim"):
        if not vec.ndim == 1:
            raise ValueError("Unsupported mat-vec for batches of vectors")
        # If the input is a vector
        if not nkjax.tree_size(S.params) == vec.size:
            raise ValueError(
                """Size mismatch between number of parameters ({nkjax.tree_size(S.params)}) 
                                and vector size {vec.size}.
                             """
            )

        _, unravel = nkjax.tree_ravel(S.params)
        vec = unravel(vec)
        ravel_result = True
    else:
        ravel_result = False

    samples = S.samples
    if jnp.ndim(samples) != 2:
        samples = samples.reshape((-1, samples.shape[-1]))

    vec = tree_cast(vec, S.params)

    mat_vec = partial(
        mat_vec_onthefly,
        forward_fn=fun,
        params=S.params,
        samples=samples,
        diag_shift=S.sr.diag_shift,
        centered=S.sr.centered,
    )

    res = mat_vec(vec)

    if ravel_result:
        res, _ = nkjax.tree_ravel(res)

    return res
