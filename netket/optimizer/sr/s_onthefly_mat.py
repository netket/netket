from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
import flax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree, Array
import netket.jax as nkjax

from .sr_onthefly_logic import mat_vec as mat_vec_onthefly, tree_cast

from .base import SR
from .s_lazy import AbstractLazySMatrix


@struct.dataclass
class SROnTheFly(SR):
    """
    Base class holding the parameters for the iterative solution of the
    SR system x = ⟨S⟩⁻¹⟨F⟩, where S is a lazy linear operator

    Tolerances are applied according to the formula
    :code:`norm(residual) <= max(tol*norm(b), atol)`
    """

    centered: bool = struct.field(pytree_node=False, default=True)
    """Uses S=⟨ΔÔᶜΔÔ⟩ if True (default), S=⟨ÔᶜΔÔ⟩ otherwise. The two forms are 
    mathematically equivalent, but might lead to different results due to numerical
    precision. The non-centered variant should be approximately 33% faster.
    """

    def create(self, vstate, **kwargs) -> "SMatrixOnTheFly":
        """
        Construct the Lazy representation of the S corresponding to this SR type.

        Args:
            vstate: The Variational State
        """
        return SMatrixOnTheFly(
            apply_fun=vstate._apply_fun,
            params=vstate.parameters,
            samples=vstate.samples,
            model_state=vstate.model_state,
            sr=self,
        )


@struct.dataclass
class SMatrixOnTheFly(AbstractLazySMatrix):
    """
    Lazy representation of an S Matrix computed by performing 2 jvp
    and 1 vjp products, using the variational state's model, the
    samples that have already been computed, and the vector.

    The S matrix is not computed yet, but can be computed by calling
    :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    def __post_init__(self):
        super().__post_init__()

        if jnp.ndim(self.samples) != 2:
            samples_r = self.samples.reshape((-1, self.samples.shape[-1]))
            object.__setattr__(self, "samples", samples_r)

    def __matmul__(self, y):
        return lazysmatrix_mat_treevec(self, y)


@jax.jit
def lazysmatrix_mat_treevec(
    S: SMatrixOnTheFly, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:
    """
    Perform the lazy mat-vec product, where vec is either a tree with the same structure as
    params or a ravelled vector
    """

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

    vec = tree_cast(vec, S.params)

    def fun(W, σ):
        return S.apply_fun({"params": W, **S.model_state}, σ)

    mat_vec = partial(
        mat_vec_onthefly,
        forward_fn=fun,
        params=S.params,
        samples=S.samples,
        diag_shift=S.sr.diag_shift,
        centered=S.sr.centered,
    )

    res = mat_vec(vec)

    if ravel_result:
        res, _ = nkjax.tree_ravel(res)

    return res
