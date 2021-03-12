# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

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

from netket.utils import rename_class
import netket.jax as nkjax

from .sr_onthefly_logic import mat_vec as mat_vec_onthefly, tree_cast

from .base import SR

Ndarray = Any


@rename_class(r"SR{Onthefly}")
@struct.dataclass
class SR_otf(SR):
    """
    Base class holding the parameters for the iterative SR on the fly method.
    """

    tol: float = 1.0e-5
    atol: float = 0.0
    maxiter: int = None
    M: Optional[Union[Callable, Ndarray]] = None
    centered: bool = struct.field(pytree_node=False, default=True)

    def create(self, *args, **kwargs):
        return LazySMatrix(*args, **kwargs)


@rename_class(r"SR{Onthefly,CG}")
@struct.dataclass
class SR_otf_cg(SR_otf):
    ...

    def solve_fun(self):
        return partial(
            jax.scipy.sparse.linalg.cg,
            tol=self.tol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self.M,
        )


@rename_class(r"SR{Onthefly,GMRES}")
@struct.dataclass
class SR_otf_gmres(SR_otf):
    restart: int = 20
    solve_method: str = struct.field(pytree_node=False, default="batched")

    def solve_fun(self):
        return partial(
            jax.scipy.sparse.linalg.gmres,
            tol=self.tol,
            atol=self.atol,
            maxiter=self.maxiter,
            M=self.M,
            restart=self.restart,
            solve_method=self.solve_method,
        )


@jax.jit
def apply_onthefly(S, grad, x0):
    # Preapply the model state so that when computing gradient we only
    # get gradient of parameeters
    def fun(W, σ):
        return S.apply_fun({"params": W, **S.model_state}, σ)

    grad = tree_cast(grad, S.params)
    # we could cache this...
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, grad)

    samples = S.samples
    if jnp.ndim(samples) != 2:
        samples = samples.reshape((-1, samples.shape[-1]))

    _mat_vec = partial(
        mat_vec_onthefly,
        forward_fn=fun,
        params=S.params,
        samples=samples,
        diag_shift=S.sr.diag_shift,
        centered=S.sr.centered,
    )
    solve_fun = S.sr.solve_fun()
    out, _ = solve_fun(_mat_vec, grad, x0=x0)
    return out


@jax.jit
def lazysmatrix_mat_treevec(S, vec):
    """
    Perform the lazy mat-vec product, where vec is either a tree with the same structure as
    params or a ravelled vector
    """

    def fun(W, σ):
        return S.apply_fun({"params": W, **S.model_state}, σ)

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
    )

    res = mat_vec(vec)

    if ravel_result:
        res, _ = nkjax.tree_ravel(res)

    return res


@struct.dataclass
class LazySMatrix:
    apply_fun: Callable = struct.field(pytree_node=False)
    params: Any
    samples: Any
    sr: SR_otf
    model_state: Any = None
    x0: Any = None

    def __matmul__(self, vec):
        return lazysmatrix_mat_treevec(self, vec)

    def __rtruediv__(self, grad):
        return self.solve(grad)

    def solve(self, grad, x0=None):
        if x0 is None:
            x0 = self.x0

        out = apply_onthefly(
            self,
            grad,
            x0,
        )

        return out

    @jax.jit
    def to_dense(self):
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        Npars = nkjax.tree_size(self.params)
        I = jax.numpy.eye(Npars)
        return jax.vmap(lambda S, x: self @ x, in_axes=(None, 0))(self, I)
