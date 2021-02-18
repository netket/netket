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

from netket.utils import rename

from .sr_onthefly_logic import mat_vec as mat_vec_onthefly, tree_cast

from .base import SR

Ndarray = Any


@rename(r"SR{Onthefly}")
@struct.dataclass
class SR_otf(SR):
    """
    Base class holding the parameters for the iterative SR on the fly method.
    """

    tol: float = 1.0e-5
    atol: float = 0.0
    maxiter: int = None
    M: Optional[Union[Callable, Ndarray]] = None

    def create(self, *args, **kwargs):
        return LazySMatrix(*args, **kwargs)


@rename(r"SR{Onthefly,CG}")
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


@rename(r"SR{Onthefly,GMRES}")
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


@partial(jax.jit, static_argnums=0)
def apply_onthefly(apply_fun, params, samples, model_state, sr, grad, x0):
    # Preapply the model state so that when computing gradient we only
    # get gradient of parameeters
    def fun(W, σ):
        return apply_fun({"params": W, **model_state}, σ)

    grad = tree_cast(grad, params)
    # we could cachee this...
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, grad)

    _mat_vec = partial(
        mat_vec_onthefly,
        forward_fn=fun,
        params=params,
        samples=samples,
        diag_shift=sr.diag_shift,
    )
    solve_fun = sr.solve_fun()
    out, _ = solve_fun(_mat_vec, grad, x0=x0)
    return out


@struct.dataclass
class LazySMatrix:
    apply_fun: Callable
    params: Any
    samples: Any
    sr: SR_otf
    model_state: Any = None
    x0: Any = None

    def __call__(self, grad):
        return self.apply(grad)

    def apply(self, grad, x0=None):
        if jnp.ndim(self.samples) != 2:
            samples = self.samples.reshape((-1, self.samples.shape[-1]))

        if x0 is None:
            x0 = self.x0

        out = apply_onthefly(
            self.apply_fun,
            self.params,
            samples,
            self.model_state,
            self.sr,
            grad,
            x0,
        )

        return out
