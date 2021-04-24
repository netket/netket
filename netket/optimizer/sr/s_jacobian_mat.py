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

from jax import numpy as jnp
from flax import struct

from netket.utils.types import PyTree

from .base import AbstractSMatrix


@struct.dataclass
class JacobianSMatrix(AbstractSMatrix):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S, 
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contaianed in
    the field `sr`.
    """

    sr: SRJacobian
    """Parameters for the solution of the system."""

    O: jnp.ndarray
    """Gradients O_ij = ∂log ψ(σ_i)/∂p_j of the neural network 
    for all samples σ_i at given values of the parameters p_j
    Average <O_j> subtracted for each parameter
    Divided through by sqrt(#samples) to normalise S matrix
    If scale is not None, normalised to unit magnitude"""

    scale: Optional[jnp.ndarray] = None
    """If not None, gives scale factors with which O is normalised"""

    x0: Optional[PyTree] = None
    """Optional initial guess for the iterative solution."""

    @jax.jit
    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        if not hasattr(vec, "ndim"):
            vec, unravel = nkjax.tree_ravel(vec)
        else:
            unravel = lambda x: x

        return unravel(
            jnp.transpose(jnp.conj(jnp.transpose(jnp.conj(S.O @ vec)) @ S.O))
            + S.sr.diag_shift * vec
        )

    @jax.jit
    def solve(self, y: PyTree, x0: Optional[PyTree] = None) -> PyTree:
        """
        Solve the linear system x=⟨S⟩⁻¹⟨y⟩ with the chosen iterataive solver.

        Args:
            y: the vector y in the system above.
            x0: optional initial guess for the solution.

        Returns:
            x: the PyTree solving the system.
            info: optional additional informations provided by the solver. Might be
                None if there are no additional informations provided.
        """

        # Ravel input PyTrees, record unravelling function too
        grad, unravel = nkjax.tree_ravel(grad)
        
        if x0 is None:
            x0 = self.x0
        if x0 is not None:
            x0, _ = nkjax.tree_ravel(x0)
            if self.scale is not None:
                x0 = x0 / self.scale

        if self.scale is not None:
            grad = grad * self.scale

        solve_fun = self.sr.solve_fun()
        _mat_vec = lambda x: self @ x
        out, info = solve_fun(_mat_vec, grad, x0=x0)

        if self.scale is not None:
            out = out * self.scale

        return unravel(out), info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        return jnp.transpose(jnp.conj(self.O)) @ self.O + self.sr.diag_shift * jnp.eye(
            self.O.shape[1]
        )



@partial(jax.jit, static_argnums=(0, 4, 5))
def gradients(
    apply_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    params: PyTree,
    samples: jnp.ndarray,
    model_state: Optional[PyTree],
    mode: str,
    rescale_shift: bool,
):
    """Calculates the gradients O_ij by backpropagating every sample separately,
    vectorising the loop using vmap
    If rescale_shift is True, columns of O are rescaled to unit magnitude, and
    scale factor [1/sqrt(S_kk)] returned as a separate vector for
    scale-invariant regularisation as per Becca & Sorella p. 143."""
    # Ravel the parameter PyTree and obtain the unravelling function
    params, unravel = nkjax.tree_ravel(params)

    if jnp.ndim(samples) != 2:
        samples = jnp.reshape(samples, (-1, samples.shape[-1]))
    n_samples = samples.shape[0]

    if mode == "holomorphic":
        # Preapply the model state so that when computing gradient
        # we only get gradient of parameters
        # Also divide through sqrt(n_samples) to normalise S matrix in the end
        def fun(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[0]
                / n_samples ** 0.5
            )

        grads = _grad_vmap_minus_mean(fun, params, samples, True)
    elif mode == "R2R":

        def fun(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[0].real
                / n_samples ** 0.5
            )

        grads = _grad_vmap_minus_mean(fun, params, samples, False)
    elif mode == "R2C":

        def fun1(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[0].real
                / n_samples ** 0.5
            )

        def fun2(W, σ):
            return (
                apply_fun({"params": unravel(W), **model_state}, σ[jnp.newaxis, :])[0].imag
                / n_samples ** 0.5
            )

        # Stack real and imaginary parts as real matrixes along the "sample"
        # axis to get Re(O†O) directly
        grads = jnp.concatenate(
            (
                _grad_vmap_minus_mean(fun1, params, samples, False),
                _grad_vmap_minus_mean(fun2, params, samples, False),
            ),
            axis=0,
        )
    else:
        raise Exception(
            "Differentation mode must be holomorphic, R2R or R2C, got {}".format(mode)
        )

    if rescale_shift:
        sqrt_Skk = jnp.linalg.norm(grads, axis=0, keepdims=True)
        return grads / sqrt_Skk, 1 / jnp.ravel(sqrt_Skk)
    else:
        return grads


@partial(jax.jit, static_argnums=(0, 3))
def _grad_vmap_minus_mean(
    fun: Callable, params: jnp.ndarray, samples: jnp.ndarray, holomorphic: bool
):
    """Calculates the gradient of a neural network for a number of samples
    efficiently using vmap(grad),
    subtracts their mean for each parameter, i.e., each column,
    and divides through with sqrt(#samples) to normalise the S matrix"""
    grads = jax.vmap(
        jax.grad(fun, holomorphic=holomorphic), in_axes=(None, 0), out_axes=0
    )(params, samples)
    return grads - jnp.sum(grads, axis=0, keepdims=True) / grads.shape[0]
