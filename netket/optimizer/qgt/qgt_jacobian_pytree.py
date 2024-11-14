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


import jax
from jax import numpy as jnp
from flax import struct

from netket.utils.types import Array, PyTree, Scalar
from netket.utils import mpi
from netket import jax as nkjax

from ..linear_operator import LinearOperator, Uninitialized

from .common import check_valid_vector_type


@struct.dataclass
class QGTJacobianPyTreeT(LinearOperator):
    """
    Semi-lazy representation of an S Matrix behaving like a linear operator.

    The matrix of gradients O is computed on initialisation, but not S,
    which can be computed by calling :code:`to_dense`.
    The details on how the ⟨S⟩⁻¹⟨F⟩ system is solved are contained in
    the field `sr`.
    """

    O: PyTree = Uninitialized
    """Centred gradients ΔO_ij = O_ij - <O_j> of the neural network, where
    O_ij = ∂log ψ(σ_i)/∂p_j, for all samples σ_i at given values of the parameters p_j
    Divided through with sqrt(#samples) to normalise S matrix
    If scale is not None, O_ij for is normalised to unit norm for each parameter j
    """

    scale: PyTree | None = None
    """If not None, contains 2-norm of each column of the gradient matrix,
    i.e., the sqrt of the diagonal elements of the S matrix
    """

    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    """Differentiation mode:
        - "real": for real-valued R->R and C->R Ansätze, splits the complex inputs
                  into real and imaginary part.
        - "complex": for complex-valued R->C and C->C Ansätze, splits the complex
                  inputs and outputs into real and imaginary part
        - "imag": For the imaginary part of the QGT. Solve in this case builds the
                  purely imaginary hermitian matrix to solve the linear system.
        - "holomorphic": for any Ansätze. Does not split complex values.
        - "auto": autoselect real or complex.
    """

    _params_structure: PyTree = struct.field(pytree_node=False, default=Uninitialized)
    """Parameters of the network. Its only purpose is to represent its own shape."""

    _in_solve: bool = struct.field(pytree_node=False, default=False)
    """Internal flag used to signal that we are inside the _solve method and matmul should
    not take apart into real and complex parts the other vector"""

    @jax.jit
    def __matmul__(self, vec: PyTree | Array) -> PyTree | Array:
        # Turn vector RHS into PyTree
        if hasattr(vec, "ndim"):
            _, unravel = nkjax.tree_ravel(self._params_structure)
            vec = unravel(vec)
            ravel = True
        else:
            ravel = False

        check_valid_vector_type(self._params_structure, vec)

        # Real-imaginary split RHS in R→R and R→C modes
        reassemble = None
        if self.mode != "holomorphic" and not self._in_solve:
            vec, reassemble = nkjax.tree_to_real(vec)

        if self.scale is not None:
            vec = jax.tree_util.tree_map(jnp.multiply, vec, self.scale)

        result = mat_vec(vec, self.O, self.diag_shift, imag_part=(self.mode == "imag"))

        if self.scale is not None:
            result = jax.tree_util.tree_map(jnp.multiply, result, self.scale)

        # Reassemble real-imaginary split as needed
        if reassemble is not None:
            result = reassemble(result)

        # Ravel PyTree back into vector as needed
        if ravel:
            result, _ = nkjax.tree_ravel(result)

        return result

    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: PyTree | None = None) -> PyTree:
        """
        Solve the linear system x=⟨S⟩⁻¹⟨y⟩ with the chosen iterative solver.

        Args:
            y: the vector y in the system above.
            x0: optional initial guess for the solution.

        Returns:
            x: the PyTree solving the system.
            info: optional additional information provided by the solver. Might be
                None if there are no additional information provided.
        """
        check_valid_vector_type(self._params_structure, y)

        # Real-imaginary split RHS in R→R and R→C modes
        if self.mode != "holomorphic":
            y, reassemble = nkjax.tree_to_real(y)
            if x0 is not None:
                x0, _ = nkjax.tree_to_real(x0)

        if self.scale is not None:
            y = jax.tree_util.tree_map(jnp.divide, y, self.scale)
            if x0 is not None:
                x0 = jax.tree_util.tree_map(jnp.multiply, x0, self.scale)

        # to pass the object LinearOperator itself down
        # but avoid rescaling, we pass down an object with
        # scale = None
        # mode=holomorphic to disable splitting the complex part
        unscaled_self = self.replace(scale=None, _in_solve=True)

        if self.mode == "imag":
            # If we want to solve the linear system IM(G)x=vec,
            # the G matrix is hermitian so its imaginary part is skew-simmetric
            # and has purely imaginary eigenvalues. This is extemely unstable
            # numerically, so we instead solve the problem
            # i Im(G) x = i Vec, such that the matrix becomes hermitian and the
            # solving algorithm are more stable.
            # Then, we take the real part of the solution.
            #
            # To multiply Im(G) by i, as Im(G) = Oₗ† Oᵣ , I can multiply
            # O by √i
            #
            sqrt_i_O = nkjax.tree_ax(jnp.sqrt(1j), unscaled_self.O)
            y_vec = nkjax.tree_ax(1j, y)
            unscaled_self = unscaled_self.replace(O=sqrt_i_O)
            out, info = solve_fun(unscaled_self, y_vec, x0=x0)
            out = jax.tree_util.tree_map(lambda x: x.real, out)
        else:
            out, info = solve_fun(unscaled_self, y, x0=x0)

        if self.scale is not None:
            out = jax.tree_util.tree_map(jnp.divide, out, self.scale)

        # Reassemble real-imaginary split as needed
        if self.mode != "holomorphic":
            out = reassemble(out)

        return out, info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
            In R→R and R→C modes, real and imaginary parts of parameters get own rows/columns
        """
        O = self.O
        if self.mode == "complex" or self.mode == "imag":
            # I want to iterate across the samples and real/imaginary part
            O = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), O)
        O = jax.vmap(lambda l: nkjax.tree_ravel(l)[0])(O)

        if self.scale is None:
            diag = jnp.eye(O.shape[-1])
        else:
            scale, _ = nkjax.tree_ravel(self.scale)
            O = O * scale[jnp.newaxis, :]
            diag = jnp.diag(scale**2)

        # concatenate samples with real/Imaginary dimension
        if self.mode == "imag":
            O = O.reshape(O.shape[0] // 2, 2, -1)

            flip_sign = jnp.array([1, -1]).reshape(1, 2, 1)
            Ol = (flip_sign * O).reshape(-1, O.shape[-1])
            Or = jnp.flip(O, axis=1).reshape(-1, O.shape[-1])
            return mpi.mpi_sum_jax(Ol.T @ Or)[0] + self.diag_shift * diag
        else:
            return mpi.mpi_sum_jax(O.T.conj() @ O)[0] + self.diag_shift * diag

    def to_real_part(self) -> "QGTJacobianPyTreeT":
        """
        Returns the operator computing real part of the complex, non holomorphic QGT.

        The real part of the QGT is used in the Stochastic Reconfiguration (SR)
        algorithm as well as in the McLachlan Variational Principles used to simulate
        Quantum Dynamics.
        See Table 1 in `"Theory of Variational Quantum Simulation" by McLachlan et al.
        <https://arxiv.org/pdf/1812.08767>`_ for more details.

        .. note::

            This function can only be called on the non-holomorphic QGT.

        .. note::

            The returned object is another :code:`QGTJacobian***` object.

            NetKet does not currently implement the *full* complex QGT object
            for non-holomorphic functions, instead you need to keep on hand
            the two separate terms obtained by calling
            :meth:`~netket.optimizer.qgt.QGTJacobianDenseT.to_real_part` and
            :meth:`~netket.optimizer.qgt.QGTJacobianDenseT.to_imag_part`.

        See also :meth:`~netket.optimizer.qgt.QGTJacobianDenseT.to_imag_part`.
        """
        if self.mode == "imag":
            return self.replace(mode="complex")
        elif self.mode == "complex":
            return self
        elif self.mode == "real":
            return self
        else:
            raise ValueError(
                "Can only convert to real part the imaginary part of the"
                "QGT, not the holomorphic or sign-less one."
            )

    def to_imag_part(self) -> "QGTJacobianPyTreeT":
        """
        Returns the operator computing imaginary part of the complex, non holomorphic QGT.

        The imaginary part of the QGT is necessary to implement the TDVP variational
        principle for the quantum dynamics.
        See Table 1 in `"Theory of Variational Quantum Simulation" by Yuan et al.
        <https://arxiv.org/pdf/1812.08767>`_ for more details.

        .. note::

            This function can only be called on the non-holomorphic QGTs
            of a complex-valued wave-function.

        .. note::

            The returned object is another :code:`QGTJacobian***` object.

            NetKet does not currently implement the *full* complex QGT object
            for non-holomorphic functions, instead you need to keep on hand
            the two separate terms obtained by calling
            :meth:`~netket.optimizer.qgt.QGTJacobianDenseT.to_real_part` and
            :meth:`~netket.optimizer.qgt.QGTJacobianDenseT.to_imag_part`.

        See also :meth:`~netket.optimizer.qgt.QGTJacobianDenseT.to_real_part`.
        """

        if self.mode == "complex":
            return self.replace(mode="imag")
        elif self.mode == "imag":
            return self
        else:
            raise ValueError(
                "Can only convert to imaginary part the real part of the"
                "QGT, not the holomorphic or sign-less one."
            )

    def __repr__(self):
        return (
            f"QGTJacobianPyTree(diag_shift={self.diag_shift}, "
            f"scale={self.scale}, mode={self.mode})"
        )


#################################################
#####           QGT internal Logic          #####
#################################################


def _jvp(oks: PyTree, v: PyTree) -> Array:
    """
    Compute the matrix-vector product between the pytree jacobian oks and the pytree vector v
    """
    td = lambda x, y: jnp.tensordot(x, y, axes=y.ndim)
    t = jax.tree_util.tree_map(td, oks, v)
    return jax.tree_util.tree_reduce(jnp.add, t)


def _vjp(oks: PyTree, w: Array) -> PyTree:
    """
    Compute the vector-matrix product between the vector w and the pytree jacobian oks
    """
    res = jax.tree_util.tree_map(lambda x: jnp.tensordot(w, x, axes=w.ndim), oks)
    return jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)  # MPI


def _mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    """
    Compute ⟨O† O⟩v = ∑ₗ ⟨Oₖᴴ Oₗ⟩ vₗ
    """
    # w = _jvp(oks, v).conjugate()
    # r = _vjp(oks, w)
    # res = nkjax.tree_conj(r)
    # res = nkjax.tree_cast(res, v)

    res = nkjax.tree_conj(_vjp(oks, _jvp(oks, v).conjugate()))
    return nkjax.tree_cast(res, v)


def _imag_mat_vec(v: PyTree, oks: PyTree) -> PyTree:
    # jvp
    oks_r = jax.tree_util.tree_map(lambda o: jnp.flip(o, axis=1), oks)
    w = _jvp(oks_r, v).conjugate()

    # prepare
    flip_sign = jnp.array([1, -1]).reshape(1, 2, 1)
    oks_l = jax.tree_util.tree_map(
        lambda o: o
        * flip_sign.reshape(tuple(2 if i == 1 else 1 for i in range(o.ndim))),
        oks,
    )

    # vjp
    res = nkjax.tree_conj(_vjp(oks_l, w))
    return nkjax.tree_cast(res, v)


def mat_vec(
    v: PyTree, centered_oks: PyTree, diag_shift: Scalar, imag_part: bool = False
) -> PyTree:
    """
    Compute (S + δ) v = 1/n ⟨ΔO† ΔO⟩v + δ v = ∑ₗ 1/n ⟨ΔOₖᴴΔOₗ⟩ vₗ + δ vₗ

    Only compatible with R→R, R→C, and holomorphic C→C
    for C→R, R&C→R, R&C→C and general C→C the parameters for generating ΔOⱼₖ should be converted to R,
    and thus also the v passed to this function as well as the output are expected to be of this form

    Args:
        v: pytree representing the vector v compatible with centered_oks
        centered_oks: pytree of gradients 1/√n ΔOⱼₖ
        diag_shift: a scalar diagonal shift δ
        imag_part: A boolean specifying whether we are applying the real or imaginary part
            of the QGT.

    Returns:
        a pytree corresponding to the sr matrix-vector product (S + δ) v
    """

    # Different implementations depending on real or imaginary part of the QGT
    if imag_part:
        S_times_v = _imag_mat_vec(v, centered_oks)
    else:
        S_times_v = _mat_vec(v, centered_oks)
    return nkjax.tree_axpy(diag_shift, v, S_times_v)
