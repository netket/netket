from functools import partial
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import n_nodes
from mpi4py import MPI

import jax
from jax.scipy.sparse.linalg import cg
from jax.tree_util import tree_flatten
from netket.vmc_common import jax_shape_for_update
import jax.numpy as jnp


class JaxSR:
    r"""
    Performs stochastic reconfiguration (SR) updates.
    """

    def __init__(
        self,
        lsq_solver=None,
        diag_shift=0.01,
        use_iterative=True,
        is_holomorphic=None,
        svd_threshold=None,
        sparse_tol=None,
        sparse_maxiter=None,
    ):

        self._lsq_solver = lsq_solver
        self._diag_shift = diag_shift
        self._use_iterative = use_iterative
        self._is_holomorphic = is_holomorphic

        # Quantities for sparse solver
        self.sparse_tol = 1.0e-5 if sparse_tol is None else sparse_tol
        self.sparse_maxiter = sparse_maxiter
        self._lsq_solver = lsq_solver
        self._x0 = None
        self._init_solver()

        # self._comm = MPI.COMM_WORLD
        if n_nodes > 1:
            raise NotImplementedError("JaxSR currently works only for serial CPU jobs")

        if not self._use_iterative:
            raise NotImplementedError("JaxSR supports only iterative solvers")

    def _init_solver(self):
        lsq_solver = self._lsq_solver

        if lsq_solver in ["gmres", "cg", "minres", "jaxcg"]:
            self._use_iterative = True
        if lsq_solver in ["ColPivHouseholder", "QR", "SVD", "Cholesky"]:
            self._use_iterative = False

        if self._use_iterative:
            if lsq_solver is None or lsq_solver == "cg":
                self._lsq_solver = "cg"
            elif lsq_solver in ["gmres", "minres"]:
                raise Warning(
                    "Conjugate gradient is the only sparse solver currently implemented in Jax. Defaulting to cg"
                )
                self._lsq_solver = "cg"
            else:
                raise RuntimeError("Unknown sparse lsq_solver " + lsq_solver + ".")

    @partial(jax.jit, static_argnums=(0,))
    def compute_update(self, oks, grad, out=None):
        r"""
        Solves the SR flow equation for the parameter update ẋ.

        The SR update is computed by solving the linear equation
           Sẋ = f
        where S is the covariance matrix of the partial derivatives
        O_i(v_j) = ∂/∂x_i log Ψ(v_j) and f is a generalized force (the loss
        gradient).

        Args:
            oks: A pytree of the jacobians ∂/∂x_i log Ψ(v_j)
            grad: A pytree of the forces f
            out: A pytree of the parameter updates
        """

        grad, oks = self.shape_for_sr(grad, oks)

        oks -= jnp.mean(oks, axis=0)

        if self.is_holomorphic is None or self.machine is None:
            raise ValueError("This SR object is not properly initialized.")

        # n_samp = _sum_inplace(jnp.atleast_1d(oks.shape[0]))
        n_samp = oks.shape[0]

        n_par = grad.shape[0]

        if out is None:
            out = jnp.zeros(n_par, dtype=jnp.complex128)

        if self._is_holomorphic:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = self._jax_cg_solve(oks, grad, n_samp)
                self._x0 = out
        else:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = self._jax_cg_solve(oks, grad.real, n_samp)
                self._x0 = jnp.real(out)

            out = out.real

        out = jax_shape_for_update(out, self.machine.parameters)

        # self._comm.bcast(out, root=0)
        # self._comm.barrier()
        return out

    @partial(jax.jit, static_argnums=(0, 3))
    def _jax_cg_solve(self, oks, grad, n_samp):
        """
        Solves the SR flow equation using the conjugate gradient method
        """

        n_par = grad.shape[0]
        if self._x0 is None:
            self._x0 = jnp.zeros(n_par, dtype=jnp.complex128)

        def mat_vec(x):
            y = jnp.matmul(oks, x) / n_samp
            y = jnp.matmul(oks.conjugate().transpose(), y)
            y = x * self._diag_shift + y

            return y

        out, _ = cg(
            mat_vec, grad, x0=self._x0, tol=self.sparse_tol, maxiter=self.sparse_maxiter
        )

        return out

    @staticmethod
    @jax.jit
    def shape_for_sr(grads, jac):
        r"""Reshapes grads and jax from tree like structures to arrays if jax_available

        Args:
            grads,jac: pytrees of jax arrays or numpy array

        Returns:
            A 1D array of gradients and a 2D array of the jacobian
        """

        grads = jnp.concatenate(tuple(fd.reshape(-1) for fd in tree_flatten(grads)[0]))
        jac = jnp.concatenate(
            tuple(fd.reshape(len(fd), -1) for fd in tree_flatten(jac)[0]), -1
        )

        return grads, jac

    def __repr__(self):
        rep = "SR(solver="

        if self._use_iterative:
            rep += "iterative"

        rep += ", is_holomorphic=" + str(self._is_holomorphic) + ")"
        return rep

    def info(self, depth=0):
        indent = " " * 4 * depth
        rep = indent
        rep += "Stochastic reconfiguration method for "
        rep += "holomorphic" if self._is_holomorphic else "real-parameter"
        rep += " wavefunctions\n"

        rep += indent + "Solver: "

        if self._use_iterative:
            rep += "iterative (Conjugate Gradient)"

        rep += "\n"

        return rep

    @property
    def is_holomorphic(self):
        return self._is_holomorphic

    @is_holomorphic.setter
    def is_holomorphic(self, is_holo):
        self._is_holomorphic = is_holo
        self._init_solver()
