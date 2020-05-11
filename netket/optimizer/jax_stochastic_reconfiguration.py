from functools import partial
from netket.stats import sum_inplace as _sum_inplace
from mpi4py import MPI

import jax
from jax.scipy.sparse.linalg import cg
from netket.vmc_common import shape_for_sr, shape_for_update
from jax.scipy.linalg import cho_factor,  cho_solve, svd, qr, solve_triangular

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
        self._svd_threshold = svd_threshold
        self._scale_invariant_pc = False
        self._S = None
        self._last_rank = None

        # Temporary arrays
        self._v_tilde = None
        self._res_t = None

        # Quantities for sparse solver
        self.sparse_tol = 1.0e-5 if sparse_tol is None else sparse_tol
        self.sparse_maxiter = sparse_maxiter
        self._lsq_solver = lsq_solver
        self._x0 = None
        self._init_solver()

        self._comm = MPI.COMM_WORLD

    def _init_solver(self):
        lsq_solver = self._lsq_solver

        if lsq_solver in ["gmres", "cg", "minres","jaxcg"]:
            self._use_iterative = True
        if lsq_solver in ["ColPivHouseholder", "QR", "SVD", "Cholesky"]:
            self._use_iterative = False

        if self._use_iterative:
            if lsq_solver is None or lsq_solver == "cg":
                self._lsq_solver = "cg"
            elif lsq_solver in ["gmres","minres"]:
                raise Warning(
                    "Conjugate gradient is the only sparse solver currently implemented in Jax. Defaulting to cg"
                    )
                self._lsq_solver = "cg"
            else:
                raise RuntimeError("Unknown sparse lsq_solver " + lsq_solver + ".")

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

        grad, oks = shape_for_sr(grad,oks)    
        
        oks -= jax.numpy.mean(oks,axis=0)

        if self.is_holomorphic is None:
            raise ValueError(
                "is_holomorphic not set: this SR object is not properly initialized."
            )

        n_samp = _sum_inplace(jax.numpy.atleast_1d(oks.shape[0]))

        n_par = grad.shape[0]

        if out is None:
            out = jax.numpy.zeros(n_par, dtype=jax.numpy.complex128)

        if self._is_holomorphic:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = self._jax_cg_solve(oks,grad,n_samp)
                self._x0 = out
            else:
                self._S = jax.numpy.matmul(oks.conj().T, oks)
                self._S = _sum_inplace(self._S)
                self._S /= float(n_samp)

                self._apply_preconditioning(grad)
                
                if self._lsq_solver == "Cholesky":
                    c, low = cho_factor(self._S, check_finite=False)
                    out = cho_solve((c, low), grad)
                if self._lsq_solver in ["QR","ColPivHouseholder"]:
                    Q, R = qr(self._S)
                    grad = jax.numpy.matmul(Q.transpose().conjugate(),grad)
                    out = solve_triangular(R,grad)
                if self._lsq_solver == "SVD":
                    U, S, V = svd(self._S)
                    grad = jax.numpy.matmul(U.transpose().conjugate(),grad)/S 
                    out = jax.numpy.matmul(V.transpose().conjugate(),grad)

                self._revert_preconditioning(out)


        else:
            if self._use_iterative:
                if self._lsq_solver == "cg":
                    out = self._jax_cg_solve(oks,grad.real,n_samp)     
                self._x0 = jax.numpy.real(out)  

            else:
                self._S = jax.numpy.matmul(oks.conj().T, oks)
                self._S = _sum_inplace(self._S)
                self._S /= float(n_samp)

                self._apply_preconditioning(grad)

                if self._lsq_solver == "Cholesky":
                    c, low = cho_factor(self._S.real, check_finite=False)
                    out = cho_solve((c, low), grad.real)
                if self._lsq_solver in ["QR","ColPivHouseholder"]:
                    Q, R = qr(self._S.real)
                    grad = jax.numpy.matmul(Q.transpose().conjugate(),grad.real)
                    out = solve_triangular(R,grad)
                if self._lsq_solver == "SVD":
                    U, S, V = svd(self._S.real)
                    grad = jax.numpy.matmul(U.transpose().conjugate(),grad.real)/S 
                    out = jax.numpy.matmul(V.transpose().conjugate(),grad)


                self._revert_preconditioning(out)
 
            out = jax.numpy.real(out)


        out = shape_for_update(out,self.machine.parameters) 

        self._comm.bcast(out, root=0)
        self._comm.barrier()
        return out

    def _jax_cg_solve(self,oks,grad,n_samp):
        """
        Solves the SR flow equation using the conjugate gradient method 
        """

        n_par = grad.shape[0]
        if self._x0 is None:
            self._x0 = jax.numpy.zeros(n_par, dtype=jax.numpy.complex128)

        cov_op = self._jax_linear_function(oks,n_samp)
        
        out, _ = cg(cov_op,grad,x0=self._x0,tol=self.sparse_tol,maxiter=self.sparse_maxiter)

        return out

    def _jax_linear_function(self,oks,n_samp):
        """
        Outputs function A(x) = Ax needed for conjugate gradient
        """
        v_tilde = self._v_tilde
        res = self._res_t 
        shift = self._diag_shift
        oks_conj = oks.conjugate()

        def matvec(oks,oks_conj,x):
            y = jax.numpy.matmul(oks,x)/n_samp
            y = jax.numpy.matmul(y,oks_conj) 
            y = x*shift + y 

            return y

        return partial(matvec,oks,oks_conj)

    def _apply_preconditioning(self, grad):
        if self._scale_invariant_pc:
            # Even if S is complex, its diagonal elements should be real since it
            # is Hermitian.
            self._diag_S = jax.numpy.sqrt(self._S.diagonal().real)

            cutoff = 1.0e-10

            index = self._diag_S <= cutoff
            self._diag_S[index] = 1.0
            self._S[index, :].fill(0.0)
            self._S[:, index].fill(0.0)
            self._S[range(len(self._S)),range(len(self._S))]
            self._S /= jax.numpy.vdot(self._diag_S, self._diag_S)
            grad /= self._diag_S

        # Apply diagonal shift
        self._S += self._diag_shift * jax.numpy.eye(self._S.shape[0])

    @property
    def last_rank(self):
        return self._last_rank

    @property
    def last_covariance_matrix(self):
        return self._S

    def _revert_preconditioning(self, out):
        if self._scale_invariant_pc:
            out /= self._diag_S
        return out

    @property
    def scale_invariant_regularization(self):
        r"""bool: Whether to use the scale-invariant regularization as described by
                    Becca and Sorella (2017), pp. 143-144.
                    https://doi.org/10.1017/9781316417041
        """
        return self._scale_invariant_pc

    @scale_invariant_regularization.setter
    def scale_invariant_regularization(self, activate):
        assert activate is True or activate is False

        if activate and self._use_iterative:
            raise NotImplementedError(
                """Scale-invariant regularization is
                   not implemented for iterative solvers at the moment."""
            )

        self._scale_invariant_pc = activate

    def __repr__(self):
        rep = "SR(solver="

        if self._use_iterative:
            rep += "iterative"
        else:
            rep += self._lsq_solver + ", diag_shift=" + str(self._diag_shift)
            if self._svd_threshold is not None:
                rep += ", threshold=" << self._svd_threshold

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
        else:
            rep += self._lsq_solver
        rep += "\n"

        return rep

    @property
    def is_holomorphic(self):
        return self._is_holomorphic

    @is_holomorphic.setter
    def is_holomorphic(self, is_holo):
        self._is_holomorphic = is_holo
        self._init_solver()
