import numpy as _np
from scipy.linalg import lstsq as _lstsq
from netket.stats import sum_on_nodes as _sum_on_nodes


class SR():
    r"""
    Performs stochastic reconfiguration (SR) updates.
    """

    def __init__(self, lsq_solver="QR", diag_shift=0.01,
                 use_iterative=False, is_holomorphic=True,
                 svd_threshold=None):
        self._lsq_solver = lsq_solver
        self._diag_shift = diag_shift
        self._use_iterative = use_iterative
        self._is_holomorphic = is_holomorphic
        self._svd_threshold = svd_threshold
        self._scale_invariant_pc = False
        self._S = None

        if 'SVD' in lsq_solver:
            self._lapack_driver = None
        elif 'ColPivHouseholder' in lsq_solver or 'QR' in lsq_solver:
            self._lapack_driver = 'gelsy'
        else:
            self._lapack_driver = None
            raise RuntimeError('Unknown lqs_solver')

        self._last_rank = None

    def compute_update(self, oks, grad, out):
        r"""
        Solves the SR flow equation for the parameter update ẋ.

        The SR update is computed by solving the linear equation
           Sẋ = f
        where S is the covariance matrix of the partial derivatives
        O_i(v_j) = ∂/∂x_i log Ψ(v_j) and f is a generalized force (the loss
        gradient).

        Args:
            oks: The matrix 𝕆 of centered log-derivatives,
               𝕆_ij = O_i(v_j) - ⟨O_i⟩.
            grad: The vector of forces f.
            out: Output array for the update ẋ.
        """

        n_samp = _sum_on_nodes(_np.atleast_1d(oks.shape[0]))

        n_par = grad.shape[0]

        if out is None:
            out = _np.empty(n_par, dtype=_np.complex128)

        if self._is_holomorphic:
            if self._use_iterative:
                raise NotImplementedError
            else:
                self._S = _np.matmul(oks.conj().T, oks, self._S)
                self._S = _sum_on_nodes(self._S)
                self._S /= float(n_samp)

                self._apply_preconditioning(grad)

                out, residuals, self._last_rank, s_vals = _lstsq(
                    self._S, grad, cond=self._svd_threshold,
                    lapack_driver=self._lapack_driver)

                self._revert_preconditioning(out)

        else:
            if self._use_iterative_:
                raise NotImplementedError
            else:
                self._S = _np.matmul(oks.conj().T, oks, self._S)
                self._S /= float(n_samp)

                self._apply_preconditioning(grad)

                out.real, residuals, self._last_rank, s_vals = _lstsq(self._S.real, grad.real,
                                                                      cond=self._svd_threshold,
                                                                      lapack_driver=self._lapack_driver)

                self._revert_preconditioning(out.real)

            out.imag.fill(0.)
        return out

    def _apply_preconditioning(self, grad):
        if self._scale_invariant_pc:
            # Even if S is complex, its diagonal elements should be real since it
            # is Hermitian.
            self._diag_S = _np.sqrt(self._S.diagonal().real)

            cutoff = 1.0e-10

            index = self._diag_S <= cutoff
            self._diag_S[index] = 1.0
            self._S[index, :].fill(0.)
            self._S[:, index].fill(0.)
            _np.fill_diagonal(self._S, 1.0)

            self._S /= _np.vdot(self._diag_S, self._diag_S)

            grad /= self._diag_S

        # Apply diagonal shift
        self._S += self._diag_shift * _np.eye(self._S.shape[0])

    @property
    def last_rank(self):
        return self._last_rank

    @property
    def last_covariance_matrix(self):
        return self._S

    def _revert_preconditioning(self, out):
        if self._scale_invariant_pc:
            out /= self._diag_S

    @property
    def scale_invariant_regularization(self):
        r"""bool: Whether to use the scale-invariant regularization as described by
                    Becca and Sorella (2017), pp. 143-144.
                    https://doi.org/10.1017/9781316417041
        """
        return self._scale_invariant_pc

    @scale_invariant_regularization.setter
    def scale_invariant_regularization(self, activate):
        assert(activate is True or activate is False)
        self._scale_invariant_pc = activate
        if(self._use_iterative):
            raise NotImplementedError("""Scale-invariant regularization is
                   not implemented for iterative solvers at the moment.""")

    def __repr__(self):
        rep = "SR(solver="

        if self._use_iterative:
            rep += "iterative"
        else:
            rep += self._lsq_solver + ", diag_shift=" + \
                str(self._diag_shift)
            if self._svd_threshold is not None:
                rep += ", threshold=" << self._svd_threshold

        rep += ", is_holomorphic=" + str(self._is_holomorphic) + ")"
        return rep

    def info(self, depth=0):
        indent = ' ' * 4 * depth
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
