import numpy as np
from numba import jit

from scipy.sparse.linalg import LinearOperator

from ._abstract_operator import AbstractOperator
from ._local_operator import LocalOperator
from ._abstract_super_operator import AbstractSuperOperator


class ImagTimeGenerator(AbstractSuperOperator):
    """
    Super-operator, generator of the imaginary time evolution for a closed (hamiltonian) system.


    The Super-Operator is defined according to the definition:

    .. math ::

        \\mathcal{L} = - \\left\\{} \\hat{H}, \\hat{\\rho}\\right\\}

    which generates the dynamics according to the equation

    .. math ::

        \\frac{d\\hat{\\rho}}{dt} = \\mathcal{L}\\hat{\\rho}

    """

    def __init__(self, hamiltonian: AbstractOperator, dtype=None):
        super().__init__(hamiltonian.hilbert)

        self._hamiltonian = hamiltonian
        self._max_conn_size = 2 * hamiltonian.max_conn_size

        self._xprime_f = np.empty((self._max_conn_size, self.hilbert.size))
        self._mels_f = np.empty(self._max_conn_size, dtype=self.dtype)

        self._dtype = hamiltonian.dtype if dtype is None else dtype
        self._impl = None

    @property
    def dtype(self):
        return self._hamiltonian.dtype

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self._max_conn_size

    @property
    def is_hermitian(self):
        return self.hamiltonian.is_hermitian

    @property
    def hamiltonian(self) -> LocalOperator:
        """The hamiltonian of this Liouvillian"""
        return self._hamiltonian

    @property
    def implementation(self):
        if self._impl is None:
            from ._imag_time_generator_impl import get_jitted_implementation

            constructor = get_jitted_implementation(
                self.hamiltonian.implementation, self.dtype
            )

            self._impl = constructor(
                self.hamiltonian.implementation,
                self._max_conn_size,
                self._xprime_f,
                self._mels_f,
            )

        return self._impl

    def get_conn_flattened(self, x, sections, pad=False):
        return self.implementation.get_conn_flattened(x, sections, pad)

    def _get_conn_flattened_closure(self):
        O = self.implementation

        def _fun(x, sections, pad=False):
            return O.get_conn_flattened(x, sections, pad)

        return jit(nopython=True)(_fun)

    def to_linear_operator(
        self, *, sparse: bool = True, append_trace: bool = False
    ) -> LinearOperator:
        r"""Returns a lazy scipy linear_operator representation of the Lindblad Super-Operator.

        The returned operator behaves like the M**2 x M**2 matrix obtained with to_dense/sparse, and accepts
        vectorised density matrices as input.

        Args:
            sparse: If True internally uses sparse matrices for the hamiltonian and jump operators,
                dense otherwise (default=True)
            append_trace: If True (default=False) the resulting operator has size M**2 + 1, and the last
                element of the input vector is the trace of the input density matrix. This is useful when
                implementing iterative methods.

        Returns:
            A linear operator taking as input vectorised density matrices and returning the product L*rho

        """
        M = self.hilbert.physical.n_states

        H = self.hamiltonian
        if sparse:
            H = H.to_sparse()
        else:
            H = H.to_dense()

        if not append_trace:
            op_size = M ** 2

            def matvec(rho_vec):
                rho = rho_vec.reshape((M, M))

                drho = np.zeros((M, M), dtype=rho.dtype)

                drho += -(rho @ H + H.conj().T @ rho)

                return drho.reshape(-1)

        else:
            # This function defines the product Liouvillian x densitymatrix, without
            # constructing the full density matrix (passed as a vector M^2).

            # An extra row is added at the bottom of the therefore M^2+1 long array,
            # with the trace of the density matrix. This is needed to enforce the
            # trace-1 condition.

            # The logic behind the use of Hnh_dag_ and Hnh_ is derived from the
            # convention adopted in local_liouvillian.cc, and inspired from reference
            # arXiv:1504.05266
            op_size = M ** 2 + 1

            def matvec(rho_vec):
                rho = rho_vec[:-1].reshape((M, M))

                out = np.zeros((M ** 2 + 1), dtype=rho.dtype)
                drho = out[:-1].reshape((M, M))

                drho += -(rho @ H + H.conj().T @ rho)

                out[-1] = rho.trace()
                return out

        L = LinearOperator((op_size, op_size), matvec=matvec, dtype=H.dtype)

        return L
