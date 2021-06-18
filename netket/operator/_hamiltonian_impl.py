import numpy as np
import numba
from numba.experimental import jitclass


class IsingImpl:
    def __init__(self, h, J, edges):
        self._h = h
        self._J = J
        self._edges = edges

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            pad (bool): Whether to use zero-valued matrix elements in order to return all equal sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        edges = self._edges
        h = self._h
        J = self._J

        n_sites = x.shape[1]
        n_conn = n_sites + 1

        x_prime = np.empty(
            (
                x.shape[0] * n_conn,
                n_sites,
            )
        )
        mels = np.empty(x.shape[0] * n_conn, dtype=type(h))

        diag_ind = 0

        for i in range(x.shape[0]):
            mels[diag_ind] = 0.0
            for k in range(edges.shape[0]):
                mels[diag_ind] += (
                    J
                    * x[
                        i,
                        edges[
                            k,
                            0,
                        ],
                    ]
                    * x[
                        i,
                        edges[
                            k,
                            1,
                        ],
                    ]
                )

            odiag_ind = 1 + diag_ind

            mels[odiag_ind : (odiag_ind + n_sites)].fill(-h)

            x_prime[diag_ind : (diag_ind + n_conn)] = np.copy(x[i])

            for j in range(n_sites):
                x_prime[j + odiag_ind][j] *= -1.0

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels


_ising_implementations = {}


def jit_ising_implementation(dtype):
    dtype = np.dtype(dtype)
    numba_dtype = numba.typeof(dtype)[:].dtype

    spec = [
        ("_edges", numba.intp[:, :]),
        ("_h", numba_dtype),
        ("_J", numba_dtype),
    ]

    return jitclass(IsingImpl, spec=spec)


def get_ising_jitted_implementation(dtype):
    dtype = np.dtype(dtype)
    if dtype not in _ising_implementations:
        _ising_implementations[dtype] = jit_ising_implementation(dtype)

    return _ising_implementations[dtype]
