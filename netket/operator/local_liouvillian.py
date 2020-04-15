from .abstract_operator import AbstractOperator
from ..hilbert import PyDoubledHilbert as DoubledHilbert

import numpy as _np
from numba import jit
import numbers


class LocalLiouvillian(AbstractOperator):
    def __init__(self, ham, jump_ops=[]):
        self._H = ham
        self._jump_ops = [op for op in jump_ops]  # to accept dicts
        self._Hnh = ham
        self._Hnh_dag = ham
        self._hilbert = DoubledHilbert(ham.hilbert)
        self._max_dissipator_conn_size = 0
        self._max_conn_size = 0

        self._compute_hnh()
        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this super-operator."""
        return self._hilbert

    @property
    def size(self):
        return self._size

    @property
    def ham(self):
        return self._H

    @property
    def ham_nh(self):
        return self._Hnh

    @property
    def jump_ops(self):
        return self._jump_ops

    def _compute_hnh(self):
        Hnh = self._H
        max_conn_size = 0
        for L in self._jump_ops:
            Hnh += -0.5j * L.conjugate().transpose() * L
            max_conn_size += L.n_operators * L._max_op_size

        self._max_dissipator_conn_size = max_conn_size ** 2

        self._Hnh = Hnh
        self._Hnh_dag = Hnh.conjugate().transpose()

        max_conn_size = (
            self._max_dissipator_conn_size + Hnh.n_operators * Hnh._max_op_size
        )
        self._max_conn_size = max_conn_size
        self._xprime = _np.empty((max_conn_size, self.hilbert.size))
        self._xr_prime = _np.empty((max_conn_size, self.hilbert.physical.size))
        self._xc_prime = _np.empty((max_conn_size, self.hilbert.physical.size))
        self._xrv = self._xprime[:, 0 : self.hilbert.physical.size]
        self._xcv = self._xprime[
            :, self.hilbert.physical.size : 2 * self.hilbert.physical.size
        ]
        self._mels = _np.empty(max_conn_size, dtype=_np.complex128)

    def add_jump_operator(self, op):
        self._jump_ops.append(op)
        self._compute_hnh()

    def add_jump_operators(self, ops):
        for op in ops:
            self._jump_ops.append(op)

        self._compute_hnh()

    def get_conn(self, x):
        n_sites = x.shape[0] // 2
        xr, xc = x[0:n_sites], x[n_sites : 2 * n_sites]
        i = 0

        batch_size = x.shape[0]

        # sec_hnh_l = np.empty(batch_size, dtype=_np.intp)
        # sec_hnh_r = np.empty(batch_size, dtype=_np.intp)

        xrp, mel_r = self._Hnh_dag.get_conn(xr)
        self._xrv[i : i + len(mel_r), :] = xrp
        self._xcv[i : i + len(mel_r), :] = xc
        self._mels[i : i + len(mel_r)] = mel_r * 1j
        i = i + len(mel_r)

        # xcp, mel_c = self._Hnh_dag.get_conn_flattened(xc, sec_hnh_r)
        xcp, mel_c = self._Hnh_dag.get_conn(xc)
        self._xrv[i : i + len(mel_c), :] = xr
        self._xcv[i : i + len(mel_c), :] = xcp
        self._mels[i : i + len(mel_r)] = mel_c * (-1j)
        i = i + len(mel_c)

        for L in self._jump_ops:
            L_xrp, L_mel_r = L.get_conn(xr)
            L_xcp, L_mel_c = L.get_conn(xc)

            nr = len(L_mel_r)
            nc = len(L_mel_c)
            # start filling batches
            for r in range(nr):
                self._xrv[i : i + nc, :] = L_xrp[r, :]
                self._xcv[i : i + nc, :] = L_xcp
                self._mels[i : i + nc] = _np.conj(L_mel_r[r]) * L_mel_c
                i = i + nc

        return _np.copy(self._xprime[0:i, :]), _np.copy(self._mels[0:i])

    def get_conn_flattened(self, x, sections):
        batch_size = x.shape[0]
        n_sites = x.shape[1]
        assert sections.shape[0] == batch_size

        xs = _np.empty(
            (self._max_conn_size * batch_size, self.hilbert.size), dtype=_np.float64
        )
        mels = _np.empty(self._max_conn_size * batch_size, dtype=_np.complex128)
        sec = 0

        for i in range(batch_size):
            xs_i, mels_i = self.get_conn(x[i, :])
            n_conns = len(mels_i)

            xs[sec : sec + n_conns, :] = xs_i
            mels[sec : sec + n_conns] = mels_i
            sections[i] = sec + n_conns
            sec = sec + n_conns

        xs.resize(sec, n_sites)
        mels.resize(sec)

        return xs, mels
