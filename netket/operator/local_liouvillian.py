from .abstract_operator import AbstractOperator
from ..hilbert import DoubledHilbert

import numpy as _np
from numba import jit
from numba.typed import List

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
        Hnh = 1.0 * self._H
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

        self._xprime_f = _np.empty((max_conn_size, self.hilbert.size))
        self._mels_f = _np.empty(max_conn_size, dtype=_np.complex128)

    def add_jump_operator(self, op):
        self._jump_ops.append(op)
        self._compute_hnh()

    def add_jump_operators(self, ops):
        for op in ops:
            self._jump_ops.append(op)

        self._compute_hnh()

    def get_conn(self, x):
        n_sites = x.shape[0] // 2
        batch_size = x.shape[0]

        xr, xc = x[0:n_sites], x[n_sites : 2 * n_sites]
        i = 0

        xrp, mel_r = self._Hnh_dag.get_conn(xr)
        self._xrv[i : i + len(mel_r), :] = xrp
        self._xcv[i : i + len(mel_r), :] = xc
        self._mels[i : i + len(mel_r)] = mel_r * 1j
        i = i + len(mel_r)

        xcp, mel_c = self._Hnh.get_conn(xc)
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
        N = x.shape[1] // 2
        n_jops = len(self.jump_ops)
        assert sections.shape[0] == batch_size

        # Separate row and column inputs
        xr, xc = x[:, 0:N], x[:, N : 2 * N]

        # Compute all flattened connections of each term
        sections_r = _np.empty(batch_size, dtype=_np.int64)
        sections_c = _np.empty(batch_size, dtype=_np.int64)
        xr_prime, mels_r = self._Hnh_dag.get_conn_flattened(xr, sections_r)
        xc_prime, mels_c = self._Hnh.get_conn_flattened(xc, sections_c)

        L_xrps = List()
        L_xcps = List()
        L_mel_rs = List()
        L_mel_cs = List()
        sections_Lr = _np.empty(batch_size * n_jops, dtype=_np.int32)
        sections_Lc = _np.empty(batch_size * n_jops, dtype=_np.int32)
        for (i, L) in enumerate(self._jump_ops):
            L_xrp, L_mel_r = L.get_conn_flattened(
                xr, sections_Lr[i * batch_size : (i + 1) * batch_size]
            )
            L_xcp, L_mel_c = L.get_conn_flattened(
                xc, sections_Lc[i * batch_size : (i + 1) * batch_size]
            )

            L_xrps.append(L_xrp)
            L_xcps.append(L_xcp)
            L_mel_rs.append(L_mel_r)
            L_mel_cs.append(L_mel_c)

        # compose everything again
        if self._xprime.shape[0] < self._max_conn_size * batch_size:
            self._xprime_f.resize(self._max_conn_size * batch_size, self.hilbert.size)
            self._mels_f.resize(self._max_conn_size * batch_size)

        return self._get_conn_flattened_kernel(
            self._xprime_f,
            self._mels_f,
            sections,
            xr,
            xc,
            sections_r,
            sections_c,
            xr_prime,
            mels_r,
            xc_prime,
            mels_c,
            L_xrps,
            L_xcps,
            L_mel_rs,
            L_mel_cs,
            sections_Lr,
            sections_Lc,
            n_jops,
            batch_size,
            N,
        )

    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(
        xs,
        mels,
        sections,
        xr,
        xc,
        sections_r,
        sections_c,
        xr_prime,
        mels_r,
        xc_prime,
        mels_c,
        L_xrps,
        L_xcps,
        L_mel_rs,
        L_mel_cs,
        sections_Lr,
        sections_Lc,
        n_jops,
        batch_size,
        N,
    ):
        sec = 0
        off = 0

        n_hr_i = 0
        n_hc_i = 0

        n_Lr_is = _np.zeros(n_jops, dtype=_np.int32)
        n_Lc_is = _np.zeros(n_jops, dtype=_np.int32)

        for i in range(batch_size):
            n_hr_f = sections_r[i]
            n_hr = n_hr_f - n_hr_i
            xs[off : off + n_hr, 0:N] = xr_prime[n_hr_i:n_hr_f, :]
            xs[off : off + n_hr, N : 2 * N] = xc[i, :]
            mels[off : off + n_hr] = 1j * mels_r[n_hr_i:n_hr_f]
            off += n_hr
            n_hr_i = n_hr_f

            n_hc_f = sections_c[i]
            n_hc = n_hc_f - n_hc_i
            xs[off : off + n_hc, N : 2 * N] = xc_prime[n_hc_i:n_hc_f, :]
            xs[off : off + n_hc, 0:N] = xr[i, :]
            mels[off : off + n_hc] = -1j * mels_c[n_hc_i:n_hc_f]
            off += n_hc
            n_hc_i = n_hc_f

            for j in range(n_jops):
                L_xrp, L_mel_r = L_xrps[j], L_mel_rs[j]
                L_xcp, L_mel_c = L_xcps[j], L_mel_cs[j]
                n_Lr_f = sections_Lr[j * batch_size + i]
                n_Lc_f = sections_Lc[j * batch_size + i]
                n_Lr_i = n_Lr_is[j]
                n_Lc_i = n_Lc_is[j]

                n_Lr = n_Lr_f - n_Lr_is[j]
                n_Lc = n_Lc_f - n_Lc_is[j]
                # start filling batches
                for r in range(n_Lr):
                    xs[off : off + n_Lc, 0:N] = L_xrp[n_Lr_i + r, :]
                    xs[off : off + n_Lc, N : 2 * N] = L_xcp[n_Lc_i:n_Lc_f, :]
                    mels[off : off + n_Lc] = (
                        _np.conj(L_mel_r[n_Lr_i + r]) * L_mel_c[n_Lc_i:n_Lc_f]
                    )
                    off = off + n_Lc

                n_Lr_is[j] = n_Lr_f
                n_Lc_is[j] = n_Lc_f

            sections[i] = off

        return _np.copy(xs[0:off, :]), _np.copy(mels[0:off])
