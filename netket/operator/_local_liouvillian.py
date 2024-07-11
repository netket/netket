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


import numpy as np
import numba
from numba import jit
from numba.typed import List

from scipy.sparse.linalg import LinearOperator

import netket.jax as nkjax
from netket.jax import canonicalize_dtypes
from netket.utils.optional_deps import import_optional_dependency
from netket.utils.types import DType

from ._discrete_operator import DiscreteOperator
from ._local_operator import LocalOperator
from ._abstract_super_operator import AbstractSuperOperator


class LocalLiouvillian(AbstractSuperOperator):
    """
    LocalLiouvillian super-operator, acting on the DoubledHilbert (tensor product) space
    ℋ⊗ℋ.

    Internally it uses :ref:`netket.operator.LocalOperator` everywhere.


    The Liouvillian is defined according to the definition:

    .. math ::

        \\mathcal{L} = -i \\left[ \\hat{H}, \\hat{\\rho}\\right] + \\sum_i \\left[ \\hat{L}_i\\hat{\\rho}\\hat{L}_i^\\dagger -
            \\left\\{ \\hat{L}_i^\\dagger\\hat{L}_i, \\hat{\\rho} \\right\\} \\right]

    which generates the dynamics according to the equation

    .. math ::

        \\frac{d\\hat{\\rho}}{dt} = \\mathcal{L}\\hat{\\rho}

    Internally, it stores the non-hermitian hamiltonian

    .. math ::

        \\hat{H}_{nh} = \\hat{H} - \\sum_i \\frac{i}{2}\\hat{L}_i^\\dagger\\hat{L}_i

    That is then composed with the jump operators in the inner kernel with the formula:

    .. math ::

        \\mathcal{L} = -i \\hat{H}_{nh}\\hat{\\rho} +i\\hat{\\rho}\\hat{H}_{nh}^\\dagger + \\sum_i \\hat{L}_i\\hat{\\rho}\\hat{L}_i^\\dagger

    """

    def __init__(
        self,
        ham: DiscreteOperator,
        jump_ops: list[DiscreteOperator] = [],
        dtype: DType | None = None,
    ):
        super().__init__(ham.hilbert)

        dtype = canonicalize_dtypes(complex, ham, *jump_ops, dtype=dtype)

        if not nkjax.is_complex_dtype(dtype):
            raise TypeError(f"A complex dtype is required (dtype={dtype} specified).")

        self._H = ham
        self._jump_ops = [op.copy(dtype=dtype) for op in jump_ops]  # to accept dicts
        self._Hnh = ham
        self._max_dissipator_conn_size = 0
        self._max_conn_size = 0

        self._dtype = dtype

        self._compute_hnh()

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return False

    @property
    def hamiltonian(self) -> LocalOperator:
        """The hamiltonian of this Liouvillian"""
        return self._H

    @property
    def hamiltonian_nh(self) -> LocalOperator:
        """The non hermitian Local Operator part of the Liouvillian"""
        return self._Hnh

    @property
    def jump_operators(self) -> list[LocalOperator]:
        """The list of local operators in this Liouvillian"""
        return self._jump_ops

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self._max_conn_size

    def _compute_hnh(self):
        # There is no i here because it's inserted in the kernel
        Hnh = np.asarray(1.0, dtype=self.dtype) * self.hamiltonian
        self._max_dissipator_conn_size = 0
        for L in self._jump_ops:
            Hnh = (
                Hnh - np.asarray(0.5j, dtype=self.dtype) * L.conjugate().transpose() @ L
            )
            self._max_dissipator_conn_size += L.max_conn_size**2

        self._Hnh = Hnh.collect().copy(dtype=self.dtype)

        max_conn_size = self._max_dissipator_conn_size + 2 * Hnh.max_conn_size
        self._max_conn_size = max_conn_size
        self._xprime = np.empty((max_conn_size, self.hilbert.size))
        self._xr_prime = np.empty((max_conn_size, self.hilbert.physical.size))
        self._xc_prime = np.empty((max_conn_size, self.hilbert.physical.size))
        self._xrv = self._xprime[:, 0 : self.hilbert.physical.size]
        self._xcv = self._xprime[
            :, self.hilbert.physical.size : 2 * self.hilbert.physical.size
        ]
        self._mels = np.empty(max_conn_size, dtype=self.dtype)

        self._xprime_f = np.empty((max_conn_size, self.hilbert.size))
        self._mels_f = np.empty(max_conn_size, dtype=self.dtype)

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

        xrp, mel_r = self._Hnh.get_conn(xr)
        self._xrv[i : i + len(mel_r), :] = xrp
        self._xcv[i : i + len(mel_r), :] = xc
        self._mels[i : i + len(mel_r)] = -1j * mel_r
        i = i + len(mel_r)

        xcp, mel_c = self._Hnh.get_conn(xc)
        self._xrv[i : i + len(mel_c), :] = xr
        self._xcv[i : i + len(mel_c), :] = xcp
        self._mels[i : i + len(mel_r)] = 1j * np.conj(mel_c)
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
                self._mels[i : i + nc] = L_mel_r[r] * np.conj(L_mel_c)
                i = i + nc

        return np.copy(self._xprime[0:i, :]), np.copy(self._mels[0:i])

    # pad option pads all sections to have the same (biggest) size.
    # to avoid using the biggest possible size, we dynamically check what is
    # the biggest size for the current size...
    # TODO: investigate if this is worthless
    def get_conn_flattened(self, x, sections, pad=False):
        batch_size = x.shape[0]
        N = x.shape[1] // 2
        n_jops = len(self.jump_operators)
        assert sections.shape[0] == batch_size

        # Separate row and column inputs
        xr, xc = x[:, 0:N], x[:, N : 2 * N]

        # Compute all flattened connections of each term
        sections_r = np.empty(batch_size, dtype=np.int64)
        sections_c = np.empty(batch_size, dtype=np.int64)
        xr_prime, mels_r = self._Hnh.get_conn_flattened(xr, sections_r)
        xc_prime, mels_c = self._Hnh.get_conn_flattened(xc, sections_c)

        if pad:
            # if else to accommodate for batches of 1 element, because
            # sections don't start from 0-index...
            # TODO: if we ever switch to 0-indexing, remove this.
            if batch_size > 1:
                max_conns_r = np.max(np.diff(sections_r))
                max_conns_c = np.max(np.diff(sections_c))
            else:
                max_conns_r = sections_r[0]
                max_conns_c = sections_c[0]
            max_conns_Lrc = 0

        #  Must type those lists otherwise, if they are empty, numba
        # cannot infer their type
        L_xrps = List.empty_list(numba.typeof(x.dtype)[:, :])
        L_xcps = List.empty_list(numba.typeof(x.dtype)[:, :])
        L_mel_rs = List.empty_list(numba.typeof(self.dtype)[:])
        L_mel_cs = List.empty_list(numba.typeof(self.dtype)[:])

        sections_Lr = np.empty(batch_size * n_jops, dtype=np.int32)
        sections_Lc = np.empty(batch_size * n_jops, dtype=np.int32)
        for i, L in enumerate(self._jump_ops):
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

            if pad:
                if batch_size > 1:
                    max_lr = np.max(
                        np.diff(sections_Lr[i * batch_size : (i + 1) * batch_size])
                    )
                    max_lc = np.max(
                        np.diff(sections_Lc[i * batch_size : (i + 1) * batch_size])
                    )
                else:
                    max_lr = sections_Lr[i * batch_size]
                    max_lc = sections_Lc[i * batch_size]

                max_conns_Lrc += max_lr * max_lc

        # compose everything again
        if self._xprime_f.dtype != x.dtype:
            self._xprime_f = np.empty(self._xprime_f.shape, dtype=x.dtype)
        if self._xprime_f.shape[0] < self._max_conn_size * batch_size:
            # refcheck=False because otherwise this errors when testing
            self._xprime_f.resize(
                self._max_conn_size * batch_size, self.hilbert.size, refcheck=False
            )
            self._mels_f.resize(self._max_conn_size * batch_size, refcheck=False)

        if pad:
            pad = max_conns_Lrc + max_conns_r + max_conns_c
        else:
            pad = 0

        self._xprime_f[:] = 0
        self._mels_f[:] = 0

        return self._get_conn_flattened_kernel(
            self._xprime_f,
            self._mels_f,
            sections,
            np.asarray(xr),
            np.asarray(xc),
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
            pad,
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
        pad,
    ):
        off = 0

        n_hr_i = 0
        n_hc_i = 0

        n_Lr_is = np.zeros(n_jops, dtype=np.int32)
        n_Lc_is = np.zeros(n_jops, dtype=np.int32)

        for i in range(batch_size):
            off_i = off
            n_hr_f = sections_r[i]
            n_hr = n_hr_f - n_hr_i
            xs[off : off + n_hr, 0:N] = xr_prime[n_hr_i:n_hr_f, :]
            xs[off : off + n_hr, N : 2 * N] = xc[i, :]
            mels[off : off + n_hr] = -1j * mels_r[n_hr_i:n_hr_f]
            off += n_hr
            n_hr_i = n_hr_f

            n_hc_f = sections_c[i]
            n_hc = n_hc_f - n_hc_i
            xs[off : off + n_hc, N : 2 * N] = xc_prime[n_hc_i:n_hc_f, :]
            xs[off : off + n_hc, 0:N] = xr[i, :]
            mels[off : off + n_hc] = 1j * np.conj(mels_c[n_hc_i:n_hc_f])
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
                    mels[off : off + n_Lc] = L_mel_r[n_Lr_i + r] * np.conj(
                        L_mel_c[n_Lc_i:n_Lc_f]
                    )
                    off = off + n_Lc

                n_Lr_is[j] = n_Lr_f
                n_Lc_is[j] = n_Lc_f

            if pad != 0:
                n_entries = off - off_i
                mels[off : off + n_entries] = 0
                off = (i + 1) * pad

            sections[i] = off

        return np.copy(xs[0:off, :]), np.copy(mels[0:off])

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

        iHnh = -1j * self.hamiltonian_nh
        if sparse:
            iHnh = iHnh.to_sparse()
            J_ops = [j.to_sparse() for j in self.jump_operators]
            J_ops_c = [
                j.conjugate().transpose().to_sparse() for j in self.jump_operators
            ]
        else:
            iHnh = iHnh.to_dense()
            J_ops = [j.to_dense() for j in self.jump_operators]
            J_ops_c = [
                j.conjugate().transpose().to_dense() for j in self.jump_operators
            ]

        if not append_trace:
            op_size = M**2

            def matvec(rho_vec):
                rho = rho_vec.reshape((M, M))

                drho = np.zeros((M, M), dtype=rho.dtype)

                drho += iHnh @ rho + rho @ iHnh.conj().T
                for J, J_c in zip(J_ops, J_ops_c):
                    drho += (J @ rho) @ J_c

                return drho.reshape(-1)

        else:
            # This function defines the product Liouvillian x density matrix, without
            # constructing the full density matrix (passed as a vector M^2).

            # An extra row is added at the bottom of the therefore M^2+1 long array,
            # with the trace of the density matrix. This is needed to enforce the
            # trace-1 condition.

            # The logic behind the use of Hnh_dag_ and Hnh_ is derived from the
            # convention adopted in local_liouvillian.cc, and inspired from reference
            # arXiv:1504.05266
            op_size = M**2 + 1

            def matvec(rho_vec):
                rho = rho_vec[:-1].reshape((M, M))

                out = np.zeros((M**2 + 1), dtype=rho.dtype)
                drho = out[:-1].reshape((M, M))

                drho += iHnh @ rho + rho @ iHnh.conj().T
                for J, J_c in zip(J_ops, J_ops_c):
                    drho += (J @ rho) @ J_c

                out[-1] = rho.trace()
                return out

        L = LinearOperator((op_size, op_size), matvec=matvec, dtype=iHnh.dtype)

        return L

    def to_qobj(self):  # -> "qutip.liouvillian"
        r"""Convert the operator to a qutip's liouvillian Qobj.

        Returns:
            A :class:`qutip.liouvillian` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        return qutip.liouvillian(
            self.hamiltonian.to_qobj(), [op.to_qobj() for op in self.jump_operators]
        )
