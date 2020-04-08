# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

from .abstract_machine import AbstractMachine
import numpy as _np

from numba import jit, optional, jitclass, int64, float64, complex128, deferred_type, typeof

__all__ = ["PyRbm"]


@jit(fastmath=True)
def _log_cosh_sum(x, out):
    x = x * _np.sign(x.real)
    for i in range(x.shape[0]):
        out[i] = _np.sum(x[i] - _np.log(2.) +
                         _np.log(1. + _np.exp(-2. * x[i])))
    return out


class PyRbm(AbstractMachine):
    r"""
    A fully connected Restricted Boltzmann Machine (RBM). This type of
    RBM has spin 1/2 hidden units and is defined by:

    .. math:: \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M \cosh
     \left(\sum_i^N W_{ij} s_i + b_j \right)

    for arbitrary local quantum numbers :math:`s_i`.
    """

    def __init__(
        self, hilbert, n_hidden=None, alpha=None,
        use_visible_bias=True, use_hidden_bias=True,
        dtype=complex
    ):
        r"""
        Constructs a new ``RbmSpin`` machine:

        Args:
           hilbert: Hilbert space object for the system.
           n_hidden: The number of hidden spin units. If n_hidden=None, the number
                   of hidden units is deduced from the parameter alpha.
           alpha: `alpha * hilbert.size` is the number of hidden spins.
                   If alpha=None, the number of hidden units must
                   be explicitely set in the argument n_hidden.
           use_visible_bias: If ``True`` then there would be a
                            bias on the visible units.
                            Default ``True``.
           use_hidden_bias: If ``True`` then there would be a
                           bias on the visible units.
                           Default ``True``.
           dtype: either complex or float, is the type used for the weights.

        Examples:
           A ``RbmSpin`` machine with hidden unit density
           alpha = 2 for a one-dimensional L=20 spin-half system:

           >>> from netket.machine import RbmSpin
           >>> from netket.hilbert import Spin
           >>> from netket.graph import Hypercube
           >>> g = Hypercube(length=20, n_dim=1)
           >>> hi = Spin(s=0.5, total_sz=0, graph=g)
           >>> ma = RbmSpin(hilbert=hi,alpha=2)
           >>> print(ma.n_par)
           860
        """

        n = hilbert.size
        if alpha < 0:
            raise ValueError("`alpha` should be non-negative")

        if(dtype is not float and dtype is not complex):
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        if alpha is None:
            m = n_hidden
        else:
            m = int(round(alpha * n))
            if n_hidden is not None:
                if n_hidden != m:
                    raise RuntimeError('''n_hidden is inconsistent with the given alpha.
                                       Remove one of the two or provide consistent values.''')

        self._w = _np.empty((m, n), dtype=self._npdtype)
        self._a = _np.empty(
            n, dtype=self._npdtype) if use_visible_bias else None
        self._b = _np.empty(
            m, dtype=self._npdtype) if use_hidden_bias else None
        self._r = _np.empty((1, m), dtype=self._npdtype)

        self.n_hidden = m
        self.n_visible = n

        self._npar = (
            self._w.size
            + (self._a.size if self._a is not None else 0)
            + (self._b.size if self._b is not None else 0)
        )

        super().__init__(hilbert)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._npar

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function given a spin
        configuration ``x``.
        """
        x = x.astype(dtype=self._npdtype)

        return self._log_val_kernel(x, out, self._w.T, self._a, self._b, self._r)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, W, a, b, r):
        if(out is None):
            out = _np.empty(x.shape[0], dtype=_np.complex128)
        r = x.dot(W)
        if b is None:
            _log_cosh_sum(r, out)
        else:
            _log_cosh_sum(r + b, out)

        if a is not None:
            out = out + x.dot(a)

        return out

    def der_log(self, x, out=None):

        if out is None:
            out = _np.empty((x.shape[0], self.n_par), dtype=_np.complex128)

        batch_size = x.shape[0]

        i = 0
        if self._a is not None:
            out[:, i: i + x.shape[1]] = x
            i += self.n_visible

        r = self._r
        r = _np.dot(x, self._w.T)
        if self._b is not None:
            r += self._b
        r = _np.tanh(r)

        if self._b is not None:
            out[:, i: i + self.n_hidden] = r
            i += self.n_hidden

        t = out[:, i: i + self._w.size]
        t.shape = (batch_size, self._w.shape[0], self._w.shape[1])
        _np.einsum("ij,il->ijl", r, x, out=t)

        return out

    def vector_jacobian_prod(self, x, vec, out=None):
        return _np.dot(_np.asmatrix(self.der_log(x)).H, vec, out)

    @property
    def is_holomorphic(self):
        r"""Complex valued RBM is a holomorphic function.
        """
        return self._dtype is complex

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        from collections import OrderedDict

        od = OrderedDict()
        if(self._dtype is complex):
            if self._a is not None:
                od["a"] = self._a.view()

            if self._b is not None:
                od["b"] = self._b.view()

            od["w"] = self._w.view()
        else:
            if self._a is not None:
                self._ac = self._a.astype(_np.complex128)
                self._a = self._ac.real.view()
                od["a"] = self._ac.view()

            if self._b is not None:
                self._bc = self._b.astype(_np.complex128)
                self._b = self._bc.real.view()
                od["b"] = self._bc.view()

            self._wc = self._w.astype(_np.complex128)
            self._w = self._wc.real.view()
            od["w"] = self._wc.view()

        return od
