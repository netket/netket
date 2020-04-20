# Copyright 2019-2020 The Simons Foundation, Inc. - All Rights Reserved.
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

from numba import (
    jit,
    float64,
    complex128,
)


@jit(fastmath=True)
def _log_cosh_sum(x, out, add_factor=None):
    x = x * _np.sign(x.real)
    if add_factor is None:
        for i in range(x.shape[0]):
            out[i] = _np.sum(x[i] - _np.log(2.0) + _np.log(1.0 + _np.exp(-2.0 * x[i])))
    else:
        for i in range(x.shape[0]):
            out[i] += add_factor * (
                _np.sum(x[i] - _np.log(2.0) + _np.log(1.0 + _np.exp(-2.0 * x[i])))
            )
    return out


class RbmSpin(AbstractMachine):
    r"""
    A fully connected Restricted Boltzmann Machine (RBM). This type of
    RBM has spin 1/2 hidden units and is defined by:

    .. math:: \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M \cosh
     \left(\sum_i^N W_{ij} s_i + b_j \right)

    for arbitrary local quantum numbers :math:`s_i`.

    The weights can be taken to be complex-valued (default option) or real-valued.
    """

    def __init__(
        self,
        hilbert,
        n_hidden=None,
        alpha=None,
        use_visible_bias=True,
        use_hidden_bias=True,
        dtype=complex,
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

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        if alpha is None:
            m = n_hidden
        else:
            if alpha < 0:
                raise ValueError("`alpha` should be non-negative")
            m = int(round(alpha * n))
            if n_hidden is not None:
                if n_hidden != m:
                    raise RuntimeError(
                        """n_hidden is inconsistent with the given alpha.
                                       Remove one of the two or provide consistent values."""
                    )

        self._w = _np.empty((n, m), dtype=self._npdtype)
        self._a = _np.empty(n, dtype=self._npdtype) if use_visible_bias else None
        self._b = _np.empty(m, dtype=self._npdtype) if use_hidden_bias else None
        self._r = _np.empty((1, m), dtype=self._npdtype)

        self.n_hidden = m

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
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The
                 length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
            """
        x = x.astype(dtype=self._npdtype)

        return self._log_val_kernel(x, out, self._w, self._a, self._b, self._r)

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, W, a, b, r):

        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
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
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.

        Returns:
            `out`
            """

        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.empty((x.shape[0], self.n_par), dtype=_np.complex128)

        batch_size = x.shape[0]
        n_visible = x.shape[1]

        i = 0
        if self._a is not None:
            out[:, i : i + n_visible] = x
            i += n_visible

        r = self._r
        r = _np.dot(x, self._w)
        if self._b is not None:
            r += self._b
        r = _np.tanh(r)

        if self._b is not None:
            out[:, i : i + self.n_hidden] = r
            i += self.n_hidden

        t = out[:, i : i + self._w.size]
        t.shape = (batch_size, self._w.shape[0], self._w.shape[1])
        _np.einsum("ij,il->ijl", x, r, out=t)

        return out

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
        if self._dtype is complex:
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


class RbmSpinReal(RbmSpin):
    r"""
    A fully connected Restricted Boltzmann Machine (RBM) with real-valued parameters.
    See RbmSpin for more details.

    """

    def __init__(
        self,
        hilbert,
        n_hidden=None,
        alpha=None,
        use_visible_bias=True,
        use_hidden_bias=True,
    ):
        r"""
        Constructs a new ``RbmSpinReal`` machine:

        Args:
           hilbert: Hilbert space object for the system.
           n_hidden: Number of hidden units.
           alpha: Hidden unit density.
           use_visible_bias: If ``True`` then there would be a
                            bias on the visible units.
                            Default ``True``.
           use_hidden_bias: If ``True`` then there would be a
                           bias on the visible units.
                           Default ``True``.

        Examples:
           A ``RbmSpinReal`` machine with hidden unit density
           alpha = 2 for a one-dimensional L=20 spin-half system:


           >>> from netket.machine import RbmSpinReal
           >>> from netket.hilbert import Spin
           >>> from netket.graph import Hypercube
           >>> g = Hypercube(length=20, n_dim=1)
           >>> hi = Spin(s=0.5, total_sz=0, graph=g)
           >>> ma = RbmSpinReal(hilbert=hi,alpha=2)
           >>> print(ma.n_par)
           860
        """
        super().__init__(
            hilbert,
            n_hidden=n_hidden,
            alpha=alpha,
            use_visible_bias=use_visible_bias,
            use_hidden_bias=use_hidden_bias,
            dtype=float,
        )


class RbmMultiVal(RbmSpin):
    r"""
        A fully connected Restricted Boltzmann Machine suitable for large local hilbert spaces.
        Local quantum numbers are passed through a one hot encoding that maps them onto
        an enlarged space of +/- 1 spins. In turn, these quantum numbers are used with a
        standard RbmSpin wave function.
    """

    def __init__(
        self,
        hilbert,
        n_hidden=None,
        alpha=None,
        use_visible_bias=True,
        use_hidden_bias=True,
        dtype=complex,
    ):

        r"""
            Constructs a new ``RbmMultiVal`` machine:

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
               A ``RbmMultiVal`` machine with hidden unit density
               alpha = 1 for a one-dimensional bosonic system:

               >>> from netket.machine import RbmMultiVal
               >>> from netket.hilbert import Boson
               >>> from netket.graph import Hypercube
               >>> g = Hypercube(length=10, n_dim=1)
               >>> hi = Boson(graph=g, n_max=3, n_bosons=8)
               >>> ma = RbmMultiVal(hilbert=hi, alpha=1, dtype=float, use_visible_bias=False)
               >>> print(ma.n_par)
               1056
        """
        local_states = _np.asarray(hilbert.local_states, dtype=int)

        # creating a fictious hilbert object to pass to the standard rbm
        l_hilbert = _np.zeros(local_states.size * hilbert.size)

        super().__init__(
            l_hilbert, n_hidden, alpha, use_visible_bias, use_hidden_bias, dtype
        )
        self.hilbert = hilbert
        self._local_states = local_states

    @staticmethod
    @jit
    def _one_hot(vec, local_states):
        if vec.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        # one hotting and converting to -/+ 1
        res = (
            vec.reshape((vec.shape[1], vec.shape[0], -1)) == local_states
        ) * 2.0 - 1.0
        return res.reshape((vec.shape[0], -1))

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The
                 length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
            """
        return super().log_val(self._one_hot(x, self._local_states), out)

    def der_log(self, x, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.

        Returns:
            `out`
            """
        return super().der_log(self._one_hot(x, self._local_states), out)


class RbmSpinPhase(AbstractMachine):
    r"""
    A fully connected Restricted Boltzmann Machine (RBM) with real-valued parameters.
    In this case, two RBMs are taken to parameterize, respectively, phase
    and amplitude of the wave-function, as introduced in Torlai et al., Nature Physics 14, 447–450(2018).
    This type of RBM has spin 1/2 hidden units and is defined by:

    .. math:: \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M
            \cosh \left(\sum_i^N W_{ij} s_i + b_j \right)

    for arbitrary local quantum numbers :math:`s_i`.
    """

    def __init__(
        self,
        hilbert,
        alpha=None,
        n_hidden=None,
        n_hidden_a=None,
        n_hidden_p=None,
        use_visible_bias=True,
        use_hidden_bias=True,
    ):
        r"""
        Constructs a new ``RbmSpinPhase`` machine:

        Args:
           hilbert: Hilbert space object for the system.
           alpha: Hidden unit density.
           n_hidden: The number of hidden spin units to be used in both RBMs. If None,
                     n_hidden_a and n_hidden_p must be specified.
           n_hidden_a: The number of hidden spin units to be used to represent the amplitude.
           n_hidden_p: The number of hidden spin units to be used to represent the phase.
           use_visible_bias: If ``True`` then there would be a
                            bias on the visible units.
                            Default ``True``.
           use_hidden_bias: If ``True`` then there would be a
                           bias on the visible units.
                           Default ``True``.

        Examples:

        """

        n = hilbert.size

        if alpha is not None:
            if alpha < 0:
                raise ValueError("`alpha` should be non-negative")
            if n_hidden is not None:
                if n_hidden != int(round(alpha * n)):
                    raise RuntimeError(
                        """n_hidden is inconsistent with the given alpha.
                                       Remove one of the two or provide consistent values."""
                    )
            n_hidden = int(round(alpha * n))

        if n_hidden is not None:
            m_a = n_hidden
            m_p = n_hidden
        else:
            m_a = n_hidden_a
            m_p = n_hidden_p

        if m_a is None or m_p is None or m_a < 0 or m_p < 0:
            raise RuntimeError("""Invalid number of hidden unit.""")

        self._wa = _np.empty((n, m_a))
        self._wp = _np.empty((n, m_p))

        self._aa = _np.empty(n) if use_visible_bias else None
        self._ap = _np.empty(n) if use_visible_bias else None

        self._ba = _np.empty(m_a) if use_hidden_bias else None
        self._bp = _np.empty(m_p) if use_hidden_bias else None

        self._ra = _np.empty((1, m_a))
        self._rp = _np.empty((1, m_p))

        self.n_hidden_a = m_a
        self.n_hidden_p = m_p

        self._npar = (
            self._wa.size
            + self._wp.size
            + (self._aa.size + self._ap.size if self._aa is not None else 0)
            + (self._ba.size + self._bp.size if self._ba is not None else 0)
        )

        super().__init__(hilbert)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._npar

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The
                 length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
            """
        return self._log_val_kernel(
            x,
            out,
            self._wa,
            self._wp,
            self._aa,
            self._ap,
            self._ba,
            self._bp,
            self._ra,
            self._rp,
        )

    @staticmethod
    @jit(nopython=True)
    def _log_val_kernel(x, out, wa, wp, aa, ap, ba, bp, ra, rp):

        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        ra = x.dot(wa)
        if ba is None:
            _log_cosh_sum(ra, out)
        else:
            _log_cosh_sum(ra + ba, out)

        rp = x.dot(wp)
        if bp is None:
            _log_cosh_sum(rp, out, add_factor=(1.0j))
        else:
            _log_cosh_sum(rp + bp, out, add_factor=(1.0j))

        if aa is not None:
            out = out + x.dot(aa) + 1.0j * x.dot(ap)

        return out

    def der_log(self, x, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.

        Returns:
            `out`
            """

        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.empty((x.shape[0], self.n_par), dtype=_np.complex128)

        batch_size = x.shape[0]

        # Amplitude parameters
        i = 0
        if self._aa is not None:
            out[:, i : i + x.shape[1]] = x
            i += self.n_visible

        r = self._ra
        r = _np.dot(x, self._wa)
        if self._ba is not None:
            r += self._ba
        r = _np.tanh(r)

        if self._ba is not None:
            out[:, i : i + self.n_hidden_a] = r
            i += self.n_hidden_a

        t = out[:, i : i + self._wa.size]
        t.shape = (batch_size, self._wa.shape[0], self._wa.shape[1])
        _np.einsum("ij,il->ijl", x, r, out=t)

        i += self._wa.size

        # Phase parameters
        if self._ap is not None:
            out[:, i : i + x.shape[1]] = 1.0j * x
            i += self.n_visible

        r = self._rp
        r = _np.dot(x, self._wp)
        if self._bp is not None:
            r += self._bp
        r = _np.tanh(r)

        if self._bp is not None:
            out[:, i : i + self.n_hidden_p] = 1.0j * r
            i += self.n_hidden_p

        t = out[:, i : i + self._wp.size]
        t.shape = (batch_size, self._wp.shape[0], self._wp.shape[1])
        _np.einsum("ij,il->ijl", x, r, out=t)
        t *= 1.0j

        return out

    @property
    def is_holomorphic(self):
        r"""This is not holomorphic.
        """
        return False

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        from collections import OrderedDict

        od = OrderedDict()
        if self._aa is not None:
            self._aac = self._aa.astype(_np.complex128)
            self._aa = self._aac.real.view()
            od["a1"] = self._aac.view()

        if self._ba is not None:
            self._bac = self._ba.astype(_np.complex128)
            self._ba = self._bac.real.view()
            od["b1"] = self._bac.view()

        self._wac = self._wa.astype(_np.complex128)
        self._wa = self._wac.real.view()
        od["w1"] = self._wac.view()

        if self._ap is not None:
            self._apc = self._ap.astype(_np.complex128)
            self._ap = self._apc.real.view()
            od["a2"] = self._apc.view()

        if self._bp is not None:
            self._bpc = self._bp.astype(_np.complex128)
            self._bp = self._bpc.real.view()
            od["b2"] = self._bpc.view()

        self._wpc = self._wp.astype(_np.complex128)
        self._wp = self._wpc.real.view()
        od["w2"] = self._wpc.view()

        return od
