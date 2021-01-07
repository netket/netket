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
from netket.graph import AbstractGraph
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

    for arbitrary local quantum numbers :math:`s_i`. Here a and b are
    called, respectively, visible and hidden bias.

    The weights can be taken to be complex-valued (default option) or real-valued.
    """

    def __init__(
        self,
        hilbert,
        n_hidden=None,
        alpha=None,
        use_visible_bias=True,
        use_hidden_bias=True,
        automorphisms=None,
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
            use_visible_bias: If ``True`` a bias on the visible units is taken.
                              Default ``True``.
            use_hidden_bias: If ``True`` bias on the hidden units is taken.
                             Default ``True``.
            automorphisms (optional): a list of automorphisms to use as symmetries.
            dtype: either complex or float, is the type used for the weights.

        Examples:
            A ``RbmSpin`` machine with hidden unit density
            alpha = 2 for a one-dimensional L=20 spin-half system:

            >>> from netket.machine import RbmSpin
            >>> from netket.hilbert import Spin
            >>> hi = Spin(s=0.5, total_sz=0, N=20)
            >>> ma = RbmSpin(hilbert=hi,alpha=2)
            >>> print(ma.n_par)
            860
        """

        super().__init__(hilbert, dtype=dtype)

        n = hilbert.size

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        self._autom, self.n_hidden, alpha_symm = self._get_hidden(
            automorphisms, hilbert, n_hidden, alpha
        )

        m = self.n_hidden

        self._w = _np.empty((n, m), dtype=self._npdtype)
        self._a = _np.empty(n, dtype=self._npdtype) if use_visible_bias else None
        self._b = _np.empty(m, dtype=self._npdtype) if use_hidden_bias else None
        self._r = _np.empty((1, m), dtype=self._npdtype)

        self._n_bare_par = (
            self._w.size
            + (self._a.size if self._a is not None else 0)
            + (self._b.size if self._b is not None else 0)
        )

        if automorphisms is None or automorphisms is False:
            self._ws = self._w.view()
            self._as = self._a.view() if use_visible_bias else None
            self._bs = self._b.view() if use_hidden_bias else None
            self._n_par = self._n_bare_par
        else:
            self._der_mat_symm, self._n_par = self._build_der_mat(
                use_visible_bias, use_hidden_bias, n, alpha_symm, self._autom
            )

            self._ws = _np.empty((n, alpha_symm), dtype=self._npdtype)
            self._as = _np.empty(1, dtype=self._npdtype) if use_visible_bias else None
            self._bs = (
                _np.empty(alpha_symm, dtype=self._npdtype) if use_hidden_bias else None
            )

            self._set_bare_parameters(
                self._a, self._b, self._w, self._as, self._bs, self._ws, self._autom
            )

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._n_par

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
        if self._autom is None:
            return self._bare_der_log(x, out)
        else:
            self._outb = self._bare_der_log(x)
            if out is None:
                out = _np.empty((x.shape[0], self._n_par), dtype=_np.complex128)
            return _np.matmul(self._outb, self._der_mat_symm, out=out)

    def _bare_der_log(self, x, out=None):

        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.empty((x.shape[0], self._n_bare_par), dtype=_np.complex128)

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
    def state_dict(self):
        return self.state_dict_with_prefix()

    def state_dict_with_prefix(self, prefix=""):
        r"""A dictionary containing the parameters of this machine"""
        from collections import OrderedDict

        od = OrderedDict()
        if self._dtype is complex:
            if self._a is not None:
                od[prefix + "a"] = self._as.view()

            if self._b is not None:
                od[prefix + "b"] = self._bs.view()

            od[prefix + "w"] = self._ws.view()
        else:
            if self._a is not None:
                self._ac = self._as.astype(_np.complex128)
                self._as = self._ac.real.view()
                od[prefix + "a"] = self._ac.view()
                if self._autom is None:
                    self._a = self._as.view()

            if self._b is not None:
                self._bc = self._bs.astype(_np.complex128)
                self._bs = self._bc.real.view()
                od[prefix + "b"] = self._bc.view()
                if self._autom is None:
                    self._b = self._bs.view()

            self._wc = self._ws.astype(_np.complex128)
            self._ws = self._wc.real.view()
            od[prefix + "w"] = self._wc.view()
            if self._autom is None:
                self._w = self._ws.view()

        return od

    @property
    def parameters(self):
        return _np.concatenate(tuple(p.reshape(-1) for p in self.state_dict.values()))

    @parameters.setter
    def parameters(self, p):
        if p.shape != (self.n_par,):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(p.shape, self.n_par)
            )

        i = 0
        for x in map(lambda x: x.reshape(-1), self.state_dict.values()):
            _np.copyto(x, p[i : i + x.size])
            i += x.size

        if self._autom is not None:
            self._set_bare_parameters(
                self._a, self._b, self._w, self._as, self._bs, self._ws, self._autom
            )

    @staticmethod
    @jit
    def _build_der_mat(use_visible_bias, use_hidden_bias, n_visible, alpha, permtable):
        perm_size = permtable.shape[0]
        n_hidden = alpha * perm_size

        n_par = int(n_visible * alpha + use_visible_bias + alpha * use_hidden_bias)
        n_bare_par = int(
            n_visible * n_hidden
            + use_visible_bias * n_visible
            + n_hidden * use_hidden_bias
        )

        der_mat_symm = _np.zeros((n_par, n_bare_par))

        k = 0
        k_bare = 0

        if use_visible_bias:
            # derivative with respect to a
            for p in range(n_visible):
                der_mat_symm[k, p] = 1
                k_bare += 1
            k += 1

        if use_hidden_bias:
            # derivatives with respect to b
            for p in range(n_hidden):
                k_symm = int(_np.floor(p / perm_size))
                der_mat_symm[k_symm + k, k_bare] = 1
                k_bare += 1

            k += alpha

        # derivatives with respect to W
        for i in range(n_visible):
            for j in range(n_hidden):
                isymm = permtable[j % perm_size, i]
                jsymm = int(_np.floor(j / perm_size))
                ksymm = jsymm + int(alpha * isymm)

                der_mat_symm[ksymm + k, k_bare] = 1

                k_bare += 1

        return der_mat_symm.T, n_par

    @staticmethod
    def _get_hidden(automorphisms, hilbert, n_hidden, alpha):
        if (automorphisms is None) or (automorphisms is False):
            if alpha is None:
                m = n_hidden
                if n_hidden < 0:
                    raise RuntimeError("n_hidden must be positive.")
            else:
                m = int(round(alpha * hilbert.size))
                if alpha < 0:
                    raise ValueError("`alpha` should be non-negative")

            if n_hidden is not None:
                if n_hidden != m:
                    raise RuntimeError(
                        """n_hidden is inconsistent with the given alpha.
                                       Remove one of the two or provide consistent values."""
                    )

            return None, m, m
        else:
            if isinstance(automorphisms, AbstractGraph):
                autom = _np.asarray(automorphisms.automorphisms())
            else:
                try:
                    autom = _np.asarray(automorphisms)
                    assert hilbert.size == autom.shape[1]
                except:
                    raise RuntimeError("Cannot find a valid automorphism array.")

            if not _np.issubdtype(type(alpha), _np.integer):
                raise ValueError(
                    "alpha must be a positive integer value when using symmetries."
                )

            alphasym = int(alpha * autom.shape[1] / autom.shape[0])
            if alphasym == 0 and alpha > 0:
                print(
                    "Warning, the given value of alpha is too small, given the size of the automorphisms."
                )
            m = int(alphasym * autom.shape[0])

            return autom, m, alphasym

    @staticmethod
    @jit
    def _set_bare_parameters(a, b, W, a_s, b_s, W_s, permtable):

        perm_size = permtable.shape[0]

        if a is not None:
            a.fill(a_s)

        if b is not None:
            for j in range(b.shape[0]):
                jsymm = int(_np.floor(j / perm_size))
                b[j] = b_s[jsymm]

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                jsymm = int(_np.floor(j / perm_size))
                W[i, j] = W_s[permtable[j % perm_size][i], jsymm]


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
        automorphisms=None,
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
            automorphisms (optional): A list of automorphisms to be taken as symmetries of the
                           rbm. They can be obtained by calling graph.automorphism()

        Examples:
           A ``RbmSpinReal`` machine with hidden unit density
           alpha = 2 for a one-dimensional L=20 spin-half system:


           >>> from netket.machine import RbmSpinReal
           >>> from netket.hilbert import Spin
           >>> g = Hypercube(length=20, n_dim=1)
           >>> hi = Spin(s=0.5, total_sz=0, N=4)
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
            automorphisms=automorphisms,
            dtype=float,
        )


class RbmSpinSymm(RbmSpin):
    r"""
    A fully connected Restricted Boltzmann Machine with lattice
    symmetries. This type of RBM has spin 1/2 hidden units and is
    defined by:

    .. math:: \Psi(s_1,\dots s_N) = e^{\sum_i^N a_i s_i} \times \Pi_{j=1}^M
    \cosh \left(\sum_i^N W_{ij} s_i + b_j \right)

    for arbitrary local quantum numbers :math:`s_i`. However, the weights
    (:math:`W_{ij}`) and biases (:math:`a_i`, :math:`b_i`) respects the
    symmetries of the lattice as specificed in hilbert.graph.automorphisms.

    The values of the weights can be chosen to be complex or real-valued.

    """

    def __init__(
        self,
        hilbert,
        automorphisms,
        alpha=None,
        use_visible_bias=True,
        use_hidden_bias=True,
        dtype=complex,
    ):
        r"""
        Constructs a new ``RbmSpinSymm`` machine with complex weights:

        Args:
           hilbert: Hilbert space object for the system.
           alpha: Hidden unit density.
           use_visible_bias: If ``True`` then there would be a
                            bias on the visible units.
                            Default ``True``.
           use_hidden_bias: If ``True`` then there would be a
                           bias on the visible units.
                           Default ``True``.

        Examples:
           A ``RbmSpinSymm`` machine with hidden unit density
           alpha = 2 for a one-dimensional L=20 spin-half system:

           >>> from netket.machine import RbmSpinSymm
           >>> from netket.hilbert import Spin
           >>> hi = Spin(s=0.5, total_sz=0, N=20)
           >>> ma = RbmSpinSymm(hilbert=hi, alpha=2)
           >>> print(ma.n_par)
           43
        """
        super().__init__(
            hilbert,
            alpha=alpha,
            use_visible_bias=use_visible_bias,
            use_hidden_bias=use_hidden_bias,
            automorphisms=automorphisms,
            dtype=dtype,
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
        automorphisms=None,
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
            automorphisms (optional): A list of automorphisms to be taken as symmetries of the
                           rbm. They can be obtained by calling graph.automorphism()
            dtype: either complex or float, is the type used for the weights.

        Examples:
            A ``RbmMultiVal`` machine with hidden unit density
            alpha = 1 for a one-dimensional bosonic system:

            >>> from netket.machine import RbmMultiVal
            >>> from netket.hilbert import Boson
            >>> hi = Boson(n_max=3, n_bosons=8, N=10)
            >>> ma = RbmMultiVal(hilbert=hi, alpha=1, dtype=float, use_visible_bias=False)
            >>> print(ma.n_par)
            1056
        """
        local_states = _np.asarray(hilbert.local_states, dtype=int)

        # creating a fictious hilbert object to pass to the standard rbm
        l_hilbert = _np.zeros(local_states.size * hilbert.size)

        # creating the symmetries for the extended space
        autom = self._make_extended_symmetry(automorphisms, hilbert)

        super().__init__(
            l_hilbert,
            n_hidden,
            alpha,
            use_visible_bias,
            use_hidden_bias,
            autom,
            dtype,
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

    @staticmethod
    def _make_extended_symmetry(automorphisms, hilbert):
        if automorphisms is not None:
            if isinstance(automorphisms, AbstractGraph):
                autom = _np.asarray(automorphisms.automorphisms())
            else:
                try:
                    autom = _np.asarray(automorphisms)
                    assert hilbert.size == autom.shape[1]
                except:
                    raise RuntimeError("Cannot find a valid automorphism array.")

            ldim = len(hilbert.local_states)

            autop = _np.zeros((autom.shape[0], autom.shape[1] * ldim), dtype=int)

            for k in range(autom.shape[0]):
                for i in range(autom.shape[1]):
                    autop[k, i * ldim : (i + 1) * ldim] = _np.arange(
                        autom[k, i] * ldim, (autom[k, i] + 1) * ldim, 1, dtype=int
                    )
        else:
            autop = None
        return autop

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
        alpha_a=None,
        alpha_p=None,
        n_hidden_a=None,
        n_hidden_p=None,
        use_visible_bias=True,
        use_hidden_bias=True,
        automorphisms=None,
    ):
        r"""
        Constructs a new ``RbmSpinPhase`` machine:

        Args:
            hilbert: Hilbert space object for the system.
            alpha: Hidden unit density.
            n_hidden: The number of hidden spin units to be used in both RBMs. If None,
                     n_hidden_a and n_hidden_p must be specified.
            alpha_a: Hidden unit density for the amplitude. If None, the general parameter alpha is used.
            alpha_p: Hidden unit density for the phase. If None, the general parameter alpha is used.
                    n_hidden_a: The number of hidden spin units to be used to represent the amplitude.
            If None, the general parameter alpha (or n_hidden) is used to derive the number of hidden units.
            n_hidden_p: The number of hidden spin units to be used to represent the phase.
            If None, the general parameter alpha (or n_hidden) is used to derive the number of hidden units.
            use_visible_bias: If ``True`` then there would be a
                            bias on the visible units.
                            Default ``True``.
            use_hidden_bias: If ``True`` then there would be a
                           bias on the visible units.
                           Default ``True``.
            automorphisms (optional): List of automorphisms to enforcee symmetries in the Boltzmann Machine.

        Examples:

        """
        if n_hidden is not None:
            n_hidden_a = n_hidden
            n_hidden_p = n_hidden

        if alpha is not None:
            alpha_a = alpha
            alpha_p = alpha

        self._rbm_a = RbmSpin(
            hilbert=hilbert,
            n_hidden=n_hidden_a,
            alpha=alpha_a,
            use_visible_bias=use_visible_bias,
            use_hidden_bias=use_hidden_bias,
            automorphisms=automorphisms,
            dtype=float,
        )
        self._rbm_p = RbmSpin(
            hilbert=hilbert,
            n_hidden=n_hidden_p,
            alpha=alpha_p,
            use_visible_bias=use_visible_bias,
            use_hidden_bias=use_hidden_bias,
            automorphisms=automorphisms,
            dtype=float,
        )

        self._n_par = self._rbm_a.n_par + self._rbm_p.n_par

        super().__init__(hilbert, dtype=float)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._n_par

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
        out = self._rbm_a.log_val(x, out)
        out += 1.0j * self._rbm_p.log_val(x)

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
        n_par = self._n_par
        n_par_a = self._rbm_a.n_par

        if out is None:
            out = _np.empty((x.shape[0], n_par), dtype=_np.complex128)

        self._rbm_a.der_log(x, out[:, :n_par_a])

        self._rbm_p.der_log(x, out[:, n_par_a:])
        out[:, n_par_a:] *= 1.0j

        return out

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        od = self._rbm_a.state_dict_with_prefix(prefix="amplitude_")
        od.update(self._rbm_p.state_dict_with_prefix(prefix="phase_"))
        return od

    @property
    def parameters(self):
        return _np.concatenate(tuple(p.reshape(-1) for p in self.state_dict.values()))

    @parameters.setter
    def parameters(self, p):
        self._rbm_a.parameters = p[: self._rbm_a.n_par]
        self._rbm_p.parameters = p[self._rbm_a.n_par :]
