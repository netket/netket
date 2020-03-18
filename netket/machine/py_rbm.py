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
from netket.utils import sum_log_cosh_complex
from .._C_netket.machine import RbmSpinKernel

__all__ = ["PyRbm"]


class PyRbm(AbstractMachine):
    """
    __Do not use me in production code!__

    A proof of concept implementation of a complex-valued RBM in pure Python.

    This class can be used as a drop-in replacement for `RbmSpin`.
    """

    def __init__(
        self, hilbert, alpha=None, use_visible_bias=True, use_hidden_bias=True
    ):
        r"""Constructs a new RBM.

        Args:
            hilbert: Hilbert space.
            alpha: `alpha * hilbert.size` is the number of hidden spins.
            use_visible_bias: specifies whether to use a bias for visible
                              spins.
            use_hidden_bias: specifies whether to use a bias for hidden spins.
        """

        n = hilbert.size
        if alpha < 0:
            raise ValueError("`alpha` should be non-negative")
        m = int(round(alpha * n))
        self._w = _np.empty([m, n], dtype=_np.complex128)
        self._a = _np.empty(n, dtype=_np.complex128) if use_visible_bias else None
        self._b = _np.empty(m, dtype=_np.complex128) if use_hidden_bias else None

        self.n_hidden = m
        self.n_visible = n

        self._r = _np.empty((1, m), dtype=_np.complex128, order="C")

        self._npar = (
            self._w.size
            + (self._a.size if self._a is not None else 0)
            + (self._b.size if self._b is not None else 0)
        )

        self._kernel = RbmSpinKernel()

        super().__init__(hilbert)

    @property
    def n_par(self):
        r"""Returns the number of parameters in the machine. We just sum the
        sizes of all the tensors we hold.
        """
        return self._npar

    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function given a spin
        configuration ``x``.
        """
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        self._kernel.log_val(x, out, self._w, self._a, self._b)

        # self._r = x.dot(self._w.T)
        # if self._b is not None:
        #     self._r += self._b
        #
        # sum_log_cosh_complex(self._r, out)
        #
        # if self._a is not None:
        #     out += _np.dot(x, self._a)

        return out

    def der_log(self, x, out=None):

        if out is None:
            out = _np.empty((x.shape[0], self.n_par), dtype=_np.complex128)

        batch_size = x.shape[0]

        i = 0
        if self._a is not None:
            out[:, i : i + x.shape[1]] = x
            i += self.n_visible

        r = _np.dot(x, self._w.T)
        if self._b is not None:
            r += self._b
        _np.tanh(r, out=r)

        if self._b is not None:
            out[:, i : i + self.n_hidden] = r
            i += self.n_hidden

        t = out[:, i : i + self._w.size]
        t.shape = (batch_size, self._w.shape[0], self._w.shape[1])
        _np.einsum("ij,il->ijl", r, x, out=t)

        return out

    def vector_jacobian_prod(self, x, vec, out=None):
        return _np.dot(_np.asmatrix(self.der_log(x)).H, vec, out)

    @property
    def is_holomorphic(self):
        r"""Complex valued RBM a holomorphic function.
        """
        return True

    @property
    def state_dict(self):
        from collections import OrderedDict

        od = OrderedDict()
        if self._a is not None:
            od["a"] = self._a.view()

        if self._b is not None:
            od["b"] = self._b.view()

        od["w"] = self._w.view()

        return od
