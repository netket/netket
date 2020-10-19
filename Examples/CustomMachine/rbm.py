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

import netket
import numpy as _np

__all__ = ["PyRbm"]


class PyRbm(netket.machine.CxxMachine):
    r"""
    __Do not use me in production code!__

    A proof of concept implementation of a complex-valued RBM in pure Python.
    This is an example of how to subclass `CxxMachine` so that the machine will
    be usable with NetKet's C++ core.

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
        # NOTE: The following call to __init__ is important!
        super(PyRbm, self).__init__(hilbert, dtype=complex)
        n = hilbert.size
        if alpha < 0:
            raise ValueError("`alpha` should be non-negative")
        m = int(round(alpha * n))
        self._w = _np.empty([m, n], dtype=_np.complex128)
        self._a = _np.empty(n, dtype=_np.complex128) if use_visible_bias else None
        self._b = _np.empty(m, dtype=_np.complex128) if use_hidden_bias else None

    def _n_par(self):
        r"""Returns the number of parameters in the machine. We just sum the
        sizes of all the tensors we hold.
        """
        return (
            self._w.size
            + (self._a.size if self._a is not None else 0)
            + (self._b.size if self._b is not None else 0)
        )

    def _log_val(self, x, out):
        r"""Computes the logarithm of the wave function given a spin
        configuration ``x``.
        """
        r = _np.dot(x, self._w.T)
        if self._b is not None:
            r += self._b

        _np.sum(PyRbm._log_cosh(r), axis=-1, out=out)

        if self._a is not None:
            out += _np.dot(x, self._a)
        # Officially, we should return
        #     self._w.shape[0] * 0.6931471805599453 + r
        # but the C++ implementation ignores the "constant factor"

    def _der_log(self, x, out):

        batch_size = x.shape[0]

        i = 0
        if self._a is not None:
            out[:, i : i + x.shape[1]] = x
            i += x.shape[1]

        r = _np.dot(x, self._w.T)
        if self._b is not None:
            r += self._b
        _np.tanh(r, out=r)

        if self._b is not None:
            out[:, i : i + self._b.shape[0]] = r
            i += self._b.shape[0]

        t = out[:, i : i + self._w.size]
        t.shape = (batch_size, self._w.shape[0], self._w.shape[1])
        _np.einsum("ij,il->ijl", r, x, out=t)

    def state_dict(self):
        from collections import OrderedDict

        return OrderedDict(
            [("a", self._a.view()), ("b", self._b.view()), ("w", self._w.view())]
        )

    @staticmethod
    def _log_cosh(x):
        # TODO: Handle big numbers properly
        return _np.log(_np.cosh(x))
