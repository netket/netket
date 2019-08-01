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
    """
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
        super(PyRbm, self).__init__(hilbert)
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
        for i in range(x.shape[0]):
            r = _np.dot(self._w, x[i])
            if self._b is not None:
                r += self._b
            r = _np.sum(PyRbm._log_cosh(r))
            if self._a is not None:
                r += _np.dot(self._a, x[i])
            # Officially, we should return
            #     self._w.shape[0] * 0.6931471805599453 + r
            # but the C++ implementation ignores the "constant factor"
            out[i] = r

    def _der_log(self, x, out):
        for j in range(x.shape[0]):
            grad = out[j]
            i = 0

            if self._a is not None:
                grad[i : i + self._a.size] = x[j]
                i += self._a.size

            tanh_stuff = _np.dot(self._w, x[j])
            if self._b is not None:
                tanh_stuff += self._b
            tanh_stuff = _np.tanh(tanh_stuff, out=tanh_stuff)

            if self._b is not None:
                grad[i : i + self._b.size] = tanh_stuff
                i += self._b.size

            tail = grad[i : i + self._w.size]
            tail.shape = (tanh_stuff.size, x[j].size)
            _np.outer(tanh_stuff, x[j], out=tail)

    def _is_holomorphic(self):
        r"""Complex valued RBM a holomorphic function.
        """
        return True

    def state_dict(self):
        from collections import OrderedDict

        return OrderedDict(
            [("a", self._a.view()), ("b", self._b.view()), ("w", self._w.view())]
        )

    @staticmethod
    def _log_cosh(x):
        # TODO: Handle big numbers properly
        return _np.log(_np.cosh(x))
