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

    def _number_parameters(self):
        r"""Returns the number of parameters in the machine. We just sum the
        sizes of all the tensors we hold.
        """
        return (
            self._w.size
            + (self._a.size if self._a is not None else 0)
            + (self._b.size if self._b is not None else 0)
        )

    def _number_visible(self):
        r"""Returns the number of visible units.
        """
        return self._w.shape[1]

    def _get_parameters(self):
        r"""Returns the parameters as a 1D tensor.

        This function tries to order parameters in the exact same way as
        ``RbmSpin`` does so that we can do stuff like

        >>> import netket
        >>> import numpy
        >>> hilbert = netket.hilbert.Spin(
                graph=netket.graph.Hypercube(length=100, n_dim=1),
                s=1/2.
            )
        >>> cxx_rbm = netket.machine.RbmSpin(hilbert, alpha=3)
        >>> py_rbm = netket.machine.PyRbm(hilbert, alpha=3)
        >>> cxx_rbm.init_random_parameters()
        >>> # Order of parameters is the same, so we can assign one to the
        >>> # other
        >>> py_rbm.parameters = cxx_rbm.parameters
        >>> x = np.array(hilbert.local_states, size=hilbert.size)
        >>> assert numpy.isclose(py_rbm.log_val(x), cxx_rbm.log_val(x))
        """
        params = tuple()
        if self._a is not None:
            params += (self._a,)
        if self._b is not None:
            params += (self._b,)
        params += (self._w.reshape(-1, order="C"),)
        return _np.concatenate(params)

    def _set_parameters(self, p):
        r"""Sets parameters from a 1D tensor.

        ``self._set_parameters(self._get_parameters())`` is an identity.
        """
        i = 0
        if self._a is not None:
            self._a[:] = p[i : i + self._a.size]
            i += self._a.size
        if self._b is not None:
            self._b[:] = p[i : i + self._b.size]
            i += self._b.size

        self._w[:] = p[i : i + self._w.size].reshape(self._w.shape, order="C")

    def log_val(self, x):
        r"""Computes the logarithm of the wave function given a spin
        configuration ``x``.
        """
        r = _np.dot(self._w, x)
        if self._b is not None:
            r += self._b
        r = _np.sum(PyRbm._log_cosh(r))
        if self._a is not None:
            r += _np.dot(self._a, x)
        # Officially, we should return
        #     self._w.shape[0] * 0.6931471805599453 + r
        # but the C++ implementation ignores the "constant factor"
        return r

    def der_log(self, x):
        r"""Computes the gradient of the logarithm of the wave function
        given a spin configuration ``x``.
        """
        grad = _np.empty(self.n_par, dtype=_np.complex128)
        i = 0

        if self._a is not None:
            grad[i : i + self._a.size] = x
            i += self._a.size

        tanh_stuff = _np.dot(self._w, x)
        if self._b is not None:
            tanh_stuff += self._b
        tanh_stuff = _np.tanh(tanh_stuff, out=tanh_stuff)

        if self._b is not None:
            grad[i : i + self._b.size] = tanh_stuff
            i += self._b.size

        out = grad[i : i + self._w.size]
        out.shape = (tanh_stuff.size, x.size)
        _np.outer(tanh_stuff, x, out=out)

        return grad

    def _is_holomorphic(self):
        r"""Complex valued RBM a holomorphic function.
        """
        return True

    def save(self, filename):
        r"""Saves machine weights to ``filename`` using ``pickle``.
        """
        import pickle

        with open(filename, "wb") as output_file:
            pickle.dump((self._w, self._a, self._b), output_file)

    def load(self, filename):
        r"""Loads machine weights from ``filename`` using ``pickle``.
        """
        import pickle

        with open(filename, "rb") as input_file:
            self._w, self._a, self._b = pickle.load(input_file)

    @staticmethod
    def _log_cosh(x):
        # TODO: Handle big numbers properly
        return _np.log(_np.cosh(x))
