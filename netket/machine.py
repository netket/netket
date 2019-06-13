from ._C_netket.machine import *

import numpy as np
import pickle


def MPSPeriodicDiagonal(hilbert, bond_dim, symperiod=-1):
    return MPSPeriodic(hilbert, bond_dim, diag=True, symperiod=symperiod)


class CxxMachine(Machine):
    def __init__(self):
        super(CxxMachine, self).__init__()

    @property
    def n_par(self):
        r"""Number of parameters in the machine.

        __Do not override this function!__ Subclasses should implement
        `CxxMachine._number_parameters` instead.
        """
        return self._number_parameters()

    @property
    def n_visible(self):
        r"""Number of "visible units". In other words the dimension of the
        vector one can pass to `log_val`.

        __Do not override this function!__ Subclasses should implement
        `CxxMachine._number_visible` instead.
        """
        return self._number_visible()

    @property
    def parameters(self):
        r"""Variational parameters as a 1D vector.

        __Do not override this attribute!__ Subclasses should implement
        `CxxMachine._get_parameters` and `CxxMachine._set_parameters` instead.
        """
        return self._get_parameters()

    @parameters.setter
    def parameters(self, p):
        self._set_parameters(p)

    @property
    def hilbert(self):
        r"""Returns the hilbert space on which the machine is defined.

        __Do not override this function!__ Subclasses should implement
        `CxxMachine._get_hilbert` instead.
        """
        return self._get_hilbert()

    ## Functions which users should override
    ###########################################################################

    def log_val(self, x):
        r"""Returns the logarithm of the wave function at point x.

        Subclasses should implement this function.

        Args:
            x: 1D vector of `float` of size `self.n_visible`.
        """
        raise NotImplementedError

    def der_log(self, x):
        r"""Returns the gradient of the logarithm of the wave function at point
        x.

        Subclasses should implement this function.

        Args:
            x: 1D vector of `float` of size `self.n_visible`.
        """
        raise NotImplementedError

    def save(self, filename):
        r"""Saves machine's state to file.

        Subclasses should implement this function.

        *Note:* the sequence of saving and loading, -- ``m.save(filename);
        m.load(filename)``, -- should be an identity.

        Args:
            filename: name of the file.
	"""
        raise NotImplementedError

    def load(self, filename):
        r"""Loads machine's state from file.

        *Note:* the sequence of saving and loading, -- ``m.save(filename);
        m.load(filename)``, -- should be an identity.

        Subclasses should implement this function.

        Args:
            filename: name of the file.
	"""
        raise NotImplementedError

    def _number_parameters(self):
        r"""Returns the number of variational parameters in the machine.

        Subclasses should implement this function, but to actually get the
        number of parameters, use `self.n_par` attribute.
        """
        raise NotImplementedError

    def _number_visible(self):
        r"""Returns the number of "visible units" in the machine.

        Subclasses should implement this function, but to actually get the
        dimension of input vector, use `self.n_visible` attribute.
        """
        raise NotImplementedError

    def _get_parameters(self):
        r"""Returns the variational parameters as a 1D complex vector.

        Subclasses should implement this function, but to actually access
        the parameters, use `self.parameters` attribute.
        """
        raise NotImplementedError

    def _set_parameters(self, p):
        r"""Sets the variational parameters.

        Subclasses should implement this function, but to actually access
        the parameters, use `self.parameters` attribute.

        Args:
            p: New variational parameters as a 1D complex vector.
        """
        raise NotImplementedError

    def _get_hilbert(self):
        r"""Returns the Hilbert space object.

        Subclasses should implement this function, but to actually access
        the Hilbert space, use `self.hilbert` attribute.
        """
        raise NotImplementedError


class PyRbm(CxxMachine):
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
            alpha: `alphe * hilbert.size` is the number of hidden spins.
            use_visible_bias: specifies whether to use a bias for visible
                              spins.
            use_hidden_bias: specifies whether to use a bias for hidden spins.
        """
        # NOTE: The following call to __init__ is important!
        super(PyRbm, self).__init__()
        n = hilbert.size
        if alpha < 0:
            raise ValueError("`alpha` should be non-negative")
        m = int(round(alpha * n))
        self._hilbert = hilbert
        self._w = np.empty([m, n], dtype=np.complex128)
        self._a = np.empty(n, dtype=np.complex128) if use_visible_bias else None
        self._b = np.empty(m, dtype=np.complex128) if use_hidden_bias else None

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
        params += (self._w.reshape(-1),)
        return np.concatenate(params)

    def _set_parameters(self, p):
        r"""Sets parameters from a 1D tensor.

        ``self._set_parameters(self._get_parameters())`` is an identity.
        """
        i = 0
        for x in [self._a, self._b, self._w]:
            if x is not None:
                x.reshape(-1)[:] = p[i : i + x.size]
                i += x.size

    def log_val(self, x):
        r = np.dot(self._w, x)
        if self._b is not None:
            r += self._b
        r = np.sum(PyRbm._log_cosh(r))
        if self._a is not None:
            r += np.dot(self._a, x)
        # Officially, we should return
        #     self._w.shape[0] * 0.6931471805599453 + r
        # but the C++ implementation ignores the "constant factor"
        return r

    def der_log(self, x):
        grad = np.empty(self.n_par, dtype=np.complex128)
        i = 0

        if self._a is not None:
            grad[i : i + self._a.size] = x
            i += self._a.size

        tanh_stuff = np.dot(self._w, x)
        if self._b is not None:
            tanh_stuff += self._b
        tanh_stuff = np.tanh(tanh_stuff)

        if self._b is not None:
            grad[i : i + self._b.size] = tanh_stuff
            i += self._b.size

        # NOTE: order='F' is important, because of the order='C' in
        # ``_get_parameters`` and ``_set_parameters``!
        grad[i : i + self._w.size] = np.outer(x, tanh_stuff).reshape(-1, order="F")
        return grad

    def _get_hilbert(self):
        return self._hilbert

    def _is_holomorphic(self):
        return True

    def save(self, filename):
        r"""Saves machine weights to ``filename`` using ``pickle``.
        """
        with open(filename, "wb") as output_file:
            pickle.dump((self._w, self._a, self._b), output_file)

    def load(self, filename):
        r"""Loads machine weights from ``filename`` using ``pickle``.
        """
        with open(filename, "rb") as input_file:
            self._w, self._a, self._b = pickle.load(input_file)

    @staticmethod
    def _log_cosh(x):
        # TODO: Handle big numbers properly
        return np.log(np.cosh(x))
