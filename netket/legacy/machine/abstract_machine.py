import abc
import numpy as _np


class AbstractMachine(abc.ABC):
    """Abstract class for NetKet machines"""

    def __init__(self, hilbert, dtype=complex):
        super().__init__()
        self.hilbert = hilbert

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype

    @abc.abstractmethod
    def log_val(self, x, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `x` and stores the result into `out`.

        Args:
            x: A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The length of `out` should be `x.shape[0]`.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
        """
        pass

    def init_random_parameters(self, seed=None, sigma=0.01):
        rgen = _np.random.RandomState(seed)
        self.parameters = rgen.normal(
            scale=sigma, size=self.n_par
        ) + 1.0j * rgen.normal(scale=sigma, size=self.n_par)

    def vector_jacobian_prod(
        self, x, vec, out=None, conjugate=True, return_jacobian=False
    ):
        r"""Computes the scalar product between gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and a vector `vec`. The result is stored into `out`.

        Args:
            x: a matrix or 3d tensor of `float64` of shape `(*, self.n_visible)` or `(*, *, self.n_visible)`.
            vec: a `complex128` vector or matrix used to compute the inner product with the jacobian.
            out: The result of the inner product, it is a vector of `complex128` and length `self.n_par`.
            conjugate (bool): If true, this computes the conjugate of the vector jacobian product.
            return_jacobian (bool): If true, the Jacobian is returned.


        Returns:
            `out` or (out,jacobian) if return_jacobian is True
        """
        vec = vec.reshape(-1)

        if x.ndim == 3:

            jacobian = _np.stack([self.der_log(xb) for xb in x])

            if conjugate:
                out = _np.tensordot(vec, jacobian.conjugate(), axes=1)
            else:
                out = _np.tensordot(vec.conjugate(), jacobian, axes=1)

        elif x.ndim == 2:

            jacobian = self.der_log(x)

            if conjugate:
                out = _np.dot(jacobian.transpose().conjugate(), vec, out)
            else:
                out = _np.dot(jacobian.transpose(), vec.conjugate(), out)

        out = out.reshape(-1)

        return (out, jacobian) if return_jacobian else out

    def jacobian_vector_prod(
        self, x, vec, out=None, conjugate=True, return_jacobian=False
    ):
        raise NotImplementedError

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
        raise NotImplementedError

    def to_array(self, normalize=True, batch_size=512):
        r"""
        Returns a numpy array representation of the machine.
        Note that, in general, the size of the array is exponential
        in the number of quantum numbers, and this operation should thus
        only be performed for low-dimensional Hilbert spaces.

        This method requires an indexable Hilbert space.

        Args:
            normalize (bool): If True, the returned array is normalized in L2 norm.
            batch_size (int): The method log_val of the machine is called on
                              batch_size states at once.

        Returns:
            numpy.array: The array machine(x) for all states x in the Hilbert space.

        """
        if self.hilbert.is_indexable:
            psi = self.log_val(self.hilbert.all_states())
            logmax = psi.real.max()
            psi = _np.exp(psi - logmax)

            if normalize:
                norm = _np.linalg.norm(psi)
                psi /= norm

            return psi
        else:
            raise RuntimeError("The hilbert space is not indexable")

    def log_norm(self, order=2):
        r"""
        Returns the log of the norm of the machine.
        Note that, in general, the size of the array is exponential
        in the number of quantum numbers, and this operation should thus
        only be performed for low-dimensional Hilbert spaces.

        This method requires an indexable Hilbert space.

        Args:
            order (int): By default order=2 is the L2 norm.

        Returns:
            float: log(|machine(x)|^order)

        """
        if self.hilbert.is_indexable:
            psi = self.log_val(self.hilbert.all_states())
            maxl = psi.real.max()
            log_n = _np.log(_np.exp(order * (psi.real - maxl)).sum())

            return log_n + maxl * order
        else:
            raise RuntimeError("The hilbert space is not indexable")

    @property
    def dtype(self):
        return self._dtype

    @property
    def has_complex_parameters(self):
        return self._dtype is complex

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self.parameters.size

    @property
    @abc.abstractmethod
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        raise NotImplementedError

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

    def numpy_flatten(self, data):
        r"""Returns a flattened numpy array representing the given data.
            This is typically used to serialize parameters and gradients.
            The default implementation attempts to return a simple reshaped view.

        Args:
            data: a contigous numpy-compatible array.

        Returns:
            numpy.ndarray: a one-dimensional array containing a view of the data
        """
        return _np.asarray(data).reshape(-1)

    def numpy_unflatten(self, data, shape_like):
        r"""Attempts a deserialization of the given numpy data.
            This is typically used to deserialize parameters and gradients.

        Args:
            data: a 1d numpy array.
            shape_like: this as in instance having the same type and shape of the desired conversion.

        Returns:
            A numpy array containing a view of data and compatible with the given shape.
        """
        return _np.asarray(data).reshape(shape_like.shape)

    def save(self, file):
        assert type(file) is str
        with open(file, "wb") as file_ob:
            _np.save(file_ob, self.numpy_flatten(self.parameters), allow_pickle=False)

    def load(self, file):
        self.parameters = self.numpy_unflatten(
            _np.load(file, allow_pickle=False), self.parameters
        )

    @property
    def input_size(self):
        return self.hilbert.size

    def __repr__(self):
        return "{}(input_size={}, n_par={}, complex_parameters={})".format(
            type(self).__name__,
            self.input_size,
            self.n_par,
            self.has_complex_parameters,
        )
