import abc
import numpy as _np


class AbstractMachine(abc.ABC):
    """Abstract class for NetKet machines"""

    def __init__(self, hilbert):
        super().__init__()
        self.hilbert = hilbert

    @abc.abstractmethod
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
        pass

    def init_random_parameters(self, seed=None, sigma=0.01):
        rgen = _np.random.RandomState(seed)
        self.parameters = rgen.normal(
            scale=sigma, size=self.n_par
        ) + 1.0j * rgen.normal(scale=sigma, size=self.n_par)

    def vector_jacobian_prod(self, x, vec, out=None):
        r"""Computes the scalar product between gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and a vector `vec`. The result is stored into `out`.

        Args:
             x: a matrix or 3d tensor of `float64` of shape `(*, self.n_visible)` or `(*, *, self.n_visible)`.
             vec: a `complex128` vector or matrix used to compute the inner product with the jacobian.
             out: The result of the inner product, it is a vector of `complex128` and length `self.n_par`.


        Returns:
             `out`
        """

        if x.ndim == 3:
            if out is None:
                out = _np.zeros(self.n_par, dtype=_np.complex128)
            else:
                out.fill(0.0)

            for xb, vb in zip(x, vec):
                out += _np.dot(self.der_log(xb).conjugate().transpose(), vb)

        elif x.ndim == 2:
            out = _np.dot(self.der_log(x).conjugate().transpose(), vec, out)

        return out

    def jacobian_vector_prod(self, v, vec, out=None):
        return NotImplementedError

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
        return NotImplementedError

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
            return RuntimeError("The hilbert space is not indexable")

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
            return RuntimeError("The hilbert space is not indexable")

    @property
    @abc.abstractmethod
    def is_holomorphic(self):
        r"""Returns whether the wave function is holomorphic."""
        return NotImplementedError

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self.parameters.size

    @property
    @abc.abstractmethod
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        return NotImplementedError

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

    def save(self, file):
        assert type(file) is str
        with open(file, "wb") as file_ob:
            _np.save(file_ob, self.parameters, allow_pickle=False)

    def load(self, file):
        self.parameters = _np.load(file, allow_pickle=False)

    @property
    def n_visible(self):
        return self.hilbert.size
