import abc
import numpy as _np


class AbstractMachine(abc.ABC):
    """Abstract class for NetKet machines"""

    def __init__(self, hilbert):
        super().__init__()
        self.hilbert = hilbert

    @abc.abstractmethod
    def log_val(self, v, out=None):
        r"""Computes the logarithm of the wave function for a batch of visible
        configurations `v` and stores the result into `out`.

        Args:
            v: Either a
                * vector of `float64` of size `self.n_visible` or
                * a matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. If `v` is a matrix then
                length of `out` should be `v.shape[0]`. If `v` is a vector,
                then length of `out` should be 1.

        Returns:
            A complex number when `v` is a vector and vector when `v` is a
            matrix.
            """
        pass

    @abc.abstractmethod
    def der_log(self, v, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `v` and stores the result into `out`.

        Args:
            v: Either a
                * vector of `float64` of size `self.n_visible` or
                * a matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`. If `v` is a matrix then of
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.
                If `v` is a vector, then `out` should be a vector of length
                `self.n_par`.

        Returns:
            `out`
            """
        pass

    @property
    @abc.abstractmethod
    def is_holomorphic(self):
        r"""Returns whether the wave function is holomorphic."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        return NotImplementedError

    @property
    def parameters(self):
        return _np.concatenate(tuple(p.reshape(-1) for p in self.state_dict().values()))

    @parameters.setter
    def parameters(self, p):
        if p.shape != (self.n_par,):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(p.shape, self.n_par)
            )
        i = 0
        for x in map(lambda x: x.reshape(-1), self.state_dict().values()):
            _np.copyto(x, p[i : i + x.size])
            i += x.size
