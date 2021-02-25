from .. import AbstractMachine
import numpy as _np


class Diagonal(AbstractMachine):
    def __init__(self, density_matrix):
        self._rho = density_matrix
        super().__init__(
            density_matrix.hilbert,
            dtype=density_matrix.dtype,
        )

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
        return self._rho.log_val(x, x, out)

    def der_log(self, x, out=None):
        return self._rho.der_log(x, x, out)

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._rho.n_par

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        return self._rho.state_dict

    @property
    def parameters(self):
        r"""The parameters of the machine."""
        return self._rho.parameters
