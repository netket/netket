from . import AbstractDensityMatrix
from .. import RbmSpin as PureRbmSpin
import numpy as _np


class RbmSpin(AbstractDensityMatrix):
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
        super().__init__(hilbert, dtype=dtype)

        if automorphisms is not None:
            if isinstance(automorphisms, netket.graph.AbstractGraph):
                automorphisms = automorphisms.automorphisms()
            import itertools

            automorphisms = [
                prod[0] + prod[1] for prod in itertools.product(autom, autom)
            ]

        input_like = _np.zeros(hilbert.size * 2)
        self._prbm = PureRbmSpin(
            input_like,
            n_hidden,
            alpha,
            use_visible_bias,
            use_hidden_bias,
            automorphisms,
            dtype,
        )
        self._plog_val = self._prbm.log_val
        self._pder_log = self._prbm.der_log

    def log_val(self, xr, xc=None, out=None):
        r"""Computes the logarithm of the density matrix for a batch of visible
        quantum numbers `(xr,xc)` and stores the result into `out`.
        Specifically, for each element of the batch i, this function should compute
        out[i]=log(rho(xr[i],xc[i])).
        If xr is None, it is assumed that xr has twice as many quantum numbers and
        contains both row and columns, stored contigously.

        Args:
            xr: A matrix of `float64` of shape `(*, self.n_visible)` if xc is given.
                If xc is None, then this should be a matrix of `float64` of shape `(*, 2*self.n_visible)`.
            xc (optional): A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. The length of `out` should be `xr.shape[0]`.

        Returns:
            A vector out[i]=log(rho(xr[i],xc[i])).
        """
        if xc is None:
            return self._plog_val(xr, out)
        else:
            return self._plog_val(_np.hstack((xr, xc)), out)

    def der_log(self, xr, xc=None, out=None):
        r"""Computes the gradient of the logarithm of the density matrix for a
        batch of visible configurations `(xr,xc)` and stores the result into `out`.

        Args:
            xr: A matrix of `float64` of shape `(*, self.n_visible)` if xc is given.
                If xc is None, then this should be a matrix of `float64` of shape `(*, 2*self.n_visible)`.
            xc (optional): A matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`.
                `out` should be a matrix of shape `(x.shape[0], self.n_par)`.

        Returns:
            `out`
        """
        if xc is None:
            return self._pder_log(xr, out)
        else:
            return self._pder_log(_np.hstack((xr, xc)), out)

    @property
    def state_dict(self):
        r"""A dictionary containing the parameters of this machine"""
        return self._prbm.state_dict
