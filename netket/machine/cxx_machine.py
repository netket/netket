from .._C_netket.machine import Machine
import numpy as _np

class CxxMachine(Machine):
    def __init__(self, hilbert):
        super(CxxMachine, self).__init__(hilbert)

    def _get_parameters(self):
        return _np.concatenate(tuple(p.reshape(-1) for p in self.state_dict().values()))

    def _set_parameters(self, p):
        if p.shape != (self._n_par(),):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(p.shape, self._n_par())
            )
        i = 0
        for x in map(lambda x: x.reshape(-1), self.state_dict().values()):
            _np.copyto(x, p[i : i + x.size])
            i += x.size

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


        Subclasses should implement this function.

        Since it is very cumbersome to properly handle all the cases, there is
        a default implementation of `log_val` which uses `_log_val` under the
        hood. `_log_val` has similar semantics to `log_val` except that it is
        guaranteed that `v` is a 2D array and that `out` is a 1D array which is
        never `None`.
        """
        if v.ndim == 2:
            assert (
                v.shape[1] == self.n_visible
            ), "v has wrong shape: {}; expected (?, {})".format(v.shape, self.n_visible)
            if out is None:
                out = _np.empty(v.shape[0], dtype=_np.complex128)
            self._log_val(v, out)
            return out
        elif v.ndim == 1:
            assert v.size == self.n_visible, "v has wrong size: {}; expected {}".format(
                v.shape, self.n_visible
            )
            if out is None:
                out = _np.empty(1, dtype=_np.complex128)
            self._log_val(v.reshape(1, -1), out)
            return out[0]
        raise ValueError(
            "v has wrong dimension: {}; expected either 1 or 2".format(v.ndim)
        )

    def _log_val(self, v, out):
        raise NotImplementedError(
            "override me for the default implementation of log_val to work"
        )

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

        Subclasses should implement this function.

        Since it is very cumbersome to properly handle all the cases, there is
        a default implementation of `der_log` which uses `_der_log` under the
        hood. `_der_log` has similar semantics to `der_log` except that it is
        guaranteed that both `v` and `out` are 2D arrays (which implies that
        `out` is never `None`).
        """
        if v.ndim == 2:
            assert (
                v.shape[1] == self.n_visible
            ), "v has wrong shape: {}; expected (?, {})".format(v.shape, self.n_visible)
            out_shape = (v.shape[0], self._n_par())
            if out is None:
                out = _np.empty(out_shape, dtype=_np.complex128)
            assert (
                out.shape == out_shape
            ), "out has wrong shape: {}; expected {}".format(out.shape, out_shape)
            self._der_log(v, out)
            return out
        elif v.ndim == 1:
            assert v.size == self.n_visible, "v has wrong size: {}; expected {}".format(
                v.shape, self.n_visible
            )
            if out is None:
                out = _np.empty(self._n_par(), dtype=_np.complex128)
            assert out.shape == (
                self._n_par(),
            ), "out has wrong shape: {}; expected {}".format(
                out.shape, (self._n_par(),)
            )
            self._der_log(v.reshape(1, -1), out.reshape(1, -1))
            return out
        raise ValueError(
            "v has wrong dimension: {}; expected either 1 or 2".format(v.ndim)
        )

    def _der_log(self, v, out):
        raise NotImplementedError

    # def save(self, filename):
    #     pass

    # def load(self, filename):
    #     pass

    def _is_holomorphic(self):
        r"""Returns whether the wave function is holomorphic.
        """
        raise NotImplementedError

    def _n_par(self):
        r"""Returns the number of variational parameters in the machine.

        Subclasses should implement this function, but to actually get the
        number of parameters, use `self.n_par` attribute.
        """
        raise NotImplementedError
