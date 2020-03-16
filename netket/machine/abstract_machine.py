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
            x: Either a
                * vector of `float64` of size `self.n_visible` or
                * a matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination vector of `complex128`. If `x` is a matrix then
                length of `out` should be `x.shape[0]`. If `x` is a vector,
                then length of `out` should be 1.

        Returns:
            A complex number when `x` is a vector and vector when `x` is a
            matrix.
            """
        pass

    @abc.abstractmethod
    def vector_jacobian_prod(selv, x, vec, out=None):
        r"""Computes the scalar product between gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and a vector `vec`. The result is stored into `out`.

        Args:
             x: a matrix of `float64` of shape `(*, self.n_visible)`.
             vec: a `complex128` vector used to compute the inner product with the jacobian.
             out: The result of the inner product, it is a vector of `complex128` and length `self.n_par`.


        Returns:
             `out`
        """
        return NotImplementedError

    def jacobian_vector_prod(self, v, vec, out=None):
        return NotImplementedError

    def der_log(self, x, out=None):
        r"""Computes the gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and stores the result into `out`.

        Args:
            x: Either a
                * vector of `float64` of size `self.n_visible` or
                * a matrix of `float64` of shape `(*, self.n_visible)`.
            out: Destination tensor of `complex128`. If `x` is a matrix then of
                `out` should be a matrix of shape `(v.shape[0], self.n_par)`.
                If `x` is a vector, then `out` should be a vector of length
                `self.n_par`.

        Returns:
            `out`
            """
        return NotImplementedError

    def to_array(self, normalize=True, b_size=512):
        if self.hilbert.is_indexable:
            all_psis = _np.zeros(self.hilbert.n_states, dtype=_np.complex128)
            batch_states = _np.zeros((b_size, self.hilbert.size))
            it = self.hilbert.states().__iter__()
            for i in range(self.hilbert.n_states // b_size + 1):
                for j in range(b_size):
                    try:
                        batch_states[j] = next(it)
                    except StopIteration:
                        batch_states.resize(j, self.hilbert.size)
                        break
                all_psis[
                    i * b_size : i * b_size + batch_states.shape[0]
                ] = self.log_val(batch_states)

                logmax = _np.max(all_psis.real)
                all_psis = _np.exp(all_psis - logmax)

            if normalize:
                norm = _np.linalg.norm(all_psis)
                all_psis = all_psis / norm

            return all_psis
        else:
            return AssertionError

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
        return _np.concatenate(tuple(p.reshape(-1) for p in self.state_dict.values()))

    @parameters.setter
    def parameters(self, p):
        if p.shape != (self.n_par,):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(
                    p.shape, self.n_par)
            )
        i = 0
        for x in map(lambda x: x.reshape(-1), self.state_dict.values()):
            _np.copyto(x, p[i: i + x.size])
            i += x.size

    def save(self, file):
        _np.save(file, self.parameters, allow_pickle=False)
