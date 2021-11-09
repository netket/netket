from typing import Optional, Callable

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import AbstractOperator


class ContinousOperator(AbstractOperator):
    r"""This class is the abstract base class for operators defined on a
    continuous Hilbert space. Users interested in implementing new
    quantum Operators for continuous Hilbert spaces should subclass
    `ContinuousOperator` and implement its interface.
    """

    def __init__(self, hilbert: AbstractHilbert, dtype: Optional[DType] = float):
        r"""
        Constructs the continuous operator acting on the given hilbert space and
        with a certain data type.

        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        self._dtype = dtype
        super().__init__(hilbert)

    @property
    def dtype(self) -> DType:
        return self._dtype

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        r"""This method defines the action of the local operator on a given quantum state
        `logpsi` for a given congfiguration `x`.
        :math:`O_{loc}(x) =  \frac{\bra{x}O{\ket{\psi}}{\bra{x}\ket{\psi}}`
        This method is executed inside of a `jax.jit` block.
        Any static data from the operator itself should be captured in the method.
        Any array should be passed through the `_pack_arguments` method in order to be
        traced by jax, and will be passed as the `data` argument.

        Args:
            logpsi: variational state
            params: parameters for the variational state
            x: a sample of particle positions
            data: additional data
        """
        raise NotImplementedError

    def _pack_arguments(self) -> Optional[PyTree]:
        r"""This methods should return a PyTree that will be passed as the `data` argument
        to the `_expect_kernel`. The PyTree should be composed of jax arrays or hashable
        objects.

        For example for the kinetic energy this method would return the masses of the
        individual particles."""
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, ContinousOperator):
            from netket.operator import SumOperator

            return SumOperator(self, other)
        else:
            return NotImplementedError

    def __rmul__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, float):
            from netket.operator import SumOperator

            return self * other
        else:
            return NotImplementedError

    def __mul__(self, other):
        if isinstance(self, ContinousOperator) and isinstance(other, float):
            from netket.operator import SumOperator

            return SumOperator(self, coefficients=other)
        else:
            return NotImplementedError
