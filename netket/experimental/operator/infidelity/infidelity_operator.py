from typing import Optional

import jax.numpy as jnp

from netket.experimental.observable import AbstractObservable

from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import VariationalState, FullSumState


class InfidelityOperator(AbstractObservable):
    """
    Infidelity operator corresponding to the projector onto the target state.
    """

    def __init__(
        self,
        target_state: VariationalState,
        *,
        cv_coeff: Optional[float] = None,
        dtype: Optional[DType] = None,
    ):
        super().__init__(target_state.hilbert)

        if not isinstance(target_state, VariationalState):
            raise TypeError("The first argument should be a variational target.")

        if cv_coeff is not None:
            cv_coeff = jnp.array(cv_coeff)

            if (not is_scalar(cv_coeff)) or jnp.iscomplex(cv_coeff):
                raise TypeError("`cv_coeff` should be a real scalar number or None.")

            if isinstance(target_state, FullSumState):
                cv_coeff = None

        self._target_state = target_state
        self._cv_coeff = cv_coeff
        self._dtype = dtype

    @property
    def target_state(self):
        return self._target_state

    @property
    def cv_coeff(self):
        return self._cv_coeff

    @property
    def dtype(self):
        return self._dtype

    @property
    def is_hermitian(self):
        return True

    def __repr__(self):
        return f"InfidelityOperator(target_state={self.target_state}, cv_coeff={self.cv_coeff})"
