from collections.abc import Callable

import flax.core as fcore

from jax.scipy.special import logsumexp

from netket.jax import HashablePartial
from netket.utils import struct, warn_deprecation
from netket.utils.types import PyTree
from netket.operator import AbstractOperator, ContinuousOperator, DiscreteJaxOperator

from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)


class LinearOperatorStateDistribution(AbstractDistribution):
    r"""
    Distribution obtained by applying a linear operator to the state.

    .. math::
        |O\psi(x)|^\alpha, O \in \text{Op}(\mathcal{H})

    """

    operator: DiscreteJaxOperator = struct.field(serialize=False)

    def __init__(self, operator: AbstractOperator, name: str = "Opsi", *, O=None):
        r"""
        Initializes the distribution
        Args:
            O: Linear operator to apply
            name: Name of the distribution (default: "Opsi")
        """
        if O is not None:
            warn_deprecation("O is a deprecated keyword argument, use operator instead")
            operator = O

        name = "default" if operator is None else name
        self.operator = operator
        super().__init__(name=name)

    @property
    def q_variables(self) -> PyTree:
        r"""
        Returns the specific variables used to define the distribution
        """
        return {"operator": self.operator}

    def __call__(self, afun: Callable, variables: PyTree):
        operator = self.operator
        if operator is None:
            return afun, variables

        new_variables = fcore.copy(variables, {"operator": operator})
        return HashablePartial(_logpsi_O_fun, afun), new_variables

    def __repr__(self):
        return f"LinearOperatorStateDistribution(operator={self.operator}, name={self.name})"


def _logpsi_O_fun(afun, new_variables, x, *args):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `operator` with
    a jax-compatible operator.
    """
    variables, O = fcore.pop(new_variables, "operator")

    if isinstance(O, ContinuousOperator):
        res = O._expect_kernel(afun, variables, x)
    else:
        xp, mels = O.get_conn_padded(x)
        xp = xp.reshape(-1, x.shape[-1])
        logpsi_xp = afun(variables, xp, *args)
        logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

        res = logsumexp(logpsi_xp, axis=-1, b=mels)
    return res
