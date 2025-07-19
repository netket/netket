from collections.abc import Callable

import flax.core as fcore

from jax.scipy.special import logsumexp
from jax import numpy as jnp

from netket.jax import HashablePartial
from netket.utils import struct
from netket.utils.types import PyTree
from netket.operator import AbstractOperator, ContinuousOperator, DiscreteJaxOperator

from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)
from advanced_drivers._src.distribution.overdispersed import (
    aux_fun,
)


class OverdispersedLinearOperatorDistribution(AbstractDistribution):
    r"""
    Distribution multiplied by linear operator:
    .. math::
        |O\psi(x)|^\alpha, O \in \text{Op}(\mathcal{H})
    """

    alpha: float
    operator: DiscreteJaxOperator = struct.field(serialize=False)

    def __init__(
        self,
        operator: AbstractOperator,
        alpha: float = 2.0,
        name: str = "overdispersedOpsi",
    ):
        r"""
        Initializes the distribution
        Args:
            O: linear operator to apply
            alpha: float giving the power of $|O\psi(x)|$
            name: name of the distribution (default: "overdispersedOpsi")
        """
        name = "default" if (operator is None and alpha == 2.0) else name
        super().__init__(name=name)

        self.operator = operator
        self.alpha = jnp.asarray(alpha)

    @property
    def q_variables(self) -> PyTree:
        r"""
        Returns the specific variables used to define the distribution
        """
        return {"operator": self.operator, "alpha": self.alpha}

    def __call__(self, afun: Callable, variables: PyTree):
        operator = self.operator
        if operator is None:
            return afun, variables

        O_afun = HashablePartial(_logpsi_O_fun, afun)
        over_O_afun = HashablePartial(aux_fun, O_afun)

        new_variables = fcore.copy(variables, self.q_variables)
        return over_O_afun, new_variables

    def __repr__(self) -> str:
        return f"OverdispersedLinearOperatorDistribution(alpha={self.alpha}, operator={self.operator}, name={self.name})"


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
