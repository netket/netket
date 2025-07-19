from collections.abc import Callable

import jax
import jax.numpy as jnp
import flax.core as fcore

from netket.utils.types import PyTree, Array
import netket.jax as nkjax

from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)


class OverdispersedDistribution(AbstractDistribution):
    r"""
    Overdispersed distribution:
    .. math::
        |\psi(x)|^\alpha, \alpha \in [0,2]

    Can help overcome the problem of a peaked wavefunction.
    """

    alpha: float

    def __init__(self, alpha: float = 2.0, name: str = "overdispersed"):
        r"""
        Initializes the distribution
        Args:
            alpha: float giving the power of $|\psi(x)|$
        """
        self.alpha = jnp.asarray(alpha)
        super().__init__(name=name)

    @property
    def q_variables(self) -> PyTree:
        return {"alpha": self.alpha}

    def __call__(self, afun: Callable, variables: PyTree):
        new_variables = fcore.copy(variables, {"alpha": self.alpha})
        return nkjax.HashablePartial(aux_fun, afun), new_variables

    def __repr__(self):
        return f"OverdispersedDistribution(alpha={self.alpha}, name={self.name})"


class OverdispersedMixtureDistribution(AbstractDistribution):
    r"""
    Mixture of of n overdispersed distributions:
    .. math::
        1/n \sum |\psi|^\alpha_k, \alpha_k \in [0,2]
    Could help stabilize the automatic tuning.
    """

    alpha: jax.Array

    def __init__(self, n: int = 2, alpha: Array = None):
        r"""
        Initializes the distribution

        Args :
            n: number of components in the mixture
            alpha: 1d array of powers to use, defaults to jnp.linspace(0,2,n)
        """
        if alpha is None:
            alpha = jnp.linspace(0, 2, n)
        else:
            # TODO: Luca this code seems wrong: alpha does not matter?
            alpha = jnp.linspace(0, 2, n)
            raise NotImplementedError(
                "@Luca this code seems wrong: alpha does not matter?"
            )

        self.alpha = jnp.asarray(alpha)
        super().__init__(name="overdispersed_mixture")

    @property
    def q_variables(self) -> PyTree:
        return {"alpha": self.alpha}

    def __call__(self, afun: Callable, variables: PyTree):
        new_variables = fcore.copy(variables, {"alpha": self.alpha})
        return nkjax.HashablePartial(aux_fun_mixture, afun), new_variables

    def __repr__(self):
        return f"OverdispersedMixtureDistribution(alpha={self.alpha}, name={self.name})"


def aux_fun(afun, new_variables, x):
    variables, alpha = fcore.pop(new_variables, "alpha")
    return (alpha / 2) * afun(variables, x)


def aux_fun_mixture(afun, alpha, new_variables, x):
    variables, alpha = fcore.pop(new_variables, "alpha")
    log_psi = afun(variables, x)
    return (1 / 2) * jnp.log(
        jnp.mean((jnp.exp(jnp.real(log_psi))[:, None]) ** alpha, axis=1)
    )
