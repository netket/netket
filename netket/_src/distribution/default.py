from collections.abc import Callable

from netket.utils.types import PyTree

from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)


class DefaultDistribution(AbstractDistribution):
    r"""
    Default distribution, no transformation applied.

    This is equivalent to not changing the default distribution used by a driver, and will
    usually default to :math:`:|\psi(x)|^2` for a complex wavefunction.
    """

    def __init__(
        self,
    ):
        super().__init__(name="default")

    def __call__(self, afun: Callable, variables: PyTree):
        return afun, variables

    @property
    def q_variables(self):
        return {}

    def __repr__(self):
        return f"DefaultDistribution(name={self.name})"
