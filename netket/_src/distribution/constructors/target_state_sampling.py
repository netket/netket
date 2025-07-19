from typing import Literal

from netket.utils import struct

from advanced_drivers._src.distribution.constructors.base import (
    AbstractDistributionConstructor,
)
from advanced_drivers._src.driver.ngd.driver_infidelity_ngd import InfidelityOptimizerNG

from advanced_drivers._src.distribution.overdispersed import OverdispersedDistribution
from advanced_drivers._src.distribution.overdisperded_linear_operator import (
    OverdispersedLinearOperatorDistribution,
)
from advanced_drivers._src.distribution.default import DefaultDistribution


class OverdispersedTargetStateDistribution(AbstractDistributionConstructor):
    r"""
    Abstract distribution class used for sampling
    """

    name: str = struct.field(pytree_node=False, serialize=False)
    r"""
    associated distribution name to keep track of the chain in the sampler
    """
    alpha: float = struct.field(pytree_node=False, serialize=False)
    r"""
    The overdispersed target state distribution (default to alpha=2.0)
    """
    sample_transformed: bool = struct.field(
        pytree_node=False, serialize=False, default=True
    )
    r"""
    If True, the distribution will sample the linearly transformed state
    like Hpsi(x) instead of the original state.

    This is the default when using importance sampling, and it more expensive by a factor
    of number of connected entries.
    """

    def __init__(
        self, alpha=None, sample_transformed: bool = True, *, name: str = None
    ):
        self.name = name
        self.alpha = alpha
        self.sample_transformed = sample_transformed

    def construct_distribution(
        self,
        driver: InfidelityOptimizerNG,
        *,
        which: Literal["state"] | Literal["target"],
    ) -> OverdispersedLinearOperatorDistribution:
        r"""
        Constructs the distribution given the driver
        Args:
            driver: the driver to use for constructing the distribution
        Returns:
            The constructed distribution
        """
        if not isinstance(driver, InfidelityOptimizerNG):
            raise TypeError(
                "The OverdispersedTargetStateDistribution only works with InfidelityOptimizerNG driver"
            )

        kwargs = {}
        if self.name is not None:
            kwargs["name"] = self.name

        if self.sample_transformed:
            operator = driver.U_target if which == "target" else driver.V_state

            return OverdispersedLinearOperatorDistribution(
                alpha=self.alpha,
                operator=operator,
                **kwargs,
            )
        else:
            if self.alpha is None:
                return DefaultDistribution()
            else:
                return OverdispersedDistribution(
                    alpha=self.alpha,
                    **kwargs,
                )
