from abc import abstractmethod

from netket.utils import struct

from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)
from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)


class AbstractDistributionConstructor(struct.Pytree):
    r"""
    Abstract distribution class used for sampling
    """

    name: str = struct.field(pytree_node=False, serialize=False)
    r"""
    associated distribution name to keep track of the chain in the sampler
    """

    def __init__(self, *, name: str):
        self.name = name

    @abstractmethod
    def construct_distribution(
        self, driver: AbstractVariationalDriver, **kwargs
    ) -> AbstractDistribution:
        """
        Constructs the distribution given the driver
        Args:
            driver: the driver to use for constructing the distribution
        Returns:
            The constructed distribution
        """
        ...
