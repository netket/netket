from typing import Tuple
from collections.abc import Callable
from abc import abstractmethod

from netket.utils import struct
from netket.utils.types import PyTree, Array


class AbstractDistribution(struct.Pytree):
    r"""
    Abstract distribution class used for sampling
    """

    name: str = struct.field(pytree_node=False, serialize=True)
    r"""
    associated distribution name to keep track of the chain in the sampler
    """

    def __init__(self, *, name: str):
        self.name = name

    @property
    @abstractmethod
    def q_variables(self) -> PyTree:
        """
        The specific variables used to define the distribution.
        """

    @abstractmethod
    def __call__(
        self, afun: Callable[[PyTree, Array], Array], variables: PyTree
    ) -> Tuple[Callable, PyTree]:
        """
        Returns the transformed pdf given then underlying state to sample from

            Args:
                afun: the log-amplitude function of the state
                variables: the variables of the state
            Returns:
                A tuple containing the log_amplitude function of the modified distribution and the appropriately modified variables
        """
        raise NotImplementedError

    def __mul__(self, other: "AbstractDistribution") -> "DistributionChain":
        """
        Returns a new distribution that is the product of this distribution and another one.
        """
        if not isinstance(other, AbstractDistribution):
            raise TypeError(
                f"Cannot multiply {type(self)} with {type(other)}. Both must be AbstractDistribution."
            )
        if isinstance(other, DistributionChain):
            return DistributionChain(self, *other.distributions)
        else:
            return DistributionChain(self, other)


class DistributionChain(AbstractDistribution):
    r"""
    A chain of distributions, composed in order
    """

    distributions: Tuple[AbstractDistribution]

    def __init__(self, *distributions: AbstractDistribution):
        self.distributions = tuple(distributions)
        names = ",".join([d.name for d in distributions])
        super().__init__(name=f"chain[{names}]")

    @property
    def q_variables(self) -> PyTree:
        """
        The specific variables used to define the distribution.
        """
        return (d.q_variables for d in self.distributions)

    def __call__(self, afun: Callable, variables: PyTree):
        for d in reversed(self.distributions):
            afun, variables = d(afun, variables)
        return afun, variables

    def __repr__(self):
        strrep = [
            "DistributionChain [first applying last, last applying the first]",
            " output",
        ]
        N_distr = len(self.distributions)

        arrow = [" ↑", " │", " │"]  # Adjust for as many lines as needed
        for i, distr in enumerate(self.distributions):
            count = N_distr - i - 1
            # Add the arrow only for the first distribution line
            if i == 0:
                _str = f"  {arrow[0]}{count} : {distr}"
            else:
                _str = f"  {arrow[min(i, len(arrow)-1)]}{count} : {distr}"
            strrep.append(_str)
        strrep.append(" input")
        return "\n".join(strrep)

    def __mul__(self, other: "AbstractDistribution") -> "DistributionChain":
        """
        Returns a new distribution that is the product of this distribution and another one.
        """
        if not isinstance(other, AbstractDistribution):
            raise TypeError(
                f"Cannot multiply {type(self)} with {type(other)}. Both must be AbstractDistribution."
            )
        if isinstance(other, DistributionChain):
            return DistributionChain(*self.distributions, *other.distributions)
        else:
            return DistributionChain(*self.distributions, other)
