from netket.hilbert.spin import Spin
from netket.sampler.rules import MetropolisRule


class GlobalSpinFlipRule(MetropolisRule):
    r"""A Metropolis rule that flips all spins in a configuration."""

    def transition(rule, sampler, machine, parameters, state, key, σ):
        if not isinstance(sampler.hilbert, Spin):
            raise TypeError(
                "GlobalSpinFlipRule is only compatible with Spin hilbert spaces"
            )
        return -σ, None
