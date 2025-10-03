from netket.sampler.rules import MetropolisRule

class GlobalFlipRule(MetropolisRule):
    r""" A Metropolis rule that flips all spins in a configuration."""
    def transition(rule, sampler, machine, parameters, state, key, σ):
        return -σ, None
