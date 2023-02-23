import jax.numpy as jnp

from netket import nn
from netket.sampler import Sampler

from ..mc_state import MCState


class MCPOVMState(MCState):
    def __init__(self, sampler: Sampler, model=None, **kwargs):
        sampler = sampler.replace(machine_pow=1)
        super().__init__(sampler, model, **kwargs)

    def to_array(self, normalize: bool = False) -> jnp.ndarray:
        return nn.to_array(
            self.hilbert, self._apply_fun, self.variables, normalize=normalize
        )

    def to_matrix(self, normalize: bool = True) -> jnp.ndarray:
        return nn.to_matrix_povm(
            self.hilbert, self._apply_fun, self.variables, normalize=normalize
        )

    def __repr__(self):
        return (
            "MCPOVMState("
            + "\n  hilbert = {},".format(self.hilbert)
            + "\n  sampler = {},".format(self.sampler)
            + "\n  n_samples = {},".format(self.n_samples)
            + "\n  n_discard_per_chain = {},".format(self.n_discard_per_chain)
            + "\n  sampler_state = {},".format(self.sampler_state)
            + "\n  n_parameters = {})".format(self.n_parameters)
        )
