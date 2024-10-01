import jax.numpy as jnp
import numpy as np

from netket import nn
from netket.sampler import Sampler
from netket.utils.optional_deps import import_optional_dependency

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

    def to_qobj(self):  # -> "qutip.Qobj"
        r"""Convert this mixed state to a qutip density matrix Qobj.

        Returns:
            A :class:`qutip.Qobj` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        hilbert = self.hilbert  # type: ignore
        q_dims = [
            list(2 for _ in range(hilbert.size)),
            list(2 for _ in range(hilbert.size)),
        ]
        return qutip.Qobj(np.asarray(self.to_matrix()), dims=q_dims)

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
