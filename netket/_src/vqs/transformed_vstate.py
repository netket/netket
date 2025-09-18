import jax
import jax.numpy as jnp

from netket.sampler import MetropolisSamplerState
from netket.jax import PRNGKey
from netket.vqs import MCState, FullSumState
from netket._src.operator.hpsi_utils import make_logpsi_op_afun

# Eventually, we should have classes TransformedMCState and TransformedFullSumState that allow access to the underlying operator and model.
# In this implementation, we have access to the operator in the new variables, but not to the original model/apply_fun.


def apply_operator(operator, vstate, *, seed=None):
    """
    Apply an operator to a variational state.

    The returned variational state wraps the model of vstate inside another model
    that simulates the application of the operator. The operator can still be
    accessed in the resulting variational state as `op_vstate.variables['operator']`.

    .. note::

        Note that is the vstate's chunk size is specified, the chunk size of the transformed vstate will
        be set to `vstate.chunk_size // operator.max_conn_size` to account for the increased memory usage.

    Args:
        operator: The operator to apply in front of the variational state ket
        vstate: The variational state (or ket)
    """

    if not isinstance(vstate, (FullSumState, MCState)):
        raise TypeError("vstate must be either an MCState or a FullSumState.")

    transformed_apply_fun, new_variables = make_logpsi_op_afun(
        vstate._apply_fun, operator, vstate.variables
    )

    if vstate.chunk_size is None:
        chunk_size = None

    else:
        chunk_size = max(vstate.chunk_size // operator.max_conn_size, 1)

    if isinstance(vstate, FullSumState):
        transformed_vstate = FullSumState(
            hilbert=vstate.hilbert,
            apply_fun=transformed_apply_fun,
            variables=new_variables,
            chunk_size=chunk_size,
        )
        return transformed_vstate

    elif isinstance(vstate, MCState):
        transformed_vstate = MCState(
            sampler=vstate.sampler,
            apply_fun=transformed_apply_fun,
            variables=new_variables,
            n_samples=vstate.n_samples,
            n_discard_per_chain=vstate.n_discard_per_chain,
            chunk_size=chunk_size,
        )
        if isinstance(vstate.sampler_state, MetropolisSamplerState):
            x = vstate.sampler_state.σ
            xp, mels = operator.get_conn_padded(x)
            seed = PRNGKey(seed)
            ids = jax.random.randint(seed, (x.shape[0],), 0, operator.max_conn_size)
            new_x = xp[jnp.arange(x.shape[0]), ids, :]
            transformed_vstate.sampler_state = new_x

        transformed_vstate.sampler_state = vstate.sampler_state.replace(σ=new_x)
        return transformed_vstate
