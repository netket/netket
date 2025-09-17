import jax
import jax.numpy as jnp

from netket.sampler import MetropolisSamplerState
from netket.jax import PRNGKey
from netket.vqs import MCState, FullSumState
from netket._src.operator.hpsi_utils import make_logpsi_op_afun

# Eventually, we should have classes TransformedMCState and TransformedFullSumState that allow access to the underlying operator and model.
# In this implementation, we have access to the operator in the new variables, but not to the original model/apply_fun.

def chunk_size_divisor(chunk_size: int, n_samples_per_rank: int):
    """Returns the largest divisor of n_samples_per_rank that is <= chunk_size."""
    divisors = [i for i in range(1, n_samples_per_rank + 1) if (n_samples_per_rank % i == 0 and i <= chunk_size)]
    return max(divisors) 


def apply_operator(operator, vstate, *, seed=None, adapt_chunk_size: bool=True):
    """
    Apply an operator to a variational state.

    The returned variational state wraps the model of vstate inside another model
    that simulates the application of the operator. The operator can still be
    accessed in the resulting variational state as `op_vstate.variables['operator']`.

    Args:
        operator: The operator to apply.
        vstate: The variational state.
        adapt_chunk_size: Whether to adapt the chunk size of the new state. Since the number of calls to the 
            model is multiplied by operator.max_conn_size, the chunk size is divided by operator.max_conn_size.
            Then, it needs to be adjusted to be a divisor of the number of samples per rank.
    """

    if not isinstance(vstate, (FullSumState, MCState)):
        raise TypeError("vstate must be either an MCState or a FullSumState.")

    transformed_apply_fun, new_variables = make_logpsi_op_afun(
        vstate._apply_fun, operator, vstate.variables
    )

    if vstate.chunk_size is None: 
        chunk_size = None

    if adapt_chunk_size and vstate.chunk_size is not None:

        chunk_size = vstate.chunk_size // operator.max_conn_size
        if isinstance(vstate, MCState):
            chunk_size = chunk_size_divisor(chunk_size, vstate.n_samples_per_rank)

    else: 
        chunk_size = vstate.chunk_size
        

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
