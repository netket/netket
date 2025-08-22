import jax
import flax

from netket import jax as nkjax
from netket.operator import ContinuousOperator
from netket.vqs import MCState, FullSumState

# Eventually, we should have classes TransformedMCState and TransformedFullSumState that allow access to the underlying operator and model.
# In this implementation, we have access to the operator in the new variables, but not to the original model/apply_fun.


def apply_operator(operator, vstate):

    if not isinstance(vstate, (FullSumState, MCState)):
        raise TypeError("vstate must be either an MCState or a FullSumState.")

    transformed_apply_fun, new_variables = make_logpsi_op_afun(
        vstate._apply_fun, operator, vstate.variables
    )
    chunk_size = (
        None
        if vstate.chunk_size is None
        else vstate.chunk_size / operator.max_conn_size
    )

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
        return transformed_vstate


def make_logpsi_op_afun(logpsi_fun, operator, variables):
    """Wraps an apply_fun into another one that multiplies it by an operator.

    This wrapper is made such that the operator is passed as the model_state
    of the new wrapped function, and therefore changes to the angles/coefficients
    of the operator should not trigger recompilation.

    Args:
        logpsi_fun: a function that takes as input variables and samples
        oeprator: a {class}`nk.operator.JaxDiscreteOperator`
        variables: The variables used to call *logpsi_fun*

    Returns:
        A tuple, where the first element is a new function with the same signature as
        the original **logpsi_fun** and a set of new variables to be used to call it.

    """
    # Wrap logpsi into logpsi_op
    logpsi_op_fun = nkjax.HashablePartial(_logpsi_op_fun, logpsi_fun)

    # Insert a new 'operator' key to store the operator. This only works
    # if operator is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(variables, {"operator": operator})

    return logpsi_op_fun, new_variables


def _logpsi_op_fun(apply_fun, variables, x, *args, **kwargs):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `operator` with
    a jax-compatible operator.
    """
    variables_applyfun, operator = flax.core.pop(variables, "operator")

    if isinstance(operator, ContinuousOperator):
        res = operator._expect_kernel(apply_fun, variables_applyfun, x)
    else:
        xp, mels = operator.get_conn_padded(x)
        xp = xp.reshape(-1, x.shape[-1])
        logpsi_xp = apply_fun(variables_applyfun, xp, *args, **kwargs)
        logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

        res = jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)
    return res


# Is it useful?
def _lazy_apply_UV_to_afun(vstate, operator, extra_hash_data=None):
    psi_logfun = vstate._apply_fun
    psi_vars = vstate.variables

    if operator is None:
        return psi_logfun, psi_vars, vstate.model
    else:
        Upsi_logfun, Upsi_vars = make_logpsi_op_afun(psi_logfun, operator, psi_vars)

    # fix the has to include forward or backward info
    # we use this to give a different hash to 'forward' and 'backward'
    # distributions, even if they are identical, and only differ in the
    # parameters.
    if extra_hash_data is not None:
        Upsi_logfun.__hash__()
        Upsi_logfun._hash = hash((Upsi_logfun._hash, extra_hash_data))

    return Upsi_logfun, Upsi_vars, Upsi_logfun
