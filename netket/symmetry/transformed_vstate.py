import jax
import flax

from netket import jax as nkjax
from netket.operator import ContinuousOperator
from netket.vqs import MCState, FullSumState


def apply_operator(operator, vstate):

    if isinstance(vstate, FullSumState):
        transformed_apply_fun, new_variables = make_logpsi_U_afun(
            vstate._apply_fun, operator, vstate.variables
        )
        transformed_vstate = FullSumState(
            hilbert=vstate.hilbert,
            apply_fun=transformed_apply_fun,
            variables=new_variables,
            chunk_size=vstate.chunk_size,
        )
        return transformed_vstate

    elif isinstance(vstate, MCState):
        transformed_apply_fun, new_variables = make_logpsi_U_afun(
            vstate._apply_fun, operator, vstate.variables
        )
        transformed_vstate = MCState(
            sampler=vstate.sampler,
            apply_fun=transformed_apply_fun,
            variables=new_variables,
            n_samples=vstate.n_samples,
            n_samples_per_rank=vstate.n_samples_per_rank,
            n_discard_per_chain=vstate.n_discard_per_chain,
            chunk_size=vstate.chunk_size,
        )
        return transformed_vstate

    else:
        raise TypeError("vstate must be either an MCState or a FullSumState.")


def make_logpsi_U_afun(logpsi_fun, U, variables):
    """Wraps an apply_fun into another one that multiplies it by an Unitary transformation U.

    This wrapper is made such that the Unitary is passed as the model_state
    of the new wrapped function, and therefore changes to the angles/coefficients
    of the Unitary should not trigger recompilation.

    Args:
        logpsi_fun: a function that takes as input variables and samples
        U: a {class}`nk.operator.JaxDiscreteOperator`
        variables: The variables used to call *logpsi_fun*

    Returns:
        A tuple, where the first element is a new function with the same signature as
        the original **logpsi_fun** and a set of new variables to be used to call it.

    """
    # wrap apply_fun into logpsi logpsi_U
    logpsiU_fun = nkjax.HashablePartial(_logpsi_U_fun, logpsi_fun)

    # Insert a new 'model_state' key to store the Unitary. This only works
    # if U is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(variables, {"unitary": U})

    return logpsiU_fun, new_variables


def _logpsi_U_fun(apply_fun, variables, x, *args):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `unitary` with
    a jax-compatible operator.
    """
    variables_applyfun, U = flax.core.pop(variables, "unitary")

    if isinstance(U, ContinuousOperator):
        res = U._expect_kernel(apply_fun, variables_applyfun, x)
    else:
        xp, mels = U.get_conn_padded(x)
        xp = xp.reshape(-1, x.shape[-1])
        logpsi_xp = apply_fun(variables_applyfun, xp, *args)
        logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

        res = jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)
    return res


def _lazy_apply_UV_to_afun(vstate, operator, extra_hash_data=None):
    ψ_logfun = vstate._apply_fun
    ψ_vars = vstate.variables

    if operator is None:
        return ψ_logfun, ψ_vars, vstate.model
    else:
        Uψ_logfun, Uψ_vars = make_logpsi_U_afun(ψ_logfun, operator, ψ_vars)

    # fix the has to include forward or backward info
    # we use this to give a different hash to 'forward' and 'backward'
    # distributions, even if they are identical, and only differ in the
    # parameters.
    if extra_hash_data is not None:
        Uψ_logfun.__hash__()
        Uψ_logfun._hash = hash((Uψ_logfun._hash, extra_hash_data))

    return Uψ_logfun, Uψ_vars, Uψ_logfun
