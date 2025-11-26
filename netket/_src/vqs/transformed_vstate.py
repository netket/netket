import jax
import jax.numpy as jnp

from flax import nnx
from flax import linen

from netket.sampler import MetropolisSamplerState
from netket.jax import PRNGKey
from netket.vqs import MCState, FullSumState
from netket.operator import ContinuousOperator, DiscreteOperator
from netket.operator._prod.base import ProductOperator
from netket._src.nn.apply_operator.linen import ApplyOperatorModuleLinen
from netket._src.nn.apply_operator.nnx import ApplyOperatorModuleNNX
from netket._src.nn.apply_operator.functional import make_logpsi_op_afun


def apply_operator(operator, vstate, *, seed=None):
    """
    Apply an operator to a variational state.

    The returned variational state wraps the model of vstate with an operator transformation.
    The implementation depends on the model type:

    - **Linen modules**: Wrapped in an ApplyOperatorModuleLinen. Operator stored in flattened form with
      leaves accessible as `op_vstate.variables['operator']['leaves']` and treedef stored in
      the module as `op_vstate._model.operator_treedef`.
    - **NNX modules**: Wrapped in an ApplyOperatorModuleNNX. Operator accessible as
      `op_vstate.model.operator`.
    - **Other (functional)**: Uses the functional approach with make_logpsi_op_afun.
      Operator accessible as `op_vstate.variables['operator']`.

    .. note::

        Note that is the vstate's chunk size is specified, the chunk size of the transformed vstate will
        be set to `vstate.chunk_size // operator.max_conn_size` to account for the increased memory usage.

    .. note::

        When applying an operator to a vstate that already has an operator applied (nested application),
        the operators are automatically combined into a ProductOperator to avoid double wrapping.
        For example, B@[A@psi] will be computed as (B*A)@psi instead of B@(A@psi).

    Args:
        operator: The operator to apply in front of the variational state ket
        vstate: The variational state (or ket)
    """

    if not isinstance(vstate, (FullSumState, MCState)):
        raise TypeError("vstate must be either an MCState or a FullSumState.")

    base_module = vstate.model
    base_variables = vstate.variables

    # sometimes we apply a joint operator...
    applied_operator = operator

    if isinstance(base_module, linen.Module):
        if isinstance(vstate.model, ApplyOperatorModuleLinen):
            existing_operator = jax.tree.unflatten(
                vstate.model.operator_treedef, vstate.variables["operator"]["leaves"]
            )
            base_module = vstate.model.base_module
            base_variables = {
                collection: params.get("base_module", params)
                for collection, params in vstate.variables.items()
                if collection != "operator"
            }
            applied_operator = ProductOperator(operator, existing_operator)

        transformed_module, new_variables = (
            ApplyOperatorModuleLinen.from_module_and_variables(
                base_module, applied_operator, base_variables
            )
        )
        transformed_apply_fun = None

    elif isinstance(base_module, nnx.Module):
        if isinstance(vstate.model, ApplyOperatorModuleNNX):
            existing_operator = vstate.model.operator
            base_module = vstate.model.base_module
            applied_operator = ProductOperator(operator, existing_operator)

        transformed_module = ApplyOperatorModuleNNX(base_module, applied_operator)
        new_variables = None
        transformed_apply_fun = None

    else:
        if "operator" in base_variables:
            raise NotImplementedError(
                "Applying an operator to a variational state that already has an operator applied "
                "is not supported for functional (apply_fun) variational states. "
                "This feature is only supported for Linen and NNX modules."
            )
        transformed_apply_fun, new_variables = make_logpsi_op_afun(
            vstate._apply_fun, applied_operator, base_variables
        )
        transformed_module = None

    if vstate.chunk_size is None:
        chunk_size = None
    else:
        chunk_size = max(vstate.chunk_size // operator.max_conn_size, 1)

    if isinstance(vstate, FullSumState):
        del seed

        return FullSumState(
            hilbert=vstate.hilbert,
            model=transformed_module,
            apply_fun=transformed_apply_fun,
            variables=new_variables,
            chunk_size=chunk_size,
        )

    elif isinstance(vstate, MCState):
        # Module-based approach (Linen or NNX)
        transformed_vstate = MCState(
            sampler=vstate.sampler,
            model=transformed_module,
            apply_fun=transformed_apply_fun,
            variables=new_variables,
            n_samples=vstate.n_samples,
            n_discard_per_chain=vstate.n_discard_per_chain,
            chunk_size=chunk_size,
            sampler_seed=seed,
        )
        if isinstance(vstate.sampler_state, MetropolisSamplerState):
            if isinstance(operator, DiscreteOperator):
                # fold in a random counter so that we don't have the same seed as sampler
                # itself.
                seed = jax.random.fold_in(PRNGKey(seed), 123)

                x = vstate.sampler_state.σ
                xp, mels = operator.get_conn_padded(x)
                ids = jax.random.randint(seed, (x.shape[0],), 0, operator.max_conn_size)
                new_x = xp[jnp.arange(x.shape[0]), ids, :]
                transformed_vstate.sampler_state = (
                    transformed_vstate.sampler_state.replace(σ=new_x)
                )
            elif isinstance(operator, ContinuousOperator):
                # TODO: Implement
                pass

        return transformed_vstate
