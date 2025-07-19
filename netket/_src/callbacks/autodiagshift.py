from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import tree

from flax import core as fcore

import netket as nk
from netket.utils import struct

from netket_pro import distributed

from advanced_drivers._src.callbacks.base import AbstractCallback

import netket.jax as nkjax


@partial(
    jax.jit,
    static_argnames=("afun", "chunk_size"),
)
def _compute_discrepancies(
    lr,
    current_loss,
    linear_term,
    quadratic_term,
    afun,
    vars,
    σ,
    updated_vars,
    updated_local_loss,
    chunk_size,
):
    afun_updated_vars = lambda σ: afun(updated_vars, σ)
    afun_updated_vars = nkjax.apply_chunked(
        afun_updated_vars, in_axes=0, chunk_size=chunk_size
    )

    afun_vars = lambda σ: afun(vars, σ)
    afun_vars = nkjax.apply_chunked(afun_vars, in_axes=0, chunk_size=chunk_size)

    weights = jnp.abs(jnp.exp(afun_updated_vars(σ) - afun_vars(σ))) ** 2
    weights /= nk.stats.mean(weights)

    updated_loss = nk.stats.mean(updated_local_loss * weights).real

    M = current_loss - lr * linear_term + lr**2 / 2 * quadratic_term

    ξ = jnp.abs((updated_loss - M) / (updated_loss + M))
    ρ = (updated_loss - current_loss) / (M - current_loss)
    return M, ξ, ρ, updated_loss


def assert_learning_rate_reachable(driver):
    if not hasattr(driver._optimizer_state, "hyperparams"):
        raise ValueError(
            r"""
            Optimizer state does not have hyperparams. To change the learning rate the optimizer should be
            defined as `optimizer = optax.inject_hyperparams(optax.sgd)(learning_rate=lr)`.
            """
        )


class PI_controller_diagshift(AbstractCallback):
    target: float = struct.field(pytree_node=False, default=0.35, serialize=False)
    resample: bool = struct.field(pytree_node=False, default=True, serialize=False)
    safety_fac: float = struct.field(pytree_node=False, default=1.0, serialize=False)
    clip_min: float = struct.field(pytree_node=False, default=0.5, serialize=False)
    clip_max: float = struct.field(pytree_node=False, default=2, serialize=False)
    diag_shift_min: float = struct.field(
        pytree_node=False, default=1e-9, serialize=False
    )
    diag_shift_max: float = struct.field(
        pytree_node=False, default=1e-1, serialize=False
    )
    order: int = struct.field(pytree_node=False, default=1, serialize=False)
    beta_1: float = struct.field(pytree_node=False, default=1.0, serialize=False)
    beta_2: float = struct.field(pytree_node=False, default=0.0, serialize=False)

    _multiplier: float = struct.field(pytree_node=False, default=1, serialize=False)
    _old_mutiplier: float = struct.field(
        pytree_node=False, default=None, serialize=False
    )

    max_step_attempts: int = struct.field(
        pytree_node=False, default=10, serialize=False
    )
    reduction_bound_low: float = struct.field(
        pytree_node=False, default=0.1, serialize=False
    )
    reduction_bound_high: float = struct.field(
        pytree_node=False, default=3, serialize=False
    )

    def __init__(
        self,
        target: float = 0.35,
        resample: bool = True,
        safety_fac: float = 1.0,
        clip_min: float = 0.5,
        clip_max: float = 2,
        diag_shift_min: float = 1e-9,
        diag_shift_max: float = 1e-1,
        order: int = 1,
        beta_1: float = 1.0,
        beta_2: float = 0.0,
        max_step_attempts: int = 10,
        reduction_bound_low: float = 0.1,
        reduction_bound_high: float = 3,
    ):
        """Proportional-Integral controller for diagonal shift.

        Args:

            target: Target value for the reduction ratio ρ.
            resample: Whether to resample the samples σ over which the loss values are compared.
            safety_fac: Safety factor to apply to the computed multiplier.
            clip_max: Maximum value for the computed multiplier.
            clip_min: Minimum value for the computed multiplier.
            diag_shift_min: Minimum value for the diagonal shift.
            diag_shift_max: Maximum value for the diagonal shift.
            order: Order of the polynomial used to compute the multiplier.
            beta_1: Exponent for the PI controller on the current multiplier.
            beta_2: Exponent for the PI controller on the old multiplier.
            max_step_attempts: Maximum number of attempts to compute a valid step.
            reduction_bound_low: Lower bound for the reduction factor ρ.
            reduction_bound_high: Upper bound for the reduction factor ρ.

        Raises:
            TypeError: If the driver does not have the `collect_quadratic_model` flag.
            ValueError: If the optimizer state does not have hyperparameters.

        This callback adjusts the diagonal shift of the geometric tensor based on the reduction ratio ρ.
        """
        self.target = target
        self.resample = resample
        self.safety_fac = safety_fac
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.diag_shift_min = diag_shift_min
        self.diag_shift_max = diag_shift_max
        self.order = order
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._old_mutiplier = 1
        self._multiplier = 1

        self.max_step_attempts = max_step_attempts
        self.reduction_bound_high = reduction_bound_high
        self.reduction_bound_low = reduction_bound_low

    def on_run_start(self, step, driver, callbacks):
        # assert_learning_rate_reachable(driver)

        if not hasattr(driver, "collect_quadratic_model"):
            raise TypeError(
                "This callback requires the driver to have the `collect_quadratic_model` flag."
            )
        driver.collect_quadratic_model = True

    def on_compute_update_end(self, step, log_data, driver):
        α = driver._optimizer_state.hyperparams["learning_rate"]
        if callable(α):
            α = α(step)

        linear_term = jnp.asarray(driver.info["linear_term"])
        quadratic_term = jnp.asarray(driver.info["quadratic_term"])

        if self.resample:
            # reset the step so that new samples are used to compare the loss functions
            driver.reset_step()

        # σ are the samples used to compare the loss function over the current parameter config and on the updated one
        # the parameter update δθ is computed in the update step over the samples σ'. σ'=σ only if self.resample==False.
        afun, vars, σ, weights, extra_args = driver._prepare_derivatives()
        chunk_size = driver.state.chunk_size

        # update parameters
        model_state, params = fcore.pop(vars, "params")
        δθ = tree.map(lambda x: -α * x, driver._dp)
        updated_pars = tree.map(jnp.add, params, δθ)
        updated_vars = {"params": updated_pars, **model_state}

        # if self.resample==True compute loss on current parameters and new samples σ.
        # if self.resample==False just take as the loss on current parameters the one previously computed in the update step.
        if self.resample:
            _, current_local_loss = driver._kernel(
                afun,
                vars,
                σ,
                *extra_args,
            )
            current_local_loss *= weights
            current_loss = nk.stats.mean(current_local_loss).real
        else:
            current_loss = driver._loss_stats.mean.real

        # compute local loss on updated parameters
        _, updated_local_loss = driver._kernel(
            afun,
            updated_vars,
            σ,
            *extra_args,
        )
        updated_local_loss *= weights

        # the weights above do not account from the fact that the original distribution being considered is the
        # one with parameters θ not θ + δθ. So we need to reweight again.
        M, ξ, ρ, updated_loss = _compute_discrepancies(
            lr=α,
            current_loss=current_loss,
            linear_term=linear_term,
            quadratic_term=quadratic_term,
            afun=afun,
            vars=vars,
            σ=σ,
            updated_vars=updated_vars,
            updated_local_loss=updated_local_loss,
            chunk_size=chunk_size,
        )

        # here we scale the diagonal shift
        bare_multiplier = (self.target - ρ) * np.abs(self.target - ρ) ** self.order + 1
        clipped_multiplier = jnp.clip(
            bare_multiplier.real, self.clip_min, self.clip_max
        )
        self._multiplier = self.safety_fac * clipped_multiplier

        self._multiplier = (
            self._multiplier**self.beta_1 / self._old_mutiplier**self.beta_2
        )

        driver.diag_shift = jnp.minimum(
            jnp.maximum(driver.diag_shift * self._multiplier, self.diag_shift_min),
            self.diag_shift_max,
        )

        fails = {
            "h_is_nan": {
                "condition": not np.isfinite(current_loss),
                "message": "Loss at current step is NaN.",
            },
            "updated_h_is_nan": {
                "condition": not np.isfinite(updated_loss),
                "message": f"Predicted loss is NaN. This is likely due to the geometric tensor or the gradient. Indeed the linear term is {linear_term} and the quadratic term is {quadratic_term}.",
            },
            "rho_out_of_bounds": {
                "condition": ρ < self.reduction_bound_low
                or ρ > self.reduction_bound_high,
                "message": f"ρ is out of bounds. The current value is {ρ}. The bounds are [{self.reduction_bound_low},{self.reduction_bound_high}]. The diagonal shift is {driver.diag_shift}. The multiplier is {self._multiplier}.",
            },
        }

        epic_fails = ["h_is_nan", "updated_h_is_nan"]

        for key, err in fails.items():
            if err["condition"]:
                driver._reject_step = True

                if key in epic_fails:
                    driver.reset_step(hard=True)

                if distributed.is_master_process():
                    print(err["message"], "\n", flush=True)

        if driver._step_attempt > self.max_step_attempts:
            driver._stop_run = True

        log_data["adaptive_diagshift"] = {
            "M": M,
            "xi": ξ,
            "rho": ρ,
            "current_h": current_loss,
            "updated_h": updated_loss,
            "multiplier": self._multiplier,
            "diag_shift": driver.diag_shift,
            "lr": α,
        }

    def on_step_end(self, step, log_data, driver):
        self._old_mutiplier = self._multiplier

        # TODO: move this to another method, because we want to log this information
        # before on_legacy_log is called.
        if log_data is not None:
            log_data["info"] = driver.info

        dp = driver._dp
        log_data["constrained_norm"] = nk.jax.tree_dot(nk.jax.tree_conj(dp), dp)
        log_data["rejected_steps"] = driver._step_attempt
