from typing import Callable, Optional, Union
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import netket.jax as nkjax
from netket.stats import mean as distributed_mean, sum as distributed_sum
from netket.utils import timing
from netket.utils.types import Array, PyTree

from advanced_drivers._src.driver.ngd.sr import _compute_sr_update
from advanced_drivers._src.driver.ngd.srt import _compute_srt_update


def _multiply_by_pdf(oks, pdf):
    """
    Computes  O'ⱼ̨ₖ = Oⱼₖ pⱼ .
    Used to multiply the log-derivatives by the probability density.
    """

    return jax.tree_util.tree_map(
        lambda x: jax.lax.broadcast_in_dim(pdf, x.shape, (0,)) * x,
        oks,
    )


@partial(jax.jit, static_argnames=("mode",))
def _prepare_input(
    O_L,
    local_grad,
    *,
    mode: str,
    pdf: Optional[Array] = None,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Prepare the input for the SR/SRt solvers.

    The local eneriges and the jacobian are reshaped, centered and normalized by the number of Monte Carlo samples.
    The complex case is handled by concatenating the real and imaginary parts of the jacobian and the local energies.

    We use [Re_x1, Im_x1, Re_x2, Im_x2, ...] so that shards are contiguous, and jax can keep track of the sharding information.
    This format is applied both to the jacobian and to the vector.

    Args:
        O_L: The jacobian of the ansatz, not centered and not multiplied by the pdf coefficients.
        local_grad: The local energies.
        pdf: The weight of every sample, potentially used for importance sampling or for fullsummation.
        mode: The mode of the jacobian: `'real'` or `'complex'`.

    Returns:
        The reshaped jacobian and the reshaped local energies.
    """
    # jacobian and local_grad are centered accounting for reweighting
    local_grad = local_grad.flatten()

    if pdf is None:
        N_mc = O_L.shape[0]
        sqrt_N_mc = jnp.array(math.sqrt(N_mc))
        # Center the jacobian and multiply by the pdf coefficients.
        # equiv to `center = True``
        O_L = O_L - distributed_mean(O_L, axis=0, keepdims=True)
        # equiv to `_sqrt_rescale = True`
        O_L = O_L / sqrt_N_mc

        de = local_grad - distributed_mean(local_grad)
        dv = 2.0 * de / sqrt_N_mc
    else:
        # Center the jacobian and multiply by the pdf coefficients.
        # equiv to `center = True``
        O_L = O_L - distributed_sum(_multiply_by_pdf(O_L, pdf), axis=0, keepdims=True)
        # equiv to `_sqrt_rescale = True`
        O_L = _multiply_by_pdf(O_L, jnp.sqrt(pdf))

        de = local_grad - (local_grad * pdf).sum()
        dv = 2.0 * de * jnp.sqrt(pdf)

    if mode == "complex":
        # Concatenate the real and imaginary derivatives of the ansatz
        # (#ns, 2, np) -> (#ns*2, np)
        O_L = jax.lax.collapse(O_L, 0, 2)

        # (#ns, 2) -> (#ns*2)
        dv2 = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)
        dv = jax.lax.collapse(dv2, 0, 2)
    elif mode == "real":
        dv = dv.real
    else:
        raise NotImplementedError()
    return O_L, dv


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "mode",
        "chunk_size",
        "collect_quadratic_model",
        "collect_gradient_statistics",
        "use_ntk",
    ),
)
def _sr_srt_common(
    log_psi,
    local_grad,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[PyTree] = None,
    chunk_size: Optional[int] = None,
    collect_quadratic_model: bool = False,
    collect_gradient_statistics: bool = False,
    use_ntk: bool = False,
    # pdf and importance_sampling_weights are very similar, as there is only a 1/N_samples scaling difference.
    # Internally we normalize to only using pdf, but externally it's handy to keep both.
    pdf: Optional[Array] = None,
    importance_sampling_weights: Optional[Array] = None,
):
    r"""
    Compute the Natural gradient update for the model specified by `log_psi({parameters, model_state}, samples)`
    and the local gradient contributions `local_grad`.

    Uses a code equivalent to QGTJacobianDense by default, or with the NTK/MinSR if `use_ntk` is True.

    Args:
        log_psi: The log of the wavefunction.
        local_grad: The local values of the estimator.
        parameters: The parameters of the model.
        model_state: The state of the model.
        samples: The samples used to compute expectation values.
        diag_shift: The diagonal shift of the stochastic reconfiguration matrix. Typical values are 1e-4 ÷ 1e-3. Can also be an optax schedule.
        proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
        momentum: Momentum used to accumulate updates in SPRING.
        linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
        mode: The mode used to compute the jacobian of the variational state. Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).
        collect_quadratic_model: Whether to collect the quadratic model. The quantities collected are the linear and quadratic term in the approximation of the loss function. They are stored in the info dictionary of the driver.
        collect_gradient_statistics: Whether to collect the statistics (mean and variance) of the gradient. They are stored in the info dictionary of the driver.
        pdf: The probability density of the samples. Used to multiply the jacobian by the pdf coefficients.
        importance_sampling_weights: The weights potentially used for importance sampling. They cannot be specified together with
            `pdf`. They have effectively the same role, where `pdf = importance_sampling_weights/samples.shape[0]`

    Returns:
        The new parameters, the old updates, and the info dictionary.
    """
    if pdf is not None and importance_sampling_weights is not None:
        raise ValueError("cannot specify both pdf and importance_sampling_weights")
    elif importance_sampling_weights is not None:
        if not importance_sampling_weights.shape == (samples.shape[0],):
            raise ValueError(
                f"importance_sampling_weights must have shape ({samples.shape[0]},) but got {importance_sampling_weights.shape}"
            )
        pdf = importance_sampling_weights / samples.shape[0]

    _, unravel_params_fn = ravel_pytree(parameters)
    _params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), parameters
    )

    jacobians = nkjax.jacobian(
        log_psi,
        parameters,
        samples,
        model_state,
        mode=mode,
        dense=True,
        center=False,
        chunk_size=chunk_size,
    )  # jacobian is NOT centered

    O_L, dv = _prepare_input(jacobians, local_grad, mode=mode, pdf=pdf)

    if old_updates is None and momentum is not None:
        old_updates = jnp.zeros(jacobians.shape[-1], dtype=jacobians.dtype)

    compute_update = _compute_srt_update if use_ntk else _compute_sr_update

    # TODO: Add support for proj_reg and momentum
    # At the moment SR does not support momentum, proj_reg.
    # We raise an error if they are passed with a value different from None.
    updates, old_updates, info = compute_update(
        O_L,
        dv,
        diag_shift=diag_shift,
        solver_fn=solver_fn,
        mode=mode,
        proj_reg=proj_reg,
        momentum=momentum,
        old_updates=old_updates,
        collect_quadratic_model=collect_quadratic_model,
        collect_gradient_statistics=collect_gradient_statistics,
        params_structure=_params_structure,
    )

    return unravel_params_fn(updates), old_updates, info


sr = partial(_sr_srt_common, use_ntk=False)

srt = partial(_sr_srt_common, use_ntk=True)
