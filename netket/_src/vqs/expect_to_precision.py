""" """

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from netket.vqs.mc import MCState
from netket.sampler import MetropolisSampler
from netket.operator import AbstractOperator
from netket._src.stats.online_stats import online_statistics
from netket._src.stats.online_stats.accumulator_batch import OnlineStatsBatch


def _summary_error_and_scale(stats) -> tuple[float, float]:
    """Return scalar error and scale summaries for scalar or batched observables."""
    s = stats.get_stats()
    if isinstance(stats, OnlineStatsBatch):
        err = float(jnp.max(jnp.abs(s.error_of_mean)))
        scale = float(jnp.max(jnp.abs(s.mean)))
    else:
        err = abs(float(s.error_of_mean))
        scale = abs(float(s.mean))
    return err, scale


def _check_not_converged(stats, atol: float | None, rtol: float | None) -> bool:
    """Return True if the stats have not yet met the requested tolerances."""
    err, scale = _summary_error_and_scale(stats)
    if atol is not None and err > atol:
        return True
    if rtol is not None and err / scale > rtol:
        return True
    return False


def _format_postfix(stats, atol: float | None, rtol: float | None) -> dict:
    """Build the tqdm postfix dict from current stats and tolerances."""
    err, scale = _summary_error_and_scale(stats)
    d = {"err": f"{err:.4g}"}
    if atol is not None:
        d["atol"] = f"{atol:.4g}"
    if rtol is not None:
        d["rel_err"] = f"{err / scale:.4g}"
        d["rtol"] = f"{rtol:.4g}"
    return d


def _accumulate_stats(state, op_leaves, active, old_stats, *, max_lag):
    active = set(active)
    result = []
    for i, (op, old) in enumerate(zip(op_leaves, old_stats)):
        if i not in active:
            result.append(old)
            continue
        le = state.local_estimators(op)
        jax.block_until_ready(le)
        result.append(online_statistics(le, old_estimator=old, max_lag=max_lag))
    return result


def expect_to_precision(
    state: MCState,
    op: AbstractOperator,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    max_iter: int = 10_000,
    max_lag: int = 64,
    verbose: bool = True,
):
    """
    Sample until the estimated standard error of the mean meets the requested
    tolerance(s).

    This uses NetKet's online_statistics to update estimates incrementally.

    At least one of ``atol`` or ``rtol`` must be specified. If both are given,
    sampling continues until *both* tolerances are satisfied simultaneously.

    Args:
        state: The MC state to sample from.
        op: The operator whose expectation value is estimated.
        atol: Desired absolute standard error of the mean. Sampling stops when
            ``error_of_mean <= atol``.
        rtol: Desired relative standard error of the mean. Sampling stops when
            ``error_of_mean / |mean| <= rtol``.
        max_iter: Maximum number of sampling iterations.
        max_lag: Max lag used for autocorrelation estimation.
        verbose: Whether to show a progress bar.
    """
    if atol is None and rtol is None:
        raise ValueError("At least one of 'atol' or 'rtol' must be specified.")
    if atol is not None and atol <= 0:
        raise ValueError("atol must be > 0.")
    if rtol is not None and rtol <= 0:
        raise ValueError("rtol must be > 0.")

    if not isinstance(state.sampler, MetropolisSampler):
        raise ValueError("Only works with MetropolisSampler.")

    _is_rank0 = jax.process_index() == 0

    # Flatten the operator pytree once; all loop internals work on plain lists.
    op_leaves, treedef = jax.tree.flatten(
        op, is_leaf=lambda x: isinstance(x, AbstractOperator)
    )

    state.sample()
    stats_list = _accumulate_stats(
        state,
        op_leaves,
        range(len(op_leaves)),
        [None] * len(op_leaves),
        max_lag=max_lag,
    )
    active = [
        i
        for i in range(len(op_leaves))
        if _check_not_converged(stats_list[i], atol, rtol)
    ]

    it = 0
    with tqdm(
        total=max_iter,
        desc="Sampling",
        unit="iter",
        disable=not verbose or not _is_rank0,
    ) as pbar:
        pbar.set_postfix(
            _format_postfix(stats_list[active[0] if active else 0], atol, rtol)
        )
        try:
            while active and it < max_iter:
                state.sample(n_discard_per_chain=0)
                stats_list = _accumulate_stats(
                    state, op_leaves, active, stats_list, max_lag=max_lag
                )
                active = [
                    i for i in active if _check_not_converged(stats_list[i], atol, rtol)
                ]

                pbar.set_postfix(
                    _format_postfix(stats_list[active[0] if active else 0], atol, rtol)
                )
                pbar.update(1)
                it += 1
        except KeyboardInterrupt:
            if _is_rank0:
                pbar.write("  Early termination requested by user.")

        # Compose messages on all ranks; only rank 0 writes.
        if it >= max_iter:
            msg = "  Reached max_iter before target precision."
            if _is_rank0:
                pbar.write(msg)
        err, _ = _summary_error_and_scale(stats_list[0])
        msg = f"  [done] error = {err:g}"
        if _is_rank0:
            pbar.write(msg)

    return treedef.unflatten(stats_list)
