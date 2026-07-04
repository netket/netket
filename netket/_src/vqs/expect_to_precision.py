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
        err = abs(float(jnp.real(s.error_of_mean)))
        scale = abs(float(jnp.real(s.mean)))
    return err, scale


def _tolerance(scale: float, atol: float | None, rtol: float | None) -> float:
    """Combined tolerance ``atol + rtol * scale``, NumPy-style.

    A missing tolerance is treated as 0, so ``atol`` acts as an absolute
    floor when the mean is close to zero.
    """
    return (atol or 0.0) + (rtol or 0.0) * scale


def _check_not_converged(stats, atol: float | None, rtol: float | None) -> bool:
    """Return True if the stats have not yet met the requested tolerance."""
    err, scale = _summary_error_and_scale(stats)
    return err > _tolerance(scale, atol, rtol)


def _broadcast_tol(tol, treedef, name: str) -> list:
    """Broadcast a scalar tolerance to all operator leaves, or validate a
    pytree of per-leaf tolerances against the operator structure.

    ``None`` is a valid leaf value, meaning "no constraint of this kind on
    this leaf".
    """
    leaves, tol_treedef = jax.tree.flatten(tol, is_leaf=lambda x: x is None)
    if tol_treedef == jax.tree.structure(0):
        # scalar (or global None): same tolerance for every operator leaf
        leaves = leaves * treedef.num_leaves
    elif tol_treedef != treedef:
        raise ValueError(
            f"'{name}' must be a scalar or a pytree matching the operator "
            f"structure {treedef}, but got {tol_treedef}."
        )
    for leaf in leaves:
        if leaf is not None and leaf <= 0:
            raise ValueError(f"'{name}' entries must be > 0 (got {leaf}).")
    return leaves


def _format_postfix(stats_list, active, atol_leaves, rtol_leaves) -> dict:
    """Build the tqdm postfix dict: convergence count (for multiple
    operators) and the err/tol of the worst non-converged leaf."""
    d = {}
    if len(stats_list) > 1:
        d["converged"] = f"{len(stats_list) - len(active)}/{len(stats_list)}"
    worst_ratio, worst_err, worst_tol = -1.0, 0.0, 0.0
    for i in active if active else range(len(stats_list)):
        err, scale = _summary_error_and_scale(stats_list[i])
        tol = _tolerance(scale, atol_leaves[i], rtol_leaves[i])
        ratio = err / tol if tol > 0.0 else float("inf")
        if ratio > worst_ratio:
            worst_ratio, worst_err, worst_tol = ratio, err, tol
    d["err"] = f"{worst_err:.4g}"
    d["tol"] = f"{worst_tol:.4g}"
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
    atol=None,
    rtol=None,
    max_iter: int = 10_000,
    max_lag: int = 64,
    verbose: bool = True,
):
    """
    Sample until the estimated standard error of the mean meets the requested
    tolerance(s).

    This uses NetKet's online_statistics to update estimates incrementally.

    At least one of ``atol`` or ``rtol`` must be specified. Sampling stops when

    .. code-block:: python

        error_of_mean <= atol + rtol * |mean|

    with a missing tolerance treated as 0 (NumPy convention): with only
    ``atol`` the criterion is absolute, with only ``rtol`` it is relative,
    and with both, ``atol`` acts as an absolute floor that keeps the
    relative criterion well-behaved when the mean is close to zero.

    When ``op`` is a pytree of operators, ``atol`` and ``rtol`` may either be
    scalars (applied to every operator) or pytrees with the same structure as
    ``op``, giving a per-operator tolerance.  A ``None`` entry means "no
    constraint of this kind for this operator"; every operator must have at
    least one non-``None`` tolerance.  Each operator stops being sampled as
    soon as its own criterion is met.

    .. code-block:: python

        ops = {"energy": H, "mag": M}
        stats = expect_to_precision(
            vs, ops,
            rtol={"energy": 0.001, "mag": 0.05},
            atol=1e-6,  # global absolute floor
        )
        stats["energy"].get_stats()

    Args:
        state: The MC state to sample from.
        op: The operator (or pytree of operators) whose expectation value is
            estimated.
        atol: Desired absolute standard error of the mean. A scalar, or a
            pytree matching the structure of ``op``.
        rtol: Desired relative standard error of the mean. A scalar, or a
            pytree matching the structure of ``op``.
        max_iter: Maximum number of sampling iterations.
        max_lag: Max lag used for autocorrelation estimation.
        verbose: Whether to show a progress bar.
    """
    if not isinstance(state.sampler, MetropolisSampler):
        raise ValueError("Only works with MetropolisSampler.")

    _is_rank0 = jax.process_index() == 0

    # Flatten the operator pytree once; all loop internals work on plain lists.
    op_leaves, treedef = jax.tree.flatten(
        op, is_leaf=lambda x: isinstance(x, AbstractOperator)
    )

    atol_leaves = _broadcast_tol(atol, treedef, "atol")
    rtol_leaves = _broadcast_tol(rtol, treedef, "rtol")
    if any(a is None and r is None for a, r in zip(atol_leaves, rtol_leaves)):
        raise ValueError(
            "At least one of 'atol' or 'rtol' must be specified for every operator."
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
        if _check_not_converged(stats_list[i], atol_leaves[i], rtol_leaves[i])
    ]

    it = 0
    with tqdm(
        total=max_iter,
        desc="Sampling",
        unit="iter",
        disable=not verbose or not _is_rank0,
    ) as pbar:
        pbar.set_postfix(_format_postfix(stats_list, active, atol_leaves, rtol_leaves))
        try:
            while active and it < max_iter:
                state.sample(n_discard_per_chain=0)
                stats_list = _accumulate_stats(
                    state, op_leaves, active, stats_list, max_lag=max_lag
                )
                active = [
                    i
                    for i in active
                    if _check_not_converged(
                        stats_list[i], atol_leaves[i], rtol_leaves[i]
                    )
                ]

                pbar.set_postfix(
                    _format_postfix(stats_list, active, atol_leaves, rtol_leaves)
                )
                pbar.update(1)
                it += 1
        except KeyboardInterrupt:
            if _is_rank0:
                pbar.write("  Early termination requested by user.")

        if verbose and _is_rank0:
            if it >= max_iter:
                pbar.write("  Reached max_iter before target precision.")
            err = max(_summary_error_and_scale(s)[0] for s in stats_list)
            pbar.write(f"  [done] max error = {err:g}")

    return treedef.unflatten(stats_list)
