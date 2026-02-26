""" """

from typing import cast
import copy
import math

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from netket.vqs.mc import MCState
from netket.sampler import MetropolisSampler
from netket.operator import AbstractOperator

from netket._src.stats.online_stats import (
    online_statistics,
    expand_max_lag,
    thin_acf_by_2,
)
from netket.utils.history import HistoryDict


def check_mc_convergence(
    state_: MCState,
    op: AbstractOperator,
    min_chain_length: int = 50,
    plot: bool = False,
    max_chain_length: int = 500,
):
    """
    Diagnose whether the Markov-chain sweep size is long enough to produce
    decorrelated samples for the expectation value of ``op``.

    **Algorithm overview**

    The function operates on a *temporary copy* of ``state_`` whose internal
    sweep size is reset to 1, exposing every elementary MC step as an
    individual sample row.  Batches of local estimators are fed one at a time
    into :func:`~netket.stats.online_statistics`, which maintains running
    estimates of the mean, variance, and the autocorrelation function (ACF)
    via the Geyer *initial positive sequence* (IPS) estimator.

    The loop continues until **all three** conditions are met:

    1. At least ``min_chain_length`` samples/chain have been accumulated.
    2. The ACF window is **not saturated** — i.e. the IPS found a non-positive
       consecutive pair, confirming that ``max_lag`` was large enough to capture
       the full tail of the ACF.
    3. The integrated autocorrelation time ``τ`` is *reliable*: the number of
       effective samples per chain is at least 50 (i.e.
       ``n_per_chain / τ ≥ 50``).

    **Adaptive coarsening when the ACF window saturates**

    If the ACF window saturates (every consecutive pair ``(ρ[2t]+ρ[2t+1])``
    is positive up to ``max_lag``), the current ``τ`` estimate is merely a
    lower bound — the chains are too short or the sweep too fine.  The
    algorithm then:

    * doubles the internal sweep size (``sweep_size *= 2``), thinning the
      Markov chain to make long-range correlations visible within the window;
    * calls :func:`~netket.stats._src.online_stats.thin_acf_by_2` to
      re-index the ACF accumulator for the coarser cadence; and
    * calls :func:`~netket.stats._src.online_stats.expand_max_lag` to restore
      the lag window width so the next iterations can probe further.

    This doubling is repeated as needed until the window is no longer
    saturated, or until ``max_chain_length`` samples/chain are exhausted.

    **Final diagnosis**

    After convergence the correlation time is re-expressed in terms of raw MC
    steps (``τ_mc = τ_acf × final_sweep_size``) and of the user's original
    sweep units (``τ_sweeps = τ_mc / orig_sweep_size``).  A sweep size is
    considered adequate when ``τ_sweeps < 1``, meaning that consecutive
    samples produced by the user's MCState are effectively independent.  The
    recommended minimum sweep size is ``2 τ_mc`` raw steps.

    Args:
        state_: The :class:`~netket.vqs.MCState` to diagnose.  A shallow copy
            is used internally; the original state is never mutated.
        op: The operator whose local estimators are used to probe correlations.
        min_chain_length: Minimum number of samples per chain to accumulate
            before the convergence check is applied.
        plot: If ``True``, display a diagnostic figure after the run.
        max_chain_length: Hard upper limit on samples per chain.  The loop is
            terminated unconditionally once this many samples have been drawn.

    Returns:
        A tuple ``(stats, hist_data)`` where ``stats`` is the final
        :class:`~netket.stats.OnlineStatistics` accumulator and ``hist_data``
        is a :class:`~netket.utils.history.HistoryDict` recording the
        evolution of key diagnostics (mean, error, R̂, τ) as the number of samples
        is increased.
    """
    if not isinstance(state_.sampler, MetropolisSampler):
        raise ValueError("check_mc_convergence only works for MetropolisSampler.")
    sampler = cast(MetropolisSampler, state_.sampler)

    # work on a copy of the state
    state = copy.copy(state_)
    del state_

    orig_sweep_size = sampler.sweep_size
    max_lag = 32

    sampler_state = state.sampler_state
    state.sampler = sampler.replace(sweep_size=1)
    state.sampler_state = sampler_state

    _is_rank0 = jax.process_index() == 0

    # Derive iteration bounds from chain-length units.
    _chain_length = state.chain_length
    min_iters = math.ceil(min_chain_length / _chain_length)
    max_iters = max_chain_length // _chain_length

    state.sample()
    O_loc = state.local_estimators(op)
    stats = online_statistics(O_loc, max_lag=max_lag)
    iter = 0
    hist_data = HistoryDict()

    with tqdm(
        desc="MC convergence",
        total=max_chain_length,
        initial=_chain_length,
        unit=" spl/chain",
        leave=True,
        disable=not _is_rank0,
    ) as pbar:
        while (
            iter < min_iters
            or acf_window_saturated(stats)
            or not tau_corr_reliable(stats)
        ):
            state.sample(n_discard_per_chain=0)
            O_loc = state.local_estimators(op)
            stats = online_statistics(O_loc, old_estimator=stats)
            _s = stats.get_stats()
            hist_data = hist_data.push(
                {
                    "mean": stats.mean,
                    "error_of_mean": _s.error_of_mean,
                    "variance": stats.variance,
                    "R_hat": stats.R_hat,
                    "tau_corr_acf": stats.tau_corr_acf,
                    "tau_corr_batch": stats.tau_corr_batch,
                    "sweep_size": state.sampler.sweep_size,
                },
                step=stats._n_samples_total,
            )

            # Always compose postfix dict (touches JAX arrays on all ranks).
            saturated = acf_window_saturated(stats)
            postfix = {
                "sweep": state.sampler.sweep_size,
                "mean": f"{stats.mean:.4g}",
                "tau": f"{stats.tau_corr_acf:.3g}",
                "R_hat": f"{stats.R_hat:.4f}",
                "sat": saturated,
            }
            pbar.set_postfix(postfix)
            pbar.update(_chain_length)

            if saturated:
                new_sweep = state.sampler.sweep_size * 2
                # Compose on all ranks; only rank 0 writes.
                msg = f"  [iter {iter}] ACF window saturated — doubling sweep size to {new_sweep}"
                if _is_rank0:
                    pbar.write(msg)
                # Re-index ACF accumulators for the coarser sample rate, then
                # restore the original window width for the new lags.
                old_max_lag = stats.max_lag
                stats = thin_acf_by_2(stats)
                stats = expand_max_lag(stats, old_max_lag)
                state.sampler = state.sampler.replace(sweep_size=new_sweep)
                state.sampler_state = sampler_state
            iter += 1

            if iter >= max_iters:
                msg = f"  Reached maximum chain length ({max_chain_length} samples/chain). Stopping."
                if _is_rank0:
                    pbar.write(msg)
                break

    # Always compute derived quantities on all ranks (avoids deadlocks).
    # tau_corr_acf is in units of (final_sweep_size MC steps); rescale to raw steps.
    final_sweep = state.sampler.sweep_size
    tau_acf = stats.tau_corr_acf
    tau_mc_steps = tau_acf * final_sweep  # correlation time in raw MC steps
    tau_sweeps = (
        tau_mc_steps / orig_sweep_size
    )  # correlation time in user's sweep units
    min_sweep_size = 2.0 * tau_mc_steps  # recommended minimum MCState.sweep_size

    good = tau_sweeps < 1.0
    sweep_emoji = "✅" if good else "❌"
    summary = (
        f"\n"
        f"---- MC Convergence Results ----\n"
        f"  Final statistics         : {stats}\n"
        f"\n"
        f"  τ_corr (MC steps)        : {tau_mc_steps:.3g}"
        + (
            f"  [τ_acf={tau_acf:.3g} × internal sweep_size={final_sweep}]\n"
            if final_sweep > 1
            else "\n"
        )
        + f"\n"
        f"  MCState.sweep_size       : {orig_sweep_size}\n"
        f"  τ_corr (MC sweeps)       : {tau_sweeps:.3g}  {sweep_emoji}"
        f"  {'(< 1 sweep — good)' if good else '(≥ 1 sweep — consider increasing sweep_size)'}\n"
        f"\n"
        f"  Minimum sweep_size       : ~{min_sweep_size:.1f}  (= 2 × τ_corr in MC steps)\n"
        f"\n"
        f"  In NQS, keep sweep_size ≥ 2τ (MC steps) so that consecutive samples\n"
        f"  are effectively independent and all samples carry useful information.\n"
        f"--------------------------------"
    )
    if _is_rank0:
        print(summary)
    if plot:
        from netket._src.vqs.plot_mc_convergence import plot_mc_convergence
        import matplotlib.pyplot as plt

        plot_mc_convergence(
            stats,
            hist_data,
            orig_sweep_size=orig_sweep_size,
            final_sweep=final_sweep,
        )
        plt.show()
    return stats, hist_data


def acf_window_saturated(estimator) -> bool:
    """True if the Geyer IPS ran out of lag window without finding a non-positive pair.
    Means tau_corr_acf is a lower bound — increase max_lag or accumulate longer chains.
    """
    rho = estimator.acf
    if rho is None:
        return False
    m = len(rho) // 2
    if m == 0:
        return False
    # Reshape to (m, 2) — static shape since max_lag is a compile-time constant —
    # then sum pairs: P[t] = rho[2t] + rho[2t+1].
    P = jnp.asarray(rho[: 2 * m]).reshape(m, 2).sum(axis=-1)  # (m,)
    return bool(jnp.all(P > 0))  # True iff IPS never found a non-positive pair


def tau_corr_reliable(estimator) -> bool:
    """True when the tau_corr estimate is likely trustworthy.

    Fails when:
    - ACF window is saturated (need more lag → increase max_lag)
    - Too few effective samples (need more data → n_effective < 50)
    """
    if acf_window_saturated(estimator):
        return False
    tau = estimator.tau_corr_acf
    if math.isnan(tau) or tau <= 0:
        return False
    n_per_chain = estimator._n_samples_total / estimator.n_chains
    return (n_per_chain / tau) >= 50


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

    def _not_converged(stats) -> bool:
        s = stats.get_stats()
        err = s.error_of_mean
        mean = s.mean
        if atol is not None and err > atol:
            return True
        if rtol is not None and err / abs(mean) > rtol:
            return True
        return False

    def _postfix(stats) -> dict:
        s = stats.get_stats()
        err = s.error_of_mean
        mean = s.mean
        d = {"err": f"{err:.4g}"}
        if atol is not None:
            d["atol"] = f"{atol:.4g}"
        if rtol is not None:
            d["rel_err"] = f"{err / abs(mean):.4g}"
            d["rtol"] = f"{rtol:.4g}"
        return d

    # initialize statistics from first batch
    state.sample()
    O_loc = state.local_estimators(op)
    stats = online_statistics(O_loc, max_lag=max_lag)

    it = 0
    with tqdm(
        total=max_iter,
        desc="Sampling",
        unit="iter",
        disable=not verbose or not _is_rank0,
    ) as pbar:
        pbar.set_postfix(_postfix(stats))
        try:
            while _not_converged(stats) and it < max_iter:
                state.sample(n_discard_per_chain=0)
                O_loc = state.local_estimators(op)
                O_loc.block_until_ready()
                stats = online_statistics(O_loc, old_estimator=stats)

                pbar.set_postfix(_postfix(stats))
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
        s = stats.get_stats()
        msg = f"  [done] error = {s.error_of_mean:g}"
        if _is_rank0:
            pbar.write(msg)

    return stats
