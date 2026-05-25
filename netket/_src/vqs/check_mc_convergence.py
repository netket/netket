""" """

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import copy
import math
import warnings

import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from netket.sampler import MetropolisSampler
from netket.operator import AbstractOperator

if TYPE_CHECKING:
    from netket.vqs.mc import MCState

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

    The loop continues until those conditions are met:

    1. The ACF window is **not saturated** — i.e. the IPS found a non-positive
       consecutive pair, confirming that ``max_lag`` was large enough to capture
       the full tail of the ACF.
    2. The integrated autocorrelation time ``τ`` is *reliable*: the number of
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

    See Also:
        :func:`thermalise_mcmc` — advance chains to stationarity before measuring.
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
                step=stats._n_samples_total // stats.n_chains,
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


def thermalise_mcmc(
    state: MCState,
    op: AbstractOperator,
    *,
    min_chain_length: int = 10,
    max_chain_length: int = 100,
    rhat_tol: float = 1.05,
    decay: float = 0.9,
    patience: int = 1,
    verbose: bool = True,
    raise_on_failure: bool = False,
):
    r"""
    Advance the Markov chains until they are thermalized (R̂ converged).

    Unlike :func:`check_mc_convergence`, this function **mutates** ``state``
    in-place: on return, ``state.sampler_state`` reflects the position of the
    chains after thermalisation.  The sampler's ``sweep_size`` is not changed.

    The function monitors the Gelman-Rubin R̂ diagnostic computed from a
    sliding EMA window of recent batches (controlled by ``decay``).
    Thermalisation is declared when R̂ < ``rhat_tol`` for ``patience``
    consecutive iterations *and* at least ``min_chain_length`` samples/chain
    have been drawn.

    .. warning::
        **Experimental functionality.** This method is subject to change
        without notice in future NetKet releases.

    .. note::
        R̂ is unreliable when the total samples/chain accumulated so far is
        small (roughly < 50).  With very short chains the between-chain
        variance is noisy and R̂ tends to be **overestimated**, so the
        function may not declare convergence until ``min_chain_length`` is
        satisfied even when the chains are already well-mixed.  If
        ``max_chain_length`` is set too low (e.g. below 50 × ``chain_length``)
        you may receive a failure warning even though the sampler is
        actually thermalized.  In that case either increase
        ``max_chain_length`` or reduce ``min_chain_length``.

    Args:
        op: The operator whose local estimators are used to monitor convergence.
            An operator is required because computing R̂ needs per-chain scalar
            values, and local estimators are the only quantity that provides
            this.  The operator does **not** need to be the one you ultimately
            care about — prefer a **cheap** observable (e.g. a single-site
            magnetisation :math:`\hat{\sigma}^z_0`, or the total magnetisation)
            over the full Hamiltonian, which may have many terms and be slow to
            evaluate.  The convergence criterion is the same regardless of
            which operator you choose.
        min_chain_length: Minimum samples/chain before the convergence check
            is applied (default: ``10``).
        max_chain_length: Hard upper limit on samples/chain (default: ``100``).
            If R̂ has not converged by this limit a :class:`UserWarning` is
            emitted (or a :class:`RuntimeError` is raised when
            ``raise_on_failure=True``).  Make sure this is comfortably above
            ``min_chain_length``; otherwise a false failure warning may be
            triggered before R̂ has had enough data to be meaningful.
        rhat_tol: R̂ threshold below which chains are considered mixed
            (default: ``1.05``).
        decay: EMA decay factor for the sliding-window R̂ (default: ``0.9``,
            effective window ≈ 10 batches).  Lower values react faster to
            recent mixing but are noisier.
        patience: Number of consecutive iterations with R̂ < ``rhat_tol``
            required before declaring convergence (default: ``1``).
        verbose: If ``True``, display a :mod:`tqdm` progress bar
            (default: ``True``).
        raise_on_failure: If ``True``, raise :class:`RuntimeError` on failure
            instead of emitting a :class:`UserWarning` (default: ``False``).

    Returns:
        A tuple ``(stats, hist_data)`` where ``stats`` is the final
        :class:`~netket._src.stats.online_stats.OnlineStats` accumulator and
        ``hist_data`` is a :class:`~netket.utils.history.HistoryDict` recording
        the evolution of mean, variance, and R̂ across iterations.

    See Also:
        :func:`check_mc_convergence` — diagnose autocorrelation without mutating state.
    """
    if not isinstance(state.sampler, MetropolisSampler):
        raise ValueError("thermalise_mcmc only works for MetropolisSampler.")

    if state.sampler.n_chains < 2:
        raise ValueError(
            "thermalise_mcmc requires at least 2 chains to compute R̂. "
            f"Current n_chains={state.sampler.n_chains}."
        )

    _is_rank0 = jax.process_index() == 0

    _chain_length = state.chain_length

    state.sample(n_discard_per_chain=0)
    O_loc = state.local_estimators(op)
    # if isinstance(O_loc, LocalEstimatorsBatch):
    #     raise TypeError(
    #         f"thermalise_mcmc requires a scalar local estimator, but "
    #         f"{type(op).__name__} returns a LocalEstimatorsBatch (K={O_loc.n_channels} "
    #         f"channels). thermalise_mcmc only supports scalar operators."
    #     )

    # Subtract 1 from both limits: one batch was already drawn before the loop.
    min_iters = max(0, math.ceil(min_chain_length / _chain_length) - 1)
    max_iters = max(0, max_chain_length // _chain_length - 1)

    stats = online_statistics(O_loc, max_lag=0, decay=decay)
    iter_count = 0
    consecutive_good = 0
    hist_data = HistoryDict()

    with tqdm(
        desc="MC thermalisation",
        total=max_chain_length,
        initial=_chain_length,
        unit=" spl/chain",
        leave=True,
        disable=not (_is_rank0 and verbose),
    ) as pbar:
        while iter_count < min_iters or consecutive_good < patience:
            state.sample(n_discard_per_chain=0)
            O_loc = state.local_estimators(op)
            stats = online_statistics(O_loc, old_estimator=stats)

            rhat = stats.R_hat
            rhat_val = float(rhat)
            _s = stats.get_stats()
            hist_data = hist_data.push(
                {
                    "mean": stats.mean,
                    "error_of_mean": _s.error_of_mean,
                    "variance": stats.variance,
                    "R_hat": rhat_val,
                },
                step=stats._n_samples_total // stats.n_chains,
            )

            good = not math.isnan(rhat_val) and rhat_val < rhat_tol
            consecutive_good = consecutive_good + 1 if good else 0

            pbar.set_postfix(
                {
                    "mean": f"{stats.mean:.4g}",
                    "R_hat": f"{rhat_val:.4f}",
                    "patience": f"{consecutive_good}/{patience}",
                }
            )
            pbar.update(_chain_length)
            iter_count += 1

            if iter_count >= max_iters:
                msg = (
                    f"thermalise_mcmc reached the maximum chain length "
                    f"({max_chain_length} samples/chain) without converging "
                    f"(R̂={rhat_val:.4f} >= {rhat_tol}). "
                    f"Consider increasing max_chain_length or sweep_size."
                )
                if _is_rank0:
                    pbar.write(msg)
                if raise_on_failure:
                    raise RuntimeError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                break

    return stats, hist_data
