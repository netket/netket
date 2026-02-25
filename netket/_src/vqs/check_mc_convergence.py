""" """

from typing import cast
import copy
import math

import jax.numpy as jnp

from netket.vqs.mc import MCState
from netket.sampler import MetropolisSampler
from netket.operator import AbstractOperator

from netket._src.stats.online_stats import (
    online_statistics,
    expand_max_lag,
    thin_acf_by_2,
)


def check_mc_convergence(
    state_: MCState,
    op: AbstractOperator,
    chain_length: int = 1000,
    plot: bool = False,
    max_iter: int = 2000,
) -> list[bool]:
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
    print("sweep size", state.sampler.sweep_size)

    state.sample()
    O_loc = state.local_estimators(op)
    stats = online_statistics(O_loc, max_lag=max_lag)
    iter = 0
    while acf_window_saturated(stats) or not tau_corr_reliable(stats):
        state.sample()
        O_loc = state.local_estimators(op)
        stats = online_statistics(O_loc, old_estimator=stats)
        print(
            f"{iter} - Operator {op}: stats={stats}, tau_corr={stats.tau_corr}, tau_corr_acf={stats.tau_corr_acf}"
        )
        print(
            f"\t\t\t\t - acf_window_saturated={acf_window_saturated(stats)}, tau_corr_relaiable={tau_corr_reliable(stats)}"
        )
        if acf_window_saturated(stats):
            new_sweep = state.sampler.sweep_size * 2
            print(
                f"\t\t\t\t - ACF window saturated: doubling sweep size to {new_sweep}"
            )
            # Re-index ACF accumulators for the coarser sample rate, then
            # restore the original window width for the new lags.
            old_max_lag = stats.max_lag
            stats = thin_acf_by_2(stats)
            stats = expand_max_lag(stats, old_max_lag)
            state.sampler = state.sampler.replace(sweep_size=new_sweep)
            state.sampler_state = sampler_state
        iter += 1

        if iter > max_iter:
            print(f"Reached maximum number of iterations ({max_iter}). Stopping.")
            break

    print("Final stats:", stats)
    print("Performed calculation with sweep size: ", state.sampler.sweep_size)
    print("Correlation time estimate:", stats.tau_corr_acf)
    print("")
    R = state.sampler.sweep_size / orig_sweep_size
    print("Effective correlation time:", stats.tau_corr_acf * R)
    print(
        "\t (For tau<1 you could reduce the sweep size to get more effective samples.)"
    )
    if plot:
        import matplotlib.pyplot as plt

        acf = stats.acf
        x = jnp.arange(len(acf)) * R
        plt.plot(x, acf)
        plt.show()
    return stats


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


def sample_until_error_of_mean(
    state: MCState,
    op: AbstractOperator,
    *,
    target_error: float,
    max_iter: int = 10_000,
    max_lag: int = 64,
    verbose: bool = True,
):
    """
    Sample until the estimated standard error of the mean falls below
    `target_error`.

    This uses NetKet's online_statistics to update estimates incrementally.

    Parameters
    ----------
    state : MCState
    op : AbstractOperator
    target_error : float
        Desired absolute standard error.
    max_iter : int
        Maximum number of sampling iterations.
    max_lag : int
        Max lag used for autocorrelation estimation.
    """

    if target_error <= 0:
        raise ValueError("target_error must be > 0.")
    if not isinstance(state.sampler, MetropolisSampler):
        raise ValueError("Only works with MetropolisSampler.")

    # initialize statistics from first batch
    state.sample()
    O_loc = state.local_estimators(op)
    stats = online_statistics(O_loc, max_lag=max_lag)

    def _scalar(x):
        return float(jnp.asarray(x))

    err = _scalar(stats.get_stats().error_of_mean)

    if verbose:
        print(f"[init] error = {err:g}")

    it = 0
    while err > target_error and it < max_iter:
        print("estimating")
        state.sample()
        O_loc = state.local_estimators(op)
        O_loc.block_until_ready()
        print("oloc")
        stats = online_statistics(O_loc, old_estimator=stats)
        print("done")

        err = _scalar(stats.get_stats().error_of_mean)

        if verbose:
            print(f"[{it:05d}] error = {err:g}  (target {target_error:g})")

        it += 1

    if it >= max_iter:
        print("Reached max_iter before target error.")

    if verbose:
        print(f"[done] error = {err:g}")

    return stats
