"""Plotting utilities for check_mc_convergence."""

import math

import numpy as np


def plot_mc_convergence(
    stats,
    hist_data,
    *,
    orig_sweep_size: int,
    final_sweep: int,
):
    """Plot diagnostics from a check_mc_convergence run.

    Produces a single figure with five sub-panels:

    - **Mean** vs number of accumulated samples (convergence of the estimator).
    - **Variance** vs number of accumulated samples.
    - **R̂** (Gelman-Rubin) vs number of accumulated samples.
    - **τ_corr** (both ACF- and batch-based) vs number of accumulated samples.
    - **Autocorrelation function** (ACF) at the final iteration, with vertical
      lines marking τ_corr (ACF) and, when available, τ_corr (batch).

    Args:
        stats: Final :class:`~netket._src.stats.online_stats.OnlineStats` object.
        hist_data: :class:`~netket.utils.history.HistoryDict` collected during
            the run.  Must contain keys ``"mean"``, ``"variance"``, ``"R_hat"``,
            ``"tau_corr_acf"``, and ``"tau_corr_batch"``.
        orig_sweep_size: Original ``MCState.sweep_size`` (before any doubling).
        final_sweep: Sweep size used at the very last iteration.

    Returns:
        The :class:`matplotlib.figure.Figure` created.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    R = final_sweep / orig_sweep_size  # lag unit: sweeps per ACF sample
    n_chains = stats.n_chains

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax_mean = fig.add_subplot(gs[0, 0])
    ax_var = fig.add_subplot(gs[0, 1])
    ax_rhat = fig.add_subplot(gs[0, 2])
    ax_tau = fig.add_subplot(gs[1, 0])
    ax_acf = fig.add_subplot(gs[1, 1:])

    # ------------------------------------------------------------------
    # Helper: extract iters and values from a History in the HistoryDict
    # ------------------------------------------------------------------
    def _get(key):
        h = hist_data[key]
        return np.asarray(h.iters), np.asarray(h.values)

    # ------------------------------------------------------------------
    # Helper: log-scale x-axis with non-overlapping tick labels
    # ------------------------------------------------------------------
    def _setup_log_xaxis(ax):
        from matplotlib.ticker import LogLocator, NullFormatter

        ax.set_xscale("log")
        # At most ~5 major ticks (one per decade); minor ticks as guides only
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=20))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="major", labelrotation=35)
        ax.tick_params(axis="x", which="minor", length=3)

    # ------------------------------------------------------------------
    # Mean  (+ error ribbon + final-mean hline)
    # ------------------------------------------------------------------
    xs, ys = _get("mean")
    _, errs = _get("error_of_mean")
    ys_real = np.real(ys)
    errs_real = np.abs(np.real(errs))
    ax_mean.fill_between(
        xs,
        ys_real - errs_real,
        ys_real + errs_real,
        alpha=0.25,
        color="C0",
        label=r"$\pm\,\sigma_\mu$",
    )
    ax_mean.plot(xs, ys_real, ".-", color="C0")
    final_mean = ys_real[-1]
    ax_mean.axhline(
        final_mean,
        color="C0",
        linewidth=1.2,
        linestyle="--",
        label=f"final = {final_mean:.4g}",
    )
    _setup_log_xaxis(ax_mean)
    ax_mean.set_xlabel("Samples")
    ax_mean.set_ylabel("Mean")
    ax_mean.set_title("Mean")
    ax_mean.legend(fontsize="small")

    # ------------------------------------------------------------------
    # Variance  (+ final-variance hline)
    # ------------------------------------------------------------------
    xs, ys = _get("variance")
    ax_var.plot(xs, ys, ".-", color="C1")
    final_var = float(ys[-1])
    ax_var.axhline(
        final_var,
        color="C1",
        linewidth=1.2,
        linestyle="--",
        label=f"final = {final_var:.4g}",
    )
    _setup_log_xaxis(ax_var)
    ax_var.set_xlabel("Samples")
    ax_var.set_ylabel("Variance")
    ax_var.set_title("Variance")
    ax_var.legend(fontsize="small")

    # ------------------------------------------------------------------
    # R̂  (Gelman-Rubin convergence diagnostic)
    # Background bands: green [1, 1.05], yellow [1.05, 1.1], red [1.1, ∞)
    # ------------------------------------------------------------------
    xs, ys = _get("R_hat")
    _, sweep_sizes_rhat = _get("sweep_size")
    xs_rhat_chain = xs * sweep_sizes_rhat / n_chains
    if not np.all(np.isnan(ys)):
        valid = ys[~np.isnan(ys)]
        data_ymax = max(float(np.max(valid)) * 1.02, 1.12)
        data_ymin = min(float(np.min(valid)) * 0.998, 0.988)

        # Background bands (drawn first, behind the data)
        ax_rhat.axhspan(1.00, 1.05, alpha=0.15, color="limegreen", zorder=0)
        ax_rhat.axhspan(1.05, 1.10, alpha=0.15, color="gold", zorder=0)
        ax_rhat.axhspan(1.10, data_ymax + 10, alpha=0.15, color="salmon", zorder=0)

        ax_rhat.plot(xs_rhat_chain, ys, ".-", color="C2", zorder=3)
        ax_rhat.axhline(1.0, color="k", linewidth=0.8, linestyle="--", zorder=2)
        ax_rhat.set_ylim(data_ymin, data_ymax)
    else:
        ax_rhat.text(
            0.5,
            0.5,
            "N/A\n(single chain)",
            ha="center",
            va="center",
            transform=ax_rhat.transAxes,
            color="gray",
        )
    _setup_log_xaxis(ax_rhat)
    ax_rhat.set_xlabel("Chain length (MC steps)")
    ax_rhat.set_ylabel(r"$\hat{R}$")
    ax_rhat.set_title(r"$\hat{R}$  (Gelman-Rubin)")

    # ------------------------------------------------------------------
    # Autocorrelation time history
    # Both tau values are in units of the current internal sweep_size at
    # the time of recording.  Multiply by sweep_size / orig_sweep_size to
    # express in original-sweep units; then the secondary y-axis (raw MC
    # steps) is simply a fixed × orig_sweep_size transform — mirroring the
    # dual x-axis on the ACF panel.
    # ------------------------------------------------------------------
    xs_acf, ys_acf = _get("tau_corr_acf")
    xs_batch, ys_batch = _get("tau_corr_batch")
    _, sweep_sizes = _get("sweep_size")

    # Convert tau to sweep units; convert x to chain length (MC steps per chain)
    ys_acf_sw = ys_acf * sweep_sizes / orig_sweep_size
    ys_batch_sw = ys_batch * sweep_sizes / orig_sweep_size
    xs_acf_chain = xs_acf * sweep_sizes / n_chains
    xs_batch_chain = xs_batch * sweep_sizes / n_chains

    if not np.all(np.isnan(ys_acf_sw)):
        ax_tau.plot(
            xs_acf_chain,
            ys_acf_sw,
            ".-",
            color="C3",
            label=r"$\tau_\mathrm{ACF}$",
        )
    if not np.all(np.isnan(ys_batch_sw)):
        ax_tau.plot(
            xs_batch_chain,
            ys_batch_sw,
            ".-",
            color="C4",
            label=r"$\tau_\mathrm{batch}$",
        )
    _setup_log_xaxis(ax_tau)
    ax_tau.set_xlabel("Chain length (MC steps)")
    ax_tau.set_ylabel("τ (sweeps)")
    ax_tau.set_title("Autocorrelation time")
    ax_tau.legend(fontsize="small")

    # Secondary y-axis: τ in raw MC steps (fixed × orig_sweep_size transform)
    secay = ax_tau.secondary_yaxis(
        "right",
        functions=(
            lambda t_sw: t_sw * orig_sweep_size,
            lambda t_mc: t_mc / orig_sweep_size,
        ),
    )
    secay.set_ylabel("τ (raw MC steps)")

    # ------------------------------------------------------------------
    # ACF panel
    # ------------------------------------------------------------------
    acf = stats.acf
    if acf is not None:
        x = np.arange(len(acf)) * R  # lag in units of original sweeps

        ax_acf.plot(x, acf, ".-", color="C0")

        # Mark points spaced one original sweep apart.
        stride = max(1, round(1.0 / R))
        ax_acf.plot(
            x[::stride],
            np.asarray(acf)[::stride],
            "x",
            color="C0",
            ms=8,
            markeredgewidth=2,
            label=f"ACF (stride = {final_sweep} steps = {R:.3g} sweeps)",
        )

        ax_acf.axhline(0, color="k", linewidth=0.8, linestyle="--")

        # τ_corr from ACF (Geyer IPS)
        tau_acf_sweeps = stats.tau_corr_acf * R
        ax_acf.axvline(
            tau_acf_sweeps,
            color="C3",
            linewidth=1.5,
            linestyle="--",
            label=rf"$\tau_\mathrm{{ACF}}$ = {tau_acf_sweeps:.2g} sweeps",
        )
        ax_acf.axvspan(0, tau_acf_sweeps, alpha=0.08, color="C3")

        # τ_corr from batch (between-chain) estimate, when available
        tau_batch = stats.tau_corr_batch
        if not math.isnan(tau_batch):
            tau_batch_sweeps = tau_batch * final_sweep / orig_sweep_size
            ax_acf.axvline(
                tau_batch_sweeps,
                color="C4",
                linewidth=1.5,
                linestyle=":",
                label=rf"$\tau_\mathrm{{batch}}$ = {tau_batch_sweeps:.2g} sweeps",
            )

        ax_acf.set_xlabel("Lag (sweeps)")
        ax_acf.set_ylabel("Autocorrelation")
        ax_acf.set_title("Autocorrelation function (final)")
        ax_acf.legend(fontsize="small")

        # Secondary x-axis in raw MC steps
        secax = ax_acf.secondary_xaxis(
            "top",
            functions=(
                lambda s: s * orig_sweep_size,
                lambda n: n / orig_sweep_size,
            ),
        )
        secax.set_xlabel("Lag (raw MC steps)")
    else:
        ax_acf.text(
            0.5,
            0.5,
            "ACF not available",
            ha="center",
            va="center",
            transform=ax_acf.transAxes,
            color="gray",
        )

    fig.suptitle("MC Convergence Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
