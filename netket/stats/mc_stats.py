import math

from numba import jit
import numpy as _np
from . import mean as _mean
from . import var as _var
from . import total_size as _total_size


def _format_decimal(value, std, var):
    if math.isfinite(std) and std > 1e-7:
        decimals = max(int(_np.ceil(-_np.log10(std))), 0)
        return (
            "{0:.{1}f}".format(value, decimals + 1),
            "{0:.{1}f}".format(std, decimals + 1),
            "{0:.{1}f}".format(var, decimals + 1),
        )
    else:
        return (
            "{0:.3e}".format(value),
            "{0:.3e}".format(std),
            "{0:.3e}".format(var),
        )


class Stats:
    """A dict-compatible class containing the result of the statistics function."""

    _NaN = float("NaN")

    def __init__(
        self,
        mean=_NaN,
        error_of_mean=_NaN,
        variance=_NaN,
        tau_corr=_NaN,
        R_hat=_NaN,
    ):
        self.mean = complex(mean) if _np.iscomplexobj(mean) else float(mean)
        self.error_of_mean = float(error_of_mean)
        self.variance = float(variance)
        self.tau_corr = float(tau_corr)
        self.R_hat = float(R_hat)

    def to_json(self):
        jsd = {}
        jsd["Mean"] = self.mean.real
        jsd["Variance"] = self.variance
        jsd["Sigma"] = self.error_of_mean
        jsd["R_hat"] = self.R_hat
        jsd["TauCorr"] = self.tau_corr
        return jsd

    def __repr__(self):
        mean, err, var = _format_decimal(self.mean, self.error_of_mean, self.variance)
        if not math.isnan(self.R_hat):
            ext = ", R̂={:.4f}".format(self.R_hat)
        else:
            ext = ""
        return "{} ± {} [σ²={}{}]".format(mean, err, var, ext)

    def __getitem__(self, name):
        if name in ("mean", "Mean"):
            return self.mean
        elif name in ("variance", "Variance"):
            return self.variance
        elif name in ("error_of_mean", "Sigma"):
            return self.error_of_mean
        elif name in ("R_hat", "R"):
            return self.R_hat
        elif name in ("tau_corr", "TauCorr"):
            return self.tau_corr


@jit(nopython=True)
def _get_blocks(data, l):
    n_blocks = int(_np.floor(data.shape[1] / float(l)))
    blocks = _np.empty(data.shape[0] * n_blocks, dtype=data.dtype)
    k = 0
    for i in range(data.shape[0]):
        for b in range(n_blocks):
            blocks[k] = data[i, b * l : (b + 1) * l].mean()
            k += 1
    return blocks


def _block_variance(data, l):
    blocks = _get_blocks(data, l)
    ts = _total_size(blocks)
    if ts > 0:
        return _var(blocks), ts
    else:
        return _np.nan, 0


def _batch_variance(data):
    b_means = _np.mean(data, axis=1)
    ts = _total_size(b_means)
    return _var(b_means), ts


def statistics(data):
    r"""
    Returns statistics of a given array (or matrix, see below) containing a stream of data.
    This is particularly useful to analyze Markov Chain data, but it can be used
    also for other type of time series.

    Args:
        data (vector or matrix): The input data. It can be real or complex valued.
                                * if a vector, it is assumed that this is a time
                                  series of data (not necessarily independent).
                                * if a matrix, it is assumed that that rows data[i]
                                  contain independent time series.

    Returns:
       Stats: A dictionary-compatible class containing the average (mean),
             the variance (variance),
             the error of the mean (error_of_mean), and an estimate of the
             autocorrelation time (tau_corr). In addition to accessing the elements with the standard
             dict sintax (e.g. res['mean']), one can also access them directly with the dot operator
             (e.g. res.mean).
    """

    stats = Stats()
    data = _np.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    mean = _mean(data)
    variance = _var(data)

    ts = _total_size(data)

    bare_var = variance

    batch_var, n_batches = _batch_variance(data)

    b_s = 32
    l_block = max(1, data.shape[1] // b_s)

    block_var, n_blocks = _block_variance(data, l_block)

    tau_batch = ((ts / n_batches) * batch_var / bare_var - 1) * 0.5
    tau_block = ((ts / n_blocks) * block_var / bare_var - 1) * 0.5

    block_good = n_blocks >= b_s and tau_block < 6 * l_block
    batch_good = n_batches >= b_s and tau_batch < 6 * data.shape[1]

    if batch_good:
        error_of_mean = _np.sqrt(batch_var / n_batches)
        tau_corr = max(0, tau_batch)
    elif block_good:
        error_of_mean = _np.sqrt(block_var / n_blocks)
        tau_corr = max(0, tau_block)
    else:
        error_of_mean = _np.nan
        tau_corr = _np.nan

    if n_batches > 1:
        N = data.shape[-1]

        # V_loc = _np.var(data, axis=-1, ddof=0)
        # W_loc = _np.mean(V_loc)
        # W = _mean(W_loc)
        # # This approximation seems to hold well enough for larger n_samples
        W = variance

        R_hat = _np.sqrt((N - 1) / N + batch_var / W)
    else:
        R_hat = float("nan")

    return Stats(mean, error_of_mean, variance, tau_corr, R_hat)
