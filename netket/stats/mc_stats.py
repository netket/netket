from numba import jit
import numpy as _np
from . import mean as _mean
from . import var as _var
from . import total_size as _total_size


def _format_decimal(value, std):
    decimals = max(int(_np.ceil(-_np.log10(std))), 0)
    return (
        "{0:.{1}f}".format(value, decimals),
        "{0:.{1}f}".format(std, decimals + 1),
    )


class Stats:
    """A dict-compatible class containing the result of the statistics function."""

    _NaN = float("NaN")

    def __init__(
        self, mean=_NaN, error_of_mean=_NaN, variance=_NaN, tau_corr=_NaN, R=_NaN
    ):
        self.mean = mean
        self.error_of_mean = error_of_mean
        self.variance = variance
        self.tau_corr = tau_corr
        self.R = R

    def to_json(self):
        jsd = {}
        jsd["Mean"] = self.mean.real
        jsd["Variance"] = self.variance
        jsd["Sigma"] = self.error_of_mean
        jsd["R"] = self.R
        jsd["TauCorr"] = self.tau_corr
        return jsd

    def __repr__(self):
        mean, err = _format_decimal(self.mean, self.error_of_mean)
        return "{} ± {} [var={:.1e}, R={:.4f}]".format(
            mean, err, self.variance, self.R
        )

    def __getitem__(self, name):
        if name is "mean" or "Mean":
            return self.mean
        elif name is "variance" or "Variance":
            return self.variance
        elif name is "error_of_mean" or "Sigma":
            return self.error_of_mean
        elif name is "R":
            return self.R
        elif name is "tau_corr" or "TauCorr":
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
        return _var(blocks) / float(ts), ts
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

    stats.mean = _mean(data)
    stats.variance = _var(data)

    ts = _total_size(data)

    bare_var = stats.variance / ts

    batch_var, n_batches = _batch_variance(data)

    b_s = 32
    l_block = max(1, data.shape[1] // b_s)

    block_var, n_blocks = _block_variance(data, l_block)

    tau_batch = (batch_var / bare_var - 1) * 0.5
    tau_block = (block_var / bare_var - 1) * 0.5

    block_good = tau_block < 6 * l_block and n_blocks >= b_s
    batch_good = tau_batch < 6 * data.shape[1] and n_batches >= b_s

    if batch_good:
        stats.error_of_mean = _np.sqrt(batch_var / ts)
        stats.tau_corr = max(0, tau_batch)
    elif block_good:
        stats.error_of_mean = _np.sqrt(block_var)
        stats.tau_corr = max(0, tau_block)
    else:
        stats.error_of_mean = _np.nan
        stats.tau_corr = _np.nan

    if n_batches > 1:
        N = data.shape[-1]
        stats.R = _np.sqrt((N - 1) / N + batch_var / stats.variance)

    return stats
