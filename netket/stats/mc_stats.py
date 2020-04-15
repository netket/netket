from numba import jit
import numpy as _np
from . import mean as _mean
from . import var as _var
from . import total_size as _total_size


class Stats(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _NaN = float("NaN")

    def __init__(
        self, mean=_NaN, error_of_mean=_NaN, variance=_NaN, tau_corr=_NaN, R=_NaN,
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
        return jsd

    def __repr__(self):
        return (
            "{:.4e}".format(self.mean.real)
            + " ± "
            + "{:.1e}".format(self.error_of_mean)
        )


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
    return _var(b_means) / float(ts), ts


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
       dict: A dictionary-compatible class containing the average (mean),
             the variance (variance),
             the error of the mean (sigma), and an estimate of the
             autocorrelation time (tauCorr). In addition to accessing the elements with the standard
             dict sintax (e.g. res['mean']), one can also access them directly with the dot operator
             (e.g. res.mean).
    """

    stats = Stats()
    data = _np.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    stats["mean"] = _mean(data)
    stats["variance"] = _var(data)

    ts = _total_size(data)

    bare_var = stats["variance"] / float(ts)

    batch_var, n_batches = _batch_variance(data)

    b_s = 32
    l_block = max(1, data.shape[1] // b_s)

    block_var, n_blocks = _block_variance(data, l_block)

    tau_batch = (batch_var / bare_var - 1) * 0.5
    tau_block = (block_var / bare_var - 1) * 0.5

    block_good = tau_block < (l_block / 10.0) and n_blocks >= b_s
    batch_good = tau_batch < (data.shape[1] / 10.0) and n_batches >= b_s

    if batch_good:
        stats["error_of_mean"] = _np.sqrt(batch_var)
        stats["tau_corr"] = max(0, tau_batch)
    elif block_good:
        stats["error_of_mean"] = _np.sqrt(block_var)
        stats["tau_corr"] = max(0, tau_block)
    else:
        stats["error_of_mean"] = _np.nan
        stats["tau_corr"] = _np.nan

    # if n_batches > 2:
    #     W = _mean(_np.var(data, axis=0))
    #     V = stats["variance"]
    #     stats["R"] = _np.sqrt(V / W)

    return stats
