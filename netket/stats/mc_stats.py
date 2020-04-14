from numba import jit
import numpy as _np
from . import mean as _mean
from . import var as _var
from . import total_size as _total_size


@jit(nopython=True)
def _get_blocks(data, l):
    n_blocks = int(_np.floor(data.shape[1] / float(l)))

    blocks = _np.empty(data.shape[0] * n_blocks, dtype=data.dtype)
    k = 0
    for i in range(data.shape[0]):
        for b in range(n_blocks):
            blocks[k] = data[i, b * l:(b + 1) * l].mean()
            k += 1
    return blocks


def _block_variance(data, l):
    blocks = _get_blocks(data, l)
    ts = _total_size(blocks)
    if(ts > 0):
        return _var(blocks) / float(ts), ts
    else:
        return _np.nan, 0


def _batch_variance(data):
    b_means = _np.mean(data, axis=1)

    ts = _total_size(b_means)
    return _var(b_means) / float(ts), ts


def statistics(data):
    stats = {}
    data = _np.atleast_1d(data)
    if(data.ndim == 1):
        data = data.reshape((1, -1))

    if(data.ndim > 2):
        raise NotImplementedError(
            "Statistics are implemented only for ndim<=2")

    stats['mean'] = _mean(data)
    stats['var'] = _var(data)

    ts = _total_size(data)

    bare_var = stats['var'] / float(ts)

    batch_var, n_batches = _batch_variance(data)

    b_s = 32
    l_block = data.shape[1] // b_s
    block_var, n_blocks = _block_variance(data, l_block)

    tau_batch = (batch_var / bare_var - 1) * 0.5
    tau_block = (block_var / bare_var - 1) * 0.5

    block_good = (tau_block < (l_block / 10.) and n_blocks >= b_s)
    batch_good = (tau_batch < (data.shape[1] / 10.) and n_batches >= b_s)

    if(batch_good):
        stats['sigma'] = _np.sqrt(batch_var)
        stats['tau_corr'] = max(0, tau_batch)
        return stats
    if(block_good):
        stats['sigma'] = _np.sqrt(block_var)
        stats['tau_corr'] = max(0, tau_block)
        return stats

    stats['sigma'] = _np.nan
    stats['tau_corr'] = _np.nan

    return stats
