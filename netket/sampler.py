from ._C_netket.sampler import *

import numpy as _np


def compute_samples(sampler, n_samples, n_discard=None, samples=None, log_values=None):

    n_chains = sampler.n_chains
    n_samples = int(_np.ceil((n_samples / n_chains)))

    if samples is None or log_values is None:
        samples = _np.ndarray(
            (n_samples, n_chains, sampler.machine.hilbert.size))
        log_values = _np.ndarray((n_samples, n_chains), dtype=_np.complex128)
    else:
        samples.resize((n_samples, n_chains, sampler.machine.hilbert.size))
        log_values.resize((n_samples, n_chains))

    if n_discard is None:
        n_discard = n_samples // 10

    # Burnout phase
    for _ in range(n_discard):
        sampler.sweep()

    # Generate samples
    for i in range(samples.shape[0]):
        sampler.sweep()
        samples[i], log_values[i] = sampler.current_state

    return samples, log_values
