from ._C_netket.sampler import *

import numpy as _np


def compute_samples(
    sampler,
    n_samples,
    n_discard=None,
    compute_log_values=False,
    samples=None,
    log_values=None,
):

    n_chains = sampler.n_chains
    n_samples = int(_np.ceil((n_samples / n_chains)))

    if samples is None or log_values is None:
        samples = _np.ndarray((n_samples, n_chains, sampler.machine.hilbert.size))
        if compute_log_values:
            log_values = _np.ndarray((n_samples, n_chains), dtype=_np.complex128)

    if n_discard is None:
        n_discard = n_samples // 10

    sweep = sampler.sweep

    # Burnout phase
    for _ in range(n_discard):
        sweep()

    if compute_log_values:
        state = sampler.current_state
        # Generate samples
        for i in range(n_samples):
            sweep()
            samples[i], log_values[i] = state

        return samples, log_values
    else:
        state = sampler.current_sample
        # Generate samples
        for i in range(n_samples):
            sweep()
            samples[i] = state

        return samples
