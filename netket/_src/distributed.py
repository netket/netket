"""Private multi-process helpers; each is a no-op in a single-process run."""

import jax
import numpy as np
from jax.experimental import multihost_utils


def _is_distributed() -> bool:
    """Whether there is more than one process to coordinate with."""
    return bool(jax.distributed.is_initialized()) and jax.process_count() > 1


def any_rank(flag: bool) -> bool:
    """Whether ``flag`` is true on any process. A collective: every process must
    call it, at the same point."""
    if not _is_distributed():
        return bool(flag)
    # A 1-d int32 buffer: a 0-d or int64 input goes wrong on the Gloo CPU backend
    # ("Received data size doesn't match expected size").
    flags = multihost_utils.process_allgather(np.asarray([flag], dtype=np.int32))
    return bool(np.any(flags))
