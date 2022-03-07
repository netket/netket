# File containing common commands for NetKet Test infrastructure

import pytest
import netket as nk

import os


def _is_true(x):
    if isinstance(x, str):
        xl = x.lower()
        if xl == "1" or x == "true":
            return True
    elif x == 1:
        return True
    else:
        return False


skipif_ci = pytest.mark.skipif(
    _is_true(os.environ.get("CI", False)), reason="Test too slow/broken on CI"
)
"""Use as a decorator to mark a test to be skipped when running on CI.
For example:

Example:
>>> @skipif_ci
>>> def test_my_serial_function():
>>>     your_serial_test()

"""

skipif_mpi = pytest.mark.skipif(nk.utils.mpi.n_nodes > 1, reason="Only run without MPI")
"""Use as a decorator to mark a test to be skipped when running under MPI.
For example:

Example:
>>> @skipif_mpi
>>> def test_my_serial_function():
>>>     your_serial_test()

"""

onlyif_mpi = pytest.mark.skipif(nk.utils.mpi.n_nodes < 2, reason="Only run with MPI")
"""Use as a decorator to mark a test to be executed only when running with at least 2 MPI
nodes.

It can be used in combination with the fixtures defined in test/conftest.py, namely
_mpi_rank, mpi_size and _mpi_comm, that retrieve the number of mpi rank, size and comm
used by netket.

If you need to trick netket into executing some code without MPI when running under MPI,
you can use the class netket_disable_mpi below

Example:

>>> @onlyif_mpi
>>> def test_my_parallel_function():
>>>     x = compute_some_stuff_with_mpi
>>>     with netket_disable_mpi():
>>>         x_serial = compute_some_stuff_withjout_mpi

"""


class netket_disable_mpi:
    """
    Temporarily disables MPI functions inside of NetKet, tricking NetKet
    into executing all code on every rank independently.

    Example:

    >>> with netket_disable_mpi():
    >>>     run_code

    """

    def __enter__(self):
        self._orig_nodes = nk.utils.mpi.n_nodes
        nk.utils.mpi.n_nodes = 1
        nk.utils.mpi.mpi.n_nodes = 1
        nk.utils.mpi.primitives.n_nodes = 1

    def __exit__(self, exc_type, exc_value, traceback):
        nk.utils.mpi.n_nodes = self._orig_nodes
        nk.utils.mpi.mpi.n_nodes = self._orig_nodes
        nk.utils.mpi.primitives.n_nodes = self._orig_nodes


def hash_for_seed(obj):
    """
    Hash any object into an int that can be used in `np.random.seed`, and does not change between Python sessions.

    Args:
      obj: any object with `repr` defined to show its states.
    """

    bs = repr(obj).encode()
    out = 0
    for b in bs:
        out = (out * 256 + b) % 4294967291  # Output in [0, 2**32 - 1]
    return out
