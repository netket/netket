from typing import Literal
import os

import pytest
import jax

@pytest.fixture
def _mpi_size(request):
    """
    Fixture returning the number of MPI nodes detected by NetKet
    """

    import netket as nk

    return nk.utils.mpi.n_nodes


@pytest.fixture
def _mpi_rank(request):
    """
    Fixture returning the MPI rank detected by NetKet
    """

    import netket as nk

    return nk.utils.mpi.rank


@pytest.fixture
def _mpi_comm(request):
    """
    Fixture returning the MPI communicator used by NetKet
    """

    from netket.utils.mpi import MPI_py_comm

    return MPI_py_comm


@pytest.fixture
def _device_count(request):
    """
    Fixture returning the number of MPI nodes detected by NetKet
    """

    from netket.jax import sharding

    return sharding.device_count()


def parse_clearcache(s: str) -> int | Literal["auto", "logical"]:
    if s in ("auto", None):
        if os.environ.get("CI", False):
            return 200
        else:
            return None
    elif s is not None:
        s = int(s)

    if s == 0:
        return None
    else:
        return s

def pytest_addoption(parser):
    parser.addoption(
        "--clear-cache-every",
        action="store",
        dest="clear_cache_every",
        metavar="clear_cache_every",
        type=parse_clearcache,
        default="auto",
        help="mpow: single, all, 1,2,3...",
    )


_n_test_since_reset : int = 0

@pytest.fixture(autouse=True)
def clear_jax_cache(request):
    """Fixture to clear jax cache every a certain number of tests.

    Used to not go OOM on Github Actions
    """
    # Setup: fill with any logic you want

    yield # this is where the testing happens

    # Teardown : fill with any logic you want
    clear_cache_every = request.config.getoption("--clear-cache-every")
    if clear_cache_every is not None:
        global _n_test_since_reset
        _n_test_since_reset += 1

        if _n_test_since_reset > clear_cache_every:
            jax.clear_caches()
            _n_test_since_reset = 0
