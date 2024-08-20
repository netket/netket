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


def parse_clearcache(s: str) -> int | None:
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
        help="...",
    )
    parser.addoption(
        "--arnn_test_rate",
        type=float,
        default=0.2,
        help="rate of running a test for ARNN",
    )
    parser.addoption(
        "--jax-distributed-mpi",
        action="store_true",
        default=False,
        help="Enable JAX distributed initialization",
    )


_n_test_since_reset: int = 0
_clear_cache_every: int = 0


@pytest.fixture(autouse=True)
def clear_jax_cache(request):
    """Fixture to clear jax cache every a certain number of tests.

    Used to not go OOM on Github Actions
    """
    # Setup: fill with any logic you want

    # this is where the testing happens
    yield

    # Teardown : fill with any logic you want
    if _clear_cache_every is not None:
        global _n_test_since_reset
        _n_test_since_reset += 1

        if _n_test_since_reset > _clear_cache_every:
            jax.clear_caches()
            _n_test_since_reset = 0


def pytest_configure(config):
    global _clear_cache_every
    _clear_cache_every = config.getoption("--clear-cache-every")
    if _clear_cache_every is not None:
        print(f"Clearing jax cache every {_clear_cache_every} tests")

    if config.getoption("--jax-distributed-mpi"):
        print("Initializing JAX distributed...")
        import jax

        jax.config.update("jax_cpu_collectives_implementation", "mpi")
        jax.distributed.initialize(cluster_detection_method="mpi4py")

        default_string = f"r{jax.process_index()}/{jax.process_count()} - "
        print(default_string, jax.devices())
        print(default_string, jax.local_devices())
