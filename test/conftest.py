import os
from pathlib import Path

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


@pytest.fixture
def tmp_path_distributed(request, tmp_path):
    """
    Equivalent to tmp_path, but works well with mpi and jax distirbuted
    """
    import netket as nk
    from netket.utils import mpi

    global MPI_pytest_comm

    if mpi.n_nodes > 1:
        tmp_path = MPI_pytest_comm.bcast(tmp_path, root=0)
    elif nk.config.netket_experimental_sharding:
        rng_key = nk.jax.PRNGKey()
        val = jax.random.randint(rng_key, (), minval=0, maxval=999999999)
        val = int(val)

        tmp_path = Path("/tmp/netket_tests") / Path(str(val))

    if mpi.rank == 0 and jax.process_index() == 0:
        tmp_path.mkdir(parents=True, exist_ok=True)
    if mpi.n_nodes > 1:
        MPI_pytest_comm.barrier()

    return tmp_path


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
        help="Enable JAX distributed initialization using MPI",
    )
    parser.addoption(
        "--jax-distributed-gloo",
        action="store_true",
        default=False,
        help="Enable JAX distributed initialization using GLOO",
    )

    parser.addoption(
        "--jax-cpu-disable-async-dispatch",
        action="store_true",
        default=False,
        help="Disable async cpu dispatch",
    )


_n_test_since_reset: int = 0
_clear_cache_every: int = 0
MPI_pytest_comm = None


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

    if config.getoption("--jax-cpu-disable-async-dispatch"):
        print("Disabling async CPU dispatch...")
        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)

    if config.getoption("--jax-distributed-mpi"):
        print("\n---------------------------------------------")
        print("Initializing JAX distributed using MPI...")
        import jax

        jax.config.update("jax_cpu_collectives_implementation", "mpi")
        jax.distributed.initialize(cluster_detection_method="mpi4py")

        default_string = f"r{jax.process_index()}/{jax.process_count()} - "
        print(default_string, jax.devices())
        print(default_string, jax.local_devices())
        print("---------------------------------------------\n", flush=True)
    elif config.getoption("--jax-distributed-gloo"):
        print("\n---------------------------------------------")
        print("Initializing JAX distributed using GLOO...")
        import jax

        jax.config.update("jax_cpu_collectives_implementation", "gloo")
        jax.distributed.initialize(cluster_detection_method="mpi4py")

        default_string = f"r{jax.process_index()}/{jax.process_count()} - "
        print(default_string, jax.devices())
        print(default_string, jax.local_devices())
        print("---------------------------------------------\n", flush=True)
    else:
        # Check for MPI
        import netket as nk

        if nk.utils.mpi.n_nodes > 1 and not nk.config.netket_experimental_sharding:
            from mpi4py import MPI

            global MPI_pytest_comm
            MPI_pytest_comm = MPI.COMM_WORLD.Create(MPI.COMM_WORLD.Get_group())
            print("Testing under MPI ...")
