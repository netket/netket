import os
from pathlib import Path

import pytest
import jax


@pytest.fixture
def _device_count(request):
    """
    Fixture returning the number of devices among which NetKet distributes
    the calculation (1 if sharding is disabled).
    """
    import netket as nk

    return jax.device_count() if nk.config.netket_sharding else 1


@pytest.fixture
def tmp_path_distributed(request, tmp_path):
    """
    Equivalent to tmp_path, but works well with jax distirbuted
    """
    import netket as nk

    if nk.config.netket_experimental_sharding:
        rng_key = nk.jax.PRNGKey()
        val = jax.random.randint(rng_key, (), minval=0, maxval=999999999)
        val = int(val)

        tmp_path = Path("/tmp/netket_tests") / Path(str(val))

    if jax.process_index() == 0:
        tmp_path.mkdir(parents=True, exist_ok=True)

    return tmp_path


def parse_clearcache(s: str) -> int | None:
    if s in ("auto", None):
        if os.environ.get("CI", False):
            return 40
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
        "--devices",
        type=int,
        default=2,
        help="Number of CPU devices to run the tests on (default: 2)",
    )
    parser.addoption(
        "--jax-distributed",
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
    # jax fixes its devices when it initialises its backend, at the first
    # jax.devices() call, e.g. when importing netket below: set them up first.
    # By default the suite runs on several CPU devices, so that every test
    # exercises the sharded code paths.
    jax.config.update("jax_num_cpu_devices", config.getoption("--devices"))
    distributed = os.environ.get("DJAXRUN", "0") == "1" or config.getoption(
        "--jax-distributed"
    )
    if distributed:
        # Every process runs the tests itself: no xdist workers, which would
        # each try to join the jax.distributed session as the same rank.
        if hasattr(config.option, "dist"):
            config.option.dist = "no"
        jax.distributed.initialize()

    if config.getoption("--jax-cpu-disable-async-dispatch"):
        print("Disabling async CPU dispatch...")
        jax.config.update("jax_cpu_enable_async_dispatch", False)

    # Not in pyproject.toml: pytest resolves the ini filters before loading this
    # conftest, and resolving these imports netket, which initialises jax.
    for category in (
        "HolomorphicUndeclaredWarning",
        "UnoptimalSRtWarning",
        "InitializePeriodicLatticeOnSmallLatticeWarning",
    ):
        config.addinivalue_line("filterwarnings", f"ignore::netket.errors.{category}")

    global _clear_cache_every
    _clear_cache_every = config.getoption("--clear-cache-every")
    if _clear_cache_every is not None:
        print(f"Clearing jax cache every {_clear_cache_every} tests")

    import netket as nk

    print("\n---------------------------------------------")
    if distributed:
        assert nk.config.netket_sharding
        print(f"r{jax.process_index()}/{jax.process_count()} -", jax.devices())
    else:
        print(f"Testing on {jax.device_count()} devices: {jax.devices()}")
    print("---------------------------------------------\n", flush=True)
