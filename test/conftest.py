import os
from pathlib import Path

import pytest
import jax


@pytest.fixture
def _device_count(request):
    """
    Fixture returning the number of MPI nodes detected by NetKet
    """

    import jax

    return jax.device_count()


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
    import os

    global _clear_cache_every
    _clear_cache_every = config.getoption("--clear-cache-every")

    djaxrun_mode = os.environ.get("DJAXRUN_MODE", "0") == "1"

    if _clear_cache_every is not None:
        print(f"Clearing jax cache every {_clear_cache_every} tests")

    if config.getoption("--jax-cpu-disable-async-dispatch"):
        print("Disabling async CPU dispatch...")
        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)

    if config.getoption("--jax-distributed") or djaxrun_mode:
        print("\n---------------------------------------------")
        print("Initializing JAX distributed using GLOO...")
        import jax

        jax.config.update("jax_cpu_collectives_implementation", "gloo")
        jax.distributed.initialize()

        default_string = f"r{jax.process_index()}/{jax.process_count()} - "
        print(default_string, jax.devices())
        print(default_string, jax.local_devices())
        print("---------------------------------------------\n", flush=True)

        os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"

        import netket as nk

        if not nk.config.netket_experimental_sharding:
            raise ValueError("problem")
    else:
        import netket as nk

        if nk.config.netket_experimental_sharding:
            import jax

            print("\n---------------------------------------------")
            print(f"Testing under non-distributed sharding with : {jax.devices()} ...")
            print("---------------------------------------------\n", flush=True)
        else:
            print("\n---------------------------------------------")
            print("Testing under standard setup ...")
            print("---------------------------------------------\n", flush=True)
