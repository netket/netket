import os
import sys
from pathlib import Path

import pytest
import jax

# The suite runs on several CPU devices by default, so that every test exercises
# the sharded code paths. The device count is fixed when jax initialises its
# backend, which happens when netket is imported (and pytest imports it while
# resolving `filterwarnings`), so it must be chosen here, at conftest import.
# A command-line option is parsed too late for that: `--devices` is read from
# sys.argv and forwarded to the xdist workers through the environment.
DEFAULT_TEST_DEVICES = 2


def _requested_devices() -> int:
    if "NETKET_TEST_DEVICES" in os.environ:
        return int(os.environ["NETKET_TEST_DEVICES"])
    for i, arg in enumerate(sys.argv):
        if arg == "--devices" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if arg.startswith("--devices="):
            return int(arg.split("=", 1)[1])
    return DEFAULT_TEST_DEVICES


if os.environ.get("DJAXRUN", "0") != "1":
    _n_devices = _requested_devices()
    os.environ["NETKET_TEST_DEVICES"] = str(_n_devices)
    jax.config.update("jax_num_cpu_devices", _n_devices)


@pytest.fixture
def _device_count(request):
    """
    Fixture returning the number of devices among which NetKet distributes
    the calculation.
    """
    return jax.device_count()


@pytest.fixture
def tmp_path_distributed(request, tmp_path):
    """
    Equivalent to tmp_path, but works well with jax distirbuted
    """
    import netket as nk

    if jax.process_count() > 1:
        rng_key = nk.jax.PRNGKey()
        val = jax.random.randint(rng_key, (), minval=0, maxval=999999999)
        val = int(val)

        tmp_path = Path("/tmp/netket_tests") / Path(str(val))

    if jax.process_index() == 0:
        tmp_path.mkdir(parents=True, exist_ok=True)

    return tmp_path


def parse_clearcache(s: str) -> int | str | None:
    if s == "file":
        return s
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
        help="Clear the jax caches every N tests, or after every test file with `file` (0: never; auto: 40 on CI)",
    )
    parser.addoption(
        "--devices",
        type=int,
        default=DEFAULT_TEST_DEVICES,
        help="Number of CPU devices to run the tests on (also NETKET_TEST_DEVICES)",
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


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """With `--clear-cache-every file`, clear the jax caches after the last test of
    every file. Under `--dist loadfile` the tests of a file share their compiled
    functions, and the memory they hold is released before the next file."""
    if _clear_cache_every == "file" and (
        nextitem is None or nextitem.path != item.path
    ):
        jax.clear_caches()


@pytest.fixture(autouse=True)
def clear_jax_cache(request):
    """Fixture to clear jax cache every a certain number of tests.

    Used to not go OOM on Github Actions
    """
    # Setup: fill with any logic you want

    # this is where the testing happens
    yield

    # Teardown : fill with any logic you want
    if isinstance(_clear_cache_every, int):
        global _n_test_since_reset
        _n_test_since_reset += 1

        if _n_test_since_reset > _clear_cache_every:
            jax.clear_caches()
            _n_test_since_reset = 0


def pytest_load_initial_conftests(early_config, parser, args):
    """Initialize JAX distributed BEFORE any plugin processes filterwarnings.

    pytest_configure runs too late: the _pytest.warnings plugin resolves
    fully-qualified warning class names (e.g. netket.errors.*) during
    pytest_configure, which imports netket, which calls jax.devices() and
    initialises the XLA backend.  jax.distributed.initialize() must be called
    before that happens, so we use this earlier hook.
    """
    djaxrun_mode = os.environ.get("DJAXRUN", "0") == "1"
    jax_distributed_flag = "--jax-distributed" in args

    if djaxrun_mode or jax_distributed_flag:
        import jax

        print("\n---------------------------------------------")
        print("Initializing JAX distributed...")
        jax.distributed.initialize()
        os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"


def pytest_configure(config):
    import os

    # Kept out of pyproject.toml: resolving these categories imports netket, and
    # pytest resolves the ini filters before loading this conftest, which would
    # initialise jax before the device count above is set.
    for category in (
        "HolomorphicUndeclaredWarning",
        "UnoptimalSRtWarning",
        "InitializePeriodicLatticeOnSmallLatticeWarning",
    ):
        config.addinivalue_line("filterwarnings", f"ignore::netket.errors.{category}")

    global _clear_cache_every
    _clear_cache_every = config.getoption("--clear-cache-every")

    djaxrun_mode = os.environ.get("DJAXRUN", "0") == "1"

    if _clear_cache_every is not None:
        print(f"Clearing jax cache every {_clear_cache_every} tests")

    if config.getoption("--jax-cpu-disable-async-dispatch"):
        print("Disabling async CPU dispatch...")
        import jax

        jax.config.update("jax_cpu_enable_async_dispatch", False)

    if djaxrun_mode or config.getoption("--jax-distributed"):
        # Multiprocess test workers can't share a JAX distributed session
        if hasattr(config.option, "numprocesses"):
            config.option.numprocesses = 0

    if config.getoption("--jax-distributed") or djaxrun_mode:
        import jax

        default_string = f"r{jax.process_index()}/{jax.process_count()} - "
        print(default_string, jax.devices())
        print(default_string, jax.local_devices())
        print("---------------------------------------------\n", flush=True)

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
