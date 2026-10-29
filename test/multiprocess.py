"""Run a function in several ``jax.distributed`` processes from a plain pytest run.

:func:`run_distributed` spawns ``nproc`` fresh processes, configures jax and calls
``jax.distributed.initialize`` in each of them *before* netket is imported, runs
``worker(rank, *args)`` and returns the per-rank results, indexed by rank::

    from test.multiprocess import run_distributed

    def _worker(rank, x):
        import jax
        return jax.process_count() * x

    def test_something():
        assert run_distributed(_worker, 5, nproc=2) == [10, 10]

The processes are *spawned*, so ``worker`` must be defined at the top level of a
module and its arguments and return value must be picklable: return numpy or
plain Python data, not jax arrays. The spawned processes re-import the module
defining ``worker``, so that module must not run jax computations, or import
netket, at import time.

Test files using it are named ``*_multiprocess.py``. Every call starts new python
interpreters and recompiles everything, so group the checks of a file into as few
calls as possible, e.g. in a module-scoped fixture.

This replaces running the whole suite under ``djaxrun``, which still works.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import random
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_PORT_RANGE = Path("/proc/sys/net/ipv4/ip_local_port_range")
# The repository root, so that the spawned processes can import `test.*`.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])


def _free_port() -> int:
    """A free port for the jax.distributed coordinator, below the kernel's
    ephemeral range.

    The coordinator binds the port only seconds later, after the spawn and the
    jax import. An ephemeral port (what binding port 0 returns) can be handed to
    another socket in between -- under xdist, other tests' gloo or gRPC
    connections -- and then the coordinator fails to bind and every rank blocks
    until jax's initialization timeout. The kernel never hands out ports below
    the ephemeral range on its own, so only an explicit bind can take those.
    """
    lo = int(_PORT_RANGE.read_text().split()[0]) if _PORT_RANGE.exists() else 32768
    start = max(1024, lo - 8192)
    rng = random.SystemRandom()
    for _ in range(100):
        port = rng.randrange(start, lo)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free coordinator port in [{start}, {lo})")


def _entrypoint(
    rank: int,
    nproc: int,
    coordinator: str,
    devices_per_process: int,
    worker: Callable[..., Any],
    args: tuple[Any, ...],
    result_queue: Any,
) -> None:
    import os
    import traceback

    os.environ["JAX_PLATFORMS"] = "cpu"
    # Inherited from the pytest process, where conftest set it for the suite.
    os.environ.pop("NETKET_TEST_DEVICES", None)

    try:
        import jax

        jax.config.update("jax_num_cpu_devices", devices_per_process)
        jax.distributed.initialize(
            coordinator_address=coordinator, num_processes=nproc, process_id=rank
        )
        result_queue.put((rank, "ok", worker(rank, *args)))
    except BaseException:  # report the traceback to the parent
        result_queue.put((rank, "error", traceback.format_exc()))
        raise


def run_distributed(
    worker: Callable[..., Any],
    *args: Any,
    nproc: int = 2,
    devices_per_process: int = 1,
    timeout: float = 300.0,
) -> list[Any]:
    """Run ``worker(rank, *args)`` in ``nproc`` spawned ``jax.distributed`` processes.

    Args:
        worker: Function to run, defined at the top level of a module.
        args: Picklable arguments passed to ``worker`` after the rank.
        nproc: Number of processes.
        devices_per_process: Number of CPU devices of every process.
        timeout: Seconds to wait for all the results.

    Returns:
        The return values of ``worker``, indexed by rank.

    Raises:
        AssertionError: with the remote traceback if a worker fails, or if a
            worker does not return before ``timeout`` (e.g. a deadlock).
    """
    # The spawned processes import `worker` (and this module) by name, as
    # `test.<...>`, using the sys.path that multiprocessing hands them. Put the
    # repository first: pytest's importlib mode does not add it, and behind the
    # standard library `test` would be CPython's own test package (shipped e.g. by
    # Homebrew's python, which macOS spawns the processes with).
    if _REPO_ROOT in sys.path:
        sys.path.remove(_REPO_ROOT)
    sys.path.insert(0, _REPO_ROOT)

    coordinator = f"127.0.0.1:{_free_port()}"
    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue()
    procs = [
        ctx.Process(
            target=_entrypoint,
            args=(
                rank,
                nproc,
                coordinator,
                devices_per_process,
                worker,
                args,
                result_queue,
            ),
        )
        for rank in range(nproc)
    ]
    for p in procs:
        p.start()

    results: dict[int, Any] = {}
    errors: dict[int, str] = {}
    deadline = time.monotonic() + timeout
    try:
        while len(results) + len(errors) < nproc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rank, status, payload = result_queue.get(timeout=min(remaining, 1.0))
            except queue_mod.Empty:
                # A process that dies before reporting (e.g. failing to import the
                # worker) never will: stop waiting once they are all gone.
                if all(p.exitcode is not None for p in procs):
                    break
                continue
            (results if status == "ok" else errors)[rank] = payload
    finally:
        for p in procs:
            p.join(timeout=10 if errors or len(results) < nproc else timeout)
            if p.is_alive():
                p.kill()

    if errors:
        report = "\n\n".join(
            f"--- rank {r} ---\n{tb}" for r, tb in sorted(errors.items())
        )
        raise AssertionError(f"distributed worker(s) failed:\n{report}")

    exitcodes = [p.exitcode for p in procs]
    missing = sorted(set(range(nproc)) - results.keys())
    assert not missing, (
        f"ranks {missing} returned no result (deadlock or crash before reporting, "
        f"see the captured stderr); exit codes={exitcodes}"
    )
    assert all(e == 0 for e in exitcodes), f"worker exit codes: {exitcodes}"
    return [results[r] for r in range(nproc)]
