"""Run functions on several ``jax.distributed`` processes from a plain pytest run.

A :class:`DistributedCluster` spawns ``nproc`` python processes, each with
``devices_per_process`` CPU devices, that call ``jax.distributed.initialize`` and
then wait for work. ``cluster.run(fn, *args)`` calls ``fn(*args)`` on every
process and returns the results, indexed by rank::

    def _energy(L):
        vs, ha = ...
        return complex(vs.expect(ha).mean)

    def test_ranks_agree(distributed):
        e0, e1 = distributed.run(_energy, 8)
        assert e0 == e1

The processes are started on the first call and then reused by every test, so
netket is imported, and most functions compiled, once per session. The
``distributed`` fixture of ``test/sharding/conftest.py`` provides the shared
cluster. If a call fails without every rank reporting back (a crash, or a rank
stuck in a collective that another one skipped), the processes are killed and the
next call starts new ones.

``fn`` and its arguments are pickled, so ``fn`` must be defined at the top level
of a module, and the arguments and results must be picklable: return numpy or
plain Python data, not jax arrays. Use ``jax.process_index()`` for the rank.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import random
import socket
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The ranks run in lockstep, so once one has returned or died the others should
# follow promptly. Those that have not after this many seconds are taken to be
# stuck in a collective that the first one never entered.
_GRACE_PERIOD = 30.0


class DistributedCluster:
    def __init__(self, nproc: int = 2, devices_per_process: int = 2):
        self.nproc = nproc
        self.devices_per_process = devices_per_process
        self._procs: list[Any] = []
        self._tasks: list[Any] = []

    def run(self, fn: Callable[..., Any], *args: Any, timeout: float = 300.0):
        """Calls ``fn(*args)`` on every rank and returns the results, by rank.

        Raises:
            AssertionError: with the remote tracebacks if ``fn`` raises on some rank,
                or if some rank dies or does not return within ``timeout``.
        """
        if not self._procs:
            self._start()
        for tasks in self._tasks:
            tasks.put((fn, args))

        replies: dict[int, tuple[bool, Any]] = {}
        deadline = time.monotonic() + timeout
        while len(replies) < self.nproc and time.monotonic() < deadline:
            try:
                rank, ok, value = self._replies.get(timeout=0.5)
                replies[rank] = (ok, value)
            except queue.Empty:
                pass
            pending = [p for r, p in enumerate(self._procs) if r not in replies]
            dead = [p for p in pending if p.exitcode is not None]
            if len(dead) == len(pending):
                break
            if replies or dead:
                deadline = min(deadline, time.monotonic() + _GRACE_PERIOD)

        errors = [f"--- rank {r} ---\n{v}" for r, (ok, v) in replies.items() if not ok]
        missing = sorted(set(range(self.nproc)) - replies.keys())
        if missing:
            exitcodes = [p.exitcode for p in self._procs]
            # The surviving ranks may be stuck, and cannot shut jax.distributed
            # down cleanly without the others: kill them all.
            self.close(kill=True)
            errors.append(
                f"ranks {missing} did not return: deadlocked or crashed (exit codes "
                f"{exitcodes}, None if still running). See the captured stderr."
            )
        if errors:
            raise AssertionError(f"{fn.__name__} failed:\n" + "\n\n".join(errors))
        return [replies[r][1] for r in range(self.nproc)]

    def _start(self):
        # The processes unpickle `fn` by importing its module, `test.<...>`, from
        # the sys.path they inherit. Pytest's importlib mode does not put the
        # repository on it, and without it first `test` is CPython's own test
        # package (shipped e.g. by Homebrew's python).
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root in sys.path:
            sys.path.remove(repo_root)
        sys.path.insert(0, repo_root)

        ctx = mp.get_context("spawn")
        coordinator = f"127.0.0.1:{_free_port()}"
        self._replies = ctx.Queue()
        self._tasks = [ctx.Queue() for _ in range(self.nproc)]
        self._procs = [
            ctx.Process(
                target=_serve,
                args=(
                    rank,
                    self.nproc,
                    self.devices_per_process,
                    coordinator,
                    self._tasks[rank],
                    self._replies,
                ),
                daemon=True,
            )
            for rank in range(self.nproc)
        ]
        for p in self._procs:
            p.start()

    def close(self, kill: bool = False):
        """Stops the processes, killing those that do not exit within 10s."""
        for tasks in self._tasks:
            tasks.put(None)
        for p in self._procs:
            p.join(timeout=0 if kill else 10)
            if p.is_alive():
                p.kill()
                p.join()
        self._procs = []


def _serve(rank, nproc, devices_per_process, coordinator, tasks, replies):
    """Main loop of every process of the cluster."""
    try:
        import jax

        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_num_cpu_devices", devices_per_process)
        jax.distributed.initialize(
            coordinator_address=coordinator, num_processes=nproc, process_id=rank
        )
    except BaseException:
        # Reported as the reply to the first task: the tests see why we failed.
        replies.put((rank, False, traceback.format_exc()))
        raise

    while (task := tasks.get()) is not None:
        try:
            fn, args = task
            replies.put((rank, True, fn(*args)))
        except BaseException:
            replies.put((rank, False, traceback.format_exc()))


def _free_port() -> int:
    """A free port for the coordinator, below the kernel's ephemeral range.

    The coordinator binds it only seconds later, once the processes have started
    and imported jax. A port from the ephemeral range (what binding port 0 gives)
    can be handed out to another socket in the meantime, e.g. to other tests'
    connections under xdist, and then every rank blocks until jax times out. The
    kernel never hands out ports below that range by itself.
    """
    port_range = Path("/proc/sys/net/ipv4/ip_local_port_range")
    lo = int(port_range.read_text().split()[0]) if port_range.exists() else 32768
    rng = random.SystemRandom()
    for _ in range(100):
        port = rng.randrange(max(1024, lo - 8192), lo)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port for the jax.distributed coordinator")
