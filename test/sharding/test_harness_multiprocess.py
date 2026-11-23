"""Tests of the multi-process harness in :mod:`test.multiprocess`."""

import pytest

from ..multiprocess import run_distributed


def _topology_worker(rank):
    import jax

    return (jax.process_index(), jax.process_count(), jax.device_count())


def _boom_worker(rank):
    raise RuntimeError(f"kaboom from rank {rank}")


def test_run_distributed_returns_per_rank_results():
    assert run_distributed(_topology_worker, nproc=2) == [(0, 2, 2), (1, 2, 2)]
    assert run_distributed(_topology_worker, nproc=2, devices_per_process=2) == [
        (0, 2, 4),
        (1, 2, 4),
    ]


def test_run_distributed_propagates_worker_errors():
    with pytest.raises(AssertionError, match="kaboom"):
        run_distributed(_boom_worker, nproc=2)
