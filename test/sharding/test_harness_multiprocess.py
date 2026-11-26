"""Tests of the multi-process harness in :mod:`test.multiprocess`."""

import os

import pytest


def _topology():
    import jax

    return jax.process_index(), jax.process_count(), jax.local_device_count()


def _fail_on_rank_1():
    import jax

    if jax.process_index() == 1:
        raise RuntimeError("kaboom")


def _die_on_rank_1():
    import jax

    if jax.process_index() == 1:
        os._exit(3)


def test_topology(distributed):
    assert distributed.run(_topology) == [(0, 2, 2), (1, 2, 2)]


def test_errors_are_reported_and_the_cluster_survives(distributed):
    with pytest.raises(AssertionError, match="kaboom"):
        distributed.run(_fail_on_rank_1)
    assert distributed.run(_topology) == [(0, 2, 2), (1, 2, 2)]


def test_a_crashed_cluster_is_restarted(distributed):
    with pytest.raises(AssertionError, match=r"ranks \[1\] did not return"):
        distributed.run(_die_on_rank_1)
    assert distributed.run(_topology) == [(0, 2, 2), (1, 2, 2)]
