"""Tests of the multi-process harness in :mod:`test.multiprocess`."""

import os

import jax
import pytest

from ..multiprocess import distributed


@distributed
def topology():
    return jax.process_index(), jax.process_count(), jax.local_device_count()


@distributed
def fail_on_rank_1():
    if jax.process_index() == 1:
        raise RuntimeError("kaboom")


@distributed
def die_on_rank_1():
    if jax.process_index() == 1:
        os._exit(3)


def test_topology():
    assert topology() == [(0, 2, 2), (1, 2, 2)]


def test_errors_are_reported_and_the_cluster_survives():
    with pytest.raises(AssertionError, match="kaboom"):
        fail_on_rank_1()
    assert topology() == [(0, 2, 2), (1, 2, 2)]


def test_a_crashed_cluster_is_restarted():
    with pytest.raises(AssertionError, match=r"ranks \[1\] did not return"):
        die_on_rank_1()
    assert topology() == [(0, 2, 2), (1, 2, 2)]
