import pytest

from ..multiprocess import DistributedCluster


@pytest.fixture(scope="session")
def distributed():
    """2 ``jax.distributed`` processes with 2 CPU devices each, shared by all tests
    (and started on first use). See :mod:`test.multiprocess`."""
    cluster = DistributedCluster(nproc=2, devices_per_process=2)
    yield cluster
    cluster.close()
