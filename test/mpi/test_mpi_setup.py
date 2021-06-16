import pytest

from .. import common


@common.onlyif_mpi
def test_mpi_setup(_mpi_rank, _mpi_size, _mpi_comm):
    rank = _mpi_rank
    size = _mpi_size
    comm = _mpi_comm

    recv = comm.bcast(rank)
    assert recv == 0, "rank={}".format(rank)

    ranks = comm.allgather(rank)
    assert len(ranks) == size
    assert ranks == list(range(size))
