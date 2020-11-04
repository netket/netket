from mpi4py import MPI
import pytest

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test MPI")
def test_is_running_with_multiple_procs():
    msg = "The Test_MPI tests should be run with more than one MPI process"
    assert size > 1, msg


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test MPI")
def test_mpi_setup():
    recv = comm.bcast(rank)
    assert recv == 0, "rank={}".format(rank)

    ranks = comm.allgather(rank)
    assert len(ranks) == size
    assert ranks == list(range(size))
