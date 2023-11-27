import pytest


@pytest.fixture
def _mpi_size(request):
    """
    Fixture returning the number of MPI nodes detected by NetKet
    """

    import netket as nk

    return nk.utils.mpi.n_nodes


@pytest.fixture
def _mpi_rank(request):
    """
    Fixture returning the MPI rank detected by NetKet
    """

    import netket as nk

    return nk.utils.mpi.rank


@pytest.fixture
def _mpi_comm(request):
    """
    Fixture returning the MPI communicator used by NetKet
    """

    from netket.utils.mpi import MPI_py_comm

    return MPI_py_comm


def pytest_addoption(parser):
    parser.addoption(
        "--arnn_test_rate",
        type=float,
        default=0.1,
        help="rate of running a test for ARNN",
    )
