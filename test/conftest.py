import pytest
import netket as nk


@pytest.fixture
def _mpi_size(request):
    """
    Fixture returning the number of MPI nodes detected by NetKet
    """
    return nk.utils.mpi.n_nodes


@pytest.fixture
def _mpi_rank(request):
    """
    Fixture returning the MPI rank detected by NetKet
    """
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
        "--legacy",
        action="store_true",
        dest="legacy",
        default=False,
        help="enable legacy tests",
    )

    parser.addoption(
        "--legacy-only",
        action="store_true",
        dest="only_legacy",
        default=False,
        help="enable legacy tests and disable everything else",
    )


def pytest_configure(config):
    if not config.option.legacy:
        setattr(config.option, "markexpr", "not legacy")

    if config.option.only_legacy:
        setattr(config.option, "markexpr", "legacy")

    if nk.utils.mpi.n_nodes > 1:
        setattr(config.option, "markexpr", "not legacy")
