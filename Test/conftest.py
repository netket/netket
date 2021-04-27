import pytest
import netket as nk

skip_if_mpi = pytest.mark.skipif(nk.utils.n_nodes > 1, reason="Only run without MPI")


@pytest.fixture
def _mpi_size(request):
    return nk.utils.n_nodes


@pytest.fixture
def _mpi_rank(request):
    return nk.utils.rank
