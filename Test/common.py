import pytest
import netket as nk

skipif_mpi = pytest.mark.skipif(nk.utils.n_nodes > 1, reason="Only run without MPI")

onlyif_mpi = pytest.mark.skipif(nk.utils.n_nodes < 2, reason="Only run with MPI")


class one_rank:
    def __enter__(self):
        self._orig_nodes = nk.utils.n_nodes
        nk.utils.n_nodes = 1

    def __exit__(self, exc_type, exc_value, traceback):
        nk.utils.n_nodes = self._orig_nodes
