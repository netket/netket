import pytest

import netket as nk

from .. import common

pytestmark = common.skipif_mpi


def test_deprecated_sr():
    with pytest.warns(FutureWarning):
        nk.optimizer.sr.SRLazyCG()

    with pytest.warns(FutureWarning):
        nk.optimizer.sr.SRLazyGMRES()
