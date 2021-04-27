from .. import common
import pytest

pytestmark = [common.skipif_mpi, pytest.mark.webtest]
