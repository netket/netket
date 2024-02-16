import pytest

import netket.nn as nknn

import flax.linen as nn

from .. import common

SEED = 123
pytestmark = common.skipif_mpi


def test_deprecated_stuff():
    with pytest.warns(FutureWarning):

        class TestModule1(nknn.Module):
            pass

    with pytest.warns(FutureWarning):

        class TestModule2(nn.Module):
            @nknn.compact
            def __call__(self, x):
                pass


def test_deprecated_layers():
    with pytest.warns(FutureWarning):
        module = nknn.Dense(features=3, dtype=complex)

    with pytest.raises(KeyError):
        nknn.Dense(features=3, param_dtype=complex)

    module2 = nn.Dense(features=3, param_dtype=complex)

    assert module == module2
