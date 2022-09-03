import pytest

import netket as nk
import netket.nn as nknn

import flax.linen as nn

from .. import common

SEED = 123
pytestmark = common.skipif_mpi


def test_deprecated_stuff():
    with pytest.warns(FutureWarning):

        class TestModule(nknn.Module):
            pass

    with pytest.warns(FutureWarning):

        class TestModule(nn.Module):
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


def test_deprecated_dtype_layers():
    g = nk.graph.Square(3)
    with pytest.warns(FutureWarning):
        module = nknn.DenseSymm(g, features=2, dtype=complex)

    with pytest.warns(FutureWarning):
        assert module.dtype == module.param_dtype

    with pytest.warns(FutureWarning):
        module = nknn.DenseEquivariant(g, features=2, dtype=complex)

    with pytest.warns(FutureWarning):
        assert module.dtype == module.param_dtype
