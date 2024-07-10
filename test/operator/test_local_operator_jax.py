import numpy as np
import jax.numpy as jnp

from netket.operator import LocalOperatorJax
from netket.operator import IsingJax
from netket.hilbert import Spin
from netket.graph import Chain
from netket.operator.spin import identity


def test_casting():
    hi = Spin(s=0.5, N=8)
    g = Chain(length=8)
    H = IsingJax(hi, g, h=1.0)

    a = 42
    b = jnp.pi

    H = b * (H + a)

    assert isinstance(H, LocalOperatorJax)


def test_empty():
    hi = Spin(1 / 2, 1)
    ha = LocalOperatorJax(hi)
    x = hi.all_states()
    xp, mels = ha.get_conn_padded(x)
    assert xp.shape == (len(x), 0, hi.size)
    assert mels.shape == (len(x), 0)


def test_identity():
    hi = Spin(1 / 2, 1)
    ha = identity(hi).to_jax_operator()
    x = hi.all_states()
    xp, mels = ha.get_conn_padded(x)
    assert xp.shape == (len(x), 1, hi.size)
    assert mels.shape == (len(x), 1)
    np.testing.assert_allclose(mels, 1)
    np.testing.assert_allclose(xp, x[:, None])
