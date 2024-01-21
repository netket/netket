import jax.numpy as jnp

from netket.operator import LocalOperatorJax
from netket.operator import IsingJax
from netket.hilbert import Spin
from netket.graph import Chain


def test_casting():
    hi = Spin(s=0.5, N=8)
    g = Chain(length=8)
    H = IsingJax(hi, g, h=1.0)

    a = 42
    b = jnp.pi

    H = b * (H + a)

    assert isinstance(H, LocalOperatorJax)
