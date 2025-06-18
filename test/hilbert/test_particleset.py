import numpy as np
import netket as nk
from netket.experimental.hilbert import ParticleSet, Electron
from netket.experimental.geometry import Cell


def _simple_particleset():
    cell = Cell(d=1, L=(1.0,), pbc=True)
    return ParticleSet([Electron(), Electron(m_z=0.5)], cell)


def test_particleset_size():
    hi = _simple_particleset()
    assert hi.size == 4


def test_positions_indices_and_random_state():
    cell = Cell(d=1, L=(1.0,), pbc=True)
    hi = ParticleSet([Electron(position=(0.2,)), Electron()], cell)

    assert hi.position_indices == (2,)
    assert hi.positions_hilbert.size == 1

    rs = hi.random_state(nk.jax.PRNGKey(0), 3)
    np.testing.assert_allclose(rs[:, 0], 0.2)


def test_fixed_spin_not_sampled():
    cell = Cell(d=1, L=(1.0,), pbc=True)
    hi = ParticleSet([Electron(m_z=0.5), Electron()], cell)

    rs = hi.random_state(nk.jax.PRNGKey(0), 4)
    np.testing.assert_allclose(rs[:, 1], 0.5)
