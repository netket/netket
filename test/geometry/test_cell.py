import jax
import jax.numpy as jnp
import netket as nk


def test_distance_jit():
    cell = nk.experimental.geometry.Cell(d=1, L=1.0, pbc=True)
    dist = cell.distance(jnp.array([0.1]), jnp.array([0.9]))
    assert jnp.isclose(dist, 0.2)
    dist_jit = jax.jit(cell.distance)(jnp.array([0.1]), jnp.array([0.9]))
    assert jnp.isclose(dist_jit, 0.2)


def test_freespace_distance():
    fs = nk.experimental.geometry.FreeSpace(d=2)
    r1 = jnp.array([1.0, 0.0])
    r2 = jnp.array([0.0, 1.0])
    assert jnp.isclose(fs.distance(r1, r2), jnp.sqrt(2.0))
