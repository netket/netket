import numpy as np
import jax.numpy as jnp
from netket.experimental.operator import (
    ParticleNumberConservingFermioperator2nd,
    ParticleNumberAndSpinConservingFermioperator2nd,
    FermionOperator2nd,
    FermiHubbardJax,
)
from netket.hilbert import SpinOrbitalFermions
from netket.graph import Hypercube
from netket.operator.fermion import destroy, create, number

from functools import partial

import pytest

from ..common import skipif_distributed


def _cast_normal_order(A):
    idx = np.array(np.where(A)).T
    idx_create = idx[:, : idx.shape[1] // 2]
    idx_destroy = idx[:, idx.shape[1] // 2 :]
    mask = (np.diff(idx_destroy) > 0).any(axis=1) | (np.diff(idx_create) > 0).any(
        axis=1
    )
    A[idx[mask].T] = 0.0
    return A


@pytest.mark.slow
@skipif_distributed
@pytest.mark.parametrize("desc", [True, False])
def test_pnc(desc):
    N = 6
    n = 3
    cutoff = 0.1
    key = np.random.randint(2**32)
    rng = np.random.default_rng(key)

    c = rng.normal()
    hij = rng.normal(size=(N,) * 2)
    hijkl = rng.normal(size=(N,) * 4)
    hijklmn = rng.normal(size=(N,) * 6)
    if desc:
        hijkl = _cast_normal_order(hijkl)
        hijklmn = _cast_normal_order(hijklmn)

    hi = SpinOrbitalFermions(N, n_fermions=n)

    terms = []
    weights = []
    terms = terms + [""]
    weights = weights + [c]
    ij = jnp.where(jnp.abs(hij) > cutoff)
    terms = terms + [f"{i}^ {j}" for i, j in list(zip(*ij))]
    weights = weights + list(hij[ij])
    ijkl = jnp.where(jnp.abs(hijkl) > cutoff)
    terms = terms + [f"{i}^ {j}^ {k} {l}" for i, j, k, l in list(zip(*ijkl))]
    weights = weights + list(hijkl[ijkl])
    ijklmn = jnp.where(jnp.abs(hijklmn) > cutoff)
    terms = terms + [
        f"{i}^ {j}^ {k}^ {l} {m} {n}" for i, j, k, l, m, n in list(zip(*ijklmn))
    ]
    weights = weights + list(hijklmn[ijklmn])
    ha = FermionOperator2nd(hi, terms=terms, weights=weights)
    if desc:
        factory = (
            ParticleNumberConservingFermioperator2nd._from_sparse_arrays_normal_order
        )
    else:
        factory = ParticleNumberConservingFermioperator2nd.from_sparse_arrays
    ha2 = factory(
        hi,
        [
            c,
            hij * (jnp.abs(hij) > cutoff),
            hijkl * (jnp.abs(hijkl) > cutoff),
            hijklmn * (jnp.abs(hijklmn) > cutoff),
        ],
    )
    np.testing.assert_allclose(ha.to_dense(), ha2.to_dense())

    ha3 = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(ha)
    np.testing.assert_allclose(ha.to_dense(), ha3.to_dense())

    ha4 = ha3.to_fermionoperator2nd()
    np.testing.assert_allclose(ha.to_dense(), ha4.to_dense())


@pytest.mark.slow
@skipif_distributed
@pytest.mark.parametrize("N", [5])
@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("s", [1 / 2, 1, 3 / 2])
def test_pnc_spin(N, n, s):
    # N = 7
    # n = 3
    # s = 1/2
    cutoff = 1e-4
    key = np.random.randint(2**32)
    rng = np.random.default_rng(key)

    hijkl = rng.normal(size=(N,) * 4)
    hij = rng.normal(size=(N,) * 2)
    c = rng.normal()

    n_spin_subsectors = int(round(2 * s + 1))

    hi = SpinOrbitalFermions(N, s=s, n_fermions_per_spin=(n,) * n_spin_subsectors)

    terms = []
    weights = []
    terms = terms + [""]
    weights = weights + [c]
    _f = lambda j, i: i + j * N
    idx_maps = [partial(_f, j) for j in range(n_spin_subsectors)]  # spin sectors
    for s in idx_maps:
        ij = jnp.where(jnp.abs(hij) > cutoff)
        terms = terms + [f"{s(i)}^ {s(j)}" for i, j in list(zip(*ij))]
        weights = weights + list(hij[ij])
    for s1 in idx_maps:
        for s2 in idx_maps:
            ijkl = jnp.where(jnp.abs(hijkl) > cutoff)
            terms = terms + [
                f"{s1(i)}^ {s2(j)}^ {s2(k)} {s1(l)}" for i, j, k, l in list(zip(*ijkl))
            ]
            weights = weights + list(hijkl[ijkl])
    ha = FermionOperator2nd(hi, terms=terms, weights=weights)

    ha2 = ParticleNumberAndSpinConservingFermioperator2nd._from_sparse_arrays_normal_order_all_sectors(
        hi, [c, hij * (jnp.abs(hij) > cutoff), hijkl * (jnp.abs(hijkl) > cutoff)]
    )
    np.testing.assert_allclose(ha.to_dense(), ha2.to_dense())

    ha3 = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(ha)
    np.testing.assert_allclose(ha.to_dense(), ha3.to_dense())


def test_fermihubbard():
    t = 1.23
    U = 3.14
    g = Hypercube(3, 2)
    hi = SpinOrbitalFermions(n_orbitals=g.n_nodes, s=1 / 2, n_fermions_per_spin=(2, 2))

    ha = FermiHubbardJax(hilbert=hi, graph=g, t=t, U=U)

    def c(site, sz):
        return destroy(hi, site, sz=sz)

    def cdag(site, sz):
        return create(hi, site, sz=sz)

    def nc(site, sz):
        return number(hi, site, sz=sz)

    ha2 = 0.0
    for sz in (-1, 1):
        for u, v in g.edges():
            ha2 += -t * cdag(u, sz) @ c(v, sz) - t * cdag(v, sz) @ c(u, sz)
    for u in g.nodes():
        ha2 += U * nc(u, 1) @ nc(u, -1)

    ha3 = ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(ha2)
    ha4 = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(ha2)

    np.testing.assert_allclose(ha2.to_dense(), ha.to_dense())
    np.testing.assert_allclose(ha2.to_dense(), ha3.to_dense())
    np.testing.assert_allclose(ha2.to_dense(), ha4.to_dense())
