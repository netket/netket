# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import netket as nk
import numpy as np
from numpy.testing import assert_equal

from netket.utils import group

from itertools import product

from .. import common

pytestmark = common.skipif_mpi

# Tests for group.py and overrides in subclasses

planar_families = [group.planar.C, group.planar.D]
planars = [fn(n) for fn in planar_families for n in range(1, 9)] + [
    group.planar.reflection_group(23),
    group.planar.glide_group([0.5, 0]).replace(unit_cell=np.eye(2)),
]
planars_proper = [True] * 8 + [False] * 10
uniaxial_families = [group.axial.C, group.axial.Ch, group.axial.S]
uniaxials = [
    fn(n, axis=np.random.standard_normal(3))
    for fn in uniaxial_families
    for n in range(1, 9)
]
uniaxials_proper = [True] * 8 + [False] * 16
impropers = [
    group.axial.inversion_group(),
    group.axial.reflection_group(axis=np.random.standard_normal(3)),
    group.axial.glide_group(axis=[0, 0, 0.5], trans=[0.5, 0, 0]).replace(
        unit_cell=np.eye(3)
    ),
]
impropers_proper = [False] * 3
screws = [
    group.axial.screw_group(360 / n, [1 / n, 0, 0]).replace(unit_cell=np.eye(3))
    for n in range(1, 9)
]
screws_proper = [True] * 8
biaxial_families = [group.axial.Cv, group.axial.D, group.axial.Dh, group.axial.Dd]
axes1 = np.random.standard_normal((32, 3))
axes2 = np.cross(axes1, np.random.standard_normal((32, 3)))
biaxials = [
    fn(n, axis=axes1[i], axis2=axes2[i])
    for i, (fn, n) in enumerate(product(biaxial_families, range(1, 9)))
]
biaxials_proper = [False] * 8 + [True] * 8 + [False] * 16
cubics = [
    group.cubic.T(),
    group.cubic.Td(),
    group.cubic.Th(),
    group.cubic.O(),
    group.cubic.Oh(),
    group.cubic.Fd3m(),
]
cubics_proper = [True, False, False, True, False, False]
point_groups = planars + uniaxials + screws + biaxials + impropers + cubics
proper = (
    planars_proper
    + uniaxials_proper
    + screws_proper
    + biaxials_proper
    + impropers_proper
    + cubics_proper
)
perms = [
    nk.graph.Hypercube(2, n_dim=3).point_group(),
    nk.graph.Square(4).space_group(),
]
groups = point_groups + perms


def assert_allclose(a, b, rtol=0, atol=1e-15, **kwargs):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)


@pytest.mark.parametrize("grp", groups)
def test_inverse(grp):
    inv = grp.inverse
    for i, j in enumerate(inv):
        assert_equal(grp._canonical(grp[i] @ grp[j]), grp._canonical(group.Identity()))


@pytest.mark.parametrize("grp", groups)
def test_product_table(grp):
    pt = grp.product_table
    # u = g^-1 h  ->  gu = h
    for i in range(len(grp)):
        for j in range(len(grp)):
            assert_equal(grp._canonical(grp[i] @ grp[pt[i, j]]), grp._canonical(grp[j]))


@pytest.mark.parametrize("grp", groups)
def test_conjugacy_table(grp):
    ct = grp.conjugacy_table
    inv = grp.inverse
    for i in range(len(grp)):
        for j, jinv in enumerate(inv):
            assert_equal(
                grp._canonical(grp[jinv] @ grp[i] @ grp[j]),
                grp._canonical(grp[ct[i, j]]),
            )


# Conjugacy class sizes and irrep dimensions taken from
# https://en.wikipedia.org/wiki/List_of_character_tables_for_chemically_important_3D_point_groups
details = [
    (group.planar.C(3), [1, 1, 1], [1, 1, 1]),
    (group.planar.C(4), [1, 1, 1, 1], [1, 1, 1, 1]),
    (group.planar.D(3), [1, 2, 3], [1, 1, 2]),
    (group.planar.D(4), [1, 2, 1, 2, 2], [1, 1, 1, 1, 2]),
    (group.planar.D(6), [1, 2, 2, 1, 3, 3], [1, 1, 1, 1, 2, 2]),
    (group.axial.C(4), [1, 1, 1, 1], [1, 1, 1, 1]),
    (group.axial.Ch(4), [1, 1, 1, 1] * 2, [1, 1, 1, 1] * 2),
    (group.axial.Cv(4), [1, 2, 1, 2, 2], [1, 1, 1, 1, 2]),
    (group.axial.S(4), [1, 1, 1, 1], [1, 1, 1, 1]),
    (group.axial.D(4), [1, 2, 1, 2, 2], [1, 1, 1, 1, 2]),
    (group.axial.Dh(4), [1, 2, 1, 2, 2] * 2, [1, 1, 1, 1, 2] * 2),
    (group.axial.Dd(4), [1, 2, 2, 2, 1, 4, 4], [1, 1, 1, 1, 2, 2, 2]),
    (group.cubic.T(), [1, 4, 4, 3], [1, 1, 1, 3]),
    (group.cubic.Td(), [1, 8, 3, 6, 6], [1, 1, 2, 3, 3]),
    (group.cubic.Th(), [1, 4, 4, 3] * 2, [1, 1, 1, 3] * 2),
    (group.cubic.O(), [1, 6, 3, 8, 6], [1, 1, 2, 3, 3]),
    (group.cubic.Oh(), [1, 6, 3, 8, 6] * 2, [1, 1, 2, 3, 3] * 2),
    (group.cubic.Fd3m(), [1, 6, 3, 8, 6] * 2, [1, 1, 2, 3, 3] * 2),
]


@pytest.mark.parametrize("grp,cls,dims", details)
def test_conjugacy_class(grp, cls, dims):
    classes, _, _ = grp.conjugacy_classes
    class_sizes = classes.sum(axis=1)

    assert_equal(np.sort(class_sizes), np.sort(cls))


@pytest.mark.parametrize("grp,cls,dims", details)
def test_character_table(grp, cls, dims):
    classes, _, _ = grp.conjugacy_classes
    class_sizes = classes.sum(axis=1)
    cht = grp.character_table_by_class

    # check that dimensions match and are sorted
    assert_allclose(
        cht[:, 0], np.sort(dims), atol=1e-10
    )  # this should not require such low tolerance

    # check orthogonality of characters
    # this also requires an high atol. it shouldn't.
    assert_allclose(
        np.eye(len(class_sizes)) * len(grp),
        cht @ np.diag(class_sizes) @ cht.T.conj(),
        atol=1e-10,
    )

    # check orthogonality of columns of the character table
    column_prod = cht.T.conj() @ cht
    assert_allclose(np.diag(np.diag(column_prod)), column_prod, atol=1e-10)


@pytest.mark.parametrize("grp,cls,dims", details)
def test_irrep_matrices(grp, cls, dims):
    irreps = grp.irrep_matrices()
    characters = grp.character_table()
    true_product_table = grp.product_table[grp.inverse]
    for i, irrep in enumerate(irreps):
        # characters are the traces of the irrep matrices
        assert_allclose(np.trace(irrep, axis1=1, axis2=2), characters[i], atol=1e-8)
        # irrep matrices respect the group multiplication rule
        assert_allclose(
            irrep[true_product_table, :, :],
            np.einsum("iab,jbc->ijac", irrep, irrep),
            atol=1e-7,
        )


# Check that rotation subgroups only contain rotations
@pytest.mark.parametrize("i,grp", list(enumerate(point_groups)))
def test_rotation_group(i, grp):
    rot = grp.rotation_group()
    assert len(rot) == (len(grp) if proper[i] else len(grp) // 2)
    for i in rot:
        assert isinstance(i, group.Identity) or i.is_proper
        assert str(i)[:3] in {"Id(", "Rot", "Scr"}


# Test for naming and generating 2D and 3D PGSymmetries

names = [
    (
        group.planar.rotation(47),
        np.asarray([[0.6819983601, -0.7313537016], [0.7313537016, 0.6819983601]]),
        "Rot(47°)",
    ),
    (
        group.planar.reflection(78),
        np.asarray([[-0.9135454576, 0.4067366431], [0.4067366431, 0.9135454576]]),
        "Refl(78°)",
    ),
    (
        group.axial.rotation(34, [1, 1, 2]),
        np.asarray(
            [
                [0.8575313105, -0.4280853559, 0.2852770227],
                [0.4850728317, 0.8575313105, -0.1713020711],
                [-0.1713020711, 0.2852770227, 0.9430125242],
            ]
        ),
        "Rot(34°)[1,1,2]",
    ),
    (
        group.axial.reflection([1, 4, 2]),
        np.asarray([[19, -8, -4], [-8, -11, -16], [-4, -16, 13]]) / 21,
        "Refl[1,4,2]",
    ),
    (
        group.axial.rotoreflection(8, [2, 3, 1]),
        np.asarray(
            [
                [0.4216200491, -0.8901676053, -0.1727372824],
                [-0.8157764537, -0.2891899754, -0.5008771663],
                [-0.3959107372, -0.3520948631, 0.8481060638],
            ]
        ),
        "RotoRefl(8°)[2,3,1]",
    ),
    (group.axial.inversion(), -np.eye(3), "Inv()"),
]


@pytest.mark.parametrize("symm,W,name", names)
def test_naming(symm, W, name):
    assert_allclose(symm.matrix, W, atol=2.0e-10)
    assert_allclose(0.0, symm.translation)
    assert str(symm) == name


names_nonsymm = [
    (
        group.planar.rotation(30).change_origin([1, 0]),
        np.asarray([[0.75**0.5, -0.5], [0.5, 0.75**0.5]]),
        np.asarray([1 - 0.75**0.5, -0.5]),
        "Rot(30°)O[1,0]",
    ),
    (
        group.planar.glide([0.5, 0]),
        np.diag([1.0, -1.0]),
        np.asarray([0.5, 0.0]),
        "Glide[1/2,0]",
    ),
    (
        group.planar.reflection(0).change_origin([0, 0.5]),
        np.diag([1.0, -1.0]),
        np.asarray([0.0, 1.0]),
        "Refl(0°)O[0,1/2]",
    ),
    (
        group.axial.rotation(-30, [0, 0, 1]).change_origin([0.5, 0, 0]),
        np.asarray([[0.75**0.5, 0.5, 0], [-0.5, 0.75**0.5, 0], [0, 0, 1]]),
        np.asarray([(1 - 0.75**0.5) / 2, 0.25, 0]),
        "Rot(30°)[0,0,-1]O[1/2,0,0]",
    ),
    (
        group.axial.screw(-30, [0, 0, 1], origin=[1 / 3, 0, 0]),
        np.asarray([[0.75**0.5, 0.5, 0], [-0.5, 0.75**0.5, 0], [0, 0, 1]]),
        np.asarray([(1 - 0.75**0.5) / 3, 1 / 6, 1]),
        "Screw(-30°)[0,0,1]O[1/3,0,0]",
    ),
    (
        group.PGSymmetry([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], [0.25, 0.25, 0.25]),
        np.asarray([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
        np.asarray([0.25, 0.25, 0.25]),
        "Screw(180°)[0,1/4,1/4]O[1/8,0,0]",
    ),
    (
        group.axial.reflection([1, 1, 0]).change_origin([1, 0, 1]),
        np.asarray([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
        np.asarray([1, 1, 0]),
        "Refl[1,1,0]O[1/2,1/2,0]",
    ),
    (
        group.axial.glide(
            axis=[0, 0, 1], trans=[1, 3**0.5 / 12, 0], origin=[0.5, 0.5, 1]
        ),
        np.diag([1, 1, -1]),
        np.asarray([1, 3**0.5 / 12, 2]),
        "Glide[1,√3/12,0]ax[0,0,1]O[0,0,1]",
    ),
]


@pytest.mark.parametrize("symm,W,w,name", names_nonsymm)
def test_naming_nonsymm(symm, W, w, name):
    assert_allclose(symm.matrix, W)
    assert_allclose(
        symm.translation,
        w,
    )
    assert str(symm) == name


# Nonsymmorphic symmetries
@pytest.mark.parametrize("grp", point_groups)
def test_change_origin(grp):
    origin = np.random.standard_normal(grp.ndim)
    grp_new = grp.change_origin(origin)
    assert_equal(grp_new.product_table, grp.product_table)
    for elem in grp_new:
        assert_allclose(elem(origin), origin, rtol=1e-15)


def test_pyrochlore():
    Fd3m = (
        group.axial.inversion_group().change_origin([1 / 8, 1 / 8, 1 / 8])
        @ group.cubic.Td()
    )
    # closure fails without specifying a unit cell
    with pytest.raises(RuntimeError):
        _ = Fd3m.product_table
    Fd3m = Fd3m.replace(
        unit_cell=np.asarray([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    )
    # canned Oh is listed in a different order
    Oh = group.axial.inversion_group() @ group.cubic.Td()
    # after specifying the unit cell, Fd3m is isomorphic to Oh
    assert_equal(Fd3m.product_table, Oh.product_table)
