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

import netket as nk

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.nn.initializers import uniform
from netket.utils.group import PermutationGroup
from netket.errors import SymmModuleInvalidInputShape

import pytest


def _setup_symm(symmetries, N, lattice=nk.graph.Chain):
    g = lattice(N)

    hi = nk.hilbert.Spin(1 / 2, g.n_nodes)

    if symmetries == "trans":
        # Only translations, N_symm = N_sites
        perms = g.translation_group()
    else:
        # All chain automorphisms, N_symm = 2 N_sites
        perms = g.space_group()

    return g, hi, perms


@pytest.mark.parametrize("mode", ["fft", "matrix"])
def test_DenseSymm_matrixInput(mode):
    g, hi, perms = _setup_symm("trans", N=8)
    shape = tuple(g.extent)
    perms_matrix = np.array(perms)
    ma = nk.nn.DenseSymm(
        symmetries=perms,
        mode=mode,
        features=2,
        shape=shape,
    )
    ma_matrix = nk.nn.DenseSymm(
        symmetries=perms_matrix,
        mode=mode,
        features=2,
        shape=shape,
    )

    assert hash(ma) == hash(ma_matrix)

    k = jax.random.PRNGKey(0)
    x = hi.numbers_to_states(np.array([0, 1, 2])).reshape(3, 1, hi.size)
    p1 = jax.jit(ma.init)(k, x)
    p2 = jax.jit(ma_matrix.init)(k, x)
    assert nk.jax.tree_size(p1) == nk.jax.tree_size(p2)

    n_symm = perms_matrix.shape[0]
    mask = np.zeros(n_symm)
    mask[np.random.choice(n_symm, n_symm // 2, replace=False)] = 1

    ma_masked = nk.nn.DenseSymm(
        symmetries=perms,
        mode=mode,
        features=2,
        shape=shape,
        mask=mask,
    )
    p3 = jax.jit(ma_masked.init)(k, x)
    assert nk.jax.tree_size(p3) < nk.jax.tree_size(p1)

    ma_masked2 = nk.nn.DenseSymm(
        symmetries=perms,
        mode=mode,
        features=2,
        shape=shape,
        mask=nk.utils.HashableArray(mask),
    )
    p4 = jax.jit(ma_masked2.init)(k, x)
    assert nk.jax.tree_size(p4) == nk.jax.tree_size(p3)
    assert hash(ma_masked) == hash(ma_masked2)


@pytest.mark.parametrize("symmetries", ["trans", "space_group"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("mode", ["fft", "matrix"])
def test_DenseSymm(symmetries, use_bias, mode):
    rng = nk.jax.PRNGSeq(0)

    g, hi, perms = _setup_symm(symmetries, N=8)

    if mode == "matrix":
        ma = nk.nn.DenseSymm(
            symmetries=perms,
            mode=mode,
            features=8,
            use_bias=use_bias,
            bias_init=uniform(),
        )
    else:
        ma = nk.nn.DenseSymm(
            symmetries=perms,
            shape=tuple(g.extent),
            mode=mode,
            features=8,
            use_bias=use_bias,
            bias_init=uniform(),
        )

    pars = ma.init(rng.next(), hi.random_state(rng.next(), (2, 1)))

    v = hi.random_state(rng.next(), (3, 1))
    vals = [ma.apply(pars, v[..., p]) for p in np.asarray(perms)]
    for val in vals:
        np.testing.assert_allclose(jnp.sort(val, -1), jnp.sort(vals[0], -1))


@pytest.mark.parametrize("symmetries", ["trans", "space_group"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("mode", ["fft", "matrix"])
def test_DenseSymm_infeatures(symmetries, use_bias, mode):
    rng = nk.jax.PRNGSeq(0)

    g, hi, perms = _setup_symm(symmetries, N=8)

    if mode == "matrix":
        ma = nk.nn.DenseSymm(
            symmetries=perms,
            mode=mode,
            features=8,
            use_bias=use_bias,
            bias_init=uniform(),
        )
    else:
        ma = nk.nn.DenseSymm(
            symmetries=perms,
            shape=tuple(g.extent),
            mode=mode,
            features=8,
            use_bias=use_bias,
            bias_init=uniform(),
        )

    pars = ma.init(rng.next(), hi.random_state(rng.next(), 2).reshape(1, 2, -1))

    v = hi.random_state(rng.next(), 6).reshape(3, 2, -1)
    vals = [ma.apply(pars, v[..., p]) for p in np.asarray(perms)]
    for val in vals:
        np.testing.assert_allclose(jnp.sort(val, -1), jnp.sort(vals[0], -1))


@pytest.mark.parametrize("mode", ["fft", "matrix", "irreps"])
def test_DenseEquivariant_creation(mode):
    g = nk.graph.Chain(8)
    space_group = g.space_group()

    def check_init(creator):
        ma = creator()
        _ = ma.init(nk.jax.PRNGKey(0), np.ones([1, 4, 16]))

    # Init with graph
    check_init(
        lambda: nk.nn.DenseEquivariant(
            symmetries=g,
            mode=mode,
            features=4,
        )
    )

    # init with space_group
    if mode == "irreps":
        check_init(
            lambda: nk.nn.DenseEquivariant(
                symmetries=space_group,
                mode=mode,
                features=4,
            )
        )
    else:
        check_init(
            lambda: nk.nn.DenseEquivariant(
                symmetries=space_group,
                shape=tuple(g.extent),
                mode=mode,
                features=4,
            )
        )

    # init with arrays
    if mode == "irreps":
        check_init(
            lambda: nk.nn.DenseEquivariant(
                symmetries=space_group.irrep_matrices(),
                mode=mode,
                features=4,
            )
        )
    elif mode == "fft":
        check_init(
            lambda: nk.nn.DenseEquivariant(
                symmetries=space_group.product_table,
                mode=mode,
                shape=(8,),
                features=4,
            )
        )
    else:
        check_init(
            lambda: nk.nn.DenseEquivariant(
                symmetries=space_group.product_table,
                mode=mode,
                features=4,
            )
        )


@pytest.mark.parametrize("symmetries", ["trans", "space_group"])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("lattice", [nk.graph.Chain, nk.graph.Square])
@pytest.mark.parametrize("mode", ["fft", "matrix", "irreps"])
@pytest.mark.parametrize("mask", [True, False])
def test_DenseEquivariant(symmetries, use_bias, lattice, mode, mask):
    rng = nk.jax.PRNGSeq(0)

    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    pt = perms.product_table
    n_symm = np.asarray(perms).shape[0]

    if mask:
        mask = np.zeros(n_symm)
        mask[np.random.choice(n_symm, n_symm // 2, replace=False)] = 1
    else:
        mask = np.ones([n_symm])

    if mode == "irreps":
        ma = nk.nn.DenseEquivariant(
            symmetries=perms,
            mode=mode,
            features=1,
            mask=mask,
            use_bias=use_bias,
            bias_init=uniform(),
        )
    else:
        ma = nk.nn.DenseEquivariant(
            symmetries=pt,
            shape=tuple(g.extent),
            mode=mode,
            features=1,
            mask=mask,
            use_bias=use_bias,
            bias_init=uniform(),
        )

    dum_input = jax.random.normal(rng.next(), (1, 1, n_symm))
    pars = ma.init(rng.next(), dum_input)

    # inv_pt computes chosen_op = gh^-1 instead of g^-1h
    chosen_op = np.random.randint(n_symm)
    inverse = PermutationGroup(
        [perms.elems[i] for i in perms.inverse], degree=g.n_nodes
    )
    inv_pt = inverse.product_table
    sym_op = np.where(inv_pt == chosen_op, 1.0, 0.0)

    v = random.normal(rng.next(), [3, 1, n_symm])
    v_trans = jnp.matmul(v, sym_op)

    out = ma.apply(pars, v)
    out_trans = ma.apply(pars, v_trans)

    # output should be involution
    np.testing.assert_allclose(jnp.matmul(out, sym_op), out_trans)


@pytest.mark.parametrize("lattice", [nk.graph.Chain, nk.graph.Square])
@pytest.mark.parametrize("symmetries", ["trans", "space_group"])
def test_modes_DenseSymm(lattice, symmetries):
    rng = nk.jax.PRNGSeq(0)
    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    ma_fft = nk.nn.DenseSymm(
        symmetries=perms,
        mode="fft",
        features=4,
        shape=tuple(g.extent),
        bias_init=uniform(),
    )
    ma_matrix = nk.nn.DenseSymm(
        symmetries=perms,
        mode="matrix",
        features=4,
        bias_init=uniform(),
    )

    dum_input = jax.random.normal(rng.next(), (3, 1, g.n_nodes))

    pars = ma_fft.init(rng.next(), dum_input)
    _ = ma_matrix.init(rng.next(), dum_input)

    np.testing.assert_allclose(
        ma_fft.apply(pars, dum_input), ma_matrix.apply(pars, dum_input)
    )

    # Test Deprecation warning
    dum_input_nofeatures = dum_input.reshape((dum_input.shape[0], dum_input.shape[2]))
    with pytest.raises(SymmModuleInvalidInputShape):
        np.testing.assert_allclose(
            ma_fft.apply(pars, dum_input), ma_fft.apply(pars, dum_input_nofeatures)
        )
    with pytest.raises(SymmModuleInvalidInputShape):
        np.testing.assert_allclose(
            ma_matrix.apply(pars, dum_input),
            ma_matrix.apply(pars, dum_input_nofeatures),
        )


@pytest.mark.parametrize("lattice", [nk.graph.Chain, nk.graph.Square])
@pytest.mark.parametrize("symmetries", ["trans", "space_group"])
def test_modes_DenseSymm_infeatures(lattice, symmetries):
    rng = nk.jax.PRNGSeq(0)
    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    ma_fft = nk.nn.DenseSymm(
        symmetries=perms,
        mode="fft",
        features=4,
        shape=tuple(g.extent),
        bias_init=uniform(),
    )
    ma_matrix = nk.nn.DenseSymm(
        symmetries=perms,
        mode="matrix",
        features=4,
        bias_init=uniform(),
    )

    dum_input = jax.random.normal(rng.next(), (1, 3, g.n_nodes))
    pars = ma_fft.init(rng.next(), dum_input)
    _ = ma_matrix.init(rng.next(), dum_input)

    np.testing.assert_allclose(
        ma_fft.apply(pars, dum_input), ma_matrix.apply(pars, dum_input)
    )


@pytest.mark.parametrize("lattice", [nk.graph.Chain, nk.graph.Square])
@pytest.mark.parametrize("symmetries", ["trans", "space_group"])
def test_modes_DenseEquivariant(lattice, symmetries):
    rng = nk.jax.PRNGSeq(0)
    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    ma_fft = nk.nn.DenseEquivariant(
        symmetries=perms,
        mode="fft",
        features=1,
        shape=tuple(g.extent),
        bias_init=uniform(),
    )
    ma_irreps = nk.nn.DenseEquivariant(
        symmetries=perms,
        mode="irreps",
        features=1,
        bias_init=uniform(),
    )
    ma_matrix = nk.nn.DenseEquivariant(
        symmetries=perms,
        mode="matrix",
        features=1,
        bias_init=uniform(),
    )

    dum_input = jax.random.normal(rng.next(), (1, 1, len(perms)))
    pars = ma_fft.init(rng.next(), dum_input)
    _ = ma_irreps.init(rng.next(), dum_input)
    _ = ma_matrix.init(rng.next(), dum_input)

    fft_out = ma_fft.apply(pars, dum_input)
    irreps_out = ma_irreps.apply(pars, dum_input)
    matrix_out = ma_matrix.apply(pars, dum_input)

    np.testing.assert_allclose(fft_out, irreps_out)
    np.testing.assert_allclose(fft_out, matrix_out)


def test_deprecated_inout_features_DenseEquivariant():
    perms = nk.graph.Chain(3).translation_group()

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            nk.nn.DenseEquivariant(
                symmetries=perms, mode="irreps", out_features=1, features=2
            )

    with pytest.warns(FutureWarning):
        nk.nn.DenseEquivariant(
            symmetries=perms,
            mode="irreps",
            out_features=1,
        )

    with pytest.warns(FutureWarning):
        nk.nn.DenseEquivariant(
            symmetries=perms, mode="irreps", in_features=3, features=1
        )
