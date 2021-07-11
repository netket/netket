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
import numpy as np
import jax.numpy as jnp

import jax

from .test_nn import _setup_symm

import pytest


@pytest.mark.parametrize("use_hidden_bias", [True, False])
@pytest.mark.parametrize("use_visible_bias", [True, False])
@pytest.mark.parametrize("symmetries", ["trans", "autom"])
def test_RBMSymm(use_hidden_bias, use_visible_bias, symmetries):
    g, hi, perms = _setup_symm(symmetries, N=8)

    ma = nk.models.RBMSymm(
        symmetries=perms,
        alpha=4,
        use_visible_bias=use_visible_bias,
        use_hidden_bias=use_hidden_bias,
        hidden_bias_init=nk.nn.initializers.uniform(),
        visible_bias_init=nk.nn.initializers.uniform(),
    )
    pars = ma.init(nk.jax.PRNGKey(), hi.random_state(nk.jax.PRNGKey()))

    print(pars)

    v = hi.random_state(jax.random.PRNGKey(1), 3)
    vals = [ma.apply(pars, v[..., p]) for p in np.asarray(perms)]

    for val in vals:
        assert jnp.allclose(val, vals[0])

    vmc = nk.VMC(
        nk.operator.Ising(hi, g, h=1.0),
        nk.optim.Sgd(0.1),
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vmc.advance(1)


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("symmetries", ["trans", "autom"])
@pytest.mark.parametrize("lattice", [nk.graph.Chain, nk.graph.Square])
@pytest.mark.parametrize("mode", ["fft", "irreps"])
def test_gcnn(use_bias, symmetries, lattice, mode):
    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    ma = nk.models.GCNN(
        symmetries=perms,
        mode=mode,
        shape=tuple(g.extent),
        layers=2,
        features=2,
        use_bias=use_bias,
        bias_init=nk.nn.initializers.uniform(),
    )

    pars = ma.init(nk.jax.PRNGKey(), hi.random_state(nk.jax.PRNGKey(), 1))

    v = hi.random_state(jax.random.PRNGKey(0), 3)
    vals = [ma.apply(pars, v[..., p]) for p in np.asarray(perms)]

    for val in vals:
        assert jnp.allclose(val, vals[0])

    vmc = nk.VMC(
        nk.operator.Ising(hi, g, h=1.0),
        nk.optim.Sgd(0.1),
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vmc.advance(1)


def test_RBMSymm_creation():
    hi = nk.hilbert.Spin(1 / 2, N=8)

    def check_init(creator):
        ma = creator()
        _ = ma.init(nk.jax.PRNGKey(0), hi.numbers_to_states(0))

    perms = [[0, 1, 2, 3, 4, 5, 6, 7]]

    # Test different permutation argument types
    check_init(lambda: nk.models.RBMSymm(symmetries=perms))
    check_init(lambda: nk.models.RBMSymm(symmetries=jnp.array(perms)))

    # wrong shape
    with pytest.raises(ValueError):
        check_init(lambda: nk.models.RBMSymm(symmetries=perms[0]))

    # init with PermutationGroup
    check_init(
        lambda: nk.models.RBMSymm(
            symmetries=nk.graph.Chain(8).translation_group(), alpha=2
        )
    )

    # alpha too small
    with pytest.raises(ValueError):
        check_init(
            lambda: nk.models.RBMSymm(
                symmetries=nk.graph.Hypercube(8, 2).automorphisms(), alpha=1
            )
        )


@pytest.mark.parametrize("mode", ["fft", "irreps"])
def test_GCNN_creation(mode):

    g = nk.graph.Chain(8)
    space_group = g.space_group()
    hi = nk.hilbert.Spin(1 / 2, N=8)

    def check_init(creator):
        ma = creator()
        _ = ma.init(nk.jax.PRNGKey(0), hi.numbers_to_states(0))

    perms = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]]

    # Init with graph
    check_init(
        lambda: nk.models.GCNN(
            symmetries=g,
            mode=mode,
            layers=2,
            features=4,
        )
    )

    # init with space_group
    if mode == "irreps":
        check_init(
            lambda: nk.models.GCNN(
                symmetries=space_group,
                mode=mode,
                layers=2,
                features=4,
            )
        )
    else:
        check_init(
            lambda: nk.models.GCNN(
                symmetries=space_group,
                shape=tuple(g.extent),
                mode=mode,
                layers=2,
                features=4,
            )
        )

    # init with arrays for sym and product_table
    if mode == "irreps":
        check_init(
            lambda: nk.models.GCNN(
                symmetries=np.asarray(space_group),
                irreps=space_group.irrep_matrices(),
                layers=2,
                features=4,
            )
        )
    else:
        check_init(
            lambda: nk.models.GCNN(
                symmetries=np.asarray(space_group),
                product_table=space_group.product_table,
                shape=(8,),
                layers=2,
                features=4,
            )
        )

    # forget irreps/product table
    with pytest.raises(ValueError):
        check_init(
            lambda: nk.models.GCNN(
                symmetries=perms[0],
                layers=2,
                features=4,
            )
        )

    # need to specify shape
    if mode == "fft":
        with pytest.raises(TypeError):
            check_init(
                lambda: nk.models.GCNN(
                    symmetries=perms,
                    product_table=np.arange(4).reshape(2, 2),
                    layers=2,
                    features=4,
                )
            )


@pytest.mark.parametrize("use_hidden_bias", [True, False])
@pytest.mark.parametrize("use_visible_bias", [True, False])
def test_RBMMultiVal(use_hidden_bias, use_visible_bias):
    N = 8
    M = 3
    hi = nk.hilbert.Fock(M, N)
    g = nk.graph.Chain(N)

    ma = nk.models.RBMMultiVal(
        alpha=2,
        n_classes=M + 1,
        use_visible_bias=use_visible_bias,
        use_hidden_bias=use_hidden_bias,
        hidden_bias_init=nk.nn.initializers.uniform(),
        visible_bias_init=nk.nn.initializers.uniform(),
    )
    _ = ma.init(nk.jax.PRNGKey(), hi.random_state(nk.jax.PRNGKey(), 1))

    vmc = nk.VMC(
        nk.operator.BoseHubbard(hi, g, U=1.0),
        nk.optim.Sgd(0.1),
        nk.sampler.MetropolisLocal(hi),
        ma,
    )
    vmc.advance(1)
