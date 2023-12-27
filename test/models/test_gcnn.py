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
import jax
from jax.nn.initializers import uniform

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


@pytest.mark.parametrize("parity", [True, False])
@pytest.mark.parametrize("symmetries", ["trans", "autom"])
@pytest.mark.parametrize("lattice", [nk.graph.Chain, nk.graph.Square])
@pytest.mark.parametrize("mode", ["fft", "irreps"])
def test_gcnn_equivariance(parity, symmetries, lattice, mode):
    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    ma = nk.models.GCNN(
        symmetries=perms,
        mode=mode,
        shape=tuple(g.extent),
        layers=2,
        features=2,
        parity=parity,
        bias_init=uniform(),
    )

    pars = ma.init(nk.jax.PRNGKey(), hi.random_state(nk.jax.PRNGKey(), 1))

    v = hi.random_state(jax.random.PRNGKey(0), 3)
    # vals = [ma.apply(pars, v[..., p]) for p in np.asarray(perms)]
    # code below implements the commented line above, but is vectorised
    v = v[..., np.asarray(perms)].transpose(1, 0, 2)
    v = v.reshape(len(perms) * 3, g.n_nodes)
    vals = ma.apply(pars, v).reshape(len(perms), 3)

    for val in vals:
        np.testing.assert_allclose(val, vals[0])


@pytest.mark.parametrize("mode", ["fft", "irreps"])
@pytest.mark.parametrize("complex_output", [True, False])
def test_gcnn(mode, complex_output):
    lattice = nk.graph.Chain
    symmetries = "trans"
    parity = True
    g, hi, perms = _setup_symm(symmetries, N=3, lattice=lattice)

    ma = nk.models.GCNN(
        symmetries=perms,
        mode=mode,
        shape=tuple(g.extent),
        layers=2,
        features=2,
        parity=parity,
        bias_init=uniform(),
        complex_output=complex_output,
    )

    vmc = nk.VMC(
        nk.operator.Ising(hi, g, h=1.0),
        nk.optimizer.Sgd(0.1),
        nk.sampler.MetropolisLocal(hi, n_chains=2, sweep_size=2),
        ma,
        n_samples=8,
    )
    vmc.advance(1)


def test_gcnn_mask():
    g = nk.graph.Chain(4)

    hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=g.n_nodes)
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)

    input_mask = np.concatenate((np.zeros(2), np.ones(2)), 0)
    hidden_mask = np.concatenate((np.zeros(4), np.ones(4)), 0)

    ma1 = nk.models.GCNN(
        symmetries=g,
        layers=2,
        features=2,
        input_mask=None,
        hidden_mask=None,
    )

    ma2 = nk.models.GCNN(
        symmetries=g,
        layers=2,
        features=2,
        input_mask=input_mask,
        hidden_mask=None,
    )

    ma3 = nk.models.GCNN(
        symmetries=g,
        layers=2,
        features=2,
        input_mask=None,
        hidden_mask=hidden_mask,
    )

    vstate1 = nk.vqs.MCState(sampler=sa, model=ma1)
    vstate2 = nk.vqs.MCState(sampler=sa, model=ma2)
    vstate3 = nk.vqs.MCState(sampler=sa, model=ma3)

    assert vstate1.n_parameters - vstate2.n_parameters == 4
    assert vstate1.n_parameters - vstate3.n_parameters == 16


@pytest.mark.parametrize("mode", ["fft", "irreps"])
def test_GCNN_creation(mode):
    g = nk.graph.Chain(8)
    space_group = g.space_group()
    hi = nk.hilbert.Spin(1 / 2, N=8)

    def check_init(creator):
        ma = creator()
        _ = ma.init(nk.jax.PRNGKey(0), hi.numbers_to_states(np.arange(2)))

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

    # character table
    check_init(
        lambda: nk.models.GCNN(
            symmetries=g,
            mode=mode,
            layers=2,
            features=4,
            characters=np.ones([len(np.asarray(space_group))]),
        )
    )

    # equal amplitudes
    check_init(
        lambda: nk.models.GCNN(
            symmetries=g, mode=mode, layers=2, features=4, equal_amplitudes=True
        )
    )

    # complex_output = False
    # fails iff parameters are real and there are negative characters
    check_init(
        lambda: nk.models.GCNN(
            symmetries=g, mode=mode, layers=2, features=4, complex_output=False
        )
    )

    # this is a valid character of g.space_group()
    # invariant under translations, flips sign under mirrors
    char = np.array([1, -1] * 8, dtype=float)

    check_init(
        lambda: nk.models.GCNN(
            symmetries=g,
            mode=mode,
            layers=2,
            features=4,
            complex_output=False,
            characters=char,
            param_dtype=complex,
        )
    )

    with pytest.raises(ValueError):
        check_init(
            lambda: nk.models.GCNN(
                symmetries=g,
                mode=mode,
                layers=2,
                features=4,
                complex_output=False,
                characters=char,
            )
        )
