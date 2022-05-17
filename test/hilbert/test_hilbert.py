# Copyright 2020 The Netket Authors. - All Rights Reserved.
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

import itertools
import netket as nk
import numpy as np
import pytest
from netket.hilbert import (
    DiscreteHilbert,
    HomogeneousHilbert,
    Particle,
    CustomHilbert,
    DoubledHilbert,
    Fock,
    Qubit,
    Spin,
)
import netket.experimental as nkx

import jax
import jax.numpy as jnp

from .. import common

pytestmark = common.skipif_mpi

hilberts = {}

# Spin 1/2
hilberts["Spin 1/2"] = Spin(s=0.5, N=20)

# Spin 1/2 with total Sz
hilberts["Spin[0.5, N=20, total_sz=1"] = Spin(s=0.5, total_sz=1.0, N=20)
hilberts["Spin[0.5, N=5, total_sz=-1.5"] = Spin(s=0.5, total_sz=-1.5, N=5)

# Spin 1/2 with total Sz
hilberts["Spin 1 with total Sz, even sites"] = Spin(s=1.0, total_sz=5.0, N=6)

# Spin 1/2 with total Sz
hilberts["Spin 1 with total Sz, odd sites"] = Spin(s=1.0, total_sz=2.0, N=7)

# Spin 3
hilberts["Spin 3"] = Spin(s=3, N=25)

# Boson
hilberts["Fock"] = Fock(n_max=5, N=41)

# Boson with total number
hilberts["Fock with total number"] = Fock(n_max=3, n_particles=110, N=120)

# Composite Fock
hilberts["Fock * Fock (indexable)"] = Fock(n_max=5, N=4) * Fock(n_max=7, N=4)
hilberts["Fock * Fock (non-indexable)"] = Fock(n_max=4, N=40) * Fock(n_max=7, N=40)

# Qubit
hilberts["Qubit"] = nk.hilbert.Qubit(100)

# Custom Hilbert
hilberts["Custom Hilbert"] = CustomHilbert(local_states=[-1232, 132, 0], N=70)

# Heisenberg 1d
hilberts["Heisenberg 1d"] = Spin(s=0.5, total_sz=0.0, N=20)

# Bose Hubbard
hilberts["Bose Hubbard"] = Fock(n_max=4, n_particles=20, N=20)

#
# Small hilbert space tests
#

# Spin 1/2
hilberts["Spin 1/2 Small"] = Spin(s=0.5, N=10)

# Spin 3
hilberts["Spin 1/2 with total Sz Small"] = Spin(s=3, total_sz=1.0, N=4)

# Boson
hilberts["Fock Small"] = Fock(n_max=3, N=5)

# Qubit
hilberts["Qubit Small"] = Qubit(N=1)

# Custom Hilbert
hilberts["Custom Hilbert Small"] = CustomHilbert(local_states=[-1232, 132, 0], N=5)

# Custom Hilbert
hilberts["DoubledHilbert[Spin]"] = DoubledHilbert(Spin(0.5, N=5))

hilberts["DoubledHilbert[Spin(total_sz=0.5)]"] = DoubledHilbert(
    Spin(0.5, N=5, total_sz=0.5)
)

hilberts["DoubledHilbert[Fock]"] = DoubledHilbert(Spin(0.5, N=5))

hilberts["DoubledHilbert[CustomHilbert]"] = DoubledHilbert(
    CustomHilbert(local_states=[-1232, 132, 0], N=5)
)

# hilberts["Tensor: Spin x Fock"] = Spin(s=0.5, N=4) * Fock(4, N=2)

hilberts["SpinOrbitalFermions"] = nkx.hilbert.SpinOrbitalFermions(3)
hilberts["SpinOrbitalFermions (spin)"] = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2)
hilberts["SpinOrbitalFermions (n_fermions)"] = nkx.hilbert.SpinOrbitalFermions(
    3, n_fermions=2
)
hilberts["SpinOrbitalFermions (n_fermions=list)"] = nkx.hilbert.SpinOrbitalFermions(
    5, s=1 / 2, n_fermions=(2, 3)
)

# Continuous space
# no pbc
hilberts["ContinuousSpaceHilbert"] = nk.hilbert.Particle(
    N=5, L=(np.inf, 10.0), pbc=(False, True)
)


all_hilbert_params = [pytest.param(hi, id=name) for name, hi in hilberts.items()]
discrete_hilbert_params = [
    pytest.param(hi, id=name)
    for name, hi in hilberts.items()
    if isinstance(hi, DiscreteHilbert)
]
homogeneous_hilbert_params = [
    pytest.param(hi, id=name)
    for name, hi in hilberts.items()
    if isinstance(hi, HomogeneousHilbert)
]
particle_hilbert_params = [
    pytest.param(hi, id=name)
    for name, hi in hilberts.items()
    if isinstance(hi, Particle)
]


#
# Tests
#
@pytest.mark.parametrize("hi", homogeneous_hilbert_params)
def test_consistent_size_homogeneous(hi: HomogeneousHilbert):
    assert hi.size > 0
    assert hi.local_size > 0
    assert len(hi.local_states) == hi.local_size
    for state in hi.local_states:
        assert np.isfinite(state).all()


@pytest.mark.parametrize("hi", particle_hilbert_params)
def test_consistent_size_particle(hi: Particle):
    assert hi.size > 0
    assert hi.n_particles > 0
    assert len(hi.extent) == (hi.size // hi.n_particles)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_random_states_discrete(hi: DiscreteHilbert):
    assert hi.random_state(jax.random.PRNGKey(13)).shape == (hi.size,)
    assert hi.random_state(jax.random.PRNGKey(13), dtype=np.float32).dtype == np.float32
    assert (
        hi.random_state(jax.random.PRNGKey(13), dtype=np.complex64).dtype
        == np.complex64
    )
    assert hi.random_state(jax.random.PRNGKey(13), 10).shape == (10, hi.size)
    assert hi.random_state(jax.random.PRNGKey(13), size=10).shape == (10, hi.size)
    # assert hi.random_state(jax.random.PRNGKey(13), size=(10,)).shape == (10, hi.size)
    # assert hi.random_state(jax.random.PRNGKey(13), size=(10, 2)).shape == (10, 2, hi.size)


@pytest.mark.parametrize("hi", homogeneous_hilbert_params)
def test_random_states_homogeneous(hi: HomogeneousHilbert):
    assert len(hi.local_states) == hi.local_size
    local_states = hi.local_states
    for i in range(100):
        rstate = hi.random_state(jax.random.PRNGKey(i * 14))
        for state in rstate:
            assert state in local_states


def test_random_states_fock_infinite():
    hi = Fock(N=2)
    rstate = hi.random_state(jax.random.PRNGKey(14), 20)
    assert np.all(rstate >= 0)
    assert rstate.shape == (20, 2)


@pytest.mark.parametrize("hi", particle_hilbert_params)
def test_random_states_particle(hi: Particle):
    assert hi.random_state(jax.random.PRNGKey(13)).shape == (hi.size,)
    assert hi.random_state(jax.random.PRNGKey(13), dtype=np.float32).dtype == np.float32
    assert (
        hi.random_state(jax.random.PRNGKey(13), dtype=np.complex64).dtype
        == np.complex64
    )
    assert hi.random_state(jax.random.PRNGKey(13), 10).shape == (10, hi.size)
    assert hi.random_state(jax.random.PRNGKey(13), size=10).shape == (10, hi.size)

    # check that boundary conditions are fulfilled if any are given
    state = hi.random_state(jax.random.PRNGKey(13))
    boundary = jnp.array(hi.n_particles * hi.pbc)
    Ls = jnp.array(hi.n_particles * hi.extent)
    extension = jnp.where(jnp.equal(boundary, False), jnp.inf, Ls)

    assert jnp.sum(
        jnp.where(jnp.equal(boundary, True), state < extension, 0)
    ) == jnp.sum(jnp.where(jnp.equal(boundary, True), 1, 0))


def test_particle_fail():
    with pytest.raises(ValueError):
        _ = Particle(N=5, L=(jnp.inf, 2.0), pbc=True)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_flip_state_discrete(hi: DiscreteHilbert):

    rng = nk.jax.PRNGSeq(1)
    N_batches = 20

    states = hi.random_state(rng.next(), N_batches)

    ids = jnp.asarray(
        jnp.floor(hi.size * jax.random.uniform(rng.next(), shape=(N_batches,))),
        dtype=int,
    )

    new_states, old_vals = nk.hilbert.random.flip_state(hi, rng.next(), states, ids)

    assert new_states.shape == states.shape

    for state in states:
        assert all(val in hi.states_at_index(i) for i, val in enumerate(state))

    states_np = np.asarray(states)
    states_new_np = np.array(new_states)

    for (row, col) in enumerate(ids):
        states_new_np[row, col] = states_np[row, col]

    np.testing.assert_allclose(states_np, states_new_np)


@pytest.mark.parametrize("hi", particle_hilbert_params)
def test_flip_state_particle(hi: Particle):
    rng = nk.jax.PRNGSeq(1)
    N_batches = 20

    states = hi.random_state(rng.next(), N_batches)

    ids = jnp.asarray(
        jnp.floor(hi.size * jax.random.uniform(rng.next(), shape=(N_batches,))),
        dtype=int,
    )

    with pytest.raises(TypeError):
        nk.hilbert.random.flip_state(hi, rng.next(), states, ids)


def test_flip_state_fock_infinite():
    hi = Fock(N=2)
    rng = nk.jax.PRNGSeq(1)
    N_batches = 20

    states = hi.random_state(rng.next(), N_batches, dtype=jnp.int64)

    ids = jnp.asarray(
        jnp.floor(hi.size * jax.random.uniform(rng.next(), shape=(N_batches,))),
        dtype=int,
    )

    new_states, old_vals = nk.hilbert.random.flip_state(hi, rng.next(), states, ids)

    assert new_states.shape == states.shape

    assert np.all(states >= 0)

    states_np = np.asarray(states)
    states_new_np = np.array(new_states)

    for (row, col) in enumerate(ids):
        states_new_np[row, col] = states_np[row, col]

    np.testing.assert_allclose(states_np, states_new_np)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_hilbert_index_discrete(hi: DiscreteHilbert):
    log_max_states = np.log(nk.hilbert._abstract_hilbert.max_states)

    if hi.is_indexable:
        local_sizes = [hi.size_at_index(i) for i in range(hi.size)]
        assert np.sum(np.log(local_sizes)) < log_max_states
        assert np.allclose(hi.states_to_numbers(hi.all_states()), range(hi.n_states))

        # batched version of number to state
        n_few = min(hi.n_states, 100)
        few_states = np.zeros(shape=(n_few, hi.size))
        for k in range(n_few):
            few_states[k] = hi.numbers_to_states(k)

        assert np.allclose(hi.numbers_to_states(np.asarray(range(n_few))), few_states)

    else:
        assert not hi.is_indexable

        with pytest.raises(RuntimeError):
            hi.n_states

    # Check that a large hilbert space raises error when constructing matrices
    g = nk.graph.Hypercube(length=100, n_dim=1)
    op = nk.operator.Heisenberg(hilbert=Spin(s=0.5, N=g.n_nodes), graph=g)

    with pytest.raises(RuntimeError):
        op.to_dense()
    with pytest.raises(RuntimeError):
        op.to_sparse()


def test_state_iteration():
    hilbert = Spin(s=0.5, N=10)

    reference = [np.array(el) for el in itertools.product([-1.0, 1.0], repeat=10)]

    for state, ref in zip(hilbert.states(), reference):
        assert np.allclose(state, ref)


def test_deprecations():
    g = nk.graph.Edgeless(3)

    with pytest.warns(FutureWarning):
        Spin(s=0.5, graph=g)

    with pytest.warns(FutureWarning):
        with pytest.raises(ValueError):
            Spin(s=0.5, graph=g, N=3)


def test_composite_hilbert_spin():
    hi1 = Spin(s=1 / 2, N=8)
    hi2 = Spin(s=3 / 2, N=8)

    hi = hi1 * hi2

    assert hi.size == hi1.size + hi2.size

    for i in range(hi.size):
        assert hi.size_at_index(i) == 2 if i < 8 else 4


def test_inhomogeneous_fock():
    hi1 = Fock(n_max=7, N=40)
    hi2 = Fock(n_max=2, N=40)
    hi = hi1 * hi2

    assert hi.size == hi1.size + hi2.size

    for i in range(0, 40):
        assert hi.size_at_index(i) == 8
        assert hi.states_at_index(i) == list(range(8))

    for i in range(40, 80):
        assert hi.size_at_index(i) == 3
        assert hi.states_at_index(i) == list(range(3))


def test_fermions():
    # size checks
    hi = nkx.hilbert.SpinOrbitalFermions(3)
    assert hi.size == 3
    assert hi.spin is None
    hi = nkx.hilbert.SpinOrbitalFermions(3, s=0)
    assert hi.size == 3
    assert hi.spin == 0.0
    hi = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    assert hi.size == 6
    assert hi.spin == 1 / 2
    hi = nkx.hilbert.SpinOrbitalFermions(3, n_fermions=2)
    assert hi.size == 3
    hi = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2, n_fermions=(2, 3))
    assert hi.size == 6

    # check the output
    hi = nkx.hilbert.SpinOrbitalFermions(5)
    assert hi.size == 5
    assert hi.n_states == 2**5
    assert hi.spin is None
    hi = nkx.hilbert.SpinOrbitalFermions(5, n_fermions=2)
    assert hi.size == 5
    assert np.all(hi.all_states().sum(axis=-1) == 2)
    hi = nkx.hilbert.SpinOrbitalFermions(5, s=1 / 2, n_fermions=(2, 1))
    assert hi.size == 10
    assert hi.spin == 1 / 2
    assert np.all(hi.all_states()[:, :5].sum(axis=-1) == 2)
    assert np.all(hi.all_states()[:, 5:].sum(axis=-1) == 1)


def test_fermion_fails():
    with pytest.raises(TypeError):
        _ = nkx.hilbert.SpinOrbitalFermions(5, n_fermions=2.7)
    with pytest.raises(TypeError):
        _ = nkx.hilbert.SpinOrbitalFermions(5, n_fermions=[1, 2])
    with pytest.raises(ValueError):
        _ = nkx.hilbert.SpinOrbitalFermions(5, n_fermions=[1, 2], s=1)


def test_fermions_states():
    import scipy.special

    hi = nkx.hilbert.SpinOrbitalFermions(5)
    assert hi.size == 5
    assert hi.n_states == 2**5

    hi = nkx.hilbert.SpinOrbitalFermions(5, n_fermions=2)
    assert hi.size == 5
    assert np.all(hi.all_states().sum(axis=-1) == 2)
    assert hi.n_states == int(scipy.special.comb(5, 2))

    hi = nkx.hilbert.SpinOrbitalFermions(5, s=1 / 2, n_fermions=2)
    assert hi.size == 10
    assert np.all(hi.all_states().sum(axis=-1) == 2)
    # distribute 2 fermions over (2*number of orbitals)
    assert hi.n_states == int(scipy.special.comb(2 * 5, 2))

    hi = nkx.hilbert.SpinOrbitalFermions(5, s=1 / 2, n_fermions=(2, 1))
    assert hi.size == 10
    assert np.all(hi.all_states()[:, :5].sum(axis=-1) == 2)
    assert np.all(hi.all_states()[:, 5:].sum(axis=-1) == 1)
    # product of all_states for -1/2 spin block and states for 1/2 block
    assert hi.n_states == int(scipy.special.comb(5, 2) * scipy.special.comb(5, 1))


def test_fermions_spin_index():
    hi = nkx.hilbert.SpinOrbitalFermions(5, s=1 / 2)
    assert hi._spin_index(-0.5) == 0  # indexing starts from -spin
    # sz=-0.5 --> block 0, sz= +0.5 --> block 1
    assert hi._spin_index(0.5) == 1
    hi = nkx.hilbert.SpinOrbitalFermions(5, s=3 / 2)
    assert hi._spin_index(-0.5) == 1  # indexing starts from -spin
    # sz=-1.5 --> block 0, sz=-0.5 --> block 1, sz= 0.5 --> block 2, ...
    assert hi._spin_index(0.5) == 2


def test_fermions_get_index():
    hi = nkx.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    # first block (-0.5) and first site (1) --> idx = 0
    assert hi._get_index(0, -0.5) == 0
    # first block (-0.5) and second site (1) --> idx = 1
    assert hi._get_index(1, -0.5) == 1
    # first block (-0.5) and first site (1) --> idx = 0 + n_orbital
    assert hi._get_index(0, +0.5) == 3
    # first block (-0.5) and second site (1) --> idx = 1 + n_orbital
    assert hi._get_index(1, +0.5) == 4
