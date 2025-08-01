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
from math import prod
from functools import partial
import netket as nk
import numpy as np
import pytest
from netket.hilbert import (
    DiscreteHilbert,
    HomogeneousHilbert,
    DoubledHilbert,
    Fock,
    Qubit,
    Spin,
)
from netket import experimental as nkx
from netket.experimental.hilbert import Particle

import jax
import jax.numpy as jnp
from jax.errors import JaxRuntimeError


from .. import common

pytestmark = common.skipif_distributed

hilberts = {}

# Spin 1/2
hilberts["Spin[1/2, N=10]"] = Spin(s=0.5, N=10)

# Spin 1/2 with total Sz
hilberts["Spin[1/2, N=10, total_sz=1"] = Spin(s=0.5, total_sz=1.0, N=10)
hilberts["Spin[1/2, N=5, total_sz=-1.5"] = Spin(s=0.5, total_sz=-1.5, N=5)

# Spin 1/2 with total Sz
hilberts["Spin[1, N=6, total_sz=5.0]"] = Spin(s=1.0, total_sz=5.0, N=6)
hilberts["Spin[1, N=7, total_sz=2.0]"] = Spin(s=1.0, total_sz=2.0, N=7)

# Spin 3
hilberts["Spin[s=3, N=25]"] = Spin(s=3, N=25)

# Boson
hilberts["Fock[max=5, N=41]"] = Fock(n_max=5, N=41)
hilberts["Fock[max=3, N=120, n_particles=110]"] = Fock(n_max=3, n_particles=110, N=120)

# Composite Fock
hilberts["Fock * Fock (indexable)"] = Fock(n_max=5, N=4) * Fock(n_max=7, N=4)
hilberts["Fock * Fock (non-indexable)"] = Fock(n_max=4, N=40) * Fock(n_max=7, N=40)

# Qubit
hilberts["Qubit"] = nk.hilbert.Qubit(100)

# Heisenberg 1d
hilberts["Heisenberg 1d"] = Spin(s=0.5, total_sz=0.0, N=10)

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

# Doubled Hilbert
hilberts["DoubledHilbert[Spin]"] = DoubledHilbert(Spin(0.5, N=5))

hilberts["DoubledHilbert[Spin(total_sz=0.5)]"] = DoubledHilbert(
    Spin(0.5, N=5, total_sz=0.5)
)

hilberts["DoubledHilbert[Fock]"] = DoubledHilbert(Spin(0.5, N=5))

# hilberts["Tensor: Spin x Fock"] = Spin(s=0.5, N=4) * Fock(4, N=2)

hilberts["SpinOrbitalFermions"] = nk.hilbert.SpinOrbitalFermions(3)
hilberts["SpinOrbitalFermions (spin)"] = nk.hilbert.SpinOrbitalFermions(3, s=1 / 2)
hilberts["SpinOrbitalFermions (n_fermions)"] = nk.hilbert.SpinOrbitalFermions(
    3, n_fermions=2
)
hilberts["SpinOrbitalFermions (n_fermions=list)"] = nk.hilbert.SpinOrbitalFermions(
    5, s=1 / 2, n_fermions_per_spin=(2, 3)
)
hilberts["SpinOrbitalFermions (polarized)"] = nk.hilbert.SpinOrbitalFermions(
    5, s=1 / 2, n_fermions_per_spin=(2, 0)
)
hilberts["SpinOrbitalFermions (higherspin)"] = nk.hilbert.SpinOrbitalFermions(
    3, s=3 / 2, n_fermions_per_spin=(2, 1, 2, 1)
)

# Continuous space
# no pbc
geo_default = nk.experimental.geometry.Cell(d=2, L=(np.inf, 10.0), pbc=(False, True))
hilberts["ContinuousSpaceHilbert"] = nk.experimental.hilbert.Particle(
    N=5, geometry=geo_default
)
hilberts["TensorContinuous"] = nk.experimental.hilbert.Particle(
    N=2, geometry=geo_default
) * nk.experimental.hilbert.Particle(N=3, geometry=geo_default)


N = 10
hilberts["ContinuousHelium"] = nk.experimental.hilbert.Particle(
    N=N, geometry=nk.experimental.geometry.Cell(d=1, L=(N / (0.3 * 2.9673),), pbc=True)
)

all_hilbert_params = [pytest.param(hi, id=name) for name, hi in hilberts.items()]
discrete_hilbert_params = [
    pytest.param(hi, id=name)
    for name, hi in hilberts.items()
    if isinstance(hi, DiscreteHilbert)
]
discrete_indexable_hilbert_params = [
    pytest.param(hi, id=name)
    for name, hi in hilberts.items()
    if (isinstance(hi, DiscreteHilbert) and hi.is_indexable)
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
@common.skipif_distributed
@pytest.mark.parametrize("hi", homogeneous_hilbert_params)
def test_consistent_size_homogeneous(hi: HomogeneousHilbert):
    assert hi.size > 0
    assert hi.local_size > 0
    assert len(hi.local_states) == hi.local_size
    for state in hi.local_states:
        assert np.isfinite(state).all()


@common.skipif_distributed
@pytest.mark.parametrize("hi", particle_hilbert_params)
def test_consistent_size_particle(hi: Particle):
    assert hi.size > 0
    assert hi.n_particles > 0
    assert hi.n_particles == sum(hi.n_per_spin)
    assert len(hi.domain) == (hi.size // hi.n_particles)


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
    np.testing.assert_allclose(
        hi.random_state(jax.random.PRNGKey(13)),
        jax.jit(hi.random_state)(jax.random.PRNGKey(13)),
    )


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_random_states_discrete_sharding(hi: DiscreteHilbert):

    # check that the sharding is correct
    s = hi.random_state(jax.random.key(13), 4)
    assert s.sharding.spec == ()

    # Check that no communication happens
    @jax.jit
    def _fun(k):
        return hi.random_state(k, 4)

    txt = _fun.lower(jax.random.key(13)).compile().as_text()
    for o in [
        "all-reduce",
        "collective-permute",
        "all-gather",
        "all-to-all",
        "reduce-scatter",
    ]:
        for l in txt.split("\n"):
            assert o not in l


@common.skipif_distributed
@pytest.mark.parametrize("hi", discrete_indexable_hilbert_params)
def test_constrained_correct(hi: DiscreteHilbert):
    n_states = hi.n_states
    assert hi.constrained == (n_states != prod(hi.shape))


@pytest.mark.parametrize("hi", homogeneous_hilbert_params)
def test_random_states_in_space_homogeneous(hi: HomogeneousHilbert):
    assert len(hi.local_states) == hi.local_size

    states = hi.random_state(jax.random.PRNGKey(1), 100)
    assert np.all((states[..., None] == np.asarray(hi._local_states)).any(-1))


@common.skipif_distributed
def test_random_states_fock_infinite():
    hi = Fock(N=2)
    rstate = hi.random_state(jax.random.key(14), 20)
    assert np.all(rstate >= 0)
    assert rstate.shape == (20, 2)


@common.skipif_distributed
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
    np.testing.assert_allclose(
        hi.random_state(jax.random.PRNGKey(13)),
        jax.jit(hi.random_state)(jax.random.PRNGKey(13)),
    )

    # check that boundary conditions are fulfilled if any are given
    state = hi.random_state(jax.random.PRNGKey(13))
    boundary = jnp.array(hi.n_particles * hi.geometry.pbc)
    Ls = jnp.array(hi.n_particles * hi.domain)
    extension = jnp.where(jnp.equal(boundary, False), jnp.inf, Ls)

    assert jnp.sum(
        jnp.where(jnp.equal(boundary, True), state < extension, 0)
    ) == jnp.sum(jnp.where(jnp.equal(boundary, True), 1, 0))


@common.skipif_distributed
def test_particle_fail():
    with pytest.raises(ValueError):
        _ = Particle(
            N=5,
            geometry=nkx.geometry.Cell(d=2, L=(jnp.inf, 2.0), pbc=True),
        )


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

    states_i = hi.states_to_local_indices(states)
    size_i = jnp.array(hi.shape)
    print(states_i.shape, size_i.shape)
    assert jnp.issubdtype(states_i.dtype, jnp.integer)
    is_pos = states_i >= 0
    assert jnp.all(is_pos)
    is_in_bounds = jnp.all(states_i <= size_i, axis=-1)
    assert jnp.all(is_in_bounds)

    states_np = np.asarray(states)
    states_new_np = np.array(new_states)

    for row, col in enumerate(ids):
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

    for row, col in enumerate(ids):
        states_new_np[row, col] = states_np[row, col]

    np.testing.assert_allclose(states_np, states_new_np)


# Check that the flip state rule works for 1-local dimension spaces
def test_flip_state_d1():
    hi = Fock(n_max=0, N=2)
    rng = nk.jax.PRNGSeq(1)
    N_batches = 20

    states = hi.random_state(rng.next(), N_batches, dtype=jnp.int64)
    np.testing.assert_allclose(states, 0)

    ids = jnp.asarray(
        jnp.floor(hi.size * jax.random.uniform(rng.next(), shape=(N_batches,))),
        dtype=int,
    )

    new_states, old_vals = nk.hilbert.random.flip_state(hi, rng.next(), states, ids)
    np.testing.assert_allclose(new_states, 0)
    np.testing.assert_allclose(old_vals, 0)


# Check that the flip state rule works for 2-local dimension spaces
def test_flip_state_d2():
    hi = Spin(0.5, N=4)
    rng = nk.jax.PRNGSeq(1)
    N_batches = 20

    states = hi.random_state(rng.next(), N_batches, dtype=jnp.int64)

    ids = jnp.asarray(
        jnp.floor(hi.size * jax.random.uniform(rng.next(), shape=(N_batches,))),
        dtype=int,
    )
    # Check it flipped the sites
    new_states, old_vals = nk.hilbert.random.flip_state(hi, rng.next(), states, ids)
    np.testing.assert_allclose(new_states[jnp.arange(N_batches), ids], -old_vals)
    # check returning correct old vals
    np.testing.assert_allclose(states[jnp.arange(N_batches), ids], old_vals)

    # Check flipping twice brings us back
    new_new_states, old_vals = nk.hilbert.random.flip_state(
        hi, rng.next(), new_states, ids
    )
    np.testing.assert_allclose(states, new_new_states)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_hilbert_index_discrete(hi: DiscreteHilbert):
    assert isinstance(hi.constrained, bool)

    log_max_states = np.log(nk.hilbert.index.max_states)

    if hi.is_indexable:
        local_sizes = [hi.size_at_index(i) for i in range(hi.size)]
        assert np.sum(np.log(local_sizes)) < log_max_states
        np.testing.assert_allclose(
            hi.states_to_numbers(hi.all_states()), range(hi.n_states)
        )

        # batched version of number to state
        n_few = min(hi.n_states, 100)
        few_states = np.zeros(shape=(n_few, hi.size))
        for k in range(n_few):
            few_states[k] = hi.numbers_to_states(k)

        np.testing.assert_allclose(
            hi.numbers_to_states(np.asarray(range(n_few))), few_states
        )

        few_states_n = np.asarray(range(n_few)).reshape(1, -1)
        np.testing.assert_allclose(
            hi.numbers_to_states(few_states_n), few_states.reshape(1, -1, hi.size)
        )

    else:
        assert not hi.is_indexable
        with pytest.raises(RuntimeError):
            hi.n_states
        with pytest.raises(RuntimeError):
            hi.all_states()


def test_hilbert_index_discrete_large_errors():
    # Check that a large hilbert space raises error when constructing matrices
    g = nk.graph.Hypercube(length=100, n_dim=1)
    op = nk.operator.Heisenberg(hilbert=Spin(s=0.5, N=g.n_nodes), graph=g)

    with pytest.raises(RuntimeError):
        op.to_dense()
    with pytest.raises(RuntimeError):
        op.to_sparse()


@pytest.mark.parametrize("hi", discrete_indexable_hilbert_params)
def test_hilbert_indexing_jax_array(hi: DiscreteHilbert):
    x_np = hi.numbers_to_states(np.array(0))
    x_jnp = hi.numbers_to_states(jnp.array(0))
    np.testing.assert_allclose(x_np, x_jnp)

    i_np = hi.states_to_numbers(x_np)
    i_jnp = hi.states_to_numbers(jnp.array(x_np))
    np.testing.assert_allclose(i_np, i_jnp)


@partial(jax.jit, static_argnums=0)
def _states_to_local_indices_jit(hilb, x):
    return hilb.states_to_local_indices(x)


@partial(jax.jit, static_argnums=0)
def _local_indices_to_states_jit(hilb, x):
    return hilb.local_indices_to_states(x)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_states_to_local_indices(hi):
    x = hi.random_state(jax.random.PRNGKey(3), (200))
    idxs = hi.states_to_local_indices(x)
    idxs_jit = _states_to_local_indices_jit(hi, x)

    np.testing.assert_allclose(idxs, idxs_jit)

    # check that the index is correct
    for s in range(hi.size):
        local_states = np.asarray(hi.states_at_index(s))
        np.testing.assert_allclose(local_states[idxs[..., s]], x[..., s])


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_local_indices_to_states(hi):
    x = hi.random_state(jax.random.PRNGKey(3), (200))
    idxs = hi.states_to_local_indices(x)

    y = hi.local_indices_to_states(idxs)
    y_jit = _local_indices_to_states_jit(hi, idxs)

    np.testing.assert_allclose(y, y_jit)

    np.testing.assert_allclose(
        x, hi.local_indices_to_states(hi.states_to_local_indices(x))
    )

    # check that the index is correct
    for s in range(hi.size):
        local_states = np.asarray(hi.states_at_index(s))
        np.testing.assert_allclose(local_states[idxs[..., s]], x[..., s])


@pytest.mark.parametrize("inverted_ordering", [True, False])
def test_spin_state_iteration(inverted_ordering: bool):
    hilbert = Spin(s=0.5, N=5, inverted_ordering=inverted_ordering)

    if inverted_ordering:
        reference = [np.array(el) for el in itertools.product([-1.0, 1.0], repeat=5)]
    else:
        reference = [np.array(el) for el in itertools.product([1.0, -1.0], repeat=5)]

    for state, ref in zip(hilbert.states(), reference):
        np.testing.assert_allclose(state, ref)


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
        # np.states_at_index(i) is a StaticRange
        np.testing.assert_array_equal(hi.states_at_index(i), np.arange(8))

    for i in range(40, 80):
        assert hi.size_at_index(i) == 3
        np.testing.assert_array_equal(hi.states_at_index(i), np.arange(3))


def test_fermions():
    # size checks
    hi = nk.hilbert.SpinOrbitalFermions(3)
    assert hi.size == 3
    assert hi.spin is None
    hi = nk.hilbert.SpinOrbitalFermions(3, s=0)
    assert hi.size == 3
    assert hi.spin == 0.0
    hi = nk.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    assert hi.size == 6
    assert hi.spin == 1 / 2
    hi = nk.hilbert.SpinOrbitalFermions(3, n_fermions=2)
    assert hi.size == 3
    hi = nk.hilbert.SpinOrbitalFermions(3, s=1 / 2, n_fermions_per_spin=(2, 3))
    assert hi.size == 6

    # check the output
    hi = nk.hilbert.SpinOrbitalFermions(5)
    assert hi.size == 5
    assert hi.n_states == 2**5
    assert hi.spin is None
    hi = nk.hilbert.SpinOrbitalFermions(5, n_fermions=2)
    assert hi.size == 5
    assert np.all(hi.all_states().sum(axis=-1) == 2)
    hi = nk.hilbert.SpinOrbitalFermions(5, s=1 / 2, n_fermions_per_spin=(2, 1))
    assert hi.size == 10
    assert hi.spin == 1 / 2
    assert np.all(hi.all_states()[:, :5].sum(axis=-1) == 2)
    assert np.all(hi.all_states()[:, 5:].sum(axis=-1) == 1)


def test_fermion_fails():
    with pytest.raises(TypeError):
        _ = nk.hilbert.SpinOrbitalFermions(5, n_fermions=2.7)
    with pytest.raises(TypeError):
        _ = nk.hilbert.SpinOrbitalFermions(5, n_fermions=[1, 2])
    with pytest.raises(ValueError):
        _ = nk.hilbert.SpinOrbitalFermions(5, n_fermions_per_spin=[1, 2])
    with pytest.raises(ValueError):
        _ = nk.hilbert.SpinOrbitalFermions(5, n_fermions_per_spin=[1, 2], s=1)


def test_fermions_states():
    import scipy.special

    hi = nk.hilbert.SpinOrbitalFermions(5)
    assert hi.size == 5
    assert not hi.constrained
    assert hi.n_states == 2**5

    hi = nk.hilbert.SpinOrbitalFermions(5, n_fermions=2)
    assert hi.size == 5
    assert hi.constrained
    assert np.all(hi.all_states().sum(axis=-1) == 2)
    assert hi.n_states == int(scipy.special.comb(5, 2))

    hi = nk.hilbert.SpinOrbitalFermions(5, s=1 / 2, n_fermions=2)
    assert hi.size == 10
    assert hi.constrained
    np.testing.assert_array_equal(hi.all_states().sum(axis=-1), 2)
    # distribute 2 fermions over (2*number of orbitals)
    assert hi.n_states == int(scipy.special.comb(2 * 5, 2))

    hi = nk.hilbert.SpinOrbitalFermions(5, s=1 / 2, n_fermions_per_spin=(2, 1))
    assert hi.size == 10
    assert hi.constrained
    np.testing.assert_array_equal(hi.all_states()[:, :5].sum(axis=-1), 2)
    np.testing.assert_array_equal(hi.all_states()[:, 5:].sum(axis=-1), 1)
    # product of all_states for -1/2 spin block and states for 1/2 block
    assert hi.n_states == int(scipy.special.comb(5, 2) * scipy.special.comb(5, 1))


def test_fermions_spin_index():
    hi = nk.hilbert.SpinOrbitalFermions(5, s=1 / 2)
    assert hi._spin_index(-1) == 0  # indexing starts from -spin
    # sz=-1 --> block 0, sz= +1 --> block 1
    assert hi._spin_index(1) == 1
    with pytest.raises(TypeError):
        hi._spin_index(-0.99)
    with pytest.raises(ValueError):
        hi._spin_index(-2)
    with pytest.raises(ValueError):
        hi._spin_index(0)

    hi = nk.hilbert.SpinOrbitalFermions(5, s=3 / 2)
    assert hi._spin_index(-3) == 0  # indexing starts from -spin
    assert hi._spin_index(-1) == 1  # indexing starts from -spin
    # sz=-3 --> block 0, sz=-0.5 --> block 1, sz= 1 --> block 2, ...
    assert hi._spin_index(1) == 2
    with pytest.raises(ValueError):
        hi._spin_index(4)
    with pytest.raises(ValueError):
        hi._spin_index(0)

    hi = nk.hilbert.SpinOrbitalFermions(5, s=1)
    assert hi._spin_index(-2) == 0  # indexing starts from -spin
    assert hi._spin_index(0) == 1  # indexing starts from -spin
    assert hi._spin_index(2) == 2
    with pytest.raises(ValueError):
        hi._spin_index(1)


def test_fermions_get_index():
    hi = nk.hilbert.SpinOrbitalFermions(3, s=1 / 2)
    # first block (-1) and first site (1) --> idx = 0
    assert hi._get_index(0, -1) == 0
    # first block (-1) and second site (1) --> idx = 1
    assert hi._get_index(1, -1) == 1
    # first block (-1) and first site (1) --> idx = 0 + n_orbital
    assert hi._get_index(0, +1) == 3
    # first block (-1) and second site (1) --> idx = 1 + n_orbital
    assert hi._get_index(1, +1) == 4


def test_no_particles():
    hi = Fock(n_max=3, n_particles=0, N=4)
    states = hi.all_states()
    assert states.shape[0] == 1
    np.testing.assert_allclose(states, 0.0)

    # same for fermions
    hi = nk.hilbert.SpinOrbitalFermions(2, s=1 / 2, n_fermions_per_spin=(0, 0))
    states = hi.all_states()
    assert states.shape[0] == 1
    np.testing.assert_allclose(states, 0.0)

    with pytest.raises(ValueError):
        # also test negative particles
        _ = Fock(n_max=3, n_particles=-1, N=4)


def test_tensor_no_recursion():
    # Issue https://github.com/netket/netket/issues/1101
    hi = nk.hilbert.Fock(3) * nk.hilbert.Spin(0.5, 2, total_sz=0.0)
    assert isinstance(hi, nk.hilbert.TensorHilbert)


def test_tensor_combination():
    hi1 = Spin(s=1 / 2, N=2) * Spin(s=1, N=2) * Fock(n_max=3, N=1)
    hi2 = Fock(n_max=3, N=1) * Spin(s=1, N=2) * Spin(s=1 / 2, N=2)
    hit = hi1 * hi2
    assert isinstance(hit, nk.hilbert.TensorHilbert)
    assert isinstance(hit, nk.hilbert._tensor_hilbert_discrete.TensorDiscreteHilbert)
    assert hit.shape == hi1.shape + hi2.shape
    assert hit.n_states == hi1.n_states * hi2.n_states
    assert len(hit._hilbert_spaces) == 5
    assert isinstance(repr(hit), str)

    hi3 = nk.experimental.hilbert.Particle(N=5, geometry=geo_default)
    hit2 = hi1 * hi3
    assert isinstance(hit2, nk.hilbert.TensorHilbert)
    assert hit2.size == hi1.size + hi3.size
    assert len(hit2._hilbert_spaces) == 4
    assert isinstance(repr(hit2), str)
    assert hit2 == nk.hilbert.TensorHilbert(hi1, hi3)

    hit3 = hit * hi3
    assert isinstance(hit3, nk.hilbert.TensorHilbert)
    assert len(hit3._hilbert_spaces) == 6
    assert isinstance(repr(hit3), str)
    assert hit3 == nk.hilbert.TensorHilbert(hit, hi3)

    hit3 = Spin(s=1 / 2, N=2) * hi3
    assert isinstance(hit3, nk.hilbert.TensorHilbert)
    assert len(hit3._hilbert_spaces) == 2
    assert isinstance(repr(hit3), str)
    assert hit3 == nk.hilbert.TensorHilbert(Spin(s=1 / 2, N=2), hi3)

    hit3 = Spin(s=1 / 2, N=2) * nk.hilbert.TensorHilbert(hi3)
    assert isinstance(hit3, nk.hilbert.TensorHilbert)
    assert len(hit3._hilbert_spaces) == 2
    assert isinstance(repr(hit3), str)
    assert hit3 == nk.hilbert.TensorHilbert(Spin(s=1 / 2, N=2), hi3)

    hit3 = nk.hilbert.TensorHilbert(hi3) * nk.hilbert.TensorHilbert(hi3)
    assert isinstance(hit3, nk.hilbert.TensorHilbert)
    assert len(hit3._hilbert_spaces) == 2
    assert isinstance(repr(hit3), str)
    assert hit3 == nk.hilbert.TensorHilbert(hi3, hi3)

    hit = hi3 * hi3
    assert isinstance(hit, nk.hilbert.TensorHilbert)
    assert len(hit._hilbert_spaces) == 2
    assert isinstance(repr(hit), str)
    assert hit == nk.hilbert.TensorHilbert(hi3, hi3)

    hit = (hi3 * hi3) * hi3
    assert isinstance(hit, nk.hilbert.TensorHilbert)
    assert len(hit._hilbert_spaces) == 3
    assert isinstance(repr(hit), str)
    assert hit == nk.hilbert.TensorHilbert(hi3 * hi3, hi3)

    hit = hi3 * (hi3 * hi3)
    assert isinstance(hit, nk.hilbert.TensorHilbert)
    assert len(hit._hilbert_spaces) == 3
    assert isinstance(repr(hit), str)
    assert hit == nk.hilbert.TensorHilbert(hi3, hi3 * hi3)

    hit = nk.hilbert.TensorHilbert(Spin(s=1 / 2, N=2))
    assert isinstance(hit, nk.hilbert._tensor_hilbert_discrete.TensorDiscreteHilbert)
    assert len(hit._hilbert_spaces) == 1
    assert isinstance(repr(hit), str)

    hit = nk.hilbert.TensorHilbert(
        nk.experimental.hilbert.Particle(N=5, geometry=geo_default)
    )
    assert isinstance(hit, nk.hilbert._tensor_hilbert.TensorGenericHilbert)
    assert len(hit._hilbert_spaces) == 1
    assert isinstance(repr(hit), str)


def test_errors():
    hi = Spin(s=1 / 2, N=2)
    with pytest.raises(TypeError):
        1 * hi
    with pytest.raises(TypeError):
        hi * 1

    hi = nk.experimental.hilbert.Particle(N=5, geometry=geo_default)
    with pytest.raises(TypeError):
        1 * hi
    with pytest.raises(TypeError):
        hi * 1


def test_pow():
    hi = Spin(s=1 / 2, N=2)
    assert hi**5 == Spin(1 / 2, N=10)


def test_constrained_eq_hash():
    hi1 = nk.hilbert.Spin(0.5, 4, total_sz=0)
    hi2 = nk.hilbert.Spin(0.5, 4, total_sz=0)
    assert hi1 == hi2
    assert hash(hi1) == hash(hi2)

    hi2 = nk.hilbert.Spin(0.5, 4, total_sz=2)
    assert hi1 != hi2
    assert hash(hi1) != hash(hi2)

    hi1 = nk.hilbert.Fock(3, 4, n_particles=2)
    hi2 = nk.hilbert.Fock(3, 4, n_particles=2)
    assert hi1 == hi2
    assert hash(hi1) == hash(hi2)

    hi2 = nk.hilbert.Fock(3, 4, n_particles=3)
    assert hi1 != hi2
    assert hash(hi1) != hash(hi2)


def test_particle_with_geometry():
    geo = nk.experimental.geometry.Cell(d=2, L=(np.inf, 10.0), pbc=(False, True))
    hi1 = nk.experimental.hilbert.Particle(N=5, geometry=geo)
    hi2 = nk.experimental.hilbert.Particle(N=5, geometry=geo_default)
    assert hi1 == hi2
    geo2 = nk.experimental.geometry.Cell(d=1, L=1.0, pbc=True)
    assert np.isclose(geo2.distance([0.1], [0.9]), 0.2)


def test_hilbert_states_outside_range_errors():
    hi = nk.hilbert.Fock(3, 2, 4)

    if jax.device_count() > 1:
        # TODO: single host this can work
        pytest.xfail("not implemented")

    with pytest.raises(JaxRuntimeError):
        # XlaRuntimeError: Numbers outside the range of allowed states.
        hi.numbers_to_states(-1)
    with pytest.raises(JaxRuntimeError):
        # XlaRuntimeError: Numbers outside the range of allowed states.
        hi.numbers_to_states(10000)
    with pytest.raises(JaxRuntimeError):
        # XlaRuntimeError: States outside the range of allowed states.
        hi.states_to_numbers(jnp.array([0, 4]))
    with pytest.raises(JaxRuntimeError):
        # XlaRuntimeError: States do not fulfill constraint.
        hi.states_to_numbers(jnp.array([0, 3]))


@partial(jax.jit, static_argnums=0)
def _states_to_numbers_jit(hi, states):
    return hi.states_to_numbers(states)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_hilbert_states_to_numbers_jit(hi: DiscreteHilbert):
    if hi.is_indexable:
        numbers0 = jnp.arange(min(10, hi.n_states))
        states0 = hi.numbers_to_states(numbers0)
        numbers1 = _states_to_numbers_jit(hi, states0)
        np.testing.assert_allclose(numbers0, numbers1)


@partial(jax.jit, static_argnums=0)
def _numbers_to_states_jit(hi, numbers):
    return hi.numbers_to_states(numbers)


@pytest.mark.parametrize("hi", discrete_hilbert_params)
def test_hilbert_numbers_to_states_jit(hi: DiscreteHilbert):
    if hi.is_indexable:
        numbers0 = jnp.arange(min(10, hi.n_states))
        states0 = hi.numbers_to_states(numbers0)
        states1 = _numbers_to_states_jit(hi, numbers0)
        np.testing.assert_allclose(states0, states1)


def test_doubled_hilbert_indexable():
    # make an indexable hilbert space
    hi = nk.hilbert.SpinOrbitalFermions(16, s=1 / 2, n_fermions=1)
    hid = DoubledHilbert(hi)
    assert hi.is_indexable  # just to verify this is indeed still the case
    assert hid.is_indexable  # then this must hold as well


def test_hilbert_dtype_int8():
    hi = nk.hilbert.SpinOrbitalFermions(4)
    assert hi.all_states().dtype == np.int8
    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions=2)
    assert hi.all_states().dtype == np.int8

    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2)
    assert hi.all_states().dtype == np.int8
    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(2, 2))
    assert hi.all_states().dtype == np.int8
