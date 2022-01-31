import jax
from jax import numpy as jnp

from netket.hilbert import Fermions2nd, LatticeFermions2nd
from netket.utils.dispatch import dispatch


def independent_permutation(key, base, batches, *args, **kwargs):
    keys = jax.random.split(key, batches)
    independent_permute = jax.vmap(
        lambda k: jax.random.permutation(k, base, *args, **kwargs)
    )
    return independent_permute(keys)


def _fermion_random_state(hilb, key, batches: int, dtype):  # noqa: F811
    if hilb.number_constrained:
        base = jnp.array([1] * hilb.n_fermions + [0] * hilb.n_holes, dtype=dtype)
        return independent_permutation(key, base, batches)
    else:
        choices = jnp.array([0, 1], dtype=dtype)
        return jax.random.choice(key, choices, shape=(batches, hilb.size), replace=True)


@dispatch
def random_state(hilb: Fermions2nd, key, batches: int, *, dtype):  # noqa: F811
    if hilb.extra_constrained:
        raise NotImplementedError("implement random state that respects constraint")
    return _fermion_random_state(hilb, key, batches, dtype)


def _fermion_flip_state_scalar(hilb, key, state, idx):
    if hilb.constrained:
        raise Exception("flip_state_scalar should not be called on constrained hilbert")
    return state.at[idx].set(1 - state[idx]), state[idx]


@dispatch
def flip_state_scalar(hilb: Fermions2nd, key, state, idx):  # noqa: F811
    return _fermion_flip_state_scalar(hilb, key, state, idx)


@dispatch
def random_state(hilb: LatticeFermions2nd, key, batches: int, *, dtype):  # noqa: F811
    if hilb.constrained_per_spin:
        n_per_spin = hilb.n_fermions_per_spin
        keys = jax.random.split(key, len(n_per_spin))
        states = [
            independent_permutation(
                keyi,
                jnp.array([1] * n + [0] * (hilb.n_sites - n), dtype=dtype),
                batches,
            )
            for keyi, n in zip(keys, n_per_spin)
        ]
        return jnp.concatenate(states, axis=-1)
    return _fermion_random_state(hilb, key, batches, dtype)


@dispatch
def flip_state_scalar(hilb: LatticeFermions2nd, key, state, idx):  # noqa: F811
    return _fermion_flip_state_scalar(hilb, key, state, idx)