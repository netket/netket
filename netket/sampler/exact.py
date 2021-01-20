import jax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb

from flax import struct

from typing import Any

from netket.nn import to_array
from netket.hilbert import AbstractHilbert

from .base import sampler, Sampler, SamplerState


@struct.dataclass
class ExactSamplerState(SamplerState):
    pdf: Any
    rng: Any


@sampler("ExactSampler")
class ExactSampler_(Sampler):
    def _init_state(sampler, machine, params, key):
        pdf = jnp.zeros(sampler.hilbert.n_states, dtype=jnp.float32)
        return ExactSamplerState(pdf=pdf, rng=key)

    def _reset(sampler, machine, parameters, state):
        pdf = jnp.absolute(
            to_array(sampler.hilbert, machine, parameters) ** sampler.machine_pow
        )
        pdf = pdf / pdf.sum()

        return state.replace(pdf=pdf)

    def _sample_next(sampler, machine, parameters, state):
        new_rng, rng = jax.random.split(state.rng)
        numbers = jax.random.choice(
            rng,
            sampler.hilbert.n_states,
            shape=(sampler.n_chains,),
            replace=True,
            p=state.pdf,
        )

        # We use a host-callback to convert integers labelling states to
        # valid state-arrays because that code is written with numba and
        # we have not yet converted it to jax.
        cb = lambda numbers: host_numbers_to_states(sampler.hilbert, numbers)

        sample = hcb.call(
            cb,
            numbers,
            result_shape=jax.ShapeDtypeStruct(
                (sampler.n_chains, sampler.hilbert.size), jnp.float64
            ),
        )

        new_state = state.replace(rng=new_rng)
        return new_state, jnp.asarray(sample, dtype=sampler.dtype)

    def __repr__(sampler):
        return (
            "ExactSampler("
            + "\n  hilbert = {},".format(sampler.hilbert)
            + "\n  n_chains = {},".format(sampler.n_chains)
            + "\n  machine_power = {})".format(sampler.machine_pow)
        )


from netket.legacy.sampler import ExactSampler as LegacyExactSampler
from netket.legacy.machine import AbstractMachine
from netket.utils import wraps_legacy


@wraps_legacy(LegacyExactSampler, "machine", AbstractMachine)
def ExactSampler(hilbert, machine_power: int = 2, sample_size: int = 8):
    """
    Constructs an exact sampler
    """
    return ExactSampler_(
        hilbert=hilbert,
        machine_pow=machine_power,
        n_chains=sample_size,
    )


def host_numbers_to_states(hilbert, numbers):
    return hilbert.numbers_to_states(numbers)
