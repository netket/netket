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

from functools import partial
from typing import Any, Callable, Optional, Union
from textwrap import dedent

import numpy as np

import jax
from flax import linen as nn
from flax import serialization
from jax import numpy as jnp

from netket.hilbert import AbstractHilbert, ContinuousHilbert

from netket.utils import mpi, wrap_afun
from netket.utils.types import PyTree, DType

from netket.utils.deprecation import warn_deprecation
from netket.utils import struct

from netket.utils.config_flags import config
from netket.jax.sharding import (
    extract_replicated,
    gather,
    distribute_to_devices_along_axis,
    device_count,
    with_samples_sharding_constraint,
)

from .base import Sampler, SamplerState
from .rules import MetropolisRule


class MetropolisSamplerState(SamplerState):
    """
    State for a Metropolis sampler.

    Contains the current configuration, the RNG state and the (optional)
    state of the transition rule.
    """

    σ: jnp.ndarray
    """Current batch of configurations in the Markov chain."""
    rng: jnp.ndarray
    """State of the random number generator (key, in jax terms)."""
    rule_state: Optional[Any]
    """Optional state of the transition rule."""

    n_steps_proc: int = struct.field(default_factory=lambda: jnp.zeros((), dtype=int))
    """Number of moves performed along the chains in this process since the last reset."""
    n_accepted_proc: jnp.ndarray
    """Number of accepted transitions among the chains in this process since the last reset."""

    def __init__(self, σ: jnp.ndarray, rng: jnp.ndarray, rule_state: Optional[Any]):
        self.σ = σ
        self.rng = rng
        self.rule_state = rule_state

        self.n_accepted_proc = with_samples_sharding_constraint(
            jnp.zeros(σ.shape[0], dtype=int)
        )
        self.n_steps_proc = jnp.zeros((), dtype=int)
        super().__init__()

    @property
    def acceptance(self) -> float:
        """The fraction of accepted moves across all chains and MPI processes.

        The rate is computed since the last reset of the sampler.
        Will return None if no sampling has been performed since then.
        """
        if self.n_steps == 0:
            return None

        return self.n_accepted / self.n_steps

    @property
    def n_steps(self) -> int:
        """Total number of moves performed across all processes since the last reset."""
        return self.n_steps_proc * mpi.n_nodes

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        # jit sum for gda
        res, _ = mpi.mpi_sum_jax(jax.jit(jnp.sum)(self.n_accepted_proc))
        return res

    def __repr__(self):
        if self.n_steps > 0:
            acc_string = f"# accepted = {self.n_accepted}/{self.n_steps} ({self.acceptance * 100}%), "
        else:
            acc_string = ""

        return f"{type(self).__name__}({acc_string}rng state={self.rng})"


# serialization when sharded
def serialize_MetropolisSamplerState_sharding(sampler_state):
    state_dict = MetropolisSamplerState._to_flax_state_dict(
        MetropolisSamplerState._pytree__static_fields, sampler_state
    )

    for prop in ["σ", "n_accepted_proc"]:
        x = state_dict.get(prop, None)
        if x is not None and isinstance(x, jax.Array) and len(x.devices()) > 1:
            state_dict[prop] = gather(x)
    state_dict = extract_replicated(state_dict)
    return state_dict


def deserialize_MetropolisSamplerState_sharding(sampler_state, state_dict):
    for prop in ["σ", "n_accepted_proc"]:
        x = state_dict[prop]
        if x is not None:
            state_dict[prop] = distribute_to_devices_along_axis(x)

    return MetropolisSamplerState._from_flax_state_dict(
        MetropolisSamplerState._pytree__static_fields, sampler_state, state_dict
    )


if config.netket_experimental_sharding:
    # when running on multiple jax processes the σ and n_accepted_proc are not fully addressable
    # however, when serializing they need to be so here we register custom handlers which
    # gather all the data to every process.
    # when deserializing we distribute the samples again to all availale devices
    # this way it is enough to serialize on process 0, and we can restart the simulation
    # also  on a different number of devices, provided the number of samples is still
    # divisible by the new number of devices
    serialization.register_serialization_state(
        MetropolisSamplerState,
        serialize_MetropolisSamplerState_sharding,
        deserialize_MetropolisSamplerState_sharding,
        override=True,
    )


def _assert_good_sample_shape(samples, shape, dtype, obj=""):
    canonical_dtype = jax.dtypes.canonicalize_dtype(dtype)
    if samples.shape != shape or samples.dtype != canonical_dtype:
        raise ValueError(
            dedent(
                f"""

            The samples returned by the {obj} have `shape={samples.shape}` and
            `dtype={samples.dtype}`, but the sampler requires `shape={shape} and
            `dtype={canonical_dtype}` (canonicalized from {dtype}).

            If you are using a custom transition rule, check that it returns the
            correct shape and dtype.

            If you are using a built-in transition rule, there might be a mismatch
            between hilbert spaces, or it's a bug in NetKet.

            """
            )
        )


def _assert_good_log_prob_shape(log_prob, n_chains_per_rank, machine):
    if log_prob.shape != (n_chains_per_rank,):
        raise ValueError(
            dedent(
                f"""

            The output of the model {machine} has `shape={log_prob.shape}`, but
            `shape=({n_chains_per_rank},)` was expected.

            This might be because of an hilbert space mismatch or because your
            model is ill-configured.

            """
            )
        )


def _round_n_chains_to_next_multiple(
    n_chains, n_chains_per_whatever, n_whatever, whatever_str
):
    # small helper function to round the number of chains to the next multiple of [whatever]
    # here [whatever] can be e.g. mpi ranks or jax devices
    # if n_chains is None and n_chains_per_whatever is None:
    #    n_chains_per_whatever = default
    if n_chains is not None and n_chains_per_whatever is not None:
        raise ValueError(
            f"Cannot specify both `n_chains` and `n_chains_per_{whatever_str}`"
        )
    elif n_chains is not None:
        n_chains_per_whatever = max(int(np.ceil(n_chains / n_whatever)), 1)
        if n_chains_per_whatever * n_whatever != n_chains:
            if mpi.rank == 0:
                import warnings

                warnings.warn(
                    f"Using {n_chains_per_whatever} chains per {whatever_str} among {n_whatever} {whatever_str}s "
                    f"(total={n_chains_per_whatever * n_whatever} instead of n_chains={n_chains}). "
                    f"To directly control the number of chains on every {whatever_str}, specify "
                    f"`n_chains_per_{whatever_str}` when constructing the sampler. "
                    f"To silence this warning, either use `n_chains_per_{whatever_str}` or use `n_chains` "
                    f"that is a multiple of the number of {whatever_str}s",
                    category=UserWarning,
                    stacklevel=2,
                )
    return n_chains_per_whatever * n_whatever


class MetropolisSampler(Sampler):
    r"""
    Metropolis-Hastings sampler for a Hilbert space according to a specific transition rule.

    The transition rule is used to generate a proposed state :math:`s^\prime`, starting from the
    current state :math:`s`. The move is accepted with probability

    .. math::

        A(s \rightarrow s^\prime) = \mathrm{min} \left( 1,\frac{P(s^\prime)}{P(s)} e^{L(s,s^\prime)} \right) ,

    where the probability being sampled from is :math:`P(s)=|M(s)|^p`. Here :math:`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.

    The dtype of the sampled states can be chosen.
    """
    rule: MetropolisRule = None
    """The Metropolis transition rule."""
    sweep_size: int = struct.field(pytree_node=False, default=None)
    """Number of sweeps for each step along the chain. Defaults to the number
    of sites in the Hilbert space."""
    n_chains: int = struct.field(pytree_node=False)
    """Total number of independent chains across all MPI ranks and/or devices."""
    reset_chains: bool = struct.field(pytree_node=False, default=False)
    """If True, resets the chain state when `reset` is called on every new sampling."""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        rule: MetropolisRule,
        *,
        n_sweeps: int = None,
        sweep_size: int = None,
        reset_chains: bool = False,
        n_chains: Optional[int] = None,
        n_chains_per_rank: Optional[int] = None,
        machine_pow: int = 2,
        dtype: DType = float,
    ):
        """
        Constructs a Metropolis Sampler.

        Args:
            hilbert: The Hilbert space to sample.
            rule: A `MetropolisRule` to generate random transitions from a given state as
                well as uniform random states.
            n_chains: The total number of independent Markov chains across all MPI ranks.
                Either specify this or `n_chains_per_rank`. If MPI is disabled, the two are equivalent;
                if MPI is enabled and `n_chains` is specified, then every MPI rank will run
                `n_chains/mpi.n_nodes` chains. In general, we recommend specifying `n_chains_per_rank`
                as it is more portable.
            n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
                               If netket_experimental_sharding is enabled this is interpreted as the number
                               of independent chains on every jax device, and the n_chains_per_rank
                               property of the sampler will return the total number of chains on all devices.
            sweep_size: Number of sweeps for each step along the chain.
                This is equivalent to subsampling the Markov chain. (Defaults to the number of sites
                in the Hilbert space.)
            reset_chains: If True, resets the chain state when `reset` is called on every
                new sampling (default = False).
            machine_pow: The power to which the machine should be exponentiated to generate
                the pdf (default = 2).
            dtype: The dtype of the states sampled (default = np.float64).
        """

        # Validate the inputs
        if not isinstance(rule, MetropolisRule):
            raise TypeError(
                f"The second positional argument, rule, must be a MetropolisRule but "
                f"`type(rule)={type(rule)}`."
            )

        if not isinstance(reset_chains, bool):
            raise TypeError("reset_chains must be a boolean.")

        if n_sweeps is not None:
            warn_deprecation(
                "Specifying `n_sweeps` when constructing sampler is deprecated. Please use `sweep_size` instead."
            )
            if sweep_size is not None:
                raise ValueError("Cannot specify both `sweep_size` and `n_sweeps`")
            sweep_size = n_sweeps

        if sweep_size is None:
            sweep_size = hilbert.size

        # Default n_chains per rank, if unset
        if n_chains is None and n_chains_per_rank is None:
            # TODO set it to a few hundred if on GPU?
            n_chains_per_rank = 16

        n_chains = _round_n_chains_to_next_multiple(
            n_chains,
            n_chains_per_rank,
            device_count(),
            "rank",
        )

        super().__init__(
            hilbert=hilbert,
            machine_pow=machine_pow,
            dtype=dtype,
        )

        self.n_chains = n_chains
        self.reset_chains = reset_chains
        self.rule = rule
        self.sweep_size = sweep_size

    @property
    def n_sweeps(self):
        warn_deprecation(
            "`MetropolisSampler.n_sweeps` is deprecated. Please use `MetropolisSampler.sweep_size` instead."
        )
        return self.sweep_size

    def sample_next(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: Optional[SamplerState] = None,
    ) -> tuple[SamplerState, jnp.ndarray]:
        """
        Samples the next state in the Markov chain.

        Args:
            machine: A Flax module or callable with the forward pass of the log-pdf.
                If it is a callable, it should have the signature
                :code:`f(parameters, σ) -> jnp.ndarray`.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler. If not specified, then initialize and reset it.

        Returns:
            state: The new state of the sampler.
            σ: The next batch of samples.

        Note:
            The return order is inverted wrt `sample` because when called inside of
            a scan function the first returned argument should be the state.
        """
        if state is None:
            state = sampler.reset(machine, parameters)

        return sampler._sample_next(wrap_afun(machine), parameters, state)

    @partial(jax.jit, static_argnums=1)
    def _init_state(sampler, machine, parameters, key):
        key_state, key_rule = jax.random.split(key)
        rule_state = sampler.rule.init_state(sampler, machine, parameters, key_rule)
        σ = jnp.zeros((sampler.n_batches, sampler.hilbert.size), dtype=sampler.dtype)
        σ = with_samples_sharding_constraint(σ)
        state = MetropolisSamplerState(σ=σ, rng=key_state, rule_state=rule_state)
        # If we don't reset the chain at every sampling iteration, then reset it
        # now.
        if not sampler.reset_chains:
            key_state, rng = jax.jit(jax.random.split)(key_state)
            σ = sampler.rule.random_state(sampler, machine, parameters, state, rng)
            _assert_good_sample_shape(
                σ,
                (sampler.n_batches, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.random_state",
            )
            σ = with_samples_sharding_constraint(σ)
            state = state.replace(σ=σ, rng=key_state)
        return state

    @partial(jax.jit, static_argnums=1)
    def _reset(sampler, machine, parameters, state):
        rng = state.rng

        if sampler.reset_chains:
            rng, key = jax.random.split(state.rng)
            σ = sampler.rule.random_state(sampler, machine, parameters, state, rng)
            _assert_good_sample_shape(
                σ,
                (sampler.n_batches, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.random_state",
            )
            σ = with_samples_sharding_constraint(σ)
        else:
            σ = state.σ

        rule_state = sampler.rule.reset(sampler, machine, parameters, state)

        return state.replace(
            σ=σ,
            rng=rng,
            rule_state=rule_state,
            n_steps_proc=jnp.zeros_like(state.n_steps_proc),
            n_accepted_proc=jnp.zeros_like(state.n_accepted_proc),
        )

    def _sample_next(sampler, machine, parameters, state):
        """
        Implementation of `sample_next` for subclasses of `MetropolisSampler`.

        If you subclass `MetropolisSampler`, you should override this and not `sample_next`
        itself, because `sample_next` contains some common logic.
        """

        def loop_body(i, s):
            # 1 to propagate for next iteration, 1 for uniform rng and n_chains for transition kernel
            s["key"], key1, key2 = jax.random.split(s["key"], 3)

            σp, log_prob_correction = sampler.rule.transition(
                sampler, machine, parameters, state, key1, s["σ"]
            )
            _assert_good_sample_shape(
                σp,
                (sampler.n_batches, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.transition",
            )
            proposal_log_prob = sampler.machine_pow * machine.apply(parameters, σp).real
            _assert_good_log_prob_shape(proposal_log_prob, sampler.n_batches, machine)

            uniform = jax.random.uniform(key2, shape=(sampler.n_batches,))
            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(
                    proposal_log_prob - s["log_prob"] + log_prob_correction
                )
            else:
                do_accept = uniform < jnp.exp(proposal_log_prob - s["log_prob"])

            # do_accept must match ndim of proposal and state (which is 2)
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["accepted"] += do_accept

            s["log_prob"] = jax.numpy.where(
                do_accept.reshape(-1), proposal_log_prob, s["log_prob"]
            )

            return s

        new_rng, rng = jax.random.split(state.rng)

        s = {
            "key": rng,
            "σ": state.σ,
            "log_prob": sampler.machine_pow * machine.apply(parameters, state.σ).real,
            # for logging
            "accepted": state.n_accepted_proc,
        }
        s = jax.lax.fori_loop(0, sampler.sweep_size, loop_body, s)

        new_state = state.replace(
            rng=new_rng,
            σ=s["σ"],
            n_accepted_proc=s["accepted"],
            n_steps_proc=state.n_steps_proc + sampler.sweep_size * sampler.n_batches,
        )

        return new_state, new_state.σ

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_chain(sampler, machine, parameters, state, chain_length):
        """
        Samples `chain_length` batches of samples along the chains.

        Internal method used for jitting calls.

        Arguments:
            sampler: The Monte Carlo sampler.
            machine: A Flax module with the forward pass of the log-pdf.
            parameters: The PyTree of parameters of the model.
            state: The current state of the sampler.
            chain_length: The length of the chains.

        Returns:
            σ: The next batch of samples.
            state: The new state of the sampler
        """
        state, samples = jax.lax.scan(
            lambda state, _: sampler.sample_next(machine, parameters, state),
            state,
            xs=None,
            length=chain_length,
        )
        # make it (n_chains, n_samples_per_chain) as expected by netket.stats.statistics
        samples = jnp.swapaxes(samples, 0, 1)
        return samples, state

    def __repr__(sampler):
        return (
            f"{type(sampler).__name__}("
            + f"\n  hilbert = {sampler.hilbert},"
            + f"\n  rule = {sampler.rule},"
            + f"\n  n_chains = {sampler.n_chains},"
            + f"\n  sweep_size = {sampler.sweep_size},"
            + f"\n  reset_chains = {sampler.reset_chains},"
            + f"\n  machine_power = {sampler.machine_pow},"
            + f"\n  dtype = {sampler.dtype}"
            + ")"
        )

    def __str__(sampler):
        return (
            f"{type(sampler).__name__}("
            + f"rule = {sampler.rule}, "
            + f"n_chains = {sampler.n_chains}, "
            + f"sweep_size = {sampler.sweep_size}, "
            + f"reset_chains = {sampler.reset_chains}, "
            + f"machine_power = {sampler.machine_pow}, "
            + f"dtype = {sampler.dtype})"
        )


def MetropolisLocal(hilbert, **kwargs) -> MetropolisSampler:
    r"""
    Sampler acting on one local degree of freedom.

    This sampler acts locally only on one local degree of freedom :math:`s_i`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s_N`,
    where :math:`s^\prime_i \neq s_i`.

    The transition probability associated to this
    sampler can be decomposed into two steps:

    1. One of the site indices :math:`i = 1\dots N` is chosen with uniform probability.

    2. Among all the possible (:math:`m - 1`) values that :math:`s^\prime_i` can take,
    one of them is chosen with uniform probability.

    For example, in the case of spin :math:`1/2` particles, :math:`m=2`
    and the possible local values are :math:`s_i = -1,+1`.
    In this case then :class:`MetropolisLocal` is equivalent to flipping a random spin.

    In the case of bosons, with occupation numbers
    :math:`s_i = 0, 1, \dots n_{\mathrm{max}}`, :class:`MetropolisLocal`
    would pick a random local occupation number uniformly between :math:`0`
    and :math:`n_{\mathrm{max}}` except the current :math:`s_i`.

    Args:
        hilbert: The Hilbert space to sample.
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from .rules import LocalRule

    return MetropolisSampler(hilbert, LocalRule(), **kwargs)


def MetropolisExchange(
    hilbert, *, clusters=None, graph=None, d_max=1, **kwargs
) -> MetropolisSampler:
    r"""
    This sampler acts locally only on two local degree of freedom :math:`s_i` and :math:`s_j`,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_j \dots s_N`,
    where in general :math:`s^\prime_i \neq s_i` and :math:`s^\prime_j \neq s_j`.
    The sites :math:`i` and :math:`j` are also chosen to be within a maximum graph
    distance of :math:`d_{\mathrm{max}}`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A pair of indices :math:`i,j = 1\dots N`, and such that
       :math:`\mathrm{dist}(i,j) \leq d_{\mathrm{max}}`,
       is chosen with uniform probability.

    2. The sites are exchanged, i.e. :math:`s^\prime_i = s_j` and :math:`s^\prime_j = s_i`.

    Notice that this sampling method generates random permutations of the quantum
    numbers, thus global quantities such as the sum of the local quantum numbers
    are conserved during the sampling.
    This scheme should be used then only when sampling in a
    region where :math:`\sum_i s_i = \mathrm{constant}` is needed,
    otherwise the sampling would be strongly not ergodic.

    Args:
        hilbert: The Hilbert space to sample.
        d_max: The maximum graph distance allowed for exchanges.
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).

    Examples:
          Sampling from a RBM machine in a 1D lattice of spin 1/2, using
          nearest-neighbor exchanges.

          >>> import netket as nk
          >>>
          >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
          >>> hi=nk.hilbert.Spin(s=0.5, N=g.n_nodes)
          >>>
          >>> # Construct a MetropolisExchange Sampler
          >>> sa = nk.sampler.MetropolisExchange(hi, graph=g)
          >>> print(sa)
          MetropolisSampler(rule = ExchangeRule(# of clusters: 200), n_chains = 16, sweep_size = 100, reset_chains = False, machine_power = 2, dtype = <class 'float'>)
    """
    from .rules import ExchangeRule

    rule = ExchangeRule(clusters=clusters, graph=graph, d_max=d_max)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisHamiltonian(hilbert, hamiltonian, **kwargs) -> MetropolisSampler:
    r"""
    Sampling based on the off-diagonal elements of a Hamiltonian (or a generic Operator).
    In this case, the transition matrix is taken to be:

    .. math::
       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    where :math:`\theta(x)` is the Heaviside step function, and :math:`\mathcal{N}(\mathbf{s})`
    is a state-dependent normalization.
    The effect of this transition probability is then to connect (with uniform probability)
    a given state :math:`\mathbf{s}` to all those states :math:`\mathbf{s}^\prime` for which the Hamiltonian has
    finite matrix elements.
    Notice that this sampler preserves by construction all the symmetries
    of the Hamiltonian. This is in generally not true for the local samplers instead.

    This sampler only works on the CPU. To use the Hamiltonian sampler with GPUs,
    you should use :class:`netket.sampler.MetropolisHamiltonianNumpy`

    Args:
        hilbert: The Hilbert space to sample.
        hamiltonian: The operator used to perform off-diagonal transition.
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).

    Examples:
       Sampling from a RBM machine in a 1D lattice of spin 1/2

       >>> import netket as nk
       >>>
       >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
       >>> hi=nk.hilbert.Spin(s=0.5, N=g.n_nodes)
       >>>
       >>> # Transverse-field Ising Hamiltonian
       >>> ha = nk.operator.Ising(hilbert=hi, h=1.0, graph=g)
       >>>
       >>> # Construct a MetropolisHamiltonian Sampler
       >>> sa = nk.sampler.MetropolisHamiltonian(hi, hamiltonian=ha)
       >>> print(sa)
       MetropolisSampler(rule = HamiltonianRuleNumba(operator=Ising(J=1.0, h=1.0; dim=100)), n_chains = 16, sweep_size = 100, reset_chains = False, machine_power = 2, dtype = <class 'float'>)
    """
    from .rules import HamiltonianRule

    rule = HamiltonianRule(hamiltonian)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisGaussian(hilbert, sigma=1.0, **kwargs) -> MetropolisSampler:
    """This sampler acts on all particle positions simultaneously
    and proposes a new state according to a Gaussian distribution
    with width `sigma`.

    Args:
        hilbert: The continuous Hilbert space to sample.
        sigma: The width of the Gaussian proposal distribution (default = 1.0).
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    if not isinstance(hilbert, ContinuousHilbert):
        raise ValueError("This sampler only works for Continuous Hilbert spaces.")

    from .rules import GaussianRule

    rule = GaussianRule(sigma)
    return MetropolisSampler(hilbert, rule, **kwargs)


def MetropolisAdjustedLangevin(
    hilbert, dt=0.001, chunk_size=None, **kwargs
) -> MetropolisSampler:
    r"""This sampler acts on all particle positions simultaneously
    and takes a Langevin step [1]:

    .. math::
       x_{t+dt} = x_t + dt \nabla_x \log p(x) \vert_{x=x_t} + \sqrt{2 dt}\eta,

    where  :math:`\eta` is normal distributed noise :math:`\eta \sim \mathcal{N}(0,1)`.
    This sampler only works for continuous Hilbert spaces.

    [1]: https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm

    Args:
        hilbert: The continuous Hilbert space to sample.
        dt: Time step size for the Langevin dynamics (noise with variance 2*dt).
        chunk_size: Chunk size to compute the gradients of the log probability.
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        sweep_size: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    if not isinstance(hilbert, ContinuousHilbert):
        raise ValueError("This sampler only works for Continuous Hilbert spaces.")

    from .rules import LangevinRule

    rule = LangevinRule(dt=dt, chunk_size=chunk_size)
    return MetropolisSampler(hilbert, rule, **kwargs)
