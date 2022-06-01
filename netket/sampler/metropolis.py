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
from typing import Any, Callable, Optional, Tuple, Union
from textwrap import dedent

import jax
from flax import linen as nn
from jax import numpy as jnp

from netket.hilbert import AbstractHilbert, ContinuousHilbert

from netket.utils import mpi, wrap_afun
from netket.utils.types import PyTree

from netket.utils.deprecation import deprecated, warn_deprecation
from netket.utils import struct

from .base import Sampler, SamplerState
from .rules import MetropolisRule


@struct.dataclass
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

    # those are initialised to 0. We want to initialise them to zero arrays because they can
    # be passed to jax jitted functions that require type invariance to avoid recompilation
    n_steps_proc: int = struct.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int64)
    )
    """Number of moves performed along the chains in this process since the last reset."""
    n_accepted_proc: int = struct.field(
        default_factory=lambda: jnp.zeros((), dtype=jnp.int64)
    )
    """Number of accepted transitions among the chains in this process since the last reset."""

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
    @deprecated(
        """Please use the attribute `.acceptance` instead of
        `.acceptance_ratio`. The new attribute `.acceptance` returns the
        acceptance ratio ∈ [0,1], instead of the current `acceptance_ratio`
        returning a percentage, which is a bug."""
    )
    def acceptance_ratio(self):
        """DEPRECATED: Please use the attribute `.acceptance` instead of
        `.acceptance_ratio`. The new attribute `.acceptance` returns the
        acceptance ratio ∈ [0,1], instead of the current `acceptance_ratio`
        returning a percentage, which is a bug.

        The percentage of accepted moves across all chains and MPI processes.

        The rate is computed since the last reset of the sampler.
        Will return None if no sampling has been performed since then.
        """
        return self.acceptance * 100

    @property
    def n_steps(self) -> int:
        """Total number of moves performed across all processes since the last reset."""
        return self.n_steps_proc * mpi.n_nodes

    @property
    def n_accepted(self) -> int:
        """Total number of moves accepted across all processes since the last reset."""
        res, _ = mpi.mpi_sum_jax(self.n_accepted_proc)
        return res

    def __repr__(self):
        if self.n_steps > 0:
            acc_string = "# accepted = {}/{} ({}%), ".format(
                self.n_accepted, self.n_steps, self.acceptance * 100
            )
        else:
            acc_string = ""

        return f"{type(self).__name__}({acc_string}rng state={self.rng})"


def _assert_good_sample_shape(samples, shape, dtype, obj=""):
    if samples.shape != shape or samples.dtype != dtype:
        raise ValueError(
            dedent(
                f"""

            The samples returned by the {obj} have `shape={samples.shape}` and
            `dtype={samples.dtype}`, but the sampler requires `shape={shape} and
            `dtype={dtype}`.

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


@struct.dataclass
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
    n_sweeps: int = struct.field(pytree_node=False, default=None)
    """Number of sweeps for each step along the chain. Defaults to the number
    of sites in the Hilbert space."""
    reset_chains: bool = struct.field(pytree_node=False, default=False)
    """If True, resets the chain state when `reset` is called on every new sampling."""

    def __pre_init__(self, hilbert: AbstractHilbert, rule: MetropolisRule, **kwargs):
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
            n_sweeps: Number of sweeps for each step along the chain.
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

        if "n_chains" not in kwargs and "n_chains_per_rank" not in kwargs:
            kwargs["n_chains_per_rank"] = 16

        # process arguments in the base
        args, kwargs = super().__pre_init__(hilbert=hilbert, **kwargs)

        kwargs["rule"] = rule

        # deprecation warnings
        if "reset_chain" in kwargs:
            warn_deprecation(
                "The keyword argument `reset_chain` is deprecated in favour of `reset_chains`"
            )
            kwargs["reset_chains"] = kwargs.pop("reset_chain")

        return args, kwargs

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.reset_chains, bool):
            raise TypeError("reset_chains must be a boolean.")

        # Default value of n_sweeps
        if self.n_sweeps is None:
            object.__setattr__(self, "n_sweeps", self.hilbert.size)

    def sample_next(
        sampler,
        machine: Union[Callable, nn.Module],
        parameters: PyTree,
        state: Optional[SamplerState] = None,
    ) -> Tuple[SamplerState, jnp.ndarray]:
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

    def _init_state(sampler, machine, params, key):
        key_state, key_rule = jax.random.split(key, 2)
        rule_state = sampler.rule.init_state(sampler, machine, params, key_rule)
        σ = jnp.zeros(
            (sampler.n_chains_per_rank, sampler.hilbert.size), dtype=sampler.dtype
        )

        state = MetropolisSamplerState(σ=σ, rng=key_state, rule_state=rule_state)

        # If we don't reset the chain at every sampling iteration, then reset it
        # now.
        if not sampler.reset_chains:
            key_state, rng = jax.random.split(key_state)
            σ = sampler.rule.random_state(sampler, machine, params, state, rng)
            _assert_good_sample_shape(
                σ,
                (sampler.n_chains_per_rank, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.random_state",
            )
            state = state.replace(σ=σ, rng=key_state)

        return state

    def _reset(sampler, machine, parameters, state):
        new_rng, rng = jax.random.split(state.rng)

        if sampler.reset_chains:
            σ = sampler.rule.random_state(sampler, machine, parameters, state, rng)
            _assert_good_sample_shape(
                σ,
                (sampler.n_chains_per_rank, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.random_state",
            )
        else:
            σ = state.σ

        rule_state = sampler.rule.reset(sampler, machine, parameters, state)

        return state.replace(
            σ=σ, rng=new_rng, rule_state=rule_state, n_steps_proc=0, n_accepted_proc=0
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
                (sampler.n_chains_per_rank, sampler.hilbert.size),
                sampler.dtype,
                f"{sampler.rule}.transition",
            )
            proposal_log_prob = sampler.machine_pow * machine.apply(parameters, σp).real
            _assert_good_log_prob_shape(
                proposal_log_prob, sampler.n_chains_per_rank, machine
            )

            uniform = jax.random.uniform(key2, shape=(sampler.n_chains_per_rank,))
            if log_prob_correction is not None:
                do_accept = uniform < jnp.exp(
                    proposal_log_prob - s["log_prob"] + log_prob_correction
                )
            else:
                do_accept = uniform < jnp.exp(proposal_log_prob - s["log_prob"])

            # do_accept must match ndim of proposal and state (which is 2)
            s["σ"] = jnp.where(do_accept.reshape(-1, 1), σp, s["σ"])
            s["accepted"] += do_accept.sum()

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
        s = jax.lax.fori_loop(0, sampler.n_sweeps, loop_body, s)

        new_state = state.replace(
            rng=new_rng,
            σ=s["σ"],
            n_accepted_proc=s["accepted"],
            n_steps_proc=state.n_steps_proc
            + sampler.n_sweeps * sampler.n_chains_per_rank,
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

        return samples, state

    def __repr__(sampler):
        return (
            f"{type(sampler).__name__}("
            + "\n  hilbert = {},".format(sampler.hilbert)
            + "\n  rule = {},".format(sampler.rule)
            + "\n  n_chains = {},".format(sampler.n_chains)
            + "\n  n_sweeps = {},".format(sampler.n_sweeps)
            + "\n  reset_chains = {},".format(sampler.reset_chains)
            + "\n  machine_power = {},".format(sampler.machine_pow)
            + "\n  dtype = {}".format(sampler.dtype)
            + ")"
        )

    def __str__(sampler):
        return (
            f"{type(sampler).__name__}("
            + "rule = {}, ".format(sampler.rule)
            + "n_chains = {}, ".format(sampler.n_chains)
            + "n_sweeps = {}, ".format(sampler.n_sweeps)
            + "reset_chains = {}, ".format(sampler.reset_chains)
            + "machine_power = {}, ".format(sampler.machine_pow)
            + "dtype = {})".format(sampler.dtype)
        )


@deprecated(
    "The module function `sample_next` is deprecated in favor of the class method `sample_next`."
)
def sample_next(
    sampler: MetropolisSampler,
    machine: Union[Callable, nn.Module],
    parameters: PyTree,
    state: Optional[SamplerState] = None,
) -> Tuple[SamplerState, jnp.ndarray]:
    """
    Samples the next state in the Markov chain.

    Args:
        sampler: The Metropolis sampler.
        machine: A Flax module or callable with the forward pass of the log-pdf.
            If it is a callable, it should have the signature :code:`f(parameters, σ) -> jnp.ndarray`.
        parameters: The PyTree of parameters of the model.
        state: The current state of the sampler. If not specified, then initialize and reset it.

    Returns:
        state: The new state of the sampler.
        σ: The next batch of samples.
    """
    return sampler.sample_next(machine, parameters, state)


def MetropolisLocal(hilbert, *args, **kwargs) -> MetropolisSampler:
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
        n_sweeps: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    from .rules import LocalRule

    return MetropolisSampler(hilbert, LocalRule(), *args, **kwargs)


def MetropolisExchange(
    hilbert, *args, clusters=None, graph=None, d_max=1, **kwargs
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
        n_sweeps: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
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
          MetropolisSampler(rule = ExchangeRule(# of clusters: 200), n_chains = 16, n_sweeps = 100, reset_chains = False, machine_power = 2, dtype = <class 'numpy.float64'>)
    """
    from .rules import ExchangeRule

    rule = ExchangeRule(clusters=clusters, graph=graph, d_max=d_max)
    return MetropolisSampler(hilbert, rule, *args, **kwargs)


def MetropolisHamiltonian(hilbert, hamiltonian, *args, **kwargs) -> MetropolisSampler:
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
        n_sweeps: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
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
       MetropolisSampler(rule = HamiltonianRule(Ising(J=1.0, h=1.0; dim=100)), n_chains = 16, n_sweeps = 100, reset_chains = False, machine_power = 2, dtype = <class 'numpy.float64'>)
    """
    from .rules import HamiltonianRule

    rule = HamiltonianRule(hamiltonian)
    return MetropolisSampler(hilbert, rule, *args, **kwargs)


def MetropolisGaussian(hilbert, sigma=1.0, *args, **kwargs) -> MetropolisSampler:
    """This sampler acts on all particle positions simultaneously
    and proposes a new state according to a Gaussian distribution
    with width `sigma`.

    Args:
        hilbert: The continuous Hilbert space to sample.
        sigma: The width of the Gaussian proposal distribution (default = 1.0).
        n_chains: The total number of independent Markov chains across all MPI ranks. Either specify this or `n_chains_per_rank`.
        n_chains_per_rank: Number of independent chains on every MPI rank (default = 16).
        n_sweeps: Number of sweeps for each step along the chain. Defaults to the number of sites in the Hilbert space.
                This is equivalent to subsampling the Markov chain.
        reset_chains: If True, resets the chain state when `reset` is called on every new sampling (default = False).
        machine_pow: The power to which the machine should be exponentiated to generate the pdf (default = 2).
        dtype: The dtype of the states sampled (default = np.float64).
    """
    if not isinstance(hilbert, ContinuousHilbert):
        raise ValueError("This sampler only works for Continuous Hilbert spaces.")

    from .rules import GaussianRule

    rule = GaussianRule(sigma)
    return MetropolisSampler(hilbert, rule, *args, **kwargs)
