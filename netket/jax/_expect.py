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

# The score function (REINFORCE) gradient estimator of an expectation

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import statistics as mpi_statistics, mean as mpi_mean, Stats
from netket.utils.types import PyTree

from netket.jax import apply_chunked, vjp as nkvjp


def expect(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    σ: jnp.ndarray,
    *expected_fun_args,
    n_chains: int | None = None,
    chunk_size: int | None = None,
    in_axes: tuple[int | None, ...] | None = None,
) -> tuple[jnp.ndarray, Stats]:
    r"""
    Computes the expectation value over a log-pdf, equivalent to

    .. math::

        \langle f \rangle = \mathbb{E}_{\sigma \sim p(x)}[f(\sigma)] = \sum_{\mathbf{x}} p(\mathbf{x}) f(\mathbf{x})

    where the evaluation of the expectation value is approximated using the sample average, with
    samples :math:`\sigma` that are assumed to be drawn from the probability distribution :math:`p(x)`.

    .. math::

        \langle f \rangle \approx \frac{1}{N} \sum_{i=1}^{N} f(\sigma_i)

    This function ensures that the backward pass is computed correctly, by first differentiating the first equation
    above, and then by approximating the expectation values again using the sample average. The resulting
    backward gradient is

    .. math::

            \nabla \langle f \rangle = \mathbb{E}_{\sigma \sim p(x)}[(\nabla \log p(\sigma)) f(\sigma) + \nabla f(\sigma)]

    where again, the expectation values are comptued using the sample average.

    .. note::

        When using this function together with MPI, you have to pay particular attention. This is because inside the function `f` that is differentiated
        a mean over the MPI ranks (`mpi_mean(term1 + term2, axis=0)`) appears. Therefore, when doing the backward pass this results in a division of the outputs
        from the previous steps by a factor equal to the number of MPI ranks, and so the final gradient on each MPI rank is rescaled as well.
        To cope with this, it is important to sum over the ranks the gradient computed after AD, for example using the function `nk.utils.mpi.mpi_sum_jax`.
        See the following example for more details.

    Example:
        Compute the energy gradient using `nk.jax.expect` on more MPI ranks.

        >>> import netket as nk
        >>> import jax
        >>> import jax.numpy as jnp
        >>>
        >>> hi = nk.hilbert.Spin(s=0.5, N=20)
        >>> graph = nk.graph.Chain(length=20)
        >>> H = nk.operator.IsingJax(hi, graph, h=1.0)
        >>> vstate = nk.vqs.MCState(sampler=nk.sampler.MetropolisLocal(hi, n_chains_per_rank=16), model=nk.models.RBM(alpha=1, param_dtype=complex), n_samples=100000)
        >>>
        >>> afun = vstate._apply_fun
        >>> pars = vstate.parameters
        >>> model_state = vstate.model_state
        >>> log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real
        >>>
        >>> σ = vstate.samples
        >>> σ = σ.reshape(-1, σ.shape[-1])
        >>>
        >>> # The function that we want to differentiate wrt pars and σ
        >>> # Note that we do not want to compute the gradient wrt model_state, so
        >>> # we capture it inside of this function.
        >>> def expect(pars, σ):
        ...
        ...     # The log probability distribution we have generated samples σ from.
        ...     def log_pdf(pars, σ):
        ...         W = {"params": pars, **model_state}
        ...         return 2 * afun(W, σ).real
        ...
        ...     def expected_fun(pars, σ):
        ...         W = {"params": pars, **model_state}
        ...         # Get connected samples
        ...         σp, mels = H.get_conn_padded(σ)
        ...         logpsi_σ = afun(W, σ)
        ...         logpsi_σp = afun(W, σp)
        ...         logHpsi_σ = jax.scipy.special.logsumexp(logpsi_σp, b=mels, axis=1)
        ...         return jnp.exp(logHpsi_σ - logpsi_σ)
        ...     return nk.jax.expect(log_pdf, expected_fun, pars, σ)[0]
        >>>
        >>> E, E_vjp_fun = nk.jax.vjp(expect, pars, σ)
        >>> grad = E_vjp_fun(jnp.ones_like(E))[0]
        >>> grad = jax.tree_util.tree_map(lambda x: nk.utils.mpi.mpi_sum_jax(x)[0], grad)

    Args:
        log_pdf: The log-pdf function from which the samples are drawn. This should output real values, and have a signature
            :code:`log_pdf(pars, σ) -> jnp.ndarray`.
        expected_fun: The function to compute the expectation value of. This should have a signature
            :code:`expected_fun(pars, σ, *expected_fun_args) -> jnp.ndarray`.
        pars: The parameters of the model.
        σ: The samples to compute the expectation value over.
        expected_fun_args: Additional arguments to pass to the expected_fun function (will be differentiated; to avoid
            differentiation, capture them as constants inside of the expected_fun).
        n_chains: The number of chains to use in the computation. If None, the number of chains is inferred from the shape of the input.
        chunk_size: The size of the chunks to use in the computation. If None, no chunking is used.
        in_axes: The axes along which to perform the chunking. If none, only the samples are chunked, otherwise this must be
            the sharding declaration of the samples and the additional arguments to the expected_fun function (must have length
            equal to the number of expected_fun_args + 2).

    Returns:
        A tuple where the first element is the scalar value containing the expectation value, and the second element is
        a :class:`netket.stats.Stats` object containing the statistics (including the mean) of the expectation value.
    """
    return _expect(
        n_chains,
        chunk_size,
        in_axes,
        log_pdf,
        expected_fun,
        pars,
        σ,
        *expected_fun_args,
    )


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def _expect(
    n_chains, chunk_size, in_axes, log_pdf, expected_fun, pars, σ, *expected_fun_args
):
    if chunk_size is not None:
        if in_axes is None:
            # only chunk samples axis
            in_axes = (None, 0) + tuple(None for _ in expected_fun_args)
        else:
            n_args = 2 + len(expected_fun_args)
            assert n_args == len(in_axes)
        expected_fun = apply_chunked(
            expected_fun, chunk_size=chunk_size, in_axes=in_axes
        )

    L_σ = expected_fun(pars, σ, *expected_fun_args)

    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ)

    return L̄_σ.mean, L̄_σ


def _expect_fwd(
    n_chains, chunk_size, in_axes, log_pdf, expected_fun, pars, σ, *expected_fun_args
):
    if chunk_size is not None:
        if in_axes is None:
            # only chunk samples axis
            in_axes = (None, 0) + tuple(None for _ in expected_fun_args)
        else:
            n_args = 2 + len(expected_fun_args)
            assert n_args == len(in_axes)
        expected_fun = apply_chunked(
            expected_fun, chunk_size=chunk_size, in_axes=in_axes
        )

    L_σ = expected_fun(pars, σ, *expected_fun_args)

    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ

    L̄_stat = mpi_statistics(L_σ_r)

    L̄_σ = L̄_stat.mean

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars, σ, expected_fun_args, ΔL_σ)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def _expect_bwd(n_chains, chunk_size, in_axes, log_pdf, expected_fun, residuals, dout):
    pars, σ, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout

    if chunk_size is None:

        def f(pars, σ, *cost_args):
            log_p = log_pdf(pars, σ)
            term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
            term2 = expected_fun(pars, σ, *cost_args)
            out = mpi_mean(term1 + term2, axis=0)
            out = out.sum()
            return out

    else:
        if in_axes is None:
            in_axes = (
                None,
                0,
            ) + tuple(None for _ in cost_args)

        def chunked_f(ΔL_σ, pars, σ, *cost_args):
            log_p = apply_chunked(log_pdf, chunk_size=chunk_size, in_axes=(None, 0))(
                pars, σ
            )
            term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
            term2 = apply_chunked(expected_fun, chunk_size=chunk_size, in_axes=in_axes)(
                pars, σ, *cost_args
            )
            out = mpi_mean(term1 + term2, axis=0)
            out = out.sum()
            return out

        # capture ΔL_σ to not differentiate through it
        f = partial(chunked_f, ΔL_σ)

    _, pb = nkvjp(f, pars, σ, *cost_args)
    grad_f = pb(dL̄)
    return grad_f


_expect.defvjp(_expect_fwd, _expect_bwd)
