from collections.abc import Callable
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket import config
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array

from netket.jax import _ntk as nt


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "chunk_size",
        "mode",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    chunk_size: int | None = None,
):
    N_mc = local_energies.size

    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # complex: (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    # real:    (Nmc) -> (Nmc,)  - no splitting
    def _apply_fn(parameters_real, samples):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate(
                (re[:, None], im[:, None]), axis=-1
            )  # shape [N_mc,2]
        else:
            return log_amp.real  # shape [N_mc, ]

    def jvp_f_chunk(parameters, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(params, samples)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # compute rhs of the linear system
    local_energies = local_energies.flatten()
    de = local_energies - jnp.mean(local_energies)

    # At the moment the final vjp is centered by centering the auxiliary vector a.
    # This is the same as centering the jacobian but may have larger variance.
    dv = 2.0 * de / jnp.sqrt(N_mc)  # shape [N_mc,]
    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # shape [N_mc,2]
    else:
        dv = jnp.real(dv)  # shape [N_mc,]

    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, 0), chunk_size=chunk_size
            )(parameters_real, old_updates, samples)

            avg = jnp.mean(acc, axis=0)
            acc = (acc - avg) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    if mode == "complex":
        dv = jax.lax.collapse(dv, 0, 2)  # shape [2*N_mc,] or [N_mc, ] if not complex

    # Collect all samples on all MPI ranks, those label the columns of the T matrix
    all_samples = samples
    if config.netket_experimental_sharding:
        samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
        all_samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    _jacobian_contraction = nt.empirical_ntk_by_jacobian(
        f=_apply_fn,
        trace_axes=(),
        vmap_axes=0,
    )

    def jacobian_contraction(samples, all_samples, parameters_real):
        if config.netket_experimental_sharding:
            parameters_real = jax.lax.pvary(parameters_real, "S")
        if chunk_size is None:
            # STRUCTURED_DERIVATIVES returns a complex array, but the imaginary part is zero
            # shape [N_mc/p.size, N_mc, 2, 2]
            return _jacobian_contraction(samples, all_samples, parameters_real).real
        else:
            _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
            ntk_local = jax.lax.map(
                lambda batch_lattice: _jacobian_contraction(
                    samples, batch_lattice, parameters_real
                ).real,
                _all_samples,
            )
            if mode == "complex":
                return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
            else:
                return rearrange(ntk_local, "nbatches i j -> i (nbatches j)")

    # If we are sharding, use shard_map manually
    if config.netket_experimental_sharding:
        mesh = jax.sharding.get_abstract_mesh()
        # SAMPLES, ALL_SAMPLES PARAMETERS_REAL
        in_specs = (P("S", None), P(), P())
        out_specs = P("S", None)

        # By default, I'm not sure whether the jacobian_contraction of NeuralTangents
        # Is correctly automatically sharded across devices. So we force it to be
        # sharded with shard map to be sure

        jacobian_contraction = jax.shard_map(
            jacobian_contraction,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
        )

    # This disables the nkjax.sharding_decorator in here, which might appear
    # in the apply function inside.
    with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
        ntk_local = jacobian_contraction(samples, all_samples, parameters_real).real

    # shape [N_mc, N_mc, 2, 2] or [N_mc, N_mc]
    if config.netket_experimental_sharding:
        ntk = jax.lax.with_sharding_constraint(
            ntk_local, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )
    else:
        ntk = ntk_local
    if mode == "complex":
        # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J
        ntk = rearrange(ntk, "i j z w -> (i z) (j w)")

    # Center the NTK by multiplying with a carefully designed matrix
    # shape [N_mc, N_mc] symmetric matrix
    delta = jnp.eye(N_mc) - 1 / N_mc
    if mode == "complex":
        # shape [2*N_mc, 2*N_mc]
        # Gets applied to the sub-blocks corresponding to the real part and imaginary part
        delta_conc = jnp.zeros((2 * N_mc, 2 * N_mc)).at[0::2, 0::2].set(delta)
        delta_conc = delta_conc.at[1::2, 1::2].set(delta)
        delta_conc = delta_conc.at[0::2, 1::2].set(0.0)
        delta_conc = delta_conc.at[1::2, 0::2].set(0.0)
    else:
        delta_conc = delta

    # shape [2*N_mc, 2*N_mc] centering the jacobian
    ntk = (delta_conc @ (ntk @ delta_conc)) / N_mc

    # add diag shift
    ntk_shifted = ntk + diag_shift * jnp.eye(ntk.shape[0])

    # add projection regularization
    if proj_reg is not None:
        ntk_shifted = ntk_shifted + proj_reg / N_mc

    # some solvers return a tuple, some others do not.
    aus_vector = solver_fn(ntk_shifted, dv)
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
    else:
        info = {}

    if info is None:
        info = {}

    # Center the vector, equivalent to centering
    # The Jacobian
    aus_vector = aus_vector / jnp.sqrt(N_mc)
    aus_vector = delta_conc @ aus_vector

    # shape [N_mc,2]
    if mode == "complex":
        aus_vector = aus_vector.reshape(-1, 2)
    # shape [N_mc // p.size,2]
    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(
                jax.sharding.get_abstract_mesh(),
                P("S", *(None,) * (aus_vector.ndim - 1)),
            ),
        )

    # _, vjp_fun = jax.vjp(f, parameters_real)
    vjp_fun = nkjax.vjp_chunked(
        _apply_fn,
        parameters_real,
        samples,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )

    (updates,) = vjp_fun(aus_vector)  # pytree [N_params,]

    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info
