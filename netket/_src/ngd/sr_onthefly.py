from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

import netket.jax as nkjax
from netket.jax import tree_axpy
from netket.optimizer.linear_operator import LinearOperator, SolverT, Uninitialized
from netket.utils import HashablePartial, timing
from netket.utils.types import Array, PyTree

from netket._src.ngd.kwargs import ensure_accepts_kwargs
from netket._src.ngd.sr_srt_common import _prepare_weights


def _broadcast_weights(weights: Array, x: jax.Array) -> jax.Array:
    return jax.lax.broadcast_in_dim(weights, x.shape, (0,))


def _match_structure(x, target):
    if jax.tree_util.tree_structure(x) == jax.tree_util.tree_structure(target):
        return x

    return jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(target),
        jax.tree_util.tree_leaves(x),
    )


def _center_and_scale_output(
    x: jax.Array,
    *,
    pdf: Array | None = None,
) -> jax.Array:
    if pdf is None:
        return (x - jnp.mean(x, axis=0, keepdims=True)) / x.shape[0]

    weights = _broadcast_weights(pdf, x)
    mean = jnp.sum(weights * x, axis=0, keepdims=True)
    return weights * (x - mean)


def _force_cotangents(
    local_energies: jax.Array,
    *,
    mode: str,
    pdf: Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    local_energies = local_energies.flatten()

    if pdf is None:
        local_energies_mean = jnp.mean(local_energies)
    else:
        local_energies_mean = jnp.sum(pdf * local_energies)
    local_energies_centered = local_energies - local_energies_mean

    if mode == "complex":
        rhs = 2.0 * jnp.stack(
            [jnp.real(local_energies_centered), jnp.imag(local_energies_centered)],
            axis=-1,
        )
    elif mode == "real":
        rhs = 2.0 * jnp.real(local_energies_centered)
    else:
        raise NotImplementedError()

    if pdf is None:
        rhs_force = rhs / local_energies.size
        dv = rhs / jnp.sqrt(local_energies.size)
    else:
        rhs_force = _broadcast_weights(pdf, rhs) * rhs
        dv = _broadcast_weights(jnp.sqrt(pdf), rhs) * rhs

    if mode == "complex":
        dv = jax.lax.collapse(dv, 0, 2)

    return rhs_force, dv


def _mat_vec(jvp_fn, v, diag_shift, *, pdf=None):
    vjp_fn = jax.linear_transpose(jvp_fn, v)

    w = jvp_fn(v)
    w = _center_and_scale_output(w, pdf=pdf)
    (res,) = vjp_fn(w)
    res = _match_structure(res, v)

    return tree_axpy(diag_shift, v, res)


@partial(jax.jit, static_argnums=0)
def mat_vec_factory(forward_fn, params, model_state, samples, pdf=None):
    def fun(W):
        return forward_fn(W, samples, model_state)

    _, jvp_fn = jax.linearize(fun, params)
    return Partial(_mat_vec, jvp_fn, pdf=pdf)


def _mat_vec_chunked(
    forward_fn, params, model_state, samples, v, diag_shift, chunk_size, pdf=None
):
    assert samples.ndim == 2

    def jvp_f_chunk(parameters, model_state, vector, samples):
        f = lambda pars: forward_fn(pars, samples, model_state)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    w = nkjax.apply_chunked(
        jvp_f_chunk, in_axes=(None, None, None, 0), chunk_size=chunk_size
    )(params, model_state, v, samples)
    w = _center_and_scale_output(w, pdf=pdf)

    vjp_fun = nkjax.vjp_chunked(
        forward_fn,
        params,
        samples,
        model_state,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=(1, 2),
    )
    (res,) = vjp_fun(w)
    res = _match_structure(res, params)

    return tree_axpy(diag_shift, v, res)


@partial(jax.jit, static_argnums=(0, 5))
def mat_vec_chunked_factory(
    forward_fn, params, model_state, samples, pdf=None, chunk_size=None
):
    return Partial(
        partial(_mat_vec_chunked, forward_fn, chunk_size=chunk_size),
        params,
        model_state,
        samples,
        pdf=pdf,
    )


def SROnTheFly_DefaultConstructor(
    apply_fun,
    parameters,
    model_state,
    samples,
    pdf=None,
    *,
    chunk_size: int | None = None,
    **kwargs,
) -> "SROnTheFlyT":
    if pdf is not None:
        if not pdf.shape == samples.shape[:-1]:
            raise ValueError(
                "The shape of pdf must match the shape of the samples, "
                f"instead you provided (pdf.shape={pdf.shape}) != "
                f"(samples.shape={samples.shape[:-1]})"
            )
        if pdf.ndim >= 2:
            pdf = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(pdf, 0, 2)

    if samples.ndim >= 3:
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)

    n_samples_per_rank = samples.shape[0] // jax.device_count()
    if chunk_size is None or chunk_size >= n_samples_per_rank:
        mv_factory = mat_vec_factory
        chunking = False
    else:
        mv_factory = HashablePartial(mat_vec_chunked_factory, chunk_size=chunk_size)
        chunking = True

    mat_vec = mv_factory(
        forward_fn=apply_fun,
        params=parameters,
        model_state=model_state,
        samples=samples,
        pdf=pdf,
    )

    return SROnTheFlyT(
        _mat_vec=mat_vec,
        _params=parameters,
        _chunking=chunking,
        **kwargs,
    )


@struct.dataclass
class SROnTheFlyT(LinearOperator):
    _mat_vec: Callable[[PyTree, float], PyTree] = Uninitialized  # type: ignore
    _params: PyTree = Uninitialized  # type: ignore
    _chunking: bool = struct.field(pytree_node=False, default=False)

    def __matmul__(self, y):
        return onthefly_mat_treevec(self, y)

    def _solve(
        self, solve_fun: SolverT, y: PyTree, *, x0: PyTree | None, **kwargs
    ) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0, **kwargs)

    def to_dense(self) -> jnp.ndarray:
        return _to_dense(self)

    def __repr__(self):
        return f"SROnTheFly(diag_shift={self.diag_shift})"


@jax.jit
def onthefly_mat_treevec(
    S: SROnTheFlyT, vec: PyTree | jnp.ndarray
) -> PyTree | jnp.ndarray:
    if hasattr(vec, "ndim"):
        if not vec.ndim == 1:
            raise ValueError("Unsupported mat-vec for chunks of vectors")
        if not nkjax.tree_size(S._params) == vec.size:
            raise ValueError(
                """Size mismatch between number of parameters ({nkjax.tree_size(S._params)})
                                and vector size {vec.size}.
                             """
            )

        _, unravel = ravel_pytree(S._params)
        vec = unravel(vec)
        ravel_result = True
    else:
        ravel_result = False

    vec = _match_structure(vec, S._params)
    vec = nkjax.tree_cast(vec, S._params)
    res = S._mat_vec(vec, S.diag_shift)

    if ravel_result:
        res, _ = ravel_pytree(res)

    return res


@jax.jit
def _solve(
    self: SROnTheFlyT, solve_fun, y: PyTree, *, x0: PyTree | None, **kwargs
) -> PyTree:
    y = _match_structure(y, self._params)
    y = nkjax.tree_cast(y, self._params)

    if x0 is None:
        x0 = jax.tree_util.tree_map(jnp.zeros_like, y)
    else:
        x0 = _match_structure(x0, self._params)

    solve_fun = ensure_accepts_kwargs(solve_fun, "dv")
    out, info = solve_fun(self, y, x0=x0, **kwargs)

    return out, info


@jax.jit
def _to_dense(self: SROnTheFlyT) -> jnp.ndarray:
    n_parameters = nkjax.tree_size(self._params)
    eye = jax.numpy.eye(n_parameters)

    if self._chunking:
        _, out = jax.lax.scan(lambda _, x: (None, self @ x), None, eye)
    else:
        out = jax.vmap(lambda x: self @ x, in_axes=0)(eye)

    if jnp.iscomplexobj(out):
        out = out.T

    return out


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
def sr_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: PyTree | None = None,
    chunk_size: int | None = None,
    weights: Array | None = None,
):
    if proj_reg is not None:
        raise ValueError("proj_reg not implemented for SR")

    if weights is not None and weights.ndim >= 2:
        weights = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(weights, 0, 2)
    if samples.ndim >= 3:
        samples = jax.jit(jax.lax.collapse, static_argnums=(1, 2))(samples, 0, 2)

    pdf, _ = _prepare_weights(weights, samples.shape[0])

    parameters_real, rss = nkjax.tree_to_real(parameters)

    def _apply_fn(parameters_real, samples, model_state):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            return jnp.stack((log_amp.real, log_amp.imag), axis=-1)
        return log_amp.real

    S = SROnTheFly_DefaultConstructor(
        _apply_fn,
        parameters_real,
        model_state,
        samples,
        pdf=pdf,
        chunk_size=chunk_size,
        diag_shift=diag_shift,
    )

    rhs_cotangent, dv = _force_cotangents(local_energies, mode=mode, pdf=pdf)

    if not S._chunking:
        _, vjp_fun = jax.vjp(
            lambda pars: _apply_fn(pars, samples, model_state),
            parameters_real,
        )
        (forces,) = vjp_fun(rhs_cotangent)
        forces = _match_structure(forces, parameters_real)
    else:
        vjp_fun = nkjax.vjp_chunked(
            _apply_fn,
            parameters_real,
            samples,
            model_state,
            chunk_size=chunk_size,
            chunk_argnums=1,
            nondiff_argnums=(1, 2),
        )
        (forces,) = vjp_fun(rhs_cotangent)
        forces = _match_structure(forces, parameters_real)

    if momentum is not None:
        if old_updates is None:
            old_updates = jax.tree_util.tree_map(jnp.zeros_like, parameters_real)
        forces = tree_axpy(diag_shift * momentum, old_updates, forces)

    updates, info = S.solve(solver_fn, forces, dv=dv)

    if momentum is not None:
        old_updates = updates

    return rss(updates), old_updates, info
