# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute empirical NNGP and NTK; approximate functions via Taylor series.

All functions in this module are applicable to any JAX functions of proper
signatures (not only those from :obj:`~neural_tangents.stax`).

NNGP and NTK are computed using :obj:`~neural_tangents.empirical_nngp_fn`,
:obj:`~neural_tangents.empirical_ntk_fn`, or
:obj:`~neural_tangents.empirical_kernel_fn` (for both). The kernels have a very
specific output shape convention that may be unexpected. Further, NTK has
multiple implementations that may perform differently depending on the task.
Please read individual functions' docstrings.

For details, please see "`Fast Finite Width Neural Tangent Kernel
<https://arxiv.org/abs/2206.08720>`_".
"""

from typing import Any, Protocol, TypeVar, Union
from collections.abc import Callable, Iterable

from functools import partial
import operator

import jax
import jax.numpy as jnp
from jax.core import ShapedArray
from jax.interpreters.ad import UndefinedPrimal

from . import utils

safe_zip = partial(zip, strict=True)

PyTree = Any
_VMapAxis = PyTree | None
VMapAxisTriple = tuple[_VMapAxis, _VMapAxis, dict[str, _VMapAxis]]
VMapAxes = Union[_VMapAxis, VMapAxisTriple]
AnalyticKernelFn = Any
EmpiricalGetKernelFn = Any
MonteCarloKernelFn = Any
InitFn = Any
LayerKernelFn = Any
MaskFn = Any
Kernel = Any
Axes = Any
_ArrayOrShape = TypeVar("_ArrayOrShape", jnp.ndarray, ShapedArray)
T = TypeVar("T")
NTTree = Union[T, list["NTTree[T]"], tuple["NTTree[T]", ...], T]
NTTrees = Union[list["NTTree[T]"], tuple["NTTree[T]", ...]]


class ApplyFn(Protocol):
    """A type alias for apply functions.

    Apply functions do computations with finite-width neural networks. They are
    functions that take a PyTree of parameters and an array of inputs and produce
    an array of outputs.
    """

    def __call__(
        self, params: PyTree, inputs: NTTree[jnp.ndarray], *args, **kwargs
    ) -> NTTree[jnp.ndarray]: ...
class EmpiricalKernelFn(Protocol):
    """A type alias for empirical kernel functions computing either NTK or NNGP.

    A kernel function that produces an empirical kernel from a single
    instantiation of a neural network specified by its parameters.

    Equivalent to `EmpiricalGetKernelFn` with `get="nngp"` or `get="ntk"`.
    """

    def __call__(
        self,
        x1: NTTree[jnp.ndarray],
        x2: NTTree[jnp.ndarray] | None,
        params: PyTree,
        **kwargs,
    ) -> NTTree[jnp.ndarray]: ...


KernelFn = Union[
    AnalyticKernelFn,
    EmpiricalKernelFn,
    EmpiricalGetKernelFn,
    MonteCarloKernelFn,
]
InternalLayer = Any
InternalLayerMasked = tuple[InitFn, ApplyFn, LayerKernelFn, MaskFn]
Layer = tuple[InitFn, ApplyFn, AnalyticKernelFn]
Kernels = Union[list[Kernel], tuple[Kernel, ...]]
"Kernel inputs/outputs of `FanOut`, `FanInSum`, etc."
_VMapAxis = PyTree | None
"A `PyTree` of integers"
VMapAxisTriple = tuple[_VMapAxis, _VMapAxis, dict[str, _VMapAxis]]
VMapAxes = Union[_VMapAxis, VMapAxisTriple]
"Specifies `(input, output, kwargs)` axes for `vmap` in empirical NTK."


def empirical_ntk_by_jacobian(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    **kwargs,
) -> EmpiricalKernelFn:
    r"""Returns a function to draw a single sample the NTK of a given network `f`.

    The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
    :math:`J` is the Jacobian :math:`df/dparams` of shape
    `full_output_shape + params.shape`.

    Compute NTK by directly instantiating Jacobians and contracting.

    For best performance:
    1) pass `x2=None` if `x1 == x2;
    2) prefer square batches (i.e `x1.shape == x2.shape`);
    3) make sure to set `vmap_axes` correctly.

    .. warning::
      Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
      subject to `trace_axes` and `diagonal_axes` parameters, which make certain
      assumptions about the outputs `f(x)` that may only be true in the infinite
      width / infinite number of samples limit, or may not apply to your
      architecture. For most precise results in the context of linearized training
      dynamics of a specific finite-width network, set both `trace_axes=()` and
      `diagonal_axes=()` to obtain the kernel exactly of shape
      `zip(f(x1).shape, f(x2).shape)`.

    For networks with multiple (i.e. lists, tuples, PyTrees) outputs, in principal
    the empirical kernels will have terms measuring the covariance between the
    outputs. Here, we ignore these cross-terms and consider each output
    separately. Please raise an issue if this feature is important to you.

    Args:
      f:
        the function whose NTK we are computing. It should have the signature
        `f(params, x, **kwargs)` where `params` is a `PyTree`, `x` is a `PyTree`,
        and `f` should also return a `PyTree`.

      trace_axes:
        output axes to trace the output kernel over, i.e. compute only the trace
        of the covariance along the respective pair of axes (one pair for each
        axis in `trace_axes`). This allows to save space and compute if you are
        only interested in the respective trace, but also improve approximation
        accuracy if you know that covariance along these pairs of axes converges
        to a `constant * identity matrix` in the limit of interest (e.g.
        infinite width or infinite `n_samples`). A common use case is the channel
        / feature / logit axis, since activation slices along such axis are i.i.d.
        and the respective covariance along the respective pair of axes indeed
        converges to a constant-diagonal matrix in the infinite width or infinite
        `n_samples` limit.
        Also related to "contracting dimensions" in XLA terms.
        (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

      diagonal_axes:
        output axes to diagonalize the output kernel over, i.e. compute only the
        diagonal of the covariance along the respective pair of axes (one pair for
        each axis in `diagonal_axes`). This allows to save space and compute, if
        off-diagonal values along these axes are not needed, but also improve
        approximation accuracy if their limiting value is known theoretically,
        e.g. if they vanish in the limit of interest (e.g. infinite
        width or infinite `n_samples`). If you further know that on-diagonal
        values converge to the same constant in your limit of interest, you should
        specify these axes in `trace_axes` instead, to save even more compute and
        gain even more accuracy. A common use case is computing the variance
        (instead of covariance) along certain axes.
        Also related to "batch dimensions" in XLA terms.
        (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

      vmap_axes:
        A triple of `(in_axes, out_axes, kwargs_axes)`
        passed to `vmap` to evaluate the empirical NTK in parallel ove these axes.
        Precisely, providing this argument implies that `f(params, x, **kwargs)`
        equals to a concatenation along `out_axes` of `f` applied to slices of
        `x` and `**kwargs` along `in_axes` and `kwargs_axes`. In other words, it
        certifies that `f` can be evaluated as a `vmap` with `out_axes=out_axes`
        over `x` (along `in_axes`) and those arguments in `**kwargs` that are
        present in `kwargs_axes.keys()` (along `kwargs_axes.values()`).

        For example if `_, f, _ = nt.stax.Aggregate()`, `f` is called via
        `f(params, x, pattern=pattern)`. By default, inputs `x`, patterns
        `pattern`, and outputs of `f` are all batched along the leading `0`
        dimension, and each output `f(params, x, pattern=pattern)[i]` only
        depends on the inputs `x[i]` and `pattern[i]`. In this case, we can
        pass `vmap_axes=(0, 0, dict(pattern=0)` to specify along which dimensions
        inputs, outputs, and keyword arguments are batched respectively.

        This allows us to evaluate Jacobians much more
        efficiently. If `vmap_axes` is not a triple, it is interpreted as
        `in_axes = out_axes = vmap_axes, kwargs_axes = {}`. For example a very
        common use case is `vmap_axes=0` for a neural network with leading (`0`)
        batch dimension, both for inputs and outputs, and no interactions between
        different elements of the batch (e.g. no BatchNorm, and, in the case of
        `nt.stax`, also no Dropout). However, if there is interaction between
        batch elements or no concept of a batch axis at all, `vmap_axes` must be
        set to `None`, to avoid wrong (and potentially silent) results.

    Returns:
      A function `ntk_fn` that computes the empirical ntk.
    """

    def sum_and_contract(fx, j1, j2):
        ndim = fx.ndim
        size = utils.size_at(fx, trace_axes)

        _diagonal_axes = utils.canonicalize_axis(diagonal_axes, ndim)
        _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

        def contract(x, y):
            param_axes = list(range(x.ndim))[ndim:]
            contract_axes = _trace_axes + param_axes
            return _dot_general(x, y, contract_axes, _diagonal_axes) / size

        return jax.tree.reduce(operator.add, jax.tree.map(contract, j1, j2))

    def ntk_fn(
        x1: PyTree, x2: PyTree | None, params: PyTree, **apply_fn_kwargs
    ) -> jnp.ndarray:
        """Computes a single sample of the empirical NTK (jacobian outer product).

        Args:
          x1:
            first batch of inputs.

          x2:
            second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
            matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.

          params:
            A `PyTree` of parameters about which we would like to compute the
            neural tangent kernel.

          **apply_fn_kwargs:
            keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
            into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
            function which will be passed to `apply_fn`. In particular, the rng key
            in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
            same (if `x1==x2`) rng keys. See the `_read_key` function for more
            details.

        Returns:
          A single sample of the empirical NTK. The shape of the kernel is "almost"
          `zip(f(x1).shape, f(x2).shape)` except for:
          1) `trace_axes` are absent as they are contracted over.
          2) `diagonal_axes` are present only once.
          All other axes are present twice.
        """
        args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
            f, apply_fn_kwargs, params, vmap_axes, x1, x2
        )

        def j_fn(x, *args):
            _kwargs = {k: v for k, v in zip(keys, args)}
            fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **_kwargs)
            jx = jax.jacobian(fx)(params)
            return jx

        if not utils.all_none(x_axis) or not utils.all_none(kw_axes):
            in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
            j_fn = jax.vmap(j_fn, in_axes=in_axes, out_axes=fx_axis)

        j1 = j_fn(x1, *args1)
        j2 = j_fn(x2, *args2) if not utils.all_none(x2) else j1
        ntk = jax.tree.map(sum_and_contract, fx1, j1, j2)
        return ntk

    return ntk_fn


## utils


def _get_f_params(
    f: Callable,
    x: PyTree,
    x_axis: PyTree,
    fx_axis: PyTree,
    kw_axes: dict[str, PyTree],
    **apply_fn_kwargs,
) -> Callable[[PyTree], PyTree]:
    x = _expand_dims(x, x_axis)

    apply_fn_kwargs = {
        k: _expand_dims(v, kw_axes[k]) if k in kw_axes else v
        for k, v in apply_fn_kwargs.items()
    }

    def _f(p: PyTree) -> PyTree:
        fx = f(p, x, **apply_fn_kwargs)
        return _squeeze(fx, fx_axis)

    return _f


def _ndim(x: PyTree) -> PyTree:
    return jax.tree.map(lambda x: x.ndim, x)


def _mod(x: PyTree | None, y: PyTree) -> PyTree:
    if x is None:
        return None
    return jax.tree.map(operator.mod, x, y)


def _squeeze(x: PyTree, axis: PyTree | None) -> PyTree:
    if axis is None:
        return x

    def squeeze(x: jnp.ndarray, axis: None | int | tuple[int, ...]) -> jnp.ndarray:
        """`np.squeeze` analog working with 0-sized axes."""
        if isinstance(axis, int):
            axis = (axis,)

        non_zero_axes = tuple()
        shift = 0

        for a in sorted(axis):
            if x.shape[a - shift] == 0:
                new_shape = x.shape[:a] + x.shape[a + 1 :]
                if utils.size_at(new_shape) == 0:
                    x = x.reshape(new_shape)
                else:
                    x = jnp.zeros(new_shape, x.dtype)

                shift += 1
            else:
                non_zero_axes += (a - shift,)

        return jnp.squeeze(x, non_zero_axes)

    return jax.tree.map(squeeze, x, axis)


def _expand_dims_array(x: _ArrayOrShape, axis: int) -> _ArrayOrShape:
    def expand(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.expand_dims(x, axis)

    if isinstance(x, ShapedArray):
        return jax.eval_shape(expand, x)

    if isinstance(x, jnp.ndarray):
        return expand(x)

    raise TypeError(type(x), x)


def _expand_dims(
    x: None | PyTree | UndefinedPrimal, axis: PyTree | None
) -> PyTree | None:
    if axis is None or x is None or isinstance(x, UndefinedPrimal):
        return x
    return jax.tree.map(_expand_dims_array, x, axis)


def _get_args(
    f: Callable,
    apply_fn_kwargs: dict[str, PyTree],
    params: PyTree,
    vmap_axes: VMapAxes,
    x1: PyTree,
    x2: PyTree,
):
    kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

    fx1 = jax.eval_shape(f, params, x1, **kwargs1)
    fx2 = fx1 if utils.all_none(x2) else jax.eval_shape(f, params, x2, **kwargs2)

    x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x1, fx1, **kwargs1)

    keys = apply_fn_kwargs.keys()
    args1 = tuple(kwargs1[k] for k in keys)
    args2 = tuple(kwargs2[k] for k in keys)
    return args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis


def _canonicalize_axes(
    vmap_axes: VMapAxes | None, x: PyTree, fx: PyTree, **kwargs
) -> VMapAxisTriple:
    if isinstance(vmap_axes, tuple) and len(vmap_axes) == 3:
        x_axis, fx_axis, kw_axes = vmap_axes
    else:
        x_axis, fx_axis, kw_axes = vmap_axes, vmap_axes, {}

    if isinstance(x_axis, int):
        x_axis = jax.tree.map(lambda _: x_axis, x)

    if isinstance(fx_axis, int):
        fx_axis = jax.tree.map(lambda _: fx_axis, fx)

    if isinstance(kw_axes, int):
        kw_axes = jax.tree.map(lambda _: kw_axes, kwargs)

    x_axis = _mod(x_axis, _ndim(x))
    fx_axis = _mod(fx_axis, _ndim(fx))
    kw_axes = _mod(kw_axes, {k: _ndim(kwargs[k]) for k in kw_axes})
    return x_axis, fx_axis, kw_axes


def _get_res_batch_dims(
    contracting_dims: Iterable[int], batch_dims: Iterable[int]
) -> list[int]:
    res_batch_dims = [2 * b - i for i, b in enumerate(batch_dims)]
    for i, b in enumerate(batch_dims):
        for c in contracting_dims:
            if b > c:
                res_batch_dims[i] -= 2
    return res_batch_dims


def _dot_general(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    contracting_dims: Axes,
    batch_dims: Axes,
    precision=None,
) -> jnp.ndarray:
    """`jax.lax.dot_general` with preserved dims order and shared lhs / rhs dims.

    Precisely, returns `jax.lax.dot_general(lhs, rhs, dimension_numbers)` where
    `dimension_numbers == ((contracting_dims, contracting_dims),
                           (batch_dims, batch_dims))`,
    but preserves the dimension order in the output. See XLA's
     `DotGeneral<https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`.

    Args:
      lhs: array.
      rhs: array, must have the same dimensionality as `lhs`.
      contracting_dims: contracting dimensions.
      batch_dims: batch dimensions.
      precision: Optional. Either `None`, which means the default precision for
        the backend, or a `Precision` enum value.

    Returns:
      Dot product result with preserved dimension order.
    """
    if lhs.ndim != rhs.ndim:
        raise ValueError(
            f"`lhs` and `rhs` must have the same dimensionality, got"
            f"`lhs.ndim == {lhs.ndim}` and `rhs.ndim == {rhs.ndim}`."
        )

    contracting_dims = utils.canonicalize_axis(contracting_dims, lhs)
    batch_dims = utils.canonicalize_axis(batch_dims, lhs)

    n_batch_dims = len(batch_dims)
    leading_batch_dims = range(n_batch_dims)

    dimension_numbers = ((contracting_dims, contracting_dims), (batch_dims, batch_dims))

    prod = jax.lax.dot_general(lhs, rhs, dimension_numbers, precision)
    prod = utils.zip_axes(prod, n_batch_dims)

    res_batch_dims = _get_res_batch_dims(contracting_dims, batch_dims)
    prod = jnp.moveaxis(prod, leading_batch_dims, res_batch_dims)
    return prod
