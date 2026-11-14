# File containing common commands for NetKet Test infrastructure

from typing import Any
from functools import partial

import pytest

import jax
import netket as nk
import numpy as np


_multiple_devices = jax.device_count() > 1

requires_multiple_devices = pytest.mark.skipif(
    not _multiple_devices, reason="Needs more than one device"
)
"""Use as a decorator to mark a test that only makes sense with several devices."""

requires_multiple_devices_single_process = pytest.mark.skipif(
    not (jax.local_device_count() > 1 and jax.process_count() == 1),
    reason="Needs several devices in a single process",
)
"""Use as a decorator to mark a test that requires multi-device sharding but is
incompatible with multi-process distributed execution (e.g. because it uses
process-local warnings or single-device reference meshes).
"""


def skipif_multiple_devices(reason: str):
    """Skip the decorated test when running on several devices, for code that is
    not supported there. A reason is mandatory."""
    return pytest.mark.skipif(_multiple_devices, reason=reason)


def xfail_multiple_devices(reason: str, strict: bool = True):
    """Mark the decorated test as known to fail on several devices. A reason is
    mandatory."""
    return pytest.mark.xfail(_multiple_devices, reason=reason, strict=strict)


class set_config:
    """
    Temporarily changes the value of the configuration `name`.

    Example:

    >>> with set_config("netket_experimental_disable_ode_jit", True):
    >>>     run_code

    """

    def __init__(self, name: str, value: Any):
        self._name = name.upper()
        self._value = value

    def __enter__(self):
        self._orig_value = nk.config.FLAGS[self._name]
        nk.config.update(self._name, self._value)

    def __exit__(self, exc_type, exc_value, traceback):
        nk.config.update(self._name, self._orig_value)


netket_experimental_fft_autocorrelation = partial(
    set_config, "NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION"
)


def hash_for_seed(obj):
    """
    Hash any object into an int that can be used in `np.random.seed`, and does not change between Python sessions.

    Args:
      obj: any object with `repr` defined to show its states.
    """

    bs = repr(obj).encode()
    out = 0
    for b in bs:
        out = (out * 256 + b) % 4294967291  # Output in [0, 2**32 - 1]
    return out


def named_parametrize(argname: str, values: list):
    param_values = [pytest.param(obj, id=f"{argname}={obj}") for obj in values]
    return pytest.mark.parametrize(argname, param_values)


def pairwise(strength: int = 2, *, cross=(), exclude=None):
    """
    Run only part of the product of the parametrizations of the decorated test,
    such that every pair of values of every two arguments still appears in at
    least one test.

    Use it on top of the usual stacked parametrizations:

    >>> @common.pairwise()
    >>> @pytest.mark.parametrize("error_norm", ["euclidean", "qgt", "maximum"])
    >>> @pytest.mark.parametrize("solver", solvers)
    >>> @pytest.mark.parametrize("propagation_type", ["real", "imag"])
    >>> def test_one_adaptive_step(solver, error_norm, propagation_type): ...

    The full product is collected, and the conftest deselects the tests that are
    not needed (a deterministic greedy selection, so the same tests run on every
    pytest-xdist worker and in every run). Use `--full-product` (or
    `NETKET_TEST_FULL_PRODUCT=1`) to run all of them.

    The arguments of a multi-argument parametrization (`"a, b"`) and the params
    of parametrized fixtures count as one argument each.

    Args:
        strength: every combination of values of any `strength` arguments
            appears. `1` runs every value of every argument at least once, which
            suits tests where an argument (e.g. the model) dominates the cost.
        cross: names of arguments whose values must also appear in all their
            combinations (in addition to the pairs with the other arguments).
        exclude: a function of the dict of the parameters of a test returning
            True for combinations that must never run (e.g. exact duplicates of
            other ones). They are deselected before the selection.

    A value is identified by its object, so two entries of a parametrization
    holding the same object (e.g. an operator listed under two names) count as
    the same value, and only one of them needs to run.
    """
    return pytest.mark.pairwise(strength=strength, cross=tuple(cross), exclude=exclude)


def _pairwise_axes(item):
    # The arguments of a parametrize mark share one index, so they are one axis.
    axes, seen = [], set()
    for mark in item.iter_markers("parametrize"):
        names = mark.args[0]
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        names = [n for n in names if n not in seen]
        if names:
            axes.append(tuple(names))
            seen.update(names)
    # params of parametrized fixtures
    axes += [(n,) for n in item.callspec.params if n not in seen]
    return axes


def select_pairwise(items, strength=2, cross=(), exclude=None):
    """
    Selects the items of `common.pairwise` (the tests of a single function):
    returns the list of items to keep, in their original order.
    """
    import itertools

    if exclude is not None:
        items = [it for it in items if not exclude(it.callspec.params)]
    if not items:
        return []

    axes = _pairwise_axes(items[0])
    t = min(strength, len(axes))
    # The groups of arguments whose combinations of values must all appear.
    groups = list(itertools.combinations(range(len(axes)), t))
    crossed = tuple(k for k, axis in enumerate(axes) if set(axis) & set(cross))
    if len(crossed) > t:
        groups.append(crossed)

    # A value is identified by its object: all the tests of a parametrization
    # share the objects of its list (`callspec.indices` is not per argument).
    # Only equality is used, so the selection is the same in every process.
    def tuples(item):
        row = [tuple(id(item.callspec.params[n]) for n in axis) for axis in axes]
        return {(g, tuple(row[k] for k in g)) for g in groups}

    candidates = [(i, tuples(it)) for i, it in enumerate(items)]
    uncovered = set().union(*(c for _, c in candidates))
    chosen = []
    while uncovered:
        # the first test covering the most new combinations
        best = max(candidates, key=lambda c: (len(c[1] & uncovered), -c[0]))
        chosen.append(best[0])
        uncovered -= best[1]
    return [items[i] for i in sorted(chosen)]


def assert_allclose(x, y, **kwargs):
    from jax.experimental import multihost_utils

    if isinstance(x, jax.Array):
        if not x.is_fully_addressable:
            x = multihost_utils.process_allgather(x)
    if isinstance(y, jax.Array):
        if not y.is_fully_addressable:
            y = multihost_utils.process_allgather(y)
    np.testing.assert_allclose(x, y, **kwargs)
