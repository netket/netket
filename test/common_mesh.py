from functools import wraps
from typing import Optional
import inspect

import math
import numpy as np

import jax
from jax.sharding import AxisType
import pytest


def create_mesh(mesh_shape, axis_names, iota_order=False, axis_types=None):
    size = math.prod(mesh_shape)
    if len(jax.devices()) < size:
        pytest.skip(f"Test requires {size} devices.")
    # if axis_types is None:
    #     axis_types = (AxisType.Explicit,) * len(axis_names)

    if iota_order:
        devices = sorted(jax.devices(), key=lambda d: d.id)
        mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
        return jax.sharding.Mesh(mesh_devices, axis_names, axis_types=axis_types)
    else:
        return jax.make_mesh(mesh_shape, axis_names, axis_types=axis_types)


def with_explicit_mesh(sizes, names, axis_types=None, iota_order=False):
    axis_types = (AxisType.Explicit,) * len(names) if axis_types is None else axis_types

    def decorator(fn):
        @wraps(fn)
        def mesh_fn(*args, **kwargs):
            mesh = create_mesh(sizes, names, iota_order, axis_types=axis_types)
            with jax.sharding.use_mesh(mesh):
                return fn(*args, **kwargs)
            return mesh_fn

        return mesh_fn

    return decorator


def mesh_name(sizes=(), names=()):
    assert len(sizes) == len(names), "Sizes and names must have the same length."
    if len(sizes) == 0 and len(names) == 0:
        return "Mesh(None)"
    _str = ",".join(f"{size}@{name}" for size, name in zip(sizes, names))
    return f"Mesh({_str})"


def with_explicit_meshes(
    sizes_and_names: list[Optional[tuple[tuple[int, ...], tuple[str, ...]]]],
):
    """
    Decorator to generate multiple test variants under different meshes.
    If an entry is None, it's treated as empty sizes/names: (), ().

    If the decorated function accepts a 'mesh' parameter, use pytest.parametrize
    to inject meshes; otherwise, dynamically generate separate test functions.
    """

    def decorator(fn):
        # check if fn accepts 'mesh'
        sig = inspect.signature(fn)
        accepts_mesh = "mesh" in sig.parameters

        if accepts_mesh:
            # build parameter list: each element is (sizes, names)
            param_list = []
            ids = []
            for param in sizes_and_names:
                if param is None:
                    sizes, names = (), ()
                    ids.append("mesh")
                else:
                    sizes, names = param
                    ids.append("_".join(names))
                param_list.append(
                    pytest.param((sizes, names), id=mesh_name(sizes, names))
                )

            @pytest.mark.parametrize("mesh", param_list, ids=ids)
            @wraps(fn)
            def wrapper(*args, mesh, **kwargs):
                sizes, names = mesh
                axis_types = (AxisType.Explicit,) * len(names)
                mesh = create_mesh(sizes, names, axis_types=axis_types)
                # enter mesh context and call original with mesh
                with jax.sharding.use_mesh(mesh):
                    return fn(*args, mesh=mesh, **kwargs)

            return wrapper

        # else: fall back to dynamic test generation
        globs = fn.__globals__
        for param in sizes_and_names:
            if param is None:
                sizes, names = (), ()
            else:
                sizes, names = param
            test_name = f"{fn.__name__}_{mesh_name(sizes, names)}"

            @wraps(fn)
            def test_wrapper(*args, _sizes=sizes, _names=names, **kwargs):
                axis_types = (AxisType.Explicit,) * len(_names)
                mesh = create_mesh(_sizes, _names, axis_types=axis_types)
                with jax.sharding.use_mesh(mesh):
                    return fn(*args, **kwargs)

            test_wrapper.__name__ = test_name
            globs[test_name] = test_wrapper

        # hide original so pytest won't collect it
        def _hidden_original(*args, **kwargs):
            pytest.skip(
                f"Original test '{fn.__name__}' hidden by with_explicit_meshes decorator"
            )

        _hidden_original.__name__ = f"_{fn.__name__}"

        return _hidden_original

    return decorator
