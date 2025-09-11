from functools import partial, wraps
import inspect

from copy import copy

import math
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, get_abstract_mesh
import pytest

from netket.vqs.mc.mc_state.state import MCState


def create_mesh(mesh_shape, axis_names, iota_order=False, axis_types=None):
    size = math.prod(mesh_shape)
    if len(jax.devices()) < size:
        pytest.skip(f"Test requires {size} devices.")
    # if axis_types is None:
    #     axis_types = (AxisType.Explicit,) * len(axis_names)
    if axis_types == "auto":
        axis_types = (AxisType.Auto,) * len(axis_names)
    elif axis_types == "explicit":
        axis_types = (AxisType.Explicit,) * len(axis_names)

    if iota_order:
        devices = sorted(jax.devices(), key=lambda d: d.id)
        mesh_devices = np.array(devices[:size]).reshape(mesh_shape)
        return jax.sharding.Mesh(mesh_devices, axis_names, axis_types=axis_types)
    else:
        return jax.make_mesh(mesh_shape, axis_names, axis_types=axis_types)


def with_mesh(sizes, names, axis_types=None, iota_order=False):
    if axis_types == "auto":
        axis_types = (AxisType.Auto,) * len(names)
    elif axis_types == "explicit":
        axis_types = (AxisType.Explicit,) * len(names)

    def decorator(fn):
        @wraps(fn)
        def mesh_fn(*args, **kwargs):
            mesh = create_mesh(sizes, names, iota_order, axis_types=axis_types)
            with jax.sharding.set_mesh(mesh):
                return fn(*args, **kwargs)
            return mesh_fn

        return mesh_fn

    return decorator


with_explicit_mesh = partial(with_mesh, axis_types="explicit")
with_auto_mesh = partial(with_mesh, axis_types="explicit")


def mesh_name(sizes=(), names=(), mode="Auto"):
    assert len(sizes) == len(names), "Sizes and names must have the same length."
    if len(sizes) == 0 and len(names) == 0:
        return "Mesh[None]"
    _str = ",".join(f"{size}@{name}" for size, name in zip(sizes, names))
    return f"Mesh[{mode}:{_str}]"


def with_meshes(
    sizes_and_names: list[tuple[tuple[int, ...], tuple[str, ...]] | None] = [],
    *,
    auto=None,
    explicit=None,
):
    """
    Decorator to generate multiple test variants under different meshes.
    If an entry is None, it's treated as empty sizes/names: (), ().

    If the decorated function accepts a 'mesh' parameter, use pytest.parametrize
    to inject meshes; otherwise, dynamically generate separate test functions.
    """
    if auto is None and explicit is None:
        auto = sizes_and_names

    meshes_specs = {
        "auto": auto if auto is not None else [],
        "explicit": explicit if explicit is not None else [],
    }

    def decorator(fn):

        sig = inspect.signature(fn)
        accepts_mesh = "mesh" in sig.parameters

        if not accepts_mesh:
            raise ValueError(
                f"Function {fn.__name__} does not accept a 'mesh' parameter. "
                "Please modify the function to accept 'mesh' or use a different decorator."
            )

        param_list = []
        for typ, sizes_and_names in meshes_specs.items():
            for param in sizes_and_names:
                if param is None:
                    sizes, names = (), ()
                else:
                    sizes, names = param
                param_list.append(
                    pytest.param((sizes, names, typ), id=mesh_name(sizes, names, typ))
                )

        @pytest.mark.parametrize("mesh", param_list)  # , ids=ids)
        @wraps(fn)
        def wrapper(*args, mesh, **kwargs):
            sizes, names, axis_type = mesh
            mesh = create_mesh(sizes, names, axis_types=axis_type)
            # enter mesh context and call original with mesh
            with jax.sharding.set_mesh(mesh):
                return fn(*args, mesh=mesh, **kwargs)

        return wrapper

    return decorator


def with_explicit_meshes(sizes_and_names):
    return with_meshes(explicit=sizes_and_names)


def with_auto_meshes(sizes_and_names):
    return with_meshes(auto=sizes_and_names)


def fix_mesh(tree):
    # out_sharding = jax.sharding.NamedSharding(jax.sharding.get_abstract_mesh(), jax.P())
    target_mesh = get_abstract_mesh()
    concrete_mesh = jax.make_mesh(
        target_mesh.axis_sizes, target_mesh.axis_names, target_mesh.axis_types
    )
    target_sharding = jax.NamedSharding(concrete_mesh, jax.P())

    def _fix(x):
        if isinstance(x, jax.Array):
            # workaround to jax bug. Maybe can be a jax.device_put in the future.
            if target_mesh.empty:
                is_new_style_key = jnp.issubdtype(x.dtype, jax.dtypes.prng_key)
                if is_new_style_key:
                    _impl = jax.random.key_impl(x)
                    x = jax.random.key_data(x)
                x = jnp.array(np.asarray(x))
                if is_new_style_key:
                    x = jax.random.wrap_key_data(x, impl=_impl)
            else:
                x = jax.device_put(x, target_sharding)

        return x

    if isinstance(tree, MCState):
        tree = copy(tree)
        tree._sampler_seed = fix_mesh(tree._sampler_seed)
        tree.variables = fix_mesh(tree.variables)
        tree.sampler = fix_mesh(tree.sampler)
        return tree
    else:
        return jax.tree.map(_fix, tree)
