from functools import wraps
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh, AxisType, auto_axes
from jax.sharding import PartitionSpec as P, NamedSharding, SingleDeviceSharding
from jax.experimental.shard_map import shard_map

from netket.utils.jax import HashablePartial


def flatten_spec(spec):
    out = []
    for s in spec:
        if isinstance(s, tuple):
            out.extend(s)
        else:
            out.append(s)
    return out


def get_sharding_spec(
    tree, *, axes: None | int | tuple[int, ...] | slice = None
) -> P | None:
    """
    Return the sharding specification of the input `x`.

    If no mesh is associated with the input, or if the input is sharded on a single device,
    return `None`.

    Args:
        x: The input whose sharding specification is to be retrieved.

    Returns:
        A `PartitionSpec` object representing the sharding specification of `x`, or `None.
    """

    def _get_sharding_spec(x):
        x_t = jax.typeof(x)
        x_s = x_t.sharding
        if isinstance(x_s, SingleDeviceSharding):
            return None
        elif isinstance(x_s, NamedSharding):
            if x_s.mesh.empty:
                return None
            spec = x_s.spec
            if axes is None:
                return spec
            elif isinstance(axes, int):
                return P(spec[axes])
            elif isinstance(axes, tuple):
                return P(*(spec[axis] for axis in axes))
            elif isinstance(axes, slice):
                return P(*spec[axes])
            else:
                raise TypeError(
                    f"Unsupported axes type: {type(axes)}. Expected int, tuple, or slice."
                )
        else:
            raise TypeError(
                f"Unsupported sharding type: {type(x_s)}. Expected SingleDeviceSharding or NamedSharding."
            )

    return jax.tree.map(_get_sharding_spec, tree)


def check_compatible_sharding(*args, axes=(0,)):
    """
    Check if all arguments have the same sharding specification.

    Args:
        *args: The arguments to check.
        axes: Optional, a list of axes to check for compatibility.

    Raises:
        ValueError: If the sharding specifications are not compatible.
    """
    specs = get_sharding_spec(args)

    def _check_compatible_sharding(*xs):
        xs_spec = get_sharding_spec(xs, axes=axes)
        if all([spec is None for spec in xs_spec]):
            return
        if any([spec is None for spec in xs_spec]):
            raise ValueError(
                "Incompatible sharding specifications found. One is replicated while others are sharded: "
            )
        if len(set(xs_spec)) > 1:
            raise ValueError(
                "Incompatible sharding specifications found: "
                f"{xs_spec}. All inputs must have the same sharding specification."
            )

    jax.tree.map(_check_compatible_sharding, *args)


def is_sharded(x, *, axes: int | None | tuple[int, ...] = None) -> bool:
    xf, _ = jax.tree.flatten(x)
    axis_spec = get_sharding_spec(xf, axes=axes)
    if all(spec is None for spec in axis_spec):
        return False
    else:
        return True


def pad_axis_for_sharding(
    array: jax.Array, *, axis, axis_name, padding_value: float | jax.Array = 0
) -> jax.Array:
    """
    Pads an array along an axis to make it divisible by the number of processes.

    Args:
        array: The array to pad.
        axis: The axis along which to pad.
        padding_value: The value to use for padding.

    Returns:
        The padded array.
    """
    if not is_sharded(array):
        return array
    if isinstance(axis_name, P):
        assert len(axis_name) == 1
        axis_name = axis_name[0]
    if axis_name is None:
        return array
    if not isinstance(axis_name, (int, tuple)):
        axis_name = (axis_name,)

    array_sharding = jax.typeof(array).sharding
    if not isinstance(array_sharding, NamedSharding):
        raise NotImplementedError(
            "Padding is only supported for NamedSharding. "
            f"Got {type(array_sharding)}."
        )
    mesh = array_sharding.mesh
    if any(an not in mesh.axis_names for an in axis_name):
        raise ValueError(
            f"Axis name {axis_name} is not present in the mesh {mesh}. "
            "Please provide a valid axis name."
        )
    array_axis_size = array.shape[axis]
    mesh_axis_size = np.prod(
        [mesh.shape[ax] for ax in axis_name if (ax in mesh.axis_names)]
    )

    if array_axis_size % mesh_axis_size != 0:
        padded_axis_size = int(
            mesh_axis_size * np.ceil(array_axis_size / mesh_axis_size)
        )
        padding_shape = [(0, 0) for _ in range(array.ndim)]
        padding_shape[axis] = (0, padded_axis_size - array_axis_size)

        array = jnp.pad(
            array,
            padding_shape,
            constant_values=padding_value,
        )
    return array


def canonicalize_sharding(
    sharding: NamedSharding | PartitionSpec | None,
    api_name: str,
    check_mesh_consistency: bool = True,
) -> NamedSharding | None:
    if sharding is None:
        return None
    if isinstance(sharding, NamedSharding) and sharding.mesh.empty:
        return None

    cur_mesh = get_abstract_mesh()
    if isinstance(sharding, PartitionSpec):
        if cur_mesh.empty:
            raise ValueError(
                "Using PartitionSpec when you are not under a mesh context is not"
                " allowed. Please pass a NamedSharding instance or enter into a mesh"
                f" context via `jax.sharding.use_mesh`. Got {sharding}"
            )
        sharding = NamedSharding(cur_mesh, sharding)
    else:
        if (
            check_mesh_consistency
            and not cur_mesh.empty
            and sharding.mesh.abstract_mesh != cur_mesh
        ):
            raise ValueError(
                f"Context mesh {cur_mesh} should match the mesh of sharding"
                f" {sharding.mesh.abstract_mesh} passed to {api_name}."
            )
        if isinstance(sharding.mesh, jax.sharding.Mesh):
            sharding = NamedSharding(sharding.mesh.abstract_mesh, sharding.spec)

    for s in flatten_spec(sharding.spec):
        if s is None:
            continue
        if sharding.mesh._name_to_type[s] in {AxisType.Auto, AxisType.Manual}:
            raise ValueError(
                f"PartitionSpec passed to {api_name} cannot contain axis"
                " names that are of type Auto or Manual. Got PartitionSpec:"
                f" {sharding.spec} with axis name: {s} of type:"
                f" {sharding.mesh._name_to_type[s]}."
            )
    return sharding


@wraps(auto_axes)
def auto_axes_maybe(
    fun, *, axes: str | tuple[str, ...] | None = None, out_sharding=None
):
    """
    Equivalent to `jax.sharding.auto_axes` but will not raise an error if the mesh is empty.
    Instead, it will call the function directly without sharding.

    Args:
        fun: The function to be decorated.
        ...
    """

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if out_sharding is None:
            if "out_sharding" in kwargs:
                _out_sharding = kwargs.pop("out_sharding")
            else:
                _out_sharding = out_sharding
        else:
            _out_sharding = out_sharding

        cur_mesh = jax.sharding.get_abstract_mesh()
        sharding_mesh = jax.make_mesh((), ())
        for i in jax.tree.leaves(_out_sharding):
            if isinstance(i, NamedSharding):
                sharding_mesh = i.mesh.abstract_mesh

        if sharding_mesh.empty and cur_mesh.empty:
            return fun(*args, **kwargs)
        else:
            return auto_axes(fun, out_sharding=_out_sharding, axes=axes)(
                *args, **kwargs
            )

    return wrapper


def auto_sharded_function_wrapper(fun, out_specs_fun, *args):
    args_flat, args_tree = jax.tree.flatten(args)
    to_flat = [jax.typeof(x) for x in args_flat]

    if all(isinstance(arg.sharding, SingleDeviceSharding) for arg in to_flat):
        return fun(*args)
    # assume NamedSharding for now
    mesh = to_flat[0].sharding.mesh

    if mesh.empty:
        return fun(*args)

    def compute_in_spec(arg):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            return P()

    # In this case, we use the shard map
    in_specs = jax.tree.map(lambda x: jax.typeof(x).sharding.spec, args)
    out_specs = out_specs_fun(*args)

    return shard_map(
        fun,
        in_specs=in_specs,
        out_specs=out_specs,
        mesh=mesh,
    )(*args)


def auto_shard_map(fun, out_specs_fun):
    """
    A decorator that automatically shards a function based on the sharding of its inputs.

    Args:
        fun: The function to be decorated.
        out_specs_fun: A function that returns the output sharding specifications based on the inputs.
            Guaranteed to be called only when the inputs are sharded.

    Example:
        @auto_shard_map
        def my_function(x, y):
            return x + y
        def output_specs(x, y):
            return jax.typeof(x).sharding.spec
        auto_shard_map(my_function, output_specs)(x, y)
    """
    return HashablePartial(
        auto_sharded_function_wrapper,
        fun,
        out_specs_fun,
    )
