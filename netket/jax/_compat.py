# Copyright 2025 The NetKet Authors - All rights reserved.
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

import jax


def aval_varying_axes(x) -> tuple:
    """
    Return the manual mesh axes over which a traced value is varying.

    JAX <= 0.9 exposed this as ``jax.typeof(x).vma``. JAX 0.10 replaced that
    attribute with ``manual_axis_type.varying``.
    """
    aval = jax.typeof(x)

    if hasattr(aval, "vma"):
        return tuple(aval.vma)

    manual_axis_type = getattr(aval, "manual_axis_type", None)
    if manual_axis_type is None:
        return ()

    varying = getattr(manual_axis_type, "varying", ())
    if not varying:
        return ()

    aval_sharding = getattr(aval, "sharding", None)
    mesh = getattr(aval_sharding, "mesh", None)
    axis_names = getattr(mesh, "axis_names", ())
    ordered_axes = tuple(axis for axis in axis_names if axis in varying)

    if len(ordered_axes) == len(varying):
        return ordered_axes

    return ordered_axes + tuple(axis for axis in varying if axis not in axis_names)


def mesh_has_axes(mesh) -> bool:
    """
    Return True when a mesh actually exposes named resource axes.

    JAX 0.10 changed rank-0 meshes so ``mesh.empty`` is no longer a reliable
    emptiness check; ``axis_names`` remains stable across versions.
    """
    axis_names = getattr(mesh, "axis_names", None)
    if axis_names is not None:
        return len(axis_names) > 0

    shape = getattr(mesh, "shape", None)
    if shape is not None:
        return len(shape) > 0

    empty = getattr(mesh, "empty", None)
    if empty is not None:
        return not empty

    return False
