# Copyright 2020 Google LLC
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

"""Utilities for defining dataclasses that can be used with jax transformations.

This code was copied and adapted from https://github.com/google/flax/struct.py.

Accessed on 03/23/2020.
"""

import dataclasses
import functools
from typing import Any

import jax


def dataclass(clz):
  """Create a class which can be passed to functional transformations.

  Jax transformations such as `jax.jit` and `jax.grad` require objects that are
  immutable and can be mapped over using the `jax.tree_util` functions.
  The `dataclass` decorator makes it easy to define custom classes that can be
  passed safely to Jax.

  Example:
    >>> from jax import jit, numpy as jnp
    >>> from neural_tangents._src.utils import dataclasses
    >>> #
    >>> @dataclasses.dataclass
    >>> class Data:
    >>>   array: jnp.ndarray
    >>>   a_boolean: bool = dataclasses.field(pytree_node=False)
    >>> #
    >>> data = Data(jnp.array([1.0]), True)
    >>> #
    >>> data.array = jnp.array([2.0])  # Data is immutable. Will raise an error.
    >>> data = data.replace(array=jnp.array([2.0]))  # Use the replace method.
    >>> #
    >>> # This class can now be used safely in Jax.
    >>> jit(lambda data: data.array if data.a_boolean else 0)(data)

  Args:
    clz: the class that will be transformed by the decorator.

  Returns:
    The new class.

  """
  data_clz = dataclasses.dataclass(frozen=True)(clz)
  meta_fields = []
  data_fields = []
  for name, field_info in data_clz.__dataclass_fields__.items():
    is_pytree_node = field_info.metadata.get('pytree_node', True)
    init = field_info.metadata.get('init', True)
    if init:
      if is_pytree_node:
        data_fields.append(name)
      else:
        meta_fields.append(name)

  def iterate_clz(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  def clz_from_iterable(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return data_clz(**kwargs)

  jax.tree_util.register_pytree_node(data_clz,
                                     iterate_clz,
                                     clz_from_iterable)

  @functools.wraps(
      dataclasses.replace,
      assigned=('__module__', '__name__', '__qualname__', '__annotations__'))
  def replace(self: data_clz, **kwargs) -> data_clz:
    """Instance method alternative to `dataclasses.replace`."""
    return dataclasses.replace(self, **kwargs)

  @functools.wraps(
      dataclasses.asdict,
      assigned=('__module__', '__name__', '__qualname__', '__annotations__'))
  def asdict(self: data_clz) -> dict[str, Any]:
    """Instance method alternative to `dataclasses.asdict`."""
    return {
        f.name: getattr(self, f.name)
        for f in dataclasses.fields(self)
    }

  @functools.wraps(
      dataclasses.astuple,
      assigned=('__module__', '__name__', '__qualname__', '__annotations__'))
  def astuple(self: data_clz) -> tuple[Any, ...]:
    """Instance method alternative to `dataclasses.astuple`."""
    return tuple(
        getattr(self, f.name)
        for f in dataclasses.fields(self)
    )

  data_clz.replace = replace
  data_clz.asdict = asdict
  data_clz.astuple = astuple

  return data_clz


def field(pytree_node: bool = True, **kwargs):
  metadata = {'pytree_node': pytree_node}
  if 'init' in kwargs:
    metadata['init'] = kwargs['init']
  return dataclasses.field(metadata=metadata, **kwargs)
