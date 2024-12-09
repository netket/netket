# Copyright 2023-2024 The NetKet Authors - All rights reserved.
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

import abc


import jax

from netket.utils import struct


class DiscreteHilbertConstraint(struct.Pytree):
    r"""
    Protocol to define an Abstract Constraint for a discete Hilbert space.

    To define a customized constraint, you must subclass this class and at least implement the
    :code:`__call__` method. The :code:`__call__` method should take as input a matrix encoding a batch of
    configurations, and return a vector of booleans specifying whether they are valid configurations
    or not.

    The :code:`__call__` method must be :code:`jax.jit`-able. If you cannot make it jax-jittable, you can implement
    it in numba/python and wrap it into a :func:`jax.pure_callback` to make it compatible with jax.

    The callback should be hashable and comparable with itself, which means it must implement :code:`__hash__` and :code:`__eq__`.
    By default, the :code:`__hash__` method is implemented by the `id` of the object, which is unique for each object,
    which will work but might lead to more recompilations in jax. If you can, you should implement a custom :code:`__hash__`

    Example:

        The following example shows a class that implements a simple constraint checking that the total sum of the
        elements in the configuration is equal to a given value. The example shows how to implement the :code:`__call__` method
        and the :code:`__hash__` and :code:`__eq__` methods.

        .. code-block:: python

            import netket as nk
            from netket.utils import struct

            import jax; import jax.numpy as jnp

            class SumConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
                # A simple constraint checking that the total sum of the elements
                # in the configuration is equal to a given value.

                # The value must be set as a pytree_node=False field, meaning
                # that it is a constant and changes to this value represent different
                # constraints.
                total_sum : float = struct.field(pytree_node=False)

                def __init__(self, total_sum):
                    self.total_sum = total_sum

                def __call__(self, x):
                    # Makes it jax-compatible
                    return jnp.sum(x, axis=-1) == self.total_sum

                def __hash__(self):
                    return hash(("SumConstraint", self.total_sum))

                def __eq__(self, other):
                    if isinstance(other, SumConstraint):
                        return self.total_sum == other.total_sum
                    return False

    Example:

        The following example shows how to implement the same function as above, but using a pure python function and
        a :func:`jax.pure_callback` to make it compatible with jax.

        .. code-block:: python

            import jax
            import jax.numpy as jnp
            import numpy as np

            import netket as nk
            from netket.utils import struct

            class SumConstraintPy(nk.hilbert.constraint.DiscreteHilbertConstraint):
                # A simple constraint checking that the total sum of the elements
                # in the configuration is equal to a given value.

                total_sum : float = struct.field(pytree_node=False)

                def __init__(self, total_sum):
                    self.total_sum = total_sum

                def __call__(self, x):
                    return jax.pure_callback(self._call_py,
                                            (jax.ShapeDtypeStruct(x.shape[:-1], bool)),
                                            x,
                                            vmap_method="expand_dims")

                def _call_py(self, x):
                    # Not Jax compatible
                    return np.sum(x, axis=-1) == self.total_sum

                def __hash__(self):
                    return hash(("SumConstraintPy", self.total_sum))

                def __eq__(self, other):
                    if isinstance(other, SumConstraintPy):
                        return self.total_sum == other.total_sum
                    return False

    """

    @abc.abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        This function should take as input a matrix encoding a batch of configurations,
        and return a vector of booleans specifying whether they are valid configurations of
        the Hilbert space or not.

        Args:
            x: 2D matrix.
        """
