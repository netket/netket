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

from netket.utils import config


def inspect(name: str, tree: jax.Array):
    """
    Internal function to inspect the sharding of an array. To be used for debugging inside
    of :func:`jax.jit`-ted functions.

    Args:
        name: A string to identify the array, usually the name, but can contain anything else.
        x: The array
    """

    def _inspect(name: str, x: jax.Array):
        if config.netket_experimental_sharding:

            def _cb(y):
                if jax.process_index() == 0:
                    print(
                        f"{name}: shape={x.shape}, sharding:",
                        y,
                        flush=True,
                    )

            jax.debug.inspect_array_sharding(x, callback=_cb)

    if isinstance(tree, jax.Array):
        _inspect(name, tree)
    else:
        jax.tree.map_with_path(
            lambda path, x: _inspect(name + jax.tree_util.keystr(path), x), tree
        )
