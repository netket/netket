# Copyright 2021 The NetKet Authors - All rights reserved.
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


def canonicalize_dtype(dtype):
    """Canonicalise a dtype, skip if None."""
    if dtype is not None:
        return jax.dtypes.canonicalize_dtype(dtype)
    return dtype


def cast_tree(tree, dtype):
    """Cast tree to given dtype, skip if None."""
    if dtype is not None:
        return jax.tree_map(lambda t: t.astype(dtype), tree)
    else:
        return tree
