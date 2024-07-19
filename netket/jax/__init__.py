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

from netket.utils import HashablePartial

from ._utils_dtype import (
    is_complex_dtype,
    dtype_complex,
    dtype_real,
    maybe_promote_to_complex,
    canonicalize_dtypes,
)

from ._utils_tree import (
    tree_ravel,
    tree_size,
    eval_shape,
    tree_leaf_isreal,
    tree_leaf_iscomplex,
    tree_ishomogeneous,
    tree_conj,
    tree_dot,
    tree_norm,
    tree_cast,
    tree_ax,
    tree_axpy,
    tree_to_real,
    compose,
)

from ._utils_random import (
    mpi_split,
    PRNGKey,
    PRNGSeq,
    batch_choice,
)


from ._vjp import vjp
from ._grad import grad, value_and_grad

from ._chunk_utils import chunk, unchunk
from ._scanmap import scan_reduce, scan_append, scan_append_reduce, scanmap
from ._vjp_chunked import vjp_chunked
from ._vmap_chunked import apply_chunked, vmap_chunked

from ._math import logsumexp_cplx, logdet_cmplx

from ._jacobian import jacobian, jacobian_default_mode

from ._sort import sort, searchsorted

from ._expect import expect

# internal sharding utilities
from . import sharding

from netket.utils import _hide_submodules

_hide_submodules(__name__, ignore="sharding")
