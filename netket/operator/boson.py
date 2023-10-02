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

from scipy import sparse as _sparse

from netket.utils.types import DType as _DType

from netket.hilbert import DiscreteHilbert as _DiscreteHilbert

from ._local_operator import LocalOperator as _LocalOperator

# export identity from here as well
from .spin import identity  # noqa: F401


def destroy(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the boson destruction operator :math:`\\hat{a}` acting on the `site`-th of
    the Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)

    D = np.array([np.sqrt(m) for m in np.arange(1, N)])
    mat = np.diag(D, 1)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def create(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = float
) -> _LocalOperator:
    """
    Builds the boson creation operator :math:`\\hat{a}^\\dagger` acting on the `site`-th
    of the Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)

    D = np.array([np.sqrt(m) for m in np.arange(1, N)])
    mat = np.diag(D, -1)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def number(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = float
) -> _LocalOperator:
    """
    Builds the number operator :math:`\\hat{a}^\\dagger\\hat{a}`  acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)

    D = np.array([m for m in np.arange(0, N)])
    mat = np.diag(D, 0)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def proj(
    hilbert: _DiscreteHilbert, site: int, n: int, dtype: _DType = float
) -> _LocalOperator:
    """
    Builds the projector operator :math:`|n\\rangle\\langle n |` acting on the
    `site`-th of the Hilbert space `hilbert` and collapsing on the state with `n`
    bosons.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)

    if n >= N:
        raise ValueError("Cannot project on a state above the cutoff.")

    D = np.array([0 for m in np.arange(0, N)])
    D[n] = 1
    mat = np.diag(D, 0)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)
