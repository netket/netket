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


def identity(hilbert: _DiscreteHilbert, dtype: _DType = None) -> _LocalOperator:
    """
    Builds the :math:`\\mathbb{I}` identity operator.

    Args:
        hilbert: The hilbert space.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return _LocalOperator(hilbert, constant=1.0, dtype=dtype)


def sigmax(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^x` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = [np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in np.arange(1, N)]
    mat = np.diag(D, 1) + np.diag(D, -1)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmay(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^y` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np
    import netket.jax as nkjax
    from netket.utils import module_version

    if module_version(np) >= (2, 0, 0):
        from numpy.exceptions import ComplexWarning
    else:
        from numpy import ComplexWarning

    if not nkjax.is_complex_dtype(dtype):
        import jax.numpy as jnp
        import warnings

        old_dtype = dtype
        dtype = jnp.promote_types(complex, old_dtype)
        if old_dtype is not None:
            warnings.warn(
                ComplexWarning(
                    f"A complex dtype is required (dtype={old_dtype} specified). "
                    f"Promoting to dtype={dtype}."
                ),
                stacklevel=2,
            )

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = np.array([1j * np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in np.arange(1, N)])
    mat = np.diag(D, -1) + np.diag(-D, 1)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmaz(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^z` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = np.array([2 * m for m in np.arange(S, -(S + 1), -1)])
    mat = np.diag(D, 0)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmam(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^{-} = \\frac{1}{2}(\\sigma^x - i \\sigma^y)` operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = np.array([np.sqrt(S2 - m * (m - 1)) for m in np.arange(S, -S, -1)])
    mat = np.diag(D, -1)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmap(
    hilbert: _DiscreteHilbert, site: int, dtype: _DType = None
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^{+} = \\frac{1}{2}(\\sigma^x + i \\sigma^y)` operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    Args:
        hilbert: The hilbert space.
        site: The site on which this operator acts.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = np.array([np.sqrt(S2 - m * (m + 1)) for m in np.arange(S - 1, -(S + 1), -1)])
    mat = np.diag(D, 1)
    mat = _sparse.coo_matrix(mat)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)
