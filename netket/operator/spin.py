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

from netket.hilbert import AbstractHilbert
from netket.utils.types import DType

from ._local_operator import LocalOperator as _LocalOperator


def sigmax(hilbert: AbstractHilbert, site: int, dtype: DType = float) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^x` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = [np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in np.arange(1, N)]
    mat = np.diag(D, 1) + np.diag(D, -1)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmay(
    hilbert: AbstractHilbert, site: int, dtype: DType = complex
) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^y` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = np.array([1j * np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in np.arange(1, N)])
    mat = np.diag(D, -1) + np.diag(-D, 1)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmaz(hilbert: AbstractHilbert, site: int, dtype: DType = float) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^z` operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    D = np.array([2 * m for m in np.arange(S, -(S + 1), -1)])
    mat = np.diag(D, 0)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmam(hilbert: AbstractHilbert, site: int, dtype: DType = float) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^{-} = \\sigma^x - i \\sigma^y` operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = np.array([np.sqrt(S2 - m * (m - 1)) for m in np.arange(S, -S, -1)])
    mat = np.diag(D, -1)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


def sigmap(hilbert: AbstractHilbert, site: int, dtype: DType = float) -> _LocalOperator:
    """
    Builds the :math:`\\sigma^{+} = \\sigma^x + i \\sigma^y` operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np

    N = hilbert.size_at_index(site)
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = np.array([np.sqrt(S2 - m * (m + 1)) for m in np.arange(S - 1, -(S + 1), -1)])
    mat = np.diag(D, 1)
    return _LocalOperator(hilbert, mat, [site], dtype=dtype)


# clean up the module
del AbstractHilbert, DType
