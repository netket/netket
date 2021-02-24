from numpy.typing import DTypeLike

from netket.hilbert import AbstractHilbert


def destroy(
    hilbert: AbstractHilbert, site: int, dtype: DTypeLike = float
) -> "LocalOperator":
    """
    Builds the boson destruction operator acting on the `site`-th of the
     Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts

    Returns:
        The resulting Local Operator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.size_at_index(site)

    D = np.array([np.sqrt(m) for m in np.arange(1, N)])
    mat = np.diag(D, 1)
    return LocalOperator(hilbert, mat, [site], dtype=dtype)


def create(
    hilbert: AbstractHilbert, site: int, dtype: DTypeLike = float
) -> "LocalOperator":
    """
    Builds the boson creation operator acting on the `site`-th of the
     Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts

    Returns:
        The resulting Local Operator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.size_at_index(site)

    D = np.array([np.sqrt(m) for m in np.arange(1, N)])
    mat = np.diag(D, -1)
    return LocalOperator(hilbert, mat, [site], dtype=dtype)


def number(
    hilbert: AbstractHilbert, site: int, dtype: DTypeLike = float
) -> "LocalOperator":
    """
    Builds the number operator acting on the `site`-th of the
    Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts

    Returns:
        The resulting Local Operator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.size_at_index(site)

    D = np.array([m for m in np.arange(0, N)])
    mat = np.diag(D, 0)
    return LocalOperator(hilbert, mat, [site], dtype=dtype)


def proj(
    hilbert: AbstractHilbert, site: int, n: int, dtype: DTypeLike = float
) -> "LocalOperator":
    """
    Builds the projector operator acting on the `site`-th of the
    Hilbert space `hilbert` and collapsing on the state with `n` bosons.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts
        n: the state on which to project

    Returns:
        the resulting operator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.size_at_index(site)

    if n >= N:
        raise ValueError("Cannot project on a state above the cutoff.")

    D = np.array([0 for m in np.arange(0, N)])
    D[n] = 1
    mat = np.diag(D, 0)
    return LocalOperator(hilbert, mat, [site], dtype=dtype)


# clean up the module
del AbstractHilbert
