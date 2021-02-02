from netket.hilbert import AbstractHilbert


def destroy(hilbert: AbstractHilbert, site: int):
    """
    Builds the boson destruction operator acting on the `site`-th of the
     Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size

    D = np.array([np.sqrt(m) for m in np.arange(1, N)])
    mat = np.diag(D, 1)
    return LocalOperator(hilbert, mat, [site])


def create(hilbert: AbstractHilbert, site: int):
    """
    Builds the boson creation operator acting on the `site`-th of the
     Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size

    D = np.array([np.sqrt(m) for m in np.arange(1, N)])
    mat = np.diag(D, -1)
    return LocalOperator(hilbert, mat, [site])


def number(hilbert: AbstractHilbert, site: int):
    """
    Builds the number operator acting on the `site`-th of the
    Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size

    D = np.array([m for m in np.arange(0, N)])
    mat = np.diag(D, 0)
    return LocalOperator(hilbert, mat, [site])


# clean up the module
del AbstractHilbert
