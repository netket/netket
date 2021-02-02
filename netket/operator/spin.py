from netket.hilbert import AbstractHilbert


def sigmax(hilbert: AbstractHilbert, site: int):
    """
    Builds the sigma_x operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size
    S = (N - 1) / 2

    D = [np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in np.arange(1, N)]
    mat = np.diag(D, 1) + np.diag(D, -1)
    return LocalOperator(hilbert, mat, [site])


def sigmay(hilbert: AbstractHilbert, site: int):
    """
    Builds the sigma_y operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size
    S = (N - 1) / 2

    D = np.array([1j * np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in np.arange(1, N)])
    mat = np.diag(D, -1) + np.diag(-D, 1)
    return LocalOperator(hilbert, mat, [site])


def sigmaz(hilbert: AbstractHilbert, site: int):
    """
    Builds the sigma_z operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size
    S = (N - 1) / 2

    D = np.array([2 * m for m in np.arange(S, -(S + 1), -1)])
    mat = np.diag(D, 0)
    return LocalOperator(hilbert, mat, [site])


def sigmam(hilbert: AbstractHilbert, site: int):
    """
    Builds the $sigma_- = sigma_x - im * sigma_y$ operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = np.array([np.sqrt(S2 - m * (m - 1)) for m in np.arange(S, -S, -1)])
    mat = np.diag(D, -1)
    return LocalOperator(hilbert, mat, [site])


def sigmap(hilbert: AbstractHilbert, site: int):
    """
    Builds the $sigma_- = sigma_x + im * sigma_y$ operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    import numpy as np
    from ._local_operator import LocalOperator

    N = hilbert.local_size
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = np.array([np.sqrt(S2 - m * (m + 1)) for m in np.arange(S - 1, -(S + 1), -1)])
    mat = np.diag(D, 1)
    return LocalOperator(hilbert, mat, [site])


# clean up the module
del AbstractHilbert
