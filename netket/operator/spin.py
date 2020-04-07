from .._C_netket.operator import LocalOperator as _LocalOperator

import numpy as _np


def sigmax(hilbert, site):
    """
    Builds the sigma_x operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size
    S = (N - 1) / 2

    D = [_np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in _np.arange(1, N)]
    mat = _np.diag(D, 1) + _np.diag(D, -1)
    return _LocalOperator(hilbert, mat, [site])


def sigmay(hilbert, site):
    """
    Builds the sigma_y operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size
    S = (N - 1) / 2

    D = _np.array(
        [1j * _np.sqrt((S + 1) * 2 * a - a * (a + 1)) for a in _np.arange(1, N)]
    )
    mat = _np.diag(D, -1) + _np.diag(-D, 1)
    return _LocalOperator(hilbert, mat, [site])


def sigmaz(hilbert, site):
    """
    Builds the sigma_z operator acting on the `site`-th of the Hilbert
    space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size
    S = (N - 1) / 2

    D = _np.array([2 * m for m in _np.arange(S, -(S + 1), -1)])
    mat = _np.diag(D, 0)
    return _LocalOperator(hilbert, mat, [site])


def sigmam(hilbert, site):
    """
    Builds the $sigma_- = sigma_x - im * sigma_y$ operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = _np.array([_np.sqrt(S2 - m * (m - 1)) for m in _np.arange(S, -S, -1)])
    mat = _np.diag(D, -1)
    return _LocalOperator(hilbert, mat, [site])


def sigmap(hilbert, site):
    """
    Builds the $sigma_- = sigma_x + im * sigma_y$ operator acting on the
    `site`-th of the Hilbert space `hilbert`.

    If `hilbert` is a non-Spin space of local dimension M, it is considered
    as a (M-1)/2 - spin space.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size
    S = (N - 1) / 2

    S2 = (S + 1) * S
    D = _np.array([_np.sqrt(S2 - m * (m + 1)) for m in _np.arange(S - 1, -(S + 1), -1)])
    mat = _np.diag(D, 1)
    return _LocalOperator(hilbert, mat, [site])
