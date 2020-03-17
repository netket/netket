from .._C_netket.operator import LocalOperator as _LocalOperator

import numpy as _np


def destroy(hilbert, site):
    """
    Builds the boson destruction operator acting on the `site`-th of the
     Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size

    D = _np.array([_np.sqrt(m) for m in _np.arange(1, N)])
    mat = _np.diag(D, 1)
    return _LocalOperator(hilbert, mat, [site])


def create(hilbert, site):
    """
    Builds the boson creation operator acting on the `site`-th of the
     Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size

    D = _np.array([_np.sqrt(m) for m in _np.arange(1, N)])
    mat = _np.diag(D, -1)
    return _LocalOperator(hilbert, mat, [site])


def number(hilbert, site):
    """
    Builds the number operator acting on the `site`-th of the
    Hilbert space `hilbert`.

    If `hilbert` is a non-Bosonic space of local dimension M, it is considered
    as a bosonic space of local dimension M.

    :param hilbert: The hilbert space
    :param site: the site on which this operator acts
    :return: a nk.operator.LocalOperator
    """
    N = hilbert.local_size

    D = _np.array([m for m in _np.arange(0, N)])
    mat = _np.diag(D, 0)
    return _LocalOperator(hilbert, mat, [site])
