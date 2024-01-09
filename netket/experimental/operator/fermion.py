# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Optional as _Optional

from netket.utils.types import DType as _DType
from netket.hilbert.abstract_hilbert import AbstractHilbert as _AbstractHilbert
from netket.experimental.operator import FermionOperator2nd as _FermionOperator2nd


def destroy(
    hilbert: _AbstractHilbert,
    site: int,
    sz: _Optional[int] = None,
    cutoff: float = 1e-10,
    dtype: _DType = None,
):
    """
    Builds the fermion destruction operator :math:`\\hat{a}` acting
    on the `site`-th of the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space.
        site: the site on which this operator acts.
        sz: spin projection quantum number. This is the eigenvalue of
            the corresponding spin-Z Pauli operator (e.g. `sz = ±1` for
            a spin-1/2, `sz ∈ [-2, -1, 1, 2]` for a spin-3/2 and
            in general `sz ∈ [-2S, -2S + 2, ... 2S-2, 2S]` for
            a spin-S )
        dtype: The datatype to use for the matrix elements.

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(hilbert, (((idx, 0),),), dtype=dtype)


def create(
    hilbert: _AbstractHilbert,
    site: int,
    sz: _Optional[int] = None,
    cutoff: float = 1e-10,
    dtype: _DType = None,
):
    """
    Builds the fermion creation operator :math:`\\hat{a}^\\dagger` acting
    on the `site`-th of the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts
        sz: spin projection quantum number. This is the eigenvalue of
            the corresponding spin-Z Pauli operator (e.g. `sz = ±1` for
            a spin-1/2, `sz ∈ [-2, -1, 1, 2]` for a spin-3/2 and
            in general `sz ∈ [-2S, -2S + 2, ... 2S-2, 2S]` for
            a spin-S )
        dtype: The datatype to use for the matrix elements.

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(hilbert, (((idx, 1),),), dtype=dtype)


def number(
    hilbert: _AbstractHilbert,
    site: int,
    sz: _Optional[int] = None,
    cutoff: float = 1e-10,
    dtype: _DType = None,
):
    """
    Builds the number operator :math:`\\hat{a}^\\dagger\\hat{a}`  acting on the
    `site`-th of the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts
        site: the site on which this operator acts
        sz: spin projection quantum number. This is the eigenvalue of
            the corresponding spin-Z Pauli operator (e.g. `sz = ±1` for
            a spin-1/2, `sz ∈ [-2, -1, 1, 2]` for a spin-3/2 and
            in general `sz ∈ [-2S, -2S + 2, ... 2S-2, 2S]` for
            a spin-S )
        dtype: The datatype to use for the matrix elements.

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(
        hilbert,
        (
            (
                (idx, 1),
                (idx, 0),
            ),
        ),
        dtype=dtype,
    )


def _get_index(hilbert: _AbstractHilbert, site: int, sz: _Optional[int] = None):
    """go from (site, spin_projection) indices to index in the (tensor) hilbert space"""
    if sz is None:
        if hasattr(hilbert, "spin") and hilbert.spin is not None:
            raise ValueError(
                "hilbert spaces with spin property require to specify the sz value to get the position in hilbert space"
            )
        return site
    elif not hasattr(hilbert, "spin"):
        raise ValueError("cannot specify sz for hilbert without spin property")
    elif hasattr(hilbert, "_get_index"):  # keep it general
        idx = hilbert._get_index(site, sz)
        if idx >= hilbert.size:
            raise IndexError(
                "requested site and sz combination is not present in the hilbert space"
            )
        return idx
    else:
        raise NotImplementedError(
            f"no method _get_index available for hilbert space {hilbert} that allows to find the position in hilbert space based on a spin projection value sz"
        )


def identity(hilbert: _AbstractHilbert, cutoff: float = 1e-10, dtype: _DType = None):
    """
    Builds the :math:`\\mathbb{I}` identity operator.

    Args:
        hilbert: The hilbert space.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`.
    """
    return _FermionOperator2nd(hilbert, constant=1, dtype=dtype, cutoff=cutoff)


def zero(hilbert: _AbstractHilbert, cutoff: float = 1e-10, dtype: _DType = None):
    """
    Builds the :math:`0` operator, which has no connected components.

    Why we provide this is a mistery, as you could just multiply by 0.

    Args:
        hilbert: The hilbert space.
        dtype: The datatype to use for the matrix elements.

    Returns:
        An instance of {class}`nk.operator.LocalOperator`."""
    return _FermionOperator2nd(hilbert, constant=0, dtype=dtype, cutoff=cutoff)
