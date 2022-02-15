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

from netket.utils.types import DType as _DType
from netket.hilbert.abstract_hilbert import AbstractHilbert as _AbstractHilbert
from netket.experimental.operator import FermionOperator2nd as _FermionOperator2nd


def destroy(hilbert: _AbstractHilbert, site: int, sz: int = None, dtype: _DType = None):
    """
    Builds the fermion destruction operator :math:`\\hat{a}` acting on the `site`-th of
    the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site (int): the site on which this operator acts
        sz (int): spin projection quantum number (e.g. sz=-0.5 for a spin-1/2 down)

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(hilbert, (f"{idx}",), dtype=dtype)


def create(hilbert: _AbstractHilbert, site: int, sz: int = None, dtype: _DType = None):
    """
    Builds the fermion creation operator :math:`\\hat{a}^\\dagger` acting on the `site`-th of
    the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site (int): the site on which this operator acts
        sz (int): spin projection quantum number (e.g. sz=-0.5 for a spin-1/2 down)

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(hilbert, (f"{idx}^",), dtype=dtype)


def number(hilbert: _AbstractHilbert, site: int, sz: int = None, dtype: _DType = None):
    """
    Builds the number operator :math:`\\hat{a}^\\dagger\\hat{a}`  acting on the
    `site`-th of the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts
        site (int): the site on which this operator acts
        sz (int): spin projection quantum number (e.g. sz=-0.5 for a spin-1/2 fermion with spin down)

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return _FermionOperator2nd(hilbert, (f"{idx}^ {idx}",), dtype=dtype)


def _get_index(hilbert: _AbstractHilbert, site: int, sz: float = None):
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
        return hilbert._get_index(site, sz)
    else:
        raise NotImplementedError(
            f"no method _get_index available for hilbert space {hilbert} that allows to find the position in hilbert space based on a spin projection value sz"
        )


def identity(hilbert: _AbstractHilbert, dtype: _DType = None):
    """identity operator"""
    return _FermionOperator2nd(hilbert, [], [], constant=1.0, dtype=dtype)


def zero(hilbert: _AbstractHilbert, dtype: _DType = None):
    """returns an object that has no contribution, meaning a constant of 0"""
    return _FermionOperator2nd(hilbert, [], [], constant=0.0, dtype=dtype)
