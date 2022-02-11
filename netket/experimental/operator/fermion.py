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

from netket.utils.types import DType
from netket.hilbert.abstract_hilbert import AbstractHilbert
from netket.experimental.operator import FermionOperator2nd


def destroy(
    hilbert: AbstractHilbert, site: int, sz: int = None, dtype: DType = complex
):
    """
    Builds the fermion destruction operator :math:`\\hat{a}` acting on the `site`-th of
    the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return FermionOperator2nd(hilbert, ("{}".format(idx),), dtype=dtype)


def create(hilbert: AbstractHilbert, site: int, sz: int = None, dtype: DType = complex):
    """
    Builds the fermion creation operator :math:`\\hat{a}^\\dagger` acting on the `site`-th of
    the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return FermionOperator2nd(hilbert, ("{}^".format(idx),), dtype=dtype)


def number(hilbert: AbstractHilbert, site: int, sz: int = None, dtype: DType = complex):
    """
    Builds the number operator :math:`\\hat{a}^\\dagger\\hat{a}`  acting on the
    `site`-th of the Hilbert space `hilbert`.

    Args:
        hilbert: The hilbert space
        site: the site on which this operator acts
        sz: spin projection

    Returns:
        The resulting FermionOperator2nd
    """
    idx = _get_index(hilbert, site, sz)
    return FermionOperator2nd(hilbert, ("{}^ {}".format(idx, idx),), dtype=dtype)


def _get_index(hilbert: AbstractHilbert, site: int, sz: float = None):
    """go from (site, spin_projection) indices to index in the (tensor) hilbert space"""
    if sz is None:
        return site
    else:  # we assume hilbert is a SpinOrbitalHilbert
        return hilbert._get_index(site, sz)
