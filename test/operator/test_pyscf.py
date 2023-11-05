# Copyright 2023 The Netket Authors. - All Rights Reserved.
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

import pytest

import numpy as np

import netket as nk
import netket.experimental as nkx


def test_pyscf():
    pytest.importorskip("pyscf")

    from pyscf import gto, scf, fci

    bond_length = 1.5109
    geometry = [
        ("Li", (0.0, 0.0, -bond_length / 2)),
        ("H", (0.0, 0.0, bond_length / 2)),
    ]
    mol = gto.M(atom=geometry, basis="STO-3G")

    mf = scf.RHF(mol).run()
    E_fci = fci.FCI(mf).kernel()[0]

    ha = nkx.operator.from_pyscf_molecule(mol, mo_coeff=mf.mo_coeff)

    assert ha.hilbert.n_orbitals == 6
    assert ha.hilbert.n_fermions == 4
    assert ha.hilbert.n_fermions_per_spin == (2, 2)

    # check that ED gives same value as FCI in pyscf
    np.testing.assert_allclose(E_fci, nk.exact.lanczos_ed(ha))
