# Copyright 2024 The Netket Authors. - All Rights Reserved.
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

import netket as nk
import numpy as np

import jax

from .. import common

pytestmark = common.skipif_distributed


def test_sum_constraint():
    c1 = nk.hilbert.constraint.SumConstraint(0.0)
    c1b = nk.hilbert.constraint.SumConstraint(0.0)
    c2 = nk.hilbert.constraint.SumConstraint(2.0)

    assert c1 == c1b
    assert c1 != c2
    assert hash(c1) == hash(c1b)
    assert hash(c1) != hash(c2)

    assert isinstance(repr(c1), str)

    with pytest.raises(TypeError):
        nk.hilbert.constraint.SumConstraint(None)


def test_sum_partition_constraint():
    with pytest.raises(TypeError):
        nk.hilbert.constraint.SumOnPartitionConstraint(1, 2)
    # Length mismatch
    with pytest.raises(ValueError):
        nk.hilbert.constraint.SumOnPartitionConstraint((1, 2), (2, 3, 3))
    # None constraint invalid
    with pytest.raises(TypeError):
        nk.hilbert.constraint.SumOnPartitionConstraint((None, 2), (2, 3))

    c1 = nk.hilbert.constraint.SumOnPartitionConstraint((1, 2), (2, 3))
    c1b = nk.hilbert.constraint.SumOnPartitionConstraint((1, 2), (2, 3))
    c2 = nk.hilbert.constraint.SumOnPartitionConstraint((1, 2), (2, 4))

    assert c1 == c1b
    assert c1 != c2
    assert hash(c1) == hash(c1b)
    assert hash(c1) != hash(c2)

    assert isinstance(repr(c1), str)


def test_extra_constraint():
    sc1 = nk.hilbert.constraint.SumConstraint(0.0)
    sc2 = nk.hilbert.constraint.SumConstraint(2.0)
    sc3 = nk.hilbert.constraint.SumConstraint(4.0)

    c1 = nk.hilbert.constraint.ExtraConstraint(sc1, sc2)
    c1b = nk.hilbert.constraint.ExtraConstraint(sc1, sc2)
    c2 = nk.hilbert.constraint.ExtraConstraint(sc1, sc3)

    assert c1 == c1b
    assert c1 != c2
    assert hash(c1) == hash(c1b)
    assert hash(c1) != hash(c2)

    assert isinstance(repr(c1), str)

    with pytest.raises(TypeError):
        nk.hilbert.constraint.ExtraConstraint(1, 2)
    with pytest.raises(TypeError):
        nk.hilbert.constraint.ExtraConstraint[int, float]


# Constraint checking that first value is 1
class CustomConstraintPy(nk.hilbert.constraint.DiscreteHilbertConstraint):
    def __call__(self, x):
        return jax.pure_callback(
            self._call_py,
            (jax.ShapeDtypeStruct(x.shape[:-1], bool)),
            x,
            vmap_method="expand_dims",
        )

    def _call_py(self, x):
        # Not Jax compatible
        return x[..., 0] == 1

    def __hash__(self):
        return hash("CustomConstraintPy")

    def __eq__(self, other):
        if isinstance(other, CustomConstraintPy):
            return True
        return False

    def __repr__(self):
        return "CustomConstraintPy()"


def test_extra_constraint_integration(recwarn):
    base_constraint = nk.hilbert.constraint.SumConstraint(0.0)
    extra_constraint = CustomConstraintPy()
    joint_constraint = nk.hilbert.constraint.ExtraConstraint(
        base_constraint, extra_constraint
    )

    hi = nk.hilbert.Spin(0.5, 4, constraint=joint_constraint)
    bare_hi = nk.hilbert.Spin(0.5, 4)

    assert hi.n_states == np.sum(joint_constraint(bare_hi.all_states()))
    assert np.all(base_constraint(hi.all_states()))
    assert np.all(extra_constraint(hi.all_states()))
    assert np.all(joint_constraint(hi.all_states()))

    with pytest.warns(nk.errors.UnoptimisedCustomConstraintRandomStateMethodWarning):
        ran_states = hi.random_state(jax.random.key(1), 100)
    assert np.all(joint_constraint(ran_states))

    nk.config.netket_random_state_fallback_warning = False
    _ = hi.random_state(jax.random.key(1), 2)
    assert len(recwarn) == 0, "Warnings were raised during the test"
    nk.config.netket_random_state_fallback_warning = True


def test_hilbert_extra_constraint():
    c = nk.hilbert.constraint.SumConstraint(0.0)
    hi = nk.hilbert.Spin(0.5, 4, constraint=c)
    assert isinstance(hi * hi, nk.hilbert.TensorHilbert)
    with pytest.raises(TypeError):
        _ = hi**2
    assert hi == nk.hilbert.Spin(0.5, 4, total_sz=0.0)
    assert isinstance(repr(hi), str)

    c = nk.hilbert.constraint.SumConstraint(1)
    hi = nk.hilbert.Fock(2, 4, constraint=c)
    assert isinstance(hi * hi, nk.hilbert.TensorHilbert)
    with pytest.raises(TypeError):
        _ = hi**2
    assert hi == nk.hilbert.Fock(2, 4, n_particles=1)
    assert isinstance(repr(hi), str)

    with pytest.raises(ValueError):
        nk.hilbert.Spin(0.5, 4, total_sz=0.0, constraint=c)
    with pytest.raises(ValueError):
        nk.hilbert.Fock(2, 4, n_particles=2, constraint=c)


def test_constraint_interface_errors():
    def myconstraint(x):
        return np.sum(x, axis=-1)

    with pytest.raises(nk.errors.InvalidConstraintInterface):
        _ = nk.hilbert.Spin(0.5, 4, constraint=myconstraint)

    class CustomConstraint(nk.hilbert.constraint.DiscreteHilbertConstraint):
        x: int

        def __init__(self, x):
            self.x = x

        def __call__(self, x):
            pass

    with pytest.raises(nk.errors.UnhashableConstraintError):
        _ = nk.hilbert.Spin(0.5, 4, constraint=CustomConstraint(1))


def test_extra_constraint_spin_orbital_fermion():
    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, n_fermions_per_spin=(1, 2))
    assert isinstance(hi.constraint, nk.hilbert.constraint.SumOnPartitionConstraint)
    assert np.all(hi.constraint(hi.all_states()))

    hi = nk.hilbert.SpinOrbitalFermions(4, s=1 / 2, constraint=CustomConstraintPy())
    assert isinstance(hi.constraint, CustomConstraintPy)
    assert np.all(hi.constraint(hi.all_states()))

    hi = nk.hilbert.SpinOrbitalFermions(
        4, s=1 / 2, n_fermions_per_spin=(1, 2), constraint=CustomConstraintPy()
    )
    assert isinstance(hi.constraint, nk.hilbert.constraint.ExtraConstraint)
    assert isinstance(
        hi.constraint,
        nk.hilbert.constraint.ExtraConstraint[
            nk.hilbert.constraint.SumOnPartitionConstraint, CustomConstraintPy
        ],
    )
    assert np.all(hi.constraint(hi.all_states()))
