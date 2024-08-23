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

import netket as nk
import numpy as np

import jax

from .. import common

pytestmark = common.skipif_distributed


def test_extra_constraint():

    # Constraint checking that first value is 1
    class CustomConstraintPy(nk.hilbert.constraint.DiscreteHilbertConstraint):

        def __call__(self, x):
            return jax.pure_callback(
                self._call_py,
                (jax.ShapeDtypeStruct(x.shape[:-1], bool)),
                x,
                vectorized=True,
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

    ran_states = hi.random_state(jax.random.key(1), 100)
    assert np.all(joint_constraint(ran_states))
