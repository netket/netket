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

from netket.utils.jax import WrappedApplyFun


def test_WrappedApplyFun():
    def fun():
        return 42

    wrapped = WrappedApplyFun(fun)
    assert wrapped.apply() == 42

    wrapped2 = WrappedApplyFun(fun)
    assert wrapped is not wrapped2
    assert wrapped == wrapped2
    assert hash(wrapped) == hash(wrapped2)
