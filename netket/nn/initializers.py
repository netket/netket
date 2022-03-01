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

from jax.nn import initializers
from netket.utils.deprecation import deprecated


_func_names = [
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "lecun_normal",
    "lecun_uniform",
    "normal",
    "ones",
    "orthogonal",
    "delta_orthogonal",
    "uniform",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
    "zeros",
]
_msg = "`netket.nn.initializers` is deprecated. Use `jax.nn.initializers` instead."
for func_name in _func_names:
    locals()[func_name] = deprecated(_msg, func_name)(getattr(initializers, func_name))
