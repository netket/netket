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

import jax


def jit_if_singleproc(f, *args, **kwargs):
    if n_nodes == 1:
        return jax.jit(f, *args, **kwargs)
    else:
        return f


def get_afun_if_module(mod_or_fun, *args, **kwargs):
    if hasattr(mod_or_fun, "apply"):
        return mod_or_fun.apply
    else:
        return mod_or_fun
