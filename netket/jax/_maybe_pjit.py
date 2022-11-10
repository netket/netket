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

from typing import Callable, Optional, Any, Sequence, Iterable, Tuple, Union

import jax

from netket import config

AxisName = Any

def pmap(fun: Callable,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: Union[int, Iterable[int]] = (),
    devices = None,  # noqa: F811
    backend: Optional[str] = None,
    axis_size: Optional[int] = None,
    donate_argnums: Union[int, Iterable[int]] = (),
    global_arg_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None,
    ):

    axis_name = "mpi"

    if config.netket_experimental_pmap:
        return jax.pmap(fun, axis_name, in_axes=in_axes, 
            out_axes=out_axes, static_broadcasted_argnums=static_broadcasted_argnums, devices=devices,
            backend=backend, axis_size=axis_size, donate_argnums=donate_argnums, global_arg_shapes=global_arg_shapes)

    else:
        return jax.jit(fun, static_argnums=static_broadcasted_argnums)