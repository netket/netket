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

from textwrap import dedent
from functools import wraps

import warnings

import flax

# contains wrappers to deprecate some flax Modules that we were wrapping prior to flax 0.5
# TODO: eventually remove this.


def _warning_string(module_name):
    return dedent(
        f"""

        ============================================================================================

        `nk.nn.{module_name}` is deprecated and you should now directly use `flax.linen`.

                       ****************************************
                       THE SYNTAX HAS ALSO CHANGED! READ BELOW!
                       ****************************************

        Use `flax.linen.{module_name}(param_dtype=...)` instead of `netket.nn.{module_name}(dtype=...)`

         1) Flax has fixed their issues with complex numbers, so we encourage you to use flax directly
         instead of `netket.nn`

         2) The meaning of `module.dtype` in Flax is different than in NetKet!
            - `param_dtype` specifies the type of parameters to be used. In most cases you will want to
                            specify this one. For example, to use complex parameters you should use
                            `param_dtype=complex`.
                            By default it is `jnp.float32`.

            - `dtype`       is an optional specification stating the precision of the calculation, and
                            it has no impact unless you are working on a GPU or TPU. For example, by
                            specifying `param_dtype=jnp.float64` and `dtype=jnp.float32`, you will store
                            parameters in double precision, but perform calculations in single precision.

        ============================================================================================

       """
    )


def deprecated_module(original_module, module_name):
    @wraps(original_module)
    def call_deprecated_module(*args, **kwargs):
        if "param_dtype" in kwargs:
            err_msg = f"""
                    *************************************************************************
                    Use `flax.linen.{module_name}` instead of `netket.nn.{module_name}`.
                    *************************************************************************

            You are specifying `param_dtype` so you should be good.
            """

            raise KeyError(dedent(err_msg))

        else:
            warnings.warn(
                _warning_string(module_name), category=FutureWarning, stacklevel=3
            )

        if "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
            return original_module(*args, param_dtype=dtype, **kwargs)
        else:
            return original_module(*args, **kwargs)

    return call_deprecated_module


def deprecated_function(original_function, function_name):
    @wraps(original_function)
    def call_deprecated_function(*args, **kwargs):
        wrn_msg = f"""
                *************************************************************************
                Use `flax.linen.{function_name}` instead of `netket.nn.{function_name}`.
                *************************************************************************
        """

        warnings.warn(dedent(wrn_msg), category=FutureWarning, stacklevel=3)

        return original_function(*args, **kwargs)

    return call_deprecated_function


DenseGeneral = deprecated_module(flax.linen.DenseGeneral, "DenseGeneral")
Dense = deprecated_module(flax.linen.Dense, "Dense")
Conv = deprecated_module(flax.linen.Dense, "Conv")
ConvTranspose = deprecated_module(flax.linen.Dense, "ConvTranspose")

Embed = deprecated_module(flax.linen.Embed, "Embed")

MultiHeadDotProductAttention = deprecated_module(
    flax.linen.MultiHeadDotProductAttention, "MultiHeadDotProductAttention"
)
SelfAttention = deprecated_module(flax.linen.SelfAttention, "SelfAttention")
dot_product_attention = deprecated_function(
    flax.linen.dot_product_attention, "dot_product_attention"
)
make_attention_mask = deprecated_function(
    flax.linen.make_attention_mask, "make_attention_mask"
)
make_causal_mask = deprecated_function(flax.linen.make_causal_mask, "make_causal_mask")
combine_masks = deprecated_function(flax.linen.combine_masks, "combine_masks")
