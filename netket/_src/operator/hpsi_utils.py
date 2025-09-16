# Copyright 2025 The NetKet Authors - All rights reserved.
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
import flax

from netket import jax as nkjax


def make_logpsi_op_afun(logpsi_fun, operator, variables):
    """Wraps an apply_fun into another one that multiplies it by an operator.

    This wrapper is made such that the operator is passed as the model_state
    of the new wrapped function, and therefore changes to the angles/coefficients
    of the operator should not trigger recompilation.

    Args:
        logpsi_fun: a function that takes as input variables and samples
        operator: a {class}`nk.operator.JaxDiscreteOperator`
        variables: The variables used to call *logpsi_fun*

    Returns:
        A tuple, where the first element is a new function with the same signature as
        the original **logpsi_fun** and a set of new variables to be used to call it.

    """
    # Wrap logpsi into logpsi_op
    logpsi_op_fun = nkjax.HashablePartial(_logpsi_op_fun, logpsi_fun)

    # Insert a new 'operator' key to store the operator. This only works
    # if operator is a pytree that can be flattened/unflattened.
    new_variables = flax.core.copy(variables, {"operator": operator})

    return logpsi_op_fun, new_variables


def _logpsi_op_fun(apply_fun, variables, x, *args, **kwargs):
    """
    This should be used as a wrapper to the original apply function, adding
    to the `variables` dictionary (in model_state) a new key `operator` with
    a jax-compatible operator.
    """
    # TODO: Move to global import
    from netket.operator import ContinuousOperator

    variables_applyfun, operator = flax.core.pop(variables, "operator")

    if isinstance(operator, ContinuousOperator):
        res = operator._expect_kernel(apply_fun, variables_applyfun, x)
    else:
        xp, mels = operator.get_conn_padded(x)
        xp = xp.reshape(-1, x.shape[-1])
        logpsi_xp = apply_fun(variables_applyfun, xp, *args, **kwargs)
        logpsi_xp = logpsi_xp.reshape(mels.shape).astype(complex)

        res = jax.scipy.special.logsumexp(logpsi_xp, axis=-1, b=mels)
    return res
